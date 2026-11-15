# 21: Keep a model's weights in GPU memory and compute there

**Status**: done   **Depends on**: 07, 11, 15

## Problem & context

Every number the engine produces today comes out of the CPU. `build.lua` compiles ggml with its CUDA
backend off, `compute.load` opens `target/tllm/libggml.so`, and every operation builds a small ggml
graph whose leaf tensors point into the mapped model file, hands that graph to a CPU backend made in
the same call, and copies the result back into a Lua table. On a machine with an NVIDIA card, the
weights never leave host memory and the card is idle.

This spec puts a loaded model's weights in the card's memory and computes there, so that a machine
with a usable NVIDIA device runs every operation that reads a weight on it and returns the numbers
the CPU returned. `Backend.matvec` from `07` takes numbers rather than weights, makes its own host
tensors and is untouched by this spec.

Three things in the tree decide the shape of the work.

`weights.load` maps the GGUF and holds a `Tensor` per name with a host pointer.
`compute.mapped_of` describes every one of those tensors once, in a long-lived ggml context, and the
operation helpers read weights through those descriptors. So the weights are placed in one function,
and it is there that they must end up in a device buffer.

Graph tensors are where the second and third changes are. `compute.run` makes the graph's tensors
through `tensor_in` and writes their numbers through `vector_of`, `scalar_of` and `host_tensor`,
which write host memory directly on the CPU path. A tensor with no buffer from the device's backend
cannot be computed there at all, and a device tensor's `data` is a device pointer, so for a graph
computed on the card the arena holds no numbers, `ggml_backend_alloc_ctx_tensors` puts every tensor
of the graph in device memory, the writes those helpers collected go in with
`ggml_backend_tensor_set`, and results come back with `ggml_backend_tensor_get`, which is also how
`read_floats` and the linear-attention readback at the end of `linear_press` read them.

The linear-attention history and state are plain `ffi` float arrays that `compute` hands to the
graph as leaf tensors, so for a graph computed on the card they are copied into device memory with
the other inputs and read back with `ggml_backend_tensor_get` like any other result. The attention
cache is a different thing: `attention_block` keeps each prefix's keys and values in host memory as
ggml tensors of the geometry's `cache_type`, and its graph reads them out of that memory, so on a
device backend that cache is copied into device memory for the length of a call and never kept
there.

`mapped_of` is called by five entry points, `attach`, `attend`, `feed_forward`, `linear_attention`
and `model`, each for the same handle. A device copy per call would be five copies of one model and
does not fit in any card, so the copy belongs to the model handle: the first entry point on a backend
copies, later ones on the same handle and the same backend reuse it. The copy is per backend as well
as per handle because one process can hold a device backend and a CPU-only one over the same handle,
and because `compute.load` may be called twice.

Nothing new is added to `weights.Handle`, which keeps the five fields `06` publishes, and the
`release` that record already carries is the call that gives the device copy back, once: `compute`
wraps it the first time it copies a handle, frees every copy of that handle and then calls what the
handle carried.

The build has one trap. ggml configures `CMAKE_CUDA_ARCHITECTURES` to `native` when nothing is
passed, which reads the architecture from a visible device at configure time. Configured where no
device node is visible (a sandbox, a container, a headless build host) it resolves to nonsense and
the build fails. The development machine's card is a GeForce RTX 3080 Laptop GPU on GA104M, compute
capability 8.6, with CUDA toolkit 13.4 and driver 615.78.08 installed, and a ggml-cuda source from
the pinned commit compiles for `sm_86` with the toolkit's own `nvcc` and the installed gcc 16.2.1, so
the build pins the architecture instead of asking the machine, as the amendment to `07/FR-001`
states.

The pinned commit is the one `07` already builds, `57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234`. With
`GGML_CUDA` on, ggml builds `libggml-cuda` beside the CPU backends, and because `07` builds the
backends as modules, `libggml.so` does not name it: the registry inside `libggml` loads the modules
from the directory they were placed in. That is what keeps a machine without an NVIDIA driver
working, since the module fails to open and finds no device instead of stopping `libggml.so` from
opening at all, and it is why `compute.load` finds the modules before it looks for a device. The
build carries the modules from the install tree beside the library it already copied, so the build
change is the flags and one more copy.

The maths surface is unchanged. Every function still takes and returns Lua numbers, `nil` and one
line stay the refusal, and no caller in `serve`, `chat` or `generate` changes: the device is picked
by the library, not by a flag, a route or a config key.

```teal
local record compute
  record Backend
    path: string
    version: string
    device: string | nil
    memory: function(Backend): integer | nil, integer | nil
    matvec: function(
      Backend, {{number}}, {number}
    ): {number} | nil, string | nil
  end
  load: function(path?: string, offload?: boolean): Backend | nil, string | nil
end
```

`device` names the device the maths runs on and is nil when the maths runs on the CPU. `memory`
returns the free bytes of the device the library found and then its total bytes, at the moment it is
called, and returns nil and nil when the library found no device. The second argument of `load`
exists so that one machine can run both paths in one process: `false` asks for the CPU even when a
card is there, which is what FR-006's case needs and what lets FR-004's case hold a device result
next to a CPU result.

## User stories

- US-001 (P1) As the engine author, I want a loaded model's weights copied into my NVIDIA card's memory once, so that the maths does not read them out of host memory.
- US-002 (P1) As the engine author, I want every operation to return the numbers the CPU returns, so that moving to the card keeps every number within a hundred-thousandth of the larger of one and the CPU's number.
- US-003 (P1) As the engine author, I want a machine with no usable card, or a model that does not fit on one, to keep computing on the CPU, so that one build runs everywhere.
- US-004 (P2) As the engine author, I want the device named and its memory reported, so that a bug report says where the maths ran and how much of the card is in use.
- US-005 (P2) As the engine author, I want the card's memory back when a model is released, so that loading another model does not exhaust it.

## Functional requirements

- FR-001 (US-001) WHEN lde builds the package, THE SYSTEM SHALL place `libggml-cuda`, the backend module the build produces, in the package's target directory beside `libggml`, `libggml-base` and the CPU backend modules.
- FR-002 (US-001) WHEN an entry point is built from a model handle, and the backend it is built on found a CUDA device, and that backend was not asked to stay on the CPU, THE SYSTEM SHALL copy every tensor of that handle into device memory and SHALL reuse that copy for every later entry point built from the same handle on the same backend.
- FR-003 (US-001) WHEN an operation of FR-004's set is computed on a backend whose weights are in device memory, THE SYSTEM SHALL compute that operation's graph on the CUDA backend, and no tensor of that graph SHALL carry a host data pointer while it is computed. The attention cache stays in host memory, so an attention block's graph copies it into device memory for the length of a call, and its tests call `compute.model(backend, handle, 2, "f32")` so the numbers they pin stand.
  Not testable: the library reports the devices it found, not which backend ran a graph, and the
  returned numbers are the same either way; a reader confirms the backend the graph is computed with
  and the buffer every tensor of the graph is bound to.
- FR-004 (US-002) WHEN an operation from `matvec`, `row`, `norm`, `rope`, `softmax`, `add`, `swiglu`, `sigmoid_gate`, `attention:block`, `feed_forward:block`, `feed_forward:scores`, `linear_attention:block` and `model:forward` is computed with the handle's weights in device memory, THE SYSTEM SHALL return the count of numbers the same operation returns on a CPU-only backend, holding at each position a number `g` whose `abs(g - c)` is at most `0.00001 * max(1, abs(c))`, where `c` is the CPU-only number at that position.
- FR-005 (US-003) WHEN no CUDA device is found, THE SYSTEM SHALL compute on the CPU and report no device name.
- FR-006 (US-003) WHEN `compute.load` is given `false` as its second argument, THE SYSTEM SHALL compute on the CPU and report no device name, whatever devices the machine has.
- FR-007 (US-003) IF the device copy of a model's weights cannot be allocated, THEN THE SYSTEM SHALL compute that model's operations on the CPU and report no device name.
  Not testable: forcing the allocation to fail needs a fixture larger than the card's free bytes,
  which means writing gigabytes on every run at a size that depends on the card; the branch is one
  allocation check, confirmed by inspection.
- FR-008 (US-004) WHEN `memory` is called on a backend whose library found a device, THE SYSTEM SHALL return that device's free bytes and then its total bytes as they stand at that moment, and SHALL return nil and nil when the library found no device.
- FR-009 (US-004) WHEN a model's weights are in device memory, THE SYSTEM SHALL set the backend's `device` to that device's name, and SHALL set it to nil when they are not.
- FR-010 (US-005) WHEN the `release` a model handle carries is called for the first time after its weights were copied to device memory, THE SYSTEM SHALL free that copy, and a second call SHALL free nothing further.

## Non-goals

- No partial offload: no layer count, no tensor split, no second device. A model fits on one card whole or it runs on the CPU.
- No device-resident attention state: the linear-attention history and state stay host buffers, copied into each operation like the other inputs, and the attention cache stays where it is, in Lua.
- No speed claim: nothing here is timed, and no throughput target is asserted.
- No other backend or architecture: no Vulkan, ROCm or Metal, and one architecture, `sm_86`, so a different card needs the flags in `build.lua` changed.

## Tests

- T-001 (FR-001, FR-002, FR-008, FR-009): `a loaded model sits in the device once`, proves with a GGUF whose tensors hold at least 128 MiB, zero-filled through the same in-process GGUF builder the weights tests use, that after `compute.attach` `/proc/self/maps` holds `target/tllm/libggml-cuda`, that the free bytes `memory` reports have fallen by at least nine tenths of the model's tensor bytes, that the total `memory` reports is not below its free bytes, and that `device` is a non-empty name; and that `compute.feed_forward` on the same handle afterwards lowers those free bytes by less than a tenth of the model's tensor bytes and leaves `device` unchanged.
- T-002 (FR-004): `the device maths matches the CPU maths`, proves that a small GGUF of each block kind gives the same numbers through a device backend and a CPU-only backend for `matvec`, `row`, `norm`, `rope`, `softmax`, `add`, `swiglu`, `sigmoid_gate`, an attention block, a `feed_forward` block and its `scores`, a `linear_attention` block and a model pass, each pair within `0.00001 * max(1, abs(c))`.
- T-003 (FR-005, FR-006): `a CPU-only load reports no device and still computes`, proves `compute.load(nil, false)` gives a backend whose `device` is nil, whose `memory` free bytes stay within 16 MiB of what they were before `attach`, and whose `attach` still gives the known numbers for `matvec` over a weight and for `norm`; and, where `memory` returns nil and nil, that a load that was not asked for the CPU gives that same nil pair and those same known numbers.
- T-004 (FR-010): `releasing the model gives the device memory back`, proves that after the handle is released the free bytes `memory` reports have risen by at least nine tenths of the model's tensor bytes, and that they stay so after a second release on the same handle.

On a machine where `compute.load()`'s `memory` returns nil and nil, the library found no device:
T-001, T-002 and T-004 skip with the line `no CUDA device`, and T-003 asserts its no-device half.

A linear block on the card has one shape limit the CPU does not: ggml's CUDA convolution needs the
channels it convolves to be a multiple of 128 and a convolution of 3, 4, 5, 7, 9 or 15 taps, so the
small block T-002 compares has 128 channels and a four tap convolution.

## Open questions

- None.

## Amendments

- None.
