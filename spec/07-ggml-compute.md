# 07: Build ggml, load it, and multiply a matrix by a vector

**Status**: done   **Depends on**: 01, 06

## Problem & context

`06` leaves the engine holding pointers into a mapped model file and nothing that can touch them.
There is no maths in the project at all: no library to call, no kernel, no way to turn two lists of
numbers into a third. A forward pass cannot be written until something computes, and every later
story (attention, a sampler, `bench`) sits on that.

This spec adds a build step and one module. `build.lua` fetches ggml v0.29.0 and builds its CPU
backend, leaving shared libraries in the package's target directory. `tllm.compute` loads that
library through `ffi` and offers exactly one operation: a matrix by a vector product.

```teal
local record compute
  record Backend
    path: string
    version: string
    matvec: function(Backend, {{number}}, {number}): {number} | nil, string | nil
  end

  load: function(path?: string): Backend | nil, string | nil
end
```

ggml is chosen over hand-written kernels because later work has to agree with what `llama.cpp`
prints, and agreeing by construction is cheaper than agreeing by reimplementation. It belongs in the
core rather than a plugin because every later story in this folder sits on this one operation, and a
plugin would need this contract in the core before it could be written.

The pin is the tag `v0.29.0`, commit `57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234`
(github.com/ggml-org/ggml). Its `CMakeLists.txt` sets `GGML_VERSION` to `0.29.0` and
`ggml_version()` returns that string, so the version the engine reports is the library's own, not a
second copy of the constant in this project. The flags are in FR-001 rather than prose because each
one is load bearing: shared libraries or there is nothing to load, tests and examples off for time,
Metal, Accelerate and BLAS off so CUDA is the only other backend built, and backends as modules the
library loads at run time, so a machine without a driver still opens it.

ggml is not vendored. `build.lua` fetches the pinned commit's source archive, unpacks it and compiles
it with ggml's own cmake, and keeps the unpacked source, the build tree and the install tree in
`target/ggml-cache`, a sibling of the copied package directory, so a rebuild reuses the objects it
already compiled instead of fetching and compiling ggml again. lde reruns a build script whenever
`src/`, `lde.json` or `build.lua` change (lde docs, build scripts), so a source edit reruns the
script and pays only what that cache does not already hold. `target/` is ignored by git, so the tree
gains one build script and no third party source.

Building needs cmake, a C and a C++ compiler, the CUDA toolkit the build compiles `libggml-cuda` with,
and network on the first build. A machine without them fails at build time, not at the first
multiplication, and nothing here falls back to a downloaded binary. The CUDA toolkit is a build
dependency only: the library that is placed does not name the driver or the CUDA runtime, so it opens
on a machine that has neither.

`build.lua` is the one file in the tree that is not Teal: lde runs a build script only when it is
named `build.lua` and only through lde's runtime, and that file sits outside the `check` and
`format` file selection, which matches `*.tl` only. `spec/README.md` states the same exception.

A package with a build script is copied into the target directory without the Teal compilation lde
otherwise runs, so `build.lua` writes the `.lua` for every `src/*.tl` itself, through the `tl`
compiler `lde.json` adds as a devDependency. Type checking is unchanged and still runs through
`ldx`.

The build produces `libggml` and `libggml-base` as shared libraries, each with a versioned file name
and symlinks, and the first names the second in its own dependency list. The CPU and CUDA backends
are modules, `libggml-cpu-*` and `libggml-cuda`, that the registry inside `libggml` loads when it
starts instead of libraries `libggml` links against, so a missing driver makes one module fail to
load and finds no device rather than stopping the library from opening at all.
Moving a library out of the build tree breaks its link to its sibling twice: glibc does not search a
dlopened library's own directory, and a library left in the build tree carries an absolute runpath
into that tree, so it keeps looking for its sibling where it was compiled. FR-001 therefore forces
the install runpath into the build tree too, and `$ORIGIN` is the only runpath that still resolves
once the files have moved, for `libggml-base` and for every module. T-001 is what proves it:
`libggml-base` is mapped from `target/tllm` and not from the cache. The file `compute.load` opens is
`libggml.so`, and the modules it loads come from the same directory.

With no path, `compute.load` opens `target/tllm/libggml.so` relative to the current directory, which
is the package root whenever lde runs the engine or its tests. Started from anywhere else it fails the
way FR-005 describes, and the caller passes the path instead. That optional path also gives FR-005 a
file that is not a library to refuse, and a build that left the file elsewhere something to point at.

The multiplication maps onto ggml's own product. A ggml tensor has its first dimension contiguous, so
the matrix becomes a tensor of `row length` by `row count` and the vector a tensor of `row length` by
one; `ggml_mul_mat` returns `row count` numbers in contiguous memory, one per matrix row, each the dot
product of that row with the vector. The call creates its own ggml context, tensor data, graph and CPU
backend, computes, reads the numbers back and frees everything it made, on the computing path and on
every refusing one. The arena the context runs in is allocated by the module and sized from the
element counts, so no matrix is silently capped at an arena size this document does not state, and
an allocation that fails comes back as a line instead of ggml's abort. A CPU backend that cannot
be created or a graph compute whose status is not success leaves no numbers to return, so that case
also frees what it made and returns nil with one line.

The shapes are checked before ggml is called, not after. `ggml_mul_mat` asserts that the first
dimensions match and aborts the process when they do not, so a plugin that passes a two element
vector would take the server down, and a zero length dimension is not something this contract lets
through either. Refusing first is what makes `US-005` true.

Nothing here knows a model, a token or a layer, and the only type involved is f32, because that is
what ggml returns from this operation.

## User stories

- US-001 (P1) As the engine author, I want ggml's CPU library built from a pinned source and placed where the engine looks, so that a fresh checkout can compute without me installing a C library by hand.
- US-002 (P1) As the engine author, I want one call that opens that library and says which file and which ggml version I am using, so that a bug report and a model run name the same maths.
- US-003 (P2) As the engine author, I want a load that fails to return one line holding the path, so that a missing or half-built library is a message rather than a crash.
- US-004 (P1) As the engine author, I want a matrix by vector product through ggml with numbers I can work out by hand, sized from its inputs and leaving nothing behind, so that a forward pass has a proven path to sit on.
- US-005 (P2) As the engine author, I want a malformed shape refused with one line, so that a plugin's mistake is a return value and the process stays up.

## Functional requirements

- FR-001 (US-001) WHEN lde refreshes the package's target directory, THE SYSTEM SHALL fetch ggml at commit `57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234`, configure it with `BUILD_SHARED_LIBS` on, `GGML_BUILD_TESTS` off, `GGML_BUILD_EXAMPLES` off, `GGML_METAL`, `GGML_ACCELERATE` and `GGML_BLAS` off, `GGML_BACKEND_DL` on, `GGML_CPU_ALL_VARIANTS` on, `GGML_CUDA` on, `CMAKE_CUDA_ARCHITECTURES` set to `86`, `CMAKE_BUILD_WITH_INSTALL_RPATH` on and `CMAKE_INSTALL_RPATH` set to `$ORIGIN` passed through the shell unexpanded, and build it.
  Not testable: the fetched commit and the flags are read from `build.lua`, and whether the build ran
  is visible in `target/tllm`, which holds the libraries afterwards or nothing.
- FR-002 (US-001) WHEN the build finishes, THE SYSTEM SHALL place `libggml` and `libggml-base`, with the versioned names and symlinks the build produces, and the backend modules the build produces, `libggml-cpu-*` and `libggml-cuda`, in the package's target directory, so that loading the placed `libggml` maps `libggml-base` from that directory and not from the build cache and loads its backends from there too.
- FR-003 (US-002) THE SYSTEM SHALL set a handle's `path` to the library file it opened: the string `target/tllm/libggml.so`, relative to the current directory, when `compute.load` is given no path, and the argument unchanged when one is given.
- FR-004 (US-002) WHEN `compute.load` returns a handle, THE SYSTEM SHALL set `version` to the version string the library reports for itself, which is `0.29.0` for the pinned source.
- FR-005 (US-003) IF the file to open cannot be loaded, THEN THE SYSTEM SHALL return nil and one line holding that path.
- FR-006 (US-004) WHEN a handle's `matvec(matrix, vector)` is called with a matrix holding at least one row and every row holding as many numbers as the vector, THE SYSTEM SHALL return one number per matrix row, in row order, each the sum over the row's positions of the row's number times the vector's number at that position.
- FR-007 (US-004) THE SYSTEM SHALL compute each returned number in single precision, so that the row `0.1`, `0.2` with the vector `1`, `1` returns `0.30000001192092896` rather than the number `0.1 + 0.2` gives in Lua.
- FR-008 (US-004) THE SYSTEM SHALL size the ggml context it creates from the matrix's and vector's element counts, so that the size of a matrix this operation accepts is bounded by memory and not by a constant.
  Not testable: telling an arena sized from the element counts from a fixed one that is large enough
  for the cases above needs a matrix past every constant worth choosing; one that does, 4096 by 4096
  f32, is 64 MiB of element data, a slow, memory-heavy call `## Tests` does not spend; a reader
  confirms the size is computed from the element counts instead.
- FR-009 (US-004) IF no ggml context can be created for that size, THEN THE SYSTEM SHALL return nil and one line holding `context`.
  Not testable: making an allocation of that size fail needs a memory limit the suite cannot set portably; the branch returns nil and one line, which a reader confirms by inspection.
- FR-010 (US-004) WHEN `matvec` returns, THE SYSTEM SHALL free every ggml context, tensor, graph and backend it created, on the computing path and on every refusing path.
  Not testable: a leak shows only as growth across many calls, which no portable assertion catches; a reader confirms one free of each object created.
- FR-011 (US-005) IF the matrix holds no row, or the vector holds no number, or any row holds no number, THEN THE SYSTEM SHALL return nil and one line holding `empty`.
- FR-012 (US-005) IF a row holds a different count of numbers than the vector, THEN THE SYSTEM SHALL return nil and one line holding that row's length, for the first such row in order, and the vector's length.
- FR-013 (US-005) IF more than one refusal applies to a call, THEN THE SYSTEM SHALL report an empty shape before a row length that differs.

## Non-goals

- Linux x86_64 only: no macOS, no Windows, no other architecture, no prebuilt download; the source is compiled.
- No other operation, type or backend: no add, normalise, rope, softmax or row gather, no quantised tensor, no attention, no Metal, Vulkan or BLAS, no thread count, no buffer reuse between calls. The CUDA backend is built and placed by the flags `21` adds, and this spec computes nothing on it.
- No model: `serve` loads a GGUF through `weights.load` under `08`, and nothing reads a weight through this library.

## Tests

- T-001 (FR-002, FR-003, FR-004): `the placed ggml loads, resolves its sibling and reports its version`, proves that with no library search path variable set, `compute.load()` returns a handle whose `path` is `target/tllm/libggml.so` and whose `version` is `0.29.0`, that `/proc/self/maps` then holds `target/tllm/libggml-base`, and that `compute.load(handle.path)` returns a handle whose `path` is that same string.
- T-002 (FR-005): `a library that cannot be loaded is refused`, proves a path that does not exist, and a file `not-a-library.so` written with the line `this is not a library`, each return nil and one line holding that path.
- T-003 (FR-006, FR-007): `a matrix times a vector comes back in single precision`, proves a handle's `matvec` gives `{6, 15}` for `{{1, 2, 3}, {4, 5, 6}}` with `{1, 1, 1}`, `{0.5}` for `{{1, -2, 0.5}}` with `{4, 2, 1}`, and `{0.30000001192092896}` for `{{0.1, 0.2}}` with `{1, 1}`.
- T-004 (FR-011, FR-012, FR-013): `a matrix of the wrong shape is refused`, proves `matvec({}, {1})` and `matvec({{}}, {1})` each return nil and one line holding `empty`, that `matvec({{1, 2}, {3}}, {1, 1})` returns nil and one line holding `1` and `2`, and that the same handle still gives `{6, 15}` for `{{1, 2, 3}, {4, 5, 6}}` with `{1, 1, 1}` afterwards.

## Open questions

- None.

## Amendments

- None.
