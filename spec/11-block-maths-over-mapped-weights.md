# 11: Do a block's maths over the mapped weights

**Status**: done   **Depends on**: 06, 07

## Problem & context

`07` offers one operation, a matrix by a vector product over two Teal lists, and it copies both
lists into a ggml graph. `06` maps a GGUF file and gives every tensor a pointer, a shape and a
type, but nothing can read those bytes. A forward pass needs both halves joined: multiply a hidden
vector by a q4_K expert matrix, gather a q8_0 embedding row, normalise with an f32 gain, rotate a
head with rope, softmax a list of scores, and gate one list with another. None of that exists.

This spec adds `compute.attach(backend, handle)` and one record, `Maths`, with eight operations.
`attach` builds one ggml context whose tensors point at the weights handle's mapping, so attaching a
23 GB model allocates tensor descriptors and no weight bytes. Each operation creates its own ggml
context, builds its inputs and result there with the named weight tensor as an input (for a
three-dimensional tensor, a descriptor over the matrix the index selects), computes on the CPU
backend, reads the result back as a Teal list, and frees that context, leaving the attached
descriptors untouched. `load` keeps the library it opened under the backend it returns, so `attach`
reaches the same library without a field the published record does not hold. No dequantiser is
written and no graph is exposed: the next spec composes these operations into one token through the
layers. The eight belong in the core because the next spec composes them and ggml is already the
core's maths; a plugin would need this contract in the core before it could be written.

The local Qwen3.6 file names the shapes this must carry: 41 blocks, 2048 embedding numbers, a fused
8192 wide attention projection, 256 experts whose gate and up tensors are 3-D with 256 matrices of
512 rows, f32 norms and routers, q8_0 projections and embeddings, q4_K gate and up experts, and q5_K
down experts. The tests use hand-made fixtures, including q8_0, q4_K and q5_K blocks, an f16 tensor,
a four-dimensional tensor and a tensor with no dimensions, because a 23 GB file cannot be a test
input. q8_0, q4_K and q5_K are separate ggml decoders with different block layouts, so the tests
cover one block of each. A quantised weight is multiplied through ggml's own quantised kernel,
which rounds the vector it is given to that type's vector format first, so a result carries that
rounding; the tests choose vectors the formats hold exactly.

Rope is the text path only. The file states `rope.dimension_count` 64, `freq_base` 10000000 and
`rope.dimension_sections` 11, 11, 10, 0, and llama.cpp builds such a block with IMRoPE
(`src/models/qwen35moe.cpp`, `build_layer_attn`, commit
`ccd360f060a0f17b87ee0d4c5d91ee84cfdff484`, raw.githubusercontent.com/ggml-org/llama.cpp). For text
every section carries the same position, and with equal positions ggml's multi-section cache is its
NEOX cache: `theta_t` is advanced by `theta_scale` once per pair (`theta_t *= theta_scale`), so pair
`i` reads `position * theta_scale^i` whichever section it falls in, and `rotate_pairs` then uses the
NEOX offset for MROPE and IMRoPE alike (ggml commit
`57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234`, `src/ggml-cpu/ops.cpp`, `ggml_mrope_cache_init` and
`rotate_pairs`). This spec therefore offers NEOX with one position, which is what one text token
needs; positions that differ per section are a later spec, and the next spec's run against the local
model is what confirms the equivalence end to end.

Refusals keep the shape `06` and `07` use: nil and one line, so a mistake in the next spec's wiring
is a return value and the process stays up. `attach` allocates an arena for tensor descriptors and a
ggml context inside it, and it refuses a backend `compute.load` did not return, because the library
it needs is kept under that handle. Both are given back with the last maths handle that reads them,
and `close` is what drops that handle's hold on the mapping, so the weights file can be released.
Only the arena failure is not in `## Tests`, because making that allocation fail needs a memory
limit the suite cannot set.

The arena each operation runs in is sized by a constant beside the operation, and a constant can be
one tensor short. ggml answers a context with no room for one more tensor with NULL rather than
placing a short one, and every later call in the same build would dereference that NULL, so the
tensor request raises at once and the operation catches the raise around its build and returns it as
one line. That covers the tensors an operation asks its context for by shape, `vector_of`,
`scalar_of`, `host_tensor` and `matrix_of`; the constant beside each operation remains the contract
for the rest. `ggml_used_mem` cannot serve as this guard: it reports what ggml placed, and ggml never
places past the buffer, so a comparison against the buffer's size can never fire.

```teal
local weights = require("tllm.weights")

local record compute
  record Backend
    path: string
    version: string
    memory: function(Backend): integer | nil, integer | nil
  end

  record Maths
    matvec: function(
      Maths, string, {number}, index?: integer
    ): {number} | nil, string | nil
    row: function(Maths, string, integer): {number} | nil, string | nil
    norm: function(
      Maths, {number}, string, number
    ): {number} | nil, string | nil
    rope: function(
      Maths, {number}, integer, integer, number
    ): {number} | nil, string | nil
    softmax: function(Maths, {number}): {number} | nil, string | nil
    add: function(Maths, {number}, {number}): {number} | nil, string | nil
    swiglu: function(
      Maths, {number}, {number}
    ): {number} | nil, string | nil
    sigmoid_gate: function(
      Maths, {number}, {number}
    ): {number} | nil, string | nil
    close: function(Maths)
  end

  load: function(path?: string, offload?: boolean): Backend | nil, string | nil
  attach: function(Backend, weights.Handle): Maths | nil, string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want the model's mapped tensors multiplied and gathered through ggml, so that a block reads q4_K and q8_0 weights without a dequantiser of my own.
- US-002 (P1) As the engine author, I want norm, rope, softmax and the gated elementwise maths with ggml's own numbers, so that one token through the layers agrees with llama.cpp.
- US-003 (P2) As the engine author, I want a name, type or shape that does not fit back as one line, so that a wiring mistake is a return value and not a crash.

## Functional requirements

- FR-001 (US-001) WHEN `compute.attach(backend, handle)` is called with a `backend` that
  `compute.load` returned, THE SYSTEM SHALL return a maths handle whose operations read the tensors
  of that weights handle by name, and nil as its second value.
- FR-002 (US-001) WHEN `maths:matvec(name, vector, index)` is called with a vector whose length is
  the named tensor's first dimension, THE SYSTEM SHALL return one number per row of the matrix that
  tensor names, each the sum over the row's positions of the row's decoded number times the
  vector's number at that position, where a f32, q8_0, q4_K or q5_K tensor's row numbers are the
  values ggml decodes from its stored bytes, a one-dimensional tensor is one row, and a
  three-dimensional tensor's `index`, counting from zero, selects the matrix at that position along
  its third dimension, and a tensor with no dimensions is one row of one number; for a non-f32
  tensor ggml converts the vector to the format its kernel uses for that weight type before the
  product, so the returned number is the sum over the vector as that format holds it.
- FR-003 (US-001) WHEN `maths:row(name, id)` is called on a two-dimensional f32, q8_0, q4_K or q5_K
  tensor with `id` counting from zero and less than its second dimension, THE SYSTEM SHALL return
  the `id`-th row's decoded numbers in order.
- FR-004 (US-002) WHEN `maths:norm(vector, gain, eps)` is called with a non-empty vector and with
  `gain` naming a one-dimensional f32 tensor of the vector's length, THE SYSTEM SHALL return, for
  each position, the gain's number times the vector's number divided by the square root of (the
  mean of the vector's squared numbers plus `eps`).
- FR-005 (US-002) WHEN `maths:rope(vector, position, dimensions, base)` is called with an even
  `dimensions` of at least two and no larger than the vector's length, a `position` of zero or more,
  and a positive `base`, THE SYSTEM SHALL return the vector with, for each `i` from zero to
  `dimensions` divided by two minus one, the number at `i` replaced by
  `x[i]*cos(theta) - x[i + dimensions/2]*sin(theta)` and the number at `i + dimensions/2` replaced
  by `x[i]*sin(theta) + x[i + dimensions/2]*cos(theta)`, where `theta` is `position` times `base`
  to the power of `-2*i/dimensions`, and every other number unchanged.
- FR-006 (US-002) WHEN `maths:softmax(vector)` is called with a non-empty vector, THE SYSTEM SHALL
  return one number per position, the exponential of the vector's number minus the vector's largest
  number divided by the sum of those exponentials.
- FR-007 (US-002) WHEN `maths:add(left, right)` is called with two non-empty lists of the same
  length, THE SYSTEM SHALL return the list holding, at each position, the sum of the two lists'
  numbers.
- FR-008 (US-002) WHEN `maths:swiglu(gate, up)` is called with two non-empty lists of the same
  length, THE SYSTEM SHALL return the list holding, at each position, the `up` number times the
  `gate` number divided by one plus the exponential of the negated `gate` number.
- FR-009 (US-002) WHEN `maths:sigmoid_gate(gate, values)` is called with two non-empty lists of the
  same length, THE SYSTEM SHALL return the list holding, at each position, the `values` number
  divided by one plus the exponential of the negated `gate` number.
- FR-010 (US-002) THE SYSTEM SHALL compute every number `matvec`, `row`, `norm`, `rope`, `softmax`,
  `add`, `swiglu` and `sigmoid_gate` return by calling ggml's own operation for that maths in a
  per-call graph, and return it as a single precision number, so that the accumulation inside each
  operation is ggml's own and not a second implementation.
- FR-011 (US-003) IF an operation names a tensor the handle does not hold, or names a tensor whose
  type the operation does not read, THEN THE SYSTEM SHALL return nil and one line holding the name
  and, when the type is at fault, that type's name.
- FR-012 (US-003) IF `matvec`'s vector length is not the tensor's first dimension, or a
  three-dimensional tensor is given no index or an index past its third dimension, or a tensor that
  is not three-dimensional is given an index, or `matvec` is given a tensor with more than three
  dimensions, or `row` is given a tensor that is not two-dimensional or an `id` past its second
  dimension, or `row` is given a negative `id`, or `matvec` is given a negative index, or `norm` is
  given an empty vector, an `eps` less than zero, a gain that does not hold one number per position
  of the vector, or a gain that is not one-dimensional, or `softmax` is given an empty list, or
  `add`, `swiglu` or `sigmoid_gate` is given lists of different lengths, or `rope` is given odd
  `dimensions`, `dimensions` less than two, `dimensions` larger than the vector, a negative
  `position`, or a `base` that is not positive, THEN THE SYSTEM SHALL return nil and one line naming
  the operation and the numbers at fault.
- FR-013 (US-003) IF the ggml context `attach` needs cannot be created, THEN THE SYSTEM SHALL return
  nil and one line holding `context`.
  Not testable: making that allocation fail needs a memory limit the suite cannot set portably; the
  branch returns nil and one line, which a reader confirms by inspection.
- FR-014 (US-003) IF more than one refusal applies to a call, THEN THE SYSTEM SHALL report a tensor
  the handle does not hold before a tensor of a type the operation does not read, and both before a
  shape or number that does not fit.
- FR-015 (US-003) IF `compute.attach` is given a backend that `compute.load` did not return, THEN THE
  SYSTEM SHALL return nil and one line naming `attach`.
- FR-016 (US-001) IF a tensor an operation asks its graph's context for by shape comes back NULL,
  THEN THE SYSTEM SHALL stop that operation's build at that tensor and return nil and one line
  naming the operation and the number of numbers the tensor asked for, so no later call in the same
  build reads through the NULL.
  Not testable: reaching the branch needs an operation whose constant is wrong, which the suite
  would have to ship; a reader confirms the raise in `graph_tensor` and the `pcall` around the build
  in `gathered` in `src/compute.tl`.
- FR-017 (US-001) WHEN a maths handle's `close` is called, THE SYSTEM SHALL give back its hold on the
  weights handle, so that a handle whose last maths handle is closed can be released, and a second
  `close` SHALL return without doing it again; IF an operation is called on a closed handle, THEN
  THE SYSTEM SHALL return nil and one line naming the operation and saying the handle is closed,
  so a closed handle can never read a released mapping.

## Non-goals

- No attention, KV cache, forward pass or sampler: one token through the layers is the next spec,
  and no layer name, head count or model family is known here.
- No SSM operations, no M-RoPE section positions and no vision: conv1d, softplus, the delta-net
  scan, and positions that differ between rope sections are later specs.
- No other type and no prefill batch: weights are f32, q8_0, q4_K or q5_K and every operation takes
  one vector. A maths handle reads the weights handle's mapping and `close` is the only lifetime
  step it takes; `06` owns what `release` refuses while a handle is still held.

## Tests

- T-001 (FR-001, FR-002, FR-003): `mapped tensors come back by name through ggml`, proves with a
  fixture holding an f32 tensor `plain` of dimensions 3 and 2 with values 1 to 6, an f32 tensor
  `bias` of dimension 3 with 2, 3 and 4, an f32 tensor `expert` of dimensions 2, 2 and 2 with 1 to
  8, an f32 tensor `deep` of dimensions 2, 2, 2 and 2 with 1 to 16, an f32 tensor `scalar` with no
  dimensions holding 7, an f16 tensor `half` of dimensions 2 and 2 with 1 to 4, a q8_0 tensor
  `embd` of dimensions 32 and 2 whose blocks have scale 1 and quants 1 to 32 and 33 to 64, a q4_K
  tensor `quart` of dimensions 256 and 2 whose blocks have scale 1, minimum 0, six-bit scale 1 and
  minimum 0 in every group and every nibble 1, and a q5_K tensor `five` of dimensions 256 and 2
  whose blocks have scale 1, minimum 0, six-bit scale 1 and minimum 0 in every group, every nibble
  1 and every high bit 0, that
  `compute.attach(backend, handle)` returns nil as its second value, `matvec("plain", {1,1,1})`
  returns `{6, 15}`, `matvec("bias", {1,1,1})` returns `{9}`, `matvec("scalar", {1})` returns
  `{7}`, `matvec("expert", {1,1}, 0)` returns `{3, 7}`, `matvec("expert", {1,1}, 1)` returns
  `{11, 15}`, `row("plain", 1)` returns `{4, 5, 6}`, `row("embd", 1)` returns 33 to 64,
  `matvec("embd", 32 numbers of 127)` returns `{67056, 197104}`, `row("quart", 0)` returns 256
  ones, `matvec("quart", 256 ones)` returns `{256, 256}`, `row("five", 0)` returns 256 ones, and
  `matvec("five", 256 ones)` returns `{256, 256}`.
- T-002 (FR-004, FR-005, FR-010): `norm and rope land on ggml's numbers`, proves with a fixture
  tensor `gain` of dimension 2 holding 1 and 2 that `norm({3,4}, "gain", 0)` returns
  `{0.8485281467437744, 2.262741804122925}`, that `norm({3,4}, "gain", 2.5)` returns
  `{0.7745966911315918, 2.0655910968780518}`, and that `rope({1,2,3,4,5,6}, 1, 4, 10000)` returns
  `{-1.9841105937957764, 1.9599006175994873, 2.4623777866363525, 4.019799709320068, 5, 6}`.
- T-003 (FR-006, FR-007, FR-008, FR-009, FR-010): `softmax, add and the gated maths land on ggml's
  numbers`, proves that `softmax({1,2,3})` returns
  `{0.09003057330846786, 0.2447284758090973, 0.6652409434318542}`, that `softmax({1000,1000})`
  returns `{0.5, 0.5}`, that `add({1,2}, {3,4})` returns `{4, 6}`, that `swiglu({0,1}, {2,3})`
  returns `{0, 2.193175792694092}`, and that `sigmoid_gate({0,1}, {2,4})` returns
  `{1, 2.924234390258789}`.
- T-004 (FR-011, FR-012, FR-014): `a name, type or shape that does not fit is refused with one
  line`, proves that `matvec("missing", {1})`, `matvec("missing", {1,1}, -1)` and
  `row("missing", 0)` return nil and a line holding `missing`, that `matvec("half", {1,1})`,
  `row("half", 0)` and `norm({1,2}, "half", 0)` on the fixture's two-dimensional f16 tensor `half`
  return nil and a line holding `half` and `f16`, that each of `matvec("plain", {1,1})`,
  `matvec("expert", {1,1})`, `matvec("expert", {1,1}, 2)`, `matvec("expert", {1,1}, -1)`,
  `matvec("plain", {1,1}, 0)`, `matvec("deep", {1,1})`, `row("bias", 0)`, `row("plain", 2)`,
  `row("plain", -1)`, `norm({1,2,3}, "gain", 0)`, `norm({1,2}, "plain", 0)`, `norm({}, "gain", 0)`,
  `norm({1,2}, "gain", -1)`, `softmax({})`, `add({1}, {1,2})`, `swiglu({1}, {1,2})`,
  `sigmoid_gate({1}, {1,2})`, `rope({1,2}, 0, 3, 10000)`, `rope({1,2}, 0, 4, 10000)`,
  `rope({1,2,3,4}, -1, 4, 10000)`, `rope({1,2,3,4}, 0, 0, 10000)` and
  `rope({1,2,3,4}, 0, 4, 0)` returns nil and one line with no newline, and that the line
  `add({1}, {1,2})` returns holds `add`, `1` and `2`.
- T-006 (FR-017): `releasing the model gives the device memory back`, proves that a weights handle
  with a maths handle over it refuses `release` with one line while the maths handle lives, that
  the same `release` after `close` returns nil and gives the device memory back, and that a second
  `close` changes nothing.
- T-005 (FR-015): `a backend compute.load did not return is refused`, proves that
  `compute.attach` given a backend built by hand returns nil and one line holding `attach`.

## Open questions

- None.

## Amendments

- None.
