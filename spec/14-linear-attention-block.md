# 14: The linear-attention block

**Status**: done   **Depends on**: 13

## Problem & context

`12` gives one block's attention half for the 10 full-attention blocks of the local
file, and `13` gives every block's feed-forward tail. The other 30 blocks of the local
model, every layer whose index counts from zero where `(index + 1) % 4` is not zero,
do not use a growing cache at all. They use a gated delta net: one causal convolution
over the projections, a per-value-head state that is decayed and corrected once per
token, and a gated norm before the output projection. Nothing of that exists, so 30 of
the 40 blocks the main pass runs cannot produce a number.

The local Qwen3.6 file states the numbers and names this must carry under `blk.N.`:
`attn_norm.weight` (2048, f32); the fused `attn_qkv.weight` (2048, 8192, q8_0) whose
8192 output numbers are 16 query heads of 128 numbers, then 16 key heads of 128
numbers, then 32 value heads of 128 numbers; `attn_gate.weight` (2048, 4096,
q8_0) as 32 value heads of 128 gate numbers; `ssm_conv1d.weight` (4, 8192, f32);
`ssm_alpha.weight` and `ssm_beta.weight` (2048, 32, f32); `ssm_a` (32, f32);
`ssm_dt.bias` (32, f32); `ssm_norm.weight` (128, f32); and `ssm_out.weight` (4096,
2048, q8_0). Its metadata gives `ssm.conv_kernel` 4, `ssm.state_size` 128,
`ssm.group_count` 16, `ssm.time_step_rank` 32 and `ssm.inner_size` 4096, and no
attention scale for this half.

llama.cpp builds such a block in `build_layer_attn_linear`
(`src/models/qwen35moe.cpp`, commit
`ccd360f060a0f17b87ee0d4c5d91ee84cfdff484`, raw.githubusercontent.com/ggml-org/llama.cpp)
and in `build_conv_state`, `build_delta_net` and `build_recurrent_attn`
(`src/models/delta-net-base.cpp`) of the same commit. It normalises the block input
with `attn_norm`, projects it once for query, key and value and once for the gate,
projects `alpha` and `beta`, adds `ssm_dt.bias` to `alpha`, applies softplus, and
multiplies by the file's `ssm_a`, which already holds `-exp(A_log)`; the result is the
per-value-head exponent the state is decayed by. The convolution reads the last
`conv_kernel - 1` projected vectors kept in the recurrent cache plus the current one,
each tap multiplying its own position (`ggml_ssm_conv`, kernel index zero reading the
oldest position), and silu follows. Query and key heads are L2 normalised
(`build_gdn_l2_norm` in `src/models/models.h` emits a root mean square norm with the
file's epsilon divided by the head length, then a scale of one over the square root of
the head length, which together divide each head by the square root of the sum of its
squares plus the epsilon). The fused op `ggml_gated_delta_net`
(`ggml/src/ggml-metal/kernels/gated_delta_net.metal`) decays a value head's state,
subtracts the state read through the key from the value, scales by the sigmoid of
`beta`, adds the key times that correction to the state, and reads the state through
the query, scaled by one over the square root of the key length. A value head reads
the key head whose index is its own modulo the key head count (`i21 % args.ne01` in
that kernel). The result is normalised per value head with `ssm_norm` as gain, gated by
silu of the gate numbers, and projected by `ssm_out`.

That fused op requires the key and value lengths to be equal and scales its scan by the
value length, so the block does not call it: it composes the same steps out of ggml's
matrix product, scale, subtract, add, `ggml_softplus`, `ggml_ssm_conv`, `ggml_rms_norm`,
`ggml_silu` and `ggml_exp`, which keeps a key length and a value length that differ
apart and scales by the key length as FR-007 says. Every number the block returns is
still ggml's own arithmetic over the mapped weights.

This spec adds `compute.linear_attention(backend, handle, geometry)` and one record,
`LinearAttention`, with one operation, `block`. No existing requirement or fixture
changes: the five numbers this needs are added to `12`'s `Geometry` as optional fields,
so the fixtures of `12` and `13`, which build a geometry without them, still hold. A
block call takes one token's hidden numbers, the block's tensor prefix and the position
the token occupies, and returns what the block's attention half adds to those numbers;
the residual add is the caller's, exactly as `12` returns it. Each prefix's
convolution history and each value head's state stay in the handle, so a later call at
the next position sees every token that came before it. The state is the one thing that
does not grow: one key-length by value-length matrix for each value head, plus
`conv_kernel - 1` projected vectors, per prefix.

The block belongs in the core for the reason `12` gives: it composes `11`'s operations
over the same ggml context, and a plugin has no way to reach that context. The handle
reads tensors by name at call time, so building one over the 23 GB model reads no
weights; state is allocated on the first block call that names its prefix.

Refusals keep the shape `11`, `12` and `13` use: nil and one line, so a mistake in the
next spec's wiring is a return value and the process stays up.

```teal
local weights = require("tllm.weights")

local record compute
  record Geometry
    heads: integer
    kv_heads: integer
    head_length: integer
    rope_dimensions: integer
    rope_base: number
    eps: number
    positions: integer
    experts_used: integer
    linear_key_heads: integer | nil
    linear_key_length: integer | nil
    linear_value_heads: integer | nil
    linear_value_length: integer | nil
    linear_conv_kernel: integer | nil
  end

  record LinearAttention
    block: function(
      LinearAttention, {Row}, string
    ): {{number}} | nil, string | nil
    release: function(LinearAttention, string)
    close: function(LinearAttention)
  end

  linear_attention: function(
    Backend, weights.Handle, Geometry
  ): LinearAttention | nil, string | nil
end
```

A `LinearAttention` handle's `close` frees the histories and states it holds and gives back its hold
on the weights mapping, refusing every operation afterwards; `12/FR-014` states that contract and
the refusal line.

## User stories

- US-001 (P1) As the engine author, I want one linear-attention block for a token computed by ggml over the mapped weights, so that the 30 linear blocks of the local model need no maths of my own.
- US-002 (P1) As the engine author, I want each block's convolution history and delta state kept per prefix, so that the next token sees everything before it.
- US-003 (P2) As the engine author, I want a name, type, shape or position that does not fit back as one line, so that a wiring mistake is a return value and not a crash.

## Functional requirements

- FR-001 (US-001) WHEN `compute.linear_attention(backend, handle, geometry)` is called
  with a backend `compute.load` returned and a geometry holding `linear_key_heads`,
  `linear_key_length`, `linear_value_heads`, `linear_value_length` and
  `linear_conv_kernel` each of one or more, a `linear_value_heads` that is a multiple of
  `linear_key_heads`, and an `eps` of zero or more, THE SYSTEM SHALL return a
  linear-attention handle, which reads the tensors of that weights handle by name at
  call time, and nil as its second value. Its tests give the geometry `cache_type`
  `f32`, so their numbers stand.
- FR-002 (US-001) WHEN that handle's `block(rows, prefix)` is called with a batch of
  `compute.Row`s, each row's `hidden` holding one number per position of
  `<prefix>attn_norm.weight`, THE SYSTEM SHALL return one contribution row per input row in
  order, each holding one number per position of that row's `hidden`, being the numbers FR-003 to
  FR-008 describe for that row, and nil as its second value.
- FR-003 (US-001) WHEN a block call begins, THE SYSTEM SHALL normalise `hidden` with
  `<prefix>attn_norm.weight` by `11/FR-004`'s norm and the geometry's `eps`, take the
  projected numbers as one number per row of `<prefix>attn_qkv.weight` in row order, the
  projected gate numbers as one number per row of `<prefix>attn_gate.weight` in row
  order, and read `<prefix>ssm_beta.weight` and `<prefix>ssm_alpha.weight` as one group
  of `hidden`-length numbers per row.
- FR-004 (US-001) WHEN the normalised hidden is ready, THE SYSTEM SHALL take each value
  head's beta number as the sum over the positions of the normalised hidden of
  `<prefix>ssm_beta.weight`'s number in that head's row times the normalised hidden's
  number, put through one over one plus the exponential of its negation, and that
  head's decay number as `<prefix>ssm_a`'s number for the head times the logarithm of
  one plus the exponential of the sum of `<prefix>ssm_dt.bias`'s number for the head and
  the sum over the positions of the normalised hidden of
  `<prefix>ssm_alpha.weight`'s number in that head's row times the normalised hidden's
  number.
- FR-005 (US-001) WHEN the projected numbers are ready, THE SYSTEM SHALL take the
  convolved numbers as, for each channel `c`, the sum over the taps `t` from zero to
  `linear_conv_kernel - 1` of `<prefix>ssm_conv1d.weight`'s number in row `c` at tap
  `t` times the projected number in channel `c` at the position
  `position - linear_conv_kernel + 1 + t`, where the projected number at a position
  below zero is zero, and then multiply each convolved number by one over one plus the
  exponential of its negation.
- FR-006 (US-001) WHEN the convolved numbers are ready, THE SYSTEM SHALL take the query
  heads as the first `linear_key_heads * linear_key_length` convolved numbers grouped
  into `linear_key_length` numbers per head in head order, the key heads as the next
  `linear_key_heads * linear_key_length` convolved numbers grouped the same way, the
  value heads as the last `linear_value_heads * linear_value_length` convolved numbers
  grouped into `linear_value_length` numbers per head in head order, divide each query
  head's numbers and each key head's numbers by the square root of the sum of their
  squares plus the geometry's `eps`, and take each value head's query head and key head
  as the head whose index is the value head's index modulo `linear_key_heads`.
- FR-007 (US-001) WHEN the query, key and value heads are ready, THE SYSTEM SHALL take,
  for each value head, that head's state as a `linear_key_length` by
  `linear_value_length` group of numbers kept under `prefix`, multiply every state
  number by the exponential of that head's decay number from FR-004, take the key
  numbers as, for each value position, the sum over key positions of the state number at
  that key position and value position times the key head's number at the key position,
  take the correction number at a value position as that head's beta number times (the
  value number at that value position less the key number at that value position), add
  to each state number at a key position and a value position the key head's number at
  that key position times the correction number at that value position, and take the
  head's scan numbers from the state so updated as one over the square root of
  `linear_key_length` times, for each value position, the sum over key positions of the
  state number at that key position and value position times the query head's number at
  the key position.
- FR-008 (US-001) WHEN a value head's scan numbers are ready, THE SYSTEM SHALL take the
  head's normed numbers as `11/FR-004`'s norm of the scan numbers with
  `<prefix>ssm_norm.weight` as gain and the geometry's `eps`, multiply each by the
  projected gate number at the same place in that head's group of FR-003's gate numbers
  divided by one plus the exponential of its negation, and apply
  `<prefix>ssm_out.weight` to the value heads' results in head order by `11/FR-002`'s
  matrix product.
- FR-009 (US-001) THE SYSTEM SHALL compute the numbers FR-003 to FR-008 describe by
  calling ggml's own operations over the mapped weights, so that the block's softplus,
  convolution, decay, state update, norms, gating and projection are ggml's arithmetic
  and not a second implementation of them.
  Not testable: the fixtures' heads hold two numbers each, and over so few numbers any
  accumulation order returns the same numbers to within the tests' tolerance, so what
  proves this is a reader finding the ggml calls the block makes.
- FR-010 (US-002) WHEN a block call is accepted, THE SYSTEM SHALL keep under a sequence and
  `prefix` the projected numbers in channel `c` at the positions
  `position - linear_conv_kernel + 2` to `position` as that sequence and prefix's convolution
  history, and the state numbers FR-007 gives as that sequence and prefix's state, one history
  and one state for each sequence and prefix, each held as single precision numbers, so that a
  call at the next position reads them and no other sequence's numbers. A sequence and prefix's
  history and state start as zeros, and a refused block call stores nothing: every refusal of
  FR-012 to FR-015 happens before those numbers are written. The history and state are kept under
  a sequence and a prefix, and are freed with that sequence as `26/FR-005` says.
- FR-017 (US-002) WHEN a block call builds its graph, THE SYSTEM SHALL build it in the one arena the
  model that built the handle holds, sized by `32/FR-005` from the largest graph a pass builds at
  the model's `positions`, and, for a handle built on its own, in one arena of its own sized by the
  same rule at the geometry's `positions`, so that the state the call reads, the state it decays,
  the product that corrects it and the state it writes, the convolution window, the heads' vectors
  and the one-row tensors between them all fit that arena and no arena is allocated per call.
- FR-011 (US-003) IF `compute.linear_attention` is given a backend that `compute.load`
  did not return, or a `geometry` missing any of the five numbers FR-001 names or
  holding `linear_key_heads`, `linear_key_length`, `linear_value_heads`,
  `linear_value_length` or `linear_conv_kernel` below one, or holding a
  `linear_value_heads` that is not a multiple of `linear_key_heads`, or holding an
  `eps` below zero, THEN THE SYSTEM SHALL return nil and one line naming
  `linear_attention` and, the geometry being at fault, the first of those names that is
  missing or at fault in the order `linear_key_heads`, `linear_key_length`,
  `linear_value_heads`, `linear_value_length`, `linear_conv_kernel`, `eps`.
- FR-012 (US-003) IF a tensor `block` reads is not in the handle, or is a tensor whose
  type is not `f32` for `attn_norm.weight`, `ssm_beta.weight`, `ssm_alpha.weight`,
  `ssm_conv1d.weight`, `ssm_a`, `ssm_dt.bias` or `ssm_norm.weight`, or is not `f32`,
  `q8_0`, `q4_K` or `q5_K` for `attn_qkv.weight`, `attn_gate.weight` or
  `ssm_out.weight`, THEN THE SYSTEM SHALL return nil and one line naming `block`, the
  first tensor in this order that the handle does not hold and, no tensor being missing,
  the first whose type is at fault together with that type's name: `attn_norm.weight`,
  `attn_qkv.weight`, `attn_gate.weight`, `ssm_beta.weight`, `ssm_alpha.weight`,
  `ssm_conv1d.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_norm.weight`, `ssm_out.weight`.
- FR-013 (US-003) IF a tensor's shape does not fit a row of `hidden` or the other tensors, that is
  if `attn_norm.weight` is not one-dimensional or does not hold one number per position
  of `hidden`; `attn_qkv.weight` is not two-dimensional, does not hold one number per
  position of `hidden` per row, or does not hold
  `2 * linear_key_heads * linear_key_length + linear_value_heads * linear_value_length`
  rows; `attn_gate.weight` is not two-dimensional, does not hold one number per position
  of `hidden` per row, or does not hold `linear_value_heads * linear_value_length` rows;
  `ssm_beta.weight` or `ssm_alpha.weight` is not two-dimensional, does not hold one
  number per position of `hidden` per row, or does not hold `linear_value_heads` rows;
  `ssm_conv1d.weight` is not two-dimensional, does not hold `linear_conv_kernel` numbers
  per row, or does not hold
  `2 * linear_key_heads * linear_key_length + linear_value_heads * linear_value_length`
  rows; `ssm_a` or `ssm_dt.bias` is not one-dimensional or does not hold
  `linear_value_heads` numbers; `ssm_norm.weight` is not one-dimensional or does not
  hold `linear_value_length` numbers; or `ssm_out.weight` is not two-dimensional, does
  not hold `linear_value_heads * linear_value_length` numbers per row, or does not hold
  one row per position of `hidden`, THEN THE SYSTEM SHALL return nil and one line naming
  `block`, the tensor and the numbers at fault. The shape checks are made against each row of the
  batch's hidden.
- FR-014 (US-003) IF a row's `position` is negative, or is not the number of positions that
  row's sequence and prefix have already taken, a sequence and prefix that have taken no position
  taking zero next, THEN THE SYSTEM SHALL return nil and one line naming `block` and that row's
  sequence and position.
- FR-015 (US-003) IF the memory a prefix's history or state or a block call's graph
  needs cannot be allocated THEN THE SYSTEM SHALL return nil and one line naming
  `block`.
  Not testable: making that allocation fail needs a memory limit the suite cannot set
  portably; the branch returns nil and one line, which a reader confirms by inspection.
- FR-016 (US-003) IF more than one refusal applies to a block call, THEN THE SYSTEM
  SHALL report a missing tensor before a tensor of a type `block` does not read, a type
  fault before a shape that does not fit, and of two faults in FR-013's list the earlier
  one, and a shape fault before a position that does not fit.

## Non-goals

- No mixed stack, no embedding lookup, no 40-block pass, no sampling, no prompt and no
  serving: spec 15 runs the whole model and serves it.
- No change to full attention or to the feed-forward tail: `12` and `13` stay as they are,
  and a stack that mixes both block kinds is spec 15's.
- No batch, no M-RoPE positions, no vision, no expert cache and no reuse or release of a
  state: one token per call, and a `LinearAttention` handle keeps its states for its
  life.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007, FR-008): `one linear
  block at position zero gives its output`, proves with a fixture holding, under
  `blk.0.` and `blk.1.`, the same f32 tensors: `attn_norm.weight` 2 numbers of 1;
  `attn_qkv.weight` 2 numbers per row over 12 rows, the rows counting from one, rows 1
  and 2 `{2,0}` and `{0,0}`, rows 3 and 4 `{0,0}` and `{1,3}`, rows 5 and 6 `{4,0}` and
  `{0,0}`, rows 7 and 8 `{0,0}` and `{1,5}`, rows 9 and 10 `{100,0}` and `{0,100}`, rows
  11 and 12 `{0,0}` and `{0,200}`; `attn_gate.weight` 2 numbers per row over 4 rows, each
  `{100,0}`; `ssm_beta.weight` 2 numbers per row over 2 rows, each `{ln 3, 0}`;
  `ssm_alpha.weight` 2 numbers per row over 2 rows, each `{1, 0}`; `ssm_dt.bias` 2
  numbers `{ln 3, 0}`; `ssm_a` 2 numbers of -1; `ssm_conv1d.weight` 2 numbers per row
  over 12 rows, each `{0.5, 1}`; `ssm_norm.weight` 2 numbers of 1;
  `ssm_out.weight` 4 numbers per row over 2 rows, rows `{1,0,1,0}` and `{0,1,0,1}`, and
  with a geometry of 1 head, 1 key/value head, heads of 2 numbers, rope over 2 numbers
  at base 10000, `eps` 0, 4 positions, 1 expert used, 2 linear key heads of 2 numbers, 2
  linear value heads of 2 numbers and a convolution kernel of 2, that
  `compute.linear_attention(backend, handle, geometry)` returns nil as its second value,
  that `block({1,1}, "blk.0.", 0)` returns `{100, 241.4213562}` to a relative 1e-5, the 100
  wide gate saturating `11/FR-008`'s silu at 100 times the gated norm, the 100 wide
  values saturating silu at their convolved numbers, and the second output adding the
  second head's gated 141.4213562 to the first head's 100, that a second handle's
  `block({3,3}, "blk.0.", 0)` returns `{100, 241.4213562}` too, the norm dividing the
  hidden by its root mean square, and that on a second handle whose geometry holds
  `eps` 1, `block({1,1}, "blk.0.", 0)` returns `{70.6498381, 170.6241588}` to within
  1e-5, the epsilon reaching the hidden norm, both L2 norms and the gated norm.
- T-002 (FR-004, FR-005, FR-007, FR-010): `a second token reads the conv history and
  the state`, proves on T-001's fixture that after `block({1,1}, "blk.0.", 0)`,
  `block({1,0}, "blk.0.", 1)` returns `{193.2903421, 251.3696764}` to a relative 1e-5, the
  second token's projected numbers
  `{2.8284271,0,0,1.4142136,5.6568542,0,0,1.4142136,141.4213562,0,0,0}` and the first
  token's `{2,0,0,4,4,0,0,6,100,100,0,200}` under the `{0.5, 1}` taps and silued to
  `{3.7469541,0,0,3.3054565,7.6532359,0,0,4.3614226,191.4213562,50,0,100}`, the second
  token's L2 normalised query and key heads `{1,0}` and `{0,1}`, the second token's
  beta number 0.8254435, the first token's state, decayed at this token by 0.0749639 at
  the first head and 0.1955703 at the second, read as `{5.6222938,5.6222938}` and
  `{0,29.3355476}` through the keys and giving `{158.9889236, 42.2535833}` and
  `{0,87.6650611}` before the 1/√2 scale, and that
  `block({1,1}, "blk.1.", 0)` on that same handle returns `{100, 241.4213562}` to within
  1e-5, `blk.1.`'s history and state being untouched by `blk.0.`'s two calls.
- T-003 (FR-010, FR-011, FR-012, FR-013, FR-014, FR-016): `a name, type, shape or
  position that does not fit is refused with one line`, proves on T-001's fixture that a
  backend built by hand is refused with one line holding `linear_attention`; that a
  geometry without `linear_conv_kernel` is refused with one line holding
  `linear_attention` and `linear_conv_kernel`; that geometries holding
  `linear_key_heads` 0, `linear_key_length` 0, `linear_value_heads` 0,
  `linear_value_length` 0, a `linear_conv_kernel` of 0, 3 `linear_value_heads` under 2
  key heads and an `eps` of -1 are each refused with one line holding `linear_attention`
  and the number or, for the 3 value heads, `linear_value_heads`; that
  `block({1,1}, "blk.9.", 0)` is refused with one line holding
  `blk.9.attn_norm.weight`; that each of `blk.0.`'s ten tensors removed in turn is
  refused with one line holding its name; that an f16 `attn_qkv.weight`, an f16
  `ssm_beta.weight` and an f16 `ssm_out.weight` are each refused with one line holding
  the tensor's name and `f16`; that `block({1}, "blk.0.", 0)` and
  `block({1,1,1}, "blk.0.", 0)`, an `attn_qkv.weight` with 11 rows, an
  `attn_gate.weight` with 3 rows, an `ssm_beta.weight` with 1 row, an `ssm_alpha.weight`
  with 3 rows, an `ssm_a` with 1 number, an `ssm_dt.bias` with 3 numbers, an
  `ssm_norm.weight` with 3 numbers, an `ssm_conv1d.weight` with 11 rows, an
  `ssm_out.weight` with 3 numbers per row, a one-dimensional `attn_qkv.weight`, a
  three-dimensional `ssm_a`, an `ssm_conv1d.weight` with 3 numbers per row and an
  `ssm_out.weight` with 1 row are each refused with one line holding the tensor at
  fault; that `block({1,1}, "blk.0.", -1)` and, on a fresh handle,
  `block({1,1}, "blk.0.", 2)` are refused with one line holding `blk.0.` and the
  position; that with `ssm_out.weight` removed and an f16 `attn_qkv.weight` the line
  holds `ssm_out.weight`, with an f16 `attn_qkv.weight` and an 11-row `attn_gate.weight`
  the line holds `attn_qkv` and `f16`, with an 11-row `attn_gate.weight` and an 11-row
  `attn_qkv.weight` the line holds `attn_qkv.weight`, and with an 11-row
  `attn_gate.weight` and position 2 the line holds `attn_gate.weight`; and that a
  refused call leaves the history and state where they were: on a fresh handle, after
  `block({1,1}, "blk.0.", 1)` is refused, `block({1,1}, "blk.0.", 0)` returns
  `{100, 241.4213562}` to a relative 1e-5, and after that, `block({1,1}, "blk.0.", 0)` is
  refused and `block({1,0}, "blk.0.", 1)` returns `{193.2903421, 251.3696764}` to a
  relative 1e-5.
- T-004 (FR-006, FR-007, FR-008): `each value head reads the key head it belongs to`,
  proves with a second fixture holding, under `blk.0.`, `attn_norm.weight` 2 numbers of
  1; `attn_qkv.weight` 2 numbers per row over 16 rows, rows 1 and 2 `{1,0}` and `{0,0}`,
  rows 3 and 4 `{1,0}` and `{0,1}`, rows 5 and 6 `{1,0}` and `{0,1}`, rows 7 and 8
  `{0,1}` and `{1,0}`, rows 9 and 10 `{100,0}` and `{0,0}`, rows 11 and 12 `{0,0}` and
  `{0,100}`, rows 13 and 14 `{100,0}` and `{0,100}`, rows 15 and 16 `{100,0}` and
  `{0,50}`; `attn_gate.weight` 2 numbers per row over 8 rows, each `{100,0}`;
  `ssm_beta.weight` and `ssm_alpha.weight` 2 numbers per row over 4 rows of zeros;
  `ssm_dt.bias` 4 zeros; `ssm_a` 4 numbers of -1; `ssm_conv1d.weight` 1 number per row
  over 16 rows, each `{1}`; `ssm_norm.weight` 2 numbers of 1; `ssm_out.weight` 8 numbers
  per row over 2 rows, rows `{1,0,1,0,1,0,1,0}` and `{0,1,0,1,0,1,0,1}`, and with a
  geometry of 1 head, 1 key/value head, heads of 2 numbers, rope over 2 numbers at base
  10000, `eps` 0, 4 positions, 1 expert used, 2 linear key heads of 2 numbers, 4 linear
  value heads of 2 numbers and a convolution kernel of 1, that
  `block({1,1}, "blk.0.", 0)` returns `{367.9124626, 304.6669094}` to a relative 1e-5, that
  `block({1,2}, "blk.0.", 1)` on that handle returns `{209.3171647, 219.4780173}` to a
  relative 1e-5, and that at position 1 value head `h` reading key head `floor(h / 2)`
  gives `{206.6115929, 221.2354228}` instead of what `h % linear_key_heads` gives.
- T-005 (FR-005, FR-010): `a three-tap convolution reads its history in order`, proves
  with T-001's fixture whose `ssm_conv1d.weight` holds 3 numbers per row over 12 rows,
  each `{7, 11, 5}`, under `blk.0.`, and a geometry of 1 head, 1 key/value head, heads of
  2 numbers, rope over 2 numbers at base 10000, `eps` 0, 4 positions, 1 expert used, 2
  linear key heads of 2 numbers, 2 linear value heads of 2 numbers and a convolution
  kernel of 3, that `block({1,1}, "blk.0.", 0)` returns `{100, 241.4213562}` to a
  relative 1e-5 and that `block({1,0}, "blk.0.", 1)` on that handle returns
  `{170.7414208, 304.1506948}` to a relative 1e-5, the second token's window holding a
  zero at tap zero, the first token's projected numbers at tap one and its own at tap two,
  which a history kept in the wrong order does not give.

## Open questions

- None.

## Amendments

- None.
