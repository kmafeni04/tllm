# 12: One block's attention over a growing cache

**Status**: done   **Depends on**: 11

## Problem & context

`11` offers eight operations over the mapped weights, each taking one vector: a matrix product, a
row gather, a norm, rope, softmax, an add and two gated products. None of them is attention, and
nothing keeps a key or a value from the token before, so a second token cannot see the first. A
block of the local model is nine tenths of the way to a token, and the missing tenth is the
attention half of it.

This spec adds `compute.attend(backend, handle, geometry)` and one record, `Attention`, with one
operation, `block`. A block call takes a block's input hidden numbers, the block's tensor prefix
and the position the token occupies, reads that block's seven tensors by name, and returns what the
block's attention half adds to the hidden numbers it was given: the residual add is the caller's.
Each block's rotated keys and values stay in the `Attention` handle under that prefix, so a later
call at the next position attends over every token that came before it. One call is one token at
one block, and a stack of blocks is a loop the caller writes. The cache size is the caller's:
`geometry.positions` is the number of positions a prefix's cache holds, and the local file's
262144 available positions, at 2 by 256 numbers for keys and as many for values, would cost
1.07 GB for each of its 10 attention blocks, so a caller that has not been asked for a long
context should say so. `attend` belongs in the core because it composes `11`'s operations over the
same ggml context and a plugin has no way to reach that context, so a plugin could not carry it
without a core contract it would have to ship first.

The local Qwen3.6 file states the numbers this must carry: 41 blocks, of which the main pass runs
40 (block 40 is the next-token draft block it leaves alone); 10 are full attention, every fourth
from block 3, and the other 30 use a different kind of attention this spec does not touch; 16 query
heads, 2 key and value heads and 256 numbers per head; `rope.dimension_count` 64 and
`rope.freq_base` 10000000; `layer_norm_rms_epsilon` 1e-6. Its attention block holds
`attn_norm.weight` (2048), `attn_q.weight` (2048, 8192) which is 16 heads of 256 query numbers
followed by 256 gate numbers, `attn_k.weight` and `attn_v.weight` (2048, 512), `attn_q_norm.weight`
and `attn_k_norm.weight` (256) and `attn_output.weight` (4096, 2048).

llama.cpp builds such a block in `build_layer_attn`
(`src/models/qwen35moe.cpp`, commit `ccd360f060a0f17b87ee0d4c5d91ee84cfdff484`,
raw.githubusercontent.com/ggml-org/llama.cpp): the single query projection carries each head's
query and its gate side by side, query and key heads are root mean square normalised separately
with the file's epsilon, rope is applied to both, 8 consecutive query heads share each of the 2 key
and value heads (`i02 = i12 / r2`, `ggml/src/ggml-cpu/ggml-cpu.c`), the scores are scaled by one
over the square root of 256 because the file states no attention scale, the gate is sigmoided and
multiplied into the attention numbers
before `attn_output.weight` is applied. Rope is `11/FR-005`'s NEOX rope with one position, which
`11` argued is what one text token needs.

Refusals keep the shape `11` uses: nil and one line, so a mistake in the next spec's wiring is a
return value and the process stays up. A cache allocates on the first block call that names its
prefix, so an `Attention` handle over a 23 GB model allocates nothing until a block is used.

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
  end

  record Attention
    block: function(
      Attention, {number}, string, integer
    ): {number} | nil, string | nil
  end

  attend: function(
    Backend, weights.Handle, Geometry
  ): Attention | nil, string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want one block's attention for a token computed by ggml over the mapped weights, so that a block's attention half needs no maths of my own.
- US-002 (P1) As the engine author, I want each block's keys and values kept from the tokens before it, so that the next token attends over everything it should and nothing it should not see.
- US-003 (P2) As the engine author, I want a name, type, shape or position that does not fit back as one line, so that a wiring mistake is a return value and not a crash.

## Functional requirements

- FR-001 (US-001) WHEN `compute.attend(backend, handle, geometry)` is called with a backend
  `compute.load` returned and `geometry` holding a `heads` of one or more, a `kv_heads` of one or
  more that divides `heads`, a `head_length` of one or more, an even `rope_dimensions` of at least
  two and at most `head_length`, a positive `rope_base`, an `eps` of zero or more and a `positions`
  of one or more, THE SYSTEM SHALL return an attention handle and nil.
- FR-002 (US-001) WHEN that handle's `block(hidden, prefix, position)` is called with `hidden`
  holding one number per position of `<prefix>attn_norm.weight`, THE SYSTEM SHALL return one number
  per position of `hidden`, being the gated attention numbers FR-003 to FR-007 describe, and nil
  as its second value.
- FR-003 (US-001) WHEN a block call begins, THE SYSTEM SHALL normalise `hidden` with
  `<prefix>attn_norm.weight` by `11/FR-004`'s norm and the geometry's `eps`, and read
  `<prefix>attn_q.weight`'s rows as one group per head in head order, the head's `head_length`
  query numbers then its `head_length` gate numbers, and `<prefix>attn_k.weight`'s and
  `<prefix>attn_v.weight`'s rows as one group of `head_length` numbers per key/value head.
- FR-004 (US-001) WHEN the query, gate, key and value head numbers are read, THE SYSTEM SHALL
  normalise each query head with `<prefix>attn_q_norm.weight` and each key head with
  `<prefix>attn_k_norm.weight` by `11/FR-004`'s norm and the geometry's `eps`, then rotate each
  query head and each key head by `11/FR-005`'s rope at `position` with the geometry's
  `rope_dimensions` and `rope_base`.
- FR-005 (US-002) WHEN a block call's keys and values are ready, THE SYSTEM SHALL keep the rotated
  key numbers and the value numbers under `prefix` at `position`, in a cache of
  `geometry.positions` positions, one cache for each prefix, each held as a single precision
  number, so that a call for a prefix attends over the positions zero to `position` and over no
  other prefix's positions. A refused block call stores nothing: every refusal of FR-009 to FR-012
  happens before those numbers are written, so a prefix's next position is the same after a refused
  call as before it.
- FR-006 (US-001) WHEN the cached positions of a prefix are known, THE SYSTEM SHALL take a query
  head's weight for a position as `11/FR-006`'s softmax, over the positions zero to `position`, of
  that query head's sum of products with that position's key head numbers times one over the
  square root of `head_length`, the key/value head being `floor(head / (heads / kv_heads))`, and
  the head's attention numbers as those weights applied to that key/value head's value numbers.
- FR-007 (US-001) WHEN the attention numbers of a query head are ready, THE SYSTEM SHALL multiply
  each by one over one plus the exponential of the negated gate number at the same place in that
  head's gate group, and apply `<prefix>attn_output.weight` to the heads' results in head order by
  `11/FR-002`'s matrix product.
- FR-008 (US-003) IF `compute.attend` is given a backend that `compute.load` did not return, or a
  `geometry` holding a number FR-001 does not admit, THEN THE SYSTEM SHALL return nil and one line
  naming `attend` and the number at fault.
- FR-009 (US-003) IF a tensor `block` reads is not in the handle, or is a tensor whose type is
  not `f32`, `q8_0`, `q4_K` or `q5_K`, THEN THE SYSTEM SHALL return nil and one line naming
  `block`, the first tensor in the order FR-003 to FR-007 read them that the handle does not hold
  and, no tensor being missing, the first whose type is at fault together with that type's name.
- FR-010 (US-003) IF a tensor's shape does not fit the geometry, that is if `attn_q.weight` does
  not hold `heads * head_length * 2` rows, `attn_k.weight` or `attn_v.weight` does not hold
  `kv_heads * head_length` rows, `attn_output.weight` does not hold `heads * head_length` numbers
  per row and one row per position of `hidden`, `attn_q_norm.weight` or `attn_k_norm.weight` does
  not hold `head_length` numbers, `hidden` does not hold the numbers `attn_norm.weight` holds, or
  any of the seven tensors holds a number of dimensions other than the one its role needs, THEN THE
  SYSTEM SHALL return nil and one line naming `block`, the tensor and the numbers at fault.
- FR-011 (US-003) IF `position` is negative, or is `geometry.positions` or more, or is not the one
  after the last position that prefix's cache took, where a prefix holding no positions takes
  position zero next, THEN THE SYSTEM SHALL return nil and one line naming `block`, the prefix and
  the position.
- FR-012 (US-003) IF the memory a prefix's cache or a block's graph needs cannot be allocated THEN
  THE SYSTEM SHALL return nil and one line naming `block`.
  Not testable: making that allocation fail needs a memory limit the suite cannot set portably; the
  branch returns nil and one line, which a reader confirms by inspection.
- FR-013 (US-003) IF more than one refusal applies to a block call, THEN THE SYSTEM SHALL report a
  missing tensor before a tensor of a type `block` does not read, a type fault before a shape that
  does not fit, and a shape fault before a position that does not fit.

## Non-goals

- No feed-forward, no router, no residual, no vocabulary scores and no sampling: this is a block's
  attention half, returning what the caller adds to the hidden numbers it passed in.
- No other kind of block: the local file's 30 linear-attention blocks (conv1d, softplus, the
  delta-net scan) and M-RoPE positions that differ between sections are later specs.
- No batch, no metadata and no route: one token per call, the geometry is the caller's, and reading
  the file's `qwen35moe.*` keys, prefilling a prompt and serving are later specs.
- No release and no reuse of a cache: one `Attention` handle holds its caches for its life, and
  freeing or reassigning them is a later spec's concern.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-007): `one block at position zero gives its output`, proves
  with a fixture holding, under `blk.0.` and `blk.1.`, the same f32 tensors: `attn_norm.weight` 4
  numbers of 1; `attn_q.weight` 8 rows of 4 numbers, the rows counting from one: rows 1, 2 and 5
  being `{1,0,0,0}`, row 6 `{-1,0,0,0}`, rows 3 and 4 zeros and rows 7 and 8 `{-1e9,0,0,0}`;
  `attn_q_norm.weight` and `attn_k_norm.weight` 2 numbers of 1; `attn_k.weight` rows `{1,0,0,0}`
  and `{0,1,0,0}`; `attn_v.weight` rows `{2,0,0,0}` and `{0,4,0,0}`; `attn_output.weight` the 4
  by 4 identity, and with a geometry of 2 heads, 1 key/value head, heads of 2 numbers, rope over 2
  numbers at base 10000, `eps` 0 and 4 positions, that `compute.attend(backend, handle, geometry)`
  returns nil as its second value, and that `block({1,1,1,1}, "blk.0.", 0)` returns `{1,2,0,0}`:
  the one cached
  position's value `{2,4}` under the first head's gate of zero and the second head's gate of
  `-1e9`, and that `block({3,3,3,3}, "blk.0.", 0)` on a second handle returns `{1,2,0,0}` too,
  because the norm divides the hidden numbers by their root mean square.
- T-002 (FR-003, FR-004, FR-005, FR-006, FR-011): `a second token attends over the first`, proves
  on T-001's fixture that after `block({1,1,1,1}, "blk.0.", 0)`,
  `block({1,0,0,0}, "blk.0.", 1)` returns `{1.5587023, 0.8825954, 0, 0}` to within 1e-5, the
  numbers the first token's key `{1,1}` and value `{2,4}`, the second token's rope-rotated query,
  key and value and the `1/sqrt(2)` scale give by hand, that `block({1,0,0,0}, "blk.1.", 0)`
  returns `{2,0,0,0}` on that same handle, blk.1.'s cache being empty while blk.0.'s holds two
  positions, that `block({1,0,0,0}, "blk.0.", 3)` returns nil and one line, position 2 being that
  prefix's next one, and that on a fresh handle whose geometry holds `eps` 1 `block({1,1,1,1},
  "blk.0.", 0)` returns `{0.7071068, 1.4142136, 0, 0}` to within 1e-5, the epsilon reaching the
  hidden norm: the hidden numbers become 0.7071068 each rather than 1, the value becomes
  `{1.4142136, 2.8284271}`, and the first head's gate of zero takes half of that, and then
  `block({1,0,0,0}, "blk.0.", 1)` on that handle returns `{0.8036810, 0.6851070, 0, 0}` to within
  1e-5, the epsilon reaching the two per-head norms as well: a hand computation that leaves `eps`
  out of those two norms gives `{0.8117628, 0.6240896, 0, 0}`, so the asserted numbers are what
  separate them. One `eps` reaches all three norms, and a handle cannot be given one value for the
  hidden norm and another for the per-head norms.
- T-003 (FR-005, FR-008, FR-009, FR-010, FR-011, FR-013): `a name, type, shape or position that
  does not fit is refused with one line`, proves on T-001's fixture that a backend built by hand is
  refused with one line holding `attend`, that a geometry of 0 heads, of 0 key/value heads, of 3
  key/value heads under 2 heads, of 0 head length, of 0, 3, 4 or 5 rope dimensions under 2-number
  heads, of base 0, of `eps` -1 and of 0 positions is each refused with one line holding `attend`,
  that `block({1,1,1,1}, "blk.9.", 0)` is refused with one line holding `blk.9.attn_norm.weight`,
  that each of `blk.0.`'s seven tensors removed in turn is refused with one line holding its name,
  that an f16 `attn_k.weight` is refused with one line holding `attn_k` and `f16`, that
  `block({1,1}, "blk.0.", 0)`, `block({1,1,1}, "blk.0.", 0)`, `attn_q.weight` with 7 rows,
  `attn_k.weight` with 3 rows, `attn_q_norm.weight` with 3 numbers, `attn_output.weight` 3 by 4,
  `attn_q.weight` holding dimensions 4, 8 and 1 and `attn_norm.weight` holding dimensions 4 and 1
  are each refused with one line holding the tensor or the number at fault, that
  `block({1,1,1,1}, "blk.0.", -1)` and `block({1,1,1,1}, "blk.0.", 4)` are refused with one line
  holding `blk.0.` and the position, that with `attn_k.weight` missing and an f16 `attn_q.weight`
  the line holds `attn_k.weight`, with an f16 `attn_k.weight` and a 7-row `attn_q.weight` the line
  holds `attn_k` and `f16`, and with a 7-row `attn_q.weight` and position 5 the line holds
  `attn_q.weight`, and that a refused call leaves the cache where it was: after
  `block({1,1}, "blk.0.", 0)` and `block({1,1,1,1}, "blk.0.", 1)` are refused on one fresh handle,
  `block({1,1,1,1}, "blk.0.", 0)` on that handle returns `{1,2,0,0}`.
- T-004 (FR-003, FR-006, FR-007): `each query head reads the key and value head it belongs to`,
  proves with a second fixture holding, under `blk.0.`, `attn_norm.weight` 4 numbers of 1;
  `attn_q.weight` 16 rows of 4 numbers,
  rows 1, 2, 9 and 10 being `{1,0,0,0}`, row 5 `{1,0,0,0}`, row 6 `{-1,0,0,0}`, row 13
  `{1,0,0,0}`, row 14 `{-1,0,0,0}` and rows 3, 4, 7, 8, 11, 12, 15, 16 zeros; `attn_q_norm.weight`
  and `attn_k_norm.weight` 2 numbers of 1; `attn_k.weight` 4 rows of 4 numbers, rows 1 and 2
  `{1,0,0,0}` and `{0,1,0,0}`, rows 3 and 4 `{1,0,0,0}` and `{1,0,0,0}`; `attn_v.weight` 4 rows of
  4 numbers, rows 1 and 2 `{2,0,0,0}` and `{0,4,0,0}`, rows 3 and 4 `{6,0,0,0}` and `{0,8,0,0}`;
  `attn_output.weight` 8 numbers per row over 4 rows, row 1 `{1,0,0,0,0,0,0,0}`, row 2
  `{0,1,0,0,0,0,0,0}`, row 3 `{0,0,0,0,1,0,0,0}` and row 4 `{0,0,0,0,0,1,0,0}`, and with a
  geometry of 4 heads, 2 key/value heads, heads of 2 numbers, rope over 2 numbers at base 10000,
  `eps` 0 and 4 positions, that `block({1,1,1,1}, "blk.0.", 0)` returns `{1,2,3,4}`: heads 0 and 1
  reading key/value head 0's value `{2,4}`, heads 2 and 3 reading key/value head 1's value
  `{6,8}`, where heads alternating between the key/value heads instead give `{1,2,1,2}`.

## Open questions

- None.

## Amendments

- None.
