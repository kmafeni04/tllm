# 13: The block's feed-forward and the scores

**Status**: done   **Depends on**: 12

## Problem & context

`12` gives one block's attention half: norm, query, key, value, rope, the cache and the
gated attention numbers. A block is not finished there. After the attention residual, every
block of the local file runs a norm, a router over 256 experts, the eight best experts'
gated matrices weighted and summed, a shared expert with its own sigmoid gate and another
residual. After the last block the model runs a final norm and a 248320 wide output
projection. None of that exists, so a token cannot leave a stack of blocks.

The local Qwen3.6 file names the numbers and tensors this must carry. It has 256 experts,
8 used per token, 512 numbers per expert and 512 numbers in the shared expert. Every one of
its 41 blocks, the 30 linear-attention blocks included, holds the same feed-forward tensors:
`post_attention_norm.weight` (2048), `ffn_gate_inp.weight` (2048, 256), `ffn_gate_inp_shexp.weight`
(2048), `ffn_gate_shexp.weight` and `ffn_up_shexp.weight` (2048, 512), `ffn_down_shexp.weight`
(512, 2048), `ffn_gate_exps.weight` and `ffn_up_exps.weight` (2048, 512, 256), and
`ffn_down_exps.weight` (512, 2048, 256). At the root it holds `output_norm.weight` (2048)
and `output.weight` (2048, 248320). Block 40 is the next-token draft block the main pass
leaves alone, and its router is BF16 rather than f32.

llama.cpp builds the block's tail in `build_layer_ffn` and `build_moe_ffn`
(`src/models/qwen35moe.cpp` and `src/llama-graph.cpp`, commit
`ccd360f060a0f17b87ee0d4c5d91ee84cfdff484`, raw.githubusercontent.com/ggml-org/llama.cpp).
The router is a softmax over every expert; the 8 largest are kept, their probabilities are
divided by the sum of the kept probabilities, and each expert's `silu(gate) * up` is
projected down and weighted. The shared expert is `silu(gate) * up`, projected down, then
multiplied by the sigmoid of a one-number gate. The block's residual is the hidden from
before `post_attention_norm`; `output_norm` and `output` come after the last block. The file
states no expert weight scale.

This spec adds `compute.feed_forward(backend, handle, geometry)` and one record,
`FeedForward`, with two operations, `block` and `scores`. `12`'s `Geometry` record gains one
number, `experts_used`, because no tensor stores how many experts one token keeps. A block is
read over `12`'s `Attention` handle as it is, so the cache the attention blocks keep is the
cache spec 15's stack reads; carrying a hidden through that stack and picking its best token
is `15`'s model pass, and this spec stops at one block's numbers.

Refusals keep the shape `11` and `12` use: nil and one line, so a mistake in the next spec's
wiring is a return value and the process stays up. These belong in the core for the reason
`12` gives: they compose `11`'s operations over the same ggml context, and a plugin has no way
to reach that context. The feed-forward handle reads tensors by name at call time, so building
one over the 23 GB model reads no weights.

```teal
local weights = require("tllm.weights")

local record compute
  record FeedForward
    block: function(
      FeedForward, {Row}, string
    ): {{number}} | nil, string | nil
    scores: function(
      FeedForward, {{number}}
    ): {{number}} | nil, string | nil
    close: function(FeedForward)
  end

  feed_forward: function(
    Backend, weights.Handle, Geometry
  ): FeedForward | nil, string | nil
end
```

A `FeedForward` handle's `close` gives back its hold on the weights mapping and refuses every
operation afterwards; `12/FR-014` states that contract and the refusal line, and it is the same for
this handle with no caches of its own to free.

## User stories

- US-001 (P1) As the engine author, I want a block's router, experts, shared expert and residual computed by ggml over the mapped weights, so that a block is a whole block and not only its attention.
- US-002 (P1) As the engine author, I want the final norm and the output projection as one operation, so that a hidden vector becomes one score per vocabulary token.
- US-003 (P1) As the engine author, I want the highest scoring token for a stack of attention blocks at one position, so that a forward pass has an answer before the linear-attention work exists.
- US-004 (P2) As the engine author, I want a name, type, shape or number that does not fit back as one line, so that a wiring mistake is a return value and not a crash.

## Functional requirements

- FR-001 (US-001) WHEN `compute.feed_forward(backend, handle, geometry)` is called with a
  `backend` that `compute.load` returned and a `geometry` whose `experts_used` is one or more
  and whose `eps` is zero or more, THE SYSTEM SHALL return a feed-forward handle and nil. Its
  tests give the geometry `cache_type` `f32`, so their numbers stand.
- FR-002 (US-001) WHEN that handle's `block(rows, prefix)` is called with a batch of
  `compute.Row`s, each row's `hidden` holding one number per position of
  `<prefix>post_attention_norm.weight`, THE SYSTEM SHALL return one contribution row per input
  row in order, each holding one number per position of that row's `hidden`, being the sum of
  that row's `hidden`, the weighted expert output FR-003 to FR-006 describe and the gated shared
  output FR-007 describes, and nil as its second value.
- FR-003 (US-001) WHEN a block call begins, THE SYSTEM SHALL normalise `hidden` with
  `<prefix>post_attention_norm.weight` by `11/FR-004`'s norm and the geometry's `eps`, and
  read `<prefix>ffn_gate_inp.weight` as one group of hidden-length numbers per row.
- FR-004 (US-001) WHEN the normalised hidden is ready, THE SYSTEM SHALL take one router
  probability per row of `<prefix>ffn_gate_inp.weight` as `11/FR-006`'s softmax over the
  rows' sums of products with the normalised hidden, and select the `experts_used` rows with
  the largest probabilities, a tie going to the row with the lower index counting from zero.
- FR-005 (US-001) WHEN the rows are selected, THE SYSTEM SHALL take each selected row's
  weight as its probability divided by the sum of the selected rows' probabilities, and the
  weighted expert output as the sum over the selected rows of that row's weight times the
  expert output FR-006 describes for that row.
- FR-006 (US-001) WHEN a selected row's expert output is computed, THE SYSTEM SHALL take the
  up list as `11/FR-002`'s matrix product of `<prefix>ffn_up_exps.weight`'s matrix at that
  row, counting from zero, with the normalised hidden, take the gate list as `11/FR-002`'s
  matrix product of `<prefix>ffn_gate_exps.weight`'s matrix at that row, counting from zero,
  with the normalised hidden, and take the expert output as `11/FR-002`'s matrix product of
  `<prefix>ffn_down_exps.weight`'s matrix at that row, counting from zero, with `11/FR-008`'s
  swiglu of the gate list first and the up list second.
- FR-007 (US-001) WHEN the shared expert is computed, THE SYSTEM SHALL take the shared output
  as `11/FR-002`'s matrix product of `<prefix>ffn_down_shexp.weight` with `11/FR-008`'s
  swiglu of `11/FR-002`'s matrix product of `<prefix>ffn_gate_shexp.weight` with the
  normalised hidden and `11/FR-002`'s matrix product of `<prefix>ffn_up_shexp.weight` with
  the normalised hidden, then multiply each of its numbers by the one number `11/FR-009`'s
  `sigmoid_gate` returns for the one-number list holding the sum over the hidden positions of
  `<prefix>ffn_gate_inp_shexp.weight`'s number times the normalised hidden's number and the
  one-number list holding 1.
- FR-008 (US-002) WHEN `scores(hidden)` is called with a batch of hidden rows, each holding one
  number per position of `output_norm.weight`, THE SYSTEM SHALL return one score row per input
  row in order, each holding one number per row of `output.weight`, being `11/FR-002`'s matrix
  product of `output.weight` with that hidden row normalised by `output_norm.weight` by
  `11/FR-004`'s norm and the geometry's `eps`.
- FR-011 (US-004) IF `compute.feed_forward` is given a backend that `compute.load` did not
  return, or a `geometry` whose `experts_used` is below one or whose `eps` is below zero,
  THEN THE SYSTEM SHALL return nil and one line naming `feed_forward` and, the geometry
  being at fault, the number.
- FR-012 (US-004) IF a tensor `block` or `scores` reads is not in the handle, or is a tensor
  whose type is not `f32` for `post_attention_norm.weight`, `ffn_gate_inp.weight`,
  `ffn_gate_inp_shexp.weight` or `output_norm.weight`, or is not a type `40` lists for the other
  seven, THEN THE SYSTEM SHALL return nil and one line naming the
  operation, the first tensor in this order that the handle does not hold, and, no tensor
  being missing, the first whose type is at fault together with that type's name:
  `post_attention_norm.weight`, `ffn_gate_inp.weight`, `ffn_up_exps.weight`,
  `ffn_gate_exps.weight`, `ffn_down_exps.weight`, `ffn_up_shexp.weight`,
  `ffn_gate_shexp.weight`, `ffn_down_shexp.weight`, `ffn_gate_inp_shexp.weight` for `block`,
  and `output_norm.weight` then `output.weight` for `scores`.
- FR-013 (US-004) IF a tensor's shape does not fit a row of `hidden` or the other tensors, that is if
  `post_attention_norm.weight` is not one-dimensional or does not hold one number per
  position of `hidden`; `ffn_gate_inp.weight` is not two-dimensional, its first dimension is
  not the number of `hidden`'s positions, or it holds no rows; `ffn_gate_inp_shexp.weight` is
  not one-dimensional or does not hold one number per position of `hidden`;
  `ffn_up_exps.weight` or `ffn_gate_exps.weight` is not three-dimensional, its first
  dimension is not the number of `hidden`'s positions, its second dimension is not the same
  expert length of one or more, or its third dimension is not the router's row count;
  `ffn_down_exps.weight` is not three-dimensional, its first dimension is not that expert
  length, its second dimension is not the number of `hidden`'s positions, or its third
  dimension is not the router's row count; `ffn_up_shexp.weight` or
  `ffn_gate_shexp.weight` is not two-dimensional, its first dimension is not the number of
  `hidden`'s positions, or its second dimension is not the same shared length of one or more;
  `ffn_down_shexp.weight` is not two-dimensional, its first dimension is not that shared
  length, or its second dimension is not the number of `hidden`'s positions;
  `experts_used` is more than the router's row count, that fault naming
  `ffn_gate_inp.weight`; `scores`'s `hidden` does not hold one
  number per position of `output_norm.weight`; `output_norm.weight` is not one-dimensional or
  does not hold the numbers `hidden` holds; or `output.weight` is not two-dimensional, its
  first dimension is not the number of `hidden`'s positions, or it holds no rows, THEN THE
  SYSTEM SHALL return nil and one line naming the operation, the tensor and the numbers at
  fault. The shape checks are made against each row of the batch's hidden.
- FR-015 (US-004) IF more than one refusal applies to a `block` or `scores` call, THEN THE
  SYSTEM SHALL report a missing tensor before a tensor of a type the operation does not read,
  a type fault before a shape that does not fit, and of two faults in FR-013's list the earlier
  one.
- FR-016 (US-004) IF the memory a `block` or `scores` call's graph needs cannot be allocated
  THEN THE SYSTEM SHALL return nil and one line naming the operation.
  Not testable: making that allocation fail needs a memory limit the suite cannot set portably;
  the branch returns nil and one line, which a reader confirms by inspection.

## Non-goals

- No linear-attention block and no delta-net work: conv1d, softplus, the gated norm and the
  delta-net scan are spec 14.
- No embedding lookup, no 40-block pass, no sampling, no prompt and no serving; block 40 and
  its BF16 router are spec 15's, and 15 amends FR-012's type list to admit a BF16 router for
  that block.
- No batch, no vision, no expert cache and no expert weight scale: every block call reads the
  named tensors and computes its experts again.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-005, FR-006, FR-007): `one block's feed-forward
  adds its experts and shared expert to the hidden`, proves with a fixture holding, under
  `blk.0.`, the block's f32 tensors: `post_attention_norm.weight` 2 numbers of 1;
  `ffn_gate_inp.weight` 2 numbers per row over 4 rows, the rows counting from one, row 1
  `{0,0}`, row 2 `{ln 2, 0}`, row 3 `{ln 3, 0}` and row 4 `{ln 4, 0}`;
  `ffn_gate_inp_shexp.weight` `{ln 3, 0}`; `ffn_up_shexp.weight` 1/100 times the 2 by 2
  identity; `ffn_gate_shexp.weight` 100 times the identity; `ffn_down_shexp.weight` rows
  `{1,0}` and `{0,2}`; `ffn_up_exps.weight` 2 by 2 by 4, 1/100 times the identity for each
  expert; `ffn_gate_exps.weight` 2 by 2 by 4, 100 times the identity for each expert;
  `ffn_down_exps.weight` 2 by 2 by 4, expert `e` counting from zero holding rows
  `{e+1, 0}` and `{2(e+1), 0}`; and, at the root, `output_norm.weight` 2 numbers of 1; and
  `output.weight` 2 numbers per row over 3 rows, `{1,0}`, `{0,1}` and `{1,1}`, and with a
  geometry of 1 head, 1 key/value head, heads of 2 numbers, rope over 2 numbers at base
  10000, `eps` 0, 1 position and 2 experts used, that
  `compute.feed_forward(backend, handle, geometry)`
  returns nil as its second value, that `block({1,1}, "blk.0.")` returns
  `{5.3214286, 9.6428571}` to within 1e-5 and `block({2,2}, "blk.0.")` returns
  `{6.3214286, 10.6428571}` to within 1e-5, the norm dividing by the root mean square so that
  both calls' expert and shared numbers are the same, the 100 wide gate saturating
  `11/FR-008`'s silu at 100 times the normalised hidden and the 1/100 up leaving the swiglu
  at the normalised hidden's square, the router keeping experts 3 and 2 with weights 4/7 and
  3/7, the experts giving `{25/7, 50/7}`, the shared gate of ln 3 giving 3/4 of `{1,2}`, and
  the residual adding the `hidden` each call was given, `{1,1}` and `{2,2}`, and that on a
  second handle whose geometry holds `eps` 1, `block({1,1}, "blk.0.")` returns
  `{3.1178393, 5.2356787}` to within 1e-5, the epsilon reaching the block's norm.
- T-002 (FR-004, FR-005): `a router tie takes the lower experts and their weights sum to
  one`, proves on T-001's fixture with `ffn_gate_inp.weight` replaced by four rows of `{0,0}`
  that `block({1,1}, "blk.0.")` returns `{3.25, 5.5}` to within 1e-5: the four probabilities
  are 1/4 each, the tie takes experts 0 and 1, their weights become 1/2 each, their down
  numbers `{1,2}` and `{2,4}` give `{1.5, 3}`, the shared expert gives `{0.75, 1.5}` and the
  residual `{1,1}`.
- T-003 (FR-008): `the final norm gives one score per token`, proves on T-001's fixture that
  `scores({3,4})` returns `{0.8485281, 1.1313708, 1.9798990}` to within 1e-5 and that on the
  `eps` 1 handle `scores({3,4})` returns `{0.8164966, 1.0886621, 1.9051587}` to within 1e-5.
- T-005 (FR-011, FR-012, FR-013, FR-015): `a name, type, shape or number that does
  not fit is refused with one line`, proves on T-001's fixture that a backend built by hand is
  refused with one line holding `feed_forward`; that a geometry with `experts_used` 0 and one
  with `eps` -1 are refused with one line holding `feed_forward`, 0 and -1; that
  `block({1,1}, "blk.9.")` is refused with one line holding
  `blk.9.post_attention_norm.weight`; that with `blk.0.ffn_gate_inp.weight` removed
  `block({1,1}, "blk.0.")` is refused with one line holding `ffn_gate_inp.weight`; that an
  f16 `ffn_gate_inp.weight` is refused with one line holding `ffn_gate_inp` and `f16`; that
  an f64 `ffn_up_exps.weight` is refused with one line holding `ffn_up_exps` and `f64`; that
  an f64 `output.weight` is refused by `scores({1,1})` with one line holding `output` and
  `f64`; that `block({1}, "blk.0.")` and `block({1,1,1}, "blk.0.")` are refused with one line
  holding `post_attention_norm.weight`; that an `ffn_gate_inp.weight` with 1 number per row is
  refused with one line holding `ffn_gate_inp.weight`; that an `ffn_down_exps.weight` with 2
  dimensions is refused with one line holding `ffn_down_exps.weight`; that an
  `ffn_down_shexp.weight` with 1 row and 2 numbers is refused with one line holding
  `ffn_down_shexp.weight`; that an `ffn_gate_exps.weight` with 3 experts under a 4-row router
  is refused with one line holding `ffn_gate_exps.weight`; that with 2 router rows, the three
  expert tensors cut to 2 experts and `experts_used` 3, `block({1,1}, "blk.0.")` is refused
  with one line holding `ffn_gate_inp.weight`; that `scores({1})` and `scores({1,1,1})` are
  refused with one line holding `output_norm.weight`; that with `output.weight` removed
  `scores({1,1})` is refused with one line holding `output.weight`; that with `output.weight`
  holding 1 number per row `scores({1,1})` is refused with one line holding `output.weight`;
  that with `blk.0.ffn_down_exps.weight` removed and an f16 `ffn_gate_inp.weight` the line
  holds `ffn_down_exps.weight`, with an f16 `ffn_gate_inp.weight` and a two-dimensional
  `ffn_down_exps.weight` the line holds `ffn_gate_inp` and `f16`, and with a two-dimensional
  `ffn_down_exps.weight` the line holds `ffn_down_exps.weight`.

## Open questions

- None.

## Amendments

- None.
