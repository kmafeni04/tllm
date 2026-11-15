# 15: The whole-model pass

**Status**: done   **Depends on**: 13, 14

## Problem & context

`13` runs a stack of attention-only blocks and `14` adds the linear-attention block; neither runs
the model. `serve` maps the file and answers `/health`, and no code path reads a token. So the
engine can do one block but not the model: nothing looks up an embedding, nothing chooses a
block's kind, and nothing reads the file's own numbers, which is why every earlier spec took its
geometry as an argument.

This spec adds `compute.model(backend, handle, positions)` and one record, `Model`, whose one
operation, `forward(token_id, position)`, takes one row of `token_embd.weight`, runs every block
the main pass holds at that position, and returns the scores `13/FR-008` describes. One token per
call: the attention caches of `12` and the delta states of `14` live in the handles, so a call at
the next position sees every token before it.

The numbers come from the file, not from this document. The local Qwen3.6 file names its
architecture `qwen35moe` in `general.architecture`, and every key below carries that value and a
dot as its prefix: `block_count` 41, `nextn_predict_layers` 1, `embedding_length` 2048,
`attention.head_count` 16, `rope.dimension_count` 64, `rope.freq_base` 10000000,
`attention.layer_norm_rms_epsilon`, `context_length`, `expert_used_count` 8, and the five `ssm`
numbers `14` uses: `conv_kernel` 4, `state_size` 128, `group_count` 16, `time_step_rank` 32 and
`inner_size` 4096. llama.cpp maps these names in `llama-arch.cpp`, and in `models/qwen35moe.cpp`
it loads the last block as a next-token draft block and does not execute it in the main pass
(`n_layer()` against `n_layer_all`), which is why the run is `block_count` minus
`nextn_predict_layers` (github.com/ggml-org/llama.cpp, commit
`ccd360f060a0f17b87ee0d4c5d91ee84cfdff484`).

Block kinds come from the same file. In that model file a layer whose index counts from zero where
`(index + 1) % 4` is not zero is linear, and 4 is what llama.cpp assumes when
`full_attention_interval` is absent, so this spec reads that key and takes a block whose
`(index + 1)` is a multiple of it as a full-attention block, every other block of the run being
linear. That is what `12` states for the local file: 10 full-attention blocks, every fourth from
block 3, and 30 linear ones.

The head length comes from the tensor the block reads, not from the width. `12` states the local
file as 16 query heads of 256 numbers each, an `attn_q.weight` of 2048 by 8192 whose rows carry one
head's query numbers and then its gate numbers, and `attn_k.weight` of 2048 by 512. So the head
length is 8192 / (2 × 16) = 256 and the key/value head count is 512 / 256 = 2; the width over the
head count, 2048 / 16 = 128, is not this model's head length and would make `12/FR-010` refuse the
8192-row query tensor. The model therefore takes the head length as the first full-attention
block's `attn_q.weight` row count divided by twice `attention.head_count`, and the key/value head
count as its `attn_k.weight` row count divided by that head length, so the geometry cannot
disagree with the tensors the block reads.

How many positions a cache holds is the caller's, the one number `12` deliberately left out of the
file: the local file's `context_length` is 262144, and `12` prices that at 1.07 GB for each of its
10 attention blocks, so `compute.model` takes a `positions` argument and the file's own context is
only the default. A caller that has not been asked for a long context says so.

A 23 GB file cannot be a test input, the reason `11` gives, so each case here builds a hand-made
file with two, four or eight numbers per hidden position, at most one full-attention block and
one linear block, and a small output projection. The numbers a pass returns are then checkable
by hand, and the two block kinds' own numbers are `12` to `14`'s.

The model holds one `Attention` when the run holds a full-attention block, one `FeedForward`, one
`LinearAttention` when the run holds a linear block, all built from the same backend, handle and
geometry, and it counts the positions it has taken. The draft block is never read, so no BF16
router type is needed and `13/FR-012` stands as it is.

Refusals keep the shape `11` to `14` use: nil and one line, so a mistake in the next spec's
wiring is a return value and the process stays up. This is the pass a prompt runs one token at a
time: spec 16 turns text into token ids and back, and spec 17 puts a chat route over it.

```teal
local weights = require("tllm.weights")

local record compute
  record Model
    positions: integer
    forward: function(
      Model, integer, integer
    ): {number} | nil, string | nil
  end

  model: function(
    Backend, weights.Handle, integer | nil, CacheType | nil
  ): Model | nil, string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want a model built from the mapped file's own keys, so that the run, the block kinds and the geometry come from the file and not from constants.
- US-002 (P1) As the engine author, I want one token through every block of the run, so that a position produces one score per vocabulary token.
- US-003 (P1) As the engine author, I want each block of the run to run as its own kind in index order with the residuals, so that the 30 linear and 10 full-attention blocks of the local file compose.
- US-004 (P2) As the engine author, I want a token, a position, a key or a number that does not fit back as one line, so that a wiring mistake is a return value and not a crash.
- US-005 (P2) As the engine author, I want positions taken one after another, so that the caches cannot silently skip or rewind.

## Functional requirements

- FR-001 (US-001) WHEN `compute.model(backend, handle, positions, cache_type)` is called with a backend that `compute.load` returned and a handle a GGUF file mapped, THE SYSTEM SHALL return a model whose `positions` is the number of positions it was built with, the `positions` argument when it is given and the file's `context_length` otherwise, and nil as its second value, and SHALL hand `cache_type` to every attention geometry it builds, `f16` when it is nil, and IF `cache_type` is given and is not one of `compute`'s `CacheType` values, `f32`, `f16`, `q8_0` and `q4_0`, THEN THE SYSTEM SHALL return nil and one line naming `model` and the type, that fault coming after FR-009's backend fault and before FR-014's read, and IF `cache_type` is `q8_0` or `q4_0` and the run holds a full-attention block whose `head_length` is not a multiple of 32, THEN THE SYSTEM SHALL return nil and one line naming the type and that `head_length`, reported once `head_length` is known and before the model is built. Its tests pass `f32`, so their numbers stand.
- FR-002 (US-001) THE SYSTEM SHALL take the run as the blocks whose indices count from zero below the file's `block_count` minus its `nextn_predict_layers`, zero when the file does not hold that key, and read no tensor of a block at or above that count.
- FR-003 (US-001) THE SYSTEM SHALL take a block of the run as a linear-attention block when `(index + 1)` modulo the file's `full_attention_interval`, four when the file does not hold it, is not zero, and as a full-attention block otherwise.
- FR-004 (US-001) THE SYSTEM SHALL take the geometry from the metadata FR-014 reads as: `embedding_length` as the width; `expert_used_count` as `experts_used`; `attention.layer_norm_rms_epsilon` as `eps`; `positions` as the argument when it is given and `context_length` otherwise; when the run holds a full-attention block, `attention.head_count` as `heads`, the row count of that block's `attn_q.weight` divided by twice `heads` as `head_length`, the row count of its `attn_k.weight` divided by `head_length` as `kv_heads`, `rope.dimension_count` as `rope_dimensions` and `rope.freq_base` as `rope_base`; and, when the run holds a linear block, `ssm.group_count` as `linear_key_heads`, `ssm.state_size` as `linear_key_length`, `ssm.time_step_rank` as `linear_value_heads`, `ssm.inner_size` divided by `linear_value_heads` as `linear_value_length` and `ssm.conv_kernel` as `linear_conv_kernel`.
- FR-005 (US-002) WHEN `model:forward(token_id, position)` is called with `token_id` a row of `token_embd.weight` holding the width's numbers and `position` the number of positions the model has taken, THE SYSTEM SHALL return one number per row of `output.weight`, being `13/FR-008`'s scores of the hidden FR-006 gives, and nil as its second value.
- FR-006 (US-003) WHEN a call begins, THE SYSTEM SHALL take the hidden as the row of `token_embd.weight` at `token_id` and, for each block of the run in index order, call `attention:block` for a full-attention block and `linear_attention:block` for a linear one with the hidden, the prefix `blk.<index>.` and `position`, add the numbers it returns to the hidden position by position, call `feed_forward:block` with that same prefix on the sum, and take its numbers as the hidden.
- FR-007 (US-005) IF `position` is negative or is not the number of positions the model has taken, THEN THE SYSTEM SHALL return nil and one line naming `forward` and the position, taking no position.
- FR-008 (US-004) IF `token_id` is negative, or is not below the row count of `token_embd.weight`, or the row it holds for `token_id` does not hold `embedding_length` numbers, or the handle does not hold `token_embd.weight`, or the tensor it holds has no rows, THEN THE SYSTEM SHALL return nil and one line naming `token_embd.weight` and the token id.
- FR-009 (US-004) IF `compute.model` is given a backend that `compute.load` did not return, or FR-014's read returns nil, or `general.architecture` is missing or is not text, or a key `block_count`, `embedding_length`, `expert_used_count`, `attention.layer_norm_rms_epsilon` or `context_length`, or, when the run holds a full-attention block, `attention.head_count`, `rope.dimension_count` or `rope.freq_base`, or, when the run holds a linear block, one of the five `ssm` keys FR-004 names, is missing or is not a number, or `nextn_predict_layers` or `full_attention_interval` is held and is not a number, or `positions` is given and is below one or above `context_length`, or a number the file holds for a key this requirement names is below one, except `attention.layer_norm_rms_epsilon` and `nextn_predict_layers`, which are at fault below zero, THEN THE SYSTEM SHALL return nil and one line naming `model` and, in the order this requirement names them, the first key or number at fault, the key fault naming that key with the file's architecture and a dot before it, except `general.architecture`, which it names bare, the number fault naming the key it came from or `positions` when that argument is at fault, and the read's own line, behind `model` and a colon, when the read is what failed.
- FR-010 (US-004) IF a call FR-006 or FR-005 makes on an attention block, a linear-attention block, a feed-forward block or the scores returns nil, THEN THE SYSTEM SHALL return nil and that line unchanged.
- FR-011 (US-005) WHEN a `forward` call returns scores, THE SYSTEM SHALL add one to the number of positions the model has taken.
- FR-012 (US-005) IF a `forward` call returns nil after a block of the run answered, THEN THE SYSTEM SHALL leave the number of positions where it was, so a later call at the same position returns nil with the line of the block that already took it.
- FR-013 (US-001) THE SYSTEM SHALL read no `attention.head_count`, `rope.dimension_count` or `rope.freq_base`, and no `attn_q.weight` or `attn_k.weight` row count, when the run holds no full-attention block, the geometry's attention numbers then being values no block reads.
- FR-014 (US-001) THE SYSTEM SHALL read the file's metadata from the handle's path and take `general.architecture` as the text every other key carries and a dot before it.
- FR-015 (US-004) IF, when the run holds a full-attention block, the handle does not hold the first full-attention block's `attn_q.weight` or `attn_k.weight`, or either holds no rows, or `attn_q.weight`'s row count is not a multiple of twice `attention.head_count`, or `attn_k.weight`'s row count is not a multiple of `head_length`, or that row count divided by `head_length`, which FR-004 takes as `kv_heads`, does not divide `attention.head_count`, or `rope.dimension_count` is below two, is odd, or is above `head_length`, or, when the run holds a linear block, `ssm.inner_size` is not a multiple of `ssm.time_step_rank`, or `ssm.time_step_rank` is not a multiple of `ssm.group_count`, or `block_count` minus `nextn_predict_layers` is below one, that fault naming `block_count`, THEN THE SYSTEM SHALL return nil and one line naming `model` and, in the order this requirement names them, the first tensor or number at fault, a fault in a tensor, in its row count, or in the `kv_heads` that row count gives naming that tensor, any other number fault naming the key it came from, after any fault FR-009 names.

## Non-goals

- No prompt, no tokenizer, no sampler, no loop and no route: `forward` takes one token id and returns scores; text in and out is spec 16, and sampling and the chat completions route are spec 17.
- No draft block and no speculative decoding: the blocks from the draft count on are never read, so `13`'s exclusion of block 40's BF16 router never bites.
- No batch, no two positions at once, no expert cache and no unload or release: one token per call, and the caches and states live as long as the model does.

## Tests

- T-001 (FR-001, FR-004, FR-005, FR-006, FR-010, FR-011): `one token through a full-attention
  model gives its scores`, proves with a fixture whose metadata is `general.architecture`
  `qwen35moe` with the keys `block_count` 1, `embedding_length` 2, `attention.head_count` 1,
  `rope.dimension_count` 2, `rope.freq_base` 10000, `attention.layer_norm_rms_epsilon` 0,
  `context_length` 4, `expert_used_count` 1 and `full_attention_interval` 1 and no `ssm` key,
  holding the f32 tensors `token_embd.weight` 2 numbers per row over 2 rows, row 1 `{1,0}` and
  row 2 `{3,4}`; under `blk.0.` `attn_norm.weight` `{1,1}`; `attn_q.weight` 2 numbers per row
  over 4 rows, rows 1 and 2 `{1,0}` and `{0,1}`, rows 3 and 4 zeros; `attn_k.weight` 2 rows
  `{1,0}` and `{0,1}`; `attn_v.weight` 2 rows `{1,0}` and `{0,1}`; `attn_q_norm.weight` and
  `attn_k_norm.weight` `{1,1}`; `attn_output.weight` 2 rows of 2 zeros;
  `post_attention_norm.weight` `{1,1}`; `ffn_gate_inp.weight` 2 rows of 2 zeros;
  `ffn_gate_inp_shexp.weight` 2 zeros; `ffn_up_shexp.weight`, `ffn_gate_shexp.weight` and
  `ffn_down_shexp.weight` 2 rows of 2 zeros; `ffn_up_exps.weight`, `ffn_gate_exps.weight` and
  `ffn_down_exps.weight` 2 numbers per row, 2 per expert and 2 experts of zeros; and at the root
  `output_norm.weight` `{1,1}` and `output.weight` 2 numbers per row over 3 rows, `{1,0}`,
  `{0,1}` and `{1,1}`, that `compute.model(backend, handle, 2)` returns a model and nil with no
  `ssm` key read, that `forward(0, 0)` returns `{1.4142136, 0, 1.4142136}` to within 1e-5 and
  nil, that `forward(1, 1)` on that model returns `{0.8485281, 1.1313708, 1.9798990}` to within
  1e-5 and nil, the block contributing zeros, the residual leaving the token's row as the hidden
  and the second call standing at position 1 because the first added one, and that on a fresh
  model over the same fixture with `expert_used_count` 3, `forward(0, 0)` returns nil and one
  line holding `ffn_gate_inp.weight`, the experts used reaching the router from that key.
- T-002 (FR-002, FR-003, FR-004, FR-006, FR-010, FR-012, FR-013): `each block of the run runs as its kind
  and its numbers are added`, proves with T-001's fixture grown to the keys `block_count` 2 and
  `full_attention_interval` 2 and the `ssm` keys `group_count` 2, `state_size` 2,
  `time_step_rank` 2, `inner_size` 4 and `conv_kernel` 2, with `token_embd.weight` a third row
  `{1,1}`, whose `blk.0.` holds `14/T-001`'s ten tensors unchanged and whose `blk.1.` holds
  T-001's `blk.0.` tensors, each block keeping T-001's zero feed-forward tensors, that
  `compute.model(backend, handle, 2)` returns a model and nil and `forward(2, 0)` returns
  `{0.5438875, 1.3054449, 1.8493324}` to within 1e-4 and nil, the linear block answering
  `{100, 241.4213562}` for the row `{1,1}`, the add giving `{101, 242.4213562}`, the zeros of the
  full block and both feed-forwards leaving it, and the final norm and projection giving those
  three numbers, so a pass that dropped the add gives `{1, 1, 2}` instead, those five `ssm`
  numbers being the geometry `14/T-001` uses, whose tensor shapes `14/FR-013` checks, so the
  model's numbers reach that block and `14`'s own tests prove what the block does with them;
  that with `blk.0.ssm_out.weight` removed the call returns nil and one line holding
  `blk.0.ssm_out.weight`; that with `blk.1.attn_v.weight` removed the call returns nil and one
  line holding `blk.1.attn_v.weight`, a second `forward(1, 0)` on that model returns nil and one
  line holding `blk.0.` and 0 because block 0 took the position before block 1 refused, so the
  blocks ran in index order; that on T-001's fixture with `block_count` 2 and
  `nextn_predict_layers` 1 the call returns `{0.8485281, 1.1313708, 1.9798990}` to within 1e-5
  while the same fixture without that key returns nil and one line holding
  `blk.1.attn_norm.weight`; and that a fixture holding `14/T-001`'s ten tensors and `13`'s zero
  feed-forward tensors under both `blk.0.` and `blk.1.`, with `ssm_out.weight` zeroed under
  both, `block_count` 2, the `ssm` keys above and no `full_attention_interval`,
  `attention.head_count`, `rope.dimension_count` or `rope.freq_base` returns
  `{0.8485281, 1.1313708, 1.9798990}` for `forward(1, 0)` to within 1e-5 under
  `compute.model(backend, handle, 2)`, both blocks being linear at the default of 4 and no
  attention key or tensor being read.
- T-003 (FR-004, FR-007, FR-008, FR-010): `a token and a position that do not fit are refused`,
  proves on T-001's fixture that `forward(1, 2)` on a model built with `positions` 2 returns nil
  and one line holding `forward` and 2, that `forward(1, 0)` on that model then returns
  `{0.8485281, 1.1313708, 1.9798990}` to within 1e-5, that a model built with no `positions`
  answers `forward(1, 0)`, `forward(1, 1)`, `forward(1, 2)` and `forward(1, 3)` with scores and
  nil and refuses `forward(1, 4)` with one line holding `blk.0.` and 4, `context_length` being
  4; that a model built with `positions` 1 answers `forward(1, 0)` with
  `{0.8485281, 1.1313708, 1.9798990}` to within 1e-5 and the `forward(1, 1)` after it with nil
  and one line holding `blk.0.` and 1, the argument being the cache's size; that `forward(-1, 0)`
  and `forward(2, 0)` on fresh models each return nil and one line holding `token_embd.weight`,
  that on a fresh model whose fixture has `token_embd.weight` removed `forward(0, 0)` returns nil
  and one line holding `token_embd.weight`, and that on a fresh model whose fixture has
  `token_embd.weight` 3 numbers per row over 2 rows `forward(0, 0)` returns nil and one line
  holding `token_embd.weight`.
- T-004 (FR-009, FR-014, FR-015): `a key or a number that does not fit is refused with one line`, proves
  on T-001's fixture that removing `qwen35moe.rope.freq_base`, removing
  `qwen35moe.attention.head_count` and removing `general.architecture` each make `compute.model`
  return nil and one line holding `model` and that removed key, that `blk.0.attn_q.weight` given
  6 rows with `attention.head_count` 2 is refused with one line holding `model` and
  `blk.0.attn_q.weight`, that `blk.0.attn_k.weight` given 6 rows with `attn_q.weight` given 64
  rows and `attention.head_count` 16 is refused with one line holding `model` and
  `blk.0.attn_k.weight`, that `rope.dimension_count` 3 and `rope.freq_base` 0 are each
  refused with one line holding `model` and `qwen35moe.rope.dimension_count` and, in the second
  call, `qwen35moe.rope.freq_base`, that `qwen35moe.block_count` 0,
  `qwen35moe.full_attention_interval` 0 and `qwen35moe.nextn_predict_layers` -1 are each refused
  with one line holding `model` and that key, that a `positions` of 0 and a `positions` of 5
  under a `context_length` of 4 are each refused with one line holding `model` and `positions`,
  that an `attention.layer_norm_rms_epsilon` of 0 builds a model and the `forward(0, 0)` after it
  returns scores, that a backend built by hand is refused with one line holding `model`, that a
  handle whose path names a file removed after it was loaded is refused with one line holding
  `model`, and that on T-002's fixture with `ssm.group_count` removed and with `ssm.inner_size`
  5, `compute.model` returns nil and one line holding `model` and `qwen35moe.ssm.group_count`
  and, in the second call, `model` and `qwen35moe.ssm.inner_size`; that with `blk.0.attn_q.weight`
  removed the model is refused with one line holding `model` and `blk.0.attn_q.weight`; that
  `rope.dimension_count` 1 and `rope.dimension_count` 4 under `attention.head_count` 1 are each
  refused with one line holding `model` and `qwen35moe.rope.dimension_count`; that on T-002's
  fixture with `ssm.time_step_rank` 3 and `ssm.inner_size` 6, `compute.model` is refused with one
  line holding `model` and `qwen35moe.ssm.time_step_rank`; and that on T-001's fixture with
  `block_count` 2 and `nextn_predict_layers` 2, `compute.model` is refused with one line holding
  `model` and `qwen35moe.block_count`.
- T-005 (FR-004): `the head length comes from the query tensor and not the width`, proves with a
  fixture whose metadata is `general.architecture` `qwen35moe` with `block_count` 1,
  `embedding_length` 4, `attention.head_count` 4, `rope.dimension_count` 2, `rope.freq_base`
  10000, `attention.layer_norm_rms_epsilon` 0, `context_length` 4, `expert_used_count` 1 and
  `full_attention_interval` 1, holding `token_embd.weight` 4 numbers per row over 2 rows, row 1
  `{1,1,1,1}` and row 2 `{2,0,0,0}`; under `blk.0.` `attn_norm.weight` `{1,1,1,1}`;
  `attn_q.weight` 4 numbers per row over 16 rows, rows 1 and 2 `{1,0,0,0}` and `{0,1,0,0}`,
  rows 5 and 6 the same, rows 9 and 10 the same, rows 13 and 14 the same, and rows 3, 4, 7, 8,
  11, 12, 15 and 16 zeros; `attn_k.weight` 2 rows `{1,0,0,0}` and `{0,1,0,0}`; `attn_v.weight`
  2 rows `{1,0,0,0}` and `{0,1,0,0}`; `attn_q_norm.weight` and `attn_k_norm.weight` `{1,1}`;
  `attn_output.weight` 4 rows of 8 zeros; `post_attention_norm.weight` `{1,1,1,1}`;
  `ffn_gate_inp.weight` 4 numbers per row over 2 rows of zeros; `ffn_gate_inp_shexp.weight` 4
  zeros; `ffn_up_shexp.weight` and `ffn_gate_shexp.weight` 4 numbers per row over 2 rows of
  zeros; `ffn_down_shexp.weight` 2 numbers per row over 4 rows of zeros; `ffn_up_exps.weight`
  and `ffn_gate_exps.weight` 4 numbers per row, 2 per expert and 2 experts of zeros;
  `ffn_down_exps.weight` 2 numbers per row, 4 per expert and 2 experts of zeros; and at the root
  `output_norm.weight` `{1,1,1,1}` and `output.weight` 4 numbers per row over 2 rows,
  `{1,0,0,0}` and `{0,1,1,0}`, that `compute.model(backend, handle, 2)` returns a model and nil
  and `forward(0, 0)` returns `{1, 2}` to within 1e-5 and nil, the 4 wide width over 4 heads
  giving a head length of 1 that `12/FR-010` refuses against the 16-row query tensor, so only 16
  rows over twice 4 heads, a head length of 2, gives a pass.
- T-006 (FR-004, FR-011): `the rope values reach the attention`, proves with a fixture whose metadata is
  `general.architecture` `qwen35moe` with `block_count` 1, `embedding_length` 8,
  `attention.head_count` 1, `rope.dimension_count` 4, `rope.freq_base` 100,
  `attention.layer_norm_rms_epsilon` 0, `context_length` 4, `expert_used_count` 1 and
  `full_attention_interval` 1, holding `token_embd.weight` 8 numbers per row over 2 rows, row 1
  `{1,1,1,1,0,0,0,0}` and row 2 `{2,1,3,1,0,0,0,0}`; under `blk.0.` `attn_norm.weight` 8 numbers
  of 1; `attn_q.weight` 8 numbers per row over 8 rows, rows 1 to 4 `{1,0,0,0,0,0,0,0}`,
  `{0,1,0,0,0,0,0,0}`, `{0,0,1,0,0,0,0,0}` and `{0,0,0,1,0,0,0,0}`, rows 5 to 8 zeros;
  `attn_k.weight` 8 numbers per row over 4 rows, those same four rows; `attn_v.weight` 8 numbers
  per row over 4 rows, those same four rows; `attn_q_norm.weight` and `attn_k_norm.weight` 4
  numbers of 1; `attn_output.weight` 4 numbers per row over 8 rows, rows
  1 to 4 `{1,0,0,0}`, `{0,1,0,0}`, `{0,0,1,0}` and `{0,0,0,1}` and rows 5 to 8 zeros;
  `post_attention_norm.weight` 8 numbers of 1; `ffn_gate_inp.weight` 8 numbers per row over 2
  rows of zeros; `ffn_gate_inp_shexp.weight` 8 zeros; `ffn_up_shexp.weight` and
  `ffn_gate_shexp.weight` 8 numbers per row over 2 rows of zeros;
  `ffn_down_shexp.weight` 2 numbers per row over 8 rows of zeros; `ffn_up_exps.weight` and
  `ffn_gate_exps.weight` 8 numbers per row, 2 per expert and 2 experts of zeros;
  `ffn_down_exps.weight` 2 numbers per row, 8 per expert and 2 experts
  of zeros; and at the root `output_norm.weight` 8 numbers of 1 and `output.weight` 8 numbers
  per row over 3 rows, `{1,0,0,0,0,0,0,0}`, `{0,1,0,0,0,0,0,0}` and `{1,1,0,0,0,0,0,0}`, that
  `compute.model(backend, handle, 2)` returns a model and nil, that its `forward(0, 0)` returns
  `{1.4142136, 1.4142136, 2.8284271}` to within 1e-5 and nil, that the `forward(1, 1)` after it
  returns `{1.4666941, 0.7843212, 2.2510153}` to within 1e-5 and nil, that on a second model
  over the same fixture with `rope.freq_base` 1 the `forward(1, 1)` after a `forward(0, 0)`
  returns a list
  differing from the first model's in at least one number by more than 1e-6, and that on a third
  model with `rope.dimension_count` 2 instead of 4 the same two calls likewise return a list
  differing from the first model's by more than 1e-6, the 4-number head rotating the numbers at
  its first and third places by the position and the numbers at its second and fourth places by
  the position times the base to the -0.5, so the 100 and the 1 part ways on that second pair,
  which the fixture's four non-zero query and key rows keep alive, and the 2 dimensions rotating
  the first and second places instead.

## Open questions

- None.

## Amendments

- None.
