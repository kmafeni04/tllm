# 16: Text in, text out, in process

**Status**: done   **Depends on**: 09, 10, 15

## Problem & context

`10` turns text into ids, `09` turns ids into text, and `15` turns a token id at a position into
one score per vocabulary entry. Nothing joins the three: there is no way to hand the engine a
prompt and get an answer back, so nothing in the project can be shown to anyone, and the chat route
of the next spec has nothing to call. The sampler, too, does not exist anywhere: `compute` only
has `top_token`, which reads the highest score and is not exported from the model.

This spec adds one module, `tllm.generate`, with one operation: `text(model, tokenizer, prompt,
options)` encodes the prompt, forwards every prompt id at its own position, samples the next token
from the scores the last forward returned, forwards that token at the next position and samples
again, until the token count is reached or an end-of-text id is sampled, and decodes the sampled
ids into the string it returns. The prompt text is never part of that string: the caller already
holds it, and the chat route will report the two apart.

Sampling has three shapes and no more: a temperature absent or zero takes the highest score, and a
temperature above zero takes one of the `top_k` highest scores with a probability proportional to
the exponent of its score divided by the temperature. A draw is needed for that second shape, so
the options carry a seed and the generator's sequence is fixed below: without it a test could only
assert that two runs agree, and a broken generator that always returns the same number would pass.

The end-of-text id comes from the model's own file, `tokenizer.ggml.eos_token_id`, which no module
reads today, so `09`'s tokenizer gains a field for it; an `end_id` in the options overrides it, and
a call with neither runs to its token count. The seed is required rather than taken from a clock,
for the same reason: the same call then gives the same text on every run, and nothing in the engine
reads the time.

```teal
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")

local record generate
  record Options
    max_tokens: integer
    seed: integer
    temperature: number | nil
    top_k: integer | nil
    end_id: integer | nil
    start: integer | nil
    on_token: function(string) | nil
  end

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options
  ): string | nil, string | nil, integer | nil, {integer} | nil
end
```

A model takes one token per position and refuses a position that is not the next, so the caller
sizes the model's positions for the prompt's ids plus `max_tokens` minus one, the last sampled id
never being forwarded; a position past them comes back as the block's own line, unchanged. Faults
keep the shape `15` uses: nil and one line, so a mistake in the next spec's wiring is a return
value and the process stays up.

A 23 GB file cannot be a test input, the reason `11` and `15` give, so each case here builds a
hand-made file carrying both the model `15/T-001` builds and a three-piece vocabulary, and the ids
a sample takes are worked out by hand from `15/T-001`'s scores.

## User stories

- US-001 (P1) As the engine author, I want one call that takes prompt text and returns the model's continuation text, so that the engine can answer without a route or a command in front of it.
- US-002 (P1) As the engine author, I want every prompt id forwarded at its own position before anything is sampled, so that the model answers from the prompt instead of from nothing.
- US-003 (P1) As the engine author, I want the next token taken greedily or from a temperature and a top-k with a seed, so that output can be predictable or varied and a test can assert the exact token.
- US-004 (P1) As a person reading the output, I want generation to stop at a token count or at end of text and to come back as text alone, so that a count cannot run away and no marker token leaks into the answer.
- US-005 (P2) As the engine author, I want a prompt, an option or a model fault back as one line, so that a mistake is a return value and the process stays up.

## Functional requirements

- FR-001 (US-001) WHEN `generate.text(model, tokenizer, prompt, options)` is called with options FR-009 accepts, THE SYSTEM SHALL return the ids it sampled decoded by `tokenizer:decode` as the first value, nil as the second, the number of ids it sampled as the third, and the ids it sampled in the order it appended them as the fourth, nil when it returns nil, holding no part of the prompt, and report each token to `on_token` as it is appended when the options carry one. Its tests build their models with `f32`, so the numbers and reply they record stand.
- FR-002 (US-002) WHEN a call begins, THE SYSTEM SHALL take the prompt's ids from `tokenizer:encode(prompt)`, call `model:forward` with each id from the options' `start` on and its index as the position, in list order, and take the first sampled token from the scores the last of those calls returns, `start` absent being 0, so a call may begin at a position the model has already reached.
- FR-003 (US-002) WHEN a token is sampled that is neither the end id nor the last `max_tokens` allows, THE SYSTEM SHALL call `model:forward` with it at the next position and sample the next token from the scores that call returns.
- FR-004 (US-003) WHEN `temperature` is absent or zero, THE SYSTEM SHALL take the sampled token as the id of the highest score, the lowest id among equal highest scores.
- FR-005 (US-003) WHEN `temperature` is above zero, THE SYSTEM SHALL take the candidates as the `top_k` highest-scoring ids, fewer when the vocabulary is smaller than `top_k`, equal scores in increasing id order, all ids when `top_k` is absent, and take the sampled token as the first candidate whose probability summed with every candidate before it exceeds one draw, the last candidate when none does, each probability being the candidate's score minus the top candidate's score, divided by `temperature`, exponentiated, and divided by the sum of the candidates' exponentiated numbers.
- FR-006 (US-003) WHEN a draw is taken, THE SYSTEM SHALL take it from a generator whose state starts at `seed` modulo 4294967296, each draw setting the state to 1664525 times the state plus 1013904223 modulo 4294967296 and being that state divided by 4294967296.
- FR-007 (US-004) WHEN the sampled token is the end id, THE SYSTEM SHALL stop without appending it, the end id being `end_id` when given and the tokenizer's `end_id` otherwise, and no end id when both are absent.
- FR-008 (US-004) WHEN `max_tokens` ids have been appended, THE SYSTEM SHALL stop.
- FR-009 (US-005) IF `max_tokens` is below one, or `temperature` is given and is below zero, or `top_k` is given and is below one, or `start` is given and is below zero, THEN THE SYSTEM SHALL return nil and one line naming `generate` and the first option at fault in that order, these checks running before the prompt is encoded, and IF `start` is given and is at or above the number of ids the prompt encodes to, THEN THE SYSTEM SHALL return nil and one line naming `generate`, `start` and that count.
- FR-010 (US-005) IF `tokenizer:encode`, `model:forward` or `tokenizer:decode` returns nil, THEN THE SYSTEM SHALL return nil and that line unchanged, the decode being either the one taken of an id as it is appended or the one taken of the whole list at the end.
- FR-011 (US-005) IF `tokenizer:encode` returns no ids, THEN THE SYSTEM SHALL return nil and one line naming `prompt`.

## Non-goals

- No route, command or config key: `serve` and the chat completions route are specs 17 and 18, and this call returns one finished string; streaming that string out as it is generated is spec 19.
- No sampler beyond the three shapes above: no top-p, no repetition or presence penalty, no stop strings, and no bias or logit list.
- No reuse of a model across calls beyond the options' `start`: a call forwards the prompt's ids from `start` on, and the caller still builds or keeps the model it passes.

## Tests

- T-001 (FR-001, FR-002, FR-004): `a prompt comes back as the text it sampled`, proves with a
  fixture whose metadata is `general.architecture` `qwen35moe` with the keys `block_count` 1,
  `embedding_length` 2, `attention.head_count` 1, `rope.dimension_count` 2, `rope.freq_base`
  10000, `attention.layer_norm_rms_epsilon` 0, `context_length` 4, `expert_used_count` 1 and
  `full_attention_interval` 1, holding the tokenizer keys `tokenizer.ggml.model` `gpt2`,
  `tokenizer.ggml.pre` `gpt2`, `tokenizer.ggml.tokens` the three pieces `a`, `b` and `c` with
  `tokenizer.ggml.token_type` of three 1s, `tokenizer.ggml.merges` an empty array and
  `tokenizer.ggml.eos_token_id` 1, and holding `15/T-001`'s tensors unchanged under `blk.0.` and
  at the root, except that `token_embd.weight` holds 2 numbers per row over 3 rows, `{1,0}`,
  `{3,4}` and `{1,0}`, that `compute.model(backend, handle, 4, "f32")` and `tokenizer.load(path)` each
  return their record and nil, that `generate.text(model, tokenizer, "a", {max_tokens = 2, seed =
  0})` returns `aa` and nil, the prefill giving `15/T-001`'s `{1.4142136, 0, 1.4142136}` and the
  greedy pick being id 0 there and after the second forward, so the prompt's own `a` is not in the
  result; that on a fresh model built with `positions` 2 `generate.text(model, tokenizer, "aa",
  {max_tokens = 1, seed = 0})` returns `a` and nil, the three-piece vocabulary encoding `aa` as
  two ids 0 standing at positions 0 and 1; and that on a fresh model built with `positions` 2
  `generate.text(model, tokenizer, "aaa", {max_tokens = 1, seed = 0})` returns nil and one line
  holding `blk.0.` and 2, a third prompt id finding no third position, that being `forward`'s
  second value unchanged.
- T-002 (FR-003, FR-005, FR-006): `a seeded draw takes the token temperature and top-k leave`,
  proves on T-001's fixture with `tokenizer.ggml.eos_token_id` removed that with `{max_tokens = 3,
  seed = 1000, temperature = 1, top_k = 2}`
  `generate.text(model, tokenizer, "a", options)` returns `ccc` and nil, ids 0 and 2 tying at
  1.4142136 as the two candidates at half each, the seed's first draw 0.6236204 above the first
  half, and id 2's row `{1,0}` repeating the scores; that with seed 0 the same options return
  `aac` and nil, the draws 0.2360680 and 0.2785669 falling below that half and the third
  0.8195338 above it; that with `{max_tokens = 1, seed = 1791, temperature = 1}` and no `top_k`
  it returns `b` and nil, the candidates being ids 0, 2 and 1 with the probabilities 0.4458083,
  0.4458083 and 0.1083834 and the draw 0.9301744 falling in the third; that with `{max_tokens =
  1, seed = 1791, temperature = 1, top_k = 5}` it returns `b` and nil, the three-piece vocabulary
  giving those same three candidates for a `top_k` above its size; that with `{max_tokens = 1,
  seed = 1791, temperature = 1, top_k = 2}` it returns `c` and nil, the same draw falling past the
  first half of two; and that with `{max_tokens = 2, seed = 1000, temperature = 0}` it returns
  `aa` and nil, the seed unused.
- T-003 (FR-007, FR-008): `generation stops at end of text or at the count`, proves on T-001's
  fixture with `tokenizer.ggml.eos_token_id` 0 that `generate.text(model, tokenizer, "a",
  {max_tokens = 4, seed = 0})` returns the empty string and nil, the first greedy sample 0 being
  the file's end id and never appended, so the count never stops it; that on the same fixture with
  `end_id = 1` in the options the call with `{max_tokens = 2, seed = 0, end_id = 1}` returns `aa`
  and nil, the given id overriding the file's 0; and that on T-001's fixture with
  `tokenizer.ggml.eos_token_id` removed the call with `{max_tokens = 2, seed = 0}` returns `aa`
  and nil, no end id stopping it before the count.
- T-004 (FR-003, FR-009, FR-010, FR-011): `a prompt and options that cannot be used are refused`,
  proves on T-001's fixture that `generate.text` with the prompt `""` and `{max_tokens = 2, seed =
  0}` returns nil and one line holding `prompt`; that `{max_tokens = 0, seed = 0}`,
  `{max_tokens = 2, seed = 0, temperature = -1}`, `{max_tokens = 2, seed = 0, top_k = 0}` and
  `{max_tokens = 2, seed = 0, start = -1}` each return nil and one line holding `generate` and, in
  turn, `max_tokens`, `temperature`, `top_k` and `start`; that `{max_tokens = 2, seed = 0, start =
  1}` on the one-id prompt `a` returns nil and one line holding `generate` and `start`; that with
  the prompt `"a\255"` and `{max_tokens = 0, seed = 0, temperature = -1}` the call returns nil and
  one line holding `generate` and `max_tokens`, the options checked in the order FR-009 names them
  before the prompt is encoded; that on a fresh model built with
  `positions` 2 the prompt `a` with `{max_tokens = 3, seed = 0}` returns nil and one line holding
  `blk.0.` and 2, the second sampled id being forwarded at a third position; that on a copy of
  T-001's fixture whose third piece stores the single byte `0xFF` with type 1 the call
  `{max_tokens = 1, seed = 1000, temperature = 1, top_k = 2}` returns nil and the line
  `tokenizer:decode` gives for `{2}`; and that the prompt `"a\255"` returns nil and a line equal
  to the one `tokenizer:encode` returns for that same text.

## Open questions

- None.

## Amendments

- None.
