# 30: Sampling that does not sort the vocabulary, and the options that narrow it

**Status**: pending   **Depends on**: 16, 18, 23, 25, 26, 29

## Problem & context

`generate` chooses a sampled token by ordering the whole vocabulary. `by_score` builds one integer
per vocabulary id and sorts all of them with a comparator closure. Measured on this machine over
248,320 scores: the full sort is 189.75 ms, a greedy argmax scan is 0.12 ms, a partial select of
the best 40 is 0.17 ms, and copying the row out of the tensor into Lua is 0.92 ms. The chat route
samples one token per step, so every reply asked for with a temperature pays that sort per token,
while a greedy reply pays nothing for it. The copy is worth keeping and the sort is not.

The sort buys two different things and only one of them is needed. With a `top_k` it buys the k
highest ids, which is a selection: the best 40 of a wide row in under a millisecond. With no
`top_k` it buys an order for the probabilities to be summed in, and that order is not needed at
all: a weighted draw is the same draw whatever order its cumulative sums follow. So no row is
ordered to pick a `top_k` and none is ordered to pick without one, the ids a `top_k` run draws stay
the ids it draws today, and a run with no `top_k` and no probability cut walks the ids in
increasing id order instead. Only `top_p` with no `top_k` still orders the whole row, a prefix of
probabilities being what it asks for, which is why FR-003's bound leaves that case out.

The rest of a sampler's options do not exist anywhere in the engine: no `top_p`, no `min_p`, no
penalty for what the run already said, and the route ignores every member it does not know,
`18/FR-005`, so a client cannot even ask for them. The selection and the options that narrow it are
one problem: which candidates a draw chooses between and what their scores are, computed from the
one score row the pass already returns.

One order, applied to every sample. The penalties of FR-006 change the scores first. A temperature
absent or zero then takes the highest score so changed. Otherwise the candidates are taken, `top_k`
by FR-001 or the whole row by FR-002, `top_p` narrows them by FR-004 and `min_p` by FR-005, and one
draw of the generator `16/FR-006` states walks what is left. `top_p`, `min_p` and `top_k` are
ignored by a greedy run: its token is the highest score, which is `16/FR-004` after the penalties.

Choosing a token with no `top_k` and no probability cut costs three walks of the row and no
ordering, which is FR-002's shape: one walk takes the highest score, one takes the sum of the
exponentials and one walks the cumulative sum the draw falls in.

One recorded reply changes: `16/T-002`'s clause that asks for
`{max_tokens = 1, seed = 1791, temperature = 1}` reads `c` rather than `b`, the ids being walked in
increasing id order now. Every other recorded reply in the suite stands: `23/T-001`'s body carrying
`"temperature":1,"seed":7` takes `"seed":2` in place of 7, its draws becoming 0.2368431 and
0.4599744 which take id 0 and then the end id 1 under the new order, so its ids, counts and lengths
stand and only its seed and those two draws change; and every reply taken with a `top_k` keeps its
ids.

```teal
local record generate
  record Options
    max_tokens: integer
    seed: integer
    temperature: number | nil
    top_k: integer | nil
    top_p: number | nil
    min_p: number | nil
    repetition_penalty: number | nil
    presence_penalty: number | nil
    frequency_penalty: number | nil
    end_id: integer | nil
    start: integer | nil
  end
end
```

The record `serve` builds from a body is unexported, so those five members join it in the code and
no snippet gains a field: `18`'s wording about the members it ignores and the shape it refuses is
what names them, and the amendments below carry it.

`16`'s non-goal "No sampler beyond the three shapes above: no top-p, no repetition or presence
penalty, no stop strings, and no bias or logit list" no longer holds: FR-004, FR-005 and FR-006 are
the top-p, min-p and penalty options it rules out, and its problem section's "Sampling has three
shapes and no more" now reads as those three shapes plus these. A non-goal is not something an
amendment entry can name, so it is said here.

The bounds below were checked against a plain LuaJIT probe on this machine: a select over 200,000
scores took about 0.9 ms and a three-pass walk over the same row about 0.7 ms, against roughly
150 ms for today's sort of that row. FR-003 pins that gap with room for a slower day.

## User stories

- US-001 (P1) As the person asking for a varied reply, I want each sampled token chosen in
  milliseconds rather than hundreds, so that a reply with a temperature is not a hundred times
  slower than a greedy one.
- US-002 (P1) As the person asking for a reply, I want to narrow the candidates by probability and
  not only by count, so that a long tail of unlikely tokens cannot be drawn.
- US-003 (P1) As the person asking for a reply, I want to discourage the tokens the run already
  said, so that it stops repeating itself.
- US-004 (P2) As a client of the chat route, I want to send those options in the body and be told
  which one is wrong, so that a bad value is a refused request and not a surprise.

## Functional requirements

- FR-001 (US-001) WHEN a run samples with a temperature above zero and a `top_k`, THE SYSTEM SHALL
  take the candidates as the `top_k` highest-scoring ids, fewer when the vocabulary is smaller,
  ordered by descending score with equal scores in increasing id order, and the sampled token as
  the first candidate whose probability summed with every candidate before it exceeds the draw,
  the last candidate when none does, each probability being the candidate's score minus the highest
  candidate's score, divided by the temperature, exponentiated, and divided by the sum of the
  candidates' exponentiated numbers, so the id a seed draws is the id it drew before this change
  and no row wider than those candidates is ordered.
- FR-002 (US-001) WHEN a run samples with a temperature above zero and none of `top_k`, `top_p` and
  `min_p`, THE SYSTEM SHALL take the candidates as every id of the vocabulary in increasing id
  order, and the sampled token as the first id whose probability summed with every id before it
  exceeds the draw, the last id when none does, each probability being that id's score minus the
  highest score, divided by the temperature, exponentiated, and divided by the sum of every id's
  exponentiated number, and the row is never ordered.
- FR-003 (US-001) WHEN a token is chosen from a row of 200,000 scores, THE SYSTEM SHALL spend under
  50 ms choosing it when the options carry `top_k` 40 and under 100 ms choosing it when they carry
  none of `top_k`, `top_p` and `min_p`, that time being the gain the model's `measure.sampling`
  takes over one token in the seconds it counts, and SHALL order no more ids than the candidates
  the options leave.
- FR-004 (US-002) WHEN the options carry a `top_p` and a temperature above zero, THE SYSTEM SHALL
  order the candidates the `top_k` leaves, or every id when no `top_k` is given, by descending
  score with equal scores in increasing id order, keep the shortest prefix of that order whose
  probabilities sum to `top_p` or more, keep the first candidate whatever `top_p` is, and draw from
  among the kept candidates with their probabilities renormalised over them.
- FR-005 (US-002) WHEN the options carry a `min_p` and a temperature above zero, THE SYSTEM SHALL
  take as its candidates the candidates `top_p` leaves, or the candidates the `top_k` leaves when no
  `top_p` is given, or every id of the vocabulary in increasing id order when neither is given, keep
  only the candidates whose probability is at least `min_p` times the highest probability among
  them, keep that candidate whatever `min_p` is, leave the kept candidates in the order they already
  have, and draw from among them with their probabilities renormalised over them.
- FR-006 (US-003) WHEN the options carry `repetition_penalty`, `presence_penalty` or
  `frequency_penalty`, THE SYSTEM SHALL change the score of every id the run holds before the
  candidates are taken, in this order: divide that id's score by `repetition_penalty` when the
  score is positive and multiply it by `repetition_penalty` when the score is negative, both when
  `repetition_penalty` is given, then subtract `presence_penalty` once when it is given, then
  subtract `frequency_penalty` once for each time that id appears when it is given, the run's ids
  being the ids the prompt holds from `start` on and every id the run has produced, a
  `repetition_penalty` of 1 or a `presence_penalty` or `frequency_penalty` of 0 changing nothing,
  and a run with a temperature absent or zero taking the highest score so changed, the lower id
  among equals.
- FR-007 (US-004) WHEN the route's body holds `top_p`, `min_p`, `repetition_penalty`,
  `presence_penalty` or `frequency_penalty`, THE SYSTEM SHALL pass it to the generation as given.
- FR-008 (US-004) IF `top_p` is given and is not a number or is zero or below or is above one, or
  `min_p` is given and is not a number or is below zero or above one, or `repetition_penalty` is
  given and is not a number or is not above zero, or `presence_penalty` or `frequency_penalty` is
  given and is not a number, THEN THE SYSTEM SHALL refuse it with one line naming that member,
  `generate.open` returning nil and a line naming `generate` and that member, after `16/FR-009`'s
  own option checks and before the prompt is encoded, in the order this requirement names them, and
  the route answering `BadRequest` as `18/FR-008` describes it.

## Non-goals

- No other sampler: no bias or logit list, no typical or mirostat sampling, and no change to the
  generator, the seed or the end id of `16`.
- No stop string and no other change to the text a reply releases: a run still answers each sampled
  token's own text as `19/FR-004` says, and cutting a reply at a string is the spec that follows.
- No penalty for a resumed conversation's earlier ids: a run holds the ids its own prompt carries
  from `start` on and the ids it produced, and nothing before `start`.

## Tests

`16/T-001`'s fixture is used by T-002 to T-004 with four pieces `a`, `b`, `c` and `d` in place of
three, `token_embd.weight` holding four rows of `{1, 0}` so every prompt gives the same hidden
numbers, `output.weight` holding the four rows `{0, 0}`, `{1, 0}`, `{2, 0}` and `{3, 0}` so the four
scores are 0, 1.4142136, 2.8284271 and 4.2426407 with ids 0 to 3, the probabilities 0.0109142,
0.0448930, 0.1846561 and 0.7595367 at temperature 1, and `tokenizer.ggml.eos_token_id` removed; a
greedy sample there is always id 3, the piece `d`. T-001 uses a wider copy of that fixture built by
loop, its `tokenizer.ggml.tokens` holding `a`, `b`, `c` and then `tostring(id)` for each id from 3 to
199,999, its `token_type` holding 200,000 ones, and both `token_embd.weight` and `output.weight`
holding 200,000 rows of `{1, 0}`, so every score is 1.4142136 and every id is a candidate at the top.
`23/T-001`'s body carrying `"temperature":1,"seed":7` takes `"seed":2` in place of 7 and records the
draws 0.2368431 and 0.4599744 in place of 0.2387808 and 0.9134933 for the reason the problem section
gives, and every other clause of every other spec's case stands.

- T-001 (FR-001, FR-003): `a wide row is sampled without ordering it`, proves on the 200,000-id
  fixture, with the model's `measure.sampling` read back after the call, that
  `generate.text(model, tokens, "a", {max_tokens = 1, seed = 0, temperature = 1, top_k = 40})`
  returns a `Text` whose `sampled` is `{9}` and leaves `measure.sampling` under 0.05, the candidates
  being ids 0 to 39 with the seed's draw 0.2360680 falling in the tenth of forty, so equal scores
  are taken in increasing id order and the wide row is never ordered; and that on a fresh model the
  same call with no `top_k` returns a `Text` whose `sampled` is `{47213}` and leaves
  `measure.sampling` under 0.10, the draw falling in the 47,214th of 200,000.
- T-002 (FR-002): `a run with no top_k walks the ids in order`, proves on the four-piece fixture
  that `generate.text(model, tokens, "a", {max_tokens = 1, seed = 2178, temperature = 1})` returns
  a `Text` whose `sampled` is `{2}`, the draw 0.0801572 falling in id 2's band of the increasing id
  order, where ordering the four scores would give id 3; and that with seed 1946 the same call
  returns `{3}`, the draw 0.9902450 past id 2's band and short of the last, where the descending
  order would give id 0.
- T-003 (FR-004, FR-005): `top-p and min-p narrow the candidates`, proves on the four-piece fixture
  that `{max_tokens = 1, seed = 1946, temperature = 1, top_k = 4}` gives `{0}`, and that
  `{max_tokens = 1, seed = 1946, temperature = 1, top_k = 4, top_p = 0.8}` gives `{2}`, the kept
  candidates being ids 3 and 2 with probabilities 0.8044297 and 0.1955703 renormalised, so the draw
  0.9902450 lands on the second; that `{max_tokens = 1, seed = 1946, temperature = 1, top_k = 4,
  top_p = 0.01}` gives `{3}`, the first candidate kept whatever `top_p` is; that
  `{max_tokens = 1, seed = 1791, temperature = 1, top_p = 0.8}` gives `{2}`, the whole row ordered
  by descending score before the prefix is taken, where the same call without `top_p` gives `{3}`
  as T-002's rule walks the ids in order; that `{max_tokens = 1, seed = 1868, temperature = 1,
  top_k = 4, min_p = 0.1}` gives `{2}`, ids 1 and 0 falling below 0.1 of the highest probability
  0.7595367 where the same call without `min_p` gives `{1}`; and that `{max_tokens = 1, seed =
  1946, temperature = 1, top_k = 4, min_p = 1}` gives `{3}`, the first candidate kept; and that
  `{max_tokens = 1, seed = 2178, temperature = 1, min_p = 0.5}` gives `{3}`, the threshold being
  0.3797683 over the highest probability 0.7595367, so only id 3 survives, where measuring against
  the first id of the increasing id order would threshold at 0.0054571, keep all four and draw
  `{2}` as T-002's rule does.
- T-004 (FR-006): `the penalties steer off what the run already holds`, proves on the four-piece
  fixture, every call greedy and taking one token, that the prompt `d` with no penalty samples
  `{3}`; that with `repetition_penalty = 1.6` the prompt `d` and the prompt `dd` both sample `{2}`,
  the score divided once; that with `presence_penalty = 1` the prompts `d` and `dd` both sample
  `{3}`, the score lowered once; that with `frequency_penalty = 1` the prompt `d` samples `{3}` and
  the prompt `dd` samples `{2}`, the score lowered once per appearance; and that
  `{max_tokens = 1, seed = 1791, temperature = 1, top_k = 2, presence_penalty = 5}` on the prompt
  `d` samples `{1}`, id 3's score 4.2426407 lowered to -0.7573593 leaving ids 2 and 1 as the two
  candidates with probabilities 0.8044297 and 0.1955703, so the penalty lands before the candidate
  cut; that `{max_tokens = 1, seed = 0, repetition_penalty = 0.5, presence_penalty = 3}` on the
  prompt `d` samples `{3}`, id 3's score becoming 5.4852814 when the division comes first where
  subtracting first would leave it 2.4852814 below id 2's 2.8284271; and that on a copy of the
  fixture whose `output.weight` holds `{-1, 0}`, `{-2, 0}`, `{-4, 0}` and `{-8, 0}`, its four
  scores being -1.4142136, -2.8284271, -5.6568542 and -11.3137084,
  `{max_tokens = 1, seed = 0, repetition_penalty = 4}` on the prompt `a` samples `{1}`, id 0's
  -1.4142136 multiplied by 4 to -5.6568542 leaving id 1's -2.8284271 the highest, where dividing
  it would leave -0.3535534 and sample `{0}`.
- T-005 (FR-007, FR-008): `the route carries the narrowing options and refuses bad values`, proves
  on `18/T-001`'s fixture, whose three scores are `16/T-001`'s `1.4142136`, 0 and `1.4142136`, whose
  probabilities at temperature 1 are 0.4458083, 0.1083834 and 0.4458083 and whose end id is 1, that
  the body `{"messages":[{"role":"user","content":"a"}],"max_tokens":1}` answers `OK` with the
  content `a`, and that each of these bodies answers `OK` with a content of its own:
  `"temperature":1,"seed":1791` gives `c`, the draw 0.9301744 falling in id 2's band of the
  increasing id order, while the same body with `"top_p":0.01` gives `a`, only the first candidate
  kept; `"temperature":1,"seed":682` gives the empty content, the draw 0.5003787 falling on the
  file's end id, while with `"min_p":0.5` it gives `c`, id 1 dropped for falling below half the
  highest probability 0.4458083; `"presence_penalty":2` gives `c`, id 0's score 1.4142136 lowered
  below id 2's, and so does `"repetition_penalty":2`; and the body whose prompt is `aa` with
  `"frequency_penalty":1` gives `c`, id 0's score lowered once per appearance, as does
  `"presence_penalty":2` on that prompt; and that bodies carrying `"top_p":"x"`, `"min_p":"x"`,
  `"repetition_penalty":"x"`, `"top_p":0`, `"top_p":1.5`, `"min_p":-1`, `"min_p":1.5`,
  `"repetition_penalty":0`, `"presence_penalty":"x"` and `"frequency_penalty":"x"` each answer
  `BadRequest` as `18/FR-008` describes it, its message holding the member at fault; and that
  `generate.open` with `{max_tokens = 1, seed = 0, top_p = 0}` returns nil and one line holding
  `generate` and `top_p`; and that `generate.open` with
  `{max_tokens = 0, seed = 0, top_p = 0}` returns nil and one line holding `max_tokens`,
  `16/FR-009`'s own checks running first, and that with the prompt `a\255` and
  `{max_tokens = 1, seed = 0, top_p = 0}` it returns nil and one line holding `top_p` rather than a
  line from `tokenizer:encode`, the new checks running before the prompt is encoded.

## Open questions

- None.

## Amendments

- `16/FR-004`: the highest score a greedy run takes is the highest score after `30/FR-006`'s
  penalties.
- `16/FR-005`: the candidates are the `top_k` highest-scoring ids ordered by descending score with
  equal scores in increasing id order, or, when `top_k` is absent, every id of the vocabulary in
  increasing id order, and the draw walks whichever order that is; the reply `16/T-002` records for
  `{max_tokens = 1, seed = 1791, temperature = 1}` becomes `c`.
- `16/FR-009`: its checks gain `30/FR-008`'s, run after the `start` check and before the prompt is
  encoded.
- `18/FR-005`: `top_p`, `min_p`, `repetition_penalty`, `presence_penalty` and `frequency_penalty`
  leave the ignored members and are honoured as `30/FR-007` says.
- `18/FR-006`: it passes those five members to the generation as well as `temperature`, `top_k` and
  `seed`.
- `18/FR-008`: its refusals gain `30/FR-008`'s, checked after the `session_id` check in the order
  `30/FR-008` names them.
