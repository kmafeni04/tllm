# 55: A reference answer from the engine that already runs this file

**Status**: done   **Depends on**: 02, 04, 16, 29, 41

## Problem & context

Nothing that comes after this can be checked for correctness. The fixtures in `12`, `13`, `15` and
`16` assert against numbers computed by hand from the same reading of the file that the code
implements, so a wrong reading of `qwen4exp` produces code and a test that agree with each other and
disagree with the model. The five `qwen4exp` mechanisms still to be built (rope sections,
hyper-connections, the ngram tables, the indexer's own cache and its mask) are five readings this
engine has never seen produce a number, and three of the ways they go wrong are answers rather than
refusals: a block that builds, rotates one band over 64 dimensions instead of three, collapses four
residual streams into one, and prints tokens.

This spec puts an answer from outside the tree under the work. One engine already runs this exact
file on this machine: strata, serving
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` as `qwen3.8-flash-next-iq3_s`. Its logprob
lines carry the chosen token's id, its natural-log probability, and the runner-up ids with theirs,
so a recording of what it answers is a set of ids and numbers, not a text a reader squints at. The
checkpoint each of those mechanisms ends on becomes "the same ids come back", not "it answered".

Three things make the recording usable rather than decorative.

It is taken with strata's speculation off, so every position has a head of its own and no position is
missing a distribution, and it is taken greedy at a fixed seed, so the ids are a property of the
weights and not of a draw.

It stores the prompt as the token ids the recording engine was fed, template included, and not as text
alone. The comparison is then about the maths, and not about whether `10`'s encoder and another
engine's agree about a boundary. The text stays in the recording for the human.

It names the file it came from by hashing both shards with sha256. A checkpoint run against the wrong
weights is a false pass, and this file has three quantisations and a 27B sibling on this machine. The
hash is read through each shard's own descriptor in chunks, not through the model's mapping, so an
84 GB pair hashes without a copy; on this NVMe that is tens of seconds, which is noise next to a run
of the model. The hash itself is one FFI call into `libcrypto`'s EVP interface, in one module.

The recording also carries, for the same three prompts, that engine's cold and steady token rates
from the same run. That is the number every entry after this one is argued against, and it is on the
screen next to this engine's own rather than remembered.

The recording's shape:

```json
{
  "engine": "strata qwen3.8-flash-next-iq3_s, speculation off",
  "date": "2026-12-01",
  "sampling": { "temperature": 0, "seed": 0, "max_tokens": 50 },
  "tolerance": 0.01,
  "model": {
    "name": "Qwen3.8-Flash-Next GSQ-RCO",
    "shards": [
      { "bytes": 54817524224, "sha256": "…" },
      { "bytes": 28800138432, "sha256": "…" }
    ]
  },
  "prompts": [
    {
      "prompt": "…text, for the human…",
      "prompt_ids": [ 1282, 306, 3410 ],
      "ids": [ 50 ids the recording engine answered ],
      "top": [
        { "position": 0, "five": [ { "id": 58, "logprob": -0.31 }, 5 entries ] }
      ],
      "cold_tokens_per_second": 12.4,
      "steady_tokens_per_second": 41.2
    }
  ]
}
```

`five` is the recording engine's five highest-probability candidates at that position, most likely
first, the chosen one among them. `position` counts the ids it answered, from 0.

## User stories

- US-001 (P1) As the person porting a model family, I want one command that answers the same three
  prompts the engine that already runs this file answered, so that I know the port is right and not
  merely fast.
- US-002 (P1) As the person running that command, I want it to refuse when the recording does not
  belong to the file I pointed at, so that a checkpoint cannot pass against the wrong weights.
- US-003 (P1) As the person who gets a disagreement, I want the first position where the answers part
  and both values there, so that I know which reading of the file to go and look at.
- US-004 (P2) As the person deciding what to optimise, I want the recording engine's cold and steady
  speed for the same prompts printed beside this engine's, so that the number to beat is on the
  screen and not in memory.

## Functional requirements

- FR-001 (US-001) WHEN `tllm reference MODEL --reference FILE` runs, THE SYSTEM SHALL resolve MODEL
  as `29/FR-001` resolves one, taking `--config` as `04/FR-001` and `04/FR-002` do, build the model
  through `serve.build` with the endpoint `29/FR-011` describes (one session, one worker, no socket,
  no worker process, no context directory), and read FILE as the recording.
- FR-002 (US-001, US-002) WHEN it reads a recording, THE SYSTEM SHALL take from it the engine and the
  date that recorded it, `sampling` with `temperature`, `seed` and `max_tokens`, a `tolerance`,
  `model` with a `name` and one `shards` entry per shard holding `bytes` and `sha256`, and `prompts`,
  each holding `prompt` text, `prompt_ids`, `ids`, `top` as one entry per position holding a
  `position` and a `five` of entries holding an `id` and a `logprob`, and
  `cold_tokens_per_second` and `steady_tokens_per_second`.
- FR-003 (US-001) THE SYSTEM SHALL keep the recording for the Flash-Next pair at
  `reference/qwen3.8-flash-next-iq3_s.json` at the package root, holding three prompts, each with the
  ids that engine answered within its `max_tokens` of 50 (one prompt ends at its own stop id, so it
  holds 30) and the five candidates at each of the first five positions, recorded greedy at seed 0
  with speculation off on this machine.
  Not testable: it is one run of the real 84 GB pair on the card strata already holds; the shape it
  holds is what FR-002 and T-001 assert.
- FR-004 (US-002) WHEN the model is loaded, THE SYSTEM SHALL hash every shard of the loaded handle
  with sha256 and compare the count and each hash against the recording's `shards` entries in order,
  and IF the count differs or any hash differs, THEN THE SYSTEM SHALL write one line naming that
  shard's path, the recorded hash and the hash the file has, and return exit code 1 having run no
  turn.
- FR-005 (US-001) WHEN the recording matches the loaded model, THE SYSTEM SHALL run one turn per
  prompt in the recording, in the recording's order, handing that prompt's `prompt_ids` to
  `generate.open_ids` with `temperature`, `seed` and `max_tokens` taken from the recording's
  `sampling`, and collect the ids that run samples.
- FR-006 (US-001, US-003) WHEN a turn has run, THE SYSTEM SHALL compare its sampled ids with the
  recording's `ids` position by position, and at each position the recording's `top` names, rank the
  scores that turn handed back for the row whose id the run chose (higher value first, the lower id
  among equal values, as `30`'s sampler ranks) and compare that ranking's five ids, and each one's
  natural-log probability over the whole vocabulary, with the recorded five.
- FR-007 (US-003) IF a position's sampled id differs from the recorded one, or a position's five ids
  are not the recorded five in the recorded order, or a recorded `logprob` and this engine's for the
  same id differ by more than the recording's `tolerance`, THEN THE SYSTEM SHALL write one line
  naming that prompt's number, that position, and the recorded value and this engine's value, and
  write no further line for that prompt's later positions.
- FR-008 (US-004) WHEN a prompt's turn has run, THE SYSTEM SHALL write one line holding the count of
  agreeing ids over the recorded count, the count of agreeing `top` positions over the recorded
  count, and that prompt's recorded `cold_tokens_per_second` and `steady_tokens_per_second` beside
  the cold and steady rates this run measured for it, every rate printed with one decimal.
- FR-009 (US-001) WHEN every prompt's ids and positions agree, THE SYSTEM SHALL write
  `reference: PASS` and return exit code 0, and IF any prompt differs, THEN THE SYSTEM SHALL write
  `reference: FAIL` and return exit code 1.
- FR-010 (US-002) IF FILE cannot be read, is not a JSON object, or misses any field FR-002 names; or
  the recording holds an id at or above the loaded model's vocabulary count; or a prompt's
  `prompt_ids` count is above the loaded model's positions, THEN THE SYSTEM SHALL write one line
  naming FILE, the missing key, the id, or the count, and return exit code 1 having run no turn.

## Non-goals

- No recording tool: this engine never talks to another engine, and the recording is taken outside
  this tree and committed as data.
- No model-key flags for `reference`: `--kv-cache`, `--ubatch`, `--vram`, `--profile` and
  `--prefetch` come from the resolved config, so the checkpoint runs on the numbers the config gives
  that model.
- No agreement measure beyond the recording: no perplexity, no distribution past the recorded five
  candidates, and no tolerance applied to ids.

## Tests

Every case calls `cli.run(args, write_out, write_err)` with writers that collect lines, on the
one-block, two-expert fixture `29`'s cases build (no `tokenizer.ggml.eos_token_id`, one token per
character of the filler), beside a recording written as JSON into a temp file. A recording's
`shards` entry holds the fixture file's own byte count and its sha256, and its `ids` and `top` hold
what that fixture actually answers.

- T-001 (FR-001, FR-002, FR-005, FR-006, FR-008, FR-009): `a recording that agrees answers PASS`,
  proves that `reference P --reference R` returns 0, writes a header line holding the recording's
  engine and date, one `prompt 1:` line holding `N of N ids`, `M of M positions` and the recording's
  two recorded rates, and a final `reference: PASS` line, and that the same call on a recording whose
  `prompt` text is a string the fixture's tokenizer cannot encode still returns 0, because the turn is
  fed `prompt_ids`.
- T-002 (FR-006, FR-007): `a differing id fails the checkpoint`, proves that a recording whose first
  recorded id differs from the fixture's returns 1 with one line holding `prompt 1`, `position 0`,
  the recorded id and the engine's id, plus a `reference: FAIL` line and no second difference line for
  that prompt; and that the same recording with that id corrected returns 0.
- T-003 (FR-006, FR-007): `the five candidates at a position are compared`, proves that a recording
  holding the fixture's five ids in another order returns 1 with one line naming that position and
  both lists; that one holding them in order with one `logprob` moved by more than the tolerance
  returns 1 with one line naming that id and both values; and that the same value moved by less than
  the tolerance returns 0.
- T-004 (FR-004): `a recording that names another file is refused`, proves that a recording whose
  `sha256` is not the loaded fixture's returns 1 with one line holding the fixture's path and both
  hashes and no `prompt` line, and that a recording naming two `shards` for that one-shard fixture
  returns 1 with one line holding both counts and no `prompt` line.
- T-005 (FR-002, FR-010): `a recording is checked before a turn runs`, proves that a recording with
  `top` removed, one with `sampling` removed, one holding an id at or above the fixture's vocabulary
  count, and one whose `prompt_ids` run past the fixture's positions each return 1 with one line
  naming `top`, `sampling`, that id, and that count, and that none writes a `prompt` line.
- T-006 (FR-001): `reference finds its model and its recording`, proves that on a config file naming
  the fixture under `m`, `reference m --config F --reference R` returns 0 with the `prompt 1:` line;
  that `--reference` naming an absent file returns 1 with one line holding that path; and that
  `reference --help` returns 0 with the usage text.
- T-007 (FR-009): `a short answer fails the checkpoint`, proves that a recording holding more `ids`
  than its own `max_tokens` lets this engine answer, with every answered id and every recorded
  position agreeing, returns 1 with one `prompt 1:` line holding `2 of 4 ids` and `2 of 2 positions`
  and a `reference: FAIL` line, and writes no difference line.

`reference/qwen3.8-flash-next-iq3_s.json` itself is not a case: it is one run of the real file, and
the specs that build those mechanisms are the ones that run it.

## Open questions

- None.

## Amendments

- None.
