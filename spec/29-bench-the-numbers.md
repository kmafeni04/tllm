# 29: Bench, the numbers before the changes

**Status**: done   **Depends on**: 02, 04, 08, 13, 15, 16, 18, 19, 24, 25

## Problem & context

`bench` is parsed and refuses: `tllm bench qwen3.8-27b` prints
`bench: not implemented (model qwen3.8-27b)` and exits 2. Nothing in the tree measures a token
rate, and the numbers that shaped the plan for this engine came from throwaway scripts: 189.75 ms
per sampled token for a full sort of 248,320 scores against 0.12 ms for a greedy scan, and 16 us of
alloc, build and free around every one of the ~1,250 ggml graphs a token runs. Those numbers decided
which work comes first, and nothing in the tree can reproduce them, so the next change is judged by
feel and a regression is invisible.

This spec makes `bench` measure. For a model it runs one streamed turn per prompt length on the
ladder, prints prompt tokens per second, decode tokens per second and the time to the first token,
and splits the decode time per token into compute, sampling and the request surface, with the
experts each token routed and the bytes those experts hold. `--dump-routing FILE` writes each
token's expert ids per layer, which is the input an expert cache policy has to be replayed against.
Nothing here changes what the engine answers, so every fixture reply in the suite stays what it is.

Four decisions shape it.

The turn goes through the engine's own request surface rather than calling the model directly. A
body is handed to `serve.stream`, which is what the `/v1/chat/completions` route calls (`18`,
`19`), and `serve.step` advances it a token per call, exactly as `src/worker.tl` drives a worker's
turn. That is what makes the request surface a measurable part of a token's cost: the time a frame
spends between the model finishing a token and the bench holding it is the figure the report calls
`ipc`. Bench drives no worker process, so that figure is the request surface's own cost, the event
encoding, framing and emit a client pays for before a worker's link is added on top. A bench that
called `generate.text` would report a cost no client pays at all.

Expert reads are counted, not timed. `weights.load` maps the whole file with `mmap`, so an expert's
bytes are fetched by the page fault inside the ggml call that reads them, and the fetch's time is
already inside compute's time. Splitting it out needs the expert tier that reads experts as blobs;
until then the honest numbers are how many experts a token routed and how many bytes those experts'
matrices hold.

The run sends `temperature` 0 and `seed` 0, so the same model and length twice gives the same
tokens, the same routing and the same expert bytes. That is what lets a difference between two runs
be attributed to a change rather than to a draw, and it is what makes a dumped routing file
replayable.

Everything bench needs to read is put on the model rather than kept in the bench: the clock
readings, the counters, and the routing sink. The alternative, a bench that wraps the model, would
have to be threaded through `serve`'s request path to be reached at all.

```teal
local record compute
  -- seconds and marks from the monotonic clock, in src/clock.tl
  record Measure
    passes: integer        -- tokens the model has passed, prompt and generated
    compute: number        -- seconds spent inside those passes
    sampling: number       -- seconds spent choosing a token from their scores
    ready: number          -- when the last token's choice finished
    routed: integer        -- experts those passes routed to
    routed_bytes: integer  -- the bytes those experts' three matrices hold
  end

  record Routing
    layer: integer
    token: integer         -- the pass count since `measure` was last zeroed
    experts: {integer}     -- ascending
  end

  record Model
    -- 15's fields, unchanged
    measure: Measure
    routing_sink: (function(Routing) | nil)
  end
end

local record bench
  record Settings
    path: string
    name: string
    lengths: {integer}
    decode: integer
    kv_cache: compute.CacheType | nil
    routing_path: string | nil
  end

  run: function(Settings, function(string), function(string)): integer
end
```

`measure` is a record the bench zeroes between lengths; nothing else writes it. `routing_sink` is
nil unless the caller sets it, and a pass hands it one record per layer, in the order the layers
run, only while it is held. `bench.run` writes its report lines through the first writer, its
refusal lines through the second, and hands back the exit code. The `generation` field of
`serve.Model` is `compute`'s model, and this is the first spec that reads it.

## User stories

- US-001 (P1) As the person deciding what to work on next, I want one command that reports prompt
  speed at the four lengths, decode speed and time to first token for a model, so that a change is
  judged by a number instead of a feeling.
- US-002 (P1) As the person deciding what to work on next, I want a token's time split into compute,
  sampling and the request surface, with the experts it routed and their bytes, so that the next
  change targets the largest part instead of the most talked about one.
- US-003 (P1) As the person writing an expert cache policy, I want every token's experts per layer
  on disk, so that a policy is replayed against a real conversation without running the model.
- US-004 (P2) As a person with a config file, I want bench to accept the model name I serve with, so
  that I measure the model I run without repeating its path.
- US-005 (P2) As the person running a change, I want the same model and length twice to route the
  same experts and hold the same bytes, so that a difference between two runs is my change.

## Functional requirements

- FR-001 (US-004) WHEN `tllm bench MODEL` runs, THE SYSTEM SHALL resolve MODEL as `serve` resolves
  one, a path when the word holds `/` or ends in `.gguf` and otherwise a name the config file
  holds, reading the file `--config` names and otherwise `04/FR-001`'s lookup paths without writing
  a starter file, and running on `04/FR-011`'s defaults when no file is found.
- FR-002 (US-001) WHEN that model is loaded, THE SYSTEM SHALL run one turn per length, ascending,
  the lengths being 512, 2048, 8192 and 32768, unless `--prefill-tokens N` is given, in which case
  the lengths are that one number.
- FR-003 (US-001, US-005) WHEN a length's turn runs, THE SYSTEM SHALL hand `serve.stream` a body
  holding `"stream":true`, one user message whose content is
  `The quick brown fox jumps over the lazy dog. ` repeated as many times as it takes for that
  content's own encoding to hold at least that length's count of ids, `max_tokens` set to
  `--decode-tokens` (128 when absent), `temperature` 0 and `seed` 0; SHALL resume
  the producer that call answers once with an `emit` that reads the clock at every payload it is
  handed and, for a payload holding a choice's `delta.content`, records that reading with the
  `measure` marks of that moment, the turn's last payload being the one holding that choice's
  `finish_reason`, then call `serve.step` until it returns 0; SHALL zero the
  `measure` of `serve.model_of(server).generation` before the turn and read that `measure` and its
  `routing_sink` there, and encode the filler with `serve.model_of(server).tokenizer`; and SHALL
  take a payload holding a choice's `delta.content` as a content frame, so the same model and
  length twice routes the same experts and reports the same expert count and bytes.
- FR-004 (US-001) WHEN a length's turn has run, THE SYSTEM SHALL write the line
  `prompt L: C tokens at R tok/s, first token M ms` and the line
  `decode L: C tokens at R tok/s`, where L is the length, the prompt C is the tokens passed up to
  and including the pass that produced the first content frame, read from `measure.tokens` as
  `33/FR-009` amends it and, when no content frame came, every token the turn passed, its R is that C over the seconds from handing the body to the first
  content frame and, when no content frame came, to the turn's end, its M is those seconds in
  milliseconds, the decode C is the content frames after the first, and its R is that C over the
  seconds from the first content frame to the last content frame, every number printed with one
  decimal except the counts and the length, and WHEN the decode C is 0 THE SYSTEM SHALL print its
  R as `0.0`.
- FR-005 (US-002) WHEN a length's turn produced at least one content frame after the first, THE
  SYSTEM SHALL write the line
  `token L: T ms, compute A ms, sampling B ms, ipc C ms` and the line
  `experts L: K routed, X MB; per turn R reads, Y MB read, S ms`, per token over those later
  frames, where T is the seconds from the first content frame to the last content frame over the
  decode count, A and B are the `measure` compute and sampling deltas between those two frames over
  that count, C is the sum over those frames of the clock when the frame reached the bench minus the
  `measure` ready mark at that moment over that count, K is the `measure` routed delta between the
  two frames over that count, and X is the `measure` routed bytes delta over that count divided by
  1000000; R, Y and S are the turn's `34/FR-003` blob reads, blob bytes divided by 1000000 and blob
  seconds in milliseconds, all read from the `measure` at the turn's end; T, A, B, C and S are each
  multiplied by 1000 and printed in milliseconds to three decimal places, K and R as whole numbers
  and X and Y to one decimal place, and the line SHALL end with `; H% hit`, H being the
  turn's `measure.expert_hits` over its `measure.routed` times 100 to one decimal, printed
  `0.0` when `routed` is 0, followed by the ahead text `38/FR-011` appends and then by the
  miss-policy text `39/FR-009` appends after the read-ahead text; and those three
  times it prints SHALL sum to within 20% of T,
  the two compared as printed.
- FR-006 (US-001) IF a length's body is refused with status `BadRequest`, THEN THE SYSTEM SHALL
  write the line `skipped L: M` where L is the length and M is what the refusal body holds, and
  run the next length; IF a length's body is answered with any other status than `OK`, or with a
  nil producer, THEN THE SYSTEM SHALL write one line naming the length and that status or the
  fact that no producer came back to the stderr writer, or one naming the length and the error a
  producer that raised on its first step reported, write no `prompt`, `decode`, `token` or
  `experts` line for it, and run the next length; and IF any length failed that way or no length
  ran, THEN THE SYSTEM SHALL exit 1, and exit 0 otherwise.
  Not testable: a producer that raises cannot be made from the command line, where `serve` is what
  builds it; a reader confirms `src/bench.tl` checks the result of resuming the producer's
  coroutine and writes the line it raises with.
- FR-007 (US-004) IF MODEL is neither a path nor a name the config file holds, or the model cannot
  be loaded, THEN THE SYSTEM SHALL write one line naming MODEL and the reason to the stderr writer
  and exit 1, having written no length's lines.
- FR-008 (US-003) WHEN `--dump-routing FILE` is given, THE SYSTEM SHALL set a routing sink on the
  model that collects one line per record, a JSON object holding the keys `layer`, `token` and
  `experts`, the first two whole numbers and the third the ids in ascending order, `token` being
  the token's index among the rows passed since `measure` was last zeroed so each length's records
  start at 0, SHALL truncate
  FILE once a length's body has been accepted and before that length's first pass, write that
  length's collected lines when its turn ends, so FILE ends up holding the run of the largest
  length that ran, and SHALL write no line and create no file when the flag is absent; IF FILE
  cannot be written, THEN THE SYSTEM SHALL write one line naming FILE to the stderr writer, run no
  further length, and exit 1.
- FR-009 (US-002) WHEN a model has made a pass, THE SYSTEM SHALL add that pass to its `measure`
  with the seconds it took and the number of ggml graphs it built, and add the rows the pass carried
  to `measure.tokens`, the field `33` declares beside `passes`, and no line of `bench`'s report and
  no count it prints changes, WHEN a token
  has been chosen from those scores, THE SYSTEM SHALL add
  that choice's seconds and its clock mark to `measure`, and WHEN a pass has routed its experts,
  THE SYSTEM SHALL add their count and the bytes their three matrices hold and, only while
  `routing_sink` is held, hand it one record per row of the pass per layer, so a run with no sink
  keeps no routing.
- FR-010 (US-002) THE SYSTEM SHALL read the monotonic clock as `ffi.C.clock_gettime` with
  `CLOCK_MONOTONIC`, filling the one two-long struct the module keeps rather than allocating one
  per read, in `src/clock.tl` and nowhere else, so every number `bench` prints comes from that one
  module. The struct carries the module's own tag because the `fs` dependency's `stat` already
  declares `timespec`.
  Not testable: a module, its allocation and where it is called, visible in the tree; the numbers
  it produces are asserted by T-001.
- FR-011 (US-004) THE SYSTEM SHALL build its server through `serve.build` with an endpoint whose
  `path` is the resolved model's path, whose `kv_cache` is the config file's value or `f16`, whose
  `sessions` is 1, whose `workers` is 1, whose `context_path` and `context_bytes` are nil, whose
  `host` is `127.0.0.1`, whose `port` is 8080, whose `ubatch` is the config file's `ubatch` or
  512 when it names none, and whose `vram` is the config file's `vram`, nil when it names none, and whose `profile` is
  the config file's `profile`, nil when it names none, and whose `prefetch` is the config file's
  `prefetch`, nil when it names none, and whose `threads` is the `--threads` value when given, then
  the config file's `threads`, then the default `44/FR-004` defines, so
  a bench run measures the tier a server would have, binds no socket, starts no worker process and
  creates no context directory.
  Not testable: `sessions`, `workers`, `context_path`, `context_bytes`, `host` and `port` exist
  only for serve's binding, spawning and context store, none of which a bench run uses, and
  `kv_cache` only shifts the numbers such a run prints, which no case pins; visible by hand,
  running `tllm bench m` under `ss -ltnp` and finding no listening socket, no child process and no
  new directory.

## Non-goals

- No expert read time and no read accounting: `mmap` makes the fetch part of compute's time, so
  only the count and the bytes are reported, and the read path stays as it is.
- No replay tool and no cache policy: the dump is written and nothing in this spec reads it.
- No pinned token rate: no case asserts a rate, so a machine that got slower cannot fail one; the
  20% of FR-005 is an accounting check on one run's own numbers, not a speed, and a case pins
  counts, line shapes and the routing, never tokens per second.

## Tests

Every case calls `cli.run(args, write_out, write_err)`, with an `environment` where a case needs
one, and with writers that collect lines, on a fixture built the way `serve`'s cases build one:
`13`'s one-block feed forward, two experts, one used, no
`tokenizer.ggml.eos_token_id` so a turn runs to its whole budget, a `tokenizer.chat_template` that
renders the message content alone, and `tokenizer.ggml.tokens` holding one token per character of
the filler text so the filler encodes to one id per character. The fixture's single block is the
only layer, so every count of layers here means that one block. T-001, T-003 and T-004 use that
fixture with `context_length` 260, T-002 and T-004 also with 4096.

- T-001 (FR-002, FR-003, FR-004, FR-005, FR-009): `a rung reports and adds up`, proves that
  `bench P --prefill-tokens 8 --decode-tokens 8` returns 0 with nothing on stderr and four lines on
  stdout matching `^prompt 8: (%d+) tokens at (%d+%.%d) tok/s, first token (%d+%.%d) ms$`,
  `^decode 8: 7 tokens at (%d+%.%d) tok/s$`,
  `^token 8: (%d+%.%d%d%d) ms, compute (%d+%.%d%d%d) ms, sampling (%d+%.%d%d%d) ms, ipc (%d+%.%d%d%d) ms$`
  and
  `^experts 8: (%d+) routed, (%d+%.%d) MB; per turn (%d+) reads, (%d+%.%d) MB read, (%d+%.%d%d%d) ms; (%d+%.%d)%% hit; (%d+) ahead, (%d+%.%d)%% landed; read ahead (%a+)$`,
  with the prompt count at least 8, the routed count 1 and compute above `0.000`, that the printed
  compute, sampling and ipc sum to within 20% of the printed total, that a second identical run
  returns 0 with the same prompt count and the same experts line apart from its milliseconds, and
  that on a copy whose `tokenizer.ggml.eos_token_id` is the
  id that copy's scores put first, `bench P --prefill-tokens 8 --decode-tokens 8` returns 0 with
  the `prompt 8:` line, a `^decode 8: 0 tokens at 0.0 tok/s$` line, and no `token 8:` and no
  `experts 8:` line.
- T-002 (FR-002, FR-006, FR-008): `a length the model cannot hold is skipped`, proves that with
  the default lengths on the 260-position fixture stdout holds `skipped 512:`, `skipped 2048:`,
  `skipped 8192:` and `skipped 32768:` lines each holding `positions`, no `prompt` line, and the
  call returns 1; that on the 4096-position fixture stdout holds `prompt 512:` and
  `prompt 2048:` as well as `skipped 8192:` and `skipped 32768:`, and the call returns 0; and that
  on a fixture with no `tokenizer.chat_template`, `--prefill-tokens 8 --decode-tokens 8` returns 1
  with one stderr line holding `8` and `InternalServerError`, no `prompt` line on stdout, and no
  file written to a `--dump-routing` path it was given.
- T-003 (FR-001, FR-007): `bench finds its model the way serve does`, proves that on a config file
  naming the fixture under `m`, `bench m --config F --prefill-tokens 8 --decode-tokens 8` returns
  0 with the `prompt 8:` line; that `bench P --prefill-tokens 8 --decode-tokens 8` given an
  `environment` naming two empty directories returns 0 with the `prompt 8:` line and leaves both
  directories empty, no `config.json` written; that `bench nosuch --config F --prefill-tokens 8`
  returns 1 with one stderr line holding `nosuch` and no line on stdout, and that a path holding a
  file that is not a GGUF returns 1 with one stderr line holding that path.
- T-004 (FR-008, FR-009): `dump-routing writes a line per layer per pass`, proves that
  `bench P --prefill-tokens 8 --decode-tokens 8 --dump-routing R` writes to R
  `prompt count + decode count` lines, one for layer 0 with `token` values 0 to
  `prompt count + decode count - 1`, each holding an `experts` list in ascending order whose every
  id is below the fixture's expert count; that on the 4096-position fixture the same call with the
  default lengths leaves R holding only the 2048 run, `prompt count + decode count` lines whose
  `token` values run 0 to that count minus one; that `--dump-routing` naming a file inside a
  directory that does not exist returns 1 with one stderr line naming that path and no `prompt`
  line; and that the same call without the flag leaves R absent.

`02/T-002`'s refusal case is gone with the amendment, applied before this spec was built: no call
exits 2 any more.

## Open questions

- None.

## Amendments

- None.
