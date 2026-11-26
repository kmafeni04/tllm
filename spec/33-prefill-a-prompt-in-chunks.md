# 33: Prefill a prompt in chunks

**Status**: done   **Depends on**: 04, 12, 14, 15, 16, 18, 19, 25, 26, 28, 29, 32

## Problem & context

A prompt is fed one token at a time. `generate.text_ids` calls `model:pass({input})` once per id
(`src/generate.tl:659`) and `serve.step` takes one row per turn per step, so a prompt of 4,096 ids is
4,096 passes: `3 * N + 2` graphs each (`32/FR-001`), every one of them reading the whole run's
weights to multiply them into a single row. That is the largest cost of a request's first token, and
it is the number `29` exists to make visible.

The batching is already built and nothing uses it. `26` let a pass hold rows, `32` built every block
over the batch a pass holds, and `26/FR-009` refuses two rows naming one sequence, so no pass has
ever held two rows of one conversation. That refusal protects one thing: `26/FR-010` holds a row to
its own sequence's positions zero to its own, and with two rows of one sequence in one graph the
earlier row would attend over the later one unless the scores past its own position are held out of
its softmax.

This spec makes a chunk of one conversation's consecutive tokens one pass. The chunk costs one graph
per block (`32`) and one read of each weight, and a 4,096 id prompt reaches its first token after
eight passes at `ubatch` 512 instead of 4,096. The pieces it needs:

- the rows of one sequence in a pass become a group of consecutive positions running up from the
  positions that sequence has taken, which is the only shape that keeps `26/FR-010`'s bound;
- an attention row's scores past its own position are held out of its softmax because the row reads
  the cache only over its sequence's positions zero to its own, whether those positions were written
  by an earlier pass or by an earlier row of the same pass, whose key and value the graph has already
  written by then;
- a linear block's convolution window and recurrent state are threaded through the graph from row to
  row and written back once at the group's end, because two rows of one graph reading and writing one
  host state have no order ggml promises them;
- the driver: `Run.next` is told how many rows it may hand and returns a list, `generate` asks for
  `ubatch` rows at a time, and `serve.step` splits `ubatch` among the live turns, each turn taking its
  whole share and at least one row;
- the knob: `ubatch` in `config.json`, 512 when unset, carried to the model and to bench, and to a
  worker by the endpoint json `28/FR-003` defines, whose decoded fields gain it.

Two numbers decide what the knob is worth and what it costs. A chunk removes work that is not
arithmetic: at 512 rows a block's weights are read once per pass instead of 512 times, which is the
gain on a bandwidth-bound run. And a chunk's block graph holds every row's attention scores: rows
times positions times heads times 4 bytes, about 335 MB for 512 rows at 4,096 positions on 40 heads
with `f32` scores. So a model's arena is grown for the rows a pass actually carries rather than sized
for the bound when the model is built (`32/FR-005` is amended to say so), and `ubatch` bounds a pass
rather than sizing one. With `workers` above one each worker builds a model of its own (`28/FR-003`),
so that arena and whatever a wide pass grows it into are counted once per worker. Sizing a chunk from
what the machine has free is a later spec's, and flash attention, which is what makes those scores
small, is nobody's.

The answer is not bit identical and this spec does not claim it is: a product over several rows picks
a different kernel and sums in a different order. What is promised is 1e-4 against the same tokens
forwarded one at a time, and the same text, ids, events and routing on every fixture. One count moves
with it: `29`'s prompt rate reads `measure.passes`, which was tokens only while a pass was one token,
so `measure` gains `tokens` and the routing dump keys on the token rather than the pass, which is
what keeps one record per token per layer.

```teal
local record compute
  -- supersedes the seven-field record `32` declares, which superseded `29`'s six
  record Measure
    passes: integer        -- passes made since `measure` was last zeroed
    tokens: integer        -- the rows those passes carried
    compute: number
    sampling: number
    ready: number
    routed: integer
    routed_bytes: integer
    graphs: integer
  end

  record Model
    -- 26's fields, unchanged
    ubatch: integer
  end

  model: function(
    backend: Backend,
    handle: weights.Handle,
    positions: integer | nil,
    cache_type: CacheType | nil,
    sequences: integer | nil,
    ubatch: integer | nil
  ): Model | nil, string | nil
end

local record generate
  record Run
    -- 25's other fields, unchanged
    next: function(Run, integer): {compute.Input} | nil
  end
end

local record serve
  record Endpoint
    -- 28's fields, unchanged
    ubatch: integer | nil
  end
end
```

`Run.next`'s argument is the most rows the caller will take in one pass, one or more, and a caller
with room for no row does not ask. A group is the rows one run hands for one sequence in one pass, and
a pass is the groups of every turn it holds, each group consecutive in the list. `Model.ubatch` is the
bound a caller asks `Run.next` for: `model:pass` takes whatever rows it is handed and enforces no
bound of its own.

## User stories

- US-001 (P1) As the engine author, I want one pass to carry a chunk of one conversation's
  consecutive tokens, so that a prompt reads each weight a handful of times instead of once per
  token.
- US-002 (P1) As the engine author, I want every row of a chunk to see exactly the positions before
  itself and no position after it, so that a chunk gives the numbers the same tokens give one at a
  time.
- US-003 (P1) As a person sending a long prompt, I want the server to prefill it in chunks, so that
  the first token does not wait for the whole prompt to walk the model one token at a time.
- US-004 (P2) As the person running the server, I want the chunk bound in `config.json`, so that I
  can trade the memory a wide chunk costs against a number I chose.
- US-005 (P1) As the engine author, I want the reply, the sampled ids, the events and the bench
  figures to say the same thing they say today, so that this is a speed change and not an answer
  change.

## Functional requirements

- FR-001 (US-001) WHEN `model:pass(rows)` is called with rows that form groups, each group one
  reserved sequence's consecutive positions running up from the number of positions that sequence
  has taken, THE SYSTEM SHALL call each block once for all of the pass's rows, return one score row
  per input row in the order given, each holding the numbers `15/FR-005` gives for that row's token
  at its position at its sequence, and add to each named sequence's taken count the rows that
  sequence was given.
- FR-002 (US-001) WHEN a pass of R rows of one sequence runs on a run of N blocks, THE SYSTEM SHALL
  build `3 * N + 2` ggml graphs for it, whatever R is, and add them to the model's `measure.graphs`,
  so a chunk costs no more graphs than one row.
- FR-003 (US-001) THE SYSTEM SHALL size a model's arena at its build from the largest graph a pass of
  one row per sequence builds and, WHEN a pass holds more rows than that arena was measured at, grow
  it to the largest graph that pass builds, both measured at the model's `positions` and the grown
  arena never below the size it replaces, that growth happening before any graph of that pass is built
  so a model built with a `ubatch` of 512 holds no arena sized for 512 rows before its first wide
  pass. The grown arena replaces the one the model held: every block of that model builds in the grown
  arena from that pass on, the device buffers the blocks of that pool hold go with it and are taken
  from the grown pool by the next graph of each block, what the pool replaced is freed once no graph of
  that model still reads it, and `close` frees the arena and the buffers the model then holds, so no
  block ever builds over a freed arena or buffer and a closed model leaves neither behind. IF the
  arena a wider pass needs cannot be allocated, THEN THE SYSTEM SHALL return nil and one line naming
  `pass` and the rows it was given, and add no position to any sequence; a buffer the grown pool cannot
  give a block is refused by the line `12/FR-012` and `14/FR-015` give for a block call's graph, which
  the pass returns unchanged. Whether a grown arena is given back when a later pass holds fewer rows
  is not decided here and no case asserts it.
  Not testable: nothing a model publishes reports its arena's size, how many allocations it made or
  which arena or buffer a graph's tensors landed in, and making an allocation fail needs a memory
  limit the suite cannot set portably; a reader confirms the sizing, the replacement of the arena and
  of the block buffers, the free at `close` in `src/compute/engine.tl` and the failed growth in
  `src/compute/blocks.tl`, as `32/FR-005`,
  `32/FR-006`, `32/FR-008` and `32/FR-009` state the same for one row per sequence.
- FR-004 (US-002) WHEN a pass holds more than one row of one sequence, THE SYSTEM SHALL take each
  row's attention weights as the softmax of that row's scores over its own sequence's positions zero
  to its own and over no later position, whatever the cache type, so no row attends over a row after
  it in its own pass.
- FR-005 (US-002) WHEN a pass holds more than one row of one sequence, THE SYSTEM SHALL give each
  linear-attention row the convolution window and the recurrent state its own position leaves, those
  being what the rows before it in the same group leave, and write the sequence's stored window and
  state once, from its group's last row.
- FR-006 (US-002) WHEN the same ids of one sequence are forwarded one row per pass and in chunks of
  two and of three rows, THE SYSTEM SHALL return score rows every number of which agrees with the
  number the one-row passes return at the same position to within 1e-4.
- FR-007 (US-001) IF the rows naming one sequence are not handed together as a group of consecutive
  positions running up from the number of positions that sequence has taken, or a row names a
  sequence the model has not reserved, or the pass holds no rows, THEN THE SYSTEM SHALL return nil
  and one line naming `pass` and that sequence, or `pass` alone for a pass of no rows, and add no
  position to any sequence.
- FR-008 (US-003) IF a row of a group reaches a position its sequence's cache cannot hold, THEN THE
  SYSTEM SHALL refuse the whole pass with the line `12/FR-011` gives for the first such row in the
  order the rows were given, and add no position to any sequence, leaving the positions and the state
  the blocks of the run already took as `15/FR-012` leaves the blocks of a pass that failed halfway.
  The refusal therefore comes from the block that reaches the bound rather than from a bound checked
  over the whole group first, so when an earlier block of the run already took the group's rows a
  later pass of that sequence is refused by that block and that conversation cannot go on, while a
  refusal before any block took them leaves the sequence able to go on.
- FR-009 (US-003) WHEN a run is asked for its next rows with a bound of one or more, THE SYSTEM SHALL
  return as many of its unforwarded rows as that bound holds, in position order and no row handed
  twice, and nil once it has no row to forward, a group it handed and has not been given scores for
  yet being no row to forward; WHEN a run's rows are accepted, THE SYSTEM SHALL take
  the scores handed with them as the scores of the last row that run was handed.
- FR-010 (US-003) WHEN `generate` drives a run, THE SYSTEM SHALL ask it for the model's `ubatch` rows
  at a time, forward the prompt's ids from `start` on in list order in those groups, sample its first
  token from the scores of the prompt's last id, and forward every sampled token as a group of one row.
- FR-011 (US-003) WHEN `serve.step` builds a pass, THE SYSTEM SHALL ask each live turn that has a row
  to forward for the model's `ubatch` divided by the number of such turns, rounded down and at least
  one row, hand one pass those turns' groups in the order the turns registered, and give each of those
  turns the scores of its own group's last row, so no turn is ever given no row and the pass holds
  more than `ubatch` rows only when there are more such turns than `ubatch`.
- FR-012 (US-004) WHEN a config file is read or none is, THE SYSTEM SHALL pass `start` the
  `ubatch` resolved for the chosen model as `53/FR-003` says, 512 when neither its entry nor the
  file names one, and IF `ubatch` is present at either level and is not a whole number of one or
  more, THEN THE SYSTEM SHALL write one line naming `ubatch` and, for the entries holding one that
  is not, every such entry in ascending byte order, through the stderr writer and return exit code
  1 without calling a start function.
- FR-013 (US-004) WHEN `serve.build` builds its model, THE SYSTEM SHALL build it with the endpoint's
  `ubatch`, 512 when the endpoint names none, and expose that number as the model's `ubatch`; and IF
  `compute.model` is given a `ubatch` below one, THEN THE SYSTEM SHALL return nil and one line naming
  `model` and `ubatch`, the number the model exposes being 512 when the argument is absent.
- FR-014 (US-005) WHEN a prompt is forwarded in chunks, THE SYSTEM SHALL sample each token from the
  scores of the same prompt position the one-row-per-step run samples from, so `Text.sampled`,
  `Text.text`, the streamed events and the text a client reads are what that run gives.
- FR-015 (US-005) WHEN a bench run prefills a prompt in chunks, THE SYSTEM SHALL report that length's
  prompt tokens per second over the tokens it passed rather than the passes it made, and hand its
  routing dump one record per token per layer with `token` the token's index among the rows passed
  since `measure` was last zeroed.
- FR-016 (US-005) THE SYSTEM SHALL answer every case of `11` to `32` with the reply, ids, events and
  lines it records today, the only changes being the step at which a prompt's rows are forwarded and
  the counts the amendments below move.
  Not testable: those cases are the assertion, they run unchanged in the same suite, and a case here
  would repeat their assertions a second time.

## Non-goals

- No logits for only the rows that need them: a pass still returns one score row per input row, so a
  prompt costs one head row per token as it does today, and dropping the rows no caller reads waits
  for a spec of its own.
- No flash attention and no chunk sized from free memory, the page cache or the expert set: a block
  graph still materialises every row's scores, and the memory budget that could bound a chunk with
  them is a later spec's.
- No change to sampling, the stop-string holdback, the event shapes, the route's body members, the
  conversation store or the cache format on disk.

## Tests

`15/T-002`'s two-block fixture (a linear `blk.0.` and a full-attention `blk.1.`, `context_length` 4,
`token_embd.weight` rows `{1,0}`, `{3,4}`, `{1,1}`) is used below as **the mixed fixture** with
`blk.1.attn_output.weight` holding the two rows `{1,0}` and `{0,1}` in place of two rows of zeros, so
both block kinds contribute numbers to every score row.

The mixed fixture turns token `0` at position `0` into the zero vector its own norm then divides by,
so the cases below read ids it answers with numbers.

Every fixture prompt is shorter than the default `ubatch`, so a turn forwards its whole prompt in the
first step that hands it rows. In the cases of `18`, `19`, `25` and `26` that step a streamed turn,
the first token's event therefore arrives in the step that forwards the prompt's last row, which for
a prompt of P ids is P-1 steps earlier than `26` records; their payloads, ids, replies, counts of
registered turns and finish reasons are otherwise what they record. The helper `16`'s cases use to
drive a run by hand calls `Run.next` once per row and hands one input to `pass`; it takes the list
`next` now returns and hands that whole list to `pass`.

- T-001 (FR-001, FR-002, FR-004, FR-005, FR-006): `a chunk gives each row the numbers it gets one at
  a time`, proves on the mixed fixture that, with two models `compute.model(backend, handle, 4,
  "f32")` each holding `""` reserved at 4 and named here the chunked model and the one-row model, one
  `pass` of the rows `{"", 1, 0}`, `{"", 2, 1}`, `{"", 1, 2}` on the chunked model returns three score
  rows each equal within 1e-4 to the row the one-row model returns for the matching one-row pass of
  that id and position, and equal within 1e-4 to the rows a third model of that shape returns for the
  same ids handed as two rows then one row; that the chunk's first score row differs from its third
  by more than 1e-3 in at least one number, so a chunk that let a row attend over a later one, or
  handed every row the same attention, fails the case; that after the chunk a pass of the single row
  `{"", 2, 3}` on the chunked model returns within 1e-4 the row the one-row model returns for its
  fourth one-row pass at position 3, so the chunk left the cache rows, the convolution window and the
  state where three single rows leave them; and that the chunk adds 8 to `measure.graphs`, the same 8
  a one-row pass adds to a model of that two-block run.
- T-002 (FR-007): `rows that do not run together are refused and take nothing`, proves on the mixed
  fixture built with `sequences` 2 and `"a"` and `"b"` reserved at 4 that a pass of `{"a", 1, 0}` and
  `{"a", 2, 2}`, a pass of `{"a", 1, 1}` and `{"a", 2, 0}`, and a pass of `{"a", 1, 0}`, `{"b", 1,
  0}` and `{"a", 2, 1}` each return nil and one line holding `pass` and `a`, and that the pass of
  `{"a", 1, 0}` and `{"a", 2, 1}` after them returns the two rows a model of that shape returns for
  those ids one at a time, within 1e-4, so no refusal took a position.
- T-003 (FR-008): `a chunk past the cache is refused as one`, proves on `15/T-001`'s fixture built
  with `compute.model(backend, handle, 4, "f32", 2)` and `"a"` reserved at 4 that a pass of the six
  rows `{"a", 0, 0}`, `{"a", 1, 1}`, `{"a", 0, 2}`, `{"a", 1, 3}`, `{"a", 0, 4}`, `{"a", 1, 5}`
  returns nil and one line holding `a` and `position 4` and not holding `5`, the first row past the
  bound being the one named, and that the pass of the four rows `{"a", 0, 0}`, `{"a", 1, 1}`,
  `{"a", 0, 2}`, `{"a", 1, 3}` after it returns each row a model of that shape returns for that row
  one at a time, within 1e-4, so the refused chunk stored no cache row and the model took no position;
  and that on the mixed fixture, whose linear block holds no bound of its own, the same six rows
  return nil and one line holding `a` and `position 4`, and the pass of those same four rows after it
  returns nil and one line holding `blk.0.` and 0, the linear block having taken the refused chunk's
  rows as `15/FR-012` leaves a block that already answered.
- T-004 (FR-009, FR-010, FR-014): `a run prefills its prompt in chunks and answers the same`, proves
  on a copy of the mixed fixture grown with the gpt2 tokenizer metadata `16/T-001`'s fixture carries,
  its `tokenizer.ggml.tokens` the three pieces `{"a", "b", "c"}`, its `tokenizer.ggml.eos_token_id`
  removed so a run answers its whole budget on any id, and its `qwen35moe.context_length` 8, so its
  score rows follow the position the way `T-001` reads them, that `generate.text_ids` with the ids
  `{1, 2, 1, 2}`, `{max_tokens = 2}` and a `seed` and `temperature` of the case's choosing returns the
  same text and the same ids on models built with `ubatch` 2 and 8 as on the model built with `ubatch`
  1; that the passes behind those three replies are 5, 3 and 2 and that `measure.tokens` is 5 on all
  three; that those options reach as far as the claim does, drawing different ids from the score rows
  that `ubatch` 1 model returns for the prompt's first row and for its last row, the case recording the
  two rows and the ids it drew, so a run that sampled from an earlier row of its group answers a
  different text; and that `generate.open_ids` on the `ubatch` 2 model hands one row for
  `next(run, 1)`, then the three remaining prompt rows for `next(run, 8)`, and samples its first token
  from the scores handed for the last of them.
- T-005 (FR-011): `a step keeps to its share and splits it fairly`, proves on `18/T-001`'s fixture
  with `sessions` 2 and an endpoint `ubatch` of 2 that the streamed body
  `{"messages":[{"role":"user","content":"aba"}],"max_tokens":2,"stream":true}` leaves its collector
  holding only the `role` event after the first `serve.step`, and holding a payload beyond it after
  the second; that on a second server of the same fixture whose endpoint names no `ubatch` the same
  body gains a payload beyond the `role` event from its first step; that the payloads both collectors
  hold are equal one to one and their content pieces are together the text `serve.answer` gives for
  that body on either server, the reach of `FR-014` being `T-004`'s; and that on a third server of the
  same fixture with `ubatch` 2 two streamed bodies whose
  contents are `aa` and `bb` each take a second step to reach their first payload beyond the `role`
  event, so each of two live turns was given one row of the two; and that on a fourth server of the
  same fixture with `ubatch` 8 a turn whose whole prompt runs as one group beside another streamed
  turn streams what the same body streams on a fifth server of the same fixture whose `ubatch` is 1,
  so a turn handed a row other than its own group's last one answers another position.
- T-006 (FR-012): `ubatch is read from the file and refused when it is not a count`, proves with the
  `04` cases' `run`, an `environment` and a start function that a file naming `ubatch` 3 reaches
  `start` with 3, that a file naming none reaches it with 512, that a file naming `ubatch` beside a
  valid `port` writes no unknown-key line, that a file naming a faulting `context` beside a faulting
  `ubatch` writes one stderr line holding `context`, and that files naming `ubatch` `"8"`, `0`, `-1`
  and `1.5` each return 1 with one stderr line holding `ubatch` and call no start function.
- T-007 (FR-013): `the endpoint's chunk bound reaches the model`, proves on `18/T-001`'s fixture that
  `serve.build` on an endpoint whose `ubatch` is 3 returns a server whose
  `serve.model_of(server).generation.ubatch` is 3 and one whose endpoint names none 512; that a
  `compute.model` built with no `ubatch` argument exposes 512; and that `compute.model` given `ubatch`
  0 returns nil and one line holding `model` and `ubatch`.
- T-008 (FR-015): `bench counts tokens and routes one record per token`, proves on `29`'s fixture
  with a config file naming `ubatch` 3 that `bench m --config F --prefill-tokens 8
  --decode-tokens 8 --dump-routing R` returns 0, prints a `prompt 8:` line whose token count is at
  least 8 and more than the passes that prompt made at that bound, and writes to R one line per token
  per layer
  whose `token` values run 0 to that prompt count plus the decode count minus one with no value
  repeated.

## Open questions

- None.

## Amendments

- None.
