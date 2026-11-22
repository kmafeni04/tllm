# 26: Several conversations' tokens in one pass

**Status**: done   **Depends on**: 12, 13, 14, 15, 16, 18, 19, 21, 23, 24, 25

## Problem & context

`25` steps every live turn once per round, but each step is one turn's `model:forward`, and a turn
has a model of its own. So a token reads every weight of the run for one conversation, two live
turns read the same 23 GB twice, and the process answers turns at the speed of answering them one
after another. `24` already holds a prefix's keys and values as a ggml tensor and computes the
attention products with `ggml_mul_mat`, which is the half of the work that was blocking a batch.

This spec makes one pass carry every live conversation's next token. One model holds
`sessions` sequences, each with its own keys and values reserved at the size `23` gives a
conversation, and one call `model:pass(rows)` takes a row per conversation and runs all of them
through a block call together, so each weight matrix is read once for the whole pass instead of
once for each conversation. `serve.step` builds that pass from every live turn, gives each turn
the scores of its own row, and writes its events as `25` already does.

Two things were checked before the shape was settled.

- The mask fits the kernel, not `12`'s signature. `ggml_soft_max_ext` already declares a mask
  argument, which `attended_heads` passes nil today, so a row's weights can be held to its own
  conversation's positions by a mask of zeros and `-inf` that ggml adds to the scores before the
  softmax, whether a pass builds one mask or one per row. What does not fit is the block call:
  `block(hidden, prefix, position)` holds one token, for one position, over one cache. A pass has a
  row per conversation, each at its own position, so the call must carry several rows. That is
  `12`'s, `13`'s, `14`'s and `15`'s contract changed, and the amendments at the end carry each
  requirement it changes.

- A pass carries at most one row per conversation. `25` forwards one prompt id per step and one
  sampled id per step, so no turn has two tokens to forward at once. That keeps each row's mask to
  "this conversation's positions up to mine", with no ordering needed among rows of one
  conversation.

Cost. A pass adds no memory: a conversation's reserve is the size `23` already gave its model, and
the pass's graph lives for the length of the call, as `12`'s block call does today. It adds no
sampling either: each turn still samples one id from one score row by `16`'s rules. The gain is
one read of each weight per pass, which is not a number this suite can pin on a hand-made file,
so what is pinned here is that batching changes no number and never costs more time than the rows
one by one; throughput on a real file is left to measurement.

```teal
local weights = require("tllm.weights")
local tokenizer = require("tllm.tokenizer")

local record compute
  record Row
    sequence: string
    hidden: {number}
    position: integer
  end

  record Input
    sequence: string
    token: integer
    position: integer
  end

  record Attention
    block: function(Attention, {Row}, string): {{number}} | nil, string | nil
    cache_bytes: function(Attention, string, string): integer | nil
  end

  record FeedForward
    block: function(
      FeedForward, {Row}, string
    ): {{number}} | nil, string | nil
    scores: function(FeedForward, {{number}}): {{number}} | nil, string | nil
  end

  record LinearAttention
    block: function(
      LinearAttention, {Row}, string
    ): {{number}} | nil, string | nil
  end

  record Model
    positions: integer
    sequences: integer
    forward: function(Model, integer, integer): {number} | nil, string | nil
    reserve: function(Model, string, integer): string | nil
    release: function(Model, string)
    cache_bytes: function(Model, string, string): integer | nil
    pass: function(Model, {Input}): {{number}} | nil, string | nil
  end

  model: function(
    Backend, weights.Handle, integer | nil, CacheType | nil, integer | nil
  ): Model | nil, string | nil
end

local record generate
  record Run
    done: boolean
    failure: string | nil
    produced: {integer}
    next: function(Run): compute.Input | nil
    accept: function(Run, {number}): string | nil, string | nil
    fail: function(Run, string)
  end

  -- 16's Options, unchanged
  open: function(
    compute.Model, tokenizer.Tokenizer, string, Options, string | nil
  ): Run | nil, string | nil

  open_ids: function(
    compute.Model,
    tokenizer.Tokenizer,
    {integer},
    Options,
    string | nil
  ): Run | nil, string | nil

  record Text
    text: string
    sampled: {integer}
  end

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options, string | nil
  ): Text | nil, string | nil
end

local record serve
  record Model
    -- 18's other fields, unchanged
    generation: compute.Model
  end

  record Session
    id: string
    length: integer
    sequence: string
    positions: integer
  end
end
```

A conversation's sequence is a name the caller chooses, `""` for the single-sequence use `15` and
`16` have today. `reserve` and `release` are what makes a long-lived model hold several
conversations without the memory growing with every conversation ever seen.

## User stories

- US-001 (P1) As the engine author, I want one pass to give each conversation's token the numbers a
  forward of its own gives, so that batching changes no reply.
- US-002 (P1) As the engine author, I want a model to hold each conversation's keys and values as a
  space I reserve, size and free, so that memory follows what is in flight and nothing leaks.
- US-003 (P1) As a person watching several agents, I want every live reply to advance in the same
  pass, each on its own stream, so that no agent waits for another's whole answer.
- US-004 (P1) As the person running the server, I want a streamed request to wait for a conversation
  slot instead of being refused, and a request that cannot wait to be told why.
- US-005 (P2) As the engine author, I want one pass never to cost more wall time than its rows one
  by one, so that sharing a pass is never a regression for one conversation.

## Functional requirements

- FR-001 (US-002) WHEN `compute.model(backend, handle, positions, cache_type, sequences)` is called
  with a `sequences` of one or more, THE SYSTEM SHALL return a model whose `sequences` is that
  number, 1 when the argument is absent, and SHALL reserve no key or value numbers for any sequence
  until FR-002 reserves it, a block call naming a sequence no `reserve` named being the one
  exception, as `12/FR-005` says.
- FR-002 (US-002) WHEN `model:reserve(sequence, positions)` is called, THE SYSTEM SHALL give that
  sequence keys and values for every full-attention block of the run at `positions` positions of
  the geometry's cache type, those positions being the bound its rows are checked against, SHALL
  leave that sequence's history and state, `14/FR-010`, starting as zeros, and SHALL return nil.
- FR-003 (US-002) IF `reserve` is given a name the model already holds, or a free name when it
  already holds as many sequences as its `sequences`, or a `positions` below one or above the
  model's `positions`, THEN THE SYSTEM SHALL return one line naming `reserve` and the name or the
  `positions` at fault, and reserve nothing.
- FR-004 (US-002) IF the memory a reservation needs cannot be allocated, THEN THE SYSTEM SHALL
  return nil and one line naming `reserve` and the sequence, holding nothing.
  Not testable: making an allocation fail needs a memory limit the suite cannot set portably, and
  `12/FR-012` and `14/FR-015` state the same fault for a block call.
- FR-005 (US-002) WHEN `model:release(sequence)` is called, THE SYSTEM SHALL free the numbers FR-002
  gave that sequence, and the history and state `14/FR-010` keeps for it, before the call returns,
  so a block call on that name starts again with zero state, `cache_bytes` gives nil for it, and
  `reserve` accepts that name again.
- FR-006 (US-002) WHEN `release` is given a name the model does not hold, THE SYSTEM SHALL free
  nothing, so a second release of one name is harmless.
- FR-007 (US-002) WHEN `model:cache_bytes(sequence, prefix)` is called, THE SYSTEM SHALL return
  the bytes `24/FR-006` states for that sequence's positions, and nil when the model holds no such
  sequence; `24/FR-005`'s `attention.cache_bytes` stays the block handle's own form for the cases
  that build a handle and no model.
- FR-008 (US-001) WHEN `model:pass(rows)` is called with rows that form groups, each group one
  reserved sequence's consecutive positions running up from the number of positions that sequence
  has taken, THE SYSTEM SHALL call each block once for all of the pass's rows, return one score row
  per input row in the order given, each holding the numbers `15/FR-005` gives for that row's token
  at its position at its sequence, and add the rows that sequence was given to that sequence's taken
  count; a pass of no rows SHALL be refused with one line naming `pass`.
- FR-009 (US-001) IF the rows naming one sequence are not handed together as a group of consecutive
  positions running up from the number of positions that sequence has taken, or a row names a
  sequence the model has not reserved, THEN THE SYSTEM SHALL return nil and one line naming `pass`
  and that sequence, and add no position to any sequence.
- FR-010 (US-001) WHEN a pass holds rows of several sequences, THE SYSTEM SHALL return each row's
  numbers as the numbers that row gets in a pass holding only it, whatever order the pass's rows
  are in, and each row's weights SHALL be the softmax of its scores over its own sequence's
  positions zero to the row's position, whether those positions came from an earlier pass or from an
  earlier row of the same pass, no other sequence's positions and no later position of its own
  sequence contributing, so no sequence's keys or values reach another sequence's scores.
- FR-011 (US-001) WHEN `model:forward(token, position)` is called, THE SYSTEM SHALL make a pass of
  one row for that token on the sequence `""`, reserving it at the model's `positions` when it is
  not reserved, and return that row's score row, or the line `reserve` gives when it cannot make
  that reservation.
- FR-012 (US-003) WHEN `serve.step(server)` is called, THE SYSTEM SHALL make one pass holding the
  groups of every registered turn that has a row to forward, each group that turn's whole share of
  the model's `ubatch` under `33/FR-011` and at least one row, in the order the turns registered,
  make no pass when no turn has one, give each such turn the scores of its own group's last row, and
  return how many turns are still registered after that pass, waiting ones included.
- FR-013 (US-003) WHEN two turns advance in one step, THE SYSTEM SHALL write each turn's events
  through that turn's own emit in the order the turns registered, and the reply each turn composes
  SHALL be the reply it gives when no other turn runs.
- FR-014 (US-003) IF a step's pass returns nil, THEN THE SYSTEM SHALL end every turn whose row was
  in that pass with that line, as `25/FR-010` ends a turn.
  Not testable: the only line a pass returns that `serve` can reach is a block's allocation fault,
  `12/FR-012` or `14/FR-015`, and making that allocation fail needs a memory limit the suite cannot
  set portably; the branch writes the line `25/FR-010` already writes.
- FR-015 (US-004) WHEN `serve.build` is called, THE SYSTEM SHALL build one model for the endpoint
  with the endpoint's `sessions` as its `sequences`, the endpoint's `ubatch` as its `ubatch` and
  the file's context length as its `positions`, hold it on that server as `serve.model_of` reports
  it, and build no model per turn;
  for an endpoint whose `workers` is more than one, `build` builds no model and holds none on that
  server, so `serve.model_of` gives nil.
- FR-016 (US-004) IF `compute.model` returns nil when `serve.build` builds it, THEN THE SYSTEM SHALL
  return that line as `build`'s second value, so `serve.start` writes it through its `write_err`
  argument and returns 1 as `25/FR-014` says, and no server answers a turn without a model.
- FR-017 (US-004) WHEN a turn stores its conversation, THE SYSTEM SHALL keep that conversation's
  sequence reserved under it, and IF the store already holds a conversation under that name whose
  sequence is not the turn's own, THEN THE SYSTEM SHALL drop that conversation and, unless a live
  turn holds it, release its sequence as FR-025 says, a sequence a live turn holds being released
  when that turn ends without storing it, as FR-024 says.
- FR-018 (US-004) WHEN a turn continues a stored conversation, THE SYSTEM SHALL forward only the
  prompt's ids past the stored length, over that conversation's own sequence.
- FR-019 (US-004) WHEN a turn's conversation holds no sequence, THE SYSTEM SHALL reserve one for it
  when the request is prepared, sized as `23/FR-004` sizes a conversation's model, named by the
  turn's `session_id` when no other live or stored sequence holds that name and by a name no other
  live or stored sequence holds otherwise.
- FR-020 (US-004) IF no sequence is free and the store holds a conversation no live turn holds,
  THEN THE SYSTEM SHALL evict the one no turn has used for longest, release its sequence as FR-025
  says, and take that place.
- FR-021 (US-004) IF no sequence is free and no stored conversation may be evicted and the reply is
  streamed, THEN THE SYSTEM SHALL write that turn's opening event, register the turn as waiting,
  write no other event, and yield.
- FR-022 (US-004) WHEN a step starts and a sequence is free, THE SYSTEM SHALL admit the
  longest-waiting turns, as many as are free, before it builds the pass, reserving each its
  sequence and forwarding its prompt's first group in that pass.
- FR-023 (US-004) IF no sequence is free and no stored conversation may be evicted and the reply is
  not streamed, THEN THE SYSTEM SHALL answer `TooManyRequests` with a body holding `error` of type
  `too_many_requests` whose message names `sessions`, storing no conversation and making no pass.
- FR-024 (US-004) WHEN a turn ends, THE SYSTEM SHALL release the sequence it reserved unless it
  stored the conversation under it.
- FR-025 (US-004) WHEN the store evicts a conversation, THE SYSTEM SHALL release that conversation's
  sequence.
- FR-026 (US-005) WHEN one pass of R rows and R passes of one row each are timed over models of the
  same fixture, at the same tokens and positions, THE SYSTEM SHALL make the one pass take no more
  wall time than the R passes together.
- FR-027 (US-004) WHEN a turn discards its conversation, THE SYSTEM SHALL release that
  conversation's sequence, whether the turn reserved it or was given a stored conversation to
  reuse, so a discarded conversation leaves no sequence reserved and a second release of it frees
  nothing as FR-006 says.
- FR-028 (US-004) IF the reservation a step makes for a waiting turn returns a line, THEN THE
  SYSTEM SHALL end that turn with that line, writing no further event, as `25/FR-010` says.
  Not testable: the only line such a reservation can return is FR-004's allocation fault, and
  making an allocation fail needs a memory limit the suite cannot set portably.

## Non-goals

- No prefill of a whole prompt in one pass and no more than one row per conversation per pass:
  `25`'s one token per step stays, so the mask needs no ordering among rows of one conversation.
- No thread, no process, no pipelining and no measurement on the 23 GB file: the pass is pinned
  against the same rows one by one, and the real file's turns per second is a later measurement.
- No change to sampling, the template, the event shapes, the budget check or the store's rules
  beyond the amendments below.

## Tests

T-001, T-002 and T-003 build `16/T-001`'s fixture with `attn_output.weight` holding the two rows
`{1, 0}` and `{0, 1}` in place of two rows of zeros, so a block's attention half contributes. That
fixture's tokens 0 and 1 are `{1, 0}` and `{3, 4}`, and every model they build names `"f32"`.

`25`'s step cases record a step that forwards one row and samples from its scores in the same call,
so their step counts, payloads and return values change: `25/T-001`'s first step returns `a` rather
than nil, its first part reaching `done` after two steps rather than three, its second part is two
steps rather than three, and its third part returns `a` from its first and second steps and the
`blk.0.` line from its third; `25/T-002`'s three calls become two with the first emitting a token;
`25/T-005`'s turns gain their tokens one step earlier and its step returns become 1 and then 0;
`25/T-006`'s step that added nothing now emits a token; `25/T-003`'s two failing steps become one,
that call returning 0 rather than 1 and then 0. `23/T-001` and `25/T-006` read
`session_of(...).positions` in place of `session_of(...).generation.positions`. Every other
assertion in those cases stands. `24/T-007`, `18/T-004` and `19/T-003` become `build` cases: a model
the server cannot build, by a cache type it cannot hold or by a file missing `embedding_length`,
leaves `build` nil rather than answering a request, so the body they recorded as
`InternalServerError` is `serve.start`'s line; their other per-request clauses, a removed template,
a `0xFF` template and an empty template, stand.

Every case that calls a block directly, in `12`, `13`, `14` and `21`, passes a one-row batch whose
row names the sequence `""`, and `24/T-001`'s `cache_bytes` calls name that sequence first; their
recorded numbers are unchanged, the sequence's positions being the geometry's `positions` as
`12/FR-005` says.

- T-001 (FR-008, FR-009, FR-010, FR-011): `a pass gives each row the numbers it gets alone`, proves
  that `compute.model(backend, handle, 4, "f32", 2)` returns a model whose `sequences` is 2, that
  `reserve("a", 4)` and `reserve("b", 4)` return nil, that a pass of the rows `{"a", 0, 0}` and
  `{"b", 1, 0}` returns two rows of three numbers, the first equal to the scores a model built with
  no `sequences` argument returns for `forward(0, 0)` and the second equal to the scores another
  such model returns for `forward(1, 0)`, to within 1e-5, and that those two score rows differ in
  their second number by more than 1e-3, so a cache shared between the sequences fails the case;
  that a second pass of the rows `{"b", 1, 1}` and `{"a", 0, 1}` returns, for `"a"`, the scores the
  `"a"` model returns for its second `forward(0, 1)` and, for `"b"`, the scores the `"b"` model
  returns for its second `forward(1, 1)`, so row order changes nothing; that on a second model of
  the same shape a pass holding two rows of `"a"` returns nil and one line holding `pass` and `a`, a
  pass holding one row of `"c"` returns nil and one line holding `pass` and `c`, a pass holding the
  row `{"a", 0, 2}` returns nil and one line holding `pass` and `a`, and a pass of the rows
  `{"a", 0, 0}` and `{"b", 1, 0}` then returns, for `"a"`, the scores the `"a"` reference model
  returns for `forward(0, 0)` and, for `"b"`, the scores the `"b"` reference model returns for
  `forward(1, 0)`, so the refused passes added no position to either sequence; that a pass holding
  no rows returns nil and one line holding `pass`; and that a fresh model built with `sequences` 2
  answers `forward(0, 0)` with the scores the `"a"` reference model returns and leaves
  `model:cache_bytes("", "blk.0.")` 64, the empty name reserved at the model's 4 positions.
- T-002 (FR-001, FR-002, FR-003, FR-005, FR-006, FR-007): `a sequence's space is reserved,
  reported and released`, proves on T-001's fixture that a model built with `sequences` 2 has
  `cache_bytes("a", "blk.0.")` nil before any reservation, that `reserve("a", 3)` returns nil and
  `cache_bytes("a", "blk.0.")` then gives 48, that `reserve("a", 3)` returns a line holding
  `reserve` and `a`, that `reserve("b", 0)` and `reserve("b", 5)` each return a line, that
  `reserve("b", 4)` returns nil and `cache_bytes("b", "blk.0.")` gives 64, that `reserve("c", 1)`
  then returns a line, that `release("a")` leaves `cache_bytes("a", "blk.0.")` nil, that a second
  `release("a")` and a `release("zzz")` free nothing, `cache_bytes("b", "blk.0.")` still 64, that
  `reserve("c", 4)` then returns nil with `cache_bytes("c", "blk.0.")` 64, that a pass naming `"a"`
  after its release returns nil and one line holding `pass` and `a`, and that a model built with no
  `sequences` argument holds one sequence, so its `reserve("a", 1)` returns nil and a second
  `reserve("b", 1)` returns a line.
- T-003 (FR-026): `a pass costs no more than its rows one by one`, proves on T-001's fixture that
  each of three fresh pairs of models built with `sequences` 4 and `threads` 4, each pair reserving
  `"a"`, `"b"`, `"c"` and `"d"` at 4 positions, times one pass of the four rows `{"a", 0, 0}`, `{"b", 0, 0}`,
  `{"c", 0, 0}` and `{"d", 0, 0}` on one model against four passes of one row each on the other,
  and that the sum of the three one-pass times is no longer than the sum of the three four-pass
  times, the times being the monotonic wall times `24/T-004` takes, after one throwaway pair has
  been run.
- T-004 (FR-012, FR-013): `two turns advance in one pass and keep their own streams`, proves on
  `18/T-001`'s fixture built from an endpoint with `sessions` 2 that `serve.stream` on
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}` and on the same body
  with content `b` each answer a producer; that resuming each producer once, with a collector per
  producer and one shared list recording which collector took each payload, leaves both suspended
  with only the `role` event collected and the list `{A, B}`; that one `serve.step` returns 2 and
  leaves the list `{A, B, A, B}`, each collector holding the `role` event and the first piece its
  body answers with; that the next `serve.step` returns 0 and leaves each collector holding the
  `role` event, its body's two pieces, the `length` finish event and `[DONE]`, with the list
  `{A, B, A, B, A, A, A, B, B, B}`; and that each collector's pieces are together the text
  `serve.answer` gives for the same body on a second server built from the same file, so batching
  changed neither reply.
- T-005 (FR-015, FR-020, FR-021, FR-022, FR-024, FR-025): `a turn waits for a sequence and starts
  when one frees`, proves on `18/T-001`'s fixture built with `sessions` 1 that
  `serve.model_of(server).generation` holds `sequences` 1; that A's producer for T-004's first body
  resumes and one `serve.step` returns 1 with A's first `a` collected, that resuming B's producer
  for the same body then leaves B's collector holding only the `role` event, that the next
  `serve.step` returns 1 and leaves B's collector at the `role` event while A's holds the `role`
  event, `a`, `a`, the `length` event and `[DONE]`; that A's end leaves no conversation, so its
  sequence is free and the next `serve.step` returns 1 and gives B's collector its first `a`, and
  that one more step returns 0 with B's collector holding `role`, `a`, `a`, `length` and `[DONE]`;
  and that on a second server of the same fixture with `sessions` 1, a streamed body carrying
  `"max_tokens":2` and `"session_id":"s"` runs to its end in two steps and is stored, holding the
  only sequence, and that `serve.stream` on T-004's first body then evicts `s` as it prepares that
  turn, so `serve.session_of(server, "s")` is nil and its collector gains its first `a` when the
  steps run; and that on a third server built with `sessions` 2, A's and B's producers for T-004's
  two bodies are resumed and one step returns 2, that C's and D's producers for those bodies then
  leave both collectors at the `role` event, that the next step returns 2 and leaves C's and D's
  collectors there while A and B end, and that the step after it returns 2 with C's and D's
  collectors each holding the first piece their body answers with, so both waiters were admitted
  in one step; and that on a fourth server built with `sessions` 1, a streamed body carrying
  `"session_id":"s"` is stepped once, T-004's first body is then resumed and waits while `s`
  finishes and is stored, and the step after that evicts `s` and gives the waiter its first piece,
  so a waiter is admitted once a stored conversation's sequence is freed.
- T-006 (FR-017, FR-018, FR-019, FR-023, FR-027): `a reply that cannot wait is told so, a stored
  conversation keeps its sequence and a discarded one frees it`, proves on `18/T-001`'s fixture
  built with `sessions` 1 that
  while a streamed turn is live `serve.answer` on
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2}` answers `TooManyRequests` with a
  body decoding to `error` of type `too_many_requests` whose message holds `sessions`, leaving
  `serve.session_of` nil and the live turn's collector unchanged; and that on a server of its own
  the body `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` answers
  `OK` with `cached_tokens` 0 and leaves `s` 2 ids and 4 positions long, that the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` answers `OK` with
  `cached_tokens` 2 and leaves `s` 4 ids long, and that `serve.session_of(server, "s").sequence`
  is `"s"` both times, so the stored conversation kept its reserved sequence and that sequence is
  named by the request's `session_id`; and that on a server built with `sessions` 2 after that
  first body was answered and a streamed copy of it was resumed and stepped once, `serve.answer`
  on the same body answers `OK` with `cached_tokens` 0 and leaves
  `serve.session_of(server, "s").sequence` a name other than `"s"`, that stepping that streamed
  turn to its end leaves `serve.session_of(server, "s").sequence` `"s"` again, and that
  `serve.model_of(server).generation:cache_bytes(name, "blk.0.")` gives nil for the name the second
  conversation held and a number for `"s"`, so the sequence the store dropped was released; and
  that on a copy of `18/T-001`'s fixture whose third piece stores the single byte `0xFF` with token
  type 1, built with `sessions` 1, the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` answers
  `OK` and stores `s` 2 ids long, that the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"temperature":1,"top_k":2,"seed":1000,
  "session_id":"s"}` answers `InternalServerError`, its first sampled id being the third piece,
  that it leaves `serve.session_of(server, "s")` nil and
  `serve.model_of(server).generation:cache_bytes("s", "blk.0.")` nil, so the discarded
  conversation freed its sequence, and that the body with no `session_id` then answers `OK`, a
  sequence being free again.
- T-007 (FR-015, FR-016): `a model the server cannot build is a start failure`, proves on a copy of
  `18/T-001`'s fixture built from an endpoint whose `kv_cache` is `"q8_0"` and whose `sessions` is 4
  that `serve.build` returns nil and one line holding `q8_0` and `2`, and that `serve.start` on the
  same endpoint returns 1 and writes that line with one trailing newline through its `write_err`
  argument.
- T-008 (FR-002, FR-005): `a released name starts from zeros`, proves on `15/T-002`'s fixture built
  with `sequences` 2 and `f32` that reserving `"a"` at 2, passing the row `{"a", 2, 0}` and
  keeping its score row, then releasing `"a"` and reserving it again at 2, the row `{"a", 2, 0}`
  passed again returns that same score row to within 1e-5, and that a second sequence `"b"`
  reserved at 2, whose row `{"b", 2, 0}` is passed for the first time after those two passes of
  `"a"`, returns that same score row too, so neither a stale state nor `"a"`'s first pass reaches
  it.

## Open questions

- None.

## Amendments

- None.
