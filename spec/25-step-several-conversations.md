# 25: Several conversations step in one process

**Status**: done   **Depends on**: 16, 19, 23

## Problem & context

`19` streams a reply from one producer that generates the whole answer inside that producer, and
`23` keeps a conversation's model between turns, but a request is still answered alone: while one
turn generates, nothing else runs, so a second client waits for the first reply to finish before
its prompt is even read.

This spec makes the serve loop step turns instead of sitting inside one: every live turn advances
one forward per round, so two agents' replies interleave one token at a time, each keeps moving,
and neither waits for the other's whole answer. It buys fairness, not wall clock: a step is one
forward, tens of milliseconds, so each reply's tokens arrive spaced by the other turns' steps, and
the time to finish several replies is the time to finish them one after another. It
uses no thread and no process, which is `28`'s ground.

Three things had to be checked before the shape was settled, and they settled it.

- The transport loop can be stepped. `tltp.Server:poll(timeout_ms?)` is a documented method and
  `tltp.Server:serve()` is a `while true` around it, its own docs saying to call it yourself when
  the server has to share control with other work, so a caller may own the loop. Polled with a
  timeout of 0, a bound server returns true and nil.
- A stream's producer cannot be resumed by us. tltp already runs every producer in a coroutine, but
  it resumes one only when its deadline passes, and a resume takes the yield, at least one second,
  so a producer that yields once per token is resumed at most once a second. The resume is local,
  so nothing outside tltp can resume it either.
- A stream's writes are its own connection's only from 0.4.1. Before it, `server_write` read the fd
  and ssl handle tltp had set for the last request it handled, and streaming a response left them
  there, so a stream resumed later wrote to whichever client was handled last. 0.4.1 binds each
  stream's emit to the connection that stream belongs to, so a write lands on the issuing turn's
  client whenever in a round it is made. That is why this spec pins 0.4.1, and why a turn carries
  its own emit and nothing else.

So the tokens are driven by the serve loop and not by the producer: the producer writes the
stream's opening event, hands its emit to the scheduler and yields, and the scheduler steps each
live turn and writes its events through that turn's emit. A turn's last event reaches the client at
generation speed, and tltp closes that connection on the next resume, up to a second later.

Generation becomes resumable. `generate.open` does the work `text` does up to the first forward and
returns a run; each step of a run is one `model:forward` and at most one appended token; `text`
becomes a loop over a run, so the one generation loop stays one loop and its sampling, seed and
end-id rules are `16`'s unchanged. `16`'s statement that `text` is the module's only operation no
longer holds. The options' `on_token` goes with it: the scheduler emits from the step's return, so
nothing in the engine passes a callback, and `16/FR-001` and `19/FR-008` lose the field rather than
keep a dead one.

A streamed turn's run is built from what `prepared` already holds, after it has encoded the prompt
and refused every fault `generate.open` could report but one: a prompt whose template renders to no
ids passes `prepared`'s budget check and is refused by the run, which FR-013 answers. A fault the
run does meet, a forward or a decode, is a line from its step and ends that turn as `19/FR-009`
says.

One conversation cannot be read by two turns at once. `prepared` hands out the stored model itself
when a prompt continues the stored ids, and `23` stores a conversation only when its turn ends, so
a second turn on that `session_id` would be given the same `compute.Model` and the two runs would
forward over each other's cache entries, silently answering both from a mixture. Two requests can
reach that in one poll, because tltp handles every ready request and resumes each stream's producer
before the loop returns, so the in-flight mark cannot wait for a step. Nothing but this spec can
produce the clash at all, so this spec rules it out: a stored conversation is in flight from the
moment `prepared` hands its model to a turn until that turn stores it, discards it, or is refused
by FR-013, and a turn naming it during that time is prepared as `23/FR-003` prepares a new
conversation.

Bookkeeping per token is small next to the forward it follows: one decode of one id and one
`json_encode` of one event, both already on `19`'s path, plus a walk over the live turns and a
re-queue of one. The scheduler's turns live in `serve`, keyed by server as the store is, and an
ended turn leaves nothing behind. The cost of several live turns is memory, not arithmetic: each
live turn holds its own model and so its own KV cache, the ~20 KB per position `23` prices, and
holds the prompt's ids and the ids it produced while it runs.

A non-streamed reply still runs inside a poll, as `18` answers it today: its body must exist
before the poll returns, so no other mode of answering it is possible. Only streamed turns
interleave, which is also the only place a client can watch progress.

```teal
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")
local tltp = require("tltp")

local record generate
  record Run
    done: boolean
    failure: string | nil
    produced: {integer}
    next: function(Run): compute.Input | nil
    accept: function(Run, {number}): string | nil, string | nil
    fail: function(Run, string)
  end

  open: function(
    compute.Model, tokenizer.Tokenizer, string, Options, sequence?: string
  ): Run | nil, string | nil
end

local record serve
  step: function(tltp.Server): integer
  round: function(tltp.Server): string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want a generation I can advance one forward at a time and leave between steps, so that a turn can be resumed instead of run to its end in one call.
- US-002 (P1) As a person watching several agents, I want every reply to advance while the others do, so that no agent waits for another's whole answer.
- US-003 (P1) As the person running the server, I want a new request accepted while turns are in flight, so that a late agent starts instead of queueing behind a long reply.
- US-004 (P1) As a chat client, I want my events in order on my own connection, so that interleaving two replies into one process mixes neither.
- US-005 (P2) As the engine author, I want a failed forward or a failed event to end the turn the way streaming ends it today, so that interleaving changes no fault.

## Functional requirements

- FR-001 (US-001) WHEN `generate.open(model, tokenizer, prompt, options, sequence)` is called, THE
  SYSTEM SHALL take the optional sequence name, `""` when it is absent, return a run and nil, and
  make no forward, the run exposing `next` and `accept` in place of `step`, and IF `16/FR-009`,
  `16/FR-010` or `16/FR-011` refuse those arguments, THEN THE SYSTEM SHALL return nil and the line
  `generate.text` gives for them.
- FR-002 (US-001) WHEN a run has an unforwarded prompt id, THE SYSTEM SHALL forward that id and only
  that id as one row of the pass, at its own position, the one `16/FR-002` gives it, and return no
  text, so a prompt of several ids takes one step per id.
- FR-003 (US-001) WHEN a row's scores are accepted, THE SYSTEM SHALL take the next token from them
  by the rules `16/FR-004`, `16/FR-005` and `16/FR-006` give, sampling happening when a row's
  scores are accepted and not after a forward of its own, and IF that token is not the end id, THEN
  THE SYSTEM SHALL append it, return its decode as the step's text, and forward it as a row at the
  next position unless that append reached `max_tokens`, and IF it is the end id, THEN THE SYSTEM
  SHALL end the run without appending it, the run also ending when `max_tokens` ids are appended; a
  run's `produced` SHALL hold the ids it appended, in the order it appended them, and its `done`
  SHALL be true once it has ended, by the end id, by the count or by a failure. The end-id ending
  is proven through `16/T-003`, which exercises a run once `text` loops over one.
- FR-004 (US-005) IF a forward or a decode in a step returns nil, THEN THE SYSTEM SHALL end the run,
  return nil as that step's text and that line unchanged, and record that line as the run's
  failure; a failed row is the pass's nil line, which reaches its run through `fail`, ending that
  run and recording the line as its `failure`; a step of a run that is done SHALL return no text and
  make no forward.
- FR-005 (US-004) WHEN the producer of a streamed turn is first resumed, THE SYSTEM SHALL write the
  opening event `19/FR-003` names, record the emit it is given, register that turn with the
  scheduler, write no other event, and yield; IF that event cannot be encoded, THEN THE SYSTEM
  SHALL register no turn, discard the conversation and return, releasing its sequence as
  `26/FR-027` says; WHEN it is resumed while that turn
  is still live, THE SYSTEM SHALL write no event and yield again, so a turn slower than tltp's
  one-second resume keeps its connection; WHEN it is resumed after that turn has ended, THE SYSTEM
  SHALL write no event and return.
- FR-006 (US-004) WHEN the scheduler writes an event for a turn, THE SYSTEM SHALL write it through
  the emit that turn's producer was given and through no other turn's emit.
- FR-007 (US-003) WHEN `serve.step(server)` is called, THE SYSTEM SHALL make one pass for every
  turn that has a row, step each turn that is still live exactly once, in the order the turns
  registered, and return how many turns are still registered after it, waiting ones included.
- FR-008 (US-002) WHEN a step of a turn appends a token and `tokenizer:decode` gives text for that
  one id, THE SYSTEM SHALL write, through that turn's own emit, the event `19/FR-004` names for that
  text.
- FR-009 (US-002) WHEN a step ends its turn, THE SYSTEM SHALL write, in this order, the event
  carrying `finish_reason` that `19/FR-005` names and then the `[DONE]` event, store the
  conversation as `23/FR-002` says, and drop that turn from the scheduler so its run, sequence
  and emit are released and it is never stepped again. The `stop` reason is proven through `19/T-001`,
  which its amendment drives through `serve.step`.
- FR-010 (US-005) IF a step returns a failure line, or `tltp.json_encode` gives nil for an event's
  data, THEN THE SYSTEM SHALL write no further event for that turn, discard its conversation as
  `23/FR-008` says, and drop the turn from the scheduler as FR-009 describes.
- FR-011 (US-003) WHEN `serve.round(server)` is called while the scheduler holds a turn, THE SYSTEM
  SHALL call `server:poll` with a timeout of 0, then call `serve.step(server)`, and return the line
  that poll gives, nil when it gives none.
- FR-012 (US-004) THE SYSTEM SHALL build against tltp 0.4.1, the first release whose stream emit
  writes to the connection that stream belongs to, `lde.json` gaining `^0.4.1` and `lde.lock`
  moving to release commit `308e1a7`; 0.4.0 writes a stream's events through the server's last
  handled connection, which two live turns would cross.
  Not testable: a dependency pin, visible in `lde.json` and `lde.lock`, and the per-stream binding
  is tltp's own test suite's.
- FR-013 (US-005) IF `generate.open` gives nil for a prepared turn, THEN THE SYSTEM SHALL answer
  status `InternalServerError` and, as `application/json`, a body holding `error` with that line and
  type `internal_error`, write no event, and answer nil as the producer.
- FR-014 (US-003) WHEN `serve.start` is called, THE SYSTEM SHALL build the server as it does today,
  report a build failure through its `write_err` argument and return 1, and WHEN the build
  succeeds, THE SYSTEM SHALL repeat `serve.round(server)` while it returns nil.
- FR-015 (US-004) WHEN a turn names a `session_id` whose stored conversation is in flight, THE
  SYSTEM SHALL give that turn a reserved sequence of its own, sized as `23/FR-004` sizes a
  conversation's sequence, waiting for a free one when none is free, read the whole prompt from
  position zero, and take `cached_tokens` as 0 when the reply is not streamed; a stored
  conversation is in flight from the moment a turn is given its sequence until that turn stores it,
  discards it, or is refused by FR-013.
- FR-016 (US-003) WHEN a round that `serve.start` repeats returns a line, THE SYSTEM SHALL write
  that line through `serve.start`'s `write_err` argument with one trailing newline and return 1.
- FR-017 (US-003) IF a round that `serve.start` repeats raises an error, THEN THE SYSTEM SHALL
  write that error through `serve.start`'s `write_err` argument with one trailing newline and
  return 1.
  Not testable: every route answers through the `on_error` `serve` installs and a failed poll is a
  line rather than a raise, so no round of a bound server raises.
- FR-018 (US-003) WHEN `serve.round(server)` is called while the scheduler holds no turn, THE
  SYSTEM SHALL call `server:poll` with a timeout of -1, then call `serve.step(server)`, and return
  the line that poll gives, nil when it gives none.
  Not testable: an idle round blocks inside `poll` until a request arrives, and a single-coroutine
  case has no moment at which to send one, a request queued before the call being answered under
  either timeout, so the value it passed is unobservable.

## Non-goals

- No thread, no process, no second core and no time slice: a round steps every live turn once
  however long that takes, so the time to finish several replies is the time to finish them one
  after another, and worker processes stay `28`'s.
- No interleaving of a non-streamed reply, and no change to the finished JSON reply, the sampling,
  the budget, the template or the shapes of the events: those are `16`'s, `17`'s, `18`'s and
  `19`'s. The store's rules change only as FR-015 and the `23/FR-001` and `23/FR-002` amendments
  say, and in no other way.
- No queueing of a second turn on a `session_id` already in flight: FR-015 gives it a model of its
  own, and both turns store in the order they end, the later one being what the store holds. No
  cancellation or rewinding of a live turn either, and no closing of a stream's connection sooner
  than tltp's one-second resume.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004): `a run advances one forward at a time and stops where it
  must`, proves on `16/T-001`'s fixture built with `positions` 4 that
  `generate.open(model, tokenizer, "a", {max_tokens = 2, seed = 0})` returns a run and nil, that its
  first step returns nil and nil, the prompt's one id forwarded, that its second and third steps
  each return `a` and nil, and that after the third `run.done` is true and `run.produced` is
  `{0, 0}`; that on the same fixture with a fresh model built with `positions` 2,
  `generate.open(model, tokenizer, "aa", {max_tokens = 1, seed = 0})` returns a run whose first two
  steps each return nil and nil, the prompt's two ids forwarded at positions 0 and 1, and whose
  third returns `a` and nil with `run.done` true and `run.produced` `{0}`; that on a fresh model
  built with `positions` 2 the call with the prompt `"a"` and `{max_tokens = 3, seed = 0}` returns
  `a` from its second step and then nil and a line holding `blk.0.` and 2 from its third, the second
  sampled id being forwarded at a third position, with `run.done` true and `run.failure` that line,
  and a further step returning nil and nil where a forward would have failed again; and that `open`
  with `{max_tokens = 0, seed = 0}` returns nil and a line holding `max_tokens`, and with the
  prompt `""` returns nil and a line holding `prompt`.
- T-002 (FR-005, FR-006, FR-007, FR-008, FR-009): `two streamed turns advance one token each in
  turn`, proves on `18/T-001`'s fixture that `serve.step(server)` returns 0 before any producer has
  run; that `serve.stream` on the bodies `{"messages":[{"role":"user","content":"a"}],"max_tokens":
  2,"stream":true,"session_id":"s1"}` and the same naming `s2` each answer `text/event-stream` with
  a producer that is not nil; that resuming each producer in a coroutine of the test's own, with a
  collector per producer and one shared list recording which collector took each payload, leaves
  both coroutines suspended with only the `role` event collected and that list `{A, B}`; and then
  that three calls of `serve.step(server)` return 2, 2 and 0, the first adding no payload, the
  second adding one token event to each collector so the list is `{A, B, A, B}`, the third adding
  the second token and then the `length` finish event and `[DONE]` to each, so A's collector holds
  the `role` event, `a`, `a`, the `length` event and `[DONE]` and the list is
  `{A, B, A, B, A, A, A, B, B, B}`, and `serve.session_of(server, "s1")` is a conversation 2 ids
  long; and that resuming each producer once more returns true, writes no payload and leaves the
  coroutine dead.
- T-003 (FR-010, FR-013): `a turn that cannot run ends its stream and drops its context`, proves
  on a copy of `18/T-001`'s fixture whose third piece stores the single byte `0xFF` with token type
  1 that after the producer of `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,
  "temperature":1,"top_k":2,"seed":1000,"session_id":"s"}` is resumed, two calls of `serve.step`
  return 1 and then 0 collecting only the `role` event, and leave `serve.session_of(server, "s")`
  nil, the first sampled id being the third piece whose decode is the line `tokenizer:decode` gives
  for `{2}`; that on a copy whose third piece is `ä` with token type 1, its decode the single byte
  `0xE4` that `tltp.json_encode` refuses, the same body resumed and stepped twice returns 1 and then
  0, leaves the collector holding the `role` event and no other payload, and
  `serve.session_of(server, "s")` nil; and on a copy whose `tokenizer.chat_template` is
  `{% for message in messages %}{% endfor %}` that `serve.stream` on the streamed body of T-002's
  first shape naming `s3` answers a producer that is nil, a status of `InternalServerError` and a
  body decoding to `error` of type `internal_error` whose message names `prompt`, no event written.
- T-004 (FR-011, FR-014, FR-016): `a round steps the transport and the turns`, proves that after
  T-002's first body on an endpoint of its own on port 9405 has its producer resumed, three calls
  of `serve.round(server)` each answer nil and leave the collector holding the `role` event, `a`,
  `a`, the `length` event and `[DONE]`, the poll binding the port and finding nothing to read; it is
  the only case in `tests/serve.test.tl` that binds a port, because tltp has no close, so that
  binding outlives the case; and that `serve.round` on a server `serve.build` made from
  `18/T-001`'s file with `host` `192.0.2.1`, which no interface here holds, answers a line holding
  `BIND`; and that `serve.start` on that same unassignable endpoint returns 1 and writes one line
  holding `BIND` and one trailing newline through its `write_err` argument.
- T-005 (FR-007): `a turn registered while others are live is stepped next`, proves on `18/T-001`'s
  fixture that after A's producer is resumed for T-002's first body and one `serve.step` call has
  returned 1, resuming the producer of the same body naming `s2` leaves B's collector holding only
  the `role` event, that the next `serve.step` returns 2 and adds A's first token to A's collector
  while B's gains nothing, and that the next returns 1 and leaves A's collector at the `role` event,
  `a`, `a`, the `length` event and `[DONE]` while B's holds the `role` event and one `a`.
- T-006 (FR-015): `a second turn on a live session builds its own model`, proves on `18/T-001`'s
  fixture that `serve.answer` on the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` answers content
  `aa` with `cached_tokens` 0 and leaves `serve.session_of(server, "s")` a conversation of 2 ids and
  4 positions on `generation`; that resuming the producer `serve.stream` gives for `23/T-001`'s
  second body (`{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"s"}`), which reuses the
  stored model as `23/FR-001` says and leaves that conversation still 2 ids long, is enough for
  `serve.answer` on that same three-message body, whose three ids continue the stored two, to answer
  content `aa` with `cached_tokens` 0 and leave
  `serve.session_of(server, "s").generation.positions` at 8, a model of its own before the live turn
  has been stepped; and that the `serve.step` call that follows then returns 1, the live turn
  forwarding the prompt's last id from the stored position and adding nothing to its collector yet.

## Open questions

- None.

## Amendments

- None.
