# 23: A conversation resumes from the context it already read

**Status**: done   **Depends on**: 04, 08, 16, 18, 19

## Problem & context

`18` builds a model per request and `16` prefills from position zero, so the second turn of a chat
reads the whole transcript again: on a long conversation almost all of a turn's work is re-reading
words the server has already seen. `18`'s non-goals say so on purpose ("no cache outlives a
request"). Nothing keys a model to a conversation, nothing keeps one between requests, and no reply
reports how much of its prompt was already read.

This spec adds a store of conversations to `serve`, keyed by a `session_id` the body may carry. A
turn whose prompt begins with the ids a stored conversation already read forwards only the ids past
them, and reports how many it skipped as `usage.prompt_tokens_details.cached_tokens`. A turn whose
prompt does not begin with them, or is exactly them, or whose model has no room for it, reads the
whole prompt and replaces the stored one, so a stale context is never answered from. A conversation
is a model plus the ids it has read, its length being the positions its model has taken, and the
store holds at most the `sessions` number `config.json` names.

The store belongs in `serve` and not in a plugin: a plugin has no route, no request pipeline and no
way to reach the caches `12` keeps inside a model, so the code that answers a request is the only
code that can keep a conversation's model alive between requests. The endpoint `build` receives
gains the cap, which is why `08`'s own calls and the ones in `tests/serve.test.tl` gain the field.

`12` allocates a prefix's cache when a block first runs over it, so a kept model holds only the
positions its own conversation reached, not the file's window: a conversation at 200 prompt ids and
a 256-token budget keeps 910 positions in each of its 10 attention blocks, and `12` prices 262144
positions at 537 MB in each of those blocks at `f16`, so that conversation costs about 18.6 MB,
which is 20 KB per position across the ten. Four conversations is about 75 MB, which is why the cap
exists and why the default is four. `24` holds each prefix's keys and values in ggml tensors of the
endpoint's `kv_cache` type, `f16` unless the file says otherwise, so that cache is where memory and
speed were won.

Several conversations in one process is the next step, and it is not this spec. A coroutine switches
only where Lua yields, and nothing on a token's path yields, so a session is stepped one token at a
time and the scheduler hands control to another session between tokens. That interleaves sessions, so
every agent keeps moving and none waits for a whole reply, and it needs no threads; what it does not
do is run two tokens at once, so the time to finish several replies is the time to finish them one
after another and no second core is used. A step is one token's forward, tens of milliseconds and not
the short slice a user interface interleaves at, so each reply's tokens arrive spaced by the other
sessions' steps: a user sees prompt starts and steady progress for every agent, which one turn at a
time cannot give, and sees slower, burstier text than one agent alone, which no amount of yielding
hides. Wall-clock needs operating-system processes, the mechanism `lde test --jobs` uses (one child
per test file, lde's own help text), and a process cannot share another's cache memory, so a
conversation that moves between them needs a context on disk. The pieces that follow are `24`, a
cache held as a ggml tensor, `25`, a coroutine scheduler, `26`, several conversations in one graph
pass, `27`, a context on disk, and `28`, worker processes. This spec is the groundwork for all five:
a conversation's context is one identifiable thing with a name, a length, and a model behind it.

Refusals and faults keep the shape `18` gives them: a 400 for a body at fault, a 500 with the
engine's line, and the same `choices` and `usage` envelope with one object added to `usage`.

```teal
local compute = require("tllm.compute")

local record serve
  -- the endpoint build receives, gaining the cap
  record Endpoint
    sessions: integer
  end

  record Session
    id: string
    length: integer
    sequence: string
    positions: integer
  end

  session_of: function(tltp.Server, string): Session | nil
end

-- tllm.generate, added to generate.Options
start: integer | nil
```

## User stories

- US-001 (P1) As a chat client in an ongoing conversation, I want each turn to read only the messages the server has not read yet, so that a long chat answers without the transcript being read again.
- US-002 (P1) As a chat client whose earlier messages changed, I want the reply to come from the messages I sent, so that a context stored for other messages never answers this one.
- US-003 (P1) As the person running the server, I want `config.json` to set how many conversations are kept, so that memory stays bounded as clients come and go.
- US-004 (P2) As the engine author, I want a failed or refused turn to leave no half-read context, so that the next turn cannot answer from one.

## Functional requirements

- FR-001 (US-001) WHEN a body's `session_id` is a string and the server holds a conversation under
  that name whose stored ids are the prompt's first ids, fewer than all of them, and which no live
  turn holds, THE SYSTEM SHALL forward only the prompt's ids past that length, each at its own
  position, and, when the reply is not streamed, take `usage`'s
  `prompt_tokens_details.cached_tokens` as that length. A stored conversation is in flight from
  the moment `prepared` hands its model to a turn until that turn stores it, discards it, or is
  refused by `25/FR-013`, and a turn naming it during that time is prepared as FR-003 does.
- FR-002 (US-001) WHEN a turn carrying a `session_id` completes with a run that ended without a
  failure line, streamed or not, THE SYSTEM SHALL store under that name the prompt's ids followed
  by every sampled id the run forwarded, the last sampled id of a turn stopped by the budget not
  being stored because no forward took its position, and report that conversation through
  `serve.session_of` with a `length` counting those ids; the conversation `session_of` reports
  carries its `sequence` and `positions` in place of a model, `positions` being the positions
  `reserve` gave that sequence. A streamed turn stores when `serve.step` ends it, `25/FR-009`'s
  case.
- FR-003 (US-001) WHEN a turn asks for a `session_id` the server holds no conversation under and
  no context file for the name is restored as `27/FR-002` says, THE SYSTEM SHALL reserve that
  conversation a sequence and read the whole prompt from position zero, taking `cached_tokens` as 0
  when the reply is not streamed; a restored file continues the conversation as FR-001 continues a
  stored one, and WHEN a restored file's ids are not the prompt's first ids, or hold all of them,
  THE SYSTEM SHALL release the sequence its load reserved as `27/FR-013` says before that
  reservation.
- FR-004 (US-001) WHEN the server reserves a conversation's sequence, THE SYSTEM SHALL reserve it
  at twice the turn's need, or the file window's positions when that is fewer, the turn's need
  being the prompt's ids plus the budget minus one, and the sequence stays reserved while the
  conversation is stored; `18/FR-009` still refuses a turn whose need is past the file window. Its
  tests' endpoints set `kv_cache` `f32`, so their recorded replies and `cached_tokens` stand.
- FR-005 (US-001) IF a turn's need is more than the positions the conversation's sequence holds,
  THEN THE SYSTEM SHALL release that sequence and reserve a larger one, read the whole prompt from
  position zero, and take `cached_tokens` as 0 when the reply is not streamed.
- FR-006 (US-002) IF a conversation's stored ids are not the prompt's first ids, or hold all of the
  prompt's ids, THEN THE SYSTEM SHALL discard that conversation, release its sequence, reserve one
  as FR-004 sizes it, read the whole prompt from position zero, and take `cached_tokens` as 0 when
  the reply is not streamed.
- FR-007 (US-003) WHEN a turn stores a conversation under a name the server holds no conversation
  under and already holds as many conversations as the endpoint's `sessions` names, THE SYSTEM SHALL
  discard the conversation no turn has used for longest, release its sequence, and only a
  conversation no live turn holds is evicted; storing or reusing a conversation counts as a use.
- FR-008 (US-004) IF the generation of a turn carrying a `session_id` returns nil, THEN THE SYSTEM
  SHALL discard that conversation and release its sequence as `26/FR-027` says and, when the reply
  was not streamed, answer the status and body `18` answers for that fault, and when it was streamed
  end the stream as `19/FR-009` says and write no further event.
- FR-009 (US-001) WHEN a body carries no `session_id`, THE SYSTEM SHALL answer as `18` does, store
  nothing and take `cached_tokens` as 0 when the reply is not streamed.
- FR-010 (US-004) IF a body is refused before its generation begins, THEN THE SYSTEM SHALL leave
  every stored conversation and its order of use unchanged.

## Non-goals

- No disk: a conversation lives in the server's memory and is gone when the process ends, which is
  `27`.
- No rewind and no partial reuse: a conversation is used whole or discarded whole, and a model is
  never cut back to an earlier position.

## Tests

- T-001 (FR-001, FR-002, FR-004, FR-009): `a second turn reads only what it has not read`, proves on
  `18/T-001`'s fixture built from an endpoint holding `sessions` 4 that the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` answers `OK`
  with `usage` of `prompt_tokens` 1, `completion_tokens` 2,
  `total_tokens` 3 and `prompt_tokens_details` of `cached_tokens` 0, and leaves
  `session_of(server, "s").length` at 2, the one prompt id and the one sampled id the turn forwarded;
  that the body `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}`, whose messages render `aaa`,
  answers the same text and reports `cached_tokens` 2, leaving that length at 4, where the same body
  naming `t` reports `cached_tokens` 0 and leaves `t` 4 long; that the first body with its
  `session_id` removed answers `cached_tokens` 0 and leaves `s` and `t` at 4; that after the first
  turn `session_of(server, "s").generation.positions` is 4, twice that turn's need of 2; and that a
  body whose single message content is 130 copies of `a` with `"max_tokens":2` and `"session_id":"w"`
  answers `OK` and leaves that conversation's model with 260 positions, twice its need of 131 being
  past the file's 260; that the body `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,
  "temperature":1,"seed":7,"session_id":"v"}` answers `OK` with `completion_tokens` 1 and leaves `v`
  2 ids long, the first draw 0.2387808 taking id 0 and the second 0.9134933 taking the end id 1, so
  the one sampled id was forwarded and is stored where a budget stop would have dropped it; and that
  the body `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"session_id":"v"}` then answers `OK` with
  `cached_tokens` 2, the two stored ids being the prompt's first two.
- T-002 (FR-003, FR-005, FR-006): `a turn that does not continue the stored context starts over`,
  proves after T-001's first two turns on `s` that a body of five `a`-content messages with
  `"max_tokens":2` and `"session_id":"s"` answers `OK` with `cached_tokens` 0 and leaves `s` 6 ids
  long, the turn's need
  of 6 being past the 4 positions the stored model holds while its 5 ids still continue the 4 stored;
  that a body of the messages `a` then
  `assistant:a` with `"max_tokens":2` and `"session_id":"s"` answers `cached_tokens` 0 and leaves `s`
  3 ids long, the 6 stored ids not being the first ids of the prompt's 2; that the body of three
  `a`-content messages with `"max_tokens":2` and `"session_id":"s"` answers `cached_tokens` 0 and
  leaves `s` 4 ids long, the 3 stored ids being all 3 of the prompt's; that after a first turn
  naming `x` the body `{"messages":[{"role":"user","content":"baa"}],"max_tokens":2,
  "session_id":"x"}` answers `cached_tokens` 0 and leaves `x` 4 ids long with a model of 8
  positions, the 2 stored ids `{0, 0}` not being the first ids of the prompt's `{1, 0, 0}` while the
  length and position checks both pass; and that a body of the single message `a` with
  `"max_tokens":2` and `"session_id":"u"` builds a model, answers `cached_tokens` 0 and leaves `u`
  2 ids long.
- T-003 (FR-007, FR-010): `the server keeps only the configured number of conversations`, proves on
  `18/T-001`'s fixture built from an endpoint holding `sessions` 2 that after the T-001 body naming
  `s1` and the same body naming `s2`, a turn of T-001's second body naming `s1` answers `cached_tokens`
  2, so a reuse counts as a use, and the store holds `s1` and `s2`; that the same body naming `s3`
  discards `s2` because `s1` was used after it, `session_of` giving a
  conversation for `s1` and `s3` and nil for `s2`; that `{"messages":[],"session_id":"s1"}` answers
  `BadRequest` and leaves `s1` 4 ids long and `s3` 4 ids long, `session_of(server, "s9")` being nil;
  and that T-001's second body naming `s4` then discards `s1` rather than `s3`, the refused turn not
  having counted as a use, `session_of` giving a conversation for `s3` and `s4` and nil for `s1`.
- T-004 (FR-008): `a failed turn leaves no context`, proves on a copy of `18/T-001`'s fixture whose
  third piece stores the single byte `0xFF` with type 1 that the body `{"messages":[{"role":"user",
  "content":"a"}],"max_tokens":2,"session_id":"s"}` answers `OK` and leaves `s` 2 ids long, that the
  body `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},
  {"role":"user","content":"a"}],"max_tokens":2,"temperature":1,"top_k":2,"seed":1000,
  "session_id":"s"}` answers `InternalServerError` with the line `tokenizer:decode` gives for `{2}`
  and leaves `session_of(server, "s")` nil, and that on a fresh server the first body followed by
  that second body with `"stream":true` added gives a producer that writes only the `role` payload
  and leaves `session_of(server, "s")` nil.
- T-005 (FR-002): `a streamed turn stores its context too`, proves on `18/T-001`'s fixture that
  `serve.stream` on T-001's first body with `"stream":true` added answers a producer, that resuming
  that producer once and then driving `serve.step` to no live turns leaves
  `session_of(server, "s").length` at 2, and that T-001's second body, naming `s` and carrying
  `"stream":false`, then reports `cached_tokens` 2.

## Open questions

- None.

## Amendments

- None.
