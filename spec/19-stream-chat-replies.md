# 19: Stream a chat reply as it is generated

**Status**: done   **Depends on**: 16, 18

## Problem & context

`18` answers `/v1/chat/completions` with one finished body, so a client waits for the whole reply
before it sees a character and, on this machine, that wait is the whole generation. `tltp` already
ships the transport: `Server:sse(fn)` answers `text/event-stream` and hands `fn` an `emit` closure
that writes every string it is given as one `data:` frame followed by a blank line.

This spec makes that route answer a stream when the body says `"stream": true`: a first event
naming the assistant role, one event per text the run releases, one event saying why the reply
stopped, and `data: [DONE]`, the four shapes the OpenAI clients expect. A body that asks for no
stream keeps answering what it answers today.

`serve.stream(server, body)` is the whole of one streamed request: it prepares the request as
`18/FR-003` does (parse, render, encode, the window model, the budget check, the generation model),
and answers the response the route hands back plus the producer inside it, that producer being nil
when the request was refused. The route becomes the one that answers the first of those two
values. The producer is the only thing a test can drive here, `mock_request` not running a
producer, so the producer is returned rather than reached for through the response; the response
itself comes from `Server:sse`, which is what sets the event framing and the stream headers, and
no case asserts the wire bytes of a frame.

Why the reply stopped is read off the number of tokens the generation appended: as many as the
budget allows means the count stopped it, fewer means an end id did, `16/FR-007` and `16/FR-008`
being the two ways a generation ends. `tltp.json_encode` writes an empty Lua table as `[]`, not
`{}`, so the event that reports why the reply stopped carries no `delta` member at all.

A delta is the text a run releases for that step, which is `tokenizer:decode` of one id joined with
any text the run held back before it; one token can hold a single byte of a multi-byte character, so
a delta can be a fragment and the client is the one that joins them. A fragment is usually not valid
UTF-8, and `tltp.json_encode` refuses such a string, which is why a stream whose event cannot be
encoded ends early rather than writing a broken frame.

```teal
local record serve
  stream: function(
    tltp.Server, string
  ): tltp.Response, tltp.StreamFn | nil
end

```

`stream` becomes a member the body may carry and one that is checked, so `18`'s wording about the
members it ignores and the shape it refuses now names it, and neither `16` nor `18` says in its
non-goals that nothing is streamed. The record `serve` builds from the body is unexported, so
`stream` joins it in the code and no snippet in `18` gains a field.

## User stories

- US-001 (P1) As a chat client, I want the reply to arrive token by token as it is generated, so that text appears without waiting for the whole answer.
- US-002 (P1) As a chat client, I want the stream to end with a marker and to say why the reply stopped, so that I can tell a full answer from a cut-off one.
- US-003 (P1) As an existing chat client that never asks for a stream, I want the same finished JSON reply as today, so that nothing I already do changes.
- US-004 (P1) As a chat client sending a body the route cannot use, I want the JSON 400 naming what is at fault, so that I can fix my request instead of reading a broken stream.

## Functional requirements

- FR-001 (US-001) WHEN a POST for `/v1/chat/completions` arrives, THE SYSTEM SHALL answer the
  response `serve.stream(server, the raw request body)` gives, which is `18/FR-002` as it now reads.
  Its tests' endpoints set `kv_cache` `f32`, so the streamed text they record stands.
- FR-002 (US-001) WHEN `serve.stream` is given a body whose `stream` member is `true` and
  `generate.open` gives a run for the prepared turn, THE SYSTEM SHALL answer the response
  `server:sse` gives for its producer, whose status is `OK`, whose content type is
  `text/event-stream`, and whose producer is not nil; WHEN `generate.open` gives nil for that
  turn, THE SYSTEM SHALL answer `InternalServerError` as `application/json`, write no event, and
  answer nil as the producer, which is `25/FR-013`'s case.
- FR-003 (US-001) WHEN the producer runs, THE SYSTEM SHALL write as its first event one whose data
  is a JSON object whose `choices` is an array holding one object whose `index` is 0 and whose
  `delta` is an object whose `role` is `assistant`.
- FR-004 (US-001) WHEN a step of a turn releases text, THE SYSTEM SHALL write, through that turn's
  own emit, an event whose data is a JSON object whose `choices` is an array holding one object
  whose `index` is 0 and whose `delta` is an object whose `content` is that text, one event per
  release and none for a token the run holds back, the event for a run's last release coming before
  the event FR-005 writes; the producer writes no token event of its own.
- FR-005 (US-002) WHEN the reply ends, THE SYSTEM SHALL have `serve.step` write, in this order, an
  event whose data is a JSON object whose `choices` is an array holding one object whose `index` is
  0, holding no `delta` member, whose `finish_reason` is `length` when as many tokens were appended
  as the budget allows and `stop` otherwise, then an event whose data is the text `[DONE]`. The
  producer writes nothing after its opening event, so it returns on a later resume and the
  connection closes up to a second after the last event.
- FR-006 (US-003) WHEN the body's `stream` member is absent or `false`, THE SYSTEM SHALL answer the
  status and the body `serve.answer` gives for that body, as `application/json`.
- FR-007 (US-004) IF the body is one `18/FR-008` refuses, or the prompt's ids plus the budget minus
  one are past the positions the file allows as `18/FR-009` says, or `chat.render`,
  `tokenizer:encode` or a reservation `reserve` refuses fails, THEN THE SYSTEM SHALL answer that
  status and body as `application/json`, write no event, and answer nil as the producer. No model
  is built per request: that build is `serve.build`'s one model, and a reservation `reserve`
  refuses is that turn's fault.
- FR-009 (US-002) IF a turn's step returns a failure line, or `tltp.json_encode` gives nil for an
  event's data, THEN THE SYSTEM SHALL have `serve.step` end that turn without writing the
  `finish_reason` event or the `[DONE]` event; a streamed turn carrying a `session_id` whose step
  fails SHALL discard that conversation, so the next turn cannot answer from a half-read context.
- FR-010 (US-002) IF a streamed turn held for a free sequence cannot be admitted when a sequence
  frees, because no name is free or the run it needs cannot be opened, THEN THE SYSTEM SHALL end
  that stream with the `finish_reason` event holding `error` and then the `[DONE]` event, so a
  client waiting for an ending receives one.
  Not testable: the branch needs the reservation or the run to fail for a turn `prepared` already
  accepted, which no case can bring about without a model whose positions run out mid-turn; a
  reader confirms `abandoned` in `src/serve.tl`.

## Non-goals

- No other member in a streamed event: no `usage`, `id`, `object`, `created` or `model`. No request
  member beyond `stream` either: no `stream_options`.
- No handling of a client that goes away, of a failed write, of chunked framing, keep-alive
  comments or sleeps, and no holding back of a token that is half a character.
- No change to the finished reply or to the sampling: temperature, top-k, seed, the 256-token
  default budget and the end-id rules are `16`'s and `18`'s.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-005): `a streamed reply arrives token by token
  and ends with done`, proves on `18/T-001`'s fixture with the body `'{"messages":[{"role":"user",
  "content":"a"}],"max_tokens":2,"stream":true}'` that `serve.stream(server, body)` answers a
  response whose `status` is `OK`, whose `content_type` is `text/event-stream`, and a producer that
  is not nil, and that resuming that producer once with an `emit` that collects its argument and
  then driving `serve.step` to no live turns gives five payloads in this order: an object whose
  `choices` holds one object whose `index` is 0 and whose `delta` is `{role = "assistant"}`; one
  whose `delta` is `{content = "a"}`; one whose `delta` is `{content = "a"}` again; one whose
  `choices` holds one object whose `index` is 0, whose `finish_reason` is `length` and whose
  `delta` is nil; and the text `[DONE]`. It also proves, on a
  copy whose `tokenizer.ggml.eos_token_id` is 0, that the same body gives three payloads only: the
  `role` object, the `finish_reason` `stop` object and `[DONE]`, the first sampled id being the end
  id and never reported.
- T-002 (FR-006): `a body that does not ask for a stream answers json`, proves on T-001's fixture
  that `serve.stream` on T-001's body with its `stream` member removed answers a response whose
  `status` is `OK`, whose `content_type` is `application/json`, whose `body` equals the body
  `serve.answer` gives for that same request, and whose producer is nil, and that the same body
  with `"stream":false` added answers the same content type and the same body.
- T-003 (FR-007): `a stream request the route cannot use answers json and writes no event`, proves
  on T-001's fixture that each of these answers a response whose producer is nil, whose
  `content_type` is `application/json` and whose body decodes to `error` with the type and the
  message named: `{"messages":[],"stream":true}` answering `BadRequest` with
  `invalid_request_error` and a message naming `messages`; T-001's body with its `stream` member
  set to `"yes"` answering `BadRequest` with a message naming `stream`; T-001's body with the
  message content six copies of `a` answering `BadRequest` with a message naming `messages`,
  `max_tokens`, `6`, `256` and `260`; on a copy whose `tokenizer.chat_template` is removed, T-001's
  body answering `InternalServerError` with `internal_error` and a message naming
  `tokenizer.chat_template`; on a copy whose `tokenizer.chat_template` is the single byte `0xFF`,
  T-001's body answering `InternalServerError` with the line `tokenizer:encode` gives for that
  text; and on a copy whose metadata holds the tokenizer keys but no `embedding_length`, T-001's
  body answering `InternalServerError` with a message naming `model` and `embedding_length`.
- T-004 (FR-009): `a stream that cannot be generated or encoded ends without done`, proves on a
  copy of T-001's fixture whose third piece stores the single byte `0xFF` with token type 1 that
  the producer `serve.stream` gives for T-001's body with `"temperature":1`, `"top_k":2`,
  `"seed":1000` and `"max_tokens":2`, resumed once and then stepped by `serve.step` until no turn
  is live, leaves exactly one payload, the `role` object, and neither a `finish_reason` payload nor
  `[DONE]`, the first sampled id being the third piece, whose decode is the line `tokenizer:decode`
  gives for `{2}`; and that on a copy whose third piece is `ä` with token type 1, its decode being
  the single byte `0xE4` that `tltp.json_encode` refuses, the same body likewise collects that one
  payload and nothing more.

## Open questions

- None.

## Amendments

- None.
