# 50: Stream a tool call

**Status**: done   **Depends on**: 19, 31, 49

## Problem & context

A streamed reply sends each text the run releases as `delta.content` the moment it is released
(`19/FR-004`), and a call is read only after the generation has ended (`49/FR-001`). The two cannot
meet: by the time the reply text is scanned for `<tool_call>`, its bytes have already gone out as
content, and an event cannot be taken back, so a harness sees prose where `49` would have answered a
call. Streaming a call is impossible until the producer can delay the text that might be one.

`31` already solved this shape for another trigger: a run holds the tail of the text it produced
while that tail could still complete a stop string, and gives it when the run ends. A call marker is
the same problem with a different trigger, so this spec gives the producing turn the same hold for
`<tool_call>`: nothing is sent while the unsent text ends in a prefix of the opening marker or holds
an opening marker with no closing marker after it; a block that completes is read with `49`'s reader
and sent as `delta.tool_calls` events; text that turns out not to be a call is sent as `content`
when it is released, or at the run's end when a hold kept it back. The marker pair and the reader
stay in `chat`, beside the template key, the one fact `49` keeps there, so the stream finds a block
by the same bytes and reads it with the same code.

A call cannot be cut by a stop string: a string inside the block would end the run before the
closing marker and the call would arrive as prose. A body that carries both one or more tool
definitions and stop strings is therefore refused with `BadRequest` naming `stop`, for a finished
reply as well as for a streamed one. That refusal is new for the finished reply too.

The deltas are the call `49/FR-003` answers, split so a client can reassemble it: `index`, `id`,
`type` and `name` in one event, the arguments text in the next, then `finish_reason` `tool_calls`
and `[DONE]`. Text outside the markers streams as `content`, so a client's content is the reply's
text with the blocks cut out, where the finished reply drops that prose as `49/FR-003` says:

```
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]}}]}
data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}
data: [DONE]
```

This spec strikes three non-goal bullets and one clause when it ships: the whole of `49`'s "No
streamed call" bullet, the whole of `47`'s "No streamed tool calls" bullet, the whole of `22`'s "No
streaming a call out of a reply" bullet, and the streaming clause of `48`'s "No streamed tool call
out of a reply" bullet, its "and `reasoning_content` stays unread" standing.

Three requirements are amended when this spec ships: `18/FR-008` gains one more check, `19/FR-004`
says what a turn's events carry and when, and `19/FR-005` names the endings a turn that has sent a
call gives.

## User stories

- US-001 (P1) As a harness author, I want a call the model writes as text to arrive as
  `delta.tool_calls` events I can reassemble, so that I run the tool while the reply is still being
  generated.
- US-002 (P2) As a streaming client, I want text that turns out not to be a call to reach me as
  `content` and the reply's text outside the markers too, so that no text is lost and nothing
  arrives twice.
- US-003 (P2) As a streaming client, I want a broken call block to end the stream with an error
  ending and `[DONE]`, so that I am not left waiting for a call that will never come.
- US-004 (P2) As the person running the server, I want a body asking for tools and stop strings
  refused with one line, so that a call the client asked for is never silently cut.

## Functional requirements

- FR-001 (US-001) WHEN a streamed reply's unsent text holds a complete block that, after its
  leading spaces, tabs, newlines and carriage returns, begins with `<function=` and holds the shape
  `49/FR-002` reads, that text being what `51/FR-006`'s hold leaves and read after that hold runs,
  THE SYSTEM SHALL send no event whose `delta.content` holds any byte of that
  block, and SHALL, for each such block in the order the blocks appear, send the bytes before it as
  FR-007 gives them, then read one call from it as `49/FR-002` reads it and cut it, its markers
  included, out of its unsent text, so a block's own bytes are never sent and the bytes after it
  are the text the turn has not yet sent, and while a reply's head has not settled no complete call
  block standing inside that held text answers a call event.
- FR-002 (US-001) WHEN a call has been read as FR-001 says, THE SYSTEM SHALL send one event whose
  `choices` is an array holding one object whose `index` is 0 and whose `delta` is an object whose
  `tool_calls` is a list holding one object whose `index` is the call's position among the reply's
  calls counting from zero, whose `id` is `call_` then that position counting from one, whose
  `type` is `function`, and whose `function` is an object whose `name` is the call's name.
- FR-003 (US-001) WHEN the event of FR-002 for a call has been sent, THE SYSTEM SHALL send one
  event whose `choices` is an array holding one object whose `index` is 0 and whose `delta` is an
  object whose `tool_calls` is a list holding one object whose `index` is that call's position and
  whose `function` is an object whose `arguments` is the call's arguments JSON text, the text
  `49/FR-003` puts in that call's `function.arguments`.
- FR-004 (US-001) WHEN a streamed turn's run ends, that turn has sent an event of FR-002 and no
  block FR-009 names has ended it, THE SYSTEM SHALL send, after the events FR-001 to FR-003 give
  for the text its last release holds and in this order, one event whose `choices` is an array
  holding one object whose `index` is 0, holding no `delta`, whose `finish_reason` is
  `tool_calls`, then one event whose data is the text `[DONE]`.
- FR-005 (US-002) WHILE a streamed turn's run is live and the text that turn has not yet sent, as
  `51/FR-006`'s hold leaves it, ends
  in a proper prefix of `<tool_call>`, THE SYSTEM SHALL send no event whose `delta.content` holds
  any of that suffix.
- FR-006 (US-002) WHILE a streamed turn's run is live and the text that turn has not yet sent, as
  `51/FR-006`'s hold leaves it, holds
  `<tool_call>` with no `</tool_call>` after it, THE SYSTEM SHALL send no event whose
  `delta.content` holds any of the bytes from that marker on.
- FR-007 (US-002) WHEN the text a streamed turn has not yet sent, as `51/FR-006`'s hold leaves it,
  holds no suffix FR-005 names,
  no marker FR-006 names, no block FR-009 names and no block FR-001 names, THE SYSTEM SHALL send
  the whole of it as one event whose `delta.content` is that text when it holds one or more bytes.
  When that text holds a block FR-001 names, the bytes it holds from the start of its buffer up to
  that block's opening marker go out under this requirement as one event whose `delta.content` is
  those bytes when they hold one or more bytes, before the events FR-002 and FR-003 give for that
  block, and the bytes after the block's closing marker are the text the turn has not yet sent, so
  the bytes before a block, the bytes between two blocks and the bytes after the last block each go
  out as their own event or as the text FR-008 releases, and no event holds bytes from two sides
  of a block. That text is the text left once a head block `51/FR-007` has sent, so a reply holding
  both sends one reasoning event ahead of its first content event.
- FR-008 (US-002) WHEN a streamed turn's run ends and FR-001 to FR-007 have run for the text its
  last release left, THE SYSTEM SHALL send the text it still holds as one event whose
  `delta.content` is that text before the turn's ending event, when that text holds one or more
  bytes and holds no block FR-009 names. That release reaches the text `51/FR-006` held behind a
  partial opening marker as well as the text FR-005 held behind a call marker's prefix, so a turn
  whose head prefix nothing completed sends that text as content and loses no byte.
- FR-009 (US-003) IF the text a streamed turn has not yet sent, as `51/FR-006`'s hold leaves it,
  holds a complete block that begins,
  after its leading spaces, tabs, newlines and carriage returns, with `<function=` and does not
  hold the shape `49/FR-002` reads, THEN THE SYSTEM SHALL send no event holding any of that block's
  bytes, SHALL send the bytes before that block as one event whose `delta.content` is those bytes
  when they hold one or more bytes, SHALL send one event whose `choices` is an array holding one
  object whose `index` is 0, holding no `delta`, whose `finish_reason` is `error`, then one event
  whose data is the text `[DONE]`, that ending standing in place of the one FR-004 gives, and SHALL
  drop the turn as `19/FR-009` drops a failed one, so a session the turn names is not stored by it,
  and while a reply's head has not settled no unreadable call block standing inside that held text
  answers that ending and dropped turn.
- FR-010 (US-004) IF the body carries a `tools` member that `47/FR-006` accepts and that holds one
  or more entries, and a `stop` member that `31/FR-007` accepts, THEN THE SYSTEM SHALL answer
  `BadRequest` as `18/FR-008` describes it with a message naming `stop`, as `application/json`,
  generating no token and answering nil as the streamed producer, that check running after
  `47/FR-006`'s and before anything is rendered or generated.

## Non-goals

- No change to the finished reply of a body carrying no `tools` member or no `stop` member: a body
  without `stream` answers the body `49/FR-003` and `49/FR-004` give, and a broken block there is
  still the `InternalServerError` `49/FR-005` gives. The one exception is FR-010's `BadRequest`, a
  request-shape refusal that is new for the finished reply too.
- No refusal beyond FR-010: a body carrying `stop` with no tools, or with an empty `tools` list, is
  not refused, so a call the model writes there can still be cut by a stop string.
- No change to sampling: no id is stopped, steered or forced by a call, no `tool_choice` makes a
  call, and the holdback changes only when text is sent, never what the model produces or what a
  session keeps.
- No other delay: a token that is half a character is still sent as it is, no `usage` joins an event
  that is not the usage event `52/FR-001` gives, and a JSON block form stays the text `49` does not
  read, streamed as content.

## Tests

T-001 and T-002 use two copies of `18`'s fixture whose tokens are three: `c` at token id 2, which
the one user message `c` encodes to, and two texts at ids 0 and 1. Their `token_embd.weight` rows are
`{1,0}`, `{0,1}` and `{0,1}`, and their `output.weight` rows are `{0,1}`, `{1,0}` and `{-1,-1}`, so
the scores of the prompt pick id 0 and the scores after id 0 pick id 1; their
`tokenizer.ggml.eos_token_id` is 2, so neither sampled id is the end id. In the first copy, called
`BLOCK_PIECES`, id 0 holds
`<tool_call><function=get_weather><parameter=city>Paris</parameter></function>` and id 1 holds
`</tool_call>`; in the second, called `HALF_MARKER`, id 0 holds `<tool_ca` and id 1 holds `ll`. The
texts are the ones `49`'s tests call `CALL` and `TWO_CALLS`, and `first_token` is the helper those
tests build. A payload is read with `tltp.json_decode` unless it is `[DONE]`.

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-007): `a streamed call arrives as tool_calls deltas`,
  proves on four fixtures with a body whose one user message content is `c`, whose `max_tokens` is
  the number named and which adds `"stream":true`: on `BLOCK_PIECES` with `max_tokens` 2, that
  `serve.stream` answers `OK` with a producer that, resumed once and then stepped by `serve.step`
  until no turn is live, writes exactly five payloads, the `role` object; one whose `delta` is
  `{tool_calls = {{index = 0, id = "call_1", type = "function", ["function"] = {name =
  "get_weather"}}}}`; one whose `delta` is `{tool_calls = {{index = 0, ["function"] = {arguments =
  '{"city":"Paris"}'}}}}`; one whose `finish_reason` is `tool_calls`, whose `index` is 0 and whose
  `delta` is nil; and the text `[DONE]`, that call being a block that id 0's text leaves unclosed,
  so the hold is what keeps it off the wire until id 1 closes it; on `first_token(CALL)` with
  `max_tokens` 1, that the payloads are, in order, the `role` object; one whose `delta` is
  `{content = "Hi"}`; the two call payloads above; one whose `delta` is `{content = "done"}`; the
  `tool_calls` `finish_reason` object; and `[DONE]`, no payload carrying any text of the block
  itself; and on `first_token` of
  `Hi<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>mid<tool_call><function=add><parameter=a>1</parameter></function></tool_call>done`
  with `max_tokens` 1, that the payloads are, in order, the `role` object; one whose `delta` is
  `{content = "Hi"}`; the two `get_weather` payloads named above; one whose `delta` is `{content =
  "mid"}`; the two `add` payloads, `{index = 1, id = "call_2", type = "function", ["function"] =
  {name = "add"}}` and `{index = 1, ["function"] = {arguments = '{"a":1}'}}`; one whose `delta`
  is `{content = "done"}`; the `tool_calls` ending and `[DONE]`, the bytes before a block, between
  two blocks and after the last block each arriving as their own content event; and on
  `first_token(TWO_CALLS)` with `max_tokens` 1, that the payloads are the `role`
  object, eight call payloads, the `tool_calls` ending and `[DONE]`, the call payloads' `index`
  members being 0, 0, 1, 1, 2, 2, 3, 3 and their `id` members `call_1`, `call_1`, `call_2`,
  `call_2`, `call_3`, `call_3`, `call_4` and `call_4`, their `function` members being `{name =
  "get_weather"}`, `{arguments = '{"city":"Paris"}'}`, `{name = "add"}`, `{arguments = '{"a":1}'}`,
  `{name = "extra"}`, `{arguments = '{"opts":{"unit":"c"}}'}`, `{name = "empty"}` and
  `{arguments = '{"opts":{}}'}`.
- T-002 (FR-005, FR-006, FR-007, FR-008): `a reply that is not a call releases its text as
  content`, proves on `HALF_MARKER` with `max_tokens` 2 that the producer writes exactly four
  payloads, the `role` object; one whose `delta` is `{content = "<tool_call"}`; one whose
  `finish_reason` is `length` and whose `delta` is nil; and `[DONE]`, the two sampled pieces
  arriving as one content event because the first ended in an opening marker's prefix; on
  `first_token` of `<tool_call><function=get_weather><parameter=city>Paris</parameter></function>`
  with `max_tokens` 1, that the payloads are the `role` object, one whose `delta` is `{content =
  that text}`, the `length` ending and `[DONE]`, an opening marker with no closing marker after it
  never being withheld from the run's ending; and on `first_token` of `x<tool_call>{"a":1` with
  `max_tokens` 1, that the payloads are the `role` object, one whose `delta` is `{content = that
  text}`, the `length` ending and `[DONE]`, the bytes before an unfinishable marker going out in
  that one event rather than alone; and on `first_token` of
  `<tool_call>{"name":"get_weather","arguments":{}}</tool_call>` with `max_tokens` 1, that the
  payloads are the `role` object, one whose `delta` is `{content = that text}`, the `length` ending
  and `[DONE]`, a complete block that is not a call block being content.
- T-003 (FR-009): `a streamed broken call block ends with an error ending`, proves on `first_token`
  of `Hi<tool_call><function=></function></tool_call>` with a body whose content is `c`, whose
  `max_tokens` is 1, whose `session_id` is `s` and which adds `"stream":true`, that the producer
  writes exactly four payloads, the `role` object; one whose `delta` is `{content = "Hi"}`, the
  bytes before the broken block; one whose `finish_reason` is `error`, whose `index` is 0 and
  holding no `delta`; and `[DONE]`, no payload holding the block's text or a `tool_calls` member,
  and that `serve.session_of(server, "s")` is nil.
- T-004 (FR-010): `a body carrying tools and stop is a bad request`, proves on `18`'s fixture that
  a body whose one user message content is `a` and which carries the one tool
  `{"type":"function","function":{"name":"b"}}` and `"stop":"a"` answers, through `serve.answer`,
  `BadRequest` with a body decoding to an object whose only member is `error`, of type
  `invalid_request_error`, whose message names `stop`, and, through `serve.stream` with
  `"stream":true` added, a response whose status is `BadRequest`, whose content type is
  `application/json` and whose producer is nil; that the same body with `"stop":["a","b"]` in place
  of `"stop":"a"` answers the same; that the same body without `stop` answers `OK` through
  `serve.answer`, and so does the body with `"stop":"a"` and no `tools`, and so does the body with
  `"stop":"a"` and `"tools":[]`, an empty list asking for no tool a stop string could cut; and that
  a body carrying both `"tools":5` and `"stop":"a"` answers `BadRequest` with a message naming
  `tools` and no `stop`, the tool check answering first.

## Open questions

- None.

## Amendments

- None.
