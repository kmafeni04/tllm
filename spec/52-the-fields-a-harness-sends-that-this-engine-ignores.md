# 52: The fields a harness sends that this engine ignores

**Status**: done   **Depends on**: 18, 19, 50

## Problem & context

A harness such as pi drives this route with the current OpenAI chat-completions schema, so its
bodies carry members the route has never heard of. `18/FR-005` answers that by ignoring every member
it does not read, which is right for most of them and wrong for two:

- `stream_options.include_usage` asks for a usage chunk at the end of a streamed reply. A finished
  body already carries `usage` (`18/FR-004`), so the same request streamed silently loses the counts,
  and a client that keeps a conversation cannot tell how much context it holds.
- `max_completion_tokens` is the name the current schema uses for the length limit. The route reads
  only `max_tokens` (`18/FR-007`), so a harness configured for the new name gets the 256-token
  default and nothing says so: the reply is simply shorter than the caller asked for.

The rest of what a harness sends by default (`store`, `prompt_cache_key`, `service_tier`,
`parallel_tool_calls`, `metadata`) changes nothing here, and refusing it would break the clients that
send it without thinking, so it stays ignored.

`stream_options` and `max_completion_tokens` therefore join the members the route reads. Asked for, a
streamed reply ends with one more event between the event that names why the reply stopped and
`[DONE]`, holding `usage` with the same counts the finished body of that same request holds;
`tltp.json_encode` writes an empty table as `[]`, so that event's `choices` is an empty array, the
shape OpenAI clients already parse. `max_completion_tokens` is the other name for the budget, and a
body carrying both names is refused rather than guessed at.

The counts a streamed turn has at its ending are the ones the finished reply answers with: the ids
the prompt encoded to, the ids that turn sampled, and the prompt ids it did not forward. The record
the route builds from the body is unexported, so both members join it in the code and no snippet in
`18` or `19` gains a field. A pooled server needs nothing of its own: `28`'s worker writes the
producer's payloads as frames and the parent relays them, so the usage event travels with them.

## User stories

- US-001 (P1) As a harness that streams a reply, I want the stream to end with the usage counts when I ask for them, so that I know how much context the conversation holds.
- US-002 (P1) As a harness built on the current OpenAI schema, I want `max_completion_tokens` to set the length limit, so that my reply is not silently cut to the default budget.
- US-003 (P2) As a harness that sends members this engine has no meaning for, I want them ignored rather than refused, so that a default-configured request still works.

## Functional requirements

- FR-001 (US-001) WHEN a body's `stream` member is `true` and its `stream_options` member is a JSON
  object holding an `include_usage` member that is `true`, THE SYSTEM SHALL end that streamed reply
  with one event after the event that names why the reply stopped and before the `[DONE]` event,
  whose data is a JSON object whose `choices` is an empty array and whose `usage` object holds
  `prompt_tokens` as the number of ids the rendered prompt encodes to, `completion_tokens` as the
  number of ids that turn sampled, `total_tokens` as their sum and `prompt_tokens_details` as an
  object whose `cached_tokens` is the number of the prompt's ids that turn did not forward, those
  four being the counts the finished body of that same request answers with.
- FR-002 (US-001) IF a streamed body carries no `stream_options` member, or one whose `include_usage`
  member is absent or `false`, or that reply never reaches an ending event as `19/FR-009` gives it,
  or ends at an ending naming `error` as `19/FR-010` and `50/FR-009` give it, THEN THE SYSTEM SHALL
  write no usage event, whatever the body asked for.
- FR-003 (US-001) IF the body's `stream_options` member is present and is not a JSON object, or holds
  an `include_usage` member that is not a boolean, THEN THE SYSTEM SHALL answer `BadRequest` with
  `18/FR-008`'s body, its message naming `stream_options`, that check coming after the `stream`
  check and before the `session_id` check.
- FR-004 (US-001) WHEN the body's `stream` member is absent or `false`, THE SYSTEM SHALL read
  `stream_options` for its shape only and answer the finished body `18/FR-004` gives, which already
  holds `usage`.
- FR-005 (US-002) WHEN a body carries `max_completion_tokens` and no `max_tokens`, THE SYSTEM SHALL
  take it as the token budget as `18/FR-007` reads `max_tokens`, and keep the 256-token default when
  a body carries neither name.
- FR-006 (US-002) IF a body carries `max_completion_tokens` and it is not a whole number of one or
  more, THEN THE SYSTEM SHALL answer `BadRequest` with `18/FR-008`'s body, its message naming
  `max_completion_tokens`, that check coming after `max_tokens`'s own.
- FR-007 (US-002) IF a body carries both `max_tokens` and `max_completion_tokens`, THEN THE SYSTEM
  SHALL answer `BadRequest` with `18/FR-008`'s body, its message naming both `max_tokens` and
  `max_completion_tokens`, that check coming after both members' own shape checks and before the
  `seed` check.
- FR-008 (US-002) WHEN the budget check `18/FR-009` refuses a body, THE SYSTEM SHALL name in its
  message the member the body carried as the budget: `max_completion_tokens` when that is the name it
  carried, and `max_tokens` when it carried `max_tokens` or neither.
- FR-009 (US-003) THE SYSTEM SHALL keep ignoring every body member it does not read, and `store`,
  `prompt_cache_key`, `service_tier`, `parallel_tool_calls` and `metadata` are among them: a body
  carrying any of them answers exactly as the same body without it.

## Non-goals

- No meaning for `store`, `prompt_cache_key`, `service_tier`, `parallel_tool_calls`, `metadata`,
  `n`, `response_format` or `logprobs`: they stay ignored, and `n > 1` stays out of the route.
- No usage in any other streamed event and no counts for a reply that ends in error: only a turn that
  reached its own ending answers them.
- No new sampling or window semantics: the 256-token default, the positions check and the sampler
  stay `16`'s and `18`'s.

## Tests

- T-001 (FR-001, FR-002): `a streamed reply asked for usage ends with a usage event`, proves on
  `18/T-001`'s fixture that `serve.stream(server, '{"messages":[{"role":"user","content":"a"}],
  "max_tokens":2,"stream":true,"stream_options":{"include_usage":true}}')` answers a response whose
  `status` is `OK` and whose `content_type` is `text/event-stream`, and that resuming its producer
  once and then driving `serve.step` to no live turns gives six payloads in this order: the
  `{role = "assistant"}` object, two `{content = "a"}` objects, the `finish_reason` `length` object,
  an object whose `choices` is an empty array and whose `usage` equals the `usage` of the body
  `serve.answer` gives for that same body with its `stream` and `stream_options` members removed
  (`prompt_tokens` 1, `completion_tokens` 2, `total_tokens` 3, `prompt_tokens_details.cached_tokens`
  0), and the text `[DONE]`; that the same body with `"include_usage":false` and the same body with
  its `stream_options` member removed both give the five payloads `19/T-001` records, with no usage
  payload; and that on the copy `19/T-004` makes, whose third piece stores the single byte `0xFF`,
  T-001's body with `"temperature":1`, `"top_k":2`, `"seed":1000` and `"max_tokens":2` leaves exactly
  one payload, the `{role = "assistant"}` object, with no usage payload and no `[DONE]`; and that on
  the `first_token` fixture `50/T-003` uses, holding that case's body (content `c`, `max_tokens` 1,
  `stream` true) with `"stream_options":{"include_usage":true}` added, the payloads are exactly the
  four `50/T-003` records, the `role` object, the `{content = "Hi"}` object, the `finish_reason`
  `error` object and `[DONE]`, with no payload holding a `usage` member.
- T-002 (FR-005, FR-006, FR-007, FR-008): `max_completion_tokens sets the budget`, proves on
  `18/T-001`'s fixture that `answer(server, '{"messages":[{"role":"user","content":"a"}],
  "max_completion_tokens":8}')` answers `OK` with `content` `a` written eight times,
  `completion_tokens` 8 and `total_tokens` 9; that a body holding `"max_completion_tokens":2.5` and
  one holding `"max_completion_tokens":0` each answer `BadRequest` with a message naming
  `max_completion_tokens`; that a body holding both `"max_tokens":4` and `"max_completion_tokens":8`
  answers `BadRequest` with a message naming both `max_tokens` and `max_completion_tokens`; that a
  body holding `"max_tokens":0` and `"max_completion_tokens":8` answers `BadRequest` with a message
  naming `max_tokens` and not `max_completion_tokens`; and that a body holding
  `"max_completion_tokens":300` answers `BadRequest` with a message naming `messages`,
  `max_completion_tokens`, `1`, `300` and `260`, one prompt id plus the 300-token budget minus one
  being past the 260 positions the file allows.
- T-003 (FR-003, FR-004, FR-009): `stream_options is checked and the unread members stay ignored`,
  proves on `18/T-001`'s fixture that each of these bodies, none of which carries a `stream` member,
  answers `BadRequest` with a message naming `stream_options`: `"stream_options":true`,
  `"stream_options":"yes"`, `"stream_options":{"include_usage":"yes"}`, `"stream_options":null` and
  `"stream_options":[]`; that a body holding
  `"stream_options":{"include_usage":true}` and no `stream` member answers the same status and body
  as the same body without it; that a body holding
  `"stream_options":{"include_usage":true,"include_obfuscation":true}` streams as T-001's body does;
  and that a body holding `"store":false`, `"prompt_cache_key":"k"`, `"service_tier":"auto"`,
  `"parallel_tool_calls":false` and `"metadata":{"x":"y"}` answers `OK` with exactly the body the
  same body without those five members answers.

## Open questions

- None.

## Amendments

- None.
