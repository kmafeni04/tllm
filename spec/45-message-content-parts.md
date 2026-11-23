# 45: A message whose content is a list of parts

**Status**: done   **Depends on**: 18

## Problem & context

The route accepts a message only when `content` is a string. Every harness built on the OpenAI
schema sends `content` as a list of typed parts, `[{"type":"text","text":"..."}]`, so a request
from one is refused with `BadRequest` before a token is generated. Nothing downstream needs the
list: a template writes one string per message, and the list carries that string as text parts.

A part of another kind, an image for instance, carries no text the engine can use, and no block
reads one, so it is refused by name rather than silently dropped: a client that sends an image
must learn that the image never reached the model.

`18/FR-008` fixes the current check and is amended. `22` records as a non-goal that a content is
only a string; that bullet is struck from `22` when this spec ships, because it stops being true.

## User stories

- US-001 (P1) As a harness author, I want a message's content accepted as text parts, so that my
  request reaches the model instead of being refused for its shape.
- US-002 (P2) As a harness author, I want a part this engine cannot read refused by its kind and
  position, so that I know what to stop sending rather than reading a generic bad request.

## Functional requirements

- FR-001 (US-001) WHEN a message's `content` is an array of one or more parts, each an object
  whose `type` is `text` and whose `text` is a string, THE SYSTEM SHALL use as that message's
  content the `text` of each part joined in the array's order with nothing between them.
- FR-002 (US-001) WHEN a message's `content` is a string, THE SYSTEM SHALL use that string
  unchanged.
- FR-003 (US-002) IF a part's `type` is a string other than `text`, THEN THE SYSTEM SHALL answer
  `BadRequest` with a body whose `error.message` names the message's position in `messages`, the
  part's position in that array, and that `type` value.
- FR-004 (US-002) IF a part is not an object whose `type` is a string, or is a part whose `type` is
  `text` and whose `text` is not a string, THEN THE SYSTEM SHALL answer `BadRequest` with a body
  whose `error.message` names the message's position in `messages` and the part's position in that
  array.
- FR-005 (US-002) IF a message's `content` is neither a string nor an array of one or more parts,
  THEN THE SYSTEM SHALL answer `BadRequest` with a body whose `error.message` names the message's
  position in `messages`.

## Non-goals

- No image, audio or video part: every `type` other than `text` is refused, and no branch that
  reads one is built.
- No other message field: `tool_calls`, `tool_call_id`, `name` and `reasoning_content` stay
  unread, and the body still carries no `tools`.
- No change to `chat.render` or to the template: a message's content is still one string.

## Tests

- T-001 (FR-001, FR-002): `text parts answer as the joined string`, proves on `18`'s fixture that
  a body whose single user message carries the part `{"type":"text","text":"a"}` answers `OK` with
  `content` of `a` written 256 times and `usage` `{prompt_tokens = 1, completion_tokens = 256,
  total_tokens = 257, prompt_tokens_details = {cached_tokens = 0}}`; that a body whose single user
  message carries the parts `{"type":"text","text":"b"}` then `{"type":"text","text":"a"}` answers
  `OK` with the same 256 `a`s and `usage` `{prompt_tokens = 2, completion_tokens = 256,
  total_tokens = 258, prompt_tokens_details = {cached_tokens = 0}}`, where the reversed order
  answers content beginning `caa`; and that on a copy whose `tokenizer.chat_template` is
  `{% for message in messages %}{% if message.role == 'system' %}a{% else %}b{% endif %}{% endfor %}`
  a body whose single user message carries the parts `a`, `b` and `c` answers `prompt_tokens` 1,
  the count the body with the one content `abc` has on that same copy, where one message per part
  would answer 3.
- T-002 (FR-003, FR-004, FR-005): `a part or message the engine cannot read is a bad request`,
  proves on `18`'s fixture that each of these answers `BadRequest` with a body decoding to `error`
  of type `invalid_request_error`: the body
  `{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"text","text":"b"},{"type":"image_url","image_url":{"url":"x"}}]}]}`,
  its message naming `messages`, `2`, `2` and `image_url`; the body
  `{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"text","text":"b"},{"type":"text"}]}]}`,
  its message naming `messages`, `2` and `2`; the body
  `{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"text":"a"},{"type":"text","text":"b"}]}]}`,
  its message naming `messages`, `2` and `1`; and the body
  `{"messages":[{"role":"user","content":"a"},{"role":"user","content":[]}]}`, its message naming
  `messages` and `2`.

## Open questions

- None.

## Amendments

- None.
