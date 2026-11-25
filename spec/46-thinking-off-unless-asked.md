# 46: Thinking off unless the request asks for it

**Status**: done   **Depends on**: 17, 18, 20, 22

## Problem & context

`17/FR-002` binds `add_generation_prompt` and nothing else, and `20/FR-001` states that
`add_vision_id`, `enable_thinking`, `preserve_thinking` and `reasoning_content` are left unbound on
purpose: that spec's job was to render a plain text conversation, and binding names no request could
supply yet would have been invented behaviour. The local Qwen-family file's template ends the
generation opener with `{%- if enable_thinking is defined and enable_thinking is false %}`, so an
unbound name fails its own `is defined` test and the template takes the `else` branch, which writes
`<think>\n`: an open thinking block. Every chat reply from those files therefore begins with the
model's reasoning. That reasoning is slow, it spends the `max_tokens` budget `16` cuts the turn at,
and it arrives inside `content`, where the harness that sent the request shows it as the answer.

The decision belongs to the request, because the template asks the caller. Clients built on the
OpenAI schema send `enable_thinking` at the top level of the body; clients built for the servers
that pass variables into a template send `chat_template_kwargs.enable_thinking`. This spec accepts
both, the nested one winning, and reads no other member of that object. Absent means false, because
a harness that wants thinking asks for it, and the guess that costs least when a caller says
nothing is the plain answer.

Two layers change. `chat.render` gains an optional boolean after the tool list and binds the
template name only when it is given, so a caller that gives nothing keeps the prompt `20/T-001` and
`22/T-001` record, and a template that never tests the name cannot tell the difference at all. The
route always resolves a value, so an HTTP request never reaches the `is defined` test that sends a
template into its open-thinking branch.

```teal
local record chat
  render: function(
    tokenizer.Tokenizer, {Message}, tools?: json.Value,
    enable_thinking?: boolean
  ): string | nil, string | nil
end
```

A member of the wrong kind is refused rather than ignored, in the style every other body member
already answers in (`stream: not a boolean`), and `request_of` runs before the model is touched, so
a refusal costs no generation. The route reads the body through `tltp.json_decode`, whose unpack
gives one entry-less table for an empty object and an empty array, so those two are the collision
this spec names and settles: no reading of the value tells them apart and neither is refused. A JSON
null arrives as the distinct sentinel that same unpack hands back, the value that makes
`{"stream":null}` a refusal today, so a null thinking member is refused like any other kind the
route cannot read. A pooled server needs nothing of its own: `28` forwards the body whole to the
worker, which resolves the member from that body exactly as this spec states.

## User stories

- US-001 (P1) As a person running a chat request, I want a reply that starts with the answer instead of the model's reasoning, so that the turn is fast and its budget is spent on what I asked for.
- US-002 (P1) As a harness author, I want to ask for thinking in the body, at the top level or inside `chat_template_kwargs`, so that my own client's shape works without a translation step.
- US-003 (P2) As a caller of `chat.render`, I want a call that gives no value to bind nothing, so that the prompts written before this change still come out byte for byte.
- US-004 (P2) As a person sending a body this engine cannot read, I want one line naming the member at fault, so that I fix that member instead of guessing which field broke the request.

## Functional requirements

- FR-001 (US-001) WHEN a chat completions request body carries neither a top-level
  `enable_thinking` member nor a `chat_template_kwargs` member holding `enable_thinking`, THE SYSTEM
  SHALL bind the template name `enable_thinking` to false for that request's render.
- FR-002 (US-002) WHEN the body carries a top-level `enable_thinking` whose value is a boolean, THE
  SYSTEM SHALL bind the template name `enable_thinking` to that value except where `FR-003` binds the
  nested one.
- FR-003 (US-002) WHEN the body carries a `chat_template_kwargs` object whose `enable_thinking`
  member is a boolean and whose top-level `enable_thinking` member is a boolean or absent, THE
  SYSTEM SHALL bind the template name `enable_thinking` to the nested value rather than to the
  top-level one, whatever that top-level boolean holds.
- FR-004 (US-002) WHEN the body carries a `chat_template_kwargs` that holds no `enable_thinking`
  member, including the entry-less table the body decoder gives for an empty object and an empty
  array alike, THE SYSTEM SHALL bind the template name `enable_thinking` from the top-level
  `enable_thinking` member when that member is a boolean, or to false as `FR-001` states when the
  body carries no such member.
- FR-005 (US-004) IF the body carries a top-level `enable_thinking` or a
  `chat_template_kwargs.enable_thinking` whose value is present and not a boolean, THEN THE SYSTEM
  SHALL answer `BadRequest` with a body whose `error.message` names that member as it is written in
  the body, either `enable_thinking` or `chat_template_kwargs.enable_thinking`, and generate no
  token, JSON null being one of those kinds. No body that `FR-005` or `FR-006` refuses reaches any
  binding `FR-001` to `FR-004` states: a thinking member the route cannot read is refused before any
  member binds or wins. That check comes after the `stop` check `18/FR-008` states, so a body
  carrying a fault that requirement already orders answers that fault's member, this one being last.
- FR-006 (US-004) IF the body carries a `chat_template_kwargs` whose value is JSON null, a string, a
  number, a boolean or an array holding one or more entries, THEN THE SYSTEM SHALL answer
  `BadRequest` with a body whose `error.message` names `chat_template_kwargs`, and generate no
  token, that check coming after `FR-005`'s. An empty object and an empty array decode to one
  entry-less table that no reading tells apart, so neither is refused here and `FR-004` reads both as
  holding no such member.
- FR-007 (US-003) WHEN `chat.render` is called with a boolean as its thinking input, THE SYSTEM
  SHALL bind the template name `enable_thinking` to that value and bind no other name that the call
  did not already bind.
- FR-008 (US-003) IF `chat.render` is called without a value for its thinking input, THEN THE
  SYSTEM SHALL leave `enable_thinking` unbound, so that the prompt `20/T-001` and `22/T-001` record
  for such a call is unchanged and a template that never tests the name writes the text it writes
  today.

## Non-goals

- No other template input: `preserve_thinking`, `reasoning_content` and `add_vision_id` stay
  unbound, every other member of `chat_template_kwargs` is unread, and no config key or CLI flag
  sets a thinking default.
- No request-side tools of this spec's own: the tool list the body may now carry is `47`'s, and
  `chat.render`'s thinking input stays the fourth argument, after that tool list.

## Tests

- T-001 (FR-007, FR-008): `the value a render call is given decides the closer the file's template
  writes`, proves on the file's own template as `20/T-001` embeds it, over the one user message
  `hi`, that `render` with the value false returns `22/T-001`'s `ONE_TURN` with its tail
  `"<|im_start|>assistant\n<think>\n"` replaced by
  `"<|im_start|>assistant\n<think>\n\n</think>\n\n"`, that `render` with the value true returns
  `ONE_TURN` itself, and that `render` given no value returns `ONE_TURN` too, the prompt that call
  already wrote; and proves on the hand-made template
  `{% if enable_thinking is defined %}t{% endif %}{% if preserve_thinking is defined %}p{% endif %}`
  that `render` with the value false writes `t` alone, that `render` with the value true writes `t`
  alone, and that `render` given no value writes nothing, so the thinking input binds its own name
  and reaches no other.
- T-002 (FR-001, FR-002, FR-003, FR-004): `the body's thinking member decides the prompt, and the
  nested one wins`, proves on a copy of `18`'s fixture whose `tokenizer.chat_template` is replaced,
  as `18/T-004` and `45/T-001` replace theirs, by
  `{% for message in messages %}{{ message.content }}{% endfor %}{% if enable_thinking is defined
  and not enable_thinking %}c{% endif %}`, so a request bound to false adds one id to the prompt;
  and proves that `18`'s own fixture keeps the template it has, so no prompt and no usage count any
  earlier spec records moves. Each case posts `{"messages":[{"role":"user","content":"a"}]}`
  plus the member named: a body carrying neither member answers `usage.prompt_tokens` 2, the false
  default of `FR-001`; `"enable_thinking":true` answers 1; `"enable_thinking":false` answers 2;
  `"chat_template_kwargs":{"enable_thinking":true}` answers 1; the top-level `true` with the nested
  `false` answers 2, the nested value winning; the top-level `true` with `"chat_template_kwargs":{}`
  answers 1, that object holding no such member; and the top-level `true` with
  `"chat_template_kwargs":[]` answers the same 1, an empty array being the entry-less table
  `FR-004` reads as holding no such member.
- T-003 (FR-005, FR-006): `a thinking member of the wrong kind is a bad request`, proves on `18`'s
  fixture that each of these answers `BadRequest` with a body decoding to `error` of type
  `invalid_request_error` and no reply content, and generates no token:
  `{"messages":[{"role":"user","content":"a"}],"enable_thinking":"yes"}`, naming
  `enable_thinking`; the same body with `"enable_thinking":1`, naming `enable_thinking`; the same
  body with `"chat_template_kwargs":{"enable_thinking":"yes"}`, naming
  `chat_template_kwargs.enable_thinking`; the same body carrying `"enable_thinking":true` and
  `"chat_template_kwargs":{"enable_thinking":"yes"}`, naming
  `chat_template_kwargs.enable_thinking`, a nested value the route cannot read being refused beside
  a readable top-level one; the same body with `"chat_template_kwargs":5`, with
  `"chat_template_kwargs":true`, with `"chat_template_kwargs":"x"` and with
  `"chat_template_kwargs":null`, each naming `chat_template_kwargs`; the same body with
  `"chat_template_kwargs":["a"]`, one array holding an entry, naming `chat_template_kwargs`; the
  same body carrying `"enable_thinking":"yes"` and `"chat_template_kwargs":5`, naming
  `enable_thinking`, `FR-005`'s check coming first; the same body carrying `"enable_thinking":null`,
  naming `enable_thinking`; the same body carrying `"enable_thinking":"yes"` and
  `"chat_template_kwargs":{"enable_thinking":true}`, naming `enable_thinking`, a nested boolean
  winning over a top-level boolean but never over a top-level value the route refuses; and the same
  body carrying `"stream":5` and `"enable_thinking":"yes"`, naming `stream`.

## Open questions

- None.

## Amendments

- None.
