# 49: A tool call out of the answered reply

**Status**: done   **Depends on**: 16, 18

## Problem & context

The route answers the generated text as `choices[0].message.content`, and nothing reads that text
for a call. A model following the file's Qwen template writes a call as

```xml
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

and gets its call answered as prose, so a harness sees no call and the tool never runs. Tool
calling cannot work end to end until a call the model wrote as text arrives as a call in the reply
body.

The form is the one the file's own template asks for, the `<function=NAME>` and
`<parameter=KEY>VALUE</parameter>` instruction `tests/chat.test.tl:79` writes and `22/T-001` pins,
and it is the form `48` renders into a history. This spec reads that form. A block whose first
text after its leading spaces, tabs, newlines and carriage returns is not `<function=` is a form
this spec does not read, so a JSON block
`<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>` is answered as today's
plain body rather than as a server error. A reader per family comes with the plugin architecture;
this spec's reader is the core's because the reply text is built inside `serve` and never reaches a
plugin, so no boundary exists for a plugin to sit behind yet, the reason `23` and `28` give for the
store and the pool. The scan itself is not family-specific: the marker pair `<tool_call>` and
`</tool_call>` and that reader are held in `chat`, beside the template key
`tokenizer.chat_template`, because the form a file's template asks for and the call read out of it
are one fact; `serve` shapes the body from what the reader returns, so a later family-specific
reader can take another form without changing the scan.

A call block is a block that, after its leading spaces, tabs, newlines and carriage returns, begins
with `<function=`. After the generation has ended, and before the body is built, the reply text is
scanned for the marker pair. Each call block's function tag gives the name, and its parameter pairs
give the arguments: a value is trimmed of the spaces, tabs, newlines and carriage returns the
template writes around it, then read with the `json` package when that text decodes as JSON, so a
number, boolean, object or list keeps its type, and kept as a string when it does not. The `json`
package is the one `47` reads the body's tool list with, because an empty object and an empty array
must stay apart and `tltp`'s unpack gives both one table. A body holding calls carries, in order,
`tool_calls` entries with ids `call_1`, `call_2`, numbered by position, `type` `function`, the
call's name and its arguments encoded with `json.encode` as JSON text, an object with no members
staying `{}` and never `[]`, an empty `content` and `finish_reason` `tool_calls`. The reply text
outside the markers is dropped from a call reply, because the model's prose is not the answer the
call carries. A reply with no call block answers the body it answers today. A call block that does
not hold the shape `FR-002` reads answers `InternalServerError` with one line naming the reply, so a
broken call is never shipped as if it were valid.

The scan reads the reply after the generation has ended and changes nothing the generation did: no
id is stopped, steered or forced, a streamed reply still sends the text pieces it sends today, and
what a session keeps is the prompt and the sampled ids it keeps today.

This amends two requirements of `18`: `FR-003` for the answered body's shape and `FR-012` for the
refusal seam. Four non-goal bullets of shipped specs are touched when this spec ships, each
becoming the bullet the list below spells out:

- `18`: its "No reply member beyond `choices` and `usage`" bullet's list becomes "no `id`, `object`,
  `created` or `model`", and one choice only.
- `22`: its "No reading a call out of a reply" bullet becomes "No streaming a call out of a reply:
  streaming one is a later spec, so generation still ends only at an end id or at the budget, the
  way `16` and `19` ship it."
- `47`: its "No tool calls out of a reply" bullet becomes "No streamed tool calls: streaming a call
  from the reply text stays later."
- `48`: its "No tool call out of a reply" bullet becomes "No streamed tool call out of a reply:
  streaming those stays later, and `reasoning_content` stays unread."

The reply a call block produces, the `usage` object being the one the generation already answers:

```json
{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}],"usage":{...}}
```

## User stories

- US-001 (P1) As a harness author, I want a call the model wrote as text to arrive as `tool_calls`
  with a name and arguments, so that I run the tool instead of reading prose.
- US-002 (P1) As a harness author, I want a reply with no call to keep the plain reply body, so that
  ordinary answers are unchanged.
- US-003 (P2) As the person running the server, I want a broken call block to be refused with one
  line, so that a broken call is never answered as if it were valid.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL scan the text of the reply an answered request returns, once
  `51/FR-001` has cut that reply's head reasoning block out of it and in that order, for the
  call markers `<tool_call>` and `</tool_call>`, a block being the text between an opening marker
  and the next closing marker after it, in the order the blocks appear, and a call block being a
  block that, after its leading spaces, tabs, newlines and carriage returns, begins with
  `<function=`, so a call block standing inside a head reasoning block answers reasoning and not a
  call.
- FR-002 (US-001) WHEN a call block, apart from its leading and trailing spaces, tabs, newlines and
  carriage returns, holds a `<function=NAME>` tag, zero or more `<parameter=KEY>VALUE</parameter>`
  pairs and then a `</function>` tag, with only spaces, tabs, newlines and carriage returns between
  them, THE SYSTEM SHALL read one call from it, whose name is NAME and whose arguments is an object
  holding one member per pair whose key is KEY and whose value is the value VALUE decodes to with
  the `json` package when VALUE trimmed of spaces, tabs, newlines and carriage returns decodes as
  JSON, and that trimmed text as a string otherwise, NAME and KEY each being one or more bytes with
  no `>`.
- FR-003 (US-001) WHEN the text `FR-001` scans holds one or more calls and no call block is refused
  by `FR-005`, that text being only what `51`'s cut leaves, so a complete call block standing inside
  a head block that ran to the end of the reply answers no call and that reply answers `51/FR-003`,
  THE SYSTEM SHALL answer `OK` with a body whose `choices` holds one object whose `index` is 0,
  whose `finish_reason` is `tool_calls` and whose `message` is an object whose `role` is `assistant`,
  whose `content` is empty, and whose `tool_calls` is a list holding, in the order `FR-002` reads
  them, one entry per call whose `id` is `call_` then the call's position in that list counting from
  one, whose `type` is `function`, and whose `function` is an object whose `name` is the call's name
  and whose `arguments` is that call's arguments encoded with `json.encode` as JSON text, an object
  with no members being `{}` and never `[]`, the body's `usage` being the object `18/FR-004`
  describes.
- FR-004 (US-002) WHEN the text `FR-001` scans holds no call block, THE SYSTEM SHALL answer `OK`
  with the body it answers today, whose `choices` holds one object whose `index` is 0 whose
  `message` is an object whose `role` is `assistant` and whose `content` is that text unchanged, and
  which holds no `finish_reason` and no `tool_calls`. That reading and this body reach only the text
  `51`'s cut left, so a reply whose head holds an open thinking pair is answered by `51/FR-002` or
  `51/FR-003` and never by this body, whatever that reply's text holds.
- FR-005 (US-003) IF a call block does not hold the shape `FR-002` reads, THEN THE SYSTEM SHALL
  answer `InternalServerError` with a body whose single member is `error`, an object whose `type` is
  `internal_error` and whose `message` is `reply: block `, the first such block's position among the
  reply's blocks counting from one, and `: not a call`.

## Non-goals

- No meaning read out of a call beyond its name and arguments: only the function tag's name and the
  parameter pairs are read, `reasoning_content` is accepted and reaches no template name and no
  prompt, and a message field of the request is
  unchanged.
- No JSON block form: a block whose first text other than spaces, tabs, newlines and carriage
  returns is `{` is a form this spec does not read, answered as today's plain body, and a reader per
  family stays with the plugin architecture.

## Tests

- T-001 (FR-001, FR-002, FR-003): `a reply holding a call answers the call it names`, proves on
  three copies of `18`'s fixture, each with a request whose one user message content is `c` and
  whose `max_tokens` is 1: on the copy whose token zero is `CALL`, its three tokens being `CALL`,
  `b` and `c`, that the body decodes to `choices` holding one object whose `index` is 0, whose
  `finish_reason` is `tool_calls` and whose `message` is `{role = "assistant", content = "",
  tool_calls = {{id = "call_1", type = "function", function = {name = "get_weather", arguments =
  '{"city":"Paris"}'}}}}`, with `usage` of `{prompt_tokens = 1, completion_tokens = 1,
  total_tokens = 2, prompt_tokens_details = {cached_tokens = 0}}`; on the copy whose token zero is
  `TWO`, that the body answers four entries whose ids are `call_1`, `call_2`, `call_3` and
  `call_4` and whose `function` members are, in order, `{name = "get_weather", arguments =
  '{"city":"Paris"}'}`, `{name = "add", arguments = '{"a":1}'}`,
  `{name = "extra", arguments = '{"opts":{"unit":"c"}}'}` and
  `{name = "empty", arguments = '{"opts":{}}'}`; and on the copy whose token zero is
  `<tool_call><function=ping></function></tool_call>`, that the body answers one call named `ping`
  whose `function.arguments` is exactly `{}` and not `[]`. `CALL` is
  `Hi<tool_call>\196\138<function=get_weather>\196\138<parameter=city>\196\138Paris\196\138</parameter>\196\138</function>\196\138</tool_call>done`,
  whose decode puts a newline around each tag and value, so the trim and the prose before and after
  the markers are proven; `TWO` is
  `<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call><tool_call><function=add><parameter=a>1</parameter></function></tool_call><tool_call><function=extra><parameter=opts>{"unit":"c"}</parameter></function></tool_call><tool_call><function=empty><parameter=opts>{}</parameter></function></tool_call>`.
- T-002 (FR-004): `a reply with no complete call keeps the plain body`, proves on `18`'s fixture
  with the body `{"messages":[{"role":"user","content":"a"}]}` that the answer decodes to `choices`
  holding one object whose `index` is 0, whose `finish_reason` is absent and whose `message` is
  `{role = "assistant", content = "a" written 256 times}`, with no `tool_calls`, and that it still
  carries the `usage` of `{prompt_tokens = 1, completion_tokens = 256, total_tokens = 257,
  prompt_tokens_details = {cached_tokens = 0}}`; on a copy whose token zero is
  `<tool_call><function=get_weather><parameter=city>Paris</parameter></function>`, that a request
  whose one user message content is `c` and whose `max_tokens` is 1 answers a `message` whose
  `content` is that token's text, with no `tool_calls` and no `finish_reason`, an opening marker
  with no closing marker after it being no complete block; and on a copy whose token zero is
  `<tool_call>{"name":"get_weather","arguments":{}}</tool_call>`, that the same request answers a
  `message` whose `content` is that token's text, with no `tool_calls` and no `finish_reason`, a
  JSON block being a form this spec does not read.
- T-003 (FR-005): `a broken call block is refused`, proves on five copies of `18`'s fixture, each
  with a request whose one user message content is `c` and whose `max_tokens` is 1, that each
  answers `InternalServerError` with a body decoding to an object whose only member is `error`, an
  object whose `type` is `internal_error`: the copy whose token zero is
  `<tool_call><function=></function></tool_call>` answers the message `reply: block 1: not a call`;
  the copies whose token zero is
  `<tool_call><function=get_weather><parameter=city>Paris</parameter></tool_call>`,
  `<tool_call><function=get_weather><parameter=>Paris</parameter></function></tool_call>` and
  `<tool_call><function=get_weather>oops</function></tool_call>` each answer
  `reply: block 1: not a call`; and the copy whose token zero is
  `<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call><tool_call><function=></function></tool_call>`
  answers `reply: block 2: not a call`, the first block being read and the broken one named by its
  position.

## Open questions

- None.

## Amendments

- None.
