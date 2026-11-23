# 17: Turn a messages list into the model's prompt text

**Status**: done   **Depends on**: 09

## Problem & context

Nothing in the project builds a prompt. `16` takes prompt text and `10` turns text into ids, so a
caller holding a conversation must spell the model's turn markers by hand, and every model file
states them its own way: `tokenizer.chat_template` holds a template whose output is the exact
string the model was trained on. The chat route of the next spec needs one call that turns a
messages list into that string.

This spec adds one module, `tllm.chat`, with one operation: `render(tokenizer, messages)` runs the
template `tokenizer.load` kept over the messages list and returns the text it writes. The module
carries the reader itself and adds no dependency. A library was the other way and none fits: lupa,
the Jinja2 implementation written in Lua, ships no rockspec, the only LuaRocks package of that
name is an unrelated Lua REPL, its repository was archived in January 2025, and it would arrive as
a vendored Lua file needing the `lpeg` C library as well. lupa's documented behaviour is the
reference for the grammar and the dialect, no code is copied from it, and only what a chat
template needs is built here. The prompt is the only text `generate.text` takes and the chat
route of the next spec needs it, so there is no boundary a plugin could sit behind yet.

The dialect keeps Jinja's tag syntax, its `-` trim markers and its `{# ... #}` comments, and runs
expressions written in Python syntax. The tags run are `{{ }}`, `{% if %}`, `{% elif %}`,
`{% else %}`, `{% for ... in ... %}`, `{% set %}` and `{% macro %}`, and the filters run are `trim`,
`length` and `tojson`; a tag outside that set, a filter outside it, and a method outside the seven
that spec 20 names are refused as `FR-007` states. A template is rendered with `messages`,
`add_generation_prompt`, `tools` when a tool list is given, `enable_thinking` when a thinking value
is given, `loop`, the names `{% set %}` binds, macro names, `namespace(...)`, `raise_exception` and
the seven string methods, which `FR-011` states and a test asserts.

`tokenizer.load` keeps `tokenizer.chat_template` as read and refuses neither its absence nor its
kind, in the shape `10/FR-012` fixed for `pre` and `merges`, so a file is never refused for a key
only chat reads. The tokenizer record gains `template: string | nil`:

```teal
local record tokenizer
  record Tokenizer
    size: integer
    end_id: integer | nil
    template: string | nil
    decode: function(Tokenizer, {integer}): string | nil, string | nil
    encode: function(Tokenizer, string): {integer} | nil, string | nil
  end

  load: function(path: string): Tokenizer | nil, string | nil
end

local record chat
  record Message
    role: string
    content: string
  end

  render: function(
    tokenizer.Tokenizer, {Message}
  ): string | nil, string | nil
end
```

`render` takes messages the caller has already shaped. `FR-009`'s checks exist because a caller
holding decoded JSON hands over tables the type system never saw, so they run at call time, and a
test reaches them with a cast Teal accepts, `({ role = "user", content = 5 }) as chat.Message`.

The Qwen3.6 file on this machine states a Python-dialect Jinja template with `messages[0]`,
`'image' in item`, `namespace()`, macros and `|tojson`; spec 20 made the reader run the constructs
its live path needs and read the rest without running it, so that file's template now renders. The
tests here use hand-made fixtures, because a 23 GB file cannot be a test input.

## User stories

- US-001 (P1) As the engine author, I want the model's own template turned into prompt text from a messages list, so that a prompt starts from the conversation shape the model was trained on.
- US-002 (P1) As the engine author, I want the prompt to end where the assistant's reply begins, so that a generate call can start from it unchanged.
- US-003 (P1) As a person with a file or a messages list chat cannot use, I want one line naming what is at fault, so that I can tell a bad model from a bad request.

## Functional requirements

- FR-001 (US-001) WHEN `render` is called with a tokenizer whose `template` is a string and one or
  more messages whose `role` and `content` are strings, THE SYSTEM SHALL return the text that
  template writes with `messages` bound to that list in order, a nil output writing nothing and any
  other output writing its Lua text, and nil as the second value.
- FR-002 (US-002) WHEN a template is rendered, THE SYSTEM SHALL bind `add_generation_prompt` to
  true, so that an `{% if add_generation_prompt %}` block writes its assistant opener.
- FR-003 (US-001) WHEN a template holds `{% if %}`, `{% elif %}` and `{% else %}`, THE SYSTEM SHALL
  write the body of the first condition that is neither nil nor false, and the `{% else %}` body
  when no condition is. The body may hold plain text, outputs and further tags.
- FR-004 (US-001) WHEN a template holds `{% for name in expression %}`, THE SYSTEM SHALL write its
  body once for each item of the list that expression gives, in order, with `name` bound to the
  item. The body may hold plain text, outputs and further tags.
- FR-005 (US-001) WHEN a tag or an output carries a `-` trim marker on a side, THE SYSTEM SHALL
  write none of the whitespace on that side, whitespace being a space, a tab, a newline or a
  carriage return.
- FR-006 (US-001) WHEN an output expression ends with `|trim`, THE SYSTEM SHALL write its value
  with the whitespace at both ends removed.
- FR-007 (US-003) IF the template holds a tag the reader does not run, a filter other than `trim`,
  `length` or `tojson`, a method other than the seven spec 20 names, a tag whose expression is
  missing or will not parse, a closing tag with no block open, a block that is never closed, or an
  expression that raises while it is evaluated, THEN THE SYSTEM SHALL return nil and one line naming
  `tokenizer.chat_template` and, for a tag or a filter, its name as written in the template. A
  construct that sits in a branch which is not taken, or in a macro that is never called, is read
  and never refused.
- FR-008 (US-003) IF the tokenizer's `template` is nil, THEN THE SYSTEM SHALL return nil and one
  line naming `tokenizer.chat_template`, this check running before the messages are read.
- FR-009 (US-003) IF the messages list holds no entry, or an entry is not a table whose `role` and
  `content` are strings, THEN THE SYSTEM SHALL return nil and one line naming `messages` and, for
  an entry, its position in the list as a number.
- FR-010 (US-001) WHEN `tokenizer.load` reads a file, THE SYSTEM SHALL set its `template` field to
  the text of the file's `tokenizer.chat_template` entry when that entry is a string and to nil
  otherwise, refusing neither its absence nor its kind, so that `FR-008` is answered by `render`
  alone.
- FR-012 (US-001) THE SYSTEM SHALL read and parse a template's text once and keep the tree under
  that text, so that a request renders the same template without lexing or parsing it again, and
  SHALL keep its refusal under that text too when the template cannot be read.
- FR-011 (US-003) WHEN an expression is evaluated, THE SYSTEM SHALL make reachable only `messages`,
  `add_generation_prompt`, `tools` from the tool list `22/FR-001` binds, `enable_thinking` from the
  thinking input `46/FR-007` binds, the name a `{% for %}` in scope binds, `loop`, the names
  `{% set %}` binds, macro names, `namespace(...)`, `raise_exception` and the seven string methods
  spec 20 names, so that a template naming anything else reaches nil and opens no file and starts
  no process. A call that gives no thinking value leaves it unbound as `46/FR-008` states, so the
  nil-reach this requirement proves still holds for such a call.

## Non-goals

- No route, command, config key or streaming: the chat completions route and its call to
  `generate.text` are spec 18, and this call returns one finished string.
- No tag or filter beyond the ones above: includes, named blocks, `import`, and every filter but
  `trim`, `length` and `tojson` are refused as `FR-007` states, and `tokenizer.chat_template.<name>`
  is not read.
- Tools, tool calls, images and videos, and the `safe` and `string` filters: spec 20 only reads
  them where a branch is never taken, `tojson` being a filter spec 22 runs, and the request inputs
  `enable_thinking`, `preserve_thinking` and `reasoning_content` are a follow-up.

## Tests

- T-001 (FR-001, FR-002, FR-004, FR-010): `messages become the prompt the template writes`, proves
  on a fixture whose metadata is a `gpt2` vocabulary `09/FR-001` to `09/FR-004` accept and
  `10/FR-012` leaves alone (`tokenizer.ggml.model` `gpt2`, `pre` `gpt2`, `tokens` the two pieces
  `a` and `b` with `token_type` of two 1s, `merges` empty) and whose `tokenizer.chat_template`
  writes, inside a `{% for message in messages %}` block, `<|im_start|>`, the role, a newline, the
  content, `<|im_end|>` and a newline, and then writes `<|im_start|>assistant` and a newline
  inside an `{% if add_generation_prompt %}` block, that `tokenizer.load` returns a tokenizer and
  nil whose `template` is that text, and that `render` of the two messages
  `{role = "system", content = "be brief"}` and `{role = "user", content = "hi"}` returns
  `<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n`
  and nil.
- T-002 (FR-001, FR-003, FR-005, FR-006): `branches, trim markers and the trim filter shape the
  output`, proves on a copy of T-001's fixture whose `tokenizer.chat_template` is
  `a {%- if add_generation_prompt -%} b {%- endif -%} c` that `render` of one valid message returns
  `abc` and nil, on a second copy whose template is
  `{% for message in messages %}{% if message.role == 'user' %}U{% elif message.role == 'assistant' %}A{% else %}S{% endif %}{{ message.content|trim }}{{ message.missing }}{% endfor %}`
  that `render` of `{role = "user", content = " hi "}` and `{role = "tool", content = "x"}`
  returns `UhiSx` and nil, the message's absent `missing` field writing nothing, and on a third
  copy whose template is `a\n{{- 5 -}}\nc` that `render` of one valid message returns `a5c` and nil,
  the trim markers on the output writing none of the whitespace either side, a newline included,
  and the number writing itself.
- T-003 (FR-007, FR-008, FR-009, FR-010, FR-011): `a template or messages that cannot be used are
  refused`, proves that on T-001's fixture with `tokenizer.chat_template` removed and an empty
  messages list, `render` returns nil and a line naming `tokenizer.chat_template`; that on a copy
  of it whose `tokenizer.chat_template` is an integer, the file loads and `render` of one valid
  message returns nil and a line naming that key; that on a copy whose template is
  `{% macro m() %}{% endmacro %}` `render` returns the empty string and nil, the macro being
  defined and never called; that
  on a copy whose template is `{% for message in messages %}` `render` returns nil and a line
  naming that key; that on a copy whose template is `{% if %}` `render` returns nil and a line
  naming that key; that on a copy whose template is `{% endif %}` `render` returns nil and a line
  naming that key; that on a copy whose template is `{{ 'x'|safe }}` `render` returns nil and a
  line naming that key and `safe`; that on a copy whose template is `{{ missing.field }}`
  `render` returns nil and a line naming that key; that on a copy whose template is
  `{{ io.open('x') }}` `render` of one valid message returns nil and a line naming that key, `io`
  reaching nothing; that on T-001's fixture `render` of the empty list returns nil and a line
  naming `messages`; and that `render` of one valid message and
  `({role = "user", content = 5}) as chat.Message` returns nil and a line naming `messages` and
  `2`; and that `render` of one valid message, a `nil` hole and one valid message returns nil and
  a line naming `messages` and `2`.

## Open questions

- None.

## Amendments

- None.
