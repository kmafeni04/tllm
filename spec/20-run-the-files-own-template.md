# 20: Run the file's own chat template

**Status**: done   **Depends on**: 17, 09

## Problem & context

`17-chat-template.md` gave `src/chat.tl` a template reader for one narrow dialect: Lua expressions,
`{% if %}`/`{% for %}`, trim markers and a `trim` filter. The local model's file carries a real
Jinja2 template, written in Python syntax, so `serve` answers every chat request with the refusal
line `tokenizer.chat_template: …` (`17/FR-007`) instead of a reply.

Closing that gap means the reader has to run the constructs that template actually uses, in Jinja's
own semantics, and nothing more. The template is on disk as `tokenizer.chat_template`, so the reader
is judged by whether it produces that file's trained-on prompt, not by how much of Jinja it knows.

The template's live path for a conversation whose messages are plain strings is small. Every other
branch (tools, images, videos, `tojson`) sits in a body that is never entered for such a
conversation, so it has to be read but never run. That is what keeps this one spec. The reader
stays in the core because the prompt has to exist before `generate` is reached, and a plugin has
no seat at that point.

## User stories

- US-001 (P1) As someone running `serve` against the local model, I want a chat request to come back
  as an answer, so that the engine is usable with the file I actually have.
- US-002 (P1) As a template author, I want the reader to compute the same prompt Jinja2 computes, so
  that the model sees the text it was trained on.
- US-003 (P2) As a template author, I want a broken template or unusable messages reported as one
  line naming the key at fault, so that I can find what to fix.

## Functional requirements

- FR-001 (US-001) WHEN `render` is called with a tokenizer whose `template` is the file's own
  `tokenizer.chat_template` and with messages whose `role` and `content` are strings, THE SYSTEM
  SHALL return the prompt that template writes for them as its first value and nil as its second,
  with `messages` bound to the list in order and `add_generation_prompt` bound to true as
  `17/FR-002` states, and with `add_vision_id`, `preserve_thinking` and `reasoning_content` left
  unbound. `tools` joins those unbound names only for a call that gives no tool list, `22/FR-002`
  states; a call that gives one binds it as `22/FR-001` states. `enable_thinking` leaves this list:
  a call that gives a thinking value binds it as `46/FR-007` states, and the unbound case these
  tests record is the call that gives none, `46/FR-008` states. Its tests' endpoints set `kv_cache`
  `f32`, so the prompt and reply they record stand.
- FR-002 (US-002) WHEN the template is read, THE SYSTEM SHALL accept Python-syntax expressions:
  `and`, `or`, `not`, `in`, `not in`, `is`, `is not`, `~`, `+`, `-`, `*`, `==`, `!=`, `<`, `<=`,
  `>`, `>=`, parentheses, string, number, `true`/`false`/`none` and list literals, attribute
  access, index and slice (`[0]`, `[-1]`, `[::-1]`), and method calls.
- FR-003 (US-002) WHEN the template holds a `{% set %}` tag, THE SYSTEM SHALL bind its name for the
  rest of the block that encloses it, where `{% if %}`, `{% elif %}` and `{% else %}` introduce no
  block of their own: a binding made inside a taken `{% if %}` is readable after `{% endif %}`, and
  a binding made in one iteration of a `{% for %}` is readable by later iterations of that loop but
  not after `{% endfor %}`.
- FR-004 (US-002) WHEN a `{% set %}` assigns an attribute of a `namespace(...)` value, THE SYSTEM
  SHALL update that namespace member, and a read after the enclosing `{% for %}` ends SHALL see the
  update; `namespace(...)` SHALL accept any number of `name=expression` initializers, evaluating
  each expression, and reading a member that was not given SHALL give the same undefined an unbound
  name gives.
- FR-005 (US-002) WHEN the template holds `{% macro name(a, b, c=false) %}`, THE SYSTEM SHALL call
  it from later expressions with fewer arguments than it declares, binding each missing parameter
  to its default, and with arguments given by name as well as by position, and the call's value
  SHALL be the text its body writes, written nowhere else, with a filter such as `|trim` applied to
  that text and the macro's own bindings reaching nothing outside the call.
- FR-006 (US-002) WHEN the template iterates `{% for %}` over a list, a string or a reversed slice,
  THE SYSTEM SHALL provide `loop.index0`, `loop.index`, `loop.first`, `loop.last`, `loop.previtem`
  and `loop.nextitem` in its body, with `loop.previtem` and `loop.nextitem` undefined outside the
  iterations where a neighbour exists, so writing them there writes the empty string as FR-009
  states.
- FR-007 (US-002) WHEN an output expression or a tag applies `|trim` or `|length`, THE SYSTEM SHALL
  apply that filter to the value before it is written or compared, `|length` on a string counting
  characters rather than bytes.
- FR-008 (US-002) WHEN the template holds a `{# … #}` comment, THE SYSTEM SHALL write nothing for
  it, and a trim marker beside it SHALL trim the neighbouring text as `17/FR-005` states.
- FR-009 (US-002) WHEN an expression reads a name that is not bound, or a field a message does not
  carry, THE SYSTEM SHALL treat it as undefined: writing it writes the empty string, using it as a
  condition is false, comparing it to a string, number or another undefined is false, and the
  `is defined`, `is undefined`, `is none`, `is string`, `is iterable`, `is mapping`, `is sequence`,
  `is true` and `is false` tests answer about it.
- FR-010 (US-002) WHEN a tag or output is inside a branch that is not taken, THE SYSTEM SHALL write
  and evaluate nothing in that branch, so that a name reached only there may stay unbound.
- FR-011 (US-003) IF the template reaches `raise_exception` with a text, THEN THE SYSTEM SHALL
  return nil and the one line `tokenizer.chat_template: <text>` with newlines collapsed.
- FR-012 (US-003) IF the reader evaluates a tag, a filter other than `trim`, `length` and `tojson`
  (`tojson` being a filter `22/FR-008` runs, while `safe` and `string` stay refused), or an
  expression it cannot evaluate, THEN THE SYSTEM SHALL return nil and one line naming
  `tokenizer.chat_template`; a construct that sits in a branch FR-010 keeps from running, or in a
  macro that is never called, is read and never refused.
- FR-013 (US-003) IF an expression reads an attribute of, indexes, or calls a method on a value that
  is undefined, THEN THE SYSTEM SHALL return nil and the one line FR-012 describes, so that
  `{{ missing.field }}` and `{{ io.open('x') }}` are both refused and a name that is only written is
  not.
- FR-014 (US-002) WHEN a condition is evaluated, THE SYSTEM SHALL treat `false`, `none`, a zero
  number, a string of length zero, a list of length zero and a table with no keys as false and
  every other value as true, as Python does.
- FR-015 (US-002) WHEN an expression calls a method, THE SYSTEM SHALL make `strip`, `lstrip`,
  `rstrip`, `split`, `startswith`, `endswith` and `replace` reachable with no argument, `split`,
  `startswith`, `endswith`, `strip`, `lstrip` and `rstrip` with one string argument and `replace`
  with two, where a separator is matched literally and never as a pattern, where `strip`, `lstrip`
  and `rstrip` with one argument remove every leading or trailing character occurring in that
  argument, where `split` with no argument splits on runs of whitespace and drops empty pieces,
  where `split` with one argument divides at every occurrence of the separator, so `s.split(C)[0]`
  is the text before its first occurrence and `s.split(C)[-1]` the text after its last, and the text
  between two occurrences is reached by neither; every other method name or argument count is
  refused as FR-012 states.
- FR-016 (US-003) IF the template reaches `raise_exception` with no argument or with a value that is
  not a string, THEN THE SYSTEM SHALL return nil and the one line FR-012 describes.
- FR-017 (US-002) WHEN a value is written, THE SYSTEM SHALL use its text form: an undefined and
  `none` write the empty string, a boolean writes `true` or `false`, a number writes Lua's `tostring`
  form, and `~` joins those text forms, so `2 ~ '. '` writes `2. `.
- FR-018 (US-003) IF `+`, `-` or `*` is applied to a pair of values it cannot operate on, THEN THE
  SYSTEM SHALL return nil and the one line FR-012 describes, so `'a' + 5` is refused rather than
  concatenated.
- FR-019 (US-003) IF a `{% for %}` iterates a value that is not a list, a string or a reversed
  slice, THEN THE SYSTEM SHALL return nil and the one line FR-012 describes, so a `{% for %}` over
  an unbound name is refused rather than writing nothing.
- FR-020 (US-002) WHEN an expression uses `in` or `not in`, THE SYSTEM SHALL answer about the right
  operand: whether a value is one of a list's items, whether a string occurs inside a string, and
  whether a key is present in a table, and SHALL refuse an operand it cannot test, such as an
  undefined, as FR-012 states.
- FR-021 (US-002) WHEN an expression applies an `is` test to a bound value, THE SYSTEM SHALL answer:
  a string is `string`, `iterable` and `sequence` but not `mapping`; a list is `iterable` and
  `sequence` but not `string` or `mapping`; a table is `mapping` and `iterable`; a number is none of
  those; `true` answers `true` and `false` answers `false`; `none` answers `none`; and a bound value
  answers `defined` and not `undefined`.

## Non-goals

- Tools, tool calls, images and videos, and the `tojson`, `safe`, `string` filters: this spec only
  has to read them without running them, so a conversation that needs them is a follow-up spec.
- `preserve_thinking` and `reasoning_content` as request inputs: the template's default path, with
  the names FR-001 leaves unbound, is what this spec produces.
- No stop rule for the closer that ends a thinking block: generation is cut only at the tokenizer's
  end token, as 16 ships it, so trimming that closer out of a reply is a follow-up spec.

## Tests

- T-001 (FR-001, FR-002, FR-010): `the file's own template renders a text conversation`, proves the
  file's template, copied from `tokenizer.chat_template` of
  `~/.local/share/llm-models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` and embedded as the
  fixture string, renders three message lists into these exact prompts. A system
  `be brief` and a user `hi` give
  `<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n`.
  Two system turns `be brief` then `and plain`, then a user `hi`, give
  `<|im_start|>system\nbe brief\nand plain<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n`.
  A user `hi` then an assistant whose content is the two pieces `plan` and `answer` with the
  closer between them, written `plan</think>answer`, give
  `<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nplan\n</think>\n\nanswer<|im_end|>\n<|im_start|>assistant\n<think>\n`.
  It also proves the never-run branches were read without being run, which shows in the
  template parsing at all, and in the rendered text holding no tool or image marker.
- T-002 (FR-003): `a set in a taken if outlives it and a set in a loop does not`, proves a name bound
  inside an `{% if %}` is readable after `{% endif %}`, and that a `{% set x = 0 %}` re-bound inside
  a loop still writes `0` after `{% endfor %}`.
- T-003 (FR-004): `namespace members survive the loop`, proves a member assigned inside a
  `{% for %}` is read after `{% endfor %}`, that `{% set ns = namespace(n=messages|length - 1) %}`
  then `{{ ns.n }}` writes the last index, and that `{{ ns.absent }}` writes nothing.
- T-004 (FR-011, FR-012, FR-013, FR-016, FR-018, FR-019, FR-020): `a template that cannot be run is
  one line`, proves `raise_exception` with a text, with no argument, and with a number, an unsupported
  tag such as `{% include %}` and an unsupported filter each give nil and one line starting
  `tokenizer.chat_template: `, and that `{{ missing.field }}` and `{{ io.open('x') }}` are refused
  the same way while `{{ missing }}` alone writes nothing, and that `{{ 'a' + 5 }}` and a
  `{% for x in missing %}` loop and `{{ 'x' in missing }}` are refused the same way.
- T-005 (FR-002, FR-009, FR-014, FR-015, FR-017, FR-020): `values, methods and operators follow
  Python`, proves `{% if '' %}` and `{% if 0 %}` take the else branch, that an empty list takes the
  else branch while a list with one item takes the if branch, that
  `{{ ' a '.strip() }}{{ 'a-b'.split('-')[1] }}`
  `{{ 'abc'.startswith('ab') }}{{ 'abc'.replace('b', 'x') }}` writes `abtrueaxc`, that
  another method name is refused, that `{{ 'abc'[0] }}{{ 'abc'[-1] }}{{ 'abc'[::-1] }}` writes
  `accba`, that `2 ~ '. '` writes `2. `, that over two messages with contents `a` and `b`,
  `{% for x in messages[::-1] %}{{ x.content }}{% endfor %}` writes `ba`, and that
  `{% if 'b' in ['a', 'b'] %}1{% endif %}{% if 'b' in 'abc' %}2{% endif %}`
  `{% if 'b' not in 'ac' %}3{% endif %}` writes `123`, and that over a message carrying only `role`
  and `content`, `{{ messages[0].reasoning_content }}` writes nothing, and that `{{ 'p1Cp2Cp3'.split('C')[0] }}`
  `{{ 'p1Cp2Cp3'.split('C')[-1] }}` writes `p1p3`, the middle piece reached by neither.
- T-006 (FR-006): `loop members follow Jinja`, proves that writing
  `{{ loop.index0 }}|{{ loop.index }}|{{ loop.first }}|{{ loop.last }}|{{ loop.previtem }}|{{ loop.nextitem }}`
  inside `{% for x in ['a', 'b'] %}` writes `0|1|true|false||b1|2|false|true|a|`, the empty slots
  being the undefined neighbours, and that the same loop over `'ab'` writes the same text, so a
  string iterates per character.
- T-007 (FR-007, FR-008): `filters and comments shape the output`, proves `{{ 'abc'|length }}`
  writes `3`, `a {#- x -#} b` writes `ab`, and `a {# x #} b` writes `a  b`.
- T-008 (FR-009, FR-021): `an undefined answers the tests`, proves a template writing the nine `is`
  tests on an unbound name, in the order FR-009 lists them and joined with `|`, writes
  `false|true|false|false|false|false|false|false|false`, that the same nine tests on the bound
  string `'x'` write `true|false|false|true|true|false|true|false|false`, and that
  `{{ '' is string }}` writes `true`.
- T-009 (FR-005): `a macro fills defaults, named arguments and its own value`, proves
  `{% macro m(a, b, c='z') %}{{ a }}{{ b }}{{ c }}{% endmacro %}` called as `{{ m('1', '2') }}`
  writes `12z` and as `{{ m('1', c='3', b='2') }}` writes `123`, and that
  `{% macro m(a) %}{{ a }}{% endmacro %}{{ m(' x ')|trim }}` writes `x` with nothing written at the
  call.

## Open questions

- None.

## Amendments

- None.
