# 48: A history that carries tool calls

**Status**: done   **Depends on**: 18, 20, 22, 45, 47

## Problem & context

After a tool call a harness replays two turns: an assistant turn carrying `tool_calls` and empty
content, then a `tool` turn carrying `tool_call_id`, the function's `name` and the result. The route
reads only `role` and `content`, so the call list is dropped, the template's tool-call branch never
runs, and the model is shown a plain assistant turn followed by tool text with no call between them.
`chat.Message` holds no place for those fields, `chat.render` binds only `role` and `content`, and
the reader refuses the `safe` and `string` filters and cannot walk a map in a `for` tag, so even a
bound call list could not reach the branch the file's own template writes. This spec makes the
history the harness replays render the text the reference template writes for it.

A call's arguments are arbitrary JSON whose member order the template writes out, so the route reads
`tool_calls` with the `json` package, as `47` reads `tools`, builds one `ToolCall` from each wire
entry's `function` member, and holds that member's `arguments` as the decoded `json.Value`.
`chat.render` converts it with `22/FR-001`'s conversion, and the template reads it as a map. The
`function` wrapper is the wire's, the shape the reference's own message writer emits (host
`github.com`): one object per call holding a `function` member whose `name` is text and whose
`arguments` is those arguments written as JSON text, which the route decodes once, keeping that
text's member order, and never re-encodes.

`18/FR-008` accepts any string `role`, so a role the engine cannot render passes
the route today, and `20/FR-019` and `20/FR-012` between them refuse the `for`
over a map and the `safe` and `string` filters that the file's own template needs for a
call. Four shipped requirements are amended: the two just named,
`18/FR-008` for the four roles, and `47/FR-007`, whose message-check list now
holds this spec's four. Four non-goal bullets are touched when this spec ships: the
first goes whole, and the three after it lose only the clause named, the rest of the bullet
standing.

- `22`: the whole of its "No tool calls in a history" bullet, which also holds the `safe`,
  `string` and map-walking clause.
- `20`: the clause "and the `tojson`, `safe`, `string` filters" of
  its first non-goal bullet, the images and videos it also names staying a non-goal.
- `45`: the clause "`tool_calls`, `tool_call_id`, `name` and `reasoning_content` stay unread" of
  its No other message field bullet, the `reasoning_content` name staying unread there.
- `47`: the history half of its "No tool calls in a history or out of a reply" bullet,
  its reply half standing.

Two halves of the reader change and stay apart: the kinds a filter writes are settled by `FR-006`, and
the two it has no form for, a map and a list, are refused at the filter by `FR-007`, so `T-006` hands
it an argument map and a list read off that map, no unbound value being involved on either path. A
message carrying no `tool_calls` keeps the call branch closed rather than refusing, because that
member's absence is the undefined `20/FR-009` reads as false in a condition and `20/FR-010`
keeps a branch not taken from evaluating, which is what lets `T-001`'s user and tool turns
render at all. The refused-filter case in `tests/chat.test.tl`, the `safe` refusal `22/T-005`
moved from `17/T-003`, names `upper`, a filter this engine still refuses, and keeps
its shape unchanged: one line naming `tokenizer.chat_template` and the filter it names, with
`tojson` in neither, that named word becoming `upper` in every assertion of it. The
filter `20/T-004` refuses there is not named in that case's own text, so the move checks
which filter it holds. An amendment entry names a requirement, so `22/T-005` is settled here
rather than in `## Amendments`.

```teal
local record chat
  record ToolCall
    name: string
    arguments: json.Value
  end

  record Message
    role: string
    content: string
    tool_calls: {ToolCall} | nil
    tool_call_id: string | nil
    name: string | nil
  end
end
```

## User stories

- US-001 (P1) As a harness author, I want to replay an assistant turn with tool calls and a tool
  turn with their results, so that the second turn of an agent conversation reaches the model.
- US-002 (P1) As a harness author, I want each call's name and arguments written into the prompt,
  so that the model sees the values it asked for when it answers the call.
- US-003 (P2) As a harness author sending a message field this engine cannot use, I want 400
  naming the message and the field at fault, and the call position where the field is a call
  list, so that I fix that message instead of guessing.

## Functional requirements

- FR-001 (US-001) WHEN a message carries `tool_calls`, THE SYSTEM SHALL bind the template name
  `tool_calls` to a list holding, in the order given, one map per call whose members are `name`, the
  call's name as text, and `arguments`, the `22/FR-001` conversion of that call's decoded arguments
  value, the calls being read as `FR-008` reads them.
- FR-002 (US-001) WHEN a message carries `tool_call_id`, THE SYSTEM SHALL bind the template name
  `tool_call_id` to its text.
- FR-003 (US-001) WHEN a message carries `name`, THE SYSTEM SHALL bind the template name `name` to
  its text.
- FR-004 (US-002) WHEN a `for` tag walks a map, THE SYSTEM SHALL yield each member's key as text, in
  the order the map holds its members, rather than refuse the walk as `20/FR-019` does today.
- FR-005 (US-002) WHEN the `safe` filter is applied to a value, THE SYSTEM SHALL yield that value
  unchanged rather than refuse the filter as `20/FR-012` does today.
- FR-006 (US-002) WHEN the `string` filter is applied to a text, THE SYSTEM SHALL write that text
  unchanged, and WHEN it is applied to a number, a boolean, none or an undefined, THE SYSTEM SHALL write
  it as `22/FR-007` writes it, the refusal reaching no value of those kinds.
- FR-007 (US-002) IF the `string` filter is applied to a map or a list, THEN THE SYSTEM SHALL
  return nil and one line naming `tokenizer.chat_template` and `string`, that refusal
  coming at the filter rather than at the read of its operand.
- FR-008 (US-001) WHEN the route builds a request from a body whose message carries a `tool_calls`
  member `FR-011` accepts, THE SYSTEM SHALL read that member with the `json` package and give
  `chat.render` one call per wire entry, each holding the entry's `function.name` and that `function`
  member's `arguments` text decoded once as a `json.Value`, in that text's member order, so that the
  history `T-001` renders reaches the model from a body.
- FR-009 (US-003) IF a message's `role` is a string other than `system`, `user`, `assistant` or
  `tool`, THEN THE SYSTEM SHALL answer `BadRequest` with a body whose `error.message` names the
  message's position in `messages` and that `role` value, and generate no token.
  Within one message the checks answer in the order `role`, `content`, `tool_calls`,
  `tool_call_id`, `name`, and the entries answer in the order `messages` holds them, so an entry at
  fault for two of them, or two entries each at fault, answers the earlier one.
- FR-010 (US-003) IF a message's `role` is `tool` and it carries no `tool_call_id`, THEN THE
  SYSTEM SHALL answer `BadRequest` with a body whose `error.message` names the message's
  position in `messages` and `tool_call_id`, and generate no token.
- FR-011 (US-003) IF a message's `tool_calls` member is not a JSON array, or holds an entry that is
  not a JSON object, or an entry whose `function` member is not a JSON object, or an entry whose
  `function.name` is not a string of one or more bytes, or an entry whose `function.arguments` is
  not a string whose text decodes as JSON, THEN THE SYSTEM SHALL answer `BadRequest` with a body
  whose `error.message` names `messages`, the message's position, `tool_calls` and, for an entry at
  fault, that entry's position counting from one, and generate no token.
- FR-012 (US-003) IF a message carries a `tool_call_id` or a `name` that is present
  and is not a string of one or more bytes, THEN THE SYSTEM SHALL answer `BadRequest` with a body
  whose `error.message` names the message's position in `messages` and that member's name, and
  generate no token.

## Non-goals

- No streamed tool call out of a reply: `reasoning_content` is accepted and reaches no template name
  and no prompt.
- No meaning read out of a call beyond its name and its arguments text: a call's `id`,
  `type` and other members pass unread, and an `arguments` member of another kind is refused rather
  than converted, which is `FR-011`'s own words and not a further check.
- No change to a plain text history: a messages list without these fields renders the prompt it
  renders today.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-005, FR-006): `a tool call and its result render the turn the
  template writes`, proves on the file's own template as `20/T-001` embeds it, over the one user
  message `{role = "user", content = "hi"}`, the assistant turn `{role = "assistant", content = "",
  tool_calls = {{name = "get_weather", arguments = json.decode(CALL)}}}` and the tool turn
  `{role = "tool", content = "sunny", tool_call_id = "call_1"}`, where `CALL` holds the JSON text
  `{"city": "Paris", "days": 2, "extra": {"unit": "c"}}`, that `render` given no tool list, no
  tool-choice input and no thinking input, so `enable_thinking` reaches the template unbound, returns
  exactly the text WANT below. The assistant turn follows the last query, so it opens with the
  thinking block `20/T-001`'s one-turn prompt ends with, whose two blank lines are the
  two newlines the closing literal carries after it, the empty reasoning itself adding none;
  `CALL` is that text as written, not re-encoded from WANT's own argument lines,
  since a value decoded and written back out need not keep the input's spacing;
  `days` is pinned as `2` because the decoder hands that digit back as a Lua number whose
  `tostring` form is `2`, the same fact `22/T-003` pins when it writes `1.0` as `1`; each parameter
  is one tag pair, `CALL`'s members in that order, `days` written through `string` as Lua's
  `tostring` form and `extra` through `tojson` and the `safe` filter unchanged; the lone tool result
  carries the user opener and the closer the template gives it; and WANT ends with that same
  generation opener:

  ```teal
  local CALL = '{"city": "Paris", "days": 2, "extra": {"unit": "c"}}'
  local WANT = table.concat({
    "<|im_start|>user\n",
    "hi<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n",
    "\n",
    "</think>\n\n",
    "<tool_call>\n",
    "<function=get_weather>\n",
    "<parameter=city>\n",
    "Paris\n",
    "</parameter>\n",
    "<parameter=days>\n",
    "2\n",
    "</parameter>\n",
    "<parameter=extra>\n",
    "{\"unit\": \"c\"}\n",
    "</parameter>\n",
    "</function>\n",
    "</tool_call><|im_end|>\n",
    "<|im_start|>user\n",
    "<tool_response>\n",
    "sunny\n",
    "</tool_response><|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n"
  }, "")
  ```

  and that the template
  `{% for message in messages %}{% if message.tool_call_id is defined %}{{ message.tool_call_id }}{% endif %}{% if message.name is defined %}{{ message.name }}{% endif %}{% endfor %}` over the same three
  messages, the tool turn carrying `get_weather` as its `name`, writes
  `call_1get_weather`, both message fields reaching the template.
- T-002 (FR-001, FR-004): `each call writes its name and every argument name`, proves on the
  template `{% for message in messages %}{% for entry in message.tool_calls %}{{ entry.name }}{% for key in entry.arguments %}{{ key }},{% endfor %}{% endfor %}{% endfor %}` over the one assistant
  message whose `tool_calls` holds the name `lookup` and the arguments value `json.decode` gives for
  `{"unit":"c","days":2}`, writes `lookup` from the call name then `unit,days,` from
  the two keys, in the order that value holds its members, one pair of name and comma per key.
- T-003 (FR-008): `a body carrying a tool call reaches the model`, proves on `18`'s
  fixture, whose `tokenizer.chat_template` is replaced by
  `{% for message in messages %}{% if message.tool_calls %}b{% else %}a{% endif %}{% endfor %}`, that the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"x","arguments":"{}"}}]}]}` answers `OK` with `usage` whose
  `prompt_tokens` is 2, the `b` proving that accepted list reached the template, and that the body
  holding that one user message alone, no `tool_calls` member anywhere in it, answers 1; and that a
  body naming `messages` twice, the first array's call `ab` and the second's `c`, answers `OK` with
  `prompt_tokens` 1 on a template writing each bound call name, so the call that reaches the model
  is the one in the array the shape checks read, the same member every other read of the body takes.
- T-004 (FR-010, FR-011, FR-012): `a message field the engine cannot use is a bad request`, proves on
  `18`'s fixture that each of these bodies answers `BadRequest` with a body decoding to an object
  whose only member is `error`, of type `invalid_request_error`:
  `{"messages":[{"role":"tool","content":"a"}]}` with a message naming `messages`, `1` and
  `tool_call_id`; `{"messages":[{"role":"tool","content":"a","tool_call_id":""}]}` and
  `{"messages":[{"role":"tool","content":"a","tool_call_id":5}]}`, each naming `messages`, `1` and
  `tool_call_id`; the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":5}]}`
  naming `messages`, `2` and `tool_calls`, a member that is not an array; the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[5]}]}`
  naming `messages`, `2`, `tool_calls` and `1`, an entry that is not an object; the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":5}]}]}`
  naming the same four, an entry whose `function` is not an object; the body
  `{"messages":[{"role":"user","content":"a","tool_call_id":5},{"role":"assistant","content":"","tool_calls":[{"type":"function"}]}]}`
  naming `messages`, `1` and `tool_call_id` and no `tool_calls`, the earlier entry's fault answering
  as `FR-009`'s order gives; the bodies
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"type":"function"}]}]}`
  and
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather"}}]}]}`
  and
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":"not json"}}]}]}`,
  each naming `messages`, `2`, `tool_calls` and `1`;
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","name":5}]}` names
  `messages`, `2` and `name`, and the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_call_id":5,"name":5}]}`,
  naming `messages`, `2` and `tool_call_id` rather than
  `name`, the order `FR-009` gives for those two members.
- T-005 (FR-009): `a role the engine does not know is a bad request`, proves on `18`'s fixture that
  `{"messages":[{"role":"wizard","content":"a"}]}` answers `BadRequest` with a body decoding to an
  object whose only member is `error`, of type `invalid_request_error`, whose message names
  `messages`, `1` and `wizard`, and that `{"messages":[{"role":"wizard","content":"a"}],"tools":5}`
  answers the same `wizard` message and names no `tools`, the message check answering before
  `47/FR-006`'s.
- T-006 (FR-007): `the string filter refuses a map and a list`, proves on the
  template
  `{% for message in messages %}{% for entry in message.tool_calls %}{{ entry.arguments | string }}{% endfor %}{% endfor %}` over the one assistant message of
  `T-002`, that `render` returns nil and one line naming `tokenizer.chat_template` and
  `string`, the argument map reaching no text form; and on the template
  `{% for message in messages %}{% for entry in message.tool_calls %}{{ entry.arguments.k | string }}{% endfor %}{% endfor %}`, over the one
  assistant message whose `tool_calls` holds the name `lookup` and the arguments
  value `json.decode` gives for `{"k":["a"]}`, that the same
  nil and the same line answer, both refusals coming from the filter at the render seam and no
  unbound read involved on either path.
- T-007 (FR-006): `the string filter writes every kind the reader holds`, proves on
  the template
  `{% for message in messages %}{% for entry in message.tool_calls %}{{ entry.arguments.n | string }}{{ entry.arguments.b | string }}{{ entry.arguments.f | string }}{{ entry.arguments.z | string }}{{ entry.arguments.absent | string }}{% endfor %}{% endfor %}` over the one assistant message whose
  `tool_calls` holds the name `x` and the arguments value `json.decode` gives
  for `{"n": 1.0, "b": true, "f": false, "z": null}`, writes `1truefalsenullnull`: the number that decoder
  hands back for `n` in Lua's `tostring` form, the same fact `22/T-003` pins when
  it writes `1.0` as `1`, the two booleans as `true` and `false`, the
  JSON null as `null`, and the member that map does not hold as `null` too, no
  value of those five kinds reaching the refusal `FR-007` states.

## Open questions

- None.

## Amendments

- None.
