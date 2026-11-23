# 22: Show the model the functions it may call

**Status**: done   **Depends on**: 17, 20

## Problem & context

The model calls a function only when the prompt shows it the functions it may call. The file's own
template already writes that block, guarded by `{% if tools and tools is iterable and tools is not
mapping %}` and writing each tool as `{{ tool | tojson }}`. `20` left both halves out on purpose:
`chat.render` binds only `messages` and `add_generation_prompt`, so `tools` reaches the template
unbound and the guard is never taken, and the reader runs no filter but `trim` and `length`, so that
branch could not be executed even by a caller that could bind it. A prompt can therefore never carry
a function definition, no conversation with tools is possible, and the two halves of tool calling
that come after this, putting a tool list into the request and reading a call back out of the reply,
have nothing to stand on.

Tool calling is three problems, not one. This is the smallest piece that stands on its own: a tool
list handed to `chat.render` comes back as the prompt the file's own template writes for it. The
request side is a later spec and the answer side is another.

The reference for behaviour is llama.cpp's own Jinja engine, read in the local checkout at
`code/clones/llama.cpp` (host `github.com`): its `tojson` writes `, ` between items and `: ` after a
key unless the template asks otherwise, leaves a non-ASCII byte alone, keeps an object's members in
the order they were given, writes `null` for none and for an undefined, and runs `safe` and `string`
through that same writer. FR-004 to FR-007 state those choices. T-001 pastes the whole expected
prompt one line per piece, so the whitespace inside and around the tools block is held rather than
left for this engine to choose; T-002 and T-003 name the exact line the block holds and lean on
`20/T-001` for the prompt that has no block, and T-004 names the exact text the filter writes on its
own. Every pasted piece is text the embedded template writes itself, in the order the reference
writes them, so the paste can be checked line by line against the template in `tests/chat.test.tl`
without running an engine at all.

The tool list has to arrive whole, not as a Lua table. A tool schema is arbitrary nested JSON, and
the tables `tltp.json_decode` hands back (it calls the decoder's `unpack`) carry no key order and
give the same empty table for an empty array as for an empty object: a `required` of an empty list
would come out as an object, and a schema's members would be written in hash order, which no exact
case could pin down. So the tool list `render` takes is a `json.Value` of the `json` package this
repository already resolves through `tltp`, whose `children` iterator gives array items and object
members in document order. `render` converts that tree into the values the reader already runs,
keeping the order a map already holds. Only that conversion sees a `json.Value`: the reader's value
tree is unchanged, `tojson` joins `trim` and `length` as a filter the reader runs over those values,
and nothing else in the engine holds a JSON handle.

The tools argument is optional, which is what keeps the two-argument calls `serve` makes today
typechecking and answering the prompts `18` and `20` fix. It is one value rather than a list of
records because the engine must not say what a tool holds: a `record Tool` would fix the members a
schema may carry, and a list of the entries' JSON texts would hand the reader one kind that `tojson`
then decodes again to write back.

```teal
local json = require("json")

local record chat
  record Message
    role: string
    content: string
  end

  render: function(
    tokenizer.Tokenizer, {Message}, tools?: json.Value
  ): string | nil, string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want a list of tool definitions bound into the template, so that the model is shown the functions it may call in the shape it was trained on.
- US-002 (P1) As the engine author, I want `tojson` to write a value as JSON, so that the schema a caller sent is the schema the model reads, its members in the order they were given and an empty list still a list.
- US-003 (P1) As a caller with no tools, I want a render call that carries no tool list to write exactly the prompt it writes today, so that a conversation without tools is unchanged.

## Functional requirements

- FR-001 (US-001) WHEN `render` is called with a tool list, THE SYSTEM SHALL bind the template name
  `tools` to that value converted to the reader's values: a JSON array to a list holding each entry
  in document order, a JSON object to a map holding each member under its own key in document order,
  a JSON string to text, a JSON number to a number, `true` and `false` to booleans and JSON null to
  none, every nested value converted the same way.
- FR-002 (US-003) WHEN `render` is called with no tool list, THE SYSTEM SHALL leave `tools` unbound,
  so that the prompt it returns is the one `20/FR-001` states.
- FR-003 (US-001) IF the tool list is not a JSON array, or an array holds an entry that is not a
  JSON object, THEN THE SYSTEM SHALL bind that value or entry as FR-001's conversion gives it and
  refuse nothing for its shape, so that the template's own guard decides whether a tools block is
  written and `tojson` writes an entry as the kind it became.
- FR-004 (US-002) WHEN the `tojson` filter is applied to a map, THE SYSTEM SHALL write `{`, then for
  each member in the order the map holds them its key as text written by FR-006, `": "`, and that
  member written by FR-004 to FR-007, every pair after the first preceded by `, `, then `}`, an empty
  map being `{}`.
- FR-005 (US-002) WHEN the `tojson` filter is applied to a list, THE SYSTEM SHALL write `[` and `]`
  around its entries, each written by FR-004 to FR-007 and every entry after the first preceded by
  `, `, an empty list being `[]`.
- FR-006 (US-002) WHEN the `tojson` filter is applied to text, THE SYSTEM SHALL write it between
  double quotes, a double quote and a backslash escaped with a backslash, a newline, a tab, a
  carriage return, a backspace and a form feed escaped as `\n`, `\t`, `\r`, `\b` and `\f`, every
  other byte below 0x20 escaped as `\u00` followed by that byte's two lowercase hex digits, and
  every other byte of the text written as it is.
- FR-007 (US-002) WHEN the `tojson` filter is applied to a number, a boolean, none or an undefined,
  THE SYSTEM SHALL write Lua's `tostring` form of the number, `true`, `false`, and `null` for both
  none and an undefined, this being the one place the reference is not followed: the reference writes
  a whole float with a fractional part, so a JSON number given as `1.0` writes `1` here and `1.0`
  there.
- FR-008 (US-002) WHEN a template applies the `tojson` filter to a value in a branch the reader runs,
  a value in a branch `20/FR-010` keeps from running not being reached by this requirement, THE
  SYSTEM SHALL run that filter rather than refuse it, so that `20/FR-012` no longer refuses the
  filter every tool of the file's template is written with.
- FR-009 (US-001) THE SYSTEM SHALL state `json` among the dependencies `lde.json` holds, at the
  version `^0.1.1` that `tltp`'s own statement already asks for and that `lde.lock` resolves to the
  `lde-org/json` checkout this spec's `json.Value` is read from, so that the package FR-001's
  conversion needs is named by this package's own statement, `lde.lock` resolves no new commit, and
  `types/json.d.tl`, which already holds that library's types, is the only declaration written.
  Not testable: a declared dependency, visible in `lde.json` and in `lde.lock`.

## Non-goals

- No request side: the body still carries no `tools` member, `serve` still passes no tool list and
  still refuses a message that is not a string `role` and `content`, so no client can reach this
  over HTTP yet. That is the next spec.
- No tool calls in a history: an assistant turn carrying `tool_calls`, a `tool` role turn,
  `tool_call_id`, `name` and `reasoning_content` as message fields, and the `safe` and `string`
  filters and the walking of a map in a `{% for %}` that the template's tool-call branch needs, are
  later.
- No reading a call out of a reply: parsing a function call from the reply text, `tool_calls` in the
  response, a `finish_reason` naming a tool call and streaming those are a later spec, so generation
  still ends only at an end id or at the budget, the way `16` and `19` ship it.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-005, FR-008): `a tool list becomes
  the tools block the file's template writes`, proves on the file's own template
  as `20/T-001` embeds it, over the one user message `hi`, that `render` with no
  tool list returns the one-turn prompt `20/T-001`'s third case ends with, the
  lone `hi` user turn and the assistant opener, and writes no tools block; and
  that `render` with the tool list `json.decode` gives for the JSON text
  `[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": []}}}]` returns exactly the text WANT below, one line of that
  prompt per piece:

  ```teal
  local WANT = table.concat({
    "<|im_start|>system\n",
    "# Tools\n",
    "\n",
    "You have access to the following functions:\n",
    "\n",
    "<tools>\n",
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": []}}}\n",
    "</tools>\n",
    "\n",
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
    "\n",
    "<tool_call>\n",
    "<function=example_function_name>\n",
    "<parameter=example_parameter_1>\n",
    "value_1\n",
    "</parameter>\n",
    "<parameter=example_parameter_2>\n",
    "This is the value for the second parameter\n",
    "that can span\n",
    "multiple lines\n",
    "</parameter>\n",
    "</function>\n",
    "</tool_call>\n",
    "\n",
    "<IMPORTANT>\n",
    "Reminder:\n",
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
    "- Required parameters MUST be specified\n",
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
    "</IMPORTANT><|im_end|>\n",
    "<|im_start|>user\n",
    "hi<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<think>\n"
  }, "")
  ```

  so that the block holds that entry's JSON text byte for byte, its `required`
  member an empty JSON array rather than an empty object and its members in the
  order that text holds them rather than sorted or hashed, FR-004 and FR-005's
  separators being what the given text already carries; that the tool list
  `[{"type": "function", "function": {"name": "get_weather", "parameters":
  {"type": "object", "properties": {"city": {"type": "string"}}, "required":
  []}}, {"type": "function", "function": {"name": "add", "parameters": {}}}]`
  returns WANT with `{"type": "function", "function": {"name": "add",
  "parameters": {}}}` on the line after the first entry's; that the list
  `[5]`, a number rather than an object, returns WANT with `5` on the entry's
  line and refuses nothing; and that WANT ends with that one-turn `hi` prompt,
  the tools block's system turn being the only turn before it.

- T-002 (FR-001, FR-002, FR-003): `which tool lists write no tools block, and
  which write one`, proves on the file's own template, over the one user message
  `hi`, that each of these returns that same one-turn `hi` prompt, holding no
  tools block: no tool list at all; the tool list `json.decode` gives for `[]`,
  an empty list being false; the tool lists it gives for `5` and `true`, neither
  iterable; the one it gives for `null`, which is false; and the one it gives
  for `{"type": "function"}`, a mapping; that the tool list `json.decode` gives
  for the JSON text `"ab"`, one string rather than an array, returns the tools
  block with `"a"` and `"b"` on the entry lines, a string being iterable and
  each of its characters a text the filter quotes; and that the list it gives
  for `[[]]`, one entry that is an empty list, returns the tools block with `[]`
  on the entry's line, none of these refused for its shape.

- T-003 (FR-001, FR-004, FR-005, FR-007): `tojson writes every value as JSON`,
  proves on the file's own template as `20/T-001` embeds it, over the one user
  message `hi`, that a tool entry whose text is the JSON text `{"type": "function", "function": {"name": "add", "parameters": {}}}`
  writes that text on its own line inside the
  block, so `{}` for its `parameters` is an empty JSON object rather than an
  empty list, each pair written with `": "` after the key and members split by
  `, `; that the same entry with a `default` member of the number `1.0` given in
  the JSON text that way writes `"default": 1`, Lua's `tostring` form rather
  than the reference's `1.0`; and that the tool list `json.decode` gives for
  `[[true, false, null, ["a", 2], {"b": [], "a": 2.5}]]`, one entry that is a
  list rather than an object, writes that entry's line as `[true, false, null,
  ["a", 2], {"b": [], "a": 2.5}]`, byte for byte, the given order of that map's
  two members and its empty `b` array included.

- T-004 (FR-004, FR-005, FR-006, FR-007): `every JSON kind, every escape and
  every key is written`, proves on a hand-made template whose only output is `{{
  tools[0] | tojson }}`, each case being that call over the one message `{role =
  "user", content = "hi"}` and a tool list whose single entry is the value under
  test, that a map holding member `b` of `1` then member `a` of `2` writes
  `{"b": 1, "a": 2}`, the given order rather than a sorted one; that a map whose
  one member's key is the text `a` then a double quote then `b` then a backslash
  then `c`, and whose value is `1`, writes `{"a\"b\\c": 1}`, that key escaped as
  FR-006 escapes a value; that a map holding member `k` of the list of the two
  texts `x` and `y` writes `{"k": ["x", "y"]}`; that an empty map writes `{}`
  and an empty list writes `[]`; that the text `a` then a double quote then `b`
  then a backslash then `c` writes that text quoted with a backslash before both
  of those characters; that a text holding a tab, a newline, a carriage return,
  a backspace and a form feed writes them as `\t`, `\n`, `\r`, `\b` and `\f`;
  that a text holding 0x01 writes `\u0001`, one holding 0x0E writes `\u000e` and
  one holding 0x1A writes `\u001a`, those hex digits lowercase; that a text
  holding the word `cafe` with an accented e writes those bytes unchanged inside
  the quotes; that the number `2` writes `2` and `2.5` writes `2.5`; that `true`
  and `false` write `true` and `false`; that a JSON null writes `null`; and that
  a second template whose only output is `{{ tools[0].absent | tojson }}`, over
  the tool list whose single entry is the map holding member `b` of `1` then
  member `a` of `2`, writes `null` for `absent`, the member that map does not
  hold.

- T-005 (FR-008): `tojson is run and safe is not`, proves that the template `{{
  'x'|tojson }}` renders the text `"x"` and nil rather than a refusal, and that
  the two refused-filter cases this spec displaces, the one in `17/T-003` and
  the one in `20/T-004`, keep refusing once the filter they name is `safe`: each
  answers nil and a line naming `tokenizer.chat_template` and `safe`, the word
  `tojson` in neither line.

## Open questions

- None.

## Amendments

- None.
