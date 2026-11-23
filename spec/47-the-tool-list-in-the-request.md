# 47: The tool list and the tool choice in the request

**Status**: pending   **Depends on**: 17, 18, 20, 22, 46

## Problem & context

`22` built the prompt half of tool calling: `chat.render` takes a tool list as a `json.Value`, binds
the template name `tools`, and the file's own template writes the block from it with `tojson`. The
route never passes one. `request_of` reads no `tools` member, and `prepared` calls
`chat.render(model.tokenizer, request.messages, nil, request.enable_thinking)`, so every prompt is a
tool-free prompt and no model is ever shown a function it may call. `22` recorded that as a non-goal
because the request side was a later spec; this is it, and that bullet is struck from `22` when this
spec ships. `46`'s non-goal that the body still carries no tool list is struck for the same reason,
its other claim, that the thinking input is the fourth, still holding. `45`'s non-goal clause that
the body still carries no `tools` is struck for the same reason; its other clause, that the other
message fields stay unread, still holds.

Two members join the body. `tools` is a list of function definitions, and `tool_choice` is the
choice a template that reads it writes into the prompt. Both are read with the `json` package
rather than `tltp.json_decode`, because a tool schema is arbitrary nested JSON: `tltp`'s unpack
loses an object's member order and gives one entry-less table for an empty object and an empty
array, which is exactly what `22` says the tool list must not become. The body is already known to
be valid JSON by the time these members are read, `request_of` having refused anything else, so the
same text can be decoded a second time without a new refusal path. `chat.render` gains one optional
input after the thinking one and binds the template name only when it is given, in the shape
`46/FR-007` and `46/FR-008` already use, so every call written before this spec answers the prompts
it answered before.

`tool_choice` is template input only: the route does not force a call, forbid one, require one or
translate `auto` into anything, so no engine policy is added and a file's template decides what the
name means. A template that never reads it cannot tell the difference. That is what lets a body's
`tool_choice` reach a template at all without this engine taking a position on sampling.

A tool definition the engine cannot use is refused by name instead of being handed over, the way
`45` refuses a content part it cannot read. What a call needs is a `function.name`; everything else
in a definition, `type`, `description` and `parameters`, is passed through as written for the model
to read. The refusal comes after `46/FR-005`'s, so a body carrying an unreadable thinking member and
an unreadable tool list answers the thinking member, and the order every earlier member's fault has
does not move. A pooled server needs nothing of its own: `28` forwards the body whole to a worker,
whose `request_of` reads these two members from it exactly as this spec states.

The route's prompt is not readable from outside `serve`, and `18`'s fixture tokenizer knows only the
tokens `a`, `b` and `c`, so it cannot encode the file's own long tools block. The route cases
therefore use a marker template that writes a fixture token when the template's own tools guard is
taken, and `22/T-001` keeps pinning the block's text on the file's own template: one case proves the
body's list reaches the guard, the other proves what the guard writes.

```teal
local record chat
  render: function(
    tokenizer.Tokenizer, {Message}, tools?: json.Value,
    enable_thinking?: boolean, tool_choice?: json.Value
  ): string | nil, string | nil
end
```

## User stories

- US-001 (P1) As a chat client, I want the `tools` I send to reach the model's prompt, so that the model can call one of my functions.
- US-002 (P1) As a chat client, I want my `tool_choice` to reach the template that writes the prompt, so that a file's template that reads it writes the instruction I asked for.
- US-003 (P2) As a chat client sending no tools, I want my prompt unchanged, so that a conversation without functions is what it is today.
- US-004 (P2) As a chat client sending a tool definition this engine cannot use, I want 400 naming `tools`, so that I fix that definition instead of guessing.

## Functional requirements

- FR-001 (US-001) WHEN the body carries a `tools` member that is a JSON array, THE SYSTEM SHALL
  give `chat.render` that member, unchanged, as its tool list, so that `22/FR-001` binds the
  template name `tools` to it.
- FR-002 (US-003) WHEN the body carries no `tools` member, THE SYSTEM SHALL give `chat.render` no
  tool list, so that `22/FR-002` leaves `tools` unbound and the prompt is the one-turn prompt
  `20/FR-001` states. A body carrying an empty array gives that array as FR-001 states; the file's
  own template writes no tools block for it because an empty list is false, as `22/T-002` records,
  and a template whose guard tests only `tools is defined` writes its own block for it.
- FR-003 (US-002) WHEN the body carries a `tool_choice` member, THE SYSTEM SHALL give `chat.render`
  that member, unchanged and whatever JSON value it holds, JSON null included, as its tool-choice
  input, which `22/FR-001` converts to none and FR-004 binds, so such a call is not refused and its
  `tool_choice` is defined; WHEN it carries none, THE SYSTEM SHALL give it none.
- FR-004 (US-002) WHEN `chat.render` is called with a tool-choice input, THE SYSTEM SHALL bind the
  template name `tool_choice` to that value converted as `22/FR-001` converts a tool list, and bind
  no other name that the call did not already bind.
- FR-005 (US-002) WHEN `chat.render` is called with no tool-choice input, THE SYSTEM SHALL leave
  `tool_choice` unbound, so that the call writes the prompt it writes today and a template that
  never reads the name cannot tell the difference.
- FR-006 (US-004) IF the body carries a `tools` member that is not a JSON array, or holds an entry
  that is not a JSON object, or an entry whose `function` member is not a JSON object, or an entry
  whose `function.name` member is not a string of one or more bytes, THEN THE SYSTEM SHALL answer
  `BadRequest` with a body whose `error.message` names `tools` and, for an entry at fault, that
  entry's position counting from one, and generate no token; JSON null being one of the kinds a
  `tools` member that is not an array is refused for.
- FR-007 (US-004) IF the body carries both a member a check of `18/FR-008` or of `46/FR-005` and
  `46/FR-006` refuses and one `FR-006` refuses, THEN THE SYSTEM SHALL answer the message the
  earlier member's fault gives, `FR-006`'s check coming after every check `18/FR-008` names and
  after `46/FR-005`'s and `46/FR-006`'s, so that no fault `FR-006` names displaces one an earlier
  member gives.

## Non-goals

- No engine policy for `tool_choice`: the route does not force, forbid or require a call, reads no
  meaning out of `auto`, `none` or `required`, and leaves the value's kind unchecked, so a template
  that never reads the name writes the prompt it writes today.
- No check on a tool definition beyond its name: `type`, `description`, `parameters` and every other
  member pass through as written, and a definition holding a member this engine does not know is
  not refused.
- No tool calls in a history or out of a reply: `tool_calls`, a `tool` role, `tool_call_id`, `name`
  and `reasoning_content` stay unread message fields, and parsing a call from the reply text,
  `tool_calls` in the response, a `finish_reason` naming a call and streaming those stay later.

## Tests

- T-001 (FR-001, FR-002): `a body's tool list reaches the template's tools block`, proves on `18`'s
  fixture, whose `tokenizer.chat_template` is replaced as `46/T-002` replaces it by
  `{% for message in messages %}{{ message.content }}{% endfor %}{% if tools and tools is iterable
  and tools is not mapping %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}`,
  that the body `{"messages":[{"role":"user","content":"a"}]}` carrying
  `"tools":[{"type":"function","function":{"name":"b"}}]` answers `prompt_tokens` 2; that the same
  body carrying `"tools":[{"type":"function","function":{"name":"b"}},{"type":"function","function":{"name":"c"}}]`
  answers 3, both entries reaching the loop; that the body with no `tools` answers 1, the count it
  answers today; and that the body carrying `"tools":[]` answers 1 too, the empty list taking the
  template's guard no further; and that on a copy whose `tokenizer.chat_template` is
  `{% for message in messages %}{{ message.content }}{% endfor %}{% if tools is defined %}b{%
  endif %}` the body carrying `"tools":[]` answers `prompt_tokens` 2 and the body carrying no
  `tools` answers 1, an empty array reaching `render` as a bound empty list and no array reaching
  it as nothing. The text of no prompt formed here is pinned: `22/T-001` pins the block the file's
  own template writes for a tool list and the prompt it writes for none, and this case holds that
  the body's list is what reaches them.

- T-002 (FR-003, FR-004, FR-005): `a body's tool_choice reaches the template, and none leaves it
  unbound`, proves on `18`'s fixture, whose `tokenizer.chat_template` is replaced by
  `{% for message in messages %}{{ message.content }}{% endfor %}{% if tool_choice is defined %}{%
  if tool_choice == 'none' %}b{% elif tool_choice is mapping %}{{ tool_choice.function.name }}{%
  else %}cc{% endif %}{% endif %}`, that `A_BODY` answers `prompt_tokens` 1, its `tool_choice`
  unbound; that `A_BODY` carrying `"tool_choice":"none"` answers 2, the text read as text; that
  `A_BODY` carrying `"tool_choice":{"type":"function","function":{"name":"c"}}` answers 2, that
  value a map the template reads the name out of; that `A_BODY` carrying `"tool_choice":"auto"`
  answers 3, the catch-all branch taken because the route hands the word over as text and turns it
  into nothing of its own; that `A_BODY` carrying `"tool_choice":null` answers 3, the JSON null
  given as FR-003 states and bound as none, so the template sees it defined; and that `A_BODY`
  carrying the one valid tool of T-001 and no `tool_choice` answers 1, a spurious binding of that
  list landing in the catch-all branch and answering 3 instead.

- T-003 (FR-006, FR-007): `a tools member the route cannot use is a bad request`, proves on `18`'s
  fixture that each of these answers `BadRequest` with a body decoding to an object whose only
  member is `error`, of type `invalid_request_error`: `A_BODY` carrying `"tools":5`,
  `"tools":null`, `"tools":{}` and `"tools":"ab"`, each message naming `tools` and none naming an
  entry position; and `A_BODY` carrying `"tools":[5]`, `"tools":[{"type":"function"}]`,
  `"tools":[{"type":"function","function":{}}]`, `"tools":[{"type":"function","function":{"name":""}}]`,
  `"tools":[{"type":"function","function":{"name":5}}]` and
  `"tools":[{"type":"function","function":{"name":"b"}},{"type":"function","function":{}}]`, each
  message naming `tools` and the entry's position counting from one; and that a body carrying both
  `"stream":5` and `"tools":5` names `stream`, one carrying both `"chat_template_kwargs":5` and
  `"tools":5` names `chat_template_kwargs`, and one carrying both `"enable_thinking":"yes"` and
  `"tools":5` names `enable_thinking`, the checks this requirement comes after answering first.

- T-004 (FR-004, FR-005): `the tool-choice input binds its own name and no other`, proves on the
  hand-made template `{% if tool_choice is defined %}t{% endif %}{% if preserve_thinking is defined
  %}p{% endif %}` over the one user message `hi`, that `chat.render` given `json.decode('"auto"')`
  as its tool-choice input writes `t` alone, and that `chat.render` given none writes nothing, so
  that input binds `tool_choice` and reaches no other name.

## Open questions

- None.

## Amendments

- `17/FR-011`: `tool_choice` joins the reachable names, bound from the tool-choice input `47/FR-004`
  names and left unbound when a call gives none as `47/FR-005` states.
- `18/FR-005`: `tools` and `tool_choice` join the honoured members, `tools` as `47/FR-001`,
  `47/FR-002` and `47/FR-006` read it and `tool_choice` as `47/FR-003` reads it.
- `18/FR-008`: the members whose value is refused gain `tools`, refused as `47/FR-006` states, that
  check coming last, after every check this requirement names and after both `46/FR-005`'s and
  `46/FR-006`'s.
