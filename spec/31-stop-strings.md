# 31: A reply that ends at a string the client names

**Status**: done   **Depends on**: 16, 18, 19, 23, 30

## Problem & context

A reply today ends in one of two ways: the file's end id, `16/FR-007`, or the budget, `16/FR-008`.
A chat client that wants the reply to stop at a marker it names, a closing tag or a role header the
model was prompted to write next, cannot: the reply runs on past the marker, and the client must
wait for the whole of it and cut the text itself. On a streamed reply even that does not work,
because the marker has already been written into an event, together with the text after it.

The cut is not a `string.find` on the finished text, and the streaming is why. A stop string can
span tokens: two ids whose pieces are `d` and `d` are the string `dd`. So a run may not release a
token's text the moment it samples it, the way `19/FR-004` has `serve.step` write one token's own
text into an event: the tail of what the run holds may be the first half of a stop string, and an
event cannot be taken back once written. A stop string can also sit inside a token whose text is
several bytes long, so a run looks for the strings anywhere in the text it has not yet released and
not only at its end. A run therefore releases nothing while the text it has produced ends in a
suffix a stop string could complete, holding every byte it has not yet released, and gives that
text when the run ends. The pieces a run gives are then together the reply's text, whether the
reply is streamed as pieces or answered in one body, and no piece holds a stop string or ends in a
suffix one could complete.

The holdback measures the text the run produced, never the prompt: a prompt that already ends in a
string the options carry does not end the run at its first step, and a resumed conversation's
stored ids are not read either.

The store is unchanged. `23/FR-002` stores the ids a run forwarded, and an id whose position no
forward took is not one of them, which is why a budget stop's last id is not stored today. The id
whose text completed a stop string ends the run at itself for the same reason: no forward takes its
position, so the next turn of that conversation does not read text the reply never showed. The id
is still one the run sampled, so `sampled` holds it and `18/FR-004`'s counts stand, that id
counting towards `completion_tokens`. A run and its `Text` therefore carry a `cut` that is true
when a stop string ended the run, which is how the route tells that last id from one an end id
followed and forwarded.

Why the reply stopped is `19/FR-005` unchanged: `length` when as many ids were appended as the
budget allows and `stop` otherwise. So a reply a stop string ends reports `stop`, except when the
string completes on the budget's last allowed id, where the counted ids reach the budget and the
report is `length`.

`30`'s non-goal says "cutting a reply at a string is the spec that follows", and this is it. `16`'s
non-goal naming stop strings no longer holds for them, which is said here because a non-goal is not
something an amendment entry can name. `19`'s non-goal about "holding back a token that is half a
character" is a different thing and is still not done: that is about a token that holds one byte of
a multi-byte character, while this holdback is about stop strings and is released when the run ends.
`19`'s problem section's "one event per token as it is appended" now reads as one event per text a
run releases, which is what the amendment to `19/FR-004` says.

```teal
local record generate
  record Options
    -- the other members stand; stop joins them
    stop: {string} | nil
  end
end
```

The record `serve` builds from a body is unexported, so `stop` joins it in the code and no snippet
gains a field: `18`'s wording about the members it ignores and the shape it refuses is what names
it.

## User stories

- US-001 (P1) As the person asking for a reply, I want it to end at a string I choose, so that I get
  the text before my marker and nothing after it.
- US-002 (P1) As a client reading a streamed reply, I want the events to hold neither the stop
  string nor a half of one, and no text to be lost, so that what I receive is exactly the reply's
  text.
- US-003 (P1) As a client of the chat route, I want to send the stop strings in the body and be told
  when one cannot be used, so that a bad value is a refused request and not a surprise.
- US-004 (P2) As a chat client in an ongoing conversation, I want the stored context to hold only
  the ids the run forwarded, so that the next turn does not continue from a token the reply cut
  away.

## Functional requirements

- FR-001 (US-001) WHEN the options carry one or more stop strings, THE SYSTEM SHALL match them
  against the text the run produced and never against the text of the prompt, and SHALL take a
  run's produced text as `tokenizer:decode` of each id it sampled, in the order it sampled them
  from the first sampled id on, joined.
- FR-002 (US-001) WHEN a sampled id's text leaves the text the run has not yet released holding one
  of those strings anywhere within it, THE SYSTEM SHALL end the run at that id, so no further id is
  sampled, SHALL keep that id in `sampled` with every id before it, and SHALL take the `Text`'s text
  as the produced text cut at the earliest position at which one of those strings starts.
- FR-003 (US-002) WHEN a sampled id's text leaves the produced text ending in a suffix that is a
  proper prefix of one of those strings, THE SYSTEM SHALL release nothing as that id's piece; WHEN
  it leaves no such suffix, THE SYSTEM SHALL release the whole of the text it had not yet released
  as that id's piece, so no piece holds a stop string and no piece ends in a suffix a stop string
  could complete.
- FR-004 (US-002) WHEN a run ends, at a stop string, at the end id or at the count, THE SYSTEM SHALL
  release the text it still holds as the run's last piece, cut at the stop position when a stop
  string ended it, and WHEN that release holds no bytes THE SYSTEM SHALL answer no piece, so the
  pieces a run gives, in order, are the `Text`'s text and no piece is empty.
- FR-005 (US-003) WHEN the body's `stop` member is a string of one or more bytes, THE SYSTEM SHALL
  read it as the list of that one string, WHEN it is an array of one to four strings of one or more
  bytes each, THE SYSTEM SHALL read it as that list, and SHALL pass it as `generate.Options.stop`.
- FR-006 (US-003) IF the options' `stop` is given and is not a list of one to four strings of one or
  more bytes each, THEN THE SYSTEM SHALL have `generate.open` return nil and one line naming
  `generate` and `stop`, that check running after `16/FR-009`'s own option checks and before the
  prompt is encoded.
- FR-007 (US-003) IF the body carries `stop` and it is neither a string of one or more bytes nor an
  array of one to four such strings, THEN THE SYSTEM SHALL answer `BadRequest` as `18/FR-008`
  describes it with a message naming `stop`, that check running after the checks `30/FR-008` names.
- FR-008 (US-004) WHEN a turn carrying a `session_id` ends at an id whose text completed a stop
  string, THE SYSTEM SHALL store under that name the prompt's ids followed by the sampled ids
  before that id, so `23/FR-002` still stores the ids the run forwarded and the id that completed
  the string ends the run at itself.

## Non-goals

- No stop ids and no other way for a run to end: the end id and the count stay `16`'s, and a client
  cannot name an id to stop at.
- No matching beyond the bytes a stop string holds: no case-insensitive match, no regular
  expression, no word or character boundary to respect, a string inside one of a token's
  multi-byte characters matching as any other bytes do.
- No cut applied to a resumed conversation's stored ids or to any text of the prompt: a run holds
  back and cuts only the text it produced itself.

## Tests

T-001 uses `30`'s four-piece fixture: pieces `a`, `b`, `c` and `d`, `token_embd.weight` holding four
rows of `{1, 0}`, `output.weight` holding the rows `{0, 0}`, `{1, 0}`, `{2, 0}` and `{3, 0}` so the
scores are 0, 1.4142136, 2.8284271 and 4.2426407, and `tokenizer.ggml.eos_token_id` removed, so
every greedy sample is id 3, the piece `d`. T-002 uses that fixture and `18/T-001`'s, T-003 and
T-004 use `18/T-001`'s, whose pieces are `a`, `b` and `c`, whose scores are 1.4142136, 0 and
1.4142136, whose end id is 1 and whose greedy reply to the prompt `a` is `aa`. T-005 uses a copy of
the four-piece fixture whose fourth piece is `xa` in place of `d`.

- T-001 (FR-001, FR-002): `a reply is cut at the stop string it completes`, proves on the four-piece
  fixture that `generate.text(model, tokens, "a", {max_tokens = 3, seed = 0, stop = {"d"}})` returns
  a `Text` whose text is `""` and whose `sampled` is `{3}`, that id being kept where an end id would
  not be; that `{max_tokens = 3, seed = 0, stop = {"dd"}}` returns text `""` and `sampled` `{3, 3}`,
  the cut reaching back into the `d` the run held when the first id was sampled; and that the prompt
  `d` with `{max_tokens = 3, seed = 0, stop = {"d"}}` returns text `""` and `sampled` `{3}`, the
  prompt's own `d` not ending the run; and that on a copy of the fixture whose fourth piece is `da`
  in place of `d`, the call `generate.text(model, tokens, "a", {max_tokens = 2, seed = 0, stop =
  {"d"}})` returns a `Text` whose text is `""` and whose `sampled` is `{3}`, the string sitting
  inside that one id's text rather than at its end, where a run matching only the end of the text it
  produced would sample a second id and answer `dada`.
- T-002 (FR-003, FR-004, FR-005): `a run holds back what a stop string could still complete`, proves
  on the four-piece fixture, opening a run and driving it one pass at a time as `25/T-001` does, that
  with `{max_tokens = 2, seed = 0, stop = {"ddx"}}` the first accept answers no piece and
  the second answers `dd` as the run ends at the count, the pieces together being the `dd` that
  `generate.text` answers with those options; and on `18/T-001`'s fixture that `serve.stream` on the
  body `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"stop":"aax"}`
  answers a producer which, resumed once and then stepped by `serve.step` until no turn is live,
  writes four payloads in this order: the object whose `delta` is `{role = "assistant"}`; one whose
  `delta` is `{content = "aa"}`; one whose `finish_reason` is `length` and whose `delta` is nil; and
  the text `[DONE]`, that content being the content the body `serve.answer` answers for the same
  request with `"stream":true` removed, so the run released nothing before the count and lost none
  of the `aa` it held.
- T-003 (FR-005, FR-006, FR-007, FR-008): `the route carries stop strings and refuses what it cannot
  use`, proves on `18/T-001`'s fixture that the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stop":"aa"}` answers `OK` with a
  content of `""` and a `usage` of `prompt_tokens` 1, `completion_tokens` 2 and `total_tokens` 3,
  the id that completed the string counting as one the run sampled; that the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stop":"aax"}` answers the same
  content as the same body whose `"stop"` is `["aax"]`, that content being `aa`, the array read as
  the list of its one string; that the first body with
  `"session_id":"s"` added answers `OK` with the empty content and leaves
  `session_of(server, "s").length` at 2, the prompt's one id and the one sampled id the run
  forwarded, the id whose text completed `aa` not being stored; that bodies carrying `"stop":""`,
  `"stop":[]`, `"stop":["a","b","c","d","e"]` and `"stop":7` each answer `BadRequest` with a message
  naming `stop`; and that `generate.open` with `{max_tokens = 1, seed = 0, stop = {}}` returns nil
  and one line naming `generate` and `stop`, that `{max_tokens = 0, seed = 0, stop = {}}` returns
  nil and one line naming `max_tokens`, `16/FR-009`'s own checks running first, and that with the
  prompt `a\255` and `{max_tokens = 1, seed = 0, stop = {}}` it returns nil and one line naming
  `stop` rather than a line from `tokenizer:encode`.
- T-004 (FR-002, FR-004): `a streamed reply that a stop string ends writes no cut text`, proves on
  `18/T-001`'s fixture that `serve.stream` on the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stream":true,"stop":"aa"}`
  answers a producer which, resumed once and then stepped by `serve.step` until no turn is live,
  writes three payloads in this order: the object whose `delta` is `{role = "assistant"}`; one whose
  `finish_reason` is `stop` and whose `delta` is nil; and the text `[DONE]`, no content event being
  written because the run released no bytes, and that the body `serve.answer` answers for that same
  request with `"stream":true` removed carries the empty content; and that the same body with
  `"max_tokens":2` writes the same three payloads except that the `finish_reason` is `length`, the
  ids the run sampled reaching the budget the string completed on.
- T-005 (FR-003): `a run holds back the shorter prefix of a longer stop string`, proves on that copy
  of the four-piece fixture whose fourth piece is `xa` that the run `generate.open` gives for the
  prompt `a` with `{max_tokens = 2, seed = 0, stop = {"abc"}}`, driven one pass at a time as
  `25/T-001` does, answers no piece and no line from its first accept, `xa` ending in the `a` that is
  a proper prefix of `abc` where a run testing only the longest such prefix would release `xa` and
  then lose the start of a string whose first byte it had already released.

## Open questions

- None.

## Amendments

- None.
