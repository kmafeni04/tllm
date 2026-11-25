# 51: The reasoning text the model wrote

**Status**: pending   **Depends on**: 10, 19, 46, 49, 50

## Problem & context

`46` asks a file's template for a plain answer, and the ask does not always hold: a file whose
generation opener is not gated on `enable_thinking` opens a thinking block anyway, and a body that
sends the value true asks for one on purpose. Either way the model's reasoning arrives inside
`content`, so the harness that sent the request shows the working as the answer. That spec's "No
reasoning channel in the reply" bullet named this gap and left it here.

The opener is not in the reply. The file's template writes `<think>\n` as the last thing before the
model answers (`tests/chat.test.tl:165-168`), and `46/T-001` pins that tail: `ONE_TURN` ends
`"<|im_start|>assistant\n<think>\n"`, and with thinking off it ends
`"<|im_start|>assistant\n<think>\n\n</think>\n\n"` instead. A reply from those files therefore
*starts inside* the block: its first text is the reasoning and the only marker in it is
`</think>`, which is the shape `20/T-001`'s replay turn already carries as
`"plan" .. fixture.think_close .. "answer"`. So a cut that waits for an opening marker at the head
of a reply reads no block from either shape, and the harness keeps the working as the answer.

What denotes reasoning is the marker pair, and one pair per reply, at its head. A pair open at the
head is one the reply itself opens, or one the prompt left open, and the reply's first closer then
ends it: everything from the head to that closer is the block. A marker that stands later in a
reply, with no pair open at the head, is ordinary text and reaches no cut, so a model that writes
`<think>` inside its answer keeps that answer whole. That reading is also what the stream can act
on: a streamed turn holds back only a head that could still be such a block, which is what lets its
events agree with the body a finished reply answers for the same text, and an answer whose head can
be no block keeps arriving one release per step in the shape `19/T-001` records, one release per step
being what that run gives when the body carries no `stop` member for `31` to withhold against.

Nothing cuts a reply today. `serve.finished` puts the run's text into
`choices[0].message.content` unchanged (`49/FR-004`), and `serve.step` sends each piece a run
releases as `delta.content` (`19/FR-004`), so a block at the head of a reply reaches the client as
the answer and the client has nothing to tell the two apart.

The replay cannot be answered either. A harness that received reasoning sends it back on the
assistant turn as a `reasoning_content` member, and `48` reads `role`, `content`, `tool_calls`,
`tool_call_id` and `name`, so that member is dropped on the way: `20/T-005` records that
`{{ messages[0].reasoning_content }}` writes nothing, and `45`'s and `48`'s non-goals named the name
as staying unread. Nothing today refuses it, so the member is already accepted, and nothing carries
it either, which is the half this spec settles: the turn is taken as the text `48` renders, and the
reasoning a harness kept is lost from the conversation it replays.

The markers are the pair the file's template writes, held in `chat` beside the call pair `49` put
there, the pair `<think>` and `</think>`: that text and the text a reply is cut at are one fact, and
`serve` shapes the body from what the reader returns. A family that names its reasoning some other
way is that reader's business, which stays with the plugin architecture for the reason `49` gives.
The pair is not the closer `17` ends a turn at: that one ends the whole turn, while this pair sits
inside it.

Two channels, joined the way `49` and `50` join theirs. A finished reply cuts the head block out of
its text and answers `choices[0].message` with the reasoning as `reasoning_content` and the text
after the closer as `content`; a reply cut inside its block before any closer answers that text as
`reasoning_content` and an empty `content`, so the two channels label the same reply the same way.
The call scan then runs on what the cut left, so a call block written inside the reasoning is no
call. A stream holds the head back while it could still be an open block, as `50/FR-005` and
`50/FR-006` hold a call marker, sends the reasoning as one `delta.reasoning_content` event ahead of
any content delta, and releases at the run's end a block the budget cut inside instead of dropping
it.

This spec strikes one bullet and replaces three clauses when it ships: the whole of `46`'s "No
reasoning channel in the reply" bullet, and the "`reasoning_content` stays unread" clause of `45`'s
"No other message field" bullet, of `48`'s first bullet and of `49`'s first, each becoming
"`reasoning_content` is accepted and reaches no template name and no prompt": the member's presence is
honoured at last, its content stays unread, and FR-011 is what says so. `50`'s first bullet holds no
such clause.

## User stories

- US-001 (P1) As a harness author, I want a reply's reasoning to arrive as `reasoning_content` and
  its answer as `content`, so that I show the answer and keep the working in its own place.
- US-002 (P1) As a streaming client, I want reasoning to arrive as `delta.reasoning_content` before
  the answer's deltas, so that I can show the two apart while the reply is still being generated.
- US-003 (P1) As a harness author, I want to replay an assistant turn that carries
  `reasoning_content`, so that a conversation this engine answered comes back to it without a fault.
- US-004 (P2) As a client of a reply with no thinking block open at its head, I want the reply I get
  today, streamed piece by piece as it arrives, so that nothing about an ordinary answer changes.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL read a reply's reasoning block from two texts, that reply and the
  prompt that turn's request rendered, and from the thinking marker pair that `chat` names beside the
  call pair `49` names, a pair standing open at that reply's head when that prompt ends with that
  pair's opening marker with nothing after it but spaces, tabs, newlines and carriage returns, or
  when that reply's first text other than spaces, tabs, newlines and carriage returns begins with
  that opening marker: the block being that reply's text from its first byte through the first closing
  marker standing at or after that head, or through the end of the reply when no such marker stands
  there, its reasoning being the text after that opening marker and before that closing marker with the
  spaces, tabs, newlines and carriage returns at either end removed and that head opening marker
  removed when the reply holds one, that reasoning being empty when the two markers stand together, one
  reply holding at most that first block, and no block being read from a reply whose head holds no open
  pair.
- FR-002 (US-001) WHEN a reply's head holds an open pair and that reply holds that pair's closing
  marker, and no call block `49/FR-001` as FR-005 runs it stands after that marker, THE SYSTEM SHALL
  answer `OK` with a body whose `choices` holds one object whose `index` is 0 and holding no
  `finish_reason`, whose `message` is an object whose `role` is `assistant`, whose
  `reasoning_content` is that block's reasoning and whose `content` is the reply's text from the byte
  after that closing marker on keeping every later marker pair it holds, the body otherwise being
  the one `18/FR-003` describes.
- FR-003 (US-001) IF a reply's head holds an open pair and that reply holds no closing marker of
  that pair, THEN THE SYSTEM SHALL answer the body FR-002 describes with that block's reasoning as
  the message's `reasoning_content` and the empty text as its `content`, so a reply the token budget
  cut inside its block answers its working as working and no text of it as an answer.
- FR-004 (US-004) IF a reply's head holds no open pair, THE SYSTEM SHALL answer the body
  `49/FR-003` or `49/FR-004` gives for that text, whose answered message holds no
  `reasoning_content` member and whose `content` is that text whole, every marker pair or lone marker
  standing later in it being text, so a model that writes the pair inside its own answer loses none
  of that answer.
- FR-005 (US-001) WHEN a reply's head holds an open pair and that pair's closing marker arrived in
  that reply, THE SYSTEM SHALL scan the text from that closing marker on for call blocks as
  `49/FR-001` scans, so a reply holding both answers `49/FR-003`'s call body with that block's
  reasoning added to its message as FR-002 names it, a call marker standing inside that block reaching
  no call, and text after that marker that leaves a call block unterminated answering no call and no
  refusal, `49/FR-005` refusing only a complete block `49/FR-001` finds there. A block that ran to the
  end of the reply leaves no text to scan, so FR-003 answers it and no call block is read from its
  text, broken or whole, `49/FR-005` refusing nothing that the cut took away.
- FR-006 (US-002) WHILE a streamed turn's run is live, no byte of that reply having gone out in any
  event of that turn, that reply's text so far holding no closing marker of that pair, and that pair
  standing open at that reply's head as FR-001 reads one, or that reply's text so far holding nothing but
  spaces, tabs, newlines and carriage returns, or that reply's first text other than spaces, tabs,
  newlines and carriage returns being one or more bytes that that marker's text continues without
  spelling whole, THE SYSTEM SHALL send no event whose `delta` holds any byte of that reply, so no step
  of a turn whose head has not settled yet sends any byte of that head, and FR-010's answer running after
  this hold in that order.
- FR-007 (US-002) WHEN the text a streamed turn has not yet sent holds that pair's closing marker, a
  pair standing open at that reply's head as FR-001 says and not yet closed there, THE SYSTEM SHALL send
  one event whose `choices` holds one object whose `index` is 0 and whose `delta` is an object whose
  `reasoning_content` is that block's reasoning as FR-001 reads it, empty when the two markers stand
  together, and which holds no `content`, before any event whose `delta` holds `content`, and SHALL
  leave the reply's text after that closing marker as the text that turn has not yet sent: one such
  event per reply, that pair being closed from that event on, and one release holding more than one
  closing marker of that pair sending that one event and leaving the reply's text after the first such
  marker as the text that turn has not yet sent, so a marker standing later in that reply streams as
  content, as FR-002 answers it finished.
- FR-008 (US-002) WHEN a streamed turn's run ends, an open pair standing at that reply's head as FR-001
  says and no event of FR-007 having closed that pair, and the text that turn has not yet sent holding no
  closing marker of that pair, THE SYSTEM SHALL send, before that turn's ending event, one event whose
  `choices` holds one object whose `index` is 0 and whose `delta` is an object whose `reasoning_content`
  is the text from that head through the end of the reply, read as FR-001 reads a block's reasoning and
  holding no `content`, that event being sent when that reasoning is empty as well, as FR-001 reads a
  block whose two markers stand together, and SHALL hold nothing back after that event: that text and
  that pair's opening marker are no longer the text that turn has not yet sent, so no later event of
  that turn carries them again.
- FR-009 (US-002) WHEN a streamed turn has sent an event of FR-007 or FR-008, THE SYSTEM SHALL send
  the text it has then not yet sent as `50/FR-007` and `50/FR-008` give it and end that turn with the
  event `19/FR-005` gives, so a reasoning event displaces neither the reply's own text nor the reason
  its run stopped.
- FR-010 (US-004) WHILE a streamed turn's run is live, no open pair standing at that reply's head as
  FR-001 reads one, and that reply's first text other than spaces, tabs, newlines and carriage
  returns being neither that pair's opening marker nor one or more bytes that its text continues
  without spelling that marker whole, THE SYSTEM SHALL answer that turn's events as `50/FR-007` and
  `50/FR-008` give them and send no `delta.reasoning_content` event for it at all, so an answer that
  opens with its own text keeps that text on the wire one release per step, and a marker standing
  later in it reaches no hold and no cut, as FR-004 answers it finished.
- FR-011 (US-003) WHEN a message entry carries a `reasoning_content` member of any kind, THE SYSTEM
  SHALL accept that entry, hand `chat.render` its `role`, `content`, `tool_calls`, `tool_call_id` and
  `name` as `48/FR-008` hands them, bind no template name from that member, and render the prompt the
  same messages without that member render, token for token, so a replayed turn reaches the model as
  the text `48` renders and its reasoning reaches no refusal and no prompt.

## Non-goals

- No reasoning read out of a file's template or its vocabulary: the pair is matched as a constant of
  the core against the reply and the rendered prompt, `49`'s call pair's equal, and no template name
  and no vocabulary entry is consulted for it, so a family that writes its reasoning another way waits
  for the plugin reader and `preserve_thinking` stays unread.
- No reasoning that reaches the prompt: a replayed `reasoning_content` member is dropped unread, no
  block is written back into a history, and what a session stores stays the prompt ids and the
  sampled ids it stores today, the block's own markers included.
- No change to what the run does: no id is stopped, steered or forced at a closer, no stop rule ends
  a run at a closing marker, a reasoning block spends the `max_tokens` budget it spends today, and a
  turn cut inside its block still ends `length`. No hold splits a character: the markers are ASCII
  bytes, so text is cut and released whole and no event of this spec carries an undecodable
  fragment.
- No other reply shape: no `reasoning` member, no usage or token split for the reasoning, no second
  block cut from a reply at all, no call block read out of the text inside the head block, and no
  reasoning read from a reply whose head holds no open pair.

## Tests

`OPEN` and `CLOSE` are the `<think>` and `</think>` literals each new case declares, the pair
`tests/chat.test.tl:227-228` exports as `think_open` and `think_close` and `20/T-001` embeds. `NL` is the
two newlines that the closer of `20/T-001`'s replayed assistant turn leaves behind it in the prompt that
test records, that closer standing inside that message's own text there. `NL` is a literal a case passes
behind its own closer in the reply text: no builder below adds bytes of its own, and `first_ws_open`'s
template is the one place that writes those two newlines, behind its opening marker. `HALF` is the first
four bytes of `OPEN`, `REST` the bytes `HALF` does not hold, and `SHORT` the first three. `PING` is the
call block text `<tool_call><function=ping></function></tool_call>` from `49/T-003`, which
`tests/serve.test.tl` writes inline, and `BROKEN` is that text with an empty function tag,
`<tool_call><function=></function></tool_call>`, a block beginning with
`<function=` and so one that `49/FR-002` refuses.

`first_token(reply)` at `tests/serve.test.tl:75` and `two_pieces(one, two)` at `:839` are that file's
own builders: token zero is the text it is given, and `two_pieces` samples its two texts as ids 0 and
1 with the end id moved to 2.

`first_open(reply)` and `open_pieces(one, two)` are those two turned into a turn whose prompt leaves the
pair open, on `tests/serve_fixtures.tl`'s `t001_meta` at `:214` and `t001_entries` at `:265`, with five
changes. `tokenizer.ggml.tokens` gains a fourth entry whose text is `OPEN`, and
`tokenizer.ggml.token_type` at `:229` marks that entry the control kind that `10/FR-014` encodes as one
id for its own text. `tokenizer.chat_template` replaces the message loop that entry carries at
`tests/serve_fixtures.tl:233-235` with a template whose whole text is the `OPEN` literal, holding no loop
over `messages` unlike the entry it replaces, so the prompt that turn renders is that one id, ends inside
the block and encodes to one id whatever the body's messages are. `token_embd.weight` at `:269` and
`output.weight` at `:271` gain a fourth column as `two_pieces` replaces its own, that column copying id
2's, so the ids those cases name are the ones sampled and no end id is sampled inside the budget.
`first_closed(reply)` is `first_open` with that template the `CLOSE` literal alone instead, a prompt that
closed its pair. `first_ws_open(reply)` is `first_open` with that fourth entry's text and that template
the `OPEN` literal followed by the two `NL` newlines instead, that text still encoding as that one id, so
the prompt ends with whitespace after the marker the way the file's own generation opener does.

`OPEN_BODY` is `{"messages":[{"role":"user","content":"a"}],"max_tokens":1}`, the shape that file's
`CALL_BODY` at `:70` carries with a budget of one, and `OPEN_STREAM(budget)` the same with
`"stream":true` and that budget. `CALL_BODY`, `CALL` and `A_BODY` are that file's own. A payload is
read with `tltp.json_decode` unless it is `[DONE]`, and a producer is resumed once, then stepped by
`serve.step` until no turn is live.

`ReplyMessage` and `StreamDelta` in `tests/serve_fixtures.tl` hold no reasoning member today, so
every case below that reads one decodes with `tltp.json_decode` into its own record, the way `49`'s
and `50`'s cases read `tool_calls`.

- T-001 (FR-001, FR-002, FR-003): `a reply that opens inside a thinking block answers reasoning and
  answer`, proves on `first_open("plan" + CLOSE + NL + "answer")` with `OPEN_BODY` that `serve.answer`
  answers `OK` whose `choices` holds one object whose `index` is 0 holding no `finish_reason`, whose
  `message` is `{role = "assistant", reasoning_content = "plan", content = NL + "answer"}`, the answer
  keeping the `NL` literal this case passes behind that closer; on `first_open(NL + "plan" + NL +
  CLOSE + "done")` with that body, that `reasoning_content` is `plan` and `content` is `done`, the
  newlines opening the prompt's block and the padding around the reasoning being removed; on
  `first_open(OPEN + "plan" + CLOSE + "out")` with that body, that `reasoning_content` is `plan` and
  `content` is `out`, an opening marker the reply repeats itself reaching no reasoning; on
  `first_ws_open("plan" + CLOSE + "done")` with that body, that `reasoning_content` is `plan` and
  `content` is `done`, a prompt ending in the newlines its template writes after the marker leaving the
  pair open at that head as the tail `46/T-001` pins; on
  `first_open(CLOSE + "out")` with that body, that `reasoning_content` is the empty text and
  `content` is `out`, a block whose two markers stand together answering an empty reasoning member;
  and on `first_open("plan only")` with that body, that `reasoning_content` is `plan only` and
  `content` is the empty text, the reply the budget cut inside its block answering no text as an
  answer.
- T-002 (FR-001, FR-002, FR-003, FR-004, FR-005): `reasoning ahead of a call is answered beside the
  call`, proves on `first_open("think" + CLOSE + CALL)` with `OPEN_BODY` that the body decodes to
  `choices` holding one object whose `finish_reason` is `tool_calls` and whose `message` is
  `{role = "assistant", content = "", reasoning_content = "think", tool_calls = {{id = "call_1", type =
  "function", function = {name = "get_weather", arguments = '{"city":"Paris"}'}}}}`, the prose around the
  call block being dropped as `49/FR-003` drops it and the scan running on the text after that closer; on
  `first_open("a" + PING + "b" + CLOSE + "c")` with `OPEN_BODY`, where `PING`'s own text holds no
  closer of the thinking pair, that `content` is `c`, `reasoning_content` is `a` followed by that
  whole call block and `b`, and the message holds no `tool_calls` and no `finish_reason`, no call
  being read out of a cut block's own text; on `first_open("plan" + CLOSE + "answer" + OPEN +
  "more" + CLOSE)` with `OPEN_BODY`, that `reasoning_content` is `plan` and `content` is `answer`
  followed by that second pair's two markers and its `more` text, only the block at the head of a
  reply being cut; on `first_token("Hi " + OPEN + "mid" + CLOSE + "done")` with `CALL_BODY`, that the
  `message`'s `content` is that whole reply text and holds no `reasoning_content`, a pair standing
  later in a reply being ordinary text; on `first_closed("plan" + CLOSE + "done")` with `OPEN_BODY`,
  that the `message`'s `content` is that whole reply text and holds no `reasoning_content`, a prompt
  that wrote the closer and not the opener leaving no pair open at a head; on
  `first_open("plan" + BROKEN)` with `OPEN_BODY`, a block the budget cut before any closer holding a
  call block that `49/FR-002` refuses, answering `OK` with `reasoning_content` that whole reply text and
  `content` the empty text rather than the `InternalServerError` `49/FR-005` gives that block; on
  `first_open("plan " + PING)` with `OPEN_BODY`, a block the same budget cut before any closer holding a
  `PING` call that `49/FR-002` reads whole, that the body answers `OK` with `reasoning_content` that
  whole reply text, `content` the empty text, and no `tool_calls` member and no `finish_reason`, so the
  scan reaches no text this spec's cut took and neither of `49`'s two bodies answers it; and on `18`'s
  own fixture with `A_BODY`, that the `message`'s `content` is `a` written 256 times, as `18/T-001`
  records it, and holds no `reasoning_content`.
- T-003 (FR-002, FR-006, FR-007, FR-009): `a streamed reply sends its reasoning before its content`,
  proves on `open_pieces(NL + "plan", CLOSE + NL + "answer")` with `OPEN_STREAM(2)` that `serve.stream`
  answers `OK` with a producer that writes exactly five payloads, in order: the `role` object; one
  whose `delta` is `{reasoning_content = "plan"}`; one whose `delta` is `{content = NL + "answer"}`;
  one whose `finish_reason` is `length`, whose `index` is 0 and whose `delta` is nil; and `[DONE]`,
  no payload holding a marker byte and no `content` delta ahead of that reasoning delta, the block's
  text being withheld across the two releases that carry it, the newlines that open the prompt's block
  and sit before its closer reaching no event, and that content payload being the text FR-002 answers for
  the same reply, the `NL` literal that case passes behind its closer included; and on
  `two_pieces(OPEN + "plan" + CLOSE, "out")`, whose pair sits whole in id 0, with the body
  `{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}`, that five payloads come
  out in order: the `role` object, one whose `delta` is `{reasoning_content = "plan"}` and one whose
  `delta` is `{content = "out"}`, the `length` ending and `[DONE]`, the pair the model wrote itself being
  the same block; and on `open_pieces("plan" + CLOSE, CALL)` with `OPEN_STREAM(2)`, a block that closes
  in the first release ahead of the call the second brings, that the producer writes exactly eight
  payloads: the `role` object; one whose `delta` is `{reasoning_content = "plan"}`; one whose `delta` is
  `{content = "Hi"}`; the two `get_weather` payloads `50/T-001` records; one whose `delta` is
  `{content = "done"}`; the `tool_calls` ending; and `[DONE]`, that reasoning event standing ahead of both
  call payloads and no payload carrying any text of the call block itself, so a reasoning event displaces
  neither that call's own events nor the prose around it nor the reason the run stopped, which is the
  `tool_calls` ending `19/FR-005` gives and not the `length` ending.
- T-004 (FR-006, FR-007, FR-008, FR-009, FR-010): `a reply cut inside its block releases that block at
  the end`, proves on `first_open("partial")` with `OPEN_STREAM(1)`, whose opener is the prompt's and
  unclosed, that the producer writes exactly four payloads: the `role` object; one whose `delta` is
  `{reasoning_content = "partial"}`, that text being the run's whole release; one whose
  `finish_reason` is `length`, whose `index` is 0 and whose `delta` is nil; and `[DONE]`, no payload
  holding a marker byte or carrying a `content` member, the text FR-006 withheld being released once
  by FR-008 and then held by nothing; and on the body
  `{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}` with
  `two_pieces(NL + SHORT, "ink>" + "plan" + CLOSE + "out")`, whose reply opens with the two newlines that
  settle no head and whose first piece is an opening marker holding only its first three bytes, completed
  by the second piece at a different byte than the half the other cases of this spec cut at, that five
  payloads come out in order: the `role` object, one whose `delta` is `{reasoning_content = "plan"}`, one
  whose `delta` is `{content = "out"}`, the `length` ending and `[DONE]`: those two newlines, that partial
  opener and the release behind it being held off the wire together, no payload of that turn holding a
  byte of the opening marker, the bytes completing it at the head moving that head to FR-007, the answer
  after the closer being `out` alone, and no byte of that reply reaching the client twice; on
  `two_pieces(HALF, "plain")`, whose head prefix nothing completes and no closer arrives, with that same
  body, FR-006's hold being the only hold on that turn, that the payloads are the `role` object and one
  whose `delta` is `{content = HALF + "plain"}`, the `length` ending and `[DONE]`, the held text going out
  as one content event at the run's end, no `delta` of that turn carrying a `reasoning_content` member and
  no byte dropped; and on `two_pieces(HALF, REST + CLOSE)`, the closer standing with nothing behind it,
  with that same body, that the payloads are the `role` object and one whose `delta` is
  `{reasoning_content = ""}`, the `length` ending and `[DONE]`, the marker completing byte by byte and a
  block whose reasoning is empty still answering a reasoning
  event; on `first_open("plan" + BROKEN)` with `OPEN_STREAM(1)` carrying a `session_id` of `s`, that
  the producer writes exactly four payloads: the `role` object; one whose `delta` is
  `{reasoning_content = "plan" + BROKEN}`, that text being the run's whole release; the `length`
  ending; and `[DONE]`, no payload carrying the `error` ending, no `delta` carrying `content`, no
  `tool_calls` event anywhere in that stream, and `serve.session_of(server, "s")` answering the
  conversation that body names rather than the nil `50/T-003` records for that same block text; and on
  `first_open("plan " + PING)` with `OPEN_STREAM(1)`, that the payloads are the `role` object, one
  whose `delta` is `{reasoning_content = "plan " + PING}`, the `length` ending and `[DONE]`, so the
  call block the finished body leaves inside its reasoning streams as that same reasoning and never as
  a call; on the body `{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}` with
  `two_pieces(OPEN + "one" + CLOSE + NL + "x", "y" + CLOSE + "z")`, a reply whose head block closes in the
  first release and whose second release brings another closing marker with no opener after it, that the
  producer writes exactly five payloads: the `role` object; one whose `delta` is
  `{reasoning_content = "one"}`; one whose `delta` is `{content = NL + "x" + "y" + CLOSE + "z"}`, that
  text being what the first closer left behind it joined to the second release and being the `content`
  FR-002 answers for that same reply text; the `length` ending; and `[DONE]`, no second payload anywhere
  in that stream holding a `reasoning_content` member and no payload holding that head block's opening
  marker, so one reply answers one reasoning event and a closer standing later in it streams as content,
  as FR-002 keeps it finished; and on `first_ws_open("plan")` with `OPEN_STREAM(1)`, whose prompt ends in
  the two newlines its own template writes behind the opener, that the producer writes exactly four
  payloads: the `role` object; one whose `delta` is `{reasoning_content = "plan"}`; the `length` ending;
  and `[DONE]`, no
  payload of that turn carrying a `content` member, a prompt ending in whitespace behind its marker
  leaving the pair open at that head and FR-006 holding back the release `50/FR-007` would otherwise send
  as content.
- T-005 (FR-004, FR-010): `a reply that opens with its answer streams that answer whole`, proves on
  `first_token("answer")` with `{"messages":[{"role":"user","content":"c"}],"max_tokens":1,
  "stream":true}` that the payloads are the `role` object, one whose `delta` is
  `{content = "answer"}`, the `length` ending and `[DONE]`; on `two_pieces("a", "b")` with
  `{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}`, a body carrying no
  `stop` member so the run withholds nothing of its own and releases one id per step, that five
  payloads come out in order: the `role` object, one whose `delta` is `{content = "a"}` and one whose
  `delta` is `{content = "b"}`, the `length` ending and `[DONE]`, one content event per release as
  `19/T-001` records for that fixture, no hold standing on a turn whose pair is open nowhere; on
  `two_pieces("answer", OPEN + "more" + CLOSE)`, whose id 0 is the answer and whose id 1 is a pair
  behind it, with that same body, that the payloads are the `role` object, one whose `delta` is
  `{content = "answer"}`, one whose `delta` is `{content = OPEN + "more" + CLOSE}`, the `length`
  ending and `[DONE]`, that closed pair reaching the client whole as the body FR-004 answers for the
  same text and as `50/FR-007` releases it, with no reasoning event standing in that stream; on
  `first_token(CLOSE + "b")` with `{"messages":[{"role":"user","content":"c"}],"max_tokens":1,
  "stream":true}`, that the payloads are the `role` object, one whose `delta` is
  `{content = CLOSE + "b"}`, the `length` ending and `[DONE]`, a reply opening with a closer that no
  prompt opened being content under every hold; and on `two_pieces("answer", OPEN + "more")`, whose
  opener stands behind the answer and is never closed, with that same budget-2 body, that the
  payloads are the `role` object, one whose `delta` is `{content = "answer"}`, one whose `delta` is
  `{content = OPEN + "more"}`, the `length` ending and `[DONE]`, that opener standing behind sent
  text reaching no hold at all.
- T-006 (FR-011): `a replayed reasoning member is accepted and reaches no prompt`, proves on `18`'s
  fixture and the template it already carries,
  `{% for message in messages %}{{ message.content }}{% endfor %}`, that the body
  `{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}` answers `OK` with
  `usage` whose `prompt_tokens` is 2, and that the same body with `"reasoning_content":"plan"` added
  to its assistant entry answers `OK` with that same `usage` whole and the same `choices`, the two ids
  of the prompt being `a` and `b` alone; and that the same body with `"reasoning_content":5` in place
  of that text answers the same `OK`, `usage` and `choices`, the member's kind being unread rather
  than refused.

## Open questions

- None.

## Amendments

- `18/FR-003`: the answered message's members become `role`, `content` and, for a turn whose reply
  head holds an open thinking pair, `reasoning_content`, that `content` being the reply's text after
  that block, or the empty text when its closing marker never arrived; and that answer turns on two
  texts rather than one, the generated reply text and the prompt the request rendered, which decides
  whether a pair stands open at that reply's head.
- `19/FR-004`: a turn's events carry, before their first `delta.content`, the one
  `delta.reasoning_content` event FR-007 or FR-008 gives, and the turn may hold a reply's head back
  past the step that released it, as `50/FR-005` already lets it hold a call marker's; that turn is
  shaped by the prompt it rendered as much as by the text it sampled, since FR-006, FR-007 and FR-008
  read whether that prompt left a pair open; and a turn whose pair stands open nowhere still writes one
  content event per release.
- `49/FR-001`: the scan runs on the reply text after this spec's head block is cut, in that order, so
  a call block standing inside a head reasoning block answers reasoning and not a call.
- `49/FR-003`: its `that reply holds one or more calls` reading reaches only the text this spec's cut
  left, so a complete call block standing inside a head block that ran to the end of the reply answers
  no call and that reply answers this spec's FR-003.
- `49/FR-004`: its `holds no call block` reading and the plain body it gives reach only the text this
  spec's cut left, so a reply whose head holds an open pair is answered by this spec's FR-002 or FR-003
  and never by this body, whatever that reply's text holds.
- `50/FR-001`, `50/FR-005`, `50/FR-006`, `50/FR-007` and `50/FR-009`: each read of the text a streamed
  turn has not yet sent runs after this spec's FR-006 hold, in that order, so while a head has not
  settled no call marker, call marker's prefix, complete call block or unreadable call block standing
  inside that held text answers a call event, a call hold of its own or the `error` ending and dropped
  turn those requirements give, and no `delta.content` event carries a byte of that held text, and that
  text goes out as this spec's FR-007, FR-008 or FR-010 gives it, so a call block inside an unsettled
  head reaches the same reading the finished body gives it.
- `50/FR-007`: the text it releases is the text left once a head block this spec's FR-007 has sent,
  so a reply holding both sends one reasoning event ahead of its first content event.
- `50/FR-008`: its run's-end release reaches the text this spec's FR-006 held behind a partial
  opening marker as well as the text `50/FR-005` held behind a call marker's prefix, so a turn whose
  head prefix nothing completed sends that text as content and loses no byte.
