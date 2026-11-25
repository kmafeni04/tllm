-- the reasoning cases spec 51 names: the head block a finished reply answers
-- and the reasoning event a streamed turn sends. The route, reply, session and
-- pool cases live in tests/serve.test.tl and tests/serve_pool.test.tl, and all
-- three build on the fixture layer tests/serve_fixtures.tl.
local test = require("lde-test")
local support = require("tests.support")
local serve = require("tllm.serve")
local tltp = require("tltp")
local mint, mint_array, mtext, mstrings =
  support.mint, support.mint_array, support.mtext, support.mstrings
local f32 = support.f32
local type Meta = support.Meta
local fixtures = require("tests.serve_fixtures")
local meta_file, replaced, build =
  fixtures.meta_file, fixtures.replaced, fixtures.build
local t001_meta, t001_entries = fixtures.t001_meta, fixtures.t001_entries
local answered, held, spelled =
  fixtures.answered, fixtures.held, fixtures.spelled
local first_token, two_pieces = fixtures.first_token, fixtures.two_pieces

-- 51: the thinking pair, the two newlines 20/T-001 leaves behind its closer,
-- the two call blocks 49/T-003 names, and the halves of the opening marker
local OPEN = "<think>"
local CLOSE = "</think>"
local NL = "\n\n"
local HALF = OPEN:sub(1, 4)
local REST = OPEN:sub(5)
local SHORT = OPEN:sub(1, 3)
local PING = "<tool_call><function=ping></function></tool_call>"
local BROKEN = "<tool_call><function=></function></tool_call>"
local OPEN_BODY = '{"messages":[{"role":"user","content":"a"}],"max_tokens":1}'

local record AnswerMessage
  role: string
  content: string
  reasoning_content: string | nil
  tool_calls: {fixtures.ReplyCall} | nil
end

local record AnswerChoice
  index: integer
  message: AnswerMessage
  finish_reason: string | nil
end

local record AnswerBody
  choices: {AnswerChoice}
  usage: fixtures.Usage
end

local record ReasonDelta
  role: string | nil
  content: string | nil
  reasoning_content: string | nil
  tool_calls: {fixtures.StreamCall} | nil
end

local record ReasonChoice
  index: integer
  delta: ReasonDelta | nil
  finish_reason: string | nil
end

local record ReasonEvent
  choices: {ReasonChoice}
end

local OPENED: ReasonEvent = {
  choices = {{index = 0, delta = {role = "assistant"}}},
}

local function reasoned(text: string): ReasonEvent
  return {choices = {{index = 0, delta = {reasoning_content = text}}}}
end

local function said(text: string): ReasonEvent
  return {choices = {{index = 0, delta = {content = text}}}}
end

local function ended(reason: string): ReasonEvent
  return {choices = {{index = 0, finish_reason = reason}}}
end

local function call_named(index: integer, id: string, name: string): ReasonEvent
  return {
    choices = {
      {
        index = 0,
        delta = {
          tool_calls = {
            {
              index = index,
              id = id,
              type = "function",
              ["function"] = {name = name},
            },
          },
        },
      },
    },
  }
end

local function call_given(index: integer, arguments: string): ReasonEvent
  return {
    choices = {
      {
        index = 0,
        delta = {
          tool_calls = {
            {index = index, ["function"] = {arguments = arguments}},
          },
        },
      },
    },
  }
end

-- the one answered choice of a body that may hold a reasoning member
local function head_of(server: tltp.Server, body: string): AnswerChoice
  local reply = serve.answer(server, body)
  test.equal(reply.status, "OK", body)
  local decoded = tltp.json_decode(reply.body) as AnswerBody
  test.equal(#decoded.choices, 1, body)
  test.equal(decoded.choices[1].index, 0, body)
  return decoded.choices[1]
end

local function open_stream(budget: integer, id?: string): string
  local tail = id is string and ',"session_id":"' .. (id as string) .. '"}'
    or "}"
  return '{"messages":[{"role":"user","content":"a"}],"max_tokens":'
    .. tostring(budget)
    .. ',"stream":true'
    .. tail
end

-- the payloads a stream writes, with its [DONE] standing last
local function same_stream(payloads: {string}, events: {ReasonEvent})
  test.equal(#payloads, #events + 1)
  test.equal(payloads[#payloads], "[DONE]")
  for index, event in ipairs(events) do
    test.deepEqual(
      tltp.json_decode(payloads[index]) as ReasonEvent, event, payloads[index]
    )
  end
end

-- no payload of this stream may hold any of these texts
local function carries_none(payloads: {string}, needles: {string})
  for _, needle in ipairs(needles) do
    for _, payload in ipairs(payloads) do
      test.equal(payload:find(needle, 1, true), nil, payload)
    end
  end
end

-- 51's fixtures: 18's fixture with one more token, the thinking marker the
-- template writes, marked the control kind 10/FR-014 encodes as one id, and a
-- template holding no loop, so the prompt is that one id and ends inside, or
-- outside, a thinking block. `two` names the second sampled id, as two_pieces
-- does, and the new column copies id 2's so the same ids are sampled.
local function open_fixture(
  one: string, two: string | nil, marker: string, template: string
): tltp.Server
  local tokens = {spelled(one), "b", "c", marker}
  local embd = {1, 0, 3, 4, 1, 0, 1, 0}
  local out = {1, 0, 0, 1, 1, 1, 1, 1}
  local eos = 1
  if not (two is nil) then
    tokens = {spelled(one), spelled(two), "c", marker}
    embd = {1, 0, 0, 1, 0, 1, 0, 1}
    out = {0, 1, 1, 0, -1, -1, -1, -1}
    eos = 2
  end
  local meta = t001_meta()
  local changes: {Meta} = {
    mstrings("tokenizer.ggml.tokens", tokens),
    mint_array("tokenizer.ggml.token_type", {1, 1, 1, 3}),
    mtext("tokenizer.chat_template", template),
    mint("tokenizer.ggml.eos_token_id", eos),
  }
  for _, entry in ipairs(changes) do
    meta = replaced(meta, entry)
  end
  local entries = support.replaced(
    support.replaced(
      t001_entries(),
      "token_embd.weight",
      f32("token_embd.weight", {2, 4}, embd)
    ),
    "output.weight",
    f32("output.weight", {2, 4}, out)
  )
  return build(meta_file(meta, entries), "m")
end

local function first_open(reply: string): tltp.Server
  return open_fixture(reply, nil, OPEN, OPEN)
end

local function first_closed(reply: string): tltp.Server
  return open_fixture(reply, nil, CLOSE, CLOSE)
end

local function first_ws_open(reply: string): tltp.Server
  return open_fixture(reply, nil, OPEN .. NL, OPEN .. NL)
end

local function open_pieces(one: string, two: string): tltp.Server
  return open_fixture(one, two, OPEN, OPEN)
end

local function streamed_body(server: tltp.Server, body: string): {string}
  local response, stream_fn = serve.stream(server, body)
  test.equal(response.status, "OK", body)
  test.truthy(stream_fn, body)
  return fixtures.driven(server, stream_fn as tltp.StreamFn)
end

test.it(
  "a reply that opens inside a thinking block answers reasoning and answer",
  function()
    local cases: {{tltp.Server, string, string}} = {
      {first_open("plan" .. CLOSE .. NL .. "answer"), "plan", NL .. "answer"},
      {first_open(NL .. "plan" .. NL .. CLOSE .. "done"), "plan", "done"},
      {first_open(OPEN .. "plan" .. CLOSE .. "out"), "plan", "out"},
      {first_ws_open("plan" .. CLOSE .. "done"), "plan", "done"},
      {first_open(CLOSE .. "out"), "", "out"},
      {first_open("plan only"), "plan only", ""},
    }
    for _, given in ipairs(cases) do
      local choice = head_of(given[1], OPEN_BODY)
      test.equal(choice.finish_reason, nil)
      test.equal(choice.message.reasoning_content, given[2])
      test.equal(choice.message.content, given[3])
    end
  end
)

test.it(
  "reasoning ahead of a call is answered beside the call",
  function()
    local ahead = head_of(
      first_open("think" .. CLOSE .. fixtures.CALL), OPEN_BODY
    )
    test.equal(ahead.finish_reason, "tool_calls")
    test.deepEqual(
      ahead.message,
      {
        role = "assistant",
        content = "",
        reasoning_content = "think",
        tool_calls = {
          {
            id = "call_1",
            type = "function",
            ["function"] = {
              name = "get_weather",
              arguments = '{"city":"Paris"}',
            },
          },
        },
      }
    )

    local cut = head_of(
      first_open("a" .. PING .. "b" .. CLOSE .. "c"), OPEN_BODY
    )
    test.equal(cut.finish_reason, nil)
    test.equal(cut.message.content, "c")
    test.equal(cut.message.reasoning_content, "a" .. PING .. "b")
    test.equal(cut.message.tool_calls, nil)

    local second = head_of(
      first_open("plan" .. CLOSE .. "answer" .. OPEN .. "more" .. CLOSE),
      OPEN_BODY
    )
    test.equal(second.message.reasoning_content, "plan")
    test.equal(second.message.content, "answer" .. OPEN .. "more" .. CLOSE)

    local later = head_of(
      first_token("Hi " .. OPEN .. "mid" .. CLOSE .. "done"), fixtures.CALL_BODY
    )
    test.equal(later.message.content, "Hi " .. OPEN .. "mid" .. CLOSE .. "done")
    test.equal(later.message.reasoning_content, nil)

    local closed = head_of(first_closed("plan" .. CLOSE .. "done"), OPEN_BODY)
    test.equal(closed.message.content, "plan" .. CLOSE .. "done")
    test.equal(closed.message.reasoning_content, nil)

    local broken = head_of(first_open("plan" .. BROKEN), OPEN_BODY)
    test.equal(broken.message.reasoning_content, "plan" .. BROKEN)
    test.equal(broken.message.content, "")

    local whole = head_of(first_open("plan " .. PING), OPEN_BODY)
    test.equal(whole.message.reasoning_content, "plan " .. PING)
    test.equal(whole.message.content, "")
    test.equal(whole.message.tool_calls, nil)
    test.equal(whole.finish_reason, nil)

    local plain = head_of(
      build(meta_file(t001_meta(), t001_entries()), "m"), fixtures.A_BODY
    )
    test.equal(plain.message.content, string.rep("a", 256))
    test.equal(plain.message.reasoning_content, nil)
  end
)

test.it(
  "a streamed reply sends its reasoning before its content",
  function()
    local pieces = streamed_body(
      open_pieces(NL .. "plan", CLOSE .. NL .. "answer"), open_stream(2)
    )
    same_stream(
      pieces, {OPENED, reasoned("plan"), said(NL .. "answer"), ended("length")}
    )
    carries_none(pieces, {"think"})

    local written = streamed_body(
      two_pieces(OPEN .. "plan" .. CLOSE, "out"),
      '{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}'
    )
    same_stream(
      written, {OPENED, reasoned("plan"), said("out"), ended("length")}
    )

    local called = streamed_body(
      open_pieces("plan" .. CLOSE, fixtures.CALL), open_stream(2)
    )
    same_stream(
      called,
      {
        OPENED,
        reasoned("plan"),
        said("Hi"),
        call_named(0, "call_1", "get_weather"),
        call_given(0, '{"city":"Paris"}'),
        said("done"),
        ended("tool_calls"),
      }
    )
    carries_none(called, {"<tool_call>", "<parameter="})
  end
)

test.it(
  "a reply cut inside its block releases that block at the end",
  function()
    same_stream(
      streamed_body(first_open("partial"), open_stream(1)),
      {OPENED, reasoned("partial"), ended("length")}
    )

    local budget = '{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}'
    local joined = streamed_body(
      two_pieces(NL .. SHORT, "ink>" .. "plan" .. CLOSE .. "out"), budget
    )
    same_stream(
      joined, {OPENED, reasoned("plan"), said("out"), ended("length")}
    )
    carries_none(joined, {"think"})

    same_stream(
      streamed_body(two_pieces(HALF, "plain"), budget),
      {OPENED, said(HALF .. "plain"), ended("length")}
    )
    same_stream(
      streamed_body(two_pieces(HALF, REST .. CLOSE), budget),
      {OPENED, reasoned(""), ended("length")}
    )

    local cut = first_open("plan" .. BROKEN)
    local broken = streamed_body(cut, open_stream(1, "s"))
    same_stream(broken, {OPENED, reasoned("plan" .. BROKEN), ended("length")})
    carries_none(broken, {'"content"', "tool_calls"})
    held(cut, "s")

    same_stream(
      streamed_body(first_open("plan " .. PING), open_stream(1)),
      {OPENED, reasoned("plan " .. PING), ended("length")}
    )

    local twice = streamed_body(
      two_pieces(OPEN .. "one" .. CLOSE .. NL .. "x", "y" .. CLOSE .. "z"),
      budget
    )
    same_stream(
      twice,
      {
        OPENED,
        reasoned("one"),
        said(NL .. "x" .. "y" .. CLOSE .. "z"),
        ended("length"),
      }
    )
    carries_none(twice, {OPEN})

    same_stream(
      streamed_body(first_ws_open("plan"), open_stream(1)),
      {OPENED, reasoned("plan"), ended("length")}
    )
  end
)

test.it(
  "a reply that opens with its answer streams that answer whole",
  function()
    local one = '{"messages":[{"role":"user","content":"c"}],"max_tokens":1,"stream":true}'
    local two = '{"messages":[{"role":"user","content":"c"}],"max_tokens":2,"stream":true}'

    same_stream(
      streamed_body(first_token("answer"), one),
      {OPENED, said("answer"), ended("length")}
    )
    same_stream(
      streamed_body(two_pieces("a", "b"), two),
      {OPENED, said("a"), said("b"), ended("length")}
    )
    same_stream(
      streamed_body(two_pieces("answer", OPEN .. "more" .. CLOSE), two),
      {OPENED, said("answer"), said(OPEN .. "more" .. CLOSE), ended("length")}
    )
    same_stream(
      streamed_body(first_token(CLOSE .. "b"), one),
      {OPENED, said(CLOSE .. "b"), ended("length")}
    )
    same_stream(
      streamed_body(two_pieces("answer", OPEN .. "more"), two),
      {OPENED, said("answer"), said(OPEN .. "more"), ended("length")}
    )
  end
)

test.it(
  "a replayed reasoning member is accepted and reaches no prompt",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local plain = '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"b"}]}'
    local base = answered(server, plain)
    test.equal(base.usage.prompt_tokens, 2)

    for _, member in ipairs(
      {',"reasoning_content":"plan"', ',"reasoning_content":5'}
    ) do
      local given = (
        plain:gsub('content":"b"}', 'content":"b"' .. member .. "}")
      )
      local same = answered(server, given)
      test.deepEqual(same.usage, base.usage, given)
      test.deepEqual(same.choices, base.choices, given)
    end
  end
)
