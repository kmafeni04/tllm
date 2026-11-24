-- the serve route, reply and session cases. The transport, worker and pool
-- cases live in tests/serve_pool.test.tl, and the fixture layer both files
-- build on is tests/serve_fixtures.tl.
local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local process = require("process")
local support = require("tests.support")
local cli = require("tllm.cli")
local compute = require("tllm.compute")
local engine = require("tllm.compute.engine")
local cores = require("tllm.cores")
local generate = require("tllm.generate")
local serve = require("tllm.serve")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local tltp = require("tltp")
local u32, u64, mint, mtext, mstrings, without_meta =
  support.u32,
  support.u64,
  support.mint,
  support.mtext,
  support.mstrings,
  support.without_meta
local defs = require("tllm.compute.types")
local type Loaded = defs.Loaded
local type Meta = support.Meta
local type Threads = defs.Threads
local fixtures = require("tests.serve_fixtures")
local type HealthBody = fixtures.HealthBody
local type ErrorBody = fixtures.ErrorBody
local type ChatReply = fixtures.ChatReply
local type StreamDelta = fixtures.StreamDelta
local type StreamEvent = fixtures.StreamEvent
local type Watcher = fixtures.Watcher
local ROOT, fixture, meta_bytes, meta_file =
  fixtures.ROOT, fixtures.fixture, fixtures.meta_bytes, fixtures.meta_file
local replaced, t001_meta, t001_entries, DEVICE_LINE =
  fixtures.replaced,
  fixtures.t001_meta,
  fixtures.t001_entries,
  fixtures.DEVICE_LINE
local answered, build, temp_dir, begun =
  fixtures.answered, fixtures.build, fixtures.temp_dir, fixtures.begun
local events_of, ROLE_EVENT, token_event, finish_event =
  fixtures.events_of,
  fixtures.ROLE_EVENT,
  fixtures.token_event,
  fixtures.finish_event
local FIRST_TURN, SECOND_TURN, unreadable_fixture, streaming =
  fixtures.FIRST_TURN,
  fixtures.SECOND_TURN,
  fixtures.unreadable_fixture,
  fixtures.streaming
local held, named, streamed = fixtures.held, fixtures.named, fixtures.streamed
local POOL_BUDGET, workers_of = fixtures.POOL_BUDGET, fixtures.workers_of

local function without_prefix(meta: {Meta}, prefix: string): {Meta}
  local result: {Meta} = {}
  for _, entry in ipairs(meta) do
    if entry.name:sub(1, #prefix) ~= prefix then
      result[#result + 1] = entry
    end
  end
  return result
end

local A_BODY = '{"messages":[{"role":"user","content":"a"}]}'

local function body_with(name: string, value: string): string
  return '{"messages":[{"role":"user","content":"a"}],"'
    .. name
    .. '":'
    .. value
    .. "}"
end

local function refused(server: tltp.Server, body: string): ErrorBody
  local reply = serve.answer(server, body)
  test.equal(reply.status, "InternalServerError", body)
  local decoded = tltp.json_decode(reply.body) as ErrorBody
  test.equal(decoded.error.type, "internal_error", body)
  return decoded
end

local function rejected(server: tltp.Server, body: string): ErrorBody
  local reply = serve.answer(server, body)
  test.equal(reply.status, "BadRequest", body)
  local decoded = tltp.json_decode(reply.body) as ErrorBody
  test.equal(decoded.error.type, "invalid_request_error", body)
  return decoded
end

local record Run
  code: integer
  err: {string}
  endpoint: serve.Endpoint | nil
  starts: integer
end

local function run(args: {string}, environment: {string: string}): Run
  local result: Run = {code = 0, err = {}, starts = 0}
  local function stub(
    endpoint: serve.Endpoint, _write_err: function(string)
  ): integer
    result.endpoint = endpoint
    result.starts = result.starts + 1
    return 0
  end
  result.code = cli.run(
    {
      args = args,
      write_out = function(_text: string) end,
      write_err = function(line: string)
        result.err[#result.err + 1] = line
      end,
      start = stub,
      environment = environment,
    }
  )
  return result
end

test.it(
  "a gguf is mapped at build and a bad file is refused",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local server = build(file_path, "m")
    test.equal(server.config.port, 9090)
    test.equal(server.config.bind_host, "127.0.0.1")

    local model = serve.model_of(server) as serve.Model
    test.truthy(model)
    test.equal(model.path, file_path)
    test.equal(model.name, "m")
    test.truthy(model.handle.tensors["token_embd.weight"])

    local unrelated = tltp.Server.new(
      {port = 9091, bind_host = "127.0.0.1", logging = false}
    )
    test.equal(serve.model_of(unrelated), nil)

    local response = server:mock_request("/health")
    test.equal(response.status, "OK")
    test.equal(response.content_type, "application/json")
    test.deepEqual(
      tltp.json_decode(response.body) as HealthBody,
      {status = "ok", model = "m"}
    )

    local maps = fs.read("/proc/self/maps") as string
    test.includes(maps, file_path)

    -- a server that runs the model itself gives the mapping back on stop
    serve.stop(server)
    test.equal(serve.model_of(server), nil)
    local after = fs.read("/proc/self/maps") as string
    test.equal(after:find(file_path, 1, true), nil)

    local not_gguf = fixture("GGML" .. u32(3) .. u64(0) .. u64(0))
    local bad, reason = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = not_gguf,
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.equal(bad, nil)
    test.includes(reason as string, not_gguf)

    local missing = ROOT .. "/missing.gguf"
    local absent, absent_reason = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = missing,
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.equal(absent, nil)
    test.includes(absent_reason as string, missing)
  end
)

test.it(
  "unrouted paths answer json 404",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    for _, path in ipairs({"/nope", "/nope/"}) do
      local response = server:mock_request(path)
      test.equal(response.status, "NotFound", path)
      test.equal(response.content_type, "application/json", path)
      test.deepEqual(
        tltp.json_decode(response.body) as ErrorBody,
        {error = {message = "not found", type = "not_found"}},
        path
      )
    end
  end
)

test.it(
  "wrong method answers json 404",
  function()
    local response = build(
      meta_file(t001_meta(), t001_entries()), "m"
    ):mock_request("/health", {method = "post"})
    test.equal(response.status, "NotFound")
    test.deepEqual(
      tltp.json_decode(response.body) as ErrorBody,
      {error = {message = "not found", type = "not_found"}}
    )
  end
)

test.it(
  "route error answers json 500",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    server:get("/boom", function(): tltp.Response error("boom") end)
    local response = server:mock_request("/boom")
    test.equal(response.status, "InternalServerError")
    test.equal(response.content_type, "application/json")
    test.deepEqual(
      tltp.json_decode(response.body) as ErrorBody,
      {error = {message = "internal error", type = "internal_error"}}
    )
  end
)

test.it(
  "serve resolves one model and names it",
  function()
    local named_dir = temp_dir()
    local named_path = named_dir .. "/configured.gguf"
    fs.write(named_path, "model")
    local configured = named_dir .. "/config.json"
    fs.write(configured, '{"models":{"m":{"path":"' .. named_path .. '"}}}')

    local served = run({"serve", "m", "--config", configured}, {})
    test.equal(served.code, 0)
    test.equal(served.starts, 1)
    local named_endpoint = served.endpoint as serve.Endpoint
    test.equal(named_endpoint.name, "m")
    test.equal(named_endpoint.path, named_path)

    local path_endpoint_dir = temp_dir()
    local spare = path_endpoint_dir .. "/spare.gguf"
    fs.write(spare, "model")
    local path_word = run({"serve", spare}, {})
    test.equal(path_word.code, 0)
    test.equal(path_word.starts, 1)
    local path_endpoint = path_word.endpoint as serve.Endpoint
    test.equal(path_endpoint.name, spare)
    test.equal(path_endpoint.path, spare)

    local lone_dir = temp_dir()
    local lone_path = lone_dir .. "/lone.gguf"
    fs.write(lone_path, "model")
    local lone_config = lone_dir .. "/config.json"
    fs.write(lone_config, '{"models":{"solo":{"path":"' .. lone_path .. '"}}}')
    local lone = run({"serve", "--config", lone_config}, {})
    test.equal(lone.code, 0)
    test.equal(lone.starts, 1)
    local lone_endpoint = lone.endpoint as serve.Endpoint
    test.equal(lone_endpoint.name, "solo")
    test.equal(lone_endpoint.path, lone_path)

    local many_dir = temp_dir()
    local zeta_path = many_dir .. "/zeta.gguf"
    local alpha_path = many_dir .. "/alpha.gguf"
    fs.write(zeta_path, "model")
    fs.write(alpha_path, "model")
    local many_config = many_dir .. "/config.json"
    fs.write(
      many_config,
      '{"models":{"Zeta":{"path":"'
        .. zeta_path
        .. '"},"alpha":{"path":"'
        .. alpha_path
        .. '"}}}'
    )
    local many = run({"serve", "--config", many_config}, {})
    test.equal(many.code, 1)
    test.equal(many.starts, 0)
    test.equal(#many.err, 1)
    test.includes(many.err[1], "Zeta")
    test.includes(many.err[1], "alpha")
    test.less(
      many.err[1]:find("Zeta") as integer, many.err[1]:find("alpha") as integer
    )
  end
)

test.it(
  "a model that cannot be opened is refused",
  function()
    local file_path = fixture("GGML" .. u32(3) .. u64(0) .. u64(0))
    local lines: {string} = {}
    local code = serve.start(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      },
      function(line: string) lines[#lines + 1] = line end
    )
    test.equal(code, 1)
    test.equal(#lines, 1)
    test.includes(lines[1], file_path)
    test.equal(lines[1]:sub(-1), "\n")
    test.equal(lines[1]:find("\n"), #lines[1])
  end
)

test.it(
  "a chat request answers the model's text with usage counts",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local full = answered(server, A_BODY)
    test.equal(#full.choices, 1)
    test.equal(full.choices[1].index, 0)
    test.deepEqual(
      full.choices[1].message,
      {role = "assistant", content = string.rep("a", 256)}
    )
    test.deepEqual(
      full.usage,
      {
        prompt_tokens = 1,
        completion_tokens = 256,
        total_tokens = 257,
        prompt_tokens_details = {cached_tokens = 0},
      }
    )

    local short = answered(server, body_with("max_tokens", "2"))
    test.equal(short.choices[1].message.content, "aa")
    test.equal(short.usage.completion_tokens, 2)
    test.equal(short.usage.total_tokens, 3)

    local stopped = build(
      meta_file(
        replaced(t001_meta(), mint("tokenizer.ggml.eos_token_id", 0)),
        t001_entries()
      ),
      "m"
    )
    local empty = answered(stopped, A_BODY)
    test.equal(empty.choices[1].message.content, "")
    test.equal(empty.usage.completion_tokens, 0)
    test.equal(empty.usage.total_tokens, 1)
  end
)

test.it(
  "text parts answer as the joined string",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local one = answered(
      server,
      '{"messages":[{"role":"user","content":[{"type":"text","text":"a"}]}]}'
    )
    test.deepEqual(
      one.choices[1].message,
      {role = "assistant", content = string.rep("a", 256)}
    )
    test.deepEqual(
      one.usage,
      {
        prompt_tokens = 1,
        completion_tokens = 256,
        total_tokens = 257,
        prompt_tokens_details = {cached_tokens = 0},
      }
    )

    local joined = answered(
      server,
      '{"messages":[{"role":"user","content":[{"type":"text","text":"b"},{"type":"text","text":"a"}]}]}'
    )
    test.deepEqual(
      joined.choices[1].message,
      {role = "assistant", content = string.rep("a", 256)}
    )
    test.deepEqual(
      joined.usage,
      {
        prompt_tokens = 2,
        completion_tokens = 256,
        total_tokens = 258,
        prompt_tokens_details = {cached_tokens = 0},
      }
    )

    local reversed = answered(
      server,
      '{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}'
    )
    test.equal(reversed.choices[1].message.content:sub(1, 3), "caa")

    local counted = build(
      meta_file(
        replaced(
          t001_meta(),
          mtext(
            "tokenizer.chat_template",
            "{% for message in messages %}{% if message.role == 'system' %}a{% else %}b{% endif %}{% endfor %}"
          )
        ),
        t001_entries()
      ),
      "m"
    )
    local parts = answered(
      counted,
      '{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"},{"type":"text","text":"c"}]}]}'
    )
    local whole = answered(
      counted, '{"messages":[{"role":"user","content":"abc"}]}'
    )
    test.equal(parts.usage.prompt_tokens, 1)
    test.equal(parts.usage.prompt_tokens, whole.usage.prompt_tokens)
  end
)

test.it(
  "the sampling numbers in the body steer the reply",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    test.equal(
      answered(
        server,
        '{"messages":[{"role":"user","content":"a"}],"temperature":1,"top_k":2,"seed":1000,"max_tokens":3}'
      ).choices[1].message.content,
      "ccc"
    )
    test.equal(
      answered(
        server,
        '{"messages":[{"role":"user","content":"a"}],"temperature":1,"top_k":2,"max_tokens":3}'
      ).choices[1].message.content,
      "aac"
    )

    local ten = answered(server, body_with("max_tokens", "10"))
    test.equal(ten.choices[1].message.content, string.rep("a", 10))
    test.equal(ten.usage.completion_tokens, 10)
    test.equal(ten.usage.total_tokens, 11)

    test.equal(
      answered(
        server,
        '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"model":"m","stream":false}'
      ).choices[1].message.content,
      "aa"
    )

    answered(server, '{"messages":[{"role":"user","content":"aaaaa"}]}')
  end
)

test.it(
  "a body the route cannot use is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local cases: {{string, {string}}} = {
      {"", {"body"}},
      {"nope", {"body"}},
      {"[]", {"body"}},
      {"{}", {"messages"}},
      {'{"messages":5}', {"messages"}},
      {'{"messages":[]}', {"messages"}},
      {'{"messages":[{"role":"user"}]}', {"messages", "1"}},
      {
        '{"messages":[{"role":"user","content":["x"]}]}',
        {"messages", "1"},
      },
      {body_with("temperature", '"hot"'), {"temperature"}},
      {body_with("temperature", "-1"), {"temperature"}},
      {body_with("top_k", "2.5"), {"top_k"}},
      {body_with("max_tokens", "0"), {"max_tokens"}},
      {body_with("seed", "1.5"), {"seed"}},
      {body_with("session_id", "5"), {"session_id"}},
      {'{"messages":[],"temperature":-1}', {"messages"}},
      {
        '{"messages":[{"role":"user","content":"a"}],"temperature":-1,"top_k":0}',
        {"temperature"},
      },
      {
        '{"messages":[{"role":"user","content":"aaaaaa"}]}',
        {"messages", "max_tokens", "6", "256", "260"},
      },
    }

    for _, given in ipairs(cases) do
      local decoded = rejected(server, given[1])
      for _, word in ipairs(given[2]) do
        test.includes(decoded.error.message, word, given[1])
      end
    end
  end
)

test.it(
  "a part or message the engine cannot read is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local cases: {{string, {string}}} = {
      {
        '{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"text","text":"b"},{"type":"image_url","image_url":{"url":"x"}}]}]}',
        {"messages", "2", "2", "image_url"},
      },
      {
        '{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"type":"text","text":"b"},{"type":"text"}]}]}',
        {"messages", "2", "2"},
      },
      {
        '{"messages":[{"role":"user","content":"a"},{"role":"user","content":[{"text":"a"},{"type":"text","text":"b"}]}]}',
        {"messages", "2", "1"},
      },
      {
        '{"messages":[{"role":"user","content":"a"},{"role":"user","content":[]}]}',
        {"messages", "2"},
      },
    }

    for _, given in ipairs(cases) do
      local decoded = rejected(server, given[1])
      for _, word in ipairs(given[2]) do
        test.includes(decoded.error.message, word, given[1])
      end
    end
  end
)

test.it(
  "the route is wired and engine failures answer json 500",
  function()
    local route = "/v1/chat/completions"
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local unmatched = server:mock_request(route)
    test.equal(unmatched.status, "NotFound")
    test.deepEqual(
      tltp.json_decode(unmatched.body) as ErrorBody,
      {error = {message = "not found", type = "not_found"}}
    )

    local no_body = server:mock_request(route, {method = "post"})
    test.equal(no_body.status, "BadRequest")
    test.equal(no_body.content_type, "application/json")
    test.includes(
      (tltp.json_decode(no_body.body) as ErrorBody).error.message, "body"
    )

    local untemplated = build(
      meta_file(
        without_meta(t001_meta(), "tokenizer.chat_template"), t001_entries()
      ),
      "m"
    )
    test.includes(
      refused(untemplated, A_BODY).error.message, "tokenizer.chat_template"
    )

    local raw_path = meta_file(
      replaced(t001_meta(), mtext("tokenizer.chat_template", "\255")),
      t001_entries()
    )
    local tokens = tokenizer.load(raw_path) as tokenizer.Tokenizer
    local _, encode_line = tokens:encode("\255")
    test.equal(refused(build(raw_path, "m"), A_BODY).error.message, encode_line)

    local blank = build(
      meta_file(
        replaced(
          t001_meta(),
          mtext(
            "tokenizer.chat_template",
            "{% for message in messages %}{% endfor %}"
          )
        ),
        t001_entries()
      ),
      "m"
    )
    test.includes(refused(blank, A_BODY).error.message, "prompt")

    local unmeasured, unmeasured_line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = meta_file(
          without_meta(t001_meta(), "qwen35moe.embedding_length"),
          t001_entries()
        ),
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.equal(unmeasured, nil)
    test.includes(unmeasured_line as string, "model")
    test.includes(unmeasured_line as string, "embedding_length")
  end
)

test.it(
  "build loads the tokenizer and holds it",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local model = serve.model_of(server) as serve.Model
    test.equal(model.tokenizer.size, 3)

    local no_tokens = meta_file(
      without_prefix(t001_meta(), "tokenizer."), t001_entries()
    )
    local refused_build, line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = no_tokens,
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.equal(refused_build, nil)
    test.includes(line as string, "tokenizer.ggml.model")
  end
)

local STREAM_BODY = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'

local function resumed(watcher: Watcher): boolean
  local ok, result = coroutine.resume(watcher.co)
  test.equal(ok, true)
  return result as boolean
end

local function driven(server: tltp.Server, stream_fn: tltp.StreamFn): {string}
  local watcher = begun(stream_fn, nil, "A")
  while serve.step(server) > 0 do
  end
  return watcher.payloads
end

local function stream_refused(
  server: tltp.Server, body: string
): tltp.Response, ErrorBody
  local response, stream_fn = serve.stream(server, body)
  test.equal(stream_fn, nil, body)
  test.equal(response.content_type, "application/json", body)
  return response, tltp.json_decode(response.body) as ErrorBody
end

test.it(
  "a streamed reply arrives token by token and ends with done",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local response, stream_fn = serve.stream(server, STREAM_BODY)
    test.equal(response.status, "OK")
    test.equal(response.content_type, "text/event-stream")
    test.truthy(stream_fn)

    local watcher = begun(stream_fn as tltp.StreamFn, nil, "A")
    test.deepEqual(events_of(watcher.payloads), {ROLE_EVENT})
    while serve.step(server) > 0 do
    end
    local payloads = watcher.payloads
    test.equal(#payloads, 5)
    local events = events_of(payloads)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], token_event("a"))
    test.deepEqual(events[3], token_event("a"))
    test.deepEqual(events[4], finish_event("length"))
    test.equal(payloads[5], "[DONE]")

    local stopped = build(
      meta_file(
        replaced(t001_meta(), mint("tokenizer.ggml.eos_token_id", 0)),
        t001_entries()
      ),
      "m"
    )
    local _, stopped_run = serve.stream(stopped, STREAM_BODY)
    local stopped_payloads = driven(stopped, stopped_run as tltp.StreamFn)
    test.equal(#stopped_payloads, 3)
    local stopped_events = events_of(stopped_payloads)
    test.deepEqual(stopped_events[1], ROLE_EVENT)
    test.deepEqual(stopped_events[2], finish_event("stop"))
    test.equal(stopped_payloads[3], "[DONE]")
  end
)

test.it(
  "a body that does not ask for a stream answers json",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local plain = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2}'
    local response, stream_fn = serve.stream(server, plain)
    test.equal(stream_fn, nil)
    test.equal(response.status, "OK")
    test.equal(response.content_type, "application/json")
    test.equal(response.body, serve.answer(server, plain).body)

    local off = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":false}'
    local off_response, off_run = serve.stream(server, off)
    test.equal(off_run, nil)
    test.equal(off_response.content_type, "application/json")
    test.equal(off_response.body, response.body)
  end
)

test.it(
  "a stream request the route cannot use answers json and writes no event",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local cases: {{string, tltp.Status, string, {string}}} = {
      {
        '{"messages":[],"stream":true}',
        "BadRequest",
        "invalid_request_error",
        {"messages"},
      },
      {
        '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":"yes"}',
        "BadRequest",
        "invalid_request_error",
        {"stream"},
      },
      {
        '{"messages":[{"role":"user","content":"aaaaaa"}],"stream":true}',
        "BadRequest",
        "invalid_request_error",
        {"messages", "max_tokens", "6", "256", "260"},
      },
    }

    for _, given in ipairs(cases) do
      local response, decoded = stream_refused(server, given[1])
      test.equal(response.status, given[2], given[1])
      test.equal(decoded.error.type, given[3], given[1])
      for _, word in ipairs(given[4]) do
        test.includes(decoded.error.message, word, given[1])
      end
    end

    local untemplated = build(
      meta_file(
        without_meta(t001_meta(), "tokenizer.chat_template"), t001_entries()
      ),
      "m"
    )
    local untemplated_response, untemplated_error = stream_refused(
      untemplated, STREAM_BODY
    )
    test.equal(untemplated_response.status, "InternalServerError")
    test.equal(untemplated_error.error.type, "internal_error")
    test.includes(untemplated_error.error.message, "tokenizer.chat_template")

    local raw_path = meta_file(
      replaced(t001_meta(), mtext("tokenizer.chat_template", "\255")),
      t001_entries()
    )
    local tokens = tokenizer.load(raw_path) as tokenizer.Tokenizer
    local _, encode_line = tokens:encode("\255")
    local raw_response, raw_error = stream_refused(
      build(raw_path, "m"), STREAM_BODY
    )
    test.equal(raw_response.status, "InternalServerError")
    test.equal(raw_error.error.message, encode_line)

    local unmeasured, unmeasured_line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = meta_file(
          without_meta(t001_meta(), "qwen35moe.embedding_length"),
          t001_entries()
        ),
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.equal(unmeasured, nil)
    test.includes(unmeasured_line as string, "model")
    test.includes(unmeasured_line as string, "embedding_length")
  end
)

test.it(
  "a stream that cannot be generated or encoded ends without done",
  function()
    local body = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"temperature":1,"top_k":2,"seed":1000}'

    local unreadable_path = meta_file(
      replaced(
        t001_meta(), mstrings("tokenizer.ggml.tokens", {"a", "b", "\255"})
      ),
      t001_entries()
    )
    local unreadable = build(unreadable_path, "m")
    local _, unreadable_run = serve.stream(unreadable, body)
    local unreadable_payloads = driven(
      unreadable, unreadable_run as tltp.StreamFn
    )
    test.equal(#unreadable_payloads, 1)
    test.deepEqual(events_of(unreadable_payloads)[1], ROLE_EVENT)

    local pieces = tokenizer.load(unreadable_path) as tokenizer.Tokenizer
    local _, decode_line = pieces:decode({2})
    test.equal(refused(unreadable, body).error.message, decode_line)

    local broken = build(
      meta_file(
        replaced(
          t001_meta(), mstrings("tokenizer.ggml.tokens", {"a", "b", "\195\164"})
        ),
        t001_entries()
      ),
      "m"
    )
    local _, broken_run = serve.stream(broken, body)
    local broken_payloads = driven(broken, broken_run as tltp.StreamFn)
    test.equal(#broken_payloads, 1)
    test.deepEqual(events_of(broken_payloads)[1], ROLE_EVENT)
  end
)

test.it(
  "a second turn reads only what it has not read",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 4)

    local first = answered(server, FIRST_TURN)
    test.equal(first.choices[1].message.content, "aa")
    test.deepEqual(
      first.usage,
      {
        prompt_tokens = 1,
        completion_tokens = 2,
        total_tokens = 3,
        prompt_tokens_details = {cached_tokens = 0},
      }
    )
    test.equal(held(server, "s").length, 2)
    test.equal(held(server, "s").positions, 4)

    local second = answered(server, SECOND_TURN)
    test.equal(second.choices[1].message.content, "aa")
    test.equal(second.usage.prompt_tokens_details.cached_tokens, 2)
    test.equal(held(server, "s").length, 4)

    local other = answered(server, named(SECOND_TURN, "t"))
    test.equal(other.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "t").length, 4)

    local loose = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2}'
    local plain = answered(server, loose)
    test.equal(plain.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 4)
    test.equal(held(server, "t").length, 4)

    local wide = answered(
      server,
      '{"messages":[{"role":"user","content":"'
        .. string.rep("a", 130)
        .. '"}],"max_tokens":2,"session_id":"w"}'
    )
    test.equal(wide.usage.completion_tokens, 2)
    test.equal(held(server, "w").positions, 260)

    local seeded = answered(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"temperature":1,"seed":2,"session_id":"v"}'
    )
    test.equal(seeded.usage.completion_tokens, 1)
    test.equal(held(server, "v").length, 2)

    local followed = answered(server, named(SECOND_TURN, "v"))
    test.equal(followed.usage.prompt_tokens_details.cached_tokens, 2)
  end
)

test.it(
  "a turn that does not continue the stored context starts over",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 4)
    answered(server, FIRST_TURN)
    answered(server, SECOND_TURN)

    local five = answered(
      server,
      '{"messages":[{"role":"user","content":"aaaaa"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    test.equal(five.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 6)

    local pair = answered(
      server,
      '{"messages":[{"role":"user","content":"a"},'
        .. '{"role":"assistant","content":"a"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    test.equal(pair.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 3)

    local three = answered(
      server,
      '{"messages":[{"role":"user","content":"a"},'
        .. '{"role":"assistant","content":"a"},'
        .. '{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    test.equal(three.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 4)

    answered(server, named(FIRST_TURN, "x"))
    local mismatched = answered(
      server,
      '{"messages":[{"role":"user","content":"baa"}],"max_tokens":2,'
        .. '"session_id":"x"}'
    )
    test.equal(mismatched.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "x").length, 4)
    test.equal(held(server, "x").positions, 8)

    local fresh = answered(server, named(FIRST_TURN, "u"))
    test.equal(fresh.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "u").length, 2)
  end
)

test.it(
  "the server keeps only the configured number of conversations",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 2)

    answered(server, named(FIRST_TURN, "s1"))
    answered(server, named(FIRST_TURN, "s2"))
    local reuse = answered(server, named(SECOND_TURN, "s1"))
    test.equal(reuse.usage.prompt_tokens_details.cached_tokens, 2)
    test.truthy(serve.session_of(server, "s1"))
    test.truthy(serve.session_of(server, "s2"))

    answered(server, named(SECOND_TURN, "s3"))
    test.truthy(serve.session_of(server, "s1"))
    test.truthy(serve.session_of(server, "s3"))
    test.equal(serve.session_of(server, "s2"), nil)

    local empty = serve.answer(server, '{"messages":[],"session_id":"s1"}')
    test.equal(empty.status, "BadRequest")
    test.equal(held(server, "s1").length, 4)
    test.equal(held(server, "s3").length, 4)
    test.equal(serve.session_of(server, "s9"), nil)

    answered(server, named(SECOND_TURN, "s4"))
    test.truthy(serve.session_of(server, "s3"))
    test.truthy(serve.session_of(server, "s4"))
    test.equal(serve.session_of(server, "s1"), nil)
  end
)

local FAILING_TURN = '{"messages":[{"role":"user","content":"a"},'
  .. '{"role":"assistant","content":"a"},'
  .. '{"role":"user","content":"a"}],"max_tokens":2,'
  .. '"temperature":1,"top_k":2,"seed":1000,"session_id":"s"}'

test.it(
  "a failed turn leaves no context",
  function()
    local file_path = unreadable_fixture()
    local server = build(file_path, "m", 4)
    answered(server, FIRST_TURN)
    test.equal(held(server, "s").length, 2)

    local pieces = tokenizer.load(file_path) as tokenizer.Tokenizer
    local _, decode_line = pieces:decode({2})
    test.equal(refused(server, FAILING_TURN).error.message, decode_line)
    test.equal(serve.session_of(server, "s"), nil)

    local fresh = build(file_path, "m", 4)
    answered(fresh, FIRST_TURN)
    local _, failing_run = serve.stream(
      fresh, (FAILING_TURN:gsub("}$", ',"stream":true}'))
    )
    local payloads = driven(fresh, failing_run as tltp.StreamFn)
    test.equal(#payloads, 1)
    test.deepEqual(events_of(payloads)[1], ROLE_EVENT)
    test.equal(serve.session_of(fresh, "s"), nil)
  end
)

test.it(
  "a streamed turn stores its context too",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 4)

    local stream_body = (FIRST_TURN:gsub("}$", ',"stream":true}'))
    local _, streamed_run = serve.stream(server, stream_body)
    test.truthy(streamed_run)
    driven(server, streamed_run as tltp.StreamFn)
    test.equal(held(server, "s").length, 2)

    local next_turn = answered(
      server, (SECOND_TURN:gsub("}$", ',"stream":false}'))
    )
    test.equal(next_turn.usage.prompt_tokens_details.cached_tokens, 2)
  end
)

test.it(
  "the configured type reaches the model the server builds",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())

    local broken, broken_line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "q8_0",
      }
    )
    test.equal(broken, nil)
    test.includes(broken_line as string, "q8_0")
    test.includes(broken_line as string, "2")

    local float = serve.answer(build(file_path, "m", 4, "f32"), A_BODY)
    test.equal(float.status, "OK")
    local half = serve.answer(build(file_path, "m", 4, "f16"), A_BODY)
    test.equal(half.status, "OK")
    test.equal(half.body, float.body)
  end
)

test.it(
  "two streamed turns advance one token each in turn",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    test.equal(serve.step(server), 0)

    local order: {string} = {}
    local first_response, first_fn = serve.stream(server, streamed("s1"))
    local second_response, second_fn = serve.stream(server, streamed("s2"))
    test.equal(first_response.content_type, "text/event-stream")
    test.equal(second_response.content_type, "text/event-stream")
    test.truthy(first_fn)
    test.truthy(second_fn)

    local first = begun(first_fn as tltp.StreamFn, order, "A")
    local second = begun(second_fn as tltp.StreamFn, order, "B")
    test.equal(coroutine.status(first.co), "suspended")
    test.equal(coroutine.status(second.co), "suspended")
    test.deepEqual(events_of(first.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(second.payloads), {ROLE_EVENT})
    test.deepEqual(order, {"A", "B"})

    test.equal(serve.step(server), 2)
    test.deepEqual(order, {"A", "B", "A", "B"})

    test.equal(serve.step(server), 0)
    test.deepEqual(order, {"A", "B", "A", "B", "A", "A", "A", "B", "B", "B"})
    local events = events_of(first.payloads)
    test.equal(#first.payloads, 5)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], token_event("a"))
    test.deepEqual(events[3], token_event("a"))
    test.deepEqual(events[4], finish_event("length"))
    test.equal(first.payloads[5], "[DONE]")
    test.equal(held(server, "s1").length, 2)

    test.equal(resumed(first), true)
    test.equal(resumed(second), true)
    test.equal(coroutine.status(first.co), "dead")
    test.equal(coroutine.status(second.co), "dead")
    test.equal(#first.payloads, 5)
  end
)

local FAILING_BODY = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
  .. '"temperature":1,"top_k":2,"seed":1000,"session_id":"s"}'

test.it(
  "a turn that cannot run ends its stream and drops its context",
  function()
    local unreadable_path = unreadable_fixture()
    local unreadable = build(unreadable_path, "m")
    local _, unreadable_run = serve.stream(unreadable, streaming(FAILING_BODY))
    test.truthy(unreadable_run)
    local unreadable_watcher = begun(unreadable_run as tltp.StreamFn, nil, "A")
    test.equal(serve.step(unreadable), 0)
    test.deepEqual(events_of(unreadable_watcher.payloads), {ROLE_EVENT})
    test.equal(serve.session_of(unreadable, "s"), nil)

    local pieces = tokenizer.load(unreadable_path) as tokenizer.Tokenizer
    local _, decode_line = pieces:decode({2})
    test.equal(refused(unreadable, FAILING_BODY).error.message, decode_line)

    local broken = build(
      meta_file(
        replaced(
          t001_meta(), mstrings("tokenizer.ggml.tokens", {"a", "b", "\195\164"})
        ),
        t001_entries()
      ),
      "m"
    )
    local _, broken_run = serve.stream(broken, streaming(FAILING_BODY))
    test.truthy(broken_run)
    local broken_watcher = begun(broken_run as tltp.StreamFn, nil, "B")
    test.equal(serve.step(broken), 0)
    test.deepEqual(events_of(broken_watcher.payloads), {ROLE_EVENT})
    test.equal(serve.session_of(broken, "s"), nil)

    local untemplated = build(
      meta_file(
        replaced(
          t001_meta(),
          mtext(
            "tokenizer.chat_template",
            "{% for message in messages %}{% endfor %}"
          )
        ),
        t001_entries()
      ),
      "m"
    )
    local response, refused_run = serve.stream(untemplated, streamed("s3"))
    test.equal(refused_run, nil)
    test.equal(response.status, "InternalServerError")
    test.equal(response.content_type, "application/json")
    local blank = tltp.json_decode(response.body) as ErrorBody
    test.equal(blank.error.type, "internal_error")
    test.includes(blank.error.message, "prompt")
  end
)

test.it(
  "a pool that cannot start leaves nothing behind",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())

    local calls = 0
    local function unused(
      _name: string, _args: {string} | nil, _opts: process.Options | nil
    ): process.Child | nil, string | nil
      calls = calls + 1
      return nil, "unused"
    end
    local built, line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = fixture(bytes),
        sessions = 4,
        kv_cache = "f32",
        workers = 2,
      },
      unused
    )
    test.equal(built, nil)
    test.includes(line as string, "workers")
    test.includes(line as string, "context")
    test.equal(calls, 0)

    local no_width = meta_bytes(
      without_meta(t001_meta(), "qwen35moe.embedding_length"), t001_entries()
    )
    local pids: {integer} = {}
    local function recording(
      name: string, args: {string} | nil, opts: process.Options | nil
    ): process.Child | nil, string | nil
      local child, failure = process.spawn(name, args, opts)
      if child is process.Child then
        pids[#pids + 1] = child.pid
      end
      return child, failure
    end
    local refused_build, reason = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = fixture(no_width),
        sessions = 4,
        kv_cache = "f32",
        context_path = temp_dir(),
        context_bytes = POOL_BUDGET,
        workers = 2,
      },
      recording
    )
    test.equal(refused_build, nil)
    test.includes(reason as string, "slot 0")
    test.includes(reason as string, "build")
    test.truthy(#pids > 0)
    for _, pid in ipairs(pids) do
      test.equal(fs.exists("/proc/" .. tostring(pid)), false)
    end

    local function refusing(
      _name: string, _args: {string} | nil, _opts: process.Options | nil
    ): process.Child | nil, string | nil
      return nil, "no process today"
    end
    local missing, why = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = fixture(bytes),
        sessions = 4,
        kv_cache = "f32",
        context_path = temp_dir(),
        context_bytes = POOL_BUDGET,
        workers = 2,
      },
      refusing
    )
    test.equal(missing, nil)
    test.includes(why as string, "slot 0")
    test.includes(why as string, "no process today")
  end
)

local memory_backend = compute.load() as compute.Backend
local device_free = memory_backend:memory()

local on_device = support.on_device

on_device(
  "a pool the card cannot hold is refused at start",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local copy = fixture(bytes)
    local mapped = weights.load(copy) as weights.Handle
    local each = compute.place(mapped, 4503599627370496, 0).resident_bytes
    mapped.release(mapped)
    test.truthy(each > 0)
    local free = device_free as integer
    local count = math.floor(free / each) + 1

    local calls = 0
    local function unused(
      _name: string, _args: {string} | nil, _opts: process.Options | nil
    ): process.Child | nil, string | nil
      calls = calls + 1
      return nil, "unused"
    end
    local refused_build, line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = copy,
        sessions = 4,
        kv_cache = "f32",
        context_path = temp_dir(),
        context_bytes = POOL_BUDGET,
        workers = count,
      },
      unused
    )
    test.equal(refused_build, nil)
    local why = line as string
    test.includes(why, tostring(count))
    local shown = tonumber(why:match("(%d+) free"))
    test.truthy(shown, why)
    test.truthy((shown as number) <= free)
    test.equal(calls, 0)
  end
)

local function free_now(): integer | nil
  local current = compute.load() as compute.Backend
  local free = current:memory()
  compute.unload(current)
  return free
end

local function tensor_total(file_path: string): integer
  local mapped = weights.load(file_path) as weights.Handle
  local total = 0
  for _, tensor in pairs(mapped.tensors) do
    total = total + tensor.bytes
  end
  mapped.release(mapped)
  return total
end

test.it(
  "build reports what the card holds",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local written: {string} = {}
    local server = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      },
      nil,
      function(line: string) written[#written + 1] = line end
    )
    test.truthy(server)
    test.equal(#written, 1)
    local line = (written[1] as string):gsub("\n$", "")
    local name, budget, held_bytes, slots, slot_bytes, unplaced = line:match(
      DEVICE_LINE
    )
    if device_free is nil then
      test.equal(name, "none")
      test.equal(tonumber(budget), 0)
      test.equal(tonumber(held_bytes), 0)
      test.equal(tonumber(slots), 0)
      test.equal(tonumber(slot_bytes), 0)
      test.equal(tonumber(unplaced), tensor_total(file_path))
    else
      test.truthy(name ~= "none" and #(name as string) > 0)
      test.greater(tonumber(budget) as number, 0)
      test.lessEqual(tonumber(held_bytes) as number, tonumber(budget) as number)
      test.greaterEqual(tonumber(slot_bytes) as integer, 0)
    end
    serve.stop(server as tltp.Server)

    local quiet: {string} = {}
    local pool, pool_failure = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
        context_path = temp_dir(),
        context_bytes = POOL_BUDGET,
        workers = 2,
      },
      nil,
      function(chunk: string) quiet[#quiet + 1] = chunk end
    )
    test.truthy(pool, pool_failure)
    test.equal(pool_failure, nil)
    test.equal(#quiet, 0)
    serve.stop(pool as tltp.Server)

    local plain = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.truthy(plain)
    serve.stop(plain as tltp.Server)
  end
)

on_device(
  "the cap reaches the model",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local before = free_now() as integer
    local capped = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
        vram = 1048576,
      }
    )
    test.truthy(capped)
    local model = serve.model_of(capped as tltp.Server) as serve.Model
    test.equal(model.generation.placement.budget, math.min(1048576, before))
    serve.stop(capped as tltp.Server)

    local free_again = free_now() as integer
    local open = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      }
    )
    test.truthy(open)
    local uncapped = serve.model_of(open as tltp.Server) as serve.Model
    test.lessEqual(
      math.abs(uncapped.generation.placement.budget - free_again),
      16 * 1024 * 1024
    )
    serve.stop(open as tltp.Server)
  end
)

test.it(
  "start hands its writer to build",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local written: {string} = {}
    local code = serve.start(
      {
        host = "192.0.2.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      },
      function(line: string) written[#written + 1] = line end
    )
    test.equal(code, 1)
    test.equal(#written, 2)
    test.equal((written[1]:gsub("\n$", "")):match(DEVICE_LINE) ~= nil, true)
    test.includes(written[2], "BIND")
  end
)

test.it(
  "the route carries the narrowing options and refuses bad values",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local function content(extra: string): string
      local reply = answered(
        server,
        '{"messages":[{"role":"user","content":"a"}],"max_tokens":1'
          .. extra
          .. "}"
      )
      return reply.choices[1].message.content
    end

    test.equal(content(""), "a")
    test.equal(content(',"temperature":1,"seed":1791'), "c")
    test.equal(content(',"temperature":1,"seed":1791,"top_p":0.01'), "a")
    test.equal(content(',"temperature":1,"seed":682'), "")
    test.equal(content(',"temperature":1,"seed":682,"min_p":0.5'), "c")
    test.equal(content(',"presence_penalty":2'), "c")
    test.equal(content(',"repetition_penalty":2'), "c")

    local function twin(extra: string): string
      local reply = answered(
        server,
        '{"messages":[{"role":"user","content":"aa"}],"max_tokens":1'
          .. extra
          .. "}"
      )
      return reply.choices[1].message.content
    end

    test.equal(twin(',"frequency_penalty":1'), "c")
    test.equal(twin(',"presence_penalty":2'), "c")

    local cases: {{string, string}} = {
      {"top_p", '"x"'},
      {"min_p", '"x"'},
      {"repetition_penalty", '"x"'},
      {"top_p", "0"},
      {"top_p", "1.5"},
      {"min_p", "-1"},
      {"min_p", "1.5"},
      {"repetition_penalty", "0"},
      {"presence_penalty", '"x"'},
      {"frequency_penalty", '"x"'},
    }
    for _, entry in ipairs(cases) do
      local body = body_with(entry[1], entry[2])
      test.includes(rejected(server, body).error.message, entry[1], body)
    end

    local model = serve.model_of(server) as serve.Model
    local refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a",
      {
        max_tokens = 1,
        seed = 0,
        top_p = 0,
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "top_p")

    refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a",
      {
        max_tokens = 0,
        seed = 0,
        top_p = 0,
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "max_tokens")

    refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a\255",
      {
        max_tokens = 1,
        seed = 0,
        top_p = 0,
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "top_p")
  end
)

test.it(
  "a run holds back what a stop string could still complete",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local body = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"stop":"aax"}'

    local response, stream_fn = serve.stream(server, body)
    test.equal(response.status, "OK")
    test.truthy(stream_fn)
    local payloads = driven(server, stream_fn as tltp.StreamFn)
    test.equal(#payloads, 4)
    local events = events_of(payloads)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], token_event("aa"))
    test.deepEqual(events[3], finish_event("length"))
    test.equal(payloads[4], "[DONE]")

    local plain = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stop":"aax"}'
    test.equal(answered(server, plain).choices[1].message.content, "aa")
  end
)

test.it(
  "the route carries stop strings and refuses what it cannot use",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local cut = answered(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stop":"aa"}'
    )
    test.equal(cut.choices[1].message.content, "")
    test.equal(cut.usage.prompt_tokens, 1)
    test.equal(cut.usage.completion_tokens, 2)
    test.equal(cut.usage.total_tokens, 3)

    local listed = answered(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stop":["aax"]}'
    )
    local one = answered(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stop":"aax"}'
    )
    test.equal(listed.choices[1].message.content, "aa")
    test.equal(
      one.choices[1].message.content, listed.choices[1].message.content
    )

    local stored = answered(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stop":"aa","session_id":"s"}'
    )
    test.equal(stored.choices[1].message.content, "")
    test.equal(held(server, "s").length, 2)

    local cases: {{string, string}} = {
      {"stop", '""'},
      {"stop", "[]"},
      {"stop", '["a","b","c","d","e"]'},
      {"stop", "7"},
    }
    for _, entry in ipairs(cases) do
      local body = body_with(entry[1], entry[2])
      test.includes(rejected(server, body).error.message, entry[1], body)
    end

    local model = serve.model_of(server) as serve.Model
    local refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a",
      {
        max_tokens = 1,
        seed = 0,
        stop = {},
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "stop")

    refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a",
      {
        max_tokens = 0,
        seed = 0,
        stop = {},
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "max_tokens")

    refused_run, line = generate.open(
      model.generation,
      model.tokenizer,
      "a\255",
      {
        max_tokens = 1,
        seed = 0,
        stop = {},
      }
    )
    test.equal(refused_run, nil)
    test.includes(line as string, "stop")
  end
)

test.it(
  "a streamed reply that a stop string ends writes no cut text",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local body = '{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stream":true,"stop":"aa"}'

    local _, stream_fn = serve.stream(server, body)
    local payloads = driven(server, stream_fn as tltp.StreamFn)
    test.equal(#payloads, 3)
    local events = events_of(payloads)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], finish_event("stop"))
    test.equal(payloads[3], "[DONE]")

    local plain = '{"messages":[{"role":"user","content":"a"}],"max_tokens":3,"stop":"aa"}'
    test.equal(answered(server, plain).choices[1].message.content, "")

    local at_count = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"stop":"aa"}'
    local _, count_fn = serve.stream(server, at_count)
    local count_payloads = driven(server, count_fn as tltp.StreamFn)
    test.equal(#count_payloads, 3)
    local count_events = events_of(count_payloads)
    test.deepEqual(count_events[1], ROLE_EVENT)
    test.deepEqual(count_events[2], finish_event("length"))
    test.equal(count_payloads[3], "[DONE]")
  end
)

local SHARE_BODY = '{"messages":[{"role":"user","content":"aba"}],"max_tokens":2,"stream":true}'

local SHARE_ANSWER = '{"messages":[{"role":"user","content":"aba"}],"max_tokens":2}'

local FIRST_HALF = '{"messages":[{"role":"user","content":"aa"}],"max_tokens":2,"stream":true}'

local SECOND_HALF = '{"messages":[{"role":"user","content":"bb"}],"max_tokens":2,"stream":true}'

local ROW_BODY = '{"messages":[{"role":"user","content":"abab"}],'
  .. '"max_tokens":1,"stream":true}'

local function pieces(events: {StreamEvent}): {string}
  local parts: {string} = {}
  for _, event in ipairs(events) do
    local delta = event.choices[1].delta
    if delta is StreamDelta then
      local piece = delta.content
      if piece is string then
        parts[#parts + 1] = piece
      end
    end
  end
  return parts
end

test.it(
  "a step keeps to its share and splits it fairly",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local chunked = build(file_path, "m", 2, "f32", nil, nil, nil, 2)
    local whole = build(file_path, "m", 2)

    local _, chunk_fn = serve.stream(chunked, SHARE_BODY)
    test.truthy(chunk_fn)
    local one = begun(chunk_fn as tltp.StreamFn, nil, "A")
    test.deepEqual(events_of(one.payloads), {ROLE_EVENT})
    serve.step(chunked)
    test.deepEqual(events_of(one.payloads), {ROLE_EVENT})
    serve.step(chunked)
    test.truthy(#one.payloads > 1, "the prompt's last row brings a payload")
    while serve.step(chunked) > 0 do
    end

    local _, whole_fn = serve.stream(whole, SHARE_BODY)
    test.truthy(whole_fn)
    local full = begun(whole_fn as tltp.StreamFn, nil, "B")
    serve.step(whole)
    test.truthy(#full.payloads > 1, "the whole prompt goes in one step")
    while serve.step(whole) > 0 do
    end

    test.deepEqual(one.payloads, full.payloads)
    local answer = serve.answer(whole, SHARE_ANSWER)
    local reply = tltp.json_decode(answer.body) as ChatReply
    test.equal(
      table.concat(pieces(events_of(one.payloads)), ""),
      reply.choices[1].message.content
    )

    local pair = build(file_path, "m", 2, "f32", nil, nil, nil, 2)
    local _, first_fn = serve.stream(pair, FIRST_HALF)
    test.truthy(first_fn)
    local first = begun(first_fn as tltp.StreamFn, nil, "A")
    local _, second_fn = serve.stream(pair, SECOND_HALF)
    test.truthy(second_fn)
    local second = begun(second_fn as tltp.StreamFn, nil, "B")
    test.equal(serve.step(pair), 2)
    test.deepEqual(events_of(first.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(second.payloads), {ROLE_EVENT})
    serve.step(pair)
    test.truthy(#first.payloads > 1, "the first turn was given its row")
    test.truthy(#second.payloads > 1, "the second turn was given its row")
    while serve.step(pair) > 0 do
    end

    -- 33/FR-011: a turn is given its own group's last row, so a turn whose group
    -- runs over several rows answers what the same prompt answers one row at a
    -- time, and a turn shares nothing with the turn stepped beside it
    local group = build(file_path, "m", 2, "f32", nil, nil, nil, 8)
    local row = build(file_path, "m", 1, "f32", nil, nil, nil, 1)
    local _, aside_fn = serve.stream(group, FIRST_HALF)
    local _, whole_fn2 = serve.stream(group, ROW_BODY)
    local _, one_fn = serve.stream(row, ROW_BODY)
    test.truthy(aside_fn)
    test.truthy(whole_fn2)
    test.truthy(one_fn)
    local aside = begun(aside_fn as tltp.StreamFn, nil, "A")
    local walked = begun(whole_fn2 as tltp.StreamFn, nil, "B")
    local solo = begun(one_fn as tltp.StreamFn, nil, "C")
    while serve.step(group) > 0 do
    end
    while serve.step(row) > 0 do
    end
    test.truthy(#aside.payloads > 1, "the turn beside it was given its row")
    test.equal(
      table.concat(pieces(events_of(walked.payloads)), ""),
      table.concat(pieces(events_of(solo.payloads)), ""),
      "the group's last row answers the prompt"
    )

    serve.stop(chunked)
    serve.stop(whole)
    serve.stop(pair)
    serve.stop(group)
    serve.stop(row)
  end
)

test.it(
  "the endpoint's chunk bound reaches the model",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local bound = build(file_path, "m", 1, "f32", nil, nil, nil, 3)
    local carried = serve.model_of(bound) as serve.Model
    test.equal(carried.generation.ubatch, 3)

    local quiet = build(file_path, "m", 1)
    test.equal((serve.model_of(quiet) as serve.Model).generation.ubatch, 512)

    local plain, plain_line = compute.model(
      carried.backend, carried.handle, nil, "f32", 1
    )
    test.equal(plain_line, nil)
    test.equal((plain as compute.Model).ubatch, 512)

    local narrow, narrow_line = compute.model(
      carried.backend, carried.handle, nil, "f32", 1, 0
    )
    test.equal(narrow, nil)
    local message = narrow_line as string
    test.includes(message, "model")
    test.includes(message, "ubatch")

    serve.stop(bound)
    serve.stop(quiet)
  end
)

-- the count a model is built with when nothing names one
local DEFAULT_THREADS = cores.machine()

-- a config file naming the entry `m` for this model, and the keys given
local function config_with(file_path: string, keys: string): string
  local location = temp_dir() .. "/config.json"
  fs.write(
    location, '{"models":{"m":{"path":"' .. file_path .. '"}}' .. keys .. "}"
  )
  return location
end

test.it(
  "the count reaches start from the file, the flag and the default",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local titled = config_with(file_path, ',"threads":3')
    local three = run({"serve", "m", "--config", titled}, {})
    test.equal(three.code, 0)
    test.equal((three.endpoint as serve.Endpoint).threads, 3)

    local flagged = run(
      {"serve", "m", "--config", titled, "--threads", "5"}, {}
    )
    test.equal((flagged.endpoint as serve.Endpoint).threads, 5)

    local plain = config_with(file_path, "")
    local machine = run({"serve", "m", "--config", plain}, {})
    test.equal((machine.endpoint as serve.Endpoint).threads, DEFAULT_THREADS)

    local pair = config_with(file_path, ',"workers":2')
    local shared = run({"serve", "m", "--config", pair}, {})
    test.equal(
      (shared.endpoint as serve.Endpoint).threads,
      math.max(1, math.floor(DEFAULT_THREADS / 2))
    )

    local leaning = config_with(
      file_path, ',"workers":' .. tostring(DEFAULT_THREADS + 1)
    )
    local one = run({"serve", "m", "--config", leaning}, {})
    test.equal((one.endpoint as serve.Endpoint).threads, 1)

    local alone = config_with(file_path, ',"workers":1')
    local whole = run({"serve", "m", "--config", alone}, {})
    test.equal((whole.endpoint as serve.Endpoint).threads, DEFAULT_THREADS)

    -- a hand-written text pads its keys with spaces, the kernel's with tabs
    test.equal(
      cores.default(
        "processor : 0\nphysical id : 0\ncore id : 0\n"
          .. "processor : 1\nphysical id : 0\ncore id : 1\n"
          .. "processor : 2\nphysical id : 0\ncore id : 2\n"
      ),
      2
    )
    test.equal(
      cores.default(
        "processor\t: 0\nprocessor\t: 1\nprocessor\t: 2\nprocessor\t: 3\n"
      ),
      3
    )
    test.equal(
      cores.default("processor\t: 0\nphysical id\t: 0\ncore id\t: 0\n"), 1
    )
    test.equal(cores.default(""), 1)
  end
)

local type Graphs = function(): integer

-- what the model under the counting planner publishes, so a case can hand the
-- wrapper the model once it exists
local record Watched
  graphs: Graphs | nil
end

-- what a model planned its graphs with: the calls the backend's
-- ggml_backend_set_n_threads entry saw, the count it was given, and the graphs
-- the model had computed when it was called
local record Planned
  calls: integer
  count: integer
  graphs: integer
end

local function counting_planner(
  backend: compute.Backend, graphs_now: function(): integer
): Planned, function()
  local entries = (engine.LOADED[backend] as Loaded).threads as Threads
  local set = ffi.cast("void (*)(ggml_backend *, int)", entries.set_n_threads)
    as function(ffi.CData, integer)
  local counts: Planned = {calls = 0, count = 0, graphs = 0}
  entries.set_n_threads = ffi.cast(
    "void (*)(ggml_backend *, int)",
    function(device: ffi.CData, count: integer)
      counts.calls = counts.calls + 1
      counts.count = count
      counts.graphs = graphs_now()
      set(device, count)
    end
  ) as ffi.CData
  local function restore()
    entries.set_n_threads = ffi.cast("void (*)(ggml_backend *, int)", set)
      as ffi.CData
  end
  return counts, restore
end

test.it(
  "the count reaches the model and its graph planner",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local counted = build(fixture(bytes), "m", nil, nil, nil, nil, nil, nil, 3)
    test.equal((serve.model_of(counted) as serve.Model).generation.threads, 3)
    serve.stop(counted)

    local quiet = build(fixture(bytes), "m")
    test.equal(
      (serve.model_of(quiet) as serve.Model).generation.threads, DEFAULT_THREADS
    )
    serve.stop(quiet)

    local backend = compute.load(nil, false) as compute.Backend
    local handle = weights.load(fixture(bytes)) as weights.Handle
    local box: Watched = {graphs = nil}
    local counts, restore = counting_planner(
      backend,
      function(): integer
        local now = box.graphs
        if now is Graphs then
          return now()
        end
        return 0
      end
    )
    local built, failure = compute.model(
      backend, handle, 32, nil, 1, nil, nil, nil, nil, 3
    )
    test.equal(failure, nil)
    local model = built as compute.Model
    test.equal(model.threads, 3)
    test.equal(counts.calls, 0)
    box.graphs = function(): integer return model.measure.graphs end
    test.equal(model:reserve("", 32), nil)
    local rows, line = model:pass({{sequence = "", token = 0, position = 0}})
    test.equal(line, nil)
    test.truthy(rows ~= nil, "the pass ran")
    test.equal(counts.calls, 1)
    test.equal(counts.count, 3)
    test.equal(counts.graphs, 0)
    model:close()
    restore()

    local plain_handle = weights.load(fixture(bytes)) as weights.Handle
    local plain_counts, plain_restore = counting_planner(
      backend, function(): integer return 0 end
    )
    local plain = compute.model(backend, plain_handle, 32, nil, 1)
      as compute.Model
    test.equal(plain.threads, DEFAULT_THREADS)
    test.equal(plain:reserve("", 32), nil)
    local plain_rows, plain_line = plain:pass(
      {{sequence = "", token = 0, position = 0}}
    )
    test.equal(plain_line, nil)
    test.truthy(plain_rows ~= nil, "the pass ran")
    test.equal(plain_counts.count, DEFAULT_THREADS)
    plain:close()
    plain_restore()
  end
)

test.it(
  "a count outside one to 512 is refused",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local bad: {string} = {'"8"', "0", "1.5", "-1", "513"}
    for _, value in ipairs(bad) do
      local location = config_with(file_path, ',"threads":' .. value)
      local bad_run = run({"serve", "m", "--config", location}, {})
      test.equal(bad_run.code, 1)
      test.equal(bad_run.starts, 0)
      test.equal(#bad_run.err, 1)
      test.includes(bad_run.err[1], "threads")
      test.includes(bad_run.err[1], location)
    end

    local flags: {{string}} = {
      {"serve", "--threads", "0"},
      {"serve", "--threads", "1.5"},
      {"serve", "--threads", "513"},
      {"bench", "m", "--threads", "513"},
    }
    for _, args in ipairs(flags) do
      local bad_run = run(args, {})
      test.equal(bad_run.code, 1)
      test.equal(#bad_run.err, 1)
      test.includes(bad_run.err[1], "--threads")
      test.equal(bad_run.starts, 0)
    end

    local backend = compute.load(nil, false) as compute.Backend
    local handle = weights.load(
      fixture(meta_bytes(t001_meta(), t001_entries()))
    ) as weights.Handle
    for _, count in ipairs({0, 513}) do
      local built, line = compute.model(
        backend, handle, 32, nil, 1, nil, nil, nil, nil, count
      )
      test.equal(built, nil)
      local refusal = line as string
      test.includes(refusal, "model")
      test.includes(refusal, "threads")
    end
  end
)

-- the Threads line of one process's status, which a model's pool raises by the
-- count it holds less the calling thread
local function threads_of_process(pid: integer): integer
  local status = fs.read("/proc/" .. tostring(pid) .. "/status") as string
  return tonumber(status:match("Threads:%s*(%d+)")) as integer
end

local function thread_counts(server: tltp.Server): {integer: integer}
  local counts: {integer: integer} = {}
  for _, worker in ipairs(workers_of(server)) do
    counts[worker.slot] = threads_of_process(worker.pid)
  end
  return counts
end

-- a pool whose workers keep their model on the CPU, so that each holds the
-- threadpool its endpoint's count makes: a card that would hold the weights
-- leaves a worker with no pool at all
local function cpu_pool(file_path: string, threads: integer): tltp.Server
  local server, failure = serve.build(
    {
      host = "127.0.0.1",
      port = 9090,
      name = "m",
      path = file_path,
      sessions = 4,
      kv_cache = "f32",
      context_path = temp_dir(),
      context_bytes = POOL_BUDGET,
      workers = 2,
      vram = 1,
      threads = threads,
    }
  )
  test.equal(failure, nil, file_path)
  return server as tltp.Server
end

test.it(
  "a worker decodes the count from its endpoint json",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local narrow = cpu_pool(fixture(bytes), 2)
    test.equal(#workers_of(narrow), 2)
    local narrow_counts = thread_counts(narrow)

    local wide = cpu_pool(fixture(bytes), 4)
    test.equal(#workers_of(wide), 2)
    local wide_counts = thread_counts(wide)
    for slot, count in pairs(narrow_counts) do
      test.equal(wide_counts[slot], count + 2)
    end

    local pids: {integer} = {}
    for _, worker in ipairs(workers_of(narrow)) do
      pids[#pids + 1] = worker.pid
    end
    for _, worker in ipairs(workers_of(wide)) do
      pids[#pids + 1] = worker.pid
    end
    serve.stop(narrow)
    serve.stop(wide)
    for _, pid in ipairs(pids) do
      test.equal(fs.exists("/proc/" .. tostring(pid)), false)
    end
  end
)

-- 18's fixture with a template whose prompt gains the id of `c` only when the
-- name `enable_thinking` reaches it bound to false.
local THINKING_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"
  .. "{% if enable_thinking is defined and not enable_thinking %}c{% endif %}"

local THINK_BODY = A_BODY:sub(1, #A_BODY - 1)

local function body_both(top: string, box: string): string
  return THINK_BODY
    .. ',"enable_thinking":'
    .. top
    .. ',"chat_template_kwargs":'
    .. box
    .. "}"
end

local function body_stream(flag: string, top: string): string
  return THINK_BODY
    .. ',"stream":'
    .. flag
    .. ',"enable_thinking":'
    .. top
    .. "}"
end

test.it(
  "the body's thinking member decides the prompt, and the nested one wins",
  function()
    local server = build(
      meta_file(
        replaced(
          t001_meta(), mtext("tokenizer.chat_template", THINKING_TEMPLATE)
        ),
        t001_entries()
      ),
      "m"
    )

    local cases: {{string, integer}} = {
      {A_BODY, 2},
      {body_with("enable_thinking", "true"), 1},
      {body_with("enable_thinking", "false"), 2},
      {body_with("chat_template_kwargs", '{"enable_thinking":true}'), 1},
      {body_both("true", '{"enable_thinking":false}'), 2},
      {body_both("true", "{}"), 1},
      {body_both("true", "[]"), 1},
    }

    for _, given in ipairs(cases) do
      local reply = answered(server, given[1])
      test.equal(reply.usage.prompt_tokens, given[2], given[1])
    end

    local plain = build(meta_file(t001_meta(), t001_entries()), "m")
    local kept = answered(plain, A_BODY)
    local plain_model = serve.model_of(plain) as serve.Model

    -- 18's fixture keeps the template it has, so no prompt and no usage
    -- count any earlier spec records moves.
    test.equal(
      plain_model.tokenizer.template,
      "{% for message in messages %}{{ message.content }}{% endfor %}"
    )
    test.equal(kept.usage.prompt_tokens, 1)
    test.equal(kept.usage.total_tokens, 257)
  end
)

test.it(
  "a thinking member of the wrong kind is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local cases: {{string, string}} = {
      {body_with("enable_thinking", '"yes"'), "enable_thinking: "},
      {body_with("enable_thinking", "1"), "enable_thinking: "},
      {
        body_with("chat_template_kwargs", '{"enable_thinking":"yes"}'),
        "chat_template_kwargs.enable_thinking: ",
      },
      {
        body_both("true", '{"enable_thinking":"yes"}'),
        "chat_template_kwargs.enable_thinking: ",
      },
      {body_with("chat_template_kwargs", "5"), "chat_template_kwargs: "},
      {body_with("chat_template_kwargs", "true"), "chat_template_kwargs: "},
      {body_with("chat_template_kwargs", '"x"'), "chat_template_kwargs: "},
      {body_with("chat_template_kwargs", "null"), "chat_template_kwargs: "},
      {body_with("chat_template_kwargs", '["a"]'), "chat_template_kwargs: "},
      {body_both('"yes"', "5"), "enable_thinking: "},
      {body_with("enable_thinking", "null"), "enable_thinking: "},
      {
        body_both('"yes"', '{"enable_thinking":true}'),
        "enable_thinking: ",
      },
      {
        body_stream("5", '"yes"'),
        "stream: ",
      },
    }

    for _, given in ipairs(cases) do
      local fault = rejected(server, given[1])
      test.equal(fault.error.message:sub(1, #given[2]), given[2], given[1])

      -- The same body is the error envelope alone: no choice and no usage.
      local reply = fault as ChatReply
      test.equal(reply.choices, nil, given[1])
      test.equal(reply.usage, nil, given[1])
    end
  end
)

-- 18's fixture with a template whose prompt gains the id of a fixture token
-- when the template's own tools guard is taken.
local TOOLS_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"
  .. "{% if tools and tools is iterable and tools is not mapping %}"
  .. "{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}"

local TOOLS_BOUND_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"
  .. "{% if tools is defined %}b{% endif %}"

local TOOL_CHOICE_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"
  .. "{% if tool_choice is defined %}"
  .. "{% if tool_choice == 'none' %}b"
  .. "{% elif tool_choice is mapping %}{{ tool_choice.function.name }}"
  .. "{% else %}cc{% endif %}{% endif %}"

local function templated(template: string): tltp.Server
  return build(
    meta_file(
      replaced(t001_meta(), mtext("tokenizer.chat_template", template)),
      t001_entries()
    ),
    "m"
  )
end

test.it(
  "a body's tool list reaches the template's tools block",
  function()
    local server = templated(TOOLS_TEMPLATE)
    local cases: {{string, integer}} = {
      {body_with("tools", '[{"type":"function","function":{"name":"b"}}]'), 2},
      {
        body_with(
          "tools",
          '[{"type":"function","function":{"name":"b"}},'
            .. '{"type":"function","function":{"name":"c"}}]'
        ),
        3,
      },
      {A_BODY, 1},
      {body_with("tools", "[]"), 1},
    }
    for _, given in ipairs(cases) do
      test.equal(
        answered(server, given[1]).usage.prompt_tokens, given[2], given[1]
      )
    end

    -- An empty array is bound and no array is not: the guard sees one defined.
    local bound = templated(TOOLS_BOUND_TEMPLATE)
    test.equal(answered(bound, body_with("tools", "[]")).usage.prompt_tokens, 2)
    test.equal(answered(bound, A_BODY).usage.prompt_tokens, 1)
  end
)

test.it(
  "a body's tool_choice reaches the template, and none leaves it unbound",
  function()
    local server = templated(TOOL_CHOICE_TEMPLATE)
    local cases: {{string, integer}} = {
      {A_BODY, 1},
      {body_with("tool_choice", '"none"'), 2},
      {
        body_with(
          "tool_choice", '{"type":"function","function":{"name":"c"}}'
        ),
        2,
      },
      {body_with("tool_choice", '"auto"'), 3},
      {body_with("tool_choice", "null"), 3},
      {body_with("tools", '[{"type":"function","function":{"name":"b"}}]'), 1},
    }
    for _, given in ipairs(cases) do
      test.equal(
        answered(server, given[1]).usage.prompt_tokens, given[2], given[1]
      )
    end
  end
)

test.it(
  "a tools member the route cannot use is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")

    local shape: {{string, string}} = {
      {body_with("tools", "5"), "tools: not an array"},
      {body_with("tools", "null"), "tools: not an array"},
      {body_with("tools", "{}"), "tools: not an array"},
      {body_with("tools", '"ab"'), "tools: not an array"},
    }
    for _, given in ipairs(shape) do
      local fault = rejected(server, given[1])
      local message = fault.error.message
      test.equal(message:sub(1, #given[2]), given[2], given[1])
      test.equal(message:find("entry", 1, true), nil, given[1])

      -- The same body is the error envelope alone: no choice and no usage.
      local reply = fault as ChatReply
      test.equal(reply.choices, nil, given[1])
      test.equal(reply.usage, nil, given[1])
    end

    local entries: {{string, string}} = {
      {body_with("tools", "[5]"), "tools: entry 1"},
      {body_with("tools", '[{"type":"function"}]'), "tools: entry 1"},
      {
        body_with("tools", '[{"type":"function","function":{}}]'),
        "tools: entry 1",
      },
      {
        body_with("tools", '[{"type":"function","function":{"name":""}}]'),
        "tools: entry 1",
      },
      {
        body_with("tools", '[{"type":"function","function":{"name":5}}]'),
        "tools: entry 1",
      },
      {
        body_with(
          "tools",
          '[{"type":"function","function":{"name":"b"}},'
            .. '{"type":"function","function":{}}]'
        ),
        "tools: entry 2",
      },
    }
    for _, given in ipairs(entries) do
      local fault = rejected(server, given[1]).error.message
      test.equal(fault:sub(1, #given[2]), given[2], given[1])
    end

    -- The checks the tools check comes after answer first.
    local earlier: {{string, string}} = {
      {
        THINK_BODY .. ',"stream":5,"tools":5}',
        "stream: not a boolean",
      },
      {
        THINK_BODY .. ',"chat_template_kwargs":5,"tools":5}',
        "chat_template_kwargs: not an object",
      },
      {
        THINK_BODY .. ',"enable_thinking":"yes","tools":5}',
        "enable_thinking: not a boolean",
      },
    }
    for _, given in ipairs(earlier) do
      local fault = rejected(server, given[1]).error.message
      test.equal(fault:sub(1, #given[2]), given[2], given[1])
    end
  end
)

-- 48/T-003: the fixture template that writes one b per accepted call list and
-- one a per message that carries none.
local CALLS_TEMPLATE = "{% for message in messages %}{% if message.tool_calls %}b"
  .. "{% else %}a{% endif %}{% endfor %}"

local CALLS_BODY = '{"messages":[{"role":"user","content":"a"},'
  .. '{"role":"assistant","content":"",'
  .. '"tool_calls":[{"type":"function","function":{"name":"x","arguments":"{}"}}]}]}'

local PLAIN_BODY = '{"messages":[{"role":"user","content":"a"}]}'

-- 48/T-003's last clause: a template that writes the name of each bound call,
-- so the array the calls were read from shows in the id count ("ab" is two of
-- the fixture's tokens, "c" is one).
local NAMES_TEMPLATE = "{% for message in messages %}"
  .. "{% if message.tool_calls %}"
  .. "{% for call in message.tool_calls %}{{ call.name }}{% endfor %}"
  .. "{% endif %}{% endfor %}"

local TWICE_BODY = '{"messages":[{"role":"assistant","content":"",'
  .. '"tool_calls":[{"type":"function","function":{"name":"ab",'
  .. '"arguments":"{}"}}]}],'
  .. '"messages":[{"role":"assistant","content":"",'
  .. '"tool_calls":[{"type":"function","function":{"name":"c",'
  .. '"arguments":"{}"}}]}]}'

test.it(
  "a body carrying a tool call reaches the model",
  function()
    local server = templated(CALLS_TEMPLATE)
    test.equal(answered(server, CALLS_BODY).usage.prompt_tokens, 2, CALLS_BODY)
    test.equal(answered(server, PLAIN_BODY).usage.prompt_tokens, 1, PLAIN_BODY)

    -- the last `messages` member is the one tltp.json_decode keeps and every
    -- shape check reads, so its call `c` is the one that reaches the model
    test.equal(
      answered(templated(NAMES_TEMPLATE), TWICE_BODY).usage.prompt_tokens,
      1,
      TWICE_BODY
    )
  end
)

-- 48/T-004: each body the message checks refuse, the words their messages
-- name, and the one word the check's own order keeps from answering.
local FIELD_CASES: {{string, {string}, string}} = {
  {
    '{"messages":[{"role":"tool","content":"a"}]}',
    {"messages", "1", "tool_call_id"},
    "",
  },
  {
    '{"messages":[{"role":"tool","content":"a","tool_call_id":""}]}',
    {"messages", "1", "tool_call_id"},
    "",
  },
  {
    '{"messages":[{"role":"tool","content":"a","tool_call_id":5}]}',
    {"messages", "1", "tool_call_id"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"","tool_calls":5}]}',
    {"messages", "2", "tool_calls"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"","tool_calls":[5]}]}',
    {"messages", "2", "tool_calls", "1"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"",'
      .. '"tool_calls":[{"type":"function","function":5}]}]}',
    {"messages", "2", "tool_calls", "1"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a","tool_call_id":5},'
      .. '{"role":"assistant","content":"",'
      .. '"tool_calls":[{"type":"function"}]}]}',
    {"messages", "1", "tool_call_id"},
    "tool_calls",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"",'
      .. '"tool_calls":[{"type":"function"}]}]}',
    {"messages", "2", "tool_calls", "1"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"",'
      .. '"tool_calls":[{"type":"function","function":{"name":"get_weather"}}]}]}',
    {"messages", "2", "tool_calls", "1"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"",'
      .. '"tool_calls":[{"type":"function","function":'
      .. '{"name":"get_weather","arguments":"not json"}}]}]}',
    {"messages", "2", "tool_calls", "1"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"","name":5}]}',
    {"messages", "2", "name"},
    "",
  },
  {
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"","tool_call_id":5,"name":5}]}',
    {"messages", "2", "tool_call_id"},
    "name",
  },
}

test.it(
  "a message field the engine cannot use is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    for _, given in ipairs(FIELD_CASES) do
      local fault = rejected(server, given[1])
      local message = fault.error.message
      for _, word in ipairs(given[2]) do
        test.includes(message, word, given[1])
      end
      if given[3] ~= "" then
        test.equal(message:find(given[3], 1, true), nil, given[1])
      end

      -- The same body is the error envelope alone: no choice and no usage.
      local reply = fault as ChatReply
      test.equal(reply.choices, nil, given[1])
      test.equal(reply.usage, nil, given[1])
    end
  end
)

test.it(
  "a role the engine does not know is a bad request",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local cases: {string} = {
      '{"messages":[{"role":"wizard","content":"a"}]}',
      '{"messages":[{"role":"wizard","content":"a"}],"tools":5}',
    }
    for _, body in ipairs(cases) do
      local message = rejected(server, body).error.message
      test.includes(message, "messages", body)
      test.includes(message, "1", body)
      test.includes(message, "wizard", body)
      test.equal(message:find("tools", 1, true), nil, body)
    end
  end
)
