local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local cli = require("tllm.cli")
local compute = require("tllm.compute")
local serve = require("tllm.serve")
local tokenizer = require("tllm.tokenizer")
local tltp = require("tltp")

local record HealthBody
  status: string
  model: string
end

local record ErrorDetail
  message: string
  type: string
end

local record ErrorBody
  error: ErrorDetail
end

local record ReplyMessage
  role: string
  content: string
end

local record Choice
  index: integer
  message: ReplyMessage
end

local record PromptDetails
  cached_tokens: integer
end

local record Usage
  prompt_tokens: integer
  completion_tokens: integer
  total_tokens: integer
  prompt_tokens_details: PromptDetails
end

local record ChatReply
  choices: {Choice}
  usage: Usage
end

local record StreamDelta
  role: string | nil
  content: string | nil
end

local record StreamChoice
  index: integer
  delta: StreamDelta | nil
  finish_reason: string | nil
end

local record StreamEvent
  choices: {StreamChoice}
end

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/s" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
end

local function u32(value: integer): string
  return string.char(
    value % 256,
    math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256,
    math.floor(value / 16777216) % 256
  )
end

local function u64(value: integer): string
  local parts: {string} = {}
  local rest = value
  for _ = 1, 8 do
    parts[#parts + 1] = string.char(rest % 256)
    rest = math.floor(rest / 256)
  end
  return table.concat(parts)
end

local function text(bytes: string): string
  return u64(#bytes) .. bytes
end

local function padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

local type Floats = {integer: number}

local FLOAT_BYTES: integer = 4

local function floats(values: {number}): string
  local parts: {string} = {}
  local cell = ffi.new("float[1]") as Floats
  for _, value in ipairs(values) do
    cell[0] = value
    parts[#parts + 1] = ffi.string(cell as ffi.CData, FLOAT_BYTES)
  end
  return table.concat(parts)
end

local record Entry
  name: string
  dims: {integer}
  code: integer
  bytes: string
end

local record Meta
  name: string
  code: integer
  bytes: string
end

local function meta_bytes(meta: {Meta}, entries: {Entry}): string
  local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(#meta)
  for _, entry in ipairs(meta) do
    head = head .. text(entry.name) .. u32(entry.code) .. entry.bytes
  end
  local body: {string} = {}
  local offset = 0
  for _, entry in ipairs(entries) do
    local dims: {string} = {}
    for _, dimension in ipairs(entry.dims) do
      dims[#dims + 1] = u64(dimension)
    end
    head = head
      .. text(entry.name)
      .. u32(#entry.dims)
      .. table.concat(dims)
      .. u32(entry.code)
      .. u64(offset)
    body[#body + 1] = padded(entry.bytes, 32)
    offset = offset + #body[#body]
  end
  return padded(head, 32) .. table.concat(body)
end

local function meta_file(meta: {Meta}, entries: {Entry}): string
  return fixture(meta_bytes(meta, entries))
end

local function zeros(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 0
  end
  return result
end

local function ones(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 1
  end
  return result
end

local function f32(name: string, dims: {integer}, values: {number}): Entry
  return {name = name, dims = dims, code = 0, bytes = floats(values)}
end

local function mint(name: string, value: integer): Meta
  return {name = name, code = 4, bytes = u32(value)}
end

local function mfloat(name: string, value: number): Meta
  return {name = name, code = 6, bytes = floats({value})}
end

local function mtext(name: string, value: string): Meta
  return {name = name, code = 8, bytes = text(value)}
end

local function array(item_code: integer, items: {string}): string
  return u32(item_code) .. u64(#items) .. table.concat(items)
end

local function mstrings(name: string, values: {string}): Meta
  local items: {string} = {}
  for _, value in ipairs(values) do
    items[#items + 1] = text(value)
  end
  return {name = name, code = 9, bytes = array(8, items)}
end

local function mint_array(name: string, values: {integer}): Meta
  local items: {string} = {}
  for _, value in ipairs(values) do
    items[#items + 1] = u32(value)
  end
  return {name = name, code = 9, bytes = array(5, items)}
end

local function both(first: {Entry}, second: {Entry}): {Entry}
  local result: {Entry} = {}
  for _, entry in ipairs(first) do
    result[#result + 1] = entry
  end
  for _, entry in ipairs(second) do
    result[#result + 1] = entry
  end
  return result
end

local function without_meta(meta: {Meta}, name: string): {Meta}
  local result: {Meta} = {}
  for _, entry in ipairs(meta) do
    if entry.name ~= name then
      result[#result + 1] = entry
    end
  end
  return result
end

local function without_prefix(meta: {Meta}, prefix: string): {Meta}
  local result: {Meta} = {}
  for _, entry in ipairs(meta) do
    if entry.name:sub(1, #prefix) ~= prefix then
      result[#result + 1] = entry
    end
  end
  return result
end

local function replaced(meta: {Meta}, entry: Meta): {Meta}
  local result = without_meta(meta, entry.name)
  result[#result + 1] = entry
  return result
end

local function t001_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 260),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", {"a", "b", "c"}),
    mint_array("tokenizer.ggml.token_type", {1, 1, 1}),
    mstrings("tokenizer.ggml.merges", {}),
    mint("tokenizer.ggml.eos_token_id", 1),
    mtext(
      "tokenizer.chat_template",
      "{% for message in messages %}{{ message.content }}{% endfor %}"
    ),
  }
end

local function attention_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
    f32(prefix .. "attn_k.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_v.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_q_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_k_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_output.weight", {2, 2}, zeros(4)),
  }
end

local function feed_forward_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "post_attention_norm.weight", {2}, ones(2)),
    f32(prefix .. "ffn_gate_inp.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ffn_gate_inp_shexp.weight", {2}, zeros(2)),
    f32(prefix .. "ffn_up_shexp.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ffn_gate_shexp.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ffn_down_shexp.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ffn_up_exps.weight", {2, 2, 2}, zeros(8)),
    f32(prefix .. "ffn_gate_exps.weight", {2, 2, 2}, zeros(8)),
    f32(prefix .. "ffn_down_exps.weight", {2, 2, 2}, zeros(8)),
  }
end

local function t001_entries(): {Entry}
  return both(
    both(attention_entries("blk.0."), feed_forward_entries("blk.0.")),
    {
      f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, 1, 0}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
    }
  )
end

local A_BODY = '{"messages":[{"role":"user","content":"a"}]}'

local function body_with(name: string, value: string): string
  return '{"messages":[{"role":"user","content":"a"}],"'
    .. name
    .. '":'
    .. value
    .. "}"
end

local function answered(server: tltp.Server, body: string): ChatReply
  local reply = serve.answer(server, body)
  test.equal(reply.status, "OK", body)
  return tltp.json_decode(reply.body) as ChatReply
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

local function build(
  file_path: string, name: string, sessions?: integer, cache?: compute.CacheType
): tltp.Server
  local server, failure = serve.build(
    {
      host = "127.0.0.1",
      port = 9090,
      name = name,
      path = file_path,
      sessions = sessions or 4,
      kv_cache = cache or "f32",
    }
  )
  test.equal(failure, nil, file_path)
  return server as tltp.Server
end

local function temp_dir(): string
  created = created + 1
  local dir = ROOT .. "/d" .. tostring(created)
  fs.mkdirAll(dir)
  return dir
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
    args,
    function(_text: string) end,
    function(line: string) result.err[#result.err + 1] = line end,
    stub,
    environment
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

    local named = run({"serve", "m", "--config", configured}, {})
    test.equal(named.code, 0)
    test.equal(named.starts, 1)
    local named_endpoint = named.endpoint as serve.Endpoint
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

    for _, held in ipairs(cases) do
      local decoded = rejected(server, held[1])
      for _, word in ipairs(held[2]) do
        test.includes(decoded.error.message, word, held[1])
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

local record Watcher
  co: thread
  payloads: {string}
end

local function begun(
  stream_fn: tltp.StreamFn, order: {string} | nil, tag: string
): Watcher
  local payloads: {string} = {}
  local co = coroutine.create(
    function(): boolean | nil
      return stream_fn(
        function(event: string | tltp.SSEMessage): string | nil, string | nil
          payloads[#payloads + 1] = event as string
          if not (order is nil) then
            order[#order + 1] = tag
          end
          return nil, nil
        end
      )
    end
  )
  local ok, reason = coroutine.resume(co)
  test.equal(ok, true, tostring(reason))
  return {co = co, payloads = payloads}
end

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

local function events_of(payloads: {string}): {StreamEvent}
  local events: {StreamEvent} = {}
  for _, payload in ipairs(payloads) do
    if payload ~= "[DONE]" then
      events[#events + 1] = tltp.json_decode(payload) as StreamEvent
    end
  end
  return events
end

local ROLE_EVENT: StreamEvent = {
  choices = {{index = 0, delta = {role = "assistant"}}},
}

local function token_event(piece: string): StreamEvent
  return {choices = {{index = 0, delta = {content = piece}}}}
end

local function finish_event(reason: string): StreamEvent
  return {choices = {{index = 0, finish_reason = reason}}}
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

    for _, held in ipairs(cases) do
      local response, decoded = stream_refused(server, held[1])
      test.equal(response.status, held[2], held[1])
      test.equal(decoded.error.type, held[3], held[1])
      for _, word in ipairs(held[4]) do
        test.includes(decoded.error.message, word, held[1])
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

local FIRST_TURN = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}'

local SECOND_TURN = '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}'

local function named(body: string, id: string): string
  return (body:gsub('"session_id":"s"', '"session_id":"' .. id .. '"'))
end

local function held(server: tltp.Server, id: string): serve.Session
  local session = serve.session_of(server, id)
  test.truthy(session, id)
  return session as serve.Session
end

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
        .. '"temperature":1,"seed":7,"session_id":"v"}'
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

local function unreadable_fixture(): string
  return meta_file(
    replaced(
      t001_meta(), mstrings("tokenizer.ggml.tokens", {"a", "b", "\255"})
    ),
    t001_entries()
  )
end

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

    local streamed = (FIRST_TURN:gsub("}$", ',"stream":true}'))
    local _, streamed_run = serve.stream(server, streamed)
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

local function streamed(id: string): string
  return '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
    .. '"stream":true,"session_id":"'
    .. id
    .. '"}'
end

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

local function streaming(body: string): string
  return (body:gsub("}$", ',"stream":true}'))
end

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

local function at(file_path: string, host: string, port: integer): tltp.Server
  local server, failure = serve.build(
    {
      host = host,
      port = port,
      name = "m",
      path = file_path,
      sessions = 4,
      kv_cache = "f32",
    }
  )
  test.equal(failure, nil, file_path)
  return server as tltp.Server
end

test.it(
  "a round steps the transport and the turns",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local server = at(file_path, "127.0.0.1", 9405)
    local _, round_run = serve.stream(server, streamed("s1"))
    test.truthy(round_run)
    local watcher = begun(round_run as tltp.StreamFn, nil, "A")

    test.equal(serve.round(server), nil)
    test.equal(serve.round(server), nil)
    test.equal(#watcher.payloads, 5)
    local events = events_of(watcher.payloads)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], token_event("a"))
    test.deepEqual(events[3], token_event("a"))
    test.deepEqual(events[4], finish_event("length"))
    test.equal(watcher.payloads[5], "[DONE]")

    local unreachable = at(file_path, "192.0.2.1", 9090)
    test.includes(serve.round(unreachable) as string, "BIND")

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
    test.equal(#written, 1)
    test.includes(written[1], "BIND")
    test.equal(#(written[1]:match("(\n*)$") as string), 1)
  end
)

test.it(
  "a turn registered while others are live is stepped next",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local order: {string} = {}
    local _, first_fn = serve.stream(server, streamed("s1"))
    local first = begun(first_fn as tltp.StreamFn, order, "A")
    test.equal(serve.step(server), 1)
    test.equal(#first.payloads, 2)
    test.deepEqual(events_of(first.payloads)[2], token_event("a"))

    local _, second_fn = serve.stream(server, streamed("s2"))
    local second = begun(second_fn as tltp.StreamFn, order, "B")
    test.deepEqual(events_of(second.payloads), {ROLE_EVENT})

    test.equal(serve.step(server), 1)
    test.equal(#first.payloads, 5)
    test.deepEqual(events_of(first.payloads)[4], finish_event("length"))
    test.equal(#second.payloads, 2)
    test.deepEqual(events_of(second.payloads)[2], token_event("a"))

    test.equal(serve.step(server), 0)
    test.equal(#second.payloads, 5)
  end
)

test.it(
  "a second turn on a live session takes its own sequence",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 4)

    local first = answered(server, FIRST_TURN)
    test.equal(first.choices[1].message.content, "aa")
    test.equal(first.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 2)
    test.equal(held(server, "s").positions, 4)

    local _, live_fn = serve.stream(
      server, (SECOND_TURN:gsub("}$", ',"stream":true}'))
    )
    test.truthy(live_fn)
    local watcher = begun(live_fn as tltp.StreamFn, nil, "A")
    test.equal(held(server, "s").length, 2)

    local second = answered(server, SECOND_TURN)
    test.equal(second.choices[1].message.content, "aa")
    test.equal(second.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").positions, 8)

    test.equal(serve.step(server), 1)
    test.equal(#watcher.payloads, 2)
    test.deepEqual(events_of(watcher.payloads)[2], token_event("a"))
    test.equal(serve.step(server), 0)
    test.equal(#watcher.payloads, 5)
  end
)

local function stream_text(watcher: Watcher): string
  local parts: {string} = {}
  for _, event in ipairs(events_of(watcher.payloads)) do
    local delta = event.choices[1].delta
    if not (delta is nil) and delta.content ~= nil then
      parts[#parts + 1] = delta.content as string
    end
  end
  return table.concat(parts)
end

test.it(
  "two turns advance in one pass and keep their own streams",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local server = build(file_path, "m", 2)
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local body_b = '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true}'

    local order: {string} = {}
    local _, a_fn = serve.stream(server, body_a)
    local _, b_fn = serve.stream(server, body_b)
    test.truthy(a_fn)
    test.truthy(b_fn)
    local a = begun(a_fn as tltp.StreamFn, order, "A")
    local b = begun(b_fn as tltp.StreamFn, order, "B")
    test.equal(coroutine.status(a.co), "suspended")
    test.equal(coroutine.status(b.co), "suspended")
    test.deepEqual(events_of(a.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})
    test.deepEqual(order, {"A", "B"})

    local check = build(file_path, "check", 2)
    local wanted_a = answered(check, body_a).choices[1].message.content
    local wanted_b = answered(check, body_b).choices[1].message.content
    local first_a = wanted_a:sub(1, 1)
    local first_b = wanted_b:sub(1, 1)

    test.equal(serve.step(server), 2)
    test.deepEqual(order, {"A", "B", "A", "B"})
    test.deepEqual(events_of(a.payloads), {ROLE_EVENT, token_event(first_a)})
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT, token_event(first_b)})

    test.equal(serve.step(server), 0)
    test.deepEqual(order, {"A", "B", "A", "B", "A", "A", "A", "B", "B", "B"})
    test.deepEqual(
      events_of(a.payloads),
      {
        ROLE_EVENT,
        token_event(first_a),
        token_event(wanted_a:sub(2, 2)),
        finish_event("length"),
      }
    )
    test.equal(a.payloads[5], "[DONE]")
    test.deepEqual(
      events_of(b.payloads),
      {
        ROLE_EVENT,
        token_event(first_b),
        token_event(wanted_b:sub(2, 2)),
        finish_event("length"),
      }
    )
    test.equal(b.payloads[5], "[DONE]")
    test.equal(stream_text(a), wanted_a)
    test.equal(stream_text(b), wanted_b)
  end
)

test.it(
  "a turn waits for a sequence and starts when one frees",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local body_b = '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true}'

    local server = build(file_path, "m", 1)
    local model = serve.model_of(server) as serve.Model
    test.equal(model.generation.sequences, 1)

    local _, a_fn = serve.stream(server, body_a)
    local a = begun(a_fn as tltp.StreamFn, nil, "A")
    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(a.payloads)[2], token_event("a"))

    local _, b_fn = serve.stream(server, body_a)
    local b = begun(b_fn as tltp.StreamFn, nil, "B")
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})

    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})
    test.equal(#a.payloads, 5)

    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(b.payloads)[2], token_event("a"))
    test.equal(serve.step(server), 0)
    test.equal(#b.payloads, 5)

    local stored = build(file_path, "stored", 1)
    local _, s_fn = serve.stream(stored, streamed("s"))
    begun(s_fn as tltp.StreamFn, nil, "S")
    test.equal(serve.step(stored), 1)
    test.equal(serve.step(stored), 0)
    test.truthy(serve.session_of(stored, "s"))

    local _, evict_fn = serve.stream(stored, body_a)
    test.truthy(evict_fn)
    test.equal(serve.session_of(stored, "s"), nil)
    local evict = begun(evict_fn as tltp.StreamFn, nil, "E")
    test.equal(serve.step(stored), 1)
    test.deepEqual(events_of(evict.payloads)[2], token_event("a"))

    local third = build(file_path, "third", 2)
    local _, third_a = serve.stream(third, body_a)
    local _, third_b_fn = serve.stream(third, body_b)
    local a_live = begun(third_a as tltp.StreamFn, nil, "A")
    local b_live = begun(third_b_fn as tltp.StreamFn, nil, "B")
    test.equal(serve.step(third), 2)
    test.equal(#a_live.payloads, 2)
    test.equal(#b_live.payloads, 2)

    local _, third_c = serve.stream(third, body_a)
    local _, third_d = serve.stream(third, body_b)
    local c = begun(third_c as tltp.StreamFn, nil, "C")
    local d = begun(third_d as tltp.StreamFn, nil, "D")
    test.deepEqual(events_of(c.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(d.payloads), {ROLE_EVENT})

    test.equal(serve.step(third), 2)
    test.deepEqual(events_of(c.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(d.payloads), {ROLE_EVENT})
    test.equal(#a_live.payloads, 5)
    test.equal(#b_live.payloads, 5)

    test.equal(serve.step(third), 2)
    test.deepEqual(events_of(c.payloads)[2], token_event("a"))
    test.deepEqual(events_of(d.payloads)[2], token_event("c"))

    local full = build(file_path, "full", 1)
    local _, storing_fn = serve.stream(full, streamed("s"))
    local storing = begun(storing_fn as tltp.StreamFn, nil, "S")
    test.equal(serve.step(full), 1)

    local _, waiting_fn = serve.stream(full, body_a)
    local waiting = begun(waiting_fn as tltp.StreamFn, nil, "W")
    test.deepEqual(events_of(waiting.payloads), {ROLE_EVENT})
    test.equal(serve.step(full), 1)
    test.equal(#storing.payloads, 5)
    test.truthy(serve.session_of(full, "s"))
    test.deepEqual(events_of(waiting.payloads), {ROLE_EVENT})

    test.equal(serve.step(full), 1)
    test.deepEqual(events_of(waiting.payloads)[2], token_event("a"))
    test.equal(serve.session_of(full, "s"), nil)
  end
)

test.it(
  "a reply that cannot wait is told so, a stored conversation keeps its sequence and a discarded one frees it",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local plain = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2}'

    local live = build(file_path, "live", 1)
    local _, live_fn = serve.stream(live, body_a)
    local watcher = begun(live_fn as tltp.StreamFn, nil, "A")
    local before = #watcher.payloads
    local reply = serve.answer(live, plain)
    test.equal(reply.status, "TooManyRequests")
    local decoded = tltp.json_decode(reply.body) as ErrorBody
    test.equal(decoded.error.type, "too_many_requests")
    test.includes(decoded.error.message, "sessions")
    test.equal(serve.session_of(live, "s"), nil)
    test.equal(#watcher.payloads, before)

    local one = build(file_path, "one", 4)
    local first = answered(one, FIRST_TURN)
    test.equal(first.usage.prompt_tokens_details.cached_tokens, 0)
    local kept = held(one, "s")
    test.equal(kept.length, 2)
    test.equal(kept.positions, 4)
    test.equal(kept.sequence, "s")

    local second = answered(one, SECOND_TURN)
    test.equal(second.usage.prompt_tokens_details.cached_tokens, 2)
    test.equal(held(one, "s").length, 4)
    test.equal(held(one, "s").sequence, "s")

    local two = build(file_path, "two", 2)
    answered(two, FIRST_TURN)
    local _, copy_fn = serve.stream(two, streaming(FIRST_TURN))
    begun(copy_fn as tltp.StreamFn, nil, "A")
    test.equal(serve.step(two), 1)

    local fresh = answered(two, FIRST_TURN)
    test.equal(fresh.usage.prompt_tokens_details.cached_tokens, 0)
    local other = held(two, "s").sequence
    test.notEqual(other, "s")

    test.equal(serve.step(two), 0)
    test.equal(held(two, "s").sequence, "s")
    local generation = (serve.model_of(two) as serve.Model).generation
    test.equal(generation:cache_bytes(other, "blk.0."), nil)
    test.truthy(generation:cache_bytes("s", "blk.0."))

    local bad = build(unreadable_fixture(), "bad", 1)
    answered(bad, FIRST_TURN)
    test.equal(held(bad, "s").length, 2)
    local failing = '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"a"},'
      .. '{"role":"user","content":"a"}],"max_tokens":2,'
      .. '"temperature":1,"top_k":2,"seed":1000,"session_id":"s"}'
    local lost_reply = serve.answer(bad, failing)
    test.equal(lost_reply.status, "InternalServerError")
    test.equal(serve.session_of(bad, "s"), nil)
    local bad_generation = (serve.model_of(bad) as serve.Model).generation
    test.equal(bad_generation:cache_bytes("s", "blk.0."), nil)
    test.equal(serve.answer(bad, plain).status, "OK")
  end
)

test.it(
  "a model the server cannot build is a start failure",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local endpoint: serve.Endpoint = {
      host = "127.0.0.1",
      port = 9090,
      name = "m",
      path = file_path,
      sessions = 4,
      kv_cache = "q8_0",
    }
    local server, line = serve.build(endpoint)
    test.equal(server, nil)
    test.includes(line as string, "q8_0")
    test.includes(line as string, "2")

    local written: {string} = {}
    local code = serve.start(
      endpoint, function(one: string) written[#written + 1] = one end
    )
    test.equal(code, 1)
    test.equal(#written, 1)
    test.includes(written[1], "q8_0")
    test.includes(written[1], "2")
    test.equal(written[1]:sub(-1), "\n")
    test.equal(written[1]:find("\n"), #written[1])
  end
)
