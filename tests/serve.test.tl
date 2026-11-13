local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local cli = require("tllm.cli")
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

local record Usage
  prompt_tokens: integer
  completion_tokens: integer
  total_tokens: integer
end

local record ChatReply
  choices: {Choice}
  usage: Usage
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

local function tokenizer_meta(): {Meta}
  return {
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", {"a"}),
    mint_array("tokenizer.ggml.token_type", {1}),
    mstrings("tokenizer.ggml.merges", {}),
  }
end

local function one_tensor_gguf(): string
  return meta_bytes(tokenizer_meta(), {f32("a", {2, 3}, zeros(6))})
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

local function build(file_path: string, name: string): tltp.Server
  local server, failure = serve.build(
    {host = "127.0.0.1", port = 9090, name = name, path = file_path}
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
    local file_path = fixture(one_tensor_gguf())
    local server = build(file_path, "m")
    test.equal(server.config.port, 9090)
    test.equal(server.config.bind_host, "127.0.0.1")

    local model = serve.model_of(server) as serve.Model
    test.truthy(model)
    test.equal(model.path, file_path)
    test.equal(model.name, "m")
    test.truthy(model.handle.tensors["a"])

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
      {host = "127.0.0.1", port = 9090, name = "m", path = not_gguf}
    )
    test.equal(bad, nil)
    test.includes(reason as string, not_gguf)

    local missing = ROOT .. "/missing.gguf"
    local absent, absent_reason = serve.build(
      {host = "127.0.0.1", port = 9090, name = "m", path = missing}
    )
    test.equal(absent, nil)
    test.includes(absent_reason as string, missing)
  end
)

test.it(
  "unrouted paths answer json 404",
  function()
    local server = build(fixture(one_tensor_gguf()), "m")
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
    local response = build(fixture(one_tensor_gguf()), "m"):mock_request(
      "/health", {method = "post"}
    )
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
    local server = build(fixture(one_tensor_gguf()), "m")
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
      {host = "127.0.0.1", port = 9090, name = "m", path = file_path},
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
      {prompt_tokens = 1, completion_tokens = 256, total_tokens = 257}
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

    local unmeasured = build(
      meta_file(
        without_meta(t001_meta(), "qwen35moe.embedding_length"), t001_entries()
      ),
      "m"
    )
    local unmeasured_reply = refused(unmeasured, A_BODY)
    test.includes(unmeasured_reply.error.message, "model")
    test.includes(unmeasured_reply.error.message, "embedding_length")
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
      {host = "127.0.0.1", port = 9090, name = "m", path = no_tokens}
    )
    test.equal(refused_build, nil)
    test.includes(line as string, "tokenizer.ggml.model")
  end
)
