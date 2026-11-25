-- the serve test layer both tests/serve.test.tl and tests/serve_pool.test.tl
-- build on: the fixture files, the reply shapes and the helpers that drive a
-- server or a stream. Every body is the one the suite already ran.
local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local serve = require("tllm.serve")
local tltp = require("tltp")
local utf8lib = require("utf8lib")

local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, zeros = support.padded, support.f32, support.zeros
local ones, mint, mfloat, mtext =
  support.ones, support.mint, support.mfloat, support.mtext
local mstrings, mint_array, both, without_meta =
  support.mstrings, support.mint_array, support.both, support.without_meta

local record serve_fixtures
  record LibC
    usleep: function(integer): integer
    kill: function(integer, integer): integer
  end
  record HealthBody
    status: string
    model: string
  end

  record ErrorDetail
    message: string
    type: string
  end

  record ErrorBody
    error: ErrorDetail
  end

  record ReplyFunction
    name: string
    arguments: string
  end

  record ReplyCall
    id: string
    type: string
    ["function"]: ReplyFunction
  end

  record ReplyMessage
    role: string
    content: string
    tool_calls: {ReplyCall} | nil
  end

  record Choice
    index: integer
    message: ReplyMessage
    finish_reason: string | nil
  end

  record PromptDetails
    cached_tokens: integer
  end

  record Usage
    prompt_tokens: integer
    completion_tokens: integer
    total_tokens: integer
    prompt_tokens_details: PromptDetails
  end

  record ChatReply
    choices: {Choice}
    usage: Usage
  end

  record StreamFunction
    name: string | nil
    arguments: string | nil
  end

  record StreamCall
    index: integer
    id: string | nil
    type: string | nil
    ["function"]: StreamFunction
  end

  record StreamDelta
    role: string | nil
    content: string | nil
    tool_calls: {StreamCall} | nil
  end

  record StreamChoice
    index: integer
    delta: StreamDelta | nil
    finish_reason: string | nil
  end

  record StreamEvent
    choices: {StreamChoice}
  end

  ROOT: string
  fixture: function(bytes: string): string
  meta_bytes: function(meta: {Meta}, entries: {Entry}): string
  meta_file: function(meta: {Meta}, entries: {Entry}): string
  replaced: function(meta: {Meta}, entry: Meta): {Meta}
  t001_meta: function(): {Meta}
  t001_entries: function(): {Entry}
  spelled: function(string): string
  A_BODY: string
  CALL_BODY: string
  CALL: string
  first_token: function(string): tltp.Server
  two_pieces: function(string, string): tltp.Server
  driven: function(tltp.Server, tltp.StreamFn): {string}
  DEVICE_LINE: string
  answered: function(server: tltp.Server, body: string): ChatReply
  build: function(
    file_path: string,
    name: string,
    sessions?: integer,
    cache?: compute.CacheType,
    context_path?: string,
    context_bytes?: integer,
    workers?: integer,
    ubatch?: integer,
    threads?: integer
  ): tltp.Server
  temp_dir: function(): string
  record Watcher
    co: thread
    payloads: {string}
  end

  begun: function(
    stream_fn: tltp.StreamFn, order: {string} | nil, tag: string
  ): Watcher
  events_of: function(payloads: {string}): {StreamEvent}
  ROLE_EVENT: StreamEvent
  token_event: function(piece: string): StreamEvent
  finish_event: function(reason: string): StreamEvent
  FIRST_TURN: string
  SECOND_TURN: string
  named: function(body: string, id: string): string
  held: function(server: tltp.Server, id: string): serve.Session
  unreadable_fixture: function(): string
  streamed: function(id: string): string
  streaming: function(body: string): string
  C: LibC
  POOL_BUDGET: integer
  workers_of: function(server: tltp.Server): {serve.Worker}
end

local type LibC = serve_fixtures.LibC
local type ChatReply = serve_fixtures.ChatReply
local type StreamEvent = serve_fixtures.StreamEvent
local type Watcher = serve_fixtures.Watcher

ffi.cdef(
  [[
int usleep(unsigned int usec);
int kill(int pid, int sig);
]]
)

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local C = ffi.C as LibC

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/s" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
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

local DEVICE_LINE = "^device: (.+), (%d+) bytes budget, (%d+) bytes held, (%d+) expert slots in (%d+) bytes, (%d+) bytes from the mapping$"

-- a reply text is stored in the vocabulary as the file's own byte symbols:
-- GPT-2 gives every byte it cannot print a codepoint of its own, and its
-- decoder turns that codepoint back into the byte
local SYMBOL: {string} = {}
do
  local displaced = 0
  for byte = 0, 255 do
    if
      (byte >= 0x21 and byte <= 0x7E)
      or (byte >= 0xA1 and byte <= 0xAC)
      or (byte >= 0xAE and byte <= 0xFF)
    then
      SYMBOL[byte + 1] = string.char(byte)
    else
      SYMBOL[byte + 1] = utf8lib.char(256 + displaced)
      displaced = displaced + 1
    end
  end
end

local function spelled(piece: string): string
  local parts: {string} = {}
  for index = 1, #piece do
    parts[index] = SYMBOL[(piece:byte(index) as integer) + 1]
  end
  return table.concat(parts)
end

local A_BODY = '{"messages":[{"role":"user","content":"a"}]}'
local CALL_BODY = '{"messages":[{"role":"user","content":"c"}],"max_tokens":1}'
local CALL = "Hi<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>done"

local function answered(server: tltp.Server, body: string): ChatReply
  local reply = serve.answer(server, body)
  test.equal(reply.status, "OK", body)
  return tltp.json_decode(reply.body) as ChatReply
end

local function build(
  file_path: string,
  name: string,
  sessions?: integer,
  cache?: compute.CacheType,
  context_path?: string,
  context_bytes?: integer,
  workers?: integer,
  ubatch?: integer,
  threads?: integer
): tltp.Server
  local server, failure = serve.build(
    {
      host = "127.0.0.1",
      port = 9090,
      name = name,
      path = file_path,
      sessions = sessions or 4,
      kv_cache = cache or "f32",
      context_path = context_path,
      context_bytes = context_bytes,
      workers = workers,
      ubatch = ubatch,
      threads = threads,
    }
  )
  test.equal(failure, nil, file_path)
  return server as tltp.Server
end

-- 49/T-001: a copy of 18's fixture whose token zero is the given text
local function first_token(reply: string): tltp.Server
  return build(
    meta_file(
      replaced(
        t001_meta(),
        mstrings("tokenizer.ggml.tokens", {spelled(reply), "b", "c"})
      ),
      t001_entries()
    ),
    "m"
  )
end

-- 50/T-001 and 50/T-002: 18's fixture with the three tokens `one`, `two` and
-- `c`, where the prompt picks id 0 and id 0 picks id 1
local function two_pieces(one: string, two: string): tltp.Server
  local meta = replaced(
    replaced(
      t001_meta(),
      mstrings("tokenizer.ggml.tokens", {spelled(one), spelled(two), "c"})
    ),
    mint("tokenizer.ggml.eos_token_id", 2)
  )
  local entries = support.replaced(
    support.replaced(
      t001_entries(),
      "token_embd.weight",
      f32("token_embd.weight", {2, 3}, {1, 0, 0, 1, 0, 1})
    ),
    "output.weight",
    f32("output.weight", {2, 3}, {0, 1, 1, 0, -1, -1})
  )
  return build(meta_file(meta, entries), "m")
end

local function temp_dir(): string
  created = created + 1
  local dir = ROOT .. "/d" .. tostring(created)
  fs.mkdirAll(dir)
  return dir
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
          if order is {string} then
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

local function events_of(payloads: {string}): {StreamEvent}
  local events: {StreamEvent} = {}
  for _, payload in ipairs(payloads) do
    if payload ~= "[DONE]" then
      events[#events + 1] = tltp.json_decode(payload) as StreamEvent
    end
  end
  return events
end

local function driven(server: tltp.Server, stream_fn: tltp.StreamFn): {string}
  local watcher = begun(stream_fn, nil, "A")
  while serve.step(server) > 0 do
  end
  return watcher.payloads
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

local function unreadable_fixture(): string
  return meta_file(
    replaced(
      t001_meta(), mstrings("tokenizer.ggml.tokens", {"a", "b", "\255"})
    ),
    t001_entries()
  )
end

local function streamed(id: string): string
  return '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
    .. '"stream":true,"session_id":"'
    .. id
    .. '"}'
end

local function streaming(body: string): string
  return (body:gsub("}$", ',"stream":true}'))
end

local POOL_BUDGET = 1048576

local function workers_of(server: tltp.Server): {serve.Worker}
  return (serve.pool_of(server) as serve.Pool).workers
end

serve_fixtures.ROOT = ROOT
serve_fixtures.C = C
serve_fixtures.fixture = fixture
serve_fixtures.meta_bytes = meta_bytes
serve_fixtures.meta_file = meta_file
serve_fixtures.replaced = replaced
serve_fixtures.t001_meta = t001_meta
serve_fixtures.t001_entries = t001_entries
serve_fixtures.spelled = spelled
serve_fixtures.A_BODY = A_BODY
serve_fixtures.CALL_BODY = CALL_BODY
serve_fixtures.CALL = CALL
serve_fixtures.first_token = first_token
serve_fixtures.two_pieces = two_pieces
serve_fixtures.driven = driven
serve_fixtures.DEVICE_LINE = DEVICE_LINE
serve_fixtures.answered = answered
serve_fixtures.build = build
serve_fixtures.temp_dir = temp_dir
serve_fixtures.begun = begun
serve_fixtures.events_of = events_of
serve_fixtures.ROLE_EVENT = ROLE_EVENT
serve_fixtures.token_event = token_event
serve_fixtures.finish_event = finish_event
serve_fixtures.FIRST_TURN = FIRST_TURN
serve_fixtures.SECOND_TURN = SECOND_TURN
serve_fixtures.named = named
serve_fixtures.held = held
serve_fixtures.unreadable_fixture = unreadable_fixture
serve_fixtures.streamed = streamed
serve_fixtures.streaming = streaming
serve_fixtures.POOL_BUDGET = POOL_BUDGET
serve_fixtures.workers_of = workers_of

return serve_fixtures
