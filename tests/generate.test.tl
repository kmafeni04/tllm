local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
local generate = require("tllm.generate")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

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

local function padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

local written = 0

local function fixture(meta: {Meta}, entries: {Entry}): string
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
  written = written + 1
  local location = ROOT .. "/generate" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
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

local function with_integer(meta: {Meta}, name: string, value: integer): {Meta}
  local result: {Meta} = {}
  local found = false
  for _, entry in ipairs(meta) do
    if entry.name == name then
      result[#result + 1] = mint(name, value)
      found = true
    else
      result[#result + 1] = entry
    end
  end
  if not found then
    result[#result + 1] = mint(name, value)
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

local function with_pieces(meta: {Meta}, pieces: {string}): {Meta}
  local result = without_meta(meta, "tokenizer.ggml.tokens")
  result[#result + 1] = mstrings("tokenizer.ggml.tokens", pieces)
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
    mint("qwen35moe.context_length", 4),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", {"a", "b", "c"}),
    mint_array("tokenizer.ggml.token_type", {1, 1, 1}),
    mstrings("tokenizer.ggml.merges", {}),
    mint("tokenizer.ggml.eos_token_id", 1),
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

local function full_block(prefix: string): {Entry}
  return both(attention_entries(prefix), feed_forward_entries(prefix))
end

local function t001_entries(): {Entry}
  return both(
    full_block("blk.0."),
    {
      f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, 1, 0}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
    }
  )
end

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function fresh(
  handle: weights.Handle, positions: integer | nil
): compute.Model
  local model, failure = compute.model(backend, handle, positions, "f32")
  test.equal(failure, nil)
  return model as compute.Model
end

local function tokens_of(handle: weights.Handle): tokenizer.Tokenizer
  local tokens, failure = tokenizer.load(handle.path)
  test.equal(failure, nil)
  return tokens as tokenizer.Tokenizer
end

test.it(
  "a prompt comes back as the text it sampled",
  function()
    local handle = loaded(t001_meta(), t001_entries())
    local tokens = tokens_of(handle)

    local answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "aa")

    answer, line = generate.text(
      fresh(handle, 2),
      tokens,
      "aa",
      {
        max_tokens = 1,
        seed = 0,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "a")

    answer, line = generate.text(
      fresh(handle, 2),
      tokens,
      "aaa",
      {
        max_tokens = 1,
        seed = 0,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "blk.0.")
    test.includes(line as string, "2")
  end
)

test.it(
  "a seeded draw takes the token temperature and top-k leave",
  function()
    local handle = loaded(
      without_meta(t001_meta(), "tokenizer.ggml.eos_token_id"), t001_entries()
    )
    local tokens = tokens_of(handle)

    local answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 3,
        seed = 1000,
        temperature = 1,
        top_k = 2,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "ccc")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 3,
        seed = 0,
        temperature = 1,
        top_k = 2,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "aac")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 1,
        seed = 1791,
        temperature = 1,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "b")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 1,
        seed = 1791,
        temperature = 1,
        top_k = 5,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "b")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 1,
        seed = 1791,
        temperature = 1,
        top_k = 2,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "c")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 1000,
        temperature = 0,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "aa")
  end
)

test.it(
  "generation stops at end of text or at the count",
  function()
    local ended = loaded(
      with_integer(t001_meta(), "tokenizer.ggml.eos_token_id", 0),
      t001_entries()
    )
    local ended_tokens = tokens_of(ended)

    local answer, line = generate.text(
      fresh(ended, 4),
      ended_tokens,
      "a",
      {
        max_tokens = 4,
        seed = 0,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "")

    answer, line = generate.text(
      fresh(ended, 4),
      ended_tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        end_id = 1,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "aa")

    local open = loaded(
      without_meta(t001_meta(), "tokenizer.ggml.eos_token_id"), t001_entries()
    )
    local open_tokens = tokens_of(open)
    answer, line = generate.text(
      fresh(open, 4),
      open_tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
      }
    )
    test.equal(line, nil)
    test.equal(answer, "aa")
  end
)

test.it(
  "a prompt and options that cannot be used are refused",
  function()
    local handle = loaded(t001_meta(), t001_entries())
    local tokens = tokens_of(handle)

    local answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "",
      {
        max_tokens = 2,
        seed = 0,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "prompt")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 0,
        seed = 0,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "max_tokens")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        temperature = -1,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "temperature")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        top_k = 0,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "top_k")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        start = -1,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "start")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        start = 1,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "start")

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a\255",
      {
        max_tokens = 0,
        seed = 0,
        temperature = -1,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "generate")
    test.includes(line as string, "max_tokens")

    answer, line = generate.text(
      fresh(handle, 2),
      tokens,
      "a",
      {
        max_tokens = 3,
        seed = 0,
      }
    )
    test.equal(answer, nil)
    test.includes(line as string, "blk.0.")
    test.includes(line as string, "2")

    local broken = loaded(
      with_pieces(t001_meta(), {"a", "b", "\255"}), t001_entries()
    )
    local broken_tokens = tokens_of(broken)
    local _, decode_line = broken_tokens:decode({2})
    answer, line = generate.text(
      fresh(broken, 4),
      broken_tokens,
      "a",
      {
        max_tokens = 1,
        seed = 1000,
        temperature = 1,
        top_k = 2,
      }
    )
    test.equal(answer, nil)
    test.equal(line, decode_line)

    local _, encode_line = tokens:encode("a\255")
    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a\255",
      {
        max_tokens = 2,
        seed = 0,
      }
    )
    test.equal(answer, nil)
    test.equal(line, encode_line)
  end
)

local record Seen
  ids: {integer}
  positions: {integer}
end

local function spied(model: compute.Model): Seen
  local seen: Seen = {ids = {}, positions = {}}
  local forward = model.forward
  model.forward = function(
    self: compute.Model, token: integer, position: integer
  ): {number} | nil, string | nil
    seen.ids[#seen.ids + 1] = token
    seen.positions[#seen.positions + 1] = position
    return forward(self, token, position)
  end
  return seen
end

local function opened(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: generate.Options
): generate.Run
  local run, line = generate.open(model, tokens, prompt, options)
  test.equal(line, nil)
  test.truthy(run)
  return run as generate.Run
end

test.it(
  "a run advances one forward at a time and stops where it must",
  function()
    local handle = loaded(t001_meta(), t001_entries())
    local tokens = tokens_of(handle)

    local wide = fresh(handle, 4)
    local wide_seen = spied(wide)
    local run = opened(wide, tokens, "a", {max_tokens = 2, seed = 0})
    local piece, line = run.step(run)
    test.equal(piece, nil)
    test.equal(line, nil)
    test.deepEqual(wide_seen.ids, {0})
    test.deepEqual(wide_seen.positions, {0})

    piece, line = run.step(run)
    test.equal(piece, "a")
    test.equal(line, nil)
    piece, line = run.step(run)
    test.equal(piece, "a")
    test.equal(line, nil)
    test.deepEqual(run.produced, {0, 0})
    test.equal(run.done, true)
    test.deepEqual(wide_seen.positions, {0, 1})

    local narrow = fresh(handle, 2)
    local narrow_seen = spied(narrow)
    local pair = opened(narrow, tokens, "aa", {max_tokens = 1, seed = 0})
    piece, line = pair.step(pair)
    test.equal(piece, nil)
    test.equal(line, nil)
    piece, line = pair.step(pair)
    test.equal(piece, nil)
    test.equal(line, nil)
    test.deepEqual(narrow_seen.ids, {0, 0})
    test.deepEqual(narrow_seen.positions, {0, 1})
    piece, line = pair.step(pair)
    test.equal(piece, "a")
    test.equal(line, nil)
    test.equal(pair.done, true)
    test.deepEqual(pair.produced, {0})

    local cramped = fresh(handle, 2)
    local cramped_seen = spied(cramped)
    local over = opened(cramped, tokens, "a", {max_tokens = 3, seed = 0})
    piece, line = over.step(over)
    test.equal(piece, nil)
    test.equal(line, nil)
    piece, line = over.step(over)
    test.equal(piece, "a")
    test.equal(line, nil)
    piece, line = over.step(over)
    test.equal(piece, nil)
    test.includes(line as string, "blk.0.")
    test.includes(line as string, "2")
    test.equal(line, over.failure)
    test.equal(over.done, true)
    test.deepEqual(cramped_seen.positions, {0, 1, 2})
    piece, line = over.step(over)
    test.equal(piece, nil)
    test.equal(line, nil)
    test.equal(#cramped_seen.positions, 3)

    local refused, refused_line = generate.open(
      fresh(handle, 4), tokens, "a", {max_tokens = 0, seed = 0}
    )
    test.equal(refused, nil)
    test.includes(refused_line as string, "max_tokens")

    refused, refused_line = generate.open(
      fresh(handle, 4), tokens, "", {max_tokens = 2, seed = 0}
    )
    test.equal(refused, nil)
    test.includes(refused_line as string, "prompt")
  end
)
