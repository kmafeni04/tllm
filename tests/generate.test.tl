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

local function with_meta(meta: {Meta}, entry: Meta): {Meta}
  local result = without_meta(meta, entry.name)
  result[#result + 1] = entry
  return result
end

local function pieces_meta(pieces: {string}): {Meta}
  local kinds: {integer} = {}
  for index = 1, #pieces do
    kinds[index] = 1
  end
  local meta = with_meta(t001_meta(), mstrings("tokenizer.ggml.tokens", pieces))
  meta = with_meta(meta, mint_array("tokenizer.ggml.token_type", kinds))
  return without_meta(meta, "tokenizer.ggml.eos_token_id")
end

local function replaced(
  entries: {Entry}, name: string, replacement: Entry
): {Entry}
  local result: {Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name == name then
      result[#result + 1] = replacement
    else
      result[#result + 1] = entry
    end
  end
  return result
end

local function repeated(values: {number}, times: integer): {number}
  local made: {number} = {}
  for _ = 1, times do
    for _, value in ipairs(values) do
      made[#made + 1] = value
    end
  end
  return made
end

local function linear_entries(prefix: string, out: {number}): {Entry}
  local LN3 = math.log(3)
  return {
    f32(prefix .. "attn_norm.weight", {2}, {1, 1}),
    f32(
      prefix .. "attn_qkv.weight",
      {2, 12},
      {
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        3,
        4,
        0,
        0,
        0,
        0,
        0,
        1,
        5,
        100,
        0,
        0,
        100,
        0,
        0,
        0,
        200,
      }
    ),
    f32(prefix .. "attn_gate.weight", {2, 4}, repeated({100, 0}, 4)),
    f32(prefix .. "ssm_beta.weight", {2, 2}, repeated({LN3, 0}, 2)),
    f32(prefix .. "ssm_alpha.weight", {2, 2}, repeated({1, 0}, 2)),
    f32(prefix .. "ssm_dt.bias", {2}, {LN3, 0}),
    f32(prefix .. "ssm_a", {2}, {-1, -1}),
    f32(prefix .. "ssm_conv1d.weight", {2, 12}, repeated({0.5, 1}, 12)),
    f32(prefix .. "ssm_norm.weight", {2}, {1, 1}),
    f32(prefix .. "ssm_out.weight", {4, 2}, out),
  }
end

local function linear_block(prefix: string, out: {number}): {Entry}
  return both(linear_entries(prefix, out), feed_forward_entries(prefix))
end

-- the mixed fixture grown with this file's tokenizer: its linear block turns
-- token zero at position zero into zeros the run divides by, so the cases that
-- read scores walk the ids this fixture answers with numbers
local function mixed_meta(): {Meta}
  local meta = with_integer(t001_meta(), "qwen35moe.block_count", 2)
  meta = with_integer(meta, "qwen35moe.full_attention_interval", 2)
  meta = with_integer(meta, "qwen35moe.ssm.group_count", 2)
  meta = with_integer(meta, "qwen35moe.ssm.state_size", 2)
  meta = with_integer(meta, "qwen35moe.ssm.time_step_rank", 2)
  meta = with_integer(meta, "qwen35moe.ssm.inner_size", 4)
  meta = with_integer(meta, "qwen35moe.ssm.conv_kernel", 2)
  meta = with_integer(meta, "qwen35moe.context_length", 8)
  return without_meta(meta, "tokenizer.ggml.eos_token_id")
end

local function mixed_entries(): {Entry}
  local entries = both(
    linear_block("blk.0.", {1, 0, 1, 0, 0, 1, 0, 1}), full_block("blk.1.")
  )
  entries = replaced(
    entries,
    "blk.1.attn_output.weight",
    f32("blk.1.attn_output.weight", {2, 2}, {1, 0, 0, 1})
  )
  return both(
    entries,
    {
      f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, 1, 1}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 3}, {0, 1, 1, 0.636, -1, -1}),
    }
  )
end

local FOUR_PIECES: {string} = {"a", "b", "c", "d"}

local function four_entries(output: {number}): {Entry}
  return both(
    full_block("blk.0."),
    {
      f32("token_embd.weight", {2, 4}, {1, 0, 1, 0, 1, 0, 1, 0}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 4}, output),
    }
  )
end

local function wide_pieces(): {string}
  local pieces: {string} = {"a", "b", "c"}
  for id = 3, 199999 do
    pieces[#pieces + 1] = tostring(id)
  end
  return pieces
end

local function wide_entries(): {Entry}
  local rows: {number} = {}
  for _ = 1, 200000 do
    rows[#rows + 1] = 1
    rows[#rows + 1] = 0
  end
  return both(
    full_block("blk.0."),
    {
      f32("token_embd.weight", {2, 200000}, rows),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 200000}, rows),
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
    test.equal((answer as generate.Text).text, "aa")

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
    test.equal((answer as generate.Text).text, "a")

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
    test.equal((answer as generate.Text).text, "ccc")

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
    test.equal((answer as generate.Text).text, "aac")

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
    test.equal((answer as generate.Text).text, "c")

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
    test.equal((answer as generate.Text).text, "b")

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
    test.equal((answer as generate.Text).text, "c")

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
    test.equal((answer as generate.Text).text, "aa")
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
    test.equal((answer as generate.Text).text, "")

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
    test.equal((answer as generate.Text).text, "aa")

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
    test.equal((answer as generate.Text).text, "aa")
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
  local pass = model.pass
  model.pass = function(
    self: compute.Model, inputs: {compute.Input}
  ): {{number}} | nil, string | nil
    for _, input in ipairs(inputs) do
      seen.ids[#seen.ids + 1] = input.token
      seen.positions[#seen.positions + 1] = input.position
    end
    return pass(self, inputs)
  end
  return seen
end

local function advance(
  model: compute.Model, run: generate.Run
): string | nil, string | nil
  local given = run.next(run, 1)
  if given is nil then
    return nil, nil
  end
  local rows, line = model:pass(given)
  if rows is nil then
    run.fail(run, line as string)
    return nil, line
  end
  local planes = rows as {{number}}
  return run.accept(run, planes[#planes])
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
  local reserved = model:reserve("", model.positions)
  if not (reserved is nil) then
    test.equal(reserved:find("is already held", 1, true), nil)
  end
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
    local piece, line = advance(wide, run)
    test.equal(piece, "a")
    test.equal(line, nil)
    test.deepEqual(wide_seen.ids, {0})
    test.deepEqual(wide_seen.positions, {0})

    piece, line = advance(wide, run)
    test.equal(piece, "a")
    test.equal(line, nil)
    test.deepEqual(run.produced, {0, 0})
    test.equal(run.done, true)
    test.deepEqual(wide_seen.ids, {0, 0})
    test.deepEqual(wide_seen.positions, {0, 1})

    local narrow = fresh(handle, 2)
    local narrow_seen = spied(narrow)
    local pair = opened(narrow, tokens, "aa", {max_tokens = 1, seed = 0})
    piece, line = advance(narrow, pair)
    test.equal(piece, nil)
    test.equal(line, nil)
    piece, line = advance(narrow, pair)
    test.equal(piece, "a")
    test.equal(line, nil)
    test.deepEqual(narrow_seen.ids, {0, 0})
    test.deepEqual(narrow_seen.positions, {0, 1})
    test.equal(pair.done, true)
    test.deepEqual(pair.produced, {0})

    local cramped = fresh(handle, 2)
    local cramped_seen = spied(cramped)
    local over = opened(cramped, tokens, "a", {max_tokens = 3, seed = 0})
    piece, line = advance(cramped, over)
    test.equal(piece, "a")
    test.equal(line, nil)
    piece, line = advance(cramped, over)
    test.equal(piece, "a")
    test.equal(line, nil)
    piece, line = advance(cramped, over)
    test.equal(piece, nil)
    test.includes(line as string, "blk.0.")
    test.includes(line as string, "2")
    test.equal(line, over.failure)
    test.equal(over.done, true)
    test.deepEqual(cramped_seen.positions, {0, 1, 2})
    piece, line = advance(cramped, over)
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

local function picked(
  handle: weights.Handle,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: generate.Options
): {integer}
  local answer, line = generate.text(fresh(handle, 4), tokens, prompt, options)
  test.equal(line, nil)
  return (answer as generate.Text).sampled
end

test.it(
  "a wide row is sampled without ordering it",
  function()
    local handle = loaded(pieces_meta(wide_pieces()), wide_entries())
    local tokens = tokens_of(handle)

    local narrow = fresh(handle, 2)
    local answer, line = generate.text(
      narrow,
      tokens,
      "a",
      {
        max_tokens = 1,
        seed = 0,
        temperature = 1,
        top_k = 40,
      }
    )
    test.equal(line, nil)
    test.deepEqual((answer as generate.Text).sampled, {9})
    test.truthy(narrow.measure.sampling < 0.05)

    local wide = fresh(handle, 2)
    answer, line = generate.text(
      wide,
      tokens,
      "a",
      {
        max_tokens = 1,
        seed = 0,
        temperature = 1,
      }
    )
    test.equal(line, nil)
    test.deepEqual((answer as generate.Text).sampled, {47213})
    test.truthy(wide.measure.sampling < 0.10)
  end
)

test.it(
  "a run with no top_k walks the ids in order",
  function()
    local handle = loaded(
      pieces_meta(FOUR_PIECES), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)

    test.deepEqual(
      picked(
        handle,
        tokens,
        "a",
        {
          max_tokens = 1,
          seed = 2178,
          temperature = 1,
        }
      ),
      {2}
    )
    test.deepEqual(
      picked(
        handle,
        tokens,
        "a",
        {
          max_tokens = 1,
          seed = 1946,
          temperature = 1,
        }
      ),
      {3}
    )
  end
)

test.it(
  "top-p and min-p narrow the candidates",
  function()
    local handle = loaded(
      pieces_meta(FOUR_PIECES), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)

    local function drew(options: generate.Options): {integer}
      return picked(handle, tokens, "a", options)
    end

    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1946,
          temperature = 1,
          top_k = 4,
        }
      ),
      {0}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1946,
          temperature = 1,
          top_k = 4,
          top_p = 0.8,
        }
      ),
      {2}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1946,
          temperature = 1,
          top_k = 4,
          top_p = 0.01,
        }
      ),
      {3}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1791,
          temperature = 1,
          top_p = 0.8,
        }
      ),
      {2}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1791,
          temperature = 1,
        }
      ),
      {3}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1868,
          temperature = 1,
          top_k = 4,
          min_p = 0.1,
        }
      ),
      {2}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1868,
          temperature = 1,
          top_k = 4,
        }
      ),
      {1}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 1946,
          temperature = 1,
          top_k = 4,
          min_p = 1,
        }
      ),
      {3}
    )
    test.deepEqual(
      drew(
        {
          max_tokens = 1,
          seed = 2178,
          temperature = 1,
          min_p = 0.5,
        }
      ),
      {3}
    )
  end
)

test.it(
  "the penalties steer off what the run already holds",
  function()
    local handle = loaded(
      pieces_meta(FOUR_PIECES), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)
    local negative = loaded(
      pieces_meta(FOUR_PIECES), four_entries({-1, 0, -2, 0, -4, 0, -8, 0})
    )
    local negative_tokens = tokens_of(negative)

    local function drew(prompt: string, options: generate.Options): {integer}
      return picked(handle, tokens, prompt, options)
    end

    test.deepEqual(drew("d", {max_tokens = 1, seed = 0}), {3})
    test.deepEqual(
      drew("d", {max_tokens = 1, seed = 0, repetition_penalty = 1.6}), {2}
    )
    test.deepEqual(
      drew("dd", {max_tokens = 1, seed = 0, repetition_penalty = 1.6}), {2}
    )
    test.deepEqual(
      drew("d", {max_tokens = 1, seed = 0, presence_penalty = 1}), {3}
    )
    test.deepEqual(
      drew("dd", {max_tokens = 1, seed = 0, presence_penalty = 1}), {3}
    )
    test.deepEqual(
      drew("d", {max_tokens = 1, seed = 0, frequency_penalty = 1}), {3}
    )
    test.deepEqual(
      drew("dd", {max_tokens = 1, seed = 0, frequency_penalty = 1}), {2}
    )
    test.deepEqual(
      drew(
        "d",
        {
          max_tokens = 1,
          seed = 1791,
          temperature = 1,
          top_k = 2,
          presence_penalty = 5,
        }
      ),
      {1}
    )
    test.deepEqual(
      drew(
        "d",
        {
          max_tokens = 1,
          seed = 0,
          repetition_penalty = 0.5,
          presence_penalty = 3,
        }
      ),
      {3}
    )
    test.deepEqual(
      picked(
        negative,
        negative_tokens,
        "a",
        {
          max_tokens = 1,
          seed = 0,
          repetition_penalty = 4,
        }
      ),
      {1}
    )
  end
)

test.it(
  "a reply is cut at the stop string it completes",
  function()
    local handle = loaded(
      pieces_meta(FOUR_PIECES), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)

    local answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 3,
        seed = 0,
        stop = {"d"},
      }
    )
    test.equal(line, nil)
    test.equal((answer as generate.Text).text, "")
    test.deepEqual((answer as generate.Text).sampled, {3})

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 3,
        seed = 0,
        stop = {"dd"},
      }
    )
    test.equal(line, nil)
    test.equal((answer as generate.Text).text, "")
    test.deepEqual((answer as generate.Text).sampled, {3, 3})

    answer, line = generate.text(
      fresh(handle, 4),
      tokens,
      "d",
      {
        max_tokens = 3,
        seed = 0,
        stop = {"d"},
      }
    )
    test.equal(line, nil)
    test.equal((answer as generate.Text).text, "")
    test.deepEqual((answer as generate.Text).sampled, {3})

    local inside = loaded(
      pieces_meta({"a", "b", "c", "da"}), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local inside_tokens = tokens_of(inside)
    answer, line = generate.text(
      fresh(inside, 4),
      inside_tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        stop = {"d"},
      }
    )
    test.equal(line, nil)
    test.equal((answer as generate.Text).text, "")
    test.deepEqual((answer as generate.Text).sampled, {3})
  end
)

test.it(
  "a run holds back what a stop string could still complete",
  function()
    local handle = loaded(
      pieces_meta(FOUR_PIECES), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)

    local model = fresh(handle, 4)
    local run = opened(
      model,
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        stop = {"ddx"},
      }
    )
    local piece, line = advance(model, run)
    test.equal(piece, nil)
    test.equal(line, nil)
    piece, line = advance(model, run)
    test.equal(piece, "dd")
    test.equal(line, nil)
    test.equal(run.done, true)
    test.deepEqual(run.produced, {3, 3})

    local answer, answer_line = generate.text(
      fresh(handle, 4),
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        stop = {"ddx"},
      }
    )
    test.equal(answer_line, nil)
    test.equal((answer as generate.Text).text, "dd")
  end
)

test.it(
  "a run holds back the shorter prefix of a longer stop string",
  function()
    local handle = loaded(
      pieces_meta({"a", "b", "c", "xa"}), four_entries({0, 0, 1, 0, 2, 0, 3, 0})
    )
    local tokens = tokens_of(handle)
    local model = fresh(handle, 4)
    local run = opened(
      model,
      tokens,
      "a",
      {
        max_tokens = 2,
        seed = 0,
        stop = {"abc"},
      }
    )
    local piece, line = advance(model, run)
    test.equal(line, nil)
    test.equal(piece, nil)
  end
)

local function bounded(handle: weights.Handle, bound: integer): compute.Model
  local model, failure = compute.model(backend, handle, nil, "f32", 1, bound)
  test.equal(failure, nil)
  return model as compute.Model
end

test.it(
  "a run prefills its prompt in chunks and answers the same",
  function()
    local handle = loaded(mixed_meta(), mixed_entries())
    local tokens = tokens_of(handle)
    local ids = {1, 2, 1, 2}
    local options: generate.Options = {max_tokens = 2, seed = 7}

    local function reply(bound: integer): generate.Text, integer, integer
      local model = bounded(handle, bound)
      local made, line = generate.text_ids(model, tokens, ids, options)
      test.equal(line, nil)
      local answer = made as generate.Text
      return answer, model.measure.passes, model.measure.tokens
    end

    local single, single_passes, single_tokens = reply(1)
    local pair, pair_passes, pair_tokens = reply(2)
    local whole, whole_passes, whole_tokens = reply(8)

    test.deepEqual(pair.sampled, single.sampled)
    test.equal(pair.text, single.text)
    test.deepEqual(whole.sampled, single.sampled)
    test.equal(whole.text, single.text)
    test.deepEqual({single_passes, pair_passes, whole_passes}, {5, 3, 2})
    test.equal(single_tokens, 5)
    test.equal(pair_tokens, 5)
    test.equal(whole_tokens, 5)

    local walked = bounded(handle, 1)
    test.equal(walked:reserve("", walked.positions), nil)
    local prompt_rows: {{number}} = {}
    for position = 1, #ids do
      local made, line = walked:pass(
        {
          {
            sequence = "",
            token = ids[position],
            position = position - 1,
          },
        }
      )
      test.equal(line, nil)
      prompt_rows[position] = (made as {{number}})[1]
    end

    -- the id a run draws with these options off the scores it samples from:
    -- the prompt's last row for the run itself, another row for a run that
    -- sampled from earlier in its group
    local function drawn(from: integer): integer
      local made, line = generate.open_ids(
        bounded(handle, 1), tokens, ids, options
      )
      test.equal(line, nil)
      local held = made as generate.Run
      for position = 1, #ids do
        local rows = held.next(held, 1) as {compute.Input}
        test.equal(#rows, 1)
        local scores = prompt_rows[position]
        if position == #ids then
          scores = prompt_rows[from]
        end
        held.accept(held, scores)
      end
      test.equal(#held.produced, 1, "one id drawn per prompt")
      return held.produced[1]
    end

    local from_last = drawn(#ids)
    local from_first = drawn(1)
    test.equal(
      from_first == from_last,
      false,
      "first "
        .. table.concat(prompt_rows[1], ",")
        .. " last "
        .. table.concat(prompt_rows[#ids], ",")
        .. " drawn "
        .. from_first
        .. "/"
        .. from_last
    )
    test.equal(single.sampled[1], from_last)

    local split = bounded(handle, 2)
    test.equal(split:reserve("", split.positions), nil)
    local made, made_line = generate.open_ids(split, tokens, ids, options)
    test.equal(made_line, nil)
    local run = made as generate.Run
    local head = run.next(run, 1) as {compute.Input}
    test.equal(#head, 1)
    local rest = run.next(run, 8) as {compute.Input}
    test.equal(#rest, 3)
    test.equal(run.next(run, 8), nil, "a group still owed hands no more rows")

    local head_scores, head_line = split:pass(head)
    test.equal(head_line, nil)
    test.equal(#(head_scores as {{number}}), 1)
    local tail_scores, tail_line = split:pass(rest)
    test.equal(tail_line, nil)
    local planes = tail_scores as {{number}}
    test.equal(#planes, 3)
    local _, accept_line = run.accept(run, planes[#planes])
    test.equal(accept_line, nil)
    test.equal(run.produced[1], single.sampled[1])
  end
)
