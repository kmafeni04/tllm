local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local clock = require("tllm.clock")
local compute = require("tllm.compute")
local generate = require("tllm.generate")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, zeros = support.padded, support.f32, support.zeros
local ones, mint, mfloat, mtext =
  support.ones, support.mint, support.mfloat, support.mtext
local mstrings, mint_array, both =
  support.mstrings, support.mint_array, support.both

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

local record Cost
  kind: compute.CacheType
  bytes: integer
end

local record Slack
  kind: compute.CacheType
  slack: number
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
  local location = ROOT .. "/cache" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

-- 32 numbers a row: one query number per head, one key and value head, and
-- every gate number the same as its query number
local function wide(prefix: string): {Entry}
  local queries: {number} = {}
  for row = 1, 128 do
    for column = 1, 32 do
      if row <= 96 and column == (row - 1) % 32 + 1 then
        queries[#queries + 1] = 1
      else
        queries[#queries + 1] = 0
      end
    end
  end
  local keyed: {number} = {}
  for row = 1, 32 do
    for column = 1, 32 do
      if column == 1 then
        keyed[#keyed + 1] = row / 32
      else
        keyed[#keyed + 1] = 0
      end
    end
  end
  local output: {number} = {}
  for row = 1, 32 do
    for column = 1, 64 do
      if column == (row - 1) % 32 + 1 then
        output[#output + 1] = 1
      else
        output[#output + 1] = 0
      end
    end
  end
  return {
    f32(prefix .. "attn_norm.weight", {32}, ones(32)),
    f32(prefix .. "attn_q.weight", {32, 128}, queries),
    f32(prefix .. "attn_k.weight", {32, 32}, keyed),
    f32(prefix .. "attn_v.weight", {32, 32}, keyed),
    f32(prefix .. "attn_q_norm.weight", {32}, ones(32)),
    f32(prefix .. "attn_k_norm.weight", {32}, ones(32)),
    f32(prefix .. "attn_output.weight", {64, 32}, output),
  }
end

-- a block of two-number heads, and the metadata a whole model reads it under
local function narrow(prefix: string): {Entry}
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

local function model_meta(width: integer): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", width),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 4),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
  }
end

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function geometry(
  positions: integer, cache_type: compute.CacheType
): compute.Geometry
  return {
    heads = 2,
    kv_heads = 1,
    head_length = 32,
    rope_dimensions = 2,
    rope_base = 10000,
    eps = 0,
    positions = positions,
    experts_used = 2,
    cache_type = cache_type,
  }
end

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function attended(
  handle: weights.Handle, shape: compute.Geometry
): compute.Attention
  local attention, failure = compute.attend(backend, handle, shape)
  test.equal(failure, nil)
  return attention as compute.Attention
end

local function blocked(
  attention: compute.Attention,
  hidden: {number},
  prefix: string,
  position: integer
): {number}
  local numbers, line = attention.block(
    attention, {{sequence = "", hidden = hidden, position = position}}, prefix
  )
  test.equal(line, nil)
  local rows = numbers as {{number}}
  return rows[1]
end

test.it(
  "a cache reports what its type costs",
  function()
    local handle = loaded({}, both(wide("blk.0."), wide("blk.1.")))
    local costs: {Cost} = {
      {kind = "f32", bytes = 1024},
      {kind = "f16", bytes = 512},
      {kind = "q8_0", bytes = 272},
      {kind = "q4_0", bytes = 144},
    }
    for _, cost in ipairs(costs) do
      local attention = attended(handle, geometry(4, cost.kind))
      test.equal(attention.cache_bytes(attention, "", "blk.0."), nil)
      blocked(attention, ones(32), "blk.0.", 0)
      test.equal(attention.cache_bytes(attention, "", "blk.0."), cost.bytes)
      test.equal(attention.cache_bytes(attention, "", "blk.2."), nil)
    end
  end
)

test.it(
  "a type the model cannot hold is refused",
  function()
    local handle = loaded(model_meta(2), narrow("blk.0."))

    local packed, packed_line = compute.model(backend, handle, 4, "q8_0")
    test.equal(packed, nil)
    test.includes(packed_line as string, "q8_0")
    test.includes(packed_line as string, "2")

    local nibbled, nibbled_line = compute.model(backend, handle, 4, "q4_0")
    test.equal(nibbled, nil)
    test.includes(nibbled_line as string, "q4_0")
    test.includes(nibbled_line as string, "2")

    local unknown, unknown_line = compute.model(
      backend, handle, 4, "q4_0x" as compute.CacheType
    )
    test.equal(unknown, nil)
    test.includes(unknown_line as string, "model")
    test.includes(unknown_line as string, "q4_0x")

    local half, half_line = compute.model(backend, handle, 4, "f16")
    test.truthy(half)
    test.equal(half_line, nil)
    local single, single_line = compute.model(backend, handle, 4, "f32")
    test.truthy(single)
    test.equal(single_line, nil)
  end
)

test.it(
  "a thousand cached positions cost a token under ten short ones",
  function()
    local handle = loaded({}, wide("blk.0."))
    local shape = geometry(1025, "f16")

    local discard = attended(handle, shape)
    blocked(discard, ones(32), "blk.0.", 0)
    blocked(discard, ones(32), "blk.0.", 1)

    local shortest = math.huge
    for _ = 1, 5 do
      local short = attended(handle, shape)
      blocked(short, ones(32), "blk.0.", 0)
      local started = clock.now()
      blocked(short, ones(32), "blk.0.", 1)
      local taken = clock.now() - started
      if taken < shortest then
        shortest = taken
      end
    end

    local long = attended(handle, shape)
    for position = 0, 1023 do
      blocked(long, ones(32), "blk.0.", position)
    end
    local started = clock.now()
    blocked(long, ones(32), "blk.0.", 1024)
    local taken = clock.now() - started

    test.less(taken, 10 * shortest)
  end
)

test.it(
  "each type's numbers stay inside their tolerance",
  function()
    local handle = loaded({}, wide("blk.0."))
    local busy: {number} = {2}
    for _ = 2, 32 do
      busy[#busy + 1] = 1
    end

    local reference = attended(handle, geometry(4, "f32"))
    blocked(reference, ones(32), "blk.0.", 0)
    local wanted = blocked(reference, busy, "blk.0.", 1)
    local largest: number = 0
    for _, value in ipairs(wanted) do
      if math.abs(value) > largest then
        largest = math.abs(value)
      end
    end

    local slacks: {Slack} = {
      {kind = "f16", slack = 1e-3},
      {kind = "q8_0", slack = 1e-2},
      {kind = "q4_0", slack = 1e-1},
    }
    for _, held in ipairs(slacks) do
      local attention = attended(handle, geometry(4, held.kind))
      blocked(attention, ones(32), "blk.0.", 0)
      local got = blocked(attention, busy, "blk.0.", 1)
      local apart = false
      for index, value in ipairs(wanted) do
        local gap = math.abs(got[index] - value)
        test.less(gap, held.slack * largest)
        if gap > 1e-5 then
          apart = true
        end
      end
      test.truthy(apart)
    end
  end
)

-- the numbers 16/FR-006's generator gives from seed 0, folded between -1 and 1
local function drawn(): function(integer): {number}
  local state: integer = 0
  return function(count: integer): {number}
    local values: {number} = {}
    for index = 1, count do
      state = (1664525 * state + 1013904223) % 4294967296
      values[index] = 2 * (state / 4294967296) - 1
    end
    return values
  end
end

local function speakable_meta(): {Meta}
  local meta = model_meta(32)
  meta[#meta + 1] = mtext("tokenizer.ggml.model", "gpt2")
  meta[#meta + 1] = mtext("tokenizer.ggml.pre", "gpt2")
  meta[#meta + 1] = mstrings("tokenizer.ggml.tokens", {"a", "b", "c"})
  meta[#meta + 1] = mint_array("tokenizer.ggml.token_type", {1, 1, 1})
  meta[#meta + 1] = mstrings("tokenizer.ggml.merges", {})
  meta[#meta + 1] = mint("tokenizer.ggml.eos_token_id", 1)
  return meta
end

local function speakable_entries(): {Entry}
  local next_numbers = drawn()
  local entries: {Entry} = {}
  local function take(name: string, dims: {integer})
    local count = 1
    for _, dimension in ipairs(dims) do
      count = count * dimension
    end
    entries[#entries + 1] = f32(name, dims, next_numbers(count))
  end
  take("blk.0.attn_norm.weight", {32})
  take("blk.0.attn_q.weight", {32, 64})
  take("blk.0.attn_k.weight", {32, 32})
  take("blk.0.attn_v.weight", {32, 32})
  take("blk.0.attn_q_norm.weight", {32})
  take("blk.0.attn_k_norm.weight", {32})
  take("blk.0.attn_output.weight", {32, 32})
  take("blk.0.post_attention_norm.weight", {32})
  take("blk.0.ffn_gate_inp.weight", {32, 32})
  take("blk.0.ffn_gate_exps.weight", {32, 32, 32})
  take("blk.0.ffn_up_exps.weight", {32, 32, 32})
  take("blk.0.ffn_down_exps.weight", {32, 32, 32})
  take("blk.0.ffn_gate_inp_shexp.weight", {32})
  take("blk.0.ffn_up_shexp.weight", {32, 32})
  take("blk.0.ffn_gate_shexp.weight", {32, 32})
  take("blk.0.ffn_down_shexp.weight", {32, 32})

  local projected: {number} = {}
  for row = 1, 3 do
    for column = 1, 32 do
      if column == 1 then
        projected[#projected + 1] = 1 / 2 ^ (row - 1)
      else
        projected[#projected + 1] = 0
      end
    end
  end
  local embedded: {number} = {}
  for row = 1, 3 do
    for _ = 1, 32 do
      embedded[#embedded + 1] = (row % 2) * 2 - 1
    end
  end
  entries[#entries + 1] = f32("output.weight", {32, 3}, projected)
  entries[#entries + 1] = f32("output_norm.weight", {32}, ones(32))
  entries[#entries + 1] = f32("token_embd.weight", {32, 3}, embedded)
  return entries
end

test.it(
  "every type gives the reply the float cache gave",
  function()
    local location = fixture(speakable_meta(), speakable_entries())
    local handle, handle_line = weights.load(location)
    test.equal(handle_line, nil)
    local mapped = handle as weights.Handle
    local pieces, pieces_line = tokenizer.load(location)
    test.equal(pieces_line, nil)
    local spoken = pieces as tokenizer.Tokenizer

    local function reply(cache_type: compute.CacheType): string
      local model, failure = compute.model(backend, mapped, 4, cache_type)
      test.equal(failure, nil)
      local output, line = generate.text(
        model as compute.Model,
        spoken,
        "a",
        {max_tokens = 2, seed = 0, end_id = 3}
      )
      test.equal(line, nil)
      return (output as generate.Text).text
    end

    local single = reply("f32")
    test.equal(#single, 2)
    for _, kind in ipairs({"f16", "q8_0", "q4_0"}) do
      test.equal(reply(kind as compute.CacheType), single)
    end
  end
)

local function long_meta(): {Meta}
  local meta = speakable_meta()
  for index, entry in ipairs(meta) do
    if entry.name == "qwen35moe.context_length" then
      meta[index] = mint(entry.name, 1056)
    end
  end
  return meta
end

test.it(
  "a save and a load cost about what their bytes cost",
  function()
    local handle = loaded(long_meta(), speakable_entries())
    local model, failure = compute.model(backend, handle, 1056, "f32", 2)
    test.equal(failure, nil)
    local generation = model as compute.Model
    test.equal(generation:reserve("a", 1056), nil)
    for position = 0, 1023 do
      local _, line = generation:pass(
        {{sequence = "a", token = 0, position = position}}
      )
      test.equal(line, nil)
    end

    local ids: {integer} = {}
    for index = 1, 1024 do
      ids[index] = 0
    end
    local path = ROOT .. "/context_perf"
    test.equal(generation:save("a", ids, path), nil)
    local held = fs.stat(path)
    test.truthy(held)
    local size = math.floor(tonumber((held as fs.Stat).size) as number)
    test.greater(size, 1024)

    local mirror = ROOT .. "/context_mirror"
    local filler = string.rep("x", size)

    local function shortest_save(): number
      local best = math.huge
      for _ = 1, 5 do
        local started = clock.now()
        local line = generation:save("a", ids, path)
        local taken = clock.now() - started
        test.equal(line, nil)
        if taken < best then
          best = taken
        end
      end
      return best
    end

    local function shortest_write(): number
      local best = math.huge
      for _ = 1, 5 do
        local started = clock.now()
        local wrote = fs.write(mirror, filler)
        local taken = clock.now() - started
        test.truthy(wrote)
        if taken < best then
          best = taken
        end
      end
      return best
    end

    local function shortest_load(): number
      local best = math.huge
      for _ = 1, 5 do
        local started = clock.now()
        local given, line = generation:load("a", path)
        local taken = clock.now() - started
        test.equal(line, nil)
        test.equal(#(given as {integer}), 1024)
        generation:release("a")
        if taken < best then
          best = taken
        end
      end
      return best
    end

    local function shortest_read(): number
      local best = math.huge
      for _ = 1, 5 do
        local started = clock.now()
        local bytes = fs.read(path)
        local taken = clock.now() - started
        test.truthy(bytes)
        if taken < best then
          best = taken
        end
      end
      return best
    end

    test.equal(generation:save("a", ids, path), nil)
    generation:release("a")
    test.truthy(generation:load("a", path))

    local shortest_saved = shortest_save()
    generation:release("a")
    local shortest_loaded = shortest_load()

    test.lessEqual(shortest_saved, 10 * shortest_write())
    test.lessEqual(shortest_loaded, 10 * shortest_read())
  end
)
