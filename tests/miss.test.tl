local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local miss = require("tllm.miss")
local weights = require("tllm.weights")
local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, mint = support.padded, support.f32, support.mint
local mfloat, mtext = support.mfloat, support.mtext

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

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
  local location = ROOT .. "/miss" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

-- 36's cache fixture: one block, three experts, one routed per pass, every
-- attention tensor leaving the feed-forward's row where the token put it
local function cache_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 32),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
  }
end

local function expert_values(): {number}
  local values: {number} = {}
  for expert = 1, 3 do
    for _ = 1, 4 do
      values[#values + 1] = expert
    end
  end
  return values
end

local function cache_entries(): {Entry}
  local experts = expert_values()
  return {
    f32("blk.0.attn_norm.weight", {2}, {1, 1}),
    f32("blk.0.attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
    f32("blk.0.attn_k.weight", {2, 2}, {1, 0, 0, 1}),
    f32("blk.0.attn_v.weight", {2, 2}, {1, 0, 0, 1}),
    f32("blk.0.attn_q_norm.weight", {2}, {1, 1}),
    f32("blk.0.attn_k_norm.weight", {2}, {1, 1}),
    f32("blk.0.attn_output.weight", {2, 2}, {0, 0, 0, 0}),
    f32("blk.0.post_attention_norm.weight", {2}, {1, 1}),
    f32("blk.0.ffn_gate_inp.weight", {2, 3}, {1, 0, 0, 1, -1, -1}),
    f32("blk.0.ffn_gate_inp_shexp.weight", {2}, {0, 0}),
    f32("blk.0.ffn_up_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32("blk.0.ffn_gate_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32("blk.0.ffn_down_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32("blk.0.ffn_up_exps.weight", {2, 2, 3}, experts),
    f32("blk.0.ffn_gate_exps.weight", {2, 2, 3}, experts),
    f32("blk.0.ffn_down_exps.weight", {2, 2, 3}, experts),
    f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, -1, -1}),
    f32("output_norm.weight", {2}, {1, 1}),
    f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
  }
end

local function profile_file(lines: {string}): string
  written = written + 1
  local location = ROOT .. "/profile" .. written .. ".jsonl"
  fs.write(location, table.concat(lines, "\n") .. "\n")
  return location
end

local function routing(layer: integer, expert: integer): string
  return '{"layer":' .. layer .. ',"token":0,"experts":[' .. expert .. "]}"
end

local backend = compute.load() as compute.Backend
local on_device = support.on_device

-- the vram that holds every fixed tensor and the reservation and leaves no
-- expert slot, so every expert a pass routes is a miss
local function miss_vram(): integer
  local handle = loaded(cache_meta(), cache_entries())
  local model = compute.model(backend, handle, nil, "f32") as compute.Model
  local vram = model.placement.resident_bytes + model.placement.reserve
  model:close()
  test.equal(handle.release(handle), nil)
  return vram
end

-- the vram that holds every fixed tensor, the reservation and the first of the
-- fixture's three expert slots
local function one_slot_vram(): integer
  local handle = loaded(cache_meta(), cache_entries())
  local model = compute.model(backend, handle, nil, "f32") as compute.Model
  local third = math.floor(model.placement.slot_bytes / 3)
  local vram = model.placement.resident_bytes + model.placement.reserve + third
  model:close()
  test.equal(handle.release(handle), nil)
  return vram
end

local function built(vram: integer, profile: string | nil): compute.Model
  local handle = loaded(cache_meta(), cache_entries())
  local model = compute.model(
    backend, handle, nil, "f32", 1, nil, vram, profile
  )
  test.truthy(model, "the missed model")
  return model as compute.Model
end

local function passed(
  model: compute.Model, token: integer, position: integer
): {number}
  local scores, line = model.forward(model, token, position)
  test.equal(line, nil)
  return scores as {number}
end

-- the id the row's scores put first
local function chosen(scores: {number}): integer
  local best = 1
  for index = 2, #scores do
    if scores[index] > scores[best] then
      best = index
    end
  end
  return best - 1
end

test.it(
  "the measured rates choose a bounded share and a stable set",
  function()
    local shares: {{number, number, number}} = {
      {56.1, 61.2, 0.35},
      {56.3, 10.3, 0.85},
      {56.2, 8.0, 0.90},
      {56.0, 22.3, 0.65},
      {56.0, 13.0, 0.85},
      {56.0, 29.0, 0.50},
      {56.0, 0.5, 0.95},
      {1.0, 200.0, 0.10},
    }
    for _, held in ipairs(shares) do
      local got = miss.share(held[1], held[2])
      test.truthy(
        math.abs(got - held[3]) <= 1e-9,
        held[1] .. " and " .. held[2] .. " gave " .. got
      )
    end

    test.equal(table.concat(miss.take({2, 0, 1}, 0.55), ","), "0,1")
    test.equal(table.concat(miss.take({2, 0, 1}, 0.20), ","), "0")
    test.equal(table.concat(miss.take({2}, 0.10), ","), "")
    test.equal(table.concat(miss.take({2}, 0.95), ","), "2")
  end
)

on_device(
  "two probes teach the policy for sixteen windows",
  function()
    local model = built(miss_vram(), nil)
    local policy = model.miss
    local measure = model.measure

    -- the first probe computes the miss on the CPU
    local started = measure.compute
    passed(model, 0, 0)
    test.equal(policy.cpu, 1)
    test.equal(policy.dma, 0)
    test.equal(policy.windows, 1)
    test.truthy(policy.pool > 0, "a positive pool rate")
    test.equal(policy.link, 0)
    test.equal(policy.share, 0)
    test.truthy(policy.cpu_seconds < measure.compute - started)
    local cpu_bytes, cpu_seconds = policy.cpu_bytes, policy.cpu_seconds
    test.truthy(cpu_bytes > 0)

    -- the second sends it by the link and leaves the CPU's totals alone
    started = measure.compute
    passed(model, 0, 1)
    test.equal(policy.cpu, 1)
    test.equal(policy.dma, 1)
    test.equal(policy.windows, 2)
    test.truthy(policy.link > 0, "a positive link rate")
    test.truthy(policy.pool > 0, "a positive pool rate")
    test.equal(policy.share, miss.share(policy.link, policy.pool))
    test.equal(policy.cpu_bytes, cpu_bytes)
    test.equal(policy.cpu_seconds, cpu_seconds)
    test.truthy(policy.dma_seconds < measure.compute - started)

    local function rates_hold(windows: integer)
      test.equal(policy.windows, windows)
      test.equal(
        policy.pool, policy.cpu_bytes / policy.cpu_seconds / 1000000000
      )
      test.equal(
        policy.link, policy.dma_bytes / policy.dma_seconds / 1000000000
      )
      test.equal(policy.share, miss.share(policy.link, policy.pool))
    end

    -- the third assigns by the share published before it
    local routed: {integer} = {}
    model.routing_sink = function(entry: compute.Routing)
      for _, expert in ipairs(entry.experts) do
        routed[#routed + 1] = expert
      end
    end
    local share = policy.share
    local cpu_before, dma_before = policy.cpu, policy.dma
    passed(model, 0, 2)
    local by_link = miss.take(routed, share)
    test.equal(policy.dma - dma_before, #by_link)
    test.equal(policy.cpu - cpu_before, #routed - #by_link)
    rates_hold(3)

    local measured = policy.cpu_bytes + policy.dma_bytes
    for pass = 4, 16 do
      passed(model, 0, pass - 1)
      rates_hold(pass)
      local now = policy.cpu_bytes + policy.dma_bytes
      test.truthy(now > measured, "calibration bytes after pass " .. pass)
      measured = now
    end

    -- the seventeenth refines nothing
    share = policy.share
    local pool, link = policy.pool, policy.link
    cpu_bytes, cpu_seconds = policy.cpu_bytes, policy.cpu_seconds
    local dma_bytes, dma_seconds = policy.dma_bytes, policy.dma_seconds
    passed(model, 0, 16)
    test.equal(policy.windows, 16)
    test.equal(policy.share, share)
    test.equal(policy.pool, pool)
    test.equal(policy.link, link)
    test.equal(policy.cpu_bytes, cpu_bytes)
    test.equal(policy.cpu_seconds, cpu_seconds)
    test.equal(policy.dma_bytes, dma_bytes)
    test.equal(policy.dma_seconds, dma_seconds)
    test.equal(policy.cpu + policy.dma, 17)
  end
)

on_device(
  "a missed expert gives the same token on either path",
  function()
    local model = built(miss_vram(), nil)
    local first = passed(model, 0, 0)
    test.equal(model.miss.dma, 0)
    local second = passed(model, 0, 1)
    test.equal(model.miss.dma, 1)
    test.equal(chosen(first), chosen(second))

    local cpu = compute.load(nil, false) as compute.Backend
    local handle = loaded(cache_meta(), cache_entries())
    local fallback = compute.model(cpu, handle, nil, "f32") as compute.Model
    local third = passed(fallback, 0, 0)
    test.equal(chosen(third), chosen(first))
    local policy = fallback.miss
    test.equal(policy.share, 0)
    test.equal(policy.pool, 0)
    test.equal(policy.link, 0)
    test.equal(policy.windows, 0)
    test.equal(policy.cpu, 0)
    test.equal(policy.dma, 0)
    fallback:close()
    test.equal(handle.release(handle), nil)
    compute.unload(cpu)
  end
)

on_device(
  "a held expert is not a miss and a hit is not a window",
  function()
    local profile = profile_file({routing(0, 0)})
    local model = built(one_slot_vram(), profile)
    local policy = model.miss

    passed(model, 0, 0)
    test.equal(policy.windows, 0)
    test.equal(policy.cpu, 0)
    test.equal(policy.dma, 0)
    test.equal(policy.cpu_bytes, 0)
    test.equal(policy.cpu_seconds, 0)
    test.equal(policy.dma_bytes, 0)
    test.equal(policy.dma_seconds, 0)

    passed(model, 1, 1)
    test.equal(policy.windows, 1)
    test.equal(policy.cpu, 1)
    test.equal(policy.dma, 0)
  end
)
