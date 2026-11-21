local fs = require("fs")
local test = require("lde-test")
local process = require("process")
local support = require("tests.support")
local cli = require("tllm.cli")
local compute = require("tllm.compute")
local serve = require("tllm.serve")
local weights = require("tllm.weights")
local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text, padded =
  support.u32, support.u64, support.text, support.padded
local f32, zeros, ones = support.f32, support.zeros, support.ones
local mint, mfloat, mtext = support.mint, support.mfloat, support.mtext

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
  local location = ROOT .. "/cache" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

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
    f32("blk.0.attn_norm.weight", {2}, ones(2)),
    f32("blk.0.attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
    f32("blk.0.attn_k.weight", {2, 2}, {1, 0, 0, 1}),
    f32("blk.0.attn_v.weight", {2, 2}, {1, 0, 0, 1}),
    f32("blk.0.attn_q_norm.weight", {2}, ones(2)),
    f32("blk.0.attn_k_norm.weight", {2}, ones(2)),
    f32("blk.0.attn_output.weight", {2, 2}, zeros(4)),
    f32("blk.0.post_attention_norm.weight", {2}, ones(2)),
    f32("blk.0.ffn_gate_inp.weight", {2, 3}, {1, 0, 0, 1, -1, -1}),
    f32("blk.0.ffn_gate_inp_shexp.weight", {2}, zeros(2)),
    f32("blk.0.ffn_up_shexp.weight", {2, 2}, zeros(4)),
    f32("blk.0.ffn_gate_shexp.weight", {2, 2}, zeros(4)),
    f32("blk.0.ffn_down_shexp.weight", {2, 2}, zeros(4)),
    f32("blk.0.ffn_up_exps.weight", {2, 2, 3}, experts),
    f32("blk.0.ffn_gate_exps.weight", {2, 2, 3}, experts),
    f32("blk.0.ffn_down_exps.weight", {2, 2, 3}, experts),
    f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, -1, -1}),
    f32("output_norm.weight", {2}, ones(2)),
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

-- the vram that holds every fixed tensor, the reservation and `thirds` of the
-- fixture's three expert slots
local function capped(handle: weights.Handle, thirds: integer): integer
  local model = compute.model(backend, handle, nil, "f32") as compute.Model
  local placement = model.placement
  local cap = placement.resident_bytes + placement.reserve + math.floor(
    placement.slot_bytes * thirds / 3
  )
  model:close()
  return cap
end

local function built(
  handle: weights.Handle, cap: integer, profile: string | nil
): compute.Model
  local model = compute.model(backend, handle, nil, "f32", 1, nil, cap, profile)
  test.truthy(model, "the capped model")
  return model as compute.Model
end

local function passed(
  model: compute.Model, token: integer, position: integer
): {number}
  local scores, line = model.forward(model, token, position)
  test.equal(line, nil)
  return scores as {number}
end

local function equal_rows(left: {number}, right: {number})
  test.equal(#left, #right)
  for index = 1, #right do
    test.equal(left[index], right[index])
  end
end

on_device(
  "a wrong profile converges on the conversation's experts",
  function()
    local handle = loaded(cache_meta(), cache_entries())
    local cap = capped(handle, 1)

    local wrong = built(handle, cap, profile_file({routing(0, 0)}))
    for position = 0, 7 do
      passed(wrong, 1, position)
    end
    test.equal(wrong.measure.expert_hits, 0)
    test.equal(wrong.measure.routed, 8)
    passed(wrong, 1, 8)
    test.equal(wrong.measure.expert_hits, 0)
    for position = 9, 16 do
      passed(wrong, 1, position)
    end
    test.equal(wrong.measure.expert_hits, 8)
    wrong:close()

    local right = built(handle, cap, profile_file({routing(0, 1)}))
    passed(right, 1, 0)
    test.equal(right.measure.expert_hits, 1)
    right:close()

    local absent = built(
      handle, cap, profile_file({routing(9, 1), routing(0, 9)})
    )
    passed(absent, 0, 0)
    test.equal(absent.measure.expert_hits, 1)
    absent:close()

    local tie = built(handle, cap, profile_file({routing(0, 0), routing(0, 1)}))
    passed(tie, 0, 0)
    test.equal(tie.measure.expert_hits, 1)
    tie:close()
    test.equal(handle.release(handle), nil)
  end
)

on_device(
  "an admitted expert is not computed before it lands",
  function()
    local handle = loaded(cache_meta(), cache_entries())
    local cap = capped(handle, 1)
    local seeded_zero = built(handle, cap, profile_file({routing(0, 0)}))
    local seeded_one = built(handle, cap, profile_file({routing(0, 1)}))
    local first: {{number}} = {}
    local second: {{number}} = {}
    for position = 0, 9 do
      first[position + 1] = passed(seeded_zero, 1, position)
      second[position + 1] = passed(seeded_one, 1, position)
      if position < 9 then
        test.equal(seeded_zero.measure.expert_hits, 0)
      end
    end
    test.equal(seeded_zero.measure.expert_hits, 1)
    test.equal(seeded_one.measure.expert_hits, 10)
    equal_rows(first[9], second[9])
    equal_rows(first[10], second[10])
    seeded_zero:close()
    seeded_one:close()
    test.equal(handle.release(handle), nil)
  end
)

on_device(
  "a contested slot keeps what the rounds chose",
  function()
    local handle = loaded(cache_meta(), cache_entries())
    local cap = capped(handle, 2)
    local profile = profile_file({routing(0, 0), routing(0, 1)})

    local first = built(handle, cap, profile)
    for position, token in ipairs({0, 0, 1, 2, 2, 2, 2, 1}) do
      passed(first, token, position - 1)
    end
    passed(first, 2, 8)
    local before = first.measure.expert_hits
    local counts: {integer} = {}
    for offset, token in ipairs({0, 1, 2}) do
      passed(first, token, 8 + offset)
      counts[offset] = first.measure.expert_hits - before
      before = first.measure.expert_hits
    end
    test.deepEqual(counts, {0, 1, 1})
    first:close()

    local second = built(handle, cap, profile)
    for position = 0, 8 do
      passed(second, 2, position)
    end
    before = second.measure.expert_hits
    passed(second, 1, 9)
    test.equal(second.measure.expert_hits - before, 0)
    before = second.measure.expert_hits
    passed(second, 0, 10)
    test.equal(second.measure.expert_hits - before, 1)
    second:close()
    test.equal(handle.release(handle), nil)
  end
)

test.it(
  "a profile is refused or carried as the run names it",
  function()
    local handle = loaded(cache_meta(), cache_entries())
    local cpu = compute.load(nil, false) as compute.Backend
    local missing = ROOT .. "/no-profile.jsonl"
    local refused, refusal = compute.model(
      cpu, handle, nil, "f32", 1, 1, nil, missing
    )
    test.equal(refused, nil)
    local line = refusal as string
    test.includes(line, "model")
    test.includes(line, "profile")
    test.includes(line, missing)

    local broken = profile_file({"{not json"})
    local refused_bad, bad_line = compute.model(
      cpu, handle, nil, "f32", 1, 1, nil, broken
    )
    test.equal(refused_bad, nil)
    local bad = bad_line as string
    test.includes(bad, "model")
    test.includes(bad, "profile")
    test.includes(bad, broken)
    test.equal(handle.release(handle), nil)
    compute.unload(cpu)

    local file_path = fixture(cache_meta(), cache_entries())
    local numbered = ROOT .. "/numbered.json"
    fs.write(
      numbered, '{"models":{"m":{"path":"' .. file_path .. '"}},"profile":1}'
    )
    local out: {string} = {}
    local err: {string} = {}
    local code = cli.run(
      {
        args = {"bench", "m", "--config", numbered},
        write_out = function(given: string) out[#out + 1] = given end,
        write_err = function(given: string) err[#err + 1] = given end,
      }
    )
    test.equal(code, 1)
    test.equal(#err, 1)
    test.includes(err[1], "profile")
    test.equal(#out, 0)

    local named = profile_file({routing(0, 1)})
    local configured = ROOT .. "/configured.json"
    fs.write(
      configured,
      '{"models":{"m":{"path":"'
        .. file_path
        .. '"}},"profile":"'
        .. named
        .. '"}'
    )
    local seen: string | nil = "unset"
    local collect = function(
      endpoint: serve.Endpoint, _err: function(string), _spawn?: process.Spawn
    ): integer
      seen = endpoint.profile
      return 0
    end
    local ran = cli.run(
      {
        args = {"serve", "m", "--config", configured},
        write_out = function(_: string) end,
        write_err = function(_: string) end,
        start = collect,
      }
    )
    test.equal(ran, 0)
    test.equal(seen, named)

    local plain = ROOT .. "/plain.json"
    fs.write(plain, '{"models":{"m":{"path":"' .. file_path .. '"}}}')
    seen = "unset"
    ran = cli.run(
      {
        args = {"serve", "m", "--config", plain},
        write_out = function(_: string) end,
        write_err = function(_: string) end,
        start = collect,
      }
    )
    test.equal(ran, 0)
    test.equal(seen, nil)
  end
)

on_device(
  "the policy adapts again after another eight rounds",
  function()
    local handle = loaded(cache_meta(), cache_entries())
    local cap = capped(handle, 1)
    local model = built(handle, cap, profile_file({routing(0, 0)}))
    for position = 0, 8 do
      passed(model, 1, position)
    end
    for position = 9, 16 do
      passed(model, 2, position)
    end
    test.equal(model.measure.expert_hits, 0)
    for position = 17, 24 do
      passed(model, 2, position)
    end
    test.equal(model.measure.expert_hits, 8)
    model:close()
    test.equal(handle.release(handle), nil)
  end
)
