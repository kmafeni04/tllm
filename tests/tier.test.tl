local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local weights = require("tllm.weights")
local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text, padded =
  support.u32, support.u64, support.text, support.padded
local f32, zeros, ones = support.f32, support.zeros, support.ones
local mint, mfloat, mtext, both =
  support.mint, support.mfloat, support.mtext, support.both
local without = support.without

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
  local location = ROOT .. "/tier" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function tensor_bytes(handle: weights.Handle): integer
  local total = 0
  for _, tensor in pairs(handle.tensors) do
    total = total + tensor.bytes
  end
  return total
end

-- the plan fixture: 41 fixed tensors of one shape, and experts twice as wide
-- in the last block as in the linear one
local PLAN_ATTENTION: {string} = {
  "attn_norm.weight",
  "attn_q.weight",
  "attn_k.weight",
  "attn_v.weight",
  "attn_q_norm.weight",
  "attn_k_norm.weight",
  "attn_output.weight",
}

local PLAN_LINEAR: {string} = {
  "attn_norm.weight",
  "attn_qkv.weight",
  "attn_gate.weight",
  "ssm_beta.weight",
  "ssm_alpha.weight",
  "ssm_conv1d.weight",
  "ssm_a",
  "ssm_dt.bias",
  "ssm_norm.weight",
  "ssm_out.weight",
}

local function flat(name: string): Entry
  return f32(name, {32}, zeros(32))
end

local function plan_entries(): {Entry}
  local entries: {Entry} = {
    flat("token_embd.weight"),
    flat("output_norm.weight"),
    flat("output.weight"),
  }
  local function add(prefix: string, names: {string})
    for _, name in ipairs(names) do
      entries[#entries + 1] = flat(prefix .. name)
    end
  end
  add("blk.0.", PLAN_ATTENTION)
  add(
    "blk.0.",
    {
      "post_attention_norm.weight",
      "ffn_gate_inp.weight",
      "ffn_gate_inp_shexp.weight",
      "ffn_up_shexp.weight",
      "ffn_gate_shexp.weight",
      "ffn_down_shexp.weight",
    }
  )
  add("blk.1.", PLAN_LINEAR)
  add(
    "blk.1.",
    {
      "post_attention_norm.weight",
      "ffn_gate_inp.weight",
      "ffn_gate_inp_shexp.weight",
      "ffn_up_shexp.weight",
      "ffn_gate_shexp.weight",
      "ffn_down_shexp.weight",
    }
  )
  local function experts(prefix: string, first: integer)
    entries[#entries + 1] = f32(
      prefix .. "ffn_up_exps.weight", {first, 1, 2}, zeros(first * 2)
    )
    entries[#entries + 1] = f32(
      prefix .. "ffn_gate_exps.weight", {first, 1, 2}, zeros(first * 2)
    )
    entries[#entries + 1] = f32(
      prefix .. "ffn_down_exps.weight", {1, first, 2}, zeros(first * 2)
    )
  end
  experts("blk.1.", 32)
  add("blk.2.", PLAN_ATTENTION)
  add("blk.2.", {"post_attention_norm.weight", "ffn_gate_inp.weight"})
  experts("blk.2.", 64)
  return entries
end

local function marked(placement: compute.Placement): {string}
  local names: {string} = {}
  for name in pairs(placement.resident) do
    names[#names + 1] = name
  end
  table.sort(names)
  return names
end

local function plan_of(
  handle: weights.Handle, budget: integer, reserve: integer
): compute.Placement
  local placement = compute.place(handle, budget, reserve)
  test.truthy(
    placement.resident_bytes + placement.slot_bytes <= budget - reserve
  )
  test.equal(
    placement.unplaced_bytes,
    tensor_bytes(handle) - placement.resident_bytes - placement.slot_bytes
  )
  return placement
end

test.it(
  "the budget fills stratum by stratum",
  function()
    local handle = loaded({}, plan_entries())
    local u = 128
    local t = u
    local f = 41 * u
    local e = 3 * 128
    local R = 1024

    test.deepEqual(marked(plan_of(handle, t + R, R)), {"token_embd.weight"})

    test.deepEqual(
      marked(plan_of(handle, t + 3 * u + R, R)),
      {
        "blk.0.attn_k.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "token_embd.weight",
      }
    )

    local sixteen = plan_of(handle, t + 16 * u + R, R)
    local names = marked(sixteen)
    test.equal(#names, 17)
    for _, name in ipairs(
      {
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.post_attention_norm.weight",
        "blk.2.attn_norm.weight",
        "blk.2.post_attention_norm.weight",
        "token_embd.weight",
      }
    ) do
      test.truthy(sixteen.resident[name], name)
    end
    for name in pairs(sixteen.resident) do
      test.truthy(name:sub(1, 6) ~= "blk.1.", name)
    end

    local routers = plan_of(handle, t + 32 * u + R, R)
    test.equal(#marked(routers), 33)
    for name in pairs(routers.resident) do
      for _, shared in ipairs(
        {
          "ffn_up_shexp.weight",
          "ffn_gate_shexp.weight",
          "ffn_down_shexp.weight",
        }
      ) do
        test.truthy(not name:find(shared, 1, true), name)
      end
    end
    for _, name in ipairs(
      {
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.1.ffn_gate_inp.weight",
        "blk.1.ffn_gate_inp_shexp.weight",
        "blk.2.ffn_gate_inp.weight",
      }
    ) do
      test.truthy(routers.resident[name], name)
    end

    local almost = plan_of(handle, f - u + R, R)
    test.equal(#marked(almost), 40)
    test.truthy(not almost.resident["output.weight"])
    local all = plan_of(handle, f + R, R)
    test.equal(#marked(all), 41)
    test.deepEqual(all.slots, {0, 0, 0})
    test.equal(all.slot_bytes, 0)

    test.deepEqual(plan_of(handle, f + e + R, R).slots, {0, 1, 0})
    test.deepEqual(plan_of(handle, f + 3 * e + R, R).slots, {0, 1, 1})
    test.deepEqual(plan_of(handle, f + 5 * e + R, R).slots, {0, 2, 1})

    local none = plan_of(handle, R, R)
    test.equal(#marked(none), 0)
    test.deepEqual(none.slots, {0, 0, 0})
  end
)

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

local function t001_entries(gate?: {number}, expert?: number): {Entry}
  local entries = both(
    attention_entries("blk.0."), feed_forward_entries("blk.0.")
  )
  for _, entry in ipairs(entries) do
    if entry.name == "blk.0.ffn_gate_inp.weight" and gate is {number} then
      entry.bytes = support.floats(gate)
    end
    if not (expert is nil) and entry.name:find("_exps.weight") then
      local values: {number} = {}
      for _ = 1, 4 do
        values[#values + 1] = expert
      end
      for _ = 1, 4 do
        values[#values + 1] = expert + 1
      end
      entry.bytes = support.floats(values)
    end
  end
  return both(
    entries,
    {
      f32("token_embd.weight", {2, 2}, {1, 0, 3, 4}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
    }
  )
end

local function t002_meta(): {Meta}
  local meta = t001_meta()
  for index, entry in ipairs(meta) do
    if entry.name == "qwen35moe.block_count" then
      meta[index] = mint(entry.name, 2)
    elseif entry.name == "qwen35moe.full_attention_interval" then
      meta[index] = mint(entry.name, 2)
    end
  end
  meta[#meta + 1] = mint("qwen35moe.ssm.group_count", 2)
  meta[#meta + 1] = mint("qwen35moe.ssm.state_size", 2)
  meta[#meta + 1] = mint("qwen35moe.ssm.time_step_rank", 2)
  meta[#meta + 1] = mint("qwen35moe.ssm.inner_size", 4)
  meta[#meta + 1] = mint("qwen35moe.ssm.conv_kernel", 2)
  return meta
end

local function linear_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_qkv.weight", {2, 12}, zeros(24)),
    f32(prefix .. "attn_gate.weight", {2, 4}, zeros(8)),
    f32(prefix .. "ssm_beta.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ssm_alpha.weight", {2, 2}, zeros(4)),
    f32(prefix .. "ssm_conv1d.weight", {2, 12}, zeros(24)),
    f32(prefix .. "ssm_a", {2}, zeros(2)),
    f32(prefix .. "ssm_dt.bias", {2}, zeros(2)),
    f32(prefix .. "ssm_norm.weight", {2}, ones(2)),
    f32(prefix .. "ssm_out.weight", {4, 2}, zeros(8)),
  }
end

local function t002_entries(): {Entry}
  return both(
    both(linear_entries("blk.0."), feed_forward_entries("blk.0.")),
    both(
      both(attention_entries("blk.1."), feed_forward_entries("blk.1.")),
      {
        f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, 1, 1}),
        f32("output_norm.weight", {2}, ones(2)),
        f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
      }
    )
  )
end

local function large_entries(): {Entry}
  local function zero(name: string, dims: {integer}): Entry
    local count = 1
    for _, dimension in ipairs(dims) do
      count = count * dimension
    end
    return {
      name = name, dims = dims, code = 0, bytes = string.rep("\0", count * 4)
    }
  end
  local entries: {Entry} = {}
  local function take(name: string, dims: {integer})
    entries[#entries + 1] = zero(name, dims)
  end
  take("blk.0.attn_norm.weight", {1024})
  take("blk.0.attn_q.weight", {1024, 2048})
  take("blk.0.attn_k.weight", {1024, 1024})
  take("blk.0.attn_v.weight", {1024, 1024})
  take("blk.0.attn_q_norm.weight", {1024})
  take("blk.0.attn_k_norm.weight", {1024})
  take("blk.0.attn_output.weight", {1024, 1024})
  take("blk.0.post_attention_norm.weight", {1024})
  take("blk.0.ffn_gate_inp.weight", {1024, 32})
  take("blk.0.ffn_up_exps.weight", {1024, 32, 32})
  take("blk.0.ffn_gate_exps.weight", {1024, 32, 32})
  take("blk.0.ffn_down_exps.weight", {32, 1024, 32})
  take("blk.0.ffn_gate_inp_shexp.weight", {1024})
  take("blk.0.ffn_up_shexp.weight", {1024, 1024})
  take("blk.0.ffn_gate_shexp.weight", {1024, 1024})
  take("blk.0.ffn_down_shexp.weight", {1024, 1024})
  take("token_embd.weight", {1024, 3})
  take("output_norm.weight", {1024})
  take("output.weight", {1024, 3})
  return entries
end

local function large_meta(): {Meta}
  local meta = t001_meta()
  for index, entry in ipairs(meta) do
    if entry.name == "qwen35moe.embedding_length" then
      meta[index] = mint(entry.name, 1024)
    end
  end
  return meta
end

local backend = compute.load() as compute.Backend

local function device_free(): integer
  return backend:memory() as integer
end

local found_device = support.found_device
local on_device = support.on_device

local function scores_of(
  model: compute.Model, token_id: integer, position: integer
): {number}
  local scores, line = model.forward(model, token_id, position)
  test.equal(line, nil)
  return scores as {number}
end

local function equal_rows(left: {number}, right: {number})
  test.equal(#left, #right)
  for index = 1, #right do
    test.equal(left[index], right[index])
  end
end

test.it(
  "a model takes the budget it is given",
  function()
    local handle = loaded(t001_meta(), t001_entries())
    local refused_model, refusal = compute.model(
      backend, handle, 4, "f32", 1, 1, 0
    )
    test.equal(refused_model, nil)
    local line = refusal as string
    test.includes(line, "model")
    test.includes(line, "0")

    local cpu = compute.load(nil, false) as compute.Backend
    local plain = compute.model(cpu, handle, 2, "f32") as compute.Model
    local placement = plain.placement
    test.equal(placement.budget, 0)
    test.equal(placement.resident_bytes, 0)
    test.equal(placement.slot_bytes, 0)
    test.equal(#marked(placement), 0)
    for _, count in ipairs(placement.slots) do
      test.equal(count, 0)
    end
    test.equal(placement.unplaced_bytes, tensor_bytes(handle))
    local row = scores_of(plain, 0, 0)
    for index, wanted in ipairs({1.4142136, 0, 1.4142136}) do
      test.less(math.abs(row[index] - wanted), 1e-5)
    end
    plain:close()
    compute.unload(cpu)
    test.equal(handle.release(handle), nil)

    if found_device() then
      local large = loaded(large_meta(), large_entries())
      local free_before = device_free()
      local big = compute.model(backend, large, nil, "f32") as compute.Model
      test.lessEqual(
        math.abs(big.placement.budget - free_before), 16 * 1024 * 1024
      )
      local held = big.placement.resident_bytes + big.placement.slot_bytes
      big:close()
      local after = device_free()
      test.greaterEqual(after - (free_before - held), math.floor(held * 9 / 10))
      big:close()
      test.greaterEqual(
        device_free() - (free_before - held), math.floor(held * 9 / 10)
      )
      test.equal(large.release(large), nil)

      local small = loaded(t001_meta(), t001_entries())
      local two = compute.model(backend, small, 2, "f32") as compute.Model
      local four = compute.model(backend, small, 4, "f32") as compute.Model
      test.greater(four.placement.reserve, two.placement.reserve)
      two:close()
      four:close()
      test.equal(small.release(small), nil)

      local broken = loaded(
        t002_meta(), without(t002_entries(), "blk.0.ssm_out.weight")
      )
      local before = device_free()
      local missing, missing_line = compute.model(backend, broken, nil, "f32")
      test.equal(missing, nil)
      test.includes(missing_line as string, "blk.0.ssm_out.weight")
      test.lessEqual(math.abs(device_free() - before), 16 * 1024 * 1024)
      test.equal(broken.release(broken), nil)
    end
  end
)

on_device(
  "one expert on the card and one off gives the same reply",
  function()
    local handle = loaded(t001_meta(), t001_entries({1, 0, 0, 1}, 1))
    local first = compute.model(backend, handle, 2, "f32") as compute.Model
    local routes: {integer} = {}
    first.routing_sink = function(routing: compute.Routing)
      routes[#routes + 1] = routing.experts[1]
    end
    local full_zero = scores_of(first, 0, 0)
    local full_one = scores_of(first, 1, 1)
    test.deepEqual(routes, {0, 1})
    test.equal(first.measure.blob_reads, 0)

    local cap = first.placement.resident_bytes + first.placement.reserve + math.floor(
      first.placement.slot_bytes / 2
    )
    first:close()

    local second = compute.model(backend, handle, 2, "f32", 1, 1, cap)
      as compute.Model
    test.deepEqual(second.placement.slots, {1})
    local capped_zero = scores_of(second, 0, 0)
    local capped_one = scores_of(second, 1, 1)
    equal_rows(capped_zero, full_zero)
    equal_rows(capped_one, full_one)
    test.greater(second.measure.blob_reads, 0)
    second:close()
    test.equal(handle.release(handle), nil)
  end
)
