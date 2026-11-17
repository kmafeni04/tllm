local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
local weights = require("tllm.weights")

local function first_row(hidden: {number}, position: integer): {compute.Row}
  return {{sequence = "", hidden = hidden, position = position}}
end

local function only(
  output: {{number}} | nil, line: string | nil
): {number} | nil, string | nil
  if output is nil then
    return nil, line
  end
  return (output as {{number}})[1], line
end

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

local type Floats = {integer: number}

local FLOAT_BYTES: integer = 4

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
  local location = ROOT .. "/device" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
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

-- deterministic numbers that do not line up with each other, so a dot product
-- of them cancels instead of growing
local function wave(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = math.sin(position * 12.9898) / 16
  end
  return result
end

local function turned(values: {number}): {number}
  local result: {number} = {}
  for position = 1, #values do
    result[position] = values[#values - position + 1]
  end
  return result
end

local function filled(value: number, count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = value
  end
  return result
end

local WIDTH: integer = 128
local HEADS: integer = 4
local HEAD_LENGTH: integer = 8
local KV_HEADS: integer = 2
local EXPERTS: integer = 2

-- a linear block whose 128 channels are a multiple of the warp a device
-- kernel convolves them with, and whose convolution has four taps
local function linear_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {WIDTH}, filled(1, WIDTH)),
    f32(prefix .. "attn_qkv.weight", {WIDTH, 128}, wave(WIDTH * 128)),
    f32(prefix .. "attn_gate.weight", {WIDTH, 64}, wave(WIDTH * 64)),
    f32(prefix .. "ssm_beta.weight", {WIDTH, 8}, wave(WIDTH * 8)),
    f32(prefix .. "ssm_alpha.weight", {WIDTH, 8}, wave(WIDTH * 8)),
    f32(prefix .. "ssm_conv1d.weight", {4, 128}, wave(4 * 128)),
    f32(prefix .. "ssm_a", {8}, filled(-0.5, 8)),
    f32(prefix .. "ssm_dt.bias", {8}, wave(8)),
    f32(prefix .. "ssm_norm.weight", {8}, filled(1, 8)),
    f32(prefix .. "ssm_out.weight", {64, WIDTH}, wave(64 * WIDTH)),
  }
end

local function attention_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {WIDTH}, filled(1, WIDTH)),
    f32(
      prefix .. "attn_q.weight",
      {WIDTH, 2 * HEADS * HEAD_LENGTH},
      wave(WIDTH * 2 * HEADS * HEAD_LENGTH)
    ),
    f32(
      prefix .. "attn_k.weight",
      {WIDTH, KV_HEADS * HEAD_LENGTH},
      wave(WIDTH * KV_HEADS * HEAD_LENGTH)
    ),
    f32(
      prefix .. "attn_v.weight",
      {WIDTH, KV_HEADS * HEAD_LENGTH},
      wave(WIDTH * KV_HEADS * HEAD_LENGTH)
    ),
    f32(prefix .. "attn_q_norm.weight", {HEAD_LENGTH}, filled(1, HEAD_LENGTH)),
    f32(prefix .. "attn_k_norm.weight", {HEAD_LENGTH}, filled(1, HEAD_LENGTH)),
    f32(
      prefix .. "attn_output.weight",
      {HEADS * HEAD_LENGTH, WIDTH},
      wave(HEADS * HEAD_LENGTH * WIDTH)
    ),
  }
end

local function feed_forward_entries(prefix: string): {Entry}
  local shared: integer = 16
  return {
    f32(prefix .. "post_attention_norm.weight", {WIDTH}, filled(1, WIDTH)),
    f32(
      prefix .. "ffn_gate_inp.weight", {WIDTH, EXPERTS}, wave(WIDTH * EXPERTS)
    ),
    f32(prefix .. "ffn_gate_inp_shexp.weight", {WIDTH}, wave(WIDTH)),
    f32(
      prefix .. "ffn_up_shexp.weight", {WIDTH, shared}, wave(WIDTH * shared)
    ),
    f32(
      prefix .. "ffn_gate_shexp.weight", {WIDTH, shared}, wave(WIDTH * shared)
    ),
    f32(
      prefix .. "ffn_down_shexp.weight", {shared, WIDTH}, wave(shared * WIDTH)
    ),
    f32(
      prefix .. "ffn_up_exps.weight",
      {WIDTH, shared, EXPERTS},
      wave(WIDTH * shared * EXPERTS)
    ),
    f32(
      prefix .. "ffn_gate_exps.weight",
      {WIDTH, shared, EXPERTS},
      wave(WIDTH * shared * EXPERTS)
    ),
    f32(
      prefix .. "ffn_down_exps.weight",
      {shared, WIDTH, EXPERTS},
      wave(shared * WIDTH * EXPERTS)
    ),
  }
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

local function model_meta(): {Meta}
  return {
    mtext("general.architecture", "devicemoe"),
    mint("devicemoe.block_count", 2),
    mint("devicemoe.embedding_length", WIDTH),
    mint("devicemoe.expert_used_count", 1),
    mfloat("devicemoe.attention.layer_norm_rms_epsilon", 0),
    mint("devicemoe.context_length", 4),
    mint("devicemoe.full_attention_interval", 2),
    mint("devicemoe.attention.head_count", HEADS),
    mint("devicemoe.rope.dimension_count", HEAD_LENGTH),
    mint("devicemoe.rope.freq_base", 10000),
    mint("devicemoe.ssm.group_count", 4),
    mint("devicemoe.ssm.state_size", 8),
    mint("devicemoe.ssm.time_step_rank", 8),
    mint("devicemoe.ssm.inner_size", 64),
    mint("devicemoe.ssm.conv_kernel", 4),
  }
end

local function model_entries(): {Entry}
  local blocks = both(
    both(linear_entries("blk.0."), feed_forward_entries("blk.0.")),
    both(attention_entries("blk.1."), feed_forward_entries("blk.1."))
  )
  return both(
    blocks,
    {
      f32("token_embd.weight", {WIDTH, 2}, wave(WIDTH * 2)),
      f32("output_norm.weight", {WIDTH}, filled(1, WIDTH)),
      f32("output.weight", {WIDTH, 4}, wave(WIDTH * 4)),
    }
  )
end

local GEOMETRY: compute.Geometry = {
  heads = HEADS,
  kv_heads = KV_HEADS,
  head_length = HEAD_LENGTH,
  rope_dimensions = HEAD_LENGTH,
  rope_base = 10000,
  eps = 0,
  positions = 4,
  experts_used = 1,
  linear_key_heads = 4,
  linear_key_length = 8,
  linear_value_heads = 8,
  linear_value_length = 8,
  linear_conv_kernel = 4,
  cache_type = "f32",
}

-- 4096 by 8192 f32 tensors hold 128 MiB
local LARGE: integer = 4096 * 8192 * FLOAT_BYTES

local function large_handle(): weights.Handle
  local handle, failure = weights.load(
    fixture(
      {},
      {
        {
          name = "blk.0.ffn_up_exps.weight",
          dims = {4096, 8192},
          code = 0,
          bytes = string.rep("\0", LARGE),
        },
      }
    )
  )
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function bytes_of(handle: weights.Handle): integer
  local total = 0
  for _, tensor in pairs(handle.tensors) do
    total = total + tensor.bytes
  end
  return total
end

local backend = compute.load() as compute.Backend

local function device_free(): integer
  local bytes = backend:memory()
  return bytes as integer
end

local function found_device(): boolean
  local bytes, total = backend:memory()
  return bytes ~= nil and total ~= nil
end

-- the cases that need a card say why they did not run
local function on_device(name: string, body: function())
  if found_device() then
    test.it(name, body)
  else
    test.skip(name .. " (no CUDA device)", function() end)
  end
end

local function agreeing(
  what: string,
  got: {number} | nil,
  got_line: string | nil,
  wanted: {number} | nil,
  wanted_line: string | nil
)
  test.equal(got_line, nil, what)
  test.equal(wanted_line, nil, what)
  test.truthy(got, what)
  test.truthy(wanted, what)
  local first = got as {number}
  local second = wanted as {number}
  test.equal(#first, #second, what)
  for index = 1, #second do
    test.less(
      math.abs(first[index] - second[index]),
      0.00001 * math.max(1, math.abs(second[index])),
      what
    )
  end
end

local function attached(b: compute.Backend, h: weights.Handle): compute.Maths
  local maths, line = compute.attach(b, h)
  test.equal(line, nil)
  test.truthy(maths)
  return maths as compute.Maths
end

local function attended(
  b: compute.Backend, h: weights.Handle
): compute.Attention
  local attention, line = compute.attend(b, h, GEOMETRY)
  test.equal(line, nil)
  test.truthy(attention)
  return attention as compute.Attention
end

local function forwarded(
  b: compute.Backend, h: weights.Handle
): compute.FeedForward
  local feed, line = compute.feed_forward(b, h, GEOMETRY)
  test.equal(line, nil)
  test.truthy(feed)
  return feed as compute.FeedForward
end

local function pressed(
  b: compute.Backend, h: weights.Handle
): compute.LinearAttention
  local linear, line = compute.linear_attention(b, h, GEOMETRY)
  test.equal(line, nil)
  test.truthy(linear)
  return linear as compute.LinearAttention
end

local function passed(b: compute.Backend, h: weights.Handle): compute.Model
  local model, line = compute.model(b, h, 2, "f32")
  test.equal(line, nil)
  test.truthy(model)
  return model as compute.Model
end

local function model_handle(): weights.Handle
  local handle, failure = weights.load(fixture(model_meta(), model_entries()))
  test.equal(failure, nil)
  return handle as weights.Handle
end

on_device(
  "a loaded model sits in the device once",
  function()
    local handle = large_handle()
    local bytes = bytes_of(handle)
    test.greaterEqual(bytes, 134217728)

    local before = device_free()
    local maths = attached(backend, handle)
    local after = device_free()
    test.greaterEqual(before - after, math.floor(bytes * 9 / 10))

    local free_after, total_after = backend:memory()
    test.greaterEqual(total_after as integer, free_after as integer)

    local name = backend.device
    test.truthy(name)
    test.greater(#(name as string), 0)

    local maps = fs.read("/proc/self/maps") as string
    test.includes(maps, "target/tllm/libggml-cuda")

    local feed_before = device_free()
    local feed = forwarded(backend, handle)
    local feed_after = device_free()
    test.less(feed_before - feed_after, math.floor(bytes / 10))
    test.equal(backend.device, name)

    maths:close()
    feed:close()
    test.equal(handle.release(handle), nil)
  end
)

on_device(
  "the device maths matches the CPU maths",
  function()
    local cpu = compute.load(nil, false) as compute.Backend
    local handle = model_handle()
    local name = backend.device
    test.truthy(name)

    local maths = attached(backend, handle)
    local cpu_maths = attached(cpu, handle)
    local probe = wave(WIDTH)
    local other = turned(probe)
    local short = wave(HEAD_LENGTH)
    local four = wave(4)

    agreeing(
      "matvec",
      maths:matvec("blk.1.attn_q.weight", probe),
      nil,
      cpu_maths:matvec("blk.1.attn_q.weight", probe),
      nil
    )
    agreeing(
      "row",
      maths:row("token_embd.weight", 1),
      nil,
      cpu_maths:row("token_embd.weight", 1),
      nil
    )
    agreeing(
      "norm",
      maths:norm(probe, "blk.0.attn_norm.weight", 0),
      nil,
      cpu_maths:norm(probe, "blk.0.attn_norm.weight", 0),
      nil
    )
    agreeing(
      "rope",
      maths:rope(short, 1, HEAD_LENGTH, 10000),
      nil,
      cpu_maths:rope(short, 1, HEAD_LENGTH, 10000),
      nil
    )
    agreeing("softmax", maths:softmax(four), nil, cpu_maths:softmax(four), nil)
    agreeing(
      "add", maths:add(probe, other), nil, cpu_maths:add(probe, other), nil
    )
    agreeing(
      "swiglu",
      maths:swiglu(probe, other),
      nil,
      cpu_maths:swiglu(probe, other),
      nil
    )
    agreeing(
      "sigmoid_gate",
      maths:sigmoid_gate(probe, other),
      nil,
      cpu_maths:sigmoid_gate(probe, other),
      nil
    )

    local attention = attended(backend, handle)
    local cpu_attention = attended(cpu, handle)
    agreeing(
      "attention block",
      only(attention:block(first_row(probe, 0), "blk.1.")),
      nil,
      only(cpu_attention:block(first_row(probe, 0), "blk.1.")),
      nil
    )

    local feed = forwarded(backend, handle)
    local cpu_feed = forwarded(cpu, handle)
    agreeing(
      "feed forward block",
      only(feed:block(first_row(probe, 0), "blk.0.")),
      nil,
      only(cpu_feed:block(first_row(probe, 0), "blk.0.")),
      nil
    )
    agreeing(
      "feed forward scores",
      only(feed:scores({probe})),
      nil,
      only(cpu_feed:scores({probe})),
      nil
    )

    local linear = pressed(backend, handle)
    local cpu_linear = pressed(cpu, handle)
    agreeing(
      "linear attention block",
      only(linear:block(first_row(probe, 0), "blk.0.")),
      nil,
      only(cpu_linear:block(first_row(probe, 0), "blk.0.")),
      nil
    )

    local model = passed(backend, handle)
    local cpu_model = passed(cpu, handle)
    agreeing(
      "the model pass at the first position",
      model:forward(0, 0),
      nil,
      cpu_model:forward(0, 0),
      nil
    )
    agreeing(
      "the model pass at the second position",
      model:forward(1, 1),
      nil,
      cpu_model:forward(1, 1),
      nil
    )
    test.equal(backend.device, name)

    maths:close()
    cpu_maths:close()
    attention:close()
    cpu_attention:close()
    feed:close()
    cpu_feed:close()
    linear:close()
    cpu_linear:close()
    model:close()
    cpu_model:close()
    test.equal(handle.release(handle), nil)
  end
)

test.it(
  "a CPU-only load reports no device and still computes",
  function()
    local cpu = compute.load(nil, false) as compute.Backend
    local handle, failure = weights.load(
      fixture(
        {},
        {
          f32("weight", {2, 2}, {1, 2, 3, 4}),
          f32("gain", {2}, {1, 2}),
        }
      )
    )
    test.equal(failure, nil)
    local loaded = handle as weights.Handle

    local before = cpu:memory()
    local maths = attached(cpu, loaded)
    test.equal(cpu.device, nil)

    local after = cpu:memory()
    if not (before is nil) and not (after is nil) then
      test.lessEqual(
        math.abs((after as integer) - (before as integer)), 16 * 1024 * 1024
      )
    end

    test.deepEqual(maths:matvec("weight", {1, 1}), {3, 7})
    test.deepEqual(
      maths:norm({3, 4}, "gain", 0), {0.8485281467437744, 2.262741804122925}
    )

    if before is nil then
      local other = compute.load() as compute.Backend
      local other_free, other_total = other:memory()
      test.equal(other_free, nil)
      test.equal(other_total, nil)
      local other_maths = attached(other, loaded)
      test.deepEqual(other_maths:matvec("weight", {1, 1}), {3, 7})
      test.deepEqual(
        other_maths:norm({3, 4}, "gain", 0),
        {0.8485281467437744, 2.262741804122925}
      )
      other_maths:close()
    end

    test.includes(loaded.release(loaded) as string, "still use it")
    maths:close()
    test.equal(loaded.release(loaded), nil)
  end
)

on_device(
  "releasing the model gives the device memory back",
  function()
    local handle = large_handle()
    local bytes = bytes_of(handle)

    local maths = attached(backend, handle)
    local loaded = device_free()
    maths:close()
    test.equal(handle.release(handle), nil)
    local released = device_free()
    test.greaterEqual(released - loaded, math.floor(bytes * 9 / 10))

    handle.release(handle)
    local again = device_free()
    test.greaterEqual(again - loaded, math.floor(bytes * 9 / 10))
  end
)
