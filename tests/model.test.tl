local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
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
  local location = ROOT .. "/model" .. written .. ".gguf"
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

local function repeated(values: {number}, times: integer): {number}
  local result: {number} = {}
  for _ = 1, times do
    for _, value in ipairs(values) do
      result[#result + 1] = value
    end
  end
  return result
end

local function f32(name: string, dims: {integer}, values: {number}): Entry
  return {name = name, dims = dims, code = 0, bytes = floats(values)}
end

local function mint(name: string, value: integer): Meta
  return {name = name, code = 4, bytes = u32(value)}
end

local function mint32(name: string, value: integer): Meta
  return {name = name, code = 5, bytes = u32(value)}
end

local function mfloat(name: string, value: number): Meta
  return {name = name, code = 6, bytes = floats({value})}
end

local function mtext(name: string, value: string): Meta
  return {name = name, code = 8, bytes = text(value)}
end

local function joined(first: {number}, second: {number}): {number}
  local result: {number} = {}
  for _, value in ipairs(first) do
    result[#result + 1] = value
  end
  for _, value in ipairs(second) do
    result[#result + 1] = value
  end
  return result
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

local function without(entries: {Entry}, name: string): {Entry}
  local result: {Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name ~= name then
      result[#result + 1] = entry
    end
  end
  return result
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

local function full_block(prefix: string): {Entry}
  return both(attention_entries(prefix), feed_forward_entries(prefix))
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

local function root_entries(rows: integer, values: {number}): {Entry}
  return {
    f32("token_embd.weight", {2, rows}, values),
    f32("output_norm.weight", {2}, ones(2)),
    f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
  }
end

local function t001_entries(): {Entry}
  return both(full_block("blk.0."), root_entries(2, {1, 0, 3, 4}))
end

local function t002_meta(): {Meta}
  local meta = with_integer(t001_meta(), "qwen35moe.block_count", 2)
  meta = with_integer(meta, "qwen35moe.full_attention_interval", 2)
  meta = with_integer(meta, "qwen35moe.ssm.group_count", 2)
  meta = with_integer(meta, "qwen35moe.ssm.state_size", 2)
  meta = with_integer(meta, "qwen35moe.ssm.time_step_rank", 2)
  meta = with_integer(meta, "qwen35moe.ssm.inner_size", 4)
  return with_integer(meta, "qwen35moe.ssm.conv_kernel", 2)
end

local function t002_entries(): {Entry}
  local entries = both(
    linear_block("blk.0.", {1, 0, 1, 0, 0, 1, 0, 1}), full_block("blk.1.")
  )
  return both(entries, root_entries(3, {1, 0, 3, 4, 1, 1}))
end

local function linear_only_meta(): {Meta}
  local meta = t002_meta()
  meta = without_meta(meta, "qwen35moe.full_attention_interval")
  meta = without_meta(meta, "qwen35moe.attention.head_count")
  meta = without_meta(meta, "qwen35moe.rope.dimension_count")
  return without_meta(meta, "qwen35moe.rope.freq_base")
end

local function linear_only_entries(): {Entry}
  local entries = both(
    linear_block("blk.0.", zeros(8)), linear_block("blk.1.", zeros(8))
  )
  return both(entries, root_entries(2, {1, 0, 3, 4}))
end

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries))
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function model_of(
  handle: weights.Handle, positions: integer | nil
): compute.Model
  local model, failure = compute.model(backend, handle, positions, "f32")
  test.equal(failure, nil)
  return model as compute.Model
end

local function refused(value: {number} | nil, line: string | nil): string
  test.equal(value, nil)
  test.truthy(line)
  local message = line as string
  test.equal(message:find("\n"), nil)
  return message
end

local function forward(
  model: compute.Model, token_id: integer, position: integer
): {number}
  local scores, line = model.forward(model, token_id, position)
  test.equal(line, nil)
  return scores as {number}
end

local function refused_forward(
  model: compute.Model, token_id: integer, position: integer
): string
  return refused(model.forward(model, token_id, position))
end

local function close(got: {number}, wanted: {number}, slack: number)
  test.equal(#got, #wanted)
  for index, number in ipairs(wanted) do
    test.less(math.abs(got[index] - number), slack)
  end
end

local function differs(got: {number}, wanted: {number})
  test.equal(#got, #wanted)
  local apart = false
  for index = 1, #got do
    if math.abs(got[index] - wanted[index]) > 1e-6 then
      apart = true
    end
  end
  test.truthy(apart)
end

test.it(
  "one token through a full-attention model gives its scores",
  function()
    local model = model_of(loaded(t001_meta(), t001_entries()), 2)
    close(forward(model, 0, 0), {1.4142136, 0, 1.4142136}, 1e-5)
    close(forward(model, 1, 1), {0.8485281, 1.1313708, 1.9798990}, 1e-5)

    local greedy = model_of(
      loaded(
        with_integer(t001_meta(), "qwen35moe.expert_used_count", 3),
        t001_entries()
      ),
      nil
    )
    local line = refused_forward(greedy, 0, 0)
    test.includes(line, "ffn_gate_inp.weight")
  end
)

test.it(
  "each block of the run runs as its kind and its numbers are added",
  function()
    local model = model_of(loaded(t002_meta(), t002_entries()), 2)
    close(forward(model, 2, 0), {0.5438875, 1.3054449, 1.8493324}, 1e-4)

    local no_out = model_of(
      loaded(t002_meta(), without(t002_entries(), "blk.0.ssm_out.weight")), nil
    )
    local out_line = refused_forward(no_out, 2, 0)
    test.includes(out_line, "blk.0.ssm_out.weight")

    local no_value = model_of(
      loaded(t002_meta(), without(t002_entries(), "blk.1.attn_v.weight")), nil
    )
    local value_line = refused_forward(no_value, 2, 0)
    test.includes(value_line, "blk.1.attn_v.weight")
    local order_line = refused_forward(no_value, 1, 0)
    test.includes(order_line, "blk.0.")
    test.includes(order_line, "position 0")

    local drafted = model_of(
      loaded(
        with_integer(
          with_integer(t001_meta(), "qwen35moe.block_count", 2),
          "qwen35moe.nextn_predict_layers",
          1
        ),
        t001_entries()
      ),
      nil
    )
    close(forward(drafted, 1, 0), {0.8485281, 1.1313708, 1.9798990}, 1e-5)

    local undrafted = model_of(
      loaded(
        with_integer(t001_meta(), "qwen35moe.block_count", 2), t001_entries()
      ),
      nil
    )
    local draft_line = refused_forward(undrafted, 1, 0)
    test.includes(draft_line, "blk.1.attn_norm.weight")

    local linear_only = model_of(
      loaded(linear_only_meta(), linear_only_entries()), 2
    )
    close(forward(linear_only, 1, 0), {0.8485281, 1.1313708, 1.9798990}, 1e-5)
  end
)

test.it(
  "a token and a position that do not fit are refused",
  function()
    local short = model_of(loaded(t001_meta(), t001_entries()), 2)
    local short_line = refused_forward(short, 1, 2)
    test.includes(short_line, "forward")
    test.includes(short_line, "2")
    close(forward(short, 1, 0), {0.8485281, 1.1313708, 1.9798990}, 1e-5)

    local full = model_of(loaded(t001_meta(), t001_entries()), nil)
    forward(full, 1, 0)
    forward(full, 1, 1)
    forward(full, 1, 2)
    forward(full, 1, 3)
    local over_line = refused_forward(full, 1, 4)
    test.includes(over_line, "blk.0.")
    test.includes(over_line, "4")

    local one = model_of(loaded(t001_meta(), t001_entries()), 1)
    close(forward(one, 1, 0), {0.8485281, 1.1313708, 1.9798990}, 1e-5)
    local next_line = refused_forward(one, 1, 1)
    test.includes(next_line, "blk.0.")
    test.includes(next_line, "1")

    local negative = refused_forward(
      model_of(loaded(t001_meta(), t001_entries()), nil), -1, 0
    )
    test.includes(negative, "token_embd.weight")

    local past = refused_forward(
      model_of(loaded(t001_meta(), t001_entries()), nil), 2, 0
    )
    test.includes(past, "token_embd.weight")

    local absent = refused_forward(
      model_of(
        loaded(t001_meta(), without(t001_entries(), "token_embd.weight")), nil
      ),
      0,
      0
    )
    test.includes(absent, "token_embd.weight")

    local wide_line = refused_forward(
      model_of(
        loaded(
          t001_meta(),
          replaced(
            t001_entries(),
            "token_embd.weight",
            f32("token_embd.weight", {3, 2}, zeros(6))
          )
        ),
        nil
      ),
      0,
      0
    )
    test.includes(wide_line, "token_embd.weight")
    test.includes(wide_line, "id 0")
  end
)

local record Bad
  meta: {Meta}
  entries: {Entry}
  marker: string
  positions: integer | nil
end

test.it(
  "a key or a number that does not fit is refused with one line",
  function()
    local with_heads_two = with_integer(
      t001_meta(), "qwen35moe.attention.head_count", 2
    )
    local with_two_blocks = with_integer(
      t001_meta(), "qwen35moe.block_count", 2
    )

    local cases: {Bad} = {
      {
        meta = without_meta(t001_meta(), "qwen35moe.rope.freq_base"),
        entries = t001_entries(),
        marker = "qwen35moe.rope.freq_base",
      },
      {
        meta = without_meta(t001_meta(), "qwen35moe.attention.head_count"),
        entries = t001_entries(),
        marker = "qwen35moe.attention.head_count",
      },
      {
        meta = without_meta(t001_meta(), "general.architecture"),
        entries = t001_entries(),
        marker = "general.architecture",
      },
      {
        meta = with_heads_two,
        entries = replaced(
          t001_entries(),
          "blk.0.attn_q.weight",
          f32("blk.0.attn_q.weight", {2, 6}, zeros(12))
        ),
        marker = "blk.0.attn_q.weight",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.attention.head_count", 16),
        entries = replaced(
          replaced(
            t001_entries(),
            "blk.0.attn_q.weight",
            f32("blk.0.attn_q.weight", {2, 64}, zeros(128))
          ),
          "blk.0.attn_k.weight",
          f32("blk.0.attn_k.weight", {2, 6}, zeros(12))
        ),
        marker = "blk.0.attn_k.weight",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.rope.dimension_count", 3),
        entries = t001_entries(),
        marker = "qwen35moe.rope.dimension_count",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.rope.freq_base", 0),
        entries = t001_entries(),
        marker = "qwen35moe.rope.freq_base",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.block_count", 0),
        entries = t001_entries(),
        marker = "qwen35moe.block_count",
      },
      {
        meta = with_integer(
          t001_meta(), "qwen35moe.full_attention_interval", 0
        ),
        entries = t001_entries(),
        marker = "qwen35moe.full_attention_interval",
      },
      {
        meta = {
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
          mint32("qwen35moe.nextn_predict_layers", -1),
        },
        entries = t001_entries(),
        marker = "qwen35moe.nextn_predict_layers",
      },
      {
        meta = t001_meta(),
        entries = t001_entries(),
        marker = "positions",
        positions = 0,
      },
      {
        meta = t001_meta(),
        entries = t001_entries(),
        marker = "positions",
        positions = 5,
      },
      {
        meta = without_meta(t002_meta(), "qwen35moe.ssm.group_count"),
        entries = t002_entries(),
        marker = "qwen35moe.ssm.group_count",
      },
      {
        meta = with_integer(t002_meta(), "qwen35moe.ssm.inner_size", 5),
        entries = t002_entries(),
        marker = "qwen35moe.ssm.inner_size",
      },
      {
        meta = t001_meta(),
        entries = without(t001_entries(), "blk.0.attn_q.weight"),
        marker = "blk.0.attn_q.weight",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.rope.dimension_count", 1),
        entries = t001_entries(),
        marker = "qwen35moe.rope.dimension_count",
      },
      {
        meta = with_integer(t001_meta(), "qwen35moe.rope.dimension_count", 4),
        entries = t001_entries(),
        marker = "qwen35moe.rope.dimension_count",
      },
      {
        meta = with_integer(
          with_integer(t002_meta(), "qwen35moe.ssm.time_step_rank", 3),
          "qwen35moe.ssm.inner_size",
          6
        ),
        entries = t002_entries(),
        marker = "qwen35moe.ssm.time_step_rank",
      },
      {
        meta = with_integer(
          with_two_blocks, "qwen35moe.nextn_predict_layers", 2
        ),
        entries = t001_entries(),
        marker = "qwen35moe.block_count",
      },
    }

    for _, bad in ipairs(cases) do
      local handle = loaded(bad.meta, bad.entries)
      local model, line = compute.model(backend, handle, bad.positions, "f32")
      test.equal(model, nil)
      local message = refused(nil, line)
      test.includes(message, "model")
      test.includes(message, bad.marker)
    end

    local forgiving = model_of(loaded(t001_meta(), t001_entries()), nil)
    test.truthy(#forward(forgiving, 0, 0) > 0)

    local forged = {
      path = "forged",
      version = "forged",
      matvec = function(
        _: compute.Backend, _matrix: {{number}}, _vector: {number}
      ): {number} | nil, string | nil
        return nil, "forged"
      end,
    } as compute.Backend
    local forged_model, forged_line = compute.model(
      forged, loaded(t001_meta(), t001_entries()), nil, "f32"
    )
    test.equal(forged_model, nil)
    test.includes(refused(nil, forged_line), "model")

    local gone = fixture(t001_meta(), t001_entries())
    local gone_handle, gone_failure = weights.load(gone)
    test.equal(gone_failure, nil)
    os.remove(gone)
    local gone_model, gone_line = compute.model(
      backend, gone_handle as weights.Handle, nil, "f32"
    )
    test.equal(gone_model, nil)
    test.includes(refused(nil, gone_line), "model")
  end
)

local function t005_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 4),
    mint("qwen35moe.attention.head_count", 4),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 4),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
  }
end

local function t005_entries(): {Entry}
  local entries = {
    f32("token_embd.weight", {4, 2}, {1, 1, 1, 1, 2, 0, 0, 0}),
    f32("blk.0.attn_norm.weight", {4}, ones(4)),
    f32(
      "blk.0.attn_q.weight",
      {4, 16},
      repeated({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 4)
    ),
    f32("blk.0.attn_k.weight", {4, 2}, {1, 0, 0, 0, 0, 1, 0, 0}),
    f32("blk.0.attn_v.weight", {4, 2}, {1, 0, 0, 0, 0, 1, 0, 0}),
    f32("blk.0.attn_q_norm.weight", {2}, ones(2)),
    f32("blk.0.attn_k_norm.weight", {2}, ones(2)),
    f32("blk.0.attn_output.weight", {8, 4}, zeros(32)),
    f32("blk.0.post_attention_norm.weight", {4}, ones(4)),
    f32("blk.0.ffn_gate_inp.weight", {4, 2}, zeros(8)),
    f32("blk.0.ffn_gate_inp_shexp.weight", {4}, zeros(4)),
    f32("blk.0.ffn_up_shexp.weight", {4, 2}, zeros(8)),
    f32("blk.0.ffn_gate_shexp.weight", {4, 2}, zeros(8)),
    f32("blk.0.ffn_down_shexp.weight", {2, 4}, zeros(8)),
    f32("blk.0.ffn_up_exps.weight", {4, 2, 2}, zeros(16)),
    f32("blk.0.ffn_gate_exps.weight", {4, 2, 2}, zeros(16)),
    f32("blk.0.ffn_down_exps.weight", {2, 4, 2}, zeros(16)),
    f32("output_norm.weight", {4}, ones(4)),
    f32("output.weight", {4, 2}, {1, 0, 0, 0, 0, 1, 1, 0}),
  }
  return entries
end

test.it(
  "the head length comes from the query tensor and not the width",
  function()
    local model = model_of(loaded(t005_meta(), t005_entries()), 2)
    close(forward(model, 0, 0), {1, 2}, 1e-5)
  end
)

local function t006_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 8),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 4),
    mint("qwen35moe.rope.freq_base", 100),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 4),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
  }
end

local function t006_entries(): {Entry}
  local identity = {
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
  }
  local gate = {
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
  }
  return {
    f32(
      "token_embd.weight",
      {8, 2},
      {1, 1, 1, 1, 0, 0, 0, 0, 2, 1, 3, 1, 0, 0, 0, 0}
    ),
    f32("blk.0.attn_norm.weight", {8}, ones(8)),
    f32("blk.0.attn_q.weight", {8, 8}, joined(identity, zeros(32))),
    f32("blk.0.attn_k.weight", {8, 4}, identity),
    f32("blk.0.attn_v.weight", {8, 4}, identity),
    f32("blk.0.attn_q_norm.weight", {4}, ones(4)),
    f32("blk.0.attn_k_norm.weight", {4}, ones(4)),
    f32("blk.0.attn_output.weight", {4, 8}, joined(gate, zeros(32))),
    f32("blk.0.post_attention_norm.weight", {8}, ones(8)),
    f32("blk.0.ffn_gate_inp.weight", {8, 2}, zeros(16)),
    f32("blk.0.ffn_gate_inp_shexp.weight", {8}, zeros(8)),
    f32("blk.0.ffn_up_shexp.weight", {8, 2}, zeros(16)),
    f32("blk.0.ffn_gate_shexp.weight", {8, 2}, zeros(16)),
    f32("blk.0.ffn_down_shexp.weight", {2, 8}, zeros(16)),
    f32("blk.0.ffn_up_exps.weight", {8, 2, 2}, zeros(32)),
    f32("blk.0.ffn_gate_exps.weight", {8, 2, 2}, zeros(32)),
    f32("blk.0.ffn_down_exps.weight", {2, 8, 2}, zeros(32)),
    f32("output_norm.weight", {8}, ones(8)),
    f32(
      "output.weight",
      {8, 3},
      {
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
      }
    ),
  }
end

test.it(
  "the rope values reach the attention",
  function()
    local model = model_of(loaded(t006_meta(), t006_entries()), 2)
    close(forward(model, 0, 0), {1.4142136, 1.4142136, 2.8284271}, 1e-5)
    local first = forward(model, 1, 1)
    close(first, {1.4666941, 0.7843212, 2.2510153}, 1e-5)

    local base_one = model_of(
      loaded(
        with_integer(t006_meta(), "qwen35moe.rope.freq_base", 1), t006_entries()
      ),
      2
    )
    forward(base_one, 0, 0)
    differs(forward(base_one, 1, 1), first)

    local narrow = model_of(
      loaded(
        with_integer(t006_meta(), "qwen35moe.rope.dimension_count", 2),
        t006_entries()
      ),
      2
    )
    forward(narrow, 0, 0)
    differs(forward(narrow, 1, 1), first)
  end
)

local function model_sequences(
  handle: weights.Handle, positions: integer, sequences: integer
): compute.Model
  local model, failure = compute.model(
    backend, handle, positions, "f32", sequences
  )
  test.equal(failure, nil)
  return model as compute.Model
end

local function contributing_entries(): {Entry}
  return replaced(
    t001_entries(),
    "blk.0.attn_output.weight",
    f32("blk.0.attn_output.weight", {2, 2}, {1, 0, 0, 1})
  )
end

test.it(
  "a pass gives each row the numbers it gets alone",
  function()
    local handle = loaded(t001_meta(), contributing_entries())
    local model = model_sequences(handle, 4, 2)
    test.equal(model.sequences, 2)
    test.equal(model:reserve("a", 4), nil)
    test.equal(model:reserve("b", 4), nil)

    local reference_a = model_of(handle, 4)
    local a0 = forward(reference_a, 0, 0)
    local reference_b = model_of(handle, 4)
    local b0 = forward(reference_b, 1, 0)
    test.greater(math.abs(b0[2] - a0[2]), 1e-3)

    local rows, line = model:pass(
      {
        {sequence = "a", token = 0, position = 0},
        {sequence = "b", token = 1, position = 0},
      }
    )
    test.equal(line, nil)
    local got = rows as {{number}}
    test.equal(#got, 2)
    test.equal(#got[1], 3)
    test.equal(#got[2], 3)
    close(got[1], a0, 1e-5)
    close(got[2], b0, 1e-5)

    local a1 = forward(reference_a, 0, 1)
    local b1 = forward(reference_b, 1, 1)
    local second, second_line = model:pass(
      {
        {sequence = "b", token = 1, position = 1},
        {sequence = "a", token = 0, position = 1},
      }
    )
    test.equal(second_line, nil)
    local other_rows = second as {{number}}
    close(other_rows[1], b1, 1e-5)
    close(other_rows[2], a1, 1e-5)

    local fresh = model_sequences(handle, 4, 2)
    test.equal(fresh:reserve("a", 4), nil)
    test.equal(fresh:reserve("b", 4), nil)
    local duplicate, duplicate_line = fresh:pass(
      {
        {sequence = "a", token = 0, position = 0},
        {sequence = "a", token = 1, position = 0},
      }
    )
    test.equal(duplicate, nil)
    test.includes(duplicate_line as string, "pass")
    test.includes(duplicate_line as string, "a")

    local unknown, unknown_line = fresh:pass(
      {{sequence = "c", token = 0, position = 0}}
    )
    test.equal(unknown, nil)
    test.includes(unknown_line as string, "pass")
    test.includes(unknown_line as string, "c")

    local ahead, ahead_line = fresh:pass(
      {{sequence = "a", token = 0, position = 2}}
    )
    test.equal(ahead, nil)
    test.includes(ahead_line as string, "pass")
    test.includes(ahead_line as string, "a")

    local kept, kept_line = fresh:pass(
      {
        {sequence = "a", token = 0, position = 0},
        {sequence = "b", token = 1, position = 0},
      }
    )
    test.equal(kept_line, nil)
    local kept_rows = kept as {{number}}
    close(kept_rows[1], a0, 1e-5)
    close(kept_rows[2], b0, 1e-5)

    local empty, empty_line = fresh:pass({})
    test.equal(empty, nil)
    test.includes(empty_line as string, "pass")

    local one = model_sequences(handle, 4, 2)
    close(forward(one, 0, 0), a0, 1e-5)
    test.equal(one:cache_bytes("", "blk.0."), 64)
  end
)

test.it(
  "a sequence's space is reserved, reported and released",
  function()
    local handle = loaded(t001_meta(), contributing_entries())
    local model = model_sequences(handle, 4, 2)
    test.equal(model:cache_bytes("a", "blk.0."), nil)
    test.equal(model:reserve("a", 3), nil)
    test.equal(model:cache_bytes("a", "blk.0."), 48)

    local again = model:reserve("a", 3) as string
    test.includes(again, "reserve")
    test.includes(again, "a")

    local zero = model:reserve("b", 0) as string
    test.truthy(zero)
    local wide = model:reserve("b", 5) as string
    test.truthy(wide)

    test.equal(model:reserve("b", 4), nil)
    test.equal(model:cache_bytes("b", "blk.0."), 64)

    local full = model:reserve("c", 1) as string
    test.truthy(full)

    model:release("a")
    test.equal(model:cache_bytes("a", "blk.0."), nil)
    model:release("a")
    model:release("zzz")
    test.equal(model:cache_bytes("b", "blk.0."), 64)

    test.equal(model:reserve("c", 4), nil)
    test.equal(model:cache_bytes("c", "blk.0."), 64)

    local released, released_line = model:pass(
      {{sequence = "a", token = 0, position = 0}}
    )
    test.equal(released, nil)
    test.includes(released_line as string, "pass")
    test.includes(released_line as string, "a")

    local single = model_of(handle, 4)
    test.equal(single:reserve("a", 1), nil)
    local second = single:reserve("b", 1) as string
    test.truthy(second)
  end
)

ffi.cdef(
  [[
struct tllm_pass_wall { long seconds; long nanos; };
int clock_gettime(int clock_id, struct tllm_pass_wall *at);
]]
)

local record PassWall
  seconds: integer
  nanos: integer
end

local record PassClocks
  clock_gettime: function(integer, ffi.CData): integer
end

local function now(): number
  local at = ffi.new("struct tllm_pass_wall") as PassWall
  (ffi.C as PassClocks).clock_gettime(1, at as ffi.CData)
  return tonumber(at.seconds) + tonumber(at.nanos) / 1e9
end

test.it(
  "a pass costs no more than its rows one by one",
  function()
    local handle = loaded(t001_meta(), contributing_entries())
    local rows: {compute.Input} = {
      {sequence = "a", token = 0, position = 0},
      {sequence = "b", token = 0, position = 0},
      {sequence = "c", token = 0, position = 0},
      {sequence = "d", token = 0, position = 0},
    }

    local function pair(): number, number
      local batched = model_sequences(handle, 4, 4)
      local single = model_sequences(handle, 4, 4)
      for _, name in ipairs({"a", "b", "c", "d"}) do
        test.equal(batched:reserve(name, 4), nil)
        test.equal(single:reserve(name, 4), nil)
      end
      collectgarbage("collect")
      local started = now()
      local result = batched:pass(rows)
      local together = now() - started
      test.truthy(result)
      local apart: number = 0
      collectgarbage("collect")
      for _, row in ipairs(rows) do
        local before = now()
        single:pass({row})
        apart = apart + (now() - before)
      end
      return together, apart
    end

    pair()
    pair()
    collectgarbage("collect")
    local together: number = 0
    local apart: number = 0
    for _ = 1, 3 do
      local one, four = pair()
      together = together + one
      apart = apart + four
    end
    test.lessEqual(together, apart)
  end
)

test.it(
  "a released name starts from zeros",
  function()
    local handle = loaded(t002_meta(), t002_entries())
    local model = model_sequences(handle, 4, 2)

    test.equal(model:reserve("a", 2), nil)
    local first, first_line = model:pass(
      {{sequence = "a", token = 2, position = 0}}
    )
    test.equal(first_line, nil)
    local wanted = (first as {{number}})[1]

    model:release("a")
    test.equal(model:reserve("a", 2), nil)
    local again, again_line = model:pass(
      {{sequence = "a", token = 2, position = 0}}
    )
    test.equal(again_line, nil)
    close((again as {{number}})[1], wanted, 1e-5)

    test.equal(model:reserve("b", 2), nil)
    local other, other_line = model:pass(
      {{sequence = "b", token = 2, position = 0}}
    )
    test.equal(other_line, nil)
    close((other as {{number}})[1], wanted, 1e-5)
  end
)
