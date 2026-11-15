local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
local weights = require("tllm.weights")

local function first_row(hidden: {number}, position: integer): {compute.Row}
  return {{sequence = "", hidden = hidden, position = position}}
end

local function first(
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

local function u16(value: integer): string
  return string.char(value % 256, math.floor(value / 256) % 256)
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

local function halves(count: integer): string
  return string.rep(u16(0x3C00), count)
end

local record Entry
  name: string
  dims: {integer}
  code: integer
  bytes: string
end

local record Broken
  make: function(): compute.Geometry
  marker: string
end

local function padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

local written = 0

local function fixture(entries: {Entry}): string
  local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(0)
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
  local location = ROOT .. "/linear_attention" .. written .. ".gguf"
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

local function f32(name: string, dims: {integer}, values: {number}): Entry
  return {name = name, dims = dims, code = 0, bytes = floats(values)}
end

local LN3 = math.log(3)

local function repeated(values: {number}, times: integer): {number}
  local result: {number} = {}
  for _ = 1, times do
    for _, value in ipairs(values) do
      result[#result + 1] = value
    end
  end
  return result
end

local function toy(prefix: string): {Entry}
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
    f32(prefix .. "ssm_out.weight", {4, 2}, {1, 0, 1, 0, 0, 1, 0, 1}),
  }
end

local function toy_entries(): {Entry}
  local entries = toy("blk.0.")
  for _, entry in ipairs(toy("blk.1.")) do
    entries[#entries + 1] = entry
  end
  return entries
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

local function loaded(entries: {Entry}): weights.Handle
  local location = fixture(entries)
  local handle, failure = weights.load(location)
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function geometry(): compute.Geometry
  return {
    heads = 1,
    kv_heads = 1,
    head_length = 2,
    rope_dimensions = 2,
    rope_base = 10000,
    eps = 0,
    positions = 4,
    experts_used = 1,
    linear_key_heads = 2,
    linear_key_length = 2,
    linear_value_heads = 2,
    linear_value_length = 2,
    linear_conv_kernel = 2,
    cache_type = "f32",
  }
end

local function copied(base: compute.Geometry): compute.Geometry
  return {
    heads = base.heads,
    kv_heads = base.kv_heads,
    head_length = base.head_length,
    rope_dimensions = base.rope_dimensions,
    rope_base = base.rope_base,
    eps = base.eps,
    positions = base.positions,
    experts_used = base.experts_used,
    linear_key_heads = base.linear_key_heads,
    linear_key_length = base.linear_key_length,
    linear_value_heads = base.linear_value_heads,
    linear_value_length = base.linear_value_length,
    linear_conv_kernel = base.linear_conv_kernel,
    cache_type = base.cache_type,
  }
end

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function linear(
  handle: weights.Handle, shape: compute.Geometry
): compute.LinearAttention
  local attention, failure = compute.linear_attention(backend, handle, shape)
  test.equal(failure, nil)
  return attention as compute.LinearAttention
end

local function refused(value: {number} | nil, line: string | nil): string
  test.equal(value, nil)
  test.truthy(line)
  local message = line as string
  test.equal(message:find("\n"), nil)
  return message
end

local function close(got: {number} | nil, wanted: {number})
  local values = got as {number}
  test.equal(#values, #wanted)
  for index, number in ipairs(wanted) do
    local slack = 1e-5 * math.max(1, math.abs(number))
    test.less(math.abs(values[index] - number), slack)
  end
end

test.it(
  "one linear block at position zero gives its output",
  function()
    local handle = loaded(toy_entries())
    local attention = linear(handle, geometry())
    close(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.0.")),
      {100, 241.4213562}
    )

    local other = linear(loaded(toy_entries()), geometry())
    close(
      first(other.block(other, first_row({3, 3}, 0), "blk.0.")),
      {100, 241.4213562}
    )

    local shape = geometry()
    shape.eps = 1
    local strict = linear(loaded(toy_entries()), shape)
    close(
      first(strict.block(strict, first_row({1, 1}, 0), "blk.0.")),
      {70.6498381, 170.6241588}
    )
  end
)

test.it(
  "a second token reads the conv history and the state",
  function()
    local attention = linear(loaded(toy_entries()), geometry())
    close(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.0.")),
      {100, 241.4213562}
    )
    close(
      first(attention.block(attention, first_row({1, 0}, 1), "blk.0.")),
      {193.2903421, 251.3696764}
    )
    close(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.1.")),
      {100, 241.4213562}
    )
  end
)

test.it(
  "a three-tap convolution reads its history in order",
  function()
    local entries = replaced(
      toy_entries(),
      "blk.0.ssm_conv1d.weight",
      f32("blk.0.ssm_conv1d.weight", {3, 12}, repeated({7, 11, 5}, 12))
    )
    local shape = geometry()
    shape.linear_conv_kernel = 3
    local attention = linear(loaded(entries), shape)
    close(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.0.")),
      {100, 241.4213562}
    )
    close(
      first(attention.block(attention, first_row({1, 0}, 1), "blk.0.")),
      {170.7414208, 304.1506948}
    )
  end
)

test.it(
  "a name, type, shape or position that does not fit is refused with one line",
  function()
    local handle = loaded(toy_entries())
    local shape = geometry()

    local forged = {
      path = "forged",
      version = "forged",
      matvec = function(
        _: compute.Backend, _matrix: {{number}}, _vector: {number}
      ): {number} | nil, string | nil
        return nil, "forged"
      end,
    } as compute.Backend
    local built, forged_line = compute.linear_attention(forged, handle, shape)
    test.equal(built, nil)
    test.includes(forged_line as string, "linear_attention")
    test.equal((forged_line as string):find("\n"), nil)

    local fieldless = copied(shape)
    fieldless.linear_conv_kernel = nil
    local fieldless_shape, fieldless_line = compute.linear_attention(
      backend, handle, fieldless
    )
    test.equal(fieldless_shape, nil)
    test.includes(fieldless_line as string, "linear_attention")
    test.includes(fieldless_line as string, "linear_conv_kernel")
    test.equal((fieldless_line as string):find("\n"), nil)

    local broken: {Broken} = {
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_key_heads = 0
          return taken
        end,
        marker = "0",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_key_length = 0
          return taken
        end,
        marker = "0",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_value_heads = 0
          return taken
        end,
        marker = "0",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_value_length = 0
          return taken
        end,
        marker = "0",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_conv_kernel = 0
          return taken
        end,
        marker = "0",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.linear_value_heads = 3
          return taken
        end,
        marker = "linear_value_heads",
      },
      {
        make = function(): compute.Geometry
          local taken = copied(shape)
          taken.eps = -1
          return taken
        end,
        marker = "-1",
      },
    }
    for _, broken_geometry in ipairs(broken) do
      local attention, line = compute.linear_attention(
        backend, handle, broken_geometry.make()
      )
      test.equal(attention, nil)
      test.includes(line as string, "linear_attention")
      test.includes(line as string, broken_geometry.marker)
    end

    local attention = linear(handle, shape)
    local line = refused(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.9."))
    )
    test.includes(line, "blk.9.attn_norm.weight")

    for _, suffix in ipairs(
      {
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
    ) do
      local missing = linear(
        loaded(without(toy_entries(), "blk.0." .. suffix)), shape
      )
      local missing_line = refused(
        first(missing.block(missing, first_row({1, 1}, 0), "blk.0."))
      )
      test.includes(missing_line, suffix)
    end

    local halves_cases = {
      {"blk.0.attn_qkv.weight", {2, 12}, 24},
      {"blk.0.ssm_beta.weight", {2, 2}, 4},
      {"blk.0.ssm_out.weight", {4, 2}, 8},
    }
    for _, half_case in ipairs(halves_cases) do
      local name = half_case[1] as string
      local half = replaced(
        toy_entries(),
        name,
        {
          name = name,
          dims = half_case[2] as {integer},
          code = 1,
          bytes = halves(half_case[3] as integer),
        }
      )
      local half_attention = linear(loaded(half), shape)
      local half_line = refused(
        first(
          half_attention.block(half_attention, first_row({1, 1}, 0), "blk.0.")
        )
      )
      test.includes(half_line, name)
      test.includes(half_line, "f16")
    end

    local shapes = {
      {{1}, "attn_norm.weight"},
      {{1, 1, 1}, "attn_norm.weight"},
    }
    for _, shape_case in ipairs(shapes) do
      local case_line = refused(
        first(
          attention.block(
            attention, first_row(shape_case[1] as {number}, 0), "blk.0."
          )
        )
      )
      test.includes(case_line, shape_case[2] as string)
    end

    local variants = {
      {"blk.0.attn_qkv.weight", {2, 11}, 22},
      {"blk.0.attn_gate.weight", {2, 3}, 6},
      {"blk.0.ssm_beta.weight", {2, 1}, 2},
      {"blk.0.ssm_alpha.weight", {2, 3}, 6},
      {"blk.0.ssm_a", {1}, 1},
      {"blk.0.ssm_dt.bias", {3}, 3},
      {"blk.0.ssm_norm.weight", {3}, 3},
      {"blk.0.ssm_conv1d.weight", {2, 11}, 22},
      {"blk.0.ssm_out.weight", {3, 2}, 6},
      {"blk.0.attn_qkv.weight", {24}, 24},
      {"blk.0.ssm_a", {2, 1, 1}, 2},
      {"blk.0.ssm_conv1d.weight", {3, 12}, 36},
      {"blk.0.ssm_out.weight", {4, 1}, 4},
    }
    for _, changed in ipairs(variants) do
      local name = changed[1] as string
      local changed_attention = linear(
        loaded(
          replaced(
            toy_entries(),
            name,
            {
              name = name,
              dims = changed[2] as {integer},
              code = 0,
              bytes = floats(zeros(changed[3] as integer)),
            }
          )
        ),
        shape
      )
      local changed_line = refused(
        first(
          changed_attention.block(
            changed_attention, first_row({1, 1}, 0), "blk.0."
          )
        )
      )
      test.includes(changed_line, name)
    end

    local position_line = refused(
      first(attention.block(attention, first_row({1, 1}, -1), "blk.0."))
    )
    test.includes(position_line, "blk.0.")
    test.includes(position_line, "-1")
    local fresh = linear(loaded(toy_entries()), shape)
    local ahead = refused(
      first(fresh.block(fresh, first_row({1, 1}, 2), "blk.0."))
    )
    test.includes(ahead, "blk.0.")
    test.includes(ahead, "2")

    local absent = replaced(
      without(toy_entries(), "blk.0.ssm_out.weight"),
      "blk.0.attn_qkv.weight",
      {
        name = "blk.0.attn_qkv.weight",
        dims = {2, 12},
        code = 1,
        bytes = halves(24),
      }
    )
    local absent_attention = linear(loaded(absent), shape)
    local absent_line = refused(
      first(
        absent_attention.block(absent_attention, first_row({1, 1}, 0), "blk.0.")
      )
    )
    test.includes(absent_line, "ssm_out.weight")

    local mistyped = replaced(
      toy_entries(),
      "blk.0.attn_qkv.weight",
      {
        name = "blk.0.attn_qkv.weight",
        dims = {2, 12},
        code = 1,
        bytes = halves(24),
      }
    )
    mistyped = replaced(
      mistyped,
      "blk.0.attn_gate.weight",
      {
        name = "blk.0.attn_gate.weight",
        dims = {2, 3},
        code = 0,
        bytes = floats(zeros(6)),
      }
    )
    local mistyped_attention = linear(loaded(mistyped), shape)
    local mistyped_line = refused(
      first(
        mistyped_attention.block(
          mistyped_attention, first_row({1, 1}, 0), "blk.0."
        )
      )
    )
    test.includes(mistyped_line, "attn_qkv")
    test.includes(mistyped_line, "f16")

    local wrong_rows = replaced(
      toy_entries(),
      "blk.0.attn_qkv.weight",
      {
        name = "blk.0.attn_qkv.weight",
        dims = {2, 11},
        code = 0,
        bytes = floats(zeros(22)),
      }
    )
    wrong_rows = replaced(
      wrong_rows,
      "blk.0.attn_gate.weight",
      {
        name = "blk.0.attn_gate.weight",
        dims = {2, 3},
        code = 0,
        bytes = floats(zeros(6)),
      }
    )
    local wrong_attention = linear(loaded(wrong_rows), shape)
    local wrong_line = refused(
      first(
        wrong_attention.block(wrong_attention, first_row({1, 1}, 0), "blk.0.")
      )
    )
    test.includes(wrong_line, "attn_qkv.weight")

    local late = replaced(
      toy_entries(),
      "blk.0.attn_gate.weight",
      {
        name = "blk.0.attn_gate.weight",
        dims = {2, 3},
        code = 0,
        bytes = floats(zeros(6)),
      }
    )
    local late_attention = linear(loaded(late), shape)
    local late_line = refused(
      first(
        late_attention.block(late_attention, first_row({1, 1}, 2), "blk.0.")
      )
    )
    test.includes(late_line, "attn_gate.weight")

    local kept = linear(loaded(toy_entries()), shape)
    refused(first(kept.block(kept, first_row({1, 1}, 1), "blk.0.")))
    close(
      first(kept.block(kept, first_row({1, 1}, 0), "blk.0.")),
      {100, 241.4213562}
    )
    refused(first(kept.block(kept, first_row({1, 1}, 0), "blk.0.")))
    close(
      first(kept.block(kept, first_row({1, 0}, 1), "blk.0.")),
      {193.2903421, 251.3696764}
    )
  end
)

test.it(
  "each value head reads the key head it belongs to",
  function()
    local entries = {
      f32("blk.0.attn_norm.weight", {2}, {1, 1}),
      f32(
        "blk.0.attn_qkv.weight",
        {2, 16},
        {
          1,
          0,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          0,
          0,
          1,
          0,
          1,
          1,
          0,
          100,
          0,
          0,
          0,
          0,
          0,
          0,
          100,
          100,
          0,
          0,
          100,
          100,
          0,
          0,
          50,
        }
      ),
      f32("blk.0.attn_gate.weight", {2, 8}, repeated({100, 0}, 8)),
      f32("blk.0.ssm_beta.weight", {2, 4}, repeated({0, 0}, 4)),
      f32("blk.0.ssm_alpha.weight", {2, 4}, repeated({0, 0}, 4)),
      f32("blk.0.ssm_dt.bias", {4}, {0, 0, 0, 0}),
      f32("blk.0.ssm_a", {4}, {-1, -1, -1, -1}),
      f32("blk.0.ssm_conv1d.weight", {1, 16}, repeated({1}, 16)),
      f32("blk.0.ssm_norm.weight", {2}, {1, 1}),
      f32(
        "blk.0.ssm_out.weight",
        {8, 2},
        {
          1,
          0,
          1,
          0,
          1,
          0,
          1,
          0,
          0,
          1,
          0,
          1,
          0,
          1,
          0,
          1,
        }
      ),
    }

    local shape = geometry()
    shape.linear_value_heads = 4
    shape.linear_conv_kernel = 1
    local attention = linear(loaded(entries), shape)
    close(
      first(attention.block(attention, first_row({1, 1}, 0), "blk.0.")),
      {367.9124626, 304.6669094}
    )
    local scan = first(
      attention.block(attention, first_row({1, 2}, 1), "blk.0.")
    ) as {number}
    close(scan, {209.3171647, 219.4780173})
    test.truthy(
      math.abs(scan[1] - 206.6115929) > 1e-4
        and math.abs(scan[2] - 221.2354228) > 1e-4
    )
  end
)
