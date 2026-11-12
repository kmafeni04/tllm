local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
local weights = require("tllm.weights")

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
  local location = ROOT .. "/attention" .. written .. ".gguf"
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

local function toy(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {4}, ones(4)),
    f32(
      prefix .. "attn_q.weight",
      {4, 8},
      {
        1,
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
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1e9,
        0,
        0,
        0,
        -1e9,
        0,
        0,
        0,
      }
    ),
    f32(prefix .. "attn_k.weight", {4, 2}, {1, 0, 0, 0, 0, 1, 0, 0}),
    f32(prefix .. "attn_v.weight", {4, 2}, {2, 0, 0, 0, 0, 4, 0, 0}),
    f32(prefix .. "attn_q_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_k_norm.weight", {2}, ones(2)),
    f32(
      prefix .. "attn_output.weight",
      {4, 4},
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
    ),
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
    heads = 2,
    kv_heads = 1,
    head_length = 2,
    rope_dimensions = 2,
    rope_base = 10000,
    eps = 0,
    positions = 4,
    experts_used = 2,
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
  }
end

local backend = compute.load() as compute.Backend

local function attended(
  handle: weights.Handle, shape: compute.Geometry
): compute.Attention
  local attention, failure = compute.attend(backend, handle, shape)
  test.equal(failure, nil)
  return attention as compute.Attention
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
    test.less(math.abs(values[index] - number), 1e-5)
  end
end

test.it(
  "one block at position zero gives its output",
  function()
    local handle = loaded(toy_entries())
    local shape = geometry()
    local attention, failure = compute.attend(backend, handle, shape)
    test.equal(failure, nil)
    local block = (attention as compute.Attention).block

    test.deepEqual(
      block(attention as compute.Attention, {1, 1, 1, 1}, "blk.0.", 0),
      {1, 2, 0, 0}
    )

    local other = attended(loaded(toy_entries()), shape)
    test.deepEqual(other.block(other, {3, 3, 3, 3}, "blk.0.", 0), {1, 2, 0, 0})
  end
)

test.it(
  "a second token attends over the first",
  function()
    local attention = attended(loaded(toy_entries()), geometry())
    local block = attention.block
    test.deepEqual(block(attention, {1, 1, 1, 1}, "blk.0.", 0), {1, 2, 0, 0})
    close(
      block(attention, {1, 0, 0, 0}, "blk.0.", 1),
      {
        1.5587023,
        0.8825954,
        0,
        0,
      }
    )

    close(block(attention, {1, 0, 0, 0}, "blk.1.", 0), {2, 0, 0, 0})

    refused(block(attention, {1, 0, 0, 0}, "blk.0.", 3))

    local shape = geometry()
    shape.eps = 1
    local strict = attended(loaded(toy_entries()), shape)
    close(
      strict.block(strict, {1, 1, 1, 1}, "blk.0.", 0),
      {
        0.7071068,
        1.4142136,
        0,
        0,
      }
    )
    close(
      strict.block(strict, {1, 0, 0, 0}, "blk.0.", 1),
      {
        0.8036810,
        0.6851070,
        0,
        0,
      }
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
    local built, forged_line = compute.attend(forged, handle, shape)
    test.equal(built, nil)
    test.includes(forged_line as string, "attend")

    local broken: {function(): compute.Geometry} = {
      function(): compute.Geometry
        local taken = copied(shape)
        taken.heads = 0
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.kv_heads = 0
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.kv_heads = 3
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.head_length = 0
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.rope_dimensions = 0
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.rope_dimensions = 3
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.rope_dimensions = 4
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.rope_dimensions = 5
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.rope_base = 0
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.eps = -1
        return taken
      end,
      function(): compute.Geometry
        local taken = copied(shape)
        taken.positions = 0
        return taken
      end,
    }
    for _, broken_geometry in ipairs(broken) do
      local attention, line = compute.attend(backend, handle, broken_geometry())
      test.equal(attention, nil)
      test.includes(line as string, "attend")
    end

    local attention = attended(handle, shape)
    local line = refused(attention.block(attention, {1, 1, 1, 1}, "blk.9.", 0))
    test.includes(line, "blk.9.attn_norm.weight")

    for _, suffix in ipairs(
      {
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_output.weight",
      }
    ) do
      local missing = attended(
        loaded(without(toy_entries(), "blk.0." .. suffix)), shape
      )
      local missing_line = refused(
        missing.block(missing, {1, 1, 1, 1}, "blk.0.", 0)
      )
      test.includes(missing_line, suffix)
    end

    local half = replaced(
      toy_entries(),
      "blk.0.attn_k.weight",
      {name = "blk.0.attn_k.weight", dims = {4, 2}, code = 1, bytes = halves(8)}
    )
    local half_attention = attended(loaded(half), shape)
    local half_line = refused(
      half_attention.block(half_attention, {1, 1, 1, 1}, "blk.0.", 0)
    )
    test.includes(half_line, "attn_k")
    test.includes(half_line, "f16")

    local shapes = {
      {{1, 1}, "attn_norm.weight"},
      {{1, 1, 1}, "attn_norm.weight"},
    }
    for _, shape_case in ipairs(shapes) do
      local case_line = refused(
        attention.block(attention, shape_case[1] as {number}, "blk.0.", 0)
      )
      test.includes(case_line, shape_case[2] as string)
    end

    local variants = {
      {
        "blk.0.attn_q.weight",
        {
          name = "blk.0.attn_q.weight",
          dims = {4, 7},
          code = 0,
          bytes = floats(zeros(28)),
        },
      },
      {
        "blk.0.attn_k.weight",
        {
          name = "blk.0.attn_k.weight",
          dims = {4, 3},
          code = 0,
          bytes = floats(zeros(12)),
        },
      },
      {
        "blk.0.attn_q_norm.weight",
        {
          name = "blk.0.attn_q_norm.weight",
          dims = {3},
          code = 0,
          bytes = floats(ones(3)),
        },
      },
      {
        "blk.0.attn_output.weight",
        {
          name = "blk.0.attn_output.weight",
          dims = {4, 3},
          code = 0,
          bytes = floats(zeros(12)),
        },
      },
      {
        "blk.0.attn_q.weight",
        {
          name = "blk.0.attn_q.weight",
          dims = {4, 8, 1},
          code = 0,
          bytes = floats(zeros(32)),
        },
      },
      {
        "blk.0.attn_norm.weight",
        {
          name = "blk.0.attn_norm.weight",
          dims = {4, 1},
          code = 0,
          bytes = floats(zeros(4)),
        },
      },
    }
    for _, changed in ipairs(variants) do
      local changed_attention = attended(
        loaded(
          replaced(toy_entries(), changed[1] as string, changed[2] as Entry)
        ),
        shape
      )
      local changed_line = refused(
        changed_attention.block(changed_attention, {1, 1, 1, 1}, "blk.0.", 0)
      )
      test.includes(changed_line, changed[1] as string)
    end

    for _, position in ipairs({-1, 4}) do
      local position_line = refused(
        attention.block(attention, {1, 1, 1, 1}, "blk.0.", position)
      )
      test.includes(position_line, "blk.0.")
      test.includes(position_line, tostring(position))
    end

    local absent = without(toy_entries(), "blk.0.attn_k.weight")
    absent = replaced(
      absent,
      "blk.0.attn_q.weight",
      {
        name = "blk.0.attn_q.weight",
        dims = {4, 8},
        code = 1,
        bytes = halves(32),
      }
    )
    local absent_attention = attended(loaded(absent), shape)
    local absent_line = refused(
      absent_attention.block(absent_attention, {1, 1, 1, 1}, "blk.0.", 0)
    )
    test.includes(absent_line, "attn_k.weight")

    local mistyped = replaced(
      toy_entries(),
      "blk.0.attn_k.weight",
      {name = "blk.0.attn_k.weight", dims = {4, 2}, code = 1, bytes = halves(8)}
    )
    mistyped = replaced(
      mistyped,
      "blk.0.attn_q.weight",
      {
        name = "blk.0.attn_q.weight",
        dims = {4, 7},
        code = 0,
        bytes = floats(zeros(28)),
      }
    )
    local mistyped_attention = attended(loaded(mistyped), shape)
    local mistyped_line = refused(
      mistyped_attention.block(mistyped_attention, {1, 1, 1, 1}, "blk.0.", 0)
    )
    test.includes(mistyped_line, "attn_k")
    test.includes(mistyped_line, "f16")

    local wrong_rows = replaced(
      toy_entries(),
      "blk.0.attn_q.weight",
      {
        name = "blk.0.attn_q.weight",
        dims = {4, 7},
        code = 0,
        bytes = floats(zeros(28)),
      }
    )
    local wrong_attention = attended(loaded(wrong_rows), shape)
    local wrong_line = refused(
      wrong_attention.block(wrong_attention, {1, 1, 1, 1}, "blk.0.", 5)
    )
    test.includes(wrong_line, "attn_q.weight")

    local fresh = attended(loaded(toy_entries()), shape)
    refused(fresh.block(fresh, {1, 1}, "blk.0.", 0))
    refused(fresh.block(fresh, {1, 1, 1, 1}, "blk.0.", 1))
    test.deepEqual(fresh.block(fresh, {1, 1, 1, 1}, "blk.0.", 0), {1, 2, 0, 0})
  end
)

test.it(
  "each query head reads the key and value head it belongs to",
  function()
    local entries = {
      f32("blk.0.attn_norm.weight", {4}, ones(4)),
      f32(
        "blk.0.attn_q.weight",
        {4, 16},
        {
          1,
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
          0,
          0,
          0,
          1,
          0,
          0,
          0,
          -1,
          0,
          0,
          0,
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
          1,
          0,
          0,
          0,
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
          -1,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
        }
      ),
      f32(
        "blk.0.attn_k.weight",
        {4, 4},
        {1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}
      ),
      f32(
        "blk.0.attn_v.weight",
        {4, 4},
        {2, 0, 0, 0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 8, 0, 0}
      ),
      f32("blk.0.attn_q_norm.weight", {2}, ones(2)),
      f32("blk.0.attn_k_norm.weight", {2}, ones(2)),
      f32(
        "blk.0.attn_output.weight",
        {8, 4},
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
        }
      ),
    }

    local shape = copied(geometry())
    shape.heads = 4
    shape.kv_heads = 2
    local attention = attended(loaded(entries), shape)
    close(attention.block(attention, {1, 1, 1, 1}, "blk.0.", 0), {1, 2, 3, 4})
  end
)
