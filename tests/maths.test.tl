local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local weights = require("tllm.weights")
local type Entry = support.Entry
local u16, u32, u64, text = support.u16, support.u32, support.u64, support.text
local padded, floats, refused = support.padded, support.floats, support.refused

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

-- 1, 2, 3 and 4 as f16
local HALVES: {integer} = {0x3C00, 0x4000, 0x4200, 0x4400}

local function halves(values: {integer}): string
  local parts: {string} = {}
  for _, value in ipairs(values) do
    parts[#parts + 1] = u16(HALVES[value])
  end
  return table.concat(parts)
end

local function quants(first: integer): string
  local parts: {string} = {}
  for value = first, first + 31 do
    parts[#parts + 1] = string.char(value)
  end
  return table.concat(parts)
end

-- scale 1, minimum 0, six-bit scale 1 and minimum 0 in every group
local K_SCALES = string.char(1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)

local function q8_0(first: integer, blocks: integer): string
  local parts: {string} = {}
  for block = 0, blocks - 1 do
    parts[#parts + 1] = u16(0x3C00) .. quants(first + block * 32)
  end
  return table.concat(parts)
end

local function q4_K(blocks: integer): string
  local parts: {string} = {}
  for _ = 1, blocks do
    parts[#parts + 1] = u16(0x3C00)
      .. u16(0)
      .. K_SCALES
      .. string.rep("\17", 128)
  end
  return table.concat(parts)
end

local function q5_K(blocks: integer): string
  local parts: {string} = {}
  for _ = 1, blocks do
    parts[#parts + 1] = u16(0x3C00)
      .. u16(0)
      .. K_SCALES
      .. string.rep("\0", 32)
      .. string.rep("\17", 128)
  end
  return table.concat(parts)
end

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
  local location = ROOT .. "/weights.gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function filled(value: number, count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = value
  end
  return result
end

local function range(first: integer, last: integer): {number}
  local result: {number} = {}
  for value = first, last do
    result[#result + 1] = value
  end
  return result
end

local location = fixture(
  {
    {
      name = "plain",
      dims = {3, 2},
      code = 0,
      bytes = floats({1, 2, 3, 4, 5, 6}),
    },
    {name = "bias", dims = {3}, code = 0, bytes = floats({2, 3, 4})},
    {
      name = "expert",
      dims = {2, 2, 2},
      code = 0,
      bytes = floats(
        {
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
        }
      ),
    },
    {
      name = "deep",
      dims = {2, 2, 2, 2},
      code = 0,
      bytes = floats(
        {
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
        }
      ),
    },
    {name = "scalar", dims = {}, code = 0, bytes = floats({7})},
    {name = "half", dims = {2, 2}, code = 1, bytes = halves({1, 2, 3, 4})},
    {name = "gain", dims = {2}, code = 0, bytes = floats({1, 2})},
    {name = "embd", dims = {32, 2}, code = 8, bytes = q8_0(1, 2)},
    {name = "quart", dims = {256, 2}, code = 12, bytes = q4_K(2)},
    {name = "five", dims = {256, 2}, code = 13, bytes = q5_K(2)},
  }
)

local handle, load_failure = weights.load(location)
test.equal(load_failure, nil)

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend
local maths = compute.attach(backend, handle as weights.Handle) as compute.Maths

test.it(
  "mapped tensors come back by name through ggml",
  function()
    local attached, failure = compute.attach(backend, handle as weights.Handle)
    test.equal(failure, nil)
    local mapped = attached as compute.Maths

    test.deepEqual(mapped:matvec("plain", {1, 1, 1}), {6, 15})
    test.deepEqual(mapped:matvec("bias", {1, 1, 1}), {9})
    test.deepEqual(mapped:matvec("scalar", {1}), {7})
    test.deepEqual(mapped:matvec("expert", {1, 1}, 0), {3, 7})
    test.deepEqual(mapped:matvec("expert", {1, 1}, 1), {11, 15})
    test.deepEqual(mapped:row("plain", 1), {4, 5, 6})
    test.deepEqual(mapped:row("embd", 1), range(33, 64))
    test.deepEqual(mapped:matvec("embd", filled(127, 32)), {67056, 197104})
    test.deepEqual(mapped:row("quart", 0), filled(1, 256))
    test.deepEqual(mapped:matvec("quart", filled(1, 256)), {256, 256})
    test.deepEqual(mapped:row("five", 0), filled(1, 256))
    test.deepEqual(mapped:matvec("five", filled(1, 256)), {256, 256})
  end
)

test.it(
  "norm and rope land on ggml's numbers",
  function()
    test.deepEqual(
      maths:norm({3, 4}, "gain", 0), {0.8485281467437744, 2.262741804122925}
    )
    test.deepEqual(
      maths:norm({3, 4}, "gain", 2.5), {0.7745966911315918, 2.0655910968780518}
    )
    test.deepEqual(
      maths:rope({1, 2, 3, 4, 5, 6}, 1, 4, 10000),
      {
        -1.9841105937957764,
        1.9599006175994873,
        2.4623777866363525,
        4.019799709320068,
        5,
        6,
      }
    )
  end
)

test.it(
  "softmax, add and the gated maths land on ggml's numbers",
  function()
    test.deepEqual(
      maths:softmax({1, 2, 3}),
      {
        0.09003057330846786,
        0.2447284758090973,
        0.6652409434318542,
      }
    )
    test.deepEqual(maths:softmax({1000, 1000}), {0.5, 0.5})
    test.deepEqual(maths:add({1, 2}, {3, 4}), {4, 6})
    test.deepEqual(maths:swiglu({0, 1}, {2, 3}), {0, 2.193175792694092})
    test.deepEqual(maths:sigmoid_gate({0, 1}, {2, 4}), {1, 2.924234390258789})
  end
)

test.it(
  "a name, type or shape that does not fit is refused with one line",
  function()
    for _, message in ipairs(
      {
        refused(maths:matvec("missing", {1})),
        refused(maths:matvec("missing", {1, 1}, -1)),
        refused(maths:row("missing", 0)),
      }
    ) do
      test.includes(message, "missing")
    end

    for _, message in ipairs(
      {
        refused(maths:matvec("half", {1, 1})),
        refused(maths:row("half", 0)),
        refused(maths:norm({1, 2}, "half", 0)),
      }
    ) do
      test.includes(message, "half")
      test.includes(message, "f16")
    end

    refused(maths:matvec("plain", {1, 1}))
    refused(maths:matvec("expert", {1, 1}))
    refused(maths:matvec("expert", {1, 1}, 2))
    refused(maths:matvec("expert", {1, 1}, -1))
    refused(maths:matvec("plain", {1, 1}, 0))
    refused(maths:matvec("deep", {1, 1}))
    refused(maths:row("bias", 0))
    refused(maths:row("plain", 2))
    refused(maths:row("plain", -1))
    refused(maths:norm({1, 2, 3}, "gain", 0))
    refused(maths:norm({1, 2}, "plain", 0))
    refused(maths:norm({}, "gain", 0))
    refused(maths:norm({1, 2}, "gain", -1))
    refused(maths:softmax({}))
    refused(maths:swiglu({1}, {1, 2}))
    refused(maths:sigmoid_gate({1}, {1, 2}))
    refused(maths:rope({1, 2}, 0, 3, 10000))
    refused(maths:rope({1, 2}, 0, 4, 10000))
    refused(maths:rope({1, 2, 3, 4}, -1, 4, 10000))
    refused(maths:rope({1, 2, 3, 4}, 0, 0, 10000))
    refused(maths:rope({1, 2, 3, 4}, 0, 4, 0))

    local mismatch = refused(maths:add({1}, {1, 2}))
    test.includes(mismatch, "add")
    test.includes(mismatch, "1")
    test.includes(mismatch, "2")
  end
)

test.it(
  "a backend compute.load did not return is refused",
  function()
    local forged = {
      path = "forged",
      version = "forged",
      matvec = function(
        _: compute.Backend, _matrix: {{number}}, _vector: {number}
      ): {number} | nil, string | nil
        return nil, "forged"
      end,
    } as compute.Backend
    local attached, failure = compute.attach(forged, handle as weights.Handle)
    test.equal(attached, nil)
    test.includes(failure as string, "attach")
  end
)
