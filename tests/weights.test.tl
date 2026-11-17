local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local gguf = require("tllm.gguf")
local weights = require("tllm.weights")

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/w" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
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

local function header(tensor_count: integer, metadata_count: integer): string
  return "GGUF" .. u32(3) .. u64(tensor_count) .. u64(metadata_count)
end

local function text(bytes: string): string
  return u64(#bytes) .. bytes
end

local function pair(name: string, code: integer, bytes: string): string
  return text(name) .. u32(code) .. bytes
end

local function tensor(
  name: string, dims: {integer}, type_code: integer, offset: integer
): string
  local parts: {string} = {}
  for _, dim in ipairs(dims) do
    parts[#parts + 1] = u64(dim)
  end
  return text(name)
    .. u32(#dims)
    .. table.concat(parts)
    .. u32(type_code)
    .. u64(offset)
end

local function padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

local function region(size: integer, fill: integer): string
  return string.rep(string.char(fill), size)
end

local function load_file(file_path: string): weights.Handle
  local handle, failure = weights.load(file_path)
  test.equal(failure, nil, file_path)
  return handle as weights.Handle
end

local function refusal(file_path: string): string
  local handle, failure = weights.load(file_path)
  test.equal(handle, nil, file_path)
  test.truthy(failure, file_path)
  local line = failure as string
  test.equal(line:find("\n"), nil, line)
  return line
end

local function refuse(bytes: string): string
  return refusal(fixture(bytes))
end

local function fd_count(): integer
  local count = 0
  local step = fs.readdir("/proc/self/fd")
  if step is nil then
    return count
  end
  while step() ~= nil do
    count = count + 1
  end
  return count
end

local function mapped(file_path: string): boolean
  local contents = fs.read("/proc/self/maps")
  if contents is nil then
    return false
  end
  return contents:find(file_path, 1, true) ~= nil
end

test.it(
  "tensors come back by name with their bytes",
  function()
    local table_bytes = header(3, 0)
      .. tensor("a.weight", {2, 3}, 0, 0)
      .. tensor("b.weight", {64, 2}, 2, 32)
      .. tensor("c.bias", {}, 0, 128)
    local data = region(24, 17)
      .. region(8, 0)
      .. region(72, 34)
      .. region(24, 0)
      .. region(4, 51)
    local file_path = fixture(padded(table_bytes, 32) .. data)

    local handle = load_file(file_path)
    test.equal(handle.path, file_path)

    local names: {string} = {}
    for name in pairs(handle.tensors) do
      names[#names + 1] = name
    end
    table.sort(names)
    test.deepEqual(names, {"a.weight", "b.weight", "c.bias"})

    local first = handle.tensors["a.weight"]
    test.deepEqual(first.dims, {2, 3})
    test.equal(first.kind.name, "f32")
    test.equal(first.elements, 6)
    test.equal(first.bytes, 24)
    test.equal(ffi.string(first.data, first.bytes), region(24, 17))

    local second = handle.tensors["b.weight"]
    test.deepEqual(second.dims, {64, 2})
    test.equal(second.kind.name, "q4_0")
    test.equal(second.elements, 128)
    test.equal(second.bytes, 72)
    test.equal(ffi.string(second.data, second.bytes), region(72, 34))

    local third = handle.tensors["c.bias"]
    test.deepEqual(third.dims, {})
    test.equal(third.kind.name, "f32")
    test.equal(third.elements, 1)
    test.equal(third.bytes, 4)
    test.equal(ffi.string(third.data, third.bytes), region(4, 51))

    handle.release(handle)
  end
)

test.it(
  "alignment and the data start come from the file",
  function()
    local aligned_table = header(1, 1)
      .. pair("general.alignment", 4, u32(64))
      .. tensor("a", {2, 3}, 0, 0)
    local aligned_path = fixture(aligned_table .. string.rep("\0", 88))
    local aligned_info = gguf.read(aligned_path) as gguf.Info
    local aligned = load_file(aligned_path)
    test.equal(aligned.alignment, 64)
    test.greaterEqual(aligned.data_start, aligned_info.table_end)
    test.equal(aligned.data_start % 64, 0)
    test.less(aligned.data_start, aligned_info.table_end + 64)
    aligned.release(aligned)

    local plain_table = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local plain_path = fixture(plain_table .. string.rep("\0", 56))
    local plain_info = gguf.read(plain_path) as gguf.Info
    local plain = load_file(plain_path)
    test.equal(plain.alignment, 32)
    test.greaterEqual(plain.data_start, plain_info.table_end)
    test.equal(plain.data_start % 32, 0)
    test.less(plain.data_start, plain_info.table_end + 32)
    plain.release(plain)
  end
)

test.it(
  "release unmaps the file",
  function()
    local table_bytes = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local file_path = fixture(table_bytes .. string.rep("\0", 56))

    local before = fd_count()
    local handle = load_file(file_path)
    test.truthy(mapped(file_path), file_path)
    test.lessEqual(fd_count(), before)

    handle.release(handle)
    test.equal(mapped(file_path), false)
    handle.release(handle)
    test.equal(mapped(file_path), false)
  end
)

test.it(
  "release waits for its holders",
  function()
    local table_bytes = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local file_path = fixture(table_bytes .. string.rep("\0", 56))

    local handle = load_file(file_path)
    weights.attach(handle)
    weights.attach(handle)
    test.includes(handle.release(handle) as string, "2")
    test.truthy(mapped(file_path), file_path)

    weights.detach(handle)
    weights.detach(handle)
    test.equal(handle.release(handle), nil)
    test.equal(mapped(file_path), false)
  end
)

test.it(
  "a file the reader refuses is refused with its line",
  function()
    local missing = ROOT .. "/missing.gguf"
    test.includes(refusal(missing), missing)

    local not_gguf = fixture("GGML" .. u32(3) .. u64(0) .. u64(0))
    test.includes(refusal(not_gguf), not_gguf)
  end
)

test.it(
  "an alignment that is not a positive integer is refused",
  function()
    local tensor_info = tensor("a", {2, 3}, 0, 0)
    local cases: {string} = {
      pair("general.alignment", 8, text("64")),
      pair("general.alignment", 4, u32(0)),
      pair("general.alignment", 12, string.char(0, 0, 0, 0, 0, 32, 80, 64)),
    }
    for _, alignment in ipairs(cases) do
      local line = refuse(header(1, 1) .. alignment .. tensor_info)
      test.includes(line, "general.alignment")
    end

    local line = refuse(
      header(1, 1)
        .. pair("general.alignment", 4, u32(0))
        .. tensor("broken", {1}, 4, 0)
    )
    test.includes(line, "general.alignment")
    test.equal(line:find("broken", 1, true), nil)
  end
)

test.it(
  "a tensor that cannot be read is refused",
  function()
    local unknown_line = refuse(
      header(1, 0) .. tensor("blk.unknown", {1}, 4, 0)
    )
    test.includes(unknown_line, "blk.unknown")
    test.includes(unknown_line, "type code 4")

    local block_line = refuse(header(1, 0) .. tensor("blk.block", {48}, 2, 0))
    test.includes(block_line, "blk.block")

    local short_table = header(1, 0) .. tensor("blk.short", {8}, 0, 0)
    local short_path = fixture(padded(short_table, 32) .. string.rep("\0", 31))
    test.includes(refusal(short_path), "blk.short")

    local duplicate_table = header(2, 0)
      .. tensor("blk.same", {8}, 0, 0)
      .. tensor("blk.same", {8}, 0, 32)
    local duplicate_line = refuse(
      padded(duplicate_table, 32) .. string.rep("\0", 80)
    )
    test.includes(duplicate_line, "blk.same")

    local order_line = refuse(
      header(2, 0)
        .. tensor("blk.first", {1}, 4, 0)
        .. tensor("blk.second", {48}, 2, 0)
    )
    test.includes(order_line, "blk.first")
    test.equal(order_line:find("blk.second", 1, true), nil)
  end
)
