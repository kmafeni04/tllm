local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local gguf = require("tllm.gguf")
local u16, u32, u64, text = support.u16, support.u32, support.u64, support.text
local array, header, pair = support.array, support.header, support.pair

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/f" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
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

local function read_file(bytes: string): gguf.Info
  local file_path = fixture(bytes)
  local info, failure = gguf.read(file_path)
  test.equal(failure, nil, file_path)
  return info as gguf.Info
end

local function refusal(file_path: string): string
  local info, failure = gguf.read(file_path)
  test.equal(info, nil, file_path)
  test.truthy(failure, file_path)
  local line = failure as string
  test.equal(line:find("\n"), nil, line)
  return line
end

local function refuse(bytes: string): string, string
  local file_path = fixture(bytes)
  return file_path, refusal(file_path)
end

local record NumericCase
  name: string
  code: integer
  kind: string
  bytes: string
  scalar: number
end

local record TypeCase
  name: string
  code: integer
  bytes: string
end

test.it(
  "header counts and keys are read",
  function()
    local info = read_file(
      header(2, 2)
        .. pair("general.architecture", 8, text("llama"))
        .. pair("llama.block_count", 4, u32(32))
        .. tensor("blk.0", {2, 3}, 0, 0)
        .. tensor("output.weight", {7}, 1, 65536)
    )

    test.equal(info.version, 3)
    test.equal(info.tensor_count, 2)
    test.equal(info.metadata_count, 2)

    local keys: {string} = {}
    for key in pairs(info.metadata) do
      keys[#keys + 1] = key
    end
    table.sort(keys)
    test.deepEqual(keys, {"general.architecture", "llama.block_count"})

    local architecture = info.metadata["general.architecture"] as gguf.Value
    test.equal(architecture.kind, "string")
    test.equal(architecture.text, "llama")

    local blocks = info.metadata["llama.block_count"] as gguf.Value
    test.equal(blocks.kind, "uint32")
    test.equal(blocks.scalar, 32)
  end
)

test.it(
  "numeric metadata keeps its type and value",
  function()
    local cases: {NumericCase} = {
      {
        name = "a",
        code = 0,
        kind = "uint8",
        bytes = string.char(200),
        scalar = 200,
      },
      {
        name = "b",
        code = 1,
        kind = "int8",
        bytes = string.char(255),
        scalar = -1,
      },
      {
        name = "c",
        code = 2,
        kind = "uint16",
        bytes = u16(40000),
        scalar = 40000,
      },
      {name = "d", code = 3, kind = "int16", bytes = u16(-2), scalar = -2},
      {
        name = "e",
        code = 4,
        kind = "uint32",
        bytes = u32(4294967295),
        scalar = 4294967295,
      },
      {name = "f", code = 5, kind = "int32", bytes = u32(-3), scalar = -3},
      {
        name = "g",
        code = 6,
        kind = "float32",
        bytes = string.char(0, 0, 192, 63),
        scalar = 1.5,
      },
      {
        name = "h",
        code = 10,
        kind = "uint64",
        bytes = u64(4294967296),
        scalar = 4294967296,
      },
      {name = "i", code = 11, kind = "int64", bytes = u64(-1), scalar = -1},
      {
        name = "j",
        code = 12,
        kind = "float64",
        bytes = string.char(0, 0, 0, 0, 0, 0, 2, 192),
        scalar = -2.25,
      },
    }

    local bytes = header(0, #cases)
    for _, case in ipairs(cases) do
      bytes = bytes .. pair(case.name, case.code, case.bytes)
    end

    local info = read_file(bytes)
    for _, case in ipairs(cases) do
      local entry = info.metadata[case.name] as gguf.Value
      test.equal(entry.kind, case.kind, case.name)
      test.equal(entry.scalar, case.scalar, case.name)
      test.equal(entry.text, nil, case.name)
      test.equal(entry.flag, nil, case.name)
      test.equal(entry.item_kind, nil, case.name)
      test.equal(entry.items, nil, case.name)
    end
  end
)

test.it(
  "string and bool metadata keep their type and value",
  function()
    local info = read_file(
      header(0, 3)
        .. pair("zeroed", 8, text("a\0b"))
        .. pair("on", 7, string.char(1))
        .. pair("off", 7, string.char(0))
    )

    local zeroed = info.metadata["zeroed"] as gguf.Value
    test.equal(zeroed.kind, "string")
    test.equal(zeroed.text, "a\0b")
    test.equal(zeroed.scalar, nil)
    test.equal(zeroed.flag, nil)
    test.equal(zeroed.item_kind, nil)
    test.equal(zeroed.items, nil)

    local on = info.metadata["on"] as gguf.Value
    test.equal(on.kind, "bool")
    test.equal(on.flag, true)
    test.equal(on.scalar, nil)
    test.equal(on.text, nil)
    test.equal(on.item_kind, nil)
    test.equal(on.items, nil)

    local off = info.metadata["off"] as gguf.Value
    test.equal(off.kind, "bool")
    test.equal(off.flag, false)
    test.equal(off.scalar, nil)
    test.equal(off.text, nil)
    test.equal(off.item_kind, nil)
    test.equal(off.items, nil)
  end
)

test.it(
  "arrays keep their element type and order",
  function()
    local info = read_file(
      header(0, 4)
        .. pair("strings", 9, array(8, {text("x"), text("yz")}))
        .. pair("ints", 9, array(5, {u32(1), u32(-2), u32(3)}))
        .. pair("empty", 9, array(6, {}))
        .. pair("nested", 9, array(9, {array(5, {u32(7), u32(8)})}))
    )

    local strings = info.metadata["strings"] as gguf.Value
    test.equal(strings.item_kind, "string")
    local string_items = strings.items as {gguf.Value}
    test.equal(#string_items, 2)
    test.equal(string_items[1].text, "x")
    test.equal(string_items[2].text, "yz")

    local ints = info.metadata["ints"] as gguf.Value
    test.equal(ints.item_kind, "int32")
    local int_items = ints.items as {gguf.Value}
    test.equal(#int_items, 3)
    test.equal(int_items[1].scalar, 1)
    test.equal(int_items[2].scalar, -2)
    test.equal(int_items[3].scalar, 3)

    local empty = info.metadata["empty"] as gguf.Value
    test.equal(empty.item_kind, "float32")
    test.equal(#(empty.items as {gguf.Value}), 0)

    local nested = info.metadata["nested"] as gguf.Value
    test.equal(nested.item_kind, "array")
    local nested_items = nested.items as {gguf.Value}
    test.equal(#nested_items, 1)
    test.equal(nested_items[1].kind, "array")
    test.equal(nested_items[1].item_kind, "int32")
    local inner = nested_items[1].items as {gguf.Value}
    test.equal(#inner, 2)
    test.equal(inner[1].scalar, 7)
    test.equal(inner[2].scalar, 8)
  end
)

test.it(
  "the tensor table is read without the tensor data",
  function()
    local info = read_file(
      header(2, 0)
        .. tensor("a.weight", {2, 3}, 0, 0)
        .. tensor("b.weight", {7}, 2, 65536)
    )

    test.equal(#info.tensors, 2)

    local first = info.tensors[1]
    test.equal(first.name, "a.weight")
    test.deepEqual(first.dims, {2, 3})
    test.equal(first.type_code, 0)
    test.equal(first.offset, 0)

    local second = info.tensors[2]
    test.equal(second.name, "b.weight")
    test.deepEqual(second.dims, {7})
    test.equal(second.type_code, 2)
    test.equal(second.offset, 65536)

    test.equal(info.table_end, info.bytes)
  end
)

test.it(
  "size and parameter count are computed",
  function()
    local product = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    test.equal(read_file(product).parameters, 6)

    local flat = header(1, 0) .. tensor("b", {7}, 0, 0)
    test.equal(read_file(flat).parameters, 7)

    local none = header(1, 0) .. tensor("c", {}, 0, 0)
    test.equal(read_file(none).parameters, 1)

    local table_bytes = header(3, 0)
      .. tensor("a", {2, 3}, 0, 0)
      .. tensor("b", {7}, 0, 0)
      .. tensor("c", {}, 0, 0)
    local table_info = read_file(table_bytes)
    test.equal(table_info.parameters, 14)
    test.equal(table_info.bytes, #table_bytes)

    local with_data = table_bytes .. string.rep("\0", 16)
    local data_info = read_file(with_data)
    test.equal(data_info.parameters, 14)
    test.equal(data_info.bytes, #with_data)
  end
)

test.it(
  "a path that is not a readable file is refused",
  function()
    local missing = ROOT .. "/missing.gguf"
    test.includes(refusal(missing), missing)

    local directory = ROOT .. "/directory"
    fs.mkdirAll(directory)
    test.includes(refusal(directory), directory)

    local file_path, line = refuse("")
    test.includes(line, file_path)
  end
)

test.it(
  "a file that is not GGUF is refused",
  function()
    local files = {"GGU", "GGML" .. u32(3) .. u64(0) .. u64(0)}
    for _, bytes in ipairs(files) do
      local file_path, line = refuse(bytes)
      test.includes(line, file_path, bytes)
      test.includes(line, "not GGUF", bytes)
    end
  end
)

test.it(
  "a version other than three is refused",
  function()
    local versions: {integer} = {2, 4}
    for _, version in ipairs(versions) do
      local _, line = refuse("GGUF" .. u32(version) .. u64(0) .. u64(0))
      test.includes(line, "version " .. tostring(version))
    end
  end
)

test.it(
  "a truncated file is refused",
  function()
    local files: {string} = {
      "GGUF",
      "GGUF" .. u32(3) .. "\0\0\0\0",
      header(0, 1) .. text("k") .. u32(8) .. u64(100) .. "abc",
      header(0, 1) .. text("k") .. u32(9) .. u32(5) .. u64(1000),
      header(1, 0) .. text("t") .. u32(1) .. u64(3) .. u32(0),
      header(0, 1099511627776),
    }
    for _, bytes in ipairs(files) do
      local file_path, line = refuse(bytes)
      test.includes(line, file_path)
      test.includes(line, "truncated")
    end
  end
)

test.it(
  "an unknown value type is refused",
  function()
    local cases: {TypeCase} = {
      {
        name = "bad.key",
        code = 13,
        bytes = header(0, 1) .. text("bad.key") .. u32(13),
      },
      {
        name = "bad.array",
        code = 99,
        bytes = header(0, 1)
          .. text("bad.array")
          .. u32(9)
          .. u32(99)
          .. u64(0),
      },
    }
    for _, case in ipairs(cases) do
      local _, line = refuse(case.bytes)
      test.includes(line, case.name)
      test.includes(line, tostring(case.code))
    end
  end
)

test.it(
  "an over-nested array is refused",
  function()
    local inner = array(5, {u32(1)})
    for _ = 1, 8 do
      inner = array(9, {inner})
    end
    local file_path, line = refuse(header(0, 1) .. pair("deep", 9, inner))
    test.includes(line, file_path)
    test.includes(line, "deep")

    local allowed = array(5, {u32(1)})
    for _ = 1, 7 do
      allowed = array(9, {allowed})
    end
    local info = read_file(header(0, 1) .. pair("deep", 9, allowed))
    local entry = info.metadata["deep"] as gguf.Value
    test.equal(entry.kind, "array")
  end
)
