local ffi = require("ffi")

local record gguf
  record Value
    kind: string
    scalar: number | nil
    text: string | nil
    flag: boolean | nil
    item_kind: string | nil
    items: {Value} | nil
  end

  record Tensor
    name: string
    dims: {integer}
    type_code: integer
    offset: integer
  end

  record Info
    version: integer
    tensor_count: integer
    metadata_count: integer
    metadata: {string: Value}
    tensors: {Tensor}
    parameters: integer
    bytes: integer
    table_end: integer
  end

  read: function(path: string): Info | nil, string | nil
end

local type Value = gguf.Value
local type Tensor = gguf.Tensor
local type Info = gguf.Info

local record Reading
  file: FILE
  path: string
  size: integer
  position: integer
end

local type Cell = {integer: ffi.CData}

local MAX_ARRAY_DEPTH = 8

local record Kind
  name: string
  width: integer | nil
end

local KINDS: {integer: Kind} = {
  [0] = {name = "uint8", width = 1},
  [1] = {name = "int8", width = 1},
  [2] = {name = "uint16", width = 2},
  [3] = {name = "int16", width = 2},
  [4] = {name = "uint32", width = 4},
  [5] = {name = "int32", width = 4},
  [6] = {name = "float32", width = 4},
  [7] = {name = "bool"},
  [8] = {name = "string"},
  [9] = {name = "array"},
  [10] = {name = "uint64", width = 8},
  [11] = {name = "int64", width = 8},
  [12] = {name = "float64", width = 8},
}

local WIDE_NAMES: {integer: string} = {
  [0] = "uint8_t",
  [1] = "int8_t",
  [2] = "uint16_t",
  [3] = "int16_t",
  [4] = "uint32_t",
  [5] = "int32_t",
  [6] = "float",
  [10] = "uint64_t",
  [11] = "int64_t",
  [12] = "double",
}

local function truncated(reading: Reading): string
  return reading.path .. ": truncated"
end

local function read_bytes(reading: Reading, count: integer): string | nil
  if count == 0 then
    return ""
  end
  if count < 0 or reading.position + count > reading.size then
    return nil
  end
  local chunk = reading.file:read(count)
  if chunk is string and #chunk == count then
    reading.position = reading.position + count
    return chunk
  end
  return nil
end

local function unsigned(bytes: string): number
  local total = 0
  for index = #bytes, 1, -1 do
    total = total * 256 + bytes:byte(index)
  end
  return total
end

local function wide(typename: string, bytes: string): number
  local buffer = ffi.new(typename .. "[1]")
  ffi.copy(buffer, bytes, #bytes)
  local cells = buffer as Cell
  return tonumber(cells[0])
end

local function read_string(reading: Reading): string | nil
  local head = read_bytes(reading, 8)
  if head is nil then
    return nil
  end
  local length = wide("uint64_t", head) as integer
  return read_bytes(reading, length)
end

local read_value: function(
  Reading, integer, string, integer
): Value | nil, string | nil
local read_array: function(Reading, string, integer): Value | nil, string | nil

function read_array(
  reading: Reading, key: string, depth: integer
): Value | nil, string | nil
  if depth > MAX_ARRAY_DEPTH then
    return nil, reading.path .. ": " .. key .. ": array nested too deeply"
  end
  local head = read_bytes(reading, 4)
  if head is nil then
    return nil, truncated(reading)
  end
  local item_code = unsigned(head) as integer
  local item_kind = KINDS[item_code]
  if item_kind is nil then
    return nil, key .. ": unknown type " .. tostring(item_code)
  end
  local count_head = read_bytes(reading, 8)
  if count_head is nil then
    return nil, truncated(reading)
  end
  local count = wide("uint64_t", count_head) as integer
  if count > reading.size - reading.position then
    return nil, truncated(reading)
  end
  local items: {Value} = {}
  for _ = 1, count do
    local item, failure = read_value(reading, item_code, key, depth)
    if item is nil then
      return nil, failure
    end
    items[#items + 1] = item
  end
  return {kind = "array", item_kind = item_kind.name, items = items}
end

function read_value(
  reading: Reading, code: integer, key: string, depth: integer
): Value | nil, string | nil
  local kind = KINDS[code]
  if kind is nil then
    return nil, key .. ": unknown type " .. tostring(code)
  end

  if code == 7 then
    local bytes = read_bytes(reading, 1)
    if not (bytes is string) then
      return nil, truncated(reading)
    end
    return {kind = "bool", flag = bytes:byte(1) ~= 0}
  end

  if code == 8 then
    local text = read_string(reading)
    if text is nil then
      return nil, truncated(reading)
    end
    return {kind = "string", text = text}
  end

  if code == 9 then
    return read_array(reading, key, depth + 1)
  end

  local width = kind.width as integer
  local bytes = read_bytes(reading, width)
  if not (bytes is string) then
    return nil, truncated(reading)
  end
  local scalar = wide(WIDE_NAMES[code] as string, bytes)
  return {kind = kind.name, scalar = scalar}
end

local function parse(reading: Reading): Info | nil, string | nil
  local version_bytes = read_bytes(reading, 4)
  if version_bytes is nil then
    return nil, truncated(reading)
  end
  local version = unsigned(version_bytes) as integer
  if version ~= 3 then
    return nil,
      reading.path .. ": version " .. tostring(version) .. " is not supported"
  end

  local tensor_head = read_bytes(reading, 8)
  if tensor_head is nil then
    return nil, truncated(reading)
  end
  local tensor_count = wide("uint64_t", tensor_head) as integer
  local metadata_head = read_bytes(reading, 8)
  if metadata_head is nil then
    return nil, truncated(reading)
  end
  local metadata_count = wide("uint64_t", metadata_head) as integer

  if tensor_count > reading.size or metadata_count > reading.size then
    return nil, truncated(reading)
  end

  local metadata: {string: Value} = {}
  for _ = 1, metadata_count do
    local key = read_string(reading)
    if key is nil then
      return nil, truncated(reading)
    end
    local code_bytes = read_bytes(reading, 4)
    if code_bytes is nil then
      return nil, truncated(reading)
    end
    local code = unsigned(code_bytes) as integer
    local entry, failure = read_value(reading, code, key, 0)
    if entry is nil then
      return nil, failure
    end
    metadata[key] = entry
  end

  local tensors: {Tensor} = {}
  local parameters = 0
  for _ = 1, tensor_count do
    local name = read_string(reading)
    if name is nil then
      return nil, truncated(reading)
    end
    local dims_head = read_bytes(reading, 4)
    if dims_head is nil then
      return nil, truncated(reading)
    end
    local dim_count = unsigned(dims_head) as integer
    if dim_count * 8 > reading.size - reading.position then
      return nil, truncated(reading)
    end
    local dims: {integer} = {}
    for _ = 1, dim_count do
      local dim_bytes = read_bytes(reading, 8)
      if dim_bytes is nil then
        return nil, truncated(reading)
      end
      dims[#dims + 1] = wide("uint64_t", dim_bytes) as integer
    end
    local type_bytes = read_bytes(reading, 4)
    if type_bytes is nil then
      return nil, truncated(reading)
    end
    local type_code = unsigned(type_bytes) as integer
    local offset_bytes = read_bytes(reading, 8)
    if offset_bytes is nil then
      return nil, truncated(reading)
    end
    local offset = wide("uint64_t", offset_bytes) as integer
    tensors[#tensors + 1] = {
      name = name,
      dims = dims,
      type_code = type_code,
      offset = offset,
    }

    local product = 1
    for _, dim in ipairs(dims) do
      product = product * dim
    end
    parameters = parameters + product
  end

  return {
    version = version,
    tensor_count = tensor_count,
    metadata_count = metadata_count,
    metadata = metadata,
    tensors = tensors,
    parameters = parameters as integer,
    bytes = reading.size,
    table_end = reading.position,
  }
end

function gguf.read(path: string): Info | nil, string | nil
  local file = io.open(path, "rb")
  if file is nil then
    return nil, path .. ": cannot be read"
  end
  local size = file:seek("end") or 0
  file:seek("set", 0)
  local magic: string | nil = file:read(4)
  if magic is nil or magic == "" then
    file:close()
    return nil, path .. ": cannot be read"
  end
  if #magic < 4 or magic ~= "GGUF" then
    file:close()
    return nil, path .. ": not GGUF"
  end
  local reading: Reading = {
    file = file,
    path = path,
    size = size,
    position = 4,
  }
  local info, failure = parse(reading)
  file:close()
  return info, failure
end

return gguf
