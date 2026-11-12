local ffi = require("ffi")
local ggml = require("tllm.ggml")
local gguf = require("tllm.gguf")

ffi.cdef(
  [[
typedef long off_t;
int open(const char *pathname, int flags, ...);
int close(int fd);
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
]]
)

local record weights
  record Tensor
    name: string
    dims: {integer}
    kind: ggml.Type
    elements: integer
    bytes: integer
    data: ffi.CData
  end

  record Handle
    path: string
    alignment: integer
    data_start: integer
    tensors: {string: Tensor}
    release: function(Handle)
  end

  load: function(path: string): Handle | nil, string | nil
end

local type Tensor = weights.Tensor
local type Handle = weights.Handle

local record LibC
  open: function(string, integer): integer
  close: function(integer)
  mmap: function(
    ffi.CData | nil, integer, integer, integer, integer, integer
  ): ffi.CData
  munmap: function(ffi.CData, integer): integer
end

local C = ffi.C as LibC

local record Planned
  info: gguf.Tensor
  kind: ggml.Type
  elements: integer
  bytes: integer
end

local ALIGNMENT_KEY = "general.alignment"
local DEFAULT_ALIGNMENT = 32
local PROT_READ = 1
local MAP_PRIVATE = 2
local O_RDONLY = 0
local MAP_FAILED = ffi.cast("void *", -1)

local function rounded_up(position: integer, alignment: integer): integer
  return math.floor((position + alignment - 1) / alignment) * alignment
end

local function fault(path: string, name: string): string
  return path .. ": tensor " .. name
end

local function alignment_of(info: gguf.Info): integer | nil, string | nil
  local entry: gguf.Value | nil = info.metadata[ALIGNMENT_KEY]
  if entry is nil then
    return DEFAULT_ALIGNMENT
  end
  local scalar = entry.scalar
  if scalar is nil or scalar < 1 or scalar % 1 ~= 0 then
    return nil, ALIGNMENT_KEY .. ": not a positive integer"
  end
  return scalar as integer
end

local function plan(
  path: string, tensor: gguf.Tensor, kind: ggml.Type
): Planned | nil, string | nil
  local first = 1
  if #tensor.dims > 0 then
    first = tensor.dims[1]
  end
  if first % kind.block_elements ~= 0 then
    return nil,
      fault(path, tensor.name)
        .. ": first dimension is not a multiple of "
        .. tostring(kind.block_elements)
  end

  local elements = 1
  local rest = 1
  for index, dimension in ipairs(tensor.dims) do
    elements = elements * dimension
    if index > 1 then
      rest = rest * dimension
    end
  end

  return {
    info = tensor,
    kind = kind,
    elements = elements,
    bytes = math.floor(first / kind.block_elements) * kind.block_bytes * rest,
  }
end

function weights.load(path: string): Handle | nil, string | nil
  local info, read_failure = gguf.read(path)
  if info is nil then
    return nil, read_failure
  end

  local alignment, alignment_failure = alignment_of(info)
  if alignment is nil then
    return nil, path .. ": " .. (alignment_failure as string)
  end
  local data_start = rounded_up(info.table_end, alignment)
  local size = info.bytes

  local planned: {Planned} = {}
  local seen: {string: boolean} = {}
  for _, tensor in ipairs(info.tensors) do
    local kind = ggml.type_of(tensor.type_code)
    if kind is nil then
      return nil,
        fault(path, tensor.name)
          .. ": type code "
          .. tostring(tensor.type_code)
          .. " is not supported"
    end
    if seen[tensor.name] then
      return nil, fault(path, tensor.name) .. ": duplicate name"
    end
    seen[tensor.name] = true

    local entry, plan_failure = plan(path, tensor, kind)
    if entry is nil then
      return nil, plan_failure
    end
    if data_start + tensor.offset + entry.bytes > size then
      return nil,
        fault(path, tensor.name) .. ": data runs past the end of the file"
    end
    planned[#planned + 1] = entry
  end

  local descriptor = C.open(path, O_RDONLY)
  if descriptor < 0 then
    return nil, path .. ": cannot be opened for mapping"
  end
  local mapping = C.mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0)
  C.close(descriptor)
  if mapping == MAP_FAILED then
    return nil, path .. ": cannot be mapped"
  end

  local base = ffi.cast("intptr_t", mapping) as integer
  local tensors: {string: Tensor} = {}
  for _, entry in ipairs(planned) do
    tensors[entry.info.name] = {
      name = entry.info.name,
      dims = entry.info.dims,
      kind = entry.kind,
      elements = entry.elements,
      bytes = entry.bytes,
      data = ffi.cast("void *", base + data_start + entry.info.offset)
        as ffi.CData,
    }
  end

  local released = false
  local function release(_: Handle)
    if released then
      return
    end
    released = true
    C.munmap(mapping, size)
  end

  return {
    path = path,
    alignment = alignment,
    data_start = data_start,
    tensors = tensors,
    release = release,
  }
end

return weights
