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
long pread(int fd, void *buf, size_t count, off_t offset);
void *realloc(void *ptr, size_t size);
void free(void *ptr);
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
    -- the mapping's address, so a tensor pointer is a position in the file,
    -- and the descriptor the whole-matrix reads go through: -1 once released
    mapping: integer
    descriptor: integer
    -- the matrices this handle has read whole, by tensor name then index
    read_matrices: {string: {integer: boolean}}
    buffer: ffi.CData | nil
    buffer_bytes: integer
    mappers: integer
    release: function(Handle): string | nil
  end

  load: function(path: string, info?: gguf.Info): Handle | nil, string | nil
  attach: function(Handle)
  detach: function(Handle)
  read: function(Handle, Tensor, index: integer): boolean | nil, string | nil
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
  pread: function(integer, ffi.CData, integer, integer): integer
  realloc: function(ffi.CData | nil, integer): ffi.CData
  free: function(ffi.CData)
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
local RELEASED = -1

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

function weights.load(
  path: string, given_info?: gguf.Info
): Handle | nil, string | nil
  local info: gguf.Info
  if given_info is nil then
    local read, read_failure = gguf.read(path)
    if read is nil then
      return nil, read_failure
    end
    info = read
  else
    info = given_info
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
  if mapping == MAP_FAILED then
    C.close(descriptor)
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
  local function release(handle: Handle): string | nil
    if released then
      return nil
    end
    -- the tensors point into the mapping, so it outlives every holder
    if handle.mappers > 0 then
      return "release: " .. handle.mappers .. " still use it"
    end
    released = true
    C.munmap(mapping, size)
    C.close(handle.descriptor)
    handle.descriptor = RELEASED
    if handle.buffer ~= nil then
      C.free(handle.buffer as ffi.CData)
      handle.buffer = nil
    end
    handle.buffer_bytes = 0
    return nil
  end

  return {
    path = path,
    alignment = alignment,
    data_start = data_start,
    tensors = tensors,
    mapping = ffi.cast("intptr_t", mapping) as integer,
    descriptor = descriptor,
    read_matrices = {},
    buffer = nil,
    buffer_bytes = 0,
    mappers = 0,
    release = release,
  }
end

function weights.attach(handle: Handle)
  handle.mappers = handle.mappers + 1
end

function weights.detach(handle: Handle)
  if handle.mappers > 0 then
    handle.mappers = handle.mappers - 1
  end
end

-- the bytes one matrix of a stacked weight holds, or the refusal of an index
-- that names no matrix
local function matrix_of(
  tensor: Tensor, index: integer
): integer | nil, string | nil
  if #tensor.dims < 3 then
    return nil, "holds no matrices"
  end
  local matrices = tensor.dims[3] as integer
  if matrices < 1 or index < 0 or index >= matrices then
    return nil, "holds no matrix " .. tostring(index)
  end
  return math.floor(tensor.bytes / matrices)
end

function weights.read(
  handle: Handle, tensor: Tensor, index: integer
): boolean | nil, string | nil
  local each, why = matrix_of(tensor, index)
  if each is nil then
    return nil, fault(handle.path, tensor.name) .. ": " .. (why as string)
  end
  local bytes = each as integer
  if handle.descriptor == RELEASED then
    return nil, fault(handle.path, tensor.name) .. ": the handle is released"
  end

  local seen = handle.read_matrices[tensor.name]
  if seen is {integer: boolean} and seen[index] then
    return false
  end

  if handle.buffer_bytes < bytes then
    local grown = C.realloc(handle.buffer, bytes)
    if grown is nil then
      return nil,
        fault(handle.path, tensor.name)
          .. ": no room for a matrix of "
          .. tostring(bytes)
          .. " bytes"
    end
    handle.buffer = grown
    handle.buffer_bytes = bytes
  end

  local position = (
    ffi.cast("intptr_t", tensor.data) as integer
  ) - handle.mapping + index * bytes
  local moved = C.pread(
    handle.descriptor, handle.buffer as ffi.CData, bytes, position
  )
  if moved < bytes then
    return nil,
      fault(handle.path, tensor.name)
        .. ": matrix "
        .. tostring(index)
        .. " holds "
        .. tostring(bytes)
        .. " bytes, read "
        .. tostring(moved)
  end

  if seen is nil then
    seen = {}
    handle.read_matrices[tensor.name] = seen
  end
  seen[index] = true
  return true
end

return weights
