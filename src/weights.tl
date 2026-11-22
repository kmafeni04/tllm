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
int getpagesize(void);
int mincore(void *addr, size_t length, unsigned char *vec);
]]
)

local record weights
  record Shard
    path: string
    bytes: integer
    alignment: integer
    data_start: integer
    -- the address the file was mapped at, 0 once given back, so a tensor
    -- pointer is a position in the file, and the descriptor whole-matrix
    -- reads go through, -1 once given back
    mapping: integer
    descriptor: integer
  end

  record Tensor
    name: string
    dims: {integer}
    kind: ggml.Type
    elements: integer
    bytes: integer
    data: ffi.CData
    shard: Shard
  end

  record Handle
    path: string
    shards: {Shard}
    tensors: {string: Tensor}
    -- the matrices this handle has read whole, by tensor name then index
    read_matrices: {string: {integer: boolean}}
    buffer: ffi.CData | nil
    buffer_bytes: integer
    mappers: integer
    release: function(Handle): string | nil
  end

  load: function(
    path: string, info?: gguf.Info, extra?: {string}
  ): Handle | nil, string | nil
  attach: function(Handle)
  detach: function(Handle)
  read: function(Handle, Tensor, index: integer): boolean | nil, string | nil
  -- true when every page the matrix spans is in the page cache, false when
  -- the index names no matrix or the check cannot be made
  resident: function(Handle, Tensor, index: integer): boolean
end

local type Shard = weights.Shard
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
  getpagesize: function(): integer
  mincore: function(ffi.CData, integer, ffi.CData): integer
end

local record Planned
  info: gguf.Tensor
  kind: ggml.Type
  elements: integer
  bytes: integer
end

local C = ffi.C as LibC

local ALIGNMENT_KEY = "general.alignment"
local SPLIT_COUNT = "split.count"
local SPLIT_NO = "split.no"
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

-- the whole number a file states for `key`: nil with no line when it states
-- none, and the refusal when it states something else
local function stated(info: gguf.Info, key: string): integer | nil, string | nil
  local entry: gguf.Value | nil = info.metadata[key]
  if entry is nil then
    return nil
  end
  local scalar = entry.scalar
  if scalar is nil or scalar % 1 ~= 0 then
    return nil, key .. " is not a whole number"
  end
  return scalar as integer
end

local function five(value: integer): string
  return string.format("%05d", value)
end

-- the path before a shard's `-NNNNN-of-MMMMM.gguf` name, or nil when this
-- path is not the first shard of a split of that count
local function stem_of(path: string, count: integer): string | nil
  local stem, digits = path:match("^(.*)%-00001%-of%-(%d+)%.gguf$")
  if not (stem is string) or not (digits is string) then
    return nil
  end
  if digits ~= five(count) then
    return nil
  end
  return stem
end

local function give_back(shard: Shard)
  if shard.mapping == 0 then
    return
  end
  C.munmap(ffi.cast("void *", shard.mapping) as ffi.CData, shard.bytes)
  if shard.descriptor ~= RELEASED then
    C.close(shard.descriptor)
  end
  shard.mapping = 0
  shard.descriptor = RELEASED
end

-- the refusal of a shard's split keys against the index and count its name
-- carries, or nil when they agree
local function shard_keys(
  header: gguf.Info,
  file_path: string,
  expected_no: integer,
  expected_count: integer
): string | nil
  local count, count_failure = stated(header, SPLIT_COUNT)
  if not (count_failure is nil) then
    return file_path .. ": " .. count_failure
  end
  if count is nil then
    return file_path
      .. ": does not state split.count "
      .. tostring(expected_count)
  end
  if count ~= expected_count then
    return file_path
      .. ": split.count "
      .. tostring(count)
      .. ", the name gives "
      .. tostring(expected_count)
  end
  local split_no, no_failure = stated(header, SPLIT_NO)
  if not (no_failure is nil) then
    return file_path .. ": " .. no_failure
  end
  if split_no is nil then
    return file_path .. ": does not state split.no " .. tostring(expected_no)
  end
  if split_no ~= expected_no then
    return file_path
      .. ": split.no "
      .. tostring(split_no)
      .. ", the name gives "
      .. tostring(expected_no)
  end
  return nil
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
  path: string, given_info?: gguf.Info, extra?: {string}
): Handle | nil, string | nil
  local shards: {Shard} = {}
  local tensors: {string: Tensor} = {}
  local names: {string: boolean} = {}
  local loaded: {string: boolean} = {}

  -- maps one file's bytes and plans its tensors; the caller checks its split
  -- keys
  local function add_file(
    file_path: string, header: gguf.Info
  ): boolean | nil, string | nil
    local alignment, alignment_failure = alignment_of(header)
    if alignment is nil then
      return nil, file_path .. ": " .. (alignment_failure as string)
    end
    local data_start = rounded_up(header.table_end, alignment)
    local size = header.bytes

    local planned: {Planned} = {}
    for _, tensor in ipairs(header.tensors) do
      local kind = ggml.type_of(tensor.type_code)
      if kind is nil then
        return nil,
          fault(file_path, tensor.name)
            .. ": type code "
            .. tostring(tensor.type_code)
            .. " is not supported"
      end
      if names[tensor.name] then
        return nil, fault(file_path, tensor.name) .. ": duplicate name"
      end
      names[tensor.name] = true

      local entry, plan_failure = plan(file_path, tensor, kind)
      if entry is nil then
        return nil, plan_failure
      end
      if data_start + tensor.offset + entry.bytes > size then
        return nil,
          fault(file_path, tensor.name)
            .. ": data runs past the end of the file"
      end
      planned[#planned + 1] = entry
    end

    local descriptor = C.open(file_path, O_RDONLY)
    if descriptor < 0 then
      return nil, file_path .. ": cannot be opened for mapping"
    end
    local mapping = C.mmap(nil, size, PROT_READ, MAP_PRIVATE, descriptor, 0)
    if mapping == MAP_FAILED then
      C.close(descriptor)
      return nil, file_path .. ": cannot be mapped"
    end

    local shard: Shard = {
      path = file_path,
      bytes = size,
      alignment = alignment,
      data_start = data_start,
      mapping = ffi.cast("intptr_t", mapping) as integer,
      descriptor = descriptor,
    }
    shards[#shards + 1] = shard
    loaded[file_path] = true

    local base = shard.mapping
    for _, entry in ipairs(planned) do
      tensors[entry.info.name] = {
        name = entry.info.name,
        dims = entry.info.dims,
        kind = entry.kind,
        elements = entry.elements,
        bytes = entry.bytes,
        data = ffi.cast("void *", base + data_start + entry.info.offset)
          as ffi.CData,
        shard = shard,
      }
    end
    return true
  end

  -- one sibling, read and checked against the index and count its name carries
  local function add_sibling(
    stem: string, index: integer, count: integer
  ): boolean | nil, string | nil
    local sibling = stem
      .. "-"
      .. five(index + 1)
      .. "-of-"
      .. five(count)
      .. ".gguf"
    if loaded[sibling] then
      return true
    end
    local read, read_failure = gguf.read(sibling)
    if read is nil then
      return nil, read_failure
    end
    local refusal = shard_keys(read, sibling, index, count)
    if not (refusal is nil) then
      return nil, refusal
    end
    return add_file(sibling, read)
  end

  -- adds a path's file and, when it states a split count above 1, its siblings
  local function add_group(
    file_path: string, info?: gguf.Info
  ): boolean | nil, string | nil
    if loaded[file_path] then
      return true
    end
    local header = info
    if header is nil then
      local read, read_failure = gguf.read(file_path)
      if read is nil then
        return nil, read_failure
      end
      header = read
    end

    local count, count_failure = stated(header, SPLIT_COUNT)
    if not (count_failure is nil) then
      return nil, file_path .. ": " .. count_failure
    end
    if count is nil or count < 2 then
      return add_file(file_path, header)
    end

    local stem = stem_of(file_path, count)
    if stem is nil then
      return nil,
        file_path
          .. ": the name does not end -00001-of-"
          .. five(count)
          .. ".gguf"
    end
    local refused = shard_keys(header, file_path, 0, count)
    if not (refused is nil) then
      return nil, refused
    end
    local added, failure = add_file(file_path, header)
    if added is nil then
      return nil, failure
    end
    for index = 1, count - 1 do
      local ok, line = add_sibling(stem, index, count)
      if ok is nil then
        return nil, line
      end
    end
    return true
  end

  local function give_all_back()
    for _, shard in ipairs(shards) do
      give_back(shard)
    end
  end

  local added, failure = add_group(path, given_info)
  if added is nil then
    give_all_back()
    return nil, failure
  end

  if not (extra is nil) then
    for _, other in ipairs(extra) do
      local ok, line = add_group(other)
      if ok is nil then
        give_all_back()
        return nil, line
      end
    end
  end

  local released = false
  local function release(handle: Handle): string | nil
    if released then
      return nil
    end
    -- the tensors point into the mappings, so they outlive every holder
    if handle.mappers > 0 then
      return "release: " .. handle.mappers .. " still use it"
    end
    released = true
    for _, shard in ipairs(handle.shards) do
      give_back(shard)
    end
    if handle.buffer ~= nil then
      C.free(handle.buffer as ffi.CData)
      handle.buffer = nil
    end
    handle.buffer_bytes = 0
    return nil
  end

  return {
    path = path,
    shards = shards,
    tensors = tensors,
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

-- whether every page from the one holding the matrix's first byte through
-- the one holding its last is in the page cache, asked without faulting a
-- page in
function weights.resident(_: Handle, tensor: Tensor, index: integer): boolean
  local each = matrix_of(tensor, index)
  if each is nil or (each as integer) < 1 then
    return false
  end
  local bytes = each as integer
  local page = C.getpagesize()
  if page < 1 then
    return false
  end
  local address = ffi.cast("intptr_t", tensor.data) as integer
  local first = (tonumber(address) as integer) + index * bytes
  local last = first + bytes - 1
  local start = first - first % page
  local count = math.floor((last - last % page - start) / page) + 1
  local pages = ffi.new("uint8_t[?]", count)
  local answer = C.mincore(
    ffi.cast("void *", start) as ffi.CData, count * page, pages
  )
  if answer ~= 0 then
    return false
  end
  local flags = ffi.string(pages as ffi.CData, count)
  for at = 1, count do
    local flag = flags:byte(at) or 0
    if flag % 2 == 0 then
      return false
    end
  end
  return true
end

function weights.read(
  handle: Handle, tensor: Tensor, index: integer
): boolean | nil, string | nil
  local shard = tensor.shard
  local each, why = matrix_of(tensor, index)
  if each is nil then
    return nil, fault(shard.path, tensor.name) .. ": " .. (why as string)
  end
  local bytes = each as integer
  if shard.descriptor == RELEASED then
    return nil, fault(shard.path, tensor.name) .. ": the handle is released"
  end

  local seen = handle.read_matrices[tensor.name]
  if seen is {integer: boolean} and seen[index] then
    return false
  end

  if handle.buffer_bytes < bytes then
    local grown = C.realloc(handle.buffer, bytes)
    if grown is nil then
      return nil,
        fault(shard.path, tensor.name)
          .. ": no room for a matrix of "
          .. tostring(bytes)
          .. " bytes"
    end
    handle.buffer = grown
    handle.buffer_bytes = bytes
  end

  local position = (
    ffi.cast("intptr_t", tensor.data) as integer
  ) - shard.mapping + index * bytes
  local moved = C.pread(
    shard.descriptor, handle.buffer as ffi.CData, bytes, position
  )
  if moved < bytes then
    return nil,
      fault(shard.path, tensor.name)
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
