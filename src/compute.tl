local ffi = require("ffi")

ffi.cdef(
  [[
typedef struct ggml_context ggml_context;
typedef struct ggml_tensor ggml_tensor;
typedef struct ggml_cgraph ggml_cgraph;
typedef struct ggml_backend ggml_backend;

struct ggml_init_params {
  size_t mem_size;
  void *mem_buffer;
  bool no_alloc;
};

const char *ggml_version(void);
size_t ggml_tensor_overhead(void);
size_t ggml_graph_overhead(void);
ggml_context *ggml_init(struct ggml_init_params params);
void ggml_free(ggml_context *ctx);
ggml_tensor *ggml_new_tensor_2d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1);
ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_cgraph *ggml_new_graph(ggml_context *ctx);
void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor);
void *ggml_get_data(const ggml_tensor *tensor);
ggml_backend *ggml_backend_cpu_init(void);
void ggml_backend_free(ggml_backend *backend);
int ggml_backend_graph_compute(ggml_backend *backend, ggml_cgraph *cgraph);

void *aligned_alloc(size_t alignment, size_t size);
void free(void *ptr);
]]
)

local type Floats = {integer: number}
local type Struct = {string | integer: ffi.CData}

local record Library
  ggml_version: function(): ffi.CData
  ggml_tensor_overhead: function(): integer
  ggml_graph_overhead: function(): integer
  ggml_init: function(ffi.CData): ffi.CData
  ggml_free: function(ffi.CData)
  ggml_new_tensor_2d: function(ffi.CData, integer, integer, integer): ffi.CData
  ggml_mul_mat: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_new_graph: function(ffi.CData): ffi.CData
  ggml_build_forward_expand: function(ffi.CData, ffi.CData)
  ggml_get_data: function(ffi.CData): ffi.CData
  ggml_backend_cpu_init: function(): ffi.CData
  ggml_backend_free: function(ffi.CData)
  ggml_backend_graph_compute: function(ffi.CData, ffi.CData): integer
end

local record Libc
  aligned_alloc: function(integer, integer): ffi.CData
  free: function(ffi.CData)
end

local record compute
  record Backend
    path: string
    version: string
    matvec: function(
      Backend, {{number}}, {number}
    ): {number} | nil, string | nil
  end

  load: function(path?: string): Backend | nil, string | nil
end

local libc = ffi.C as Libc
local DEFAULT_PATH = "target/tllm/libggml.so"
local F32_BYTES: integer = 4
-- GGML_MEM_ALIGN in the pinned ggml source
local ARENA_ALIGN: integer = 16
local GGML_TYPE_F32: integer = 0
local GGML_STATUS_SUCCESS: integer = 0

local function load_library(location: string): Library
  return ffi.load(location) as Library
end

local function refusal_line(location: string, reason: string): string
  local detail = (reason:gsub("^.-%.lua:%d+: ", ""))
  return "compute: cannot load " .. location .. ": " .. (detail:gsub("\n", " "))
end

local function probe_version(lib: Library): string
  return ffi.string(lib.ggml_version())
end

local function round_up(value: integer, alignment: integer): integer
  local remainder = value % alignment
  if remainder == 0 then
    return value
  end
  return value + alignment - remainder
end

local function shape_refusal(matrix: {{number}}, vector: {number}): string | nil
  if #matrix == 0 then
    return "matvec: empty matrix"
  end
  if #vector == 0 then
    return "matvec: empty vector"
  end
  for index, row in ipairs(matrix) do
    if #row == 0 then
      return "matvec: row " .. index .. " is empty"
    end
  end
  for index, row in ipairs(matrix) do
    if #row ~= #vector then
      return "matvec: row "
        .. index
        .. " holds "
        .. #row
        .. " numbers, the vector holds "
        .. #vector
    end
  end
  return nil
end

local function matvec(
  lib: Library, matrix: {{number}}, vector: {number}
): {number} | nil, string | nil
  local refusal = shape_refusal(matrix, vector)
  if refusal then
    return nil, refusal
  end

  local rows = #matrix
  local columns = #vector
  local data_bytes = (rows * columns + columns + rows) * F32_BYTES
  -- three tensors and a graph, plus one tensor overhead of slack
  local overhead = 4 * lib.ggml_tensor_overhead() + lib.ggml_graph_overhead()
  local mem_size = data_bytes + overhead

  local arena_size = round_up(mem_size, ARENA_ALIGN)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena == nil then
    return nil, "matvec: no context of " .. arena_size .. " bytes"
  end

  local params = ffi.new("struct ggml_init_params") as Struct
  params.mem_size = arena_size
  params.mem_buffer = arena
  local ctx = lib.ggml_init(params)

  local matrix_tensor = lib.ggml_new_tensor_2d(
    ctx, GGML_TYPE_F32, columns, rows
  )
  local vector_tensor = lib.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, 1)
  local product = lib.ggml_mul_mat(ctx, matrix_tensor, vector_tensor)
  local graph = lib.ggml_new_graph(ctx)
  lib.ggml_build_forward_expand(graph, product)

  local cells = ffi.cast("float*", lib.ggml_get_data(matrix_tensor)) as Floats
  local cell = 0
  for _, row in ipairs(matrix) do
    for _, value in ipairs(row) do
      cells[cell] = value
      cell = cell + 1
    end
  end

  local weights = ffi.cast("float*", lib.ggml_get_data(vector_tensor)) as Floats
  for position, value in ipairs(vector) do
    weights[position - 1] = value
  end

  local backend = lib.ggml_backend_cpu_init()
  local status: integer = GGML_STATUS_SUCCESS
  if backend ~= nil then
    status = lib.ggml_backend_graph_compute(backend, graph)
    lib.ggml_backend_free(backend)
  end

  local products = ffi.cast("float*", lib.ggml_get_data(product)) as Floats
  local result: {number} = {}
  for row = 0, rows - 1 do
    result[row + 1] = products[row]
  end

  lib.ggml_free(ctx)
  libc.free(arena)

  if backend == nil then
    return nil, "matvec: no backend"
  end
  if status ~= GGML_STATUS_SUCCESS then
    return nil, "matvec: compute returned " .. status
  end
  return result
end

function compute.load(path?: string): compute.Backend | nil, string | nil
  local location = path or DEFAULT_PATH
  local opened, lib = pcall(load_library, location)
  if not opened then
    return nil, refusal_line(location, lib as string)
  end
  local handle = lib as Library

  local known, version = pcall(probe_version, handle)
  if not known then
    return nil, refusal_line(location, version as string)
  end

  local backend = {
    path = location,
    version = version as string,
  } as compute.Backend
  backend.matvec = function(
    _: compute.Backend, matrix: {{number}}, vector: {number}
  ): {number} | nil, string | nil
    return matvec(handle, matrix, vector)
  end
  return backend
end

return compute
