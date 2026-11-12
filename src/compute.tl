local ffi = require("ffi")
local weights = require("tllm.weights")

-- struct ggml_tensor here is the fields up to data, byte-exact for the pinned
-- ggml commit: tensor_of writes the mapping into the library's tensor memory
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

struct ggml_tensor {
  int32_t type;
  void *buffer;
  int64_t ne[4];
  size_t nb[4];
  int32_t op;
  int32_t op_params[16];
  int32_t flags;
  void *src[10];
  void *view_src;
  size_t view_offs;
  void *data;
};

const char *ggml_version(void);
size_t ggml_tensor_overhead(void);
size_t ggml_graph_overhead(void);
ggml_context *ggml_init(struct ggml_init_params params);
void ggml_free(ggml_context *ctx);
ggml_tensor *ggml_new_tensor_2d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1);
ggml_tensor *ggml_new_tensor_4d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_get_rows(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_rms_norm(ggml_context *ctx, ggml_tensor *a, float eps);
ggml_tensor *ggml_mul(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_rope_ext(
  ggml_context *ctx, ggml_tensor *a, ggml_tensor *b, ggml_tensor *c,
  int n_dims, int mode, int n_ctx, float freq_base, float freq_scale,
  float ext_factor, float attn_factor, float beta_fast, float beta_slow);
ggml_tensor *ggml_soft_max(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_add(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_swiglu_split(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_sigmoid(ggml_context *ctx, ggml_tensor *a);
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
local type Ints = {integer: integer}
local type Params = {string | integer: ffi.CData | boolean}

local record Library
  ggml_version: function(): ffi.CData
  ggml_tensor_overhead: function(): integer
  ggml_graph_overhead: function(): integer
  ggml_init: function(Params): ffi.CData
  ggml_free: function(ffi.CData)
  ggml_new_tensor_2d: function(ffi.CData, integer, integer, integer): ffi.CData
  ggml_new_tensor_4d: function(
    ffi.CData, integer, integer, integer, integer, integer
  ): ffi.CData
  ggml_mul_mat: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_get_rows: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_rms_norm: function(ffi.CData, ffi.CData, number): ffi.CData
  ggml_mul: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_rope_ext: function(
    ffi.CData,
    ffi.CData,
    ffi.CData,
    ffi.CData | nil,
    integer,
    integer,
    integer,
    number,
    number,
    number,
    number,
    number,
    number
  ): ffi.CData
  ggml_soft_max: function(ffi.CData, ffi.CData): ffi.CData
  ggml_add: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_swiglu_split: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_sigmoid: function(ffi.CData, ffi.CData): ffi.CData
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

  record Maths
    matvec: function(
      Maths, string, {number}, index?: integer
    ): {number} | nil, string | nil
    row: function(Maths, string, integer): {number} | nil, string | nil
    norm: function(
      Maths, {number}, string, number
    ): {number} | nil, string | nil
    rope: function(
      Maths, {number}, integer, integer, number
    ): {number} | nil, string | nil
    softmax: function(Maths, {number}): {number} | nil, string | nil
    add: function(Maths, {number}, {number}): {number} | nil, string | nil
    swiglu: function(Maths, {number}, {number}): {number} | nil, string | nil
    sigmoid_gate: function(
      Maths, {number}, {number}
    ): {number} | nil, string | nil
  end

  load: function(path?: string): Backend | nil, string | nil
  attach: function(Backend, weights.Handle): Maths | nil, string | nil
end

local type Tensor = weights.Tensor

local record TensorView
  data: ffi.CData
end

local record Mapped
  lib: Library
  tensors: {string: Tensor}
  descriptors: {string: ffi.CData}
end

local libc = ffi.C as Libc
local DEFAULT_PATH = "target/tllm/libggml.so"
local F32_BYTES: integer = 4
local I32_BYTES: integer = 4
-- GGML_MEM_ALIGN in the pinned ggml source
local ARENA_ALIGN: integer = 16
-- room for the context and graph objects ggml aligns past the tensors
local ARENA_SLACK: integer = 8192
local GGML_TYPE_F32: integer = 0
local GGML_TYPE_I32: integer = 26
local GGML_STATUS_SUCCESS: integer = 0
local GGML_ROPE_TYPE_NEOX: integer = 2

local WEIGHTS: {string: boolean} = {
  f32 = true,
  q8_0 = true,
  q4_K = true,
  q5_K = true,
}

local F32_ONLY: {string: boolean} = {f32 = true}

-- the library each loaded backend reads, so attach needs no field the
-- published Backend record does not hold
local LIBRARIES: {compute.Backend: Library} = {}

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

-- tensors is the number of tensors the operation builds plus spares: an arena
-- one tensor short makes ggml dereference NULL instead of returning a refusal
local function arena_for(
  lib: Library, tensors: integer, data_bytes: integer
): integer
  local objects = lib.ggml_tensor_overhead() * (tensors + 8)
  local bytes = lib.ggml_graph_overhead() + objects + data_bytes + ARENA_SLACK
  return round_up(bytes, ARENA_ALIGN)
end

local function arena_params(size: integer): Params
  local params = ffi.new("struct ggml_init_params") as Params
  params.mem_size = size
  return params
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

  local params = arena_params(arena_size)
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

  local vector_cells = ffi.cast("float*", lib.ggml_get_data(vector_tensor))
    as Floats
  for position, value in ipairs(vector) do
    vector_cells[position - 1] = value
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

local function write_floats(cells: Floats, values: {number})
  for position, value in ipairs(values) do
    cells[position - 1] = value
  end
end

local function floats_of(
  lib: Library, tensor: ffi.CData, count: integer
): {number}
  local cells = ffi.cast("float*", lib.ggml_get_data(tensor)) as Floats
  local result: {number} = {}
  for position = 0, count - 1 do
    result[position + 1] = cells[position]
  end
  return result
end

local function tensor_of(
  lib: Library,
  ctx: ffi.CData,
  dims: {integer},
  code: integer,
  location: ffi.CData
): ffi.CData
  local ne: {integer} = {1, 1, 1, 1}
  for index, size in ipairs(dims) do
    ne[index] = size
  end
  local tensor = lib.ggml_new_tensor_4d(ctx, code, ne[1], ne[2], ne[3], ne[4])
  local view = tensor as TensorView
  view.data = location
  return tensor
end

local function vector_of(
  lib: Library, ctx: ffi.CData, values: {number}
): ffi.CData
  local tensor = lib.ggml_new_tensor_4d(ctx, GGML_TYPE_F32, #values, 1, 1, 1)
  write_floats(ffi.cast("float*", lib.ggml_get_data(tensor)) as Floats, values)
  return tensor
end

local function run(
  lib: Library,
  operation: string,
  arena_size: integer,
  count: integer,
  build: function(ffi.CData): ffi.CData
): {number} | nil, string | nil
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena == nil then
    return nil, operation .. ": no context of " .. arena_size .. " bytes"
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  local ctx = lib.ggml_init(params)
  local graph = lib.ggml_new_graph(ctx)
  local result = build(ctx)
  lib.ggml_build_forward_expand(graph, result)

  local backend = lib.ggml_backend_cpu_init()
  local status: integer = GGML_STATUS_SUCCESS
  if backend ~= nil then
    status = lib.ggml_backend_graph_compute(backend, graph)
    lib.ggml_backend_free(backend)
  end

  local numbers: {number} = {}
  if backend ~= nil and status == GGML_STATUS_SUCCESS then
    numbers = floats_of(lib, result, count)
  end

  lib.ggml_free(ctx)
  libc.free(arena)

  if backend == nil then
    return nil, operation .. ": no backend"
  end
  if status ~= GGML_STATUS_SUCCESS then
    return nil, operation .. ": compute returned " .. status
  end
  return numbers
end

local function looked_up(
  mapped: Mapped, operation: string, name: string, readable: {string: boolean}
): Tensor | nil, string | nil
  local tensor = mapped.tensors[name]
  if tensor == nil then
    return nil, operation .. ": no tensor " .. name
  end
  if not readable[tensor.kind.name] then
    return nil, operation .. ": tensor " .. name .. " is " .. tensor.kind.name
  end
  return tensor
end

local function matrix_vector(
  mapped: Mapped, name: string, vector: {number}, index?: integer
): {number} | nil, string | nil
  local tensor, refusal = looked_up(mapped, "matvec", name, WEIGHTS)
  if not (tensor is Tensor) then
    return nil, refusal
  end

  local dims = tensor.dims
  local depth = #dims
  if depth > 3 then
    return nil, "matvec: tensor " .. name .. " holds " .. depth .. " dimensions"
  end

  local rows: integer = 1
  local columns: integer = 1
  if depth >= 1 then
    columns = dims[1]
  end
  if depth >= 2 then
    rows = dims[2]
  end

  local location = tensor.data
  if depth == 3 then
    if index == nil then
      return nil,
        "matvec: tensor "
          .. name
          .. " holds "
          .. dims[3]
          .. " matrices and needs an index"
    end
    if index < 0 then
      return nil, "matvec: index " .. index
    end
    if index >= dims[3] then
      return nil, "matvec: index " .. index .. " of " .. dims[3]
    end
    local stride = math.floor(tensor.bytes / dims[3])
    local base = ffi.cast("intptr_t", tensor.data) as integer
    location = ffi.cast("void *", base + index * stride) as ffi.CData
  elseif index ~= nil then
    return nil,
      "matvec: tensor "
        .. name
        .. " holds "
        .. depth
        .. " dimensions, index "
        .. index
  end

  if #vector ~= columns then
    return nil,
      "matvec: vector holds "
        .. #vector
        .. " numbers, tensor "
        .. name
        .. " holds "
        .. columns
  end

  local lib = mapped.lib
  local data_bytes = (rows + columns) * F32_BYTES
  return run(
    lib,
    "matvec",
    arena_for(lib, 4, data_bytes),
    rows,
    function(ctx: ffi.CData): ffi.CData
      local weight = mapped.descriptors[name]
      if depth == 3 then
        weight = tensor_of(
          lib, ctx, {columns, rows}, tensor.kind.code, location
        )
      end
      return lib.ggml_mul_mat(ctx, weight, vector_of(lib, ctx, vector))
    end
  )
end

local function row(
  mapped: Mapped, name: string, id: integer
): {number} | nil, string | nil
  local tensor, refusal = looked_up(mapped, "row", name, WEIGHTS)
  if not (tensor is Tensor) then
    return nil, refusal
  end

  local dims = tensor.dims
  if #dims ~= 2 then
    return nil, "row: tensor " .. name .. " holds " .. #dims .. " dimensions"
  end
  if id < 0 then
    return nil, "row: id " .. id
  end
  if id >= dims[2] then
    return nil, "row: id " .. id .. " of " .. dims[2]
  end

  local lib = mapped.lib
  local columns = dims[1]
  return run(
    lib,
    "row",
    arena_for(lib, 4, columns * F32_BYTES + I32_BYTES),
    columns,
    function(ctx: ffi.CData): ffi.CData
      local index = lib.ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1)
      local cells = ffi.cast("int32_t*", lib.ggml_get_data(index)) as Ints
      cells[0] = id
      return lib.ggml_get_rows(ctx, mapped.descriptors[name], index)
    end
  )
end

local function normalise(
  mapped: Mapped, vector: {number}, gain_name: string, eps: number
): {number} | nil, string | nil
  local tensor, refusal = looked_up(mapped, "norm", gain_name, F32_ONLY)
  if not (tensor is Tensor) then
    return nil, refusal
  end

  local count = #vector
  if count == 0 then
    return nil, "norm: vector holds 0 numbers"
  end
  if eps < 0 then
    return nil, "norm: eps " .. eps
  end
  if #tensor.dims ~= 1 then
    return nil,
      "norm: gain " .. gain_name .. " holds " .. #tensor.dims .. " dimensions"
  end
  if tensor.dims[1] ~= count then
    return nil,
      "norm: gain "
        .. gain_name
        .. " holds "
        .. tensor.dims[1]
        .. " numbers, the vector holds "
        .. count
  end

  local lib = mapped.lib
  return run(
    lib,
    "norm",
    arena_for(lib, 6, count * 3 * F32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      local scaled = lib.ggml_rms_norm(ctx, vector_of(lib, ctx, vector), eps)
      return lib.ggml_mul(ctx, scaled, mapped.descriptors[gain_name])
    end
  )
end

local function rotary(
  mapped: Mapped,
  vector: {number},
  position: integer,
  dimensions: integer,
  base: number
): {number} | nil, string | nil
  local count = #vector
  if dimensions < 2 then
    return nil, "rope: dimensions " .. dimensions
  end
  if dimensions % 2 ~= 0 then
    return nil, "rope: dimensions " .. dimensions
  end
  if dimensions > count then
    return nil,
      "rope: dimensions " .. dimensions .. " of " .. count .. " numbers"
  end
  if position < 0 then
    return nil, "rope: position " .. position
  end
  if base <= 0 then
    return nil, "rope: base " .. base
  end

  local lib = mapped.lib
  return run(
    lib,
    "rope",
    arena_for(lib, 6, count * 2 * F32_BYTES + I32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      local input = vector_of(lib, ctx, vector)
      local places = lib.ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1)
      local cells = ffi.cast("int32_t*", lib.ggml_get_data(places)) as Ints
      cells[0] = position
      return lib.ggml_rope_ext(
        ctx,
        input,
        places,
        nil,
        dimensions,
        GGML_ROPE_TYPE_NEOX,
        0,
        base,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0
      )
    end
  )
end

local function softmax(
  mapped: Mapped, vector: {number}
): {number} | nil, string | nil
  local count = #vector
  if count == 0 then
    return nil, "softmax: vector holds 0 numbers"
  end

  local lib = mapped.lib
  return run(
    lib,
    "softmax",
    arena_for(lib, 4, count * 2 * F32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      return lib.ggml_soft_max(ctx, vector_of(lib, ctx, vector))
    end
  )
end

local function add(
  mapped: Mapped, left: {number}, right: {number}
): {number} | nil, string | nil
  if #left ~= #right then
    return nil, "add: " .. #left .. " and " .. #right
  end

  local lib = mapped.lib
  local count = #left
  return run(
    lib,
    "add",
    arena_for(lib, 4, count * 3 * F32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      local first = vector_of(lib, ctx, left)
      return lib.ggml_add(ctx, first, vector_of(lib, ctx, right))
    end
  )
end

local function swiglu(
  mapped: Mapped, gate: {number}, up: {number}
): {number} | nil, string | nil
  if #gate ~= #up then
    return nil, "swiglu: " .. #gate .. " and " .. #up
  end

  local lib = mapped.lib
  local count = #gate
  return run(
    lib,
    "swiglu",
    arena_for(lib, 4, count * 3 * F32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      local first = vector_of(lib, ctx, gate)
      return lib.ggml_swiglu_split(ctx, first, vector_of(lib, ctx, up))
    end
  )
end

local function sigmoid_gate(
  mapped: Mapped, gate: {number}, values: {number}
): {number} | nil, string | nil
  if #gate ~= #values then
    return nil, "sigmoid_gate: " .. #gate .. " and " .. #values
  end

  local lib = mapped.lib
  local count = #gate
  return run(
    lib,
    "sigmoid_gate",
    arena_for(lib, 6, count * 4 * F32_BYTES),
    count,
    function(ctx: ffi.CData): ffi.CData
      local gates = lib.ggml_sigmoid(ctx, vector_of(lib, ctx, gate))
      return lib.ggml_mul(ctx, gates, vector_of(lib, ctx, values))
    end
  )
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
  LIBRARIES[backend] = handle
  return backend
end

function compute.attach(
  backend: compute.Backend, handle: weights.Handle
): compute.Maths | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib == nil then
    return nil, "attach: backend was not loaded by compute.load"
  end
  local count = 0
  for _ in pairs(handle.tensors) do
    count = count + 1
  end

  local arena_size = arena_for(lib, count, 0)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena == nil then
    return nil, "attach: no context of " .. arena_size .. " bytes"
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  params.no_alloc = true
  local ctx = lib.ggml_init(params)

  local descriptors: {string: ffi.CData} = {}
  for name, tensor in pairs(handle.tensors) do
    descriptors[name] = tensor_of(
      lib, ctx, tensor.dims, tensor.kind.code, tensor.data
    )
  end

  local mapped: Mapped = {
    lib = lib,
    tensors = handle.tensors,
    descriptors = descriptors,
  }

  local maths = {} as compute.Maths
  maths.matvec = function(
    _: compute.Maths, name: string, vector: {number}, index?: integer
  ): {number} | nil, string | nil
    return matrix_vector(mapped, name, vector, index)
  end
  maths.row = function(
    _: compute.Maths, name: string, id: integer
  ): {number} | nil, string | nil
    return row(mapped, name, id)
  end
  maths.norm = function(
    _: compute.Maths, vector: {number}, gain: string, eps: number
  ): {number} | nil, string | nil
    return normalise(mapped, vector, gain, eps)
  end
  maths.rope = function(
    _: compute.Maths,
    vector: {number},
    position: integer,
    dimensions: integer,
    base: number
  ): {number} | nil, string | nil
    return rotary(mapped, vector, position, dimensions, base)
  end
  maths.softmax = function(
    _: compute.Maths, vector: {number}
  ): {number} | nil, string | nil
    return softmax(mapped, vector)
  end
  maths.add = function(
    _: compute.Maths, left: {number}, right: {number}
  ): {number} | nil, string | nil
    return add(mapped, left, right)
  end
  maths.swiglu = function(
    _: compute.Maths, gate: {number}, up: {number}
  ): {number} | nil, string | nil
    return swiglu(mapped, gate, up)
  end
  maths.sigmoid_gate = function(
    _: compute.Maths, gate: {number}, values: {number}
  ): {number} | nil, string | nil
    return sigmoid_gate(mapped, gate, values)
  end
  return maths
end

return compute
