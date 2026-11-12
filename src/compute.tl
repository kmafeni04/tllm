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

  record Geometry
    heads: integer
    kv_heads: integer
    head_length: integer
    rope_dimensions: integer
    rope_base: number
    eps: number
    positions: integer
  end

  record Attention
    block: function(
      Attention, {number}, string, integer
    ): {number} | nil, string | nil
  end

  load: function(path?: string): Backend | nil, string | nil
  attach: function(Backend, weights.Handle): Maths | nil, string | nil
  attend: function(
    Backend, weights.Handle, Geometry
  ): Attention | nil, string | nil
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

local record Cache
  keys: ffi.CData
  values: ffi.CData
  count: integer
end

local record AttentionState
  mapped: Mapped
  geometry: compute.Geometry
  caches: {string: Cache}
end

local record BlockTensors
  norm: Tensor
  query: Tensor
  key: Tensor
  value: Tensor
  query_norm: Tensor
  key_norm: Tensor
  output: Tensor
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
  if arena is nil then
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
  if not (backend is nil) then
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

  if backend is nil then
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

local function mapped_of(
  lib: Library, handle: weights.Handle, operation: string
): Mapped | nil, string | nil
  local count = 0
  for _ in pairs(handle.tensors) do
    count = count + 1
  end

  local arena_size = arena_for(lib, count, 0)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena is nil then
    return nil, operation .. ": no context of " .. arena_size .. " bytes"
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

  return {
    lib = lib,
    tensors = handle.tensors,
    descriptors = descriptors,
  }
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
  if arena is nil then
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
  if not (backend is nil) then
    status = lib.ggml_backend_graph_compute(backend, graph)
    lib.ggml_backend_free(backend)
  end

  local numbers: {number} = {}
  if not (backend is nil) and status == GGML_STATUS_SUCCESS then
    numbers = floats_of(lib, result, count)
  end

  lib.ggml_free(ctx)
  libc.free(arena)

  if backend is nil then
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
  if tensor is nil then
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
    if index is nil then
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
  elseif not (index is nil) then
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

local function geometry_refusal(geometry: compute.Geometry): string | nil
  if geometry.heads < 1 then
    return "attend: heads " .. geometry.heads
  end
  if geometry.kv_heads < 1 then
    return "attend: kv_heads " .. geometry.kv_heads
  end
  if geometry.heads % geometry.kv_heads ~= 0 then
    return "attend: kv_heads "
      .. geometry.kv_heads
      .. " of "
      .. geometry.heads
      .. " heads"
  end
  if geometry.head_length < 1 then
    return "attend: head_length " .. geometry.head_length
  end
  local dimensions = geometry.rope_dimensions
  if
    dimensions < 2 or dimensions % 2 ~= 0 or dimensions > geometry.head_length
  then
    return "attend: rope_dimensions "
      .. dimensions
      .. " of "
      .. geometry.head_length
  end
  if geometry.rope_base <= 0 then
    return "attend: rope_base " .. geometry.rope_base
  end
  if geometry.eps < 0 then
    return "attend: eps " .. geometry.eps
  end
  if geometry.positions < 1 then
    return "attend: positions " .. geometry.positions
  end
  return nil
end

local function block_failure(line: string): string
  return "block: " .. (line:gsub("^[%a_]+:%s*", ""))
end

local function slice(values: {number}, first: integer, count: integer): {number}
  local result: {number} = {}
  for offset = 0, count - 1 do
    result[offset + 1] = values[first + offset + 1]
  end
  return result
end

local function tensor_names(prefix: string): {string}
  return {
    prefix .. "attn_norm.weight",
    prefix .. "attn_q.weight",
    prefix .. "attn_k.weight",
    prefix .. "attn_v.weight",
    prefix .. "attn_q_norm.weight",
    prefix .. "attn_k_norm.weight",
    prefix .. "attn_output.weight",
  }
end

local function count_refusal(
  name: string, got: integer, wanted: integer, unit: string
): string | nil
  if got ~= wanted then
    return "block: tensor "
      .. name
      .. " holds "
      .. got
      .. " "
      .. unit
      .. ", the geometry needs "
      .. wanted
  end
  return nil
end

local function read_tensors(
  mapped: Mapped,
  prefix: string,
  hidden_count: integer,
  geometry: compute.Geometry
): BlockTensors | nil, string | nil
  local names = tensor_names(prefix)
  for _, name in ipairs(names) do
    local tensor = mapped.tensors[name]
    if tensor is nil then
      return nil, "block: no tensor " .. name
    end
  end
  for _, name in ipairs(names) do
    local kind = (mapped.tensors[name] as Tensor).kind.name
    if not WEIGHTS[kind] then
      return nil, "block: tensor " .. name .. " is " .. kind
    end
  end

  local norm_name = prefix .. "attn_norm.weight"
  local query_name = prefix .. "attn_q.weight"
  local key_name = prefix .. "attn_k.weight"
  local value_name = prefix .. "attn_v.weight"
  local query_norm_name = prefix .. "attn_q_norm.weight"
  local key_norm_name = prefix .. "attn_k_norm.weight"
  local output_name = prefix .. "attn_output.weight"

  local norm = mapped.tensors[norm_name] as Tensor
  local query = mapped.tensors[query_name] as Tensor
  local key = mapped.tensors[key_name] as Tensor
  local value = mapped.tensors[value_name] as Tensor
  local query_norm = mapped.tensors[query_norm_name] as Tensor
  local key_norm = mapped.tensors[key_norm_name] as Tensor
  local output = mapped.tensors[output_name] as Tensor

  local heights = geometry.heads * geometry.head_length
  local widths = geometry.kv_heads * geometry.head_length

  local refusal = count_refusal(norm_name, #norm.dims, 1, "dimensions")
    or count_refusal(
      norm_name, (norm.dims[1] or 0) as integer, hidden_count, "numbers"
    )
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(query_name, #query.dims, 2, "dimensions")
    or count_refusal(
      query_name, (query.dims[1] or 0) as integer, hidden_count, "numbers"
    ) or count_refusal(
    query_name, (query.dims[2] or 0) as integer, heights * 2, "rows"
  )
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(key_name, #key.dims, 2, "dimensions")
    or count_refusal(
      key_name, (key.dims[1] or 0) as integer, hidden_count, "numbers"
    ) or count_refusal(key_name, (key.dims[2] or 0) as integer, widths, "rows")
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(value_name, #value.dims, 2, "dimensions")
    or count_refusal(
      value_name, (value.dims[1] or 0) as integer, hidden_count, "numbers"
    ) or count_refusal(
    value_name, (value.dims[2] or 0) as integer, widths, "rows"
  )
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(query_norm_name, #query_norm.dims, 1, "dimensions")
    or count_refusal(
      query_norm_name,
      (query_norm.dims[1] or 0) as integer,
      geometry.head_length,
      "numbers"
    )
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(key_norm_name, #key_norm.dims, 1, "dimensions")
    or count_refusal(
      key_norm_name,
      (key_norm.dims[1] or 0) as integer,
      geometry.head_length,
      "numbers"
    )
  if refusal then
    return nil, refusal
  end
  refusal = count_refusal(output_name, #output.dims, 2, "dimensions")
    or count_refusal(
      output_name, (output.dims[1] or 0) as integer, heights, "numbers"
    ) or count_refusal(
    output_name, (output.dims[2] or 0) as integer, hidden_count, "rows"
  )
  if refusal then
    return nil, refusal
  end

  return {
    norm = norm,
    query = query,
    key = key,
    value = value,
    query_norm = query_norm,
    key_norm = key_norm,
    output = output,
  }
end

local function position_refusal(
  prefix: string, position: integer, expected: integer, capacity: integer
): string | nil
  if position < 0 then
    return "block: prefix " .. prefix .. " position " .. position
  end
  if position >= capacity then
    return "block: prefix "
      .. prefix
      .. " position "
      .. position
      .. " of "
      .. capacity
  end
  if position ~= expected then
    return "block: prefix "
      .. prefix
      .. " position "
      .. position
      .. ", the next is "
      .. expected
  end
  return nil
end

local function cache_of(
  state: AttentionState, prefix: string
): Cache | nil, string | nil
  local stored = state.caches[prefix]
  if not (stored is nil) then
    return stored
  end

  local geometry = state.geometry
  local count = geometry.positions * geometry.kv_heads * geometry.head_length
  local made_keys, keys = pcall(ffi.new, "float[?]", count)
  local made_values, values = pcall(ffi.new, "float[?]", count)
  if not (made_keys and made_values) then
    return nil, "block: no cache of " .. (count * F32_BYTES * 2) .. " bytes"
  end

  local cache = {
    keys = keys as ffi.CData,
    values = values as ffi.CData,
    count = 0,
  } as Cache
  state.caches[prefix] = cache
  return cache
end

local function attention_block(
  state: AttentionState, hidden: {number}, prefix: string, position: integer
): {number} | nil, string | nil
  local geometry = state.geometry
  local mapped = state.mapped
  local tensors, refusal = read_tensors(mapped, prefix, #hidden, geometry)
  if not (tensors is BlockTensors) then
    return nil, refusal
  end

  local cache = state.caches[prefix]
  local expected = 0
  if not (cache is nil) then
    expected = cache.count
  end
  local bad_position = position_refusal(
    prefix, position, expected, geometry.positions
  )
  if bad_position then
    return nil, bad_position
  end
  if cache is nil then
    local made, failure = cache_of(state, prefix)
    if not (made is Cache) then
      return nil, failure
    end
    cache = made
  end

  local heads = geometry.heads
  local kv_heads = geometry.kv_heads
  local width = geometry.head_length
  local eps = geometry.eps

  local normalised, norm_failure = normalise(
    mapped, hidden, prefix .. "attn_norm.weight", eps
  )
  if normalised is nil then
    return nil, block_failure(norm_failure as string)
  end
  local queries, query_failure = matrix_vector(
    mapped, prefix .. "attn_q.weight", normalised
  )
  if queries is nil then
    return nil, block_failure(query_failure as string)
  end
  local read_keys, key_failure = matrix_vector(
    mapped, prefix .. "attn_k.weight", normalised
  )
  if read_keys is nil then
    return nil, block_failure(key_failure as string)
  end
  local read_values, value_failure = matrix_vector(
    mapped, prefix .. "attn_v.weight", normalised
  )
  if read_values is nil then
    return nil, block_failure(value_failure as string)
  end

  local query_norm_name = prefix .. "attn_q_norm.weight"
  local key_norm_name = prefix .. "attn_k_norm.weight"
  local rotated_queries: {{number}} = {}
  local gates: {{number}} = {}
  for head = 0, heads - 1 do
    local offset = head * width * 2
    local raw = slice(queries, offset, width)
    local scaled, norm_line = normalise(mapped, raw, query_norm_name, eps)
    if scaled is nil then
      return nil, block_failure(norm_line as string)
    end
    local turned, rope_line = rotary(
      mapped, scaled, position, geometry.rope_dimensions, geometry.rope_base
    )
    if turned is nil then
      return nil, block_failure(rope_line as string)
    end
    rotated_queries[head + 1] = turned
    gates[head + 1] = slice(queries, offset + width, width)
  end

  local rotated_keys: {{number}} = {}
  local value_heads: {{number}} = {}
  for head = 0, kv_heads - 1 do
    local offset = head * width
    local raw = slice(read_keys, offset, width)
    local scaled, norm_line = normalise(mapped, raw, key_norm_name, eps)
    if scaled is nil then
      return nil, block_failure(norm_line as string)
    end
    local turned, rope_line = rotary(
      mapped, scaled, position, geometry.rope_dimensions, geometry.rope_base
    )
    if turned is nil then
      return nil, block_failure(rope_line as string)
    end
    rotated_keys[head + 1] = turned
    value_heads[head + 1] = slice(read_values, offset, width)
  end

  local key_cells = ffi.cast("float*", cache.keys) as Floats
  local value_cells = ffi.cast("float*", cache.values) as Floats
  local scale = 1 / math.sqrt(width)
  local group = heads / kv_heads
  local combined: {number} = {}

  for head = 0, heads - 1 do
    local kv_head = math.floor(head / group)
    local query = rotated_queries[head + 1]
    local scores: {number} = {}
    for past = 0, position - 1 do
      local total: number = 0
      for offset = 0, width - 1 do
        total = total + query[offset + 1] * key_cells[(
          past * kv_heads + kv_head
        ) * width + offset]
      end
      scores[past + 1] = total * scale
    end
    local current: number = 0
    local current_keys = rotated_keys[kv_head + 1]
    for offset = 0, width - 1 do
      current = current + query[offset + 1] * current_keys[offset + 1]
    end
    scores[position + 1] = current * scale

    local attention_weights, soft_line = softmax(mapped, scores)
    if attention_weights is nil then
      return nil, block_failure(soft_line as string)
    end

    local attended: {number} = {}
    local value_numbers = value_heads[kv_head + 1]
    for offset = 0, width - 1 do
      local total: number = 0
      for past = 0, position - 1 do
        total = total + attention_weights[past + 1] * value_cells[(
          past * kv_heads + kv_head
        ) * width + offset]
      end
      total = total + attention_weights[position + 1] * value_numbers[offset + 1]
      attended[offset + 1] = total
    end

    local gated, gate_line = sigmoid_gate(mapped, gates[head + 1], attended)
    if gated is nil then
      return nil, block_failure(gate_line as string)
    end
    for offset = 1, width do
      combined[#combined + 1] = gated[offset]
    end
  end

  local result, output_failure = matrix_vector(
    mapped, prefix .. "attn_output.weight", combined
  )
  if result is nil then
    return nil, block_failure(output_failure as string)
  end

  for head = 0, kv_heads - 1 do
    for offset = 0, width - 1 do
      local index = (position * kv_heads + head) * width + offset
      key_cells[index] = rotated_keys[head + 1][offset + 1]
      value_cells[index] = value_heads[head + 1][offset + 1]
    end
  end
  cache.count = position + 1
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
  LIBRARIES[backend] = handle
  return backend
end

function compute.attach(
  backend: compute.Backend, handle: weights.Handle
): compute.Maths | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib is nil then
    return nil, "attach: backend was not loaded by compute.load"
  end
  local mapped, map_failure = mapped_of(lib, handle, "attach")
  if mapped is nil then
    return nil, map_failure
  end

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

function compute.attend(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.Attention | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib is nil then
    return nil, "attend: backend was not loaded by compute.load"
  end
  local refusal = geometry_refusal(geometry)
  if refusal then
    return nil, refusal
  end

  local mapped, map_failure = mapped_of(lib, handle, "attend")
  if mapped is nil then
    return nil, map_failure
  end

  local state: AttentionState = {
    mapped = mapped,
    geometry = geometry,
    caches = {},
  }
  local attention = {} as compute.Attention
  attention.block = function(
    _: compute.Attention, hidden: {number}, prefix: string, position: integer
  ): {number} | nil, string | nil
    return attention_block(state, hidden, prefix, position)
  end
  return attention
end

return compute
