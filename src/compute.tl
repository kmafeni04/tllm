local ffi = require("ffi")
local gguf = require("tllm.gguf")
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
ggml_tensor *ggml_new_tensor_3d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_tensor *ggml_view_2d(ggml_context *ctx, ggml_tensor *a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
ggml_tensor *ggml_reshape_2d(ggml_context *ctx, ggml_tensor *a, int64_t ne0, int64_t ne1);
ggml_tensor *ggml_reshape_3d(ggml_context *ctx, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_tensor *ggml_repeat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_concat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b, int dim);
ggml_tensor *ggml_scale(ggml_context *ctx, ggml_tensor *a, float s);
ggml_tensor *ggml_sub(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
ggml_tensor *ggml_silu(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_softplus(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_exp(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_ssm_conv(ggml_context *ctx, ggml_tensor *sx, ggml_tensor *c);
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
  ggml_new_tensor_3d: function(
    ffi.CData, integer, integer, integer, integer
  ): ffi.CData
  ggml_view_2d: function(
    ffi.CData, ffi.CData, integer, integer, integer, integer
  ): ffi.CData
  ggml_reshape_2d: function(ffi.CData, ffi.CData, integer, integer): ffi.CData
  ggml_reshape_3d: function(
    ffi.CData, ffi.CData, integer, integer, integer
  ): ffi.CData
  ggml_repeat: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_concat: function(ffi.CData, ffi.CData, ffi.CData, integer): ffi.CData
  ggml_scale: function(ffi.CData, ffi.CData, number): ffi.CData
  ggml_sub: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_silu: function(ffi.CData, ffi.CData): ffi.CData
  ggml_softplus: function(ffi.CData, ffi.CData): ffi.CData
  ggml_exp: function(ffi.CData, ffi.CData): ffi.CData
  ggml_ssm_conv: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
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
    experts_used: integer
    linear_key_heads: integer | nil
    linear_key_length: integer | nil
    linear_value_heads: integer | nil
    linear_value_length: integer | nil
    linear_conv_kernel: integer | nil
  end

  record Attention
    block: function(
      Attention, {number}, string, integer
    ): {number} | nil, string | nil
  end

  record FeedForward
    block: function(FeedForward, {number}, string): {number} | nil, string | nil
    scores: function(FeedForward, {number}): {number} | nil, string | nil
  end

  record LinearAttention
    block: function(
      LinearAttention, {number}, string, integer
    ): {number} | nil, string | nil
  end

  record Model
    positions: integer
    forward: function(Model, integer, integer): {number} | nil, string | nil
  end

  load: function(path?: string): Backend | nil, string | nil
  attach: function(Backend, weights.Handle): Maths | nil, string | nil
  attend: function(
    Backend, weights.Handle, Geometry
  ): Attention | nil, string | nil
  feed_forward: function(
    Backend, weights.Handle, Geometry
  ): FeedForward | nil, string | nil
  linear_attention: function(
    Backend, weights.Handle, Geometry
  ): LinearAttention | nil, string | nil
  model: function(
    Backend, weights.Handle, integer | nil
  ): Model | nil, string | nil
  top_token: function(
    Attention, FeedForward, {number}, {string}, integer
  ): integer | nil, string | nil
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

local record FeedForwardState
  mapped: Mapped
  geometry: compute.Geometry
end

local record LinearShape
  key_heads: integer
  key_length: integer
  value_heads: integer
  value_length: integer
  conv_kernel: integer
end

local record LinearKept
  history: ffi.CData
  state: ffi.CData
  count: integer
end

local record LinearState
  mapped: Mapped
  shape: LinearShape
  eps: number
  prefixes: {string: LinearKept}
end

local record ModelPlan
  mapped: Mapped
  kinds: {boolean}
  count: integer
  width: integer
  positions: integer
  attention: compute.Attention | nil
  linear: compute.LinearAttention | nil
  feed_forward: compute.FeedForward
end

local record ModelState
  mapped: Mapped
  kinds: {boolean}
  count: integer
  width: integer
  attention: compute.Attention | nil
  linear: compute.LinearAttention | nil
  feed_forward: compute.FeedForward
  taken: integer
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

local BLOCK_NAMES: {string} = {
  "post_attention_norm.weight",
  "ffn_gate_inp.weight",
  "ffn_up_exps.weight",
  "ffn_gate_exps.weight",
  "ffn_down_exps.weight",
  "ffn_up_shexp.weight",
  "ffn_gate_shexp.weight",
  "ffn_down_shexp.weight",
  "ffn_gate_inp_shexp.weight",
}

local BLOCK_TYPES: {string: {string: boolean}} = {
  ["post_attention_norm.weight"] = F32_ONLY,
  ["ffn_gate_inp.weight"] = F32_ONLY,
  ["ffn_up_exps.weight"] = WEIGHTS,
  ["ffn_gate_exps.weight"] = WEIGHTS,
  ["ffn_down_exps.weight"] = WEIGHTS,
  ["ffn_up_shexp.weight"] = WEIGHTS,
  ["ffn_gate_shexp.weight"] = WEIGHTS,
  ["ffn_down_shexp.weight"] = WEIGHTS,
  ["ffn_gate_inp_shexp.weight"] = F32_ONLY,
}

local SCORE_NAMES: {string} = {"output_norm.weight", "output.weight"}

local SCORE_TYPES: {string: {string: boolean}} = {
  ["output_norm.weight"] = F32_ONLY,
  ["output.weight"] = WEIGHTS,
}

local LINEAR_NAMES: {string} = {
  "attn_norm.weight",
  "attn_qkv.weight",
  "attn_gate.weight",
  "ssm_beta.weight",
  "ssm_alpha.weight",
  "ssm_conv1d.weight",
  "ssm_a",
  "ssm_dt.bias",
  "ssm_norm.weight",
  "ssm_out.weight",
}

local LINEAR_TYPES: {string: {string: boolean}} = {
  ["attn_norm.weight"] = F32_ONLY,
  ["attn_qkv.weight"] = WEIGHTS,
  ["attn_gate.weight"] = WEIGHTS,
  ["ssm_beta.weight"] = F32_ONLY,
  ["ssm_alpha.weight"] = F32_ONLY,
  ["ssm_conv1d.weight"] = F32_ONLY,
  ["ssm_a"] = F32_ONLY,
  ["ssm_dt.bias"] = F32_ONLY,
  ["ssm_norm.weight"] = F32_ONLY,
  ["ssm_out.weight"] = WEIGHTS,
}

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

local function named_failure(operation: string, line: string): string
  return operation .. ": " .. (line:gsub("^[%a_]+:%s*", ""))
end

local function block_failure(line: string): string
  return named_failure("block", line)
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

local function count_line(
  operation: string, name: string, got: integer, wanted: integer, unit: string
): string
  return operation
    .. ": tensor "
    .. name
    .. " holds "
    .. got
    .. " "
    .. unit
    .. ", the geometry needs "
    .. wanted
end

local function count_refusal(
  name: string, got: integer, wanted: integer, unit: string
): string | nil
  if got ~= wanted then
    return count_line("block", name, got, wanted, unit)
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

local function first_fault(
  mapped: Mapped,
  operation: string,
  prefix: string,
  names: {string},
  types: {string: {string: boolean}}
): string | nil
  for _, suffix in ipairs(names) do
    local tensor = mapped.tensors[prefix .. suffix]
    if tensor is nil then
      return operation .. ": no tensor " .. prefix .. suffix
    end
  end
  for _, suffix in ipairs(names) do
    local kind = (mapped.tensors[prefix .. suffix] as Tensor).kind.name
    if not types[suffix][kind] then
      return operation .. ": tensor " .. prefix .. suffix .. " is " .. kind
    end
  end
  return nil
end

local function block_shape_refusal(
  mapped: Mapped, prefix: string, count: integer, experts_used: integer
): string | nil
  local function dims_of(suffix: string): {integer}
    return (mapped.tensors[prefix .. suffix] as Tensor).dims
  end
  local function name_of(suffix: string): string
    return prefix .. suffix
  end

  local norm = name_of("post_attention_norm.weight")
  local norm_dims = dims_of("post_attention_norm.weight")
  local refusal = count_refusal(norm, #norm_dims, 1, "dimensions")
    or count_refusal(norm, (norm_dims[1] or 0) as integer, count, "numbers")
  if refusal then
    return refusal
  end

  local router = name_of("ffn_gate_inp.weight")
  local router_dims = dims_of("ffn_gate_inp.weight")
  refusal = count_refusal(router, #router_dims, 2, "dimensions")
    or count_refusal(router, (router_dims[1] or 0) as integer, count, "numbers")
  if refusal then
    return refusal
  end
  local rows = router_dims[2] as integer
  if rows < 1 then
    return "block: tensor " .. router .. " holds no rows"
  end

  local shared_router = name_of("ffn_gate_inp_shexp.weight")
  local shared_router_dims = dims_of("ffn_gate_inp_shexp.weight")
  refusal = count_refusal(shared_router, #shared_router_dims, 1, "dimensions")
    or count_refusal(
      shared_router, (shared_router_dims[1] or 0) as integer, count, "numbers"
    )
  if refusal then
    return refusal
  end

  local up_experts = name_of("ffn_up_exps.weight")
  local up_experts_dims = dims_of("ffn_up_exps.weight")
  refusal = count_refusal(up_experts, #up_experts_dims, 3, "dimensions")
    or count_refusal(
      up_experts, (up_experts_dims[1] or 0) as integer, count, "numbers"
    )
  if refusal then
    return refusal
  end
  local expert_length = up_experts_dims[2] as integer
  if expert_length < 1 then
    return "block: tensor " .. up_experts .. " holds no expert numbers"
  end
  refusal = count_refusal(
    up_experts, (up_experts_dims[3] or 0) as integer, rows, "matrices"
  )
  if refusal then
    return refusal
  end

  local gate_experts = name_of("ffn_gate_exps.weight")
  local gate_experts_dims = dims_of("ffn_gate_exps.weight")
  refusal = count_refusal(gate_experts, #gate_experts_dims, 3, "dimensions")
    or count_refusal(
      gate_experts, (gate_experts_dims[1] or 0) as integer, count, "numbers"
    ) or count_refusal(
    gate_experts,
    (gate_experts_dims[2] or 0) as integer,
    expert_length,
    "numbers per expert"
  ) or count_refusal(
    gate_experts, (gate_experts_dims[3] or 0) as integer, rows, "matrices"
  )
  if refusal then
    return refusal
  end

  local down_experts = name_of("ffn_down_exps.weight")
  local down_experts_dims = dims_of("ffn_down_exps.weight")
  refusal = count_refusal(down_experts, #down_experts_dims, 3, "dimensions")
    or count_refusal(
      down_experts,
      (down_experts_dims[1] or 0) as integer,
      expert_length,
      "numbers per expert"
    ) or count_refusal(
    down_experts, (down_experts_dims[2] or 0) as integer, count, "numbers"
  ) or count_refusal(
    down_experts, (down_experts_dims[3] or 0) as integer, rows, "matrices"
  )
  if refusal then
    return refusal
  end

  local up_shared = name_of("ffn_up_shexp.weight")
  local up_shared_dims = dims_of("ffn_up_shexp.weight")
  refusal = count_refusal(up_shared, #up_shared_dims, 2, "dimensions")
    or count_refusal(
      up_shared, (up_shared_dims[1] or 0) as integer, count, "numbers"
    )
  if refusal then
    return refusal
  end
  local shared_length = up_shared_dims[2] as integer
  if shared_length < 1 then
    return "block: tensor " .. up_shared .. " holds no shared numbers"
  end

  local gate_shared = name_of("ffn_gate_shexp.weight")
  local gate_shared_dims = dims_of("ffn_gate_shexp.weight")
  refusal = count_refusal(gate_shared, #gate_shared_dims, 2, "dimensions")
    or count_refusal(
      gate_shared, (gate_shared_dims[1] or 0) as integer, count, "numbers"
    ) or count_refusal(
    gate_shared,
    (gate_shared_dims[2] or 0) as integer,
    shared_length,
    "numbers per row"
  )
  if refusal then
    return refusal
  end

  local down_shared = name_of("ffn_down_shexp.weight")
  local down_shared_dims = dims_of("ffn_down_shexp.weight")
  refusal = count_refusal(down_shared, #down_shared_dims, 2, "dimensions")
    or count_refusal(
      down_shared,
      (down_shared_dims[1] or 0) as integer,
      shared_length,
      "numbers per row"
    ) or count_refusal(
    down_shared, (down_shared_dims[2] or 0) as integer, count, "numbers"
  )
  if refusal then
    return refusal
  end

  if experts_used > rows then
    return "block: tensor "
      .. router
      .. " holds "
      .. rows
      .. " rows, experts_used is "
      .. experts_used
  end
  return nil
end

local function chosen_experts(
  probabilities: {number}, count: integer
): {integer}
  local taken: {integer} = {}
  local used: {integer: boolean} = {}
  for _ = 1, count do
    local best = 0
    for index, probability in ipairs(probabilities) do
      if
        not used[index] and (best == 0 or probability > probabilities[best])
      then
        best = index
      end
    end
    used[best] = true
    taken[#taken + 1] = best - 1
  end
  return taken
end

local function expert_output(
  mapped: Mapped, prefix: string, normalised: {number}, expert: integer
): {number} | nil, string | nil
  local up, up_line = matrix_vector(
    mapped, prefix .. "ffn_up_exps.weight", normalised, expert
  )
  if up is nil then
    return nil, up_line
  end
  local gate, gate_line = matrix_vector(
    mapped, prefix .. "ffn_gate_exps.weight", normalised, expert
  )
  if gate is nil then
    return nil, gate_line
  end
  local mixed, mixed_line = swiglu(mapped, gate, up)
  if mixed is nil then
    return nil, mixed_line
  end
  return matrix_vector(mapped, prefix .. "ffn_down_exps.weight", mixed, expert)
end

local function shared_output(
  mapped: Mapped, prefix: string, normalised: {number}
): {number} | nil, string | nil
  local up, up_line = matrix_vector(
    mapped, prefix .. "ffn_up_shexp.weight", normalised
  )
  if up is nil then
    return nil, up_line
  end
  local gate, gate_line = matrix_vector(
    mapped, prefix .. "ffn_gate_shexp.weight", normalised
  )
  if gate is nil then
    return nil, gate_line
  end
  local mixed, mixed_line = swiglu(mapped, gate, up)
  if mixed is nil then
    return nil, mixed_line
  end
  local down, down_line = matrix_vector(
    mapped, prefix .. "ffn_down_shexp.weight", mixed
  )
  if down is nil then
    return nil, down_line
  end
  local router, router_line = matrix_vector(
    mapped, prefix .. "ffn_gate_inp_shexp.weight", normalised
  )
  if router is nil then
    return nil, router_line
  end
  local gates, gates_line = sigmoid_gate(mapped, router, {1})
  if gates is nil then
    return nil, gates_line
  end
  local gated = gates[1]
  for position = 1, #down do
    down[position] = down[position] * gated
  end
  return down
end

local function feed_forward_block(
  state: FeedForwardState, hidden: {number}, prefix: string
): {number} | nil, string | nil
  local mapped = state.mapped
  local fault = first_fault(mapped, "block", prefix, BLOCK_NAMES, BLOCK_TYPES)
  if fault then
    return nil, fault
  end
  local count = #hidden
  fault = block_shape_refusal(
    mapped, prefix, count, state.geometry.experts_used
  )
  if fault then
    return nil, fault
  end

  local normalised, norm_line = normalise(
    mapped, hidden, prefix .. "post_attention_norm.weight", state.geometry.eps
  )
  if normalised is nil then
    return nil, named_failure("block", norm_line as string)
  end

  local votes, vote_line = matrix_vector(
    mapped, prefix .. "ffn_gate_inp.weight", normalised
  )
  if votes is nil then
    return nil, named_failure("block", vote_line as string)
  end
  local probabilities, probability_line = softmax(mapped, votes)
  if probabilities is nil then
    return nil, named_failure("block", probability_line as string)
  end

  local selected = chosen_experts(probabilities, state.geometry.experts_used)
  local total: number = 0
  for _, expert in ipairs(selected) do
    total = total + probabilities[expert + 1]
  end

  local experts: {number} = {}
  for position = 1, count do
    experts[position] = 0
  end
  for _, expert in ipairs(selected) do
    local weight = probabilities[expert + 1] / total
    local output, output_line = expert_output(
      mapped, prefix, normalised, expert
    )
    if output is nil then
      return nil, named_failure("block", output_line as string)
    end
    for position = 1, count do
      experts[position] = experts[position] + weight * output[position]
    end
  end

  local shared, shared_line = shared_output(mapped, prefix, normalised)
  if shared is nil then
    return nil, named_failure("block", shared_line as string)
  end

  local mixed, mixed_line = add(mapped, hidden, experts)
  if mixed is nil then
    return nil, named_failure("block", mixed_line as string)
  end
  local result, result_line = add(mapped, mixed, shared)
  if result is nil then
    return nil, named_failure("block", result_line as string)
  end
  return result
end

local function feed_forward_scores(
  state: FeedForwardState, hidden: {number}
): {number} | nil, string | nil
  local mapped = state.mapped
  local fault = first_fault(mapped, "scores", "", SCORE_NAMES, SCORE_TYPES)
  if fault then
    return nil, fault
  end

  local norm_name = "output_norm.weight"
  local output_name = "output.weight"
  local norm = mapped.tensors[norm_name] as Tensor
  local output = mapped.tensors[output_name] as Tensor
  local count = #hidden
  local function expected(
    name: string, got: integer, wanted: integer, unit: string
  ): string | nil
    if got ~= wanted then
      return count_line("scores", name, got, wanted, unit)
    end
    return nil
  end

  local refusal = expected(norm_name, #norm.dims, 1, "dimensions")
    or expected(norm_name, (norm.dims[1] or 0) as integer, count, "numbers")
  if refusal then
    return nil, refusal
  end
  refusal = expected(output_name, #output.dims, 2, "dimensions")
    or expected(output_name, (output.dims[1] or 0) as integer, count, "numbers")
  if refusal then
    return nil, refusal
  end
  if (output.dims[2] or 0) < 1 then
    return nil, "scores: tensor " .. output_name .. " holds no rows"
  end

  local normalised, norm_line = normalise(
    mapped, hidden, norm_name, state.geometry.eps
  )
  if normalised is nil then
    return nil, named_failure("scores", norm_line as string)
  end
  local values, values_line = matrix_vector(mapped, output_name, normalised)
  if values is nil then
    return nil, named_failure("scores", values_line as string)
  end
  return values
end

local function linear_shape_of(
  geometry: compute.Geometry
): LinearShape | nil, string | nil
  local key_heads = geometry.linear_key_heads
  if key_heads is nil then
    return nil, "linear_attention: geometry has no linear_key_heads"
  end
  if key_heads < 1 then
    return nil, "linear_attention: linear_key_heads " .. key_heads
  end
  local key_length = geometry.linear_key_length
  if key_length is nil then
    return nil, "linear_attention: geometry has no linear_key_length"
  end
  if key_length < 1 then
    return nil, "linear_attention: linear_key_length " .. key_length
  end
  local value_heads = geometry.linear_value_heads
  if value_heads is nil then
    return nil, "linear_attention: geometry has no linear_value_heads"
  end
  if value_heads < 1 then
    return nil, "linear_attention: linear_value_heads " .. value_heads
  end
  if value_heads % key_heads ~= 0 then
    return nil,
      "linear_attention: linear_value_heads "
        .. value_heads
        .. " of "
        .. key_heads
        .. " linear_key_heads"
  end
  local value_length = geometry.linear_value_length
  if value_length is nil then
    return nil, "linear_attention: geometry has no linear_value_length"
  end
  if value_length < 1 then
    return nil, "linear_attention: linear_value_length " .. value_length
  end
  local conv_kernel = geometry.linear_conv_kernel
  if conv_kernel is nil then
    return nil, "linear_attention: geometry has no linear_conv_kernel"
  end
  if conv_kernel < 1 then
    return nil, "linear_attention: linear_conv_kernel " .. conv_kernel
  end
  if geometry.eps < 0 then
    return nil, "linear_attention: eps " .. geometry.eps
  end
  return {
    key_heads = key_heads,
    key_length = key_length,
    value_heads = value_heads,
    value_length = value_length,
    conv_kernel = conv_kernel,
  }
end

local function linear_shape_refusal(
  mapped: Mapped, prefix: string, count: integer, shape: LinearShape
): string | nil
  local head_rows = shape.key_heads * shape.key_length
  local rows = 2 * head_rows + shape.value_heads * shape.value_length
  local channels = shape.value_heads * shape.value_length

  local function dims_of(suffix: string): {integer}
    return (mapped.tensors[prefix .. suffix] as Tensor).dims
  end
  local function name_of(suffix: string): string
    return prefix .. suffix
  end

  local norm = name_of("attn_norm.weight")
  local norm_dims = dims_of("attn_norm.weight")
  local refusal = count_refusal(norm, #norm_dims, 1, "dimensions")
    or count_refusal(norm, (norm_dims[1] or 0) as integer, count, "numbers")
  if refusal then
    return refusal
  end

  local qkv = name_of("attn_qkv.weight")
  local qkv_dims = dims_of("attn_qkv.weight")
  refusal = count_refusal(qkv, #qkv_dims, 2, "dimensions")
    or count_refusal(qkv, (qkv_dims[1] or 0) as integer, count, "numbers")
    or count_refusal(qkv, (qkv_dims[2] or 0) as integer, rows, "rows")
  if refusal then
    return refusal
  end

  local gate = name_of("attn_gate.weight")
  local gate_dims = dims_of("attn_gate.weight")
  refusal = count_refusal(gate, #gate_dims, 2, "dimensions")
    or count_refusal(gate, (gate_dims[1] or 0) as integer, count, "numbers")
    or count_refusal(gate, (gate_dims[2] or 0) as integer, channels, "rows")
  if refusal then
    return refusal
  end

  local beta = name_of("ssm_beta.weight")
  local beta_dims = dims_of("ssm_beta.weight")
  refusal = count_refusal(beta, #beta_dims, 2, "dimensions")
    or count_refusal(beta, (beta_dims[1] or 0) as integer, count, "numbers")
    or count_refusal(
      beta, (beta_dims[2] or 0) as integer, shape.value_heads, "rows"
    )
  if refusal then
    return refusal
  end

  local alpha = name_of("ssm_alpha.weight")
  local alpha_dims = dims_of("ssm_alpha.weight")
  refusal = count_refusal(alpha, #alpha_dims, 2, "dimensions")
    or count_refusal(alpha, (alpha_dims[1] or 0) as integer, count, "numbers")
    or count_refusal(
      alpha, (alpha_dims[2] or 0) as integer, shape.value_heads, "rows"
    )
  if refusal then
    return refusal
  end

  local conv = name_of("ssm_conv1d.weight")
  local conv_dims = dims_of("ssm_conv1d.weight")
  refusal = count_refusal(conv, #conv_dims, 2, "dimensions")
    or count_refusal(
      conv, (conv_dims[1] or 0) as integer, shape.conv_kernel, "numbers per row"
    ) or count_refusal(conv, (conv_dims[2] or 0) as integer, rows, "rows")
  if refusal then
    return refusal
  end

  local a = name_of("ssm_a")
  local a_dims = dims_of("ssm_a")
  refusal = count_refusal(a, #a_dims, 1, "dimensions")
    or count_refusal(
      a, (a_dims[1] or 0) as integer, shape.value_heads, "numbers"
    )
  if refusal then
    return refusal
  end

  local dt = name_of("ssm_dt.bias")
  local dt_dims = dims_of("ssm_dt.bias")
  refusal = count_refusal(dt, #dt_dims, 1, "dimensions")
    or count_refusal(
      dt, (dt_dims[1] or 0) as integer, shape.value_heads, "numbers"
    )
  if refusal then
    return refusal
  end

  local ssm_norm = name_of("ssm_norm.weight")
  local ssm_norm_dims = dims_of("ssm_norm.weight")
  refusal = count_refusal(ssm_norm, #ssm_norm_dims, 1, "dimensions")
    or count_refusal(
      ssm_norm,
      (ssm_norm_dims[1] or 0) as integer,
      shape.value_length,
      "numbers"
    )
  if refusal then
    return refusal
  end

  local out = name_of("ssm_out.weight")
  local out_dims = dims_of("ssm_out.weight")
  refusal = count_refusal(out, #out_dims, 2, "dimensions")
    or count_refusal(
      out, (out_dims[1] or 0) as integer, channels, "numbers per row"
    ) or count_refusal(out, (out_dims[2] or 0) as integer, count, "rows")
  if refusal then
    return refusal
  end
  return nil
end

local function linear_prefix_of(
  state: LinearState, prefix: string
): LinearKept | nil, string | nil
  local shape = state.shape
  local head_rows = shape.key_heads * shape.key_length
  local rows = 2 * head_rows + shape.value_heads * shape.value_length
  -- one float stands in for an empty history, so the field never holds nil
  local history_floats = (shape.conv_kernel - 1) * rows
  if history_floats < 1 then
    history_floats = 1
  end
  local state_floats = shape.key_length * shape.value_length * shape.value_heads

  local made_history, history = pcall(ffi.new, "float[?]", history_floats)
  if not made_history then
    return nil,
      "block: no history of " .. (history_floats * F32_BYTES) .. " bytes"
  end
  local made_state, buffer = pcall(ffi.new, "float[?]", state_floats)
  if not made_state then
    return nil, "block: no state of " .. (state_floats * F32_BYTES) .. " bytes"
  end

  local kept: LinearKept = {
    history = history as ffi.CData,
    state = buffer as ffi.CData,
    count = 0,
  } as LinearKept
  state.prefixes[prefix] = kept
  return kept
end

local function linear_press(
  state: LinearState, hidden: {number}, prefix: string, position: integer
): {number} | nil, string | nil
  local mapped = state.mapped
  local shape = state.shape
  local fault = first_fault(mapped, "block", prefix, LINEAR_NAMES, LINEAR_TYPES)
  if fault then
    return nil, fault
  end
  fault = linear_shape_refusal(mapped, prefix, #hidden, shape)
  if fault then
    return nil, fault
  end

  local kept = state.prefixes[prefix]
  local expected = 0
  if not (kept is nil) then
    expected = kept.count
  end
  if position < 0 then
    return nil, "block: prefix " .. prefix .. " position " .. position
  end
  if position ~= expected then
    return nil,
      "block: prefix "
        .. prefix
        .. " position "
        .. position
        .. ", the next is "
        .. expected
  end
  if kept is nil then
    local made, failure = linear_prefix_of(state, prefix)
    if not (made is LinearKept) then
      return nil, failure
    end
    kept = made
  end

  local lib = mapped.lib
  local descriptors = mapped.descriptors
  local key_heads = shape.key_heads
  local key_length = shape.key_length
  local value_heads = shape.value_heads
  local value_length = shape.value_length
  local kernel = shape.conv_kernel
  local rows = 2 * key_heads * key_length + value_heads * value_length
  local channels = value_heads * value_length
  local count = #hidden
  local eps = state.eps
  local state_floats = key_length * value_length * value_heads
  local head_bytes = key_length * F32_BYTES
  local value_head_bytes = value_length * F32_BYTES
  local head_scale = 1 / math.sqrt(key_length)

  -- four copies of the state, the convolution window and the heads' vectors,
  -- each with room to spare
  local vector_floats = count + kernel * rows + 2 * channels
  local head_floats = key_length * value_heads
  local data_floats = 4 * state_floats + 8 * vector_floats + 8 * head_floats
  local arena_size = arena_for(lib, 96, data_floats * F32_BYTES)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena is nil then
    return nil, "block: no context of " .. arena_size .. " bytes"
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  local ctx = lib.ggml_init(params)
  local graph = lib.ggml_new_graph(ctx)

  local normalised = lib.ggml_mul(
    ctx,
    lib.ggml_rms_norm(ctx, vector_of(lib, ctx, hidden), eps),
    descriptors[prefix .. "attn_norm.weight"]
  )
  local projected = lib.ggml_mul_mat(
    ctx, descriptors[prefix .. "attn_qkv.weight"], normalised
  )
  local gates = lib.ggml_mul_mat(
    ctx, descriptors[prefix .. "attn_gate.weight"], normalised
  )
  local betas = lib.ggml_sigmoid(
    ctx,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "ssm_beta.weight"], normalised)
  )
  local decay = lib.ggml_exp(
    ctx,
    lib.ggml_mul(
      ctx,
      lib.ggml_softplus(
        ctx,
        lib.ggml_add(
          ctx,
          lib.ggml_mul_mat(
            ctx, descriptors[prefix .. "ssm_alpha.weight"], normalised
          ),
          descriptors[prefix .. "ssm_dt.bias"]
        )
      ),
      descriptors[prefix .. "ssm_a"]
    )
  )

  local current = lib.ggml_reshape_3d(ctx, projected, 1, rows, 1)
  local window = current
  if kernel > 1 then
    local past = lib.ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel - 1, rows, 1)
    local past_view = past as TensorView
    past_view.data = kept.history
    window = lib.ggml_concat(ctx, past, current, 0)
  end
  local convolved = lib.ggml_silu(
    ctx,
    lib.ggml_ssm_conv(ctx, window, descriptors[prefix .. "ssm_conv1d.weight"])
  )

  local queries = lib.ggml_scale(
    ctx,
    lib.ggml_rms_norm(
      ctx,
      lib.ggml_view_2d(ctx, convolved, key_length, key_heads, head_bytes, 0),
      eps / key_length
    ),
    head_scale
  )
  local reads = lib.ggml_scale(
    ctx,
    lib.ggml_rms_norm(
      ctx,
      lib.ggml_view_2d(
        ctx,
        convolved,
        key_length,
        key_heads,
        head_bytes,
        key_heads * head_bytes
      ),
      eps / key_length
    ),
    head_scale
  )
  local values = lib.ggml_reshape_3d(
    ctx,
    lib.ggml_view_2d(
      ctx,
      convolved,
      value_length,
      value_heads,
      value_head_bytes,
      2 * key_heads * head_bytes
    ),
    value_length,
    1,
    value_heads
  )

  local shapes = lib.ggml_new_tensor_2d(
    ctx, GGML_TYPE_F32, key_length, value_heads
  )
  local head_queries = lib.ggml_reshape_3d(
    ctx, lib.ggml_repeat(ctx, queries, shapes), key_length, 1, value_heads
  )
  local head_reads = lib.ggml_reshape_3d(
    ctx, lib.ggml_repeat(ctx, reads, shapes), key_length, 1, value_heads
  )

  local stored_state = lib.ggml_new_tensor_3d(
    ctx, GGML_TYPE_F32, key_length, value_length, value_heads
  )
  local state_view = stored_state as TensorView
  state_view.data = kept.state
  local decayed = lib.ggml_mul(
    ctx, stored_state, lib.ggml_reshape_3d(ctx, decay, 1, 1, value_heads)
  )
  local corrections = lib.ggml_mul(
    ctx,
    lib.ggml_sub(ctx, values, lib.ggml_mul_mat(ctx, decayed, head_reads)),
    lib.ggml_reshape_3d(ctx, betas, 1, 1, value_heads)
  )
  local updated = lib.ggml_add(
    ctx,
    decayed,
    lib.ggml_mul_mat(
      ctx,
      lib.ggml_reshape_3d(ctx, head_reads, 1, key_length, value_heads),
      lib.ggml_reshape_3d(ctx, corrections, 1, value_length, value_heads)
    )
  )
  local scanned = lib.ggml_scale(
    ctx, lib.ggml_mul_mat(ctx, updated, head_queries), head_scale
  )
  local gated = lib.ggml_mul(
    ctx,
    lib.ggml_mul(
      ctx,
      lib.ggml_rms_norm(ctx, scanned, eps),
      descriptors[prefix .. "ssm_norm.weight"]
    ),
    lib.ggml_silu(
      ctx, lib.ggml_reshape_3d(ctx, gates, value_length, 1, value_heads)
    )
  )
  local output = lib.ggml_mul_mat(
    ctx,
    descriptors[prefix .. "ssm_out.weight"],
    lib.ggml_reshape_2d(ctx, gated, channels, 1)
  )

  lib.ggml_build_forward_expand(graph, output)
  lib.ggml_build_forward_expand(graph, updated)

  local backend = lib.ggml_backend_cpu_init()
  local status: integer = GGML_STATUS_SUCCESS
  if not (backend is nil) then
    status = lib.ggml_backend_graph_compute(backend, graph)
    lib.ggml_backend_free(backend)
  end

  local numbers: {number} = {}
  if not (backend is nil) and status == GGML_STATUS_SUCCESS then
    numbers = floats_of(lib, output, count)

    local next_state = ffi.cast("float*", lib.ggml_get_data(updated)) as Floats
    local kept_state = ffi.cast("float*", kept.state) as Floats
    for cell = 0, state_floats - 1 do
      kept_state[cell] = next_state[cell]
    end
    if kernel > 1 then
      -- the past tensor reads a channel's taps consecutively, oldest first
      local slots = kernel - 1
      local history = ffi.cast("float*", kept.history) as Floats
      local fresh = ffi.cast("float*", lib.ggml_get_data(projected)) as Floats
      for channel = 0, rows - 1 do
        for slot = 0, slots - 2 do
          history[channel * slots + slot] = history[channel * slots + slot + 1]
        end
        history[channel * slots + slots - 1] = fresh[channel]
      end
    end
    kept.count = position + 1
  end

  lib.ggml_free(ctx)
  libc.free(arena)

  if backend is nil then
    return nil, "block: no backend"
  end
  if status ~= GGML_STATUS_SUCCESS then
    return nil, "block: compute returned " .. status
  end
  return numbers
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

function compute.feed_forward(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.FeedForward | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib is nil then
    return nil, "feed_forward: backend was not loaded by compute.load"
  end
  if geometry.experts_used < 1 then
    return nil, "feed_forward: experts_used " .. geometry.experts_used
  end
  if geometry.eps < 0 then
    return nil, "feed_forward: eps " .. geometry.eps
  end

  local mapped, map_failure = mapped_of(lib, handle, "feed_forward")
  if mapped is nil then
    return nil, map_failure
  end

  local state: FeedForwardState = {
    mapped = mapped,
    geometry = geometry,
  }
  local feed_forward = {} as compute.FeedForward
  feed_forward.block = function(
    _: compute.FeedForward, hidden: {number}, prefix: string
  ): {number} | nil, string | nil
    return feed_forward_block(state, hidden, prefix)
  end
  feed_forward.scores = function(
    _: compute.FeedForward, hidden: {number}
  ): {number} | nil, string | nil
    return feed_forward_scores(state, hidden)
  end
  return feed_forward
end

function compute.linear_attention(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.LinearAttention | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib is nil then
    return nil, "linear_attention: backend was not loaded by compute.load"
  end
  local shape, geometry_line = linear_shape_of(geometry)
  if not (shape is LinearShape) then
    return nil, geometry_line
  end

  local mapped, map_failure = mapped_of(lib, handle, "linear_attention")
  if mapped is nil then
    return nil, map_failure
  end

  local state: LinearState = {
    mapped = mapped,
    shape = shape,
    eps = geometry.eps,
    prefixes = {},
  }
  local linear = {} as compute.LinearAttention
  linear.block = function(
    _: compute.LinearAttention,
    hidden: {number},
    prefix: string,
    position: integer
  ): {number} | nil, string | nil
    return linear_press(state, hidden, prefix, position)
  end
  return linear
end

local function summed(left: {number}, right: {number}): {number}
  local result: {number} = {}
  for index, value in ipairs(left) do
    result[index] = value + (right[index] or 0)
  end
  return result
end

local function required_number(
  info: gguf.Info, name: string
): number | nil, string | nil
  local entry = info.metadata[name]
  if entry is nil then
    return nil, "model: no key " .. name
  end
  local scalar = entry.scalar
  if scalar is nil then
    return nil, "model: key " .. name .. " is not a number"
  end
  return scalar, nil
end

-- a held key comes back as a number, an absent one as nil and nil, and a key
-- holding text as nil and the line
local function held_number(
  info: gguf.Info, name: string
): number | nil, string | nil
  local entry = info.metadata[name]
  if entry is nil then
    return nil, nil
  end
  local scalar = entry.scalar
  if scalar is nil then
    return nil, "model: key " .. name .. " is not a number"
  end
  return scalar, nil
end

local function below_one(value: number, name: string): string | nil
  if value < 1 then
    return "model: " .. name .. " is " .. value
  end
  return nil
end

local function architecture_of(info: gguf.Info): string | nil, string | nil
  local entry = info.metadata["general.architecture"]
  if entry is nil then
    return nil, "model: no key general.architecture"
  end
  local name = entry.text
  if name is nil then
    return nil, "model: key general.architecture is not text"
  end
  return name, nil
end

local function model_plan(
  lib: Library,
  backend: compute.Backend,
  handle: weights.Handle,
  positions: integer | nil
): ModelPlan | nil, string | nil
  local info, read_line = gguf.read(handle.path)
  if info is nil then
    return nil, "model: " .. (read_line as string)
  end
  local architecture, arch_line = architecture_of(info)
  if architecture is nil then
    return nil, arch_line
  end
  local arch = architecture as string
  local function key(name: string): string
    return arch .. "." .. name
  end

  local block_count, block_line = required_number(info, key("block_count"))
  if block_count is nil then
    return nil, block_line
  end
  local refusal = below_one(block_count, key("block_count"))
  if refusal then
    return nil, refusal
  end

  local width, width_line = required_number(info, key("embedding_length"))
  if width is nil then
    return nil, width_line
  end
  refusal = below_one(width, key("embedding_length"))
  if refusal then
    return nil, refusal
  end

  local experts, experts_line = required_number(info, key("expert_used_count"))
  if experts is nil then
    return nil, experts_line
  end
  refusal = below_one(experts, key("expert_used_count"))
  if refusal then
    return nil, refusal
  end

  local eps, eps_line = required_number(
    info, key("attention.layer_norm_rms_epsilon")
  )
  if eps is nil then
    return nil, eps_line
  end
  if eps < 0 then
    return nil,
      "model: " .. key("attention.layer_norm_rms_epsilon") .. " is " .. eps
  end

  local context, context_line = required_number(info, key("context_length"))
  if context is nil then
    return nil, context_line
  end
  refusal = below_one(context, key("context_length"))
  if refusal then
    return nil, refusal
  end

  local nextn, nextn_line = held_number(info, key("nextn_predict_layers"))
  if nextn_line then
    return nil, nextn_line
  end
  local skipped = 0
  if not (nextn is nil) then
    if nextn < 0 then
      return nil, "model: " .. key("nextn_predict_layers") .. " is " .. nextn
    end
    skipped = math.floor(nextn)
  end

  local interval, interval_line = held_number(
    info, key("full_attention_interval")
  )
  if interval_line then
    return nil, interval_line
  end
  local step = 4
  if not (interval is nil) then
    refusal = below_one(interval, key("full_attention_interval"))
    if refusal then
      return nil, refusal
    end
    step = math.floor(interval)
  end

  local count = math.floor(block_count) - skipped
  local kinds: {boolean} = {}
  local has_full = false
  local has_linear = false
  local first_full = 0
  local found_full = false
  for index = 0, count - 1 do
    local full = (index + 1) % step == 0
    kinds[index + 1] = full
    if full then
      has_full = true
      if not found_full then
        first_full = index
        found_full = true
      end
    else
      has_linear = true
    end
  end

  local heads: number = 1
  local rope_dimensions: number = 2
  local rope_base: number = 1
  if has_full then
    local value, line = required_number(info, key("attention.head_count"))
    if value is nil then
      return nil, line
    end
    heads = value
    value, line = required_number(info, key("rope.dimension_count"))
    if value is nil then
      return nil, line
    end
    rope_dimensions = value
    value, line = required_number(info, key("rope.freq_base"))
    if value is nil then
      return nil, line
    end
    rope_base = value
  end

  local group_count: number = 1
  local state_size: number = 1
  local time_step_rank: number = 1
  local inner_size: number = 1
  local conv_kernel: number = 1
  if has_linear then
    local value, line = required_number(info, key("ssm.group_count"))
    if value is nil then
      return nil, line
    end
    group_count = value
    value, line = required_number(info, key("ssm.state_size"))
    if value is nil then
      return nil, line
    end
    state_size = value
    value, line = required_number(info, key("ssm.time_step_rank"))
    if value is nil then
      return nil, line
    end
    time_step_rank = value
    value, line = required_number(info, key("ssm.inner_size"))
    if value is nil then
      return nil, line
    end
    inner_size = value
    value, line = required_number(info, key("ssm.conv_kernel"))
    if value is nil then
      return nil, line
    end
    conv_kernel = value
  end

  refusal = below_one(heads, key("attention.head_count"))
    or below_one(rope_dimensions, key("rope.dimension_count"))
    or below_one(rope_base, key("rope.freq_base"))
  if refusal then
    return nil, refusal
  end

  if has_linear then
    refusal = below_one(group_count, key("ssm.group_count"))
      or below_one(state_size, key("ssm.state_size"))
      or below_one(time_step_rank, key("ssm.time_step_rank"))
      or below_one(inner_size, key("ssm.inner_size"))
      or below_one(conv_kernel, key("ssm.conv_kernel"))
    if refusal then
      return nil, refusal
    end
  end

  local taken = math.floor(context)
  if not (positions is nil) then
    if positions < 1 or positions > context then
      return nil, "model: positions is " .. positions
    end
    taken = positions
  end

  local mapped, map_line = mapped_of(lib, handle, "model")
  if mapped is nil then
    return nil, map_line
  end

  local head_length: integer = 1
  local kv_heads: integer = 1
  if has_full then
    local prefix = "blk." .. first_full .. "."
    local query = mapped.tensors[prefix .. "attn_q.weight"]
    if query is nil then
      return nil, "model: no tensor " .. prefix .. "attn_q.weight"
    end
    local key_tensor = mapped.tensors[prefix .. "attn_k.weight"]
    if key_tensor is nil then
      return nil, "model: no tensor " .. prefix .. "attn_k.weight"
    end
    local query_rows = (query.dims[2] or 0) as integer
    local key_rows = (key_tensor.dims[2] or 0) as integer
    if query_rows < 1 then
      return nil, "model: tensor " .. prefix .. "attn_q.weight holds no rows"
    end
    if key_rows < 1 then
      return nil, "model: tensor " .. prefix .. "attn_k.weight holds no rows"
    end
    local head_count = math.floor(heads)
    local twice_heads = 2 * head_count
    if query_rows % twice_heads ~= 0 then
      return nil,
        "model: tensor "
          .. prefix
          .. "attn_q.weight holds "
          .. query_rows
          .. " rows, not "
          .. twice_heads
    end
    head_length = math.floor(query_rows / twice_heads)
    if key_rows % head_length ~= 0 then
      return nil,
        "model: tensor "
          .. prefix
          .. "attn_k.weight holds "
          .. key_rows
          .. " rows, not "
          .. head_length
    end
    kv_heads = math.floor(key_rows / head_length)
    if head_count % kv_heads ~= 0 then
      return nil,
        "model: tensor "
          .. prefix
          .. "attn_k.weight gives "
          .. kv_heads
          .. " key/value heads of "
          .. head_count
    end
    local dimensions = math.floor(rope_dimensions)
    if dimensions < 2 or dimensions % 2 ~= 0 or dimensions > head_length then
      return nil,
        "model: "
          .. key("rope.dimension_count")
          .. " is "
          .. dimensions
          .. " of "
          .. head_length
    end
  end

  local linear_key_heads: integer = 1
  local linear_key_length: integer = 1
  local linear_value_heads: integer = 1
  local linear_value_length: integer = 1
  local linear_conv_kernel: integer = 1
  if has_linear then
    local heads_count = math.floor(time_step_rank)
    local inner = math.floor(inner_size)
    local groups = math.floor(group_count)
    if inner % heads_count ~= 0 then
      return nil,
        "model: "
          .. key("ssm.inner_size")
          .. " is "
          .. inner
          .. " of "
          .. heads_count
    end
    if heads_count % groups ~= 0 then
      return nil,
        "model: "
          .. key("ssm.time_step_rank")
          .. " is "
          .. heads_count
          .. " of "
          .. groups
    end
    linear_key_heads = groups
    linear_key_length = math.floor(state_size)
    linear_value_heads = heads_count
    linear_value_length = math.floor(inner / heads_count)
    linear_conv_kernel = math.floor(conv_kernel)
  end

  if count < 1 then
    return nil, "model: " .. key("block_count") .. " leaves no blocks"
  end

  local geometry: compute.Geometry = {
    heads = math.floor(heads),
    kv_heads = kv_heads,
    head_length = head_length,
    rope_dimensions = math.floor(rope_dimensions),
    rope_base = rope_base,
    eps = eps,
    positions = taken,
    experts_used = math.floor(experts),
    linear_key_heads = linear_key_heads,
    linear_key_length = linear_key_length,
    linear_value_heads = linear_value_heads,
    linear_value_length = linear_value_length,
    linear_conv_kernel = linear_conv_kernel,
  }

  local feed_forward, feed_line = compute.feed_forward(
    backend, handle, geometry
  )
  if feed_forward is nil then
    return nil, "model: " .. (feed_line as string)
  end

  local attention: compute.Attention | nil = nil
  if has_full then
    local made, line = compute.attend(backend, handle, geometry)
    if made is nil then
      return nil, "model: " .. (line as string)
    end
    attention = made
  end

  local linear: compute.LinearAttention | nil = nil
  if has_linear then
    local made, line = compute.linear_attention(backend, handle, geometry)
    if made is nil then
      return nil, "model: " .. (line as string)
    end
    linear = made
  end

  return {
    mapped = mapped,
    kinds = kinds,
    count = count,
    width = math.floor(width),
    positions = taken,
    attention = attention,
    linear = linear,
    feed_forward = feed_forward as compute.FeedForward,
  }
end

local function embedding_row(
  mapped: Mapped, width: integer, token_id: integer
): {number} | nil, string | nil
  local tensor = mapped.tensors["token_embd.weight"]
  if tensor is nil then
    return nil, "forward: no tensor token_embd.weight for token " .. token_id
  end
  local dims = tensor.dims
  if #dims ~= 2 then
    return nil,
      "forward: token_embd.weight id "
        .. token_id
        .. " holds "
        .. #dims
        .. " dimensions"
  end
  local rows = (dims[2] or 0) as integer
  if rows < 1 then
    return nil, "forward: token_embd.weight id " .. token_id .. " holds no rows"
  end
  if token_id < 0 or token_id >= rows then
    return nil,
      "forward: tensor token_embd.weight id " .. token_id .. " of " .. rows
  end
  if (dims[1] or 0) ~= width then
    return nil,
      "forward: token_embd.weight id "
        .. token_id
        .. " holds "
        .. (dims[1] or 0)
        .. " numbers per row, the width is "
        .. width
  end
  local values, line = row(mapped, "token_embd.weight", token_id)
  if values is nil then
    return nil, named_failure("forward", line as string)
  end
  return values
end

function compute.model(
  backend: compute.Backend, handle: weights.Handle, positions: integer | nil
): compute.Model | nil, string | nil
  local lib = LIBRARIES[backend]
  if lib is nil then
    return nil, "model: backend was not loaded by compute.load"
  end
  local plan, plan_line = model_plan(lib, backend, handle, positions)
  if plan is nil then
    return nil, plan_line
  end

  local state: ModelState = {
    mapped = plan.mapped,
    kinds = plan.kinds,
    count = plan.count,
    width = plan.width,
    attention = plan.attention,
    linear = plan.linear,
    feed_forward = plan.feed_forward,
    taken = 0,
  }

  local model = {} as compute.Model
  model.positions = plan.positions
  model.forward = function(
    _: compute.Model, token_id: integer, position: integer
  ): {number} | nil, string | nil
    if position < 0 or position ~= state.taken then
      return nil,
        "forward: position " .. position .. ", the next is " .. state.taken
    end

    local hidden, embed_line = embedding_row(
      state.mapped, state.width, token_id
    )
    if hidden is nil then
      return nil, embed_line
    end

    for index = 0, state.count - 1 do
      local prefix = "blk." .. index .. "."
      local contribution: {number} | nil
      local line: string | nil
      if state.kinds[index + 1] then
        local attention = state.attention as compute.Attention
        contribution, line = attention:block(hidden, prefix, position)
      else
        local linear = state.linear as compute.LinearAttention
        contribution, line = linear:block(hidden, prefix, position)
      end
      if contribution is nil then
        return nil, line
      end
      local blocked, block_line = state.feed_forward:block(
        summed(hidden, contribution), prefix
      )
      if blocked is nil then
        return nil, block_line
      end
      hidden = blocked
    end

    local scores, score_line = state.feed_forward:scores(hidden)
    if scores is nil then
      return nil, score_line
    end
    state.taken = state.taken + 1
    return scores, nil
  end
  return model
end

function compute.top_token(
  attention: compute.Attention,
  feed_forward: compute.FeedForward,
  hidden: {number},
  prefixes: {string},
  position: integer
): integer | nil, string | nil
  if #prefixes == 0 then
    return nil, "top_token: no prefixes"
  end

  local running = hidden
  for _, prefix in ipairs(prefixes) do
    local attended, attend_line = attention:block(running, prefix, position)
    if attended is nil then
      return nil, attend_line
    end
    local blocked, block_line = feed_forward:block(
      summed(running, attended), prefix
    )
    if blocked is nil then
      return nil, block_line
    end
    running = blocked
  end

  local values, score_line = feed_forward:scores(running)
  if values is nil then
    return nil, score_line
  end

  local best = 0
  local largest = values[1]
  for index = 2, #values do
    if values[index] > largest then
      largest = values[index]
      best = index - 1
    end
  end
  return best
end

return compute
