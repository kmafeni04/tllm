local ffi = require("ffi")
local fs = require("fs")
local clock = require("tllm.clock")
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
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;

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
size_t ggml_graph_overhead_custom(size_t size, bool grads);
ggml_context *ggml_init(struct ggml_init_params params);
void ggml_free(ggml_context *ctx);
ggml_tensor *ggml_new_tensor_2d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1);
ggml_tensor *ggml_new_tensor_4d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
ggml_tensor *ggml_new_tensor_3d(ggml_context *ctx, int type, int64_t ne0, int64_t ne1, int64_t ne2);
ggml_tensor *ggml_view_2d(ggml_context *ctx, ggml_tensor *a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
ggml_tensor *ggml_view_3d(ggml_context *ctx, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
ggml_tensor *ggml_cpy(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
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
ggml_tensor *ggml_transpose(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_cast(ggml_context *ctx, ggml_tensor *a, int type);
ggml_tensor *ggml_cont(ggml_context *ctx, ggml_tensor *a);
ggml_tensor *ggml_soft_max_ext(
  ggml_context *ctx, ggml_tensor *a, ggml_tensor *mask, float scale,
  float max_bias);
size_t ggml_quantize_chunk(
  int type, const float *src, void *dst, int64_t start, int64_t nrows,
  int64_t n_per_row, const float *imatrix);
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
ggml_cgraph *ggml_new_graph_custom(ggml_context *ctx, size_t size, bool grads);
void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor);
void *ggml_get_data(const ggml_tensor *tensor);
size_t ggml_nbytes(const ggml_tensor *tensor);
size_t ggml_used_mem(const ggml_context *ctx);
ggml_tensor *ggml_get_first_tensor(const ggml_context *ctx);
ggml_tensor *ggml_get_next_tensor(const ggml_context *ctx, ggml_tensor *tensor);
void ggml_backend_free(ggml_backend *backend);
int ggml_backend_graph_compute(ggml_backend *backend, ggml_cgraph *cgraph);
ggml_backend_dev_t ggml_backend_dev_by_type(int type);
void ggml_backend_load_all_from_path(const char *dir_path);
ggml_backend *ggml_backend_dev_init(ggml_backend_dev_t device, const char *params);
const char *ggml_backend_dev_name(ggml_backend_dev_t device);
void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t *free, size_t *total);
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ggml_context *ctx, ggml_backend *backend);
size_t ggml_backend_alloc_ctx_tensors_from_buft_size(ggml_context *ctx, ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend *backend);
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void *ptr, size_t size);
ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend *backend, size_t size);
void *ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);
size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer);
size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer);
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const ggml_tensor *tensor);
int ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ggml_tensor *tensor, void *addr);
int ggml_backend_view_init(ggml_tensor *tensor);
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer);
void ggml_backend_tensor_set(ggml_tensor *tensor, const void *data, size_t offset, size_t size);
void ggml_backend_tensor_get(const ggml_tensor *tensor, void *data, size_t offset, size_t size);

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
  ggml_graph_overhead_custom: function(integer, boolean): integer
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
  ggml_transpose: function(ffi.CData, ffi.CData): ffi.CData
  ggml_cast: function(ffi.CData, ffi.CData, integer): ffi.CData
  ggml_cont: function(ffi.CData, ffi.CData): ffi.CData
  ggml_soft_max_ext: function(
    ffi.CData, ffi.CData, ffi.CData | nil, number, number
  ): ffi.CData
  ggml_quantize_chunk: function(
    integer, ffi.CData, ffi.CData, integer, integer, integer, ffi.CData | nil
  ): integer
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
  ggml_new_graph_custom: function(ffi.CData, integer, boolean): ffi.CData
  ggml_build_forward_expand: function(ffi.CData, ffi.CData)
  ggml_get_data: function(ffi.CData): ffi.CData
  ggml_nbytes: function(ffi.CData): integer
  ggml_used_mem: function(ffi.CData): integer
  ggml_get_first_tensor: function(ffi.CData): ffi.CData
  ggml_get_next_tensor: function(ffi.CData, ffi.CData): ffi.CData
  ggml_cpy: function(ffi.CData, ffi.CData, ffi.CData): ffi.CData
  ggml_view_3d: function(
    ffi.CData, ffi.CData, integer, integer, integer, integer, integer, integer
  ): ffi.CData
  ggml_backend_free: function(ffi.CData)
  ggml_backend_graph_compute: function(ffi.CData, ffi.CData): integer
  ggml_backend_dev_by_type: function(integer): ffi.CData
  ggml_backend_load_all_from_path: function(string)
  ggml_backend_dev_init: function(ffi.CData, ffi.CData | nil): ffi.CData
  ggml_backend_dev_name: function(ffi.CData): ffi.CData
  ggml_backend_dev_memory: function(ffi.CData, ffi.CData, ffi.CData)
  ggml_backend_alloc_ctx_tensors: function(ffi.CData, ffi.CData): ffi.CData
  ggml_backend_alloc_ctx_tensors_from_buft_size: function(
    ffi.CData, ffi.CData
  ): integer
  ggml_backend_get_default_buffer_type: function(ffi.CData): ffi.CData
  ggml_backend_cpu_buffer_type: function(): ffi.CData
  ggml_backend_cpu_buffer_from_ptr: function(ffi.CData, integer): ffi.CData
  ggml_backend_alloc_buffer: function(ffi.CData, integer): ffi.CData
  ggml_backend_buffer_get_base: function(ffi.CData): ffi.CData
  ggml_backend_buffer_get_size: function(ffi.CData): integer
  ggml_backend_buffer_get_alignment: function(ffi.CData): integer
  ggml_backend_buffer_get_alloc_size: function(ffi.CData, ffi.CData): integer
  ggml_backend_tensor_alloc: function(ffi.CData, ffi.CData, ffi.CData): integer
  ggml_backend_view_init: function(ffi.CData): integer
  ggml_backend_buffer_free: function(ffi.CData)
  ggml_backend_tensor_set: function(ffi.CData, ffi.CData, integer, integer)
  ggml_backend_tensor_get: function(ffi.CData, ffi.CData, integer, integer)
end

local record Libc
  aligned_alloc: function(integer, integer): ffi.CData
  free: function(ffi.CData)
end

local record compute
  enum CacheType
    "f32"
    "f16"
    "q8_0"
    "q4_0"
  end

  record Backend
    path: string
    version: string
    device: string | nil
    memory: function(Backend): integer | nil, integer | nil
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
    close: function(Maths)
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
    cache_type: CacheType
  end

  record Row
    sequence: string
    hidden: {number}
    position: integer
  end

  record Input
    sequence: string
    token: integer
    position: integer
  end

  record Attention
    block: function(Attention, {Row}, string): {{number}} | nil, string | nil
    cache_bytes: function(Attention, string, string): integer | nil
    reserve: function(Attention, string, integer, {string}): string | nil
    release: function(Attention, string)
    snapshot: function(
      Attention, string, string, integer
    ): string | nil, string | nil
    restore: function(Attention, string, string, integer, string): string | nil
    close: function(Attention)
  end

  record Measure
    passes: integer
    tokens: integer
    compute: number
    sampling: number
    ready: number
    routed: integer
    routed_bytes: integer
    graphs: integer
  end

  record Routing
    layer: integer
    token: integer
    experts: {integer}
  end

  record Watch
    layer: integer
    token: integer
    measure: Measure
    sink: (function(Routing) | nil)
  end

  record FeedForward
    block: function(
      FeedForward, {Row}, string, watch?: {Watch}
    ): {{number}} | nil, string | nil
    scores: function(FeedForward, {{number}}): {{number}} | nil, string | nil
    close: function(FeedForward)
  end

  record LinearAttention
    block: function(
      LinearAttention, {Row}, string
    ): {{number}} | nil, string | nil
    release: function(LinearAttention, string)
    snapshot: function(
      LinearAttention, string, string, integer
    ): string | nil, string | nil
    restore: function(
      LinearAttention, string, string, integer, string
    ): string | nil
    close: function(LinearAttention)
  end

  record Model
    positions: integer
    sequences: integer
    ubatch: integer
    measure: Measure
    routing_sink: (function(Routing) | nil)
    forward: function(Model, integer, integer): {number} | nil, string | nil
    reserve: function(Model, string, integer): string | nil
    release: function(Model, string)
    cache_bytes: function(Model, string, string): integer | nil
    capacity: function(Model, string): integer | nil
    pass: function(Model, {Input}): {{number}} | nil, string | nil
    save: function(Model, string, {integer}, string): string | nil
    load: function(Model, string, string): {integer} | nil, string | nil
    close: function(Model)
  end

  load: function(path?: string, offload?: boolean): Backend | nil, string | nil
  unload: function(Backend)
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
    Backend,
    weights.Handle,
    integer | nil,
    CacheType | nil,
    sequences?: integer,
    ubatch?: integer
  ): Model | nil, string | nil
end

local type Tensor = weights.Tensor

local record TensorView
  data: ffi.CData
  view_src: ffi.CData
end

-- a tensor a graph answers with and the numbers it holds, a count of zero
-- being a node the graph runs and answers with no number
local record Output
  tensor: ffi.CData
  count: integer
end

-- one graph a handle builds, for the measuring pass that sizes its arena; the
-- kind names the graphs a block's device buffer has to hold
local record Build
  kind: string
  build: function(Graph): {Output}
end

-- the device copy of one model's weights: one buffer, its descriptors and the
-- backend the graphs over them are computed on
local record Copy
  lib: Library
  backend: ffi.CData
  ctx: ffi.CData
  arena: ffi.CData
  buffer: ffi.CData
  tensors: {string: ffi.CData}
end

local record Loaded
  lib: Library
  device: ffi.CData | nil
  offload: boolean
end

-- tensors written into host memory at once, or into the device buffer of the
-- backend the graph runs on once the buffer exists; `measure` builds a graph
-- for its size only, writing no number into any tensor
local record Graph
  lib: Library
  ctx: ffi.CData
  device: ffi.CData | nil
  pending: {function()}
  measure: string | nil
  label: string | nil
  -- a context that places no number of its own has no tensor to write into
  -- until the graph's buffer holds them, so every input waits in `pending`
  deferred: boolean
end

-- the one host arena a handle writes its graphs over, and the device buffers
-- its graphs run in, one per block, each sized from the largest graph built in
-- it: `host` is re-initialised by every graph and freed only by the handle
local record Pool
  lib: Library
  host: ffi.CData | nil
  bytes: integer
  -- the rows its graphs were measured at, so a call holding more rows than that
  -- is given an arena sized from itself rather than one that cannot hold it
  rows: integer
  -- the tensors the largest of those graphs made, so the next sizing counts a
  -- row's nodes from a measurement rather than by building one to count
  tensors: integer
  -- the nodes the graph object this arena holds room for
  nodes: integer
  backend: ffi.CData | nil
  cpu: ffi.CData | nil
  device_bytes: {string: integer}
  devices: {string: ffi.CData}
  held: {string: integer}
end

-- the arena a run of blocks builds its graphs in, held through one box so a
-- model that grows its arena for a wider pass gives the grown arena to every
-- block of its run: the pool inside is re-initialised by every graph and
-- freed only by the handle that holds the box
local record Box
  pool: Pool | nil
  owned: boolean
end

local record Holder
  hold: function()
  close: function()
end

local record Mapped
  lib: Library
  tensors: {string: Tensor}
  descriptors: {string: ffi.CData}
  device: Copy | nil
  arena: ffi.CData | nil
  ctx: ffi.CData | nil
  hold: function()
  close: function()
end

-- one prefix's keys and values: two buffers of the geometry's cache type in
-- host memory, the size they hold and the positions they took
local record Cache
  keys: ffi.CData
  values: ffi.CData
  bytes: integer
  count: integer
end

local record AttentionState
  mapped: Mapped
  geometry: compute.Geometry
  caches: {string: {string: Cache}}
  capacities: {string: integer}
  pool: Box
  measure: compute.Measure | nil
  owner: boolean
end

local record FeedForwardState
  mapped: Mapped
  geometry: compute.Geometry
  pool: Box
  measure: compute.Measure | nil
  owner: boolean
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
  prefixes: {string: {string: LinearKept}}
  pool: Box
  measure: compute.Measure | nil
  owner: boolean
end

local record ModelPlan
  mapped: Mapped
  kinds: {boolean}
  count: integer
  width: integer
  positions: integer
  sequences: integer
  pool: Box
  sized: function(integer): {Build}
  measure: compute.Measure
  attention: compute.Attention | nil
  linear: compute.LinearAttention | nil
  feed_forward: compute.FeedForward
end

local record SequenceState
  capacity: integer
  taken: integer
end

local record ModelState
  mapped: Mapped
  kinds: {boolean}
  count: integer
  width: integer
  attention: compute.Attention | nil
  linear: compute.LinearAttention | nil
  feed_forward: compute.FeedForward
  sequences: {string: SequenceState}
  limit: integer
  positions: integer
  fingerprint: integer
  cache_code: integer
  full_prefixes: {string}
  linear_prefixes: {string}
  pool: Box
  sized: function(integer): {Build}
  measure: compute.Measure
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
-- the build struct ggml_tensor is mirrored from; another version's layout is
-- not what this file reads and writes
local EXPECTED_VERSION = "0.29.0"
local F32_BYTES: integer = 4
local I32_BYTES: integer = 4
-- GGML_MEM_ALIGN in the pinned ggml source
local ARENA_ALIGN: integer = 16
-- TENSOR_ALIGNMENT of the CPU backend's buffers in the pinned ggml source,
-- which a graph's numbers are laid out on
local TENSOR_ALIGN: integer = 64
-- room for the context and graph objects ggml aligns past the tensors
local ARENA_SLACK: integer = 8192
-- the tensors a measuring context is opened with before anything is measured:
-- a graph's nodes are counted per row of the batch, with room to spare
local TENSORS_PER_ROW: integer = 256
local TENSOR_SPARE: integer = 4096
-- the rows a pass carries when nothing names a bound of its own
local DEFAULT_UBATCH: integer = 512
local GGML_TYPE_F32: integer = 0
local GGML_TYPE_F16: integer = 1
local GGML_TYPE_Q4_0: integer = 2
local GGML_TYPE_Q8_0: integer = 8
local GGML_TYPE_I32: integer = 26
local GGML_STATUS_SUCCESS: integer = 0
local GGML_ROPE_TYPE_NEOX: integer = 2
-- GGML_BACKEND_DEVICE_TYPE_CPU and _GPU in the pinned ggml source
local GGML_DEVICE_CPU: integer = 0
local GGML_DEVICE_GPU: integer = 1

-- what a cached number costs in each type the cache may hold: the bytes of
-- one block and how many numbers a block carries
local record CacheKind
  code: integer
  block_elements: integer
  block_bytes: integer
end

local CACHE_KINDS: {string: CacheKind} = {
  f32 = {code = GGML_TYPE_F32, block_elements = 1, block_bytes = 4},
  f16 = {code = GGML_TYPE_F16, block_elements = 1, block_bytes = 2},
  q8_0 = {code = GGML_TYPE_Q8_0, block_elements = 32, block_bytes = 34},
  q4_0 = {code = GGML_TYPE_Q4_0, block_elements = 32, block_bytes = 18},
}

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
local LOADED: {compute.Backend: Loaded} = {}
local COPIES: {weights.Handle: {compute.Backend: Copy}} = {}
local GUARDED: {weights.Handle: boolean} = {}
-- the directories whose backend modules have been found already
local SEARCHED: {string: boolean} = {}

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

-- tensors is the number of tensors the operation builds plus spares, and a
-- graph of that many nodes is reserved beside them: an arena one tensor short
-- makes ggml dereference NULL instead of returning a refusal
local function arena_for(
  lib: Library, tensors: integer, data_bytes: integer
): integer
  local overhead = tonumber(lib.ggml_tensor_overhead()) as integer
  local graph = tonumber(lib.ggml_graph_overhead()) as integer
  local wide = tonumber(lib.ggml_graph_overhead_custom(tensors, false))
    as integer
  if wide > graph then
    graph = wide
  end
  local objects = overhead * (tensors + 8)
  local bytes = graph + objects + data_bytes + ARENA_SLACK
  return round_up(bytes, ARENA_ALIGN)
end

local function arena_params(size: integer): Params
  local params = ffi.new("struct ggml_init_params") as Params
  params.mem_size = size
  return params
end

-- the parameters a graph's context is opened with: the pool's arena, and a
-- context that places no number of its own when the graph runs on a device
local function pool_params(
  size: integer, block: ffi.CData, device: boolean
): Params
  local params = arena_params(size)
  params.mem_buffer = block
  params.no_alloc = device
  return params
end

-- a block the collector cannot reach on its own, given back with the cache
-- that holds it
local function finalised(block: ffi.CData): ffi.CData
  return ffi.gc(block, function(held: ffi.CData) libc.free(held) end)
end

-- ggml places the context inside the buffer it is given, so the buffer and the
-- context go back together, the context first
local function arena_finalised(
  lib: Library, block: ffi.CData, ctx: ffi.CData
): ffi.CData
  return ffi.gc(
    block, function(held: ffi.CData) lib.ggml_free(ctx); libc.free(held) end
  )
end

-- the backends are modules loaded from the directory the library sits in, so a
-- machine without a driver still opens libggml and simply finds no device
local function find_backends(lib: Library, location: string)
  local directory = location:match("^(.*)/") or "."
  if SEARCHED[directory] then
    return
  end
  SEARCHED[directory] = true
  lib.ggml_backend_load_all_from_path(directory)
end

local function cpu_backend(lib: Library): ffi.CData | nil
  local device = lib.ggml_backend_dev_by_type(GGML_DEVICE_CPU)
  if device is nil then
    return nil
  end
  return lib.ggml_backend_dev_init(device, nil)
end

local function write_floats(cells: Floats, values: {number})
  for position, value in ipairs(values) do
    cells[position - 1] = value
  end
end

local function read_floats(
  lib: Library, tensor: ffi.CData, count: integer, device: ffi.CData | nil
): {number}
  local cells: Floats
  if device is nil then
    cells = ffi.cast("float*", lib.ggml_get_data(tensor)) as Floats
  else
    local buffer = ffi.new("float[?]", count) as ffi.CData
    lib.ggml_backend_tensor_get(tensor, buffer, 0, count * F32_BYTES)
    cells = buffer as Floats
  end
  local result: {number} = {}
  for position = 0, count - 1 do
    result[position + 1] = cells[position]
  end
  return result
end

local function read_bytes(
  lib: Library,
  tensor: ffi.CData,
  destination: ffi.CData,
  bytes: integer,
  device: ffi.CData | nil,
  at?: integer
)
  local offset = at or 0
  if device is nil then
    local base = ffi.cast("intptr_t", lib.ggml_get_data(tensor)) as integer
    ffi.copy(destination, ffi.cast("void *", base + offset) as ffi.CData, bytes)
  else
    lib.ggml_backend_tensor_get(tensor, destination, offset, bytes)
  end
end

local function tensor_in(
  lib: Library, ctx: ffi.CData, dims: {integer}, code: integer
): ffi.CData
  local ne: {integer} = {1, 1, 1, 1}
  for index, size in ipairs(dims) do
    ne[index] = size
  end
  return lib.ggml_new_tensor_4d(ctx, code, ne[1], ne[2], ne[3], ne[4])
end

-- what a build raises when its context has no room for the tensor it asked for
local record ArenaError
  arena: string | nil
end

-- the line a short arena raises, naming the block or the pass's graph it
-- stopped and the numbers the tensor asked for
local function arena_line(graph: Graph, detail: string): string
  local label = graph.label
  if label is nil then
    return detail
  end
  return label .. ": " .. detail
end

-- ggml answers a context with no room for one more tensor with NULL, and every
-- later call in the same build would dereference it, so a short arena stops the
-- build here and the operation's pcall turns the raise into one line
local function graph_tensor(
  graph: Graph, dims: {integer}, code: integer
): ffi.CData
  local tensor = tensor_in(graph.lib, graph.ctx, dims, code)
  if tensor is nil then
    error(
      {
        arena = arena_line(
          graph, "no room for " .. tostring(dims[1] or 0) .. " numbers"
        ),
      },
      0
    )
  end
  return tensor
end

-- the result of a ggml operation: a graph of a block or of a pass checks every
-- one of them, so a NULL is a line and never a pointer a later node reads
local function node(graph: Graph, tensor: ffi.CData): ffi.CData
  if tensor is nil then
    error({arena = arena_line(graph, "no room for a node's result")}, 0)
  end
  return tensor
end

-- a tensor of f32 numbers, written into host memory now or into the device
-- buffer once the graph's buffer exists
local function vector_of(graph: Graph, values: {number}): ffi.CData
  local tensor = graph_tensor(graph, {#values}, GGML_TYPE_F32)
  if graph.measure ~= nil then
    return tensor
  end
  local device = graph.device
  if device is nil and not graph.deferred then
    write_floats(
      ffi.cast("float*", graph.lib.ggml_get_data(tensor)) as Floats, values
    )
  else
    graph.pending[#graph.pending + 1] = function()
      local cells = ffi.new("float[?]", #values) as Floats
      write_floats(cells, values)
      graph.lib.ggml_backend_tensor_set(
        tensor, cells as ffi.CData, 0, #values * F32_BYTES
      )
    end
  end
  return tensor
end

local function scalar_of(graph: Graph, value: integer): ffi.CData
  local tensor = graph_tensor(graph, {1}, GGML_TYPE_I32)
  if graph.measure ~= nil then
    return tensor
  end
  local device = graph.device
  if device is nil and not graph.deferred then
    local cells = ffi.cast("int32_t*", graph.lib.ggml_get_data(tensor)) as Ints
    cells[0] = value
  else
    graph.pending[#graph.pending + 1] = function()
      local cells = ffi.new("int32_t[1]") as Ints
      cells[0] = value
      graph.lib.ggml_backend_tensor_set(
        tensor, cells as ffi.CData, 0, I32_BYTES
      )
    end
  end
  return tensor
end

-- a tensor of i32 numbers, one per row, written like any other input
local function ints_of(graph: Graph, values: {integer}): ffi.CData
  local tensor = graph_tensor(graph, {#values}, GGML_TYPE_I32)
  if graph.measure ~= nil then
    return tensor
  end
  local device = graph.device
  if device is nil and not graph.deferred then
    local cells = ffi.cast("int32_t*", graph.lib.ggml_get_data(tensor)) as Ints
    for position, value in ipairs(values) do
      cells[position - 1] = value
    end
  else
    graph.pending[#graph.pending + 1] = function()
      local cells = ffi.new("int32_t[?]", #values) as Ints
      for position, value in ipairs(values) do
        cells[position - 1] = value
      end
      graph.lib.ggml_backend_tensor_set(
        tensor, cells as ffi.CData, 0, #values * I32_BYTES
      )
    end
  end
  return tensor
end

-- a tensor ggml reads out of memory the caller already holds: a slice of the
-- mapped file, or a history buffer compute keeps
local function host_tensor(
  graph: Graph,
  dims: {integer},
  code: integer,
  location: ffi.CData,
  bytes: integer
): ffi.CData
  local tensor = graph_tensor(graph, dims, code)
  if graph.measure == "device" then
    return tensor
  end
  local device = graph.device
  if device is nil then
    (tensor as TensorView).data = location
  else
    graph.pending[#graph.pending + 1] = function()
      graph.lib.ggml_backend_tensor_set(tensor, location, 0, bytes)
    end
  end
  return tensor
end

local function device_of_copy(copy: Copy | nil): ffi.CData | nil
  if copy is nil then
    return nil
  end
  return copy.backend
end

local record Session
  lib: Library
  copy: Copy | nil
  pool: Pool | nil
  arena: ffi.CData
  ctx: ffi.CData
  graph: Graph
  backend: ffi.CData | nil
  buffer: ffi.CData | nil
end

-- the one arena a handle builds its graphs in, and the device buffers its
-- blocks' graphs run in: the arena is allocated here and freed by the handle
local function pool_new(
  lib: Library,
  copy: Copy | nil,
  host_bytes: integer,
  device_bytes: {string: integer},
  rows: integer,
  tensors: integer,
  nodes: integer
): Pool | nil
  -- the numbers a graph keeps sit in a buffer after its descriptors, so the
  -- arena starts and ends on the alignment that buffer needs
  local size = round_up(host_bytes + ARENA_SLACK, TENSOR_ALIGN)
  local block = libc.aligned_alloc(TENSOR_ALIGN, size)
  if block is nil then
    return nil
  end
  return {
    lib = lib,
    host = finalised(block),
    bytes = size,
    rows = rows,
    tensors = tensors,
    nodes = nodes,
    backend = device_of_copy(copy),
    device_bytes = device_bytes,
    devices = {},
    held = {},
  } as Pool
end

local function pool_free(pool: Pool)
  local host = pool.host
  if not (host is nil) then
    pool.host = nil
    ffi.gc(host, nil)
    libc.free(host)
  end
  for prefix, buffer in pairs(pool.devices) do
    pool.lib.ggml_backend_buffer_free(buffer)
    pool.devices[prefix] = nil
    pool.held[prefix] = nil
  end
  pool.device_bytes = {}
  local cpu = pool.cpu
  if not (cpu is nil) then
    pool.cpu = nil
    pool.lib.ggml_backend_free(cpu)
  end
end

-- the device buffer one block's graphs run in, allocated the first time that
-- block builds a graph, grown to the largest graph that block builds, and
-- reused by every later pass
local function pool_device(
  pool: Pool, prefix: string, kind: string
): ffi.CData | nil
  local backend = pool.backend
  if backend is nil then
    return nil
  end
  local want = round_up(
    (pool.device_bytes[kind] or 0) + ARENA_ALIGN, ARENA_ALIGN
  )
  local buffer = pool.devices[prefix]
  if not (buffer is nil) then
    if (pool.held[prefix] or 0) >= want then
      return buffer
    end
    pool.lib.ggml_backend_buffer_free(buffer)
    pool.devices[prefix] = nil
  end
  local made = pool.lib.ggml_backend_alloc_buffer(backend, want)
  if made is nil then
    return nil
  end
  pool.devices[prefix] = made
  pool.held[prefix] = want
  return made
end

local function pool_backend(pool: Pool): ffi.CData | nil
  local device = pool.backend
  if not (device is nil) then
    return device
  end
  local held = pool.cpu
  if not (held is nil) then
    return held
  end
  local made = cpu_backend(pool.lib)
  pool.cpu = made
  return made
end

-- a context over the pool's arena, holding no number of its own when the
-- weights are in device memory
local function pooled(
  pool: Pool, prefix: string, kind: string
): Session | nil, string | nil
  local lib = pool.lib
  local block = pool.host
  if block is nil then
    return nil, "no arena"
  end
  local device = pool.backend
  local ctx = lib.ggml_init(pool_params(pool.bytes, block, true))
  if ctx is nil then
    return nil, "no context of " .. pool.bytes .. " bytes"
  end
  local buffer: ffi.CData | nil = nil
  if not (device is nil) then
    buffer = pool_device(pool, prefix, kind)
    if buffer is nil then
      lib.ggml_free(ctx)
      return nil, "no device buffer"
    end
  end
  return {
    lib = lib,
    pool = pool,
    arena = block,
    ctx = ctx,
    graph = {
      lib = lib,
      ctx = ctx,
      device = device,
      pending = {},
      label = prefix,
      deferred = true,
    },
    backend = device,
    buffer = buffer,
  } as Session
end

-- hands the graph's own tensors the bytes of the block's device buffer, in the
-- order ggml made them, and stops with a line where the buffer is one short
local function placed(session: Session): string | nil
  local lib = session.lib
  local buffer = session.buffer
  if buffer is nil then
    return nil
  end
  local alignment = tonumber(lib.ggml_backend_buffer_get_alignment(buffer))
    as integer
  local size = tonumber(lib.ggml_backend_buffer_get_size(buffer)) as integer
  local base = lib.ggml_backend_buffer_get_base(buffer)
  local offset: integer = 0
  local tensor = lib.ggml_get_first_tensor(session.ctx)
  while not (tensor is nil) do
    local held = tensor as TensorView
    -- a tensor whose numbers the caller already holds keeps them, and a view of
    -- one keeps the numbers its source holds
    if held.data == nil then
      if held.view_src == nil then
        local bytes = round_up(
          tonumber(lib.ggml_backend_buffer_get_alloc_size(buffer, tensor))
            as integer,
          alignment
        )
        if offset + bytes > size then
          return arena_line(
            session.graph, "no room for " .. bytes .. " bytes of a node"
          )
        end
        local address = ffi.cast("intptr_t", base) as integer
        lib.ggml_backend_tensor_alloc(
          buffer, tensor, ffi.cast("void *", address + offset) as ffi.CData
        )
        offset = offset + bytes
      else
        lib.ggml_backend_view_init(tensor)
      end
    end
    tensor = lib.ggml_get_next_tensor(session.ctx, tensor)
  end
  return nil
end

local function opened(
  mapped: Mapped, tensors: integer, data_bytes: integer
): Session | nil, string | nil
  local lib = mapped.lib
  local copy = mapped.device
  -- with a device copy the only host memory a graph needs is its arena
  local host_bytes = data_bytes
  if not (copy is nil) then
    host_bytes = 0
  end

  local arena_size = arena_for(lib, tensors, host_bytes)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena is nil then
    return nil, "no context of " .. arena_size .. " bytes"
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  if not (copy is nil) then
    params.no_alloc = true
  end
  local ctx = lib.ggml_init(params)
  if ctx is nil then
    libc.free(arena)
    return nil, "no context of " .. arena_size .. " bytes"
  end

  return {
    lib = lib,
    copy = copy,
    arena = arena,
    ctx = ctx,
    graph = {
      lib = lib, ctx = ctx, device = device_of_copy(copy), pending = {}
    },
  } as Session
end

-- the graph's own tensors are allocated here, in device memory when the
-- weights are there, so that no tensor of the graph holds a host pointer
local function computed(
  session: Session, operation: string, cgraph: ffi.CData
): string | nil
  local lib = session.lib
  local pool = session.pool
  if not (pool is nil) then
    local engine = pool_backend(pool)
    if engine is nil then
      return operation .. ": no backend"
    end
    session.backend = engine
    local device_backend = pool.backend
    if device_backend is nil then
      local used = round_up(
        tonumber(lib.ggml_used_mem(session.ctx)) as integer, TENSOR_ALIGN
      )
      local held = session.arena
      if used >= pool.bytes then
        return operation .. ": no room for the graph's numbers"
      end
      local base = ffi.cast("intptr_t", held) as integer
      session.buffer = lib.ggml_backend_cpu_buffer_from_ptr(
        ffi.cast("void *", base + used) as ffi.CData, pool.bytes - used
      )
      local made_buffer = session.buffer
      if made_buffer is nil then
        return operation .. ": no room for the graph's numbers"
      end
    end
    local placed_line = placed(session)
    if not (placed_line is nil) then
      return operation .. ": " .. placed_line
    end
    for _, write in ipairs(session.graph.pending) do
      write()
    end
    local status = lib.ggml_backend_graph_compute(engine, cgraph)
    if status ~= GGML_STATUS_SUCCESS then
      return operation .. ": compute returned " .. status
    end
    return nil
  end

  local copy = session.copy
  if copy is nil then
    session.backend = cpu_backend(lib)
  else
    local buffer = lib.ggml_backend_alloc_ctx_tensors(session.ctx, copy.backend)
    if buffer is nil then
      return operation .. ": no device buffer"
    end
    session.buffer = buffer
    session.backend = copy.backend
  end

  local backend = session.backend
  if backend is nil then
    return operation .. ": no backend"
  end

  for _, write in ipairs(session.graph.pending) do
    write()
  end

  local status = lib.ggml_backend_graph_compute(backend, cgraph)
  if status ~= GGML_STATUS_SUCCESS then
    return operation .. ": compute returned " .. status
  end
  return nil
end

local function closed(session: Session)
  local lib = session.lib
  local pool = session.pool
  if not (pool is nil) then
    -- the pool's arena and device buffer outlive the graph that used them, and
    -- the buffer a graph's numbers were given was this call's own
    local buffer = session.buffer
    local on_device = pool.backend
    if on_device is nil and not (buffer is nil) then
      lib.ggml_backend_buffer_free(buffer)
    end
    lib.ggml_free(session.ctx)
    return
  end
  local buffer = session.buffer
  if not (buffer is nil) then
    lib.ggml_backend_buffer_free(buffer)
  end
  local backend = session.backend
  local copy = session.copy
  if copy is nil and not (backend is nil) then
    lib.ggml_backend_free(backend)
  end
  lib.ggml_free(session.ctx)
  libc.free(session.arena)
end

local function gathered(
  mapped: Mapped,
  operation: string,
  tensors: integer,
  data_bytes: integer,
  build: function(Graph): {Output}
): {{number}} | nil, string | nil
  local session, open_line = opened(mapped, tensors, data_bytes)
  if session is nil then
    return nil, operation .. ": " .. (open_line as string)
  end

  local ok, outputs = pcall(build, session.graph)
  if not ok then
    closed(session)
    local reason = outputs
    if type(reason) == "table" then
      local held = reason as ArenaError
      local line = held.arena
      if line is string then
        return nil, operation .. ": " .. line
      end
    end
    error(reason, 0)
  end
  local made = outputs as {Output}

  local cgraph = session.lib.ggml_new_graph(session.ctx)
  for _, output in ipairs(made) do
    session.lib.ggml_build_forward_expand(cgraph, output.tensor)
  end

  local failure = computed(session, operation, cgraph)
  local planes: {{number}} = {}
  if failure is nil then
    local device = device_of_copy(session.copy)
    for index, output in ipairs(made) do
      planes[index] = read_floats(
        session.lib, output.tensor, output.count, device
      )
    end
  end
  closed(session)

  if failure then
    return nil, failure
  end
  return planes
end

local function run(
  mapped: Mapped,
  operation: string,
  tensors: integer,
  data_bytes: integer,
  count: integer,
  build: function(Graph): ffi.CData
): {number} | nil, string | nil
  local planes, failure = gathered(
    mapped,
    operation,
    tensors,
    data_bytes,
    function(graph: Graph): {Output}
      return {{tensor = build(graph), count = count}}
    end
  )
  if planes is nil then
    return nil, failure
  end
  return planes[1], nil
end

-- the tensors one build makes, walked out of the context it built them in:
-- a row's nodes repeat for every row of the batch, so the count of one row's
-- build tells how many a batch of any size asks the arena for
-- the bytes one graph asks for, measured by building it in a context that
-- places no number: the arena holds the descriptors and the graph object, and
-- the numbers go into a buffer whose bytes this asks ggml for
local function footprint_of(
  lib: Library,
  copy: Copy | nil,
  device: boolean,
  build: function(Graph): {Output},
  tensors: integer
): integer, integer, integer
  local buft: ffi.CData = lib.ggml_backend_cpu_buffer_type()
  local mode = "host"
  if device then
    local engine = device_of_copy(copy) as ffi.CData
    mode = "device"
    buft = lib.ggml_backend_get_default_buffer_type(engine)
  end

  local size = arena_for(lib, tensors, 0)
  local block = libc.aligned_alloc(ARENA_ALIGN, size)
  if block is nil then
    return 0, 0, 0
  end
  local ctx = lib.ggml_init(pool_params(size, block, true))
  if ctx is nil then
    libc.free(block)
    return 0, 0, 0
  end
  local graph: Graph = {
    lib = lib,
    ctx = ctx,
    pending = {},
    measure = mode,
  }
  if not pcall(build, graph) then
    lib.ggml_free(ctx)
    libc.free(block)
    return 0, 0, 0
  end

  -- the graph object a pass's build makes comes out of the same arena, and
  -- the tensors it left behind are what the arena is then sized from
  lib.ggml_new_graph_custom(ctx, tensors, false)
  local used = tonumber(lib.ggml_used_mem(ctx)) as integer
  local numbers = tonumber(
    lib.ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)
  ) as integer
  local made = 0
  local tensor = lib.ggml_get_first_tensor(ctx)
  while not (tensor is nil) do
    made = made + 1
    tensor = lib.ggml_get_next_tensor(ctx, tensor)
  end
  lib.ggml_free(ctx)
  libc.free(block)
  if device then
    return used, numbers, made
  end
  return used + numbers, 0, made
end

-- the arena and the device buffers a handle holds, each the worst case of one
-- graph: never the sum of a run's blocks
local function sized_pool(
  lib: Library,
  copy: Copy | nil,
  builds: {Build},
  rows: integer,
  per_row: integer | nil,
  floor: integer | nil
): Pool | nil
  -- a graph's nodes repeat for every row it holds, so one measurement carries
  -- to the next size: a measuring context that runs short stops the process
  -- rather than hand back NULL, so its room is never grown by trial and error
  local tensors = TENSOR_SPARE + rows * TENSORS_PER_ROW
  if not (per_row is nil) then
    tensors = per_row * rows + TENSOR_SPARE
  end

  local host: integer = 0
  local device: {string: integer} = {}
  local made = 0
  for _, entry in ipairs(builds) do
    local host_bytes, device_bytes, count = footprint_of(
      lib, copy, false, entry.build, tensors
    )
    if host_bytes > host then
      host = host_bytes
    end
    if count > made then
      made = count
    end
    if not (copy is nil) then
      host_bytes, device_bytes, count = footprint_of(
        lib, copy, true, entry.build, tensors
      )
      if host_bytes > host then
        host = host_bytes
      end
      if device_bytes > (device[entry.kind] or 0) then
        device[entry.kind] = device_bytes
      end
      if count > made then
        made = count
      end
    end
  end
  if not (floor is nil) and floor > host then
    host = floor
  end
  if host == 0 then
    return nil
  end
  return pool_new(lib, copy, host, device, rows, made, tensors)
end

-- a graph built in the pool's arena and computed, its tensors readable until
-- the next graph takes the arena over
local record Computed
  lib: Library
  device: ffi.CData | nil
  outputs: {Output}
  session: Session
end

-- one graph of one block, or of the pass, in the arena the pool holds: the
-- arena is written over for every graph, so nothing is read through a tensor
-- of an earlier one
local function graph_run(
  pool: Pool,
  prefix: string,
  kind: string,
  operation: string,
  measure: compute.Measure | nil,
  build: function(Graph): {Output}
): Computed | nil, string | nil
  local session, open_line = pooled(pool, prefix, kind)
  if session is nil then
    return nil, operation .. ": " .. (open_line as string)
  end

  local ok, made = pcall(build, session.graph)
  if not ok then
    closed(session)
    local reason = made
    if type(reason) == "table" then
      local held = reason as ArenaError
      local line = held.arena
      if line is string then
        return nil, operation .. ": " .. line
      end
    end
    error(reason, 0)
  end
  local outputs = made as {Output}
  local cgraph = session.lib.ggml_new_graph_custom(
    session.ctx, pool.nodes, false
  )
  for _, output in ipairs(outputs) do
    session.lib.ggml_build_forward_expand(cgraph, output.tensor)
  end

  local failure = computed(session, operation, cgraph)
  if failure then
    closed(session)
    return nil, failure
  end
  if not (measure is nil) then
    measure.graphs = measure.graphs + 1
  end
  return {
    lib = session.lib,
    device = pool.backend,
    outputs = outputs,
    session = session,
  } as Computed
end

local function graph_closed(made: Computed)
  closed(made.session)
end

-- the numbers of one output tensor, read before the next graph takes the arena
local function graph_numbers(made: Computed, output: Output): {number}
  return read_floats(made.lib, output.tensor, output.count, made.device)
end

-- the numbers of every output the build asked to read, in that order
local function graph_planes(made: Computed): {{number}}
  local planes: {{number}} = {}
  for _, output in ipairs(made.outputs) do
    if output.count > 0 then
      planes[#planes + 1] = graph_numbers(made, output)
    end
  end
  return planes
end

local function free_copy(copy: Copy)
  copy.lib.ggml_backend_buffer_free(copy.buffer)
  copy.lib.ggml_free(copy.ctx)
  libc.free(copy.arena)
  copy.lib.ggml_backend_free(copy.backend)
end

-- every tensor of the handle goes into the device once, and the copy stays
-- with the handle so each later entry point over it reuses it
local function device_copy(loaded: Loaded, handle: weights.Handle): Copy | nil
  local lib = loaded.lib
  local found = loaded.device as ffi.CData
  local backend = lib.ggml_backend_dev_init(found, nil)
  if backend is nil then
    return nil
  end

  local count = 0
  for _ in pairs(handle.tensors) do
    count = count + 1
  end

  local arena_size = arena_for(lib, count, 0)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena is nil then
    lib.ggml_backend_free(backend)
    return nil
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  params.no_alloc = true
  local ctx = lib.ggml_init(params)
  if ctx is nil then
    libc.free(arena)
    lib.ggml_backend_free(backend)
    return nil
  end

  local tensors: {string: ffi.CData} = {}
  for name, tensor in pairs(handle.tensors) do
    tensors[name] = tensor_in(lib, ctx, tensor.dims, tensor.kind.code)
  end

  local buffer = lib.ggml_backend_alloc_ctx_tensors(ctx, backend)
  if buffer is nil then
    lib.ggml_free(ctx)
    libc.free(arena)
    lib.ggml_backend_free(backend)
    return nil
  end

  for name, tensor in pairs(handle.tensors) do
    lib.ggml_backend_tensor_set(tensors[name], tensor.data, 0, tensor.bytes)
  end

  return {
    lib = lib,
    backend = backend,
    ctx = ctx,
    arena = arena,
    buffer = buffer,
    tensors = tensors,
  } as Copy
end

local function release_copies(handle: weights.Handle)
  local held = COPIES[handle]
  if held is nil then
    return
  end
  COPIES[handle] = nil
  for _, copy in pairs(held) do
    free_copy(copy)
  end
end

-- the release a handle already carries gives the device copy back first, once
local function freeing(
  handle: weights.Handle, plain: function(weights.Handle): string | nil
): string | nil
  if handle.mappers > 0 then
    return "release: " .. handle.mappers .. " still use it"
  end
  release_copies(handle)
  -- the handle is a table key only while it holds something, so releasing it
  -- keeps nothing alive
  GUARDED[handle] = nil
  return plain(handle)
end

local function guard_release(handle: weights.Handle)
  if GUARDED[handle] then
    return
  end
  GUARDED[handle] = true
  local plain = handle.release
  handle.release = function(h: weights.Handle): string | nil
    return freeing(h, plain)
  end
end

-- one holder's mark on a handle, taken when the record that reads the mapping
-- exists and given back by its close, so the file outlives every reader
local function holder(handle: weights.Handle): Holder
  local attached = false
  return {
    hold = function()
      if attached then
        return
      end
      attached = true
      weights.attach(handle)
    end,
    close = function()
      if not attached then
        return
      end
      attached = false
      weights.detach(handle)
    end,
  }
end

local function mapped_of(
  backend: compute.Backend,
  loaded: Loaded,
  handle: weights.Handle,
  operation: string
): Mapped | nil, string | nil
  local lib = loaded.lib
  backend.device = nil
  guard_release(handle)
  local found = loaded.device
  if loaded.offload and not (found is nil) then
    local held = COPIES[handle]
    local copy: Copy | nil = nil
    if not (held is nil) then
      copy = held[backend]
    end
    if copy is nil then
      copy = device_copy(loaded, handle)
      if not (copy is nil) then
        if held is nil then
          held = {}
          COPIES[handle] = held
        end
        held[backend] = copy
      end
    end
    if not (copy is nil) then
      backend.device = ffi.string(lib.ggml_backend_dev_name(found as ffi.CData))
      local mark = holder(handle)
      return {
        lib = lib,
        tensors = handle.tensors,
        descriptors = copy.tensors,
        device = copy,
        hold = mark.hold,
        close = mark.close,
      }
    end
  end

  local count = 0
  for _ in pairs(handle.tensors) do
    count = count + 1
  end

  local arena_size = arena_for(lib, count, 0)
  local arena_block = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena_block is nil then
    return nil, operation .. ": no context of " .. arena_size .. " bytes"
  end
  local params = arena_params(arena_size)
  params.mem_buffer = arena_block
  params.no_alloc = true
  local ctx = lib.ggml_init(params)
  if ctx is nil then
    libc.free(arena_block)
    return nil, operation .. ": no context of " .. arena_size .. " bytes"
  end

  -- the collector gives the arena and, with it, the context back once no
  -- record reads either of them
  local arena = arena_finalised(lib, arena_block, ctx)

  local descriptors: {string: ffi.CData} = {}
  for name, tensor in pairs(handle.tensors) do
    local descriptor = tensor_in(lib, ctx, tensor.dims, tensor.kind.code)
    (descriptor as TensorView).data = tensor.data
    descriptors[name] = descriptor
  end

  local mark = holder(handle)
  return {
    lib = lib,
    tensors = handle.tensors,
    descriptors = descriptors,
    arena = arena,
    ctx = ctx,
    hold = mark.hold,
    close = mark.close,
  }
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
  local stride: integer = 0
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
    stride = math.floor(tensor.bytes / dims[3])
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
    mapped,
    "matvec",
    4,
    data_bytes,
    rows,
    function(graph: Graph): ffi.CData
      local weight = mapped.descriptors[name]
      if depth == 3 then
        weight = host_tensor(
          graph, {columns, rows}, tensor.kind.code, location, stride
        )
      end
      return lib.ggml_mul_mat(graph.ctx, weight, vector_of(graph, vector))
    end
  )
end

-- several vectors written as the columns of one f32 tensor, so a weight
-- matrix is read once for the whole batch
local function matrix_of(
  graph: Graph, rows: {{number}}, columns: integer
): ffi.CData
  local tensor = graph_tensor(graph, {columns, #rows}, GGML_TYPE_F32)
  if graph.measure ~= nil then
    return tensor
  end
  local count = columns * #rows
  local device = graph.device
  if device is nil and not graph.deferred then
    local cells = ffi.cast("float*", graph.lib.ggml_get_data(tensor)) as Floats
    local at: integer = 0
    for _, row in ipairs(rows) do
      for position = 1, columns do
        cells[at] = row[position]
        at = at + 1
      end
    end
  else
    graph.pending[#graph.pending + 1] = function()
      local cells = ffi.new("float[?]", count) as Floats
      local at: integer = 0
      for _, row in ipairs(rows) do
        for position = 1, columns do
          cells[at] = row[position]
          at = at + 1
        end
      end
      graph.lib.ggml_backend_tensor_set(
        tensor, cells as ffi.CData, 0, count * F32_BYTES
      )
    end
  end
  return tensor
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
    mapped,
    "row",
    4,
    columns * F32_BYTES + I32_BYTES,
    columns,
    function(graph: Graph): ffi.CData
      return lib.ggml_get_rows(
        graph.ctx, mapped.descriptors[name], scalar_of(graph, id)
      )
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
    mapped,
    "norm",
    6,
    count * 3 * F32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      local scaled = lib.ggml_rms_norm(graph.ctx, vector_of(graph, vector), eps)
      return lib.ggml_mul(graph.ctx, scaled, mapped.descriptors[gain_name])
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
    mapped,
    "rope",
    6,
    count * 2 * F32_BYTES + I32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      local input = vector_of(graph, vector)
      return lib.ggml_rope_ext(
        graph.ctx,
        input,
        scalar_of(graph, position),
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
    mapped,
    "softmax",
    4,
    count * 2 * F32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      return lib.ggml_soft_max(graph.ctx, vector_of(graph, vector))
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
    mapped,
    "add",
    4,
    count * 3 * F32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      local first = vector_of(graph, left)
      return lib.ggml_add(graph.ctx, first, vector_of(graph, right))
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
    mapped,
    "swiglu",
    4,
    count * 3 * F32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      local first = vector_of(graph, gate)
      return lib.ggml_swiglu_split(graph.ctx, first, vector_of(graph, up))
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
    mapped,
    "sigmoid_gate",
    6,
    count * 4 * F32_BYTES,
    count,
    function(graph: Graph): ffi.CData
      local gates = lib.ggml_sigmoid(graph.ctx, vector_of(graph, gate))
      return lib.ggml_mul(graph.ctx, gates, vector_of(graph, values))
    end
  )
end

local function cache_kind_of(name: string): CacheKind | nil
  return CACHE_KINDS[name]
end

-- the bytes one cached row of this type costs
local function cache_row_size(kind: CacheKind, width: integer): integer
  return math.floor(width / kind.block_elements) * kind.block_bytes
end

local function geometry_refusal(geometry: compute.Geometry): string | nil
  local kind = cache_kind_of(geometry.cache_type)
  if kind is nil then
    return "attend: cache_type " .. tostring(geometry.cache_type)
  end
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

local function slice(values: {number}, first: integer, count: integer): {number}
  local result: {number} = {}
  for offset = 0, count - 1 do
    result[offset + 1] = values[first + offset + 1]
  end
  return result
end

-- whether the handle holds every one of these tensors: a measuring build reads
-- them all, and a graph over a missing one would be built from nothing
local function holds(mapped: Mapped, names: {string}): boolean
  for _, name in ipairs(names) do
    if mapped.tensors[name] == nil then
      return false
    end
  end
  return true
end

local function prefixed(prefix: string, suffixes: {string}): {string}
  local names: {string} = {}
  for _, suffix in ipairs(suffixes) do
    names[#names + 1] = prefix .. suffix
  end
  return names
end

local ATTENTION_NAMES: {string} = {
  "attn_norm.weight",
  "attn_q.weight",
  "attn_k.weight",
  "attn_v.weight",
  "attn_q_norm.weight",
  "attn_k_norm.weight",
  "attn_output.weight",
}

local function tensor_names(prefix: string): {string}
  return prefixed(prefix, ATTENTION_NAMES)
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
  sequence: string,
  prefix: string,
  position: integer,
  expected: integer,
  capacity: integer
): string | nil
  if position < 0 then
    return "block: sequence "
      .. sequence
      .. " prefix "
      .. prefix
      .. " position "
      .. position
  end
  if position >= capacity then
    return "block: sequence "
      .. sequence
      .. " prefix "
      .. prefix
      .. " position "
      .. position
      .. " of "
      .. capacity
  end
  if position ~= expected then
    return "block: sequence "
      .. sequence
      .. " prefix "
      .. prefix
      .. " position "
      .. position
      .. ", the next is "
      .. expected
  end
  return nil
end

local function caches_in(
  state: AttentionState, sequence: string
): {string: Cache}
  local held = state.caches[sequence]
  if held is nil then
    held = {}
    state.caches[sequence] = held
  end
  return held
end

local function cache_capacity(state: AttentionState, sequence: string): integer
  return state.capacities[sequence] or state.geometry.positions
end

local function cache_of(
  state: AttentionState, sequence: string, prefix: string
): Cache | nil, string | nil
  local held = caches_in(state, sequence)
  local stored = held[prefix]
  if not (stored is nil) then
    return stored
  end

  local geometry = state.geometry
  local kind = cache_kind_of(geometry.cache_type) as CacheKind
  local rows = cache_capacity(state, sequence) * geometry.kv_heads
  local size = rows * cache_row_size(kind, geometry.head_length)
  local keys_block = libc.aligned_alloc(
    ARENA_ALIGN, round_up(size, ARENA_ALIGN)
  )
  local values_block = libc.aligned_alloc(
    ARENA_ALIGN, round_up(size, ARENA_ALIGN)
  )
  if keys_block is nil or values_block is nil then
    if not (keys_block is nil) then
      libc.free(keys_block)
    end
    if not (values_block is nil) then
      libc.free(values_block)
    end
    return nil, "block: no cache of " .. (size * 2) .. " bytes"
  end

  local cache = {
    keys = finalised(keys_block),
    values = finalised(values_block),
    bytes = size * 2,
    count = 0,
  } as Cache
  held[prefix] = cache
  return cache
end

local function cache_forget(cache: Cache)
  local keys = ffi.gc(cache.keys, nil)
  local values = ffi.gc(cache.values, nil)
  libc.free(keys)
  libc.free(values)
end

local function attention_release(state: AttentionState, sequence: string)
  local held = state.caches[sequence]
  if held is nil then
    state.capacities[sequence] = nil
    return
  end
  for prefix, cache in pairs(held) do
    cache_forget(cache)
    held[prefix] = nil
  end
  state.caches[sequence] = nil
  state.capacities[sequence] = nil
end

local function attention_closed(state: AttentionState)
  local sequences: {string} = {}
  for sequence in pairs(state.caches) do
    sequences[#sequences + 1] = sequence
  end
  for _, sequence in ipairs(sequences) do
    attention_release(state, sequence)
  end
  state.mapped.close()
end

local function attention_reserve(
  state: AttentionState,
  sequence: string,
  positions: integer,
  prefixes: {string}
): string | nil
  if positions > state.geometry.positions then
    return "reserve: positions "
      .. positions
      .. " of "
      .. state.geometry.positions
  end
  state.capacities[sequence] = positions
  for _, prefix in ipairs(prefixes) do
    local made, failure = cache_of(state, sequence, prefix)
    if made is nil then
      attention_release(state, sequence)
      return failure
    end
  end
  return nil
end

-- the bytes one block's keys, or its values, hold for this many positions
local function attention_bytes(
  state: AttentionState, positions: integer
): integer
  local geometry = state.geometry
  local kind = cache_kind_of(geometry.cache_type) as CacheKind
  return positions * geometry.kv_heads * cache_row_size(
    kind, geometry.head_length
  )
end

local function attention_snapshot(
  state: AttentionState, sequence: string, prefix: string, positions: integer
): string | nil, string | nil
  local held = state.caches[sequence]
  local cache: Cache | nil = nil
  if not (held is nil) then
    cache = held[prefix]
  end
  if cache is nil then
    return nil, prefix .. " holds no cache"
  end
  local stored = cache as Cache
  if stored.count ~= positions then
    return nil,
      prefix .. " took " .. stored.count .. " positions, not " .. positions
  end
  local count = attention_bytes(state, positions)
  return ffi.string(stored.keys, count) .. ffi.string(stored.values, count), nil
end

local function attention_restore(
  state: AttentionState,
  sequence: string,
  prefix: string,
  positions: integer,
  payload: string
): string | nil
  local count = attention_bytes(state, positions)
  if #payload ~= 2 * count then
    return prefix .. " holds " .. #payload .. " bytes, not " .. (2 * count)
  end
  local cache, failure = cache_of(state, sequence, prefix)
  if cache is nil then
    return failure
  end
  local stored = cache as Cache
  local base = ffi.cast("intptr_t", payload) as integer
  ffi.copy(stored.keys, payload, count)
  ffi.copy(
    stored.values, ffi.cast("const char *", base + count) as ffi.CData, count
  )
  stored.count = positions
  return nil
end

-- the cache tensors one row's graph reads and appends to: the host buffer
-- itself on the CPU path, a copy of the filled part on the device path, where
-- the rows the graph appended are copied back once it has computed
local record Appended
  keys: ffi.CData
  values: ffi.CData
end

local function row_cache(
  graph: Graph,
  cache: Cache,
  kind: CacheKind,
  width: integer,
  kv_heads: integer,
  past: integer
): ffi.CData, ffi.CData
  local row_bytes = cache_row_size(kind, width)
  local rows = past * kv_heads
  return host_tensor(
    graph, {width, rows}, kind.code, cache.keys, rows * row_bytes
  ),
    host_tensor(graph, {width, rows}, kind.code, cache.values, rows * row_bytes)
end

-- one head group's attention numbers: the row's own keys and values, the
-- queries of every head the group carries, and ggml's own scores and softmax
-- over the positions up to this row's
local function group_attended(
  graph: Graph,
  keys: ffi.CData,
  values: ffi.CData,
  queries: ffi.CData,
  kind: CacheKind,
  width: integer,
  kv_heads: integer,
  past: integer,
  group: integer,
  head: integer,
  scale: number
): ffi.CData
  local lib = graph.lib
  local ctx = graph.ctx
  local row_bytes = cache_row_size(kind, width)
  local values_over = node(
    graph,
    lib.ggml_view_2d(
      ctx, values, width, past, kv_heads * row_bytes, head * row_bytes
    )
  )
  local numbers = values_over
  if kind.block_elements > 1 then
    numbers = node(graph, lib.ggml_cast(ctx, values_over, GGML_TYPE_F32))
  end
  local turned = node(
    graph, lib.ggml_cont(ctx, node(graph, lib.ggml_transpose(ctx, numbers)))
  )
  local keys_over = node(
    graph,
    lib.ggml_view_2d(
      ctx, keys, width, past, kv_heads * row_bytes, head * row_bytes
    )
  )
  local carried = node(
    graph,
    lib.ggml_view_2d(
      ctx,
      queries,
      width,
      group,
      width * F32_BYTES,
      head * group * width * F32_BYTES
    )
  )
  local scores = node(graph, lib.ggml_mul_mat(ctx, keys_over, carried))
  local probs = node(graph, lib.ggml_soft_max_ext(ctx, scores, nil, scale, 0))
  return node(graph, lib.ggml_mul_mat(ctx, turned, probs))
end

-- one block's attention half for every row of the batch in one graph: the
-- batch's projections, its query heads' and key heads' norms and its rope are
-- one node each over the whole batch, and only the cache rows and the scores
-- that follow a row's own position are that row's own
local function attention_build(
  state: AttentionState,
  graph: Graph,
  entries: {compute.Row},
  prefix: string,
  caches: {Cache},
  appended: {Appended}
): {Output}
  local lib = graph.lib
  local ctx = graph.ctx
  local geometry = state.geometry
  local descriptors = state.mapped.descriptors
  local kind = cache_kind_of(geometry.cache_type) as CacheKind
  local heads = geometry.heads
  local kv_heads = geometry.kv_heads
  local width = geometry.head_length
  local group = math.floor(heads / kv_heads)
  local eps = geometry.eps
  local row_bytes = cache_row_size(kind, width)
  local scale = 1 / math.sqrt(width)
  local rows = #entries

  local hiddens: {{number}} = {}
  local positions: {integer} = {}
  for index, entry in ipairs(entries) do
    hiddens[index] = entry.hidden
    positions[index] = entry.position
  end

  local normalised = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(
          ctx,
          matrix_of(graph, hiddens, #(entries[1] as compute.Row).hidden),
          eps
        )
      ),
      descriptors[prefix .. "attn_norm.weight"]
    )
  )
  local query = node(
    graph,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_q.weight"], normalised)
  )
  local read_keys = node(
    graph,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_k.weight"], normalised)
  )
  local read_values = node(
    graph,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_v.weight"], normalised)
  )

  -- the query projection carries each head's query and gate side by side
  local grouped = node(
    graph, lib.ggml_reshape_3d(ctx, query, width, 2 * heads, rows)
  )
  local raw_queries = node(
    graph,
    lib.ggml_cont(
      ctx,
      node(
        graph,
        lib.ggml_view_3d(
          ctx,
          grouped,
          width,
          heads,
          rows,
          2 * width * F32_BYTES,
          2 * width * heads * F32_BYTES,
          0
        )
      )
    )
  )
  local gates = node(
    graph,
    lib.ggml_cont(
      ctx,
      node(
        graph,
        lib.ggml_view_3d(
          ctx,
          grouped,
          width,
          heads,
          rows,
          2 * width * F32_BYTES,
          2 * width * heads * F32_BYTES,
          width * F32_BYTES
        )
      )
    )
  )

  -- one rope over every head of every row, each row at its own position
  local at_positions = ints_of(graph, positions)
  local queries = node(
    graph,
    lib.ggml_rope_ext(
      ctx,
      node(
        graph,
        lib.ggml_mul(
          ctx,
          node(graph, lib.ggml_rms_norm(ctx, raw_queries, eps)),
          descriptors[prefix .. "attn_q_norm.weight"]
        )
      ),
      at_positions,
      nil,
      geometry.rope_dimensions,
      GGML_ROPE_TYPE_NEOX,
      0,
      geometry.rope_base,
      1.0,
      0.0,
      1.0,
      0.0,
      0.0
    )
  )
  local read_keys_3d = node(
    graph, lib.ggml_reshape_3d(ctx, read_keys, width, kv_heads, rows)
  )
  local rotated_keys = node(
    graph,
    lib.ggml_rope_ext(
      ctx,
      node(
        graph,
        lib.ggml_mul(
          ctx,
          node(graph, lib.ggml_rms_norm(ctx, read_keys_3d, eps)),
          descriptors[prefix .. "attn_k_norm.weight"]
        )
      ),
      at_positions,
      nil,
      geometry.rope_dimensions,
      GGML_ROPE_TYPE_NEOX,
      0,
      geometry.rope_base,
      1.0,
      0.0,
      1.0,
      0.0,
      0.0
    )
  )
  local read_values_3d = node(
    graph, lib.ggml_reshape_3d(ctx, read_values, width, kv_heads, rows)
  )

  local outputs: {Output} = {}
  for index, entry in ipairs(entries) do
    local cache = caches[index]
    local past = entry.position + 1
    local at = entry.position * kv_heads * row_bytes
    local keys, values = row_cache(graph, cache, kind, width, kv_heads, past)
    appended[index] = {keys = keys, values = values}

    -- this row's position of the batch's key and value rows goes into the cache,
    -- and these two writes are expanded before the nodes that read the cache so
    -- the graph runs them first
    outputs[#outputs + 1] = {
      tensor = node(
        graph,
        lib.ggml_cpy(
          ctx,
          node(
            graph,
            lib.ggml_view_3d(
              ctx,
              rotated_keys,
              width,
              kv_heads,
              1,
              width * F32_BYTES,
              kv_heads * width * F32_BYTES,
              (index - 1) * kv_heads * width * F32_BYTES
            )
          ),
          node(
            graph, lib.ggml_view_2d(ctx, keys, width, kv_heads, row_bytes, at)
          )
        )
      ),
      count = 0,
    }
    outputs[#outputs + 1] = {
      tensor = node(
        graph,
        lib.ggml_cpy(
          ctx,
          node(
            graph,
            lib.ggml_view_3d(
              ctx,
              read_values_3d,
              width,
              kv_heads,
              1,
              width * F32_BYTES,
              kv_heads * width * F32_BYTES,
              (index - 1) * kv_heads * width * F32_BYTES
            )
          ),
          node(
            graph, lib.ggml_view_2d(ctx, values, width, kv_heads, row_bytes, at)
          )
        )
      ),
      count = 0,
    }

    local row_shape = width * heads * F32_BYTES
    local carried = node(
      graph,
      lib.ggml_reshape_2d(
        ctx,
        node(
          graph,
          lib.ggml_view_3d(
            ctx,
            queries,
            width,
            heads,
            1,
            width * F32_BYTES,
            row_shape,
            (index - 1) * row_shape
          )
        ),
        width,
        heads
      )
    )
    local row_gates = node(
      graph,
      lib.ggml_reshape_2d(
        ctx,
        node(
          graph,
          lib.ggml_view_3d(
            ctx,
            gates,
            width,
            heads,
            1,
            width * F32_BYTES,
            row_shape,
            (index - 1) * row_shape
          )
        ),
        width,
        heads
      )
    )

    local attended: {ffi.CData} = {}
    for head = 0, kv_heads - 1 do
      attended[head + 1] = group_attended(
        graph,
        keys,
        values,
        carried,
        kind,
        width,
        kv_heads,
        past,
        group,
        head,
        scale
      )
    end

    local combined = attended[1]
    for head = 2, kv_heads do
      combined = node(graph, lib.ggml_concat(ctx, combined, attended[head], 1))
    end
    local gated = node(
      graph,
      lib.ggml_mul(ctx, node(graph, lib.ggml_sigmoid(ctx, row_gates)), combined)
    )
    local flat = node(graph, lib.ggml_reshape_2d(ctx, gated, width * heads, 1))
    outputs[#outputs + 1] = {
      tensor = node(
        graph,
        lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_output.weight"], flat)
      ),
      count = #entry.hidden,
    }
  end
  return outputs
end

-- the rows the graph appended, copied back into the host cache it read
local function attention_kept(
  made: Computed,
  entries: {compute.Row},
  caches: {Cache},
  appended: {Appended},
  kind: CacheKind,
  geometry: compute.Geometry
)
  local device = made.device
  if device is nil then
    return
  end
  local row_bytes = cache_row_size(kind, geometry.head_length)
  local bytes = geometry.kv_heads * row_bytes
  for index, entry in ipairs(entries) do
    local held = appended[index]
    local cache = caches[index]
    local at = entry.position * bytes
    read_bytes(made.lib, held.keys, cache.keys, bytes, device, at)
    read_bytes(made.lib, held.values, cache.values, bytes, device, at)
  end
end

-- a pointer a measuring build hands a host tensor: no number behind it is
-- ever read, and it stands for every such tensor of every measuring graph
local MEASURE_POINTER: ffi.CData | nil = nil

local function measure_pointer(): ffi.CData | nil
  if MEASURE_POINTER is nil then
    local block = libc.aligned_alloc(ARENA_ALIGN, ARENA_ALIGN)
    if block is nil then
      return nil
    end
    MEASURE_POINTER = finalised(block)
  end
  return MEASURE_POINTER
end

local function blank_hidden(count: integer): {number}
  local hidden: {number} = {}
  for at = 1, count do
    hidden[at] = 0
  end
  return hidden
end

-- the rows a floor of a handle's graphs is measured over: as many as a pass
-- holds, at the first position
local function blank_rows(width: integer, rows: integer): {compute.Row}
  local entries: {compute.Row} = {}
  local hidden = blank_hidden(width)
  for index = 1, rows do
    entries[index] = {sequence = "", hidden = hidden, position = 0}
  end
  return entries
end

-- every expert a router could choose, so the measuring build makes as many
-- expert nodes as a real one
local function even_votes(experts: integer): {number}, number
  local probabilities: {number} = {}
  for index = 1, experts do
    probabilities[index] = 1
  end
  return probabilities, experts
end

-- the width of a block's hidden numbers, as the tensor named with this suffix
local function hidden_width(mapped: Mapped, suffix: string): integer | nil
  for name, tensor in pairs(mapped.tensors) do
    if #name > #suffix and name:sub(-#suffix) == suffix then
      return tensor.dims[1]
    end
  end
  return nil
end

-- the prefix of a block holding this suffix, every block of a run naming the
-- same shapes
local function prefix_named(mapped: Mapped, suffix: string): string | nil
  for name in pairs(mapped.tensors) do
    if #name > #suffix and name:sub(-#suffix) == suffix then
      return name:sub(1, #name - #suffix)
    end
  end
  return nil
end

-- the largest graph an attention handle builds: every row of a pass at the
-- geometry's last position, where the scores a row holds for its heads are
-- widest
local function attention_measure(
  state: AttentionState, prefix: string, rows: integer
): Build | nil
  local geometry = state.geometry
  local norm = state.mapped.tensors[prefix .. "attn_norm.weight"]
  if norm is nil then
    return nil
  end
  local width = (norm.dims[1] or 0) as integer
  local scratch = measure_pointer()
  if scratch is nil then
    return nil
  end
  local kept = {keys = scratch, values = scratch, bytes = 0, count = 0} as Cache
  local entries: {compute.Row} = {}
  local caches: {Cache} = {}
  for index = 1, rows do
    entries[index] = {
      sequence = "",
      hidden = blank_hidden(width),
      position = geometry.positions - 1,
    }
    caches[index] = kept
  end
  return {
    kind = "attention",
    build = function(graph: Graph): {Output}
      return attention_build(state, graph, entries, prefix, caches, {})
    end,
  }
end

-- the one arena an attention handle holds, sized from that graph
local function attention_sized(
  state: AttentionState, prefix: string, rows: integer
): Pool | nil
  if not holds(state.mapped, prefixed(prefix, ATTENTION_NAMES)) then
    return nil
  end
  local entry = attention_measure(state, prefix, rows)
  if entry is nil then
    return nil
  end
  return sized_pool(
    state.mapped.lib, state.mapped.device, {entry}, rows, nil, nil
  )
end

-- the arena a graph is built in: the model's, which holds every row a pass can
-- carry, or the handle's own, sized again from this call's rows when the call
-- holds more rows than the arena it already has was measured at
local function grown(
  box: Box, owned: boolean, rows: integer, size: function(integer): Pool | nil
): Pool | nil
  local held = box.pool
  if not owned then
    return held
  end
  if not (held is nil) and held.rows >= rows then
    return held
  end
  local wider = size(rows)
  if wider is nil then
    return nil
  end
  if not (held is nil) then
    pool_free(held)
  end
  box.pool = wider
  return wider
end

local function attention_arena(
  state: AttentionState, prefix: string, rows: integer
): Pool | nil
  return grown(
    state.pool,
    state.owner,
    rows,
    function(most: integer): Pool | nil
      return attention_sized(state, prefix, most)
    end
  )
end

local function attention_block(
  state: AttentionState, rows: {compute.Row}, prefix: string
): {{number}} | nil, string | nil
  if #rows == 0 then
    return {}, nil
  end
  local geometry = state.geometry
  local mapped = state.mapped
  local kind = cache_kind_of(geometry.cache_type) as CacheKind
  local checked: {integer: boolean} = {}
  -- the position each sequence's next row of this call must name: a group runs
  -- on from the row before it rather than from the cache alone
  local running: {string: integer} = {}

  for _, entry in ipairs(rows) do
    if not checked[#entry.hidden] then
      local read, refusal = read_tensors(
        mapped, prefix, #entry.hidden, geometry
      )
      if not (read is BlockTensors) then
        return nil, refusal
      end
      checked[#entry.hidden] = true
      if geometry.head_length % kind.block_elements ~= 0 then
        return nil,
          "block: cache "
            .. geometry.cache_type
            .. " cannot hold a head of "
            .. geometry.head_length
            .. " numbers"
      end
    end
    local cache: Cache | nil = nil
    local held = state.caches[entry.sequence]
    if not (held is nil) then
      cache = held[prefix]
    end
    local expected = running[entry.sequence]
    if expected is nil then
      expected = 0
      if not (cache is nil) then
        expected = (cache as Cache).count
      end
    end
    local bad_position = position_refusal(
      entry.sequence,
      prefix,
      entry.position,
      expected,
      cache_capacity(state, entry.sequence)
    )
    if bad_position then
      return nil, bad_position
    end
    running[entry.sequence] = entry.position + 1
  end

  for _, entry in ipairs(rows) do
    local held = state.caches[entry.sequence]
    local cache: Cache | nil = nil
    if not (held is nil) then
      cache = held[prefix]
    end
    if cache is nil then
      local made, failure = cache_of(state, entry.sequence, prefix)
      if not (made is Cache) then
        return nil, failure
      end
    end
  end

  local pool = attention_arena(state, prefix, #rows)
  if pool is nil then
    return nil, "block: no arena for its graph"
  end

  local caches: {Cache} = {}
  for index, entry in ipairs(rows) do
    caches[index] = (state.caches[entry.sequence] as {string: Cache})[prefix]
  end
  local appended: {Appended} = {}

  local made, failure = graph_run(
    pool,
    prefix,
    "attention",
    "block",
    state.measure,
    function(graph: Graph): {Output}
      return attention_build(state, graph, rows, prefix, caches, appended)
    end
  )
  if made is nil then
    return nil, failure
  end
  local planes = graph_planes(made)
  attention_kept(made, rows, caches, appended, kind, geometry)
  graph_closed(made)

  for _, entry in ipairs(rows) do
    local kept = (state.caches[entry.sequence] as {string: Cache})[prefix]
    kept.count = entry.position + 1
  end
  return planes, nil
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

-- the matrix the index selects of a three-dimensional weight, or the whole
-- descriptor otherwise, as `11/FR-002`'s matrix product reads it
local function weight_of(
  graph: Graph, mapped: Mapped, name: string, index?: integer
): ffi.CData
  local tensor = mapped.tensors[name] as Tensor
  local dims = tensor.dims
  if #dims < 3 then
    return mapped.descriptors[name]
  end
  local stride = math.floor(tensor.bytes / (dims[3] as integer))
  local base = ffi.cast("intptr_t", tensor.data) as integer
  local location = ffi.cast("void *", base + (index or 0) * stride) as ffi.CData
  return host_tensor(
    graph,
    {dims[1] as integer, dims[2] as integer},
    tensor.kind.code,
    location,
    stride
  )
end

-- the norm and the router of one row, whose softmax the host chooses from
local function feed_forward_votes(
  state: FeedForwardState, graph: Graph, entry: compute.Row, prefix: string
): ffi.CData
  local lib = graph.lib
  local ctx = graph.ctx
  local mapped = state.mapped
  local normalised = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(
          ctx, vector_of(graph, entry.hidden), state.geometry.eps
        )
      ),
      mapped.descriptors[prefix .. "post_attention_norm.weight"]
    )
  )
  local votes = node(
    graph,
    lib.ggml_mul_mat(
      ctx, mapped.descriptors[prefix .. "ffn_gate_inp.weight"], normalised
    )
  )
  return node(graph, lib.ggml_soft_max(ctx, votes))
end

-- the numbers a block's feed forward leaves one row holding: the chosen
-- experts weighted by their router numbers, the shared expert gated by its own
-- router, and both residual adds, all in one graph
local function feed_forward_choice(
  state: FeedForwardState,
  graph: Graph,
  entry: compute.Row,
  prefix: string,
  probabilities: {number},
  total: number
): ffi.CData
  local lib = graph.lib
  local ctx = graph.ctx
  local mapped = state.mapped
  local descriptors = mapped.descriptors
  local normalised = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(
          ctx, vector_of(graph, entry.hidden), state.geometry.eps
        )
      ),
      descriptors[prefix .. "post_attention_norm.weight"]
    )
  )

  local experts: ffi.CData | nil = nil
  for _, expert in ipairs(
    chosen_experts(probabilities, state.geometry.experts_used)
  ) do
    local up = node(
      graph,
      lib.ggml_mul_mat(
        ctx,
        weight_of(graph, mapped, prefix .. "ffn_up_exps.weight", expert),
        normalised
      )
    )
    local gate = node(
      graph,
      lib.ggml_mul_mat(
        ctx,
        weight_of(graph, mapped, prefix .. "ffn_gate_exps.weight", expert),
        normalised
      )
    )
    local mixed = node(graph, lib.ggml_swiglu_split(ctx, gate, up))
    local down = node(
      graph,
      lib.ggml_mul_mat(
        ctx,
        weight_of(graph, mapped, prefix .. "ffn_down_exps.weight", expert),
        mixed
      )
    )
    local weighted = node(
      graph, lib.ggml_scale(ctx, down, probabilities[expert + 1] / total)
    )
    if experts is nil then
      experts = weighted
    else
      experts = node(graph, lib.ggml_add(ctx, experts, weighted))
    end
  end

  local shared_up = node(
    graph,
    lib.ggml_mul_mat(
      ctx, descriptors[prefix .. "ffn_up_shexp.weight"], normalised
    )
  )
  local shared_gate = node(
    graph,
    lib.ggml_mul_mat(
      ctx, descriptors[prefix .. "ffn_gate_shexp.weight"], normalised
    )
  )
  local shared_mixed = node(
    graph, lib.ggml_swiglu_split(ctx, shared_gate, shared_up)
  )
  local shared_down = node(
    graph,
    lib.ggml_mul_mat(
      ctx, descriptors[prefix .. "ffn_down_shexp.weight"], shared_mixed
    )
  )
  local router = node(
    graph,
    lib.ggml_mul_mat(
      ctx, descriptors[prefix .. "ffn_gate_inp_shexp.weight"], normalised
    )
  )
  local gated = node(
    graph,
    lib.ggml_mul(
      ctx, node(graph, lib.ggml_sigmoid(ctx, router)), vector_of(graph, {1})
    )
  )
  local shared = node(graph, lib.ggml_mul(ctx, shared_down, gated))
  local mixed_hidden = node(
    graph,
    lib.ggml_add(ctx, vector_of(graph, entry.hidden), experts as ffi.CData)
  )
  return node(graph, lib.ggml_add(ctx, mixed_hidden, shared))
end

-- one row's vocabulary scores: its norm and the output projection
local function feed_forward_row_scores(
  state: FeedForwardState, graph: Graph, hidden: {number}
): ffi.CData
  local lib = graph.lib
  local ctx = graph.ctx
  local normalised = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(ctx, vector_of(graph, hidden), state.geometry.eps)
      ),
      state.mapped.descriptors["output_norm.weight"]
    )
  )
  return node(
    graph,
    lib.ggml_mul_mat(ctx, state.mapped.descriptors["output.weight"], normalised)
  )
end

-- the graphs a feed forward handle builds for a block, over the rows a pass
-- holds, and none of them when the block's tensors are not all there
local function feed_forward_block_builds(
  state: FeedForwardState, prefix: string, width: integer, rows: integer
): {Build}
  local mapped = state.mapped
  if not holds(mapped, prefixed(prefix, BLOCK_NAMES)) then
    return {}
  end
  local router = mapped.tensors[prefix .. "ffn_gate_inp.weight"] as Tensor
  local experts = router.dims[2] as integer
  if state.geometry.experts_used > experts then
    return {}
  end
  local entries = blank_rows(width, rows)
  local probabilities, total = even_votes(state.geometry.experts_used)
  return {
    {
      kind = "votes",
      build = function(graph: Graph): {Output}
        local outputs: {Output} = {}
        for index, entry in ipairs(entries) do
          outputs[index] = {
            tensor = feed_forward_votes(state, graph, entry, prefix),
            count = experts,
          }
        end
        return outputs
      end,
    },
    {
      kind = "choice",
      build = function(graph: Graph): {Output}
        local outputs: {Output} = {}
        for index, entry in ipairs(entries) do
          outputs[index] = {
            tensor = feed_forward_choice(
              state, graph, entry, prefix, probabilities, total
            ),
            count = #entry.hidden,
          }
        end
        return outputs
      end,
    },
  }
end

-- the graphs the scores of a pass build, over the rows a pass holds
local function feed_forward_score_builds(
  state: FeedForwardState, width: integer, rows: integer
): {Build}
  local mapped = state.mapped
  if not holds(mapped, SCORE_NAMES) then
    return {}
  end
  local hidden: {{number}} = {}
  local entry = blank_hidden(width)
  for index = 1, rows do
    hidden[index] = entry
  end
  local vocab = (mapped.tensors["output.weight"] as Tensor).dims[2] as integer
  return {
    {
      kind = "scores",
      build = function(graph: Graph): {Output}
        local outputs: {Output} = {}
        for index, scores in ipairs(hidden) do
          outputs[index] = {
            tensor = feed_forward_row_scores(state, graph, scores),
            count = vocab,
          }
        end
        return outputs
      end,
    },
  }
end

local function feed_forward_builds(
  state: FeedForwardState, prefix: string, width: integer, rows: integer
): {Build}
  local builds = feed_forward_block_builds(state, prefix, width, rows)
  for _, entry in ipairs(feed_forward_score_builds(state, width, rows)) do
    builds[#builds + 1] = entry
  end
  return builds
end

-- the one arena a handle built on its own holds, sized from the largest block
-- graph and the largest scores graph at this call's rows, whichever of the two
-- the call is: the block's width comes from the tensors it reads, so a scores
-- call sizes the block graph correctly too
local function feed_forward_ready(
  state: FeedForwardState, prefix: string | nil, width: integer, rows: integer
): Pool | nil
  local builds: {Build} = {}
  local block_prefix = prefix
  if block_prefix is nil then
    block_prefix = prefix_named(state.mapped, "post_attention_norm.weight")
  end
  local block_width = hidden_width(state.mapped, "post_attention_norm.weight")
  if not (block_prefix is nil) and not (block_width is nil) then
    for _, entry in ipairs(
      feed_forward_block_builds(state, block_prefix, block_width, rows)
    ) do
      builds[#builds + 1] = entry
    end
  end
  for _, entry in ipairs(feed_forward_score_builds(state, width, rows)) do
    builds[#builds + 1] = entry
  end
  if #builds == 0 then
    return nil
  end
  return sized_pool(
    state.mapped.lib, state.mapped.device, builds, rows, nil, nil
  )
end

-- what a router's choice reports: how many experts it routed to, the bytes of
-- their weights, and the line the routing sink draws from it
local function expert_bytes(
  mapped: Mapped, prefix: string, selected: {integer}
): integer
  local total = 0
  for _, name in ipairs(
    {"ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight"}
  ) do
    local tensor = mapped.tensors[prefix .. name]
    if not (tensor is nil) then
      local count = 1
      if #tensor.dims >= 3 then
        count = tensor.dims[3] as integer
      end
      if count < 1 then
        count = 1
      end
      total = total + math.floor(tensor.bytes / count) * #selected
    end
  end
  return total
end

local function recorded(
  watch: compute.Watch, mapped: Mapped, prefix: string, selected: {integer}
)
  watch.measure.routed = watch.measure.routed + #selected
  watch.measure.routed_bytes = watch.measure.routed_bytes + expert_bytes(
    mapped, prefix, selected
  )
  local sink = watch.sink
  if sink is nil then
    return
  end
  local ids: {integer} = {}
  for index, expert in ipairs(selected) do
    ids[index] = expert
  end
  table.sort(ids)
  sink({layer = watch.layer, token = watch.token, experts = ids})
end

local function feed_forward_arena(
  state: FeedForwardState, prefix: string | nil, width: integer, rows: integer
): Pool | nil
  return grown(
    state.pool,
    state.owner,
    rows,
    function(most: integer): Pool | nil
      return feed_forward_ready(state, prefix, width, most)
    end
  )
end

local function feed_forward_block(
  state: FeedForwardState,
  rows: {compute.Row},
  prefix: string,
  watchs: {compute.Watch} | nil
): {{number}} | nil, string | nil
  local mapped = state.mapped
  local geometry = state.geometry
  local checked: {integer: boolean} = {}
  for _, entry in ipairs(rows) do
    if not checked[#entry.hidden] then
      local fault = first_fault(
        mapped, "block", prefix, BLOCK_NAMES, BLOCK_TYPES
      )
      if fault then
        return nil, fault
      end
      fault = block_shape_refusal(
        mapped, prefix, #entry.hidden, geometry.experts_used
      )
      if fault then
        return nil, fault
      end
      checked[#entry.hidden] = true
    end
  end

  local router = mapped.tensors[prefix .. "ffn_gate_inp.weight"] as Tensor
  local experts = router.dims[2] as integer
  local width = #((rows[1] as compute.Row).hidden)
  local pool = feed_forward_arena(state, prefix, width, #rows)
  if pool is nil then
    return nil, "block: no arena for its graphs"
  end

  local made, failure = graph_run(
    pool,
    prefix,
    "votes",
    "block",
    state.measure,
    function(graph: Graph): {Output}
      local outputs: {Output} = {}
      for index, entry in ipairs(rows) do
        outputs[index] = {
          tensor = feed_forward_votes(state, graph, entry, prefix),
          count = experts,
        }
      end
      return outputs
    end
  )
  if made is nil then
    return nil, failure
  end
  local votes = graph_planes(made)
  graph_closed(made)

  local totals: {number} = {}
  for index, probabilities in ipairs(votes) do
    local chosen = chosen_experts(probabilities, geometry.experts_used)
    local total: number = 0
    for _, expert in ipairs(chosen) do
      total = total + probabilities[expert + 1]
    end
    totals[index] = total
    if not (watchs is nil) then
      recorded((watchs as {compute.Watch})[index], mapped, prefix, chosen)
    end
  end

  local second, second_failure = graph_run(
    pool,
    prefix,
    "choice",
    "block",
    state.measure,
    function(graph: Graph): {Output}
      local outputs: {Output} = {}
      for index, entry in ipairs(rows) do
        outputs[index] = {
          tensor = feed_forward_choice(
            state, graph, entry, prefix, votes[index], totals[index]
          ),
          count = #entry.hidden,
        }
      end
      return outputs
    end
  )
  if second is nil then
    return nil, second_failure
  end
  local planes = graph_planes(second)
  graph_closed(second)
  return planes, nil
end

local function feed_forward_scores(
  state: FeedForwardState, hidden: {{number}}
): {{number}} | nil, string | nil
  if #hidden == 0 then
    return {}, nil
  end
  local mapped = state.mapped
  local checked: {integer: boolean} = {}
  for _, entry in ipairs(hidden) do
    if not checked[#entry] then
      local fault = first_fault(mapped, "scores", "", SCORE_NAMES, SCORE_TYPES)
      if fault then
        return nil, fault
      end
      local count = #entry
      local norm_name = "output_norm.weight"
      local output_name = "output.weight"
      local norm = mapped.tensors[norm_name] as Tensor
      local output = mapped.tensors[output_name] as Tensor
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
        or expected(
          output_name, (output.dims[1] or 0) as integer, count, "numbers"
        )
      if refusal then
        return nil, refusal
      end
      if (output.dims[2] or 0) < 1 then
        return nil, "scores: tensor " .. output_name .. " holds no rows"
      end
      checked[#entry] = true
    end
  end

  local vocab = (mapped.tensors["output.weight"] as Tensor).dims[2] as integer
  local width = #(hidden[1] as {number})
  local pool = feed_forward_arena(state, nil, width, #hidden)
  if pool is nil then
    return nil, "scores: no arena for its graphs"
  end
  local made, failure = graph_run(
    pool as Pool,
    "",
    "scores",
    "scores",
    state.measure,
    function(graph: Graph): {Output}
      local outputs: {Output} = {}
      for index, entry in ipairs(hidden) do
        outputs[index] = {
          tensor = feed_forward_row_scores(state, graph, entry),
          count = vocab,
        }
      end
      return outputs
    end
  )
  if made is nil then
    return nil, failure
  end
  local planes = graph_planes(made)
  graph_closed(made)
  return planes, nil
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

local function prefixes_in(
  state: LinearState, sequence: string
): {string: LinearKept}
  local held = state.prefixes[sequence]
  if held is nil then
    held = {}
    state.prefixes[sequence] = held
  end
  return held
end

-- the floats a linear block's history and its state hold
local function linear_floats(shape: LinearShape): integer, integer
  local head_rows = shape.key_heads * shape.key_length
  local rows = 2 * head_rows + shape.value_heads * shape.value_length
  local history = (shape.conv_kernel - 1) * rows
  -- one float stands in for an empty history, so the field never holds nil
  if history < 1 then
    history = 1
  end
  local state = shape.key_length * shape.value_length * shape.value_heads
  return history, state
end

local function linear_prefix_of(
  state: LinearState, sequence: string, prefix: string
): LinearKept | nil, string | nil
  local shape = state.shape
  local history_floats, state_floats = linear_floats(shape)

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
  prefixes_in(state, sequence)[prefix] = kept
  return kept
end

local function linear_release(state: LinearState, sequence: string)
  local held = state.prefixes[sequence]
  if held is nil then
    return
  end
  for prefix, kept in pairs(held) do
    kept.history = nil as ffi.CData
    kept.state = nil as ffi.CData
    held[prefix] = nil
  end
  state.prefixes[sequence] = nil
end

local function linear_closed(state: LinearState)
  local sequences: {string} = {}
  for sequence in pairs(state.prefixes) do
    sequences[#sequences + 1] = sequence
  end
  for _, sequence in ipairs(sequences) do
    linear_release(state, sequence)
  end
  state.mapped.close()
end

local function linear_snapshot(
  state: LinearState, sequence: string, prefix: string, positions: integer
): string | nil, string | nil
  local held = state.prefixes[sequence]
  local kept: LinearKept | nil = nil
  if not (held is nil) then
    kept = held[prefix]
  end
  if kept is nil and positions == 0 then
    return "", nil
  end
  if kept is nil then
    return nil, prefix .. " holds no history"
  end
  local active = kept as LinearKept
  if active.count ~= positions then
    return nil,
      prefix .. " took " .. active.count .. " positions, not " .. positions
  end
  if positions == 0 then
    return "", nil
  end
  local history_floats, state_floats = linear_floats(state.shape)
  return ffi.string(active.history, history_floats * F32_BYTES)
    .. ffi.string(active.state, state_floats * F32_BYTES), nil
end

local function linear_restore(
  state: LinearState,
  sequence: string,
  prefix: string,
  positions: integer,
  payload: string
): string | nil
  local history_floats, state_floats = linear_floats(state.shape)
  local wanted = 0
  if positions > 0 then
    wanted = (history_floats + state_floats) * F32_BYTES
  end
  if #payload ~= wanted then
    return prefix .. " holds " .. #payload .. " bytes, not " .. wanted
  end
  if positions == 0 then
    return nil
  end
  local kept, failure = linear_prefix_of(state, sequence, prefix)
  if kept is nil then
    return failure
  end
  local active = kept as LinearKept
  local history_bytes = history_floats * F32_BYTES
  local base = ffi.cast("intptr_t", payload) as integer
  ffi.copy(active.history, payload, history_bytes)
  ffi.copy(
    active.state,
    ffi.cast("const char *", base + history_bytes) as ffi.CData,
    state_floats * F32_BYTES
  )
  active.count = positions
  return nil
end

local record LinearBuilt
  output: ffi.CData
  updated: ffi.CData
  projected: ffi.CData
end

-- what the rows before this one in its group leave it: the state the row
-- before it left and every projection the group made, oldest first
local record LinearCarry
  state: ffi.CData | nil
  taps: {ffi.CData}
end

-- one row's linear-attention graph: the tensors it answers with, the state it
-- leaves, and the projection the history takes its newest tap from
local function linear_build(
  state: LinearState,
  graph: Graph,
  entry: compute.Row,
  prefix: string,
  active: LinearKept,
  carry: LinearCarry
): LinearBuilt
  local lib = graph.lib
  local ctx = graph.ctx
  local mapped = state.mapped
  local descriptors = mapped.descriptors
  local shape = state.shape
  local key_heads = shape.key_heads
  local key_length = shape.key_length
  local value_heads = shape.value_heads
  local value_length = shape.value_length
  local kernel = shape.conv_kernel
  local rows = 2 * key_heads * key_length + value_heads * value_length
  local channels = value_heads * value_length
  local eps = state.eps
  local state_floats = key_length * value_length * value_heads
  local head_bytes = key_length * F32_BYTES
  local value_head_bytes = value_length * F32_BYTES
  local head_scale = 1 / math.sqrt(key_length)

  local normalised = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(graph, lib.ggml_rms_norm(ctx, vector_of(graph, entry.hidden), eps)),
      descriptors[prefix .. "attn_norm.weight"]
    )
  )
  local projected = node(
    graph,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_qkv.weight"], normalised)
  )
  local gates = node(
    graph,
    lib.ggml_mul_mat(ctx, descriptors[prefix .. "attn_gate.weight"], normalised)
  )
  local betas = node(
    graph,
    lib.ggml_sigmoid(
      ctx,
      node(
        graph,
        lib.ggml_mul_mat(
          ctx, descriptors[prefix .. "ssm_beta.weight"], normalised
        )
      )
    )
  )
  local decay = node(
    graph,
    lib.ggml_exp(
      ctx,
      node(
        graph,
        lib.ggml_mul(
          ctx,
          node(
            graph,
            lib.ggml_softplus(
              ctx,
              node(
                graph,
                lib.ggml_add(
                  ctx,
                  node(
                    graph,
                    lib.ggml_mul_mat(
                      ctx, descriptors[prefix .. "ssm_alpha.weight"], normalised
                    )
                  ),
                  descriptors[prefix .. "ssm_dt.bias"]
                )
              )
            )
          ),
          descriptors[prefix .. "ssm_a"]
        )
      )
    )
  )

  -- the taps of this row's window: the stored window's oldest taps the group
  -- does not carry, then the rows before it, then its own projection
  local current = node(graph, lib.ggml_reshape_3d(ctx, projected, 1, rows, 1))
  local window = current
  if kernel > 1 then
    local slots = kernel - 1
    local mine = #carry.taps
    local used = mine
    if used > slots then
      used = slots
    end
    local parts: {ffi.CData} = {}
    if slots > used then
      local stored = host_tensor(
        graph,
        {slots, rows, 1},
        GGML_TYPE_F32,
        active.history,
        slots * rows * F32_BYTES
      )
      parts[#parts + 1] = node(
        graph,
        lib.ggml_view_3d(
          ctx,
          stored,
          slots - used,
          rows,
          1,
          slots * F32_BYTES,
          slots * rows * F32_BYTES,
          used * F32_BYTES
        )
      )
    end
    for tap = mine - used + 1, mine do
      parts[#parts + 1] = node(
        graph, lib.ggml_reshape_3d(ctx, carry.taps[tap], 1, rows, 1)
      )
    end
    parts[#parts + 1] = current
    window = parts[1]
    for part = 2, #parts do
      window = node(graph, lib.ggml_concat(ctx, window, parts[part], 0))
    end
  end
  local convolved = node(
    graph,
    lib.ggml_silu(
      ctx,
      node(
        graph,
        lib.ggml_ssm_conv(
          ctx, window, descriptors[prefix .. "ssm_conv1d.weight"]
        )
      )
    )
  )

  local queries = node(
    graph,
    lib.ggml_scale(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(
          ctx,
          node(
            graph,
            lib.ggml_view_2d(
              ctx, convolved, key_length, key_heads, head_bytes, 0
            )
          ),
          eps / key_length
        )
      ),
      head_scale
    )
  )
  local reads = node(
    graph,
    lib.ggml_scale(
      ctx,
      node(
        graph,
        lib.ggml_rms_norm(
          ctx,
          node(
            graph,
            lib.ggml_view_2d(
              ctx,
              convolved,
              key_length,
              key_heads,
              head_bytes,
              key_heads * head_bytes
            )
          ),
          eps / key_length
        )
      ),
      head_scale
    )
  )
  local values = node(
    graph,
    lib.ggml_reshape_3d(
      ctx,
      node(
        graph,
        lib.ggml_view_2d(
          ctx,
          convolved,
          value_length,
          value_heads,
          value_head_bytes,
          2 * key_heads * head_bytes
        )
      ),
      value_length,
      1,
      value_heads
    )
  )

  local shapes = graph_tensor(graph, {key_length, value_heads}, GGML_TYPE_F32)
  local head_queries = node(
    graph,
    lib.ggml_reshape_3d(
      ctx,
      node(graph, lib.ggml_repeat(ctx, queries, shapes)),
      key_length,
      1,
      value_heads
    )
  )
  local head_reads = node(
    graph,
    lib.ggml_reshape_3d(
      ctx,
      node(graph, lib.ggml_repeat(ctx, reads, shapes)),
      key_length,
      1,
      value_heads
    )
  )

  local carried = carry.state
  local stored_state: ffi.CData
  if carried is nil then
    stored_state = host_tensor(
      graph,
      {key_length, value_length, value_heads},
      GGML_TYPE_F32,
      active.state,
      state_floats * F32_BYTES
    )
  else
    stored_state = carried
  end
  local decayed = node(
    graph,
    lib.ggml_mul(
      ctx,
      stored_state,
      node(graph, lib.ggml_reshape_3d(ctx, decay, 1, 1, value_heads))
    )
  )
  local corrections = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_sub(
          ctx, values, node(graph, lib.ggml_mul_mat(ctx, decayed, head_reads))
        )
      ),
      node(graph, lib.ggml_reshape_3d(ctx, betas, 1, 1, value_heads))
    )
  )
  local updated = node(
    graph,
    lib.ggml_add(
      ctx,
      decayed,
      node(
        graph,
        lib.ggml_mul_mat(
          ctx,
          node(
            graph,
            lib.ggml_reshape_3d(ctx, head_reads, 1, key_length, value_heads)
          ),
          node(
            graph,
            lib.ggml_reshape_3d(ctx, corrections, 1, value_length, value_heads)
          )
        )
      )
    )
  )
  local scanned = node(
    graph,
    lib.ggml_scale(
      ctx, node(graph, lib.ggml_mul_mat(ctx, updated, head_queries)), head_scale
    )
  )
  local gated = node(
    graph,
    lib.ggml_mul(
      ctx,
      node(
        graph,
        lib.ggml_mul(
          ctx,
          node(graph, lib.ggml_rms_norm(ctx, scanned, eps)),
          descriptors[prefix .. "ssm_norm.weight"]
        )
      ),
      node(
        graph,
        lib.ggml_silu(
          ctx,
          node(
            graph, lib.ggml_reshape_3d(ctx, gates, value_length, 1, value_heads)
          )
        )
      )
    )
  )
  local output = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      descriptors[prefix .. "ssm_out.weight"],
      node(graph, lib.ggml_reshape_2d(ctx, gated, channels, 1))
    )
  )

  return {output = output, updated = updated, projected = projected}
end

-- the graphs a linear handle builds, each over the rows a pass holds
local function linear_builds(
  state: LinearState, prefix: string, width: integer, rows: integer
): {Build}
  if not holds(state.mapped, prefixed(prefix, LINEAR_NAMES)) then
    return {}
  end
  local scratch = measure_pointer()
  if scratch is nil then
    return {}
  end
  local kept = {history = scratch, state = scratch, count = 0} as LinearKept
  local entries = blank_rows(width, rows)
  local build = function(graph: Graph): {Output}
    local outputs: {Output} = {}
    local carry: LinearCarry = {taps = {}}
    for _, entry in ipairs(entries) do
      local made = linear_build(state, graph, entry, prefix, kept, carry)
      carry = {state = made.updated, taps = carry.taps}
      carry.taps[#carry.taps + 1] = made.projected
      outputs[#outputs + 1] = {tensor = made.output, count = #entry.hidden}
      outputs[#outputs + 1] = {tensor = made.updated, count = 0}
      outputs[#outputs + 1] = {tensor = made.projected, count = 0}
    end
    return outputs
  end
  return {{kind = "linear", build = build}}
end

local function linear_arena(
  state: LinearState, prefix: string, width: integer, rows: integer
): Pool | nil
  return grown(
    state.pool,
    state.owner,
    rows,
    function(most: integer): Pool | nil
      local builds = linear_builds(state, prefix, width, most)
      if #builds == 0 then
        return nil
      end
      return sized_pool(
        state.mapped.lib, state.mapped.device, builds, most, nil, nil
      )
    end
  )
end

local function linear_block(
  state: LinearState, rows: {compute.Row}, prefix: string
): {{number}} | nil, string | nil
  if #rows == 0 then
    return {}, nil
  end
  local mapped = state.mapped
  local shape = state.shape
  local checked: {integer: boolean} = {}
  local kepts: {LinearKept} = {}
  -- the position each sequence's next row of this call must name: a group runs
  -- on from the row before it rather than from the kept state alone
  local running: {string: integer} = {}

  for index, entry in ipairs(rows) do
    if not checked[#entry.hidden] then
      local fault = first_fault(
        mapped, "block", prefix, LINEAR_NAMES, LINEAR_TYPES
      )
      if fault then
        return nil, fault
      end
      fault = linear_shape_refusal(mapped, prefix, #entry.hidden, shape)
      if fault then
        return nil, fault
      end
      checked[#entry.hidden] = true
    end
    local held = state.prefixes[entry.sequence]
    local kept: LinearKept | nil = nil
    if not (held is nil) then
      kept = held[prefix]
    end
    local expected = 0
    if not (kept is nil) then
      expected = (kept as LinearKept).count
    end
    if entry.position < 0 then
      return nil, "block: prefix " .. prefix .. " position " .. entry.position
    end
    local next_position = running[entry.sequence]
    if not (next_position is nil) then
      expected = next_position
    end
    if entry.position ~= expected then
      return nil,
        "block: prefix "
          .. prefix
          .. " position "
          .. entry.position
          .. ", the next is "
          .. expected
    end
    running[entry.sequence] = entry.position + 1
    if kept is nil then
      local made, failure = linear_prefix_of(state, entry.sequence, prefix)
      if not (made is LinearKept) then
        return nil, failure
      end
      kept = made
    end
    kepts[index] = kept as LinearKept
  end

  local width = #((rows[1] as compute.Row).hidden)
  local pool = linear_arena(state, prefix, width, #rows)
  if pool is nil then
    return nil, "block: no arena for its graph"
  end

  local made, failure = graph_run(
    pool,
    prefix,
    "linear",
    "block",
    state.measure,
    function(graph: Graph): {Output}
      local outputs: {Output} = {}
      local carry: LinearCarry = {taps = {}}
      for index, entry in ipairs(rows) do
        if index > 1 and rows[index - 1].sequence ~= entry.sequence then
          carry = {taps = {}}
        end
        local built = linear_build(
          state, graph, entry, prefix, kepts[index], carry
        )
        carry = {state = built.updated, taps = carry.taps}
        carry.taps[#carry.taps + 1] = built.projected
        local count = #entry.hidden
        outputs[#outputs + 1] = {tensor = built.output, count = count}
        outputs[#outputs + 1] = {tensor = built.updated, count = 0}
        outputs[#outputs + 1] = {tensor = built.projected, count = 0}
      end
      return outputs
    end
  )
  if made is nil then
    return nil, failure
  end

  local kernel = shape.conv_kernel
  local slots = kernel - 1
  local value_cells = shape.value_heads * shape.value_length
  local channels = 2 * shape.key_heads * shape.key_length + value_cells
  local _, state_floats = linear_floats(shape)
  local outputs: {{number}} = {}
  for index, entry in ipairs(rows) do
    local active = kepts[index]
    outputs[index] = graph_numbers(made, made.outputs[index * 3 - 2])
    local group = index < #rows and rows[index + 1].sequence == entry.sequence
    if not group then
      -- the window and the state a sequence keeps are what its group's last
      -- row leaves them
      local first = index
      while first > 1 and rows[first - 1].sequence == entry.sequence do
        first = first - 1
      end
      read_bytes(
        made.lib,
        made.outputs[index * 3 - 1].tensor,
        active.state,
        state_floats * F32_BYTES,
        made.device
      )
      if kernel > 1 then
        local fresh = index - first + 1
        if fresh > slots then
          fresh = slots
        end
        local older = slots - fresh
        local taps: {Floats} = {}
        for tap = 1, fresh do
          local cells = ffi.new("float[?]", channels) as ffi.CData
          read_bytes(
            made.lib,
            made.outputs[(index - fresh + tap) * 3].tensor,
            cells,
            channels * F32_BYTES,
            made.device
          )
          taps[tap] = cells as Floats
        end
        -- the past tensor reads a channel's taps consecutively, oldest first
        local history = ffi.cast("float*", active.history) as Floats
        for channel = 0, channels - 1 do
          for slot = 0, older - 1 do
            local ahead = channel * slots + slot + fresh
            history[channel * slots + slot] = history[ahead]
          end
          for tap = 1, fresh do
            history[channel * slots + older + tap - 1] = taps[tap][channel]
          end
        end
      end
      active.count = entry.position + 1
    end
  end
  graph_closed(made)
  return outputs, nil
end

function compute.load(
  path?: string, offload?: boolean
): compute.Backend | nil, string | nil
  local location = path or DEFAULT_PATH
  local made, lib = pcall(load_library, location)
  if not made then
    return nil, refusal_line(location, lib as string)
  end
  local handle = lib as Library

  local known, version = pcall(probe_version, handle)
  if not known then
    return nil, refusal_line(location, version as string)
  end
  if (version as string) ~= EXPECTED_VERSION then
    return nil,
      refusal_line(
        location,
        "version "
          .. (version as string)
          .. " is not the pinned "
          .. EXPECTED_VERSION
      )
  end

  find_backends(handle, location)

  -- the card is reported even when the caller asked to stay on the CPU, and
  -- used only when they did not
  local found = handle.ggml_backend_dev_by_type(GGML_DEVICE_GPU)

  local backend = {
    path = location,
    version = version as string,
  } as compute.Backend
  backend.memory = function(_: compute.Backend): integer | nil, integer | nil
    if found is nil then
      return nil, nil
    end
    local free_bytes = ffi.new("size_t[1]")
    local total_bytes = ffi.new("size_t[1]")
    handle.ggml_backend_dev_memory(found, free_bytes, total_bytes)
    local sizes = free_bytes as Ints
    local totals = total_bytes as Ints
    return tonumber(sizes[0]) as integer, tonumber(totals[0]) as integer
  end
  LOADED[backend] = {
    lib = handle,
    device = found,
    offload = offload ~= false,
  }
  return backend
end

-- forgets the library a backend opened, so the pool's parent keeps no device
function compute.unload(backend: compute.Backend)
  LOADED[backend] = nil
end

function compute.attach(
  backend: compute.Backend, handle: weights.Handle
): compute.Maths | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "attach: backend was not loaded by compute.load"
  end
  local mapped, map_failure = mapped_of(backend, loaded, handle, "attach")
  if mapped is nil then
    return nil, map_failure
  end
  local attached: Mapped = mapped
  attached.hold()

  local finished = false
  local function shut(operation: string): string | nil
    if finished then
      return operation .. ": the maths are closed"
    end
    return nil
  end

  local maths = {} as compute.Maths
  maths.matvec = function(
    _: compute.Maths, name: string, vector: {number}, index?: integer
  ): {number} | nil, string | nil
    local line = shut("matvec")
    if not (line is nil) then
      return nil, line
    end
    return matrix_vector(mapped, name, vector, index)
  end
  maths.row = function(
    _: compute.Maths, name: string, id: integer
  ): {number} | nil, string | nil
    local line = shut("row")
    if not (line is nil) then
      return nil, line
    end
    return row(mapped, name, id)
  end
  maths.norm = function(
    _: compute.Maths, vector: {number}, gain: string, eps: number
  ): {number} | nil, string | nil
    local line = shut("norm")
    if not (line is nil) then
      return nil, line
    end
    return normalise(mapped, vector, gain, eps)
  end
  maths.rope = function(
    _: compute.Maths,
    vector: {number},
    position: integer,
    dimensions: integer,
    base: number
  ): {number} | nil, string | nil
    local line = shut("rope")
    if not (line is nil) then
      return nil, line
    end
    return rotary(mapped, vector, position, dimensions, base)
  end
  maths.softmax = function(
    _: compute.Maths, vector: {number}
  ): {number} | nil, string | nil
    local line = shut("softmax")
    if not (line is nil) then
      return nil, line
    end
    return softmax(mapped, vector)
  end
  maths.add = function(
    _: compute.Maths, left: {number}, right: {number}
  ): {number} | nil, string | nil
    local line = shut("add")
    if not (line is nil) then
      return nil, line
    end
    return add(mapped, left, right)
  end
  maths.swiglu = function(
    _: compute.Maths, gate: {number}, up: {number}
  ): {number} | nil, string | nil
    local line = shut("swiglu")
    if not (line is nil) then
      return nil, line
    end
    return swiglu(mapped, gate, up)
  end
  maths.sigmoid_gate = function(
    _: compute.Maths, gate: {number}, values: {number}
  ): {number} | nil, string | nil
    local line = shut("sigmoid_gate")
    if not (line is nil) then
      return nil, line
    end
    return sigmoid_gate(mapped, gate, values)
  end
  maths.close = function(_: compute.Maths)
    if finished then
      return
    end
    finished = true
    attached.close()
  end
  return maths
end

-- the attention handle over a mapped handle: `pool` is the one the model that
-- built it holds, and a handle built on its own is given one sized from its own
-- largest graph
local function attention_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure: compute.Measure | nil
): compute.Attention | nil, string | nil
  local refusal = geometry_refusal(geometry)
  if refusal then
    return nil, refusal
  end
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "attend: backend was not loaded by compute.load"
  end

  local mapped, map_failure = mapped_of(backend, loaded, handle, "attend")
  if mapped is nil then
    return nil, map_failure
  end
  mapped.hold()

  local state: AttentionState = {
    mapped = mapped,
    geometry = geometry,
    caches = {},
    capacities = {},
    pool = pool,
    measure = measure,
    owner = pool.owned,
  }
  local finished = false
  local function shut(operation: string): string | nil
    if finished then
      return operation .. ": the attention is closed"
    end
    return nil
  end

  local attention = {} as compute.Attention
  attention.block = function(
    _: compute.Attention, rows: {compute.Row}, prefix: string
  ): {{number}} | nil, string | nil
    local line = shut("block")
    if not (line is nil) then
      return nil, line
    end
    return attention_block(state, rows, prefix)
  end
  attention.cache_bytes = function(
    _: compute.Attention, sequence: string, prefix: string
  ): integer | nil
    if finished then
      return nil
    end
    local held = state.caches[sequence]
    if held is nil then
      return nil
    end
    local cache = held[prefix]
    if cache is nil then
      return nil
    end
    return cache.bytes
  end
  attention.reserve = function(
    _: compute.Attention,
    sequence: string,
    positions: integer,
    prefixes: {string}
  ): string | nil
    local line = shut("reserve")
    if not (line is nil) then
      return line
    end
    return attention_reserve(state, sequence, positions, prefixes)
  end
  attention.release = function(_: compute.Attention, sequence: string)
    if finished then
      return
    end
    attention_release(state, sequence)
  end
  attention.snapshot = function(
    _: compute.Attention, sequence: string, prefix: string, positions: integer
  ): string | nil, string | nil
    local line = shut("snapshot")
    if not (line is nil) then
      return nil, line
    end
    return attention_snapshot(state, sequence, prefix, positions)
  end
  attention.restore = function(
    _: compute.Attention,
    sequence: string,
    prefix: string,
    positions: integer,
    payload: string
  ): string | nil
    local line = shut("restore")
    if not (line is nil) then
      return line
    end
    return attention_restore(state, sequence, prefix, positions, payload)
  end
  attention.close = function(_: compute.Attention)
    if finished then
      return
    end
    finished = true
    attention_closed(state)
    if state.owner then
      local held = state.pool.pool
      if not (held is nil) then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return attention
end

function compute.attend(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.Attention | nil, string | nil
  return attention_of(
    backend, handle, geometry, {pool = nil, owned = true}, nil
  )
end

-- the feed forward handle over a mapped handle, sharing a model's arena when
-- one is given and holding one of its own otherwise
local function feed_forward_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure: compute.Measure | nil
): compute.FeedForward | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "feed_forward: backend was not loaded by compute.load"
  end
  if geometry.experts_used < 1 then
    return nil, "feed_forward: experts_used " .. geometry.experts_used
  end
  if geometry.eps < 0 then
    return nil, "feed_forward: eps " .. geometry.eps
  end

  local mapped, map_failure = mapped_of(backend, loaded, handle, "feed_forward")
  if mapped is nil then
    return nil, map_failure
  end
  mapped.hold()

  local state: FeedForwardState = {
    mapped = mapped,
    geometry = geometry,
    pool = pool,
    measure = measure,
    owner = pool.owned,
  }
  local finished = false
  local feed_forward = {} as compute.FeedForward
  feed_forward.block = function(
    _: compute.FeedForward,
    rows: {compute.Row},
    prefix: string,
    watch?: {compute.Watch}
  ): {{number}} | nil, string | nil
    if finished then
      return nil, "block: the feed forward is closed"
    end
    return feed_forward_block(state, rows, prefix, watch)
  end
  feed_forward.scores = function(
    _: compute.FeedForward, hidden: {{number}}
  ): {{number}} | nil, string | nil
    if finished then
      return nil, "scores: the feed forward is closed"
    end
    return feed_forward_scores(state, hidden)
  end
  feed_forward.close = function(_: compute.FeedForward)
    if finished then
      return
    end
    finished = true
    state.mapped.close()
    if state.owner then
      local held = state.pool.pool
      if not (held is nil) then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return feed_forward
end

function compute.feed_forward(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.FeedForward | nil, string | nil
  return feed_forward_of(
    backend, handle, geometry, {pool = nil, owned = true}, nil
  )
end

-- the linear-attention handle over a mapped handle, sharing a model's arena
-- when one is given and holding one of its own otherwise
local function linear_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure: compute.Measure | nil
): compute.LinearAttention | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "linear_attention: backend was not loaded by compute.load"
  end
  local shape, geometry_line = linear_shape_of(geometry)
  if not (shape is LinearShape) then
    return nil, geometry_line
  end

  local mapped, map_failure = mapped_of(
    backend, loaded, handle, "linear_attention"
  )
  if mapped is nil then
    return nil, map_failure
  end
  mapped.hold()

  local state: LinearState = {
    mapped = mapped,
    shape = shape,
    eps = geometry.eps,
    prefixes = {},
    pool = pool,
    measure = measure,
    owner = pool.owned,
  }
  local finished = false
  local function shut(operation: string): string | nil
    if finished then
      return operation .. ": the linear attention is closed"
    end
    return nil
  end

  local linear = {} as compute.LinearAttention
  linear.block = function(
    _: compute.LinearAttention, rows: {compute.Row}, prefix: string
  ): {{number}} | nil, string | nil
    local line = shut("block")
    if not (line is nil) then
      return nil, line
    end
    return linear_block(state, rows, prefix)
  end
  linear.release = function(_: compute.LinearAttention, sequence: string)
    if finished then
      return
    end
    linear_release(state, sequence)
  end
  linear.snapshot = function(
    _: compute.LinearAttention,
    sequence: string,
    prefix: string,
    positions: integer
  ): string | nil, string | nil
    local line = shut("snapshot")
    if not (line is nil) then
      return nil, line
    end
    return linear_snapshot(state, sequence, prefix, positions)
  end
  linear.restore = function(
    _: compute.LinearAttention,
    sequence: string,
    prefix: string,
    positions: integer,
    payload: string
  ): string | nil
    local line = shut("restore")
    if not (line is nil) then
      return line
    end
    return linear_restore(state, sequence, prefix, positions, payload)
  end
  linear.close = function(_: compute.LinearAttention)
    if finished then
      return
    end
    finished = true
    linear_closed(state)
    if state.owner then
      local held = state.pool.pool
      if not (held is nil) then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return linear
end

function compute.linear_attention(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.LinearAttention | nil, string | nil
  return linear_of(backend, handle, geometry, {pool = nil, owned = true}, nil)
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

-- every row's embedding numbers in one graph, so a pass of several rows reads
-- the embedding tensor once
local function embedding_build(
  mapped: Mapped, graph: Graph, ids: {integer}
): ffi.CData
  return node(
    graph,
    graph.lib.ggml_get_rows(
      graph.ctx, mapped.descriptors["token_embd.weight"], ints_of(graph, ids)
    )
  )
end

-- the graphs a pass of `rows` rows builds, each measured at the model's whole
-- position count, where a row's attention scores are widest: the largest of
-- them sizes the arena that pass runs in
local function model_builds(
  mapped: Mapped,
  geometry: compute.Geometry,
  kinds: {boolean},
  count: integer,
  width: integer,
  rows: integer
): {Build}
  local first_full: integer | nil = nil
  local first_linear: integer | nil = nil
  for index = 1, count do
    if kinds[index] then
      if first_full is nil then
        first_full = index - 1
      end
    elseif first_linear is nil then
      first_linear = index - 1
    end
  end

  local builds: {Build} = {}
  -- the graph gathering a pass's embedding rows, whose ids the build reads as
  -- its own inputs
  if holds(mapped, {"token_embd.weight"}) then
    local ids: {integer} = {}
    for index = 1, rows do
      ids[index] = 0
    end
    builds[#builds + 1] = {
      kind = "embed",
      build = function(graph: Graph): {Output}
        return {
          {tensor = embedding_build(mapped, graph, ids), count = rows * width},
        }
      end,
    }
  end
  local feed_forward_state: FeedForwardState = {
    mapped = mapped,
    geometry = geometry,
    pool = {pool = nil, owned = false},
    owner = false,
  }
  local full_prefix = prefix_named(mapped, "post_attention_norm.weight")
  local hidden_count = hidden_width(mapped, "post_attention_norm.weight")
  if not (full_prefix is nil) and not (hidden_count is nil) then
    for _, entry in ipairs(
      feed_forward_builds(feed_forward_state, full_prefix, hidden_count, rows)
    ) do
      builds[#builds + 1] = entry
    end
  end
  if not (first_full is nil) then
    local attention_state: AttentionState = {
      mapped = mapped,
      geometry = geometry,
      caches = {},
      capacities = {},
      pool = {pool = nil, owned = false},
      owner = false,
    }
    local prefix = "blk." .. (first_full as integer) .. "."
    if holds(mapped, prefixed(prefix, ATTENTION_NAMES)) then
      local entry = attention_measure(attention_state, prefix, rows)
      if not (entry is nil) then
        builds[#builds + 1] = entry
      end
    end
  end
  if not (first_linear is nil) then
    local linear_state: LinearState = {
      mapped = mapped,
      shape = (linear_shape_of(geometry) as LinearShape),
      eps = geometry.eps,
      prefixes = {},
      pool = {pool = nil, owned = false},
      owner = false,
    }
    local prefix = "blk." .. (first_linear as integer) .. "."
    local linear_count = hidden_width(mapped, "attn_norm.weight")
    if not (linear_count is nil) then
      for _, entry in ipairs(
        linear_builds(linear_state, prefix, linear_count, rows)
      ) do
        builds[#builds + 1] = entry
      end
    end
  end
  return builds
end

local function model_plan(
  backend: compute.Backend,
  handle: weights.Handle,
  positions: integer | nil,
  cache_type: compute.CacheType | nil,
  sequences: integer
): ModelPlan | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "model: backend was not loaded by compute.load"
  end
  local cached: compute.CacheType = cache_type or "f16"
  local kind = cache_kind_of(cached)
  if kind is nil then
    return nil, "model: cache_type " .. cached
  end
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

  local mapped, map_line = mapped_of(backend, loaded, handle, "model")
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
    local held = cache_kind_of(cached) as CacheKind
    if head_length % held.block_elements ~= 0 then
      return nil,
        "model: cache "
          .. cached
          .. " cannot hold a head of "
          .. head_length
          .. " numbers"
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
    cache_type = cached,
  }

  -- the one arena the model's graphs are built in, sized from the largest
  -- graph a pass builds at the model's whole position count
  local measure: compute.Measure = {
    passes = 0,
    tokens = 0,
    compute = 0,
    sampling = 0,
    ready = 0,
    routed = 0,
    routed_bytes = 0,
    graphs = 0,
  }
  local sized = function(rows: integer): {Build}
    return model_builds(mapped, geometry, kinds, count, math.floor(width), rows)
  end
  local builds = sized(sequences)
  local arena = sized_pool(
    mapped.lib, mapped.device, builds, sequences, nil, nil
  )
  if arena is nil then
    return nil, "model: no arena for its graphs"
  end
  local box: Box = {pool = arena, owned = false}

  local feed_forward, feed_line = feed_forward_of(
    backend, handle, geometry, box, measure
  )
  if feed_forward is nil then
    pool_free(arena)
    return nil, "model: " .. (feed_line as string)
  end

  local attention: compute.Attention | nil = nil
  if has_full then
    local made, line = attention_of(backend, handle, geometry, box, measure)
    if made is nil then
      feed_forward:close()
      pool_free(arena)
      return nil, "model: " .. (line as string)
    end
    attention = made
  end

  local linear: compute.LinearAttention | nil = nil
  if has_linear then
    local made, line = linear_of(backend, handle, geometry, box, measure)
    if made is nil then
      feed_forward:close()
      local blocked = attention
      if not (blocked is nil) then
        blocked:close()
      end
      pool_free(arena)
      return nil, "model: " .. (line as string)
    end
    linear = made
  end

  mapped.hold()
  return {
    mapped = mapped,
    kinds = kinds,
    count = count,
    width = math.floor(width),
    positions = taken,
    sequences = sequences,
    pool = box,
    sized = sized,
    measure = measure,
    attention = attention,
    linear = linear,
    feed_forward = feed_forward as compute.FeedForward,
  }
end

-- what the embedding row of this token refuses, before any graph is built
local function embedding_refusal(
  mapped: Mapped, width: integer, token_id: integer
): string | nil
  local tensor = mapped.tensors["token_embd.weight"]
  if tensor is nil then
    return "forward: no tensor token_embd.weight for token " .. token_id
  end
  local dims = tensor.dims
  if #dims ~= 2 then
    return "forward: token_embd.weight id "
      .. token_id
      .. " holds "
      .. #dims
      .. " dimensions"
  end
  local rows = (dims[2] or 0) as integer
  if rows < 1 then
    return "forward: token_embd.weight id " .. token_id .. " holds no rows"
  end
  if token_id < 0 or token_id >= rows then
    return "forward: tensor token_embd.weight id " .. token_id .. " of " .. rows
  end
  if (dims[1] or 0) ~= width then
    return "forward: token_embd.weight id "
      .. token_id
      .. " holds "
      .. (dims[1] or 0)
      .. " numbers per row, the width is "
      .. width
  end
  return nil
end

local function sequence_count(state: ModelState): integer
  local count = 0
  for _ in pairs(state.sequences) do
    count = count + 1
  end
  return count
end

local function sequence_reserve(
  state: ModelState, sequence: string, positions: integer
): string | nil
  if state.sequences[sequence] ~= nil then
    return "reserve: sequence " .. sequence .. " is already held"
  end
  if sequence_count(state) >= state.limit then
    return "reserve: sequence "
      .. sequence
      .. ", "
      .. state.limit
      .. " sequences are held"
  end
  if positions < 1 or positions > state.positions then
    return "reserve: positions " .. positions .. " for " .. sequence
  end
  local attention = state.attention
  if not (attention is nil) and #state.full_prefixes > 0 then
    local line = attention:reserve(sequence, positions, state.full_prefixes)
    if not (line is nil) then
      return named_failure("reserve", line)
    end
  end
  state.sequences[sequence] = {capacity = positions, taken = 0}
  return nil
end

local function sequence_release(state: ModelState, sequence: string)
  if state.sequences[sequence] == nil then
    return
  end
  local attention = state.attention
  if not (attention is nil) then
    attention:release(sequence)
  end
  local linear = state.linear
  if not (linear is nil) then
    linear:release(sequence)
  end
  state.sequences[sequence] = nil
end

local CONTEXT_MAGIC: integer = 0x54584354
local CONTEXT_VERSION: integer = 1
local HASH_MODULUS: integer = 4294967296

local function packed_u32(value: integer): string
  return string.char(
    value % 256,
    math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256,
    math.floor(value / 16777216) % 256
  )
end

local function read_u32(text: string, at: integer): integer | nil
  if at < 0 or at + 4 > #text then
    return nil
  end
  local first, second, third, fourth = text:byte(at + 1, at + 4)
  return first + second * 256 + third * 65536 + fourth * 16777216
end

local function salted(hash: integer, word: string): integer
  local running = hash
  for index = 1, #word do
    running = (running * 31 + word:byte(index)) % HASH_MODULUS
  end
  return running
end

-- what a file records of the model it came from: every tensor's name, its
-- dimensions and its type, in name order
local function fingerprint_of(tensors: {string: Tensor}): integer
  local names: {string} = {}
  for name in pairs(tensors) do
    names[#names + 1] = name
  end
  table.sort(names)
  local hash = 2166136261
  hash = salted(hash, "tllm.context/" .. CONTEXT_VERSION)
  for _, name in ipairs(names) do
    local tensor = tensors[name] as Tensor
    hash = salted(hash, name)
    for _, dimension in ipairs(tensor.dims) do
      hash = salted(hash, tostring(dimension) .. ",")
    end
    hash = salted(hash, tensor.kind.name .. ";")
  end
  return hash
end

-- the rows of one sequence in one pass are a group: consecutive positions
-- running up from the number of positions that sequence has taken
local function group_refusal(
  state: ModelState, inputs: {compute.Input}
): string | nil
  local named: {string: boolean} = {}
  for index, input in ipairs(inputs) do
    local held = state.sequences[input.sequence]
    if held is nil then
      return "pass: sequence " .. input.sequence .. " is not reserved"
    end
    local first = index == 1 or inputs[index - 1].sequence ~= input.sequence
    local next_position = (held as SequenceState).taken
    if first then
      if named[input.sequence] then
        return "pass: sequence " .. input.sequence .. " is handed twice"
      end
      named[input.sequence] = true
    else
      next_position = inputs[index - 1].position + 1
    end
    if input.position ~= next_position then
      return "pass: sequence "
        .. input.sequence
        .. " position "
        .. input.position
        .. ", the next is "
        .. next_position
    end
  end
  return nil
end

-- the arena a pass runs in: the model's, grown to the largest graph a pass of
-- these rows builds when it holds more rows than the arena was measured at,
-- the blocks of the run building over the arena the box then holds
local function pass_arena(
  state: ModelState, rows: integer
): Pool | nil, string | nil
  local box = state.pool
  local held = box.pool
  if held is nil then
    return nil, "pass: no arena for its graphs"
  end
  if held.rows >= rows then
    return held, nil
  end
  -- sized from a measurement of this run's own largest graph, and never
  -- smaller than the arena it replaces
  local old = held as Pool
  local wider = sized_pool(
    old.lib,
    state.mapped.device,
    state.sized(rows),
    rows,
    math.ceil(old.tensors / math.max(old.rows, 1)),
    old.bytes
  )
  if wider is nil then
    return nil, "pass: no arena for " .. rows .. " rows"
  end
  pool_free(held as Pool)
  box.pool = wider
  return wider, nil
end

local function model_pass(
  state: ModelState,
  inputs: {compute.Input},
  measure: compute.Measure,
  sink: (function(compute.Routing) | nil)
): {{number}} | nil, string | nil
  if #inputs == 0 then
    return nil, "pass: no rows"
  end
  local refusal = group_refusal(state, inputs)
  if not (refusal is nil) then
    return nil, refusal
  end
  local arena, no_room = pass_arena(state, #inputs)
  if arena is nil then
    return nil, no_room
  end

  local started = clock.now()
  local first = measure.tokens

  local ids: {integer} = {}
  for index, input in ipairs(inputs) do
    local refused = embedding_refusal(state.mapped, state.width, input.token)
    if not (refused is nil) then
      return nil, refused
    end
    ids[index] = input.token
  end

  local embedded, embed_line = graph_run(
    arena as Pool,
    "",
    "embed",
    "forward",
    measure,
    function(graph: Graph): {Output}
      return {
        {
          tensor = embedding_build(state.mapped, graph, ids),
          count = #ids * state.width,
        },
      }
    end
  )
  if embedded is nil then
    return nil, embed_line
  end
  local numbers = graph_numbers(embedded, embedded.outputs[1])
  graph_closed(embedded)

  local rows: {compute.Row} = {}
  for index, input in ipairs(inputs) do
    rows[index] = {
      sequence = input.sequence,
      hidden = slice(numbers, (index - 1) * state.width, state.width),
      position = input.position,
    }
  end

  for index = 0, state.count - 1 do
    local prefix = "blk." .. index .. "."
    local contributions: {{number}} | nil
    local line: string | nil
    if state.kinds[index + 1] then
      local attention = state.attention as compute.Attention
      contributions, line = attention:block(rows, prefix)
    else
      local linear = state.linear as compute.LinearAttention
      contributions, line = linear:block(rows, prefix)
    end
    if contributions is nil then
      return nil, line
    end

    local mixed: {compute.Row} = {}
    for row_index, entry in ipairs(rows) do
      mixed[row_index] = {
        sequence = entry.sequence,
        hidden = summed(entry.hidden, contributions[row_index]),
        position = entry.position,
      }
    end
    local watchs: {compute.Watch} = {}
    for row_index = 1, #mixed do
      watchs[row_index] = {
        layer = index,
        token = first + row_index - 1,
        measure = measure,
        sink = sink,
      }
    end
    local blocked, block_line = state.feed_forward:block(mixed, prefix, watchs)
    if blocked is nil then
      return nil, block_line
    end
    local carried: {compute.Row} = {}
    for row_index, entry in ipairs(mixed) do
      carried[row_index] = {
        sequence = entry.sequence,
        hidden = blocked[row_index],
        position = entry.position,
      }
    end
    rows = carried
  end

  local hidden_rows: {{number}} = {}
  for index, entry in ipairs(rows) do
    hidden_rows[index] = entry.hidden
  end
  local scores, score_line = state.feed_forward:scores(hidden_rows)
  if scores is nil then
    return nil, score_line
  end

  for _, input in ipairs(inputs) do
    local held = state.sequences[input.sequence] as SequenceState
    held.taken = held.taken + 1
  end
  measure.passes = measure.passes + 1
  measure.tokens = first + #inputs
  measure.compute = measure.compute + (clock.now() - started)
  return scores, nil
end

function compute.model(
  backend: compute.Backend,
  handle: weights.Handle,
  positions: integer | nil,
  cache_type: compute.CacheType | nil,
  sequences?: integer,
  ubatch?: integer
): compute.Model | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "model: backend was not loaded by compute.load"
  end
  local limit = sequences or 1
  if limit < 1 then
    return nil, "model: sequences is " .. limit
  end
  local rows = ubatch or DEFAULT_UBATCH
  if rows < 1 then
    return nil, "model: ubatch is " .. rows
  end
  local cached: compute.CacheType = cache_type or "f16"
  local plan, plan_line = model_plan(
    backend, handle, positions, cache_type, limit
  )
  if plan is nil then
    return nil, plan_line
  end

  local full_prefixes: {string} = {}
  local linear_prefixes: {string} = {}
  for index = 0, plan.count - 1 do
    local prefix = "blk." .. index .. "."
    if plan.kinds[index + 1] then
      full_prefixes[#full_prefixes + 1] = prefix
    else
      linear_prefixes[#linear_prefixes + 1] = prefix
    end
  end

  local state: ModelState = {
    mapped = plan.mapped,
    kinds = plan.kinds,
    count = plan.count,
    width = plan.width,
    attention = plan.attention,
    linear = plan.linear,
    feed_forward = plan.feed_forward,
    sequences = {},
    limit = limit,
    positions = plan.positions,
    fingerprint = fingerprint_of(plan.mapped.tensors),
    cache_code = (cache_kind_of(cached) as CacheKind).code,
    full_prefixes = full_prefixes,
    linear_prefixes = linear_prefixes,
    pool = plan.pool,
    sized = plan.sized,
    measure = plan.measure,
  }

  local model = {} as compute.Model
  model.positions = plan.positions
  model.sequences = limit
  model.ubatch = rows
  model.measure = plan.measure
  local measure = model.measure
  model.routing_sink = nil
  local finished = false
  local function shut(operation: string): string | nil
    if finished then
      return operation .. ": the model is closed"
    end
    return nil
  end

  model.forward = function(
    _: compute.Model, token_id: integer, position: integer
  ): {number} | nil, string | nil
    local closed_line = shut("forward")
    if not (closed_line is nil) then
      return nil, closed_line
    end
    if state.sequences[""] == nil then
      local line = sequence_reserve(state, "", state.positions)
      if not (line is nil) then
        return nil, line
      end
    end
    local held = state.sequences[""] as SequenceState
    if position < 0 or position ~= held.taken then
      return nil,
        "forward: position " .. position .. ", the next is " .. held.taken
    end
    local scores, line = model_pass(
      state,
      {{sequence = "", token = token_id, position = position}},
      measure,
      model.routing_sink
    )
    if scores is nil then
      return nil, line
    end
    return scores[1], nil
  end
  model.reserve = function(
    _: compute.Model, sequence: string, capacity: integer
  ): string | nil
    local closed_line = shut("reserve")
    if not (closed_line is nil) then
      return closed_line
    end
    return sequence_reserve(state, sequence, capacity)
  end
  model.release = function(_: compute.Model, sequence: string)
    if finished then
      return
    end
    sequence_release(state, sequence)
  end
  model.cache_bytes = function(
    _: compute.Model, sequence: string, prefix: string
  ): integer | nil
    if finished then
      return nil
    end
    if state.sequences[sequence] == nil then
      return nil
    end
    local attention = state.attention
    if attention is nil then
      return nil
    end
    return attention:cache_bytes(sequence, prefix)
  end
  model.capacity = function(_: compute.Model, sequence: string): integer | nil
    if finished then
      return nil
    end
    local held = state.sequences[sequence]
    if held is nil then
      return nil
    end
    return held.capacity
  end
  -- gives the caches, the arenas and the handle back: nothing of this model
  -- outlives it, and the handle can be released once it is closed
  model.close = function(_: compute.Model)
    if finished then
      return
    end
    finished = true
    local attention = state.attention
    if not (attention is nil) then
      attention:close()
    end
    local linear = state.linear
    if not (linear is nil) then
      linear:close()
    end
    state.feed_forward:close()
    state.mapped.close()
    local pool = state.pool.pool
    if not (pool is nil) then
      pool_free(pool)
      state.pool.pool = nil
    end
  end
  model.save = function(
    _: compute.Model, sequence: string, ids: {integer}, path: string
  ): string | nil
    local closed_line = shut("save")
    if not (closed_line is nil) then
      return closed_line
    end
    local held = state.sequences[sequence]
    if held is nil then
      return "save: sequence " .. sequence .. " is not reserved"
    end
    local taken = held.taken
    if #ids ~= taken then
      return "save: sequence "
        .. sequence
        .. " took "
        .. taken
        .. " positions, not "
        .. #ids
        .. " ids"
    end

    local partial = path .. ".part"
    local made_file = io.open(partial, "wb")
    if made_file is nil then
      return "save: cannot write " .. path
    end
    local file = made_file
    local function failed(line: string): string
      file:close()
      os.remove(partial)
      return line
    end

    local header: {string} = {}
    header[#header + 1] = packed_u32(CONTEXT_MAGIC)
    header[#header + 1] = packed_u32(CONTEXT_VERSION)
    header[#header + 1] = packed_u32(#ids)
    for _, id in ipairs(ids) do
      header[#header + 1] = packed_u32(id)
    end
    header[#header + 1] = packed_u32(state.fingerprint)
    header[#header + 1] = packed_u32(state.cache_code)
    header[#header + 1] = packed_u32(held.capacity)
    header[#header + 1] = packed_u32(taken)
    header[#header + 1] = packed_u32(state.count)
    if not file:write(table.concat(header)) then
      return failed("save: cannot write " .. path)
    end

    -- one block at a time, so a turn's cache never stands in memory twice
    for index = 0, state.count - 1 do
      local prefix = "blk." .. index .. "."
      local full = state.kinds[index + 1]
      local kind = 1
      if not full then
        kind = 0
      end
      local payload: string | nil = nil
      local line: string | nil = nil
      if full then
        payload, line = (state.attention as compute.Attention):snapshot(
          sequence, prefix, taken
        )
      else
        payload, line = (state.linear as compute.LinearAttention):snapshot(
          sequence, prefix, taken
        )
      end
      if payload is nil then
        return failed("save: sequence " .. sequence .. ": " .. (line as string))
      end
      local bytes = payload as string
      if not file:write(
        packed_u32(kind), packed_u32(taken), packed_u32(#bytes), bytes
      ) then
        return failed("save: cannot write " .. path)
      end
    end

    file:close()
    if not os.rename(partial, path) then
      os.remove(partial)
      return "save: cannot write " .. path
    end
    return nil
  end
  model.load = function(
    _: compute.Model, sequence: string, path: string
  ): {integer} | nil, string | nil
    local closed_line = shut("load")
    if not (closed_line is nil) then
      return nil, closed_line
    end
    local source = fs.read(path)
    if source is nil then
      return nil, "load: cannot read " .. path
    end
    local text = source as string
    local function refused(reason: string): string
      return "load: " .. path .. ": " .. reason
    end

    local magic = read_u32(text, 0)
    if magic ~= CONTEXT_MAGIC then
      return nil, refused("not a context file")
    end
    local version = read_u32(text, 4)
    if version is nil then
      return nil, refused("shorter than its version")
    end
    if version ~= CONTEXT_VERSION then
      return nil, refused("version " .. version .. " is not known")
    end
    local count = read_u32(text, 8)
    if count is nil then
      return nil, refused("shorter than its ids")
    end
    local at = 12 + 4 * count
    if at > #text then
      return nil, refused("shorter than its " .. count .. " ids")
    end
    local ids: {integer} = {}
    for index = 1, count do
      ids[index] = read_u32(text, 12 + 4 * (index - 1)) as integer
    end

    local fingerprint = read_u32(text, at)
    if fingerprint is nil then
      return nil, refused("shorter than its fingerprint")
    end
    local cache_code = read_u32(text, at + 4)
    if cache_code is nil then
      return nil, refused("shorter than its cache type")
    end
    local capacity = read_u32(text, at + 8)
    if capacity is nil then
      return nil, refused("shorter than its capacity")
    end
    local taken = read_u32(text, at + 12)
    if taken is nil then
      return nil, refused("shorter than its positions")
    end
    local blocks = read_u32(text, at + 16)
    if blocks is nil then
      return nil, refused("shorter than its blocks")
    end
    at = at + 20

    if fingerprint ~= state.fingerprint then
      return nil, refused("another model's tensors")
    end
    if cache_code ~= state.cache_code then
      return nil, refused("another cache type")
    end
    if capacity > state.positions then
      return nil,
        refused("capacity " .. capacity .. " past the " .. state.positions)
    end
    if taken > capacity then
      return nil, refused("taken " .. taken .. " past capacity " .. capacity)
    end
    if count ~= taken then
      return nil, refused(count .. " ids for " .. taken .. " positions")
    end
    if blocks ~= state.count then
      return nil, refused(blocks .. " blocks for " .. state.count)
    end

    local reserve_line = sequence_reserve(state, sequence, capacity)
    if not (reserve_line is nil) then
      return nil, reserve_line
    end
    local function give_up(reason: string): string
      sequence_release(state, sequence)
      return refused(reason)
    end

    for index = 0, state.count - 1 do
      local prefix = "blk." .. index .. "."
      local kind = read_u32(text, at)
      if kind is nil then
        return nil, give_up("shorter than " .. prefix)
      end
      local took = read_u32(text, at + 4)
      if took is nil then
        return nil, give_up("shorter than " .. prefix)
      end
      local length = read_u32(text, at + 8)
      if length is nil then
        return nil, give_up("shorter than " .. prefix)
      end
      at = at + 12
      local wanted = 0
      if state.kinds[index + 1] then
        wanted = 1
      end
      if kind ~= wanted then
        return nil, give_up(prefix .. " holds another kind of block")
      end
      if took ~= taken then
        return nil,
          give_up(prefix .. " took " .. took .. " positions, not " .. taken)
      end
      if at + length > #text then
        return nil, give_up("shorter than " .. prefix)
      end
      local payload = text:sub(at + 1, at + length)
      at = at + length
      local line: string | nil = nil
      if wanted == 1 then
        line = (state.attention as compute.Attention):restore(
          sequence, prefix, took, payload
        )
      else
        line = (state.linear as compute.LinearAttention):restore(
          sequence, prefix, took, payload
        )
      end
      if not (line is nil) then
        return nil, give_up(line as string)
      end
    end
    (state.sequences[sequence] as SequenceState).taken = taken
    return ids, nil
  end
  model.pass = function(
    _: compute.Model, inputs: {compute.Input}
  ): {{number}} | nil, string | nil
    local closed_line = shut("pass")
    if not (closed_line is nil) then
      return nil, closed_line
    end
    return model_pass(state, inputs, measure, model.routing_sink)
  end
  return model
end

return compute
