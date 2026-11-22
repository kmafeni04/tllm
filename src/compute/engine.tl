-- the ggml library and the plumbing every block builds on: the backends a
-- machine offers, the host arena a session builds graphs in, the pools a run's
-- graphs live in, the graph lifecycle, and the copies of weights on a device

local ffi = require("ffi")
local clock = require("tllm.clock")
local compute = require("tllm.compute.api")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Floats = {integer: number}
local type Ints = {integer: integer}

local type ArenaError = defs.ArenaError
local type Builder = defs.Builder
local type Build = defs.Build
local type Computed = defs.Computed
local type Copy = defs.Copy
local type Graph = defs.Graph
local type Holder = defs.Holder
local type Library = defs.Library
local type Loaded = defs.Loaded
local type Mapped = defs.Mapped
local type Output = defs.Output
local type Params = defs.Params
local type Pool = defs.Pool
local type Session = defs.Session
local type TensorView = defs.TensorView
local type Threads = defs.Threads

-- struct ggml_tensor here is the fields up to data, byte-exact for the pinned
-- ggml commit: tensor_of writes the mapping into the library's tensor memory
ffi.cdef(
  [[
typedef struct ggml_context ggml_context;
typedef struct ggml_tensor ggml_tensor;
typedef struct ggml_cgraph ggml_cgraph;
typedef struct ggml_backend ggml_backend;
typedef struct ggml_backend_reg ggml_backend_reg;
typedef struct ggml_threadpool ggml_threadpool;
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

struct ggml_threadpool_params {
  bool cpumask[512];
  int n_threads;
  int prio;
  uint32_t poll;
  bool strict_cpu;
  bool paused;
};

void ggml_threadpool_params_init(
  struct ggml_threadpool_params *params, int n_threads);
ggml_backend_reg *ggml_backend_reg_by_name(const char *name);
void *ggml_backend_reg_get_proc_address(
  ggml_backend_reg *reg, const char *name);

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
int ggml_backend_graph_compute_async(ggml_backend *backend, ggml_cgraph *cgraph);
void ggml_backend_synchronize(ggml_backend *backend);
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
local record Libc
  aligned_alloc: function(integer, integer): ffi.CData
  free: function(ffi.CData)
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
-- the threadpool entries are built into the CPU module ggml loads at run time
-- and not into the library handle, so they are looked up through the registry
-- entry the library answers: a library without that module answers none
local function threads_of(lib: Library): Threads | nil
  local reg = lib.ggml_backend_reg_by_name("CPU")
  if reg is nil then
    return nil
  end
  local make = lib.ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new")
  local give = lib.ggml_backend_reg_get_proc_address(
    reg, "ggml_threadpool_free"
  )
  local set = lib.ggml_backend_reg_get_proc_address(
    reg, "ggml_backend_cpu_set_threadpool"
  )
  local count = lib.ggml_backend_reg_get_proc_address(
    reg, "ggml_backend_set_n_threads"
  )
  if make is nil or give is nil or set is nil or count is nil then
    return nil
  end
  return {
    make = make,
    give = give,
    set = set,
    set_n_threads = count,
    n_threads = nil,
    pool = nil,
  } as Threads
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
  bytes: integer,
  counted?: boolean
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
      local started = clock.now()
      graph.lib.ggml_backend_tensor_set(tensor, location, 0, bytes)
      if counted == true then
        graph.copied = (graph.copied or 0) + (clock.now() - started)
      end
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
-- the one arena a handle builds its graphs in, and the device buffers its
-- blocks' graphs run in: the arena is allocated here and freed by the handle
local function pool_new(
  lib: Library,
  copy: Copy | nil,
  host_bytes: integer,
  device_bytes: {string: integer},
  device_held: {string: integer},
  rows: integer,
  tensors: integer,
  nodes: integer,
  threads?: Threads
): Pool | nil
  -- the numbers a graph keeps sit in a buffer after its descriptors, so the
  -- arena starts and ends on the alignment that buffer needs
  local size = round_up(host_bytes + ARENA_SLACK, TENSOR_ALIGN)
  local block = libc.aligned_alloc(TENSOR_ALIGN, size)
  if block is nil then
    return nil
  end
  local pool = {
    lib = lib,
    host = finalised(block),
    bytes = size,
    rows = rows,
    tensors = tensors,
    nodes = nodes,
    backend = device_of_copy(copy),
    device_bytes = device_bytes,
    device_held = device_held,
    devices = {},
    held = {},
    threads = nil,
  } as Pool
  -- the model's one threadpool, made with the arena beside it so that no graph
  -- of its passes pays a create and a join of its own: the entries are copied,
  -- because the pool that replaces this one makes its own from them, and the
  -- count is the model's rather than the library's
  if threads is Threads then
    local count = threads.n_threads
    if not (count is nil) then
      local params = ffi.new("struct ggml_threadpool_params") as ffi.CData
      lib.ggml_threadpool_params_init(params, count)
      local make = ffi.cast(
        "ggml_threadpool *(*)(struct ggml_threadpool_params *)", threads.make
      ) as function(ffi.CData): ffi.CData
      local made = make(params)
      if not (made is nil) then
        -- the pool is given back with the arena even where the pool that holds
        -- it is dropped without its free
        local give = ffi.cast("void (*)(ggml_threadpool *)", threads.give)
          as function(ffi.CData)
        pool.threads = {
          make = threads.make,
          give = threads.give,
          set = threads.set,
          set_n_threads = threads.set_n_threads,
          n_threads = count,
          pool = ffi.gc(made, function(held: ffi.CData) give(held) end),
        } as Threads
      end
    end
  end
  return pool
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
  pool.device_held = {}
  local cpu = pool.cpu
  if not (cpu is nil) then
    pool.cpu = nil
    pool.lib.ggml_backend_free(cpu)
  end
  -- the pool goes after the backend made over it, and the threadpool that
  -- backend held goes with it, so a closed model leaves no thread behind
  local threads = pool.threads
  if threads is Threads then
    pool.threads = nil
    local threadpool = threads.pool
    if not (threadpool is nil) then
      threads.pool = nil
      ffi.gc(threadpool, nil)
      local give = ffi.cast("void (*)(ggml_threadpool *)", threads.give)
        as function(ffi.CData)
      give(threadpool)
    end
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
  if made is nil then
    return nil
  end
  -- the pool the graphs run on is given to the backend the first time that
  -- backend is made, before its first graph
  local threads = pool.threads
  if not (threads is Threads) then
    return made
  end
  local threadpool = threads.pool
  if threadpool is nil then
    return made
  end
  -- the pinned ggml plans each graph from the backend's own n_threads, which
  -- ggml_backend_cpu_init leaves at GGML_DEFAULT_N_THREADS: the pool's count is
  -- set here, before the first graph this backend computes
  local count = threads.n_threads
  if not (count is nil) then
    local set_count = ffi.cast(
      "void (*)(ggml_backend *, int)", threads.set_n_threads
    ) as function(ffi.CData, integer)
    set_count(made, count)
  end
  local set = ffi.cast(
    "void (*)(ggml_backend *, ggml_threadpool *)", threads.set
  ) as function(ffi.CData, ffi.CData)
  set(made, threadpool)
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
  if copy is Copy then
    host_bytes = 0
  end

  local arena_size = arena_for(lib, tensors, host_bytes)
  local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
  if arena is nil then
    return nil, "no context of " .. arena_size .. " bytes"
  end

  local params = arena_params(arena_size)
  params.mem_buffer = arena
  if copy is Copy then
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
  session: Session, operation: string, cgraph: ffi.CData, issued?: boolean
): string | nil
  local lib = session.lib
  local pool = session.pool
  if pool is Pool then
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
    if placed_line is string then
      return operation .. ": " .. placed_line
    end
    for _, write in ipairs(session.graph.pending) do
      write()
    end
    local status: integer
    if issued == true then
      status = lib.ggml_backend_graph_compute_async(engine, cgraph)
    else
      status = lib.ggml_backend_graph_compute(engine, cgraph)
    end
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
  if pool is Pool then
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
): integer, integer, integer, integer
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
    return 0, 0, 0, 0
  end

  -- the graph object a pass's build makes comes out of the same arena, and
  -- the tensors it left behind are what the arena is then sized from
  lib.ggml_new_graph_custom(ctx, tensors, false)
  local used = tonumber(lib.ggml_used_mem(ctx)) as integer
  local numbers = tonumber(
    lib.ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)
  ) as integer
  local made = 0
  local held = 0
  local tensor = lib.ggml_get_first_tensor(ctx)
  while not (tensor is nil) do
    local view = tensor as TensorView
    if device and view.data == nil and view.view_src == nil then
      held = held + (tonumber(lib.ggml_nbytes(tensor)) as integer)
    end
    made = made + 1
    tensor = lib.ggml_get_next_tensor(ctx, tensor)
  end
  lib.ggml_free(ctx)
  libc.free(block)
  if device then
    return used, numbers, made, held
  end
  return used + numbers, 0, made, 0
end
-- the arena and the device buffers a handle holds, each the worst case of one
-- graph: never the sum of a run's blocks
local function sized_pool(
  lib: Library,
  copy: Copy | nil,
  builds: {Build},
  rows: integer,
  per_row: integer | nil,
  floor: integer | nil,
  threads?: Threads
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
  local device_held: {string: integer} = {}
  local made = 0
  for _, entry in ipairs(builds) do
    local host_bytes, device_bytes, count, held = footprint_of(
      lib, copy, false, entry.build, tensors
    )
    if host_bytes > host then
      host = host_bytes
    end
    if count > made then
      made = count
    end
    if copy is Copy then
      host_bytes, device_bytes, count, held = footprint_of(
        lib, copy, true, entry.build, tensors
      )
      if host_bytes > host then
        host = host_bytes
      end
      if device_bytes > (device[entry.kind] or 0) then
        device[entry.kind] = device_bytes
      end
      if held > (device_held[entry.kind] or 0) then
        device_held[entry.kind] = held
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
  return pool_new(
    lib, copy, host, device, device_held, rows, made, tensors, threads
  )
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
  build: Builder,
  issued?: boolean
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

  local failure = computed(session, operation, cgraph, issued)
  if failure then
    closed(session)
    return nil, failure
  end
  if measure is compute.Measure then
    measure.graphs = measure.graphs + 1
  end
  return {
    lib = session.lib,
    device = pool.backend,
    outputs = outputs,
    session = session,
  } as Computed
end
-- waits for a graph issued on a device, which its numbers are read after
local function graph_wait(made: Computed)
  local session = made.session
  local backend = session.backend
  if backend is nil then
    return
  end
  session.lib.ggml_backend_synchronize(backend)
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

return {
  load_library = load_library,
  refusal_line = refusal_line,
  probe_version = probe_version,
  round_up = round_up,
  arena_for = arena_for,
  arena_params = arena_params,
  pool_params = pool_params,
  finalised = finalised,
  arena_finalised = arena_finalised,
  find_backends = find_backends,
  read_bytes = read_bytes,
  tensor_in = tensor_in,
  graph_tensor = graph_tensor,
  node = node,
  vector_of = vector_of,
  scalar_of = scalar_of,
  ints_of = ints_of,
  host_tensor = host_tensor,
  pool_free = pool_free,
  threads_of = threads_of,
  run = run,
  sized_pool = sized_pool,
  graph_run = graph_run,
  graph_wait = graph_wait,
  graph_closed = graph_closed,
  graph_numbers = graph_numbers,
  graph_planes = graph_planes,
  free_copy = free_copy,
  device_copy = device_copy,
  guard_release = guard_release,
  holder = holder,
  libc = libc,
  DEFAULT_PATH = DEFAULT_PATH,
  EXPECTED_VERSION = EXPECTED_VERSION,
  F32_BYTES = F32_BYTES,
  I32_BYTES = I32_BYTES,
  ARENA_ALIGN = ARENA_ALIGN,
  GGML_TYPE_F32 = GGML_TYPE_F32,
  GGML_TYPE_F16 = GGML_TYPE_F16,
  GGML_TYPE_Q4_0 = GGML_TYPE_Q4_0,
  GGML_TYPE_Q8_0 = GGML_TYPE_Q8_0,
  GGML_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX,
  GGML_DEVICE_GPU = GGML_DEVICE_GPU,
  LOADED = LOADED,
  COPIES = COPIES,
}
