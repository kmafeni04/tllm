local ffi = require("ffi")
local compute = require("tllm.compute.api")
local reads = require("tllm.reads")
local weights = require("tllm.weights")

-- what the compute modules pass each other: a graph and the arena it is built
-- in, a mapped handle and its device copy, and the state each block kind and
-- the model keep. Every module reads these, so they live here rather than in
-- any one of them.
local type Tensor = weights.Tensor

local record Types
  type Params = {string | integer: ffi.CData | boolean}

  record Library
    ggml_version: function(): ffi.CData
    ggml_tensor_overhead: function(): integer
    ggml_graph_overhead: function(): integer
    ggml_graph_overhead_custom: function(integer, boolean): integer
    ggml_init: function(Params): ffi.CData
    ggml_free: function(ffi.CData)
    ggml_new_tensor_2d: function(
      ffi.CData, integer, integer, integer
    ): ffi.CData
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
    ggml_backend_graph_compute_async: function(ffi.CData, ffi.CData): integer
    ggml_backend_synchronize: function(ffi.CData)
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
    ggml_backend_tensor_alloc: function(
      ffi.CData, ffi.CData, ffi.CData
    ): integer
    ggml_backend_view_init: function(ffi.CData): integer
    ggml_backend_buffer_free: function(ffi.CData)
    ggml_backend_tensor_set: function(ffi.CData, ffi.CData, integer, integer)
    ggml_backend_tensor_get: function(ffi.CData, ffi.CData, integer, integer)
    ggml_threadpool_params_init: function(ffi.CData, integer)
    ggml_backend_reg_by_name: function(string): ffi.CData | nil
    ggml_backend_reg_get_proc_address: function(
      ffi.CData, string
    ): ffi.CData | nil
  end

  -- the model's one threadpool: the entries the loaded CPU backend module
  -- answers for it, which the library handle does not export, and the pool
  -- they made, made beside the arena that pool's graphs are built in
  record Threads
    make: ffi.CData
    give: ffi.CData
    set: ffi.CData
    -- the registry's ggml_backend_set_n_threads, which the backend that plans
    -- the model's graphs is told the pool's count through
    set_n_threads: ffi.CData
    -- the count the pool is made with: a model's own, nil before one is chosen
    n_threads: integer | nil
    pool: ffi.CData | nil
  end

  record TensorView
    data: ffi.CData
    view_src: ffi.CData
  end

  -- a tensor a graph answers with and the numbers it holds, a count of zero
  -- being a node the graph runs and answers with no number
  record Output
    tensor: ffi.CData
    count: integer
  end

  -- one graph a handle builds, for the measuring pass that sizes its arena; the
  -- kind names the graphs a block's device buffer has to hold
  record Build
    kind: string
    build: function(Graph): {Output}
  end

  -- the device copy of one model's weights: one buffer, its descriptors and the
  -- backend the graphs over them are computed on
  record Copy
    lib: Library
    backend: ffi.CData
    ctx: ffi.CData
    arena: ffi.CData
    buffer: ffi.CData
    tensors: {string: ffi.CData}
    experts: {string: {ffi.CData | nil}} | nil
  end

  record Loaded
    lib: Library
    device: ffi.CData | nil
    offload: boolean
    -- the threadpool entries this library answers, resolved once here so that
    -- a model makes its one pool from them; nil where it found no CPU module
    threads: Threads | nil
  end

  -- tensors written into host memory at once, or into the device buffer of the
  -- backend the graph runs on once the buffer exists; `measure` builds a graph
  -- for its size only, writing no number into any tensor
  record Graph
    lib: Library
    ctx: ffi.CData
    device: ffi.CData | nil
    pending: {function()}
    -- the seconds the graph spent copying expert matrices to the device
    copied: number | nil
    measure: string | nil
    label: string | nil
    -- a context that places no number of its own has no tensor to write into
    -- until the graph's buffer holds them, so every input waits in `pending`
    deferred: boolean
  end

  -- one graph's build: the tensors it answers with, in the order they are read
  type Builder = function(Graph): {Output}

  -- the one host arena a handle writes its graphs over, and the device buffers
  -- its graphs run in, one per block, each sized from the largest graph built in
  -- it: `host` is re-initialised by every graph and freed only by the handle
  record Pool
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
    device_held: {string: integer}
    devices: {string: ffi.CData}
    held: {string: integer}
    -- the threadpool the model's graphs run on, made beside this arena and
    -- freed with it: nil for every arena but a model's own, which is every
    -- arena a block handle or the maths build
    threads: Threads | nil
  end

  -- the arena a run of blocks builds its graphs in, held through one box so a
  -- model that grows its arena for a wider pass gives the grown arena to every
  -- block of its run: the pool inside is re-initialised by every graph and
  -- freed only by the handle that holds the box
  record Box
    pool: Pool | nil
    owned: boolean
  end

  record Holder
    hold: function()
    close: function()
  end

  -- the expert cache of one model: the experts its blocks' slots hold, the
  -- routes the current window counted, and the slots a pass will fill once its
  -- arithmetic has returned
  record ExpertBlock
    prefix: string
    slots: {integer}
    counts: {integer: integer}
    last: {integer: integer}
    pending: {integer: integer} | nil
  end

  record ExpertCache
    blocks: {ExpertBlock}
    rounds: integer
    round: integer
  end

  -- the experts a window named and how many of their matrices landed whole,
  -- one record per expert, and the reads handed over and not yet reaped
  record AheadExpert
    wanted: integer
    landed: integer
    counted: boolean
  end

  record AheadPending
    prefix: string
    expert: integer
    wanted: integer
  end

  -- the queue a model took for read-ahead, the state it publishes, and the
  -- window it decides by: what was handed over since the build or the last
  -- decision
  record Ahead
    published: compute.ReadAhead
    queue: reads.Queue | nil
    -- the value a person forced, nil when the gate decides
    forced: boolean | nil
    expert_bytes: integer
    passes: integer
    named: integer
    landed: integer
    experts: {string: AheadExpert}
    pending: {string: AheadPending}
  end

  record Mapped
    lib: Library
    tensors: {string: Tensor}
    descriptors: {string: ffi.CData}
    handle: weights.Handle
    device: Copy | nil
    arena: ffi.CData | nil
    ctx: ffi.CData | nil
    -- a model's plan: the tensors its own buffer holds, the copy they live in
    -- and the cache of the experts its slots hold; nil for every other entry
    resident: {string: boolean} | nil
    cache: ExpertCache | nil
    plan: compute.Placement | nil
    hold: function()
    close: function()
  end

  -- one prefix's keys and values: two buffers of the geometry's cache type in
  -- host memory, the size they hold and the positions they took
  record Cache
    keys: ffi.CData
    values: ffi.CData
    bytes: integer
    count: integer
  end

  record AttentionState
    mapped: Mapped
    geometry: compute.Geometry
    caches: {string: {string: Cache}}
    capacities: {string: integer}
    pool: Box
    measure: compute.Measure | nil
    owner: boolean
  end

  record FeedForwardState
    mapped: Mapped
    geometry: compute.Geometry
    pool: Box
    measure: compute.Measure | nil
    owner: boolean
    ahead: Ahead | nil
    miss: MissWindow | nil
    -- whether the page-cache path is on for the model this state serves
    zero_copy: boolean
    -- the mapping's own view of the handle, which the misses computed on the
    -- CPU read rather than the card's copy, and the arena their graphs build in
    host: Mapped | nil
    cpupool: Pool | nil
  end

  -- one model's miss policy and what the pass being calibrated has measured
  record MissWindow
    policy: compute.MissPolicy
    missed: boolean
  end

  record LinearShape
    key_heads: integer
    key_length: integer
    value_heads: integer
    value_length: integer
    conv_kernel: integer
  end

  record LinearKept
    history: ffi.CData
    state: ffi.CData
    count: integer
  end

  record LinearState
    mapped: Mapped
    shape: LinearShape
    eps: number
    prefixes: {string: {string: LinearKept}}
    pool: Box
    measure: compute.Measure | nil
    owner: boolean
  end

  record ModelPlan
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
    placement: compute.Placement
    copy: Copy | nil
    ahead: Ahead
    miss: MissWindow
    zero_copy: boolean
  end

  record SequenceState
    capacity: integer
    taken: integer
  end

  record ModelState
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
    placement: compute.Placement
    copy: Copy | nil
    ahead: Ahead
    miss: MissWindow
  end

  record BlockTensors
    norm: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    query_norm: Tensor
    key_norm: Tensor
    output: Tensor
  end

  -- what a cached number costs in each type the cache may hold: the bytes of
  -- one block and how many numbers a block carries
  record CacheKind
    code: integer
    block_elements: integer
    block_bytes: integer
  end

  -- what a build raises when its context has no room for the tensor it asked for
  record ArenaError
    arena: string | nil
  end

  record Session
    lib: Library
    copy: Copy | nil
    pool: Pool | nil
    arena: ffi.CData
    ctx: ffi.CData
    graph: Graph
    backend: ffi.CData | nil
    buffer: ffi.CData | nil
  end

  -- a graph built in the pool's arena and computed, its tensors readable until
  -- the next graph takes the arena over
  record Computed
    lib: Library
    device: ffi.CData | nil
    outputs: {Output}
    session: Session
  end

  record Tiers
    attention: {string}
    linear: {string}
    routers: {string}
    shared: {string}
    experts: {string}
    block_numbers: function(weights.Handle): {integer}
    fixed_order: function(weights.Handle): {string}
    experts_held: function({string: Tensor}, string): integer
    expert_charge: function({string: Tensor}, string, integer): integer
    same_placement: function(compute.Placement, compute.Placement): boolean
    place: function(weights.Handle, integer, integer): compute.Placement
    reserve_of: function(Pool, {boolean}, integer): integer
    placed_copy: function(
      Loaded, weights.Handle, compute.Placement, {{integer}}
    ): Copy | nil
    mapped_with_copy: function(
      Loaded, weights.Handle, Copy, compute.Placement
    ): Mapped
    borrowed: function(Mapped): Mapped
    weight_of: function(Graph, Mapped, string, index?: integer): ffi.CData
  end

  -- the cache tensors one row's graph reads and appends to: the host buffer
  -- itself on the CPU path, a copy of the filled part on the device path, where
  -- the rows the graph appended are copied back once it has computed
  record Appended
    keys: ffi.CData
    values: ffi.CData
  end

  record LinearBuilt
    output: ffi.CData
    updated: ffi.CData
    projected: ffi.CData
  end

  -- what the rows before this one in its group leave it: the state the row
  -- before it left and every projection the group made, oldest first
  record LinearCarry
    state: ffi.CData | nil
    taps: {ffi.CData}
  end
end

return Types
