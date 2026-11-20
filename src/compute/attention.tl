-- a full-attention block: its keys and values per sequence in host memory,
-- their snapshots, and the graphs of one row's scored and weighted values

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local scaffold = require("tllm.compute.blocks")
local engine = require("tllm.compute.engine")
local mathx = require("tllm.compute.maths")
local oncard = require("tllm.compute.placement")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Appended = defs.Appended
local type AttentionState = defs.AttentionState
local type BlockTensors = defs.BlockTensors
local type Box = defs.Box
local type Build = defs.Build
local type Cache = defs.Cache
local type CacheKind = defs.CacheKind
local type Computed = defs.Computed
local type Graph = defs.Graph
local type Mapped = defs.Mapped
local type Output = defs.Output
local type Pool = defs.Pool

local ARENA_ALIGN = engine.ARENA_ALIGN
local F32_BYTES = engine.F32_BYTES
local GGML_ROPE_TYPE_NEOX = engine.GGML_ROPE_TYPE_NEOX
local GGML_TYPE_F32 = engine.GGML_TYPE_F32
local LOADED = engine.LOADED
local finalised = engine.finalised
local graph_closed = engine.graph_closed
local graph_planes = engine.graph_planes
local graph_run = engine.graph_run
local host_tensor = engine.host_tensor
local ints_of = engine.ints_of
local libc = engine.libc
local node = engine.node
local pool_free = engine.pool_free
local read_bytes = engine.read_bytes
local round_up = engine.round_up
local sized_pool = engine.sized_pool

local mapped_of = oncard.mapped_of
local tier = oncard.tier

local cache_kind_of = mathx.cache_kind_of
local cache_row_size = mathx.cache_row_size
local geometry_refusal = mathx.geometry_refusal
local holds = mathx.holds
local matrix_of = mathx.matrix_of
local prefixed = mathx.prefixed

local ATTENTION_NAMES = mathx.ATTENTION_NAMES
local blank_hidden = scaffold.blank_hidden
local grown = scaffold.grown
local measure_pointer = scaffold.measure_pointer
local position_refusal = scaffold.position_refusal
local read_tensors = scaffold.read_tensors

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
  if stored is Cache then
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
  if held is {string: Cache} then
    cache = held[prefix]
  end
  if cache is nil then
    return nil, prefix .. " holds no cache"
  end
  local stored = cache
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
  local mapped = state.mapped
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
      tier.weight_of(graph, mapped, prefix .. "attn_norm.weight")
    )
  )
  local query = node(
    graph,
    lib.ggml_mul_mat(
      ctx, tier.weight_of(graph, mapped, prefix .. "attn_q.weight"), normalised
    )
  )
  local read_keys = node(
    graph,
    lib.ggml_mul_mat(
      ctx, tier.weight_of(graph, mapped, prefix .. "attn_k.weight"), normalised
    )
  )
  local read_values = node(
    graph,
    lib.ggml_mul_mat(
      ctx, tier.weight_of(graph, mapped, prefix .. "attn_v.weight"), normalised
    )
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
          tier.weight_of(graph, mapped, prefix .. "attn_q_norm.weight")
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
          tier.weight_of(graph, mapped, prefix .. "attn_k_norm.weight")
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
        lib.ggml_mul_mat(
          ctx,
          tier.weight_of(graph, mapped, prefix .. "attn_output.weight"),
          flat
        )
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
    if held is {string: Cache} then
      cache = held[prefix]
    end
    local expected = running[entry.sequence]
    if expected is nil then
      expected = 0
      if cache is Cache then
        expected = cache.count
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
    if held is {string: Cache} then
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
-- the attention handle over a mapped handle: `pool` is the one the model that
-- built it holds, and a handle built on its own is given one sized from its own
-- largest graph
local function attention_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure?: compute.Measure,
  given?: Mapped
): compute.Attention | nil, string | nil
  local refusal = geometry_refusal(geometry)
  if refusal then
    return nil, refusal
  end
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "attend: backend was not loaded by compute.load"
  end

  local mapped: Mapped | nil = nil
  if given is Mapped then
    mapped = tier.borrowed(given)
  else
    local made, map_failure = mapped_of(backend, loaded, handle, "attend")
    if made is nil then
      return nil, map_failure
    end
    mapped = made
  end
  local attachment = mapped as Mapped
  attachment.hold()

  local state: AttentionState = {
    mapped = attachment,
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
    if line is string then
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
    if line is string then
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
    if line is string then
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
    if line is string then
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
      if held is Pool then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return attention
end

return {
  attention_of = attention_of,
  attention_measure = attention_measure,
}
