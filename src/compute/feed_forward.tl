-- the feed-forward half of a block: the router's votes, the experts it
-- chooses, the shared and expert matrices they read, and one row's scores

local ffi = require("ffi")
local clock = require("tllm.clock")
local compute = require("tllm.compute.api")
local scaffold = require("tllm.compute.blocks")
local cachex = require("tllm.compute.cache")
local engine = require("tllm.compute.engine")
local mathx = require("tllm.compute.maths")
local oncard = require("tllm.compute.placement")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Tensor = weights.Tensor

local type Box = defs.Box
local type Build = defs.Build
local type Copy = defs.Copy
local type FeedForwardState = defs.FeedForwardState
local type Graph = defs.Graph
local type Mapped = defs.Mapped
local type ExpertCache = defs.ExpertCache
local type Output = defs.Output
local type Pool = defs.Pool

local LOADED = engine.LOADED
local graph_closed = engine.graph_closed
local graph_planes = engine.graph_planes
local graph_run = engine.graph_run
local node = engine.node
local pool_free = engine.pool_free
local sized_pool = engine.sized_pool
local vector_of = engine.vector_of

local mapped_of = oncard.mapped_of
local tier = oncard.tier

local BLOCK_NAMES = mathx.BLOCK_NAMES
local BLOCK_TYPES = mathx.BLOCK_TYPES
local SCORE_NAMES = mathx.SCORE_NAMES
local SCORE_TYPES = mathx.SCORE_TYPES
local holds = mathx.holds
local prefixed = mathx.prefixed

local blank_hidden = scaffold.blank_hidden
local blank_rows = scaffold.blank_rows
local count_line = scaffold.count_line
local count_refusal = scaffold.count_refusal
local even_votes = scaffold.even_votes
local first_fault = scaffold.first_fault
local grown = scaffold.grown
local hidden_width = scaffold.hidden_width
local prefix_named = scaffold.prefix_named

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
      tier.weight_of(graph, mapped, prefix .. "post_attention_norm.weight")
    )
  )
  local votes = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "ffn_gate_inp.weight"),
      normalised
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
      tier.weight_of(graph, mapped, prefix .. "post_attention_norm.weight")
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
        tier.weight_of(graph, mapped, prefix .. "ffn_up_exps.weight", expert),
        normalised
      )
    )
    local gate = node(
      graph,
      lib.ggml_mul_mat(
        ctx,
        tier.weight_of(
          graph, mapped, prefix .. "ffn_gate_exps.weight", expert
        ),
        normalised
      )
    )
    local mixed = node(graph, lib.ggml_swiglu_split(ctx, gate, up))
    local down = node(
      graph,
      lib.ggml_mul_mat(
        ctx,
        tier.weight_of(
          graph, mapped, prefix .. "ffn_down_exps.weight", expert
        ),
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
      ctx,
      tier.weight_of(graph, mapped, prefix .. "ffn_up_shexp.weight"),
      normalised
    )
  )
  local shared_gate = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "ffn_gate_shexp.weight"),
      normalised
    )
  )
  local shared_mixed = node(
    graph, lib.ggml_swiglu_split(ctx, shared_gate, shared_up)
  )
  local shared_down = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "ffn_down_shexp.weight"),
      shared_mixed
    )
  )
  local router = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "ffn_gate_inp_shexp.weight"),
      normalised
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
      tier.weight_of(graph, state.mapped, "output_norm.weight")
    )
  )
  return node(
    graph,
    lib.ggml_mul_mat(
      ctx, tier.weight_of(graph, state.mapped, "output.weight"), normalised
    )
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
  if block_prefix is string and not (block_width is nil) then
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
    if tensor is Tensor then
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
-- the whole read of the chosen experts: one request per matrix this handle has
-- not read yet, filling the page cache behind the pointer the choice graph
-- multiplies by. The graph over a device copy reads no mapping, so it reads no
-- matrix
local function read_experts(
  mapped: Mapped,
  prefix: string,
  chosen: {integer: boolean},
  experts: integer,
  measure: compute.Measure | nil
): string | nil
  local plan = mapped.plan
  local copy = mapped.device
  if plan is nil and copy is Copy then
    return nil
  end
  local held = mapped.cache
  local block = cachex.block_of(prefix)
  local started = clock.now()
  local reads = 0
  local bytes = 0
  for _, name in ipairs(tier.experts) do
    local tensor = mapped.tensors[prefix .. name]
    if tensor is Tensor and #tensor.dims >= 3 then
      local each = math.floor(tensor.bytes / (tensor.dims[3] as integer))
      for expert = 0, experts - 1 do
        local in_slot = false
        if held is ExpertCache then
          in_slot = cachex.slot_of(held, block, expert) ~= nil
        end
        if chosen[expert] and not in_slot then
          local moved, line = weights.read(mapped.handle, tensor, expert)
          if moved is nil then
            return line as string
          end
          if moved == true then
            reads = reads + 1
            bytes = bytes + each
          end
        end
      end
    end
  end
  if reads > 0 and measure is compute.Measure then
    measure.blob_reads = measure.blob_reads + reads
    measure.blob_bytes = measure.blob_bytes + bytes
    measure.blob_seconds = measure.blob_seconds + (clock.now() - started)
  end
  return nil
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
  local chosen: {integer: boolean} = {}
  local block_cache = mapped.cache
  local block = cachex.block_of(prefix)
  for index, probabilities in ipairs(votes) do
    local routed = chosen_experts(probabilities, geometry.experts_used)
    local total: number = 0
    for _, expert in ipairs(routed) do
      total = total + probabilities[expert + 1]
      chosen[expert] = true
      if block_cache is ExpertCache then
        local was_held = cachex.record(block_cache, block, expert)
        local measure = state.measure
        if was_held and measure is compute.Measure then
          measure.expert_hits = measure.expert_hits + 1
        end
      end
    end
    totals[index] = total
    if watchs is {compute.Watch} then
      recorded(watchs[index], mapped, prefix, routed)
    end
  end

  local read_line = read_experts(mapped, prefix, chosen, experts, state.measure)
  if read_line is string then
    return nil, read_line
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
-- the feed forward handle over a mapped handle, sharing a model's arena when
-- one is given and holding one of its own otherwise
local function feed_forward_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure?: compute.Measure,
  given?: Mapped
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

  local mapped: Mapped | nil = nil
  if given is Mapped then
    mapped = tier.borrowed(given)
  else
    local made, map_failure = mapped_of(backend, loaded, handle, "feed_forward")
    if made is nil then
      return nil, map_failure
    end
    mapped = made
  end
  local attachment = mapped as Mapped
  attachment.hold()

  local state: FeedForwardState = {
    mapped = attachment,
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
      if held is Pool then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return feed_forward
end

return {
  feed_forward_of = feed_forward_of,
  feed_forward_builds = feed_forward_builds,
}
