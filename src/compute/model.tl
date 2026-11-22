-- the whole model: the plan a gguf file and a cache type give, the per-block
-- handles a pass drives, and the conversation files a model saves and reads

local ffi = require("ffi")
local fs = require("fs")
local clock = require("tllm.clock")
local compute = require("tllm.compute.api")
local attn = require("tllm.compute.attention")
local scaffold = require("tllm.compute.blocks")
local cachex = require("tllm.compute.cache")
local engine = require("tllm.compute.engine")
local ff = require("tllm.compute.feed_forward")
local lin = require("tllm.compute.linear")
local mathx = require("tllm.compute.maths")
local oncard = require("tllm.compute.placement")
local defs = require("tllm.compute.types")
local gguf = require("tllm.gguf")
local missx = require("tllm.miss")
local prefetch = require("tllm.prefetch")
local reads = require("tllm.reads")
local weights = require("tllm.weights")

local type Tensor = weights.Tensor

local type AttentionState = defs.AttentionState
local type Box = defs.Box
local type Build = defs.Build
local type CacheKind = defs.CacheKind
local type Copy = defs.Copy
local type FeedForwardState = defs.FeedForwardState
local type Graph = defs.Graph
local type LinearShape = defs.LinearShape
local type LinearState = defs.LinearState
local type Mapped = defs.Mapped
local type ExpertCache = defs.ExpertCache
local type MissWindow = defs.MissWindow
local type ModelPlan = defs.ModelPlan
local type ModelState = defs.ModelState
local type Output = defs.Output
local type Pool = defs.Pool
local type Ahead = defs.Ahead
local type SequenceState = defs.SequenceState

local LOADED = engine.LOADED
local free_copy = engine.free_copy
local graph_closed = engine.graph_closed
local graph_numbers = engine.graph_numbers
local graph_run = engine.graph_run
local ints_of = engine.ints_of
local node = engine.node
local pool_free = engine.pool_free
local sized_pool = engine.sized_pool

local mapped_of = oncard.mapped_of
local tier = oncard.tier

local BLOCK_NAMES = mathx.BLOCK_NAMES
local cache_kind_of = mathx.cache_kind_of
local holds = mathx.holds
local prefixed = mathx.prefixed

local ATTENTION_NAMES = mathx.ATTENTION_NAMES
local hidden_width = scaffold.hidden_width
local prefix_named = scaffold.prefix_named

local attention_measure = attn.attention_measure
local attention_of = attn.attention_of

local feed_forward_builds = ff.feed_forward_builds
local feed_forward_of = ff.feed_forward_of

local linear_builds = lin.linear_builds
local linear_of = lin.linear_of
local linear_shape_of = lin.linear_shape_of

-- the rows a pass carries when nothing names a bound of its own
local DEFAULT_UBATCH: integer = 512
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
      graph.ctx,
      tier.weight_of(graph, mapped, "token_embd.weight"),
      ints_of(graph, ids)
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
  -- a second block exists when a block of the run can name a following one's
  -- experts, and the prediction's nodes are the same shape whatever it is
  local following: string | nil = nil
  if count >= 2 then
    following = "blk.1."
  end
  if full_prefix is string and not (hidden_count is nil) then
    for _, entry in ipairs(
      feed_forward_builds(
        feed_forward_state, full_prefix, hidden_count, rows, following
      )
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
      if entry is Build then
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
-- the bytes every expert matrix of the handle holds, and the bytes of its
-- largest one
local function expert_bytes_of(handle: weights.Handle): integer, integer
  local total = 0
  local largest = 0
  for name, tensor in pairs(handle.tensors) do
    local expert = false
    for _, suffix in ipairs(tier.experts) do
      if #name > #suffix and name:sub(-#suffix) == suffix then
        expert = true
        break
      end
    end
    if expert and #tensor.dims >= 3 and (tensor.dims[3] or 0) >= 1 then
      local each = math.floor(tensor.bytes / (tensor.dims[3] as integer))
      if each > largest then
        largest = each
      end
      total = total + tensor.bytes
    end
  end
  return total, largest
end
-- the gate the page-cache path is on by: the machine's free bytes over the
-- bytes every expert matrix of the handle holds, at least 0.55 of them
local ZERO_COPY_ROOM: number = 0.55

local function zero_copy_of(expert_bytes: integer, free: integer | nil): boolean
  if expert_bytes < 1 or not (free is integer) then
    return false
  end
  return (free as integer) / expert_bytes >= ZERO_COPY_ROOM
end
-- the queue a model takes for read-ahead and the state the gate opens it in:
-- a model given `prefetch` false takes none, one whose handle holds no expert
-- matrix takes none, and a platform without the interface takes none
local function read_ahead_of(
  handle: weights.Handle,
  experts_used: integer,
  forced: boolean | nil,
  free: integer | nil
): Ahead
  local total, largest = expert_bytes_of(handle)
  local ahead: Ahead = {
    published = {open = false, depth = 0, bytes = 0},
    queue = nil,
    forced = forced,
    expert_bytes = total,
    passes = 0,
    named = 0,
    landed = 0,
    experts = {},
    pending = {},
  }
  if total < 1 or largest < 1 or forced == false then
    return ahead
  end
  if not reads.offered() then
    return ahead
  end
  local depth = 3 * experts_used
  if depth < 4 then
    depth = 4
  end
  local queue = reads.open(depth, largest)
  if queue is nil then
    return ahead
  end
  ahead.queue = queue
  ahead.published.depth = queue.depth
  ahead.published.bytes = queue.bytes
  if forced == true then
    ahead.published.open = true
    return ahead
  end
  local room: number | nil = nil
  if free is integer then
    room = free / total
  end
  ahead.published.open = prefetch.decide(room, 0, 0, false)
  return ahead
end
local function give_back(ahead: Ahead)
  local queue = ahead.queue
  if queue is reads.Queue then
    ahead.queue = nil
    reads.close(queue)
  end
end
-- the window a decision counts: what was named and landed since the build or
-- the last decision
local function decided(ahead: Ahead)
  if not (ahead.forced == nil) then
    return
  end
  local queue = ahead.queue
  if not (queue is reads.Queue) then
    return
  end
  local free = prefetch.memory()
  local room: number | nil = nil
  if free is integer and ahead.expert_bytes > 0 then
    room = free / ahead.expert_bytes
  end
  ahead.published.open = prefetch.decide(
    room, ahead.named, ahead.landed, ahead.published.open
  )
  ahead.named = 0
  ahead.landed = 0
  ahead.experts = {}
end
local function model_plan(
  backend: compute.Backend,
  handle: weights.Handle,
  positions: integer | nil,
  cache_type: compute.CacheType | nil,
  sequences: integer,
  rows: integer,
  vram: integer | nil,
  profile: string | nil,
  forced: boolean | nil
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

  local head_length: integer = 1
  local kv_heads: integer = 1
  if has_full then
    local prefix = "blk." .. first_full .. "."
    local query = handle.tensors[prefix .. "attn_q.weight"]
    if query is nil then
      return nil, "model: no tensor " .. prefix .. "attn_q.weight"
    end
    local key_tensor = handle.tensors[prefix .. "attn_k.weight"]
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

  -- the widest pass the model may be handed: ubatch rows, and no more than
  -- every sequence at every position it holds
  local widest = rows
  local most_rows = sequences * taken
  if widest > most_rows then
    widest = most_rows
  end

  -- the machine's free bytes for the page cache, read once at the build: the
  -- room read-ahead opens on and the one the page-cache path is gated by
  local free_bytes = prefetch.memory()
  -- the queue read-ahead hands reads to, after every fault the model's own
  -- keys and shapes name and before the profile is read
  local ahead = read_ahead_of(handle, geometry.experts_used, forced, free_bytes)
  local zero_copy = zero_copy_of(ahead.expert_bytes, free_bytes)
  local miss: MissWindow = {
    policy = {
      share = 0,
      pool = 0,
      link = 0,
      windows = 0,
      cpu = 0,
      dma = 0,
      cpu_bytes = 0,
      cpu_seconds = 0,
      dma_bytes = 0,
      dma_seconds = 0,
    },
    missed = false,
  }
  local function give_up(line: string): nil, string
    give_back(ahead)
    return nil, line
  end

  -- the profile is read whether or not a slot is placed, after every fault
  -- the model's own keys name and before anything is copied to the card
  local profile_counts: {integer: {integer: integer}} | nil = nil
  if profile is string then
    local counts, profile_line = cachex.read(profile)
    if counts is nil then
      return give_up(profile_line as string)
    end
    profile_counts = counts
  end
  local numbers = tier.block_numbers(handle)
  local limits: {integer} = {}
  local prefixes: {string} = {}
  for index, block_number in ipairs(numbers) do
    local prefix = "blk." .. block_number .. "."
    prefixes[index] = prefix
    limits[index] = tier.experts_held(handle.tensors, prefix)
  end
  local function slots_holding(planned: compute.Placement): {{integer}}
    return cachex.seeds(profile_counts, limits, planned.slots)
  end
  local function cache_of(planned: compute.Placement): ExpertCache
    return cachex.make(slots_holding(planned), prefixes)
  end

  -- what the card may take, and the plan that says what sits there
  local found = loaded.device
  local use_device = loaded.offload and not (found is nil)
  local budget: integer = 0
  if use_device then
    local free = backend:memory()
    budget = free or 0
    if not (vram is nil) and vram < budget then
      budget = vram
    end
  end

  local mapped: Mapped | nil = nil
  local arena: Pool | nil = nil
  local placement: compute.Placement | nil = nil
  local copy: Copy | nil = nil

  local function mapped_total(): integer
    local sum = 0
    for _, tensor in pairs(handle.tensors) do
      sum = sum + tensor.bytes
    end
    return sum
  end

  local function on_the_cpu(
    budget_bytes: integer, reserve_bytes: integer
  ): string | nil
    -- an empty resident set, keeping the numbers the card answered: FR-004
    -- keeps the budget and the reservation, FR-011 clears only the tensors
    local bare: compute.Placement = {
      budget = budget_bytes,
      reserve = reserve_bytes,
      resident = {},
      resident_bytes = 0,
      slots = {},
      slot_bytes = 0,
      unplaced_bytes = mapped_total(),
    }
    for index = 1, count do
      bare.slots[index] = 0
    end
    placement = bare
    local plain = mapped_of(backend, loaded, handle, "model", false)
    if plain is nil then
      return "model: no context for its tensors"
    end
    plain.plan = placement
    plain.resident = placement.resident
    mapped = plain
    local made = sized_pool(
      plain.lib,
      nil,
      model_builds(
        plain, geometry, kinds, count, math.floor(width), sequences
      ),
      sequences,
      nil,
      nil
    )
    if made is nil then
      return "model: no arena for its graphs"
    end
    arena = made
    return nil
  end

  if not use_device or budget < 1 then
    local line = on_the_cpu(0, 0)
    if line is string then
      return give_up(line)
    end
  else
    -- a model whose tensors the tier would place must hold every fixed tensor
    -- of the blocks it runs; one that does not is refused before it copies
    for index = 0, count - 1 do
      local prefix = "blk." .. index .. "."
      local full = kinds[index + 1]
      local names = tier.linear
      if full then
        names = tier.attention
      end
      for _, name in ipairs(names) do
        if handle.tensors[prefix .. name] == nil then
          return give_up("model: no tensor " .. prefix .. name)
        end
      end
      for _, name in ipairs(BLOCK_NAMES) do
        if handle.tensors[prefix .. name] == nil then
          return give_up("model: no tensor " .. prefix .. name)
        end
      end
    end

    local total_bytes = mapped_total()
    placement = tier.place(handle, budget, 0)
    local stable = false
    local no_experts = false
    for _ = 1, 24 do
      local planned = placement as compute.Placement
      local seeds = slots_holding(planned)
      copy = tier.placed_copy(loaded, handle, planned, seeds)
      if copy is nil and planned.slot_bytes > 0 then
        -- the card refused the expert slots: keep the tensors it already took
        local bare = {
          budget = planned.budget,
          reserve = planned.reserve,
          resident = planned.resident,
          resident_bytes = planned.resident_bytes,
          slots = {},
          slot_bytes = 0,
          unplaced_bytes = math.max(0, total_bytes - planned.resident_bytes),
        } as compute.Placement
        for index = 1, #planned.slots do
          bare.slots[index] = 0
        end
        copy = tier.placed_copy(loaded, handle, bare, slots_holding(bare))
        if copy is Copy then
          placement = bare
          no_experts = true
        end
      end
      if copy is nil then
        local line = on_the_cpu(budget, planned.reserve)
        if line is string then
          return give_up(line)
        end
        stable = true
        break
      end
      mapped = tier.mapped_with_copy(loaded, handle, copy, placement)
      local probe = sized_pool(
        mapped.lib,
        copy,
        model_builds(
          mapped, geometry, kinds, count, math.floor(width), widest
        ),
        widest,
        nil,
        nil
      )
      if probe is nil then
        free_copy(copy)
        mapped.close()
        return give_up("model: no arena for its graphs")
      end
      local reserve = tier.reserve_of(probe, kinds, count)
      local following = tier.place(handle, budget, reserve)
      if no_experts then
        for index = 1, #following.slots do
          following.slots[index] = 0
        end
        following.slot_bytes = 0
        following.unplaced_bytes = math.max(
          0, total_bytes - following.resident_bytes
        )
      end
      if tier.same_placement(placement, following) then
        placement = following
        pool_free(probe)
        arena = sized_pool(
          mapped.lib,
          copy,
          model_builds(
            mapped, geometry, kinds, count, math.floor(width), sequences
          ),
          sequences,
          nil,
          nil
        )
        if arena is nil then
          free_copy(copy)
          mapped.close()
          return give_up("model: no arena for its graphs")
        end
        stable = true
        break
      end
      pool_free(probe)
      free_copy(copy)
      copy = nil
      mapped = nil
      placement = following
    end
    if not stable then
      -- a plan the card keeps changing is no plan; the CPU holds the model
      local line = on_the_cpu(budget, (placement as compute.Placement).reserve)
      if line is string then
        return give_up(line)
      end
    end
  end

  local plan = placement as compute.Placement
  local final_mapped = mapped as Mapped
  local final_arena = arena as Pool
  final_mapped.cache = cache_of(plan)
  if use_device and (plan.resident_bytes > 0 or plan.slot_bytes > 0) then
    backend.device = ffi.string(
      loaded.lib.ggml_backend_dev_name(loaded.device as ffi.CData)
    )
  else
    backend.device = nil
  end

  -- the one arena the model's graphs are built in, sized from the largest
  -- graph a pass of one row per sequence builds at the model's positions
  local measure: compute.Measure = {
    passes = 0,
    tokens = 0,
    compute = 0,
    sampling = 0,
    ready = 0,
    routed = 0,
    routed_bytes = 0,
    expert_hits = 0,
    blob_reads = 0,
    blob_bytes = 0,
    blob_seconds = 0,
    direct_reads = 0,
    graphs = 0,
    prefetch_reads = 0,
    prefetch_named = 0,
    prefetch_landed = 0,
  }
  local sized = function(wanted: integer): {Build}
    return model_builds(
      final_mapped, geometry, kinds, count, math.floor(width), wanted
    )
  end
  local box: Box = {pool = final_arena, owned = false}

  local function abandon()
    if copy is Copy then
      free_copy(copy)
      copy = nil
    end
    final_mapped.close()
  end

  local feed_forward, feed_line = feed_forward_of(
    backend,
    handle,
    geometry,
    box,
    measure,
    final_mapped,
    ahead,
    miss,
    zero_copy
  )
  if feed_forward is nil then
    pool_free(final_arena)
    abandon()
    give_back(ahead)
    return nil, "model: " .. (feed_line as string)
  end

  local attention: compute.Attention | nil = nil
  if has_full then
    local made, line = attention_of(
      backend, handle, geometry, box, measure, final_mapped
    )
    if made is nil then
      feed_forward:close()
      pool_free(final_arena)
      abandon()
      give_back(ahead)
      return nil, "model: " .. (line as string)
    end
    attention = made
  end

  local linear: compute.LinearAttention | nil = nil
  if has_linear then
    local made, line = linear_of(
      backend, handle, geometry, box, measure, final_mapped
    )
    if made is nil then
      feed_forward:close()
      local blocked = attention
      if blocked is compute.Attention then
        blocked:close()
      end
      pool_free(final_arena)
      abandon()
      give_back(ahead)
      return nil, "model: " .. (line as string)
    end
    linear = made
  end

  final_mapped.hold()
  return {
    mapped = final_mapped,
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
    placement = plan,
    copy = copy,
    ahead = ahead,
    miss = miss,
    zero_copy = zero_copy,
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
  if attention is compute.Attention and #state.full_prefixes > 0 then
    local line = attention:reserve(sequence, positions, state.full_prefixes)
    if line is string then
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
  if attention is compute.Attention then
    attention:release(sequence)
  end
  local linear = state.linear
  if linear is compute.LinearAttention then
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
-- the pass is a calibration window when it missed: its measured bytes and
-- seconds refine the share the next misses use, and the sixteenth leaves the
-- policy where it stands
local function calibrated(work: MissWindow)
  if not work.missed then
    return
  end
  work.missed = false
  local policy = work.policy
  if policy.windows >= 16 then
    return
  end
  policy.windows = policy.windows + 1
  if policy.cpu_seconds > 0 then
    policy.pool = policy.cpu_bytes / policy.cpu_seconds / 1000000000
  end
  if policy.windows >= 2 and policy.dma_seconds > 0 then
    policy.link = policy.dma_bytes / policy.dma_seconds / 1000000000
    policy.share = missx.share(policy.link, policy.pool)
  end
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
  if refusal is string then
    return nil, refusal
  end
  local arena, no_room = pass_arena(state, #inputs)
  if arena is nil then
    return nil, no_room
  end

  -- read-ahead decides once every sixteen completed passes, and a person who
  -- forced it is asked nothing
  local ahead = state.ahead
  if ahead.passes >= 16 then
    ahead.passes = 0
    decided(ahead)
  end

  local started = clock.now()
  local first = measure.tokens

  local ids: {integer} = {}
  for index, input in ipairs(inputs) do
    local refused = embedding_refusal(state.mapped, state.width, input.token)
    if refused is string then
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

  local cache = state.mapped.cache
  if cache is ExpertCache then
    cachex.begin(cache)
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
    local following: string | nil = nil
    if index < state.count - 1 then
      following = "blk." .. (index + 1) .. "."
    end
    local blocked, block_line = state.feed_forward:block(
      mixed, prefix, watchs, following
    )
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
  if cache is ExpertCache then
    local copy_line = cachex.finish(cache, state.mapped)
    if copy_line is string then
      return nil, copy_line
    end
  end
  calibrated(state.miss)

  for _, input in ipairs(inputs) do
    local held = state.sequences[input.sequence] as SequenceState
    held.taken = held.taken + 1
  end
  measure.passes = measure.passes + 1
  ahead.passes = ahead.passes + 1
  measure.tokens = first + #inputs
  measure.compute = measure.compute + (clock.now() - started)
  return scores, nil
end
local function build(
  backend: compute.Backend,
  handle: weights.Handle,
  positions: integer | nil,
  cache_type: compute.CacheType | nil,
  sequences?: integer,
  ubatch?: integer,
  vram?: integer,
  profile?: string,
  prefetch_value?: boolean
): compute.Model | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "model: backend was not loaded by compute.load"
  end
  if not (vram is nil) and vram < 1 then
    return nil, "model: vram " .. vram
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
    backend,
    handle,
    positions,
    cache_type,
    limit,
    rows,
    vram,
    profile,
    prefetch_value
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
    placement = plan.placement,
    copy = plan.copy,
    ahead = plan.ahead,
    miss = plan.miss,
  }

  local model = {} as compute.Model
  model.positions = plan.positions
  model.sequences = limit
  model.ubatch = rows
  model.measure = plan.measure
  model.placement = plan.placement
  model.read_ahead = plan.ahead.published
  model.zero_copy = plan.zero_copy
  model.miss = plan.miss.policy
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
    if closed_line is string then
      return nil, closed_line
    end
    if state.sequences[""] == nil then
      local line = sequence_reserve(state, "", state.positions)
      if line is string then
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
    if closed_line is string then
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
    if attention is compute.Attention then
      attention:close()
    end
    local linear = state.linear
    if linear is compute.LinearAttention then
      linear:close()
    end
    state.feed_forward:close()
    local pool = state.pool.pool
    if pool is Pool then
      pool_free(pool)
      state.pool.pool = nil
    end
    give_back(state.ahead)
    state.mapped.close()
    local copy = state.copy
    if copy is Copy then
      state.copy = nil
      free_copy(copy)
    end
  end
  model.save = function(
    _: compute.Model, sequence: string, ids: {integer}, path: string
  ): string | nil
    local closed_line = shut("save")
    if closed_line is string then
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
    if closed_line is string then
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
    if reserve_line is string then
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
      if line is string then
        return nil, give_up(line)
      end
    end
    (state.sequences[sequence] as SequenceState).taken = taken
    return ids, nil
  end
  model.pass = function(
    _: compute.Model, inputs: {compute.Input}
  ): {{number}} | nil, string | nil
    local closed_line = shut("pass")
    if closed_line is string then
      return nil, closed_line
    end
    return model_pass(state, inputs, measure, model.routing_sink)
  end
  return model
end

return {build = build}
