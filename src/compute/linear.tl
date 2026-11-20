-- a linear-attention block: the convolution window and recurrent state it
-- carries per sequence, their snapshots, and the graphs that advance them

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local scaffold = require("tllm.compute.blocks")
local engine = require("tllm.compute.engine")
local mathx = require("tllm.compute.maths")
local oncard = require("tllm.compute.placement")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Floats = {integer: number}
local type Tensor = weights.Tensor

local type Box = defs.Box
local type Build = defs.Build
local type Graph = defs.Graph
local type LinearBuilt = defs.LinearBuilt
local type LinearCarry = defs.LinearCarry
local type LinearKept = defs.LinearKept
local type LinearShape = defs.LinearShape
local type LinearState = defs.LinearState
local type Mapped = defs.Mapped
local type Output = defs.Output
local type Pool = defs.Pool

local F32_BYTES = engine.F32_BYTES
local GGML_TYPE_F32 = engine.GGML_TYPE_F32
local LOADED = engine.LOADED
local graph_closed = engine.graph_closed
local graph_numbers = engine.graph_numbers
local graph_run = engine.graph_run
local graph_tensor = engine.graph_tensor
local host_tensor = engine.host_tensor
local node = engine.node
local pool_free = engine.pool_free
local read_bytes = engine.read_bytes
local sized_pool = engine.sized_pool
local vector_of = engine.vector_of

local mapped_of = oncard.mapped_of
local tier = oncard.tier

local LINEAR_NAMES = mathx.LINEAR_NAMES
local LINEAR_TYPES = mathx.LINEAR_TYPES
local holds = mathx.holds
local prefixed = mathx.prefixed

local blank_rows = scaffold.blank_rows
local count_refusal = scaffold.count_refusal
local first_fault = scaffold.first_fault
local grown = scaffold.grown
local measure_pointer = scaffold.measure_pointer

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
  if held is {string: LinearKept} then
    kept = held[prefix]
  end
  if kept is nil and positions == 0 then
    return "", nil
  end
  if kept is nil then
    return nil, prefix .. " holds no history"
  end
  local active = kept
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
      tier.weight_of(graph, mapped, prefix .. "attn_norm.weight")
    )
  )
  local projected = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "attn_qkv.weight"),
      normalised
    )
  )
  local gates = node(
    graph,
    lib.ggml_mul_mat(
      ctx,
      tier.weight_of(graph, mapped, prefix .. "attn_gate.weight"),
      normalised
    )
  )
  local betas = node(
    graph,
    lib.ggml_sigmoid(
      ctx,
      node(
        graph,
        lib.ggml_mul_mat(
          ctx,
          tier.weight_of(graph, mapped, prefix .. "ssm_beta.weight"),
          normalised
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
                      ctx,
                      tier.weight_of(
                        graph, mapped, prefix .. "ssm_alpha.weight"
                      ),
                      normalised
                    )
                  ),
                  tier.weight_of(graph, mapped, prefix .. "ssm_dt.bias")
                )
              )
            )
          ),
          tier.weight_of(graph, mapped, prefix .. "ssm_a")
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
          ctx,
          window,
          tier.weight_of(graph, mapped, prefix .. "ssm_conv1d.weight")
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
          tier.weight_of(graph, mapped, prefix .. "ssm_norm.weight")
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
      tier.weight_of(graph, mapped, prefix .. "ssm_out.weight"),
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
    if held is {string: LinearKept} then
      kept = held[prefix]
    end
    local expected = 0
    if kept is LinearKept then
      expected = kept.count
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
-- the linear-attention handle over a mapped handle, sharing a model's arena
-- when one is given and holding one of its own otherwise
local function linear_of(
  backend: compute.Backend,
  handle: weights.Handle,
  geometry: compute.Geometry,
  pool: Box,
  measure?: compute.Measure,
  given?: Mapped
): compute.LinearAttention | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "linear_attention: backend was not loaded by compute.load"
  end
  local shape, geometry_line = linear_shape_of(geometry)
  if not (shape is LinearShape) then
    return nil, geometry_line
  end

  local mapped: Mapped | nil = nil
  if given is Mapped then
    mapped = tier.borrowed(given)
  else
    local made, map_failure = mapped_of(
      backend, loaded, handle, "linear_attention"
    )
    if made is nil then
      return nil, map_failure
    end
    mapped = made
  end
  local attachment = mapped as Mapped
  attachment.hold()

  local state: LinearState = {
    mapped = attachment,
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
    if line is string then
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
    if line is string then
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
    if line is string then
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
      if held is Pool then
        pool_free(held)
        state.pool.pool = nil
      end
    end
  end
  return linear
end

return {
  linear_of = linear_of,
  linear_shape_of = linear_shape_of,
  linear_builds = linear_builds,
}
