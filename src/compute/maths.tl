-- the arithmetic over a mapped handle, one graph at a time, the tensor names
-- each block kind asks the mapping for, and what a cached row costs in each
-- cache type

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local engine = require("tllm.compute.engine")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Tensor = weights.Tensor
local type Floats = {integer: number}

local type CacheKind = defs.CacheKind
local type Graph = defs.Graph
local type Mapped = defs.Mapped

local F32_BYTES = engine.F32_BYTES
local GGML_ROPE_TYPE_NEOX = engine.GGML_ROPE_TYPE_NEOX
local GGML_TYPE_F16 = engine.GGML_TYPE_F16
local GGML_TYPE_F32 = engine.GGML_TYPE_F32
local GGML_TYPE_Q4_0 = engine.GGML_TYPE_Q4_0
local GGML_TYPE_Q8_0 = engine.GGML_TYPE_Q8_0
local I32_BYTES = engine.I32_BYTES
local graph_tensor = engine.graph_tensor
local host_tensor = engine.host_tensor
local run = engine.run
local scalar_of = engine.scalar_of
local vector_of = engine.vector_of

local CACHE_KINDS: {string: CacheKind} = {
  f32 = {code = GGML_TYPE_F32, block_elements = 1, block_bytes = 4},
  f16 = {code = GGML_TYPE_F16, block_elements = 1, block_bytes = 2},
  q8_0 = {code = GGML_TYPE_Q8_0, block_elements = 32, block_bytes = 34},
  q4_0 = {code = GGML_TYPE_Q4_0, block_elements = 32, block_bytes = 18},
}
local WEIGHTS: {string: boolean} = {
  f32 = true,
  f16 = true,
  bf16 = true,
  q4_0 = true,
  q4_1 = true,
  q5_0 = true,
  q5_1 = true,
  q8_0 = true,
  q2_K = true,
  q3_K = true,
  q4_K = true,
  q5_K = true,
  q6_K = true,
  iq2_xxs = true,
  iq2_xs = true,
  iq2_s = true,
  iq3_xxs = true,
  iq3_s = true,
  iq1_s = true,
  iq1_m = true,
  iq4_nl = true,
  iq4_xs = true,
  tq1_0 = true,
  tq2_0 = true,
  q1_0 = true,
  q2_0 = true,
  mxfp4 = true,
  nvfp4 = true,
  f8_e4m3 = true,
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
local ATTENTION_NAMES: {string} = {
  "attn_norm.weight",
  "attn_q.weight",
  "attn_k.weight",
  "attn_v.weight",
  "attn_q_norm.weight",
  "attn_k_norm.weight",
  "attn_output.weight",
}
local ATTENTION_TYPES: {string: {string: boolean}} = {
  ["attn_norm.weight"] = F32_ONLY,
  ["attn_q.weight"] = WEIGHTS,
  ["attn_k.weight"] = WEIGHTS,
  ["attn_v.weight"] = WEIGHTS,
  ["attn_q_norm.weight"] = F32_ONLY,
  ["attn_k_norm.weight"] = F32_ONLY,
  ["attn_output.weight"] = WEIGHTS,
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

return {
  BLOCK_NAMES = BLOCK_NAMES,
  BLOCK_TYPES = BLOCK_TYPES,
  SCORE_NAMES = SCORE_NAMES,
  SCORE_TYPES = SCORE_TYPES,
  ATTENTION_NAMES = ATTENTION_NAMES,
  ATTENTION_TYPES = ATTENTION_TYPES,
  LINEAR_NAMES = LINEAR_NAMES,
  LINEAR_TYPES = LINEAR_TYPES,
  matrix_vector = matrix_vector,
  matrix_of = matrix_of,
  row = row,
  normalise = normalise,
  rotary = rotary,
  softmax = softmax,
  add = add,
  swiglu = swiglu,
  sigmoid_gate = sigmoid_gate,
  cache_kind_of = cache_kind_of,
  cache_row_size = cache_row_size,
  geometry_refusal = geometry_refusal,
  holds = holds,
  prefixed = prefixed,
}
