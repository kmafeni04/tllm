-- what the three block kinds share: the count and position refusals, the
-- buffers a build starts from, and the arena a block's graphs live in

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local engine = require("tllm.compute.engine")
local mathx = require("tllm.compute.maths")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Tensor = weights.Tensor

local type BlockTensors = defs.BlockTensors
local type Box = defs.Box
local type Mapped = defs.Mapped
local type Pool = defs.Pool

local ARENA_ALIGN = engine.ARENA_ALIGN
local finalised = engine.finalised
local libc = engine.libc
local pool_free = engine.pool_free

local ATTENTION_NAMES = mathx.ATTENTION_NAMES
local WEIGHTS = mathx.WEIGHTS
local prefixed = mathx.prefixed

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
  if held is Pool and held.rows >= rows then
    return held
  end
  local wider = size(rows)
  if wider is nil then
    return nil
  end
  if held is Pool then
    pool_free(held)
  end
  box.pool = wider
  return wider
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
return {
  count_line = count_line,
  count_refusal = count_refusal,
  read_tensors = read_tensors,
  position_refusal = position_refusal,
  measure_pointer = measure_pointer,
  blank_hidden = blank_hidden,
  blank_rows = blank_rows,
  even_votes = even_votes,
  hidden_width = hidden_width,
  prefix_named = prefix_named,
  grown = grown,
  first_fault = first_fault,
}
