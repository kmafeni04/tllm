-- the expert cache: the experts a profile seeds each slot with, the routes a
-- window counts, and the slot a pass fills once its arithmetic has returned

local ffi = require("ffi")
local fs = require("fs")
local json = require("json")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type ExpertBlock = defs.ExpertBlock
local type ExpertCache = defs.ExpertCache
local type Copy = defs.Copy
local type Mapped = defs.Mapped
local type Tensor = weights.Tensor
local type JsonValue = json.Value
local type JsonDocument = json.Document

local MATRICES: {string} = {
  "ffn_up_exps.weight",
  "ffn_gate_exps.weight",
  "ffn_down_exps.weight",
}

-- a pass is a round, and the cache adapts after eight of them
local WINDOW: integer = 8

local cache = {}

-- the routing record a line holds, or nil and nil when the line is not one
local function record_of(line: string): integer | nil, {integer} | nil
  local parsed, document = pcall(json.Document.parse, line)
  if not parsed then
    return nil, nil
  end
  local held = document as JsonDocument
  local root = held:root()
  local layer_value: JsonValue | nil = nil
  local experts_value: JsonValue | nil = nil
  if root:is("object") then
    for key, value in root:children() do
      if key == "layer" then
        layer_value = value
      elseif key == "experts" then
        experts_value = value
      end
    end
  end

  local layer: integer | nil = nil
  if layer_value is JsonValue then
    local held_layer = layer_value:val()
    if held_layer is number and held_layer % 1 == 0 then
      layer = held_layer as integer
    end
  end
  local experts: {integer} | nil = nil
  if experts_value is JsonValue and experts_value:is("array") then
    local found: {integer} = {}
    local complete = true
    for _, item in experts_value:children() do
      if item is JsonValue then
        local expert = item:val()
        if expert is number and expert % 1 == 0 then
          found[#found + 1] = expert as integer
        else
          complete = false
          break
        end
      else
        complete = false
        break
      end
    end
    if complete then
      experts = found
    end
  end
  held:free()
  return layer, experts
end

-- the routes a dump holds, by layer and expert: one JSON object per line, so a
-- line that is not a routing record is refused with the file and its line number
function cache.read(
  file_path: string
): {integer: {integer: integer}} | nil, string | nil
  local text = fs.read(file_path)
  if text is nil then
    return nil, "model: profile " .. file_path .. ": cannot be read"
  end
  local counts: {integer: {integer: integer}} = {}
  local line_number = 0
  for line in (text as string):gmatch("([^\n]+)") do
    line_number = line_number + 1
    local layer, experts = record_of((line:gsub("\r$", "")))
    if layer is nil or experts is nil then
      return nil,
        "model: profile "
          .. file_path
          .. ": line "
          .. line_number
          .. " is not a routing record"
    end
    local counted = counts[layer]
    if counted is nil then
      counted = {}
      counts[layer] = counted
    end
    for _, expert in ipairs(experts) do
      counted[expert] = (counted[expert] or 0) + 1
    end
  end
  return counts, nil
end

-- the experts a block's slots start with: the ones the profile routed most
-- often, the lower id first on a tie, then the lowest ids it did not name
function cache.seeds(
  counts: {integer: {integer: integer}} | nil,
  limits: {integer},
  slots: {integer}
): {{integer}}
  local result: {{integer}} = {}
  for index, wanted in ipairs(slots) do
    local limit = limits[index] or 0
    local chosen: {integer} = {}
    local taken: {integer: boolean} = {}
    local ranked: {integer} = {}
    if wanted > 0 and not (counts is nil) then
      local counted = counts[index - 1]
      for expert, routes in pairs(counted) do
        if expert >= 0 and expert < limit and routes > 0 then
          ranked[#ranked + 1] = expert
        end
      end
      table.sort(
        ranked,
        function(left: integer, right: integer): boolean
          local a = counted[left] or 0
          local b = counted[right] or 0
          if a ~= b then
            return a > b
          end
          return left < right
        end
      )
    end
    for _, expert in ipairs(ranked) do
      if #chosen >= wanted then
        break
      end
      chosen[#chosen + 1] = expert
      taken[expert] = true
    end
    for expert = 0, limit - 1 do
      if #chosen >= wanted then
        break
      end
      if not taken[expert] then
        chosen[#chosen + 1] = expert
        taken[expert] = true
      end
    end
    result[index] = chosen
  end
  return result
end

function cache.make(seeds: {{integer}}, prefixes: {string}): ExpertCache
  local blocks: {ExpertBlock} = {}
  for index, prefix in ipairs(prefixes) do
    local slots: {integer} = {}
    for slot, expert in ipairs(seeds[index] or {}) do
      slots[slot] = expert
    end
    blocks[index] = {
      prefix = prefix,
      slots = slots,
      counts = {},
      last = {},
      pending = nil,
    }
  end
  return {blocks = blocks, rounds = 0, round = 0}
end

local function ranked(block: ExpertBlock): {integer: integer} | nil
  local wanted = #block.slots
  if wanted == 0 then
    return nil
  end
  local candidates: {integer: boolean} = {}
  for _, expert in ipairs(block.slots) do
    if expert >= 0 then
      candidates[expert] = true
    end
  end
  for expert in pairs(block.counts) do
    candidates[expert] = true
  end
  local order: {integer} = {}
  for expert in pairs(candidates) do
    order[#order + 1] = expert
  end
  if #order == 0 then
    return nil
  end
  local counts, last = block.counts, block.last
  table.sort(
    order,
    function(left: integer, right: integer): boolean
      local a = counts[left] or 0
      local b = counts[right] or 0
      if a ~= b then
        return a > b
      end
      if a > 0 then
        local first = last[left] or -1
        local second = last[right] or -1
        if first ~= second then
          return first > second
        end
      end
      return left < right
    end
  )
  local kept: {integer: boolean} = {}
  local chosen = #order
  if chosen > wanted then
    chosen = wanted
  end
  for index = 1, chosen do
    kept[order[index]] = true
  end

  local pending: {integer: integer} = {}
  local in_slot: {integer: boolean} = {}
  local free: {integer} = {}
  for slot, expert in ipairs(block.slots) do
    if expert >= 0 and kept[expert] then
      pending[slot] = expert
      in_slot[expert] = true
    else
      pending[slot] = -1
      free[#free + 1] = slot
    end
  end
  local admitted = 0
  for index = 1, chosen do
    local expert = order[index]
    if not in_slot[expert] then
      admitted = admitted + 1
      local target = free[admitted]
      if target is nil then
        return nil
      end
      pending[target] = expert
    end
  end
  return pending
end

-- the block position a `blk.N.` prefix names, one past N, or 0 when it names
-- no block at all
function cache.block_of(prefix: string): integer
  local found = prefix:match("^blk%.(%d+)%.")
  if found is string then
    return (tonumber(found) as integer) + 1
  end
  return 0
end

-- the slot an expert already holds, when it holds one
function cache.slot_of(
  entry: ExpertCache, index: integer, expert: integer
): integer | nil
  local block = entry.blocks[index]
  if block is nil then
    return nil
  end
  for slot, held in ipairs(block.slots) do
    if held == expert then
      return slot
    end
  end
  return nil
end

-- a pass begins: it is round `round`, and after eight completed rounds the
-- window's routes and held experts choose each block's slots
function cache.begin(entry: ExpertCache)
  entry.round = entry.round + 1
  local adapting = entry.rounds >= WINDOW
  if adapting then
    entry.rounds = 0
  end
  for _, block in ipairs(entry.blocks) do
    block.pending = nil
    if adapting then
      block.pending = ranked(block)
      block.counts = {}
      block.last = {}
    end
  end
end

-- one route of the current pass, answering whether the block held the expert
-- when the pass began
function cache.record(
  entry: ExpertCache, index: integer, expert: integer
): boolean
  local block = entry.blocks[index]
  if block is nil then
    return false
  end
  block.counts[expert] = (block.counts[expert] or 0) + 1
  block.last[expert] = entry.round
  return cache.slot_of(entry, index, expert) ~= nil
end

local function copied(
  copy: Copy,
  tensors: {string: Tensor},
  prefix: string,
  slot: integer,
  expert: integer
): string | nil
  local matrices = copy.experts
  if matrices is nil then
    return "model: no expert slots"
  end
  for _, name in ipairs(MATRICES) do
    local tensor = tensors[prefix .. name]
    if tensor is Tensor and #tensor.dims >= 3 then
      local count = tensor.dims[3] as integer
      local each = math.floor(tensor.bytes / count)
      local held = matrices[name]
      local descriptor: ffi.CData | nil = nil
      if held is {ffi.CData | nil} then
        descriptor = held[slot]
      end
      if descriptor is nil then
        return "model: no slot for " .. prefix .. name
      end
      local base = ffi.cast("intptr_t", tensor.data) as integer
      local location = ffi.cast("void *", base + expert * each) as ffi.CData
      copy.lib.ggml_backend_tensor_set(descriptor, location, 0, each)
    end
  end
  return nil
end

-- the pass's arithmetic has returned: fill each slot the window chose with the
-- expert's three matrices from the mapping, then mark that expert held
function cache.finish(entry: ExpertCache, mapped: Mapped): string | nil
  for _, block in ipairs(entry.blocks) do
    local pending = block.pending
    block.pending = nil
    if pending is {integer: integer} then
      local copy = mapped.device
      for slot, expert in pairs(pending) do
        if block.slots[slot] ~= expert then
          if expert >= 0 and copy is Copy then
            local line = copied(
              copy, mapped.tensors, block.prefix, slot, expert
            )
            if line is string then
              return line
            end
          end
          block.slots[slot] = expert
        end
      end
    end
  end
  entry.rounds = entry.rounds + 1
  return nil
end

return cache
