-- where a model's weights live: a handle's tensors mapped over the file, the
-- copies a card holds under a byte budget, and the expert slots that budget
-- fills block by block

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local cachex = require("tllm.compute.cache")
local engine = require("tllm.compute.engine")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Tensor = weights.Tensor

local type Copy = defs.Copy
local type Graph = defs.Graph
local type Loaded = defs.Loaded
local type Mapped = defs.Mapped
local type ExpertCache = defs.ExpertCache
local type Pool = defs.Pool
local type TensorView = defs.TensorView
local type Tiers = defs.Tiers

local ARENA_ALIGN = engine.ARENA_ALIGN
local COPIES = engine.COPIES
local arena_finalised = engine.arena_finalised
local arena_for = engine.arena_for
local arena_params = engine.arena_params
local device_copy = engine.device_copy
local guard_release = engine.guard_release
local holder = engine.holder
local host_tensor = engine.host_tensor
local libc = engine.libc
local pool_params = engine.pool_params
local round_up = engine.round_up
local tensor_in = engine.tensor_in

local tier = (
  function(): Tiers
    -- the fixed tensors a budget may take, in the order the eight tiers give
    local TIER_ATTENTION: {string} = {
      "attn_norm.weight",
      "attn_q.weight",
      "attn_k.weight",
      "attn_v.weight",
      "attn_q_norm.weight",
      "attn_k_norm.weight",
      "attn_output.weight",
      "post_attention_norm.weight",
    }

    local TIER_LINEAR: {string} = {
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
      "post_attention_norm.weight",
    }

    local TIER_ROUTERS: {string} = {
      "ffn_gate_inp.weight",
      "ffn_gate_inp_shexp.weight",
    }

    local TIER_SHARED: {string} = {
      "ffn_up_shexp.weight",
      "ffn_gate_shexp.weight",
      "ffn_down_shexp.weight",
    }

    local EXPERT_NAMES: {string} = {
      "ffn_up_exps.weight",
      "ffn_gate_exps.weight",
      "ffn_down_exps.weight",
    }

    -- the block numbers a handle names, in block order
    local function block_numbers(handle: weights.Handle): {integer}
      local seen: {integer: boolean} = {}
      for name in pairs(handle.tensors) do
        local found = name:match("^blk%.(%d+)%.")
        if found is string then
          seen[tonumber(found) as integer] = true
        end
      end
      local numbers: {integer} = {}
      for number in pairs(seen) do
        numbers[#numbers + 1] = number
      end
      table.sort(numbers)
      return numbers
    end

    local function full_block(
      tensors: {string: Tensor}, prefix: string
    ): boolean
      return tensors[prefix .. "attn_q.weight"] ~= nil
    end

    local function linear_shaped(
      tensors: {string: Tensor}, prefix: string
    ): boolean
      return tensors[prefix .. "attn_qkv.weight"] ~= nil
    end

    local function fixed_order(handle: weights.Handle): {string}
      local tensors = handle.tensors
      local numbers = block_numbers(handle)
      local order: {string} = {"token_embd.weight"}
      local function add(prefix: string, names: {string})
        for _, name in ipairs(names) do
          local entered = tensors[prefix .. name]
          if entered is Tensor then
            order[#order + 1] = prefix .. name
          end
        end
      end
      for _, number in ipairs(numbers) do
        local prefix = "blk." .. number .. "."
        if full_block(tensors, prefix) then
          add(prefix, TIER_ATTENTION)
        end
      end
      for _, number in ipairs(numbers) do
        local prefix = "blk." .. number .. "."
        if linear_shaped(tensors, prefix) then
          add(prefix, TIER_LINEAR)
        end
      end
      for _, number in ipairs(numbers) do
        add("blk." .. number .. ".", TIER_ROUTERS)
      end
      for _, number in ipairs(numbers) do
        add("blk." .. number .. ".", TIER_SHARED)
      end
      order[#order + 1] = "output_norm.weight"
      order[#order + 1] = "output.weight"
      return order
    end

    -- the experts a block holds, or none when it holds no expert matrix
    local function experts_held(
      tensors: {string: Tensor}, prefix: string
    ): integer
      local most = 0
      for _, name in ipairs(EXPERT_NAMES) do
        local tensor = tensors[prefix .. name]
        if tensor is Tensor and #tensor.dims >= 3 then
          local count = tensor.dims[3] as integer
          if count > most then
            most = count
          end
        end
      end
      return most
    end

    -- what one expert is charged: its three matrices, each rounded up to 128
    local function expert_charge(
      tensors: {string: Tensor}, prefix: string, index: integer
    ): integer
      local total = 0
      for _, name in ipairs(EXPERT_NAMES) do
        local tensor = tensors[prefix .. name]
        if tensor is Tensor and #tensor.dims >= 3 then
          local count = tensor.dims[3] as integer
          if index < count then
            total = total + round_up(math.floor(tensor.bytes / count), 128)
          end
        end
      end
      return total
    end

    local function same_placement(
      left: compute.Placement, right: compute.Placement
    ): boolean
      for name in pairs(left.resident) do
        if not right.resident[name] then
          return false
        end
      end
      for name in pairs(right.resident) do
        if not left.resident[name] then
          return false
        end
      end
      if #left.slots ~= #right.slots then
        return false
      end
      for index, count in ipairs(left.slots) do
        if right.slots[index] ~= count then
          return false
        end
      end
      return true
    end

    -- the plan for a budget and a reservation, over a handle's tensors
    local function place(
      handle: weights.Handle, budget: integer, reserve: integer
    ): compute.Placement
      local tensors = handle.tensors
      local numbers = block_numbers(handle)
      local placement: compute.Placement = {
        budget = budget,
        reserve = reserve,
        resident = {},
        resident_bytes = 0,
        slots = {},
        slot_bytes = 0,
        unplaced_bytes = 0,
      }
      local total = 0
      for _, tensor in pairs(tensors) do
        total = total + tensor.bytes
      end
      local limits: {integer} = {}
      for index, number in ipairs(numbers) do
        placement.slots[index] = 0
        limits[index] = experts_held(tensors, "blk." .. number .. ".")
      end

      local remaining = budget - reserve
      for _, name in ipairs(fixed_order(handle)) do
        local tensor = tensors[name]
        if tensor is Tensor then
          local charge = round_up(tensor.bytes, 128)
          if charge <= remaining then
            placement.resident[name] = true
            placement.resident_bytes = placement.resident_bytes + charge
            remaining = remaining - charge
          end
        end
      end

      while true do
        local taken = 0
        for index, number in ipairs(numbers) do
          local expert = placement.slots[index]
          if expert < limits[index] then
            local charge = expert_charge(
              tensors, "blk." .. number .. ".", expert
            )
            if charge <= remaining then
              placement.slots[index] = expert + 1
              placement.slot_bytes = placement.slot_bytes + charge
              remaining = remaining - charge
              taken = taken + 1
            end
          end
        end
        if taken == 0 then
          break
        end
      end

      placement.unplaced_bytes = math.max(
        0, total - placement.resident_bytes - placement.slot_bytes
      )
      return placement
    end

    -- the bytes a pool's own graph buffers need: one per block and one for
    -- the pass's embedding and score graphs together
    local function reserve_of(
      pool: Pool, kinds: {boolean}, count: integer
    ): integer
      local function held(kind: string): integer
        local bytes = pool.device_held[kind]
        if bytes is nil or bytes == 0 then
          return 0
        end
        return round_up(bytes, ARENA_ALIGN)
      end
      -- one buffer serves every graph of one prefix: a block's three graphs,
      -- and the pass's embedding and score graphs
      local function most(names: {string}): integer
        local bytes = 0
        for _, name in ipairs(names) do
          if held(name) > bytes then
            bytes = held(name)
          end
        end
        return bytes
      end
      local total = most({"embed", "scores"})
      for index = 1, count do
        local half = "linear"
        if kinds[index] then
          half = "attention"
        end
        total = total + most({half, "votes", "choice"})
      end
      return total
    end

    -- the device copy a model's placement names: the tensors it marks and every
    -- expert matrix it places, in one buffer of the model's own
    local function placed_copy(
      loaded: Loaded,
      handle: weights.Handle,
      placement: compute.Placement,
      seeds: {{integer}}
    ): Copy | nil
      local lib = loaded.lib
      local tensors = handle.tensors
      local numbers = block_numbers(handle)
      local size = placement.resident_bytes + placement.slot_bytes
      if size <= 0 then
        return nil
      end

      local descriptor_count = 0
      for _, name in ipairs(fixed_order(handle)) do
        if placement.resident[name] then
          descriptor_count = descriptor_count + 1
        end
      end
      for index, number in ipairs(numbers) do
        local prefix = "blk." .. number .. "."
        local slots = placement.slots[index]
        for _ = 0, slots - 1 do
          for _, name in ipairs(EXPERT_NAMES) do
            local entered = tensors[prefix .. name]
            if entered is Tensor then
              descriptor_count = descriptor_count + 1
            end
          end
        end
      end
      if descriptor_count == 0 then
        return nil
      end

      local found = loaded.device as ffi.CData
      local backend = lib.ggml_backend_dev_init(found, nil)
      if backend is nil then
        return nil
      end
      local arena_size = arena_for(lib, descriptor_count, 0)
      local arena = libc.aligned_alloc(ARENA_ALIGN, arena_size)
      if arena is nil then
        lib.ggml_backend_free(backend)
        return nil
      end
      local ctx = lib.ggml_init(pool_params(arena_size, arena, true))
      if ctx is nil then
        libc.free(arena)
        lib.ggml_backend_free(backend)
        return nil
      end
      local buffer = lib.ggml_backend_alloc_buffer(backend, size)
      if buffer is nil then
        lib.ggml_free(ctx)
        libc.free(arena)
        lib.ggml_backend_free(backend)
        return nil
      end
      local base = lib.ggml_backend_buffer_get_base(buffer)
      local base_at = ffi.cast("intptr_t", base) as integer

      local function failed(): nil
        lib.ggml_backend_buffer_free(buffer)
        lib.ggml_free(ctx)
        libc.free(arena)
        lib.ggml_backend_free(backend)
        return nil
      end

      local descriptors: {string: ffi.CData} = {}
      local experts: {string: {ffi.CData | nil}} = {}
      local offset: integer = 0
      local function give(
        dims: {integer}, code: integer, location: ffi.CData, bytes: integer
      ): ffi.CData | nil
        local descriptor = tensor_in(lib, ctx, dims, code)
        if descriptor is nil then
          return nil
        end
        local address = ffi.cast("void *", base_at + offset) as ffi.CData
        if lib.ggml_backend_tensor_alloc(buffer, descriptor, address) ~= 0 then
          return nil
        end
        lib.ggml_backend_tensor_set(descriptor, location, 0, bytes)
        offset = offset + round_up(bytes, 128)
        return descriptor
      end

      for _, name in ipairs(fixed_order(handle)) do
        if placement.resident[name] then
          local tensor = tensors[name]
          if tensor is Tensor then
            local descriptor = give(
              tensor.dims, tensor.kind.code, tensor.data, tensor.bytes
            )
            if descriptor is nil then
              return failed()
            end
            descriptors[name] = descriptor
          end
        end
      end

      for index, number in ipairs(numbers) do
        local prefix = "blk." .. number .. "."
        local slots = placement.slots[index]
        local chosen = seeds[index] or {}
        for slot = 0, slots - 1 do
          local expert = chosen[slot + 1]
          for _, name in ipairs(EXPERT_NAMES) do
            local tensor = tensors[prefix .. name]
            if
              tensor is Tensor and #tensor.dims >= 3 and not (expert is nil)
            then
              local count = tensor.dims[3] as integer
              local each = math.floor(tensor.bytes / count)
              local from = ffi.cast("intptr_t", tensor.data) as integer
              local location = ffi.cast("void *", from + expert * each)
                as ffi.CData
              local descriptor = give(
                {tensor.dims[1] as integer, tensor.dims[2] as integer},
                tensor.kind.code,
                location,
                each
              )
              if descriptor is nil then
                return failed()
              end
              local held = experts[name]
              if held is nil then
                held = {}
                experts[name] = held
              end
              held[slot + 1] = descriptor
            end
          end
        end
      end

      return {
        lib = lib,
        backend = backend,
        ctx = ctx,
        arena = arena,
        buffer = buffer,
        tensors = descriptors,
        experts = experts,
      } as Copy
    end
    -- a mapped handle over a model's own placement copy
    local function mapped_with_copy(
      loaded: Loaded,
      handle: weights.Handle,
      copy: Copy,
      placement: compute.Placement
    ): Mapped
      local mark = holder(handle)
      return {
        lib = loaded.lib,
        tensors = handle.tensors,
        descriptors = copy.tensors,
        handle = handle,
        device = copy,
        resident = placement.resident,
        cache = nil,
        plan = placement,
        hold = mark.hold,
        close = mark.close,
      }
    end

    -- the mark one block takes on a handle a model has already mapped
    local function borrowed(mapped: Mapped): Mapped
      local mark = holder(mapped.handle)
      return {
        lib = mapped.lib,
        tensors = mapped.tensors,
        descriptors = mapped.descriptors,
        handle = mapped.handle,
        device = mapped.device,
        arena = mapped.arena,
        ctx = mapped.ctx,
        resident = mapped.resident,
        cache = mapped.cache,
        plan = mapped.plan,
        hold = mark.hold,
        close = mark.close,
      }
    end

    -- the device descriptor of the slot holding this expert, or nil when no
    -- slot does
    local function slot_descriptor(
      mapped: Mapped, name: string, expert: integer
    ): ffi.CData | nil
      local held = mapped.cache
      if not (held is ExpertCache) then
        return nil
      end
      local copy = mapped.device
      if not (copy is Copy) then
        return nil
      end
      local matrices = copy.experts
      if matrices is nil then
        return nil
      end
      local descriptors = matrices[name]
      if descriptors is nil then
        return nil
      end
      local slot = cachex.slot_of(held, cachex.block_of(name), expert)
      if slot is nil then
        return nil
      end
      return descriptors[slot]
    end

    -- the descriptor a graph multiplies by: the model's device copy when its plan
    -- marks the tensor, the mapping for the length of the call when it does not
    local function weight_of(
      graph: Graph, mapped: Mapped, name: string, index?: integer
    ): ffi.CData
      local tensor = mapped.tensors[name] as Tensor
      local dims = tensor.dims
      local plan = mapped.plan
      if #dims < 3 then
        if plan is compute.Placement and not plan.resident[name] then
          return host_tensor(
            graph, dims, tensor.kind.code, tensor.data, tensor.bytes
          )
        end
        return mapped.descriptors[name]
      end

      if plan is compute.Placement then
        local descriptor = slot_descriptor(mapped, name, index or 0)
        if not (descriptor is nil) then
          return descriptor
        end
      end

      local stride = math.floor(tensor.bytes / (dims[3] as integer))
      local base = ffi.cast("intptr_t", tensor.data) as integer
      local location = ffi.cast("void *", base + (index or 0) * stride)
        as ffi.CData
      return host_tensor(
        graph,
        {dims[1] as integer, dims[2] as integer},
        tensor.kind.code,
        location,
        stride
      )
    end

    return {
      attention = TIER_ATTENTION,
      linear = TIER_LINEAR,
      routers = TIER_ROUTERS,
      shared = TIER_SHARED,
      experts = EXPERT_NAMES,
      block_numbers = block_numbers,
      fixed_order = fixed_order,
      experts_held = experts_held,
      expert_charge = expert_charge,
      same_placement = same_placement,
      place = place,
      reserve_of = reserve_of,
      placed_copy = placed_copy,
      mapped_with_copy = mapped_with_copy,
      borrowed = borrowed,
      weight_of = weight_of,
    }
  end
)()
local function mapped_of(
  backend: compute.Backend,
  loaded: Loaded,
  handle: weights.Handle,
  operation: string,
  use_copy?: boolean
): Mapped | nil, string | nil
  local lib = loaded.lib
  backend.device = nil
  guard_release(handle)
  local found = loaded.device
  local offload = loaded.offload
  if use_copy == false then
    offload = false
  end
  if offload and not (found is nil) then
    local held = COPIES[handle]
    local copy: Copy | nil = nil
    if held is {compute.Backend: Copy} then
      copy = held[backend]
    end
    if copy is nil then
      copy = device_copy(loaded, handle)
      if copy is Copy then
        if held is nil then
          held = {}
          COPIES[handle] = held
        end
        held[backend] = copy
      end
    end
    if copy is Copy then
      backend.device = ffi.string(lib.ggml_backend_dev_name(found as ffi.CData))
      local mark = holder(handle)
      return {
        lib = lib,
        tensors = handle.tensors,
        descriptors = copy.tensors,
        handle = handle,
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
    handle = handle,
    arena = arena,
    ctx = ctx,
    hold = mark.hold,
    close = mark.close,
  }
end

return {tier = tier, mapped_of = mapped_of}
