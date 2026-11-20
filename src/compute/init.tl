-- the published module: the table api.tl declares, filled in from the
-- modules below, so every other file keeps requiring `tllm.compute`

local ffi = require("ffi")
local compute = require("tllm.compute.api")
local attn = require("tllm.compute.attention")
local engine = require("tllm.compute.engine")
local ff = require("tllm.compute.feed_forward")
local lin = require("tllm.compute.linear")
local mathx = require("tllm.compute.maths")
local mdl = require("tllm.compute.model")
local oncard = require("tllm.compute.placement")
local defs = require("tllm.compute.types")
local weights = require("tllm.weights")

local type Ints = {integer: integer}

local type Library = defs.Library
local type Mapped = defs.Mapped

local DEFAULT_PATH = engine.DEFAULT_PATH
local EXPECTED_VERSION = engine.EXPECTED_VERSION
local GGML_DEVICE_GPU = engine.GGML_DEVICE_GPU
local LOADED = engine.LOADED
local find_backends = engine.find_backends
local load_library = engine.load_library
local probe_version = engine.probe_version
local refusal_line = engine.refusal_line

local mapped_of = oncard.mapped_of
local tier = oncard.tier

local add = mathx.add
local matrix_vector = mathx.matrix_vector
local normalise = mathx.normalise
local rotary = mathx.rotary
local row = mathx.row
local sigmoid_gate = mathx.sigmoid_gate
local softmax = mathx.softmax
local swiglu = mathx.swiglu

local attention_of = attn.attention_of

local feed_forward_of = ff.feed_forward_of

local linear_of = lin.linear_of

function compute.place(
  handle: weights.Handle, budget: integer, reserve: integer
): compute.Placement
  return tier.place(handle, budget, reserve)
end

function compute.load(
  path?: string, offload?: boolean
): compute.Backend | nil, string | nil
  local location = path or DEFAULT_PATH
  local made, lib = pcall(load_library, location)
  if not made then
    return nil, refusal_line(location, lib as string)
  end
  local handle = lib as Library

  local known, version = pcall(probe_version, handle)
  if not known then
    return nil, refusal_line(location, version as string)
  end
  if (version as string) ~= EXPECTED_VERSION then
    return nil,
      refusal_line(
        location,
        "version "
          .. (version as string)
          .. " is not the pinned "
          .. EXPECTED_VERSION
      )
  end

  find_backends(handle, location)

  -- the card is reported even when the caller asked to stay on the CPU, and
  -- used only when they did not
  local found = handle.ggml_backend_dev_by_type(GGML_DEVICE_GPU)

  local backend = {
    path = location,
    version = version as string,
  } as compute.Backend
  backend.memory = function(_: compute.Backend): integer | nil, integer | nil
    if found is nil then
      return nil, nil
    end
    local free_bytes = ffi.new("size_t[1]")
    local total_bytes = ffi.new("size_t[1]")
    handle.ggml_backend_dev_memory(found, free_bytes, total_bytes)
    local sizes = free_bytes as Ints
    local totals = total_bytes as Ints
    return tonumber(sizes[0]) as integer, tonumber(totals[0]) as integer
  end
  LOADED[backend] = {
    lib = handle,
    device = found,
    offload = offload ~= false,
  }
  return backend
end

-- forgets the library a backend opened, so the pool's parent keeps no device
function compute.unload(backend: compute.Backend)
  LOADED[backend] = nil
end

function compute.attach(
  backend: compute.Backend, handle: weights.Handle
): compute.Maths | nil, string | nil
  local loaded = LOADED[backend]
  if loaded is nil then
    return nil, "attach: backend was not loaded by compute.load"
  end
  local mapped, map_failure = mapped_of(backend, loaded, handle, "attach")
  if mapped is nil then
    return nil, map_failure
  end
  local attached: Mapped = mapped
  attached.hold()

  local finished = false
  local function shut(operation: string): string | nil
    if finished then
      return operation .. ": the maths are closed"
    end
    return nil
  end

  local maths = {} as compute.Maths
  maths.matvec = function(
    _: compute.Maths, name: string, vector: {number}, index?: integer
  ): {number} | nil, string | nil
    local line = shut("matvec")
    if line is string then
      return nil, line
    end
    return matrix_vector(mapped, name, vector, index)
  end
  maths.row = function(
    _: compute.Maths, name: string, id: integer
  ): {number} | nil, string | nil
    local line = shut("row")
    if line is string then
      return nil, line
    end
    return row(mapped, name, id)
  end
  maths.norm = function(
    _: compute.Maths, vector: {number}, gain: string, eps: number
  ): {number} | nil, string | nil
    local line = shut("norm")
    if line is string then
      return nil, line
    end
    return normalise(mapped, vector, gain, eps)
  end
  maths.rope = function(
    _: compute.Maths,
    vector: {number},
    position: integer,
    dimensions: integer,
    base: number
  ): {number} | nil, string | nil
    local line = shut("rope")
    if line is string then
      return nil, line
    end
    return rotary(mapped, vector, position, dimensions, base)
  end
  maths.softmax = function(
    _: compute.Maths, vector: {number}
  ): {number} | nil, string | nil
    local line = shut("softmax")
    if line is string then
      return nil, line
    end
    return softmax(mapped, vector)
  end
  maths.add = function(
    _: compute.Maths, left: {number}, right: {number}
  ): {number} | nil, string | nil
    local line = shut("add")
    if line is string then
      return nil, line
    end
    return add(mapped, left, right)
  end
  maths.swiglu = function(
    _: compute.Maths, gate: {number}, up: {number}
  ): {number} | nil, string | nil
    local line = shut("swiglu")
    if line is string then
      return nil, line
    end
    return swiglu(mapped, gate, up)
  end
  maths.sigmoid_gate = function(
    _: compute.Maths, gate: {number}, values: {number}
  ): {number} | nil, string | nil
    local line = shut("sigmoid_gate")
    if line is string then
      return nil, line
    end
    return sigmoid_gate(mapped, gate, values)
  end
  maths.close = function(_: compute.Maths)
    if finished then
      return
    end
    finished = true
    attached.close()
  end
  return maths
end

function compute.attend(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.Attention | nil, string | nil
  return attention_of(
    backend, handle, geometry, {pool = nil, owned = true}, nil
  )
end

function compute.feed_forward(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.FeedForward | nil, string | nil
  return feed_forward_of(
    backend, handle, geometry, {pool = nil, owned = true}, nil
  )
end

function compute.linear_attention(
  backend: compute.Backend, handle: weights.Handle, geometry: compute.Geometry
): compute.LinearAttention | nil, string | nil
  return linear_of(backend, handle, geometry, {pool = nil, owned = true}, nil)
end

compute.model = mdl.build

return compute
