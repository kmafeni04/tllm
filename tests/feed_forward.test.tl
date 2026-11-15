local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")
local weights = require("tllm.weights")

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

local function u16(value: integer): string
  return string.char(value % 256, math.floor(value / 256) % 256)
end

local function u32(value: integer): string
  return string.char(
    value % 256,
    math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256,
    math.floor(value / 16777216) % 256
  )
end

local function u64(value: integer): string
  local parts: {string} = {}
  local rest = value
  for _ = 1, 8 do
    parts[#parts + 1] = string.char(rest % 256)
    rest = math.floor(rest / 256)
  end
  return table.concat(parts)
end

local function text(bytes: string): string
  return u64(#bytes) .. bytes
end

local type Floats = {integer: number}

local FLOAT_BYTES: integer = 4

local function floats(values: {number}): string
  local parts: {string} = {}
  local cell = ffi.new("float[1]") as Floats
  for _, value in ipairs(values) do
    cell[0] = value
    parts[#parts + 1] = ffi.string(cell as ffi.CData, FLOAT_BYTES)
  end
  return table.concat(parts)
end

local function halves(count: integer): string
  return string.rep(u16(0x3C00), count)
end

local record Entry
  name: string
  dims: {integer}
  code: integer
  bytes: string
end

local function padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

local written = 0

local function fixture(entries: {Entry}): string
  local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(0)
  local body: {string} = {}
  local offset = 0
  for _, entry in ipairs(entries) do
    local dims: {string} = {}
    for _, dimension in ipairs(entry.dims) do
      dims[#dims + 1] = u64(dimension)
    end
    head = head
      .. text(entry.name)
      .. u32(#entry.dims)
      .. table.concat(dims)
      .. u32(entry.code)
      .. u64(offset)
    body[#body + 1] = padded(entry.bytes, 32)
    offset = offset + #body[#body]
  end
  written = written + 1
  local location = ROOT .. "/feed_forward" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function zeros(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 0
  end
  return result
end

local function ones(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 1
  end
  return result
end

local function f32(name: string, dims: {integer}, values: {number}): Entry
  return {name = name, dims = dims, code = 0, bytes = floats(values)}
end

local function expert_identity(scale: number, experts: integer): {number}
  local values: {number} = {}
  for _ = 1, experts do
    values[#values + 1] = scale
    values[#values + 1] = 0
    values[#values + 1] = 0
    values[#values + 1] = scale
  end
  return values
end

local function expert_down(experts: integer): {number}
  local values: {number} = {}
  for expert = 1, experts do
    values[#values + 1] = expert
    values[#values + 1] = 0
    values[#values + 1] = 2 * expert
    values[#values + 1] = 0
  end
  return values
end

-- two positions, four experts and a three row vocabulary; every gate is 100
-- times the identity and every up a hundredth, so the swiglu is the squared
-- normalised hidden and the router's four probabilities are 1, 2, 3 and 4
-- over ten
local function toy(prefix: string): {Entry}
  return {
    f32(prefix .. "post_attention_norm.weight", {2}, ones(2)),
    f32(
      prefix .. "ffn_gate_inp.weight",
      {2, 4},
      {
        0,
        0,
        math.log(2),
        0,
        math.log(3),
        0,
        math.log(4),
        0,
      }
    ),
    f32(prefix .. "ffn_gate_inp_shexp.weight", {2}, {math.log(3), 0}),
    f32(prefix .. "ffn_up_shexp.weight", {2, 2}, {0.01, 0, 0, 0.01}),
    f32(prefix .. "ffn_gate_shexp.weight", {2, 2}, {100, 0, 0, 100}),
    f32(prefix .. "ffn_down_shexp.weight", {2, 2}, {1, 0, 0, 2}),
    f32(prefix .. "ffn_up_exps.weight", {2, 2, 4}, expert_identity(0.01, 4)),
    f32(prefix .. "ffn_gate_exps.weight", {2, 2, 4}, expert_identity(100, 4)),
    f32(prefix .. "ffn_down_exps.weight", {2, 2, 4}, expert_down(4)),
  }
end

local function toy_entries(): {Entry}
  local entries = toy("blk.0.")
  entries[#entries + 1] = f32("output_norm.weight", {2}, ones(2))
  entries[#entries + 1] = f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1})
  return entries
end

-- attention's toy block from 12's T-001
local function attention(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {4}, ones(4)),
    f32(
      prefix .. "attn_q.weight",
      {4, 8},
      {
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1e9,
        0,
        0,
        0,
        -1e9,
        0,
        0,
        0,
      }
    ),
    f32(prefix .. "attn_k.weight", {4, 2}, {1, 0, 0, 0, 0, 1, 0, 0}),
    f32(prefix .. "attn_v.weight", {4, 2}, {2, 0, 0, 0, 0, 4, 0, 0}),
    f32(prefix .. "attn_q_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_k_norm.weight", {2}, ones(2)),
    f32(
      prefix .. "attn_output.weight",
      {4, 4},
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
    ),
  }
end

-- one expert joining two blocks, its gate saturating so its output is 100
-- times the normalised hidden's square; the shared expert is all zeros
local function stacked(prefix: string): {Entry}
  return {
    f32(prefix .. "post_attention_norm.weight", {4}, ones(4)),
    f32(
      prefix .. "ffn_gate_inp.weight",
      {4, 2},
      {
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      }
    ),
    f32(prefix .. "ffn_gate_inp_shexp.weight", {4}, zeros(4)),
    f32(prefix .. "ffn_up_shexp.weight", {4, 4}, zeros(16)),
    f32(prefix .. "ffn_gate_shexp.weight", {4, 4}, zeros(16)),
    f32(prefix .. "ffn_down_shexp.weight", {4, 4}, zeros(16)),
    f32(
      prefix .. "ffn_up_exps.weight",
      {4, 4, 2},
      {
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      }
    ),
    f32(
      prefix .. "ffn_gate_exps.weight",
      {4, 4, 2},
      {
        100,
        0,
        0,
        0,
        0,
        100,
        0,
        0,
        0,
        0,
        100,
        0,
        0,
        0,
        0,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      }
    ),
    f32(
      prefix .. "ffn_down_exps.weight",
      {4, 4, 2},
      {
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      }
    ),
  }
end

local function appended(entries: {Entry}, more: {Entry}): {Entry}
  for _, entry in ipairs(more) do
    entries[#entries + 1] = entry
  end
  return entries
end

local function stacked_entries(): {Entry}
  local entries = appended(attention("blk.0."), attention("blk.1."))
  entries = appended(entries, stacked("blk.0."))
  entries = appended(entries, stacked("blk.1."))
  entries[#entries + 1] = f32("output_norm.weight", {4}, ones(4))
  entries[#entries + 1] = f32(
    "output.weight",
    {4, 3},
    {
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
    }
  )
  return entries
end

local function without(entries: {Entry}, name: string): {Entry}
  local result: {Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name ~= name then
      result[#result + 1] = entry
    end
  end
  return result
end

local function replaced(
  entries: {Entry}, name: string, replacement: Entry
): {Entry}
  local result: {Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name == name then
      result[#result + 1] = replacement
    else
      result[#result + 1] = entry
    end
  end
  return result
end

local function loaded(entries: {Entry}): weights.Handle
  local location = fixture(entries)
  local handle, failure = weights.load(location)
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function geometry(): compute.Geometry
  return {
    heads = 1,
    kv_heads = 1,
    head_length = 2,
    rope_dimensions = 2,
    rope_base = 10000,
    eps = 0,
    positions = 1,
    experts_used = 2,
    cache_type = "f32",
  }
end

local function stacked_geometry(): compute.Geometry
  return {
    heads = 2,
    kv_heads = 1,
    head_length = 2,
    rope_dimensions = 2,
    rope_base = 10000,
    eps = 0,
    positions = 4,
    experts_used = 1,
    cache_type = "f32",
  }
end

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function forwarded(
  handle: weights.Handle, shape: compute.Geometry
): compute.FeedForward
  local feed_forward, failure = compute.feed_forward(backend, handle, shape)
  test.equal(failure, nil)
  return feed_forward as compute.FeedForward
end

local function attended(
  handle: weights.Handle, shape: compute.Geometry
): compute.Attention
  local attention_handle, failure = compute.attend(backend, handle, shape)
  test.equal(failure, nil)
  return attention_handle as compute.Attention
end

local function refused(value: {number} | nil, line: string | nil): string
  test.equal(value, nil)
  test.truthy(line)
  local message = line as string
  test.equal(message:find("\n"), nil)
  return message
end

local function close(got: {number} | nil, wanted: {number})
  local values = got as {number}
  test.equal(#values, #wanted)
  for index, number in ipairs(wanted) do
    test.less(math.abs(values[index] - number), 1e-5)
  end
end

test.it(
  "one block's feed-forward adds its experts and shared expert to the hidden",
  function()
    local feed_forward = forwarded(loaded(toy_entries()), geometry())
    local block = feed_forward.block
    close(block(feed_forward, {1, 1}, "blk.0."), {5.3214286, 9.6428571})
    close(block(feed_forward, {2, 2}, "blk.0."), {6.3214286, 10.6428571})

    local strict_shape = geometry()
    strict_shape.eps = 1
    local strict = forwarded(loaded(toy_entries()), strict_shape)
    close(strict.block(strict, {1, 1}, "blk.0."), {3.1178393, 5.2356787})
  end
)

test.it(
  "a router tie takes the lower experts and their weights sum to one",
  function()
    local entries = replaced(
      toy_entries(),
      "blk.0.ffn_gate_inp.weight",
      f32("blk.0.ffn_gate_inp.weight", {2, 4}, zeros(8))
    )
    local feed_forward = forwarded(loaded(entries), geometry())
    close(feed_forward.block(feed_forward, {1, 1}, "blk.0."), {3.25, 5.5})
  end
)

test.it(
  "the final norm gives one score per token",
  function()
    local feed_forward = forwarded(loaded(toy_entries()), geometry())
    close(
      feed_forward.scores(feed_forward, {3, 4}),
      {0.8485281, 1.1313708, 1.9798990}
    )

    local strict_shape = geometry()
    strict_shape.eps = 1
    local strict = forwarded(loaded(toy_entries()), strict_shape)
    close(strict.scores(strict, {3, 4}), {0.8164966, 1.0886621, 1.9051587})
  end
)

test.it(
  "a stack of attention blocks gives the top token",
  function()
    local shape = stacked_geometry()
    local handle = loaded(stacked_entries())
    local attention_handle = attended(handle, shape)
    local feed_forward = forwarded(handle, shape)
    test.deepEqual(
      compute.top_token(
        attention_handle, feed_forward, {1, 1, 1, 1}, {"blk.0.", "blk.1."}, 0
      ),
      2
    )

    local tied_entries = replaced(
      stacked_entries(),
      "output.weight",
      f32(
        "output.weight",
        {4, 2},
        {
          1,
          0,
          0,
          0,
          1,
          0,
          0,
          0,
        }
      )
    )
    local tied_handle = loaded(tied_entries)
    local tied = attended(tied_handle, shape)
    local tied_feed = forwarded(tied_handle, shape)
    test.deepEqual(
      compute.top_token(
        tied, tied_feed, {1, 1, 1, 1}, {"blk.0.", "blk.1."}, 0
      ),
      0
    )

    local fresh_handle = loaded(stacked_entries())
    local fresh = attended(fresh_handle, shape)
    local fresh_feed = forwarded(fresh_handle, shape)
    local index, line = compute.top_token(
      fresh, fresh_feed, {1, 1, 1, 1}, {"blk.0.", "blk.1."}, 1
    )
    test.equal(index, nil)
    test.includes(line as string, "blk.0.")
    test.includes(line as string, "1")
  end
)

test.it(
  "a name, type, shape or number that does not fit is refused with one line",
  function()
    local handle = loaded(toy_entries())
    local shape = geometry()

    local forged = {
      path = "forged",
      version = "forged",
      matvec = function(
        _: compute.Backend, _matrix: {{number}}, _vector: {number}
      ): {number} | nil, string | nil
        return nil, "forged"
      end,
    } as compute.Backend
    local built, forged_line = compute.feed_forward(forged, handle, shape)
    test.equal(built, nil)
    test.includes(forged_line as string, "feed_forward")

    local no_experts = geometry()
    no_experts.experts_used = 0
    local missing_experts, experts_line = compute.feed_forward(
      backend, handle, no_experts
    )
    test.equal(missing_experts, nil)
    test.includes(experts_line as string, "feed_forward")
    test.includes(experts_line as string, "0")

    local negative_eps = geometry()
    negative_eps.eps = -1
    local bad_eps, eps_line = compute.feed_forward(
      backend, handle, negative_eps
    )
    test.equal(bad_eps, nil)
    test.includes(eps_line as string, "feed_forward")
    test.includes(eps_line as string, "-1")

    local feed_forward = forwarded(handle, shape)
    local block = feed_forward.block

    local other_prefix_line = refused(block(feed_forward, {1, 1}, "blk.9."))
    test.includes(other_prefix_line, "blk.9.post_attention_norm.weight")

    local missing_router = forwarded(
      loaded(without(toy_entries(), "blk.0.ffn_gate_inp.weight")), shape
    )
    local missing_router_line = refused(
      missing_router.block(missing_router, {1, 1}, "blk.0.")
    )
    test.includes(missing_router_line, "ffn_gate_inp.weight")

    local half_router = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_gate_inp.weight",
          {
            name = "blk.0.ffn_gate_inp.weight",
            dims = {2, 4},
            code = 1,
            bytes = halves(8),
          }
        )
      ),
      shape
    )
    local half_router_line = refused(
      half_router.block(half_router, {1, 1}, "blk.0.")
    )
    test.includes(half_router_line, "ffn_gate_inp")
    test.includes(half_router_line, "f16")

    local half_up = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_up_exps.weight",
          {
            name = "blk.0.ffn_up_exps.weight",
            dims = {2, 2, 4},
            code = 1,
            bytes = halves(16),
          }
        )
      ),
      shape
    )
    local half_up_line = refused(half_up.block(half_up, {1, 1}, "blk.0."))
    test.includes(half_up_line, "ffn_up_exps")
    test.includes(half_up_line, "f16")

    local half_output = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "output.weight",
          {
            name = "output.weight",
            dims = {2, 3},
            code = 1,
            bytes = halves(6),
          }
        )
      ),
      shape
    )
    local half_output_line = refused(half_output.scores(half_output, {1, 1}))
    test.includes(half_output_line, "output")
    test.includes(half_output_line, "f16")

    for _, hidden in ipairs({ones(1), ones(3)}) do
      local short_line = refused(block(feed_forward, hidden, "blk.0."))
      test.includes(short_line, "post_attention_norm.weight")
    end

    local narrow_router = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_gate_inp.weight",
          f32("blk.0.ffn_gate_inp.weight", {1, 4}, zeros(4))
        )
      ),
      shape
    )
    local narrow_router_line = refused(
      narrow_router.block(narrow_router, {1, 1}, "blk.0.")
    )
    test.includes(narrow_router_line, "ffn_gate_inp.weight")

    local flat_down = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_down_exps.weight",
          f32("blk.0.ffn_down_exps.weight", {2, 2}, zeros(4))
        )
      ),
      shape
    )
    local flat_down_line = refused(flat_down.block(flat_down, {1, 1}, "blk.0."))
    test.includes(flat_down_line, "ffn_down_exps.weight")

    local narrow_shared = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_down_shexp.weight",
          f32("blk.0.ffn_down_shexp.weight", {2, 1}, zeros(2))
        )
      ),
      shape
    )
    local narrow_shared_line = refused(
      narrow_shared.block(narrow_shared, {1, 1}, "blk.0.")
    )
    test.includes(narrow_shared_line, "ffn_down_shexp.weight")

    local short_gate = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_gate_exps.weight",
          f32("blk.0.ffn_gate_exps.weight", {2, 2, 3}, zeros(12))
        )
      ),
      shape
    )
    local short_gate_line = refused(
      short_gate.block(short_gate, {1, 1}, "blk.0.")
    )
    test.includes(short_gate_line, "ffn_gate_exps.weight")

    local few_experts = toy_entries()
    few_experts = replaced(
      few_experts,
      "blk.0.ffn_gate_inp.weight",
      f32("blk.0.ffn_gate_inp.weight", {2, 2}, zeros(4))
    )
    few_experts = replaced(
      few_experts,
      "blk.0.ffn_up_exps.weight",
      f32("blk.0.ffn_up_exps.weight", {2, 2, 2}, expert_identity(0.01, 2))
    )
    few_experts = replaced(
      few_experts,
      "blk.0.ffn_gate_exps.weight",
      f32("blk.0.ffn_gate_exps.weight", {2, 2, 2}, expert_identity(100, 2))
    )
    few_experts = replaced(
      few_experts,
      "blk.0.ffn_down_exps.weight",
      f32("blk.0.ffn_down_exps.weight", {2, 2, 2}, expert_down(2))
    )
    local few_shape = geometry()
    few_shape.experts_used = 3
    local few = forwarded(loaded(few_experts), few_shape)
    local few_line = refused(few.block(few, {1, 1}, "blk.0."))
    test.includes(few_line, "ffn_gate_inp.weight")

    for _, hidden in ipairs({ones(1), ones(3)}) do
      local score_line = refused(feed_forward.scores(feed_forward, hidden))
      test.includes(score_line, "output_norm.weight")
    end

    local no_output = forwarded(
      loaded(without(toy_entries(), "output.weight")), shape
    )
    local no_output_line = refused(no_output.scores(no_output, {1, 1}))
    test.includes(no_output_line, "output.weight")

    local flat_output = forwarded(
      loaded(
        replaced(
          toy_entries(), "output.weight", f32("output.weight", {1, 3}, zeros(3))
        )
      ),
      shape
    )
    local flat_output_line = refused(flat_output.scores(flat_output, {1, 1}))
    test.includes(flat_output_line, "output.weight")

    local absent_down = without(toy_entries(), "blk.0.ffn_down_exps.weight")
    absent_down = replaced(
      absent_down,
      "blk.0.ffn_gate_inp.weight",
      {
        name = "blk.0.ffn_gate_inp.weight",
        dims = {2, 4},
        code = 1,
        bytes = halves(8),
      }
    )
    local absent = forwarded(loaded(absent_down), shape)
    local absent_line = refused(absent.block(absent, {1, 1}, "blk.0."))
    test.includes(absent_line, "ffn_down_exps.weight")

    local mistyped = replaced(
      toy_entries(),
      "blk.0.ffn_gate_inp.weight",
      {
        name = "blk.0.ffn_gate_inp.weight",
        dims = {2, 4},
        code = 1,
        bytes = halves(8),
      }
    )
    mistyped = replaced(
      mistyped,
      "blk.0.ffn_down_exps.weight",
      f32("blk.0.ffn_down_exps.weight", {2, 2}, zeros(4))
    )
    local mistyped_feed = forwarded(loaded(mistyped), shape)
    local mistyped_line = refused(
      mistyped_feed.block(mistyped_feed, {1, 1}, "blk.0.")
    )
    test.includes(mistyped_line, "ffn_gate_inp")
    test.includes(mistyped_line, "f16")

    local flat_only = forwarded(
      loaded(
        replaced(
          toy_entries(),
          "blk.0.ffn_down_exps.weight",
          f32("blk.0.ffn_down_exps.weight", {2, 2}, zeros(4))
        )
      ),
      shape
    )
    local flat_only_line = refused(flat_only.block(flat_only, {1, 1}, "blk.0."))
    test.includes(flat_only_line, "ffn_down_exps.weight")

    local shape_stack = stacked_geometry()
    local stack_attention = attended(loaded(stacked_entries()), shape_stack)
    local stack_feed = forwarded(loaded(stacked_entries()), shape_stack)
    local empty_index, empty_line = compute.top_token(
      stack_attention, stack_feed, {1, 1, 1, 1}, {}, 0
    )
    test.equal(empty_index, nil)
    test.includes(empty_line as string, "top_token")

    local no_block_router = loaded(
      without(stacked_entries(), "blk.1.ffn_gate_inp.weight")
    )
    local router_index, router_line = compute.top_token(
      attended(no_block_router, shape_stack),
      forwarded(no_block_router, shape_stack),
      {1, 1, 1, 1},
      {"blk.0.", "blk.1."},
      0
    )
    test.equal(router_index, nil)
    test.includes(router_line as string, "blk.1.ffn_gate_inp.weight")

    local no_block_norm = loaded(
      without(stacked_entries(), "blk.1.attn_norm.weight")
    )
    local norm_index, norm_line = compute.top_token(
      attended(no_block_norm, shape_stack),
      forwarded(no_block_norm, shape_stack),
      {1, 1, 1, 1},
      {"blk.0.", "blk.1."},
      0
    )
    test.equal(norm_index, nil)
    test.includes(norm_line as string, "blk.1.attn_norm.weight")
  end
)
