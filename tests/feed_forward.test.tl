local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local weights = require("tllm.weights")
local type Entry = support.Entry
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, zeros = support.padded, support.f32, support.zeros
local ones, halves, without, replaced =
  support.ones, support.halves, support.without, support.replaced
local first, first_row, refused =
  support.first, support.first_row, support.refused

local function score_one(
  feed_forward: compute.FeedForward, hidden: {number}
): {number} | nil, string | nil
  local rows, line = feed_forward.scores(feed_forward, {hidden})
  if rows is nil then
    return nil, line
  end
  return (rows as {{number}})[1], line
end

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

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
-- one expert joining two blocks, its gate saturating so its output is 100
-- times the normalised hidden's square; the shared expert is all zeros

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

-- the numbers below are the CPU backend's, so these stay on the CPU
local backend = compute.load(nil, false) as compute.Backend

local function forwarded(
  handle: weights.Handle, shape: compute.Geometry
): compute.FeedForward
  local feed_forward, failure = compute.feed_forward(backend, handle, shape)
  test.equal(failure, nil)
  return feed_forward as compute.FeedForward
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
    close(
      first(block(feed_forward, first_row({1, 1}, 0), "blk.0.")),
      {5.3214286, 9.6428571}
    )
    close(
      first(block(feed_forward, first_row({2, 2}, 0), "blk.0.")),
      {6.3214286, 10.6428571}
    )

    local strict_shape = geometry()
    strict_shape.eps = 1
    local strict = forwarded(loaded(toy_entries()), strict_shape)
    close(
      first(strict.block(strict, first_row({1, 1}, 0), "blk.0.")),
      {3.1178393, 5.2356787}
    )
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
    close(
      first(feed_forward.block(feed_forward, first_row({1, 1}, 0), "blk.0.")),
      {3.25, 5.5}
    )
  end
)

test.it(
  "the final norm gives one score per token",
  function()
    local feed_forward = forwarded(loaded(toy_entries()), geometry())
    close(score_one(feed_forward, {3, 4}), {0.8485281, 1.1313708, 1.9798990})

    local strict_shape = geometry()
    strict_shape.eps = 1
    local strict = forwarded(loaded(toy_entries()), strict_shape)
    close(score_one(strict, {3, 4}), {0.8164966, 1.0886621, 1.9051587})
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

    local other_prefix_line = refused(
      first(block(feed_forward, first_row({1, 1}, 0), "blk.9."))
    )
    test.includes(other_prefix_line, "blk.9.post_attention_norm.weight")

    local missing_router = forwarded(
      loaded(without(toy_entries(), "blk.0.ffn_gate_inp.weight")), shape
    )
    local missing_router_line = refused(
      first(
        missing_router.block(missing_router, first_row({1, 1}, 0), "blk.0.")
      )
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
      first(half_router.block(half_router, first_row({1, 1}, 0), "blk.0."))
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
    local half_up_line = refused(
      first(half_up.block(half_up, first_row({1, 1}, 0), "blk.0."))
    )
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
    local half_output_line = refused(score_one(half_output, {1, 1}))
    test.includes(half_output_line, "output")
    test.includes(half_output_line, "f16")

    for _, hidden in ipairs({ones(1), ones(3)}) do
      local short_line = refused(
        first(block(feed_forward, first_row(hidden, 0), "blk.0."))
      )
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
      first(narrow_router.block(narrow_router, first_row({1, 1}, 0), "blk.0."))
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
    local flat_down_line = refused(
      first(flat_down.block(flat_down, first_row({1, 1}, 0), "blk.0."))
    )
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
      first(narrow_shared.block(narrow_shared, first_row({1, 1}, 0), "blk.0."))
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
      first(short_gate.block(short_gate, first_row({1, 1}, 0), "blk.0."))
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
    local few_line = refused(
      first(few.block(few, first_row({1, 1}, 0), "blk.0."))
    )
    test.includes(few_line, "ffn_gate_inp.weight")

    for _, hidden in ipairs({ones(1), ones(3)}) do
      local score_line = refused(score_one(feed_forward, hidden))
      test.includes(score_line, "output_norm.weight")
    end

    local no_output = forwarded(
      loaded(without(toy_entries(), "output.weight")), shape
    )
    local no_output_line = refused(score_one(no_output, {1, 1}))
    test.includes(no_output_line, "output.weight")

    local flat_output = forwarded(
      loaded(
        replaced(
          toy_entries(), "output.weight", f32("output.weight", {1, 3}, zeros(3))
        )
      ),
      shape
    )
    local flat_output_line = refused(score_one(flat_output, {1, 1}))
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
    local absent_line = refused(
      first(absent.block(absent, first_row({1, 1}, 0), "blk.0."))
    )
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
      first(mistyped_feed.block(mistyped_feed, first_row({1, 1}, 0), "blk.0."))
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
    local flat_only_line = refused(
      first(flat_only.block(flat_only, first_row({1, 1}, 0), "blk.0."))
    )
    test.includes(flat_only_line, "ffn_down_exps.weight")
  end
)
