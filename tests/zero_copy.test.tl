local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local compute = require("tllm.compute")
local weights = require("tllm.weights")

local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text, padded =
  support.u32, support.u64, support.text, support.padded
local f32, zeros, ones = support.f32, support.zeros, support.ones
local mint, mfloat, mtext, both =
  support.mint, support.mfloat, support.mtext, support.both
local without = support.without

ffi.cdef([[int truncate(const char *path, long length);]])
local record Raw
  truncate: function(string, integer): integer
end
local C = ffi.C as Raw

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

local written = 0

local function fixture(meta: {Meta}, entries: {Entry}): string
  local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(#meta)
  for _, entry in ipairs(meta) do
    head = head .. text(entry.name) .. u32(entry.code) .. entry.bytes
  end
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
  local location = ROOT .. "/zero" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return location
end

local function tensor_info(entry: Entry, offset: integer): string
  local dims: {string} = {}
  for _, dimension in ipairs(entry.dims) do
    dims[#dims + 1] = u64(dimension)
  end
  return text(entry.name)
    .. u32(#entry.dims)
    .. table.concat(dims)
    .. u32(entry.code)
    .. u64(offset)
end

-- the fixture's header alone, with a filler entry whose length puts the body's
-- first byte on a page boundary, and the file extended over bytes it does not
-- hold, so every tensor lies in a hole
local function sparse_fixture(meta: {Meta}, entries: {Entry}): string
  local function head_of(filler_bytes: string | nil): string
    local count = #meta + 1
    local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(count)
    for _, entry in ipairs(meta) do
      head = head .. text(entry.name) .. u32(entry.code) .. entry.bytes
    end
    if filler_bytes is string then
      head = head .. text("filler") .. u32(8) .. text(filler_bytes as string)
    end
    local offset = 0
    for _, entry in ipairs(entries) do
      head = head .. tensor_info(entry, offset)
      offset = offset + #padded(entry.bytes, 32)
    end
    return head
  end

  local body_bytes = 0
  for _, entry in ipairs(entries) do
    body_bytes = body_bytes + #padded(entry.bytes, 32)
  end
  local bare = #head_of("")
  local filler = string.rep("f", (4096 - bare % 4096) % 4096)
  local head = head_of(filler)
  test.equal(#head % 4096, 0)

  written = written + 1
  local location = ROOT .. "/zero" .. written .. ".gguf"
  fs.write(location, head)
  test.equal(C.truncate(location, #head + body_bytes), 0)
  return location
end

-- 15/T-001's metadata
local function base_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 4),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
  }
end

-- 38's routing fixture has two blocks and a second expert for each to choose
local function routing_meta(): {Meta}
  local meta = support.with_integer(base_meta(), "qwen35moe.block_count", 2)
  meta = support.with_integer(meta, "qwen35moe.context_length", 65)
  return support.with_integer(meta, "qwen35moe.full_attention_interval", 1)
end

local function attention_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
    f32(prefix .. "attn_k.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_v.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_q_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_k_norm.weight", {2}, ones(2)),
    f32(prefix .. "attn_output.weight", {2, 2}, zeros(4)),
  }
end

-- the first expert's matrices hold 1 and the second's hold 2, as `38`'s
-- routing fixture writes them
local function expert_values(experts: integer): {number}
  local values: {number} = {}
  for expert = 1, experts do
    for _ = 1, 4 do
      values[#values + 1] = expert
    end
  end
  return values
end

local function block_entries(prefix: string, count: integer): {Entry}
  local entries = both(
    attention_entries(prefix),
    {
      f32(prefix .. "post_attention_norm.weight", {2}, ones(2)),
      f32(prefix .. "ffn_gate_inp.weight", {2, count}, {1, 0, 0, 1}),
      f32(prefix .. "ffn_gate_inp_shexp.weight", {2}, zeros(2)),
      f32(prefix .. "ffn_up_shexp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_gate_shexp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_down_shexp.weight", {2, 2}, zeros(4)),
    }
  )
  for _, name in ipairs(
    {"ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight"}
  ) do
    entries = both(
      entries,
      {
        f32(prefix .. name, {2, 2, count}, expert_values(count)),
      }
    )
  end
  return entries
end

local ROOT_ENTRIES: {Entry} = {
  f32("token_embd.weight", {2, 2}, {1, 0, 3, 4}),
  f32("output_norm.weight", {2}, ones(2)),
  f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
}

-- the resident fixture: one block, two experts, its expert matrices 96 bytes in
-- all and written in full
local function resident_entries(): {Entry}
  return both(block_entries("blk.0.", 2), ROOT_ENTRIES)
end

-- the evicted fixture: the same block with expert matrices a kilobyte wide,
-- each matrix lying in a hole
local function evicted_entries(): {Entry}
  local entries = resident_entries()
  local function wide(name: string, dims: {integer}): Entry
    return f32(name, dims, zeros(dims[1] * dims[2] * dims[3]))
  end
  entries = support.replaced(
    entries,
    "blk.0.ffn_up_exps.weight",
    wide("blk.0.ffn_up_exps.weight", {2, 1024, 2})
  )
  entries = support.replaced(
    entries,
    "blk.0.ffn_gate_exps.weight",
    wide("blk.0.ffn_gate_exps.weight", {2, 1024, 2})
  )
  return support.replaced(
    entries,
    "blk.0.ffn_down_exps.weight",
    wide("blk.0.ffn_down_exps.weight", {1024, 2, 2})
  )
end

local EXPERT_NAMES: {string} = {
  "ffn_up_exps.weight",
  "ffn_gate_exps.weight",
  "ffn_down_exps.weight",
}

local function loaded(path: string): weights.Handle
  local handle, failure = weights.load(path)
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function built(handle: weights.Handle): compute.Model
  local backend = compute.load(nil, false) as compute.Backend
  local model, failure = compute.model(
    backend, handle, nil, "f32", 1, 1, nil, nil, false
  )
  test.equal(failure, nil)
  return model as compute.Model
end

local function scores_of(
  model: compute.Model, token_id: integer, position: integer
): {number}
  local scores, line = model:forward(token_id, position)
  test.equal(line, nil)
  return scores as {number}
end

local function same_scores(left: {number}, right: {number})
  test.equal(#left, #right)
  for index, value in ipairs(left) do
    test.equal(value, right[index])
  end
end

local function closed(model: compute.Model, handle: weights.Handle)
  model:close()
  test.equal(handle.release(handle), nil)
end

local function expert_at(handle: weights.Handle, name: string): weights.Tensor
  return handle.tensors["blk.0." .. name] as weights.Tensor
end

-- the body under a stubbed `weights.resident`, restored even when it fails
local type ResidentAnswer = function(
  weights.Handle, weights.Tensor, integer
): boolean

local function with_resident(answer: ResidentAnswer, body: function())
  local real = weights.resident
  weights.resident = answer
  local ok, failure = pcall(function(): string | nil body(); return nil end)
  weights.resident = real
  if not ok then
    error(failure as string, 0)
  end
end

test.it(
  "a resident blob is computed where it lies",
  function()
    local read_scores: {number} = {}

    -- 52 bytes over the fixture's 96 expert bytes is a room of 0.54
    support.with_memory(
      function(): integer | nil return 52 end,
      function()
        local handle = loaded(fixture(base_meta(), resident_entries()))
        local model = built(handle)
        test.equal(model.zero_copy, false)
        read_scores = scores_of(model, 0, 0)
        test.equal(model.measure.blob_reads, 3)
        test.equal(model.measure.direct_reads, 0)
        closed(model, handle)
      end
    )

    support.with_memory(
      function(): integer | nil return 53 end,
      function()
        local path = fixture(base_meta(), resident_entries())
        local handle = loaded(path)
        local model = built(handle)
        test.equal(model.zero_copy, true)
        local where_scores = scores_of(model, 0, 0)
        test.equal(model.measure.blob_reads, 0)
        test.equal(model.measure.direct_reads, 3)
        test.equal(model.measure.blob_bytes, 0)
        test.equal(model.measure.blob_seconds, 0)
        same_scores(where_scores, read_scores)
        closed(model, handle)
      end
    )

    support.with_memory(
      function(): integer | nil return nil end,
      function()
        local handle = loaded(fixture(base_meta(), resident_entries()))
        local model = built(handle)
        test.equal(model.zero_copy, false)
        closed(model, handle)
      end
    )

    -- a handle that holds no expert matrix has no room to weigh
    local routing = both(block_entries("blk.0.", 2), block_entries("blk.1.", 2))
    local bare = both(routing, ROOT_ENTRIES)
    for _, prefix in ipairs({"blk.0.", "blk.1."}) do
      for _, name in ipairs(EXPERT_NAMES) do
        bare = without(bare, prefix .. name)
      end
    end
    support.with_memory(
      function(): integer | nil return 53 end,
      function()
        local handle = loaded(fixture(routing_meta(), bare))
        local model = built(handle)
        test.equal(model.zero_copy, false)
        closed(model, handle)
      end
    )
  end
)

test.it(
  "an evicted blob takes the read path",
  function()
    support.with_memory(
      function(): integer | nil return 49152 end,
      function()
        local handle = loaded(sparse_fixture(base_meta(), evicted_entries()))
        local model = built(handle)
        test.equal(model.zero_copy, true)
        for _, name in ipairs(EXPERT_NAMES) do
          local tensor = expert_at(handle, name)
          test.equal(weights.resident(handle, tensor, 0), false, name)
        end
        local up = expert_at(handle, "ffn_up_exps.weight")
        test.equal(weights.resident(handle, up, 2), false, "no matrix 2")
        test.equal(weights.resident(handle, up, 0), false, "asked twice")

        scores_of(model, 0, 0)
        test.equal(model.measure.direct_reads, 0)
        test.equal(model.measure.blob_reads, 3)
        closed(model, handle)
      end
    )
  end
)

test.it(
  "a blob in the page cache is checked again",
  function()
    support.with_memory(
      function(): integer | nil return 53 end,
      function()
        local handle = loaded(fixture(base_meta(), resident_entries()))
        local model = built(handle)
        test.equal(model.zero_copy, true)
        local first = scores_of(model, 0, 0)
        test.equal(model.measure.direct_reads, 3)
        test.equal(model.measure.blob_reads, 0)

        with_resident(
          function(
            _handle: weights.Handle, _tensor: weights.Tensor, _index: integer
          ): boolean
            return false
          end,
          function()
            same_scores(scores_of(model, 0, 1), first)
            test.equal(model.measure.blob_reads, 3)
            test.equal(model.measure.direct_reads, 3)
          end
        )

        same_scores(scores_of(model, 0, 2), first)
        test.equal(model.measure.blob_reads, 3)
        test.equal(model.measure.direct_reads, 3)
        closed(model, handle)
      end
    )
  end
)
