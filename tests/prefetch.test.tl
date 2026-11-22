local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local cli = require("tllm.cli")
local compute = require("tllm.compute")
local prefetch = require("tllm.prefetch")
local reads = require("tllm.reads")
local serve = require("tllm.serve")
local weights = require("tllm.weights")
local tltp = require("tltp")

local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text, padded =
  support.u32, support.u64, support.text, support.padded
local f32, zeros, ones = support.f32, support.zeros, support.ones
local mint, mfloat, mtext, mstrings, mint_array =
  support.mint,
  support.mfloat,
  support.mtext,
  support.mstrings,
  support.mint_array
local both, without, replaced = support.both, support.without, support.replaced

ffi.cdef([[int truncate(const char *path, long length);]])
local record Raw
  truncate: function(string, integer): integer
end
local C = ffi.C as Raw

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)

test.afterAll(function() fs.rmdir(ROOT) end)

-- one handle on the library keeps its backend modules registered while the
-- cases below open their own
local _kept = compute.load() as compute.Backend

local written = 0

local record Fixture
  path: string
  offset: {string: integer}
end

local function fixture(meta: {Meta}, entries: {Entry}): Fixture
  local head = "GGUF" .. u32(3) .. u64(#entries) .. u64(#meta)
  for _, entry in ipairs(meta) do
    head = head .. text(entry.name) .. u32(entry.code) .. entry.bytes
  end
  local offsets: {string: integer} = {}
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
    offsets[entry.name] = offset
    body[#body + 1] = padded(entry.bytes, 32)
    offset = offset + #body[#body]
  end
  local head_bytes = #padded(head, 32)
  for name, at in pairs(offsets) do
    offsets[name] = head_bytes + at
  end
  written = written + 1
  local location = ROOT .. "/p" .. written .. ".gguf"
  fs.write(location, padded(head, 32) .. table.concat(body))
  return {path = location, offset = offsets}
end

local function with_integer(meta: {Meta}, name: string, value: integer): {Meta}
  local result: {Meta} = {}
  local found = false
  for _, entry in ipairs(meta) do
    if entry.name == name then
      result[#result + 1] = mint(name, value)
      found = true
    else
      result[#result + 1] = entry
    end
  end
  if not found then
    result[#result + 1] = mint(name, value)
  end
  return result
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

local function routing_meta(): {Meta}
  local meta = with_integer(base_meta(), "qwen35moe.block_count", 2)
  meta = with_integer(meta, "qwen35moe.context_length", 65)
  return with_integer(meta, "qwen35moe.full_attention_interval", 1)
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

local function expert_values(experts: integer): {number}
  local values: {number} = {}
  for expert = 1, experts do
    for _ = 1, 4 do
      values[#values + 1] = expert
    end
  end
  return values
end

local function routing_block(prefix: string, count: integer): {Entry}
  local entries = both(
    attention_entries(prefix),
    {
      f32(prefix .. "post_attention_norm.weight", {2}, ones(2)),
      f32(
        prefix .. "ffn_gate_inp.weight",
        {2, count},
        count == 2 and {1, 0, 0, 1} or {1, 0, 0, 1, 0, 0, 0, 0}
      ),
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

local function root_entries(rows: integer, values: {number}): {Entry}
  return {
    f32("token_embd.weight", {2, rows}, values),
    f32("output_norm.weight", {2}, ones(2)),
    f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
  }
end

local function routing_entries(): {Entry}
  local entries = both(routing_block("blk.0.", 2), routing_block("blk.1.", 2))
  return both(entries, root_entries(2, {1, 0, 3, 4}))
end

-- a routing fixture whose block 1 expert matrices are a megabyte each, so the
-- anonymous block the queue's buffers live in stands apart in /proc/self/maps
local function large_matrix(rows: integer, columns: integer): string
  return string.rep("\0", rows * columns * 2 * 4)
end

local function large_routing_entries(): {Entry}
  local entries = routing_entries()
  local function big(name: string, dims: {integer}, bytes: string): Entry
    return {name = name, dims = dims, code = 0, bytes = bytes}
  end
  entries = replaced(
    entries,
    "blk.1.ffn_up_exps.weight",
    big("blk.1.ffn_up_exps.weight", {2, 131072, 2}, large_matrix(2, 131072))
  )
  entries = replaced(
    entries,
    "blk.1.ffn_gate_exps.weight",
    big("blk.1.ffn_gate_exps.weight", {2, 131072, 2}, large_matrix(2, 131072))
  )
  return replaced(
    entries,
    "blk.1.ffn_down_exps.weight",
    big("blk.1.ffn_down_exps.weight", {131072, 2, 2}, large_matrix(131072, 2))
  )
end

local function flipped_entries(): {Entry}
  return replaced(
    routing_entries(),
    "blk.1.attn_output.weight",
    f32("blk.1.attn_output.weight", {2, 2}, {-1000, 0, 0, 0})
  )
end

-- 29's fixture with one or two blocks, holding the tokenizer keys a bench run
-- reads
local FILLER = "The quick brown fox jumps over the lazy dog. "
local TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"

local function pieces(): {string}
  local seen: {string: boolean} = {}
  local list: {string} = {}
  for index = 1, #FILLER do
    local char = FILLER:sub(index, index)
    if not seen[char] then
      seen[char] = true
      if char == " " then
        list[#list + 1] = "\196\160"
      else
        list[#list + 1] = char
      end
    end
  end
  return list
end

local function bench_meta(blocks: integer): {Meta}
  local tokens = pieces()
  local types: {integer} = {}
  for index = 1, #tokens do
    types[index] = 1
  end
  local meta: {Meta} = {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", blocks),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 260),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", tokens),
    mint_array("tokenizer.ggml.token_type", types),
    mstrings("tokenizer.ggml.merges", {}),
    mtext("tokenizer.chat_template", TEMPLATE),
  }
  return meta
end

local function bench_block(prefix: string): {Entry}
  return both(
    attention_entries(prefix),
    {
      f32(prefix .. "post_attention_norm.weight", {2}, ones(2)),
      f32(prefix .. "ffn_gate_inp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_gate_inp_shexp.weight", {2}, zeros(2)),
      f32(prefix .. "ffn_up_shexp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_gate_shexp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_down_shexp.weight", {2, 2}, zeros(4)),
      f32(prefix .. "ffn_up_exps.weight", {2, 2, 2}, zeros(8)),
      f32(prefix .. "ffn_gate_exps.weight", {2, 2, 2}, zeros(8)),
      f32(prefix .. "ffn_down_exps.weight", {2, 2, 2}, zeros(8)),
    }
  )
end

local function bench_entries(blocks: integer): {Entry}
  local tokens = pieces()
  local count = #tokens
  local embedding: {number} = {}
  for _ = 1, count do
    embedding[#embedding + 1] = 1
    embedding[#embedding + 1] = 1
  end
  local output: {number} = {}
  for index = 0, count - 1 do
    output[#output + 1] = 1
    output[#output + 1] = index / count
  end
  local entries = bench_block("blk.0.")
  if blocks > 1 then
    entries = both(entries, bench_block("blk.1."))
  end
  return both(
    entries,
    {
      f32("token_embd.weight", {2, count}, embedding),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, count}, output),
    }
  )
end

-- 18/T-001's fixture, as serve's cases build it
local function serve_meta(): {Meta}
  return {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", 260),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", {"a", "b", "c"}),
    mint_array("tokenizer.ggml.token_type", {1, 1, 1}),
    mstrings("tokenizer.ggml.merges", {}),
    mint("tokenizer.ggml.eos_token_id", 1),
    mtext("tokenizer.chat_template", TEMPLATE),
  }
end

local function serve_entries(): {Entry}
  return both(
    bench_block("blk.0."),
    {
      f32("token_embd.weight", {2, 3}, {1, 0, 3, 4, 1, 0}),
      f32("output_norm.weight", {2}, ones(2)),
      f32("output.weight", {2, 3}, {1, 0, 0, 1, 1, 1}),
    }
  )
end

local function loaded(meta: {Meta}, entries: {Entry}): weights.Handle
  local handle, failure = weights.load(fixture(meta, entries).path)
  test.equal(failure, nil)
  return handle as weights.Handle
end

local function built(
  handle: weights.Handle,
  vram?: integer,
  profile?: string,
  prefetch_value?: boolean,
  sequences?: integer,
  threads?: integer
): compute.Model
  local backend = compute.load(nil, false) as compute.Backend
  local model, failure = compute.model(
    backend,
    handle,
    nil,
    "f32",
    sequences or 1,
    1,
    vram,
    profile,
    prefetch_value,
    threads
  )
  test.equal(failure, nil)
  return model as compute.Model
end

local function largest_expert(handle: weights.Handle): integer
  local largest = 0
  for name, tensor in pairs(handle.tensors) do
    if name:match("^blk%.%d+%.ffn_.*_exps%.weight$") and #tensor.dims >= 3 then
      local each = math.floor(tensor.bytes / (tensor.dims[3] as integer))
      if each > largest then
        largest = each
      end
    end
  end
  return largest
end

local function closed(model: compute.Model, handle: weights.Handle)
  model:close()
  test.equal(handle.release(handle), nil)
end

local function with_decide(
  answer: function(number | nil, integer, integer, boolean): boolean,
  body: function()
)
  local real = prefetch.decide
  prefetch.decide = answer
  local ok, failure = pcall(function(): string | nil body(); return nil end)
  prefetch.decide = real
  if not ok then
    error(failure as string, 0)
  end
end

-- read-ahead open on a machine whose page cache holds none of the experts, so
-- every chosen matrix takes `34`'s read
local function plenty(body: function())
  with_decide(
    function(
      _room: number | nil, _named: integer, _landed: integer, _open: boolean
    ): boolean
      return true
    end,
    function()
      support.with_memory(function(): integer | nil return 0 end, body)
    end
  )
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

local function needs_interface(name: string, body: function())
  if reads.offered() then
    test.it(name, body)
  else
    test.skip(name .. " (no submission interface)", function() end)
  end
end

-- the descriptors and the mapped rings a queue and a model hold
local function descriptors(): integer
  local count = 0
  local step = fs.readdir("/proc/self/fd")
  if step is nil then
    return count
  end
  while step() ~= nil do
    count = count + 1
  end
  return count
end

-- the mappings a queue and a model leave behind: the io_uring rings, the file
-- mapping the handle took and any unnamed block at least `block` bytes, which
-- is the size the queue's buffers stand at
local function rings(block: number): integer
  local contents = fs.read("/proc/self/maps") as string
  local count = 0
  for line in contents:gmatch("[^\n]+") do
    local first, last = line:match("^(%x+)%-(%x+) %S+ %x+ %x+:%x+ %d+%s*$")
    if line:find("io_uring", 1, true) or line:find(" rw-s ", 1, true) then
      count = count + 1
    elseif first is string and last is string then
      local size = (tonumber(last, 16) as number) - (
        tonumber(first, 16) as number
      )
      if size >= block then
        count = count + 1
      end
    end
  end
  return count
end

needs_interface(
  "read ahead hands over and moves no score",
  function()
    plenty(
      function()
        local handle = loaded(routing_meta(), routing_entries())
        local model = built(handle)
        test.equal(model.read_ahead.open, true)

        local records: {string} = {}
        model.routing_sink = function(routing: compute.Routing)
          records[#records + 1] = routing.layer
            .. ":"
            .. routing.token
            .. ":"
            .. table.concat(routing.experts, ",")
        end
        local graphs = model.measure.graphs
        local scores = scores_of(model, 0, 0)
        test.equal(model.read_ahead.open, true)
        test.equal(model.measure.prefetch_reads, 3)
        test.equal(model.measure.prefetch_named, 1)
        test.equal(model.measure.prefetch_landed, 1)
        test.equal(model.measure.blob_reads, 3)
        test.equal(model.measure.graphs - graphs, 8)
        closed(model, handle)

        local plain_handle = loaded(routing_meta(), routing_entries())
        local plain = built(plain_handle, nil, nil, false)
        local plain_records: {string} = {}
        plain.routing_sink = function(routing: compute.Routing)
          plain_records[#plain_records + 1] = routing.layer
            .. ":"
            .. routing.token
            .. ":"
            .. table.concat(routing.experts, ",")
        end
        local plain_graphs = plain.measure.graphs
        local plain_scores = scores_of(plain, 0, 0)
        test.equal(plain.measure.blob_reads, 6)
        test.equal(plain.measure.prefetch_reads, 0)
        test.equal(plain.measure.prefetch_named, 0)
        test.equal(plain.measure.prefetch_landed, 0)
        test.equal(plain.measure.graphs - plain_graphs, 8)
        same_scores(scores, plain_scores)
        test.equal(#records, 2)
        test.equal(#plain_records, #records)
        for index, line in ipairs(records) do
          test.equal(line, plain_records[index])
        end
        closed(plain, plain_handle)
      end
    )
  end
)

needs_interface(
  "a wrong prediction costs a read and not an answer",
  function()
    plenty(
      function()
        local handle = loaded(routing_meta(), flipped_entries())
        local model = built(handle)
        local scores = scores_of(model, 0, 0)
        test.equal(model.measure.prefetch_reads, 3)
        test.equal(model.measure.prefetch_named, 1)
        test.equal(model.measure.prefetch_landed, 1)
        local reads_made = model.measure.blob_reads
        closed(model, handle)

        local plain_handle = loaded(routing_meta(), flipped_entries())
        local plain = built(plain_handle, nil, nil, false)
        local plain_scores = scores_of(plain, 0, 0)
        same_scores(scores, plain_scores)
        test.equal(reads_made, plain.measure.blob_reads)
        closed(plain, plain_handle)
      end
    )
  end
)

needs_interface(
  "the gate opens and closes over sixteen passes",
  function()
    local figures: {integer | nil} = {192, 96, 116, 135, nil}
    local at = 0
    local handle = loaded(routing_meta(), routing_entries())
    support.with_memory(
      function(): integer | nil at = at + 1; return figures[at] end,
      function()
        local model = built(handle)
        test.equal(model.read_ahead.open, true)
        for position = 0, 15 do
          scores_of(model, 0, position)
        end
        test.equal(model.read_ahead.open, true)
        scores_of(model, 0, 16)
        test.equal(model.read_ahead.open, false)
        for position = 17, 32 do
          scores_of(model, 0, position)
        end
        test.equal(model.read_ahead.open, false)
        for position = 33, 48 do
          scores_of(model, 0, position)
        end
        test.equal(model.read_ahead.open, true)
        for position = 49, 64 do
          scores_of(model, 0, position)
        end
        test.equal(model.read_ahead.open, false)
        closed(model, handle)
      end
    )

    -- the gate's rule as a value
    test.equal(prefetch.decide(0.65, 0, 0, false), true)
    test.equal(prefetch.decide(0.6, 4, 1, true), false)
    test.equal(prefetch.decide(0.64, 4, 4, false), false)
    test.equal(prefetch.decide(0.64, 4, 4, true), true)
    test.equal(prefetch.decide(0.7, 4, 1, true), false)
    test.equal(prefetch.decide(0.6, 4, 4, false), false)
    test.equal(prefetch.decide(0.6, 4, 4, true), true)
    test.equal(prefetch.decide(nil, 0, 0, true), false)

    -- a person's decide is asked once at the build and once every sixteen passes
    local calls = 0
    with_decide(
      function(
        _room: number | nil, _named: integer, _landed: integer, _open: boolean
      ): boolean
        calls = calls + 1
        return true
      end,
      function()
        support.with_memory(
          function(): integer | nil return 0 end,
          function()
            local forced_handle = loaded(routing_meta(), routing_entries())
            local forced = built(forced_handle)
            test.equal(forced.read_ahead.open, true)
            test.equal(calls, 1)
            for position = 0, 15 do
              scores_of(forced, 0, position)
            end
            test.equal(calls, 1)
            scores_of(forced, 0, 16)
            test.equal(calls, 2)
            closed(forced, forced_handle)
          end
        )
      end
    )

    -- the real figure and the real decision agree
    local figure = prefetch.memory()
    if figure is nil then
      print("no memory figure")
    else
      local real_handle = loaded(routing_meta(), routing_entries())
      local real_model = built(real_handle)
      test.equal(
        real_model.read_ahead.open, prefetch.decide(figure / 192, 0, 0, false)
      )
      closed(real_model, real_handle)
    end
  end
)

local function on_device(name: string, body: function())
  if not support.found_device() then
    test.skip(name .. " (no CUDA device)", function() end)
    return
  end
  needs_interface(name, body)
end

on_device(
  "read ahead skips what the slot holds",
  function()
    local card = compute.load() as compute.Backend
    plenty(
      function()
        local first_handle = loaded(routing_meta(), routing_entries())
        local first = compute.model(
          card, first_handle, nil, "f32", 1, 1, nil, nil, false
        ) as compute.Model
        local placement = first.placement
        local vram = placement.resident_bytes + placement.reserve + math.floor(
          placement.slot_bytes / 2
        )
        closed(first, first_handle)

        local handle = loaded(routing_meta(), routing_entries())
        local ahead = compute.model(
          card, handle, nil, "f32", 1, 1, vram, nil, true
        ) as compute.Model
        test.equal(ahead.read_ahead.open, true)
        local scores = scores_of(ahead, 0, 0)
        test.equal(ahead.measure.prefetch_reads, 0)
        test.equal(ahead.measure.prefetch_named, 0)
        test.equal(ahead.measure.prefetch_landed, 0)
        closed(ahead, handle)

        local plain_handle = loaded(routing_meta(), routing_entries())
        local plain = compute.model(
          card, plain_handle, nil, "f32", 1, 1, vram, nil, false
        ) as compute.Model
        local plain_scores = scores_of(plain, 0, 0)
        same_scores(scores, plain_scores)
        closed(plain, plain_handle)
      end
    )
    compute.unload(card)
  end
)

local function config_file(
  name: string, path: string, prefetch_value: string | nil, vram?: integer
): string
  written = written + 1
  local location = ROOT .. "/" .. name .. written .. ".json"
  local body = '{"models":{"m":{"path":"' .. path .. '"}}'
  if prefetch_value is string then
    body = body .. ',"prefetch":' .. prefetch_value
  end
  if not (vram is nil) then
    body = body .. ',"vram":' .. vram
  end
  fs.write(location, body .. "}")
  return location
end

-- the vram that holds every fixed tensor and leaves no expert slot, so the
-- experts a turn routes are misses the read ahead can predict, on a machine
-- whose card would otherwise keep them all resident
local function no_slot_vram(path: string): integer
  local card = compute.load() as compute.Backend
  local handle, failure = weights.load(path)
  test.equal(failure, nil)
  local held = handle as weights.Handle
  local model = compute.model(card, held, nil, "f16") as compute.Model
  local placement = model.placement
  local vram = placement.resident_bytes + placement.reserve
  model:close()
  test.equal(held.release(held), nil)
  compute.unload(card)
  return vram
end

local record Ran
  code: integer
  out: {string}
  err: {string}
end

local function ran(
  args: {string}, start?: function(serve.Endpoint, function(string)): integer
): Ran
  local result: Ran = {code = 0, out = {}, err = {}}
  local environment = {
    XDG_CONFIG_HOME = ROOT .. "/config",
    XDG_DATA_HOME = ROOT .. "/data",
    HOME = ROOT,
  }
  fs.mkdirAll(environment.XDG_CONFIG_HOME)
  fs.mkdirAll(environment.XDG_DATA_HOME)
  result.code = cli.run(
    {
      args = args,
      write_out = function(line: string)
        result.out[#result.out + 1] = (line:gsub("\n$", ""))
      end,
      write_err = function(line: string)
        result.err[#result.err + 1] = (line:gsub("\n$", ""))
      end,
      environment = environment,
      start = start,
    }
  )
  return result
end

local function experts_at(out: {string}, length: integer): string | nil
  for _, line in ipairs(out) do
    if line:match("^experts " .. length .. ": ") then
      return line
    end
  end
  return nil
end

needs_interface(
  "the key forced, refused and reaching the model",
  function()
    support.with_memory(
      function(): integer | nil return 0 end,
      function()
        local handle = loaded(routing_meta(), routing_entries())
        local forced = built(handle, nil, nil, true)
        test.equal(forced.read_ahead.open, true)
        closed(forced, handle)

        local closed_handle = loaded(routing_meta(), routing_entries())
        local stopped = built(closed_handle, nil, nil, false)
        test.equal(stopped.read_ahead.open, false)
        test.equal(stopped.read_ahead.depth, 0)
        test.equal(stopped.read_ahead.bytes, 0)
        scores_of(stopped, 0, 0)
        test.equal(stopped.measure.prefetch_reads, 0)
        closed(stopped, closed_handle)
      end
    )

    -- the file's key reaches the start function and the model serve builds
    local file_path = fixture(serve_meta(), serve_entries()).path
    local named = config_file("forced", file_path, "true")
    local seen: boolean | nil = nil
    local collected = function(
      endpoint: serve.Endpoint, _err: function(string)
    ): integer
      seen = endpoint.prefetch
      return 0
    end
    test.equal(ran({"serve", "m", "--config", named}, collected).code, 0)
    test.equal(seen, true)

    local plain = config_file("plain", file_path, nil)
    seen = nil
    test.equal(ran({"serve", "m", "--config", plain}, collected).code, 0)
    test.equal(seen, nil)

    local text_value = config_file("text", file_path, '"true"')
    local text_run = ran({"serve", "m", "--config", text_value}, collected)
    test.equal(text_run.code, 1)
    test.equal(#text_run.err, 1)
    test.includes(text_run.err[1], "prefetch")
    local number_value = config_file("number", file_path, "1")
    local number_run = ran({"serve", "m", "--config", number_value}, collected)
    test.equal(number_run.code, 1)
    test.equal(#number_run.err, 1)
    test.includes(number_run.err[1], "prefetch")

    -- the bench Settings carries it, and the line proves the model held it open
    local bench_file = fixture(bench_meta(2), bench_entries(2)).path
    local bench_config = config_file("bench", bench_file, "true")
    local got = ran(
      {
        "bench",
        "m",
        "--config",
        bench_config,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(got.code, 0, table.concat(got.err, "\n"))
    local line = experts_at(got.out, 8) as string
    test.truthy(line, table.concat(got.out, "\n"))
    test.truthy(
      line:match(
        "read ahead open; 0%.0%% over PCIe, pool 0%.0 GB/s, "
          .. "link 0%.0 GB/s; %d+ direct$"
      ),
      line
    )

    -- serve gives the endpoint's value to the model it builds
    local server, failure = serve.build(
      {
        host = "127.0.0.1",
        port = 8090,
        name = "m",
        path = file_path,
        sessions = 1,
        kv_cache = "f32",
        workers = 1,
        ubatch = 512,
        prefetch = true,
      }
    )
    test.equal(failure, nil)
    local model = serve.model_of(server as tltp.Server) as serve.Model
    test.equal(model.generation.read_ahead.open, true)
    serve.stop(server as tltp.Server)

    local off, off_failure = serve.build(
      {
        host = "127.0.0.1",
        port = 8091,
        name = "m",
        path = file_path,
        sessions = 1,
        kv_cache = "f32",
        workers = 1,
        ubatch = 512,
        prefetch = false,
      }
    )
    test.equal(off_failure, nil)
    local off_model = serve.model_of(off as tltp.Server) as serve.Model
    test.equal(off_model.generation.read_ahead.open, false)
    serve.stop(off as tltp.Server)
  end
)

needs_interface(
  "bench reports the ahead reads",
  function()
    local one = fixture(bench_meta(1), bench_entries(1)).path
    local off = config_file("off", one, "false")
    local got = ran(
      {
        "bench",
        "m",
        "--config",
        off,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(got.code, 0, table.concat(got.err, "\n"))
    local line = experts_at(got.out, 8) as string
    test.truthy(line, table.concat(got.out, "\n"))
    test.truthy(
      line:match(
        "; 0 ahead, 0%.0%% landed; read ahead closed; 0%.0%% over PCIe, "
          .. "pool 0%.0 GB/s, link 0%.0 GB/s; %d+ direct$"
      ),
      line
    )

    local two = fixture(bench_meta(2), bench_entries(2)).path
    local vram: integer | nil = nil
    if support.found_device() then
      vram = no_slot_vram(two)
    end
    local on = config_file("on", two, "true", vram)
    local ahead = ran(
      {
        "bench",
        "m",
        "--config",
        on,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(ahead.code, 0, table.concat(ahead.err, "\n"))
    line = experts_at(ahead.out, 8) as string
    test.truthy(line, table.concat(ahead.out, "\n"))
    test.truthy(
      line:match(
        "; 3 ahead, 100%.0%% landed; read ahead open; "
          .. "%d+%.%d%% over PCIe, pool %d+%.%d GB/s, link %d+%.%d GB/s; "
          .. "%d+ direct$"
      ),
      line
    )
  end
)

needs_interface(
  "a short read ahead closes and the demand read answers",
  function()
    -- the down matrix is the file's last tensor, so cutting the file a few bytes
    -- into its first matrix is a read that lands short
    local with_down = both(
      without(routing_entries(), "blk.1.ffn_down_exps.weight"),
      {
        f32("blk.1.ffn_down_exps.weight", {2, 2, 2}, expert_values(2)),
      }
    )
    local backing = fixture(routing_meta(), with_down)
    local handle = weights.load(backing.path) as weights.Handle
    test.equal(
      C.truncate(
        backing.path, (backing.offset["blk.1.ffn_down_exps.weight"] or 0) + 4
      ),
      0
    )
    support.with_memory(
      function(): integer | nil return 0 end,
      function()
        local model = built(handle, nil, nil, true, 2)
        test.equal(model.read_ahead.open, true)
        local scores, line = model:forward(0, 0)
        test.equal(scores, nil)
        test.includes(line as string, "blk.1.ffn_down_exps.weight")
        test.equal(model.read_ahead.open, false)
        test.equal(model.measure.prefetch_landed, 0)
        test.equal(model:reserve("again", 1), nil)
        local again, again_line = model:pass(
          {{sequence = "again", token = 0, position = 0}}
        )
        test.equal(again, nil)
        test.includes(again_line as string, "blk.1.ffn_down_exps.weight")
        closed(model, handle)
      end
    )
  end
)

needs_interface(
  "the queue is taken, sized and given back",
  function()
    plenty(
      function()
        local backing = fixture(routing_meta(), large_routing_entries())
        collectgarbage("collect")
        local before_fd = descriptors()
        local before_rings = rings(1048576)
        local handle = weights.load(backing.path) as weights.Handle
        local model = built(handle, nil, nil, nil, nil, 1)
        test.truthy(model.read_ahead.depth >= 4)
        test.truthy(model.read_ahead.bytes >= largest_expert(handle))
        model:close()
        test.equal(handle.release(handle), nil)
        collectgarbage("collect")
        test.equal(descriptors(), before_fd, "no descriptor left behind")
        test.equal(rings(1048576), before_rings, "no mapped block left behind")
        model:close()
        test.equal(
          descriptors(), before_fd, "a second close gives nothing back"
        )
        test.equal(
          rings(1048576), before_rings, "a second close gives nothing back"
        )
      end
    )

    -- three experts per row need nine entries, and four is the floor
    local wide = both(routing_block("blk.0.", 4), routing_block("blk.1.", 4))
    local wide_handle = loaded(
      with_integer(routing_meta(), "qwen35moe.expert_used_count", 3),
      both(wide, root_entries(2, {1, 0, 3, 4}))
    )
    plenty(
      function()
        local model = built(wide_handle)
        test.truthy(model.read_ahead.depth >= 9)
        closed(model, wide_handle)
      end
    )

    -- no expert matrix, no queue
    local bare_entries = routing_entries()
    for _, name in ipairs(
      {
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.1.ffn_up_exps.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.1.ffn_down_exps.weight",
      }
    ) do
      bare_entries = without(bare_entries, name)
    end
    local bare_handle = loaded(routing_meta(), bare_entries)
    local bare = built(bare_handle, nil, nil, true)
    test.equal(bare.read_ahead.open, false)
    test.equal(bare.read_ahead.depth, 0)
    test.equal(bare.read_ahead.bytes, 0)
    bare:forward(0, 0)
    test.equal(bare.measure.prefetch_reads, 0)
    closed(bare, bare_handle)

    -- a model that cannot be built gives the queue back
    local failing = loaded(routing_meta(), large_routing_entries())
    collectgarbage("collect")
    local before_fd = descriptors()
    local before_rings = rings(1048576)
    plenty(
      function()
        local made, failure = compute.model(
          compute.load(nil, false) as compute.Backend,
          failing,
          nil,
          "f32",
          1,
          1,
          nil,
          ROOT .. "/no-such-profile.jsonl",
          true
        )
        test.equal(made, nil)
        test.truthy(failure)
        collectgarbage("collect")
        test.equal(descriptors(), before_fd, "no descriptor left behind")
        test.equal(rings(1048576), before_rings, "no mapped block left behind")
      end
    )
    test.equal(failing.release(failing), nil)
  end
)
