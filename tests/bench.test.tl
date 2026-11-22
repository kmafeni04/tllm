local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local cli = require("tllm.cli")
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local tltp = require("tltp")
local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, mint = support.padded, support.f32, support.mint
local mfloat, mtext, mstrings, mint_array =
  support.mfloat, support.mtext, support.mstrings, support.mint_array
local both, replaced = support.both, support.replaced

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function path_for(name: string): string
  created = created + 1
  return ROOT .. "/" .. name .. tostring(created)
end

local function fixture(bytes: string): string
  local file_path = path_for("s") .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
end

local function meta_bytes(meta: {Meta}, entries: {Entry}): string
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
  return padded(head, 32) .. table.concat(body)
end

local function meta_file(meta: {Meta}, entries: {Entry}): string
  return fixture(meta_bytes(meta, entries))
end

local FILLER = "The quick brown fox jumps over the lazy dog. "

local TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"

-- the vocabulary holds one token per byte symbol of the filler, so the filler
-- encodes to one id per character
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

local function meta(
  context: integer, eos: integer | nil, templated: boolean
): {Meta}
  local tokens = pieces()
  local types: {integer} = {}
  for index = 1, #tokens do
    types[index] = 1
  end
  local entries: {Meta} = {
    mtext("general.architecture", "qwen35moe"),
    mint("qwen35moe.block_count", 1),
    mint("qwen35moe.embedding_length", 2),
    mint("qwen35moe.attention.head_count", 1),
    mint("qwen35moe.rope.dimension_count", 2),
    mint("qwen35moe.rope.freq_base", 10000),
    mfloat("qwen35moe.attention.layer_norm_rms_epsilon", 0),
    mint("qwen35moe.context_length", context),
    mint("qwen35moe.expert_used_count", 1),
    mint("qwen35moe.full_attention_interval", 1),
    mtext("tokenizer.ggml.model", "gpt2"),
    mtext("tokenizer.ggml.pre", "gpt2"),
    mstrings("tokenizer.ggml.tokens", tokens),
    mint_array("tokenizer.ggml.token_type", types),
    mstrings("tokenizer.ggml.merges", {}),
  }
  if templated then
    entries[#entries + 1] = mtext("tokenizer.chat_template", TEMPLATE)
  end
  if not (eos is nil) then
    entries[#entries + 1] = mint("tokenizer.ggml.eos_token_id", eos)
  end
  return entries
end

local function attention_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "attn_norm.weight", {2}, {1, 1}),
    f32(prefix .. "attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
    f32(prefix .. "attn_k.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_v.weight", {2, 2}, {1, 0, 0, 1}),
    f32(prefix .. "attn_q_norm.weight", {2}, {1, 1}),
    f32(prefix .. "attn_k_norm.weight", {2}, {1, 1}),
    f32(prefix .. "attn_output.weight", {2, 2}, {0, 0, 0, 0}),
  }
end

local function feed_forward_entries(prefix: string): {Entry}
  return {
    f32(prefix .. "post_attention_norm.weight", {2}, {1, 1}),
    f32(prefix .. "ffn_gate_inp.weight", {2, 2}, {0, 0, 0, 0}),
    f32(prefix .. "ffn_gate_inp_shexp.weight", {2}, {0, 0}),
    f32(prefix .. "ffn_up_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32(prefix .. "ffn_gate_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32(prefix .. "ffn_down_shexp.weight", {2, 2}, {0, 0, 0, 0}),
    f32(prefix .. "ffn_up_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
    f32(prefix .. "ffn_gate_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
    f32(prefix .. "ffn_down_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
  }
end

local function entries(): {Entry}
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
  return both(
    both(attention_entries("blk.0."), feed_forward_entries("blk.0.")),
    {
      f32("token_embd.weight", {2, count}, embedding),
      f32("output_norm.weight", {2}, {1, 1}),
      f32("output.weight", {2, count}, output),
    }
  )
end

local record Result
  code: integer
  out: {string}
  err: {string}
end

local function empty_environment(): {string: string}
  local base = path_for("e")
  fs.mkdirAll(base .. "/config")
  fs.mkdirAll(base .. "/data")
  return {
    XDG_CONFIG_HOME = base .. "/config",
    XDG_DATA_HOME = base .. "/data",
    HOME = base,
  }
end

local function run(args: {string}, environment?: {string: string}): Result
  local result: Result = {code = 0, out = {}, err = {}}
  local given_environment = environment
  if given_environment is nil then
    given_environment = empty_environment()
  end
  result.code = cli.run(
    {
      args = args,
      write_out = function(given: string)
        result.out[#result.out + 1] = (given:gsub("\n$", ""))
      end,
      write_err = function(given: string)
        result.err[#result.err + 1] = (given:gsub("\n$", ""))
      end,
      environment = given_environment,
    }
  )
  return result
end

-- the id the copy's own scores put first, read through the same cache the
-- bench run uses
local function first_id(file_path: string, content: string): integer
  local tokens = tokenizer.load(file_path) as tokenizer.Tokenizer
  local ids = tokens:encode(content) as {integer}
  local handle = weights.load(file_path) as weights.Handle
  local backend = compute.load() as compute.Backend
  local model = compute.model(backend, handle, nil, "f16", 1) as compute.Model
  local scores: {number} = {}
  for position, id in ipairs(ids) do
    local planes, line = model:forward(id, position - 1)
    test.equal(line, nil)
    test.truthy(planes)
    scores = planes as {number}
  end
  local best = 1
  for index = 2, #scores do
    if scores[index] > scores[best] then
      best = index
    end
  end
  return best - 1
end

local PROMPT_8 = "^prompt 8: (%d+) tokens at (%d+%.%d) tok/s, "
  .. "first token (%d+%.%d) ms$"

local TOKEN_8 = "^token 8: (%d+%.%d%d%d) ms, compute (%d+%.%d%d%d) ms, "
  .. "sampling (%d+%.%d%d%d) ms, ipc (%d+%.%d%d%d) ms$"

local EXPERTS_8 = "^experts 8: (%d+) routed, (%d+%.%d) MB; per turn (%d+) "
  .. "reads, (%d+%.%d) MB read, (%d+%.%d%d%d) ms; (%d+%.%d)%% hit; "
  .. "(%d+) ahead, (%d+%.%d)%% landed; read ahead (%a+); "
  .. "(%d+%.%d)%% over PCIe, pool (%d+%.%d) GB/s, link (%d+%.%d) GB/s; "
  .. "(%d+) direct$"

test.it(
  "a rung reports and adds up",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    local args = {
      "bench",
      file_path,
      "--prefill-tokens",
      "8",
      "--decode-tokens",
      "8",
    }
    local first = run(args)
    test.equal(first.code, 0)
    test.equal(#first.err, 0)
    test.equal(#first.out, 4, table.concat(first.out, "\n"))

    local count_text, _, first_ms = first.out[1]:match(PROMPT_8)
    test.truthy(count_text, first.out[1])
    local prompt_count = tonumber(count_text as string) as integer
    test.truthy(prompt_count >= 8, first.out[1])
    test.truthy(tonumber(first_ms as string), first.out[1])
    test.truthy(first.out[2]:match("^decode 8: 7 tokens at (%d+%.%d) tok/s$"))
    local total_text, compute_text, sampling_text, ipc_text = first.out[3]:match(
      TOKEN_8
    )
    test.truthy(total_text, first.out[3])
    local total = tonumber(total_text as string) as number
    local compute_time = tonumber(compute_text as string) as number
    local sampling_time = tonumber(sampling_text as string) as number
    local ipc = tonumber(ipc_text as string) as number
    test.truthy(compute_time > 0, first.out[3])
    test.truthy(
      math.abs(compute_time + sampling_time + ipc - total) <= total * 0.2,
      first.out[3]
    )
    local routed, _, _, _ = first.out[4]:match(EXPERTS_8)
    test.equal(tonumber(routed as string), 1, first.out[4])

    local second = run(args)
    test.equal(second.code, 0)
    test.truthy(second.out[1]:match(PROMPT_8))
    test.equal(
      tonumber((second.out[1]:match(PROMPT_8)) as string), prompt_count
    )
    -- the milliseconds are a clock reading, so the two runs share every other
    -- figure of the experts line
    local again_routed, again_bytes, again_reads, again_read = second.out[4]:match(
      EXPERTS_8
    )
    local _, first_bytes, first_reads, first_read = first.out[4]:match(
      EXPERTS_8
    )
    test.equal(again_routed, routed, second.out[4])
    test.equal(again_bytes, first_bytes, second.out[4])
    test.equal(again_reads, first_reads, second.out[4])
    test.equal(again_read, first_read, second.out[4])

    local stopped = meta_file(
      meta(260, first_id(file_path, FILLER), true), entries()
    )
    local ended = run(
      {
        "bench",
        stopped,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(ended.code, 0)
    test.equal(#ended.err, 0)
    test.equal(#ended.out, 2, table.concat(ended.out, "\n"))
    test.truthy(ended.out[1]:match(PROMPT_8), ended.out[1])
    test.truthy(
      ended.out[2]:match("^decode 8: 0 tokens at 0%.0 tok/s$"), ended.out[2]
    )
  end
)

test.it(
  "a length the model cannot hold is skipped",
  function()
    local small = meta_file(meta(260, nil, true), entries())
    local refused = run({"bench", small})
    test.equal(refused.code, 1)
    test.equal(#refused.out, 4)
    local lengths = {512, 2048, 8192, 32768}
    for index, length in ipairs(lengths) do
      local line = refused.out[index]
      test.truthy(line:match("^skipped " .. length .. ": "), line)
      test.includes(line, "positions")
    end

    local roomy = meta_file(meta(4096, nil, true), entries())
    local held = run({"bench", roomy})
    test.equal(held.code, 0)
    local seen = table.concat(held.out, "\n")
    test.includes(seen, "prompt 512:")
    test.includes(seen, "prompt 2048:")
    test.includes(seen, "skipped 8192:")
    test.includes(seen, "skipped 32768:")
    test.equal(seen:find("skipped 512:", 1, true), nil)
    test.equal(seen:find("skipped 2048:", 1, true), nil)

    local untemplated = meta_file(meta(260, nil, false), entries())
    local dump = path_for("r") .. ".jsonl"
    local broken = run(
      {
        "bench",
        untemplated,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
        "--dump-routing",
        dump,
      }
    )
    test.equal(broken.code, 1)
    test.equal(#broken.err, 1)
    test.includes(broken.err[1], "8")
    test.includes(broken.err[1], "InternalServerError")
    for _, line in ipairs(broken.out) do
      test.equal(line:find("^prompt"), nil)
    end
    test.falsy(fs.exists(dump))
  end
)

test.it(
  "bench finds its model the way serve does",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    local config_path = path_for("c") .. ".json"
    fs.write(config_path, '{"models":{"m":{"path":"' .. file_path .. '"}}}')
    local named = run(
      {
        "bench",
        "m",
        "--config",
        config_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(named.code, 0, table.concat(named.err, "\n"))
    test.truthy(named.out[1]:match("^prompt 8: "), named.out[1])

    local environment = empty_environment()
    local plain = run(
      {
        "bench",
        file_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      },
      environment
    )
    test.equal(plain.code, 0)
    test.truthy(plain.out[1]:match("^prompt 8: "), plain.out[1])
    for _, name in ipairs({"XDG_CONFIG_HOME", "XDG_DATA_HOME"}) do
      local dir = environment[name] as string
      local next_entry = fs.readdir(dir) as function(): fs.DirEntry | nil
      test.truthy(next_entry, dir)
      test.equal(next_entry(), nil, dir)
    end

    local unknown = run(
      {"bench", "nosuch", "--config", config_path, "--prefill-tokens", "8"}
    )
    test.equal(unknown.code, 1)
    test.equal(#unknown.out, 0)
    test.equal(#unknown.err, 1)
    test.includes(unknown.err[1], "nosuch")

    local not_gguf = path_for("n") .. ".gguf"
    fs.write(not_gguf, "not a gguf file")
    local refused = run({"bench", not_gguf, "--prefill-tokens", "8"})
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], not_gguf)
  end
)

local record DumpLine
  layer: integer
  token: integer
  experts: {integer}
end

local function dumped(file_path: string): {DumpLine}
  local content = fs.read(file_path) as string
  local lines: {DumpLine} = {}
  for line in content:gmatch("[^\n]+") do
    lines[#lines + 1] = tltp.json_decode(line) as DumpLine
  end
  return lines
end

local function count_of(lines: {string}, prefix: string): integer
  for _, line in ipairs(lines) do
    local found = line:match("^" .. prefix .. ": (%d+) tokens")
    if found then
      return tonumber(found as string) as integer
    end
  end
  return -1
end

test.it(
  "dump-routing writes a line per layer per pass",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    local dump = path_for("r") .. ".jsonl"
    local first = run(
      {
        "bench",
        file_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
        "--dump-routing",
        dump,
      }
    )
    test.equal(first.code, 0)
    local prompt_count = count_of(first.out, "prompt 8")
    local decode_count = count_of(first.out, "decode 8")
    test.equal(decode_count, 7)
    local lines = dumped(dump)
    test.equal(#lines, prompt_count + decode_count)
    for index, line in ipairs(lines) do
      test.equal(line.layer, 0)
      test.equal(line.token, index - 1)
      test.truthy(#line.experts >= 1)
      for position, expert in ipairs(line.experts) do
        test.truthy(expert < 2)
        if position > 1 then
          test.truthy(expert > line.experts[position - 1])
        end
      end
    end

    local roomy = meta_file(meta(4096, nil, true), entries())
    local kept_path = path_for("r") .. ".jsonl"
    local wide = run({"bench", roomy, "--dump-routing", kept_path})
    test.equal(wide.code, 0)
    local wide_prompt = count_of(wide.out, "prompt 2048")
    local wide_decode = count_of(wide.out, "decode 2048")
    test.truthy(wide_prompt > 0)
    local kept = dumped(kept_path)
    test.equal(#kept, wide_prompt + wide_decode)
    for index, line in ipairs(kept) do
      test.equal(line.token, index - 1)
    end

    local missing = ROOT .. "/nodir/r.jsonl"
    local unwritable = run(
      {
        "bench",
        file_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
        "--dump-routing",
        missing,
      }
    )
    test.equal(unwritable.code, 1)
    test.equal(#unwritable.err, 1)
    test.includes(unwritable.err[1], missing)
    test.equal(#unwritable.out, 0)

    local absent = path_for("r") .. ".jsonl"
    local plain = run(
      {
        "bench",
        file_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(plain.code, 0)
    test.falsy(fs.exists(absent))
  end
)

test.it(
  "bench counts tokens and routes one record per token",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    local config_path = path_for("c") .. ".json"
    fs.write(
      config_path, '{"models":{"m":{"path":"' .. file_path .. '"}},"ubatch":3}'
    )
    local dump = path_for("r") .. ".jsonl"
    local ran = run(
      {
        "bench",
        "m",
        "--config",
        config_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
        "--dump-routing",
        dump,
      }
    )
    test.equal(ran.code, 0)

    local prompt_count = count_of(ran.out, "prompt 8")
    local decode_count = count_of(ran.out, "decode 8")
    test.truthy(prompt_count >= 8, "the prompt counts the tokens it passed")

    local lines = dumped(dump)
    test.equal(#lines, prompt_count + decode_count)
    local seen: {integer: boolean} = {}
    for _, line in ipairs(lines) do
      test.equal(line.layer, 0)
      test.truthy(line.token >= 0, "token " .. line.token)
      test.less(line.token, #lines)
      test.equal(seen[line.token], nil, "token " .. line.token .. " twice")
      seen[line.token] = true
    end
    for index = 0, #lines - 1 do
      test.equal(seen[index], true, "no record for token " .. index)
    end
  end
)

-- the expert tier of a fixture wide enough that its read figure is more than
-- a rounding note
local WIDE = 12500
local WIDE_MATRIX = 2 * WIDE * 4

local function wide_entries(): {Entry}
  local empty: {number} = {}
  for _ = 1, 4 * WIDE do
    empty[#empty + 1] = 0
  end
  local stacked = replaced(
    entries(),
    "blk.0.ffn_up_exps.weight",
    f32("blk.0.ffn_up_exps.weight", {2, WIDE, 2}, empty)
  )
  stacked = replaced(
    stacked,
    "blk.0.ffn_gate_exps.weight",
    f32("blk.0.ffn_gate_exps.weight", {2, WIDE, 2}, empty)
  )
  return replaced(
    stacked,
    "blk.0.ffn_down_exps.weight",
    f32("blk.0.ffn_down_exps.weight", {WIDE, 2, 2}, empty)
  )
end

test.it(
  "bench reports the expert reads",
  function()
    -- no room for the experts in the page cache, so a turn's chosen matrices
    -- are read rather than computed where they lie
    support.with_memory(
      function(): integer | nil return 0 end,
      function()
        local file_path = meta_file(meta(260, nil, true), wide_entries())
        local got = run(
          {
            "bench",
            file_path,
            "--prefill-tokens",
            "8",
            "--decode-tokens",
            "8",
          }
        )
        test.equal(got.code, 0)
        test.equal(#got.out, 4, table.concat(got.out, "\n"))

        local _, _, reads, read_bytes, ms = got.out[4]:match(EXPERTS_8)
        test.truthy(reads, got.out[4])

        -- the same load the bench server makes, run over this fixture here, so
        -- the figures are compared with a measure rather than with a guess
        -- about this machine: the turn routes one expert, as this pass does
        local handle = weights.load(file_path) as weights.Handle
        local backend = compute.load() as compute.Backend
        local model = compute.model(backend, handle, nil, "f16", 1)
          as compute.Model
        local planes, line = model:forward(0, 0)
        test.equal(line, nil)
        test.truthy(planes ~= nil, "the pass ran")
        local whole = model.measure.blob_reads
        test.equal(tonumber(reads as string), whole, got.out[4])
        test.equal(
          read_bytes,
          string.format("%.1f", model.measure.blob_bytes / 1000000),
          got.out[4]
        )
        if whole > 0 then
          test.equal(
            read_bytes,
            string.format("%.1f", whole * WIDE_MATRIX / 1000000),
            got.out[4]
          )
          test.truthy((tonumber(ms as string) as number) > 0, got.out[4])
        end

        model:close()
        test.equal(handle.release(handle), nil)
      end
    )
  end
)

local function profile_file(lines: string): string
  local file_path = path_for("p") .. ".jsonl"
  fs.write(file_path, lines)
  return file_path
end

local function experts_at(out: {string}, length: integer): string | nil
  for _, line in ipairs(out) do
    if line:match("^experts " .. length .. ": ") then
      return line
    end
  end
  return nil
end

-- the vram that holds every fixed tensor, the reservation and half of the
-- fixture's two expert slots
local function half_slot_vram(file_path: string): integer
  local handle = weights.load(file_path) as weights.Handle
  local backend = compute.load() as compute.Backend
  local model = compute.model(backend, handle, nil, "f16", 1) as compute.Model
  local placement = model.placement
  local vram = placement.resident_bytes + placement.reserve + math.floor(
    placement.slot_bytes / 2
  )
  model:close()
  compute.unload(backend)
  test.equal(handle.release(handle), nil)
  return vram
end

test.it(
  "bench reports the hit rate and the profile reaches the model",
  function()
    if not support.found_device() then
      test.skip(
        "bench reports the hit rate and the profile reaches the model"
          .. " (no CUDA device)",
        function() end
      )
      return
    end

    local file_path = meta_file(meta(260, nil, true), entries())
    local vram = half_slot_vram(file_path)
    local named = profile_file('{"layer":0,"token":0,"experts":[1]}\n')
    local seeded = path_for("c") .. ".json"
    fs.write(
      seeded,
      '{"models":{"m":{"path":"'
        .. file_path
        .. '"}},"vram":'
        .. vram
        .. ',"profile":"'
        .. named
        .. '"}'
    )
    local got = run(
      {
        "bench",
        "m",
        "--config",
        seeded,
        "--prefill-tokens",
        "4",
        "--decode-tokens",
        "4",
      }
    )
    test.equal(got.code, 0, table.concat(got.err, "\n"))
    local line = experts_at(got.out, 4) as string
    test.truthy(line, table.concat(got.out, "\n"))
    test.truthy(
      line:match(
        "; 0%.0%% hit; 0 ahead, 0%.0%% landed; read ahead (%a+); "
          .. "%d+%.%d%% over PCIe, pool %d+%.%d GB/s, link %d+%.%d GB/s; "
          .. "%d+ direct$"
      ),
      line
    )

    local plain = path_for("c") .. ".json"
    fs.write(
      plain,
      '{"models":{"m":{"path":"' .. file_path .. '"}},"vram":' .. vram .. "}"
    )
    local default = run(
      {
        "bench",
        "m",
        "--config",
        plain,
        "--prefill-tokens",
        "4",
        "--decode-tokens",
        "4",
      }
    )
    test.equal(default.code, 0, table.concat(default.err, "\n"))
    line = experts_at(default.out, 4) as string
    test.truthy(line, table.concat(default.out, "\n"))
    test.truthy(
      line:match(
        "; 100%.0%% hit; 0 ahead, 0%.0%% landed; read ahead (%a+); "
          .. "0%.0%% over PCIe, pool 0%.0 GB/s, link 0%.0 GB/s; %d+ direct$"
      ),
      line
    )

    local long_path = meta_file(meta(4096, nil, true), entries())
    local long_vram = half_slot_vram(long_path)
    local long = path_for("c") .. ".json"
    fs.write(
      long,
      '{"models":{"m":{"path":"'
        .. long_path
        .. '"}},"vram":'
        .. long_vram
        .. "}"
    )
    local ladder = run({"bench", "m", "--config", long})
    test.equal(ladder.code, 0, table.concat(ladder.err, "\n"))
    line = experts_at(ladder.out, 2048) as string
    test.truthy(line, table.concat(ladder.out, "\n"))
    test.truthy(
      line:match(
        "; 100%.0%% hit; 0 ahead, 0%.0%% landed; read ahead (%a+); "
          .. "0%.0%% over PCIe, pool 0%.0 GB/s, link 0%.0 GB/s; %d+ direct$"
      ),
      line
    )
  end
)

-- the vram that holds every fixed tensor and the reservation and leaves no
-- expert slot, so every expert a turn routes is a miss
local function no_slot_vram(file_path: string): integer
  local handle = weights.load(file_path) as weights.Handle
  local backend = compute.load() as compute.Backend
  local model = compute.model(backend, handle, nil, "f16", 1) as compute.Model
  local placement = model.placement
  local vram = placement.resident_bytes + placement.reserve
  model:close()
  compute.unload(backend)
  test.equal(handle.release(handle), nil)
  return vram
end

test.it(
  "bench reports the split and its evidence",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    local on_card = support.found_device()
    local vram = 1048576
    if on_card then
      vram = no_slot_vram(file_path)
    end
    local config_path = path_for("c") .. ".json"
    fs.write(
      config_path,
      '{"models":{"m":{"path":"' .. file_path .. '"}},"vram":' .. vram .. "}"
    )
    local got = run(
      {
        "bench",
        "m",
        "--config",
        config_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
      }
    )
    test.equal(got.code, 0, table.concat(got.err, "\n"))
    local line = experts_at(got.out, 8) as string
    test.truthy(line, table.concat(got.out, "\n"))
    local share, pool, link = line:match(
      "; (%d+%.%d)%% over PCIe, pool (%d+%.%d) GB/s, link (%d+%.%d) GB/s; "
        .. "%d+ direct$"
    )
    test.truthy(share, line)
    if not on_card then
      test.equal(share, "0.0")
      test.equal(pool, "0.0")
      test.equal(link, "0.0")
      return
    end
    local percent = tonumber(share as string) as number
    test.truthy(percent >= 10.0 and percent <= 95.0, line)
  end
)

test.it(
  "bench reports the direct matrices",
  function()
    local file_path = meta_file(meta(260, nil, true), entries())
    -- a card that holds every fixed tensor and no expert slot, so a turn's
    -- chosen matrices come from the mapping
    local body = '{"models":{"m":{"path":"'
      .. file_path
      .. '"}},"prefetch":false'
    if support.found_device() then
      body = body .. ',"vram":' .. no_slot_vram(file_path)
    end
    local config_path = path_for("c") .. ".json"
    fs.write(config_path, body .. "}")
    local args = {
      "bench",
      "m",
      "--config",
      config_path,
      "--prefill-tokens",
      "8",
      "--decode-tokens",
      "8",
    }
    local function tail(): string, string
      local got = run(args)
      test.equal(got.code, 0, table.concat(got.err, "\n"))
      local line = experts_at(got.out, 8) as string
      test.truthy(line, table.concat(got.out, "\n"))
      local reads, direct = line:match(
        "per turn (%d+) reads, .-; (%d+) direct$"
      )
      test.truthy(direct, line)
      return reads as string, direct as string
    end

    -- the fixture's expert matrices hold 96 bytes in all
    support.with_memory(
      function(): integer | nil return 96 end,
      function()
        local reads, direct = tail()
        test.equal(tonumber(reads), 0)
        test.greater(tonumber(direct) as number, 0)
      end
    )

    support.with_memory(
      function(): integer | nil return 0 end,
      function()
        local reads, direct = tail()
        test.equal(tonumber(direct), 0)
        test.greater(tonumber(reads) as number, 0)
      end
    )
  end
)
