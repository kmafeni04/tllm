local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local cli = require("tllm.cli")
local generate = require("tllm.generate")
local serve = require("tllm.serve")
local sha256 = require("tllm.sha256")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local tltp = require("tltp")

local type Entry = support.Entry
local type Meta = support.Meta
local u32, u64, text = support.u32, support.u64, support.text
local padded, f32, mint = support.padded, support.f32, support.mint
local mfloat, mtext, mstrings, mint_array =
  support.mfloat, support.mtext, support.mstrings, support.mint_array
local both = support.both

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

-- 29's fixture: one block, two experts, no end id, and a template that renders
-- the message content alone
local function meta(context: integer): {Meta}
  local tokens = pieces()
  local types: {integer} = {}
  for _ = 1, #tokens do
    types[#types + 1] = 1
  end
  return {
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
    mtext("tokenizer.chat_template", TEMPLATE),
  }
end

local function entries(): {Entry}
  local count = #pieces()
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
    both(
      {
        f32("blk.0.attn_norm.weight", {2}, {1, 1}),
        f32("blk.0.attn_q.weight", {2, 4}, {1, 0, 0, 1, 0, 0, 0, 0}),
        f32("blk.0.attn_k.weight", {2, 2}, {1, 0, 0, 1}),
        f32("blk.0.attn_v.weight", {2, 2}, {1, 0, 0, 1}),
        f32("blk.0.attn_q_norm.weight", {2}, {1, 1}),
        f32("blk.0.attn_k_norm.weight", {2}, {1, 1}),
        f32("blk.0.attn_output.weight", {2, 2}, {0, 0, 0, 0}),
      },
      {
        f32("blk.0.post_attention_norm.weight", {2}, {1, 1}),
        f32("blk.0.ffn_gate_inp.weight", {2, 2}, {0, 0, 0, 0}),
        f32("blk.0.ffn_gate_inp_shexp.weight", {2}, {0, 0}),
        f32("blk.0.ffn_up_shexp.weight", {2, 2}, {0, 0, 0, 0}),
        f32("blk.0.ffn_gate_shexp.weight", {2, 2}, {0, 0, 0, 0}),
        f32("blk.0.ffn_down_shexp.weight", {2, 2}, {0, 0, 0, 0}),
        f32("blk.0.ffn_up_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
        f32("blk.0.ffn_gate_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
        f32("blk.0.ffn_down_exps.weight", {2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0}),
      }
    ),
    {
      f32("token_embd.weight", {2, count}, embedding),
      f32("output_norm.weight", {2}, {1, 1}),
      f32("output.weight", {2, count}, output),
    }
  )
end

local function model_file(context: integer): string
  return fixture(meta_bytes(meta(context), entries()))
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

-- the recording's shape, every field optional so a case can remove one
local record Candidate
  id: number | nil
  logprob: number | nil
end

local record Place
  position: number | nil
  five: {Candidate} | nil
end

local record Prompt
  prompt: string | nil
  prompt_ids: {integer} | nil
  ids: {integer} | nil
  top: {Place} | nil
  cold_tokens_per_second: number | nil
  steady_tokens_per_second: number | nil
end

local record Shard
  bytes: integer | nil
  sha256: string | nil
end

local record Named
  name: string | nil
  shards: {Shard} | nil
end

local record Sampling
  temperature: number | nil
  seed: integer | nil
  max_tokens: integer | nil
end

local record Sheet
  engine: string | nil
  date: string | nil
  sampling: Sampling | nil
  tolerance: number | nil
  model: Named | nil
  prompts: {Prompt} | nil
end

local function written(sheet: Sheet): string
  local sheet_path = path_for("r") .. ".json"
  fs.write(sheet_path, tltp.json_encode(sheet) as string)
  return sheet_path
end

local function shard_of(file_path: string): Shard
  local handle = weights.load(file_path) as weights.Handle
  local held = handle.shards[1]
  local found = sha256.descriptor(held.descriptor, held.bytes)
  handle.release(handle)
  return {bytes = held.bytes, sha256 = found as string}
end

-- the natural-log probability of one id over the whole row
local function logprob(values: {number}, id: integer): number
  local top = values[1]
  for index = 2, #values do
    if values[index] > top then
      top = values[index]
    end
  end
  local total: number = 0
  for index = 1, #values do
    total = total + math.exp(values[index] - top)
  end
  return (values[id + 1] - top) - math.log(total)
end

local record Answered
  ids: {integer}
  top: {Place}
end

-- what the fixture itself answers: the ids it samples, and the five candidates
-- at each of the first `places` of them
local function answered(
  file_path: string, ids: {integer}, most: integer, places: integer
): Answered
  local server = serve.build(
    {
      host = "127.0.0.1",
      port = 8080,
      name = "m",
      path = file_path,
      sessions = 1,
      kv_cache = nil,
      context_path = nil,
      context_bytes = nil,
      workers = 1,
      ubatch = nil,
      vram = nil,
      profile = nil,
      prefetch = nil,
      threads = nil,
    }
  ) as tltp.Server
  local model = serve.model_of(server) as serve.Model
  local generation = model.generation
  generation:release("")
  test.equal(generation:reserve("", generation.positions), nil)
  local open = generate.open_ids(
    generation,
    model.tokenizer,
    ids,
    {max_tokens = most, seed = 0, temperature = 0},
    ""
  ) as generate.Run

  local top: {Place} = {}
  while not open.done do
    local rows = open.next(open, generation.ubatch)
    if rows is nil then
      break
    end
    local given, line = generation:pass(rows)
    test.equal(line, nil)
    local scores = given as {{number}}
    local row = scores[#scores]
    local before = #open.produced
    open.accept(open, row)
    local count = #open.produced
    if count > before and count <= places then
      local five: {Candidate} = {}
      for _, id in ipairs(generate.top(row, 5)) do
        five[#five + 1] = {id = id, logprob = logprob(row, id)}
      end
      top[#top + 1] = {position = count - 1, five = five}
    end
  end
  serve.stop(server)
  return {ids = open.produced, top = top}
end

local function sheet_of(
  file_path: string, ids: {integer}, most: integer, places: integer
): Sheet
  local given = answered(file_path, ids, most, places)
  return {
    engine = "strata qwen3.8-flash-next-iq3_s, speculation off",
    date = "2026-12-01",
    sampling = {temperature = 0, seed = 0, max_tokens = most},
    tolerance = 0.01,
    model = {
      name = "Qwen3.8-Flash-Next GSQ-RCO",
      shards = {shard_of(file_path)},
    },
    prompts = {
      {
        prompt = FILLER:sub(1, #ids),
        prompt_ids = ids,
        ids = given.ids,
        top = given.top,
        cold_tokens_per_second = 12.4,
        steady_tokens_per_second = 41.2,
      },
    },
  }
end

local function prompt_ids(file_path: string, count: integer): {integer}
  local tokens = tokenizer.load(file_path) as tokenizer.Tokenizer
  return tokens:encode(FILLER:sub(1, count)) as {integer}
end

local function differences(out: {string}): integer
  local count = 0
  for _, line in ipairs(out) do
    if line:match("^prompt %d+ position ") then
      count = count + 1
    end
  end
  return count
end

local function listed(values: {Candidate}): string
  local parts: {string} = {}
  for _, candidate in ipairs(values) do
    parts[#parts + 1] = tostring(candidate.id as number)
  end
  return table.concat(parts, ", ")
end

test.it(
  "a recording that agrees answers PASS",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local got = run(
      {
        "reference",
        file_path,
        "--reference",
        written(sheet_of(file_path, ids, 2, 2)),
      }
    )
    test.equal(got.code, 0, table.concat(got.out, "\n"))
    test.equal(#got.err, 0, table.concat(got.err, "\n"))
    test.equal(#got.out, 3, table.concat(got.out, "\n"))
    test.includes(got.out[1], "strata qwen3.8-flash-next-iq3_s")
    test.includes(got.out[1], "2026-12-01")
    test.includes(got.out[2], "prompt 1:")
    test.includes(got.out[2], "2 of 2 ids")
    test.includes(got.out[2], "2 of 2 positions")
    test.includes(got.out[2], "12.4")
    test.includes(got.out[2], "41.2")
    test.equal(got.out[3], "reference: PASS")

    -- the turn is fed prompt_ids, so the text is only for the human
    local sheet = sheet_of(file_path, ids, 2, 2)
    (sheet.prompts as {Prompt})[1].prompt = "\228\189\160"
    local unreadable = run(
      {"reference", file_path, "--reference", written(sheet)}
    )
    test.equal(unreadable.code, 0, table.concat(unreadable.out, "\n"))
    test.includes(unreadable.out[2], "2 of 2 ids")
  end
)

test.it(
  "a differing id fails the checkpoint",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local sheet = sheet_of(file_path, ids, 2, 2)
    local sampled = (sheet.prompts as {Prompt})[1].ids as {integer}
    local mine = sampled[1]
    local other = mine - 1
    if other < 0 then
      other = mine + 1
    end
    sampled[1] = other
    local got = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(got.code, 1)
    test.equal(differences(got.out), 1, table.concat(got.out, "\n"))
    test.includes(got.out[2], "prompt 1")
    test.includes(got.out[2], "position 0")
    test.includes(got.out[2], tostring(other))
    test.includes(got.out[2], tostring(mine))
    test.equal(got.out[#got.out], "reference: FAIL")

    sampled[1] = mine
    local fixed = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(fixed.code, 0, table.concat(fixed.out, "\n"))
    test.equal(differences(fixed.out), 0, table.concat(fixed.out, "\n"))
  end
)

test.it(
  "the five candidates at a position are compared",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local sheet = sheet_of(file_path, ids, 2, 2)
    local places = (sheet.prompts as {Prompt})[1].top as {Place}
    local five = places[2].five as {Candidate}
    local high = five[1]
    local low = five[2]

    five[1] = low
    five[2] = high
    local swapped = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(swapped.code, 1)
    test.equal(differences(swapped.out), 1, table.concat(swapped.out, "\n"))
    test.includes(swapped.out[2], "position 1")
    test.includes(swapped.out[2], listed({low, high}))
    test.includes(swapped.out[2], listed({high, low}))

    five[1] = high
    five[2] = low
    local kept = high.logprob as number
    high.logprob = kept + 0.5
    local apart = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(apart.code, 1, table.concat(apart.out, "\n"))
    test.equal(differences(apart.out), 1, table.concat(apart.out, "\n"))
    test.includes(apart.out[2], "position 1")
    test.includes(apart.out[2], tostring(high.id as number))
    test.includes(apart.out[2], string.format("%.4f", kept + 0.5))
    test.includes(apart.out[2], string.format("%.4f", kept))

    high.logprob = kept + 0.001
    local close = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(close.code, 0, table.concat(close.out, "\n"))
  end
)

test.it(
  "a recording that names another file is refused",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local sheet = sheet_of(file_path, ids, 2, 2)
    local shards = (sheet.model as Named).shards as {Shard}
    local held = shards[1].sha256 as string
    local wanted = "0000000000000000000000000000000000000000000000000000000000000000"
    shards[1].sha256 = wanted
    local wrong = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(wrong.code, 1)
    test.equal(#wrong.out, 0, table.concat(wrong.out, "\n"))
    test.equal(#wrong.err, 1, table.concat(wrong.err, "\n"))
    test.includes(wrong.err[1], file_path)
    test.includes(wrong.err[1], wanted)
    test.includes(wrong.err[1], held)

    shards[1].sha256 = held
    shards[2] = {bytes = 1, sha256 = held}
    local twice = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(twice.code, 1)
    test.equal(#twice.out, 0, table.concat(twice.out, "\n"))
    test.equal(#twice.err, 1, table.concat(twice.err, "\n"))
    test.includes(twice.err[1], "names 2 shards")
    test.includes(twice.err[1], "has 1")
  end
)

test.it(
  "a recording is checked before a turn runs",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local words = (tokenizer.load(file_path) as tokenizer.Tokenizer).size

    local no_places = sheet_of(file_path, ids, 2, 2)
    (no_places.prompts as {Prompt})[1].top = nil
    local top = run({"reference", file_path, "--reference", written(no_places)})
    test.equal(top.code, 1)
    test.equal(#top.out, 0, table.concat(top.out, "\n"))
    test.includes(top.err[1], "top")

    local no_sampling = sheet_of(file_path, ids, 2, 2)
    no_sampling.sampling = nil
    local sampling = run(
      {"reference", file_path, "--reference", written(no_sampling)}
    )
    test.equal(sampling.code, 1)
    test.equal(#sampling.out, 0, table.concat(sampling.out, "\n"))
    test.includes(sampling.err[1], "sampling")

    local high = sheet_of(file_path, ids, 2, 2)
    local sampled = ((high.prompts as {Prompt})[1]).ids as {integer}
    sampled[1] = words
    local beyond = run({"reference", file_path, "--reference", written(high)})
    test.equal(beyond.code, 1)
    test.equal(#beyond.out, 0, table.concat(beyond.out, "\n"))
    test.includes(beyond.err[1], tostring(words))

    local long = sheet_of(file_path, ids, 2, 2)
    local given: {integer} = {}
    for index = 1, 300 do
      given[index] = ids[(index % #ids) + 1]
    end
    ((long.prompts as {Prompt})[1]).prompt_ids = given
    local past = run({"reference", file_path, "--reference", written(long)})
    test.equal(past.code, 1)
    test.equal(#past.out, 0, table.concat(past.out, "\n"))
    test.includes(past.err[1], "300")
    test.includes(past.err[1], "260")
  end
)

test.it(
  "reference finds its model and its recording",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local sheet_path = written(sheet_of(file_path, ids, 2, 2))
    local config_path = path_for("c") .. ".json"
    fs.write(config_path, '{"models":{"m":{"path":"' .. file_path .. '"}}}')

    local named = run(
      {"reference", "m", "--config", config_path, "--reference", sheet_path}
    )
    test.equal(named.code, 0, table.concat(named.out, "\n"))
    test.includes(named.out[2], "prompt 1:")

    local missing = path_for("none") .. ".json"
    local absent = run({"reference", file_path, "--reference", missing})
    test.equal(absent.code, 1)
    test.equal(#absent.err, 1, table.concat(absent.err, "\n"))
    test.includes(absent.err[1], missing)

    local help = run({"reference", "--help"})
    test.equal(help.code, 0)
    test.equal(#help.err, 0)
    test.includes(help.out[1], "--reference")
  end
)

test.it(
  "a short answer fails the checkpoint",
  function()
    local file_path = model_file(260)
    local ids = prompt_ids(file_path, 1)
    local sheet = sheet_of(file_path, ids, 2, 2)
    local sampled = (sheet.prompts as {Prompt})[1].ids as {integer}
    sampled[#sampled + 1] = sampled[1]
    sampled[#sampled + 1] = sampled[2]

    local got = run({"reference", file_path, "--reference", written(sheet)})
    test.equal(got.code, 1, table.concat(got.out, "\n"))
    test.equal(differences(got.out), 0, table.concat(got.out, "\n"))
    test.includes(got.out[2], "prompt 1:")
    test.includes(got.out[2], "2 of 4 ids")
    test.includes(got.out[2], "2 of 2 positions")
    test.equal(got.out[3], "reference: FAIL")
  end
)
