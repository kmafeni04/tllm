local fs = require("fs")
local clock = require("tllm.clock")
local compute = require("tllm.compute")
local cores = require("tllm.cores")
local generate = require("tllm.generate")
local serve = require("tllm.serve")
local sha256 = require("tllm.sha256")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local tltp = require("tltp")

local record reference
  record Settings
    path: string
    name: string
    recording: string
    kv_cache: compute.CacheType | nil
    ubatch: integer | nil
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
    threads: integer | nil
  end

  run: function(Settings, function(string), function(string)): integer
end

local record LooseObject
end

local type Loose = nil | boolean | number | string | LooseObject

-- every field of a recording is optional to the compiler and checked at run
-- time, because FR-010 refuses the recording that misses one
local record Five
  id: number | nil
  logprob: number | nil
end

local record Place
  position: number | nil
  five: {Five} | nil
end

local record Prompt
  prompt: string | nil
  prompt_ids: {number} | nil
  ids: {number} | nil
  top: {Place} | nil
  cold_tokens_per_second: number | nil
  steady_tokens_per_second: number | nil
end

local record Shard
  bytes: number | nil
  sha256: string | nil
end

local record Named
  name: string | nil
  shards: {Shard} | nil
end

local record Sampling
  temperature: number | nil
  seed: number | nil
  max_tokens: number | nil
end

local record Sheet
  engine: string | nil
  date: string | nil
  sampling: Sampling | nil
  tolerance: number | nil
  model: Named | nil
  prompts: {Prompt} | nil
end

local record Checked
  ids: integer
  places: integer
  recorded: integer
  recorded_places: integer
  cold: number
  steady: number
  line: string | nil
  failure: string | nil
end

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function whole(value: number | nil): boolean
  return value is number and value % 1 == 0
end

local function as_integers(values: {number}): {integer}
  local ids: {integer} = {}
  for _, value in ipairs(values) do
    ids[#ids + 1] = math.floor(value) as integer
  end
  return ids
end

local function listed(values: {integer}): string
  local parts: {string} = {}
  for _, value in ipairs(values) do
    parts[#parts + 1] = tostring(value)
  end
  return table.concat(parts, ", ")
end

local function checked(sheet: Sheet): string | nil
  local engine = sheet.engine
  local date = sheet.date
  if not (engine is string) then
    return "engine"
  end
  if not (date is string) then
    return "date"
  end

  local sampling = sheet.sampling
  if sampling is nil then
    return "sampling"
  end
  local heat = sampling.temperature
  if
    not (heat is number)
    or not whole(sampling.seed)
    or not whole(sampling.max_tokens)
  then
    return "sampling"
  end

  local tolerance = sheet.tolerance
  if not (tolerance is number) then
    return "tolerance"
  end

  local named = sheet.model
  if named is nil then
    return "model"
  end
  local family = named.name
  local shards = named.shards
  if not (family is string) then
    return "model"
  end
  if shards is nil or #shards < 1 then
    return "shards"
  end
  for _, shard in ipairs(shards) do
    local hash = shard.sha256
    if not whole(shard.bytes) or not (hash is string) then
      return "shards"
    end
  end

  local prompts = sheet.prompts
  if prompts is nil or #prompts < 1 then
    return "prompts"
  end
  for _, each in ipairs(prompts) do
    local text = each.prompt
    local given = each.prompt_ids
    local ids = each.ids
    local places = each.top
    local cold = each.cold_tokens_per_second
    local steady = each.steady_tokens_per_second
    if not (text is string) then
      return "prompt"
    end
    if given is nil then
      return "prompt_ids"
    end
    if ids is nil then
      return "ids"
    end
    if places is nil then
      return "top"
    end
    if not (cold is number) or not (steady is number) then
      return "tokens_per_second"
    end
    for _, id in ipairs(given) do
      if not whole(id) then
        return "prompt_ids"
      end
    end
    for _, id in ipairs(ids) do
      if not whole(id) then
        return "ids"
      end
    end
    for _, place in ipairs(places) do
      local five = place.five
      if not whole(place.position) or five is nil or #five < 1 then
        return "top"
      end
      for _, candidate in ipairs(five) do
        local logprob = candidate.logprob
        if not whole(candidate.id) or not (logprob is number) then
          return "top"
        end
      end
    end
  end
  return nil
end

local function beyond(ids: {number}, words: integer): number | nil
  for _, id in ipairs(ids) do
    if id >= words then
      return id
    end
  end
end

-- the first recorded candidate id at or above the vocabulary count, or nil
local function far_candidate(each: Prompt, words: integer): number | nil
  for _, place in ipairs(each.top as {Place}) do
    local ids: {number} = {}
    for _, candidate in ipairs(place.five as {Five}) do
      ids[#ids + 1] = candidate.id as number
    end
    local high = beyond(ids, words)
    if high is number then
      return high
    end
  end
end

local function out_of_range(
  sheet: Sheet, words: integer, positions: integer
): string | nil
  for _, each in ipairs(sheet.prompts as {Prompt}) do
    local given = each.prompt_ids as {number}
    if #given > positions then
      return "prompt holds "
        .. tostring(#given)
        .. " ids, the model holds "
        .. tostring(positions)
        .. " positions"
    end
    local high = beyond(given, words)
    if high is nil then
      high = beyond(each.ids as {number}, words)
    end
    if high is nil then
      high = far_candidate(each, words)
    end
    if high is number then
      return "id "
        .. tostring(high)
        .. ", the model holds "
        .. tostring(words)
        .. " ids"
    end
  end
end

-- FR-004: every shard of the loaded handle, hashed through its own descriptor
local function named_shards(sheet: Sheet, handle: weights.Handle): string | nil
  local recorded = (sheet.model as Named).shards as {Shard}
  if #recorded ~= #handle.shards then
    return "the recording names "
      .. tostring(#recorded)
      .. " shards, the model has "
      .. tostring(#handle.shards)
  end
  for index, shard in ipairs(handle.shards) do
    local wanted = (recorded[index].sha256 as string):lower()
    local found = sha256.descriptor(shard.descriptor, shard.bytes)
    if found is nil then
      return shard.path .. " sha256 could not be read"
    end
    if (found as string) ~= wanted then
      return shard.path
        .. " sha256 "
        .. wanted
        .. ", the file has "
        .. (found as string)
    end
  end
end

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

-- the first disagreement at one position, or nil when that position agrees
local function compared(
  ordinal: integer,
  position: integer,
  scores: {number},
  chosen: integer,
  each: Prompt,
  place: Place | nil,
  tolerance: number
): string | nil
  local where = "prompt "
    .. tostring(ordinal)
    .. " position "
    .. tostring(position)
  local recorded = (each.ids as {number})[position + 1]
  if recorded ~= (chosen as number) then
    if recorded is nil then
      return where
        .. ": recorded "
        .. tostring(#(each.ids as {number}))
        .. " ids, engine id "
        .. tostring(chosen)
    end
    return where
      .. ": recorded id "
      .. tostring(recorded)
      .. ", engine id "
      .. tostring(chosen)
  end
  if place is nil then
    return nil
  end
  local five = place.five as {Five}
  local mine = generate.top(scores, #five)
  local wanted: {integer} = {}
  for _, candidate in ipairs(five) do
    wanted[#wanted + 1] = math.floor(candidate.id as number) as integer
  end
  for index = 1, #five do
    if mine[index] ~= wanted[index] then
      return where
        .. ": recorded ids "
        .. listed(wanted)
        .. ", engine ids "
        .. listed(mine)
    end
  end
  for index = 1, #five do
    local id = wanted[index]
    local here = logprob(scores, id)
    local theirs = five[index].logprob as number
    if math.abs(here - theirs) > tolerance then
      return where
        .. ": recorded id "
        .. tostring(id)
        .. " logprob "
        .. string.format("%.4f", theirs)
        .. ", engine "
        .. string.format("%.4f", here)
    end
  end
end

-- a turn that never ran: nothing agreed, and `line` says why
local function no_turn(each: Prompt, held: integer, line: string): Checked
  return {
    ids = 0,
    places = 0,
    recorded = #(each.ids as {number}),
    recorded_places = held,
    cold = 0,
    steady = 0,
    line = nil,
    failure = line,
  }
end

-- one prompt's turn, its ids and its recorded positions compared as it runs,
-- so no score row outlives the pass that made it
local function turn(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  each: Prompt,
  sampling: Sampling,
  tolerance: number,
  ordinal: integer
): Checked
  local places: {integer: Place} = {}
  local held = 0
  for _, place in ipairs(each.top as {Place}) do
    places[math.floor(place.position as number) as integer] = place
    held = held + 1
  end
  local options: generate.Options = {
    max_tokens = math.floor(sampling.max_tokens as number) as integer,
    seed = math.floor(sampling.seed as number) as integer,
    temperature = sampling.temperature,
  }

  model:release("")
  local room = model.positions
  local held_line = model:reserve("", room)
  if held_line is string then
    return no_turn(each, held, held_line)
  end

  local run, open_line = generate.open_ids(
    model, tokens, as_integers(each.prompt_ids as {number}), options, ""
  )
  if run is nil then
    return no_turn(each, held, open_line as string)
  end

  local agreed_ids = 0
  local agreed_places = 0
  local line: string | nil = nil
  local failure: string | nil = nil
  local first: number | nil = nil
  local last: number | nil = nil
  local started = clock.now()
  while not run.done do
    local rows = run.next(run, model.ubatch)
    if rows is nil then
      break
    end
    local scores, pass_line = model:pass(rows)
    if scores is nil then
      failure = pass_line as string
      break
    end
    local rows_out = scores as {{number}}
    local row = rows_out[#rows_out]
    local before = #run.produced
    local _, accept_line = run.accept(run, row)
    if accept_line is string then
      failure = accept_line
      break
    end
    local count = #run.produced
    if count > before then
      local at = clock.now()
      if first is nil then
        first = at
      end
      last = at
      local position = count - 1
      local place: Place | nil = places[position]
      local difference = compared(
        ordinal, position, row, run.produced[count], each, place, tolerance
      )
      if difference is string then
        if line is nil then
          line = difference
        end
      else
        agreed_ids = agreed_ids + 1
        if place is Place then
          agreed_places = agreed_places + 1
        end
      end
    end
  end
  local ended = clock.now()
  model:release("")

  local count = #run.produced
  local cold = 0.0
  if count > 0 and ended > started then
    cold = count / (ended - started)
  end
  local steady = 0.0
  if count > 1 and first is number and last is number then
    local span = last - first
    if span > 0 then
      steady = (count - 1) / span
    end
  end
  return {
    ids = agreed_ids,
    places = agreed_places,
    recorded = #(each.ids as {number}),
    recorded_places = held,
    cold = cold,
    steady = steady,
    line = line,
    failure = failure,
  }
end

function reference.run(
  settings: reference.Settings,
  write_out: function(string),
  write_err: function(string)
): integer
  local text = fs.read(settings.recording)
  if not (text is string) then
    write_err(one_line("reference: cannot read " .. settings.recording))
    return 1
  end

  local decoded = tltp.json_decode(text) as Loose
  if not (decoded is LooseObject) then
    write_err(
      one_line("reference: " .. settings.recording .. " is not a JSON object")
    )
    return 1
  end

  local sheet = decoded as Sheet
  local key = checked(sheet)
  if key is string then
    write_err(
      one_line("reference: " .. settings.recording .. " has no " .. key)
    )
    return 1
  end

  local server, failure = serve.build(
    {
      host = "127.0.0.1",
      port = 8080,
      name = settings.name,
      path = settings.path,
      sessions = 1,
      kv_cache = settings.kv_cache,
      context_path = nil,
      context_bytes = nil,
      workers = 1,
      ubatch = settings.ubatch,
      vram = settings.vram,
      profile = settings.profile,
      prefetch = settings.prefetch,
      threads = settings.threads or cores.machine(),
    }
  )
  if server is nil then
    write_err(
      one_line("reference: " .. settings.name .. ": " .. (failure as string))
    )
    return 1
  end

  local model = serve.model_of(server) as serve.Model
  local generation = model.generation

  -- FR-004 and FR-010: a recording that does not name this file, or that holds
  -- a shape or an id this model cannot answer, is refused before a turn runs
  local shard_line = named_shards(sheet, model.handle)
  if shard_line is string then
    write_err(one_line("reference: " .. shard_line))
    serve.stop(server)
    return 1
  end

  local range_line = out_of_range(
    sheet, model.tokenizer.size, generation.positions
  )
  if range_line is string then
    write_err(
      one_line("reference: " .. settings.recording .. " holds " .. range_line)
    )
    serve.stop(server)
    return 1
  end

  write_out(
    one_line(
      "recording: "
        .. (sheet.engine as string)
        .. ", recorded "
        .. (sheet.date as string)
    )
  )
  local sampling = sheet.sampling as Sampling
  local tolerance = sheet.tolerance as number
  local differed = false
  for ordinal, each in ipairs(sheet.prompts as {Prompt}) do
    local ran = turn(
      generation, model.tokenizer, each, sampling, tolerance, ordinal
    )
    local refused = ran.failure
    local difference = ran.line
    if refused is string then
      write_err(
        one_line("reference: prompt " .. tostring(ordinal) .. ": " .. refused)
      )
      differed = true
    else
      if difference is string then
        write_out(one_line(difference))
        differed = true
      end
      if ran.ids ~= ran.recorded or ran.places ~= ran.recorded_places then
        differed = true
      end
      write_out(
        one_line(
          "prompt "
            .. tostring(ordinal)
            .. ": "
            .. tostring(ran.ids)
            .. " of "
            .. tostring(ran.recorded)
            .. " ids, "
            .. tostring(ran.places)
            .. " of "
            .. tostring(ran.recorded_places)
            .. " positions, recorded "
            .. string.format("%.1f", each.cold_tokens_per_second as number)
            .. " cold "
            .. string.format("%.1f", each.steady_tokens_per_second as number)
            .. " steady tok/s, engine "
            .. string.format("%.1f", ran.cold)
            .. " cold "
            .. string.format("%.1f", ran.steady)
            .. " steady tok/s"
        )
      )
    end
  end

  serve.stop(server)
  if differed then
    write_out(one_line("reference: FAIL"))
    return 1
  end
  write_out(one_line("reference: PASS"))
  return 0
end

return reference
