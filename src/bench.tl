local fs = require("fs")
local clock = require("tllm.clock")
local compute = require("tllm.compute")
local serve = require("tllm.serve")
local tokenizer = require("tllm.tokenizer")
local tltp = require("tltp")

local record bench
  record Settings
    path: string
    name: string
    lengths: {integer}
    decode: integer
    kv_cache: compute.CacheType | nil
    routing_path: string | nil
    ubatch: integer | nil
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
  end

  run: function(Settings, function(string), function(string)): integer
end

local record LooseObject
end

local type Loose = nil | boolean | number | string | LooseObject
local type Fields = {string: Loose}

local record EventChoice
  index: integer
  delta: Fields | nil
  finish_reason: Loose
end

local record EventBody
  choices: {EventChoice}
end

local record Frame
  at: number
  compute: number
  sampling: number
  ready: number
  routed: integer
  bytes: integer
  tokens: integer
end

local record Report
  started: number
  ended: number
  tokens: integer
  reads: integer
  read_bytes: integer
  read_seconds: number
  direct: integer
  routes: integer
  hits: integer
  ahead: integer
  named: integer
  landed: integer
  open: boolean
  miss: compute.MissPolicy
  frames: {Frame}
end

local FILLER = "The quick brown fox jumps over the lazy dog. "

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function filler(
  tokens: tokenizer.Tokenizer, length: integer
): string | nil, string | nil
  local aside, aside_line = tokens:encode(FILLER)
  if aside is nil then
    return nil, aside_line
  end
  local each = #(aside as {integer})
  if each < 1 then
    each = 1
  end
  local content = string.rep(FILLER, math.ceil(length / each))
  while true do
    local ids, line = tokens:encode(content)
    if ids is nil then
      return nil, line
    end
    if #(ids as {integer}) >= length then
      return content, nil
    end
    content = content .. FILLER
  end
end

local function zeroed(measure: compute.Measure)
  measure.passes = 0
  measure.tokens = 0
  measure.compute = 0
  measure.sampling = 0
  measure.ready = 0
  measure.routed = 0
  measure.routed_bytes = 0
  measure.expert_hits = 0
  measure.blob_reads = 0
  measure.blob_bytes = 0
  measure.blob_seconds = 0
  measure.direct_reads = 0
  measure.prefetch_reads = 0
  measure.prefetch_named = 0
  measure.prefetch_landed = 0
end

local function driven(
  server: tltp.Server,
  stream_fn: tltp.StreamFn,
  measure: compute.Measure,
  started: number,
  ahead: compute.ReadAhead,
  miss: compute.MissPolicy
): Report | nil, string | nil
  local frames: {Frame} = {}
  local function emit(payload: string): string | nil, string | nil
    local at = clock.now()
    local decoded = tltp.json_decode(payload) as EventBody | nil
    if decoded is nil then
      return nil, nil
    end
    local choices = (decoded as EventBody).choices
    if choices is nil or #choices == 0 then
      return nil, nil
    end
    local delta = choices[1].delta
    if delta is nil then
      return nil, nil
    end
    local text = delta.content
    if not (text is string) then
      return nil, nil
    end
    frames[#frames + 1] = {
      at = at,
      compute = measure.compute,
      sampling = measure.sampling,
      ready = measure.ready,
      routed = measure.routed,
      bytes = measure.routed_bytes,
      tokens = measure.tokens,
    }
    return nil, nil
  end

  local entry = stream_fn as function(tltp.EmitFn): boolean | nil
  local producer = coroutine.create(entry)
  -- a producer that raises never registers its turn, so nothing would be
  -- measured and the zeros would read as a pass
  local resumed, why = coroutine.resume(producer, emit)
  if not resumed then
    return nil, tostring(why)
  end
  while serve.step(server) > 0 do
  end
  local ended = clock.now()
  return {
    started = started,
    ended = ended,
    tokens = measure.tokens,
    reads = measure.blob_reads,
    read_bytes = measure.blob_bytes,
    read_seconds = measure.blob_seconds,
    direct = measure.direct_reads,
    routes = measure.routed,
    hits = measure.expert_hits,
    ahead = measure.prefetch_reads,
    named = measure.prefetch_named,
    landed = measure.prefetch_landed,
    open = ahead.open,
    miss = miss,
    frames = frames,
  }
end

local function reported(
  length: integer, report: Report, write_out: function(string)
)
  local frames = report.frames
  local count = #frames
  local prompt_count = report.tokens
  local prompt_end = report.ended
  if count > 0 then
    prompt_count = frames[1].tokens
    prompt_end = frames[1].at
  end
  local prompt_seconds = prompt_end - report.started
  local prompt_rate = 0.0
  if prompt_seconds > 0 then
    prompt_rate = prompt_count / prompt_seconds
  end
  write_out(
    string.format(
      "prompt %d: %d tokens at %.1f tok/s, first token %.1f ms\n",
      length,
      prompt_count,
      prompt_rate,
      prompt_seconds * 1000
    )
  )

  local decode_count = 0
  if count > 0 then
    decode_count = count - 1
  end
  local decode_seconds = 0.0
  if decode_count > 0 then
    decode_seconds = frames[count].at - frames[1].at
  end
  local decode_rate = 0.0
  if decode_count > 0 and decode_seconds > 0 then
    decode_rate = decode_count / decode_seconds
  end
  write_out(
    string.format(
      "decode %d: %d tokens at %.1f tok/s\n", length, decode_count, decode_rate
    )
  )

  if decode_count < 1 then
    return
  end
  local first = frames[1]
  local last = frames[count]
  local total = (last.at - first.at) / decode_count * 1000
  local compute_time = (last.compute - first.compute) / decode_count * 1000
  local sampling_time = (last.sampling - first.sampling) / decode_count * 1000
  local ipc: number = 0
  for index = 2, count do
    ipc = ipc + (frames[index].at - frames[index].ready)
  end
  ipc = ipc / decode_count * 1000
  write_out(
    string.format(
      "token %d: %.3f ms, compute %.3f ms, sampling %.3f ms, ipc %.3f ms\n",
      length,
      total,
      compute_time,
      sampling_time,
      ipc
    )
  )
  local routed = math.floor((last.routed - first.routed) / decode_count)
  local megabytes = (last.bytes - first.bytes) / decode_count / 1000000
  local hits = 0.0
  if report.routes > 0 then
    hits = report.hits / report.routes * 100
  end
  local landed = 0.0
  if report.named > 0 then
    landed = report.landed / report.named * 100
  end
  local state = "closed"
  if report.open then
    state = "open"
  end
  local policy = report.miss
  write_out(
    string.format(
      "experts %d: %d routed, %.1f MB; per turn %d reads, %.1f MB read, "
        .. "%.3f ms; %.1f%% hit; %d ahead, %.1f%% landed; read ahead %s; "
        .. "%.1f%% over PCIe, pool %.1f GB/s, link %.1f GB/s; %d direct\n",
      length,
      routed,
      megabytes,
      report.reads,
      report.read_bytes / 1000000,
      report.read_seconds * 1000,
      hits,
      report.ahead,
      landed,
      state,
      policy.share * 100,
      policy.pool,
      policy.link,
      report.direct
    )
  )
end

function bench.run(
  settings: bench.Settings,
  write_out: function(string),
  write_err: function(string)
): integer
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
    }
  )
  if server is nil then
    write_err(
      one_line("bench: " .. settings.name .. ": " .. (failure as string))
    )
    return 1
  end

  local model = serve.model_of(server) as serve.Model
  local generation = model.generation
  local measure = generation.measure
  local ran = 0
  local failed = false

  for _, length in ipairs(settings.lengths) do
    local content, content_line = filler(model.tokenizer, length)
    if content is nil then
      write_err(
        one_line("bench: " .. length .. ": " .. (content_line as string))
      )
      failed = true
    else
      local body = tltp.json_encode(
        {
          messages = {{role = "user", content = content}},
          max_tokens = settings.decode,
          temperature = 0,
          seed = 0,
          stream = true,
        }
      ) as string
      zeroed(measure)
      local started = clock.now()
      local response, stream_fn = serve.stream(server, body)
      if response.status ~= "OK" then
        if response.status == "BadRequest" then
          write_out(one_line("skipped " .. length .. ": " .. response.body))
        else
          write_err(one_line("bench: " .. length .. ": " .. response.status))
          failed = true
        end
      elseif stream_fn is nil then
        write_err(one_line("bench: " .. length .. ": no producer"))
        failed = true
      else
        ran = ran + 1
        local lines: {string} | nil = nil
        local routing_path = settings.routing_path
        if routing_path is string and not fs.write(routing_path, "") then
          write_err(one_line("bench: cannot write " .. routing_path))
          return 1
        end
        if routing_path is string then
          lines = {}
          generation.routing_sink = function(routing: compute.Routing)
            local held = lines
            if held is nil then
              return
            end
            held[#held + 1] = tltp.json_encode(
              {
                layer = routing.layer,
                token = routing.token,
                experts = routing.experts,
              }
            ) as string
          end
        end

        local report, report_line = driven(
          server,
          stream_fn,
          measure,
          started,
          generation.read_ahead,
          generation.miss
        )
        generation.routing_sink = nil
        if lines is {string} then
          local text = ""
          if #lines > 0 then
            text = table.concat(lines, "\n") .. "\n"
          end
          fs.write(routing_path as string, text)
        end
        if report is nil then
          write_err(
            one_line("bench: " .. length .. ": " .. (report_line as string))
          )
          failed = true
        else
          reported(length, report, write_out)
        end
      end
    end
  end

  if failed or ran == 0 then
    return 1
  end
  return 0
end

return bench
