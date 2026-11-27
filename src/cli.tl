local process = require("process")
local targs = require("targs")
local bench = require("tllm.bench")
local config = require("tllm.config")
local cores = require("tllm.cores")
local reference = require("tllm.reference")
local serve = require("tllm.serve")

local cli = {}

local type ModelEntry = config.ModelEntry
local type Config = config.Config

-- 54: the five model keys a run may give for the model it names
local function model_options(command: targs.Parser)
  command:option({name = "kv-cache", placeholder = "TYPE"})
  command:option({name = "ubatch", placeholder = "N", type = "number"})
  command:option({name = "vram", placeholder = "N", type = "number"})
  command:option({name = "profile", placeholder = "FILE"})
  command:flag({name = "prefetch"})
  command:flag({name = "no-prefetch"})
end

local function build(): targs.Parser
  local parser = targs.new(
    {
      name = "tllm",
      description = "An LLM inference engine",
    }
  )

  local serve_parser = parser:command(
    {
      name = "serve",
      description = "Serve a model",
    }
  )
  serve_parser:argument({name = "model", max_length = 1})
  serve_parser:option({name = "config", placeholder = "FILE"})
  serve_parser:option({name = "host", placeholder = "HOST"})
  serve_parser:option(
    {
      name = "port",
      placeholder = "PORT",
      type = "number",
    }
  )
  serve_parser:option(
    {
      name = "threads",
      placeholder = "N",
      type = "number",
    }
  )
  model_options(serve_parser)

  local bench_parser = parser:command(
    {
      name = "bench",
      description = "Benchmark a model",
    }
  )
  bench_parser:argument({name = "model", max_length = 1, required = true})
  bench_parser:option({name = "config", placeholder = "FILE"})
  bench_parser:option(
    {
      name = "prefill-tokens",
      placeholder = "N",
      type = "number",
    }
  )
  bench_parser:option(
    {
      name = "decode-tokens",
      placeholder = "N",
      type = "number",
    }
  )
  bench_parser:option({name = "dump-routing", placeholder = "FILE"})
  bench_parser:option(
    {
      name = "threads",
      placeholder = "N",
      type = "number",
    }
  )
  model_options(bench_parser)

  local reference_parser = parser:command(
    {
      name = "reference",
      description = "Check a model against a recorded answer",
    }
  )
  reference_parser:argument({name = "model", max_length = 1, required = true})
  reference_parser:option({name = "config", placeholder = "FILE"})
  reference_parser:option({name = "reference", placeholder = "FILE"})

  return parser
end

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function model_of(command: targs.Parser): string | number | nil
  local values = command:get_argument("model")
  if values is {string | number} and #values > 0 then
    return values[1]
  end
end

local function sorted_names(models: {string: ModelEntry}): {string}
  local names: {string} = {}
  for name in pairs(models) do
    names[#names + 1] = name
  end
  table.sort(names)
  return names
end

local record Chosen
  name: string
  path: string
  entry: ModelEntry | nil
end

local type Starter = function(
  endpoint: serve.Endpoint, write_err: function(string), spawn?: process.Spawn
): integer

local record Settings
  args: {string}
  write_out: function(string)
  write_err: function(string)
  start: Starter | nil
  environment: {string: string} | nil
end

local function chosen_model(
  command: string,
  models: {string: ModelEntry},
  word: string | nil,
  path_word: boolean,
  write_err: function(string)
): Chosen | nil
  if word is string and path_word then
    if not config.has_file(word) then
      write_err(
        one_line(
          command .. ": model path " .. word .. " is not a readable file"
        )
      )
      return nil
    end
    return {name = word, path = word, entry = nil}
  end
  if word is string then
    local configured = models[word]
    if configured is nil then
      write_err(one_line(command .. ": unknown model " .. word))
      return nil
    end
    return {name = word, path = configured.path, entry = configured}
  end

  local names = sorted_names(models)
  if #names == 0 then
    write_err(one_line(command .. ": no model configured"))
    return nil
  end
  if #names > 1 then
    write_err(
      one_line(
        command
          .. ": no model given; configured models are "
          .. table.concat(names, ", ")
      )
    )
    return nil
  end
  local only = models[names[1]]
  return {name = names[1], path = only.path, entry = only}
end

-- 02/FR-003: the model word, the config file and the model it names, the same
-- read for every command
local function resolved(
  command: targs.Parser,
  name: string,
  environment: {string: string},
  starter: boolean,
  write_err: function(string)
): Chosen | nil, Config | nil
  local word_value = model_of(command)
  local word: string | nil = nil
  if word_value is string then
    word = word_value
  end

  local path_word = false
  if word is string then
    path_word = config.is_model_path(word)
  end

  local explicit = command:get_option("config")
  local config_path: string | nil = nil
  if explicit is string then
    config_path = explicit
  end

  local settings, config_line = config.load(
    environment, config_path, path_word, starter
  )
  if settings is nil then
    write_err(one_line(config_line as string))
    return nil, nil
  end

  local chosen = chosen_model(name, settings.models, word, path_word, write_err)
  if chosen is nil then
    return nil, nil
  end
  return chosen, settings
end

local function process_environment(): {string: string}
  local environment: {string: string} = {}
  local xdg = os.getenv("XDG_CONFIG_HOME")
  if xdg is string then
    environment.XDG_CONFIG_HOME = xdg
  end
  local data = os.getenv("XDG_DATA_HOME")
  if data is string then
    environment.XDG_DATA_HOME = data
  end
  local home = os.getenv("HOME")
  if home is string then
    environment.HOME = home
  end
  return environment
end

-- 44/FR-006: the flag's count, nil when the flag was not given, and the
-- refusal when it is not one ggml can hold
local function flag_threads(
  command: targs.Parser, write_err: function(string)
): integer | nil, boolean
  local value = command:get_option("threads")
  if not (value is number) then
    return nil, false
  end
  if not cores.fits(value) then
    write_err(
      one_line(
        "'--threads' expects an integer from 1 to 512, got " .. tostring(value)
      )
    )
    return nil, true
  end
  return value as integer, false
end

-- 54/FR-006, FR-007: a count flag checked by the rule the config checks that
-- key by, and the refusal line naming the flag and its value
local function count_flag(
  command: targs.Parser, name: string, write_err: function(string)
): integer | nil, boolean
  local value = command:get_option(name)
  if not (value is number) then
    return nil, false
  end
  if not config.whole_count(value) then
    write_err(
      one_line(
        "'--"
          .. name
          .. "' expects an integer of one or more, got "
          .. tostring(value)
      )
    )
    return nil, true
  end
  return value as integer, false
end

-- 54/FR-001..FR-005, FR-008: the model keys the run gives, and the refusal
local function model_flags(
  command: targs.Parser, write_err: function(string)
): config.Level | nil, boolean
  local flags: config.Level = {}

  local cache = command:get_option("kv-cache")
  if cache is string then
    local kind = config.cache_type(cache)
    if kind is nil then
      write_err(one_line("'--kv-cache' expects a cache type, got " .. cache))
      return nil, true
    end
    flags.kv_cache = kind
  end

  local chunk, chunk_refused = count_flag(command, "ubatch", write_err)
  if chunk_refused then
    return nil, true
  end
  flags.ubatch = chunk

  local room, room_refused = count_flag(command, "vram", write_err)
  if room_refused then
    return nil, true
  end
  flags.vram = room

  local profile = command:get_option("profile")
  if profile is string then
    flags.profile = profile
  end

  local on = command:get_flag("prefetch")
  local off = command:get_flag("no-prefetch")
  if on and off then
    write_err(one_line("'--prefetch' and '--no-prefetch' cannot both be given"))
    return nil, true
  end
  if on then
    flags.prefetch = true
  elseif off then
    flags.prefetch = false
  end
  return flags, false
end

-- 44/FR-005: the default shared among the file's workers
local function shared(count: integer, workers: integer): integer
  local each = math.floor(count / workers)
  if each < 1 then
    return 1
  end
  return each
end

local function serve_code(
  command: targs.Parser,
  write_err: function(string),
  start: function(
    endpoint: serve.Endpoint, write_err: function(string), spawn?: process.Spawn
  ): integer,
  environment: {string: string}
): integer
  local given_port: integer | nil = nil
  local port = command:get_option("port")
  if port is number then
    if not (port % 1 == 0 and port >= 1 and port <= 65535) then
      write_err(
        one_line(
          "'--port' expects an integer from 1 to 65535, got " .. tostring(port)
        )
      )
      return 1
    end
    given_port = port as integer
  end

  local given_host: string | nil = nil
  local host = command:get_option("host")
  if host is string then
    if not config.valid_host(host) then
      write_err(
        one_line("'--host' expects a dotted IPv4 address, got " .. host)
      )
      return 1
    end
    given_host = host
  end

  local flag_count, refused = flag_threads(command, write_err)
  if refused then
    return 1
  end

  -- 54/FR-009: before the model word is resolved and the file is read
  local flags, flag_refused = model_flags(command, write_err)
  if flag_refused then
    return 1
  end

  local chosen, settings = resolved(
    command, "serve", environment, true, write_err
  )
  if chosen is nil or settings is nil then
    return 1
  end

  local bind_host = given_host or settings.host or "127.0.0.1"
  local bind_port = given_port or settings.port or 8080

  local threads = flag_count or settings.threads
  if threads is nil then
    threads = shared(cores.machine(), settings.workers or 1)
  end

  local values = config.model_values(settings, chosen.entry, flags)
  return start(
    {
      host = bind_host,
      port = bind_port,
      name = chosen.name,
      path = chosen.path,
      sessions = settings.sessions or 4,
      kv_cache = values.kv_cache,
      context_path = settings.context_path,
      context_bytes = settings.context_bytes,
      workers = settings.workers or 1,
      ubatch = values.ubatch,
      vram = values.vram,
      profile = values.profile,
      prefetch = values.prefetch,
      threads = threads,
    },
    write_err
  )
end

local function bench_code(
  command: targs.Parser,
  write_out: function(string),
  write_err: function(string),
  environment: {string: string}
): integer
  local lengths: {integer} = {512, 2048, 8192, 32768}
  local prefill = command:get_option("prefill-tokens")
  if prefill is number then
    if not config.whole_count(prefill) then
      write_err(
        one_line(
          "'--prefill-tokens' expects a positive integer, got "
            .. tostring(prefill)
        )
      )
      return 1
    end
    lengths = {prefill as integer}
  end

  local decode = 128
  local count = command:get_option("decode-tokens")
  if count is number then
    if not config.whole_count(count) then
      write_err(
        one_line(
          "'--decode-tokens' expects a positive integer, got "
            .. tostring(count)
        )
      )
      return 1
    end
    decode = count as integer
  end

  local flag_count, refused = flag_threads(command, write_err)
  if refused then
    return 1
  end

  -- 54/FR-009: before the model word is resolved and the file is read
  local flags, flag_refused = model_flags(command, write_err)
  if flag_refused then
    return 1
  end

  local chosen, settings = resolved(
    command, "bench", environment, false, write_err
  )
  if chosen is nil or settings is nil then
    return 1
  end

  local routing = command:get_option("dump-routing")
  local routing_path: string | nil = nil
  if routing is string then
    routing_path = routing
  end

  local values = config.model_values(settings, chosen.entry, flags)
  return bench.run(
    {
      path = chosen.path,
      name = chosen.name,
      lengths = lengths,
      decode = decode,
      kv_cache = values.kv_cache,
      routing_path = routing_path,
      ubatch = values.ubatch,
      vram = values.vram,
      profile = values.profile,
      prefetch = values.prefetch,
      threads = flag_count or settings.threads or cores.machine(),
    },
    write_out,
    write_err
  )
end

local function reference_code(
  command: targs.Parser,
  write_out: function(string),
  write_err: function(string),
  environment: {string: string}
): integer
  local recording = command:get_option("reference")
  if not (recording is string) then
    write_err(one_line("'--reference' expects a recording file"))
    return 1
  end

  local chosen, settings = resolved(
    command, "reference", environment, false, write_err
  )
  if chosen is nil or settings is nil then
    return 1
  end

  local values = config.model_values(settings, chosen.entry, nil)
  return reference.run(
    {
      path = chosen.path,
      name = chosen.name,
      recording = recording as string,
      kv_cache = values.kv_cache,
      ubatch = values.ubatch,
      vram = values.vram,
      profile = values.profile,
      prefetch = values.prefetch,
      threads = settings.threads,
    },
    write_out,
    write_err
  )
end

function cli.run(settings: Settings): integer
  local args = settings.args
  local write_out = settings.write_out
  local write_err = settings.write_err
  local start = settings.start
  local environment = settings.environment
  local parser = build()
  local ok, failure = parser:pparse(args)
  if not ok then
    local parsed = failure as targs.Parser.Error
    if parsed.kind == "exit" then
      write_out(one_line(parsed.message))
      return parsed.code
    end
    write_err(one_line(parsed.message))
    return parsed.code
  end

  local serve_command = parser:get_command("serve")
  local bench_command = parser:get_command("bench")

  if serve_command then
    return serve_code(
      serve_command,
      write_err,
      start or serve.start,
      environment or process_environment()
    )
  end

  if bench_command then
    return bench_code(
      bench_command, write_out, write_err, environment or process_environment()
    )
  end

  local reference_command = parser:get_command("reference")
  if reference_command then
    return reference_code(
      reference_command,
      write_out,
      write_err,
      environment or process_environment()
    )
  end

  write_out(one_line(parser:get_help()))
  return 1
end

return cli
