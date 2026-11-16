local targs = require("targs")
local compute = require("tllm.compute")
local config = require("tllm.config")
local serve = require("tllm.serve")

local cli = {}

local record Entry
  label: string
  text: string
end

local record Outcome
  code: integer
  text: string
  to_err: boolean
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

  local bench = parser:command(
    {
      name = "bench",
      description = "Benchmark a model",
    }
  )
  bench:argument({name = "model", max_length = 1, required = true})
  bench:option({name = "config", placeholder = "FILE"})
  bench:option(
    {
      name = "prefill-tokens",
      placeholder = "N",
      type = "number",
    }
  )
  bench:option(
    {
      name = "decode-tokens",
      placeholder = "N",
      type = "number",
    }
  )

  return parser
end

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function add(
  entries: {Entry}, label: string, value: string | number | nil
)
  if not (value is nil) then
    local entry: Entry = {label = label}
    if value is string then
      entry.text = value
    else
      entry.text = tostring(value)
    end
    entries[#entries + 1] = entry
  end
end

local function refusal(command: string, entries: {Entry}): string
  if #entries == 0 then
    return command .. ": not implemented"
  end
  local parts: {string} = {}
  for _, entry in ipairs(entries) do
    parts[#parts + 1] = entry.label .. " " .. entry.text
  end
  return command .. ": not implemented (" .. table.concat(parts, ", ") .. ")"
end

local function model_of(command: targs.Parser): string | number | nil
  local values = command:get_argument("model")
  if values is {string | number} and #values > 0 then
    return values[1]
  end
end

local function sorted_names(models: {string: string}): {string}
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
end

local function chosen_model(
  models: {string: string},
  word: string | nil,
  path_word: boolean,
  write_err: function(string)
): Chosen | nil
  if word is string then
    if path_word then
      if not config.has_file(word) then
        write_err(
          one_line("serve: model path " .. word .. " is not a readable file")
        )
        return nil
      end
      return {name = word, path = word}
    end
    local configured = models[word]
    if configured is nil then
      write_err(one_line("serve: unknown model " .. word))
      return nil
    end
    return {name = word, path = configured}
  end

  local names = sorted_names(models)
  if #names == 0 then
    write_err(one_line("serve: no model configured"))
    return nil
  end
  if #names > 1 then
    write_err(
      one_line(
        "serve: no model given; configured models are "
          .. table.concat(names, ", ")
      )
    )
    return nil
  end
  return {name = names[1], path = models[names[1]]}
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

local function serve_code(
  command: targs.Parser,
  write_err: function(string),
  start: function(
    endpoint: serve.Endpoint, write_err: function(string)
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

  local settings = config.load(environment, config_path, path_word, write_err)
  if settings is nil then
    return 1
  end

  local bind_host = given_host or settings.host or "127.0.0.1"
  local bind_port = given_port or settings.port or 8080

  local cache_type: compute.CacheType = "f16"
  local file_cache = settings.kv_cache
  if not (file_cache is nil) then
    cache_type = file_cache
  end

  local chosen = chosen_model(settings.models, word, path_word, write_err)
  if chosen is nil then
    return 1
  end

  return start(
    {
      host = bind_host,
      port = bind_port,
      name = chosen.name,
      path = chosen.path,
      sessions = settings.sessions or 4,
      kv_cache = cache_type,
      context_path = settings.context_path,
      context_bytes = settings.context_bytes,
      workers = settings.workers or 1,
    },
    write_err
  )
end

local function bench_outcome(command: targs.Parser): Outcome
  local entries: {Entry} = {}
  add(entries, "model", model_of(command))
  add(entries, "config", command:get_option("config"))

  for _, name in ipairs({"prefill-tokens", "decode-tokens"}) do
    local count = command:get_option(name)
    if count is number and not (count % 1 == 0 and count >= 1) then
      return {
        code = 1,
        text = "'--"
          .. name
          .. "' expects a positive integer, got "
          .. tostring(count),
        to_err = true,
      }
    end
    add(entries, name, count)
  end

  return {code = 2, text = refusal("bench", entries), to_err = false}
end

function cli.run(
  args: {string},
  write_out: function(string),
  write_err: function(string),
  start?: (function(
    endpoint: serve.Endpoint, write_err: function(string)
  ): integer),
  environment?: {string: string}
): integer
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
  local bench = parser:get_command("bench")

  if serve_command then
    return serve_code(
      serve_command,
      write_err,
      start or serve.start,
      environment or process_environment()
    )
  end

  local outcome: Outcome
  if bench then
    outcome = bench_outcome(bench)
  else
    write_out(one_line(parser:get_help()))
    return 1
  end

  if outcome.to_err then
    write_err(one_line(outcome.text))
  else
    write_out(one_line(outcome.text))
  end
  return outcome.code
end

return cli
