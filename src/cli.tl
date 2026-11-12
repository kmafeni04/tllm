local targs = require("targs")

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

  local serve = parser:command(
    {
      name = "serve",
      description = "Serve a model",
    }
  )
  serve:argument({name = "model", max_length = 1})
  serve:option({name = "config", placeholder = "FILE"})
  serve:option({name = "host", placeholder = "HOST"})
  serve:option(
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
    entries[#entries + 1] = {label = label, text = tostring(value)}
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

local function serve_outcome(command: targs.Parser): Outcome
  local entries: {Entry} = {}
  add(entries, "model", model_of(command))
  add(entries, "config", command:get_option("config"))
  add(entries, "host", command:get_option("host"))

  local port = command:get_option("port")
  if port is number and not (port % 1 == 0 and port >= 1 and port <= 65535) then
    return {
      code = 1,
      text = "'--port' expects an integer from 1 to 65535, got "
        .. tostring(port),
      to_err = true,
    }
  end
  add(entries, "port", port)

  return {code = 2, text = refusal("serve", entries), to_err = false}
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
  args: {string}, write_out: function(string), write_err: function(string)
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

  local serve = parser:get_command("serve")
  local bench = parser:get_command("bench")

  local outcome: Outcome
  if serve then
    outcome = serve_outcome(serve)
  elseif bench then
    outcome = bench_outcome(bench)
  else
    write_err(one_line("no command given; expected 'serve' or 'bench'"))
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
