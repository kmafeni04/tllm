local weights = require("tllm.weights")
local tltp = require("tltp")

local record serve
  record Model
    name: string
    path: string
    handle: weights.Handle
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
  end

  model_of: function(tltp.Server): Model | nil
  build: function(Endpoint): tltp.Server | nil, string | nil
  start: function(Endpoint, function(string)): integer
end

local models: {tltp.Server: serve.Model} = {}

function serve.model_of(server: tltp.Server): serve.Model | nil
  return models[server]
end

local function error_body(message: string, kind: string): string
  return tltp.json_encode({error = {message = message, type = kind}}) as string
end

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function install(server: tltp.Server, name: string)
  server:get(
    "/health",
    function(self: tltp.Server): tltp.Response
      return self:json(
        "OK", tltp.json_encode({status = "ok", model = name}) as string
      )
    end
  )

  server.handle_404 = function(self: tltp.Server): tltp.Response
    return self:json("NotFound", error_body("not found", "not_found"))
  end

  server.default_route = function(self: tltp.Server): tltp.Response
    return self:handle_404()
  end

  server.on_error = function(
    self: tltp.Server, _err: string, _trace?: string
  ): tltp.Response
    return self:json(
      "InternalServerError", error_body("internal error", "internal_error")
    )
  end
end

function serve.build(endpoint: serve.Endpoint): tltp.Server | nil, string | nil
  local handle, failure = weights.load(endpoint.path)
  if handle is nil then
    return nil, failure
  end

  local server = tltp.Server.new(
    {port = endpoint.port, bind_host = endpoint.host, logging = false}
  )
  models[server] = {
    name = endpoint.name,
    path = endpoint.path,
    handle = handle,
  }
  install(server, endpoint.name)
  return server, nil
end

function serve.start(
  endpoint: serve.Endpoint, write_err: function(string)
): integer
  local server, failure = serve.build(endpoint)
  if server is nil then
    write_err(one_line(failure as string))
    return 1
  end

  local raised: string | nil
  local ok = xpcall(
    function() server:serve() end,
    function(reason: string) raised = tostring(reason) end
  )
  if not ok then
    write_err(one_line(raised or "server failed"))
    return 1
  end
  return 0
end

return serve
