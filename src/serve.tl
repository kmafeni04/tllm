local tltp = require("tltp")

local record serve
  record Endpoint
    host: string
    port: integer
  end

  build: function(Endpoint): tltp.Server
  start: function(Endpoint, function(string)): integer
end

local function error_body(message: string, kind: string): string
  return tltp.json_encode({error = {message = message, type = kind}}) as string
end

local function install(server: tltp.Server)
  server:get(
    "/health",
    function(self: tltp.Server): tltp.Response
      return self:json("OK", tltp.json_encode({status = "ok"}) as string)
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

function serve.build(endpoint: serve.Endpoint): tltp.Server
  local server = tltp.Server.new(
    {port = endpoint.port, bind_host = endpoint.host, logging = false}
  )
  install(server)
  return server
end

function serve.start(
  endpoint: serve.Endpoint, write_err: function(string)
): integer
  local server = serve.build(endpoint)
  local failure: string | nil
  local ok = xpcall(
    function() server:serve() end,
    function(reason: string) failure = tostring(reason) end
  )
  if not ok then
    write_err((failure or "server failed") .. "\n")
    return 1
  end
  return 0
end

return serve
