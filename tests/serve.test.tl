local test = require("lde-test")
local serve = require("tllm.serve")
local tltp = require("tltp")

local record HealthBody
  status: string
end

local record ErrorDetail
  message: string
  type: string
end

local record ErrorBody
  error: ErrorDetail
end

local function build(): tltp.Server
  return serve.build({host = "127.0.0.1", port = 9090})
end

test.it(
  "builds a server for the given host and port",
  function()
    local server = build()
    test.equal(server.config.port, 9090)
    test.equal(server.config.bind_host, "127.0.0.1")
    local response = server:mock_request("/health")
    test.equal(response.status, "OK")
  end
)

test.it(
  "health answers json",
  function()
    local response = build():mock_request("/health")
    test.equal(response.status, "OK")
    test.equal(response.content_type, "application/json")
    test.deepEqual(
      tltp.json_decode(response.body) as HealthBody, {status = "ok"}
    )
  end
)

test.it(
  "unrouted paths answer json 404",
  function()
    local server = build()
    for _, path in ipairs({"/nope", "/nope/"}) do
      local response = server:mock_request(path)
      test.equal(response.status, "NotFound", path)
      test.equal(response.content_type, "application/json", path)
      test.deepEqual(
        tltp.json_decode(response.body) as ErrorBody,
        {error = {message = "not found", type = "not_found"}},
        path
      )
    end
  end
)

test.it(
  "wrong method answers json 404",
  function()
    local response = build():mock_request("/health", {method = "post"})
    test.equal(response.status, "NotFound")
    test.deepEqual(
      tltp.json_decode(response.body) as ErrorBody,
      {error = {message = "not found", type = "not_found"}}
    )
  end
)

test.it(
  "route error answers json 500",
  function()
    local server = build()
    server:get("/boom", function(): tltp.Response error("boom") end)
    local response = server:mock_request("/boom")
    test.equal(response.status, "InternalServerError")
    test.equal(response.content_type, "application/json")
    test.deepEqual(
      tltp.json_decode(response.body) as ErrorBody,
      {error = {message = "internal error", type = "internal_error"}}
    )
  end
)
