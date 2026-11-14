local fs = require("fs")
local test = require("lde-test")
local cli = require("tllm.cli")
local serve = require("tllm.serve")

local function run_with(
  args: {string}, start?: function(serve.Endpoint, function(string)): integer
): integer, {string}, {string}
  local out: {string} = {}
  local err: {string} = {}
  local code = cli.run(
    args,
    function(text: string) out[#out + 1] = text end,
    function(text: string) err[#err + 1] = text end,
    start
  )
  return code, out, err
end

test.it(
  "dispatcher returns a code through supplied writers",
  function()
    local real_stdout = io.stdout
    local real_stderr = io.stderr
    local real_output = io.output()
    local out_scratch = io.tmpfile()
    local err_scratch = io.tmpfile()

    io.stdout = out_scratch
    io.stderr = err_scratch
    io.output(out_scratch)

    local failure_code, failure_out, failure_err = run_with({"bogus"})
    local refusal_code, refusal_out, refusal_err = run_with(
      {"bench", "qwen3.8-27b"}
    )

    io.stdout = real_stdout
    io.stderr = real_stderr
    io.output(real_output)

    out_scratch:seek("set")
    err_scratch:seek("set")
    local stream_out = out_scratch:read("*a")
    local stream_err = err_scratch:read("*a")
    out_scratch:close()
    err_scratch:close()

    test.equal(failure_code, 1)
    test.equal(#failure_out, 0)
    test.equal(#failure_err, 1)
    test.equal(refusal_code, 2)
    test.equal(#refusal_out, 1)
    test.equal(#refusal_err, 0)
    test.equal(stream_out, "")
    test.equal(stream_err, "")
  end
)

test.it(
  "bench echoes the values it was given",
  function()
    local code, out, err = run_with(
      {
        "bench",
        "qwen3.8-27b",
        "--config",
        "cfg.json",
        "--prefill-tokens",
        "512",
        "--decode-tokens",
        "128",
      }
    )

    test.equal(code, 2)
    test.equal(#err, 0)
    test.equal(#out, 1)
    test.equal(
      out[1],
      "bench: not implemented (model qwen3.8-27b, config cfg.json, prefill-tokens 512, decode-tokens 128)\n"
    )
  end
)

test.it(
  "a bare invocation prints help and exits one",
  function()
    local code, out, err = run_with({})
    test.equal(code, 1)
    test.equal(#err, 0)
    test.includes(table.concat(out), "Usage: tllm")
  end
)

test.it(
  "an unknown command is a usage error",
  function()
    local code, out, err = run_with({"wat"})
    test.equal(code, 1)
    test.equal(#out, 0)
    test.equal(#err, 1)
  end
)

test.it(
  "bad arguments exit one",
  function()
    local cases: {{string}} = {
      {"bench"},
      {"bench", "qwen3.8-27b", "--prefill-tokens", "0"},
    }

    for _, args in ipairs(cases) do
      local code, out, err = run_with(args)
      test.equal(code, 1, table.concat(args, " "))
      test.equal(#out, 0, table.concat(args, " "))
      test.equal(#err, 1, table.concat(args, " "))
    end

    local code, out, err = run_with(
      {"serve", "--port", "0"}, function(): integer return 99 end
    )
    test.equal(code, 1, "serve --port 0")
    test.equal(#out, 0, "serve --port 0")
    test.equal(#err, 1, "serve --port 0")
  end
)

test.it(
  "serve starts with the given host and port",
  function()
    local out: {string} = {}
    local err: {string} = {}
    local received: serve.Endpoint | nil
    local received_err: function(string) | nil

    local function stub(
      endpoint: serve.Endpoint, write_err: function(string)
    ): integer
      received = endpoint
      received_err = write_err
      return 7
    end

    local function write_out(text: string)
      out[#out + 1] = text
    end

    local function write_err(text: string)
      err[#err + 1] = text
    end

    local base = os.tmpname()
    os.remove(base)
    fs.mkdirAll(base .. "/tllm")
    local model = base .. "/m.gguf"
    fs.write(model, "model")
    fs.write(
      base .. "/tllm/config.json", '{"models":{"m":{"path":"' .. model .. '"}}}'
    )
    local environment = {XDG_CONFIG_HOME = base}

    local code = cli.run(
      {"serve", "--host", "127.0.0.1", "--port", "9090"},
      write_out,
      write_err,
      stub,
      environment
    )

    test.equal(code, 7)
    test.equal(#out, 0)
    test.equal(#err, 0)
    local endpoint = received as serve.Endpoint
    test.equal(endpoint.host, "127.0.0.1")
    test.equal(endpoint.port, 9090)
    test.equal(endpoint.name, "m")
    test.equal(received_err, write_err)

    cli.run({"serve", "m"}, write_out, write_err, stub, environment)

    endpoint = received as serve.Endpoint
    test.equal(endpoint.host, "127.0.0.1")
    test.equal(endpoint.port, 8080)
    test.equal(endpoint.name, "m")

    fs.rmdir(base)
  end
)

test.it(
  "bad host is a usage error",
  function()
    for _, host in ipairs({"localhost", "127.0.0.999", "127.0.0.01"}) do
      local called = false
      local code, out, err = run_with(
        {"serve", "--host", host},
        function(): integer called = true; return 0 end
      )

      test.equal(code, 1, host)
      test.equal(#out, 0, host)
      test.equal(#err, 1, host)
      test.falsy(called, host)
    end
  end
)

test.it(
  "help returns zero",
  function()
    local code, out, err = run_with({"--help"})
    test.equal(code, 0)
    test.equal(#err, 0)
    test.includes(table.concat(out), "Usage: tllm")

    local serve_code, serve_out, serve_err = run_with({"serve", "--help"})
    test.equal(serve_code, 0)
    test.equal(#serve_err, 0)
    test.includes(table.concat(serve_out), "Usage: tllm serve")
  end
)
