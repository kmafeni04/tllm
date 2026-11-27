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
    {
      args = args,
      write_out = function(text: string) out[#out + 1] = text end,
      write_err = function(text: string) err[#err + 1] = text end,
      start = start,
    }
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
      {"bench", "qwen3.8-27b", "--config", "/nonexistent/config.json"}
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
    test.equal(refusal_code, 1)
    test.equal(#refusal_out, 0)
    test.equal(#refusal_err, 1)
    test.equal(stream_out, "")
    test.equal(stream_err, "")
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
      {
        args = {"serve", "--host", "127.0.0.1", "--port", "9090"},
        write_out = write_out,
        write_err = write_err,
        start = stub,
        environment = environment,
      }
    )

    test.equal(code, 7)
    test.equal(#out, 0)
    test.equal(#err, 0)
    local endpoint = received as serve.Endpoint
    test.equal(endpoint.host, "127.0.0.1")
    test.equal(endpoint.port, 9090)
    test.equal(endpoint.name, "m")
    test.equal(received_err, write_err)

    cli.run(
      {
        args = {"serve", "m"},
        write_out = write_out,
        write_err = write_err,
        start = stub,
        environment = environment,
      }
    )

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

test.it(
  "every command accepts its options",
  function()
    local config_path = os.tmpname()
    os.remove(config_path)

    local started = false
    local function stub(
      _endpoint: serve.Endpoint, _write_err: function(string)
    ): integer
      started = true
      return 0
    end

    local serve_code, serve_out, serve_err = run_with(
      {
        "serve",
        "m",
        "--config",
        config_path,
        "--host",
        "127.0.0.1",
        "--port",
        "9090",
      },
      stub
    )
    test.equal(serve_code, 1)
    test.equal(#serve_out, 0)
    test.equal(#serve_err, 1)
    test.includes(serve_err[1], config_path)
    test.falsy(started)

    local bench_code, bench_out, bench_err = run_with(
      {
        "bench",
        "m",
        "--config",
        config_path,
        "--prefill-tokens",
        "8",
        "--decode-tokens",
        "8",
        "--dump-routing",
        config_path .. ".routing",
      }
    )
    test.equal(bench_code, 1)
    test.equal(#bench_out, 0)
    test.equal(#bench_err, 1)
    test.includes(bench_err[1], config_path)
  end
)

local record Run
  code: integer
  out: {string}
  err: {string}
  endpoint: serve.Endpoint | nil
  starts: integer
end

local function run(args: {string}, environment: {string: string}): Run
  local result: Run = {code = 0, out = {}, err = {}, starts = 0}
  local function stub(
    endpoint: serve.Endpoint, _write_err: function(string)
  ): integer
    result.endpoint = endpoint
    result.starts = result.starts + 1
    return 0
  end
  result.code = cli.run(
    {
      args = args,
      write_out = function(text: string)
        result.out[#result.out + 1] = text
      end,
      write_err = function(text: string)
        result.err[#result.err + 1] = text
      end,
      start = stub,
      environment = environment,
    }
  )
  return result
end

local function model_word(): string, string
  local dir = os.tmpname()
  os.remove(dir)
  fs.mkdirAll(dir)
  local model = dir .. "/m.gguf"
  fs.write(model, "model")
  return dir, model
end

test.it(
  "a flag on a path word reaches the endpoint and the keys it does not name keep their defaults",
  function()
    local dir, model = model_word()

    local cache = run({"serve", model, "--kv-cache", "q8_0"}, {})
    test.equal(cache.code, 0, table.concat(cache.err, "\n"))
    local endpoint = cache.endpoint as serve.Endpoint
    test.equal(endpoint.kv_cache, "q8_0")
    test.equal(endpoint.ubatch, 512)
    test.equal(endpoint.vram, nil)
    test.equal(endpoint.profile, nil)
    test.equal(endpoint.prefetch, nil)

    local all = run(
      {
        "serve",
        model,
        "--ubatch",
        "8",
        "--vram",
        "4096",
        "--profile",
        "P",
        "--prefetch",
      },
      {}
    )
    test.equal(all.code, 0, table.concat(all.err, "\n"))
    endpoint = all.endpoint as serve.Endpoint
    test.equal(endpoint.ubatch, 8)
    test.equal(endpoint.vram, 4096)
    test.equal(endpoint.profile, "P")
    test.equal(endpoint.prefetch, true)
    test.equal(endpoint.kv_cache, "f16")

    local file_path = dir .. "/prefetch.json"
    fs.write(file_path, '{"prefetch":true,"models":{}}')
    local off = run(
      {"serve", model, "--config", file_path, "--no-prefetch"}, {}
    )
    test.equal(off.code, 0, table.concat(off.err, "\n"))
    test.equal((off.endpoint as serve.Endpoint).prefetch, false)

    fs.rmdir(dir)
  end
)

test.it(
  "a flag beats an entry that names the key",
  function()
    local dir, model = model_word()
    local file_path = dir .. "/entry.json"
    fs.write(
      file_path,
      '{"models":{"m":{"path":"'
        .. model
        .. '","kv_cache":"q4_0","ubatch":64,"vram":2048,"profile":"E",'
        .. '"prefetch":true}}}'
    )

    local cache = run(
      {"serve", "m", "--config", file_path, "--kv-cache", "f32"}, {}
    )
    test.equal(cache.code, 0, table.concat(cache.err, "\n"))
    local endpoint = cache.endpoint as serve.Endpoint
    test.equal(endpoint.kv_cache, "f32")
    test.equal(endpoint.ubatch, 64)
    test.equal(endpoint.vram, 2048)
    test.equal(endpoint.profile, "E")
    test.equal(endpoint.prefetch, true)

    local rest = run(
      {
        "serve",
        "m",
        "--config",
        file_path,
        "--ubatch",
        "4",
        "--vram",
        "1",
        "--profile",
        "Q",
        "--no-prefetch",
      },
      {}
    )
    test.equal(rest.code, 0, table.concat(rest.err, "\n"))
    endpoint = rest.endpoint as serve.Endpoint
    test.equal(endpoint.ubatch, 4)
    test.equal(endpoint.vram, 1)
    test.equal(endpoint.profile, "Q")
    test.equal(endpoint.prefetch, false)
    test.equal(endpoint.kv_cache, "q4_0")

    fs.rmdir(dir)
  end
)

test.it(
  "a flag value that breaks its key's rule is refused naming the flag",
  function()
    local dir, model = model_word()
    local cases: {{string}} = {
      {"--kv-cache", "q8_0x"},
      {"--kv-cache", "F16"},
      {"--ubatch", "0"},
      {"--ubatch", "abc"},
      {"--vram", "0"},
      {"--prefetch", "--no-prefetch"},
    }
    for _, flags in ipairs(cases) do
      local args: {string} = {"serve", model}
      for _, flag in ipairs(flags) do
        args[#args + 1] = flag
      end
      local got = run(args, {})
      local given = table.concat(args, " ")
      test.equal(got.code, 1, given)
      test.equal(#got.out, 0, given)
      test.equal(#got.err, 1, given)
      test.equal(got.starts, 0, given)
      for _, flag in ipairs(flags) do
        test.includes(got.err[1], flag, given)
      end
    end

    local cache = run(
      {"serve", model, "--kv-cache", "q8_0x", "--ubatch", "0"}, {}
    )
    test.equal(#cache.err, 1)
    test.includes(cache.err[1], "--kv-cache")
    test.falsy(cache.err[1]:find("--ubatch", 1, true), cache.err[1])

    local room = run(
      {"serve", model, "--vram", "0", "--prefetch", "--no-prefetch"}, {}
    )
    test.equal(#room.err, 1)
    test.includes(room.err[1], "--vram")
    test.falsy(room.err[1]:find("--prefetch", 1, true), room.err[1])

    local port = run({"serve", model, "--port", "0", "--ubatch", "0"}, {})
    test.equal(#port.err, 1)
    test.includes(port.err[1], "--port")
    test.falsy(port.err[1]:find("--ubatch", 1, true), port.err[1])

    fs.rmdir(dir)
  end
)

test.it(
  "the flag fault is reported before the model word and the file",
  function()
    local dir, model = model_word()

    local unknown = run({"serve", "bogus", "--ubatch", "0"}, {})
    test.equal(unknown.code, 1)
    test.equal(#unknown.err, 1)
    test.includes(unknown.err[1], "--ubatch")
    test.falsy(unknown.err[1]:find("unknown model", 1, true), unknown.err[1])

    local file_path = dir .. "/both.json"
    fs.write(
      file_path,
      '{"kv_cache":"f16","models":{"m":{"path":"'
        .. model
        .. '","kv_cache":"q8_0"}}}'
    )
    local at_both = run(
      {"serve", "m", "--config", file_path, "--kv-cache", "q4x"}, {}
    )
    test.equal(at_both.code, 1)
    test.equal(#at_both.err, 1)
    test.includes(at_both.err[1], "--kv-cache")
    test.falsy(at_both.err[1]:find("top level", 1, true), at_both.err[1])

    fs.rmdir(dir)
  end
)
