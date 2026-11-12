local test = require("lde-test")
local cli = require("tllm.cli")

local function run_with(args: {string}): integer, {string}, {string}
  local out: {string} = {}
  local err: {string} = {}
  local code = cli.run(
    args,
    function(text: string) out[#out + 1] = text end,
    function(text: string) err[#err + 1] = text end
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
    local refusal_code, refusal_out, refusal_err = run_with({"serve"})

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
  "serve echoes the values it was given",
  function()
    local code, out, err = run_with(
      {
        "serve",
        "qwen3.8-27b",
        "--config",
        "cfg.json",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
      }
    )

    test.equal(code, 2)
    test.equal(#err, 0)
    test.equal(#out, 1)
    test.equal(
      out[1],
      "serve: not implemented (model qwen3.8-27b, config cfg.json, host 127.0.0.1, port 8080)\n"
    )
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
  "refusal line without values",
  function()
    local code, out, err = run_with({"serve"})

    test.equal(code, 2)
    test.equal(#err, 0)
    test.equal(#out, 1)
    test.equal(out[1], "serve: not implemented\n")
  end
)

test.it(
  "missing and unknown commands are usage errors",
  function()
    local missing_code, missing_out, missing_err = run_with({})
    test.equal(missing_code, 1)
    test.equal(#missing_out, 0)
    test.equal(#missing_err, 1)

    local unknown_code, unknown_out, unknown_err = run_with({"wat"})
    test.equal(unknown_code, 1)
    test.equal(#unknown_out, 0)
    test.equal(#unknown_err, 1)
  end
)

test.it(
  "bad arguments exit one",
  function()
    local cases: {{string}} = {
      {"bench"},
      {"serve", "--port", "0"},
      {"bench", "qwen3.8-27b", "--prefill-tokens", "0"},
    }

    for _, args in ipairs(cases) do
      local code, out, err = run_with(args)
      test.equal(code, 1, table.concat(args, " "))
      test.equal(#out, 0, table.concat(args, " "))
      test.equal(#err, 1, table.concat(args, " "))
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
