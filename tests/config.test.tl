local fs = require("fs")
local test = require("lde-test")
local cli = require("tllm.cli")
local serve = require("tllm.serve")

local STARTER = '{"host":"127.0.0.1","port":8080,"models":{}}'

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

local function temp_dir(): string
  local base = os.tmpname()
  os.remove(base)
  fs.mkdirAll(base)
  return base
end

local function cleanup(dirs: {string})
  for _, dir in ipairs(dirs) do
    fs.rmdir(dir)
  end
end

local function model_file(dir: string): string
  local file_path = dir .. "/m.gguf"
  fs.write(file_path, "model")
  return file_path
end

local function config_text(model_path: string, extras: string): string
  local models = '"models":{"m":{"path":"' .. model_path .. '"}}'
  if extras == "" then
    return "{" .. models .. "}"
  end
  return "{" .. extras .. "," .. models .. "}"
end

local function write_config(xdg: string, text: string): string
  local file_path = xdg .. "/tllm/config.json"
  fs.mkdirAll(xdg .. "/tllm")
  fs.write(file_path, text)
  return file_path
end

test.it(
  "config lookup reads xdg then home, --config replaces it",
  function()
    local xdg = temp_dir()
    local home = temp_dir()
    local third = temp_dir()
    local model = model_file(third)
    local xdg_config = write_config(xdg, config_text(model, '"port":9001'))
    fs.mkdirAll(home .. "/.config/tllm")
    local home_config = home .. "/.config/tllm/config.json"
    fs.write(home_config, config_text(model, '"port":9002'))
    local environment = {XDG_CONFIG_HOME = xdg, HOME = home}

    local found = run({"serve", "m"}, environment)
    test.equal(found.code, 0)
    test.equal((found.endpoint as serve.Endpoint).port, 9001)

    fs.delete(xdg_config)
    local fallback = run({"serve", "m"}, environment)
    test.equal((fallback.endpoint as serve.Endpoint).port, 9002)

    fs.write(xdg_config, config_text(model, '"port":9001'))
    local custom = third .. "/custom.json"
    fs.write(custom, config_text(model, '"port":9003'))
    local replaced = run({"serve", "m", "--config", custom}, environment)
    test.equal((replaced.endpoint as serve.Endpoint).port, 9003)

    cleanup({xdg, home, third})
  end
)

test.it(
  "an unreadable config path is refused",
  function()
    local xdg = temp_dir()
    local home = temp_dir()
    local environment = {XDG_CONFIG_HOME = xdg, HOME = home}

    local missing = xdg .. "/nope.json"
    local absent = run({"serve", "--config", missing}, environment)
    test.equal(absent.code, 1)
    test.equal(#absent.err, 1)
    test.includes(absent.err[1], missing)
    test.equal(absent.starts, 0)
    test.falsy(fs.exists(xdg .. "/tllm/config.json"))
    test.falsy(fs.exists(home .. "/.config/tllm/config.json"))

    local directory = temp_dir()
    local refused = run({"serve", "--config", directory}, environment)
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], directory)
    test.equal(refused.starts, 0)

    cleanup({xdg, home, directory})
  end
)

test.it(
  "first run writes the starter config and exits one",
  function()
    local xdg = temp_dir()
    local home = temp_dir()
    local starter = run({"serve"}, {XDG_CONFIG_HOME = xdg, HOME = home})
    local xdg_config = xdg .. "/tllm/config.json"
    test.equal(starter.code, 1)
    test.equal(starter.starts, 0)
    test.equal(#starter.err, 1)
    test.includes(starter.err[1], xdg_config)
    test.equal(fs.read(xdg_config), STARTER)

    local empty_xdg = temp_dir()
    local home_only = temp_dir()
    local fallback = run({"serve"}, {XDG_CONFIG_HOME = "", HOME = home_only})
    local home_config = home_only .. "/.config/tllm/config.json"
    test.equal(fallback.code, 1)
    test.equal(fallback.starts, 0)
    test.equal(#fallback.err, 1)
    test.includes(fallback.err[1], home_config)
    test.equal(fs.read(home_config), STARTER)

    local blocker = empty_xdg .. "/blocker"
    fs.write(blocker, "not a directory")
    local blocked = run({"serve"}, {XDG_CONFIG_HOME = blocker, HOME = ""})
    test.equal(blocked.code, 1)
    test.equal(#blocked.err, 1)
    test.equal(blocked.starts, 0)

    local no_environment = run({"serve"}, {})
    test.equal(no_environment.code, 1)
    test.equal(#no_environment.err, 1)
    test.equal(no_environment.starts, 0)

    cleanup({xdg, home, empty_xdg, home_only})
  end
)

test.it(
  "text that is not json is refused",
  function()
    local xdg = temp_dir()
    local config = write_config(xdg, "{")
    local refused = run({"serve"}, {XDG_CONFIG_HOME = xdg})
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], config)
    test.equal(refused.starts, 0)
    cleanup({xdg})
  end
)

test.it(
  "an unknown key is refused",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    write_config(
      xdg, '{"port":8080,"prot":1,"models":{"m":{"path":"' .. model .. '"}}}'
    )
    local refused = run({"serve"}, {XDG_CONFIG_HOME = xdg})
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], "prot")
    test.equal(refused.starts, 0)
    cleanup({xdg})
  end
)

test.it(
  "bad config values are refused",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local environment = {XDG_CONFIG_HOME = xdg}

    local function refused(text: string): Run
      write_config(xdg, text)
      return run({"serve"}, environment)
    end

    local function with_model(extras: string): string
      return "{" .. extras .. ',"models":{"alpha":{"path":"' .. model .. '"}}}'
    end

    local values = {
      {with_model('"host":"localhost"'), "host"},
      {with_model('"port":0'), "port"},
      {with_model('"port":"8080"'), "port"},
      {'{"port":8080,"models":"nope"}', "models"},
      {with_model('"sessions":0'), "sessions"},
      {with_model('"sessions":"2"'), "sessions"},
    }
    for _, case in ipairs(values) do
      local result = refused(case[1])
      test.equal(result.code, 1, case[1])
      test.equal(#result.err, 1, case[1])
      test.includes(result.err[1], case[2], case[1])
      test.equal(result.starts, 0, case[1])
    end

    local entries = {
      '{"port":8080,"models":{"alpha":{}}}',
      '{"port":8080,"models":{"alpha":{"path":"' .. model .. '","extra":1}}}',
    }
    for _, text in ipairs(entries) do
      local result = refused(text)
      test.equal(result.code, 1, text)
      test.equal(#result.err, 1, text)
      test.includes(result.err[1], "alpha", text)
      test.equal(result.starts, 0, text)
    end

    local unknown_first = refused('{"prot":1,"port":0,"models":{}}')
    test.includes(unknown_first.err[1], "prot")

    local host_first = refused('{"host":"localhost","port":0,"models":{}}')
    test.includes(host_first.err[1], "host")

    cleanup({xdg})
  end
)

test.it(
  "a configured model with no file is refused",
  function()
    local xdg = temp_dir()
    local environment = {XDG_CONFIG_HOME = xdg}

    local missing = xdg .. "/missing.gguf"
    write_config(xdg, '{"models":{"alpha":{"path":"' .. missing .. '"}}}')
    local absent = run({"serve"}, environment)
    test.equal(absent.code, 1)
    test.equal(#absent.err, 1)
    test.includes(absent.err[1], "alpha")
    test.includes(absent.err[1], missing)
    test.equal(absent.starts, 0)

    local directory = temp_dir()
    write_config(xdg, '{"models":{"alpha":{"path":"' .. directory .. '"}}}')
    local refused = run({"serve"}, environment)
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], "alpha")
    test.includes(refused.err[1], directory)
    test.equal(refused.starts, 0)

    cleanup({xdg, directory})
  end
)

test.it(
  "a model path is served without a config",
  function()
    local xdg = temp_dir()
    local home = temp_dir()
    local environment = {XDG_CONFIG_HOME = xdg, HOME = home}
    local model = model_file(xdg)

    local served = run({"serve", model}, environment)
    test.equal(served.code, 0)
    test.equal(served.starts, 1)
    local endpoint = served.endpoint as serve.Endpoint
    test.equal(endpoint.host, "127.0.0.1")
    test.equal(endpoint.port, 8080)
    test.equal(endpoint.sessions, 4)
    test.falsy(fs.exists(xdg .. "/tllm/config.json"))
    test.falsy(fs.exists(home .. "/.config/tllm/config.json"))

    local missing = xdg .. "/gone.gguf"
    local absent = run({"serve", missing}, environment)
    test.equal(absent.code, 1)
    test.equal(#absent.err, 1)
    test.equal(absent.starts, 0)

    local upper = xdg .. "/up.GGUF"
    fs.write(upper, "model")
    write_config(xdg, config_text(model, ""))
    local named = run({"serve", "up.GGUF"}, environment)
    test.equal(named.code, 1)
    test.equal(#named.err, 1)
    test.equal(named.err[1], "serve: unknown model up.GGUF\n")
    test.equal(named.starts, 0)

    local custom = xdg .. "/custom.json"
    fs.write(custom, config_text(model, ""))
    local explicit = run({"serve", "m", "--config", custom}, {})
    test.equal(explicit.code, 0)
    test.equal(explicit.starts, 1)

    cleanup({xdg, home})
  end
)

test.it(
  "a model name must be configured",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local config = write_config(xdg, config_text(model, ""))
    local environment = {XDG_CONFIG_HOME = xdg}

    local typo = run({"serve", "typo"}, environment)
    test.equal(typo.code, 1)
    test.equal(#typo.err, 1)
    test.includes(typo.err[1], "typo")
    test.equal(typo.starts, 0)

    local resolved = run({"serve", "m"}, environment)
    test.equal(resolved.code, 0)
    test.equal(resolved.starts, 1)

    fs.write(config, '{"models":{}}')
    local empty = run({"serve"}, environment)
    test.equal(empty.code, 1)
    test.equal(#empty.err, 1)
    test.equal(empty.starts, 0)

    cleanup({xdg})
  end
)

test.it(
  "flags beat the file and the file beats the defaults",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local config = write_config(
      xdg, config_text(model, '"host":"127.0.0.2","port":9100,"sessions":2')
    )
    local environment = {XDG_CONFIG_HOME = xdg}

    local from_file = run({"serve", "m"}, environment)
    test.equal(from_file.code, 0)
    local endpoint = from_file.endpoint as serve.Endpoint
    test.equal(endpoint.host, "127.0.0.2")
    test.equal(endpoint.port, 9100)
    test.equal(endpoint.sessions, 2)

    local flag_port = run({"serve", "m", "--port", "9090"}, environment)
    local overridden = flag_port.endpoint as serve.Endpoint
    test.equal(overridden.host, "127.0.0.2")
    test.equal(overridden.port, 9090)

    fs.write(config, config_text(model, ""))
    local defaults = run({"serve", "m"}, environment)
    local fallback = defaults.endpoint as serve.Endpoint
    test.equal(fallback.host, "127.0.0.1")
    test.equal(fallback.port, 8080)
    test.equal(fallback.sessions, 4)

    fs.write(config, config_text(model, '"host":"localhost","port":9100'))
    local bad_host = run({"serve", "m", "--host", "127.0.0.1"}, environment)
    test.equal(bad_host.code, 1)
    test.equal(#bad_host.err, 1)
    test.equal(bad_host.starts, 0)

    cleanup({xdg})
  end
)

test.it(
  "the configured type reaches the server, and one it does not name is refused",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local config = write_config(
      xdg, '{"host":"127.0.0.1","port":8080,"kv_cache":"q8_0","models":{}}'
    )
    local environment = {XDG_CONFIG_HOME = xdg}

    local named = run({"serve", model}, environment)
    test.equal(named.code, 0)
    test.equal((named.endpoint as serve.Endpoint).kv_cache, "q8_0")

    fs.write(config, '{"host":"127.0.0.1","port":8080,"models":{}}')
    local unset = run({"serve", model}, environment)
    test.equal(unset.code, 0)
    test.equal((unset.endpoint as serve.Endpoint).kv_cache, "f16")

    fs.write(
      config, '{"host":"127.0.0.1","port":8080,"kv_cache":"q8_0x","models":{}}'
    )
    local refused = run({"serve", model}, environment)
    test.equal(refused.code, 1)
    test.equal(#refused.err, 1)
    test.includes(refused.err[1], "kv_cache")
    test.equal(refused.starts, 0)

    cleanup({xdg})
  end
)

test.it(
  "the context directory and its budget come from config",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local named_path = xdg .. "/named.json"
    local context = '"context":{"path":"D","bytes":4096}'
    fs.write(named_path, config_text(model, context))
    local plain_path = xdg .. "/plain.json"
    fs.write(plain_path, config_text(model, ""))
    local off_path = xdg .. "/off.json"
    fs.write(off_path, config_text(model, '"context":false'))
    local environment = {XDG_DATA_HOME = "X", HOME = "H"}

    local named = run({"serve", "m", "--config", named_path}, environment)
    test.equal(named.code, 0)
    test.equal(named.starts, 1)
    local named_endpoint = named.endpoint as serve.Endpoint
    test.equal(named_endpoint.context_path, "D")
    test.equal(named_endpoint.context_bytes, 4096)

    local dropped = run({"serve", "m", "--config", named_path}, {})
    test.equal(dropped.code, 0)
    local dropped_endpoint = dropped.endpoint as serve.Endpoint
    test.equal(dropped_endpoint.context_path, "D")
    test.equal(dropped_endpoint.context_bytes, 4096)

    local defaulted = run({"serve", "m", "--config", plain_path}, environment)
    test.equal(defaulted.code, 0)
    local defaulted_endpoint = defaulted.endpoint as serve.Endpoint
    test.equal(defaulted_endpoint.context_path, "X/tllm/contexts")
    test.equal(defaulted_endpoint.context_bytes, 1073741824)

    local homed = run({"serve", "m", "--config", plain_path}, {HOME = "H"})
    test.equal(homed.code, 0)
    local homed_endpoint = homed.endpoint as serve.Endpoint
    test.equal(homed_endpoint.context_path, "H/.local/share/tllm/contexts")
    test.equal(homed_endpoint.context_bytes, 1073741824)

    local nowhere = run({"serve", "m", "--config", plain_path}, {})
    test.equal(nowhere.code, 0)
    local nowhere_endpoint = nowhere.endpoint as serve.Endpoint
    test.equal(nowhere_endpoint.context_path, nil)
    test.equal(nowhere_endpoint.context_bytes, nil)

    local off = run({"serve", "m", "--config", off_path}, environment)
    test.equal(off.code, 0)
    local off_endpoint = off.endpoint as serve.Endpoint
    test.equal(off_endpoint.context_path, nil)
    test.equal(off_endpoint.context_bytes, nil)

    local bad: {string} = {
      "true",
      "1",
      '{"bytes":4096}',
      '{"path":"","bytes":4096}',
      '{"path":"D"}',
      '{"path":"D","bytes":0}',
      '{"path":"D","bytes":"4096"}',
      '{"path":"D","bytes":4096,"extra":1}',
    }
    local bad_path = xdg .. "/bad.json"
    for _, shape in ipairs(bad) do
      fs.write(bad_path, config_text(model, '"context":' .. shape))
      local refused = run({"serve", "m", "--config", bad_path}, environment)
      test.equal(refused.code, 1, shape)
      test.equal(#refused.err, 1, shape)
      test.includes(refused.err[1], "context", shape)
      test.equal(refused.starts, 0, shape)
    end

    local both_path = xdg .. "/both.json"
    fs.write(both_path, config_text(model, '"sessions":0,"context":true'))
    local both = run({"serve", "m", "--config", both_path}, environment)
    test.equal(both.code, 1)
    test.equal(#both.err, 1)
    test.includes(both.err[1], "sessions")
    test.equal(both.starts, 0)

    cleanup({xdg})
  end
)

test.it(
  "the worker count comes from config",
  function()
    local dir = temp_dir()
    local model_path = model_file(dir)
    local with_three = dir .. "/three.json"
    fs.write(
      with_three, '{"workers":3,"models":{"m":{"path":"' .. model_path .. '"}}}'
    )
    local three = run({"serve", "m", "--config", with_three}, {})
    test.equal(three.code, 0)
    test.equal(three.starts, 1)
    test.equal((three.endpoint as serve.Endpoint).workers, 3)

    local without = dir .. "/without.json"
    fs.write(without, config_text(model_path, ""))
    local one = run({"serve", "m", "--config", without}, {})
    test.equal(one.code, 0)
    test.equal(one.starts, 1)
    test.equal((one.endpoint as serve.Endpoint).workers, 1)

    local bad_path = dir .. "/bad-workers.json"
    local bad: {string} = {"0", "-1", "1.5", '"2"', "true"}
    for _, value in ipairs(bad) do
      fs.write(bad_path, config_text(model_path, '"workers":' .. value))
      local refused = run({"serve", "m", "--config", bad_path}, {})
      test.equal(refused.code, 1, value)
      test.equal(#refused.err, 1, value)
      test.includes(refused.err[1], "workers", value)
      test.equal(refused.starts, 0, value)
    end

    cleanup({dir})
  end
)

test.it(
  "ubatch is read from the file and refused when it is not a count",
  function()
    local xdg = temp_dir()
    local model = model_file(xdg)
    local config = write_config(xdg, config_text(model, '"ubatch":3'))
    local environment = {XDG_CONFIG_HOME = xdg}

    local named = run({"serve", "m"}, environment)
    test.equal(named.code, 0)
    test.equal((named.endpoint as serve.Endpoint).ubatch, 3)

    fs.write(config, config_text(model, ""))
    local unset = run({"serve", "m"}, environment)
    test.equal(unset.code, 0)
    test.equal((unset.endpoint as serve.Endpoint).ubatch, 512)

    fs.write(config, config_text(model, '"ubatch":3,"port":9101'))
    local beside = run({"serve", "m"}, environment)
    test.equal(beside.code, 0)
    test.equal(#beside.err, 0)
    test.equal((beside.endpoint as serve.Endpoint).ubatch, 3)

    fs.write(
      config, config_text(model, '"ubatch":0,"context":{"path":"","bytes":0}')
    )
    local first = run({"serve", "m"}, environment)
    test.equal(first.code, 1)
    test.equal(#first.err, 1)
    test.includes(first.err[1], "context")
    test.equal(first.starts, 0)

    for _, given in ipairs({'"8"', "0", "-1", "1.5"}) do
      fs.write(config, config_text(model, '"ubatch":' .. given))
      local refused = run({"serve", "m"}, environment)
      test.equal(refused.code, 1, given)
      test.equal(#refused.err, 1, given)
      test.includes(refused.err[1], "ubatch")
      test.equal(refused.starts, 0, given)
    end

    cleanup({xdg})
  end
)
