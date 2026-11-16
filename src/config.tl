local fs = require("fs")
local compute = require("tllm.compute")
local tltp = require("tltp")

local config = {}

local STARTER = '{"host":"127.0.0.1","port":8080,"models":{}}'

local record ConfigObject
end

local type ConfigValue = string | number | boolean | nil | ConfigObject
local type Fields = {string: ConfigValue}

local record Settings
  host: string | nil
  port: integer | nil
  models: {string: string}
  sessions: integer | nil
  kv_cache: compute.CacheType | nil
  context_path: string | nil
  context_bytes: integer | nil
  context_off: boolean
  context_set: boolean
end

local DEFAULT_CONTEXT_BYTES: integer = 1073741824

local CACHE_TYPES: {string: compute.CacheType} = {
  f32 = "f32",
  f16 = "f16",
  q8_0 = "q8_0",
  q4_0 = "q4_0",
}

local function one_line(text: string): string
  return text .. "\n"
end

local function refuse(write_err: function(string), text: string)
  write_err(one_line(text))
end

function config.valid_host(host: string): boolean
  if not host:match("^%d+%.%d+%.%d+%.%d+$") then
    return false
  end
  for part in host:gmatch("%d+") do
    if #part > 1 and part:sub(1, 1) == "0" then
      return false
    end
    local value = tonumber(part) as number
    if value > 255 then
      return false
    end
  end
  return true
end

function config.is_model_path(word: string): boolean
  return word:sub(-5) == ".gguf" or word:find("/", 1, true) ~= nil
end

function config.has_file(file_path: string): boolean
  return fs.isfile(file_path)
end

local function lookup_paths(environment: {string: string}): {string}
  local paths: {string} = {}
  local xdg = environment.XDG_CONFIG_HOME
  if xdg is string and xdg ~= "" then
    paths[#paths + 1] = xdg .. "/tllm/config.json"
  end
  local home = environment.HOME
  if home is string and home ~= "" then
    paths[#paths + 1] = home .. "/.config/tllm/config.json"
  end
  return paths
end

local function read_file(file_path: string): string | nil
  if not fs.isfile(file_path) then
    return nil
  end
  return fs.read(file_path)
end

local function unknown_key(fields: Fields): string | nil
  local allowed: {string: boolean} = {
    host = true,
    port = true,
    models = true,
    sessions = true,
    kv_cache = true,
    context = true,
  }
  local keys: {string} = {}
  for key in pairs(fields) do
    local name = tostring(key)
    if not allowed[name] then
      keys[#keys + 1] = name
    end
  end
  table.sort(keys)
  return keys[1]
end

local function model_names(fields: Fields): {string}
  local names: {string} = {}
  for key in pairs(fields) do
    names[#names + 1] = tostring(key)
  end
  table.sort(names)
  return names
end

local function parse(
  file_path: string, text: string, write_err: function(string)
): Settings | nil
  local decoded = tltp.json_decode(text) as ConfigValue
  if not (decoded is ConfigObject) then
    refuse(write_err, "config " .. file_path .. ": not a json object")
    return nil
  end

  local fields = decoded as Fields
  local unknown = unknown_key(fields)
  if unknown is string then
    refuse(write_err, "config " .. file_path .. ": unknown key " .. unknown)
    return nil
  end

  local settings: Settings = {
    host = nil,
    port = nil,
    models = {},
    sessions = nil,
    kv_cache = nil,
    context_path = nil,
    context_bytes = nil,
    context_off = false,
    context_set = false,
  }

  local host = fields.host
  if host is string then
    if not config.valid_host(host) then
      refuse(
        write_err,
        "config " .. file_path .. ": host must be a dotted IPv4 address"
      )
      return nil
    end
    settings.host = host
  elseif not (host is nil) then
    refuse(
      write_err,
      "config " .. file_path .. ": host must be a dotted IPv4 address"
    )
    return nil
  end

  local port = fields.port
  if port is number then
    if not (port % 1 == 0 and port >= 1 and port <= 65535) then
      refuse(
        write_err,
        "config " .. file_path .. ": port must be an integer from 1 to 65535"
      )
      return nil
    end
    settings.port = port as integer
  elseif not (port is nil) then
    refuse(
      write_err,
      "config " .. file_path .. ": port must be an integer from 1 to 65535"
    )
    return nil
  end

  local models_value = fields.models
  if models_value is ConfigObject then
    local entries = models_value as Fields
    for _, name in ipairs(model_names(entries)) do
      local entry = entries[name]
      if not (entry is ConfigObject) then
        refuse(
          write_err,
          "config "
            .. file_path
            .. ": model "
            .. name
            .. " must be an object holding only a path string"
        )
        return nil
      end
      local entry_fields = entry as Fields
      local has_extra = false
      for key in pairs(entry_fields) do
        if key ~= "path" then
          has_extra = true
          break
        end
      end
      local model_path = entry_fields.path
      if not (model_path is string) or has_extra then
        refuse(
          write_err,
          "config "
            .. file_path
            .. ": model "
            .. name
            .. " must be an object holding only a path string"
        )
        return nil
      end
      settings.models[name] = model_path
    end
  elseif not (models_value is nil) then
    refuse(
      write_err,
      "config "
        .. file_path
        .. ": models must be an object of entries holding a path string"
    )
    return nil
  end

  local sessions = fields.sessions
  if sessions is number then
    if not (sessions % 1 == 0 and sessions >= 1) then
      refuse(
        write_err,
        "config " .. file_path .. ": sessions must be an integer of one or more"
      )
      return nil
    end
    settings.sessions = sessions as integer
  elseif not (sessions is nil) then
    refuse(
      write_err,
      "config " .. file_path .. ": sessions must be an integer of one or more"
    )
    return nil
  end

  local kv_cache = fields.kv_cache
  if kv_cache is string then
    local held = CACHE_TYPES[kv_cache]
    if held is nil then
      refuse(
        write_err,
        "config " .. file_path .. ": kv_cache must be f32, f16, q8_0 or q4_0"
      )
      return nil
    end
    settings.kv_cache = held
  elseif not (kv_cache is nil) then
    refuse(
      write_err,
      "config " .. file_path .. ": kv_cache must be f32, f16, q8_0 or q4_0"
    )
    return nil
  end

  local function bad_context(): nil
    refuse(
      write_err,
      "config "
        .. file_path
        .. ": context must be false or an object holding a path string"
        .. " and a bytes integer"
    )
    return nil
  end

  local context = fields.context
  if context is boolean then
    if context then
      return bad_context()
    end
    settings.context_off = true
  elseif context is ConfigObject then
    local entry = context as Fields
    local context_path = entry.path
    local context_bytes = entry.bytes
    local extra = false
    for key in pairs(entry) do
      if key ~= "path" and key ~= "bytes" then
        extra = true
        break
      end
    end
    if
      extra
      or not (context_path is string)
      or #(context_path as string) == 0
      or not (context_bytes is number)
      or context_bytes % 1 ~= 0
      or context_bytes < 1
    then
      return bad_context()
    end
    settings.context_set = true
    settings.context_path = context_path as string
    settings.context_bytes = context_bytes as integer
  elseif not (context is nil) then
    return bad_context()
  end

  local names: {string} = {}
  for name in pairs(settings.models) do
    names[#names + 1] = name
  end
  table.sort(names)
  for _, name in ipairs(names) do
    local model_path = settings.models[name]
    if not fs.isfile(model_path) then
      refuse(
        write_err,
        "config "
          .. file_path
          .. ": model "
          .. name
          .. ": "
          .. model_path
          .. " is not a readable file"
      )
      return nil
    end
  end

  return settings
end

local function write_starter(file_path: string, write_err: function(string))
  local parent = file_path:match("^(.*)/") or "."
  fs.mkdirAll(parent)
  if not fs.write(file_path, STARTER) then
    refuse(write_err, "config: cannot write " .. file_path)
    return
  end
  refuse(write_err, "config: wrote a starter file at " .. file_path)
end

local function context_of(
  settings: Settings | nil, environment: {string: string}
): string | nil, integer | nil
  if settings is Settings then
    if settings.context_off then
      return nil, nil
    end
    if settings.context_set then
      return settings.context_path, settings.context_bytes
    end
  end
  local data = environment.XDG_DATA_HOME
  if data is string and data ~= "" then
    return data .. "/tllm/contexts", DEFAULT_CONTEXT_BYTES
  end
  local home = environment.HOME
  if home is string and home ~= "" then
    return home .. "/.local/share/tllm/contexts", DEFAULT_CONTEXT_BYTES
  end
  return nil, nil
end

local function loaded(
  settings: Settings | nil, environment: {string: string}
): boolean, string | nil, integer | nil, {string: string}, integer | nil, compute.CacheType | nil, string | nil, integer | nil
  local context_path, context_bytes = context_of(settings, environment)
  if settings is Settings then
    return true,
      settings.host,
      settings.port,
      settings.models,
      settings.sessions,
      settings.kv_cache,
      context_path,
      context_bytes
  end
  return false, nil, nil, {}, nil, nil, context_path, context_bytes
end

function config.load(
  environment: {string: string},
  explicit: string | nil,
  path_word: boolean,
  write_err: function(string)
): boolean, string | nil, integer | nil, {string: string}, integer | nil, compute.CacheType | nil, string | nil, integer | nil
  if explicit is string then
    local text = read_file(explicit)
    if text is nil then
      refuse(write_err, "config " .. explicit .. ": cannot be read")
      return loaded(nil, environment)
    end
    return loaded(parse(explicit, text, write_err), environment)
  end

  local paths = lookup_paths(environment)
  local chosen: string | nil = nil
  for _, candidate in ipairs(paths) do
    if fs.exists(candidate) then
      chosen = candidate
      break
    end
  end

  if chosen is string then
    local text = read_file(chosen)
    if text is nil then
      refuse(write_err, "config " .. chosen .. ": cannot be read")
      return loaded(nil, environment)
    end
    return loaded(parse(chosen, text, write_err), environment)
  end

  if path_word then
    local context_path, context_bytes = context_of(nil, environment)
    return true, nil, nil, {}, nil, nil, context_path, context_bytes
  end

  if #paths == 0 then
    refuse(
      write_err, "config: no config directory; set XDG_CONFIG_HOME or HOME"
    )
    return loaded(nil, environment)
  end

  write_starter(paths[1], write_err)
  return loaded(nil, environment)
end

return config
