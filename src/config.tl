local fs = require("fs")
local compute = require("tllm.compute")
local cores = require("tllm.cores")
local tltp = require("tltp")

local record config
  -- what a models entry holds: its path and the values of its own
  record ModelEntry
    path: string
    kv_cache: compute.CacheType | nil
    ubatch: integer | nil
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
  end

  -- what a start function is given once the file and environment are resolved
  record Config
    host: string | nil
    port: integer | nil
    models: {string: ModelEntry}
    sessions: integer | nil
    kv_cache: compute.CacheType | nil
    context_path: string | nil
    context_bytes: integer | nil
    workers: integer | nil
    ubatch: integer | nil
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
    threads: integer | nil
  end

  -- one level's five model values, each absent when that level names none
  record Level
    kv_cache: compute.CacheType | nil
    ubatch: integer | nil
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
  end

  -- the values a model is served with, once its levels are read
  record ModelValues
    kv_cache: compute.CacheType
    ubatch: integer
    vram: integer | nil
    profile: string | nil
    prefetch: boolean | nil
  end

  -- 53/FR-001: one key's owning level and the levels it may appear at
  record KeyLevel
    wins: string
    top: boolean
    entry: boolean
  end

  valid_host: function(host: string): boolean
  is_model_path: function(word: string): boolean
  has_file: function(file_path: string): boolean
  levels: function(): {string: KeyLevel}
  -- 54/FR-006, 54/FR-007: the rules a value is checked by, at either file
  -- level or on a flag: the cache type for kv_cache, the count for the rest
  cache_type: function(value: string): compute.CacheType | nil
  whole_count: function(value: number): boolean
  model_values: function(
    settings: Config, entry: ModelEntry | nil, flags: Level | nil
  ): ModelValues
  load: function(
    environment: {string: string},
    explicit: string | nil,
    path_word: boolean,
    starter?: boolean
  ): Config | nil, string | nil
end

local type Config = config.Config
local type Level = config.Level
local type ModelEntry = config.ModelEntry
local type ModelValues = config.ModelValues
local type KeyLevel = config.KeyLevel

local STARTER = '{"host":"127.0.0.1","port":8080,"models":{}}'

local record ConfigObject
end

local type ConfigValue = string | number | boolean | nil | ConfigObject
local type Fields = {string: ConfigValue}

local DEFAULT_CONTEXT_BYTES: integer = 1073741824
-- 04/FR-011: the chunk bound a file names none keeps
local DEFAULT_UBATCH: integer = 512

local CACHE_TYPES: {string: compute.CacheType} = {
  f32 = "f32",
  f16 = "f16",
  q8_0 = "q8_0",
  q4_0 = "q4_0",
}

local function fault(text: string): nil, string
  return nil, text
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

-- 53/FR-001: every key the file may hold, the level whose value wins for it,
-- and the levels it may appear at. A key that skips this table fails a test.
local KEYS: {string: KeyLevel} = {
  host = {wins = "top", top = true, entry = false},
  port = {wins = "top", top = true, entry = false},
  models = {wins = "top", top = true, entry = false},
  sessions = {wins = "top", top = true, entry = false},
  context = {wins = "top", top = true, entry = false},
  workers = {wins = "top", top = true, entry = false},
  threads = {wins = "top", top = true, entry = false},
  path = {wins = "entry", top = false, entry = true},
  kv_cache = {wins = "entry", top = true, entry = true},
  ubatch = {wins = "entry", top = true, entry = true},
  vram = {wins = "entry", top = true, entry = true},
  profile = {wins = "entry", top = true, entry = true},
  prefetch = {wins = "entry", top = true, entry = true},
}

-- the keys a models entry owns, in the order their faults are reported
local MODEL_KEYS: {string} = {
  "kv_cache",
  "ubatch",
  "vram",
  "profile",
  "prefetch",
}

local VALUE_FAULT: {string: string} = {
  kv_cache = "kv_cache must be f32, f16, q8_0 or q4_0",
  ubatch = "ubatch must be an integer of one or more",
  vram = "vram must be an integer of one or more",
  profile = "profile must be a string",
  prefetch = "prefetch must be a boolean",
}

-- the entry names of one models table, in ascending byte order
local function model_names(fields: Fields): {string}
  local names: {string} = {}
  for key in pairs(fields) do
    names[#names + 1] = tostring(key)
  end
  table.sort(names)
  return names
end

local function held_by(entries: Fields, key: string): {string}
  local names: {string} = {}
  for _, name in ipairs(model_names(entries)) do
    local entry = entries[name]
    if entry is ConfigObject then
      local value = (entry as Fields)[key]
      if not (value is nil) then
        names[#names + 1] = name
      end
    end
  end
  return names
end

-- 04/FR-006: a top-level name the table does not allow at the top level
local function unknown_key(fields: Fields): string | nil
  local keys: {string} = {}
  for key in pairs(fields) do
    local name = tostring(key)
    local rule = KEYS[name]
    if rule is nil or not rule.top then
      keys[#keys + 1] = name
    end
  end
  table.sort(keys)
  return keys[1]
end

-- 53/FR-006: a model key named at both levels of one file
local function both_levels(fields: Fields, entries: Fields): string | nil
  for _, key in ipairs(MODEL_KEYS) do
    local given = fields[key]
    if not (given is nil) then
      local names = held_by(entries, key)
      if #names > 0 then
        return key
          .. " is set at the top level and in models "
          .. table.concat(names, ", ")
      end
    end
  end
  return nil
end

-- 53/FR-005: a name in an entry that the table does not allow in an entry
local function entry_key(entries: Fields): string | nil
  local holders: {string: {string}} = {}
  for _, name in ipairs(model_names(entries)) do
    local entry = entries[name]
    if entry is ConfigObject then
      for key in pairs(entry as Fields) do
        local name_key = tostring(key)
        local rule = KEYS[name_key]
        if rule is nil or not rule.entry then
          local list = holders[name_key]
          if list is nil then
            list = {}
            holders[name_key] = list
          end
          list[#list + 1] = name
        end
      end
    end
  end
  local keys: {string} = {}
  for name_key in pairs(holders) do
    keys[#keys + 1] = name_key
  end
  table.sort(keys)
  local first = keys[1]
  if first is nil then
    return nil
  end
  return first
    .. " is not allowed in a models entry ("
    .. table.concat(holders[first] as {string}, ", ")
    .. ")"
end

local function default_context(
  environment: {string: string}
): string | nil, integer | nil
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

local function entries_of(fields: Fields): Fields | nil
  local models = fields.models
  if models is ConfigObject then
    return models as Fields
  end
  return nil
end

function config.cache_type(value: string): compute.CacheType | nil
  return CACHE_TYPES[value]
end

function config.whole_count(value: number): boolean
  return value % 1 == 0 and value >= 1
end

-- 53/FR-002: the rule a model key's value is checked by, at either level
local function value_ok(value: ConfigValue, key: string): boolean
  if key == "kv_cache" then
    if not (value is string) then
      return false
    end
    local kind = config.cache_type(value)
    return not (kind is nil)
  end
  if key == "profile" then
    return value is string
  end
  if key == "prefetch" then
    return value is boolean
  end
  return value is number and config.whole_count(value)
end

-- keep a checked value under its own key, in an entry or the file's top level
local function keep(held: ModelEntry, key: string, value: ConfigValue)
  if key == "kv_cache" then
    held.kv_cache = CACHE_TYPES[value as string]
  elseif key == "profile" then
    held.profile = value as string
  elseif key == "prefetch" then
    held.prefetch = value as boolean
  elseif key == "ubatch" then
    held.ubatch = value as integer
  else
    held.vram = value as integer
  end
end

local function parse(
  file_path: string, text: string, environment: {string: string}
): Config | nil, string | nil
  local decoded = tltp.json_decode(text) as ConfigValue
  if not (decoded is ConfigObject) then
    return fault("config " .. file_path .. ": not a json object")
  end

  local fields = decoded as Fields
  local unknown = unknown_key(fields)
  if unknown is string then
    return fault("config " .. file_path .. ": unknown key " .. unknown)
  end

  local entries = entries_of(fields)
  if not (entries is nil) then
    local both = both_levels(fields, entries)
    if both is string then
      return fault("config " .. file_path .. ": " .. both)
    end
    local misplaced = entry_key(entries)
    if misplaced is string then
      return fault("config " .. file_path .. ": " .. misplaced)
    end
  end

  local settings: Config = {models = {}}

  local host = fields.host
  if host is string then
    if not config.valid_host(host) then
      return fault(
        "config " .. file_path .. ": host must be a dotted IPv4 address"
      )
    end
    settings.host = host
  elseif not (host is nil) then
    return fault(
      "config " .. file_path .. ": host must be a dotted IPv4 address"
    )
  end

  local port = fields.port
  if port is number then
    if not (port % 1 == 0 and port >= 1 and port <= 65535) then
      return fault(
        "config " .. file_path .. ": port must be an integer from 1 to 65535"
      )
    end
    settings.port = port as integer
  elseif not (port is nil) then
    return fault(
      "config " .. file_path .. ": port must be an integer from 1 to 65535"
    )
  end

  local models_value = fields.models
  if models_value is ConfigObject then
    local given = models_value as Fields
    local at_fault: {string: {string}} = {}
    for _, name in ipairs(model_names(given)) do
      local entry = given[name]
      local model_path: ConfigValue = nil
      if entry is ConfigObject then
        model_path = (entry as Fields).path
      end
      if not (model_path is string) then
        return fault(
          "config "
            .. file_path
            .. ": model "
            .. name
            .. " must be an object holding a path string"
        )
      end
      local held: ModelEntry = {path = model_path}
      for _, key in ipairs(MODEL_KEYS) do
        local value = (entry as Fields)[key]
        if not (value is nil) then
          if not value_ok(value, key) then
            local names = at_fault[key]
            if names is nil then
              names = {}
              at_fault[key] = names
            end
            names[#names + 1] = name
          else
            keep(held, key, value)
          end
        end
      end
      settings.models[name] = held
    end
    for _, key in ipairs(MODEL_KEYS) do
      local names = at_fault[key]
      if not (names is nil) then
        return fault(
          "config "
            .. file_path
            .. ": "
            .. VALUE_FAULT[key]
            .. " in models "
            .. table.concat(names as {string}, ", ")
        )
      end
    end
  elseif not (models_value is nil) then
    return fault(
      "config "
        .. file_path
        .. ": models must be an object of entries holding a path string"
    )
  end

  local sessions = fields.sessions
  if sessions is number then
    if not config.whole_count(sessions) then
      return fault(
        "config " .. file_path .. ": sessions must be an integer of one or more"
      )
    end
    settings.sessions = sessions as integer
  elseif not (sessions is nil) then
    return fault(
      "config " .. file_path .. ": sessions must be an integer of one or more"
    )
  end

  local kv_cache = fields.kv_cache
  if not (kv_cache is nil) then
    if not value_ok(kv_cache, "kv_cache") then
      return fault("config " .. file_path .. ": " .. VALUE_FAULT.kv_cache)
    end
    settings.kv_cache = CACHE_TYPES[kv_cache as string]
  end

  local function bad_context(): nil, string
    return fault(
      "config "
        .. file_path
        .. ": context must be false or an object holding a path string"
        .. " and a bytes integer"
    )
  end

  local context = fields.context
  local context_off = false
  local context_set = false
  if context is boolean then
    if context then
      return bad_context()
    end
    context_off = true
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
    context_set = true
    settings.context_path = context_path as string
    settings.context_bytes = context_bytes as integer
  elseif not (context is nil) then
    return bad_context()
  end

  local workers = fields.workers
  if workers is number then
    if not config.whole_count(workers) then
      return fault(
        "config " .. file_path .. ": workers must be an integer of one or more"
      )
    end
    settings.workers = workers as integer
  elseif not (workers is nil) then
    return fault(
      "config " .. file_path .. ": workers must be an integer of one or more"
    )
  end

  local ubatch = fields.ubatch
  if not (ubatch is nil) then
    if not value_ok(ubatch, "ubatch") then
      return fault("config " .. file_path .. ": " .. VALUE_FAULT.ubatch)
    end
    settings.ubatch = ubatch as integer
  end

  local vram = fields.vram
  if not (vram is nil) then
    if not value_ok(vram, "vram") then
      return fault("config " .. file_path .. ": " .. VALUE_FAULT.vram)
    end
    settings.vram = vram as integer
  end

  local profile = fields.profile
  if not (profile is nil) then
    if not value_ok(profile, "profile") then
      return fault("config " .. file_path .. ": " .. VALUE_FAULT.profile)
    end
    settings.profile = profile as string
  end

  local prefetch = fields.prefetch
  if not (prefetch is nil) then
    if not value_ok(prefetch, "prefetch") then
      return fault("config " .. file_path .. ": " .. VALUE_FAULT.prefetch)
    end
    settings.prefetch = prefetch as boolean
  end

  local threads = fields.threads
  local bad_threads = "config "
    .. file_path
    .. ": threads must be an integer from 1 to 512"
  if threads is number then
    if not cores.fits(threads) then
      return fault(bad_threads)
    end
    settings.threads = threads as integer
  elseif not (threads is nil) then
    return fault(bad_threads)
  end

  local names: {string} = {}
  for name in pairs(settings.models) do
    names[#names + 1] = name
  end
  table.sort(names)
  for _, name in ipairs(names) do
    local entry = settings.models[name]
    if not fs.isfile(entry.path) then
      return fault(
        "config "
          .. file_path
          .. ": model "
          .. name
          .. ": "
          .. entry.path
          .. " is not a readable file"
      )
    end
  end

  if not (context_off or context_set) then
    settings.context_path, settings.context_bytes = default_context(environment)
  end
  return settings, nil
end

local function write_starter(file_path: string): string
  local parent = file_path:match("^(.*)/") or "."
  fs.mkdirAll(parent)
  if not fs.write(file_path, STARTER) then
    return "config: cannot write " .. file_path
  end
  return "config: wrote a starter file at " .. file_path
end

local function fresh(environment: {string: string}): Config
  local context_path, context_bytes = default_context(environment)
  return {
    models = {},
    context_path = context_path,
    context_bytes = context_bytes,
  }
end

-- one level's values over the ones already resolved
local function overlay(values: ModelValues, held: Level | nil)
  if held is nil then
    return
  end
  local cache = held.kv_cache
  if not (cache is nil) then
    values.kv_cache = cache
  end
  local ubatch = held.ubatch
  if not (ubatch is nil) then
    values.ubatch = ubatch
  end
  local vram = held.vram
  if not (vram is nil) then
    values.vram = vram
  end
  local profile = held.profile
  if not (profile is nil) then
    values.profile = profile
  end
  local prefetch = held.prefetch
  if not (prefetch is nil) then
    values.prefetch = prefetch
  end
end

-- 53/FR-003, 54/FR-001..FR-005: the values a model is served with: the run's
-- flags, then its entry, then the file's top level, then the default
function config.model_values(
  settings: Config, entry: ModelEntry | nil, flags: Level | nil
): ModelValues
  local values: ModelValues = {kv_cache = "f16", ubatch = DEFAULT_UBATCH}
  overlay(
    values,
    {
      kv_cache = settings.kv_cache,
      ubatch = settings.ubatch,
      vram = settings.vram,
      profile = settings.profile,
      prefetch = settings.prefetch,
    }
  )
  if not (entry is nil) then
    overlay(
      values,
      {
        kv_cache = entry.kv_cache,
        ubatch = entry.ubatch,
        vram = entry.vram,
        profile = entry.profile,
        prefetch = entry.prefetch,
      }
    )
  end
  overlay(values, flags)
  return values
end

-- 53/FR-001: the level table itself, so a test can name exactly what it holds
function config.levels(): {string: KeyLevel}
  return KEYS
end

function config.load(
  environment: {string: string},
  explicit: string | nil,
  path_word: boolean,
  starter?: boolean
): Config | nil, string | nil
  if explicit is string then
    local text = read_file(explicit)
    if text is nil then
      return fault("config " .. explicit .. ": cannot be read")
    end
    local settings, line = parse(explicit, text, environment)
    if settings is nil then
      return nil, line
    end
    return settings, nil
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
      return fault("config " .. chosen .. ": cannot be read")
    end
    local settings, line = parse(chosen, text, environment)
    if settings is nil then
      return nil, line
    end
    return settings, nil
  end

  if path_word or starter == false then
    return fresh(environment), nil
  end

  if #paths == 0 then
    return fault("config: no config directory; set XDG_CONFIG_HOME or HOME")
  end

  return nil, write_starter(paths[1])
end

return config
