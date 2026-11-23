local fs = require("fs")
local json = require("json")
local process = require("process")
local chat = require("tllm.chat")
local compute = require("tllm.compute")
local cores = require("tllm.cores")
local generate = require("tllm.generate")
local gguf = require("tllm.gguf")
local pause = require("tllm.pause")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")
local tltp = require("tltp")

local record serve
  record Model
    name: string
    path: string
    handle: weights.Handle
    tokenizer: tokenizer.Tokenizer
    backend: compute.Backend
    kv_cache: compute.CacheType | nil
    generation: compute.Model
    context_path: string | nil
    context_bytes: integer | nil
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
    sessions: integer
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

  record Worker
    slot: integer
    pid: integer
    sessions: {string}
  end

  record Pool
    dir: string
    workers: {Worker}
  end

  record Session
    id: string
    length: integer
    sequence: string
    positions: integer
  end

  record Reply
    status: tltp.Status
    body: string
  end

  model_of: function(tltp.Server): Model | nil
  build: function(
    endpoint: Endpoint, spawn?: process.Spawn, write_out?: function(string)
  ): tltp.Server | nil, string | nil
  start: function(
    endpoint: Endpoint, write_err: function(string), spawn?: process.Spawn
  ): integer
  answer: function(tltp.Server, string): Reply
  stream: function(tltp.Server, string): tltp.Response, tltp.StreamFn | nil
  session_of: function(tltp.Server, string): Session | nil
  pool_of: function(tltp.Server): Pool | nil
  stop: function(tltp.Server)
  step: function(tltp.Server): integer
  round: function(tltp.Server): string | nil
end

local record LooseObject
end

local type Loose = nil | boolean | number | string | LooseObject
local type Fields = {string: Loose}

local record Request
  messages: {chat.Message}
  temperature: number | nil
  top_k: integer | nil
  top_p: number | nil
  min_p: number | nil
  repetition_penalty: number | nil
  presence_penalty: number | nil
  frequency_penalty: number | nil
  max_tokens: integer
  seed: integer
  stream: boolean
  session_id: string | nil
  stop: {string} | nil
  enable_thinking: boolean
  tools: json.Value | nil
  tool_choice: json.Value | nil
end

local record Held
  ids: {integer}
  session: serve.Session
end

local record Store
  held: {string: Held}
  order: {string}
  limit: integer
end

local record Prepared
  server: tltp.Server
  model: serve.Model
  request: Request
  prompt: string
  ids: {integer}
  sequence: string
  positions: integer
  cached: integer
  store: Store | nil
  waiting: boolean
  reserved: boolean
end

-- why a request was not admitted: the route turns this into a reply
local record Fault
  status: tltp.Status
  message: string
end

local record EventChoice
  index: integer
  delta: {string: string} | nil
  finish_reason: string | nil
end

local record Event
  choices: {EventChoice}
end

local record Turn
  ready: Prepared
  run: generate.Run | nil
  emit: tltp.EmitFn
  live: boolean
  waiting: boolean
end

local record PoolTurn
  body: string
  session_id: string | nil
  emit: tltp.EmitFn | nil
  live: boolean
  assigned: boolean
end

local record PoolWorker
  slot: integer
  pid: integer
  sessions: {string}
  limit: integer
  dir: string
  child: process.Child
  ready: boolean
  replacement: boolean
  turn: PoolTurn | nil
  job: string | nil
  kind: string | nil
  offset: integer
end

local record PoolState
  dir: string
  workers: {PoolWorker}
  turns: {PoolTurn}
  next_turn: integer
  watcher: fs.Watcher | nil
  spawn: process.Spawn
  endpoint_json: string
  limit: integer
end

local record PoolHeader
  kind: string
  status: tltp.Status | nil
  body: string | nil
end

local record EndpointFields
  host: string
  port: integer
  name: string
  path: string
  sessions: integer
  kv_cache: compute.CacheType | nil
  context_path: string | nil
  context_bytes: integer | nil
  ubatch: integer | nil
  vram: integer | nil
  profile: string | nil
  prefetch: boolean | nil
  threads: integer | nil
end

local DEFAULT_BUDGET: integer = 256
-- more bytes than any model file holds, so place takes every fixed tensor
local EVERY_FIXED: integer = 4503599627370496
local CHAT_PATH = "/v1/chat/completions"
local CONTEXT_SUFFIX = ".context"

local record ContextFile
  name: string
  size: integer
  modifyTime: number
end

-- a file name is at most 255 bytes, so a long id is stored under a digest of
-- itself rather than a name that cannot be written
local LONGEST_NAME: integer = 100

-- two 32-bit hashes over the same id, so a name collision needs one in 2^64
-- rather than one in 2^32, which a client choosing its own ids could reach
local function digest(text: string): string
  local first: integer = 5381
  local second: integer = 2166136261
  for index = 1, #text do
    local byte = text:byte(index) as integer
    first = (first * 33 + byte) % 4294967296
    second = ((second * 131 + byte) % 4294967296 + index) % 4294967296
  end
  return string.format("%08x%08x", first, second)
end

-- the session id as text a file name can hold, so no id reaches another path
local function escaped(id: string): string
  local parts: {string} = {}
  for index = 1, #id do
    local byte = id:byte(index) as integer
    local char = string.char(byte)
    if char:match("^[%w._-]$") then
      parts[#parts + 1] = char
    else
      parts[#parts + 1] = string.format("%%%02X", byte)
    end
  end
  return table.concat(parts)
end

local models: {tltp.Server: serve.Model} = {}
local stores: {tltp.Server: Store} = {}
local turns: {tltp.Server: {Turn}} = {}
local reservations: {tltp.Server: {string: boolean}} = {}
local held: {tltp.Server: {string: boolean}} = {}
local pools: {tltp.Server: PoolState} = {}

function serve.model_of(server: tltp.Server): serve.Model | nil
  return models[server]
end

function serve.session_of(server: tltp.Server, id: string): serve.Session | nil
  local store = stores[server]
  if store is nil then
    return nil
  end
  local found = store.held[id]
  if found is nil then
    return nil
  end
  return found.session
end

local function error_body(message: string, kind: string): string
  return tltp.json_encode({error = {message = message, type = kind}}) as string
end

local function as_reply(refusal: Fault): serve.Reply
  local kind = "internal_error"
  if refusal.status == "BadRequest" then
    kind = "invalid_request_error"
  elseif refusal.status == "TooManyRequests" then
    kind = "too_many_requests"
  end
  return {status = refusal.status, body = error_body(refusal.message, kind)}
end

local function bad_request(message: string): Fault
  return {status = "BadRequest", message = message}
end

local function internal(message: string): Fault
  return {status = "InternalServerError", message = message}
end

local function entry_fault(index: integer): string
  return "messages: entry "
    .. tostring(index)
    .. " is not an object with string role and content"
end

local function content_fault(index: integer): string
  return "messages: entry " .. tostring(index) .. " has no usable content"
end

local function part_fault(index: integer, part: integer): string
  return "messages: entry "
    .. tostring(index)
    .. " part "
    .. tostring(part)
    .. " is not an object with string type and text"
end

local function content_of(
  value: Loose, index: integer
): string | nil, string | nil
  if value is string then
    return value
  end
  if not (value is LooseObject) then
    return nil, content_fault(index)
  end
  local parts = value as {Loose}
  if #parts == 0 then
    return nil, content_fault(index)
  end
  local texts: {string} = {}
  for part_index = 1, #parts do
    local part = parts[part_index]
    if not (part is LooseObject) then
      return nil, part_fault(index, part_index)
    end
    local fields = part as Fields
    local kind = fields["type"]
    local text = fields.text
    if not (kind is string) or (kind == "text" and not (text is string)) then
      return nil, part_fault(index, part_index)
    end
    if kind ~= "text" then
      return nil,
        "messages: entry "
          .. tostring(index)
          .. " part "
          .. tostring(part_index)
          .. " has type "
          .. (kind as string)
          .. ", only text parts are read"
    end
    texts[#texts + 1] = text as string
  end
  return table.concat(texts)
end

local function messages_of(value: Loose): {chat.Message} | nil, string | nil
  if value is nil then
    return nil, "messages: the body holds none"
  end
  if not (value is LooseObject) then
    return nil, "messages: not an array"
  end
  local entries = value as {Loose}
  if #entries == 0 then
    return nil, "messages: the list is empty"
  end
  local messages: {chat.Message} = {}
  for index = 1, #entries do
    local entry = entries[index]
    if not (entry is LooseObject) then
      return nil, entry_fault(index)
    end
    local fields = entry as Fields
    local role = fields.role
    if not (role is string) then
      return nil, entry_fault(index)
    end
    local content, content_line = content_of(fields.content, index)
    if content is nil then
      return nil, content_line
    end
    messages[index] = {role = role as string, content = content}
  end
  return messages, nil
end

local function whole(value: Loose): integer | nil
  if not (value is number) or value % 1 ~= 0 then
    return nil
  end
  return value as integer
end

-- the one kind a body member of the wrong type never becomes: the sentinel
-- json_decode hands back for JSON null, which is a table like an object is
local NULL = json.null as LooseObject

-- 46/FR-001 to 46/FR-006: the thinking value this request binds, or the one
-- line naming the member the route cannot read. A member of either name is
-- refused before any member binds or wins.
local function thinking_of(given: Fields): boolean | nil, string | nil
  local top = given.enable_thinking
  if not (top is nil) and not (top is boolean) then
    return nil, "enable_thinking: not a boolean"
  end

  local box = given.chat_template_kwargs
  if box is nil then
    return top == true, nil
  end
  if not (box is LooseObject) or box == NULL then
    return nil, "chat_template_kwargs: not an object"
  end

  local member = (box as Fields).enable_thinking
  if member is boolean then
    return member, nil
  end
  if not (member is nil) then
    return nil, "chat_template_kwargs.enable_thinking: not a boolean"
  end
  if #(box as {Loose}) > 0 then
    return nil, "chat_template_kwargs: not an object"
  end
  return top == true, nil
end

-- 47/FR-001 and 47/FR-003: the two members are read from a second decode of
-- the same text because a tool schema is arbitrary nested JSON, and tltp's
-- unpack loses an object's member order and gives one table for an empty
-- object and an empty array.
local function member_of(
  container: json.Value, wanted: string
): json.Value | nil
  for name, child in container:children() do
    if name == wanted then
      return child as json.Value
    end
  end
  return nil
end

local function tool_fault(position: integer, entry: json.Value): string | nil
  local at = "tools: entry " .. tostring(position)
  if not entry:is("object") then
    return at .. " is not an object"
  end
  local member = member_of(entry, "function")
  if member is nil or not member:is("object") then
    return at .. " holds no function object"
  end
  local name = member_of(member, "name")
  if name is nil or not name:is("string") or #(name:val() as string) < 1 then
    return at .. " holds no function.name of one or more bytes"
  end
  return nil
end

-- 47/FR-001, 47/FR-002, 47/FR-003 and 47/FR-006: the tool list and tool choice
-- this request hands the template, or the one line naming the entry the route
-- cannot use. A body carrying no such member gives nothing for it.
local function tools_of(
  body: string
): json.Value | nil, json.Value | nil, string | nil
  local root = json.decode(body)
  local given = member_of(root, "tools")
  local choice = member_of(root, "tool_choice")
  if given is nil then
    return nil, choice, nil
  end
  if not given:is("array") then
    return nil, nil, "tools: not an array"
  end
  local position = 0
  for _, entry in given:children() do
    position = position + 1
    local line = tool_fault(position, entry as json.Value)
    if line is string then
      return nil, nil, line
    end
  end
  return given, choice, nil
end

local function request_of(body: string): Request | nil, string | nil
  local decoded = tltp.json_decode(body) as Loose
  -- the decoder gives the same empty table for {} and [], so the first
  -- non-space character is what tells an object from an array
  local opening = body:find("%S")
  if
    decoded is nil
    or opening is nil
    or body:sub(opening, opening) ~= "{"
    or not (decoded is LooseObject)
  then
    return nil, "body: not a JSON object"
  end

  local fields = decoded as Fields

  local messages, messages_line = messages_of(fields.messages)
  if messages is nil then
    return nil, messages_line
  end

  local temperature: number | nil = nil
  local temperature_value = fields.temperature
  if temperature_value is number then
    if temperature_value < 0 then
      return nil, "temperature: below zero"
    end
    temperature = temperature_value
  elseif not (temperature_value is nil) then
    return nil, "temperature: not a number"
  end

  local top_k: integer | nil = nil
  local top_k_value = fields.top_k
  if not (top_k_value is nil) then
    top_k = whole(top_k_value)
    if top_k is nil or top_k < 1 then
      return nil, "top_k: not a whole number of one or more"
    end
  end

  local budget = DEFAULT_BUDGET
  local max_tokens_value = fields.max_tokens
  if not (max_tokens_value is nil) then
    local held_budget = whole(max_tokens_value)
    if held_budget is nil or held_budget < 1 then
      return nil, "max_tokens: not a whole number of one or more"
    end
    budget = held_budget
  end

  local seed = 0
  local seed_value = fields.seed
  if not (seed_value is nil) then
    local seed_number = whole(seed_value)
    if seed_number is nil then
      return nil, "seed: not a whole number"
    end
    seed = seed_number
  end

  local stream = false
  local stream_value = fields.stream
  if stream_value is boolean then
    stream = stream_value
  elseif not (stream_value is nil) then
    return nil, "stream: not a boolean"
  end

  local session_id: string | nil = nil
  local session_value = fields.session_id
  if session_value is string then
    session_id = session_value
  elseif not (session_value is nil) then
    return nil, "session_id: not a string"
  end

  local top_p: number | nil = nil
  local top_p_value = fields.top_p
  if top_p_value is number then
    if top_p_value <= 0 or top_p_value > 1 then
      return nil, "top_p: not above zero and at most one"
    end
    top_p = top_p_value
  elseif not (top_p_value is nil) then
    return nil, "top_p: not a number"
  end

  local min_p: number | nil = nil
  local min_p_value = fields.min_p
  if min_p_value is number then
    if min_p_value < 0 or min_p_value > 1 then
      return nil, "min_p: not between zero and one"
    end
    min_p = min_p_value
  elseif not (min_p_value is nil) then
    return nil, "min_p: not a number"
  end

  local repetition_penalty: number | nil = nil
  local repetition_value = fields.repetition_penalty
  if repetition_value is number then
    if repetition_value <= 0 then
      return nil, "repetition_penalty: not above zero"
    end
    repetition_penalty = repetition_value
  elseif not (repetition_value is nil) then
    return nil, "repetition_penalty: not a number"
  end

  local presence_penalty: number | nil = nil
  local presence_value = fields.presence_penalty
  if presence_value is number then
    presence_penalty = presence_value
  elseif not (presence_value is nil) then
    return nil, "presence_penalty: not a number"
  end

  local frequency_penalty: number | nil = nil
  local frequency_value = fields.frequency_penalty
  if frequency_value is number then
    frequency_penalty = frequency_value
  elseif not (frequency_value is nil) then
    return nil, "frequency_penalty: not a number"
  end

  local stop: {string} | nil = nil
  local stop_value = fields.stop
  if not (stop_value is nil) then
    if stop_value is string then
      if #stop_value < 1 then
        return nil, "stop: not a string of one or more bytes"
      end
      stop = {stop_value}
    elseif stop_value is LooseObject then
      local given = stop_value as {Loose}
      if #given < 1 or #given > 4 then
        return nil,
          "stop: not a string of one or more bytes or an array of one to four"
      end
      local markers: {string} = {}
      for _, entry in ipairs(given) do
        if not (entry is string) or #(entry as string) < 1 then
          return nil,
            "stop: not a string of one or more bytes or an array of one to four"
        end
        markers[#markers + 1] = entry as string
      end
      stop = markers
    else
      return nil,
        "stop: not a string of one or more bytes or an array of one to four"
    end
  end

  local thinking, thinking_line = thinking_of(fields)
  if not (thinking is boolean) then
    return nil, thinking_line
  end

  local held_tools: json.Value | nil = nil
  local tool_choice: json.Value | nil = nil
  if fields.tools ~= nil or fields.tool_choice ~= nil then
    local tools_line: string | nil
    held_tools, tool_choice, tools_line = tools_of(body)
    if tools_line is string then
      return nil, tools_line
    end
  end

  return {
    messages = messages,
    temperature = temperature,
    top_k = top_k,
    top_p = top_p,
    min_p = min_p,
    repetition_penalty = repetition_penalty,
    presence_penalty = presence_penalty,
    frequency_penalty = frequency_penalty,
    max_tokens = budget,
    seed = seed,
    stream = stream,
    session_id = session_id,
    stop = stop,
    enable_thinking = thinking,
    tools = held_tools,
    tool_choice = tool_choice,
  }, nil
end

local function budget_fault(
  ids: integer, budget: integer, positions: integer
): Fault
  return bad_request(
    "messages: the prompt's "
      .. tostring(ids)
      .. " ids plus the max_tokens budget "
      .. tostring(budget)
      .. " minus one is past the "
      .. tostring(positions)
      .. " positions the file allows"
  )
end

local function is_held(server: tltp.Server, name: string): boolean
  local marks = held[server]
  return marks is {string: boolean} and marks[name] == true
end

local function mark_held(server: tltp.Server, name: string)
  local marks = held[server]
  if marks is nil then
    marks = {}
    held[server] = marks
  end
  marks[name] = true
end

local function unmark_held(server: tltp.Server, name: string)
  local marks = held[server]
  if marks is nil then
    return
  end
  marks[name] = nil
end

local function is_reserved(server: tltp.Server, name: string): boolean
  local marks = reservations[server]
  return marks is {string: boolean} and marks[name] == true
end

local function free_slots(server: tltp.Server): integer
  local model = models[server]
  if model is nil then
    return 0
  end
  local count = 0
  local marks = reservations[server]
  if marks is {string: boolean} then
    for _ in pairs(marks) do
      count = count + 1
    end
  end
  return model.generation.sequences - count
end

local function release_name(server: tltp.Server, name: string)
  local model = models[server]
  if model is serve.Model then
    model.generation:release(name)
  end
  local marks = reservations[server]
  if marks is {string: boolean} then
    marks[name] = nil
  end
end

local function begins_with(ids: {integer}, prefix: {integer}): boolean
  for index = 1, #prefix do
    if ids[index] ~= prefix[index] then
      return false
    end
  end
  return true
end

local function file_name(id: string): string
  local escaped_id = escaped(id)
  if #escaped_id <= LONGEST_NAME then
    return escaped_id .. CONTEXT_SUFFIX
  end
  return "long-" .. digest(id) .. CONTEXT_SUFFIX
end

local function context_file(model: serve.Model, id: string): string | nil
  local dir = model.context_path
  if dir is nil then
    return nil
  end
  return dir .. "/" .. file_name(id)
end

local function context_files(dir: string): {ContextFile}
  local files: {ContextFile} = {}
  local next_entry = fs.readdir(dir)
  if next_entry is nil then
    return files
  end
  while true do
    local entry = next_entry()
    if entry is nil then
      break
    end
    local name = (entry as fs.DirEntry).name
    if name:sub(-#CONTEXT_SUFFIX) == CONTEXT_SUFFIX then
      -- the file's own mode tells its type, where a directory entry's may not
      local details = fs.stat(dir .. "/" .. name)
      if details is fs.Stat then
        local info = details
        if info.type == "file" then
          files[#files + 1] = {
            name = name,
            size = math.floor(tonumber(info.size) as number),
            modifyTime = tonumber(info.modifyTime) as number,
          }
        end
      end
    end
  end
  return files
end

-- the directory holds no more than the budget: the oldest files go first,
-- and a file that alone holds more than the budget goes alone
local function trimmed(model: serve.Model, name: string)
  local dir = model.context_path
  local budget = model.context_bytes
  if dir is nil or budget is nil then
    return
  end
  local files = context_files(dir)
  local total = 0
  local fresh: ContextFile | nil = nil
  for _, file in ipairs(files) do
    total = total + file.size
    if file.name == name then
      fresh = file
    end
  end
  if fresh is nil then
    return
  end
  local newest = fresh as ContextFile
  if newest.size > budget then
    fs.delete(dir .. "/" .. newest.name)
    return
  end
  if total <= budget then
    return
  end
  table.sort(
    files,
    function(left: ContextFile, right: ContextFile): boolean
      if left.modifyTime ~= right.modifyTime then
        return left.modifyTime < right.modifyTime
      end
      return left.name < right.name
    end
  )
  for _, file in ipairs(files) do
    if total <= budget then
      break
    end
    if file.name ~= newest.name then
      local path = dir .. "/" .. file.name
      fs.delete(path)
      if fs.exists(path) then
        break
      end
      total = total - file.size
    end
  end
end

-- the conversation's file, written for a turn that stored it
local function stored(
  model: serve.Model, sequence: string, ids: {integer}, id: string
)
  local path = context_file(model, id)
  if path is nil then
    return
  end
  local line = model.generation:save(sequence, ids, path)
  if line is string then
    return
  end
  trimmed(model, file_name(id))
end

local function free_name(server: tltp.Server, base: string): string
  local attempt = 0
  while attempt < 100000 do
    local name = base
    if attempt > 0 then
      if base == "" then
        name = "turn" .. attempt
      else
        name = base .. "#" .. attempt
      end
    end
    if not is_reserved(server, name) then
      return name
    end
    attempt = attempt + 1
  end
  return base
end

-- the conversation the file under this id holds, when it continues the prompt
local function restored_session(
  server: tltp.Server, id: string, ids: {integer}, needed: integer
): Held | nil
  local model = models[server]
  if model is nil then
    return nil
  end
  local path = context_file(model, id)
  if path is nil or not fs.isfile(path) then
    return nil
  end
  local name = free_name(server, id)
  local file_ids = model.generation:load(name, path)
  if file_ids is nil then
    return nil
  end
  local restored = file_ids as {integer}
  local capacity = model.generation:capacity(name)
  if
    not (#restored < #ids and begins_with(ids, restored))
    or not (capacity is integer)
    or needed > (capacity as integer)
  then
    release_name(server, name)
    return nil
  end
  return {
    ids = restored,
    session = {
      id = id,
      length = #restored,
      sequence = name,
      positions = capacity as integer,
    },
  }
end

local function forgotten(store: Store, id: string)
  store.held[id] = nil
  for index = #store.order, 1, -1 do
    if store.order[index] == id then
      table.remove(store.order, index)
    end
  end
end

local function used(store: Store, id: string)
  for index = #store.order, 1, -1 do
    if store.order[index] == id then
      table.remove(store.order, index)
    end
  end
  store.order[#store.order + 1] = id
end

-- the oldest stored conversation no live turn holds
local function evict_one(server: tltp.Server, store: Store): boolean
  for _, id in ipairs(store.order) do
    local held_conversation = store.held[id]
    if
      held_conversation is Held
      and not is_held(server, held_conversation.session.sequence)
    then
      forgotten(store, id)
      release_name(server, held_conversation.session.sequence)
      return true
    end
  end
  return false
end

local function reserve_sequence(
  server: tltp.Server, base: string, positions: integer
): string | nil, string | nil
  local model = models[server]
  if model is nil then
    return nil, "model: this server holds none"
  end
  local attempt = 0
  while attempt < 100000 do
    local name = base
    if attempt > 0 then
      if base == "" then
        name = "turn" .. attempt
      else
        name = base .. "#" .. attempt
      end
    end
    if not is_reserved(server, name) then
      local line = model.generation:reserve(name, positions)
      if line is nil then
        local marks = reservations[server]
        if marks is nil then
          marks = {}
          reservations[server] = marks
        end
        marks[name] = true
        return name, nil
      end
      if line:find("is already held", 1, true) == nil then
        return nil, line
      end
    end
    attempt = attempt + 1
  end
  return nil, "reserve: no free sequence name"
end

local function too_many(server: tltp.Server): Fault
  local store = stores[server]
  local limit = 0
  if store is Store then
    limit = store.limit
  end
  return {
    status = "TooManyRequests",
    message = "the server holds no free session; sessions is "
      .. tostring(limit),
  }
end

local function prepared(
  server: tltp.Server, body: string
): Prepared | nil, Fault | nil
  local model = models[server]
  if model is nil then
    return nil, internal("model: this server holds none")
  end

  local request, request_line = request_of(body)
  if request is nil then
    return nil, bad_request(request_line as string)
  end

  local prompt, render_line = chat.render(
    model.tokenizer,
    request.messages,
    request.tools,
    request.enable_thinking,
    request.tool_choice
  )
  if prompt is nil then
    return nil, internal(render_line as string)
  end

  local ids, encode_line = model.tokenizer:encode(prompt)
  if ids is nil then
    return nil, internal(encode_line as string)
  end

  local generation = model.generation
  local window = generation.positions
  local budget = request.max_tokens
  local needed = #ids + budget - 1
  if needed > window then
    return nil, budget_fault(#ids, budget, window)
  end

  local id = request.session_id
  local store: Store | nil = stores[server]
  local sequence = ""
  local positions = needed * 2
  if positions > window then
    positions = window
  end
  local cached = 0
  local waiting = false
  local reserved = false
  local found_session: serve.Session | nil = nil

  if id is string and store is Store then
    local found = store.held[id]
    if found is Held then
      local session = found.session
      if
        session.length < #ids
        and needed <= session.positions
        and begins_with(ids, found.ids)
        and not is_held(server, session.sequence)
      then
        found_session = session
      elseif not is_held(server, session.sequence) then
        forgotten(store, id)
        release_name(server, session.sequence)
      end
    else
      if #store.order >= store.limit then
        evict_one(server, store)
      end
      if #store.order < store.limit then
        local revived = restored_session(server, id, ids, needed)
        if revived is Held then
          local hold = revived
          store.held[id] = hold
          used(store, id)
          found_session = hold.session
        end
      end
    end
  end

  if found_session is serve.Session then
    sequence = found_session.sequence
    positions = found_session.positions
    cached = found_session.length
    local marks = reservations[server]
    if marks is nil then
      marks = {}
      reservations[server] = marks
    end
    marks[sequence] = true
    mark_held(server, sequence)
    return {
      server = server,
      model = model,
      request = request,
      prompt = prompt,
      ids = ids,
      sequence = sequence,
      positions = positions,
      cached = cached,
      store = store,
      waiting = false,
      reserved = true,
    }, nil
  end

  if free_slots(server) < 1 and store is Store then
    evict_one(server, store)
  end

  local slots = free_slots(server)
  if slots < 1 and not request.stream then
    return nil, too_many(server)
  end
  if slots < 1 then
    waiting = true
    sequence = id or ""
  else
    local name, reserve_line = reserve_sequence(server, id or "", positions)
    if name is nil then
      return nil, internal(reserve_line as string)
    end
    sequence = name
    mark_held(server, sequence)
    reserved = true
  end

  return {
    server = server,
    model = model,
    request = request,
    prompt = prompt,
    ids = ids,
    sequence = sequence,
    positions = positions,
    cached = cached,
    store = store,
    waiting = waiting,
    reserved = reserved,
  }, nil
end

local function options_of(ready: Prepared): generate.Options
  local request = ready.request
  return {
    max_tokens = request.max_tokens,
    seed = request.seed,
    temperature = request.temperature,
    top_k = request.top_k,
    top_p = request.top_p,
    min_p = request.min_p,
    repetition_penalty = request.repetition_penalty,
    presence_penalty = request.presence_penalty,
    frequency_penalty = request.frequency_penalty,
    start = ready.cached,
    stop = request.stop,
  }
end

local function forwarded(
  ready: Prepared, sampled: {integer}, cut: boolean
): {integer}
  local count = #sampled
  if cut or count == ready.request.max_tokens then
    count = count - 1
  end
  local ids: {integer} = {}
  for index = 1, #ready.ids do
    ids[index] = ready.ids[index]
  end
  for index = 1, count do
    ids[#ids + 1] = sampled[index]
  end
  return ids
end

local function released(ready: Prepared)
  unmark_held(ready.server, ready.sequence)
  if ready.reserved then
    release_name(ready.server, ready.sequence)
    ready.reserved = false
  end
end

local function kept(ready: Prepared, sampled: {integer}, cut: boolean)
  local store = ready.store
  local id = ready.request.session_id
  if id is string and store is Store then
    local existing = store.held[id]
    if existing is Held and existing.session.sequence ~= ready.sequence then
      forgotten(store, id)
      if not is_held(ready.server, existing.session.sequence) then
        release_name(ready.server, existing.session.sequence)
      end
    end
    if store.held[id] == nil and #store.order >= store.limit then
      evict_one(ready.server, store)
    end
    local ids = forwarded(ready, sampled, cut)
    store.held[id] = {
      ids = ids,
      session = {
        id = id,
        length = #ids,
        sequence = ready.sequence,
        positions = ready.positions,
      },
    }
    used(store, id)
    stored(ready.model, ready.sequence, ids, id)
    unmark_held(ready.server, ready.sequence)
    ready.reserved = false
    return
  end
  released(ready)
end

local function lost(ready: Prepared)
  local store = ready.store
  local id = ready.request.session_id
  if id is string and store is Store then
    local existing = store.held[id]
    if existing is Held and existing.session.sequence == ready.sequence then
      forgotten(store, id)
    end
  end
  released(ready)
end

local function finished(ready: Prepared): serve.Reply
  local answer, text_line = generate.text_ids(
    ready.model.generation,
    ready.model.tokenizer,
    ready.ids,
    options_of(ready),
    ready.sequence
  )
  if answer is nil then
    lost(ready)
    return as_reply(internal(text_line as string))
  end

  local sampled = answer.sampled
  local completion = #sampled
  kept(ready, sampled, answer.cut)
  return {
    status = "OK",
    body = tltp.json_encode(
      {
        choices = {
          {index = 0, message = {role = "assistant", content = answer.text}},
        },
        usage = {
          prompt_tokens = #ready.ids,
          completion_tokens = completion,
          total_tokens = #ready.ids + completion,
          prompt_tokens_details = {cached_tokens = ready.cached},
        },
      }
    ) as string,
  }
end

local function event_data(choice: EventChoice): string | nil
  local data = tltp.json_encode({choices = {choice}} as Event)
  return data
end

local function registered(server: tltp.Server): {Turn}
  local list = turns[server]
  if list is nil then
    list = {}
    turns[server] = list
  end
  return list
end

local function dropped(server: tltp.Server, turn: Turn)
  turn.live = false
  local list = turns[server]
  if list is nil then
    return
  end
  for index = #list, 1, -1 do
    if list[index] == turn then
      table.remove(list, index)
    end
  end
end

local function producer(
  server: tltp.Server, ready: Prepared, run: generate.Run | nil
): tltp.StreamFn
  return function(emit: tltp.EmitFn): boolean | nil
    local opening = event_data({index = 0, delta = {role = "assistant"}})
    if opening is nil then
      lost(ready)
      return true
    end
    emit(opening)

    local turn: Turn = {
      ready = ready,
      run = run,
      emit = emit,
      live = true,
      waiting = ready.waiting,
    }
    local list = registered(server)
    list[#list + 1] = turn

    coroutine.yield()
    while turn.live do
      coroutine.yield()
    end
    return true
  end
end

-- a turn a slot never took ends the stream it has already opened, so a client
-- waiting for its ending sees one instead of a reply that stops mid-frame
local function abandoned(server: tltp.Server, turn: Turn)
  local ending = event_data({index = 0, finish_reason = "error"})
  if ending is string then
    turn.emit(ending)
    turn.emit("[DONE]")
  end
  dropped(server, turn)
end

local function admit(server: tltp.Server, turn: Turn): boolean
  local ready = turn.ready
  local name = reserve_sequence(server, ready.sequence, ready.positions)
  if name is nil then
    lost(ready)
    abandoned(server, turn)
    return false
  end
  ready.sequence = name
  ready.reserved = true
  mark_held(server, name)

  local run = generate.open_ids(
    ready.model.generation,
    ready.model.tokenizer,
    ready.ids,
    options_of(ready),
    name
  )
  if run is nil then
    lost(ready)
    abandoned(server, turn)
    return false
  end
  turn.run = run
  turn.waiting = false
  return true
end

-- a wait that ends even when a dying worker writes no file
local function pool_pause(state: PoolState)
  local watcher = state.watcher
  if watcher is fs.Watcher then
    watcher:poll()
  end
  pause.milliseconds(2)
end

local function clear_files(dir: string)
  local next_entry = fs.readdir(dir)
  if next_entry is nil then
    return
  end
  while true do
    local found = next_entry()
    if found is nil then
      break
    end
    local entry = found as fs.DirEntry
    if entry.type ~= "dir" then
      fs.delete(dir .. "/" .. entry.name)
    end
  end
end

-- process.Child:poll gives nil for a live child and for one a signal ended,
-- so the pid's own state is the other half of the answer
local function ended(worker: PoolWorker): boolean
  local code = worker.child:poll()
  if not (code is nil) then
    return true
  end
  local text = fs.read("/proc/" .. tostring(worker.pid) .. "/stat")
  if text is nil then
    return process.platform == "linux"
  end
  return text:match("%) (%a)") == "Z"
end

local function failed_line(worker: PoolWorker): string | nil
  local path = worker.dir .. "/failed"
  if not fs.isfile(path) then
    return nil
  end
  local text = fs.read(path)
  if text is nil then
    return "the worker gave no line"
  end
  return (text:gsub("%s+$", ""))
end

local function worker_holds(worker: PoolWorker, id: string): boolean
  for _, held_id in ipairs(worker.sessions) do
    if held_id == id then
      return true
    end
  end
  return false
end

local function served_by(worker: PoolWorker, id: string)
  for index = #worker.sessions, 1, -1 do
    if worker.sessions[index] == id then
      table.remove(worker.sessions, index)
    end
  end
  worker.sessions[#worker.sessions + 1] = id
  while #worker.sessions > worker.limit do
    table.remove(worker.sessions, 1)
  end
end

local function pool_spawn(
  state: PoolState, slot: integer, replacement: boolean
): string | nil
  local dir = state.dir .. "/w" .. tostring(slot)
  fs.mkdirAll(dir)
  clear_files(dir)
  local child, failure = state.spawn(
    "lde",
    {"run", "src/worker.tl", state.dir, tostring(slot)},
    {stdin = state.endpoint_json, stdout = "pipe", stderr = "pipe"}
  )
  if child is nil then
    return failure or "cannot be started"
  end
  state.workers[#state.workers + 1] = {
    slot = slot,
    pid = child.pid,
    sessions = {},
    limit = state.limit,
    dir = dir,
    child = child,
    ready = false,
    replacement = replacement,
    turn = nil,
    job = nil,
    kind = nil,
    offset = 0,
  }
  return nil
end

local function pool_built(state: PoolState): string | nil
  for _, worker in ipairs(state.workers) do
    while not worker.ready do
      if fs.isfile(worker.dir .. "/ready") then
        worker.ready = true
        worker.replacement = false
        break
      end
      local line = failed_line(worker)
      if line is string then
        return "slot " .. tostring(worker.slot) .. ": " .. line
      end
      if ended(worker) then
        return "slot " .. tostring(worker.slot) .. ": worker exited"
      end
      pool_pause(state)
    end
  end
  return nil
end

local function choose_worker(
  state: PoolState, id: string | nil
): PoolWorker | nil
  if id is string then
    for _, worker in ipairs(state.workers) do
      if worker_holds(worker, id) then
        if worker.ready and worker.turn == nil then
          return worker
        end
        return nil
      end
    end
  end
  local best: PoolWorker | nil = nil
  local best_count = math.huge
  for _, worker in ipairs(state.workers) do
    if worker.ready and worker.turn == nil then
      local count = #worker.sessions
      local better = best is nil
      if not better then
        local other = best as PoolWorker
        better = count < best_count
          or (count == best_count and worker.slot < other.slot)
      end
      if better then
        best = worker
        best_count = count
      end
    end
  end
  return best
end

local function dropped_turn(state: PoolState, turn: PoolTurn)
  turn.live = false
  for index = #state.turns, 1, -1 do
    if state.turns[index] == turn then
      table.remove(state.turns, index)
    end
  end
end

local function released_worker(worker: PoolWorker)
  local job = worker.job
  if job is string then
    local target = worker.dir .. "/" .. job
    fs.delete(target .. ".in")
    fs.delete(target .. ".out")
    fs.delete(target .. ".done")
    fs.delete(target .. ".go")
  end
  worker.turn = nil
  worker.job = nil
  worker.kind = nil
  worker.offset = 0
end

local function dispatched(state: PoolState, worker: PoolWorker, turn: PoolTurn)
  state.next_turn = state.next_turn + 1
  local name = "turn" .. tostring(state.next_turn)
  local target = worker.dir .. "/" .. name
  fs.write(target .. ".tmp", turn.body)
  fs.move(target .. ".tmp", target .. ".in")
  worker.job = name
  worker.kind = nil
  worker.offset = 0
  worker.turn = turn
  turn.assigned = true
  local id = turn.session_id
  if id is string then
    served_by(worker, id)
  end
end

-- what the worker has written past the offset this turn has read, so a step
-- reads the new frame and not the whole turn again
local function tail_of(path: string, offset: integer): string | nil
  local made_file = io.open(path, "rb")
  if made_file is nil then
    return nil
  end
  local file = made_file
  if not file:seek("set", offset) then
    file:close()
    return nil
  end
  local rest = file:read("a")
  file:close()
  return rest
end

-- one frame off the worker's output file, leaving the rest for later steps
local function next_frame(worker: PoolWorker): string | nil
  local job = worker.job
  if job is nil then
    return nil
  end
  local rest = tail_of(worker.dir .. "/" .. job .. ".out", worker.offset)
  if rest is nil then
    return nil
  end
  local colon = rest:find(":", 1, true)
  if colon is nil then
    return nil
  end
  local count = tonumber(rest:sub(1, colon - 1)) as integer | nil
  if count is nil then
    return nil
  end
  local start = colon + 1
  if start + count - 1 > #rest then
    return nil
  end
  local part = rest:sub(start, start + count - 1)
  worker.offset = worker.offset + start + count - 1
  return part
end

local function absorbed(worker: PoolWorker): {string}
  local parts: {string} = {}
  while true do
    local part = next_frame(worker)
    if part is nil then
      break
    end
    parts[#parts + 1] = part
  end
  return parts
end

local function drained(worker: PoolWorker): boolean
  local job = worker.job
  if job is nil then
    return true
  end
  local rest = tail_of(worker.dir .. "/" .. job .. ".out", worker.offset)
  if rest is nil then
    return true
  end
  return rest == ""
end

local function pool_advance(state: PoolState, worker: PoolWorker)
  local turn = worker.turn
  if turn is nil then
    return
  end
  if worker.kind == nil then
    local opening = next_frame(worker)
    if opening is nil then
      return
    end
    local header = tltp.json_decode(opening) as PoolHeader | nil
    if header is nil then
      worker.kind = "json"
    elseif header.kind == "json" then
      local send = turn.emit
      if send is tltp.EmitFn then
        send(header.body or "")
      end
      worker.kind = "json"
    else
      worker.kind = "stream"
    end
  end

  local parts = absorbed(worker)
  for _, part in ipairs(parts) do
    local send = turn.emit
    if send is tltp.EmitFn then
      send(part)
    end
  end

  local job = worker.job
  if job is nil then
    return
  end
  if fs.isfile(worker.dir .. "/" .. job .. ".done") and drained(worker) then
    dropped_turn(state, turn)
    released_worker(worker)
    return
  end
  -- the worker takes one step for every tick, and a tick is only written
  -- once the payloads of the last one have been read
  if worker.kind ~= "json" and #parts == 0 then
    local tick = worker.dir .. "/" .. job .. ".go"
    if not fs.exists(tick) then
      fs.write(tick, "")
    end
  end
end

local function pool_reap(state: PoolState)
  local index = 1
  while index <= #state.workers do
    local worker = state.workers[index]
    if ended(worker) then
      worker.child:wait()
      local turn = worker.turn
      if turn is PoolTurn then
        dropped_turn(state, turn)
      end
      table.remove(state.workers, index)
      if not worker.replacement then
        pool_spawn(state, worker.slot, true)
      end
    else
      index = index + 1
    end
  end
end

local function pool_dispatch(state: PoolState)
  for _, turn in ipairs(state.turns) do
    if turn.live and not turn.assigned then
      local worker = choose_worker(state, turn.session_id)
      if worker is PoolWorker then
        dispatched(state, worker, turn)
      end
    end
  end
end

local function pool_ready(state: PoolState)
  for _, worker in ipairs(state.workers) do
    if not worker.ready and fs.isfile(worker.dir .. "/ready") then
      worker.ready = true
      -- it built its server, so this slot is healthy again and its next death
      -- is answered with a replacement like any other
      worker.replacement = false
    end
  end
end

local function pool_step(state: PoolState): integer
  pool_ready(state)
  pool_reap(state)
  for _, worker in ipairs(state.workers) do
    if worker.turn ~= nil then
      pool_advance(state, worker)
    end
  end
  pool_dispatch(state)
  local live = 0
  for _, turn in ipairs(state.turns) do
    if turn.live then
      live = live + 1
    end
  end
  return live
end

local function asked_stream(body: string): boolean
  local decoded = tltp.json_decode(body) as Loose
  if not (decoded is LooseObject) then
    return false
  end
  local fields = decoded as Fields
  return fields.stream == true
end

local function session_of_body(body: string): string | nil
  local request = request_of(body)
  if request is nil then
    return nil
  end
  return (request as Request).session_id
end

local function pool_answer(
  server: tltp.Server, state: PoolState, body: string
): serve.Reply
  local job_body = body
  if asked_stream(body) then
    local decoded = tltp.json_decode(body) as Loose
    local fields = decoded as Fields
    fields.stream = nil
    job_body = tltp.json_encode(fields) as string
  end
  local turn: PoolTurn = {
    body = job_body,
    session_id = session_of_body(body),
    emit = nil,
    live = true,
    assigned = false,
  }
  local worker: PoolWorker | nil = nil
  while worker is nil do
    pool_reap(state)
    worker = choose_worker(state, turn.session_id)
    if worker is nil then
      serve.step(server)
      pool_pause(state)
    end
  end
  local held_worker = worker as PoolWorker
  dispatched(state, held_worker, turn)
  local job = held_worker.job as string
  while not fs.isfile(held_worker.dir .. "/" .. job .. ".done") do
    if ended(held_worker) then
      local slot = held_worker.slot
      pool_reap(state)
      return as_reply(internal("slot " .. tostring(slot) .. " went away"))
    end
    pool_pause(state)
  end
  local slot = held_worker.slot
  local first = next_frame(held_worker)
  released_worker(held_worker)
  if first is nil then
    return as_reply(internal("slot " .. tostring(slot) .. " gave no reply"))
  end
  local header = tltp.json_decode(first) as PoolHeader | nil
  if header is nil then
    return as_reply(internal("slot " .. tostring(slot) .. " gave no reply"))
  end
  local reply_body = header.body
  if reply_body is nil then
    return as_reply(internal("slot " .. tostring(slot) .. " gave no reply"))
  end
  local status: tltp.Status = "InternalServerError"
  local given = header.status
  if given is string then
    status = given as tltp.Status
  end
  return {status = status, body = reply_body}
end

local function pool_producer(state: PoolState, turn: PoolTurn): tltp.StreamFn
  return function(emit: tltp.EmitFn): boolean | nil
    turn.emit = emit
    state.turns[#state.turns + 1] = turn
    coroutine.yield()
    while turn.live do
      coroutine.yield()
    end
    return true
  end
end

local function pool_stream(
  server: tltp.Server, state: PoolState, body: string
): tltp.Response, tltp.StreamFn | nil
  if not asked_stream(body) then
    local reply = pool_answer(server, state, body)
    return server:json(reply.status, reply.body), nil
  end
  local turn: PoolTurn = {
    body = body,
    session_id = session_of_body(body),
    emit = nil,
    live = true,
    assigned = false,
  }
  local stream_fn = pool_producer(state, turn)
  return server:sse(stream_fn), stream_fn
end

function serve.step(server: tltp.Server): integer
  local state = pools[server]
  if state is PoolState then
    return pool_step(state)
  end
  local list = turns[server]
  if list is nil then
    return 0
  end
  local snapshot: {Turn} = {}
  for index = 1, #list do
    snapshot[index] = list[index]
  end

  for _, turn in ipairs(snapshot) do
    if turn.live and turn.waiting then
      if free_slots(server) < 1 then
        local store = stores[server]
        if store is nil then
          break
        end
        if not evict_one(server, store) then
          break
        end
      end
      admit(server, turn)
    end
  end

  local model = models[server] as serve.Model
  -- every turn that has a row takes its whole share of the model's bound, so a
  -- step hands one pass the groups of them all
  local with_rows = 0
  for _, turn in ipairs(list) do
    local driven = turn.run
    if turn.live and not turn.waiting and driven is generate.Run then
      with_rows = with_rows + 1
    end
  end
  local share = 1
  if with_rows > 0 then
    share = math.floor(model.generation.ubatch / with_rows)
    if share < 1 then
      share = 1
    end
  end

  local active: {Turn} = {}
  local lasts: {integer} = {}
  local inputs: {compute.Input} = {}
  for _, turn in ipairs(list) do
    if turn.live and not turn.waiting then
      local run = turn.run
      if run is generate.Run then
        local rows = run.next(run, share)
        if rows is {compute.Input} then
          for _, input in ipairs(rows) do
            inputs[#inputs + 1] = input
          end
          active[#active + 1] = turn
          lasts[#active] = #inputs
        end
      end
    end
  end

  if #inputs > 0 then
    local rows = model.generation:pass(inputs)
    if rows is nil then
      for _, turn in ipairs(active) do
        lost(turn.ready)
        dropped(server, turn)
      end
    else
      local planes = rows as {{number}}
      for index, turn in ipairs(active) do
        local run = turn.run as generate.Run
        local piece, accept_line = run.accept(run, planes[lasts[index]])
        if accept_line is string then
          lost(turn.ready)
          dropped(server, turn)
        else
          if piece is string then
            local data = event_data({index = 0, delta = {content = piece}})
            if data is nil then
              lost(turn.ready)
              dropped(server, turn)
            else
              turn.emit(data)
            end
          end
          if turn.live and run.done then
            local reason = "stop"
            if #run.produced >= turn.ready.request.max_tokens then
              reason = "length"
            end
            local ending = event_data({index = 0, finish_reason = reason})
            if ending is nil then
              lost(turn.ready)
              dropped(server, turn)
            else
              turn.emit(ending)
              turn.emit("[DONE]")
              kept(turn.ready, run.produced, run.cut)
              dropped(server, turn)
            end
          end
        end
      end
    end
  end

  local live = 0
  for _, turn in ipairs(list) do
    if turn.live then
      live = live + 1
    end
  end
  return live
end

function serve.round(server: tltp.Server): string | nil
  local list = turns[server]
  local timeout = -1
  if list is {Turn} and #list > 0 then
    timeout = 0
  end
  local state = pools[server]
  if state is PoolState and #state.turns > 0 then
    timeout = 0
  end
  local _, line = server:poll(timeout)
  serve.step(server)
  return line
end

function serve.answer(server: tltp.Server, body: string): serve.Reply
  local state = pools[server]
  if state is PoolState then
    return pool_answer(server, state, body)
  end
  local ready, refusal = prepared(server, body)
  if ready is nil then
    return as_reply(refusal as Fault)
  end
  if ready.waiting then
    return as_reply(too_many(server))
  end
  return finished(ready)
end

function serve.stream(
  server: tltp.Server, body: string
): tltp.Response, tltp.StreamFn | nil
  local state = pools[server]
  if state is PoolState then
    return pool_stream(server, state, body)
  end
  local ready, refusal = prepared(server, body)
  if ready is nil then
    local reply = as_reply(refusal as Fault)
    return server:json(reply.status, reply.body), nil
  end

  if not ready.request.stream then
    local reply = as_reply(too_many(server))
    if not ready.waiting then
      reply = finished(ready)
    end
    return server:json(reply.status, reply.body), nil
  end

  -- a turn that has to wait builds its run when a slot frees, not now
  local run: generate.Run | nil = nil
  if not ready.waiting then
    local run_line: string | nil = nil
    run, run_line = generate.open_ids(
      ready.model.generation,
      ready.model.tokenizer,
      ready.ids,
      options_of(ready),
      ready.sequence
    )
    if run is nil then
      released(ready)
      local reply = as_reply(internal(run_line as string))
      return server:json(reply.status, reply.body), nil
    end
  end
  local stream_fn = producer(server, ready, run)
  return server:sse(stream_fn), stream_fn
end

local function one_line(text: string): string
  return (text:gsub("%s+$", "")) .. "\n"
end

local function install(server: tltp.Server, name: string)
  server:get(
    "/health",
    function(asked: tltp.Server): tltp.Response
      return asked:json(
        "OK", tltp.json_encode({status = "ok", model = name}) as string
      )
    end
  )

  server:post(
    CHAT_PATH,
    function(asked: tltp.Server): tltp.Response
      local response = serve.stream(asked, asked.req.body)
      return response
    end
  )

  server.handle_404 = function(asked: tltp.Server): tltp.Response
    return asked:json("NotFound", error_body("not found", "not_found"))
  end

  server.default_route = function(asked: tltp.Server): tltp.Response
    return asked:handle_404()
  end

  server.on_error = function(
    asked: tltp.Server, _err: string, _trace?: string
  ): tltp.Response
    return asked:json(
      "InternalServerError", error_body("internal error", "internal_error")
    )
  end
end

local function endpoint_json(endpoint: serve.Endpoint): string
  local fields: EndpointFields = {
    host = endpoint.host,
    port = endpoint.port,
    name = endpoint.name,
    path = endpoint.path,
    sessions = endpoint.sessions,
    kv_cache = endpoint.kv_cache,
    context_path = endpoint.context_path,
    context_bytes = endpoint.context_bytes,
    ubatch = endpoint.ubatch,
    vram = endpoint.vram,
    profile = endpoint.profile,
    prefetch = endpoint.prefetch,
    threads = endpoint.threads,
  }
  return tltp.json_encode(fields) as string
end

local function slot_under(base: string): string | nil
  if not fs.isdir(base) then
    return nil
  end
  for _ = 1, 32 do
    local made = os.tmpname()
    os.remove(made)
    local dir = base .. "/" .. (made:match("[^/]+$") as string)
    if fs.mkdir(dir) then
      return dir
    end
  end
  return nil
end

-- the pool's scratch directory: made by this call or not at all, and readable
-- only by this process, because every request body and reply passes through it
local function scratch_directory(): string | nil, string | nil
  -- a set-but-unusable XDG_RUNTIME_DIR (a read-only /run in a container, a
  -- runtime directory this user does not own) must not take the pool down, so
  -- every candidate is tried rather than the first one that is merely set
  local bases: {string} = {}
  local runtime = os.getenv("XDG_RUNTIME_DIR")
  local temp = os.getenv("TMPDIR")
  if runtime is string and runtime ~= "" then
    bases[#bases + 1] = runtime
  end
  if temp is string and temp ~= "" then
    bases[#bases + 1] = temp
  end
  bases[#bases + 1] = "/tmp"
  for _, base in ipairs(bases) do
    local dir = slot_under(base)
    if dir is string then
      -- 0700: fs.mkdir applies the umask, which leaves the directory readable
      -- by every local user on a machine with no runtime directory
      if not fs.chmod(dir, 448) then
        fs.rmdir(dir)
        return nil, "build: cannot restrict " .. dir
      end
      return dir, nil
    end
  end
  return nil,
    "build: cannot make a scratch directory under " .. table.concat(bases, ", ")
end

local function pool_build(
  endpoint: serve.Endpoint, count: integer, spawn: process.Spawn
): tltp.Server | nil, string | nil
  local context_path = endpoint.context_path
  if context_path is nil then
    return nil, "build: workers need a context: no context_path"
  end
  fs.mkdirAll(context_path)
  if not fs.isdir(context_path) then
    return nil, "build: cannot create " .. context_path
  end

  local backend, backend_failure = compute.load()
  if backend is nil then
    return nil, backend_failure
  end
  local free = backend:memory()
  compute.unload(backend)
  if free is integer then
    local mapped, map_failure = weights.load(endpoint.path)
    if mapped is nil then
      return nil, map_failure
    end
    local each = compute.place(mapped, EVERY_FIXED, 0).resident_bytes
    mapped.release(mapped)
    local needed = each * count
    if needed > free then
      return nil,
        "build: "
          .. tostring(count)
          .. " workers need "
          .. tostring(needed)
          .. " bytes; the device has "
          .. tostring(free)
          .. " free"
    end
  end

  local dir, dir_line = scratch_directory()
  if dir is nil then
    return nil, dir_line as string
  end
  for slot = 0, count - 1 do
    fs.mkdirAll(dir .. "/w" .. tostring(slot))
  end
  local watcher = fs.watch(
    dir, function(_event: fs.WatchEvent, _name: string) end, {recursive = true}
  )

  local server = tltp.Server.new(
    {port = endpoint.port, bind_host = endpoint.host, logging = false}
  )
  pools[server] = {
    dir = dir,
    workers = {},
    turns = {},
    next_turn = 0,
    watcher = watcher,
    spawn = spawn,
    endpoint_json = endpoint_json(endpoint),
    limit = endpoint.sessions,
  }
  install(server, endpoint.name)

  for slot = 0, count - 1 do
    local spawn_failure = pool_spawn(pools[server] as PoolState, slot, false)
    if spawn_failure is string then
      serve.stop(server)
      return nil, "build: slot " .. tostring(slot) .. ": " .. spawn_failure
    end
  end

  local fault = pool_built(pools[server] as PoolState)
  if fault is string then
    serve.stop(server)
    return nil, "build: " .. fault
  end
  return server, nil
end

function serve.pool_of(server: tltp.Server): serve.Pool | nil
  local state = pools[server]
  if state is nil then
    return nil
  end
  local workers: {serve.Worker} = {}
  for _, worker in ipairs(state.workers) do
    local sessions: {string} = {}
    for index = 1, #worker.sessions do
      sessions[index] = worker.sessions[index]
    end
    workers[#workers + 1] = {
      slot = worker.slot,
      pid = worker.pid,
      sessions = sessions,
    }
  end
  return {dir = state.dir, workers = workers}
end

-- gives every process, mapping and cache the server owns back, whether it runs
-- the model itself or behind a pool
function serve.stop(server: tltp.Server)
  local state = pools[server]
  if state is PoolState then
    pools[server] = nil
    local watcher = state.watcher
    if watcher is fs.Watcher then
      watcher:close()
    end
    for _, worker in ipairs(state.workers) do
      worker.child:kill()
    end
    for _, worker in ipairs(state.workers) do
      worker.child:wait()
    end
    fs.rmdir(state.dir)
  end

  local model = models[server]
  if model is serve.Model then
    model.generation:close()
    model.handle.release(model.handle)
    compute.unload(model.backend)
  end
  models[server] = nil
  stores[server] = nil
  turns[server] = nil
  reservations[server] = nil
  held[server] = nil
end

function serve.build(
  endpoint: serve.Endpoint, spawn?: process.Spawn, write_out?: function(string)
): tltp.Server | nil, string | nil
  local count = endpoint.workers
  if count is integer and count > 1 then
    local launcher = spawn
    if launcher is nil then
      launcher = process.spawn
    end
    return pool_build(endpoint, count as integer, launcher)
  end
  local context_path = endpoint.context_path
  if context_path is string then
    fs.mkdirAll(context_path)
    if not fs.isdir(context_path) then
      return nil, "build: cannot create " .. context_path
    end
  end

  -- one walk of the metadata, shared by the weights and the vocabulary
  local info, info_failure = gguf.read(endpoint.path)
  if info is nil then
    return nil, info_failure
  end

  local handle, failure = weights.load(endpoint.path, info)
  if handle is nil then
    return nil, failure
  end

  local tokens, token_failure = tokenizer.load(endpoint.path, info)
  if tokens is nil then
    return nil, token_failure
  end

  local backend, backend_failure = compute.load()
  if backend is nil then
    return nil, backend_failure
  end

  local generation, generation_failure = compute.model(
    backend,
    handle,
    nil,
    endpoint.kv_cache,
    endpoint.sessions,
    endpoint.ubatch,
    endpoint.vram,
    endpoint.profile,
    endpoint.prefetch,
    endpoint.threads or cores.machine()
  )
  if generation is nil then
    return nil, generation_failure
  end

  if write_out is function then
    local placement = generation.placement
    local slots = 0
    for _, each_slot in ipairs(placement.slots) do
      slots = slots + each_slot
    end
    local name = "none"
    local found = backend.device
    if found is string then
      name = found
    end
    write_out(
      one_line(
        "device: "
          .. name
          .. ", "
          .. tostring(placement.budget)
          .. " bytes budget, "
          .. tostring(placement.resident_bytes)
          .. " bytes held, "
          .. tostring(slots)
          .. " expert slots in "
          .. tostring(placement.slot_bytes)
          .. " bytes, "
          .. tostring(placement.unplaced_bytes)
          .. " bytes from the mapping"
      )
    )
  end

  local server = tltp.Server.new(
    {port = endpoint.port, bind_host = endpoint.host, logging = false}
  )
  models[server] = {
    name = endpoint.name,
    path = endpoint.path,
    handle = handle,
    tokenizer = tokens,
    backend = backend,
    kv_cache = endpoint.kv_cache,
    generation = generation,
    context_path = endpoint.context_path,
    context_bytes = endpoint.context_bytes,
  }
  stores[server] = {held = {}, order = {}, limit = endpoint.sessions}
  install(server, endpoint.name)
  return server, nil
end

function serve.start(
  endpoint: serve.Endpoint, write_err: function(string), spawn?: process.Spawn
): integer
  local server, failure = serve.build(endpoint, spawn, write_err)
  if server is nil then
    write_err(one_line(failure as string))
    return 1
  end

  local raised: string | nil
  local ok, code = xpcall(
    function(): integer
      local result = 0
      while result == 0 do
        local line = serve.round(server as tltp.Server)
        if line is string then
          write_err(one_line(line))
          result = 1
        end
      end
      return result
    end,
    function(reason: string): string
      raised = tostring(reason)
      return raised
    end
  )
  serve.stop(server as tltp.Server)
  if not ok then
    write_err(one_line(raised or "server failed"))
    return 1
  end
  return code as integer
end

return serve
