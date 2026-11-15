local chat = require("tllm.chat")
local compute = require("tllm.compute")
local generate = require("tllm.generate")
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
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
    sessions: integer
    kv_cache: compute.CacheType | nil
  end

  record Session
    id: string
    length: integer
    generation: compute.Model
  end

  record Reply
    status: tltp.Status
    body: string
  end

  model_of: function(tltp.Server): Model | nil
  build: function(Endpoint): tltp.Server | nil, string | nil
  start: function(Endpoint, function(string)): integer
  answer: function(tltp.Server, string): Reply
  stream: function(tltp.Server, string): tltp.Response, tltp.StreamFn | nil
  session_of: function(tltp.Server, string): Session | nil
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
  max_tokens: integer
  seed: integer
  stream: boolean
  session_id: string | nil
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
  generation: compute.Model
  cached: integer
  store: Store | nil
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
  run: generate.Run
  emit: tltp.EmitFn
  live: boolean
end

local DEFAULT_BUDGET: integer = 256
local CHAT_PATH = "/v1/chat/completions"

local models: {tltp.Server: serve.Model} = {}
local stores: {tltp.Server: Store} = {}
local turns: {tltp.Server: {Turn}} = {}
local in_flight: {tltp.Server: {string: boolean}} = {}

function serve.model_of(server: tltp.Server): serve.Model | nil
  return models[server]
end

function serve.session_of(server: tltp.Server, id: string): serve.Session | nil
  local store = stores[server]
  if store is nil then
    return nil
  end
  local held = store.held[id]
  if held is nil then
    return nil
  end
  return held.session
end

local function error_body(message: string, kind: string): string
  return tltp.json_encode({error = {message = message, type = kind}}) as string
end

local function rejected(message: string): serve.Reply
  return {
    status = "BadRequest",
    body = error_body(message, "invalid_request_error"),
  }
end

local function failed(message: string): serve.Reply
  return {
    status = "InternalServerError",
    body = error_body(message, "internal_error"),
  }
end

local function entry_fault(index: integer): string
  return "messages: entry "
    .. tostring(index)
    .. " is not an object with string role and content"
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
    local held = entry as Fields
    local role = held.role
    local content = held.content
    if not (role is string) or not (content is string) then
      return nil, entry_fault(index)
    end
    messages[index] = {role = role as string, content = content as string}
  end
  return messages, nil
end

local function whole(value: Loose): integer | nil
  if not (value is number) or value % 1 ~= 0 then
    return nil
  end
  return value as integer
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
  if not (temperature_value is nil) then
    if not (temperature_value is number) then
      return nil, "temperature: not a number"
    end
    if temperature_value < 0 then
      return nil, "temperature: below zero"
    end
    temperature = temperature_value
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
    local held = whole(seed_value)
    if held is nil then
      return nil, "seed: not a whole number"
    end
    seed = held
  end

  local stream = false
  local stream_value = fields.stream
  if not (stream_value is nil) then
    if not (stream_value is boolean) then
      return nil, "stream: not a boolean"
    end
    stream = stream_value as boolean
  end

  local session_id: string | nil = nil
  local session_value = fields.session_id
  if not (session_value is nil) then
    if not (session_value is string) then
      return nil, "session_id: not a string"
    end
    session_id = session_value as string
  end

  return {
    messages = messages,
    temperature = temperature,
    top_k = top_k,
    max_tokens = budget,
    seed = seed,
    stream = stream,
    session_id = session_id,
  }, nil
end

local function budget_fault(
  ids: integer, budget: integer, positions: integer
): serve.Reply
  return rejected(
    "messages: the prompt's "
      .. tostring(ids)
      .. " ids plus the max_tokens budget "
      .. tostring(budget)
      .. " minus one is past the "
      .. tostring(positions)
      .. " positions the file allows"
  )
end

local function marked(server: tltp.Server, id: string): boolean
  local marks = in_flight[server]
  return marks ~= nil and marks[id] == true
end

local function unmark(server: tltp.Server, id: string)
  local marks = in_flight[server]
  if marks is nil then
    return
  end
  marks[id] = nil
end

local function begins_with(ids: {integer}, prefix: {integer}): boolean
  for index = 1, #prefix do
    if ids[index] ~= prefix[index] then
      return false
    end
  end
  return true
end

local function prepared(
  server: tltp.Server, body: string
): Prepared | nil, serve.Reply | nil
  local model = models[server]
  if model is nil then
    return nil, failed("model: this server holds none")
  end

  local request, request_line = request_of(body)
  if request is nil then
    return nil, rejected(request_line as string)
  end

  local prompt, render_line = chat.render(model.tokenizer, request.messages)
  if prompt is nil then
    return nil, failed(render_line as string)
  end

  local ids, encode_line = model.tokenizer:encode(prompt)
  if ids is nil then
    return nil, failed(encode_line as string)
  end

  local window, window_line = compute.model(
    model.backend, model.handle, nil, model.kv_cache
  )
  if window is nil then
    return nil, failed(window_line as string)
  end

  local budget = request.max_tokens
  local needed = #ids + budget - 1
  if needed > window.positions then
    return nil, budget_fault(#ids, budget, window.positions)
  end

  local id = request.session_id
  local store: Store | nil = nil
  local generation: compute.Model | nil = nil
  local cached = 0

  if id is string then
    store = stores[server]
    if store is Store then
      local held = store.held[id]
      if not (held is nil) then
        if
          held.session.length < #ids
          and needed <= held.session.generation.positions
          and begins_with(ids, held.ids)
          and not marked(server, id)
        then
          generation = held.session.generation
          cached = held.session.length
        end
      end
    end
  end

  if generation is nil then
    local taken = needed
    if id is string then
      taken = needed * 2
      if taken > window.positions then
        taken = window.positions
      end
    end
    local built, generation_line = compute.model(
      model.backend, model.handle, taken, model.kv_cache
    )
    if built is nil then
      return nil, failed(generation_line as string)
    end
    generation = built
  end

  if id is string then
    local marks = in_flight[server]
    if marks is nil then
      marks = {}
      in_flight[server] = marks
    end
    marks[id] = true
  end

  return {
    server = server,
    model = model,
    request = request,
    prompt = prompt,
    ids = ids,
    generation = generation,
    cached = cached,
    store = store,
  }, nil
end

local function options_of(ready: Prepared): generate.Options
  local request = ready.request
  return {
    max_tokens = request.max_tokens,
    seed = request.seed,
    temperature = request.temperature,
    top_k = request.top_k,
    start = ready.cached,
  }
end

local function forwarded(ready: Prepared, sampled: {integer}): {integer}
  local count = #sampled
  if count == ready.request.max_tokens then
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

local function released(ready: Prepared)
  local id = ready.request.session_id
  if id is string then
    unmark(ready.server, id)
  end
end

local function kept(ready: Prepared, sampled: {integer})
  local store = ready.store
  local id = ready.request.session_id
  if id is string and store is Store then
    local held = store.held[id]
    if held is nil then
      if #store.order >= store.limit then
        forgotten(store, store.order[1])
      end
      held = {
        ids = {},
        session = {
          id = id,
          length = 0,
          generation = ready.generation,
        },
      }
      store.held[id] = held
    end
    held.ids = forwarded(ready, sampled)
    held.session.length = #held.ids
    held.session.generation = ready.generation
    used(store, id)
  end
  released(ready)
end

local function lost(ready: Prepared)
  local store = ready.store
  local id = ready.request.session_id
  if id is string and store is Store then
    forgotten(store, id)
  end
  released(ready)
end

local function finished(ready: Prepared): serve.Reply
  local text, text_line, completion, sampled = generate.text(
    ready.generation, ready.model.tokenizer, ready.prompt, options_of(ready)
  )
  if text is nil then
    lost(ready)
    return failed(text_line as string)
  end

  kept(ready, sampled as {integer})
  return {
    status = "OK",
    body = tltp.json_encode(
      {
        choices = {
          {index = 0, message = {role = "assistant", content = text}},
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
  server: tltp.Server, ready: Prepared, run: generate.Run
): tltp.StreamFn
  return function(emit: tltp.EmitFn): boolean | nil
    local opening = event_data({index = 0, delta = {role = "assistant"}})
    if opening is nil then
      lost(ready)
      return true
    end
    emit(opening)

    local turn: Turn = {ready = ready, run = run, emit = emit, live = true}
    local list = registered(server)
    list[#list + 1] = turn

    coroutine.yield()
    while turn.live do
      coroutine.yield()
    end
    return true
  end
end

local function advance(server: tltp.Server, turn: Turn)
  local run = turn.run
  local piece, line = run.step(run)
  if not (line is nil) then
    lost(turn.ready)
    dropped(server, turn)
    return
  end

  if not (piece is nil) then
    local data = event_data({index = 0, delta = {content = piece}})
    if data is nil then
      lost(turn.ready)
      dropped(server, turn)
      return
    end
    turn.emit(data)
  end

  if not run.done then
    return
  end

  local reason = "stop"
  if #run.produced >= turn.ready.request.max_tokens then
    reason = "length"
  end
  local ending = event_data({index = 0, finish_reason = reason})
  if ending is nil then
    lost(turn.ready)
    dropped(server, turn)
    return
  end
  turn.emit(ending)
  turn.emit("[DONE]")
  kept(turn.ready, run.produced)
  dropped(server, turn)
end

function serve.step(server: tltp.Server): integer
  local list = turns[server]
  if list is nil then
    return 0
  end
  local order: {Turn} = {}
  for index = 1, #list do
    order[index] = list[index]
  end
  local live = 0
  for _, turn in ipairs(order) do
    if turn.live then
      advance(server, turn)
    end
    if turn.live then
      live = live + 1
    end
  end
  return live
end

function serve.round(server: tltp.Server): string | nil
  local list = turns[server]
  local timeout = -1
  if not (list is nil) and #list > 0 then
    timeout = 0
  end
  local _, line = server:poll(timeout)
  serve.step(server)
  return line
end

function serve.answer(server: tltp.Server, body: string): serve.Reply
  local ready, failure = prepared(server, body)
  if ready is nil then
    return failure as serve.Reply
  end
  return finished(ready)
end

function serve.stream(
  server: tltp.Server, body: string
): tltp.Response, tltp.StreamFn | nil
  local ready, failure = prepared(server, body)
  if ready is nil then
    local fault = failure as serve.Reply
    return server:json(fault.status, fault.body), nil
  end

  if not ready.request.stream then
    local reply = finished(ready)
    return server:json(reply.status, reply.body), nil
  end

  local run, open_line = generate.open(
    ready.generation, ready.model.tokenizer, ready.prompt, options_of(ready)
  )
  if run is nil then
    released(ready)
    local fault = failed(open_line as string)
    return server:json(fault.status, fault.body), nil
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
    function(self: tltp.Server): tltp.Response
      return self:json(
        "OK", tltp.json_encode({status = "ok", model = name}) as string
      )
    end
  )

  server:post(
    CHAT_PATH,
    function(self: tltp.Server): tltp.Response
      local response = serve.stream(self, self.req.body)
      return response
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

  local tokens, token_failure = tokenizer.load(endpoint.path)
  if tokens is nil then
    return nil, token_failure
  end

  local backend, backend_failure = compute.load()
  if backend is nil then
    return nil, backend_failure
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
  }
  stores[server] = {held = {}, order = {}, limit = endpoint.sessions}
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
  local ok, code = xpcall(
    function(): integer
      local result = 0
      while result == 0 do
        local line = serve.round(server as tltp.Server)
        if not (line is nil) then
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
  if not ok then
    write_err(one_line(raised or "server failed"))
    return 1
  end
  return code as integer
end

return serve
