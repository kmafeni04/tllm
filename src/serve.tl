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
    generation: compute.Model
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
    sequence: string
    positions: integer
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
  sequence: string
  positions: integer
  cached: integer
  store: Store | nil
  waiting: boolean
  reserved: boolean
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

local DEFAULT_BUDGET: integer = 256
local CHAT_PATH = "/v1/chat/completions"

local models: {tltp.Server: serve.Model} = {}
local stores: {tltp.Server: Store} = {}
local turns: {tltp.Server: {Turn}} = {}
local reservations: {tltp.Server: {string: boolean}} = {}
local held: {tltp.Server: {string: boolean}} = {}

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
    local fields = entry as Fields
    local role = fields.role
    local content = fields.content
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
    local seed_number = whole(seed_value)
    if seed_number is nil then
      return nil, "seed: not a whole number"
    end
    seed = seed_number
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

local function is_held(server: tltp.Server, name: string): boolean
  local marks = held[server]
  return marks ~= nil and marks[name] == true
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
  return marks ~= nil and marks[name] == true
end

local function free_slots(server: tltp.Server): integer
  local model = models[server]
  if model is nil then
    return 0
  end
  local count = 0
  local marks = reservations[server]
  if not (marks is nil) then
    for _ in pairs(marks) do
      count = count + 1
    end
  end
  return model.generation.sequences - count
end

local function release_name(server: tltp.Server, name: string)
  local model = models[server]
  if not (model is nil) then
    model.generation:release(name)
  end
  local marks = reservations[server]
  if not (marks is nil) then
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
      not (held_conversation is nil)
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

local function too_many(server: tltp.Server): serve.Reply
  local store = stores[server]
  local limit = 0
  if not (store is nil) then
    limit = store.limit
  end
  return {
    status = "TooManyRequests",
    body = error_body(
      "the server holds no free session; sessions is " .. tostring(limit),
      "too_many_requests"
    ),
  }
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
    if not (found is nil) then
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
    end
  end

  if not (found_session is nil) then
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

  if free_slots(server) < 1 then
    if not request.stream then
      return nil, too_many(server)
    end
    waiting = true
    sequence = id or ""
  else
    local name, reserve_line = reserve_sequence(server, id or "", positions)
    if name is nil then
      return nil, failed(reserve_line as string)
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

local function released(ready: Prepared)
  unmark_held(ready.server, ready.sequence)
  if ready.reserved then
    release_name(ready.server, ready.sequence)
    ready.reserved = false
  end
end

local function kept(ready: Prepared, sampled: {integer})
  local store = ready.store
  local id = ready.request.session_id
  if id is string and store is Store then
    local existing = store.held[id]
    if
      not (existing is nil) and existing.session.sequence ~= ready.sequence
    then
      forgotten(store, id)
      if not is_held(ready.server, existing.session.sequence) then
        release_name(ready.server, existing.session.sequence)
      end
    end
    if store.held[id] == nil and #store.order >= store.limit then
      evict_one(ready.server, store)
    end
    local ids = forwarded(ready, sampled)
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
    if
      not (existing is nil) and existing.session.sequence == ready.sequence
    then
      forgotten(store, id)
    end
  end
  released(ready)
end

local function finished(ready: Prepared): serve.Reply
  local text, text_line, completion, sampled = generate.text(
    ready.model.generation,
    ready.model.tokenizer,
    ready.prompt,
    options_of(ready),
    ready.sequence
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

local function admit(server: tltp.Server, turn: Turn): boolean
  local ready = turn.ready
  local name = reserve_sequence(server, ready.sequence, ready.positions)
  if name is nil then
    lost(ready)
    dropped(server, turn)
    return false
  end
  ready.sequence = name
  ready.reserved = true
  mark_held(server, name)

  local run = generate.open(
    ready.model.generation,
    ready.model.tokenizer,
    ready.prompt,
    options_of(ready),
    name
  )
  if run is nil then
    lost(ready)
    dropped(server, turn)
    return false
  end
  turn.run = run
  turn.waiting = false
  return true
end

function serve.step(server: tltp.Server): integer
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

  local active: {Turn} = {}
  local inputs: {compute.Input} = {}
  for _, turn in ipairs(list) do
    if turn.live and not turn.waiting then
      local run = turn.run
      if not (run is nil) then
        local input = run.next(run)
        if not (input is nil) then
          active[#active + 1] = turn
          inputs[#inputs + 1] = input
        end
      end
    end
  end

  if #inputs > 0 then
    local model = models[server] as serve.Model
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
        local piece, accept_line = run.accept(run, planes[index])
        if not (accept_line is nil) then
          lost(turn.ready)
          dropped(server, turn)
        else
          if not (piece is nil) then
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
              kept(turn.ready, run.produced)
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
  if ready.waiting then
    return too_many(server)
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
    if ready.waiting then
      local fault = too_many(server)
      return server:json(fault.status, fault.body), nil
    end
    local reply = finished(ready)
    return server:json(reply.status, reply.body), nil
  end

  local run, open_line = generate.open(
    ready.model.generation,
    ready.model.tokenizer,
    ready.prompt,
    options_of(ready),
    ready.sequence
  )
  if run is nil then
    released(ready)
    local fault = failed(open_line as string)
    return server:json(fault.status, fault.body), nil
  end

  local held_run: generate.Run | nil = run
  if ready.waiting then
    held_run = nil
  end
  local stream_fn = producer(server, ready, held_run)
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

  local generation, generation_failure = compute.model(
    backend, handle, nil, endpoint.kv_cache, endpoint.sessions
  )
  if generation is nil then
    return nil, generation_failure
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
