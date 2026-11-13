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
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
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
end

local record Prepared
  model: serve.Model
  request: Request
  prompt: string
  ids: {integer}
  generation: compute.Model
end

local record EventChoice
  index: integer
  delta: {string: string} | nil
  finish_reason: string | nil
end

local record Event
  choices: {EventChoice}
end

local DEFAULT_BUDGET: integer = 256
local CHAT_PATH = "/v1/chat/completions"

local models: {tltp.Server: serve.Model} = {}

function serve.model_of(server: tltp.Server): serve.Model | nil
  return models[server]
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

  return {
    messages = messages,
    temperature = temperature,
    top_k = top_k,
    max_tokens = budget,
    seed = seed,
    stream = stream,
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

  local window, window_line = compute.model(model.backend, model.handle, nil)
  if window is nil then
    return nil, failed(window_line as string)
  end

  local budget = request.max_tokens
  local needed = #ids + budget - 1
  if needed > window.positions then
    return nil, budget_fault(#ids, budget, window.positions)
  end

  local generation, generation_line = compute.model(
    model.backend, model.handle, needed
  )
  if generation is nil then
    return nil, failed(generation_line as string)
  end

  return {
    model = model,
    request = request,
    prompt = prompt,
    ids = ids,
    generation = generation,
  }, nil
end

local function options_of(
  ready: Prepared, on_token: function(string) | nil
): generate.Options
  local request = ready.request
  return {
    max_tokens = request.max_tokens,
    seed = request.seed,
    temperature = request.temperature,
    top_k = request.top_k,
    on_token = on_token,
  }
end

local function finished(ready: Prepared): serve.Reply
  local text, text_line, sampled = generate.text(
    ready.generation,
    ready.model.tokenizer,
    ready.prompt,
    options_of(ready, nil)
  )
  if text is nil then
    return failed(text_line as string)
  end

  local completion = sampled as integer
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
        },
      }
    ) as string,
  }
end

local function event_data(choice: EventChoice): string | nil
  local data = tltp.json_encode({choices = {choice}} as Event)
  return data
end

local function producer(ready: Prepared): tltp.StreamFn
  return function(emit: tltp.EmitFn): boolean | nil
    local opening = event_data({index = 0, delta = {role = "assistant"}})
    if opening is nil then
      return
    end
    emit(opening)

    local broken = false
    local text, _, sampled = generate.text(
      ready.generation,
      ready.model.tokenizer,
      ready.prompt,
      options_of(
        ready,
        function(piece: string)
          if broken then
            return
          end
          local data = event_data({index = 0, delta = {content = piece}})
          if data is nil then
            broken = true
            return
          end
          emit(data)
        end
      )
    )
    if broken or text is nil then
      return
    end

    local reason = "stop"
    if (sampled as integer) >= ready.request.max_tokens then
      reason = "length"
    end
    local ending = event_data({index = 0, finish_reason = reason})
    if ending is nil then
      return
    end
    emit(ending)
    emit("[DONE]")
    return true
  end
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

  local run = producer(ready)
  return server:sse(run), run
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
  }
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
  local ok = xpcall(
    function() server:serve() end,
    function(reason: string) raised = tostring(reason) end
  )
  if not ok then
    write_err(one_line(raised or "server failed"))
    return 1
  end
  return 0
end

return serve
