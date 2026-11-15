local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")

local record generate
  record Options
    max_tokens: integer
    seed: integer
    temperature: number | nil
    top_k: integer | nil
    end_id: integer | nil
    start: integer | nil
  end

  record Run
    done: boolean
    failure: string | nil
    produced: {integer}
    next: function(Run): compute.Input | nil
    accept: function(Run, {number}): string | nil, string | nil
    fail: function(Run, string)
  end

  open: function(
    compute.Model, tokenizer.Tokenizer, string, Options, sequence?: string
  ): Run | nil, string | nil

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options, sequence?: string
  ): string | nil, string | nil, integer | nil, {integer} | nil
end

local type Options = generate.Options

local STATE_MODULUS = 4294967296
local STATE_MULTIPLIER = 1664525
local STATE_INCREMENT = 1013904223

local record Stream
  state: number
end

local record Progress
  model: compute.Model
  tokens: tokenizer.Tokenizer
  ids: {integer}
  sequence: string
  end_id: integer | nil
  max_tokens: integer
  options: Options
  stream: Stream
  position: integer
  index: integer
  pending: integer | nil
end

local function drawn(stream: Stream): number
  stream.state = (
    STATE_MULTIPLIER * stream.state + STATE_INCREMENT
  ) % STATE_MODULUS
  return stream.state / STATE_MODULUS
end

local function greediest(scores: {number}): integer
  local best = 1
  for index = 2, #scores do
    if scores[index] > scores[best] then
      best = index
    end
  end
  return best - 1
end

local function by_score(scores: {number}): {integer}
  local order: {integer} = {}
  for index = 1, #scores do
    order[index] = index
  end
  table.sort(
    order,
    function(left: integer, right: integer): boolean
      local first = scores[left]
      local second = scores[right]
      if first ~= second then
        return first > second
      end
      return left < right
    end
  )
  return order
end

local function sampled(
  scores: {number}, options: Options, stream: Stream
): integer
  local temperature = options.temperature as number
  local order = by_score(scores)
  local count = #order
  local top_k = options.top_k
  if not (top_k is nil) and top_k < count then
    count = top_k
  end
  local top = scores[order[1]]
  local weights: {number} = {}
  local total: number = 0
  for place = 1, count do
    local weight = math.exp((scores[order[place]] - top) / temperature)
    weights[place] = weight
    total = total + weight
  end
  local draw = drawn(stream)
  local running: number = 0
  for place = 1, count do
    running = running + weights[place] / total
    if running > draw then
      return order[place] - 1
    end
  end
  return order[count] - 1
end

local function chosen(
  scores: {number}, options: Options, stream: Stream
): integer
  local temperature = options.temperature
  if temperature is nil or temperature == 0 then
    return greediest(scores)
  end
  return sampled(scores, options, stream)
end

local function faulted(options: Options): string | nil
  if options.max_tokens < 1 then
    return "generate: max_tokens "
      .. tostring(options.max_tokens)
      .. " is below one"
  end
  local temperature = options.temperature
  if not (temperature is nil) and temperature < 0 then
    return "generate: temperature " .. tostring(temperature) .. " is below zero"
  end
  local top_k = options.top_k
  if not (top_k is nil) and top_k < 1 then
    return "generate: top_k " .. tostring(top_k) .. " is below one"
  end
  local start = options.start
  if not (start is nil) and start < 0 then
    return "generate: start " .. tostring(start) .. " is below zero"
  end
  return nil
end

-- a name the model already holds is the one refusal a caller can safely ignore
local function ensure_reserved(
  model: compute.Model, sequence: string
): string | nil
  local line = model:reserve(sequence, model.positions)
  if line is nil then
    return nil
  end
  if line:find("is already held", 1, true) ~= nil then
    return nil
  end
  return line
end

function generate.open(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options,
  sequence?: string
): generate.Run | nil, string | nil
  local fault = faulted(options)
  if not (fault is nil) then
    return nil, fault
  end

  local ids, encode_line = tokens:encode(prompt)
  if ids is nil then
    return nil, encode_line
  end
  if #ids == 0 then
    return nil, "generate: prompt holds no token ids"
  end

  local start = options.start or 0
  if start >= #ids then
    return nil,
      "generate: start "
        .. tostring(start)
        .. " is not below the prompt's "
        .. tostring(#ids)
        .. " ids"
  end

  local end_id = options.end_id
  if end_id is nil then
    end_id = tokens.end_id
  end

  local progress: Progress = {
    model = model,
    tokens = tokens,
    ids = ids,
    sequence = sequence or "",
    end_id = end_id,
    max_tokens = options.max_tokens,
    options = options,
    stream = {state = options.seed % STATE_MODULUS},
    position = start,
    index = start + 1,
    pending = nil,
  }

  local produced: {integer} = {}

  local function next_input(self: generate.Run): compute.Input | nil
    if self.done then
      return nil
    end
    if progress.index <= #progress.ids then
      return {
        sequence = progress.sequence,
        token = progress.ids[progress.index],
        position = progress.position,
      }
    end
    local token = progress.pending
    if token is nil then
      return nil
    end
    return {
      sequence = progress.sequence,
      token = token,
      position = progress.position,
    }
  end

  local function accept(
    self: generate.Run, scores: {number}
  ): string | nil, string | nil
    if self.done then
      return nil, nil
    end

    local was_prompt = progress.index <= #progress.ids
    progress.position = progress.position + 1
    if was_prompt then
      progress.index = progress.index + 1
    else
      progress.pending = nil
    end

    if progress.index <= #progress.ids then
      return nil, nil
    end

    local token_id = chosen(scores, progress.options, progress.stream)
    if token_id == progress.end_id then
      self.done = true
      return nil, nil
    end

    produced[#produced + 1] = token_id
    local piece, piece_line = progress.tokens:decode({token_id})
    if piece is nil then
      self.done = true
      self.failure = piece_line
      return nil, piece_line
    end
    if #produced >= progress.max_tokens then
      self.done = true
      return piece, nil
    end
    progress.pending = token_id
    return piece, nil
  end

  local function fail(self: generate.Run, line: string)
    self.done = true
    self.failure = line
  end

  local run: generate.Run = {
    done = false,
    failure = nil,
    produced = produced,
    next = next_input,
    accept = accept,
    fail = fail,
  }
  return run, nil
end

function generate.text(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options,
  sequence?: string
): string | nil, string | nil, integer | nil, {integer} | nil
  local run, open_line = generate.open(model, tokens, prompt, options, sequence)
  if run is nil then
    return nil, open_line
  end

  local name = sequence or ""
  local reserve_line = ensure_reserved(model, name)
  if not (reserve_line is nil) then
    return nil, reserve_line
  end

  while not run.done do
    local input = run.next(run)
    if input is nil then
      break
    end
    local rows, pass_line = model:pass({input})
    if rows is nil then
      run.fail(run, pass_line as string)
      break
    end
    local scores = rows as {{number}}
    local _, accept_line = run.accept(run, scores[1])
    if not (accept_line is nil) then
      break
    end
  end

  local failure = run.failure
  if not (failure is nil) then
    return nil, failure
  end

  local answer, decode_line = tokens:decode(run.produced)
  if answer is nil then
    return nil, decode_line
  end
  return answer, nil, #run.produced, run.produced
end

return generate
