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
    step: function(Run): string | nil, string | nil
  end

  open: function(
    compute.Model, tokenizer.Tokenizer, string, Options
  ): Run | nil, string | nil

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options
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
  end_id: integer | nil
  max_tokens: integer
  options: Options
  stream: Stream
  position: integer
  index: integer
  scores: {number} | nil
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

function generate.open(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options
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
    end_id = end_id,
    max_tokens = options.max_tokens,
    options = options,
    stream = {state = options.seed % STATE_MODULUS},
    position = start,
    index = start + 1,
    scores = nil,
  }

  local produced: {integer} = {}

  local function step(self: generate.Run): string | nil, string | nil
    if self.done then
      return nil, nil
    end

    if progress.index <= #progress.ids then
      local scores, forward_line = progress.model:forward(
        progress.ids[progress.index], progress.position
      )
      if scores is nil then
        self.done = true
        self.failure = forward_line
        return nil, forward_line
      end
      progress.scores = scores
      progress.position = progress.position + 1
      progress.index = progress.index + 1
      return nil, nil
    end

    local scores = progress.scores as {number}
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

    local scores_next, next_line = progress.model:forward(
      token_id, progress.position
    )
    if scores_next is nil then
      self.done = true
      self.failure = next_line
      return nil, next_line
    end
    progress.scores = scores_next
    progress.position = progress.position + 1
    return piece, nil
  end

  local run: generate.Run = {
    done = false,
    failure = nil,
    produced = produced,
    step = step,
  }
  return run, nil
end

function generate.text(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options
): string | nil, string | nil, integer | nil, {integer} | nil
  local run, open_line = generate.open(model, tokens, prompt, options)
  if run is nil then
    return nil, open_line
  end

  while not run.done do
    local _, step_line = run.step(run)
    if not (step_line is nil) then
      return nil, step_line
    end
  end

  local answer, decode_line = tokens:decode(run.produced)
  if answer is nil then
    return nil, decode_line
  end
  return answer, nil, #run.produced, run.produced
end

return generate
