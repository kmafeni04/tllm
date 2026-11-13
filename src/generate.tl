local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")

local record generate
  record Options
    max_tokens: integer
    seed: integer
    temperature: number | nil
    top_k: integer | nil
    end_id: integer | nil
    on_token: function(string) | nil
  end

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options
  ): string | nil, string | nil, integer | nil
end

local type Options = generate.Options

local STATE_MODULUS = 4294967296
local STATE_MULTIPLIER = 1664525
local STATE_INCREMENT = 1013904223

local record Stream
  state: number
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
  return nil
end

function generate.text(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options
): string | nil, string | nil, integer | nil
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

  local end_id = options.end_id
  if end_id is nil then
    end_id = tokens.end_id
  end

  local scores: {number} | nil = nil
  local line: string | nil = nil
  for index, id in ipairs(ids) do
    scores, line = model:forward(id, index - 1)
    if scores is nil then
      return nil, line
    end
  end

  local stream: Stream = {state = options.seed % STATE_MODULUS}
  local produced: {integer} = {}
  local position = #ids
  while #produced < options.max_tokens do
    local token_id = chosen(scores as {number}, options, stream)
    if token_id == end_id then
      break
    end
    produced[#produced + 1] = token_id
    local piece, piece_line = tokens:decode({token_id})
    if piece is nil then
      return nil, piece_line
    end
    local on_token = options.on_token
    if not (on_token is nil) then
      on_token(piece)
    end
    if #produced >= options.max_tokens then
      break
    end
    scores, line = model:forward(token_id, position)
    if scores is nil then
      return nil, line
    end
    position = position + 1
  end

  local answer, decode_line = tokens:decode(produced)
  if answer is nil then
    return nil, decode_line
  end
  return answer, nil, #produced
end

return generate
