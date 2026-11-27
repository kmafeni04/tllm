local clock = require("tllm.clock")
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")

local record generate
  record Options
    max_tokens: integer
    seed: integer
    temperature: number | nil
    top_k: integer | nil
    top_p: number | nil
    min_p: number | nil
    repetition_penalty: number | nil
    presence_penalty: number | nil
    frequency_penalty: number | nil
    end_id: integer | nil
    start: integer | nil
    stop: {string} | nil
  end

  record Run
    done: boolean
    cut: boolean
    failure: string | nil
    produced: {integer}
    next: function(Run, integer): {compute.Input} | nil
    accept: function(Run, {number}): string | nil, string | nil
    fail: function(Run, string)
  end

  record Text
    text: string
    sampled: {integer}
    cut: boolean
  end

  open: function(
    compute.Model, tokenizer.Tokenizer, string, Options, sequence?: string
  ): Run | nil, string | nil

  open_ids: function(
    compute.Model, tokenizer.Tokenizer, {integer}, Options, sequence?: string
  ): Run | nil, string | nil

  text: function(
    compute.Model, tokenizer.Tokenizer, string, Options, sequence?: string
  ): Text | nil, string | nil

  text_ids: function(
    compute.Model, tokenizer.Tokenizer, {integer}, Options, sequence?: string
  ): Text | nil, string | nil

  -- the count highest-scoring ids in the order a sample ranks them: the higher
  -- score first, the lower id among equal scores
  top: function(scores: {number}, count: integer): {integer}
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
  held: {integer: integer}
  stream: Stream
  position: integer
  index: integer
  pending: integer | nil
  handed: integer
  unreleased: string
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

-- the penalties move a score once for each time the run holds its id
local function penalised(
  scores: {number}, options: Options, held: {integer: integer}
): {number}
  local repetition = options.repetition_penalty
  local presence = options.presence_penalty
  local frequency = options.frequency_penalty
  if repetition is nil and presence is nil and frequency is nil then
    return scores
  end
  local values: {number} = {}
  for index = 1, #scores do
    local value = scores[index]
    local count = held[index - 1] or 0
    if count > 0 then
      if repetition is number then
        if value > 0 then
          value = value / repetition
        elseif value < 0 then
          value = value * repetition
        end
      end
      if presence is number then
        value = value - presence
      end
      if frequency is number then
        value = value - frequency * count
      end
    end
    values[index] = value
  end
  return values
end

-- higher score first, the lower id among equal scores
local function outranks(
  values: {number}, first: integer, second: integer
): boolean
  local left = values[first + 1]
  local right = values[second + 1]
  if left ~= right then
    return left > right
  end
  return first < second
end

local function sifted(
  values: {number}, heap: {integer}, root: integer, size: integer
)
  while true do
    local child = root * 2
    if child > size then
      return
    end
    if child + 1 <= size and outranks(values, heap[child], heap[child + 1]) then
      child = child + 1
    end
    if not outranks(values, heap[root], heap[child]) then
      return
    end
    heap[root], heap[child] = heap[child], heap[root]
    root = child
  end
end

local function raised(values: {number}, heap: {integer}, size: integer)
  local at = size
  while at > 1 do
    local parent = math.floor(at / 2)
    if not outranks(values, heap[parent], heap[at]) then
      return
    end
    heap[parent], heap[at] = heap[at], heap[parent]
    at = parent
  end
end

-- the count best ids in descending order, the wide row walked once and
-- never ordered
local function selected(values: {number}, count: integer): {integer}
  local heap: {integer} = {}
  local held = 0
  for index = 1, #values do
    local id = index - 1
    if held < count then
      held = held + 1
      heap[held] = id
      raised(values, heap, held)
    elseif outranks(values, id, heap[1]) then
      heap[1] = id
      sifted(values, heap, 1, held)
    end
  end
  table.sort(
    heap,
    function(left: integer, right: integer): boolean
      return outranks(values, left, right)
    end
  )
  return heap
end

local function ordered(values: {number}): {integer}
  local order: {integer} = {}
  for index = 1, #values do
    order[index] = index - 1
  end
  table.sort(
    order,
    function(left: integer, right: integer): boolean
      return outranks(values, left, right)
    end
  )
  return order
end

-- three walks of the row and one draw, the ids in increasing order
local function across(
  values: {number}, temperature: number, stream: Stream
): integer
  local count = #values
  local top = values[1]
  for index = 2, count do
    if values[index] > top then
      top = values[index]
    end
  end
  local total: number = 0
  for index = 1, count do
    total = total + math.exp((values[index] - top) / temperature)
  end
  local draw = drawn(stream)
  local running: number = 0
  for index = 1, count do
    running = running + math.exp((values[index] - top) / temperature) / total
    if running > draw then
      return index - 1
    end
  end
  return count - 1
end

local function narrowed(
  values: {number}, options: Options, temperature: number, stream: Stream
): integer
  local top_k = options.top_k
  local top_p = options.top_p
  local min_p = options.min_p
  local order: {integer}
  local top: number
  if not (top_k is nil) then
    local size = top_k
    if size > #values then
      size = #values
    end
    order = selected(values, size)
    top = values[order[1] + 1]
  elseif top_p is number then
    order = ordered(values)
    top = values[order[1] + 1]
  else
    order = {}
    for index = 1, #values do
      order[index] = index - 1
    end
    top = values[1]
    for index = 2, #values do
      if values[index] > top then
        top = values[index]
      end
    end
  end

  local count = #order
  local weights: {number} = {}
  for index = 1, count do
    weights[index] = math.exp((values[order[index] + 1] - top) / temperature)
  end

  if top_p is number then
    local total: number = 0
    for index = 1, count do
      total = total + weights[index]
    end
    local running: number = 0
    for index = 1, count do
      running = running + weights[index] / total
      if running >= top_p then
        count = index
        break
      end
    end
  end

  if min_p is number then
    local highest = weights[1]
    for index = 2, count do
      if weights[index] > highest then
        highest = weights[index]
      end
    end
    local threshold = min_p * highest
    local kept = 0
    for index = 1, count do
      local weight = weights[index]
      if weight >= threshold then
        kept = kept + 1
        order[kept] = order[index]
        weights[kept] = weight
      end
    end
    count = kept
  end

  local total: number = 0
  for index = 1, count do
    total = total + weights[index]
  end
  local draw = drawn(stream)
  local running: number = 0
  for index = 1, count do
    running = running + weights[index] / total
    if running > draw then
      return order[index]
    end
  end
  return order[count]
end

local function chosen(
  scores: {number}, options: Options, held: {integer: integer}, stream: Stream
): integer
  local values = penalised(scores, options, held)
  local temperature = options.temperature
  if temperature is nil or temperature == 0 then
    return greediest(values)
  end
  local top_k = options.top_k
  local top_p = options.top_p
  local min_p = options.min_p
  if top_k is nil and top_p is nil and min_p is nil then
    return across(values, temperature, stream)
  end
  return narrowed(values, options, temperature, stream)
end

local function faulted(options: Options): string | nil
  if options.max_tokens < 1 then
    return "generate: max_tokens "
      .. tostring(options.max_tokens)
      .. " is below one"
  end
  local temperature = options.temperature
  if temperature is number and temperature < 0 then
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
  local top_p = options.top_p
  if top_p is number and (top_p <= 0 or top_p > 1) then
    return "generate: top_p "
      .. tostring(top_p)
      .. " is not above zero and at most one"
  end
  local min_p = options.min_p
  if min_p is number and (min_p < 0 or min_p > 1) then
    return "generate: min_p "
      .. tostring(min_p)
      .. " is not between zero and one"
  end
  local repetition = options.repetition_penalty
  if repetition is number and repetition <= 0 then
    return "generate: repetition_penalty "
      .. tostring(repetition)
      .. " is not above zero"
  end
  local stop = options.stop
  if stop is nil then
    return nil
  end
  local usable = #stop >= 1 and #stop <= 4
  local index = 1
  while usable and index <= #stop do
    if #stop[index] < 1 then
      usable = false
    end
    index = index + 1
  end
  if not usable then
    return "generate: stop is not one to four strings of one or more bytes"
  end
  return nil
end

-- where the earliest stop string starts inside the text not yet released
local function stop_start(text: string, stops: {string} | nil): integer | nil
  if stops is nil then
    return nil
  end
  local first: integer | nil = nil
  for _, marker in ipairs(stops) do
    local at = text:find(marker, 1, true)
    if not (at is nil) and (first is nil or at < first) then
      first = at
    end
  end
  return first
end

-- the text ends in a proper prefix of a stop string, so no piece is given yet
local function withheld(text: string, stops: {string} | nil): boolean
  if stops is nil then
    return false
  end
  for _, marker in ipairs(stops) do
    local longest = #marker - 1
    if #text < longest then
      longest = #text
    end
    for length = longest, 1, -1 do
      if text:sub(#text - length + 1) == marker:sub(1, length) then
        return true
      end
    end
  end
  return false
end

-- the text a run gives as a piece, cut at the stop position when given
local function released(progress: Progress, at: integer | nil): string | nil
  local text = progress.unreleased
  progress.unreleased = ""
  if at is integer then
    text = text:sub(1, at - 1)
  end
  if text == "" then
    return nil
  end
  return text
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

-- a caller that already encoded its prompt hands the ids in, so one turn pays
-- for one encoding; the two entry points keep the text and the ids apart rather
-- than taking both and trusting them to agree
function generate.open_ids(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  ids: {integer},
  options: Options,
  sequence?: string
): generate.Run | nil, string | nil
  local fault = faulted(options)
  if fault is string then
    return nil, fault
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

  local held: {integer: integer} = {}
  for index = start + 1, #ids do
    local id = ids[index]
    held[id] = (held[id] or 0) + 1
  end

  local progress: Progress = {
    model = model,
    tokens = tokens,
    ids = ids,
    sequence = sequence or "",
    end_id = end_id,
    max_tokens = options.max_tokens,
    options = options,
    held = held,
    stream = {state = options.seed % STATE_MODULUS},
    position = start,
    index = start + 1,
    pending = nil,
    handed = 0,
    unreleased = "",
  }

  local produced: {integer} = {}

  -- the rows this run has yet to forward, as many as the caller's bound holds:
  -- consecutive positions, none of them taken until accept
  local function next_input(
    self: generate.Run, most: integer
  ): {compute.Input} | nil
    if self.done or most < 1 then
      return nil
    end
    local rows: {compute.Input} = {}
    local at = progress.index + progress.handed
    while #rows < most and at <= #progress.ids do
      rows[#rows + 1] = {
        sequence = progress.sequence,
        token = progress.ids[at],
        position = progress.position + progress.handed + #rows,
      }
      at = at + 1
    end
    if #rows == 0 then
      local token = progress.pending
      if token is nil or progress.handed > 0 then
        return nil
      end
      rows[1] = {
        sequence = progress.sequence,
        token = token,
        position = progress.position,
      }
    end
    progress.handed = progress.handed + #rows
    return rows
  end

  local function accept(
    self: generate.Run, scores: {number}
  ): string | nil, string | nil
    if self.done then
      return nil, nil
    end

    local rows = progress.handed
    progress.handed = 0
    local prompt_rows = 0
    while
      prompt_rows < rows and progress.index + prompt_rows <= #progress.ids
    do
      prompt_rows = prompt_rows + 1
    end
    progress.index = progress.index + prompt_rows
    progress.position = progress.position + rows
    if prompt_rows < rows then
      progress.pending = nil
    end

    if progress.index <= #progress.ids then
      return nil, nil
    end

    local started = clock.now()
    local token_id = chosen(
      scores, progress.options, progress.held, progress.stream
    )
    local ended = clock.now()
    local measure = progress.model.measure
    measure.sampling = measure.sampling + (ended - started)
    measure.ready = ended
    if token_id == progress.end_id then
      self.done = true
      return released(progress, nil), nil
    end

    produced[#produced + 1] = token_id
    progress.held[token_id] = (progress.held[token_id] or 0) + 1
    local piece, piece_line = progress.tokens:decode({token_id})
    if piece is nil then
      self.done = true
      self.failure = piece_line
      return nil, piece_line
    end
    progress.unreleased = progress.unreleased .. piece

    local stops = progress.options.stop
    local at = stop_start(progress.unreleased, stops)
    if at is integer then
      self.done = true
      self.cut = true
      return released(progress, at), nil
    end
    if #produced >= progress.max_tokens then
      self.done = true
      return released(progress, nil), nil
    end
    progress.pending = token_id
    if withheld(progress.unreleased, stops) then
      return nil, nil
    end
    return released(progress, nil), nil
  end

  local function fail(self: generate.Run, line: string)
    self.done = true
    self.failure = line
  end

  local run: generate.Run = {
    done = false,
    cut = false,
    failure = nil,
    produced = produced,
    next = next_input,
    accept = accept,
    fail = fail,
  }
  return run, nil
end

function generate.open(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options,
  sequence?: string
): generate.Run | nil, string | nil
  local fault = faulted(options)
  if fault is string then
    return nil, fault
  end
  local ids, encode_line = tokens:encode(prompt)
  if ids is nil then
    return nil, encode_line
  end
  return generate.open_ids(model, tokens, ids, options, sequence)
end

-- the count best ids of one score row, in the order a sample ranks them
function generate.top(scores: {number}, count: integer): {integer}
  return selected(scores, count)
end

function generate.text_ids(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  ids: {integer},
  options: Options,
  sequence?: string
): generate.Text | nil, string | nil
  local run, open_line = generate.open_ids(
    model, tokens, ids, options, sequence
  )
  if run is nil then
    return nil, open_line
  end

  local name = sequence or ""
  local reserve_line = ensure_reserved(model, name)
  if reserve_line is string then
    return nil, reserve_line
  end

  local pieces: {string} = {}
  while not run.done do
    local given = run.next(run, model.ubatch)
    if given is nil then
      break
    end
    local rows, pass_line = model:pass(given)
    if rows is nil then
      run.fail(run, pass_line as string)
      break
    end
    local scores = rows as {{number}}
    local piece, accept_line = run.accept(run, scores[#scores])
    if accept_line is string then
      break
    end
    if piece is string then
      pieces[#pieces + 1] = piece
    end
  end

  local failure = run.failure
  if failure is string then
    return nil, failure
  end

  return {text = table.concat(pieces), sampled = run.produced, cut = run.cut},
    nil
end

function generate.text(
  model: compute.Model,
  tokens: tokenizer.Tokenizer,
  prompt: string,
  options: Options,
  sequence?: string
): generate.Text | nil, string | nil
  local fault = faulted(options)
  if fault is string then
    return nil, fault
  end
  local ids, encode_line = tokens:encode(prompt)
  if ids is nil then
    return nil, encode_line
  end
  return generate.text_ids(model, tokens, ids, options, sequence)
end

return generate
