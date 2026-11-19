local gguf = require("tllm.gguf")
local unicode = require("tllm.unicode")
local utf8lib = require("utf8lib")

local record tokenizer
  record Tokenizer
    size: integer
    end_id: integer | nil
    template: string | nil
    decode: function(Tokenizer, {integer}): string | nil, string | nil
    encode: function(Tokenizer, string): {integer} | nil, string | nil
  end

  load: function(path: string, info?: gguf.Info): Tokenizer | nil, string | nil
end

local type Tokenizer = tokenizer.Tokenizer

local record Vocabulary
  pieces: {string}
  types: {integer}
  ids: {string: integer}
  pre: gguf.Value | nil
  merges: gguf.Value | nil
end

local MODEL_KEY = "tokenizer.ggml.model"
local TOKENS_KEY = "tokenizer.ggml.tokens"
local TYPES_KEY = "tokenizer.ggml.token_type"
local PRE_KEY = "tokenizer.ggml.pre"
local MERGES_KEY = "tokenizer.ggml.merges"
local END_KEY = "tokenizer.ggml.eos_token_id"
local TEMPLATE_KEY = "tokenizer.chat_template"
local SCHEME = "gpt2"
local QWEN_SCHEME = "qwen35"
local DEFAULT_TYPE = 1
local APOSTROPHE = 0x27
local SPACE = 0x20
local CARRIAGE_RETURN = 0x0D
local LINE_FEED = 0x0A

local INTEGER_KINDS: {string: boolean} = {
  uint8 = true,
  int8 = true,
  uint16 = true,
  int16 = true,
  uint32 = true,
  int32 = true,
  uint64 = true,
  int64 = true,
}

local BYTE_OF_CODEPOINT: {integer: integer} = {}

do
  local displaced = 0
  for byte = 0, 255 do
    local keeps_codepoint = (byte >= 0x21 and byte <= 0x7E)
      or (byte >= 0xA1 and byte <= 0xAC)
      or (byte >= 0xAE and byte <= 0xFF)
    if keeps_codepoint then
      BYTE_OF_CODEPOINT[byte] = byte
    else
      BYTE_OF_CODEPOINT[256 + displaced] = byte
      displaced = displaced + 1
    end
  end
end

local BYTE_SYMBOL: {integer: string} = {}

for codepoint, byte in pairs(BYTE_OF_CODEPOINT) do
  BYTE_SYMBOL[byte] = utf8lib.char(codepoint)
end

local function printable(value: string): string
  local cleaned: string = value:gsub("[%z\1-\31\127]", "?")
  return cleaned
end

local function described(entry: gguf.Value): string
  local entry_text = entry.text
  if entry_text is string then
    return printable(entry_text)
  end
  local entry_scalar = entry.scalar
  if entry_scalar is number then
    return tostring(entry_scalar)
  end
  local entry_flag = entry.flag
  if entry_flag is boolean then
    return tostring(entry_flag)
  end
  return entry.kind
end

local function gpt2_bytes(piece: string): string | nil, string | nil
  if not utf8lib.valid(piece) then
    return nil, "not valid UTF-8"
  end
  local parts: {string} = {}
  for _, codepoint in utf8lib.codes(piece) do
    local byte = BYTE_OF_CODEPOINT[codepoint]
    if byte is nil then
      return nil, "codepoint " .. tostring(codepoint) .. " has no GPT-2 byte"
    end
    parts[#parts + 1] = string.char(byte)
  end
  return table.concat(parts), nil
end

local function pieces_of(
  path: string, entry: gguf.Value | nil
): {string} | nil, string | nil
  local fault = path .. ": " .. TOKENS_KEY
  if entry is nil then
    return nil, fault .. ": missing"
  end
  local items = entry.items
  if items is nil then
    return nil, fault .. ": not an array of strings"
  end
  local pieces: {string} = {}
  for _, item in ipairs(items) do
    if item.kind ~= "string" or item.text == nil then
      return nil, fault .. ": not an array of strings"
    end
    pieces[#pieces + 1] = item.text as string
  end
  return pieces, nil
end

local function types_of(
  path: string, entry: gguf.Value | nil, count: integer
): {integer} | nil, string | nil
  local types: {integer} = {}
  if entry is nil then
    for index = 1, count do
      types[index] = DEFAULT_TYPE
    end
    return types, nil
  end
  local fault = path
    .. ": "
    .. TYPES_KEY
    .. ": not an array of "
    .. tostring(count)
    .. " integers"
  local items = entry.items
  local item_kind = entry.item_kind
  if
    items is nil
    or item_kind is nil
    or not INTEGER_KINDS[item_kind]
    or #items ~= count
  then
    return nil, fault
  end
  for index, item in ipairs(items) do
    types[index] = item.scalar as integer
  end
  return types, nil
end

local function decode_ids(
  vocabulary: Vocabulary, ids: {integer}
): string | nil, string | nil
  local size = #vocabulary.pieces
  local parts: {string} = {}
  for _, id in ipairs(ids) do
    if id < 0 or id >= size then
      return nil,
        "id "
          .. tostring(id)
          .. " is outside a vocabulary of "
          .. tostring(size)
          .. " entries"
    end
    local piece = vocabulary.pieces[id + 1]
    local kind = vocabulary.types[id + 1]
    if kind == 1 then
      local bytes, failure = gpt2_bytes(piece)
      if bytes is nil then
        return nil, "id " .. tostring(id) .. ": " .. (failure as string)
      end
      parts[#parts + 1] = bytes as string
    elseif kind == 4 then
      parts[#parts + 1] = piece
    elseif kind ~= 2 and kind ~= 3 and kind ~= 5 then
      return nil, "id " .. tostring(id) .. " has token type " .. tostring(kind)
    end
  end
  return table.concat(parts), nil
end

local function end_id_of(entry: gguf.Value | nil): integer | nil
  if entry is nil then
    return nil
  end
  local scalar = entry.scalar
  if scalar is nil or not INTEGER_KINDS[entry.kind] then
    return nil
  end
  return scalar as integer
end

local function identifiers_of(pieces: {string}): {string: integer}
  local ids: {string: integer} = {}
  for index, piece in ipairs(pieces) do
    if ids[piece] == nil then
      ids[piece] = index - 1
    end
  end
  return ids
end

local function scheme_of(
  path: string, entry: gguf.Value | nil
): string | nil, string | nil
  local fault = path .. ": " .. PRE_KEY
  if entry is nil then
    return nil, fault .. ": missing"
  end
  local scheme = entry.text
  if scheme is nil then
    return nil, fault .. ": " .. described(entry)
  end
  if scheme ~= SCHEME and scheme ~= QWEN_SCHEME then
    return nil, fault .. ": " .. printable(scheme)
  end
  return scheme, nil
end

local function merge_ranks(
  path: string, entry: gguf.Value | nil
): {string: integer} | nil, string | nil
  local fault = path .. ": " .. MERGES_KEY
  if entry is nil then
    return nil, fault .. ": missing"
  end
  local items = entry.items
  if items is nil then
    return nil, fault .. ": not an array of strings"
  end
  local ranks: {string: integer} = {}
  for index, item in ipairs(items) do
    if item.kind ~= "string" or item.text == nil then
      return nil, fault .. ": not an array of strings"
    end
    local merge = item.text as string
    if merge:find(" ", 2) == nil then
      return nil,
        fault
          .. ": "
          .. printable(merge)
          .. " has no space after its first character"
    end
    if ranks[merge] == nil then
      ranks[merge] = index - 1
    end
  end
  return ranks, nil
end

local record Codepoints
  values: {integer}
  starts: {integer}
end

local function codepoints_of(text: string): Codepoints | nil
  if not utf8lib.valid(text) then
    return nil
  end
  local values: {integer} = {}
  local starts: {integer} = {}
  for position, codepoint in utf8lib.codes(text) do
    values[#values + 1] = codepoint
    starts[#starts + 1] = position
  end
  return {values = values, starts = starts}
end

local function folded(codepoint: integer): integer
  if codepoint >= 0x41 and codepoint <= 0x5A then
    return codepoint + 0x20
  end
  return codepoint
end

local function apostrophe_end(
  codepoints: {integer}, position: integer, any_case: boolean
): integer | nil
  if codepoints[position] ~= APOSTROPHE then
    return nil
  end
  local count = #codepoints
  if position + 1 > count then
    return nil
  end
  local first = codepoints[position + 1]
  if any_case then
    first = folded(first)
  end
  if first == 0x73 or first == 0x74 or first == 0x6D or first == 0x64 then
    return position + 1
  end
  if position + 2 > count then
    return nil
  end
  local second = codepoints[position + 2]
  if any_case then
    second = folded(second)
  end
  if
    (first == 0x72 and second == 0x65)
    or (first == 0x76 and second == 0x65)
    or (first == 0x6C and second == 0x6C)
  then
    return position + 2
  end
  return nil
end

local function letter_run(codepoints: {integer}, start: integer): integer
  local last = start
  while last + 1 <= #codepoints and unicode.is_letter(codepoints[last + 1]) do
    last = last + 1
  end
  return last
end

local function digit_run(codepoints: {integer}, start: integer): integer
  local last = start
  while last + 1 <= #codepoints and unicode.is_number(codepoints[last + 1]) do
    last = last + 1
  end
  return last
end

local function mark_run(codepoints: {integer}, start: integer): integer
  local last = start
  while last + 1 <= #codepoints do
    local codepoint = codepoints[last + 1]
    if not (unicode.is_letter(codepoint) or unicode.is_mark(codepoint)) then
      break
    end
    last = last + 1
  end
  return last
end

local function punctuation_run(
  codepoints: {integer}, start: integer, with_marks: boolean
): integer
  local last = start
  while last + 1 <= #codepoints do
    local codepoint = codepoints[last + 1]
    local stops = unicode.is_whitespace(codepoint)
      or unicode.is_letter(codepoint)
      or unicode.is_number(codepoint)
    if not with_marks then
      stops = stops or unicode.is_mark(codepoint)
    end
    if stops then
      break
    end
    last = last + 1
  end
  return last
end

local function whitespace_run(codepoints: {integer}, position: integer): integer
  local count = 0
  while
    position + count <= #codepoints
    and unicode.is_whitespace(codepoints[position + count])
  do
    count = count + 1
  end
  return count
end

local function trailing_whitespace_end(
  codepoints: {integer}, position: integer, run: integer
): integer | nil
  if run == 0 then
    return nil
  end
  if run > 1 and position + run <= #codepoints then
    return position + run - 2
  end
  return position + run - 1
end

local function split_gpt2(codepoints: {integer}): {{integer}}
  local words: {{integer}} = {}
  local count = #codepoints
  local position = 1
  while position <= count do
    local last = apostrophe_end(codepoints, position, false)
    if last is nil then
      local start = position
      if codepoints[position] == SPACE then
        start = position + 1
      end
      if start <= count then
        local next_codepoint = codepoints[start]
        if unicode.is_letter(next_codepoint) then
          last = letter_run(codepoints, start)
        elseif unicode.is_number(next_codepoint) then
          last = digit_run(codepoints, start)
        elseif not (
          unicode.is_whitespace(next_codepoint)
          or unicode.is_letter(next_codepoint)
          or unicode.is_number(next_codepoint)
        ) then
          last = punctuation_run(codepoints, start, true)
        end
      end
    end
    if last is nil then
      last = trailing_whitespace_end(
        codepoints, position, whitespace_run(codepoints, position)
      )
    end
    if last is nil then
      last = position
    end
    words[#words + 1] = {position, last}
    position = last + 1
  end
  return words
end

local function split_qwen35(codepoints: {integer}): {{integer}}
  local words: {{integer}} = {}
  local count = #codepoints
  local position = 1
  while position <= count do
    local codepoint = codepoints[position]
    local last = apostrophe_end(codepoints, position, true)
    if
      last is nil
      and not (
        codepoint == CARRIAGE_RETURN
          or codepoint == LINE_FEED
          or unicode.is_number(codepoint)
      )
    then
      if unicode.is_letter(codepoint) or unicode.is_mark(codepoint) then
        last = mark_run(codepoints, position)
      elseif position + 1 <= count then
        local follower = codepoints[position + 1]
        if unicode.is_letter(follower) or unicode.is_mark(follower) then
          last = mark_run(codepoints, position + 1)
        end
      end
    end
    if last is nil and unicode.is_number(codepoint) then
      last = position
    end
    if last is nil then
      local start = position
      if codepoint == SPACE then
        start = position + 1
      end
      if start <= count then
        local follower = codepoints[start]
        if not (
          unicode.is_whitespace(follower)
          or unicode.is_letter(follower)
          or unicode.is_mark(follower)
          or unicode.is_number(follower)
        ) then
          last = punctuation_run(codepoints, start, false)
          while
            last + 1 <= count
            and (
              codepoints[last + 1] == CARRIAGE_RETURN
              or codepoints[last + 1] == LINE_FEED
            )
          do
            last = last + 1
          end
        end
      end
    end
    if last is nil then
      local run = whitespace_run(codepoints, position)
      for offset = 0, run - 1 do
        local held = codepoints[position + offset]
        if held == CARRIAGE_RETURN or held == LINE_FEED then
          last = position + offset
        end
      end
      if last is nil then
        last = trailing_whitespace_end(codepoints, position, run)
      end
    end
    if last is nil then
      last = position
    end
    words[#words + 1] = {position, last}
    position = last + 1
  end
  return words
end

local record Candidate
  rank: integer
  at: integer
  left_stamp: integer
  right_stamp: integer
end

local function sooner(left: Candidate, right: Candidate): boolean
  if left.rank ~= right.rank then
    return left.rank < right.rank
  end
  return left.at < right.at
end

local function offered(heap: {Candidate}, item: Candidate)
  local at = #heap + 1
  heap[at] = item
  while at > 1 do
    local parent = math.floor(at / 2)
    if not sooner(heap[at], heap[parent]) then
      break
    end
    heap[at], heap[parent] = heap[parent], heap[at]
    at = parent
  end
end

local function taken(heap: {Candidate}): Candidate | nil
  local count = #heap
  if count == 0 then
    return nil
  end
  local top = heap[1]
  heap[1] = heap[count]
  heap[count] = nil
  count = count - 1
  local at = 1
  while true do
    local left = at * 2
    local right = left + 1
    local next_at = at
    if left <= count and sooner(heap[left], heap[next_at]) then
      next_at = left
    end
    if right <= count and sooner(heap[right], heap[next_at]) then
      next_at = right
    end
    if next_at == at then
      break
    end
    heap[at], heap[next_at] = heap[next_at], heap[at]
    at = next_at
  end
  return top
end

-- the pairs still worth merging sit in a heap, so one word costs its merges
-- times log of its length rather than a scan of the word per merge: a
-- paragraph with no space, or a long identifier, is one word
local function merged(symbols: {string}, ranks: {string: integer}): {string}
  local count = #symbols
  if count < 2 then
    return symbols
  end

  local texts: {string} = {}
  local nexts: {integer} = {}
  local prevs: {integer} = {}
  local stamps: {integer} = {}
  local alive: {boolean} = {}
  for index = 1, count do
    texts[index] = symbols[index]
    nexts[index] = index + 1
    prevs[index] = index - 1
    stamps[index] = 0
    alive[index] = true
  end

  local heap: {Candidate} = {}
  local function offered_pair(at: integer)
    if at < 1 or not alive[at] then
      return
    end
    local partner = nexts[at]
    if partner > count or not alive[partner] then
      return
    end
    local rank = ranks[texts[at] .. " " .. texts[partner]]
    if rank is nil then
      return
    end
    offered(
      heap,
      {
        rank = rank as integer,
        at = at,
        left_stamp = stamps[at],
        right_stamp = stamps[partner],
      }
    )
  end
  for index = 1, count - 1 do
    offered_pair(index)
  end

  while true do
    local item = taken(heap)
    if item is nil then
      break
    end
    local held = item as Candidate
    local at = held.at
    local partner = nexts[at]
    local fresh = alive[at]
      and partner <= count
      and alive[partner]
      and stamps[at] == held.left_stamp
      and stamps[partner] == held.right_stamp
    if fresh then
      texts[at] = texts[at] .. texts[partner]
      stamps[at] = stamps[at] + 1
      alive[partner] = false
      local after = nexts[partner]
      nexts[at] = after
      if after <= count then
        prevs[after] = at
      end
      offered_pair(at)
      offered_pair(prevs[at])
    end
  end

  local kept: {string} = {}
  local at = 1
  while at <= count do
    kept[#kept + 1] = texts[at]
    at = nexts[at]
  end
  return kept
end

local function append_word(
  vocabulary: Vocabulary,
  path: string,
  word: string,
  ranks: {string: integer},
  ids: {integer}
): string | nil
  local symbols: {string} = {}
  for index = 1, #word do
    symbols[#symbols + 1] = BYTE_SYMBOL[word:byte(index) as integer]
  end
  for _, symbol in ipairs(merged(symbols, ranks)) do
    local id = vocabulary.ids[symbol]
    if not (id is nil) then
      ids[#ids + 1] = id
    else
      for _, codepoint in utf8lib.codes(symbol) do
        local fallback = vocabulary.ids[utf8lib.char(codepoint)]
        if fallback is nil then
          return path .. ": no token stores codepoint " .. tostring(codepoint)
        end
        ids[#ids + 1] = fallback
      end
    end
  end
  return nil
end

local function encode_text(
  vocabulary: Vocabulary, path: string, text: string, ranks: {string: integer}
): {integer} | nil, string | nil
  local scheme, scheme_failure = scheme_of(path, vocabulary.pre)
  if scheme is nil then
    return nil, scheme_failure
  end
  local points = codepoints_of(text)
  if points is nil then
    return nil, path .. ": the text is not valid UTF-8"
  end
  local words: {{integer}}
  if scheme == SCHEME then
    words = split_gpt2(points.values)
  else
    words = split_qwen35(points.values)
  end
  local ids: {integer} = {}
  for _, word in ipairs(words) do
    local stop = #text
    if word[2] < #points.starts then
      stop = points.starts[word[2] + 1] - 1
    end
    local failure = append_word(
      vocabulary, path, text:sub(points.starts[word[1]], stop), ranks, ids
    )
    if failure is string then
      return nil, failure
    end
  end
  return ids, nil
end

function tokenizer.load(
  path: string, given_info?: gguf.Info
): Tokenizer | nil, string | nil
  local info: gguf.Info
  if given_info is nil then
    local read, read_failure = gguf.read(path)
    if read is nil then
      return nil, read_failure
    end
    info = read
  else
    info = given_info
  end

  local model = info.metadata[MODEL_KEY]
  if model is nil then
    return nil, path .. ": " .. MODEL_KEY .. ": missing"
  end
  if model.text ~= SCHEME then
    return nil, path .. ": " .. MODEL_KEY .. ": " .. described(model)
  end

  local pieces, pieces_failure = pieces_of(path, info.metadata[TOKENS_KEY])
  if pieces is nil then
    return nil, pieces_failure
  end
  local types, types_failure = types_of(path, info.metadata[TYPES_KEY], #pieces)
  if types is nil then
    return nil, types_failure
  end

  local ranks: {string: integer} | nil = nil
  local ranks_failure: string | nil = nil
  local vocabulary: Vocabulary = {
    pieces = pieces,
    types = types,
    ids = identifiers_of(pieces),
    pre = info.metadata[PRE_KEY],
    merges = info.metadata[MERGES_KEY],
  }
  local template_entry = info.metadata[TEMPLATE_KEY]
  local template: string | nil = nil
  if template_entry is gguf.Value then
    template = template_entry.text
  end

  return {
    size = #pieces,
    end_id = end_id_of(info.metadata[END_KEY]),
    template = template,
    decode = function(_: Tokenizer, ids: {integer}): string | nil, string | nil
      return decode_ids(vocabulary, ids)
    end,
    encode = function(_: Tokenizer, text: string): {integer} | nil, string | nil
      local built = ranks
      if built is nil and ranks_failure is nil then
        built, ranks_failure = merge_ranks(path, vocabulary.merges)
        ranks = built
      end
      if built is nil then
        return nil, ranks_failure
      end
      return encode_text(vocabulary, path, text, built)
    end,
  }, nil
end

return tokenizer
