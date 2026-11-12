local gguf = require("tllm.gguf")

local record tokenizer
  record Tokenizer
    size: integer
    decode: function(Tokenizer, {integer}): string | nil, string | nil
  end

  load: function(path: string): Tokenizer | nil, string | nil
end

local type Tokenizer = tokenizer.Tokenizer

local record Vocabulary
  pieces: {string}
  types: {integer}
end

local MODEL_KEY = "tokenizer.ggml.model"
local TOKENS_KEY = "tokenizer.ggml.tokens"
local TYPES_KEY = "tokenizer.ggml.token_type"
local SCHEME = "gpt2"
local DEFAULT_TYPE = 1

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

local function described(entry: gguf.Value): string
  if entry.text ~= nil then
    return entry.text as string
  end
  if entry.scalar ~= nil then
    return tostring(entry.scalar)
  end
  if entry.flag ~= nil then
    return tostring(entry.flag)
  end
  return entry.kind
end

local function utf8_at(
  text: string, index: integer
): integer | nil, integer | nil
  local lead = text:byte(index)
  if lead is nil then
    return nil, nil
  end
  if lead < 0x80 then
    return lead, index + 1
  end

  local width: integer
  local codepoint: integer
  if lead >= 0xC2 and lead <= 0xDF then
    width = 2
    codepoint = lead - 0xC0
  elseif lead >= 0xE0 and lead <= 0xEF then
    width = 3
    codepoint = lead - 0xE0
  elseif lead >= 0xF0 and lead <= 0xF4 then
    width = 4
    codepoint = lead - 0xF0
  else
    return nil, nil
  end

  for offset = 1, width - 1 do
    local continuation = text:byte(index + offset)
    if continuation is nil or continuation < 0x80 or continuation > 0xBF then
      return nil, nil
    end
    codepoint = codepoint * 0x40 + (continuation - 0x80)
  end

  if width == 2 and codepoint < 0x80 then
    return nil, nil
  end
  if
    width == 3
    and (codepoint < 0x800 or (codepoint >= 0xD800 and codepoint <= 0xDFFF))
  then
    return nil, nil
  end
  if width == 4 and (codepoint < 0x10000 or codepoint > 0x10FFFF) then
    return nil, nil
  end
  return codepoint, index + width
end

local function gpt2_bytes(piece: string): string | nil, string | nil
  local parts: {string} = {}
  local index = 1
  while index <= #piece do
    local codepoint, next_index = utf8_at(piece, index)
    if codepoint is nil then
      return nil, "not valid UTF-8"
    end
    local byte = BYTE_OF_CODEPOINT[codepoint]
    if byte is nil then
      return nil, "codepoint " .. tostring(codepoint) .. " has no GPT-2 byte"
    end
    parts[#parts + 1] = string.char(byte)
    index = next_index as integer
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

function tokenizer.load(path: string): Tokenizer | nil, string | nil
  local info, read_failure = gguf.read(path)
  if info is nil then
    return nil, read_failure
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

  local vocabulary: Vocabulary = {pieces = pieces, types = types}
  return {
    size = #pieces,
    decode = function(_: Tokenizer, ids: {integer}): string | nil, string | nil
      return decode_ids(vocabulary, ids)
    end,
  }, nil
end

return tokenizer
