local json = require("json")
local tokenizer = require("tllm.tokenizer")
local utf8lib = require("utf8lib")

local record chat
  record ToolCall
    name: string
    arguments: json.Value
  end

  record Message
    role: string
    content: string
    tool_calls: {ToolCall} | nil
    tool_call_id: string | nil
    name: string | nil
  end

  record ReplyCall
    name: string
    arguments: string
  end

  record ReplyBlock
    is_call: boolean
    call: ReplyCall | nil
  end

  -- 51/FR-001: what one reply's head holds. `pending` says the head has not
  -- settled yet, so it could still open a pair; `open` says one stands there;
  -- `closed` says that pair's closing marker stands in the reply, and then
  -- `reasoning` and `rest` are the block's text and what that marker left.
  record ReplyHead
    pending: boolean
    open: boolean
    closed: boolean
    reasoning: string
    rest: string
  end

  -- the marker pair a scanned block is found by, so a stream holds and sends
  -- the same bytes the scan reads
  call_open: string
  call_close: string
  -- the marker pair a reply's reasoning block is found by
  think_open: string
  think_close: string
  read_block: function(string): ReplyBlock
  read_calls: function(string): {ReplyCall} | nil, string | nil
  read_head: function(reply: string, prompt: string): ReplyHead
  render: function(
    tokenizer.Tokenizer,
    {Message},
    tools?: json.Value,
    enable_thinking?: boolean,
    tool_choice?: json.Value
  ): string | nil, string | nil
end

local type Msg = chat.Message

local KEY = "tokenizer.chat_template"
local WS = "[ \t\r\n]"
local CALL_OPEN = "<tool_call>"
local CALL_CLOSE = "</tool_call>"
local THINK_OPEN = "<think>"
local THINK_CLOSE = "</think>"
local FUNCTION_OPEN = "<function="
local FUNCTION_CLOSE = "</function>"
local PARAMETER_OPEN = "<parameter="
local PARAMETER_CLOSE = "</parameter>"

chat.call_open = CALL_OPEN
chat.call_close = CALL_CLOSE
chat.think_open = THINK_OPEN
chat.think_close = THINK_CLOSE

local record Expr
  form: string
  text: string
  num: number
  truth: boolean
  reverse: boolean
  parts: {Expr}
  labels: {string}
  left: Expr | nil
  right: Expr | nil
  head: Expr | nil
  tail: Expr | nil
end

local record Branch
  test: Expr | nil
  body: {Node}
end

local record Macro
  name: string
  params: {string}
  optional: {string: Expr}
  body: {Node}
end

local record Node
  form: string
  text: string
  value: Expr | nil
  member: string
  binding: string
  entry: Macro | nil
  body: {Node}
  branches: {Branch}
end

-- Values carry their kind so undefined, Python's none and a missing field stay
-- apart without any use of `any`.
local record Val
  kind: string
  truth: boolean
  num: number
  text: string
  items: {Val}
  fields: {string: Val}
  order: {string}
  code: Macro | nil
end

local record Scope
  names: {string: Val}
  parent: Scope | nil
end

local record Runner
  root: Scope
  frame: Scope
  parts: {string}
  fault: string
  broken: boolean
  depth: integer
end

-- Template pieces
local record Cursor
  at: integer
end

local record Piece
  form: string
  source: string
  name: string
  cut_left: boolean
  cut_right: boolean
end

local record Atom
  form: string
  text: string
  num: number
end

local record Parser
  atoms: {Atom}
  mark: integer
end

local METHODS: {string: integer} = {
  strip = 2,
  lstrip = 2,
  rstrip = 2,
  split = 2,
  startswith = 2,
  endswith = 2,
  replace = 3,
}

local TESTS: {string: boolean} = {
  defined = true,
  undefined = true,
  none = true,
  string = true,
  iterable = true,
  mapping = true,
  sequence = true,
  ["true"] = true,
  ["false"] = true,
}

local SIGNS: {string: boolean} = {
  ["=="] = true,
  ["!="] = true,
  ["<="] = true,
  [">="] = true,
  ["<"] = true,
  [">"] = true,
  ["+"] = true,
  ["-"] = true,
  ["*"] = true,
  ["~"] = true,
  ["("] = true,
  [")"] = true,
  ["["] = true,
  ["]"] = true,
  [","] = true,
  ["."] = true,
  [":"] = true,
  ["|"] = true,
  ["="] = true,
}

local UNBOUND: Val = {
  kind = "unbound",
  truth = false,
  num = 0,
  text = "",
  items = {},
  fields = {},
  order = {},
  code = nil,
}

local NOTHING: Val = {
  kind = "none",
  truth = false,
  num = 0,
  text = "",
  items = {},
  fields = {},
  order = {},
  code = nil,
}

local function say(kind: string): Val
  return {
    kind = kind,
    truth = false,
    num = 0,
    text = "",
    items = {},
    fields = {},
    order = {},
    code = nil,
  }
end

local function as_text(text: string): Val
  local held = say("text")
  held.text = text
  held.truth = text ~= ""
  return held
end

local function as_number(amount: number): Val
  local held = say("number")
  held.num = amount
  held.truth = amount ~= 0
  return held
end

local function as_boolean(flag: boolean): Val
  local held = say("boolean")
  held.truth = flag
  return held
end

local function as_list(items: {Val}): Val
  local held = say("list")
  held.items = items
  held.truth = #items > 0
  return held
end

local function as_map(): Val
  return say("map")
end

local function map_put(held: Val, name: string, value: Val): nil
  local found = held.fields[name]
  if found is nil then
    held.order[#held.order + 1] = name
  end
  held.fields[name] = value
  held.truth = true
end

local of_json: function(held: json.Value): Val

local function as_call(call: chat.ToolCall): Val
  local held = as_map()
  map_put(held, "name", as_text(call.name))
  map_put(held, "arguments", of_json(call.arguments))
  return held
end

local function as_calls(calls: {chat.ToolCall}): Val
  local items: {Val} = {}
  for _, call in ipairs(calls) do
    items[#items + 1] = as_call(call)
  end
  return as_list(items)
end

local function as_message(message: Msg): Val
  local held = as_map()
  map_put(held, "role", as_text(message.role))
  map_put(held, "content", as_text(message.content))
  local calls = message.tool_calls
  if not (calls is nil) then
    map_put(held, "tool_calls", as_calls(calls))
  end
  local call_id = message.tool_call_id
  if call_id is string then
    map_put(held, "tool_call_id", as_text(call_id))
  end
  local given = message.name
  if given is string then
    map_put(held, "name", as_text(given))
  end
  return held
end

of_json = function(held: json.Value): Val
  if held:is("array") then
    local items: {Val} = {}
    for _, child in held:children() do
      items[#items + 1] = of_json(child as json.Value)
    end
    return as_list(items)
  end
  if held:is("object") then
    local map = as_map()
    for name, child in held:children() do
      map_put(map, name as string, of_json(child as json.Value))
    end
    return map
  end
  if held:is("null") then
    return NOTHING
  end
  if held:is("boolean") then
    return as_boolean(held:val() as boolean)
  end
  if held:is("number") then
    return as_number(held:val() as number)
  end
  return as_text(held:val() as string)
end

local JSON_ESCAPES: {string: string} = {
  ['"'] = '\\"',
  ["\\"] = "\\\\",
  ["\n"] = "\\n",
  ["\t"] = "\\t",
  ["\r"] = "\\r",
  ["\b"] = "\\b",
  ["\f"] = "\\f",
}

local function quoted(text: string): string
  local pieces: {string} = {'"'}
  for index = 1, #text do
    local one = text:sub(index, index)
    local held = JSON_ESCAPES[one]
    if held is nil then
      local code = one:byte() as integer
      if code < 0x20 then
        pieces[#pieces + 1] = string.format("\\u%04x", code)
      else
        pieces[#pieces + 1] = one
      end
    else
      pieces[#pieces + 1] = held
    end
  end
  pieces[#pieces + 1] = '"'
  return table.concat(pieces)
end

local json_written: function(held: Val): string

json_written = function(held: Val): string
  local kind = held.kind
  if kind == "text" then
    return quoted(held.text)
  end
  if kind == "number" then
    return tostring(held.num)
  end
  if kind == "boolean" then
    return held.truth and "true" or "false"
  end
  if kind == "list" then
    local pieces: {string} = {}
    for _, item in ipairs(held.items) do
      pieces[#pieces + 1] = json_written(item)
    end
    return "[" .. table.concat(pieces, ", ") .. "]"
  end
  if kind == "map" then
    local pieces: {string} = {}
    for _, name in ipairs(held.order) do
      pieces[#pieces + 1] = quoted(name)
        .. ": "
        .. json_written(held.fields[name] as Val)
    end
    return "{" .. table.concat(pieces, ", ") .. "}"
  end
  return "null"
end

local function fault(detail: string): string
  local cleaned: string = detail:gsub("[\r\n]+", " ")
  return KEY .. ": " .. cleaned
end

local function halt(run: Runner, detail: string): string
  local message = fault(detail)
  if not run.broken then
    run.broken = true
    run.fault = message
  end
  return run.fault
end

local function fail(run: Runner, detail: string): Val
  if not run.broken then
    run.broken = true
    run.fault = fault(detail)
  end
  return UNBOUND
end

local function bare(text: string): string
  local head: string = text:gsub("^" .. WS .. "+", "")
  return (head:gsub(WS .. "+$", ""))
end

local function representable(held: json.Value): boolean
  if held:is("number") then
    local amount = held:val() as number
    return amount == amount and amount ~= math.huge and amount ~= -math.huge
  end
  if held:is("array") or held:is("object") then
    for _, child in held:children() do
      if not representable(child) then
        return false
      end
    end
  end
  return true
end

-- 49/FR-002: a value keeps its JSON type when its text decodes as JSON and is
-- representable, and is text otherwise
local function encoded(text: string): string
  local decoded, value = pcall(json.decode, text, {json5 = false})
  if decoded and representable(value as json.Value) then
    return json.encode(value as json.Value)
  end
  return json.encode(text)
end

local function unspaced(text: string, at: integer): integer
  return text:find("[^ \t\r\n]", at) or #text + 1
end

-- 49/FR-002: one call out of a block that begins with `<function=`
local function read_call(text: string): chat.ReplyCall | nil
  local name_end = text:find(">", #FUNCTION_OPEN + 1, true)
  if name_end is nil then
    return nil
  end
  local name = text:sub(#FUNCTION_OPEN + 1, name_end - 1)
  if #name == 0 then
    return nil
  end

  local arguments: {string} = {}
  local at = name_end + 1
  while true do
    at = unspaced(text, at)
    if text:sub(at, at + #FUNCTION_CLOSE - 1) == FUNCTION_CLOSE then
      if unspaced(text, at + #FUNCTION_CLOSE) <= #text then
        return nil
      end
      return {
        name = name,
        arguments = "{" .. table.concat(arguments, ",") .. "}",
      }
    end
    if text:sub(at, at + #PARAMETER_OPEN - 1) ~= PARAMETER_OPEN then
      return nil
    end
    local key_end = text:find(">", at + #PARAMETER_OPEN, true)
    if key_end is nil then
      return nil
    end
    local key = text:sub(at + #PARAMETER_OPEN, key_end - 1)
    if #key == 0 then
      return nil
    end
    local value_end = text:find(PARAMETER_CLOSE, key_end + 1, true)
    if value_end is nil then
      return nil
    end
    local value = bare(text:sub(key_end + 1, value_end - 1))
    arguments[#arguments + 1] = json.encode(key) .. ":" .. encoded(value)
    at = value_end + #PARAMETER_CLOSE
  end
end

-- 49/FR-001: one block's inner text read as the call it holds, a block that
-- does not begin with `<function=` being no call block at all
function chat.read_block(text: string): chat.ReplyBlock
  local block = bare(text)
  if block:sub(1, #FUNCTION_OPEN) ~= FUNCTION_OPEN then
    return {is_call = false, call = nil}
  end
  return {is_call = true, call = read_call(block)}
end

-- 49/FR-001 and 49/FR-005: every call block in the reply, or the line naming
-- the first block that does not hold the shape a call has
function chat.read_calls(text: string): {chat.ReplyCall} | nil, string | nil
  local calls: {chat.ReplyCall} = {}
  local at = 1
  local position = 0
  while true do
    local open_at = text:find(CALL_OPEN, at, true)
    if open_at is nil then
      break
    end
    local close_at = text:find(CALL_CLOSE, open_at + #CALL_OPEN, true)
    if close_at is nil then
      break
    end
    position = position + 1
    local block = chat.read_block(text:sub(open_at + #CALL_OPEN, close_at - 1))
    local call = block.call
    if block.is_call and call is nil then
      return nil, "reply: block " .. tostring(position) .. ": not a call"
    end
    if call is chat.ReplyCall then
      calls[#calls + 1] = call
    end
    at = close_at + #CALL_CLOSE
  end
  return calls, nil
end

-- 51/FR-001: the reasoning block at a reply's head, read from that reply and
-- from the prompt the request rendered, which may be the half of the pair that
-- stands open
function chat.read_head(reply: string, prompt: string): chat.ReplyHead
  local after = 1
  local open = (prompt:gsub(WS .. "+$", "")):sub(-#THINK_OPEN) == THINK_OPEN
  local pending = false
  local head = reply:find("[^ \t\r\n]")
  if not (head is nil) then
    if reply:sub(head, head + #THINK_OPEN - 1) == THINK_OPEN then
      open = true
      after = head + #THINK_OPEN
    elseif not open then
      local front = reply:sub(head)
      pending = #front < #THINK_OPEN and THINK_OPEN:sub(1, #front) == front
    end
  else
    pending = not open
  end

  local close_at = reply:find(THINK_CLOSE, after, true)
  local end_at = #reply
  local rest = ""
  if not (close_at is nil) then
    end_at = close_at - 1
    rest = reply:sub(close_at + #THINK_CLOSE)
  end
  return {
    pending = pending,
    open = open,
    closed = not (close_at is nil),
    reasoning = bare(reply:sub(after, end_at)),
    rest = rest,
  }
end

local function letters(text: string): {string}
  local held: {string} = {}
  for _, codepoint in utf8lib.codes(text) do
    held[#held + 1] = utf8lib.char(codepoint)
  end
  return held
end

local function earliest(
  first: integer | nil, second: integer | nil
): integer | nil
  if first is nil then
    return second
  end
  if second is nil then
    return first
  end
  if first < second then
    return first
  end
  return second
end

local function piece_body(piece: Piece): string
  local word = piece.name
  return bare(piece.source:sub(#word + 1))
end

local function split_pieces(template: string): {Piece} | nil, string | nil
  local pieces: {Piece} = {}
  local length = #template
  local index = 1
  while index <= length do
    local output_at = template:find("{{", index, true)
    local tag_at = template:find("{%", index, true)
    local note_at = template:find("{#", index, true)
    local opening = earliest(earliest(output_at, tag_at), note_at)
    if opening is nil then
      pieces[#pieces + 1] = {
        form = "text",
        source = template:sub(index),
        name = "",
        cut_left = false,
        cut_right = false,
      }
      index = length + 1
    else
      local spot = opening as integer
      if spot > index then
        pieces[#pieces + 1] = {
          form = "text",
          source = template:sub(index, spot - 1),
          name = "",
          cut_left = false,
          cut_right = false,
        }
      end
      local head = template:sub(spot, spot + 1)
      local closer = "%}"
      local form = "tag"
      local label = "tag"
      if head == "{{" then
        closer = "}}"
        form = "output"
        label = "output"
      elseif head == "{#" then
        closer = "#}"
        form = "comment"
        label = "comment"
      end
      local cursor = spot + 2
      local cut_left = template:sub(cursor, cursor) == "-"
      if cut_left then
        cursor = cursor + 1
      end
      local stop_at = template:find(closer, cursor, true)
      if stop_at is nil then
        return nil, fault("an unclosed " .. label)
      end
      local cut_right = template:sub(stop_at - 1, stop_at - 1) == "-"
      local stop = stop_at - 1
      if cut_right then
        stop = stop_at - 2
      end
      local source = bare(template:sub(cursor, stop))
      local name = ""
      if form == "tag" then
        name = source:match("^(%a+)") or ""
        if name == "" then
          return nil, fault("a tag holds no name")
        end
      elseif form == "output" and source == "" then
        return nil, fault("an output holds no expression")
      end
      pieces[#pieces + 1] = {
        form = form,
        source = source,
        name = name,
        cut_left = cut_left,
        cut_right = cut_right,
      }
      index = stop_at + 2
    end
  end

  for position, held in ipairs(pieces) do
    if held.cut_left and position > 1 then
      local before = pieces[position - 1]
      if before.form == "text" then
        before.source = before.source:gsub(WS .. "+$", "")
      end
    end
    if held.cut_right and position < #pieces then
      local after = pieces[position + 1]
      if after.form == "text" then
        after.source = after.source:gsub("^" .. WS .. "+", "")
      end
    end
  end
  return pieces, nil
end

local function next_atom(source: string, walk: Cursor): Atom | nil, string | nil
  local index = walk.at
  local length = #source
  local one = source:sub(index, index)
  if one == " " or one == "\t" or one == "\n" or one == "\r" then
    walk.at = index + 1
    return nil, nil
  end
  if one == "'" or one == '"' then
    local pieces: {string} = {}
    local cursor = index + 1
    local closed = false
    while cursor <= length do
      local next_one = source:sub(cursor, cursor)
      if next_one == "\\" then
        local pair = source:sub(cursor + 1, cursor + 1)
        if pair == "n" then
          pieces[#pieces + 1] = "\n"
        elseif pair == "t" then
          pieces[#pieces + 1] = "\t"
        elseif pair == "r" then
          pieces[#pieces + 1] = "\r"
        elseif pair == "\\" then
          pieces[#pieces + 1] = "\\"
        else
          pieces[#pieces + 1] = pair
        end
        cursor = cursor + 2
      elseif next_one == one then
        closed = true
        break
      else
        pieces[#pieces + 1] = next_one
        cursor = cursor + 1
      end
    end
    if not closed then
      return nil, fault("an expression holds an unclosed string")
    end
    walk.at = cursor + 1
    return {form = "text", text = table.concat(pieces), num = 0}, nil
  end
  if one:match("%d") then
    local digits = source:match("^[%d%._]+", index)
    local amount = tonumber(digits as string)
    if amount is nil then
      return nil, fault("an expression holds a bad number")
    end
    walk.at = index + #(digits as string)
    return {form = "number", text = "", num = amount as number}, nil
  end
  if one == "." and source:sub(index + 1, index + 1):match("%d") then
    local digits = source:match("^[%d%._]+", index)
    local amount = tonumber(digits as string)
    if amount is nil then
      return nil, fault("an expression holds a bad number")
    end
    walk.at = index + #(digits as string)
    return {form = "number", text = "", num = amount as number}, nil
  end
  if one:match("[%a_]") then
    local word = source:match("^[%a_][%w_]*", index) as string
    walk.at = index + #word
    return {form = "word", text = word, num = 0}, nil
  end
  local pair = source:sub(index, index + 1)
  if
    SIGNS[pair]
    and (pair == "==" or pair == "!=" or pair == "<=" or pair == ">=")
  then
    walk.at = index + 2
    return {form = "sign", text = pair, num = 0}, nil
  end
  if SIGNS[one] then
    walk.at = index + 1
    return {form = "sign", text = one, num = 0}, nil
  end
  return nil, fault("an expression holds an unusable character")
end

local function split_atoms(source: string): {Atom} | nil, string | nil
  local atoms: {Atom} = {}
  local walk: Cursor = {at = 1}
  while walk.at <= #source do
    local atom, failure = next_atom(source, walk)
    if failure is string then
      return nil, failure
    end
    if atom is Atom then
      atoms[#atoms + 1] = atom
    end
  end
  atoms[#atoms + 1] = {form = "done", text = "", num = 0}
  return atoms, nil
end

local function here(parser: Parser): Atom
  return parser.atoms[parser.mark]
end

local function ahead(parser: Parser, step: integer): Atom
  local held = parser.atoms[parser.mark + step]
  if held is nil then
    return parser.atoms[#parser.atoms]
  end
  return held
end

local function is_word(parser: Parser, wanted: string): boolean
  local atom = here(parser)
  return atom.form == "word" and atom.text == wanted
end

local function is_sign(parser: Parser, wanted: string): boolean
  local atom = here(parser)
  return atom.form == "sign" and atom.text == wanted
end

local function take_word(parser: Parser, wanted: string): boolean
  if is_word(parser, wanted) then
    parser.mark = parser.mark + 1
    return true
  end
  return false
end

local function take_sign(parser: Parser, wanted: string): boolean
  if is_sign(parser, wanted) then
    parser.mark = parser.mark + 1
    return true
  end
  return false
end

local function new_expr(form: string): Expr
  return {
    form = form,
    text = "",
    num = 0,
    truth = false,
    reverse = false,
    parts = {},
    labels = {},
    left = nil,
    right = nil,
    head = nil,
    tail = nil,
  }
end

local read_expression: function(parser: Parser): Expr | nil, string | nil

local function read_items(parser: Parser): Expr | nil, string | nil
  parser.mark = parser.mark + 1
  local held = new_expr("list")
  if take_sign(parser, "]") then
    return held, nil
  end
  while true do
    local item, failure = read_expression(parser)
    if item is nil then
      return nil, failure
    end
    held.parts[#held.parts + 1] = item
    local comma = take_sign(parser, ",")
    if take_sign(parser, "]") then
      return held, nil
    end
    if not comma then
      return nil, fault("a list is not closed")
    end
  end
end

local function read_atom(parser: Parser): Expr | nil, string | nil
  local atom = here(parser)
  if atom.form == "text" then
    parser.mark = parser.mark + 1
    local held = new_expr("text")
    held.text = atom.text
    return held, nil
  end
  if atom.form == "number" then
    parser.mark = parser.mark + 1
    local held = new_expr("number")
    held.num = atom.num
    return held, nil
  end
  if atom.form == "word" then
    local word = atom.text
    parser.mark = parser.mark + 1
    if word == "true" or word == "True" then
      local held = new_expr("boolean")
      held.truth = true
      return held, nil
    end
    if word == "false" or word == "False" then
      return new_expr("boolean"), nil
    end
    if word == "none" or word == "None" then
      return new_expr("none"), nil
    end
    local held = new_expr("name")
    held.text = word
    return held, nil
  end
  if atom.form == "sign" then
    if atom.text == "(" then
      parser.mark = parser.mark + 1
      local inner, failure = read_expression(parser)
      if inner is nil then
        return nil, failure
      end
      if not take_sign(parser, ")") then
        return nil, fault("an expression holds an unclosed parenthesis")
      end
      return inner, nil
    end
    if atom.text == "[" then
      return read_items(parser)
    end
  end
  return nil, fault("an expression starts with something unusable")
end

local record Arguments
  args: {Expr}
  labels: {string}
end

local function read_arguments(parser: Parser): Arguments | nil, string | nil
  local args: {Expr} = {}
  local labels: {string} = {}
  if take_sign(parser, ")") then
    return {args = args, labels = labels}, nil
  end
  while true do
    local label = ""
    local first = here(parser)
    local second = ahead(parser, 1)
    if
      first.form == "word"
      and second.form == "sign"
      and second.text == "="
      and not (ahead(parser, 2).form == "sign" and ahead(parser, 2).text == "=")
    then
      label = first.text
      parser.mark = parser.mark + 2
    end
    local value, failure = read_expression(parser)
    if value is nil then
      return nil, failure
    end
    args[#args + 1] = value
    labels[#labels + 1] = label
    local comma = take_sign(parser, ",")
    if take_sign(parser, ")") then
      return {args = args, labels = labels}, nil
    end
    if not comma then
      return nil, fault("a call is not closed")
    end
  end
end

local function read_bound(parser: Parser): Expr | nil, string | nil
  if is_sign(parser, "]") or is_sign(parser, ":") then
    return nil, nil
  end
  return read_expression(parser)
end

local function read_postfix(parser: Parser): Expr | nil, string | nil
  local node, failure = read_atom(parser)
  if node is nil then
    return nil, failure
  end
  while true do
    local atom = here(parser)
    if atom.form ~= "sign" then
      return node, nil
    end
    if atom.text == "." then
      parser.mark = parser.mark + 1
      local held = here(parser)
      if held.form ~= "word" then
        return nil, fault("a dot needs a name after it")
      end
      parser.mark = parser.mark + 1
      if is_sign(parser, "(") then
        parser.mark = parser.mark + 1
        local arguments, bad = read_arguments(parser)
        if arguments is nil then
          return nil, bad
        end
        local call = new_expr("method")
        call.left = node
        call.text = held.text
        call.parts = arguments.args
        call.labels = arguments.labels
        node = call
      else
        local access = new_expr("field")
        access.left = node
        access.text = held.text
        node = access
      end
    elseif atom.text == "[" then
      parser.mark = parser.mark + 1
      local first: Expr | nil = nil
      if not is_sign(parser, ":") then
        local head, head_failure = read_expression(parser)
        if head is nil then
          return nil, head_failure
        end
        first = head
      end
      if not is_sign(parser, ":") then
        if not take_sign(parser, "]") then
          return nil, fault("an index is not closed")
        end
        local hit = new_expr("index")
        hit.left = node
        hit.right = first
        node = hit
      else
        parser.mark = parser.mark + 1
        local stop, stop_failure = read_bound(parser)
        if stop_failure is string then
          return nil, stop_failure
        end
        local stride: Expr | nil = nil
        if take_sign(parser, ":") then
          local tail, tail_failure = read_bound(parser)
          if tail_failure is string then
            return nil, tail_failure
          end
          stride = tail
        end
        if not take_sign(parser, "]") then
          return nil, fault("a slice is not closed")
        end
        local cut = new_expr("slice")
        cut.left = node
        cut.head = first
        cut.right = stop
        cut.tail = stride
        node = cut
      end
    elseif atom.text == "(" then
      parser.mark = parser.mark + 1
      local arguments, bad = read_arguments(parser)
      if arguments is nil then
        return nil, bad
      end
      local call = new_expr("call")
      call.left = node
      call.parts = arguments.args
      call.labels = arguments.labels
      node = call
    elseif atom.text == "|" then
      parser.mark = parser.mark + 1
      local held = here(parser)
      if held.form ~= "word" then
        return nil, fault("a filter needs a name after it")
      end
      parser.mark = parser.mark + 1
      local pipe = new_expr("filter")
      pipe.left = node
      pipe.text = held.text
      node = pipe
    else
      return node, nil
    end
  end
end

local COMPARES: {string: boolean} = {
  ["=="] = true,
  ["!="] = true,
  ["<"] = true,
  ["<="] = true,
  [">"] = true,
  [">="] = true,
}

local function read_product(parser: Parser): Expr | nil, string | nil
  if is_sign(parser, "-") then
    parser.mark = parser.mark + 1
    local inner, failure = read_product(parser)
    if inner is nil then
      return nil, failure
    end
    local held = new_expr("negate")
    held.left = inner
    return held, nil
  end
  return read_postfix(parser)
end

local function read_scale(parser: Parser): Expr | nil, string | nil
  local left, failure = read_product(parser)
  if left is nil then
    return nil, failure
  end
  while true do
    local atom = here(parser)
    if not (atom.form == "sign" and (atom.text == "*" or atom.text == "/")) then
      return left, nil
    end
    local op = atom.text
    parser.mark = parser.mark + 1
    local right, right_failure = read_product(parser)
    if right is nil then
      return nil, right_failure
    end
    local held = new_expr("binary")
    held.text = op
    held.left = left
    held.right = right
    left = held
  end
end

local function read_sum(parser: Parser): Expr | nil, string | nil
  local left, failure = read_scale(parser)
  if left is nil then
    return nil, failure
  end
  while true do
    local atom = here(parser)
    if not (atom.form == "sign" and (atom.text == "+" or atom.text == "-")) then
      return left, nil
    end
    local op = atom.text
    parser.mark = parser.mark + 1
    local right, right_failure = read_scale(parser)
    if right is nil then
      return nil, right_failure
    end
    local held = new_expr("binary")
    held.text = op
    held.left = left
    held.right = right
    left = held
  end
end

local function read_join(parser: Parser): Expr | nil, string | nil
  local left, failure = read_sum(parser)
  if left is nil then
    return nil, failure
  end
  if not is_sign(parser, "~") then
    return left, nil
  end
  parser.mark = parser.mark + 1
  local right, right_failure = read_join(parser)
  if right is nil then
    return nil, right_failure
  end
  local held = new_expr("binary")
  held.text = "~"
  held.left = left
  held.right = right
  return held, nil
end

local function read_test(parser: Parser, left: Expr): Expr | nil, string | nil
  parser.mark = parser.mark + 1
  local reverse = take_word(parser, "not")
  local atom = here(parser)
  if atom.form ~= "word" then
    return nil, fault("a test needs a name after it")
  end
  if TESTS[atom.text] == nil then
    return nil, fault("a test named " .. atom.text .. " is not run")
  end
  parser.mark = parser.mark + 1
  local held = new_expr("test")
  held.left = left
  held.text = atom.text
  held.reverse = reverse
  return held, nil
end

local function read_judge(parser: Parser): Expr | nil, string | nil
  local left, failure = read_join(parser)
  if left is nil then
    return nil, failure
  end
  while true do
    local atom = here(parser)
    if atom.form == "sign" and COMPARES[atom.text] == true then
      local op = atom.text
      parser.mark = parser.mark + 1
      local right, right_failure = read_join(parser)
      if right is nil then
        return nil, right_failure
      end
      local held = new_expr("binary")
      held.text = op
      held.left = left
      held.right = right
      left = held
    elseif atom.form == "word" and atom.text == "in" then
      parser.mark = parser.mark + 1
      local right, right_failure = read_join(parser)
      if right is nil then
        return nil, right_failure
      end
      local held = new_expr("binary")
      held.text = "in"
      held.left = left
      held.right = right
      left = held
    elseif
      atom.form == "word"
      and atom.text == "not"
      and ahead(parser, 1).text == "in"
    then
      parser.mark = parser.mark + 2
      local right, right_failure = read_join(parser)
      if right is nil then
        return nil, right_failure
      end
      local held = new_expr("binary")
      held.text = "not in"
      held.left = left
      held.right = right
      left = held
    elseif atom.form == "word" and atom.text == "is" then
      local held, held_failure = read_test(parser, left)
      if held is nil then
        return nil, held_failure
      end
      left = held as Expr
    else
      return left, nil
    end
  end
end

local function read_unless(parser: Parser): Expr | nil, string | nil
  if is_word(parser, "not") then
    parser.mark = parser.mark + 1
    local inner, failure = read_unless(parser)
    if inner is nil then
      return nil, failure
    end
    local held = new_expr("logic")
    held.text = "not"
    held.left = inner
    return held, nil
  end
  return read_judge(parser)
end

local function read_and(parser: Parser): Expr | nil, string | nil
  local left, failure = read_unless(parser)
  if left is nil then
    return nil, failure
  end
  while is_word(parser, "and") do
    parser.mark = parser.mark + 1
    local right, right_failure = read_unless(parser)
    if right is nil then
      return nil, right_failure
    end
    local held = new_expr("logic")
    held.text = "and"
    held.left = left
    held.right = right
    left = held
  end
  return left, nil
end

local function read_union(parser: Parser): Expr | nil, string | nil
  local left, failure = read_and(parser)
  if left is nil then
    return nil, failure
  end
  while is_word(parser, "or") do
    parser.mark = parser.mark + 1
    local right, right_failure = read_and(parser)
    if right is nil then
      return nil, right_failure
    end
    local held = new_expr("logic")
    held.text = "or"
    held.left = left
    held.right = right
    left = held
  end
  return left, nil
end

read_expression = function(parser: Parser): Expr | nil, string | nil
  local left, failure = read_union(parser)
  if left is nil then
    return nil, failure
  end
  if take_word(parser, "if") then
    local test, test_failure = read_union(parser)
    if test is nil then
      return nil, test_failure
    end
    local other: Expr | nil = nil
    if take_word(parser, "else") then
      local tail, tail_failure = read_expression(parser)
      if tail is nil then
        return nil, tail_failure
      end
      other = tail
    end
    local held = new_expr("choice")
    held.left = left
    held.right = test
    held.tail = other
    return held, nil
  end
  return left, nil
end

local function parse_source(source: string): Expr | nil, string | nil
  local atoms, failure = split_atoms(source)
  if atoms is nil then
    return nil, failure
  end
  local parser: Parser = {atoms = atoms as {Atom}, mark = 1}
  local node, parse_failure = read_expression(parser)
  if node is nil then
    return nil, parse_failure
  end
  if here(parser).form ~= "done" then
    return nil, fault("an expression holds extra text")
  end
  return node, nil
end

-- Statements
local IF_STOPS: {string: boolean} = {
  ["else"] = true,
  elif = true,
  endif = true,
}
local ELSE_STOPS: {string: boolean} = {endif = true}
local FOR_STOPS: {string: boolean} = {endfor = true}
local MACRO_STOPS: {string: boolean} = {endmacro = true}

local function new_node(form: string): Node
  return {
    form = form,
    text = "",
    value = nil,
    member = "",
    binding = "",
    entry = nil,
    body = {},
    branches = {},
  }
end

local record Block
  nodes: {Node}
  stop: Piece | nil
end

local read_block: function(
  pieces: {Piece}, walk: Cursor, stops: {string: boolean}
): Block | nil, string | nil

local function read_set(body: string): Node | nil, string | nil
  local target, source = body:match("^(.-)%s*=%s*(.+)$")
  if target is nil or source is nil then
    return nil, fault("tag 'set' holds no '<name> = <value>'")
  end
  local name = bare(target as string)
  local word, member = name:match("^(%a[%w_]*)%.(%a[%w_]*)$")
  local expression, failure = parse_source(bare(source as string))
  if expression is nil then
    return nil, failure
  end
  local held = new_node("set")
  held.value = expression
  if word is string then
    held.binding = word
    held.member = member
    return held, nil
  end
  if name:match("^%a[%w_]*$") == nil then
    return nil, fault("tag 'set' holds an unusable target")
  end
  held.binding = name
  return held, nil
end

local function read_macro(body: string): Macro | nil, string | nil
  local word = body:match("^%a[%w_]*")
  if word is nil then
    return nil, fault("tag 'macro' holds no name")
  end
  local held: Macro = {
    name = word as string,
    params = {},
    optional = {},
    body = {},
  }
  local after = bare(body:sub(#(word as string) + 1))
  if after:sub(1, 1) ~= "(" then
    return nil, fault("tag 'macro' holds no parameter list")
  end
  local stop = after:find(")", 1, true)
  if stop is nil then
    return nil, fault("a macro parameter list is not closed")
  end
  local inner = bare(after:sub(2, (stop as integer) - 1))
  if inner ~= "" then
    for piece in (inner as string):gmatch("[^,]+") do
      local text = bare(piece)
      local name, default = text:match("^(%a[%w_]*)%s*=%s*(.+)$")
      if name is nil and text:match("^%a[%w_]*$") == nil then
        return nil, fault("a macro parameter holds an unusable name")
      end
      if name is nil then
        held.params[#held.params + 1] = text
      else
        local expression, failure = parse_source(bare(default as string))
        if expression is nil then
          return nil, failure
        end
        held.params[#held.params + 1] = name as string
        held.optional[name as string] = expression
      end
    end
  end
  return held, nil
end

read_block = function(
  pieces: {Piece}, walk: Cursor, stops: {string: boolean}
): Block | nil, string | nil
  local nodes: {Node} = {}
  while walk.at <= #pieces do
    local piece = pieces[walk.at]
    walk.at = walk.at + 1
    if piece.form == "text" then
      local held = new_node("text")
      held.text = piece.source
      nodes[#nodes + 1] = held
    elseif piece.form == "comment" then
      -- a comment writes nothing
    elseif piece.form == "output" then
      local expression, failure = parse_source(piece.source)
      if expression is nil then
        return nil, failure
      end
      local held = new_node("output")
      held.value = expression
      nodes[#nodes + 1] = held
    else
      local name = piece.name
      if stops[name] then
        return {nodes = nodes, stop = piece}, nil
      end
      if name == "if" then
        local body = piece_body(piece)
        if body == "" then
          return nil, fault("tag 'if' holds no expression")
        end
        local test, failure = parse_source(body)
        if test is nil then
          return nil, failure
        end
        local branches: {Branch} = {}
        local taken_once = false
        while not taken_once do
          local inner, inner_failure = read_block(pieces, walk, IF_STOPS)
          if inner is nil then
            return nil, inner_failure
          end
          local stop = inner.stop
          if stop is nil then
            return nil, fault("tag 'if' is never closed")
          end
          branches[#branches + 1] = {test = test, body = inner.nodes}
          if stop.name == "elif" then
            local next_body = piece_body(stop)
            if next_body == "" then
              return nil, fault("tag 'elif' holds no expression")
            end
            local next_test, next_failure = parse_source(next_body)
            if next_test is nil then
              return nil, next_failure
            end
            test = next_test
          elseif stop.name == "else" then
            local other, other_failure = read_block(pieces, walk, ELSE_STOPS)
            if other is nil then
              return nil, other_failure
            end
            local other_stop = other.stop
            if other_stop is nil then
              return nil, fault("tag 'if' is never closed")
            end
            branches[#branches + 1] = {test = nil, body = other.nodes}
            taken_once = true
          else
            taken_once = true
          end
        end
        local held = new_node("if")
        held.branches = branches
        nodes[#nodes + 1] = held
      elseif name == "for" then
        local body = piece_body(piece)
        local binding, source = body:match("^(.-)%s+in%s+(.+)$")
        if binding is nil or source is nil then
          return nil, fault("tag 'for' holds no '<name> in <list>'")
        end
        if (binding as string):match("^%a[%w_]*$") == nil then
          return nil, fault("tag 'for' holds no name")
        end
        local list, failure = parse_source(bare(source as string))
        if list is nil then
          return nil, failure
        end
        local inner, inner_failure = read_block(pieces, walk, FOR_STOPS)
        if inner is nil then
          return nil, inner_failure
        end
        local inner_stop = inner.stop
        if inner_stop is nil then
          return nil, fault("tag 'for' is never closed")
        end
        local held = new_node("for")
        held.binding = binding as string
        held.value = list
        held.body = inner.nodes
        nodes[#nodes + 1] = held
      elseif name == "set" then
        local held, failure = read_set(piece_body(piece))
        if held is nil then
          return nil, failure
        end
        nodes[#nodes + 1] = held as Node
      elseif name == "macro" then
        local entry, failure = read_macro(piece_body(piece))
        if entry is nil then
          return nil, failure
        end
        local inner, inner_failure = read_block(pieces, walk, MACRO_STOPS)
        if inner is nil then
          return nil, inner_failure
        end
        local inner_stop = inner.stop
        if inner_stop is nil then
          return nil, fault("tag 'macro' is never closed")
        end
        local held_entry = entry as Macro
        held_entry.body = inner.nodes
        local held = new_node("macro")
        held.binding = held_entry.name
        held.text = held_entry.name
        held.entry = held_entry
        nodes[#nodes + 1] = held
      else
        local held = new_node("refused")
        held.text = name
        nodes[#nodes + 1] = held
      end
    end
  end
  return {nodes = nodes, stop = nil}, nil
end

-- Evaluation
local function lookup(frame: Scope | nil, name: string): Val
  local walk = frame
  while walk is Scope do
    local held = walk.names[name]
    if held is Val then
      return held
    end
    walk = walk.parent
  end
  return UNBOUND
end

local function written(held: Val): string
  if held.kind == "text" then
    return held.text
  end
  if held.kind == "number" then
    return tostring(held.num)
  end
  if held.kind == "boolean" then
    return held.truth and "true" or "false"
  end
  if held.kind == "list" then
    local pieces: {string} = {}
    for _, item in ipairs(held.items) do
      pieces[#pieces + 1] = written(item)
    end
    return "[" .. table.concat(pieces, ", ") .. "]"
  end
  if held.kind == "map" then
    local pieces: {string} = {}
    for _, name in ipairs(held.order) do
      pieces[#pieces + 1] = name .. "=" .. written(held.fields[name] as Val)
    end
    return "{" .. table.concat(pieces, ", ") .. "}"
  end
  return ""
end

local function spot_for(size: integer, wanted: number): integer | nil
  local whole = math.floor(wanted)
  if whole ~= wanted then
    return nil
  end
  local spot = whole + 1
  if whole < 0 then
    spot = size + whole + 1
  end
  if spot < 1 or spot > size then
    return nil
  end
  return spot
end

local function slice_spots(
  size: integer, low: Val | nil, high: Val | nil, stride: Val | nil
): {integer} | nil, string | nil
  local step = 1
  if stride is Val then
    if stride.kind ~= "number" then
      return nil, "a slice step must be a number"
    end
    step = math.floor(stride.num)
    if step ~= stride.num or step == 0 then
      return nil, "a slice step must be a whole number other than zero"
    end
  end
  local start: integer | nil = nil
  local to: integer | nil = nil
  if low is Val then
    if low.kind ~= "number" then
      return nil, "a slice bound must be a number"
    end
    start = math.floor(low.num)
    if start < 0 then
      start = size + start
    end
  end
  if high is Val then
    if high.kind ~= "number" then
      return nil, "a slice bound must be a number"
    end
    to = math.floor(high.num)
    if to < 0 then
      to = size + to
    end
  end
  local spots: {integer} = {}
  if step > 0 then
    local walk = start or 0
    if walk < 0 then
      walk = 0
    end
    local limit = to or size
    if limit > size then
      limit = size
    end
    while walk < limit do
      spots[#spots + 1] = walk + 1
      walk = walk + step
    end
  else
    local walk = (start or (size - 1))
    if walk >= size then
      walk = size - 1
    end
    local limit = to or -1
    if limit < -1 then
      limit = -1
    end
    while walk > limit do
      spots[#spots + 1] = walk + 1
      walk = walk + step
    end
  end
  return spots, nil
end

local function field_of(run: Runner, owner: Val, name: string): Val
  if owner.kind == "unbound" then
    return fail(run, "an expression reads a field of a name that is not bound")
  end
  if owner.kind == "map" then
    local held = owner.fields[name]
    if held is nil then
      return UNBOUND
    end
    return held
  end
  return fail(run, "an expression reads a field of " .. owner.kind)
end

local function index_of(run: Runner, owner: Val, wanted: Val): Val
  if owner.kind == "unbound" then
    return fail(run, "an expression indexes a name that is not bound")
  end
  if owner.kind == "text" then
    if wanted.kind ~= "number" then
      return fail(run, "an expression indexes a string by a non number")
    end
    local marks = letters(owner.text)
    local spot = spot_for(#marks, wanted.num)
    if spot is nil then
      return UNBOUND
    end
    return as_text(marks[spot])
  end
  if owner.kind == "list" then
    if wanted.kind ~= "number" then
      return fail(run, "an expression indexes a list by a non number")
    end
    local spot = spot_for(#owner.items, wanted.num)
    if spot is nil then
      return UNBOUND
    end
    return owner.items[spot]
  end
  if owner.kind == "map" then
    if wanted.kind ~= "text" then
      return fail(run, "an expression indexes a table by a non string")
    end
    local held = owner.fields[wanted.text]
    if held is nil then
      return UNBOUND
    end
    return held
  end
  return fail(run, "an expression indexes " .. owner.kind)
end

local function slice_of(
  run: Runner, owner: Val, low: Val | nil, high: Val | nil, stride: Val | nil
): Val
  if owner.kind == "unbound" then
    return fail(run, "an expression slices a name that is not bound")
  end
  if owner.kind == "text" then
    local marks = letters(owner.text)
    local spots, failure = slice_spots(#marks, low, high, stride)
    if spots is nil then
      return fail(run, failure as string)
    end
    local picked: {string} = {}
    for _, spot in ipairs(spots as {integer}) do
      picked[#picked + 1] = marks[spot]
    end
    return as_text(table.concat(picked))
  end
  if owner.kind == "list" then
    local spots, failure = slice_spots(#owner.items, low, high, stride)
    if spots is nil then
      return fail(run, failure as string)
    end
    local picked: {Val} = {}
    for _, spot in ipairs(spots as {integer}) do
      picked[#picked + 1] = owner.items[spot]
    end
    return as_list(picked)
  end
  return fail(run, "an expression slices " .. owner.kind)
end

local function equal(left: Val, right: Val): boolean
  if left.kind == "unbound" or right.kind == "unbound" then
    return false
  end
  if left.kind == "number" and right.kind == "number" then
    return left.num == right.num
  end
  if left.kind == "boolean" and right.kind == "boolean" then
    return left.truth == right.truth
  end
  if left.kind == "text" and right.kind == "text" then
    return left.text == right.text
  end
  if left.kind == "none" and right.kind == "none" then
    return true
  end
  return left == right
end

local function order_of(run: Runner, op: string, left: Val, right: Val): Val
  local step: number | nil = nil
  if left.kind == "number" and right.kind == "number" then
    step = left.num - right.num
  elseif left.kind == "text" and right.kind == "text" then
    if left.text == right.text then
      step = 0
    elseif left.text < right.text then
      step = -1
    else
      step = 1
    end
  else
    return fail(run, "an expression orders values that cannot be ordered")
  end
  local walk = step as number
  if op == "<" then
    return as_boolean(walk < 0)
  end
  if op == "<=" then
    return as_boolean(walk <= 0)
  end
  if op == ">" then
    return as_boolean(walk > 0)
  end
  return as_boolean(walk >= 0)
end

local function inside(run: Runner, needle: Val, owner: Val): Val
  if owner.kind == "unbound" then
    return fail(
      run, "an expression tests membership of a name that is not bound"
    )
  end
  if owner.kind == "list" then
    for _, item in ipairs(owner.items) do
      if equal(item, needle) then
        return as_boolean(true)
      end
    end
    return as_boolean(false)
  end
  if owner.kind == "text" then
    if needle.kind ~= "text" then
      return fail(
        run, "an expression tests a value that is not text in a string"
      )
    end
    return as_boolean(owner.text:find(needle.text, 1, true) ~= nil)
  end
  if owner.kind == "map" then
    if needle.kind ~= "text" then
      return fail(run, "an expression tests a key that is not text")
    end
    return as_boolean(owner.fields[needle.text] ~= nil)
  end
  return fail(run, "an expression tests membership of " .. owner.kind)
end

local function test_of(run: Runner, name: string, held: Val): Val
  if name == "defined" then
    return as_boolean(held.kind ~= "unbound")
  end
  if name == "undefined" then
    return as_boolean(held.kind == "unbound")
  end
  if name == "none" then
    return as_boolean(held.kind == "none")
  end
  if name == "string" then
    return as_boolean(held.kind == "text")
  end
  if name == "iterable" then
    return as_boolean(
      held.kind == "text" or held.kind == "list" or held.kind == "map"
    )
  end
  if name == "mapping" then
    return as_boolean(held.kind == "map")
  end
  if name == "sequence" then
    return as_boolean(held.kind == "text" or held.kind == "list")
  end
  if name == "true" then
    return as_boolean(held.kind == "boolean" and held.truth)
  end
  if name == "false" then
    return as_boolean(held.kind == "boolean" and not held.truth)
  end
  return fail(run, "a test named " .. name .. " is not run")
end

local function string_form(held: Val): string | nil
  local kind = held.kind
  if kind == "text" then
    return held.text
  end
  if kind == "number" then
    return tostring(held.num)
  end
  if kind == "boolean" then
    return held.truth and "true" or "false"
  end
  if kind == "none" or kind == "unbound" then
    return "null"
  end
  return nil
end

local function filter_of(run: Runner, name: string, held: Val): Val
  if name == "trim" then
    return as_text(bare(written(held)))
  end
  if name == "tojson" then
    return as_text(json_written(held))
  end
  if name == "safe" then
    return held
  end
  if name == "string" then
    local form = string_form(held)
    if form is nil then
      return fail(run, "filter 'string' has no text form")
    end
    return as_text(form as string)
  end
  if name == "length" then
    if held.kind == "text" then
      local size = utf8lib.len(held.text)
      return as_number(size as number)
    end
    if held.kind == "list" then
      return as_number(#held.items)
    end
    if held.kind == "map" then
      local count = 0
      for _ in pairs(held.fields) do
        count = count + 1
      end
      return as_number(count)
    end
    return fail(run, "filter 'length' cannot size " .. held.kind)
  end
  return fail(run, "filter '" .. name .. "' is not run")
end

local function holds(one: string, marks: {string}): boolean
  for _, held in ipairs(marks) do
    if held == one then
      return true
    end
  end
  return false
end

local function trim_marks(
  text: string, marks: {string}, front: boolean, back: boolean
): string
  local held = letters(text)
  local first = 1
  local last = #held
  if front then
    while first <= last and holds(held[first], marks) do
      first = first + 1
    end
  end
  if back then
    while last >= first and holds(held[last], marks) do
      last = last - 1
    end
  end
  local picked: {string} = {}
  for index = first, last do
    picked[#picked + 1] = held[index]
  end
  return table.concat(picked)
end

local function pieces_on(text: string, mark: string): {string}
  local held: {string} = {}
  if mark == "" then
    held[#held + 1] = text
    return held
  end
  local start = 1
  while true do
    local spot = text:find(mark, start, true)
    if spot is nil then
      held[#held + 1] = text:sub(start)
      break
    end
    held[#held + 1] = text:sub(start, spot - 1)
    start = spot + #mark
  end
  return held
end

local function pieces_apart(text: string): {string}
  local held: {string} = {}
  for piece in text:gmatch("%S+") do
    held[#held + 1] = piece
  end
  return held
end

local function patch_all(text: string, needle: string, patch: string): string
  if needle == "" then
    return text
  end
  local held: {string} = {}
  local start = 1
  while true do
    local spot = text:find(needle, start, true)
    if spot is nil then
      held[#held + 1] = text:sub(start)
      break
    end
    held[#held + 1] = text:sub(start, spot - 1)
    held[#held + 1] = patch
    start = spot + #needle
  end
  return table.concat(held)
end

local function method_of(
  run: Runner, owner: Val, name: string, args: {Val}
): Val
  if owner.kind == "unbound" then
    return fail(
      run, "an expression calls '" .. name .. "' on a name that is not bound"
    )
  end
  local wanted = METHODS[name]
  if wanted is nil then
    return fail(run, "method '" .. name .. "' is not run")
  end
  if owner.kind ~= "text" then
    return fail(run, "an expression calls '" .. name .. "' on " .. owner.kind)
  end
  if #args + 1 > (wanted as integer) then
    return fail(run, "method '" .. name .. "' holds too many arguments")
  end
  if name == "strip" or name == "lstrip" or name == "rstrip" then
    if #args == 0 then
      if name == "strip" then
        return as_text(bare(owner.text))
      end
      if name == "lstrip" then
        return as_text((owner.text:gsub("^" .. WS .. "+", "")))
      end
      return as_text((owner.text:gsub(WS .. "+$", "")))
    end
    local marks = args[1]
    if marks.kind ~= "text" then
      return fail(run, "method '" .. name .. "' needs a string argument")
    end
    if marks.text == "" then
      return as_text(owner.text)
    end
    return as_text(
      trim_marks(
        owner.text, letters(marks.text), name ~= "rstrip", name ~= "lstrip"
      )
    )
  end
  if name == "split" then
    local held: {string} = {}
    if #args == 0 then
      held = pieces_apart(owner.text)
    else
      if args[1].kind ~= "text" then
        return fail(run, "method 'split' needs a string argument")
      end
      held = pieces_on(owner.text, args[1].text)
    end
    local items: {Val} = {}
    for _, piece in ipairs(held) do
      items[#items + 1] = as_text(piece)
    end
    return as_list(items)
  end
  if name == "startswith" or name == "endswith" then
    local mark = ""
    if #args > 0 then
      if args[1].kind ~= "text" then
        return fail(run, "method '" .. name .. "' needs a string argument")
      end
      mark = args[1].text
    end
    if mark == "" then
      return as_boolean(true)
    end
    if name == "startswith" then
      return as_boolean(owner.text:sub(1, #mark) == mark)
    end
    if #owner.text < #mark then
      return as_boolean(false)
    end
    return as_boolean(owner.text:sub(#owner.text - #mark + 1) == mark)
  end
  if #args == 0 then
    return as_text(owner.text)
  end
  if #args < 2 then
    return fail(run, "method 'replace' needs two string arguments")
  end
  if args[1].kind ~= "text" or args[2].kind ~= "text" then
    return fail(run, "method 'replace' needs two string arguments")
  end
  return as_text(patch_all(owner.text, args[1].text, args[2].text))
end

local evaluate: function(run: Runner, node: Expr): Val

local walk_nodes: function(run: Runner, nodes: {Node}): string | nil

local function call_macro(
  run: Runner, held: Macro, args: {Val}, labels: {string}
): Val
  if run.depth > 20 then
    return fail(run, "a macro call nests too deeply")
  end
  local names: {string: Val} = {}
  for _, name in ipairs(held.params) do
    names[name] = UNBOUND
  end
  for _, name in ipairs(held.params) do
    local default = held.optional[name]
    if default is Expr then
      names[name] = evaluate(run, default)
      if run.broken then
        return UNBOUND
      end
    end
  end
  for position, value in ipairs(args) do
    local mark = labels[position]
    if mark == "" then
      local name = held.params[position]
      if name is nil then
        return fail(run, "a macro call holds too many arguments")
      end
      names[name] = value
    else
      if names[mark] == nil then
        return fail(run, "a macro call names " .. mark)
      end
      names[mark] = value
    end
  end
  local frame: Scope = {names = names, parent = run.root}
  local saved_frame = run.frame
  local saved_parts = run.parts
  local held_parts: {string} = {}
  run.frame = frame
  run.parts = held_parts
  run.depth = run.depth + 1
  local failure = walk_nodes(run, held.body)
  run.depth = run.depth - 1
  run.frame = saved_frame
  run.parts = saved_parts
  if failure is string then
    run.broken = true
    run.fault = failure
    return UNBOUND
  end
  return as_text(table.concat(held_parts))
end

evaluate = function(run: Runner, node: Expr): Val
  if run.broken then
    return UNBOUND
  end
  local form = node.form
  if form == "text" then
    return as_text(node.text)
  end
  if form == "number" then
    return as_number(node.num)
  end
  if form == "boolean" then
    return as_boolean(node.truth)
  end
  if form == "none" then
    return NOTHING
  end
  if form == "name" then
    return lookup(run.frame, node.text)
  end
  if form == "list" then
    local items: {Val} = {}
    for _, item in ipairs(node.parts) do
      items[#items + 1] = evaluate(run, item)
      if run.broken then
        return UNBOUND
      end
    end
    return as_list(items)
  end
  local owner = evaluate(run, node.left as Expr)
  if run.broken then
    return UNBOUND
  end
  if form == "field" then
    return field_of(run, owner, node.text)
  end
  if form == "index" then
    local wanted = evaluate(run, node.right as Expr)
    if run.broken then
      return UNBOUND
    end
    return index_of(run, owner, wanted)
  end
  if form == "slice" then
    local low: Val | nil = nil
    if node.head ~= nil then
      low = evaluate(run, node.head as Expr)
    end
    local high: Val | nil = nil
    if node.right ~= nil then
      high = evaluate(run, node.right as Expr)
    end
    local stride: Val | nil = nil
    if node.tail ~= nil then
      stride = evaluate(run, node.tail as Expr)
    end
    if run.broken then
      return UNBOUND
    end
    return slice_of(run, owner, low, high, stride)
  end
  if form == "method" then
    local args: {Val} = {}
    for _, item in ipairs(node.parts) do
      args[#args + 1] = evaluate(run, item)
      if run.broken then
        return UNBOUND
      end
    end
    return method_of(run, owner, node.text, args)
  end
  if form == "filter" then
    return filter_of(run, node.text, owner)
  end
  if form == "test" then
    local answer = test_of(run, node.text, owner)
    if run.broken then
      return UNBOUND
    end
    if node.reverse then
      return as_boolean(not answer.truth)
    end
    return answer
  end
  if form == "negate" then
    if owner.kind ~= "number" then
      return fail(run, "an expression negates " .. owner.kind)
    end
    return as_number(-owner.num)
  end
  if form == "logic" then
    if node.text == "not" then
      return as_boolean(not owner.truth)
    end
    if node.text == "and" then
      if not owner.truth then
        return owner
      end
      local right = evaluate(run, node.right as Expr)
      if run.broken then
        return UNBOUND
      end
      return right
    end
    if owner.truth then
      return owner
    end
    local right = evaluate(run, node.right as Expr)
    if run.broken then
      return UNBOUND
    end
    return right
  end
  if form == "choice" then
    local test = evaluate(run, node.right as Expr)
    if run.broken then
      return UNBOUND
    end
    if test.truth then
      return evaluate(run, node.left as Expr)
    end
    if node.tail ~= nil then
      return evaluate(run, node.tail as Expr)
    end
    return UNBOUND
  end
  if form == "binary" then
    local right = evaluate(run, node.right as Expr)
    if run.broken then
      return UNBOUND
    end
    local op = node.text
    if op == "in" then
      return inside(run, owner, right)
    end
    if op == "not in" then
      local answer = inside(run, owner, right)
      if run.broken then
        return UNBOUND
      end
      return as_boolean(not answer.truth)
    end
    if op == "==" then
      return as_boolean(equal(owner, right))
    end
    if op == "!=" then
      return as_boolean(not equal(owner, right))
    end
    if op == "<" or op == "<=" or op == ">" or op == ">=" then
      return order_of(run, op, owner, right)
    end
    if op == "~" then
      return as_text(written(owner) .. written(right))
    end
    if op == "+" then
      if owner.kind == "number" and right.kind == "number" then
        return as_number(owner.num + right.num)
      end
      if owner.kind == "text" and right.kind == "text" then
        return as_text(owner.text .. right.text)
      end
      if owner.kind == "list" and right.kind == "list" then
        local items: {Val} = {}
        for _, item in ipairs(owner.items) do
          items[#items + 1] = item
        end
        for _, item in ipairs(right.items) do
          items[#items + 1] = item
        end
        return as_list(items)
      end
      return fail(
        run, "an expression adds " .. owner.kind .. " and " .. right.kind
      )
    end
    if op == "-" then
      if owner.kind == "number" and right.kind == "number" then
        return as_number(owner.num - right.num)
      end
      return fail(run, "an expression subtracts non numbers")
    end
    if op == "*" then
      if owner.kind == "number" and right.kind == "number" then
        return as_number(owner.num * right.num)
      end
      return fail(run, "an expression multiplies non numbers")
    end
    return fail(run, "an expression applies " .. op)
  end
  if form == "call" then
    local callee = node.left as Expr
    if callee.form == "name" then
      if callee.text == "raise_exception" then
        local args: {Val} = {}
        for _, item in ipairs(node.parts) do
          args[#args + 1] = evaluate(run, item)
          if run.broken then
            return UNBOUND
          end
        end
        if #args ~= 1 or args[1].kind ~= "text" then
          return fail(run, "raise_exception needs one text argument")
        end
        return fail(run, args[1].text)
      end
      if callee.text == "namespace" then
        local held = as_map()
        for position, item in ipairs(node.parts) do
          local value = evaluate(run, item)
          if run.broken then
            return UNBOUND
          end
          local mark = node.labels[position]
          if mark == "" then
            return fail(run, "namespace() needs named initializers")
          end
          map_put(held, mark, value)
        end
        return held
      end
    end
    local args: {Val} = {}
    for _, item in ipairs(node.parts) do
      args[#args + 1] = evaluate(run, item)
      if run.broken then
        return UNBOUND
      end
    end
    if owner.kind == "unbound" then
      return fail(run, "an expression calls a name that is not bound")
    end
    if owner.kind == "macro" then
      return call_macro(run, owner.code as Macro, args, node.labels)
    end
    return fail(run, "an expression calls " .. owner.kind)
  end
  return fail(run, "an expression holds " .. form)
end

local function loop_object(size: integer, position: integer, items: {Val}): Val
  local held = as_map()
  map_put(held, "index0", as_number(position - 1))
  map_put(held, "index", as_number(position))
  map_put(held, "first", as_boolean(position == 1))
  map_put(held, "last", as_boolean(position == size))
  if position > 1 then
    map_put(held, "previtem", items[position - 1])
  end
  if position < size then
    map_put(held, "nextitem", items[position + 1])
  end
  return held
end

local function walk_over(run: Runner, held: Val): {Val} | nil
  if held.kind == "list" then
    return held.items
  end
  if held.kind == "map" then
    local keys: {Val} = {}
    for _, name in ipairs(held.order) do
      keys[#keys + 1] = as_text(name)
    end
    return keys
  end
  if held.kind == "text" then
    local items: {Val} = {}
    for _, one in ipairs(letters(held.text)) do
      items[#items + 1] = as_text(one)
    end
    return items
  end
  fail(run, "tag 'for' cannot walk " .. held.kind)
  return nil
end

walk_nodes = function(run: Runner, nodes: {Node}): string | nil
  for _, node in ipairs(nodes) do
    if run.broken then
      return run.fault
    end
    local form = node.form
    if form == "text" then
      run.parts[#run.parts + 1] = node.text
    elseif form == "output" then
      local held = evaluate(run, node.value as Expr)
      if run.broken then
        return run.fault
      end
      run.parts[#run.parts + 1] = written(held)
    elseif form == "if" then
      for _, branch in ipairs(node.branches) do
        local taken = false
        if branch.test == nil then
          taken = true
        else
          local answer = evaluate(run, branch.test as Expr)
          if run.broken then
            return run.fault
          end
          taken = answer.truth
        end
        if taken then
          local failure = walk_nodes(run, branch.body)
          if failure is string then
            return failure
          end
          break
        end
      end
    elseif form == "for" then
      local held = evaluate(run, node.value as Expr)
      if run.broken then
        return run.fault
      end
      local items = walk_over(run, held)
      if items is nil then
        return run.fault
      end
      local list = items as {Val}
      local frame: Scope = {names = {}, parent = run.frame}
      local saved = run.frame
      run.frame = frame
      for position = 1, #list do
        frame.names[node.binding] = list[position]
        frame.names.loop = loop_object(#list, position, list)
        local failure = walk_nodes(run, node.body)
        if failure is string then
          run.frame = saved
          return failure
        end
      end
      run.frame = saved
    elseif form == "set" then
      local held = evaluate(run, node.value as Expr)
      if run.broken then
        return run.fault
      end
      if node.member == "" then
        run.frame.names[node.binding] = held
      else
        local owner = lookup(run.frame, node.binding)
        if owner.kind == "unbound" then
          return halt(
            run, "tag 'set' writes a member of a name that is not bound"
          )
        end
        if owner.kind ~= "map" then
          return halt(run, "tag 'set' writes through " .. owner.kind)
        end
        map_put(owner, node.member, held)
      end
    elseif form == "macro" then
      local held = say("macro")
      held.truth = true
      held.code = node.entry
      run.frame.names[node.binding] = held
    else
      return halt(run, "tag '" .. node.text .. "' is not run")
    end
  end
  return nil
end

local templates: {string: Block} = {}
local template_failures: {string: string} = {}

function chat.render(
  built: tokenizer.Tokenizer,
  messages: {Msg},
  tools?: json.Value,
  enable_thinking?: boolean,
  tool_choice?: json.Value
): string | nil, string | nil
  local template = built.template
  if template is nil then
    return nil, fault("the file states none")
  end

  if #messages == 0 then
    return nil, "messages: the list is empty"
  end
  for index = 1, #messages do
    local message = messages[index]
    if type(message) ~= "table" then
      return nil,
        "messages: entry "
          .. tostring(index)
          .. " is not a table with string role and content"
    end
    if type(message.role) ~= "string" or type(message.content) ~= "string" then
      return nil,
        "messages: entry "
          .. tostring(index)
          .. " is not a table with string role and content"
    end
  end

  -- the template is a constant of the file, so its tree is lexed and parsed
  -- once and read again for every request that renders it
  local block = templates[template as string]
  if block is nil then
    local refused_template = template_failures[template as string]
    if refused_template is string then
      return nil, refused_template
    end
    local pieces, lex_failure = split_pieces(template as string)
    if pieces is nil then
      template_failures[template as string] = lex_failure as string
      return nil, lex_failure
    end

    local walk: Cursor = {at = 1}
    local stops: {string: boolean} = {}
    local read, parse_failure = read_block(pieces as {Piece}, walk, stops)
    if read is nil then
      template_failures[template as string] = parse_failure as string
      return nil, parse_failure
    end
    block = read
    templates[template as string] = block
  end

  local held_block = block as Block
  local stop = held_block.stop
  if stop is Piece then
    return nil, fault("tag '" .. tostring(stop.name) .. "' is out of place")
  end

  local items: {Val} = {}
  for _, message in ipairs(messages) do
    items[#items + 1] = as_message(message)
  end
  local names: {string: Val} = {
    messages = as_list(items),
    add_generation_prompt = as_boolean(true),
  }
  if tools is json.Value then
    names.tools = of_json(tools)
  end
  if enable_thinking is boolean then
    names.enable_thinking = as_boolean(enable_thinking)
  end
  if tool_choice is json.Value then
    names.tool_choice = of_json(tool_choice)
  end
  local root: Scope = {names = names, parent = nil}
  local run: Runner = {
    root = root,
    frame = root,
    parts = {},
    fault = "",
    broken = false,
    depth = 0,
  }
  local failure = walk_nodes(run, held_block.nodes)
  if failure is string then
    return nil, failure
  end
  return table.concat(run.parts), nil
end

return chat
