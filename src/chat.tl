local tokenizer = require("tllm.tokenizer")

local record chat
  record Message
    role: string
    content: string
  end

  render: function(tokenizer.Tokenizer, {Message}): string | nil, string | nil
end

local type Tokenizer = tokenizer.Tokenizer
local type Message = chat.Message
local type Value = nil | boolean | number | string | Message
local type Chunk = function(): Value

local IF_STOPS: {string: boolean} = {
  ["elif"] = true,
  ["else"] = true,
  ["endif"] = true,
}
local ELSE_STOPS: {string: boolean} = {["endif"] = true}
local FOR_STOPS: {string: boolean} = {["endfor"] = true}

local KEY = "tokenizer.chat_template"
local EDGE = "[ \t\r\n]"

local record Token
  kind: string
  text: string
  name: string | nil
  left_trim: boolean
  right_trim: boolean
end

local record Branch
  call: Chunk | nil
  body: {Node}
end

local record Node
  kind: string
  text: string | nil
  call: Chunk | nil
  filters: {string} | nil
  branches: {Branch} | nil
  name: string | nil
  body: {Node} | nil
end

local record State
  tokens: {Token}
  index: integer
  environment: {string: Value}
end

local function refused(index: integer): string
  return "messages: entry "
    .. tostring(index)
    .. " is not a table with string role and content"
end

local function fault(detail: string): string
  local cleaned: string = detail:gsub("[\r\n]+", " ")
  return KEY .. ": " .. cleaned
end

local function stripped(text: string): string
  local leading: string = text:gsub("^" .. EDGE .. "+", "")
  local both: string = leading:gsub(EDGE .. "+$", "")
  return both
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

local function lexed(template: string): {Token} | nil, string | nil
  local tokens: {Token} = {}
  local length = #template
  local index = 1
  while index <= length do
    local output_at = template:find("{{", index, true)
    local tag_at = template:find("{%", index, true)
    local comment_at = template:find("{#", index, true)
    local opening = earliest(earliest(output_at, tag_at), comment_at)
    if opening is nil then
      tokens[#tokens + 1] = {
        kind = "text",
        text = template:sub(index),
        left_trim = false,
        right_trim = false,
      }
      index = length + 1
    elseif opening == comment_at then
      return nil, fault("tag '#' is not run")
    else
      if opening > index then
        tokens[#tokens + 1] = {
          kind = "text",
          text = template:sub(index, opening - 1),
          left_trim = false,
          right_trim = false,
        }
      end
      local is_output = template:sub(opening, opening + 1) == "{{"
      local closer = "%}"
      local label = "tag"
      if is_output then
        closer = "}}"
        label = "output"
      end
      local cursor = opening + 2
      local left_trim = template:sub(cursor, cursor) == "-"
      if left_trim then
        cursor = cursor + 1
      end
      local close_at = template:find(closer, cursor, true)
      if close_at is nil then
        return nil, fault("an unclosed " .. label)
      end
      local right_trim = template:sub(close_at - 1, close_at - 1) == "-"
      local stop = close_at - 1
      if right_trim then
        stop = close_at - 2
      end
      local content = stripped(template:sub(cursor, stop))
      if is_output then
        if content == "" then
          return nil, fault("an output holds no expression")
        end
        tokens[#tokens + 1] = {
          kind = "output",
          text = content,
          left_trim = left_trim,
          right_trim = right_trim,
        }
      else
        local name = content:match("^(%a+)")
        if name is nil then
          return nil, fault("a tag holds no name")
        end
        tokens[#tokens + 1] = {
          kind = "tag",
          text = content,
          name = name,
          left_trim = left_trim,
          right_trim = right_trim,
        }
      end
      index = close_at + 2
    end
  end

  for position, held in ipairs(tokens) do
    if held.left_trim and position > 1 then
      local previous = tokens[position - 1]
      if previous.kind == "text" then
        local cleaned: string = previous.text:gsub(EDGE .. "+$", "")
        previous.text = cleaned
      end
    end
    if held.right_trim and position < #tokens then
      local following = tokens[position + 1]
      if following.kind == "text" then
        local cleaned: string = following.text:gsub("^" .. EDGE .. "+", "")
        following.text = cleaned
      end
    end
  end
  return tokens, nil
end

local function split_filters(
  source: string
): string | nil, {string} | nil, string | nil
  local pieces: {string} = {}
  local start = 1
  local quote: string | nil = nil
  local escaped = false
  local depth = 0
  for position = 1, #source do
    local character = source:sub(position, position)
    if quote is nil then
      if character == "'" or character == '"' then
        quote = character
      elseif character == "(" or character == "[" or character == "{" then
        depth = depth + 1
      elseif character == ")" or character == "]" or character == "}" then
        if depth > 0 then
          depth = depth - 1
        end
      elseif character == "|" and depth == 0 then
        pieces[#pieces + 1] = source:sub(start, position - 1)
        start = position + 1
      end
    elseif escaped then
      escaped = false
    elseif character == "\\" then
      escaped = true
    elseif character == quote then
      quote = nil
    end
  end
  pieces[#pieces + 1] = source:sub(start)

  local filters: {string} = {}
  for index = 2, #pieces do
    local name = stripped(pieces[index])
    if name ~= "trim" then
      return nil, nil, fault("filter '" .. name .. "' is not run")
    end
    filters[#filters + 1] = name
  end
  return stripped(pieces[1]), filters, nil
end

local function compiled(
  expression: string, environment: {string: Value}, label: string | nil
): Chunk | nil, string | nil
  local chunk: Chunk | nil = load(
    "return (" .. expression .. ")", "chat_template", "t", environment
  )
  if chunk is nil then
    local named = ""
    if not (label is nil) then
      named = label .. ": "
    end
    return nil, fault(named .. "an expression will not parse")
  end
  return chunk, nil
end

local function output_node(
  token: Token, environment: {string: Value}
): Node | nil, string | nil
  local expression, filters, failure = split_filters(token.text)
  if expression is nil then
    return nil, failure
  end
  local call, compile_failure = compiled(expression as string, environment, nil)
  if call is nil then
    return nil, compile_failure
  end
  local held_filters = filters as {string}
  return {kind = "output", call = call, filters = held_filters}, nil
end

local function tag_expression(token: Token): string
  local name = token.name as string
  return stripped(token.text:sub(#name + 1))
end

local function parse_nodes(
  state: State, stops: {string: boolean}
): {Node} | nil, Token | nil, string | nil
  local nodes: {Node} = {}
  while state.index <= #state.tokens do
    local token = state.tokens[state.index]
    state.index = state.index + 1
    if token.kind == "text" then
      nodes[#nodes + 1] = {kind = "text", text = token.text}
    elseif token.kind == "output" then
      local node, failure = output_node(token, state.environment)
      if node is nil then
        return nil, nil, failure
      end
      nodes[#nodes + 1] = node as Node
    else
      local name = token.name as string
      if stops[name] then
        return nodes, token, nil
      elseif name == "if" then
        local condition = tag_expression(token)
        if condition == "" then
          return nil, nil, fault("tag 'if' holds no expression")
        end
        local call, failure = compiled(condition, state.environment, "if")
        if call is nil then
          return nil, nil, failure
        end
        local branches: {Branch} = {}
        local closed = false
        while not closed do
          local body, stop, body_failure = parse_nodes(state, IF_STOPS)
          if body is nil then
            return nil, nil, body_failure
          end
          local held_body = body as {Node}
          branches[#branches + 1] = {call = call, body = held_body}
          if stop is nil then
            return nil, nil, fault("tag 'if' is never closed")
          end
          local stop_name = stop.name as string
          if stop_name == "elif" then
            local next_condition = tag_expression(stop)
            if next_condition == "" then
              return nil, nil, fault("tag 'elif' holds no expression")
            end
            call, failure = compiled(next_condition, state.environment, "elif")
            if call is nil then
              return nil, nil, failure
            end
          elseif stop_name == "else" then
            local else_body, else_stop, else_failure = parse_nodes(
              state, ELSE_STOPS
            )
            if else_body is nil then
              return nil, nil, else_failure
            end
            if else_stop is nil then
              return nil, nil, fault("tag 'if' is never closed")
            end
            local held_else = else_body as {Node}
            branches[#branches + 1] = {body = held_else}
            closed = true
          else
            closed = true
          end
        end
        nodes[#nodes + 1] = {kind = "if", branches = branches}
      elseif name == "for" then
        local binding, source = tag_expression(token):match(
          "^([%a_][%w_]*)%s+in%s+(.+)$"
        )
        if binding is nil or source is nil then
          return nil, nil, fault("tag 'for' holds no '<name> in <list>'")
        end
        local call, failure = compiled(
          source as string, state.environment, "for"
        )
        if call is nil then
          return nil, nil, failure
        end
        local body, stop, body_failure = parse_nodes(state, FOR_STOPS)
        if body is nil then
          return nil, nil, body_failure
        end
        if stop is nil then
          return nil, nil, fault("tag 'for' is never closed")
        end
        local held_body = body as {Node}
        nodes[#nodes + 1] = {
          kind = "for",
          name = binding as string,
          call = call,
          body = held_body,
        }
      else
        return nil, nil, fault("tag '" .. name .. "' is not run")
      end
    end
  end
  return nodes, nil, nil
end

local function rendered(
  nodes: {Node}, environment: {string: Value}, parts: {string}
): string | nil
  for _, node in ipairs(nodes) do
    if node.kind == "text" then
      parts[#parts + 1] = node.text as string
    elseif node.kind == "output" then
      local call = node.call as Chunk
      local ok, value = pcall(call)
      if not ok then
        return fault("an output raised: " .. tostring(value))
      end
      if not (value is nil) then
        local written = tostring(value)
        for _, filter in ipairs(node.filters as {string}) do
          if filter == "trim" then
            written = stripped(written)
          end
        end
        parts[#parts + 1] = written
      end
    elseif node.kind == "if" then
      for _, branch in ipairs(node.branches as {Branch}) do
        local branch_call = branch.call
        local taken = false
        if branch_call is nil then
          taken = true
        else
          local ok, value = pcall(branch_call)
          if not ok then
            return fault("tag 'if' raised: " .. tostring(value))
          end
          taken = not (value is nil) and value ~= false
        end
        if taken then
          local failure = rendered(branch.body, environment, parts)
          if not (failure is nil) then
            return failure
          end
          break
        end
      end
    elseif node.kind == "for" then
      local call = node.call as Chunk
      local ok, value = pcall(call)
      if not ok then
        return fault("tag 'for' raised: " .. tostring(value))
      end
      if type(value) ~= "table" then
        return fault("tag 'for' needs a list")
      end
      local name = node.name as string
      local previous = environment[name]
      for _, item in ipairs(value as {Message}) do
        environment[name] = item
        local failure = rendered(node.body as {Node}, environment, parts)
        if not (failure is nil) then
          environment[name] = previous
          return failure
        end
      end
      environment[name] = previous
    end
  end
  return nil
end

function chat.render(
  built: Tokenizer, messages: {Message}
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
      return nil, refused(index)
    end
    if type(message.role) ~= "string" or type(message.content) ~= "string" then
      return nil, refused(index)
    end
  end

  local tokens, lex_failure = lexed(template as string)
  if tokens is nil then
    return nil, lex_failure
  end

  -- the environment holds a message list under `messages`, but Value has one
  -- table arm, so this cast is what the compiler accepts
  local environment: {string: Value} = {
    messages = messages as Message,
    add_generation_prompt = true,
  }
  local state: State = {
    tokens = tokens as {Token},
    index = 1,
    environment = environment,
  }
  local nothing: {string: boolean} = {}
  local nodes, stop, parse_failure = parse_nodes(state, nothing)
  if nodes is nil then
    return nil, parse_failure
  end
  if not (stop is nil) then
    return nil, fault("tag '" .. tostring(stop.name) .. "' is out of place")
  end

  local parts: {string} = {}
  local render_failure = rendered(nodes as {Node}, environment, parts)
  if not (render_failure is nil) then
    return nil, render_failure
  end
  return table.concat(parts), nil
end

return chat
