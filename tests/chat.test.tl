local fs = require("fs")
local test = require("lde-test")
local chat = require("tllm.chat")
local tokenizer = require("tllm.tokenizer")

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function u32(value: integer): string
  return string.char(
    value % 256,
    math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256,
    math.floor(value / 16777216) % 256
  )
end

local function u64(value: integer): string
  local parts: {string} = {}
  local rest = value
  for _ = 1, 8 do
    parts[#parts + 1] = string.char(rest % 256)
    rest = math.floor(rest / 256)
  end
  return table.concat(parts)
end

local function header(tensor_count: integer, metadata_count: integer): string
  return "GGUF" .. u32(3) .. u64(tensor_count) .. u64(metadata_count)
end

local function text(bytes: string): string
  return u64(#bytes) .. bytes
end

local function pair(name: string, code: integer, bytes: string): string
  return text(name) .. u32(code) .. bytes
end

local function array(item_code: integer, items: {string}): string
  return u32(item_code) .. u64(#items) .. table.concat(items)
end

local function strings(items: {string}): string
  local encoded: {string} = {}
  for _, item in ipairs(items) do
    encoded[#encoded + 1] = text(item)
  end
  return array(8, encoded)
end

local function ints(values: {integer}): string
  local encoded: {string} = {}
  for _, value in ipairs(values) do
    encoded[#encoded + 1] = u32(value)
  end
  return array(5, encoded)
end

local function fixture(extra: {string}): string
  local entries: {string} = {
    pair("tokenizer.ggml.model", 8, text("gpt2")),
    pair("tokenizer.ggml.pre", 8, text("gpt2")),
    pair("tokenizer.ggml.tokens", 9, strings({"a", "b"})),
    pair("tokenizer.ggml.token_type", 9, ints({1, 1})),
    pair("tokenizer.ggml.merges", 9, strings({})),
  }
  for _, entry in ipairs(extra) do
    entries[#entries + 1] = entry
  end
  created = created + 1
  local file_path = ROOT .. "/c" .. tostring(created) .. ".gguf"
  fs.write(file_path, header(0, #entries) .. table.concat(entries))
  return file_path
end

local function template_entry(template: string): string
  return pair("tokenizer.chat_template", 8, text(template))
end

local function loaded(extra: {string}): tokenizer.Tokenizer
  local file_path = fixture(extra)
  local built, failure = tokenizer.load(file_path)
  test.equal(failure, nil, file_path)
  return built as tokenizer.Tokenizer
end

local function with_template(template: string): tokenizer.Tokenizer
  return loaded({template_entry(template)})
end

local PROMPT = "{% for message in messages %}<|im_start|>"
  .. "{{ message.role }}\n{{ message.content }}<|im_end|>\n"
  .. "{% endfor %}{% if add_generation_prompt %}"
  .. "<|im_start|>assistant\n{% endif %}"

local PROMPT_TEXT = "<|im_start|>system\nbe brief<|im_end|>\n"
  .. "<|im_start|>user\nhi<|im_end|>\n"
  .. "<|im_start|>assistant\n"

local ONE: {chat.Message} = {{role = "user", content = "hi"}}

test.it(
  "messages become the prompt the template writes",
  function()
    local built = loaded({template_entry(PROMPT)})
    test.equal(built.template, PROMPT)

    local prompt, failure = chat.render(
      built,
      {
        {role = "system", content = "be brief"},
        {role = "user", content = "hi"},
      }
    )
    test.equal(failure, nil)
    test.equal(prompt, PROMPT_TEXT)
  end
)

test.it(
  "branches, trim markers and the trim filter shape the output",
  function()
    local trimmed = with_template(
      "a {%- if add_generation_prompt -%} b {%- endif -%} c"
    )
    local written, failure = chat.render(trimmed, ONE)
    test.equal(failure, nil)
    test.equal(written, "abc")

    local branched = with_template(
      "{% for message in messages %}"
        .. "{% if message.role == 'user' %}U"
        .. "{% elif message.role == 'assistant' %}A"
        .. "{% else %}S{% endif %}"
        .. "{{ message.content|trim }}{{ message.missing }}{% endfor %}"
    )
    written, failure = chat.render(
      branched,
      {
        {role = "user", content = " hi "},
        {role = "tool", content = "x"},
      }
    )
    test.equal(failure, nil)
    test.equal(written, "UhiSx")

    local numbered = with_template("a\n{{- 5 -}}\nc")
    written, failure = chat.render(numbered, ONE)
    test.equal(failure, nil)
    test.equal(written, "a5c")
  end
)

test.it(
  "a template or messages that cannot be used are refused",
  function()
    local bare = loaded({})
    test.equal(bare.template, nil)
    local written, failure = chat.render(bare, {})
    test.equal(written, nil)
    test.includes(failure as string, "tokenizer.chat_template")

    local numeric = loaded({pair("tokenizer.chat_template", 5, u32(7))})
    written, failure = chat.render(numeric, ONE)
    test.equal(written, nil)
    test.includes(failure as string, "tokenizer.chat_template")

    local _, macro_failure = chat.render(
      with_template("{% macro m() %}{% endmacro %}"), ONE
    )
    test.truthy(macro_failure)
    test.includes(macro_failure as string, "tokenizer.chat_template")
    test.includes(macro_failure as string, "macro")

    local _, unclosed_for = chat.render(
      with_template("{% for message in messages %}"), ONE
    )
    test.includes(unclosed_for as string, "tokenizer.chat_template")

    local _, bare_if = chat.render(with_template("{% if %}"), ONE)
    test.includes(bare_if as string, "tokenizer.chat_template")

    local _, stray_endif = chat.render(with_template("{% endif %}"), ONE)
    test.includes(stray_endif as string, "tokenizer.chat_template")

    local _, tojson = chat.render(with_template("{{ 'x'|tojson }}"), ONE)
    test.includes(tojson as string, "tokenizer.chat_template")
    test.includes(tojson as string, "tojson")

    local _, absent = chat.render(with_template("{{ missing.field }}"), ONE)
    test.includes(absent as string, "tokenizer.chat_template")

    local _, io_reach = chat.render(with_template("{{ io.open('x') }}"), ONE)
    test.includes(io_reach as string, "tokenizer.chat_template")

    local promised = loaded({template_entry(PROMPT)})
    local _, empty = chat.render(promised, {})
    test.truthy(empty)
    test.includes(empty as string, "messages")

    local broken = {role = "user", content = 5} as chat.Message
    local _, mixed = chat.render(promised, {ONE[1], broken})
    test.includes(mixed as string, "messages")
    test.includes(mixed as string, "2")

    local holed = {ONE[1], nil, ONE[1]} as {chat.Message}
    local _, gap = chat.render(promised, holed)
    test.includes(gap as string, "messages")
    test.includes(gap as string, "2")
  end
)
