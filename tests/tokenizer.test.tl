local fs = require("fs")
local test = require("lde-test")
local gguf = require("tllm.gguf")
local tokenizer = require("tllm.tokenizer")

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/t" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
end

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

local function vocabulary(pieces: {string}, kinds: {integer}): string
  return header(0, 3)
    .. pair("tokenizer.ggml.model", 8, text("gpt2"))
    .. pair("tokenizer.ggml.tokens", 9, strings(pieces))
    .. pair("tokenizer.ggml.token_type", 9, ints(kinds))
end

local function loaded(bytes: string): tokenizer.Tokenizer
  local file_path = fixture(bytes)
  local built, failure = tokenizer.load(file_path)
  test.equal(failure, nil, file_path)
  return built as tokenizer.Tokenizer
end

local function refused(bytes: string): string
  local file_path = fixture(bytes)
  local built, failure = tokenizer.load(file_path)
  test.equal(built, nil, file_path)
  test.truthy(failure, file_path)
  return failure as string
end

test.it(
  "a gpt2 vocabulary decodes ids to text",
  function()
    local built = loaded(
      vocabulary(
        {
          "Hello",
          "Ġworld",
          "ĠcafÃ©",
          "<|im_end|>",
          "<|probe|>",
          "Ġpad",
          "spare",
        },
        {1, 1, 1, 3, 2, 4, 5}
      )
    )
    test.equal(built.size, 7)

    local decoded, failure = built:decode({0, 1, 2})
    test.equal(decoded, "Hello world café")
    test.equal(failure, nil)

    decoded, failure = built:decode({3, 4, 6})
    test.equal(decoded, "")
    test.equal(failure, nil)

    decoded, failure = built:decode({5})
    test.equal(decoded, "Ġpad")
    test.equal(failure, nil)

    decoded, failure = built:decode({})
    test.equal(decoded, "")
    test.equal(failure, nil)

    local plain = loaded(
      header(0, 2)
        .. pair("tokenizer.ggml.model", 8, text("gpt2"))
        .. pair("tokenizer.ggml.tokens", 9, strings({"Ġend"}))
    )
    local plain_decoded, plain_failure = plain:decode({0})
    test.equal(plain_decoded, " end")
    test.equal(plain_failure, nil)
  end
)

test.it(
  "a vocabulary that cannot be used is refused",
  function()
    local broken = fixture("GGML")
    local _, reader_failure = gguf.read(broken)
    local info, failure = tokenizer.load(broken)
    test.equal(info, nil)
    test.equal(failure, reader_failure)

    local missing_model = refused(
      header(0, 1) .. pair("general.architecture", 8, text("llama"))
    )
    test.includes(missing_model, "tokenizer.ggml.model")

    local wrong_model = refused(
      header(0, 1) .. pair("tokenizer.ggml.model", 8, text("gemma4"))
    )
    test.includes(wrong_model, "gemma4")

    local missing_tokens = refused(
      header(0, 1) .. pair("tokenizer.ggml.model", 8, text("gpt2"))
    )
    test.includes(missing_tokens, "tokenizer.ggml.tokens")

    local wrong_tokens = refused(
      header(0, 2)
        .. pair("tokenizer.ggml.model", 8, text("gpt2"))
        .. pair("tokenizer.ggml.tokens", 9, ints({5}))
    )
    test.includes(wrong_tokens, "tokenizer.ggml.tokens")

    local wrong_types = refused(
      header(0, 3)
        .. pair("tokenizer.ggml.model", 8, text("gpt2"))
        .. pair("tokenizer.ggml.tokens", 9, strings({"a", "b"}))
        .. pair("tokenizer.ggml.token_type", 9, ints({1}))
    )
    test.includes(wrong_types, "tokenizer.ggml.token_type")
  end
)

test.it(
  "an id that cannot be decoded is refused",
  function()
    local built = loaded(
      vocabulary({"Hello", "π", "<0x0A>", "spare", "tail"}, {1, 1, 6, 5, 5})
    )

    local decoded, failure = built:decode({0, 9})
    test.equal(decoded, nil)
    test.includes(failure as string, "9")
    test.includes(failure as string, "5")

    decoded, failure = built:decode({-1})
    test.equal(decoded, nil)
    test.includes(failure as string, "-1")

    decoded, failure = built:decode({2})
    test.equal(decoded, nil)
    test.includes(failure as string, "2")
    test.includes(failure as string, "6")

    decoded, failure = built:decode({1})
    test.equal(decoded, nil)
    test.includes(failure as string, "1")

    local raw = loaded(vocabulary({"a", "b", "c", "\255"}, {1, 1, 1, 1}))
    local raw_decoded, raw_failure = raw:decode({3})
    test.equal(raw_decoded, nil)
    test.includes(raw_failure as string, "3")
  end
)
