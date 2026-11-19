local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local clock = require("tllm.clock")
local gguf = require("tllm.gguf")
local tokenizer = require("tllm.tokenizer")
local u32, text, array = support.u32, support.text, support.array
local header, pair = support.header, support.pair

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

    local with_end = loaded(
      header(0, 4)
        .. pair("tokenizer.ggml.model", 8, text("gpt2"))
        .. pair("tokenizer.ggml.tokens", 9, strings({"a"}))
        .. pair("tokenizer.ggml.token_type", 9, ints({1}))
        .. pair("tokenizer.ggml.eos_token_id", 8, text("1"))
    )
    test.equal(with_end.end_id, nil)
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

local function unit_kinds(count: integer): {integer}
  local kinds: {integer} = {}
  for index = 1, count do
    kinds[index] = 1
  end
  return kinds
end

local function encoding_file(
  pre: string | nil, pieces: {string}, extra: {string}
): string
  local parts: {string} = {
    pair("tokenizer.ggml.model", 8, text("gpt2")),
    pair("tokenizer.ggml.tokens", 9, strings(pieces)),
    pair("tokenizer.ggml.token_type", 9, ints(unit_kinds(#pieces))),
  }
  if pre is string then
    parts[#parts + 1] = pair("tokenizer.ggml.pre", 8, text(pre))
  end
  for _, entry in ipairs(extra) do
    parts[#parts + 1] = entry
  end
  return header(0, #parts) .. table.concat(parts)
end

local function merges_entry(merges: {string}): string
  return pair("tokenizer.ggml.merges", 9, strings(merges))
end

test.it(
  "text becomes the ids its merges give",
  function()
    local pieces = {"a", "b", "c", "ab", "bc", "Ġ"}
    local built = loaded(
      encoding_file("gpt2", pieces, {merges_entry({"a b", "b c"})})
    )

    local ids, failure = built:encode("abc")
    test.deepEqual(ids, {3, 2})
    test.equal(failure, nil)

    local reversed = loaded(
      encoding_file("gpt2", pieces, {merges_entry({"b c", "a b"})})
    )
    ids, failure = reversed:encode("abc")
    test.deepEqual(ids, {0, 4})
    test.equal(failure, nil)

    ids, failure = built:encode("a b")
    test.deepEqual(ids, {0, 5, 1})
    test.equal(failure, nil)

    local decoded, decode_failure = built:decode({0, 5, 1})
    test.equal(decoded, "a b")
    test.equal(decode_failure, nil)

    local fallback = loaded(
      encoding_file("gpt2", {"a", "b"}, {merges_entry({"a b"})})
    )
    ids, failure = fallback:encode("ab")
    test.deepEqual(ids, {0, 1})
    test.equal(failure, nil)

    ids, failure = built:encode("")
    test.deepEqual(ids, {})
    test.equal(failure, nil)
  end
)

test.it(
  "each scheme splits text into its own words",
  function()
    local pieces = {"A", "'", "S", "'S", "1", "2", "12", "a", "Ċ", "ĊĊ"}
    local merges = {merges_entry({"' S", "1 2", "Ċ Ċ"})}
    local gpt2 = loaded(encoding_file("gpt2", pieces, merges))
    local qwen = loaded(encoding_file("qwen35", pieces, merges))

    local ids, failure = gpt2:encode("A'S")
    test.deepEqual(ids, {0, 1, 2})
    test.equal(failure, nil)

    ids, failure = gpt2:encode("12")
    test.deepEqual(ids, {6})
    test.equal(failure, nil)

    ids, failure = gpt2:encode("a\n\na")
    test.deepEqual(ids, {7, 8, 8, 7})
    test.equal(failure, nil)

    ids, failure = qwen:encode("A'S")
    test.deepEqual(ids, {0, 3})
    test.equal(failure, nil)

    ids, failure = qwen:encode("12")
    test.deepEqual(ids, {4, 5})
    test.equal(failure, nil)

    ids, failure = qwen:encode("a\n\na")
    test.deepEqual(ids, {7, 9, 7})
    test.equal(failure, nil)
  end
)

test.it(
  "text that cannot be encoded is refused",
  function()
    local no_merges = loaded(encoding_file("gpt2", {"a"}, {}))
    local ids, failure = no_merges:encode("a")
    test.equal(ids, nil)
    test.includes(failure as string, "tokenizer.ggml.merges")

    local numeric_merges = loaded(
      encoding_file(
        "gpt2", {"a", "b"}, {pair("tokenizer.ggml.merges", 9, ints({5}))}
      )
    )
    ids, failure = numeric_merges:encode("a")
    test.equal(ids, nil)
    test.includes(failure as string, "tokenizer.ggml.merges")

    local no_space = loaded(
      encoding_file("gpt2", {"a", "b"}, {merges_entry({"\nab"})})
    )
    ids, failure = no_space:encode("a")
    test.equal(ids, nil)
    test.includes(failure as string, "tokenizer.ggml.merges")
    test.equal((failure as string):find("\n"), nil)

    local no_pre = loaded(encoding_file(nil, {"a"}, {merges_entry({})}))
    ids, failure = no_pre:encode("a")
    test.equal(ids, nil)
    test.includes(failure as string, "tokenizer.ggml.pre")

    local llama3 = loaded(encoding_file("llama3", {"a"}, {merges_entry({})}))
    ids, failure = llama3:encode("a")
    test.equal(ids, nil)
    test.includes(failure as string, "llama3")

    local missing_piece = loaded(
      encoding_file("gpt2", {"a"}, {merges_entry({})})
    )
    ids, failure = missing_piece:encode("ab")
    test.equal(ids, nil)
    test.includes(failure as string, "98")

    local valid = loaded(encoding_file("gpt2", {"a"}, {merges_entry({})}))
    ids, failure = valid:encode("\255")
    test.equal(ids, nil)
    test.includes(failure as string, "not valid UTF-8")
  end
)

test.it(
  "one long word costs its merges, not a scan per merge",
  function()
    local pairs_wanted = 4096
    local pieces = {"a", "b"}
    local merges = {"a b"}
    local size: integer = 2
    while size <= pairs_wanted do
      pieces[#pieces + 1] = string.rep("ab", size)
      merges[#merges + 1] = string.rep("ab", math.floor(size / 2))
        .. " "
        .. string.rep("ab", math.floor(size / 2))
      size = size * 2
    end
    local built = loaded(encoding_file("gpt2", pieces, {merges_entry(merges)}))

    local started = clock.now()
    local ids, failure = built:encode(string.rep("ab", pairs_wanted))
    local spent = clock.now() - started

    test.equal(failure, nil)
    test.deepEqual(ids, {#pieces - 1})
    test.less(spent, 0.4)
  end
)

test.it(
  "a combining mark stays with the letters beside it",
  function()
    local built = loaded(
      encoding_file(
        "qwen35", {"e", "Ì", "ģ", "eÌ", "5"}, {merges_entry({"e Ì"})}
      )
    )
    local ids, failure = built:encode("e\204\1295")
    test.deepEqual(ids, {3, 2, 4})
    test.equal(failure, nil)
  end
)
