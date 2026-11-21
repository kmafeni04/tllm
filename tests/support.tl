local ffi = require("ffi")
local test = require("lde-test")
local compute = require("tllm.compute")

local FLOAT_BYTES: integer = 4

local type Floats = {integer: number}

local record support
  FLOAT_BYTES: integer

  record Entry
    name: string
    dims: {integer}
    code: integer
    bytes: string
  end

  record Meta
    name: string
    code: integer
    bytes: string
  end

  u16: function(integer): string
  u32: function(integer): string
  u64: function(integer): string
  text: function(string): string
  array: function(integer, {string}): string
  header: function(integer, integer): string
  pair: function(string, integer, string): string
  padded: function(string, integer): string
  floats: function({number}): string
  f32: function(string, {integer}, {number}): Entry
  f64: function(string, {integer}, {number}): Entry
  q6_K: function(Entry): Entry
  zeros: function(integer): {number}
  ones: function(integer): {number}
  halves: function(integer): string
  mint: function(string, integer): Meta
  mfloat: function(string, number): Meta
  mtext: function(string, string): Meta
  mstrings: function(string, {string}): Meta
  mint_array: function(string, {integer}): Meta
  both: function({Entry}, {Entry}): {Entry}
  without: function({Entry}, string): {Entry}
  without_meta: function({Meta}, string): {Meta}
  replaced: function({Entry}, string, Entry): {Entry}
  first: function({{number}} | nil, string | nil): {number} | nil, string | nil
  first_row: function({number}, integer): {compute.Row}
  refused: function({number} | nil, string | nil): string
  found_device: function(): boolean
  on_device: function(string, function())
end

function support.u16(value: integer): string
  return string.char(value % 256, math.floor(value / 256) % 256)
end

function support.u32(value: integer): string
  return string.char(
    value % 256,
    math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256,
    math.floor(value / 16777216) % 256
  )
end

function support.u64(value: integer): string
  local parts: {string} = {}
  local rest = value
  for _ = 1, 8 do
    parts[#parts + 1] = string.char(rest % 256)
    rest = math.floor(rest / 256)
  end
  return table.concat(parts)
end

function support.text(bytes: string): string
  support.FLOAT_BYTES = FLOAT_BYTES

  return support.u64(#bytes) .. bytes
end

function support.array(item_code: integer, items: {string}): string
  support.FLOAT_BYTES = FLOAT_BYTES

  return support.u32(item_code) .. support.u64(#items) .. table.concat(items)
end

function support.header(tensor_count: integer, metadata_count: integer): string
  return "GGUF"
    .. support.u32(3)
    .. support.u64(tensor_count)
    .. support.u64(metadata_count)
end

function support.pair(name: string, code: integer, bytes: string): string
  support.FLOAT_BYTES = FLOAT_BYTES

  return support.text(name) .. support.u32(code) .. bytes
end

function support.padded(prefix: string, alignment: integer): string
  if #prefix % alignment == 0 then
    return prefix
  end
  return prefix .. string.rep("\0", alignment - #prefix % alignment)
end

function support.floats(values: {number}): string
  local parts: {string} = {}
  local cell = ffi.new("float[1]") as Floats
  for _, value in ipairs(values) do
    cell[0] = value
    parts[#parts + 1] = ffi.string(cell as ffi.CData, FLOAT_BYTES)
  end
  return table.concat(parts)
end

function support.f32(
  name: string, dims: {integer}, values: {number}
): support.Entry
  return {name = name, dims = dims, code = 0, bytes = support.floats(values)}
end

function support.f64(
  name: string, dims: {integer}, values: {number}
): support.Entry
  local parts: {string} = {}
  local cell = ffi.new("double[1]") as Floats
  for _, value in ipairs(values) do
    cell[0] = value
    parts[#parts + 1] = ffi.string(cell as ffi.CData, 8)
  end
  return {name = name, dims = dims, code = 28, bytes = table.concat(parts)}
end

local record Quantizer
  ggml_quantize_chunk: function(
    integer, ffi.CData, ffi.CData, integer, integer, integer, ffi.CData | nil
  ): integer
end

function support.q6_K(entry: support.Entry): support.Entry
  local lib = ffi.load("target/tllm/libggml.so") as Quantizer
  local width = entry.dims[1]
  test.equal(#entry.bytes % (width * FLOAT_BYTES), 0)
  test.equal(width % 256, 0)
  local rows = math.floor(#entry.bytes / (width * FLOAT_BYTES))
  local bytes = math.floor(width / 256) * 210 * rows
  local source = ffi.new("float[?]", width * rows) as ffi.CData
  local destination = ffi.new("uint8_t[?]", bytes) as ffi.CData
  ffi.copy(source, entry.bytes, #entry.bytes)
  local written = lib.ggml_quantize_chunk(
    14, source, destination, 0, rows, width, nil
  )
  test.equal(written, bytes)
  return {
    name = entry.name,
    dims = entry.dims,
    code = 14,
    bytes = ffi.string(destination, bytes),
  }
end

function support.zeros(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 0
  end
  return result
end

function support.ones(count: integer): {number}
  local result: {number} = {}
  for position = 1, count do
    result[position] = 1
  end
  return result
end

function support.halves(count: integer): string
  return string.rep(support.u16(0x3C00), count)
end

function support.mint(name: string, value: integer): support.Meta
  return {name = name, code = 4, bytes = support.u32(value)}
end

function support.mfloat(name: string, value: number): support.Meta
  return {name = name, code = 6, bytes = support.floats({value})}
end

function support.mtext(name: string, value: string): support.Meta
  return {name = name, code = 8, bytes = support.text(value)}
end

function support.mstrings(name: string, values: {string}): support.Meta
  local items: {string} = {}
  for _, value in ipairs(values) do
    items[#items + 1] = support.text(value)
  end
  return {name = name, code = 9, bytes = support.array(8, items)}
end

function support.mint_array(name: string, values: {integer}): support.Meta
  local items: {string} = {}
  for _, value in ipairs(values) do
    items[#items + 1] = support.u32(value)
  end
  return {name = name, code = 9, bytes = support.array(5, items)}
end

function support.both(
  first: {support.Entry}, second: {support.Entry}
): {support.Entry}
  local result: {support.Entry} = {}
  for _, entry in ipairs(first) do
    result[#result + 1] = entry
  end
  for _, entry in ipairs(second) do
    result[#result + 1] = entry
  end
  return result
end

function support.without(
  entries: {support.Entry}, name: string
): {support.Entry}
  local result: {support.Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name ~= name then
      result[#result + 1] = entry
    end
  end
  return result
end

function support.without_meta(
  meta: {support.Meta}, name: string
): {support.Meta}
  local result: {support.Meta} = {}
  for _, entry in ipairs(meta) do
    if entry.name ~= name then
      result[#result + 1] = entry
    end
  end
  return result
end

function support.replaced(
  entries: {support.Entry}, name: string, replacement: support.Entry
): {support.Entry}
  local result: {support.Entry} = {}
  for _, entry in ipairs(entries) do
    if entry.name == name then
      result[#result + 1] = replacement
    else
      result[#result + 1] = entry
    end
  end
  return result
end

function support.first(
  output: {{number}} | nil, line: string | nil
): {number} | nil, string | nil
  if output is nil then
    return nil, line
  end
  return (output as {{number}})[1], line
end

function support.first_row(hidden: {number}, position: integer): {compute.Row}
  return {{sequence = "", hidden = hidden, position = position}}
end

function support.refused(value: {number} | nil, line: string | nil): string
  test.equal(value, nil)
  test.truthy(line)
  local message = line as string
  test.equal(message:find("\n"), nil)
  return message
end

-- the gate every device case uses: a machine whose library found a device
function support.found_device(): boolean
  local backend = compute.load() as compute.Backend
  local bytes, total = backend:memory()
  compute.unload(backend)
  return not (bytes is nil) and not (total is nil)
end

function support.on_device(name: string, body: function())
  if support.found_device() then
    test.it(name, body)
  else
    test.skip(name .. " (no CUDA device)", function() end)
  end
end

support.FLOAT_BYTES = FLOAT_BYTES

return support
