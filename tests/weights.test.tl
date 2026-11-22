local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local gguf = require("tllm.gguf")
local weights = require("tllm.weights")
local u32, u64, text, header =
  support.u32, support.u64, support.text, support.header
local pair, padded = support.pair, support.padded

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/w" .. tostring(created) .. ".gguf"
  fs.write(file_path, bytes)
  return file_path
end

local function tensor(
  name: string, dims: {integer}, type_code: integer, offset: integer
): string
  local parts: {string} = {}
  for _, dim in ipairs(dims) do
    parts[#parts + 1] = u64(dim)
  end
  return text(name)
    .. u32(#dims)
    .. table.concat(parts)
    .. u32(type_code)
    .. u64(offset)
end

local function region(size: integer, fill: integer): string
  return string.rep(string.char(fill), size)
end

local function load_file(file_path: string): weights.Handle
  local handle, failure = weights.load(file_path)
  test.equal(failure, nil, file_path)
  return handle as weights.Handle
end

local function refusal(file_path: string): string
  local handle, failure = weights.load(file_path)
  test.equal(handle, nil, file_path)
  test.truthy(failure, file_path)
  local line = failure as string
  test.equal(line:find("\n"), nil, line)
  return line
end

local function refuse(bytes: string): string
  return refusal(fixture(bytes))
end

local function fd_count(): integer
  local count = 0
  local step = fs.readdir("/proc/self/fd")
  if step is nil then
    return count
  end
  while step() ~= nil do
    count = count + 1
  end
  return count
end

local function mapped(file_path: string): boolean
  local contents = fs.read("/proc/self/maps")
  if contents is nil then
    return false
  end
  return contents:find(file_path, 1, true) ~= nil
end

local function named(name: string, bytes: string): string
  local file_path = ROOT .. "/" .. name
  fs.write(file_path, bytes)
  return file_path
end

-- a shard's header: its split index and count, and its own alignment when it
-- states one
local function shard_header(
  tensor_count: integer,
  split_no: integer,
  count: integer,
  alignment: integer | nil
): string
  local metadata = pair("split.no", 4, u32(split_no))
    .. pair("split.count", 4, u32(count))
  local metadata_count = 2
  if not (alignment is nil) then
    metadata = metadata .. pair("general.alignment", 4, u32(alignment))
    metadata_count = 3
  end
  return header(tensor_count, metadata_count) .. metadata
end

local function mapped_regions(prefix: string): integer
  local contents = fs.read("/proc/self/maps")
  if contents is nil then
    return 0
  end
  local count = 0
  for line in contents:gmatch("[^\n]+") do
    if line:find(prefix, 1, true) then
      count = count + 1
    end
  end
  return count
end

local function read_only(file_path: string): boolean
  local contents = fs.read("/proc/self/maps")
  if contents is nil then
    return false
  end
  for line in contents:gmatch("[^\n]+") do
    if line:find(file_path, 1, true) then
      local perms = line:match("^%S+%s+(%S+)")
      return perms is string and perms:sub(1, 2) == "r-"
    end
  end
  return false
end

-- a refusal that leaves the process with the descriptors and mapped regions
-- it held before the call
local function refusal_clean(path: string, extra?: {string}): string
  local descriptors = fd_count()
  local regions = mapped_regions(ROOT)
  local handle, failure = weights.load(path, nil, extra)
  test.equal(handle, nil, path)
  test.truthy(failure, path)
  local line = failure as string
  test.equal(line:find("\n"), nil, line)
  test.equal(fd_count(), descriptors, path .. ": no descriptor left behind")
  test.equal(
    mapped_regions(ROOT), regions, path .. ": no mapped region left behind"
  )
  return line
end

local record Split
  first: string
  second: string
end

local function healthy_split(prefix: string): Split
  local first = named(
    prefix .. "-00001-of-00002.gguf",
    padded(shard_header(1, 0, 2, nil) .. tensor("x.weight", {2, 2}, 0, 0), 32)
      .. region(16, 17)
  )
  local second = named(
    prefix .. "-00002-of-00002.gguf",
    padded(shard_header(1, 1, 2, nil) .. tensor("y.weight", {2, 2}, 0, 0), 32)
      .. region(16, 34)
  )
  return {first = first, second = second}
end

test.it(
  "tensors come back by name with their bytes",
  function()
    local table_bytes = header(3, 0)
      .. tensor("a.weight", {2, 3}, 0, 0)
      .. tensor("b.weight", {64, 2}, 2, 32)
      .. tensor("c.bias", {}, 0, 128)
    local data = region(24, 17)
      .. region(8, 0)
      .. region(72, 34)
      .. region(24, 0)
      .. region(4, 51)
    local file_path = fixture(padded(table_bytes, 32) .. data)

    local handle = load_file(file_path)
    test.equal(handle.path, file_path)
    test.equal(#handle.shards, 1)
    local only = handle.shards[1]
    test.equal(only.path, file_path)

    local names: {string} = {}
    for name in pairs(handle.tensors) do
      names[#names + 1] = name
    end
    table.sort(names)
    test.deepEqual(names, {"a.weight", "b.weight", "c.bias"})

    local first = handle.tensors["a.weight"]
    test.equal(first.shard, only)
    test.deepEqual(first.dims, {2, 3})
    test.equal(first.kind.name, "f32")
    test.equal(first.elements, 6)
    test.equal(first.bytes, 24)
    test.equal(ffi.string(first.data, first.bytes), region(24, 17))

    local second = handle.tensors["b.weight"]
    test.deepEqual(second.dims, {64, 2})
    test.equal(second.kind.name, "q4_0")
    test.equal(second.elements, 128)
    test.equal(second.bytes, 72)
    test.equal(ffi.string(second.data, second.bytes), region(72, 34))

    local third = handle.tensors["c.bias"]
    test.deepEqual(third.dims, {})
    test.equal(third.kind.name, "f32")
    test.equal(third.elements, 1)
    test.equal(third.bytes, 4)
    test.equal(ffi.string(third.data, third.bytes), region(4, 51))

    handle.release(handle)
  end
)

test.it(
  "alignment and the data start come from the file",
  function()
    local aligned_table = header(1, 1)
      .. pair("general.alignment", 4, u32(64))
      .. tensor("a", {2, 3}, 0, 0)
    local aligned_path = fixture(aligned_table .. string.rep("\0", 88))
    local aligned_info = gguf.read(aligned_path) as gguf.Info
    local aligned = load_file(aligned_path)
    local aligned_shard = aligned.shards[1]
    test.equal(aligned_shard.alignment, 64)
    test.greaterEqual(aligned_shard.data_start, aligned_info.table_end)
    test.equal(aligned_shard.data_start % 64, 0)
    test.less(aligned_shard.data_start, aligned_info.table_end + 64)
    aligned.release(aligned)

    local plain_table = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local plain_path = fixture(plain_table .. string.rep("\0", 56))
    local plain_info = gguf.read(plain_path) as gguf.Info
    local plain = load_file(plain_path)
    local plain_shard = plain.shards[1]
    test.equal(plain_shard.alignment, 32)
    test.greaterEqual(plain_shard.data_start, plain_info.table_end)
    test.equal(plain_shard.data_start % 32, 0)
    test.less(plain_shard.data_start, plain_info.table_end + 32)
    plain.release(plain)
  end
)

test.it(
  "release unmaps the file",
  function()
    local table_bytes = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local file_path = fixture(table_bytes .. string.rep("\0", 56))

    local handle = load_file(file_path)
    test.truthy(mapped(file_path), file_path)

    handle.release(handle)
    test.equal(mapped(file_path), false)
    handle.release(handle)
    test.equal(mapped(file_path), false)
  end
)

local STACK = header(1, 0) .. tensor("stack.weight", {2, 2, 2}, 0, 0)

local record Closing
  close: function(integer): integer
end

test.it(
  "a whole-matrix read returns the mapping's bytes",
  function()
    local file_path = fixture(
      padded(STACK, 32) .. region(16, 17) .. region(16, 34)
    )
    local handle = load_file(file_path)
    local stack = handle.tensors["stack.weight"] as weights.Tensor
    local each = math.floor(stack.bytes / 2)

    local moved, failure = weights.read(handle, stack, 1)
    test.equal(moved, true, failure)
    test.equal(handle.buffer_bytes, each)
    test.equal(ffi.string(handle.buffer as ffi.CData, each), region(each, 34))
    test.notEqual(
      ffi.string(handle.buffer as ffi.CData, each), ffi.string(stack.data, each)
    )
    handle.release(handle)
  end
)

test.it(
  "a read that cannot be made is refused with its tensor",
  function()
    local file_path = fixture(padded(STACK, 32) .. region(32, 17))

    local gone = load_file(file_path)
    local stack = gone.tensors["stack.weight"] as weights.Tensor
    gone.release(gone)
    local moved, failure = weights.read(gone, stack, 0)
    test.equal(moved, nil, failure)
    test.includes(failure as string, "stack.weight")

    -- a descriptor the whole read would go through, closed under the handle
    local closed = load_file(file_path)
    local kept = closed.tensors["stack.weight"] as weights.Tensor
    (ffi.C as Closing).close(kept.shard.descriptor)
    local read, at_fault = weights.read(closed, kept, 0)
    test.equal(read, nil, at_fault)
    test.includes(at_fault as string, "stack.weight")
  end
)

test.it(
  "release closes the read descriptor and frees its buffer",
  function()
    local before = fd_count()
    local file_path = fixture(padded(STACK, 32) .. region(32, 17))
    local handle = load_file(file_path)
    test.equal(fd_count(), before + 1, "one descriptor for the whole read")

    local stack = handle.tensors["stack.weight"] as weights.Tensor
    test.equal(weights.read(handle, stack, 0), true)
    test.truthy(handle.buffer ~= nil, "the read's buffer")

    handle.release(handle)
    test.equal(fd_count(), before, "no descriptor left behind")
    test.equal(handle.buffer, nil)
    test.equal(handle.buffer_bytes, 0)
  end
)

test.it(
  "release waits for its holders",
  function()
    local table_bytes = header(1, 0) .. tensor("a", {2, 3}, 0, 0)
    local file_path = fixture(table_bytes .. string.rep("\0", 56))

    local handle = load_file(file_path)
    weights.attach(handle)
    weights.attach(handle)
    test.includes(handle.release(handle) as string, "2")
    test.truthy(mapped(file_path), file_path)

    weights.detach(handle)
    weights.detach(handle)
    test.equal(handle.release(handle), nil)
    test.equal(mapped(file_path), false)
  end
)

test.it(
  "a file the reader refuses is refused with its line",
  function()
    local missing = ROOT .. "/missing.gguf"
    test.includes(refusal(missing), missing)

    local not_gguf = fixture("GGML" .. u32(3) .. u64(0) .. u64(0))
    test.includes(refusal(not_gguf), not_gguf)
  end
)

test.it(
  "an alignment that is not a positive integer is refused",
  function()
    local tensor_info = tensor("a", {2, 3}, 0, 0)
    local cases: {string} = {
      pair("general.alignment", 8, text("64")),
      pair("general.alignment", 4, u32(0)),
      pair("general.alignment", 12, string.char(0, 0, 0, 0, 0, 32, 80, 64)),
    }
    for _, alignment in ipairs(cases) do
      local line = refuse(header(1, 1) .. alignment .. tensor_info)
      test.includes(line, "general.alignment")
    end

    local line = refuse(
      header(1, 1)
        .. pair("general.alignment", 4, u32(0))
        .. tensor("broken", {1}, 4, 0)
    )
    test.includes(line, "general.alignment")
    test.equal(line:find("broken", 1, true), nil)
  end
)

test.it(
  "a tensor that cannot be read is refused",
  function()
    local unknown_line = refuse(
      header(1, 0) .. tensor("blk.unknown", {1}, 4, 0)
    )
    test.includes(unknown_line, "blk.unknown")
    test.includes(unknown_line, "type code 4")

    local block_line = refuse(header(1, 0) .. tensor("blk.block", {48}, 2, 0))
    test.includes(block_line, "blk.block")

    local short_table = header(1, 0) .. tensor("blk.short", {8}, 0, 0)
    local short_path = fixture(padded(short_table, 32) .. string.rep("\0", 31))
    test.includes(refusal(short_path), "blk.short")

    local duplicate_table = header(2, 0)
      .. tensor("blk.same", {8}, 0, 0)
      .. tensor("blk.same", {8}, 0, 32)
    local duplicate_line = refuse(
      padded(duplicate_table, 32) .. string.rep("\0", 80)
    )
    test.includes(duplicate_line, "blk.same")

    local order_line = refuse(
      header(2, 0)
        .. tensor("blk.first", {1}, 4, 0)
        .. tensor("blk.second", {48}, 2, 0)
    )
    test.includes(order_line, "blk.first")
    test.equal(order_line:find("blk.second", 1, true), nil)
  end
)

test.it(
  "a split and an extra file answer what one file answers",
  function()
    local first_path = named(
      "split-00001-of-00002.gguf",
      padded(
        shard_header(2, 0, 2, nil)
          .. tensor("a.weight", {2, 3}, 0, 0)
          .. tensor("b.weight", {64, 2}, 2, 24),
        32
      ) .. region(24, 17)
        .. region(72, 34)
    )
    local second_path = named(
      "split-00002-of-00002.gguf",
      padded(shard_header(1, 1, 2, 64) .. tensor("c.weight", {4, 5}, 0, 0), 64)
        .. region(80, 51)
    )
    local extra_path = named(
      "extra.gguf",
      padded(header(1, 0) .. tensor("d.weight", {8}, 0, 0), 32)
        .. region(32, 68)
    )
    local one_path = named(
      "one.gguf",
      padded(
        header(4, 0)
          .. tensor("a.weight", {2, 3}, 0, 0)
          .. tensor("b.weight", {64, 2}, 2, 24)
          .. tensor("c.weight", {4, 5}, 0, 96)
          .. tensor("d.weight", {8}, 0, 176),
        32
      ) .. region(24, 17)
        .. region(72, 34)
        .. region(80, 51)
        .. region(32, 68)
    )

    local data_of: {string: string} = {
      ["a.weight"] = region(24, 17),
      ["b.weight"] = region(72, 34),
      ["c.weight"] = region(80, 51),
      ["d.weight"] = region(32, 68),
    }
    local file_of: {string: string} = {
      ["a.weight"] = first_path,
      ["b.weight"] = first_path,
      ["c.weight"] = second_path,
      ["d.weight"] = extra_path,
    }

    local one = load_file(one_path)
    test.equal(#one.shards, 1)

    local handle, failure = weights.load(first_path, nil, {extra_path})
    test.equal(failure, nil, failure)
    local merged = handle as weights.Handle

    test.equal(#merged.shards, 3)
    local by_path: {string: weights.Shard} = {}
    for _, shard in ipairs(merged.shards) do
      by_path[shard.path] = shard
      test.equal(shard.bytes, #(fs.read(shard.path) as string), shard.path)
      local info = gguf.read(shard.path) as gguf.Info
      test.greaterEqual(shard.data_start, info.table_end)
      test.equal(shard.data_start % shard.alignment, 0, shard.path)
      local wanted = 32
      if shard.path == second_path then
        wanted = 64
      end
      test.equal(shard.alignment, wanted, shard.path)
    end
    test.truthy(by_path[first_path] ~= nil, first_path)
    test.truthy(by_path[second_path] ~= nil, second_path)
    test.truthy(by_path[extra_path] ~= nil, extra_path)

    local names: {string} = {}
    for name in pairs(merged.tensors) do
      names[#names + 1] = name
    end
    table.sort(names)
    test.deepEqual(names, {"a.weight", "b.weight", "c.weight", "d.weight"})

    for _, name in ipairs(names) do
      local entry = merged.tensors[name]
      local same = one.tensors[name]
      test.deepEqual(entry.dims, same.dims)
      test.equal(entry.kind.name, same.kind.name)
      test.equal(entry.elements, same.elements)
      test.equal(entry.bytes, same.bytes)
      test.equal(entry.shard.path, file_of[name])
      test.equal(ffi.string(entry.data, entry.bytes), data_of[name])
    end

    local again, again_failure = weights.load(first_path, nil, {first_path})
    test.equal(again_failure, nil, again_failure)
    local held = again as weights.Handle
    test.equal(#held.shards, 2)
    local held_names: {string} = {}
    for name in pairs(held.tensors) do
      held_names[#held_names + 1] = name
    end
    table.sort(held_names)
    test.deepEqual(held_names, {"a.weight", "b.weight", "c.weight"})

    merged.release(merged)
    held.release(held)
    one.release(one)
  end
)

test.it(
  "a broken split names its file",
  function()
    local lonely = named(
      "lonely-00001-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local missing_path = ROOT .. "/lonely-00002-of-00002.gguf"
    local missing_line = refusal_clean(lonely)
    test.includes(missing_line, missing_path)
    test.includes(missing_line, "cannot be read")

    local wrong_count = named(
      "count-00001-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local count_three = named(
      "count-00002-of-00002.gguf", shard_header(0, 1, 3, nil)
    )
    local count_line = refusal_clean(wrong_count)
    test.includes(count_line, count_three)
    test.includes(count_line, "3")

    local text_first = named(
      "text-00001-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local text_second = named(
      "text-00002-of-00002.gguf",
      header(0, 2)
        .. pair("split.no", 4, u32(1))
        .. pair("split.count", 8, text("2"))
    )
    local text_line = refusal_clean(text_first)
    test.includes(text_line, text_second)
    test.includes(text_line, "split.count")

    local magic_first = named(
      "magic-00001-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local magic_second = named(
      "magic-00002-of-00002.gguf", "GGML" .. u32(3) .. u64(0) .. u64(0)
    )
    local magic_line = refusal_clean(magic_first)
    test.includes(magic_line, magic_second)
    test.includes(magic_line, "not GGUF")

    local no_first = named("no-00001-of-00002.gguf", shard_header(0, 0, 2, nil))
    local no_second = named(
      "no-00002-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local no_line = refusal_clean(no_first)
    test.includes(no_line, no_second)
    test.includes(no_line, "split.no 0")

    local second_named = named(
      "second-00002-of-00002.gguf", shard_header(0, 1, 2, nil)
    )
    local second_line = refusal_clean(second_named)
    test.includes(second_line, second_named)
    test.includes(second_line, "-00001-of-00002.gguf")

    local cut_first = named(
      "cut-00001-of-00002.gguf", shard_header(0, 0, 2, nil)
    )
    local cut_second = named(
      "cut-00002-of-00002.gguf",
      padded(
        shard_header(1, 1, 2, nil) .. tensor("cut.weight", {2, 2}, 0, 0), 32
      ) .. region(15, 17)
    )
    local cut_line = refusal_clean(cut_first)
    test.includes(cut_line, cut_second)
    test.includes(cut_line, "cut.weight")

    local order_first = named(
      "order-00001-of-00002.gguf",
      padded(
        shard_header(1, 0, 2, nil) .. tensor("order.weight", {2, 2}, 0, 0), 32
      ) .. region(15, 17)
    )
    local order_line = refusal_clean(order_first)
    test.includes(order_line, order_first)
    test.includes(order_line, "order.weight")
    test.equal(
      order_line:find(ROOT .. "/order-00002-of-00002.gguf", 1, true), nil
    )

    local healthy = healthy_split("extra")
    local absent = ROOT .. "/extra-missing.gguf"
    test.includes(refusal_clean(healthy.first, {absent}), absent)
  end
)

test.it(
  "a matrix in the second shard is read whole",
  function()
    local first = named("stack-00001-of-00002.gguf", shard_header(0, 0, 2, nil))
    named(
      "stack-00002-of-00002.gguf",
      padded(
        shard_header(1, 1, 2, 64) .. tensor("stack.weight", {2, 2, 2}, 0, 0), 64
      ) .. region(16, 17)
        .. region(16, 34)
    )
    local handle = load_file(first)
    local stack = handle.tensors["stack.weight"] as weights.Tensor
    test.equal(stack.shard.path, ROOT .. "/stack-00002-of-00002.gguf")

    local each = math.floor(stack.bytes / 2)
    test.equal(weights.read(handle, stack, 1), true)
    test.equal(handle.buffer_bytes, each)
    test.equal(ffi.string(handle.buffer as ffi.CData, each), region(each, 34))
    test.notEqual(
      ffi.string(handle.buffer as ffi.CData, each), ffi.string(stack.data, each)
    )
    handle.release(handle)
  end
)

test.it(
  "release gives back every shard",
  function()
    local split = healthy_split("release")
    local handle = load_file(split.first)
    test.truthy(read_only(split.first), split.first)
    test.truthy(read_only(split.second), split.second)

    test.equal(handle.release(handle), nil)
    test.equal(mapped(split.first), false)
    test.equal(mapped(split.second), false)
    test.equal(handle.release(handle), nil)
    test.equal(mapped(split.first), false)
    test.equal(mapped(split.second), false)
  end
)
