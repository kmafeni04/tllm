local ffi = require("ffi")
local fs = require("fs")
local test = require("lde-test")
local support = require("tests.support")
local reads = require("tllm.reads")
local weights = require("tllm.weights")
local u32, u64, text, header =
  support.u32, support.u64, support.text, support.header
local padded = support.padded

-- the limits, the truncation and the reads go through the same C library the
-- primitive does, so the case can take a descriptor or an address range away
ffi.cdef(
  [[
struct tllm_rlimit {
  unsigned long cur;
  unsigned long max;
};
int getrlimit(int resource, struct tllm_rlimit *rlim);
int setrlimit(int resource, const struct tllm_rlimit *rlim);
int truncate(const char *path, long length);
]]
)

local record Raw
  open: function(string, integer): integer
  close: function(integer): integer
  getrlimit: function(integer, ffi.CData): integer
  setrlimit: function(integer, ffi.CData): integer
  truncate: function(string, integer): integer
end

local record Limit
  cur: integer
  max: integer
end

local C = ffi.C as Raw

local NOFILE = 7
local AS = 9
local PAGE_BYTES = 4096
local MIB = 1048576
local STACKED = "blk.0.ffn_up_exps.weight"
local PLAIN = "blk.0.attn_q.weight"

local ROOT = os.tmpname()
os.remove(ROOT)
fs.mkdirAll(ROOT)
local created = 0

test.afterAll(function() fs.rmdir(ROOT) end)

local function fixture(bytes: string): string
  created = created + 1
  local file_path = ROOT .. "/r" .. tostring(created) .. ".gguf"
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

-- the file a stack of `matrices` matrices of `each` bytes holds, its own
-- matrix filled with its own byte
local function stack_file(matrices: integer, each: integer): string
  local parts: {string} = {
    padded(
      header(1, 0)
        .. tensor(STACKED, {math.floor(each / 4), 1, matrices}, 0, 0),
      32
    ),
  }
  for matrix = 1, matrices do
    parts[#parts + 1] = region(each, 16 + matrix)
  end
  return table.concat(parts)
end

local function load_file(file_path: string): weights.Handle
  local handle, failure = weights.load(file_path)
  test.equal(failure, nil, file_path)
  return handle as weights.Handle
end

local function stack_of(handle: weights.Handle): weights.Tensor
  return handle.tensors[STACKED] as weights.Tensor
end

local function opened(depth: integer, bytes: integer): reads.Queue
  local queue, failure = reads.open(depth, bytes)
  test.equal(failure, nil, failure)
  return queue as reads.Queue
end

local function needs_interface(name: string, body: function())
  if reads.offered() then
    test.it(name, body)
  else
    test.skip(name .. " (no submission interface)", function() end)
  end
end

local function descriptors(): integer
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

-- the mappings a queue owns: its rings, which the kernel names or maps shared,
-- and its one block of buffers, an unnamed mapping at least as large as the
-- block; a JIT code area, a collector arena or a grown heap is none of those
local function regions(block: number): integer
  local contents = fs.read("/proc/self/maps") as string
  local count = 0
  for line in contents:gmatch("[^\n]+") do
    local first, last = line:match("^(%x+)%-(%x+) %S+ %x+ %x+:%x+ %d+%s*$")
    if line:find("io_uring", 1, true) or line:find(" rw-s ", 1, true) then
      count = count + 1
    elseif first is string and last is string then
      local size = (tonumber(last, 16) as number) - (
        tonumber(first, 16) as number
      )
      if size >= block then
        count = count + 1
      end
    end
  end
  return count
end

local function address_space(): integer
  local contents = fs.read("/proc/self/statm") as string
  local digits = contents:match("^(%d+)") as string
  return (tonumber(digits) as integer) * PAGE_BYTES
end

local function limits_of(resource: integer): Limit
  local cell = ffi.new("struct tllm_rlimit[1]")
  C.getrlimit(resource, cell)
  local view = ffi.cast(
    "struct tllm_rlimit *", ffi.cast("intptr_t", cell) as integer
  ) as Limit
  return {cur = view.cur, max = view.max}
end

local function set_limits(resource: integer, limits: Limit)
  local cell = ffi.new("struct tllm_rlimit[1]")
  local view = ffi.cast(
    "struct tllm_rlimit *", ffi.cast("intptr_t", cell) as integer
  ) as Limit
  view.cur = limits.cur
  view.max = limits.max
  C.setrlimit(resource, cell)
end

needs_interface(
  "a handed-over read answers the bytes it landed",
  function()
    local each = 256
    local file_path = fixture(stack_file(2, each))
    local handle = load_file(file_path)
    local stack = stack_of(handle)
    local queue = opened(2, each)

    test.equal(reads.submit(queue, handle, stack, 0), true)
    test.equal(reads.submit(queue, handle, stack, 1), true)

    local rows, failure = reads.wait(queue)
    test.equal(failure, nil, failure)
    local landed = rows as {reads.Landed}
    test.equal(#landed, 2)

    test.equal(landed[1].tensor, STACKED)
    test.equal(landed[1].index, 0)
    test.equal(landed[1].bytes, each)
    test.equal(landed[1].fault, nil)

    test.equal(landed[2].tensor, STACKED)
    test.equal(landed[2].index, 1)
    test.equal(landed[2].bytes, each)
    test.equal(landed[2].fault, nil)
    test.equal(queue.pending, 0)

    local again = reads.wait(queue) as {reads.Landed}
    test.equal(#again, 0)

    reads.close(queue)
    handle.release(handle)
  end
)

needs_interface(
  "a full queue reaps before it takes another",
  function()
    local each = 512
    local file_path = fixture(stack_file(5, each))
    local handle = load_file(file_path)
    local stack = stack_of(handle)
    local queue = opened(4, each)

    for index = 0, 4 do
      test.equal(reads.submit(queue, handle, stack, index), true)
      test.lessEqual(queue.pending, queue.depth)
      if index == 3 then
        test.equal(queue.pending, 4)
      end
    end

    local rows = reads.wait(queue) as {reads.Landed}
    test.equal(#rows, 5)
    for index = 0, 4 do
      test.equal(rows[index + 1].tensor, STACKED)
      test.equal(rows[index + 1].index, index)
      test.equal(rows[index + 1].bytes, each)
    end
    test.equal(queue.pending, 0)

    reads.close(queue)
    handle.release(handle)
  end
)

needs_interface(
  "a short handed-over read names its tensor and its index",
  function()
    local each = 300
    local bytes = stack_file(2, each)
    local file_path = fixture(bytes)
    local handle = load_file(file_path)
    local stack = stack_of(handle)
    C.truncate(file_path, #bytes - 20)

    local queue = opened(1, each)
    test.equal(reads.submit(queue, handle, stack, 1), true)

    local rows = reads.wait(queue) as {reads.Landed}
    test.equal(#rows, 1)
    local row = rows[1]
    test.equal(row.tensor, STACKED)
    test.equal(row.index, 1)
    test.equal(row.bytes, each - 20)

    local fault = row.fault as string
    test.equal(fault:find("\n"), nil)
    test.includes(fault, STACKED)
    test.includes(fault, "matrix 1")
    test.includes(fault, tostring(each - 20))

    reads.close(queue)
    handle.release(handle)
  end
)

needs_interface(
  "closing a queue leaves nothing behind",
  function()
    local each = MIB
    local file_path = fixture(stack_file(4, each))
    local handle = load_file(file_path)
    local stack = stack_of(handle)

    -- the allocations this case counts are the ones a second run makes
    local warm = opened(4, each)
    test.equal(reads.submit(warm, handle, stack, 0), true)
    reads.close(warm)

    local openings = descriptors()
    local mappings = regions(4 * MIB)

    local queue = opened(4, each)
    test.equal(reads.submit(queue, handle, stack, 0), true)
    test.equal(reads.submit(queue, handle, stack, 1), true)
    reads.close(queue)
    test.equal(descriptors(), openings, "no descriptor left behind")
    test.equal(regions(4 * MIB), mappings, "no mapped region left behind")

    reads.close(queue)
    test.equal(descriptors(), openings, "a second close gives nothing back")
    test.equal(regions(4 * MIB), mappings, "a second close gives nothing back")

    local moved, closed = reads.submit(queue, handle, stack, 0)
    test.equal(moved, nil)
    test.includes(closed as string, "closed")
    local rows, waited = reads.wait(queue)
    test.equal(rows, nil)
    test.includes(waited as string, "closed")

    local in_flight = opened(2, each)
    test.equal(reads.submit(in_flight, handle, stack, 2), true)
    reads.close(in_flight)
    test.equal(
      descriptors(), openings, "an in-flight read leaves no descriptor"
    )
    test.equal(regions(4 * MIB), mappings, "an in-flight read leaves no region")

    local broken = opened(1, each)
    C.close(broken.descriptor)
    local none, refused = reads.wait(broken)
    test.equal(none, nil)
    test.includes(refused as string, "reap was refused")
    test.equal(broken.shut, true)
    local refused_submit, told = reads.submit(broken, handle, stack, 0)
    test.equal(refused_submit, nil)
    test.includes(told as string, "closed")
    reads.close(broken)
    test.equal(descriptors(), openings, "a shut queue leaves no descriptor")
    test.equal(regions(4 * MIB), mappings, "a shut queue leaves no region")

    handle.release(handle)
  end
)

needs_interface(
  "a queue or a matrix it cannot take is refused",
  function()
    local none, no_depth = reads.open(0, 4096)
    test.equal(none, nil)
    test.includes(no_depth as string, "depth 0")
    local empty, no_bytes = reads.open(2, 0)
    test.equal(empty, nil)
    test.includes(no_bytes as string, "0 bytes")

    local table_bytes = header(3, 0)
      .. tensor(PLAIN, {64, 2}, 0, 0)
      .. tensor(STACKED, {2, 1, 2}, 0, 512)
      .. tensor("blk.0.small.weight", {2, 1, 1}, 0, 520)
    local file_path = fixture(
      padded(table_bytes, 32)
        .. region(512, 17)
        .. region(16, 34)
        .. region(8, 51)
    )
    local handle = load_file(file_path)
    local queue = opened(2, 1024)

    local plain, plain_line = reads.submit(
      queue, handle, handle.tensors[PLAIN], 0
    )
    test.equal(plain, nil)
    test.includes(plain_line as string, PLAIN)
    test.includes(plain_line as string, "matrix 0")

    local past, past_line = reads.submit(queue, handle, stack_of(handle), 2)
    test.equal(past, nil)
    test.includes(past_line as string, STACKED)
    test.includes(past_line as string, "matrix 2")

    local narrow = opened(1, 4)
    local small, size_line = reads.submit(
      narrow, handle, handle.tensors["blk.0.small.weight"], 0
    )
    test.equal(small, nil)
    test.includes(size_line as string, "holds 8 bytes")
    test.includes(size_line as string, "a buffer holds 4")
    test.includes(size_line as string, "matrix 0")
    reads.close(narrow)

    local released = load_file(file_path)
    local released_stack = released.tensors[STACKED] as weights.Tensor
    C.close(released_stack.shard.descriptor)
    local gone, gone_line = reads.submit(queue, released, released_stack, 0)
    test.equal(gone, nil)
    test.includes(gone_line as string, "released")
    test.includes(gone_line as string, STACKED)

    reads.close(queue)
    handle.release(handle)
    released.release(released)
  end
)

needs_interface(
  "an interface that cannot be set up takes nothing",
  function()
    local before = descriptors()
    local saved = limits_of(NOFILE)
    set_limits(NOFILE, {cur = before, max = saved.max})

    local spent: {integer} = {}
    for _ = 1, 4096 do
      local taken = C.open("/dev/null", 0)
      if taken < 0 then
        break
      end
      spent[#spent + 1] = taken
    end

    local queue, failure = reads.open(1, 4096)
    test.equal(queue, nil)
    test.includes(failure as string, "cannot be set up")

    for _, taken in ipairs(spent) do
      C.close(taken)
    end
    set_limits(NOFILE, saved)
    test.equal(descriptors(), before)

    local after = opened(1, 4096)
    reads.close(after)
  end
)

needs_interface(
  "a queue it cannot buffer gives back what it took",
  function()
    local warm = opened(4, 4096)
    reads.close(warm)

    local saved = limits_of(AS)
    local function unbuffered(): string
      set_limits(AS, {cur = address_space() + 16 * MIB, max = saved.max})
      local queue, failure = reads.open(4, 1024 * 1024 * MIB)
      set_limits(AS, saved)
      test.equal(queue, nil)
      return failure as string
    end

    -- the allocations this case counts are the ones a later run makes
    unbuffered()
    unbuffered()
    local openings = descriptors()
    local mappings = regions(math.huge)
    local line = unbuffered()

    test.includes(line, "buffer")
    test.includes(line, tostring(1024 * 1024 * MIB))
    test.equal(descriptors(), openings, "no descriptor left behind")
    test.equal(regions(math.huge), mappings, "no mapped region left behind")
  end
)

needs_interface(
  "a handover the platform refuses shuts the queue",
  function()
    local each = 256
    local file_path = fixture(stack_file(2, each))
    local handle = load_file(file_path)
    local stack = stack_of(handle)

    local queue = opened(1, each)
    C.close(queue.descriptor)
    local moved, posted = reads.submit(queue, handle, stack, 0)
    test.equal(moved, nil)
    local line = posted as string
    test.includes(line, STACKED)
    test.includes(line, "matrix 0")
    test.includes(line, "refused the post")
    test.equal(queue.shut, true)
    reads.close(queue)

    local second = opened(1, each)
    test.equal(reads.submit(second, handle, stack, 0), true)
    C.close(second.descriptor)
    local refused, reason = reads.submit(second, handle, stack, 1)
    test.equal(refused, nil)
    test.includes(reason as string, "reap was refused")
    test.equal(second.pending, 1)
    test.equal(second.shut, true)
    reads.close(second)

    handle.release(handle)
  end
)

needs_interface(
  "a handed-over read that fails names its tensor and its index",
  function()
    local each = 256
    local file_path = fixture(stack_file(2, each))
    local handle = load_file(file_path)
    local stack = stack_of(handle)
    -- a descriptor the platform refuses to read while fcntl still answers
    C.close(stack.shard.descriptor)
    stack.shard.descriptor = C.open(file_path, 1)
    local queue = opened(1, each)

    test.equal(reads.submit(queue, handle, stack, 1), true)
    local rows = reads.wait(queue) as {reads.Landed}
    test.equal(#rows, 1)
    test.equal(rows[1].tensor, STACKED)
    test.equal(rows[1].index, 1)
    test.equal(rows[1].bytes, 0)
    local fault = rows[1].fault as string
    test.equal(fault:find("\n"), nil)
    test.includes(fault, STACKED)
    test.includes(fault, "matrix 1")
    test.includes(fault, "read 0")
    test.equal(queue.pending, 0)

    C.close(stack.shard.descriptor)
    stack.shard.descriptor = -1
    reads.close(queue)
    handle.release(handle)
  end
)
