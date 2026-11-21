local ffi = require("ffi")
local weights = require("tllm.weights")

-- the layouts the kernel reads and writes here, at the offsets EINVAL-free
-- io_uring ABI fixes: one of the 64-byte entries, one 16-byte completion and
-- the 120-byte parameters `io_uring_setup` fills
ffi.cdef(
  [[
long syscall(long number, ...);
int fcntl(int fd, int cmd, ...);

struct tllm_sqe {
  uint64_t head;
  uint64_t off;
  uint64_t addr;
  uint64_t size;
  uint64_t tag;
  uint64_t rest[3];
};

struct tllm_cqe {
  uint64_t tag;
  int32_t res;
  uint32_t res_flags;
};
]]
)

local record Held
  slot: integer
  tensor: string
  index: integer
  path: string
  wanted: integer
  landed: integer | nil
  fault: string | nil
end

local record reads
  record Queue
    depth: integer
    bytes: integer
    pending: integer
    descriptor: integer
    shut: boolean
    closed: boolean
    -- the mapped rings and entries, -1 once given back, and the one block of
    -- buffers the reads land in
    submission: integer
    submission_bytes: integer
    completion: integer
    completion_bytes: integer
    entries: integer
    entries_bytes: integer
    buffers: integer
    buffers_bytes: integer
    -- ring positions, read and written in the mapped memory
    sq_tail: integer
    sq_array: integer
    sq_mask: integer
    cq_head: integer
    cq_tail: integer
    cqes: integer
    cq_mask: integer
    -- entries no read is using, the reads handed over by their tag order
    free: {integer}
    held: {integer: Held}
    order: {integer}
    next: integer
  end

  record Landed
    tensor: string
    index: integer
    bytes: integer
    fault: string | nil
  end

  offered: function(): boolean
  open: function(depth: integer, bytes: integer): Queue | nil, string | nil
  submit: function(
    Queue, weights.Handle, weights.Tensor, index: integer
  ): boolean | nil, string | nil
  wait: function(Queue): {Landed} | nil, string | nil
  close: function(Queue)
end

local type Handle = weights.Handle
local type Tensor = weights.Tensor
local type Queue = reads.Queue

local record LibC
  syscall: function(
    ffi.CType, ffi.CType, ffi.CType | nil, ...: ffi.CType
  ): ffi.CData
  fcntl: function(integer, integer): integer
  mmap: function(
    ffi.CData | nil, integer, integer, integer, integer, integer
  ): ffi.CData
  munmap: function(ffi.CData, integer): integer
  close: function(integer)
end

local record Entry
  head: integer
  off: integer
  addr: integer
  size: integer
  tag: integer
end

local record Completion
  tag: integer
  res: integer
end

local C = ffi.C as LibC

local SETUP = 425
local ENTER = 426
local GETEVENTS = 1
local READ = 22
local F_GETFD = 1
local EINTR = 4
local EBADF = 9
local PROT = 3
local SHARED_POPULATED = 0x8001
-- the buffers the kernel writes into are a mapping of their own, so a read
-- that lands after the queue gave them back cannot touch the heap
local MAP_PRIVATE_ANONYMOUS = 0x22
local SQ_RING = 0
local CQ_RING = 0x8000000
local SQES = 0x10000000
local PARAMS_BYTES = 120
local SQE_BYTES = 64
local CQE_BYTES = 16
local RELEASED = -1

local CLOSED = "reads: the queue is closed"
local REFUSED = "reads: the reap was refused"
local SETUP_FAILED = "reads: the submission interface cannot be set up"

-- u32 fields of the ring headers and of the setup parameters, read and written
-- through the mapped memory: a 32-bit store, never a wider one that could
-- overwrite a field the kernel moved meanwhile
local function u32_at(address: integer): integer
  local raw = ffi.string(ffi.cast("void *", address) as ffi.CData, 4)
  local value = 0
  for place = 4, 1, -1 do
    value = value * 256 + (raw:byte(place) as integer)
  end
  return value
end

local function put_u32(address: integer, value: integer)
  local rest = value % 4294967296
  ffi.copy(
    ffi.cast("void *", address) as ffi.CData,
    string.char(
      rest % 256,
      math.floor(rest / 256) % 256,
      math.floor(rest / 65536) % 256,
      math.floor(rest / 16777216) % 256
    ),
    4
  )
end

local function call_setup(entries: integer, params: integer): integer
  return C.syscall(
    ffi.cast("long", SETUP),
    ffi.cast("long", entries),
    ffi.cast("void *", params)
  ) as integer
end

local function call_enter(
  descriptor: integer, submit: integer, complete: integer, flags: integer
): integer
  return C.syscall(
    ffi.cast("long", ENTER),
    ffi.cast("long", descriptor),
    ffi.cast("long", submit),
    ffi.cast("long", complete),
    ffi.cast("long", flags),
    nil,
    ffi.cast("long", 0)
  ) as integer
end

-- the address a mapping was made at, or -1 when the kernel refused it
local function map(descriptor: integer, size: integer, offset: integer): integer
  return ffi.cast(
    "intptr_t", C.mmap(nil, size, PROT, SHARED_POPULATED, descriptor, offset)
  ) as integer
end

-- the one block of buffers a queue's reads land in, or -1 when the kernel
-- refused the space
local function map_buffers(size: integer): integer
  return ffi.cast(
    "intptr_t", C.mmap(nil, size, PROT, MAP_PRIVATE_ANONYMOUS, -1, 0)
  ) as integer
end

local function unmap(address: integer, size: integer)
  if address ~= RELEASED then
    C.munmap(ffi.cast("void *", address) as ffi.CData, size)
  end
end

local function line_of(
  path: string, name: string, index: integer, why: string
): string
  return path
    .. ": tensor "
    .. name
    .. ", matrix "
    .. tostring(index)
    .. ": "
    .. why
end

-- the bytes one matrix of a stacked weight holds, or the refusal of an index
-- that names no matrix
local function matrix_of(
  tensor: Tensor, index: integer
): integer | nil, string | nil
  if #tensor.dims < 3 then
    return nil, "holds no matrices"
  end
  local matrices = tensor.dims[3]
  if matrices < 1 or index < 0 or index >= matrices then
    return nil, "holds no matrix " .. tostring(index)
  end
  return math.floor(tensor.bytes / matrices)
end

-- a queue the kernel refused to reap is one no entry may be touched on again,
-- and a descriptor a caller closed is not closed a second time by `close`
local function shut(queue: Queue, failure: integer)
  queue.shut = true
  if failure == -EBADF then
    queue.descriptor = RELEASED
  end
end

local function refuse(queue: Queue, failure: integer): string
  shut(queue, failure)
  return REFUSED
end

local function finish(queue: Queue, tag: integer, landed: integer)
  local held = queue.held[tag]
  queue.pending = queue.pending - 1
  queue.free[#queue.free + 1] = held.slot
  if landed >= held.wanted then
    held.landed = landed
    return
  end
  local moved = landed
  if landed < 0 then
    moved = 0
  end
  held.landed = moved
  held.fault = line_of(
    held.path,
    held.tensor,
    held.index,
    "holds " .. tostring(held.wanted) .. " bytes, read " .. tostring(moved)
  )
end

-- takes every completion the ring holds and frees the entries they landed in
local function drain(queue: Queue)
  local head = u32_at(queue.cq_head)
  local tail = u32_at(queue.cq_tail)
  while head ~= tail do
    local done = ffi.cast(
      "struct tllm_cqe *", queue.cqes + head % (queue.cq_mask + 1) * CQE_BYTES
    ) as Completion
    finish(queue, tonumber(done.tag) as integer, tonumber(done.res) as integer)
    head = head + 1
  end
  put_u32(queue.cq_head, head)
end

-- waits for one read to land and hands its entry back
local function refill(queue: Queue): integer | nil, string | nil
  while true do
    local answer = call_enter(queue.descriptor, 0, 1, GETEVENTS)
    if answer < 0 and answer ~= -EINTR then
      return nil, refuse(queue, answer)
    end
    drain(queue)
    if #queue.free > 0 then
      return table.remove(queue.free)
    end
  end
end

local function free_slot(queue: Queue): integer | nil, string | nil
  if #queue.free > 0 then
    return table.remove(queue.free)
  end
  return refill(queue)
end

-- reaps every read handed over and not yet answered, and refuses a queue whose
-- descriptor no longer answers
local function reap_pending(queue: Queue): string | nil
  local answer = call_enter(queue.descriptor, 0, 0, 0)
  if answer < 0 and answer ~= -EINTR then
    return refuse(queue, answer)
  end
  drain(queue)
  while queue.pending > 0 do
    answer = call_enter(queue.descriptor, 0, 1, GETEVENTS)
    if answer < 0 and answer ~= -EINTR then
      return refuse(queue, answer)
    end
    drain(queue)
  end
  return nil
end

local function post(
  queue: Queue,
  slot: integer,
  tag: integer,
  descriptor: integer,
  position: integer,
  count: integer
): integer
  local tail = u32_at(queue.sq_tail)
  local index = tail % (queue.sq_mask + 1)
  local entry = ffi.cast("struct tllm_sqe *", queue.entries + index * SQE_BYTES)
    as Entry
  entry.head = READ + descriptor * 4294967296
  entry.off = position
  entry.addr = queue.buffers + slot * queue.bytes
  entry.size = count
  entry.tag = tag
  put_u32(queue.sq_array + index * 4, index)
  put_u32(queue.sq_tail, tail + 1)
  return call_enter(queue.descriptor, 1, 0, 0)
end

local probed: boolean | nil = nil

function reads.offered(): boolean
  if probed is nil then
    local answer = false
    if ffi.os == "Linux" then
      local params = ffi.new("uint8_t[?]", PARAMS_BYTES)
      local descriptor = call_setup(1, ffi.cast("intptr_t", params) as integer)
      if descriptor >= 0 then
        C.close(descriptor)
        answer = true
      end
    end
    probed = answer
  end
  return probed as boolean
end

function reads.open(depth: integer, bytes: integer): Queue | nil, string | nil
  if depth < 1 then
    return nil, "reads: cannot take a queue of depth " .. tostring(depth)
  end
  if bytes < 1 then
    return nil, "reads: cannot take a buffer of " .. tostring(bytes) .. " bytes"
  end
  if not reads.offered() then
    return nil, SETUP_FAILED
  end

  local params = ffi.new("uint8_t[?]", PARAMS_BYTES)
  local params_at = ffi.cast("intptr_t", params) as integer
  local descriptor = call_setup(depth, params_at)
  if descriptor < 0 then
    return nil, SETUP_FAILED
  end

  local entry_count = u32_at(params_at)
  local completion_count = u32_at(params_at + 4)
  local array_at = u32_at(params_at + 64)
  local cqes_at = u32_at(params_at + 100)
  local submission_bytes = array_at + entry_count * 4
  local completion_bytes = cqes_at + completion_count * CQE_BYTES
  local entries_bytes = entry_count * SQE_BYTES

  local submission = map(descriptor, submission_bytes, SQ_RING)
  local completion = map(descriptor, completion_bytes, CQ_RING)
  local entries = map(descriptor, entries_bytes, SQES)
  if
    submission == RELEASED or completion == RELEASED or entries == RELEASED
  then
    unmap(submission, submission_bytes)
    unmap(completion, completion_bytes)
    unmap(entries, entries_bytes)
    C.close(descriptor)
    return nil, SETUP_FAILED
  end

  local buffer_bytes = entry_count * bytes
  local buffers = map_buffers(buffer_bytes)
  if buffers == RELEASED then
    unmap(submission, submission_bytes)
    unmap(completion, completion_bytes)
    unmap(entries, entries_bytes)
    C.close(descriptor)
    return nil, "reads: cannot take buffers of " .. tostring(bytes) .. " bytes"
  end

  local queue: Queue = {
    depth = entry_count,
    bytes = bytes,
    pending = 0,
    descriptor = descriptor,
    shut = false,
    closed = false,
    submission = submission,
    submission_bytes = submission_bytes,
    completion = completion,
    completion_bytes = completion_bytes,
    entries = entries,
    entries_bytes = entries_bytes,
    buffers = buffers,
    buffers_bytes = buffer_bytes,
    sq_tail = submission + u32_at(params_at + 44),
    sq_array = submission + array_at,
    sq_mask = u32_at(submission + u32_at(params_at + 48)),
    cq_head = completion + u32_at(params_at + 80),
    cq_tail = completion + u32_at(params_at + 84),
    cqes = completion + cqes_at,
    cq_mask = u32_at(completion + u32_at(params_at + 88)),
    free = {},
    held = {},
    order = {},
    next = 0,
  }
  for slot = 0, entry_count - 1 do
    queue.free[#queue.free + 1] = slot
  end
  return queue
end

function reads.submit(
  queue: Queue, handle: Handle, tensor: Tensor, index: integer
): boolean | nil, string | nil
  if queue.shut then
    return nil, CLOSED
  end
  local each, why = matrix_of(tensor, index)
  if each is nil then
    return nil, line_of(handle.path, tensor.name, index, why as string)
  end
  local count = each as integer
  if count > queue.bytes then
    return nil,
      line_of(
        handle.path,
        tensor.name,
        index,
        "holds "
          .. tostring(count)
          .. " bytes, a buffer holds "
          .. tostring(queue.bytes)
      )
  end
  if
    handle.descriptor == RELEASED or C.fcntl(handle.descriptor, F_GETFD) < 0
  then
    return nil,
      line_of(handle.path, tensor.name, index, "the handle is released")
  end

  local slot, refused = free_slot(queue)
  if slot is nil then
    return nil, refused
  end

  local tag = queue.next
  queue.next = tag + 1
  local position = (
    ffi.cast("intptr_t", tensor.data) as integer
  ) - handle.mapping + index * count
  local answer = post(queue, slot, tag, handle.descriptor, position, count)
  if answer < 1 then
    shut(queue, answer)
    queue.free[#queue.free + 1] = slot
    return nil,
      line_of(handle.path, tensor.name, index, "the platform refused the post")
  end

  queue.held[tag] = {
    slot = slot,
    tensor = tensor.name,
    index = index,
    path = handle.path,
    wanted = count,
    landed = nil,
    fault = nil,
  }
  queue.order[#queue.order + 1] = tag
  queue.pending = queue.pending + 1
  return true
end

function reads.wait(queue: Queue): {reads.Landed} | nil, string | nil
  if queue.shut then
    return nil, CLOSED
  end
  local refused = reap_pending(queue)
  if refused ~= nil then
    return nil, refused
  end
  local rows: {reads.Landed} = {}
  for _, tag in ipairs(queue.order) do
    local held = queue.held[tag]
    rows[#rows + 1] = {
      tensor = held.tensor,
      index = held.index,
      bytes = held.landed as integer,
      fault = held.fault,
    }
  end
  queue.order = {}
  queue.held = {}
  return rows
end

function reads.close(queue: Queue)
  if queue.closed then
    return
  end
  if not queue.shut and queue.pending > 0 then
    reap_pending(queue)
  end
  queue.shut = true
  queue.closed = true
  unmap(queue.submission, queue.submission_bytes)
  unmap(queue.completion, queue.completion_bytes)
  unmap(queue.entries, queue.entries_bytes)
  queue.submission = RELEASED
  queue.completion = RELEASED
  queue.entries = RELEASED
  if queue.descriptor ~= RELEASED then
    C.close(queue.descriptor)
    queue.descriptor = RELEASED
  end
  if queue.buffers ~= RELEASED then
    unmap(queue.buffers, queue.buffers_bytes)
    queue.buffers = RELEASED
  end
end

return reads
