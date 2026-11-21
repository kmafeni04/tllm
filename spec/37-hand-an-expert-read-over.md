# 37: Hand an expert read over and reap it later

**Status**: pending   **Depends on**: 06, 34

## Problem & context

`34` reads each chosen expert's matrix whole with one `pread` on the calling thread, so the pass
stops for the read. A cold 1.5 MB expert costs about 280 us at the 5.3 GB/s a whole read reaches, and
for those 280 us a pass that could be multiplying does nothing. Read size is settled by `34`; what is
missing is a way to start the read and learn about it later, which is what lets a pass compute while
the disk moves bytes.

The mechanism is Linux's `io_uring`: one queue of entries, a submission ring and a completion ring
mapped into the process, and a `pread`-shaped read posted into it. The kernel does the read while Lua
keeps working, and the completion says how many bytes landed. No thread is needed, and nothing has
to be installed for the tree to load.

This is the dangerous piece of the plan. The rings are mapped memory, the buffers a read lands in
belong to the kernel from the moment the read is posted until it completes, and a buffer freed or
reused one step early is a write into memory this process gave back. So every buffer has exactly one
read in it at a time and is freed exactly once, the rings and the descriptor are given back by the
one call that owns them, and a machine or a container where the interface cannot be set up keeps
`34`'s blocking read with no other change. That keeps the tree runnable on a Windows or macOS port,
where the interface does not exist at all.

This spec is the primitive only: a queue that hands a matrix read over and answers what landed. The
policy that uses it, a router prediction one block early, the page-cache gate and the bench figures,
is the spec after this one, and it needs nothing here but `offered`, `open`, `submit`, `wait` and
`close`.

```teal
-- src/reads.tl
local record reads
  record Queue
    depth: integer    -- entries the queue holds, as the platform took them
    bytes: integer    -- what one buffer of the queue holds
    pending: integer  -- reads handed over and not yet reaped
    descriptor: integer -- the queue's descriptor, -1 once it has been given
                      -- back or closed by a caller
    shut: boolean     -- no entry is touched once it is set, by `close` or by a
                      -- refused reap
    closed: boolean   -- set once `close` has returned, so a second `close`
                      -- gives nothing back
  end

  -- what one read leaves behind once it is reaped
  record Landed
    tensor: string
    index: integer
    bytes: integer
    fault: string | nil
  end

  -- whether this platform offers a way to hand a read over at all
  offered: function(): boolean
  -- a queue of at least `depth` entries, one buffer of `bytes` for each entry
  -- it holds
  open: function(depth: integer, bytes: integer): Queue | nil, string | nil
  -- hand one matrix of one tensor over, and return without waiting for the
  -- read it posted
  submit: function(
    Queue, weights.Handle, weights.Tensor, index: integer
  ): boolean | nil, string | nil
  -- reap every read handed over since the last wait, in handover order
  wait: function(Queue): {Landed} | nil, string | nil
  -- reap what is outstanding, then give the entries, buffers and
  -- descriptor back
  close: function(Queue)
end
```

The offsets are `34`'s: a matrix of a stacked tensor sits `index * (tensor.bytes / dims[3])` bytes
into the tensor's own region, and `weights.Handle.mapping` is the address the file was mapped at, so
a tensor's `data` says which position of the file the read asks for. A reaped read's bytes are
whatever the file holds there; nothing here copies them into a graph, because the graph `34` serves
reads the mapping and the kernel already has those pages.

## User stories

- US-001 (P1) As the engine author, I want a matrix read to start now and finish later, so that a
  pass can compute while the disk moves the bytes.
- US-002 (P1) As the engine author, I want a buffer a read landed in to stay mine until that read is
  reaped and to be freed exactly once, so that nothing writes into memory this process gave back.
- US-003 (P1) As a person running the engine where the platform offers no submission interface, I
  want experts read as they are today, so that nothing new has to be installed to run it.
- US-004 (P2) As a person whose model file is damaged, I want a handed-over read that lands short or
  fails to be reported with the line `34/FR-004` writes, so that a broken file is named and not
  computed from.

## Functional requirements

- FR-001 (US-001) WHEN a queue is opened with a depth of one or more and a byte count of one or
  more, THE SYSTEM SHALL take entries of the platform's submission interface, as many as it gives
  for that depth and no fewer than the depth, SHALL take one buffer of the byte count given for each
  entry it holds, and SHALL count those entries in `depth`; IF the depth is below one, or the byte
  count is below one, THEN THE SYSTEM SHALL answer nil and one line naming the size it was given; IF
  any buffer cannot be taken, THEN THE SYSTEM SHALL give back everything it took, the entries
  included, before it answers and SHALL answer nil and one line naming what could not be taken.
- FR-002 (US-001) WHEN a matrix is handed over, THE SYSTEM SHALL ask the platform to read that
  matrix whole, at the position `34/FR-001` reads it from, into a buffer no other read is using, and
  SHALL answer without waiting for it; IF the tensor holds no matrices or the index names none, THEN
  THE SYSTEM SHALL answer nil and one line naming the tensor and the index; IF the matrix holds more
  bytes than one buffer of the queue holds, THEN THE SYSTEM SHALL answer nil and one line naming the
  tensor, the index and both byte counts; IF the queue is shut, THEN THE SYSTEM SHALL answer nil and
  one line saying the queue is closed, before it touches an entry; IF the handle's descriptor is
  closed, THEN THE SYSTEM SHALL answer nil and one line saying the handle is released, as `34/FR-004`
  refuses a released handle; IF the platform refuses the post, THEN THE SYSTEM SHALL shut the queue
  and answer nil and one line naming the tensor, the index and the refused post.
- FR-003 (US-001) WHEN `wait` is called, THE SYSTEM SHALL answer one row for each read handed over
  since the last `wait`, including each read FR-005 reaped inside a handover, and none for a read it
  already answered, in the order the reads were handed over, each row holding the tensor's name, the
  matrix's index, the number of bytes that landed and the row's fault, SHALL free the entry of every
  read it reaps itself so that a queue whose every read has been answered holds every entry free, and
  SHALL count `pending` as the reads handed over and not yet reaped; IF the queue is shut, THEN THE
  SYSTEM SHALL answer nil and that same line; IF the platform refuses to reap, THEN THE SYSTEM SHALL
  shut the queue and answer nil and one line saying so.
- FR-004 (US-004) IF a reaped read landed fewer bytes than its matrix holds, or failed, THEN THE
  SYSTEM SHALL give that row one line naming the tensor and the matrix's index, and the bytes that
  landed, 0 where the read failed.
- FR-005 (US-002) WHEN a matrix is handed over and no entry of the queue is free, THE SYSTEM SHALL
  wait for the reads handed over earliest until an entry is free, reaping every read that has landed
  on the way, before it hands the matrix over, so the queue never holds more reads than its depth and
  no buffer holds two reads at once; IF the platform refuses that reap, THEN THE SYSTEM SHALL shut
  the queue, hand nothing over and answer nil and one line saying the reap was refused.
- FR-006 (US-002) WHEN a queue is closed, THE SYSTEM SHALL wait for every read it still holds in
  flight and reap it, then unmap the submission and completion rings, close the descriptor the queue
  still holds, answering nothing when it holds none, free every one of its buffers and the queue
  itself, SHALL give nothing back a second time when `close` has been called twice whatever `shut`
  holds, and SHALL touch no weights handle, so nothing a handed-over read took outlives the queue;
  the reads `close` reaps itself are answered to no caller; IF the queue's reap is refused while
  closing, THEN THE SYSTEM SHALL still unmap its rings, close the descriptor it still holds and free
  its buffers.
- FR-007 (US-003) WHERE the platform offers no submission interface, THE SYSTEM SHALL answer
  `offered` false and leave every matrix to `34/FR-001`'s blocking read on the calling thread.
  Not testable: no case can take a kernel's submission interface away; the fallback is the read path
  `34`'s cases already cover, and a reader confirms the one probe in `src/reads.tl`.
- FR-008 (US-003) IF the platform offers the interface but it cannot be set up for this process,
  THEN THE SYSTEM SHALL answer nil and one line saying so and take nothing.

## Non-goals

- No thread, no dependency that must be present to load, and no second copy a graph reads: the
  buffers exist so the kernel has somewhere to put the bytes, and the buffer only fills the page
  cache behind the mapping the graph reads.
- No policy: no prediction, no gate, no counter and no config key here, and no decision about which
  matrices are worth handing over, which the spec after this one makes.
- No change to how a read on demand is made, to the mapping, to `34`'s record of the matrices a
  handle has read, or to the card path, where a graph reads a device copy and no mapping at all.

## Tests

- T-001 (FR-001, FR-002, FR-003): `a handed-over read answers the bytes it landed`, proves on a
  machine where `reads.offered` answers true, otherwise skipping with the line
  `no submission interface`, that a queue of depth 2 with each buffer the bytes a matrix holds, over
  a fixture tensor holding two matrices, answers a `wait` after both matrices are handed over two
  rows in handover order, the second naming that tensor, its index 1 and the bytes its second matrix
  holds, the position those bytes come from being the one `34`'s own case proves; that the queue's
  `pending` is 0 afterwards; and that a second `wait` answers no row.
- T-002 (FR-005): `a full queue reaps before it takes another`, proves on the same machine and a
  fixture tensor holding five matrices that a queue of depth 4, with each buffer the bytes a matrix
  holds, takes every one of the five handovers with `pending` at most the queue's `depth`
  after each and 4 after the fourth, and that a `wait` answers five rows, one per matrix, in the order
  they were handed over, and `pending` 0 after it.
- T-003 (FR-004): `a short handed-over read names its tensor and its index`, proves on the same
  machine and a fixture cut a few bytes inside its last matrix that the row for that matrix holds one
  line naming the tensor and the index, and the bytes that landed.
- T-004 (FR-002, FR-003, FR-006): `closing a queue leaves nothing behind`, proves on the same machine
  that the process holds no more descriptors and no more mapped regions after `close` than it held
  before `open`, counted from `/proc/self/fd` and `/proc/self/maps`, the queue opened with a depth of
  4 and buffers of 1 MiB each over a fixture tensor whose matrices are 1 MiB each, so a buffer the
  case leaked stands as a mapped region of its own, a fraction of that tensor handed over before the
  close; that a second `close` runs without a fault; that a `submit` and a `wait` on the closed queue
  each answer nil and one line saying the queue is closed, rather than touching its rings; that a
  queue closed while a matrix handed to it a step earlier is still in flight leaves no descriptor and
  no mapped region behind and raises nothing, the reap of that read being what `close` does before it
  frees; and that a `wait` on a queue whose descriptor the case closed with `ffi.C.close` answers nil
  and one line saying the reap was refused, with the queue shut afterwards so a `submit` on it answers
  nil and one line saying the queue is closed, and a `close` on it afterwards raises nothing.
- T-005 (FR-001, FR-002): `a queue or a matrix it cannot take is refused`, proves on the same machine
  that `open` with a depth of 0 and with a byte count of 0 each answer nil and one line holding the
  size it was given, that `submit` on a two-dimensional tensor and on an index past the end of a
  stacked tensor each answer nil and one line naming the tensor and the index, and that `submit` of a
  matrix of 8 bytes on a queue of depth 1 with 4 bytes per buffer answers nil and one line naming the
  tensor, the index and both byte counts, and that `submit` of a matrix of a handle whose descriptor
  the case closed answers nil and one line saying the handle is released.
- T-006 (FR-008): `an interface that cannot be set up takes nothing`, proves on a machine where
  `reads.offered` answers true, otherwise skipping with the line `no submission interface`, that
  `open` answers nil and one line saying so when the process's soft `RLIMIT_NOFILE` has been lowered
  to its own descriptor count and spent, and that the descriptors the case took and the limit it
  lowered are given back afterwards, so the queue the next case opens is unaffected.
- T-007 (FR-001): `a queue it cannot buffer gives back what it took`, proves on a machine where
  `reads.offered` answers true, otherwise skipping with the line `no submission interface`, that
  `open` with a depth of 4 and a byte count larger than the address space the process is allowed
  answers nil and one line naming what could not be taken, that the descriptors and the mapped
  regions the process holds afterwards are what it held before the call, counted from
  `/proc/self/fd` and `/proc/self/maps`, and that the soft `RLIMIT_AS` the case lowered to make the
  allocation fail is given back afterwards.
- T-008 (FR-002, FR-005): `a handover the platform refuses shuts the queue`, proves on the same
  machine that a queue of depth 1 whose descriptor the case closed with `ffi.C.close` answers a
  `submit` nil and one line naming the tensor, the index and the refused post, with the queue shut
  afterwards; and that a second queue of depth 1, handed one matrix before its descriptor is closed,
  answers a second `submit` nil and one line saying the reap was refused, with `pending` 1
  afterwards, the queue shut, and a `close` on it raising nothing.

## Open questions

- Whether the primitive and the policy that uses it are one document (options: one spec, the
  primitive here and the policy after it, the policy here and the primitive before it).
  (decided: the primitive here, the lookahead, gate, counters, config key and bench line in the spec
  after this one)

## Amendments

- `34/FR-001`: a matrix a caller hands over may be read by the platform's submission interface
  instead of one blocking read on the calling thread, so `34`'s non-goal refusing `io_uring` and
  asynchronous reads gives way to this one path, with no new dependency and still no thread, and a
  matrix read on demand stays one blocking read.
