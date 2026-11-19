# 34: Whole-blob expert reads

**Status**: done   **Depends on**: 06, 13, 21, 26, 29, 33

## Problem & context

Every weight is read through the mapping `06` made: `weights.load` maps the file and `compute`'s
`weight_of` hands ggml a pointer into that mapping, so an expert's 1.5 MB is fetched by the page
fault inside the matrix multiply that walks it, one small page at a time. 4 KB requests were
measured at 0.49 GB/s against 5.3 GB/s for 1 MB requests, so a cold expert's arithmetic waits on the
disk for several times what its bytes are worth.

The fix is request size, and it is not parallelism. One read of the whole matrix gets the 5.3 GB/s
figure on one thread; queue depth was measured to buy nothing (io_uring was identical to `pread` at
every size and depth), and OS threads would cost a C file and a Linux-only build. So each matrix of
each chosen expert is read whole with one `pread` on the calling thread, before the graph that
multiplies by it runs, and nothing else about how a pass reads weights changes.

A handle remembers the matrices it has read and does not read one twice, which is what bounds what
this buys: the first pass over a model whose experts the page cache holds pays one request per
matrix instead of a fault per 4 KB, and every later pass over those experts pays nothing at all.
Reading them again every token would copy the same 36 MB the kernel already holds, which costs more
than the faults it saves. A model whose experts do not fit in memory is evicted between passes, and
this spec leaves that model as it is today: the only thing that could tell the handle its bytes are
gone is the kernel's own residency table, which costs a call per matrix per token and cannot be
pinned from a test.

The graph keeps reading the mapping and no byte is copied into it. The read only fills the page
cache behind the pointer `weight_of` already hands out, so a matrix costs its copy once in a
handle's lifetime and a later zero-copy read path over that same mapping stays open.

The seam already exists. `compute` chooses a block's experts on the host between the block's votes
graph and its choice graph (`src/compute.tl:3920`), and `expert_bytes` already takes the chosen set
apart into the three matrices an expert holds, so their offsets and sizes are known before the graph
that reads them is built. What is missing is the descriptor to read from (`weights.load` closes the
one it opened for `mmap`, which `06/FR-013` now keeps open on the handle), the buffer a read lands
in, the handle's record of what it has read, and three fields in `compute.Measure` beside the
`tokens` field `33` added, so `29`'s bench can report the reads.

```teal
local record weights
  record Handle
    -- 06's fields, plus:
    mapping: integer
    descriptor: integer
    read_matrices: {string: {integer: boolean}}
    buffer: ffi.CData | nil
    buffer_bytes: integer
  end

  -- reads matrix `index` of a three-dimensional tensor whole into the handle's
  -- buffer, from the tensor's own region, `index * (tensor.bytes / dims[3])`
  -- bytes into it, and only when this handle has not read it before; answers
  -- whether it read it
  read: function(Handle, Tensor, index: integer): boolean | nil, string | nil
end

local record compute
  record Measure
    -- 29's fields, unchanged
    blob_reads: integer
    blob_bytes: integer
    blob_seconds: number
  end
end
```

The buffer is allocated by a handle's first read, grown when a larger matrix comes, reused for
every later read of that handle and freed by `release` beside the mapping. `mapping` is the address
the mapping was made at, so a tensor's `data` says which position of the file the whole read asks
for.

## User stories

- US-001 (P1) As a person running a model whose weights are on disk, I want a chosen expert's
  weights read in one request, so that a cold pass is not spent fetching them a page at a time.
- US-002 (P1) As a person running several conversations in one pass, I want an expert two rows
  choose read once, so that the disk moves it once and not once per row.
- US-003 (P2) As a person measuring a change, I want bench to report those reads, so that I can see
  what the disk costs a turn.

## Functional requirements

- FR-001 (US-001) WHEN a pass has chosen a block's experts and reads them from the mapping, THE
  SYSTEM SHALL read each matrix of each chosen expert whole, one read at that matrix's offset in the
  model file, before the graph that multiplies by it runs; a chosen matrix the model's placement
  marks resident SHALL NOT be read whole, and every other chosen matrix SHALL be read whole whether
  or not the model's fixed weights are on the card.
- FR-002 (US-001, US-002) WHEN this handle has already read a matrix, THE SYSTEM SHALL NOT read it
  again, whether the next row asking for it is another row of the same pass or a pass that comes
  later.
- FR-003 (US-001) WHEN a matrix is read, THE SYSTEM SHALL add one to `measure.blob_reads`, that
  matrix's bytes to `measure.blob_bytes` and the read's seconds to `measure.blob_seconds`.
- FR-004 (US-001) IF a read returns fewer bytes than the matrix holds, or fails, or the handle it is
  asked of has been released, THEN THE SYSTEM SHALL answer nil and one line naming the tensor.
- FR-005 (US-001) WHEN a model handle is released, THE SYSTEM SHALL close the descriptor its reads
  use, once, and free the buffer they read into.
- FR-006 (US-003) WHEN `bench` runs a length's turn, THE SYSTEM SHALL write at the end of that
  length's `experts` line, when `29/FR-005` writes it, the reads the turn made, the bytes those
  reads moved and their milliseconds, as the `measure` values at that turn's end.
- FR-007 (US-001) IF a read a pass asked for answers nil, THEN THE SYSTEM SHALL stop the pass and
  return nil and that same line, before any graph that reads the matrix is built.

## Non-goals

- No threads, no io_uring, no asynchronous read and no new dependency: every read is one `pread` on
  the calling thread, because depth was measured to buy nothing and request size is what binds.
- No change to the mapping, nor to any tensor but the routed experts' three matrices: the graph
  keeps reading the mapping, which is what makes the second pass over an expert free and what
  leaves a zero-copy read path over that mapping open.
- No change to the card: `21` copies a model's weights into device memory when it loads, a graph
  over that copy reads no mapping, and that copy stays exactly as it is.

## Tests

- T-001 (FR-001): `a whole-matrix read returns the mapping's bytes`, proves a fixture tensor that
  holds two matrices, read at index 1 through `weights.read`, holds in the handle's buffer the bytes
  the mapping holds at that matrix's own offset, which differ from the bytes the same tensor's first
  matrix holds.
- T-002 (FR-002): `a matrix is read once for the handle`, proves a second pass over a model whose
  expert an earlier pass routed counts no read, and a pass that routes an expert for the first time
  counts it.
- T-003 (FR-003): `a pass counts the expert reads`, proves one row's pass over the fixture counts one
  read per chosen matrix and the bytes they hold, and moves the read seconds off zero.
- T-004 (FR-004, FR-007): `a short matrix read is refused with its tensor`, proves a fixture cut a
  few bytes inside its last expert matrix answers nil and one line holding that tensor's name, and
  that the pass that asked for it returns nil and that same line.
- T-005 (FR-004): `a read that cannot be made is refused with its tensor`, proves `weights.read`
  after `release` answers nil and one line naming the tensor instead of reading from the closed
  descriptor, and that on a fresh handle whose `descriptor` a case closes with `ffi.C.close`,
  `weights.read` answers nil and one line naming the tensor.
- T-006 (FR-005): `release closes the read descriptor and frees its buffer`, proves the process
  holds exactly one more descriptor after `load` and no more than it held before `load` after
  `release`, and that the handle's `buffer` is nil and its `buffer_bytes` 0 once `release` has
  returned.
- T-007 (FR-006): `bench reports the expert reads`, proves a bench run's `experts` line carries the
  turn's reads, the bytes read and the milliseconds beside the routed count and bytes, and that its
  MB read figure is the turn's `measure.blob_bytes` value over 1000000 to one decimal place.
- T-008 (FR-001): `a device model reads no matrix whole`, gated on `on_device` as
  `tests/device.test.tl` gates its cases, proves `measure.blob_reads` is 0 after a pass over a model
  whose backend holds a device copy.

## Open questions

- None.

## Amendments

- None.
