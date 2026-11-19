# 32: One graph per block per pass

**Status**: done   **Depends on**: 11, 12, 13, 14, 15, 29

## Problem & context

Every operation the engine runs builds a ggml graph of its own. `matrix_vector`, `normalise`,
`rotary`, `softmax`, `add`, `swiglu` and `sigmoid_gate` each aligned_alloc an arena, call
`ggml_init` over it, build a graph of a few nodes, compute, read the result back into a Lua table
and free the context and the buffer. `gathered`, `run` and `opened` in `src/compute.tl` are that
cycle, and `29` quotes it at 16 us of alloc, build and free. One row through one full-attention
block of the local file builds 58 of those cycles for the attention half, 36 of them one norm and
one rope per head, and 43 for the feed-forward; a token lands near `29`'s 1,250 graphs, which at
16 us is 20 to 40 ms of overhead before any arithmetic.

This spec makes a block's work one graph per pass instead of one graph per operation. A graph is
built over the whole batch of rows and heads, so a head's norm and rope become nodes of the block's
graph rather than graphs of their own, all query heads sharing one gain tensor and one rope
parameter. Its arena is the model's, allocated once and initialised again for every graph, so the
16 us cycle loses its arena allocation and its arena free and the rest of it is paid once per block
instead of once per operation. A run of the local file's 40 blocks goes from about 1,250 graphs a
token to 122, and at `29`'s 16 us from 20 to 40 ms to about 2 ms.

Three graphs per block, not one, and the reasons are the contracts `12`, `13` and `14` publish. The
block's attention or linear half returns what it adds to the hidden numbers, and the model pass
adds that to the hidden in Lua, where the add costs no graph at all; the residual add therefore
cannot sit in a graph that also reads the normed sum. The router's softmax is read back and its
`experts_used` best rows chosen on the host, `13`'s rule and `13`'s tested numbers, so the
feed-forward cannot be one graph without putting the choice inside ggml. What is left is one graph
for a block's attention or linear half, one for its norm and router, one for its chosen experts and
shared expert with both residual adds, and for the pass itself one graph gathering the batch's
embedding rows and one computing its scores.

What a block's graph reads and writes is decided here too. It reads the block's weights, the
pass's rows for that block, and the state the block keeps: for an attention block each row's
sequence cache up to that row's position, and it appends that position's key and value rows itself;
for a linear block each row's sequence history and state, and writes back what that row's position
leaves them. Before this spec those numbers crossed into and out of Lua per operation, the cache
rows quantised on the host and the linear history shifted there too.
One row per sequence, which is what `26` allows, so a pass of several conversations is still one
graph per block: the rows' state tensors and caches are separate host tensors of one graph.

One arena per model, sized once from the largest of the graphs a pass builds at the model's whole
position count, because the scores a graph holds for each of the batch's heads grow with the
position. A block handle built on its own, as `12`, `13` and `14` publish one, gets one arena of
its own by the same rule, made at its first graph since that is the first time the tensors it reads
are known. Placing a graph in an arena again is what the cost drops by, and it is also what it
costs: every tensor of an earlier graph is gone once the arena is initialised again, so each graph
builds its own descriptors and its numbers are read back into Lua before the next one is built. The
feed-forward's second graph therefore normalises the row again rather than reading the first
graph's norm: the router's numbers are the only ones that cross between them. The guard that a
graph bigger than its arena is a line and not a write past the buffer stays `11/FR-016`'s check.
`11` found that `ggml_used_mem` cannot serve as that guard, since it reports what ggml placed and
ggml never places past its buffer, so a comparison against the buffer's size can never fire; that
holds here, and the guard is the NULL check. Its reach does change: `11`'s per-call builds lean on
the constant beside each operation, and no constant can be the contract for a graph of a block, so
every node's result is checked and a NULL raises.

On the device path of `21` the same reasoning applies to device memory: a block gets one device
buffer holding its graphs' tensors, sized the same way from the largest graph the block builds at
the geometry's whole position count, instead of one per operation. The cache stays in host memory as
`21/FR-003` states, and because the graph now appends the position's key and value rows itself, the
block copies back the rows it appended once the graph has computed.

The numbers must not move. `11` to `15` agree the per-operation results and the one-line refusals a
pass returns, and every case of theirs runs unchanged; `29`'s report keeps its line shapes, its
counts and its routing dump, so what changes in it is the time a token takes and `measure.graphs`,
which it does not print, not one number a case pins.

```teal
local record compute
  record Measure
    passes: integer
    compute: number
    sampling: number
    ready: number
    routed: integer
    routed_bytes: integer
    graphs: integer
  end
end
```

`graphs` is the number of ggml graphs the model's passes have built since `measure` was last
zeroed, beside `passes`, which is the same count of passes. This snippet supersedes the six-field
record `29` declares, and `29/FR-009` is amended to say so.

## User stories

- US-001 (P1) As the engine author, I want each block of a pass and the pass itself to run a fixed few ggml graphs, so that a token stops paying 16 us per operation.
- US-002 (P1) As the engine author, I want every head's norm and rope computed in one node of the block's graph over the batch, so that the graph count does not grow with the head count or the row count.
- US-003 (P1) As the engine author, I want the block's graph to read and write its own cache rows, history and state, so that a pass does not carry those numbers through Lua between operations.
- US-004 (P2) As the engine author, I want one arena per model, written over again for every graph, so that building a graph costs no arena allocation and no arena free.
- US-005 (P1) As the engine author, I want every number and every refusal `11` to `15` agree to stay as it is, so that this is a speed change and not a maths change.

## Functional requirements

- FR-001 (US-001) WHEN a model's pass runs, THE SYSTEM SHALL build three ggml graphs for each block
  of the run, one for the block's attention or linear half and two for its feed-forward, and two
  more for the pass, one gathering its rows' embedding numbers and one computing its scores, so a
  pass of a run of N blocks builds `3 * N + 2` graphs, and SHALL add to the model's
  `measure.graphs` the number of graphs it built, being `3 * N + 2` for a pass that returns scores
  and the graphs it built before a refusal for one that does not.
- FR-002 (US-002) WHEN a block's graph is built, THE SYSTEM SHALL compute the geometry's query heads'
  norms and rope and its key heads' norms and rope in one node each over all of the batch's heads,
  so that a pass builds `3 * N + 2` graphs whatever the geometry's `heads` and `kv_heads` and
  whatever number of rows the pass holds.
- FR-003 (US-003) WHEN a block's graph is built, THE SYSTEM SHALL hand it the block's weights and the
  pass's rows and SHALL have the graph read the state the block keeps and write it back, that being
  each row's sequence cache up to that row's position with that position's key and value rows
  appended for an attention block, and each row's sequence history and state as that row's position
  leaves them for a linear block, so that no cache row, history or state number is read or written
  through a Lua list.
  Not testable: what a graph reads and writes is memory, and the numbers a later position returns
  from what the graph wrote are the numbers `12`, `14` and `15` already pin; a reader confirms the
  cache and state tensors the block's graph is built with.
- FR-004 (US-004) WHEN a model builds a graph for a pass, THE SYSTEM SHALL place that graph's tensors
  in the one arena the model holds, initialised again over the buffer it already has, SHALL read that
  graph's numbers back into Lua before the arena is initialised again, and SHALL reference no tensor
  of an earlier graph after that initialisation, so that a pass two hundred deep returns the numbers
  a pass one deep returned, all two hundred fit the arena allocated when the model was built, and no
  number is read through a tensor the buffer no longer backs.
- FR-005 (US-004) THE SYSTEM SHALL size a model's arena at its build from the largest of the graphs
  FR-001 counts a pass of one row per sequence to build at the model's `positions`, so an arena is
  the worst case of one graph and not the sum of the run's blocks, and WHEN a pass holds more rows
  than that arena was measured at, replace it with one sized the same way for that pass, as
  `33/FR-003` says, the block handles of that model building over the arena the model then holds and
  the block device buffers that pool holds going with it.
  Not testable: nothing a model publishes reports its arena's size or how many allocations it made;
  a reader confirms the one allocation and the largest-graph sizing in the tree.
- FR-006 (US-004) WHEN a block handle is built outside a model, THE SYSTEM SHALL give it one arena of
  its own, sized by FR-005's rule from the largest graph the handle builds at its geometry's
  `positions`, made at the handle's first graph because that is the first time the tensors it reads
  and the rows it is given are known, so that between being built and being used it holds none.
  Not testable: `12`'s, `13`'s and `14`'s cases build such a handle and run unchanged, and nothing
  reports its arena's size; a reader confirms the handle's one arena.
- FR-007 (US-004) IF a graph asks an arena for a tensor it has no room for, THEN THE SYSTEM SHALL
  stop that graph's build at that tensor and return nil and one line naming the block or the pass's
  graph and the numbers the tensor asked for, for a NULL a node returns as for a NULL a build asks
  for, a node's result having no number to name and the line then saying it had no room.
  Not testable: reaching the branch needs an arena one tensor short, which the sizing makes
  unreachable from outside; the branch is the raise in `graph_tensor` and the `pcall` around a
  graph's build, confirmed by inspection, and `11/FR-016` states the same for one operation.
- FR-008 (US-004) WHEN a block's graph runs on a backend whose weights are in device memory or
  whose model's placement marks at least one tensor, whatever the block's own weights, THE
  SYSTEM SHALL allocate one device buffer for that block's graphs, sized by FR-005's rule as
  `33/FR-003` amends it from the largest graph that block builds at the geometry's `positions`, and
  reuse it for every later pass's graph of that block, a pass holding more rows replacing the
  buffers of the pool a model grew so the next graph of each block takes the buffer the grown pool
  gives it and no block keeps a buffer across that replacement, and SHALL copy back the key and value rows the graph appended so the rest of
  the cache stays in host memory.
  Not testable: the library reports neither which backend buffer a graph's tensors landed in nor how
  many buffers were allocated, and `21/FR-003` makes the same argument; the copy-back is what
  `12/T-002` and `15/T-001` catch on a machine whose weights are in device memory, and a reader
  confirms the one device allocation.
- FR-009 (US-004) WHEN a model's `close` or a block handle's `close` is called, THE SYSTEM SHALL free
  the arena that handle holds, a block a model built holding none of its own, and every device buffer
  FR-008 allocated for the blocks that handle holds or, for a handle built outside a model, for
  itself, before the call returns, so that a closed model and a closed handle leave no arena and no
  device buffer behind and a second `close` changes nothing.
  Not testable: nothing reports an arena's size, the allocations a handle made or which buffer a
  graph's tensors landed in; a reader confirms the free in each close path, as `11`'s Problem &
  context states it for the maths handle's arena.
- FR-010 (US-005) THE SYSTEM SHALL return, for every operation of a pass, the numbers `11` to `15`
  agree it returns, and for every fault they agree it refuses, nil and the line they agree.
  Not testable: the cases of `11`, `12`, `13`, `14`, `15` and `29` assert exactly these and run
  unchanged in the same suite, so a case here would be those assertions twice.
- FR-011 (US-004) WHEN a block handle built outside a model is given `reserve` for a sequence at more
  positions than its geometry's `positions`, THE SYSTEM SHALL return one line naming `reserve` and
  the positions and reserve nothing, so that the one arena FR-006 gives the handle holds every graph
  it builds.

## Non-goals

- No change to the weights mapping, the cache format on disk, the sampler, the routes or the worker
  pool: this spec changes how a block's maths is built and nothing else.
- No batching beyond `26`: one row per sequence in a pass and no two rows of one sequence in one
  graph, and the router's `experts_used` choice stays `13`'s host rule over `13`'s numbers.
- No CUDA kernel of our own and no new line in `bench`'s report: `29`'s line shapes, counts and
  routing dump stay as they are, and the maths handle of `11` keeps its per-call graphs.

## Tests

- T-001 (FR-001): `a pass builds three graphs per block and two for itself`, proves that on
  `15/T-001`'s fixture a `forward` adds 5 to the model's `measure.graphs`, and that on `15/T-002`'s
  two-block fixture a `forward` adds 8.
- T-002 (FR-002): `neither heads nor rows multiply the graphs`, proves that on `15/T-005`'s
  four-query-head fixture a `forward` adds 5 to `measure.graphs`, the 4 query heads' and 1 key
  head's norms and rope of `12`'s geometry being nodes of the block's graph and not graphs of their
  own, and that on `15/T-002`'s two-block fixture built with `sequences` 2, with the sequences `"a"`
  and `"b"` reserved, a `pass` of two rows, one for `"a"` and one for `"b"` each at position 0,
  adds 8, the same as a pass of one row.
- T-003 (FR-004): `two hundred passes fit the arena built with the model`, proves that on
  `15/T-001`'s fixture with `context_length` 256, built by `compute.model(backend, handle, 256)`,
  two hundred `forward(0, p)` calls for p from 0 to 199 each return `{1.4142136, 0, 1.4142136}` to
  within 1e-5 and nil, the block contributing zeros while its score tensors grow with every pass,
  so the arena the model allocated when it was built serves a pass two hundred deep and every one
  before it, and proves nothing about the read-back order FR-004 states, which a reader confirms
  and no number here would catch.
- T-004 (FR-011): `a reserve above the handle's geometry is refused`, proves that on a handle
  `compute.attend` built with `positions` 2, `12/T-001`'s tensors under `blk.0.` being what a block
  reads, `reserve("s", 3)` returns one line naming `reserve` and 3, and that `reserve("s", 2)`
  returns nil, so no sequence of a standalone handle holds more positions than the one arena FR-006
  sized for it.

## Open questions

- None.

## Amendments

- None.
