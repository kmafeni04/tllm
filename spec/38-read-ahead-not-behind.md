# 38: Read ahead, not behind

**Status**: pending   **Depends on**: 04, 15, 21, 29, 32, 33, 34, 35, 36, 37

## Problem & context

`34` reads a chosen expert's matrices whole before the graph that multiplies by them, so every cold
expert still stops the pass for its bytes. `37` gave `reads.tl` a queue that hands a whole-matrix
read to the platform and reaps it later, and nothing uses it, because nothing yet knows which
matrices to hand over. Guessing is the only other option: a model has thousands of experts and a
pass touches a handful, so handing over the wrong ones costs disk time for bytes no graph reads.

The pass already says what the next block will route, almost. A block leaves the row the next
block's attention takes in, and that block's router reads what its attention leaves. When a
block's attention adds little, the row a block leaves is close to the row the next block's router
sees, so scoring the next block's router on that row names most of the experts the next block
routes. That is a prediction, not an answer: it changes no number a pass returns, and when it is
wrong the following block reads the experts it really routed on demand, as it does today.

Handing those reads over when a block returns, and reaping them before the graph that multiplies by
them, lets the next block's attention and vote graphs run while the disk moves the bytes. The
prediction rides in the third graph `32/FR-001` builds for a block, the choice graph holding the
chosen experts and the shared expert with both residual adds, because that graph already holds the
row the block leaves. A pass still builds `3 * N + 2` graphs, the prediction being nodes of a graph the block already
builds, and it runs on every pass, even with the gate shut, because the block's graph keeps one
shape and one arena size whatever the gate holds; a shut gate hands nothing over.

The gate is what keeps read-ahead from competing with the page cache. The model sizes the machine's
free memory against the bytes its expert matrices hold: when that room is large the page cache can
hold what an ahead read fetches and bring, when it is small the read evicts what the run needs.
Landings decide the rest, because a prediction that keeps naming the wrong experts costs reads for
nothing. Both are checked over passes rather than every pass, so a run does not flap, and a person
can force the state when they know better than the machine.

The queue is the model's, not the weights handle's: `37` takes one for a byte count, and the model
is the thing that knows its largest expert matrix and the block boundaries the reads are handed over
and reaped at. A platform without the submission interface keeps `34`'s blocking read.

```teal
-- src/prefetch.tl
local record prefetch
  -- the bytes the machine has free for a page cache, or nil where it does not say
  memory: function(): integer | nil
  -- open read-ahead, close it, or leave it as it is, from the room a window had
  decide: function(
    room: number | nil, named: integer, landed: integer, open: boolean
  ): boolean
end

local record compute
  record Measure
    -- 36's fields, unchanged
    prefetch_reads: integer   -- matrices handed over since measure was zeroed
    prefetch_named: integer   -- experts a handover named since measure was zeroed
    prefetch_landed: integer  -- those named experts that landed whole
  end

  record ReadAhead
    open: boolean            -- whether a block is handing reads over
    depth: integer           -- entries the queue holds, 0 when there is none
    bytes: integer           -- what one buffer holds, 0 when there is none
  end

  record Model
    -- 36's fields, unchanged
    read_ahead: ReadAhead
  end

  model: function(
    backend: Backend,
    handle: weights.Handle,
    positions: integer | nil,
    cache_type: CacheType | nil,
    sequences?: integer,
    ubatch?: integer,
    vram?: integer,
    profile?: string,
    prefetch?: boolean
  ): Model | nil, string | nil
end

local record serve
  record Endpoint
    -- 35's and 36's fields, unchanged
    prefetch: boolean | nil
  end
end

local record bench
  record Settings
    -- 36's fields, unchanged
    prefetch: boolean | nil
  end
end
```

This snippet supersedes the `Measure` record and the `model` declaration `36` leaves, the
`Endpoint` record `36` declares and the `Settings` record `36` declares. `prefetch.decide` is the
gate's rule as a value, so a case gives it figures and checks what read-ahead does; `prefetch.memory`
answers `MemAvailable` from `/proc/meminfo`, in bytes, or nil. `read_ahead.depth` and `read_ahead.bytes`
are the queue the model took, so a case reads the size it was given without reaching into the queue.

## User stories

- US-001 (P1) As a person running a model whose experts are on disk, I want the next block's expert
  reads handed over while this block computes, so that a cold pass waits less for the disk.
- US-002 (P1) As the engine author, I want the prediction, the handovers and a failed reap to change
  no number and no refusal a pass returns, so that read-ahead is a speed change.
- US-003 (P1) As the person running the server, I want read-ahead to open and close with what the
  machine can hold and how well the prediction lands, so that it helps the page cache instead of
  fighting it.
- US-004 (P2) As the person running the engine, I want to force read-ahead on or off, so that a run
  is what I asked for.
- US-005 (P2) As the person benchmarking a change, I want the ahead reads and their landings in the
  report, so that I can see what read-ahead bought.

## Functional requirements

- FR-001 (US-001) WHEN a pass runs a block that a following block of the run comes after, THE SYSTEM
  SHALL name, for each row that block carried, the `experts_used` experts the following block's
  `post_attention_norm.weight` and `ffn_gate_inp.weight` choose for the row the block leaves, scored
  as `13/FR-003` and chosen by `13/FR-004`'s rule, a tie going to the lower expert, and SHALL name no
  expert when it runs the pass's first or last block.
- FR-002 (US-002) WHEN a pass of N blocks runs, THE SYSTEM SHALL build `3 * N + 2` graphs, FR-001's
  naming being nodes of the third graph each block already builds, so a pass costs no graph for the
  prediction.
- FR-003 (US-001) WHEN a block that a following block comes after returns and read-ahead is open,
  THE SYSTEM SHALL hand over one whole-matrix read for each matrix of each expert FR-001 named for
  the following block that `34/FR-001` would read whole and that this handle has not read whole, one
  handover for an expert whatever rows named it, and SHALL add one to `measure.prefetch_reads` for
  each matrix handed over.
- FR-004 (US-001) WHEN the following block is about to build the graph that multiplies by its chosen
  experts, THE SYSTEM SHALL wait for every read FR-003 handed over for it, whatever experts that
  block routes, record every matrix that landed whole as one the handle has read so `34/FR-002`
  reads it no further, add to `measure.prefetch_named` each expert at least one of whose matrices was
  handed over, and add to `measure.prefetch_landed` each of those experts whose every handover landed
  whole.
- FR-005 (US-002) IF a read FR-003 handed over lands short or fails, or the wait for those reads is
  refused, THEN THE SYSTEM SHALL close read-ahead until the model's next decision, a model given
  `prefetch` true having none and so staying closed for its life, record no matrix of that read, and
  leave that matrix to the following block's demand read, which answers the line `34/FR-007` gives;
  the line `37/FR-004` gives a row that did not land whole SHALL be dropped, so only the demand
  read's line reaches a caller.
- FR-006 (US-002) THE SYSTEM SHALL return the score rows, the records the routing sink is handed and
  the refusals a pass of the same rows makes with read-ahead closed, whether read-ahead is open or
  named experts no block of that pass routes.
- FR-007 (US-003) WHEN a model is built without a `prefetch` value, THE SYSTEM SHALL decide
  read-ahead's state at the build and when a pass begins after 16 completed passes since the build
  or the last decision, opening read-ahead where the room is at least 0.65 and at least half the
  experts named and landed since the build or the last decision landed whole, a window that named
  none counting as having landed all, closing it where the room is below 0.55 or fewer than half
  those experts landed whole, and leaving it as it is otherwise; the room SHALL be the bytes
  `prefetch.memory` answers over the bytes the model's expert matrices hold, taken at the build and
  at every decision, and IF it answers nil or the handle holds no expert matrix, THEN THE SYSTEM
  SHALL close read-ahead.
- FR-008 (US-003) WHEN a model whose handle holds an expert matrix is built without `prefetch` false
  and `reads.offered` answers true, THE SYSTEM SHALL take a queue of at least `3 * experts_used`
  entries and at least four, with one buffer per entry sized for the largest expert matrix of the
  handle, after the faults `15/FR-009`, `15/FR-015` and `35/FR-009` name, and before the `profile`
  is read, and report its depth and buffer size in `read_ahead`; a model that holds no queue SHALL
  report read-ahead closed and hand nothing over.
- FR-009 (US-004) WHEN `compute.model` is given `prefetch` true, THE SYSTEM SHALL hold read-ahead
  open from the build wherever the model holds a queue and make no decision at the gate; WHEN it is
  given false, THE SYSTEM SHALL hold read-ahead closed, take no queue and hand nothing over.
- FR-010 (US-004) WHEN a config file names `prefetch` a boolean, THE SYSTEM SHALL pass it to the
  start function, to the `bench.Settings` a bench run builds and to the endpoint `bench` and `serve`
  build a model from, so `compute.model` is given it; IF it is set and is not a boolean, THEN THE
  SYSTEM SHALL write one line naming `prefetch` and return exit code 1 without calling a start
  function.
- FR-011 (US-005) WHEN `bench` writes a length's `experts` line, THE SYSTEM SHALL end it with
  `; A ahead, P% landed; read ahead S`, where A is the turn's `measure.prefetch_reads`, P is its
  `measure.prefetch_landed` over its `measure.prefetch_named` times 100 to one decimal place,
  printed `0.0` when `prefetch_named` is 0, and S is `open` or `closed` as the model's
  `read_ahead.open` stands at the turn's end.
- FR-012 (US-002) WHEN a block's third graph is built, THE SYSTEM SHALL build FR-001's naming into
  it on every pass of the model, whether read-ahead is open or closed.
  Not testable: no case can tell whether the naming ran while read-ahead was closed, the counters it
  feeds being ones a shut gate never increments and the graph count being the same either way; a
  reader confirms the nodes in the third graph.
- FR-013 (US-004) WHEN a worker decodes its endpoint json, THE SYSTEM SHALL carry the `prefetch` the
  json holds into the endpoint it builds.
  Not testable: the model lives in a worker process and nothing a worker publishes reports its
  read-ahead; a reader confirms the field in `src/worker.tl`'s `EndpointJson` record and its
  `endpoint_of` table.
- FR-014 (US-003) WHEN a model that holds a queue is closed, or when `compute.model` returns nil
  after taking one, THE SYSTEM SHALL reap the reads the queue holds and give it back as `37/FR-006`
  gives one back, a second close giving nothing further and the weights handle closing none of it, so
  a model that cannot be built leaves no descriptor, mapped ring or buffer behind.
- FR-015 (US-003) IF `reads.open` answers no queue while `reads.offered` answered true, THEN THE
  SYSTEM SHALL hold read-ahead closed, report `read_ahead` with depth and bytes 0, and hand nothing
  over.
  Not testable: a process whose soft `RLIMIT_NOFILE` is spent cannot read the model's metadata, so no
  built model reaches a refused `reads.open`; `37/FR-008` pins the refusal and a reader confirms the
  branch leaves read-ahead closed with no queue.

## Non-goals

- No prediction made by a pass's first or last block, no lookahead of two blocks, and no prediction
  of a shared expert or of anything the router's choice does not already name.
- No change to the read a pass waits on, to the shape of `34`'s record of what a handle has read,
  which FR-004 marks into, to the slots `35` and `36` place, or to the card copy.
- No second knob: the queue's depth and buffer size, the window's length, the two room ratios and
  half the landings are fixed here, and only `prefetch` is a person's.

## Tests

**The routing fixture** is `15/T-001`'s fixture with two full-attention blocks: `block_count` 2,
`full_attention_interval` 1, `context_length` 65, `blk.1.` holding `blk.0.`'s entries, both blocks'
`ffn_gate_inp.weight` holding `{1, 0}` over `{0, 1}`, both blocks' three expert tensors holding 1
for expert 0 and 2 for expert 1, its `token_embd.weight` rows `{1, 0}` and `{3, 4}`. Every
attention tensor is `15/T-001`'s, whose `attn_output.weight` is zeros, so block 1's attention adds
nothing and its router reads the row block 0 leaves, which is the row the prediction scores. A pass of
token 0 routes expert 0 in both blocks of this fixture.

**The flipping fixture** is the routing fixture whose `blk.1.attn_output.weight` holds `{-1000, 0, 0,
0}`, so block 1's attention pushes the row's first number far below zero, the block routes expert 1
where the prediction named expert 0, and the two disagree.

**The two-block bench fixture** is `29`'s fixture with `block_count` 2, `full_attention_interval` 1
and its one block's entries repeated under `blk.1.`, so a turn's prediction names the expert its
following block routes.

Every case that needs read-ahead open, or a queue taken, stubs `prefetch.memory` to answer a figure,
or `prefetch.decide` to answer true, and restores both afterwards; every case that talks to a queue
runs only where `reads.offered` answers true, otherwise skipping with the line `no submission
interface`, as `37`'s cases do. A case that does not ask for a card builds its model with
`compute.load(nil, false)`, as `35`'s CPU cases do.

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-006): `read ahead hands over and moves no score`, proves
  on the routing fixture, with `prefetch.memory` answering 1000000, that a pass of token 0 at
  position 0 on a model built with no `vram` leaves `read_ahead.open` true, adds 3 to
  `measure.prefetch_reads`, 1 to `measure.prefetch_named` and 1 to `measure.prefetch_landed`, adds 3
  to `measure.blob_reads` and no more, the following block's three matrices having landed whole, and
  returns score rows every number of which equals what a second model of the fixture built with
  `prefetch` false returns for the same pass, that second model counting 6 blob reads, 0 ahead
  reads, no named expert and no landing, that a `routing_sink` each model is given collects the same
  records for that pass, one per block and row, each naming the experts that block routed, and that
  each model's pass adds 8 to `measure.graphs`, the two-block count `32/FR-001` gives.
- T-002 (FR-006): `a wrong prediction costs a read and not an answer`, proves on the flipping
  fixture, with `prefetch.memory` answering 1000000, that a pass of token 0 at position 0 adds 3 to
  `measure.prefetch_reads`, names and lands one expert, and returns score rows every number of which
  equals what a second model of the fixture built with `prefetch` false returns for the same pass,
  and that both models' `measure.blob_reads` are equal, the ahead read having named expert 0 while
  the block routed expert 1 and read it on demand.
- T-003 (FR-007): `the gate opens and closes over sixteen passes`, proves on the routing fixture,
  whose expert matrices hold 192 bytes in all, with `prefetch.memory` stubbed, that a model whose
  build saw 192 bytes, a room of 1.0, has `read_ahead.open` true and stays open over 16 passes; that
  its 17th pass, begun with the stub answering 96 bytes, a room of 0.5, is begun with read-ahead
  closed; that the 33rd pass, begun with 116 bytes, a room of 0.6, leaves it closed; that the 49th
  pass, begun with 135 bytes, a room of 0.7, leaves it open; that the 65th pass, begun with the stub
  answering nil, leaves it closed; that `prefetch.decide` given 0.65 and a window that named nothing
  answers true, given 0.6 and a window that landed 1 of 4 answers false, given 0.64 with all landed
  answers the state read-ahead is in, true when it is open and false when it is closed, given 0.7 with
  1 of 4 landed answers false, given 0.6 with all landed answers true
  when read-ahead is open and false when it is closed, and given nil answers false; that a model
  whose build saw room 0.0 and whose `prefetch.decide` is replaced by one that answers true has
  `read_ahead.open` true, that stub having been called once by the build and not again over 16
  passes, and twice once the 17th pass begins; and that a model built with the real `prefetch.memory`
  has the state the real `prefetch.decide` gives for the room the real figure makes, skipping that
  last part with the line `no memory figure` where the real `prefetch.memory` answers nil.
- T-004 (FR-003, FR-006): `read ahead skips what the slot holds`, proves, on a machine whose library
  found a device, gated on `on_device` as `35`'s cases gate, on the routing fixture, that a model
  built with a `vram` of a first model's `resident_bytes` plus its `reserve` plus half its
  `slot_bytes`, being one slot per block, whose block 1 slot holds expert 0, the expert the
  prediction names, makes a pass of token 0 at position 0 with 0 ahead reads, no named expert and no
  landing, and returns score rows every number of which equals what the same model built with that
  same `vram` and `prefetch` false returns.
- T-005 (FR-009, FR-010): `the key forced, refused and reaching the model`, proves on the routing
  fixture and a `prefetch.memory` stub answering 0, so the gate alone would close read-ahead, that a
  model built with `prefetch` true has `read_ahead.open` true and one built with false has it false
  and a `read_ahead` whose depth and bytes are 0, and that a pass over the false model counts no
  ahead read; that a config file naming `prefetch` true reaches the start function with true, and
  that `bench m --config F` on a config naming the two-block bench fixture and `prefetch` true
  writes its `experts 8:` line ending `read ahead open`, so the bench `Settings` carries it; that
  `cli.run` with the settings `{"serve", "m", "--config", F}` on a file whose `prefetch` is true
  calls `start` with an endpoint whose `prefetch` is true and with nil when the file sets none; that
  a file whose `prefetch` is `"true"` and one whose `prefetch` is 1 each return 1 with one stderr
  line holding `prefetch` and call no start function; and that `serve.build` on `18/T-001`'s fixture
  with an endpoint naming `prefetch` true gives a server whose
  `serve.model_of(server).generation.read_ahead.open` is true and one naming false gives false.
- T-006 (FR-011): `bench reports the ahead reads`, proves that `bench m --config F --prefill-tokens
  8 --decode-tokens 8` on a config naming `29`'s fixture unmodified and `prefetch` false returns 0
  with an `experts 8:` line ending `; 0 ahead, 0.0% landed; read ahead closed`, and that the same
  call on a config naming the two-block bench fixture and `prefetch` true writes an `experts 8:`
  line ending `; 3 ahead, 100.0% landed; read ahead open`, so the line's figures are the turn's own.
- T-007 (FR-005): `a short read ahead closes and the demand read answers`, proves on the routing
  fixture whose `blk.1.ffn_down_exps.weight` is cut a few bytes inside its first expert's matrix,
  with `prefetch` true so read-ahead is open, that the pass of token 0 at position 0 returns nil and
  one line naming `blk.1.ffn_down_exps.weight`, that `read_ahead.open` is false afterwards and
  `measure.prefetch_landed` is 0, and that a second pass returns nil and the same line rather than
  computing from bytes that did not land.
- T-008 (FR-008, FR-009, FR-014): `the queue is taken, sized and given back`, proves on the routing
  fixture, with the handle's largest expert matrix `m` bytes and `reads.offered` true, that a model
  built with `prefetch.memory` answering 1000000 has a `read_ahead` whose depth is at least 4 and
  whose bytes are at least `m`, and that its `close` together with the handle's `release` leaves the
  descriptors and the mapped regions the process held before the build, a second `close` changing
  neither; that a four-expert copy of the routing fixture, both blocks' `ffn_gate_inp.weight` holding
  the four rows `{1, 0}`, `{0, 1}`, `{0, 0}` and `{0, 0}`, its `expert_used_count` 3 and its three
  expert tensors holding four experts, has a `read_ahead` whose depth is at least 9; that a model of
  the routing fixture with its six expert tensors removed, built with `prefetch` true, reports
  read-ahead closed with depth and bytes 0 and hands nothing over, no queue being taken where the
  handle holds no expert matrix; and that `compute.model` over the routing fixture given a `profile`
  naming a path that does not exist, built with `prefetch` true, returns nil and leaves the
  descriptors and mapped regions the process held before the call.

## Open questions

- What text ends the `experts` line, and what does the last word print (options: `; A ahead, P%
  landed; read ahead open` with the state word, `; A ahead, P% landed; read ahead S` with the
  landing share again). (decided: `; A ahead, P% landed; read ahead open` with `open` or `closed`)
- Where the gate lives and what a case may replace (options: a `prefetch` module publishing `memory`
  and a pure `decide`, only `memory` published). (decided: `src/prefetch.tl` publishes `memory` and
  the pure `decide`)

## Amendments

- `04/FR-006`: `prefetch` joins the top-level keys a config file may hold.
- `04/FR-007`: a file whose `prefetch` is set and is not a boolean is refused, checked after
  `profile`.
- `04/FR-011`: the file's `prefetch` is passed to `start`, nil when the file sets none.
- `29/FR-005`: the `experts` line ends with `; A ahead, P% landed; read ahead S` as `38/FR-011`
  prints it, and `29/T-001`'s pattern for that line grows with it.
- `29/FR-011`: the endpoint `bench` builds carries the config's `prefetch`, nil when it names none.
- `32/FR-003`: a block's third graph also reads the following block's `post_attention_norm.weight`
  and `ffn_gate_inp.weight`, which `38/FR-001` scores that block's rows with.
- `34/FR-003`: an ahead read counts one in `measure.prefetch_reads` and nothing in `blob_reads`,
  `blob_bytes` or `blob_seconds`, which count only the reads a pass waits on.
- `36/FR-010`: the line's hit text is followed by the ahead text `38/FR-011` appends, so the three
  `experts` endings `36/T-003` expects grow with it.
