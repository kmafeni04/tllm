# 42: Zero-copy from the page cache

**Status**: pending   **Depends on**: 15, 29, 34, 35, 38, 39

## Problem & context

`34` reads a chosen expert's matrix whole with one `pread` into the handle's buffer, but the
graph never reads that buffer: it reads the mapping. The read exists only to fill the page cache
behind the mapping's pointer. On a cold blob that is the fastest way to fetch the bytes. On a warm
blob it copies bytes the kernel already holds. The benchmark behind this change measures warm
`mmap` computing in place at 37.8 GB/s against 11.9 GB/s for warm `pread`, so every warm matrix
pays a copy that buys nothing.

This spec adds one path in front of `34`'s read. A blob here is one matrix, the unit `34` reads.
Before a chosen matrix is read whole, the engine asks `mincore` whether every page the matrix spans
is already in the page cache. When it is, and the path is on, the matrix is left to the graph that
reads the mapping, with no copy into the handle's buffer. When any page is absent, the check fails,
or the path is off, `34`'s read runs exactly as it does today and fills the cache.

The path is on only when the page cache has room for the experts. The room is the figure `38`
computes: `prefetch.memory` over the bytes the handle's expert matrices hold. At `0.55` and above
the path is on; below it, or when the figure is nil, or when the handle holds no expert matrix, it
is off. The gate is settled when the model is built, so a later spec, `43`, can print it at
start-up. This spec publishes it as `model.zero_copy` and counts its use each turn in
`measure.direct_reads`.

The path is checked each time a matrix would be read whole, not once per handle. A matrix taken
from the page cache is not recorded as read, so a later pass asks again and takes `34`'s read if the
pages are gone. Only an actual `34` read still marks a matrix read. The path applies to the demand
read a pass waits on; the reads `38` hands over ahead of time are unchanged, and `blob_reads`
keeps its meaning as the reads that actually happened.

```teal
local record weights
  -- true when every page a matrix spans is in the page cache, false when an
  -- index names no matrix or the check cannot be made
  resident: function(Handle, Tensor, index: integer): boolean
end

local record compute
  record Measure
    -- 39's fields, unchanged
    direct_reads: integer
  end

  record Model
    -- 39's fields, unchanged
    zero_copy: boolean
  end
end
```

`weights.resident` is the check as a value, so a case can give it a page-cache answer without
evicting a file, and `model.zero_copy` is the gate as a value, so a case can read what the build
decided.

## User stories

- US-001 (P1) As a person running a model whose experts are in the page cache, I want the pass to compute from the mapping instead of copying the same bytes again, so that a warm pass is not slowed by a redundant copy.
- US-002 (P1) As the engine author, I want an evicted matrix to take `34`'s read, so that a cold blob is still fetched whole and a pass is never slower than it is today.
- US-003 (P2) As the person benchmarking a change, I want bench to report the matrices taken straight from the mapping, so that I can see which read path a turn used.

## Functional requirements

- FR-001 (US-001) WHEN a pass would read whole a chosen matrix that this handle has not read whole and no expert-cache slot holds, and the model's `zero_copy` is true, and every page the matrix spans is in the page cache, THE SYSTEM SHALL leave that matrix to the graph that reads the mapping without reading it into the handle's buffer, SHALL add one to `measure.direct_reads`, and SHALL add nothing to `measure.blob_reads`, `measure.blob_bytes` or `measure.blob_seconds`.
- FR-002 (US-001, US-002) WHEN a pass would read whole a chosen matrix and the model's `zero_copy` is false or `weights.resident` answers false, THE SYSTEM SHALL read the matrix whole as `34/FR-001` reads it and SHALL add nothing to `measure.direct_reads`.
- FR-003 (US-002) WHEN a matrix is taken by the page-cache path, THE SYSTEM SHALL not record it in the handle's read matrices, so a later pass asks `weights.resident` about it again.
- FR-004 (US-001) WHEN a matrix this handle has already read whole is chosen again, THE SYSTEM SHALL take neither read path and SHALL add nothing to `measure.direct_reads`.
- FR-005 (US-001) WHEN `compute.model` builds a model, THE SYSTEM SHALL set its `zero_copy` true when `prefetch.memory` answers a figure and that figure over the bytes the handle's expert matrices hold is at least 0.55, and false when that figure is below 0.55, when `prefetch.memory` answers nil, or when the handle holds no expert matrix.
- FR-006 (US-001, US-002) WHEN `weights.resident` is asked whether a matrix is resident, THE SYSTEM SHALL answer true only when every page from the page holding the matrix's first byte through the page holding its last byte is in the page cache, SHALL answer false when the index names no matrix, and SHALL not fault any page in.
- FR-007 (US-003) WHEN `bench` writes a length's `experts` line, THE SYSTEM SHALL append after the miss-policy text `39/FR-009` appends `; N direct`, where N is that length's `measure.direct_reads`, and the line SHALL end after it.
- FR-008 (US-001) IF the residency check cannot be made, THEN THE SYSTEM SHALL answer false.
  Not testable: a `mincore` error cannot be forced without unmapping the shard under a live tensor, which the project's memory-safety rule forbids; a reader confirms the error arm answers false.

## Non-goals

- No start-up line naming the path: `43` reports `model.zero_copy` at start-up, and this spec stops at the field, the counter and the bench text.
- No re-decision of the gate over the model's life and no config key: `zero_copy` is what the build decided, and the path has no switch of its own.
- No change to what `38` hands over, to routing, to the card copy or to the graph's arithmetic: the page-cache path removes a copy the graph never read, and a matrix `38` hands over still lands a read.

## Tests

**The resident fixture** is `15/T-001`'s fixture with its three expert tensors holding 1 for the
first expert and 2 for the second, as `38`'s routing fixture does, written in full, so its expert
matrices, which hold 96 bytes in all, are in the page cache. T-001 and T-003 use it with `prefetch`
false, so `38` takes no queue, and with `prefetch.memory` stubbed for the gate.

**The evicted fixture** is the same fixture with its `ffn_up_exps.weight` and
`ffn_gate_exps.weight` enlarged to `{2, 1024, 2}` and its `ffn_down_exps.weight` to `{1024, 2, 2}`,
and its file written only up to its header and then extended to its full length, so every expert
matrix lies in a hole and no expert matrix is resident whole. T-002 uses it with `prefetch` false.

- T-001 (FR-001, FR-005): `a resident blob is computed where it lies`, proves on the resident fixture that with `prefetch.memory` answering 52 a model has `zero_copy` false and a pass of token 0 at position 0 counts 3 blob reads and 0 direct reads, and that with the stub answering 53 a second model over a freshly loaded handle has `zero_copy` true and a pass counts 0 blob reads and 3 direct reads, and that this second pass's `measure.blob_bytes` and `measure.blob_seconds` are 0, and that the two passes return score rows every number of which is equal, so the same greedy token; and proves that a model of the same fixture whose stub answers nil has `zero_copy` false, and that a model of `38`'s routing fixture with its six expert tensors removed, with the stub answering 53, has `zero_copy` false.
- T-002 (FR-002, FR-006): `an evicted blob takes the read path`, proves on the evicted fixture, with `prefetch.memory` answering 49152, that `weights.resident` answers false for each of the three chosen matrices and false for an index the tensor does not hold, that asking it again before any read for one of those matrices answers false, so the check faulted no page in, and that a pass counts 0 direct reads and 3 blob reads.
- T-003 (FR-003, FR-004): `a blob in the page cache is checked again`, proves on the resident fixture, with `prefetch.memory` answering 53, that a first pass counts 3 direct reads and 0 blob reads, that after `weights.resident` is replaced by one answering false a second pass counts 3 more blob reads and no more direct reads and returns the same score rows, and that after `weights.resident` is restored a third pass counts neither a blob read nor a direct read, the handle having read those matrices whole.
- T-004 (FR-007): `bench reports the direct matrices`, proves that `bench m --config F --prefill-tokens 8 --decode-tokens 8` on a config naming `29`'s fixture and `prefetch` false, with `prefetch.memory` answering 96, writes an `experts 8:` line ending `; N direct` with N above 0 and its reads 0, and that the same call with the stub answering 0 writes an `experts 8:` line ending `; 0 direct` with its reads above 0.

## Open questions

- None.

## Amendments

- `34/FR-001`: a chosen matrix whose every page is in the page cache, when the model's `zero_copy` is true, is not read whole; it is left to the graph that reads the mapping, and only every other chosen matrix is read whole.
- `35/FR-006`: the matrices it asks to be read whole are those `34/FR-001` still reads whole after the page-cache exception, so a resident matrix its placement does not mark is not read whole.
- `38/FR-003`: the predicted matrices it hands over still include one the page-cache path would take, so read-ahead does not check residency and its handovers are unchanged.
- `39/FR-005`: the whole-matrix demand read before either miss path is the one `34/FR-001` asks for after the page-cache exception; a resident matrix needs no demand read before it is computed.
