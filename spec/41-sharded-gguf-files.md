# 41: Sharded GGUF files

**Status**: pending   **Depends on**: 05, 06, 34, 37

## Problem & context

Flash-Next ships as two GGUF files that state `split.no` 0 and 1 of `split.count` 2 and hold 84 GB
together, and as an MTP head in a third file that is not part of the split. `gguf.read` reads one
file's header and `weights.load` maps one file, so the loader sees only the tensors of whatever file
it is given: a model built from one shard is missing every tensor in the other, and the build fails
after the map.

The change is in `weights.load`, which already owns the mapping and the tensor table. Given a path
whose file states a split count above 1, it finds the siblings from the path's
`-NNNNN-of-MMMMM.gguf` name, reads each sibling's header through `gguf.read` (05), maps each shard
read-only (06) and merges the tensor tables, each tensor remembering the file that states it.
Offsets and `general.alignment` are each shard's own, so a tensor's data begins in its own shard at
its own offset. A file that is not part of the split is loaded the same way and merged into the same
handle, so a tensor does not care which file holds it. Until a config key names such a file, the
tests are that argument's only caller. `gguf.read` stays a one-file reader: every metadata key a
model needs is in the first shard, and nothing here walks a header twice.

Every read of a tensor's bytes takes the tensor's shard and not the handle's one file: `weights.read`
(34) and a handed-over read (37) both use that shard's mapping and descriptor. The `mincore`
residency check the brief names belongs to spec 42, which is not written yet; the per-tensor shard
this spec adds is what such a check reads, and its amendment waits for that spec.

The shards are mapped, never copied, so the real pair costs address space and page cache the way one
mapped file does today.

```teal
local record weights
  record Shard
    path: string
    bytes: integer
    alignment: integer
    data_start: integer
    mapping: integer
    descriptor: integer
  end

  record Tensor
    -- 06's fields, plus the file that states this tensor
    shard: Shard
  end

  record Handle
    -- 06's fields, with `shards` in place of `alignment`, `data_start`,
    -- `mapping` and `descriptor`
    shards: {Shard}
  end

  load: function(
    path: string, info?: gguf.Info, extra?: {string}
  ): Handle | nil, string | nil
end
```

A shard is found by its file name. The first path's name ends `-00001-of-MMMMM.gguf`, where MMMMM is
the `split.count` its metadata states, and the shard whose `split.no` is N is named
`-NNNNN-of-MMMMM.gguf` with NNNNN equal to N + 1: llama.cpp writes 1-based file names beside
0-based `split.no` (github.com/ggml-org/llama.cpp, `tools/gguf-split/gguf-split.cpp`). Each shard
states the same `split.count` and its own `split.no`, and only the first states the metadata a model
build reads; a later shard states its split keys and, when it chooses, its own `general.alignment`.

A shard that is missing, that states a count or an index other than its name's, or whose tensor runs
past its own last byte is refused with one line naming that file. The refusals keep `06`'s order:
the first path's own reader failure, then the split's name and count, then each shard in order, its
alignment before its tensors, and its tensors in file order.

## User stories

- US-001 (P1) As a person with a model split across files, I want one load to find and use every shard, so that the model loads at all.
- US-002 (P1) As the engine author, I want every tensor to point into the file that holds it at that file's own offset, so that mapping and arithmetic do not care which file holds a weight.
- US-003 (P1) As a person with a damaged or incomplete split, I want one line naming the file at fault, so that I can tell a bad shard from a bad loader.
- US-004 (P2) As a person whose model has a file outside the split, I want it merged into the same handle, so that its tensors are just as reachable.
- US-005 (P1) As the engine author, I want release to give back every shard, so that a dropped model leaves nothing mapped or open.

## Functional requirements

- FR-001 (US-001) WHEN `weights.load(path)` is called on a path whose file states a `split.count` M above 1, THE SYSTEM SHALL read the header of each sibling named `-NNNNN-of-MMMMM.gguf` for NNNNN from 00002 to M, in that order, and merge their tensors after the first file's.
- FR-002 (US-004) WHEN `weights.load` is given `extra`, THE SYSTEM SHALL load each extra path by FR-001's and FR-003's rules in the order given, expanding a path whose file states a split count above 1, merge its tensors after those already loaded, and skip a path whose string is the string of a path this load already loaded, its siblings included.
- FR-003 (US-001) WHEN no file the path loads states a split count above 1, THE SYSTEM SHALL map that one file and give each of its tensors its offset from that file's own data start.
- FR-004 (US-002) THE SYSTEM SHALL give the handle one `shards` entry per file loaded, holding that file's path, its size, its alignment as `06/FR-004` computes it and its data start, and give each tensor the `shard` entry of the file that states it.
- FR-005 (US-002) THE SYSTEM SHALL map each shard read-only and privately, with `PROT_READ|MAP_PRIVATE`, and keep that shard's descriptor and the address it was mapped at in its `shards` entry.
- FR-006 (US-003) IF a shard's path cannot be read, or its header does not state `split.no` and `split.count` as whole-number scalars, or its `split.no` is not the index its name gives, or its `split.count` is not the count its name gives, or the first path's name does not end `-00001-of-MMMMM.gguf` with the `split.count` its file states, THEN THE SYSTEM SHALL return nil and one line naming that file, with the count or index it states where it states one, and that line SHALL be `gguf.read`'s own when the reader refused the file.
- FR-007 (US-003) IF a tensor's data would end past its own shard's last byte, THEN THE SYSTEM SHALL return nil and one line naming that shard's path and the tensor.
- FR-008 (US-002) WHEN `weights.read` is called for a tensor, THE SYSTEM SHALL read from that tensor's shard at the tensor's offset in that shard, through that shard's descriptor.
- FR-009 (US-005) WHEN `release` is called and no holder remains, THE SYSTEM SHALL unmap every shard and close every shard's descriptor once, and a second call SHALL return without unmapping or closing again.
- FR-010 (US-001) WHEN `weights.load` is called on the real two-shard 84 GB Flash-Next pair, THE SYSTEM SHALL return a handle.
  Not testable: the pair is not in the repository and is 84 GB; a reader runs `weights.load` on it by hand and watches it return a handle.
- FR-011 (US-005) IF a shard or an extra is refused after an earlier shard was mapped, THEN THE SYSTEM SHALL unmap every shard already mapped and close every descriptor it kept before it returns nil and the line, so a refused load leaves no mapping and no descriptor behind.
- FR-012 (US-003) IF more than one refusal applies to a load, THEN THE SYSTEM SHALL report the first path's own reader failure before a shard's, a shard's name or split-key refusal before its alignment or its tensors, an earlier shard's refusal before a later shard's, and within a shard the order `06/FR-012` gives.

## Non-goals

- No change to `gguf.read`, to the tokenizer and model metadata, which come from the first shard, or to `config` and `serve`, which still name one path: extra files are a caller's argument.
- No `split.tensors.count` check and no merge of the shards' metadata: a tensor missing inside a present shard is caught by the model build, as today.
- No change to the card copy or the placement: `35` copies `data` and `bytes` as it does for one file, and the merged handle offers it the same tensors.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004): `a split and an extra file answer what one file answers`, proves on one set of tensors that a fixture split across two files, the second stating a different `general.alignment`, plus a third non-split file given as an extra, gives the same names, dimensions, kind names, element counts and byte counts as the same tensors in one file, that reading each tensor's bytes at its `data` gives that tensor's own bytes, that each tensor's `shard.path` is the file that states it, that `#handle.shards` is 3 with each entry's `path` one of the three files, its `bytes` that file's length on disk, the second shard's `alignment` the 64 its `general.alignment` states and the other two 32, and each `data_start` a multiple of its entry's `alignment` at or after its own `table_end`, that a load of the first path with that same path passed again as an extra answers a handle of its two shards and their tensors, with no refusal naming a duplicate tensor, and that the one-file fixture alone still loads the same way.
- T-002 (FR-002, FR-006, FR-007, FR-011, FR-012): `a broken split names its file`, proves a missing sibling, a sibling stating `split.count` 3 in a two-file split, a sibling stating `split.count` as a string, a sibling whose first four bytes are not `GGUF`, a sibling stating `split.no` 0 again, a first path named `-00002-of-00002.gguf`, and a shard holding a tensor whose data ends one byte past that shard's last byte each return nil and one line naming the file at fault, with the count, index or tensor in the line where it applies and the reader's own line for the file it refused, that a split whose first shard holds such a tensor and whose second shard does not exist answers the first shard's tensor line rather than the missing second shard's, that a load of a healthy two-file split given an extra path that does not exist answers nil and one line naming that path, and that each of those refusals leaves the process holding the descriptors and mapped regions it held before the call, counted from `/proc/self/fd` and `/proc/self/maps`, so the shards mapped before a later refusal are given back.
- T-003 (FR-008): `a matrix in the second shard is read whole`, proves a two-file fixture whose second file holds a stacked tensor of two matrices with distinct bytes answers `weights.read(handle, tensor, 1)` true and leaves in `handle.buffer` the bytes that file's mapping holds at that matrix's offset, which differ from the first matrix's bytes.
- T-004 (FR-005, FR-009): `release gives back every shard`, proves both shard paths appear in `/proc/self/maps` with read-only permissions after `load`, neither appears after `release`, and a second `release` returns nil with them still absent.

## Open questions

- None.

## Amendments

- `06/FR-003`: the handle also holds `shards`, one entry per file loaded, each tensor also holds `shard`, the entry of the file that states it, and `weights.load` takes a third, optional `extra: {string}` of files to merge; `06`'s non-goal that a split model is loaded one file at a time is removed.
- `06/FR-004`: the alignment and data start are each shard's own, from that shard's `general.alignment` or 32 and its own `table_end`, one pair per `shards` entry, and the handle holds neither.
- `06/FR-005`: a tensor's `data` points into its own shard at that shard's data start plus the tensor's offset.
- `06/FR-007`: `release` unmaps every shard's bytes, not one file's.
- `06/FR-010`: the past-the-end check is against the tensor's own shard and its line names that file, and the duplicate-name check spans every loaded file.
- `06/FR-011`: for a split, a shard that cannot be opened or mapped unmaps and closes every shard the load already mapped before it answers nil.
- `06/FR-013`: each shard's descriptor is kept in its `shards` entry and `release` closes every one.
- `34/FR-001`: the read is at that matrix's offset in the tensor's shard, through that shard's descriptor.
- `34/FR-004`: a closed shard descriptor is refused as a released handle is.
- `34/FR-005`: `release` closes every shard's descriptor once; `34/T-006`'s descriptor count becomes one per shard.
- `37/FR-002`: the position and the descriptor are the tensor's shard's, and a closed shard descriptor is what "released" means; the `Handle.mapping` its Problem & context names is the tensor's shard's mapping.
