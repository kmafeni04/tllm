# 06: Map a GGUF file's weights and know each ggml type's block shape

**Status**: done   **Depends on**: 01, 05

## Problem & context

`05` turns a GGUF file's header, metadata and tensor table into a record, and stops there: no
weight byte is reachable. Every tensor info says where its bytes are and what type they are, but
nothing turns an offset into a pointer, nothing knows that `general.alignment` shifts the data
start, and nothing in the project can say that 256 elements of `q4_K` occupy 144 bytes. A loader, a
`bench` that times a load and a plugin that wants a tensor by name all need those two things and
have neither.

This spec adds two modules and no command. `tllm.ggml` is the type table: one lookup from a ggml
type code to the type's name, its block element count and its block byte size. `tllm.weights`
opens a GGUF file, maps its bytes once, and returns a handle holding each tensor's pointer, shape
and size. Nothing is dequantised, nothing is copied and no model family is involved.

```teal
local record ggml
  record Type
    code: integer
    name: string
    block_elements: integer
    block_bytes: integer
  end

  type_of: function(code: integer): Type | nil
end

local record weights
  record Tensor
    name: string
    dims: {integer}
    kind: ggml.Type
    elements: integer
    bytes: integer
    data: ffi.CData
  end

  record Handle
    path: string
    alignment: integer
    data_start: integer
    tensors: {string: Tensor}
    release: function(Handle)
  end

  load: function(path: string): Handle | nil, string | nil
end
```

The type table is the first place the core knows a quant format, so it is data, not code: one row
per code, copied from ggml's `src/ggml.c` `type_traits` and the block structs in
`src/ggml-common.h` (github.com/ggml-org/ggml). Codes 4, 5, 31, 32, 33, 36, 37 and 38 are in ggml's
enum with a block size of zero because support was removed from files, so they are not in the table.
The table grows with ggml, which is why an unlisted code is answered with nil rather than refused:
a caller can tell "newer than this engine" from "wrong", and `load` refuses the tensor by name.

```
code  name       block_elements  block_bytes
0     f32        1               4
1     f16        1               2
2     q4_0       32              18
3     q4_1       32              20
6     q5_0       32              22
7     q5_1       32              24
8     q8_0       32              34
9     q8_1       32              36
10    q2_K       256             84
11    q3_K       256             110
12    q4_K       256             144
13    q5_K       256             176
14    q6_K       256             210
15    q8_K       256             292
16    iq2_xxs    256             66
17    iq2_xs     256             74
18    iq3_xxs    256             98
19    iq1_s      256             50
20    iq4_nl     32              18
21    iq3_s      256             110
22    iq2_s      256             82
23    iq4_xs     256             136
24    i8         1               1
25    i16        1               2
26    i32        1               4
27    i64        1               8
28    f64        1               8
29    iq1_m      256             56
30    bf16       1               2
34    tq1_0      256             54
35    tq2_0      256             66
39    mxfp4      32              17
40    nvfp4      64              36
41    q1_0       128             18
42    q2_0       64              18
43    f8_e4m3    1               1
```

The names are ggml's own, mixed case and all, so a line printed from this table matches what
`llama.cpp` prints.

`weights.load(path)` calls `gguf.read` for the tensor table and the file's size, then maps the whole
file read-only through POSIX `mmap` using the `ffi` that `05` already imports. Mapping the whole
file rather than each tensor's region is the only choice that costs one call: `mmap` refuses an
offset that is not a page boundary, and a tensor offset is only aligned to `general.alignment`.
The mapping is the reason the handle exists: a 16-gigabyte model costs no heap, the kernel reads a
tensor's pages when the engine first touches them, and one `release` gives the file back. Nothing
is copied, so the pointer a tensor hands out stays valid until `release` and never after. The
descriptor `mmap` needs is closed before `load` returns, on the failing path as well, so the
handle holds the mapping and nothing else, which is what `US-004` asks: a load leaves nothing
behind.

`05` stops at the end of the tensor table and, as amended by this spec, exposes that position as
`table_end`. The loader needs it to compute `data_start`: the first byte of tensor data is the
table's end rounded up to the next multiple of the alignment the file states, or 32 when it states
none, which is ggml's default.

The refusals are ordered as `05` orders its own: a file the reader refuses fails there, then the
alignment before any tensor, then the tensors in file order. A tensor is refused when its bytes
cannot be read: its type code is not in the table, its first dimension is not a multiple of the
block, its data would run past the file's last byte, or another tensor info already took its name.
A tensor with no dimensions counts as one dimension of 1, as `05` counts it when summing
`parameters`.

Tests build their GGUF fixtures byte by byte into a temp directory, as `05`'s do, and the type
table is checked against the rows above.

## User stories

- US-001 (P1) As the engine author, I want to ask what a ggml type code means, so that the loader and a later forward pass agree on a tensor's shape from the same table.
- US-002 (P1) As the engine author, I want one call that turns a GGUF file's tensor table into pointers and sizes, so that the model's weights are in reach without copying the file.
- US-003 (P1) As a person with a model whose tensors do not fit its header, I want one line naming the tensor at fault, so that I can tell a damaged file from a broken loader.
- US-004 (P2) As the engine author, I want to hand the file's bytes back when a model is dropped, so that a load leaves nothing behind.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL return, for each type code the table in this document lists, a record whose `code` is that code and whose `name`, `block_elements` and `block_bytes` are the name, block element count and block byte size the table states.
- FR-002 (US-001) IF `type_of` is called with a type code the table does not list, THEN THE SYSTEM SHALL return nil.
- FR-003 (US-002) WHEN `weights.load(path)` is called on a file whose header `gguf.read` accepts, THE SYSTEM SHALL return a handle whose `path` is the path passed to `load` and whose `tensors` holds one entry per tensor info in the file keyed by the info's name, each entry holding `name`, `dims` and `kind`, the type table entry for the info's type code.
- FR-004 (US-002) THE SYSTEM SHALL set `alignment` to the scalar of the metadata entry `general.alignment` when the file states that key and to 32 when it does not, and `data_start` to the smallest position at or after the record's `table_end` that is a multiple of `alignment`.
- FR-005 (US-002) THE SYSTEM SHALL give each tensor a `data` field pointing at the file's byte `data_start` plus the tensor's offset bytes into the file.
- FR-006 (US-002) THE SYSTEM SHALL set a tensor's `elements` to the product of its dimensions, a tensor with no dimensions holding 1, and its `bytes` to the tensor's first dimension, or 1 when it has none, divided by the block element count, multiplied by the block byte size and by the product of the remaining dimensions, that product being 1 when there are none.
- FR-007 (US-004) WHEN the handle's `release` is called, THE SYSTEM SHALL unmap the file's bytes so that they no longer appear in the process's mappings, and a second call on the same handle SHALL return without unmapping again.
- FR-008 (US-003) IF `gguf.read` returns a failure for the path, THEN THE SYSTEM SHALL return nil and that line unchanged.
- FR-009 (US-003) IF `general.alignment` is present and its entry has no `scalar`, or its `scalar` is less than 1 or is not a whole number, THEN THE SYSTEM SHALL return nil and one line naming that key.
- FR-010 (US-003) IF a tensor info's type code is not in the type table, or its first dimension, or 1 when it has no dimensions, is not a multiple of that type's block element count, or its data would end past the file's last byte, or two tensor infos share a name, THEN THE SYSTEM SHALL return nil and one line naming the tensor, and the type code when the type code is at fault.
- FR-011 (US-003) IF the file cannot be opened for mapping, or the mapping fails, THEN THE SYSTEM SHALL return nil and one line naming the path, leaving no descriptor open.
  Not testable: forcing `open` or `mmap` to fail needs a descriptor limit the suite cannot set
  portably; the branch returns nil and one line holding the path and closes the descriptor, which a
  reader confirms by inspection.
- FR-012 (US-003) IF more than one refusal applies to a file the reader accepts, THEN THE SYSTEM SHALL report the alignment before a tensor, and the first tensor in file order when several tensors are at fault.
- FR-013 (US-004) WHEN the mapping succeeds, THE SYSTEM SHALL close the descriptor `load` opened for it before returning.

## Non-goals

- No dequantisation and no forward pass: a tensor's bytes stay as stored, no type is decoded to floats, and no model family, layer or plugin is named.
- No bench: `bench` loads no model, and `serve` maps the model through `weights.load` under `08`.
- No copy and no shard: nothing is read into a buffer, nothing is cached, the mapping is read-only, and a model split across several files is loaded one file at a time.

## Tests

- T-001 (FR-001): `every ggml type code keeps its block shape`, proves each of the 36 listed codes gives a record whose `code` is the code and whose `name`, `block_elements` and `block_bytes` are the table's row.
- T-002 (FR-002): `a code with no type has no entry`, proves codes 4, 5, 31, 32, 33, 36, 37, 38, 99 and -1 each give nil.
- T-003 (FR-003, FR-005, FR-006): `tensors come back by name with their bytes`, proves on a fixture holding an f32 tensor of dimensions 2 and 3, a q4_0 tensor of dimensions 64 and 2 and an f32 tensor with no dimensions, with the data written at aligned offsets, that `path` is the fixture's path, that `tensors` holds exactly those three names, that each `dims` and `kind.name` is as the file states, that `elements` and `bytes` are 6 and 24 for the first, 128 and 72 for the second and 1 and 4 for the third, and that reading `bytes` from each `data` gives that tensor's bytes.
- T-004 (FR-004): `alignment and the data start come from the file`, proves a fixture stating `general.alignment` 64 returns `alignment` 64 and a `data_start` at or after the `table_end` that `gguf.read` gives for it, a multiple of 64 and less than that `table_end` plus 64, and that the same tensor table with no `general.alignment` returns `alignment` 32 and a `data_start` at or after its `table_end`, a multiple of 32 and less than that `table_end` plus 32.
- T-005 (FR-007, FR-013): `release unmaps the file`, proves the fixture's path appears in `/proc/self/maps` after `load`, no longer appears after `release`, that a second `release` returns with the path still absent, and that the process holds no more `/proc/self/fd` entries after `load` than before it.
- T-006 (FR-008): `a file the reader refuses is refused with its line`, proves a path that does not exist and a file whose first four bytes are `GGML` each return nil and one line holding the path.
- T-007 (FR-009, FR-012): `an alignment that is not a positive integer is refused`, proves `general.alignment` as a string, as 0 and as 64.5 each return nil and one line holding `general.alignment`, and that a fixture stating `general.alignment` 0 with a tensor of type 4 returns the alignment line, not the tensor line.
- T-008 (FR-010, FR-012): `a tensor that cannot be read is refused`, proves a tensor of type 4, a q4_0 tensor whose first dimension is 48, a tensor whose data ends one byte past the file, and two tensor infos sharing a name each return nil and one line naming the tensor, with the type code in the line for the first, and that a fixture whose first tensor is of type 4 and whose second is a q4_0 tensor of first dimension 48 returns nil and one line naming the first.

## Open questions

- None.

## Amendments

- None.
