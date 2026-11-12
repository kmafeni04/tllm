# 05: Read a GGUF file's header, metadata and tensor table

**Status**: done   **Depends on**: 01, 02

## Problem & context

Nothing in the project knows the GGUF format. The loader, a `bench` that times a load, and a plugin
that wants `llama.block_count` all need the same thing first: the file's header, its metadata and
its tensor table, read from bytes. None of them can be written until that exists.

This spec adds one module, `tllm.gguf`, exporting `read`. It parses the part of a GGUF file that
comes before the weights and returns it as one record. It reads no weight, maps nothing, holds no
state, and changes no existing file. No command prints it: a command whose only job is to print a
file's metadata is scaffolding that would be deleted the moment the loader exists, so the reader is
a module and the tests are its only caller until serve loads a model.

The format facts below come from ggml's `docs/gguf.md` (github.com/ggml-org/ggml). The file opens
with the four bytes `GGUF`, then a 32-bit version, then two 64-bit counts, tensors first and
metadata pairs second. That many key/value pairs follow, each a string key, a 32-bit type code and a
value. Then one tensor info per tensor: a string name, a 32-bit dimension count, that many 64-bit
dimensions, a 32-bit ggml type code, and a 64-bit offset relative to the start of the tensor data.
A string is a 64-bit length and that many bytes, with no terminator. An array is a 32-bit element
type, a 64-bit element count and that many elements, and arrays nest. Values are little-endian, and
the format gives no way to tell a big-endian file apart, so little-endian is what this reads.

```teal
local record gguf
  record Value
    kind: string
    scalar: number | nil
    text: string | nil
    flag: boolean | nil
    item_kind: string | nil
    items: {Value} | nil
  end

  record Tensor
    name: string
    dims: {integer}
    type_code: integer
    offset: integer
  end

  record Info
    version: integer
    tensor_count: integer
    metadata_count: integer
    metadata: {string: Value}
    tensors: {Tensor}
    parameters: integer
    bytes: integer
    table_end: integer
  end

  read: function(path: string): Info | nil, string | nil
end
```

A `Value` sets the one member its `kind` names and leaves the rest nil, so a caller narrows on
`kind` and reads the member that is there: `scalar` for the ten numeric kinds, `text` for a string,
`flag` for a bool, `items` and `item_kind` for an array. `item_kind` is set even when the array is
empty, because the file states it and the elements cannot. `scalar` is a LuaJIT number, so a 64-bit
value above 2^53 arrives rounded; the reader does not pretend to hold more than a double.

`type_code` stays the raw ggml integer rather than a name. That list grows with every ggml release,
and which tensors a model has is a model family's business, which is a plugin's, not the core's.

`parameters` is the sum over tensors of the product of the dimensions, so it is the number of
weights whatever the quantisation, and `bytes` is the file's size. A tensor with no dimensions
counts as one, as ggml treats a scalar.

Reading only what it needs is the module's reason to exist rather than a detail: `fs.read` returns
the whole file as one string (`types/fs.d.tl`), and a model file is tens of gigabytes, so the
reader opens the file and reads forward, stopping at the end of the tensor table. The test that
comes closest is a fixture whose last byte is the last byte of the last tensor info; that the
reader does not read the tensor data is not observable from its result, so it is stated here and
not asserted by a test.

The bytes come from `io` in Teal's standard library, which declares `open` and the `read` and
`seek` methods on a file, so no dependency is added and nothing here touches `fs` or JSON. The
reader asks for the file's size with one `seek("end")` and returns to the start before reading, so
`bytes` is known before the first pair and every stated length and count is checked against that
size.

The two 64-bit kinds and the two float kinds are decoded through LuaJIT's `ffi`, which copies the
bytes into a buffer and casts them to `int64_t`, `uint64_t`, `float` or `double`, so no IEEE754 or
two's-complement arithmetic is written by hand and a value is rounded only as far as a LuaJIT
number rounds it. `ffi` needs no declaration of this project's own: the pinned `luajit-tl-type`
rock puts `ffi.d.tl` in `target/`, which `tlconfig.lua` already has on the include path.

A failure returns nil and one line naming what is at fault, and writes to no stream, since the
reader has no caller to print for yet. The wording is the caller's to shape except where a
requirement names it; the tests check that the line holds the path, the version, the key and code
at fault, or says the file is not GGUF or is truncated.

Two traps. `io.open` succeeds on a directory on Linux and the read is what fails, so a directory is
refused at the first read rather than at the open: the reader knows only that no byte came back, and
its line does not claim to know why. A stated length or count can be far larger than the file, so
every length and count is measured against the file's size before it is used. The refusals are
ordered: path, then magic, then version, then the header and the tables.

`read(path)` is the only entry point. A loader that memory-maps the weights parses from memory
rather than from a stream, and splitting the parser out for it is that spec's job, taken with a
second caller in hand rather than guessed at now.

The tests build their files byte by byte into a temp directory, because no GGUF file ships with the
repository and one of any real size would not belong in it.

## User stories

- US-001 (P1) As the engine author, I want one module that turns a GGUF file's header, metadata and tensor table into a record, so that the loader and the plugins have one place that knows the format.
- US-002 (P1) As the engine author, I want each metadata value typed as the file states it, so that a plugin can read a setting without guessing what it is.
- US-003 (P1) As a person with a file that is not GGUF or is damaged, I want one line saying what is at fault, so that I can tell a bad file from a bad reader.
- US-004 (P2) As the engine author, I want the file's size and its total parameter count, so that I can report and budget a model before loading it.

## Functional requirements

- FR-001 (US-001) WHEN `require("tllm.gguf").read(path)` is called on a file whose first four bytes are `GGUF` and whose version is 3, THE SYSTEM SHALL return a record whose `version`, `tensor_count` and `metadata_count` hold the values the header states.
- FR-002 (US-001) THE SYSTEM SHALL give that record a `metadata` table holding one entry per key the file states, keyed by the key's bytes as a string, with no entry for a key the file does not state.
- FR-003 (US-002) WHEN a metadata value carries type code 0, 1, 2, 3, 4, 5, 6, 10, 11 or 12, THE SYSTEM SHALL give its entry `kind` `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `float32`, `uint64`, `int64` or `float64` respectively, `scalar` holding the value as a number, and nil for `text`, `flag`, `item_kind` and `items`.
- FR-004 (US-002) WHEN a metadata value carries type code 8, THE SYSTEM SHALL give its entry `kind` `string`, `text` holding the bytes as a string including any zero byte, and nil for `scalar`, `flag`, `item_kind` and `items`.
- FR-005 (US-002) WHEN a metadata value carries type code 7, THE SYSTEM SHALL give its entry `kind` `bool`, `flag` true when the byte is not zero and false when it is, and nil for `scalar`, `text`, `item_kind` and `items`.
- FR-006 (US-002) WHEN a metadata value carries type code 9, THE SYSTEM SHALL give its entry `kind` `array`, `item_kind` naming the element type code as `FR-003`, `FR-004`, `FR-005` and this requirement name it, and `items` holding one entry per element in file order, each decoded by those requirements and this one, and WHEN the array holds no element, THE SYSTEM SHALL still set `item_kind` and give `items` an empty list.
- FR-007 (US-001) THE SYSTEM SHALL give the record a `tensors` list holding one entry per tensor in file order, each holding `name`, `dims` as the dimensions in file order, `type_code` as the code the file states, and `offset` as the offset the file states, and a `table_end` holding the position in the file of the first byte after the last tensor info.
- FR-008 (US-004) THE SYSTEM SHALL set `bytes` to the file's size in bytes and `parameters` to the sum over `tensors` of the product of its `dims`, a tensor whose `dims` holds no entry counting as one.
- FR-009 (US-003) IF the path cannot be opened, or opens but no byte can be read from it, THEN THE SYSTEM SHALL return nil and one line naming the path.
- FR-010 (US-003) IF at least one byte and fewer than four can be read from the path, or the first four bytes are not `GGUF`, THEN THE SYSTEM SHALL return nil and one line naming the path and saying the file is not GGUF.
- FR-011 (US-003) IF the version the header states is not 3, THEN THE SYSTEM SHALL return nil and one line naming the version found.
- FR-012 (US-003) IF the file ends before its header is complete, or before the metadata pairs and tensor infos that header states are complete, or a stated string length, array element count or dimension count runs past the end of the file, THEN THE SYSTEM SHALL return nil and one line naming the path and saying the file is truncated.
- FR-013 (US-003) IF a metadata value's type code, or an array's element type code, is not one of 0 to 12, THEN THE SYSTEM SHALL return nil and one line naming the key and the code.
- FR-014 (US-001) WHEN the file ends immediately after its tensor table, THE SYSTEM SHALL return the record, so that no byte of tensor data is needed.
- FR-015 (US-003) IF an array nests more than 8 levels deep, THEN THE SYSTEM SHALL return nil and one line naming the key.

## Non-goals

- No tensor data: no weight is read, mapped or held, and alignment, the start of the data and the meaning of a type code are the loader's business.
- No model family knowledge: no `general.architecture` dispatch, no per-architecture key, no default for a missing key and no check that a required key is present. The reader reports what the file states.
- No surface and no exact 64-bit integers: no command, route or printed line, no big-endian file, and no value held beyond what a LuaJIT number holds.

## Tests

- T-001 (FR-001, FR-002): `header counts and keys are read`, proves a fixture with version 3, two tensor infos and two metadata pairs returns `version` 3, `tensor_count` 2, `metadata_count` 2, and a `metadata` table whose keys are exactly those two and whose entries decode.
- T-002 (FR-003): `numeric metadata keeps its type and value`, proves one pair of each of the ten numeric type codes returns the matching `kind` and the value as `scalar`, where the fixture writes -1 as int8, -2 as int16, -3 as int32, -1 as int64, 4294967295 as uint32, 1.5 as float32 and -2.25 as float64, and that `text`, `flag`, `item_kind` and `items` are nil on each.
- T-003 (FR-004, FR-005): `string and bool metadata keep their type and value`, proves a string holding an embedded zero byte comes back whole as `text` with `kind` `string`, a bool byte of 1 gives `flag` true, a bool byte of 0 gives `flag` false, and the other members are nil on each.
- T-004 (FR-006): `arrays keep their element type and order`, proves an array of two strings, an array of three int32, an empty float32 array and an array holding an array each decode with the `item_kind` the file states and `items` in file order.
- T-005 (FR-007, FR-014): `the tensor table is read without the tensor data`, proves on a fixture whose last byte is the last byte of its last tensor info that two tensor infos come back in file order with their names, dimensions, type codes and offsets as stated, and that `table_end` equals `bytes`.
- T-006 (FR-008): `size and parameter count are computed`, proves `bytes` equals the fixture's length and `parameters` is 6 for a 2 by 3 tensor, 7 for a 7 tensor, and 1 for a tensor with no dimensions, with the three in one file summing to 14, and that the same three tensor infos followed by 16 bytes of tensor data still give `parameters` 14 and a `bytes` equal to the whole file's length rather than the length of the table.
- T-007 (FR-009): `a path that is not a readable file is refused`, proves a path that does not exist, a directory and an empty file each return nil and one line holding the path.
- T-008 (FR-010): `a file that is not GGUF is refused`, proves a three-byte file and a file whose first four bytes are `GGML` each return nil and one line holding the path and saying the file is not GGUF.
- T-009 (FR-011): `a version other than three is refused`, proves version 2 and version 4 each return nil and one line holding the version found.
- T-010 (FR-012): `a truncated file is refused`, proves a four-byte file holding only `GGUF`, a file cut inside the tensor count, a metadata string whose stated length runs past the end, an array whose stated count runs past the end, a file cut inside the tensor table, and a header stating a metadata count of 1099511627776 each return nil and one line holding the path and saying the file is truncated.
- T-011 (FR-013): `an unknown value type is refused`, proves a pair with type code 13 and an array whose element type code is 99 each return nil and one line holding the key and the code.
- T-012 (FR-015): `an over-nested array is refused`, proves an array nested 9 deep returns nil and one line holding the key, and an array nested 8 deep decodes.

## Open questions

- None.

## Amendments

- None.
