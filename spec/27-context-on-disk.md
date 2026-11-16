# 27: A conversation on disk comes back after a restart

**Status**: done   **Depends on**: 04, 14, 15, 18, 19, 23, 24, 26

## Problem & context

`23` keeps conversations in the server's memory, so a restart loses them and the next turn reads
the whole transcript again; `23`'s non-goals said disk was this spec. `26` gave each conversation a
named sequence, so this spec writes that sequence's numbers to one file when a turn stores the
conversation, and restores them when a turn names the store holds nothing under, so a restarted
server continues where the old one stopped. It is also the mechanism `28` needs to move a
conversation between worker processes.

What a conversation's model holds is more than keys and values. `24` holds each full-attention
block's keys and values in host tensors, and `14` holds each linear block's convolution history and
delta state in host float arrays. On the local file the linear blocks dominate: 30 blocks of
`128 * 128 * 32` numbers is 2 MiB a block, about 60 MiB a conversation whatever its length, plus
2.8 MiB of history. The attention numbers add 20 KB per position at `f16` and 40 KB at `f32`
across the ten full-attention blocks, so an 8192-position conversation is about 230 MB at `f16`
and about 400 MB at `f32`. A file that left the linear blocks out would still load and give a
wrong answer, which is why they are in it.

Everything is host memory: `24`'s caches live in host buffers even when `21` offloads a block's
graph to the device, so a save reads no device memory and a load writes none. Both are plain
copies of bytes, so their cost is the file's size at whatever speed the directory's storage gives,
and the suite pins that ratio rather than a time on one machine.

Both calls belong on the model, not in a plugin, for the reason `23` and `24` gave for the cache
itself: a save reads each block's keys and values, history and state out of the buffers the model
allocated, and a load writes them back with the geometry the model built from the file's metadata,
so nothing outside `compute` can name them. Where the files go and how much room they get cannot
leave either, because a plugin has no point at which a turn stores: only `serve` knows, so only
`serve` can trim the directory, and `context` names just those two values.

Saving is automatic: every stored turn writes its conversation, and a turn for a conversation the
server does not hold reads that file back. Each file is named for its session id and ends in
`.context`, so the directory holds nothing the server did not write there. The place and the
budget have defaults, because a feature nobody configures should still work: the directory is
`$XDG_DATA_HOME/tllm/contexts`, or `$HOME/.local/share/tllm/contexts` when the first is absent or
empty and `HOME` holds a value, and the budget is 1 GiB, which is about four 8192-position
conversations at `f16`. `config.json` overrides either with a `context` object, and
`"context": false` keeps everything in memory as `23` did.

A file records the model it came from. A fingerprint of the file's tensor entries and metadata,
taken when the model is built, is what a load compares, along with the cache type and the
conversation's capacity, so a file for another model or another `kv_cache` is refused rather than read as numbers
that belong to another model. A refused or missing file is not a fault the client sees: the turn
reads its whole prompt, as `23` already does when a stored conversation does not continue it, and
the next stored turn overwrites the file.

```teal
local record compute
  record Model
    -- 26's fields, unchanged
    capacity: function(Model, string): integer | nil
    save: function(Model, string, {integer}, string): string | nil
    load: function(Model, string, string): {integer} | nil, string | nil
  end
end

local record serve
  record Endpoint
    -- 18's other fields, unchanged
    context_path: string | nil
    context_bytes: integer | nil
  end

  record Model
    -- 18's other fields, unchanged
    context_path: string | nil
    context_bytes: integer | nil
  end
end
```

`config.load` gains the file's context path and budget beside `kv_cache` in the value it returns,
and `cli` passes them in the endpoint `start` receives, which is what `serve.build` gets. A
`serve.Endpoint` literal that names no context leaves both fields nil, which is the same in-memory
server `23` built, so no existing literal in the suite changes.

`capacity` is the one field beside the two calls, because a turn that restored a file must record
the positions `load` reserved rather than the positions its own need would have asked for: the
capacity is in the file, `load` returns only the ids, and a turn that guessed a larger capacity
would pass the next position past the window it restored.

## User stories

- US-001 (P1) As a chat client, I want my conversation to be there after the server restarts, so that it resumes without the transcript being read again.
- US-002 (P1) As the engine author, I want the file to hold everything a conversation's model holds, so that a restored conversation answers with the numbers it gave before.
- US-003 (P1) As the person running the server, I want to name where contexts go and how much disk they may use, so that they survive without filling the disk.
- US-004 (P2) As the person running the server, I want a session id from a client never to reach a file outside the context directory, so that a request cannot read or write elsewhere.
- US-005 (P2) As the person running the server, I want a save and a load to cost about what their bytes cost, so that a long conversation does not stall a turn.

## Functional requirements

- FR-001 (US-001) WHEN a turn stores a conversation under a `session_id` and the server's endpoint
  names a context directory, THE SYSTEM SHALL write that conversation's file for that id into the
  directory before answering, replacing any file that id already has.
- FR-002 (US-001) WHEN a body's `session_id` is a string, the store holds no conversation under it,
  and a file for it that `load` accepts holds the prompt's first ids, fewer than all of them, THE
  SYSTEM SHALL put the restored conversation in the store under that id, forward only the ids past
  its length at their positions, and take `cached_tokens` as that length when the reply is not
  streamed.
- FR-003 (US-002) WHEN `model:save(sequence, ids, path)` is called for a sequence the model has
  reserved whose every block took its taken positions and whose `ids` are as many as those
  positions, THE SYSTEM SHALL write one file at `path` holding the format's version, the ids, a
  fingerprint the engine takes over the model's tensor entries and their metadata, the numbers'
  cache type, the sequence's capacity and taken positions, each full-attention block's keys and
  values for those positions, and each linear block's history, state and count, and return nil.
- FR-004 (US-002) IF `save` is called for a sequence the model has not reserved, or a block of the
  run took other than its taken positions, or `ids` is not as many as those positions, THEN THE
  SYSTEM SHALL return one line naming `save` and the sequence and write no file.
- FR-005 (US-002) WHEN `model:load(sequence, path)` accepts a file, THE SYSTEM SHALL reserve the
  sequence at the file's capacity, give it the file's taken positions and ids, restore every block's
  numbers and count, and return the ids and nil, so a pass after it returns the numbers the model
  that saved the file returns for the same pass; IF that reservation returns a line, THEN THE SYSTEM
  SHALL return that line and reserve nothing.
- FR-006 (US-002) IF a file cannot be read, is shorter than the leading version, fingerprint, cache
  type, capacity and taken positions it declares, holds a version the engine does not know, was
  written by another model because its fingerprint or cache type differs from the loaded model's, or
  holds a capacity more than the model's `positions`, THEN THE SYSTEM SHALL return nil and one line
  naming `load`, and leave the model as it was.
- FR-007 (US-003) WHEN a config file is read or none is, THE SYSTEM SHALL pass the endpoint the
  directory and budget that `context` names when it is an object holding a `path` string of one or
  more characters, a `bytes` integer of one or more and no other key, `$XDG_DATA_HOME/tllm/contexts`
  and 1073741824 bytes when `context` is absent and `XDG_DATA_HOME` holds a value,
  `$HOME/.local/share/tllm/contexts` and 1073741824 bytes when `context` is absent, `XDG_DATA_HOME`
  is absent or empty and `HOME` holds a value, and nil for both when `context` is `false` or when
  `context` is absent and neither variable holds a value, taking the two variables from the
  environment `config.load` takes; IF `context` is neither an object of that shape nor `false`,
  THEN THE SYSTEM SHALL write one line naming `context` through the stderr writer and return exit
  code 1 without calling a start function.
- FR-008 (US-003) WHEN a conversation's file is written and the files whose names end in `.context`
  would hold more than the endpoint's `context_bytes`, THE SYSTEM SHALL delete those files with the
  oldest modification times until they hold no more than that, counting and deleting no file of any
  other name, a file this server wrote later going after one it wrote earlier when the file system
  gives both the same modification time, which `fs.stat` does to the second; IF the new file alone
  holds more than that, THEN THE SYSTEM SHALL delete the new file and no other; IF a delete fails,
  THEN THE SYSTEM SHALL leave that file in place, change nothing else and leave the reply as FR-011
  leaves one.
- FR-009 (US-004) WHEN a file is named for a session id, THE SYSTEM SHALL place it directly in the
  context directory as that id's escaped text followed by `.context`, so that two different ids get
  two different files whatever text the ids hold.
- FR-010 (US-005) WHEN a conversation is saved to a file and loaded back from it, THE SYSTEM SHALL
  take no more than ten times the wall time `fs.write` takes for a string of that file's byte length
  and no more than ten times the wall time `fs.read` takes for one, each the shortest of five runs
  after one throwaway save and load.
- FR-011 (US-001) WHEN a conversation's write to its file returns a line, THE SYSTEM SHALL answer the
  turn as `23/FR-002` answers it and leave the store holding the conversation the turn stored.
- FR-012 (US-003) WHEN `serve.build` receives an endpoint whose `context_path` is not nil, THE SYSTEM
  SHALL create that directory when it does not exist; IF it cannot create it, THEN THE SYSTEM SHALL
  return nil and one line naming `build` and that path, so `serve.start` writes that line and returns
  1.
- FR-013 (US-001) WHEN a file is accepted and its ids are not the prompt's first ids, or hold all of
  them, THE SYSTEM SHALL release the sequence `load` reserved before `23/FR-003` reserves the turn's
  own, so no reservation outlives the turn that made it.

## Non-goals

- No compatibility with llama.cpp's state files and no format an older or newer tllm reads: the file
  carries a version, and a version the engine does not know is refused.
- No partial save, no save of a live turn and no rewind: the file is the conversation as the turn
  that stored it left it, whole or not at all.
- No conversion: a file for another model, another cache type or a larger window is refused, never
  read as if it were this model's.

## Tests

T-002, T-003 and T-007 build `15/T-002`'s fixture and `24/T-006`'s fixture with `compute.model`
directly; the other cases use `18/T-001`'s fixture and its `build` helper, which gains the two
context arguments, or `04`'s config surface. No number already recorded by another case changes.
"The extended body" below is the body whose `messages` are
`{"role":"user","content":"a"}`, then `{"role":"assistant","content":"a"}` (the text the
fixture's model completes with), then `{"role":"user","content":"a"}`, with `"max_tokens":2`.

- T-001 (FR-001, FR-002, FR-011, FR-012, FR-013): `a restart continues a conversation from its
  file`, proves on `18/T-001`'s fixture built with `sessions` 4, `kv_cache` `f32`, a context
  directory that does not exist yet and `context_bytes` 1000000 that `serve.build` returns a server
  and the directory then exists, that `serve.stream` on
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"s"}`
  answers a producer whose resume followed by `serve.step` until it returns 0 leaves the directory
  holding one file, that the file it holds copied into a second directory leaves that directory
  holding one file of the same byte length, that a server built from the same file, directory and
  budget answers the extended body naming `s` with `OK`, `cached_tokens` 2, `prompt_tokens` 3,
  `completion_tokens` 2 and `total_tokens` 5, that body's text being the text a server built from
  the same file without a context answers for that body after it answered
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2}`; that the same body with
  `"session_id":"t"` answers `cached_tokens` 0 and leaves the directory holding two files, that the
  body with no `session_id` answers `cached_tokens` 0 and leaves two files; that a second server
  built from the same model file with `kv_cache` `f16` over the second directory answers the
  extended body naming `s` with `OK` and `cached_tokens` 0 and leaves that directory holding one
  file whose byte length differs from the one that was copied there, so a file another cache type
  wrote is refused to the client and overwritten by the next stored turn; that `serve.build` with
  `context_path` a path whose parent is a regular file returns nil and one line holding that path;
  and that a server whose context directory was replaced after `serve.build` by a regular file of
  that name answers `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"f"}`
  with `OK` and the usage the same body gets on a server built from that file without a context,
  then the extended body naming `f` with `cached_tokens` 2, so a write that returned a line left
  the reply and the store as `23` answers them; and that a server built over a fourth directory
  answers `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}` with
  `OK`, that a server rebuilt over that directory answers
  `{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"session_id":"s"}` with `OK` and
  `cached_tokens` 0 and then that same body naming `t`, `u` and `v` each with `OK` on a model of 4
  sequences, so the sequence the uncontinued file's load reserved came back instead of starving the
  model.
- T-002 (FR-003, FR-004, FR-005): `a saved conversation gives back the numbers it stopped with`,
  proves on `15/T-002`'s fixture built with `compute.model(backend, handle, 2, "f32", 2)` that a
  model's `reserve("a", 2)` returns nil, that its pass of the row
  `{sequence = "a", token = 2, position = 0}` returns a score row, that `save("a", {5}, path)`
  returns nil and the file holds bytes, that a second model's `load("a", path)` returns `{5}` and
  nil, that its `cache_bytes("a", "blk.0.")` equals the first model's, that the first model and
  then the second each pass the row `{sequence = "a", token = 1, position = 1}`, the two score rows
  being equal number by number within 1e-6, so a file missing the linear block's history or state,
  the attention numbers or the taken count fails the case; that a third model with `reserve("b",
  2)` and `save("b", {}, path2)` returns nil, that a fourth model's `load("b", path2)` returns `{}`
  and nil with `cache_bytes("b", "blk.0.")` equal to the third model's, and that the third model and
  then the fourth each pass `{sequence = "b", token = 2, position = 0}` with equal score rows within
  1e-6, so a conversation with nothing taken keeps its capacity; that the first model's
  `save("zzz", {1}, path3)` returns one line holding `save` and `zzz` and writes no file, that its
  `save("a", {1}, path4)` returns a line and writes no file, the sequence having taken two
  positions, and that on a model over that fixture with `blk.1.attn_v.weight` removed, whose
  `reserve("a", 2)` returns nil and whose pass of one row fails after `blk.0.` took the position,
  `save("a", {}, path5)` returns one line holding `save` and `a` and writes no file, so the only
  fault left is the block that took no position.
- T-003 (FR-006): `a file that is not this model's is refused`, proves on `15/T-002`'s fixture at
  `positions` 2, `"f32"` and 2 sequences, with a conversation of one position saved as in T-002,
  that `load` returns nil and one line holding `load` for each of: a model of the same fixture whose
  `positions` is 1, a model of the same fixture with `kv_cache` `"f16"`, a model of the same fixture
  with `output.weight` holding 2 rows in place of 3, a path holding the first half of the file's
  bytes, a path holding the text `not a context file`, and a path that names no file; that
  `reserve("a", 1)` followed by `load("a", path)` over a file at capacity 2
  returns one line holding `reserve` and `a`, and
  that on a model with `sequences` 1 holding `"b"`, `load("a", path)` returns one line holding
  `reserve` and `a`; and that after the first refusal the model that refused has
  `cache_bytes("a", "blk.0.")` nil and its `reserve("a", 1)` returns nil, so no refusal left a
  sequence held.
- T-004 (FR-007): `the context directory and its budget come from config`, proves on `04`'s config
  surface, every case passing an environment of its own and none reading the real one, for the
  reason `04/FR-001` gives that a test reading the real `~/.config` would write into a person's
  home, that with an environment holding `XDG_DATA_HOME` `X` and `HOME` `H` a file holding
  `{"context":{"path":"D","bytes":4096},"models":{"m":{"path":"P"}}}` calls `start` once with an
  endpoint whose `context_path` is `D` and `context_bytes` 4096, that the same file with an
  environment holding neither variable passes `D` and 4096, so a named path is never dropped, that
  the same file without `context` passes `X/tllm/contexts` and 1073741824, that the same file with
  an environment holding `HOME` `H` and no `XDG_DATA_HOME` passes `H/.local/share/tllm/contexts` and
  1073741824, that the same file with an environment holding neither and no `context` passes nil for
  both, that a file holding `"context":false` passes nil for both, and that a file whose `context`
  is `true`, `1`, `{"bytes":4096}`, `{"path":"","bytes":4096}`, `{"path":"D"}`,
  `{"path":"D","bytes":0}`, `{"path":"D","bytes":"4096"}` or
  `{"path":"D","bytes":4096,"extra":1}` each returns 1 with one stderr line holding `context` and
  calls no start function, and that a file holding both `"sessions":0` and a bad `context` writes
  one line holding `sessions`.
- T-005 (FR-008): `the files stay inside the configured budget`, proves on `18/T-001`'s fixture,
  whose `sessions` is 4, every server built with `kv_cache` `f32`, with the bodies
  `{"messages":[{"role":"user","content":"<130 copies of
  a>"}],"max_tokens":2,"session_id":"s1"}`, that body holding `<120 copies of a>` and `s2` and that
  body holding `<110 copies of a>` and `s3`, that a server built with a first directory and
  `context_bytes` 10000000 answers the `s1` body and leaves that directory holding one file, whose
  size `fs.stat` gives, that a server built with a second directory and that size as its
  `context_bytes` answers the `s1`, `s2` and `s3` bodies in that order, each with `OK`, and after
  each leaves the second directory holding one file whose size is no more than that, that a server
  built over the second directory answers the body whose `messages` are `<110 copies of a>`,
  `{"role":"assistant","content":"a"}` and `{"role":"user","content":"a"}` naming `s3` with
  `cached_tokens` 111, and then the same shape of body holding `<130 copies of a>`, `{"role":
  "assistant","content":"a"}` and `{"role":"user","content":"a"}` naming `s1` with `cached_tokens`
  0, which would be 131 were its file still there, so the oldest file is the one that went and not
  the newest, and that a server built with a third directory and `context_bytes` 1 answers the `s1`
  body with `OK` and leaves that directory holding no file. FR-008's failed delete is not asserted:
  a delete that fails while a write to the same directory succeeded needs a permission or state
  change between the two inside one turn, which the suite cannot make.
- T-006 (FR-002, FR-009): `a session id cannot name a file outside the directory`, proves on
  `18/T-001`'s fixture with a context directory and `context_bytes` 10000000 that the body
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"../escaped"}`, then
  that body with each of `"x/y"`, `"x_y"`, `"a/b"`, `"a%2Fb"` and `"."` answers each `OK` and
  leaves the directory holding six files, each whose name ends in `.context`, that the directory's
  parent holds no file named `escaped.context` and the directory itself holds no directory, and
  that a server built from the same file and directory answers the extended body naming
  `../escaped` with `cached_tokens` 2, the extended body naming `x/y` with `cached_tokens` 2 and
  the extended body naming `x_y` with `cached_tokens` 2, so two ids one character apart keep their
  own files and neither reads the other's.
- T-007 (FR-010): `a save and a load cost about what their bytes cost`, proves on `24/T-006`'s
  fixture with `context_length` 1056, built by `compute.model(backend, handle, 1056, "f32", 2)`,
  that after `reserve("a", 1056)` and the passes of the rows `{sequence = "a", token = 0,
  position = p}` for each `p` from 0 to 1023, `save("a", 1024 ids, path)` returns nil and the
  file's byte length is what `fs.stat` gives, that the shortest of five saves of that conversation
  is no more than ten times the shortest of five `fs.write` calls of a string of that byte length
  to a file in the same directory, and that the shortest of five `load("a", path)` calls, each
  followed by `release("a")`, is no more than ten times the shortest of five `fs.read` calls of
  that path, the timings being the monotonic wall times `24/T-004` takes, after one throwaway save
  and load have been run.

## Open questions

- None.

## Amendments

- None.
