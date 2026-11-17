# 28: Serve from worker processes

**Status**: done   **Depends on**: 04, 18, 21, 23, 25, 26, 27

## Problem & context

`25` and `26` keep every conversation in one process: `25` gives each live turn a step per
round, and `26` puts several conversations into one pass so each weight matrix is read once. Both
buy fairness, and `26` buys throughput per weight read, but the maths still runs on one core and
one turn's slow work holds every other turn. A person watching two agents waits for the second
while the first decodes, whatever the coroutines do.

This spec lets the person running the server move the maths into worker processes: one child
process per worker slot, each running the server `18`, `23`, `26` and `27` already build, with its
own model, its own store and its own caches. The parent keeps the listening socket, holds the queue
and never computes. A conversation is pinned to the worker that served it, so a follow-up turn
lands where the conversation already sits and nothing moves in the normal case; when a worker goes
away, `27` has already written each stored conversation to a file and any other worker restores it,
so a conversation comes back without a new mechanism.

Two things about the weights decide the cost. `weights.load` maps the `.gguf` with `mmap` and
`MAP_PRIVATE` over read-only pages, so on the CPU every worker maps the same file and the operating
system keeps one copy of those pages; what each worker adds is its own caches, which `26` already
sizes per conversation. A card is different: `21` copies a handle's tensors into device memory once
per process, and ggml offers no way to hand a device buffer to another process, so every worker
makes its own copy of the weights and no arrangement of this spec loads them once. That copy is
usually the binding limit, which is why the worker count is the configured number and why a pool
whose copies cannot fit the card is refused at start with one sentence instead of failing inside a
worker later.

The tools decide the shape of the parent's side. lde's `process` library spawns a child, polls it
and kills it, gives it one input string on its standard input at spawn, and hands back the child's
output only once it has exited (`Child:wait` in `src/init.lua` of `lde-org/process`). Two links are
possible: a socket from a registry or luarocks library, or files in a scratch directory, a job
file the worker takes, an output file the parent reads as it grows, a tick the parent writes to
give the worker one step and a done file that ends the turn. This spec takes the files, so the
pool's only new dependency is lde's `process` (the `fs` it also uses is already carried), at the
cost of a poll interval of a few milliseconds per payload. The parent reads only the bytes the
worker has written past the offset it has already read, so a streamed reply costs the bytes it
carries and not the whole file again per step. It is also the shape lde's own
`childPool` uses to hand work to children
(`packages/lde-core/src/util/childpool.lua` in the lde repository).

Neither the tick nor the watcher wakes on a clock, so a parent waiting for a reply and a worker
waiting for a tick would each spin a core. Lua's standard library has no sleep and `fs`'s watcher
only wakes on a change, so both sides wait through `src/pause.tl`, a few lines over `ffi` that ask
the system for a bounded pause; that module is the only place in the pool that reaches `ffi`.

It also settles a question no earlier spec settles: whether one machine can have the ggml backend
open in more than one process at once. The pool's own first case answers it, because two workers
each load the library, map the file and answer.

The turn itself does not change. The parent reads two fields of a body, `stream` and `session_id`,
and hands the body to a worker; the worker answers exactly as it answers today, so its status
where the client is given one, its body and its payloads reach the client unchanged. Every rule
about messages, templates, ids, budgets, `sessions` and `TooManyRequests` stays where it is, in the
worker, with the cases that already pin it. A reply that is not streamed holds the parent until it
arrives, as it does today. A body that asks for a stream is answered with `Server:sse` before a
worker has judged it, so a refusal that worker gives reaches that stream as one payload holding
its error body rather than as a JSON status; a body that asks for no stream always carries the
worker's status and body.

The worker takes one step per round the parent gives it. The parent writes a tick into the worker's
directory once it has read the payloads the last tick produced, and the worker steps the turn once
for that tick, so a streamed turn advances at the parent's pace and a worker killed between two
ticks loses a turn that has not finished.

The pool belongs in `serve` and nowhere else. A plugin never sees a live connection, the listening
socket or the queue a request waits in, and the model a conversation's keys and values live in is
`compute`'s, so a plugin has no point at which it could decide which worker a body goes to or hold a
request until one is free. What a plugin could carry is a different transport for the parent's
side, and this spec keeps the file link in the core because it is the one the rest of the pool's
tests use.

Each worker is built with the endpoint's model, `sessions` and context as they stand, so `sessions`
counts the conversations one worker keeps and the pool's total is `workers` times that, in host
caches and on the card. Whether a queue of fresh prompts is served better by this or by `26`'s
shared pass is a measurement, and this spec asserts no throughput.

```teal
local record serve
  record Endpoint
    -- 27's fields, unchanged
    workers: integer | nil
  end

  record Worker
    slot: integer
    pid: integer
    sessions: {string}
  end

  record Pool
    dir: string
    workers: {Worker}
  end

  -- nil for a server whose turns run in this process, as 27's server does
  pool_of: function(tltp.Server): Pool | nil
  -- the shape of process.spawn, so a caller can start the workers itself
  build: function(
    endpoint: Endpoint, spawn?: process.Spawn
  ): tltp.Server | nil, string | nil
  stop: function(tltp.Server)
  -- 26's step, unchanged: one round of the pool's work, returning the live turns
  step: function(tltp.Server): integer
end
```

`workers` is nil in every `Endpoint` literal written before this spec, and nil is one worker, so no
existing literal changes. `stop` is what `serve.start` calls when its loop ends, and what a caller
uses to end a pool it built. `process.Child` is the handle `process.spawn` gives: a `pid`, a
`poll` returning the exit code once the child is gone, a `wait` and a `kill`. `process.Spawn` is
declared beside `process.Child` in `types/process.d.tl`: Teal does not carry a type alias written
inside a record to another module, so the shape `process.spawn` already has lives with `process`
rather than in `serve`.

## User stories

- US-001 (P1) As the person running the server, I want each conversation's maths to run in a
  process of its own, so that two conversations get their tokens at the same time and a worker
  going away takes one reply down, not the server.
- US-002 (P1) As a chat client, I want my streamed reply to keep arriving and my session to keep its
  place, so that a busy server or a lost worker does not lose my conversation.
- US-003 (P1) As the person running the server, I want the worker count in `config.json` and no
  worker processes unless I ask, so that a plain setup behaves exactly as it does today.
- US-004 (P2) As the person running the server, I want startup to refuse when the workers' copies
  cannot fit in the card, so that the failure is one sentence at boot.
- US-005 (P2) As the person running the server, I want the server to stop every worker it started,
  so that stopping it leaves no process and no scratch directory behind.

## Functional requirements

- FR-001 (US-003) WHEN a config file is read or none is, THE SYSTEM SHALL pass the endpoint the
  count `workers` names when it is a whole number of one or more, and 1 when the key is absent; IF
  `workers` is present and is not a whole number of one or more, THEN THE SYSTEM SHALL write one
  line naming `workers` through the stderr writer and return exit code 1 without calling a start
  function.
- FR-002 (US-003) WHEN `serve.build` is given an endpoint whose `workers` is nil or 1, THE SYSTEM
  SHALL build the server `27` builds, with the model in that process, no worker process and
  `pool_of` nil.
- FR-003 (US-001) WHEN `serve.build` is given an endpoint whose `workers` is more than one, THE
  SYSTEM SHALL start that many worker processes, each started by the spawn function as
  `spawn("lde", {"run", "src/worker.tl", dir, tostring(slot)}, {stdin = endpoint_json, stdout =
  "pipe", stderr = "pipe"})`, `process.spawn` when none is given in its place, `dir` being the
  directory the pool makes and `slot` the worker's own, `endpoint_json` the endpoint's fields with
  `workers` nil as JSON, each server built with `workers` nil and bound to no socket, SHALL return
  only after every worker has reported its server built, and SHALL return a server whose
  `pool_of` gives one worker per process with its slot, its pid and the sessions it serves, whose
  `model_of` and `session_of` give nil, and after which the calling process holds no mapping of
  the model file; IF that endpoint's `context_path` is nil, THEN THE SYSTEM SHALL return nil and
  one line naming `workers` and `context` and start no worker.
- FR-004 (US-001) WHEN a request reaches a server whose `pool_of` is not nil, THE SYSTEM SHALL hand
  its body to a worker that has no turn in flight and has reported its server built, and forward
  that worker's reply: a body that asks for no stream reaches the client with the status and the
  body the worker gave, a body that asks for a stream gets the worker's payloads at the request's
  emit in the order the worker wrote them, and a refusal the worker gives for such a body reaches
  that stream as one payload holding its error body, keeping `18/FR-001`'s routes on that server,
  so `/health` answers `OK` with the endpoint's model name and any other path answers `18`'s
  `NotFound` body, whatever the pool's workers are doing.
- FR-005 (US-002) WHEN a request's `session_id` is a string and a running worker's sessions hold
  that id, THE SYSTEM SHALL hand the request to that worker, holding it until that worker has no
  turn in flight as FR-007 holds a request, and SHALL NOT hand it to another worker while that
  worker runs.
- FR-006 (US-002) WHEN a worker serves a request whose `session_id` is a string, THE SYSTEM SHALL
  hold that id in that worker's sessions, dropping the id it served longest ago first when the list
  already holds as many ids as the endpoint's `sessions`, so no worker's list grows past that
  number.
- FR-007 (US-002) WHEN a request reaches a pool whose workers all have a turn in flight, THE SYSTEM
  SHALL hold the request until a worker has none, give no worker a second turn while it has one,
  emit nothing for a held request that is streamed, and keep forwarding the payloads of the turns
  already live while a request is held, except while a reply that is not streamed has a worker
  answering it, which holds the parent as it does today.
- FR-008 (US-001) WHEN a worker's process exits, THE SYSTEM SHALL end the turn that worker had in
  flight writing no further payload for it, answer a request it was serving without streaming with
  `18`'s `InternalServerError` body, whose message names that slot, drop it from `pool_of` and start
  a replacement in its slot through the same spawn function, giving the replacement no body until
  it has reported its server built, so a later request for a session it had served is served by
  another worker which restores the conversation as `27/FR-002` restores one the server does not
  hold; IF the replacement cannot be started, THEN THE SYSTEM SHALL leave that slot with no worker,
  so `pool_of` gives one worker fewer, start no further replacement for it and write nothing, that
  loss being what `pool_of` shows. A replacement that reports its server built is a worker like any
  other: its own death starts a replacement in its slot, so a pool that loses one worker per slot
  over its life does not shrink one slot at a time.
- FR-014 (US-002) IF a held turn cannot be admitted when a worker frees, THEN THE SYSTEM SHALL end
  its stream with `19/FR-010`'s `error` and `[DONE]` events, so the wait ends with a reply rather
  than with a stream that stops mid-frame.
  Not testable: reachable through `19/FR-010`'s narrow branch only; a reader confirms `abandoned` in
  `src/serve.tl`.
- FR-009 (US-005) WHEN `serve.stop(server)` is called, THE SYSTEM SHALL stop every worker that
  server started and remove the directory it made for them, so `pool_of` gives nil afterwards and
  no worker process outlives the call, and for a server that runs the model itself SHALL also give
  back what `serve.build` took: the model's caches, its hold on the mapping, the mapping itself and
  the library the backend opened, forgetting the server's model, store, turns, reservations and
  held names, so a server built and stopped leaves nothing behind; `serve.start` SHALL call it when
  the loop it runs ends, and SHALL take an optional third argument, the spawn function it passes to
  `serve.build`.
- FR-013 (US-005) WHEN `serve.build` makes the pool's scratch directory, THE SYSTEM SHALL make it
  with a name this call chose, an atomic create, so a directory another user had already made at
  that path is not adopted, and the mode 0700, so only the process that made it can list or read
  the request bodies and replies the workers keep there; IF the create or the mode fails, THEN THE
  SYSTEM SHALL remove what it made and return nil and one line naming the directory, rather than
  letting the workers write into someone else's path or a world-readable one.
- FR-010 (US-005) IF a worker cannot build its server, or its process cannot be started, THEN THE
  SYSTEM SHALL return nil and one line naming that worker's slot as `slot` and the number, and the
  fault, after stopping every worker already started and removing the directory, so a pool either
  starts whole or leaves nothing behind.
- FR-011 (US-004) WHEN `serve.build` builds a pool and the backend `compute.load` opens finds a
  device, THE SYSTEM SHALL compare that device's free bytes from `memory` with `workers` copies of
  the sum of the model file's tensor byte lengths, SHALL drop that backend before it returns
  whether or not the copies fit, and IF the free bytes are fewer, THEN THE SYSTEM SHALL return nil
  and one line naming the number of workers and the bytes the device has free, and start no
  worker.
- FR-012 (US-003) THE SYSTEM SHALL declare lde's `process` in `lde.json` at its registry version
  and declare its shape in `types/process.d.tl`, so that starting, polling and stopping a worker go
  through it and nothing in the pool runs `os.execute`; the only `ffi` the pool reaches is the
  bounded pause in `src/pause.tl`, which the parent's wait and the worker's wait call instead of
  spinning.
  Not testable: a dependency and a type declaration, visible in `lde.json`, `lde.lock` and the
  tree.

## Non-goals

- No single copy of the weights across processes and no measurement: every worker loads the model
  itself, and nothing here says whether this or `26`'s shared pass serves a prompt queue faster.
- No returning the probing process's own device memory to the card: `compute.unload` forgets the
  backend, and the ggml library it opened and any device context it created stay until the process
  exits. `FR-011` compares the device's free bytes and drops the backend; it does not reclaim them.
- No moving a conversation while its worker lives, and no rebalancing: a session moves only because
  its worker went away.
- No cleanup after a parent killed uncleanly: a worker left behind by a `SIGKILL` of its parent is
  the operator's to stop.

## Tests

T-001, T-002, T-003, T-005 and the first half of T-006 build `18/T-001`'s fixture through the serve
cases' `build` helper with the endpoint each case names; T-006's second half, T-008, T-009 and
T-010 name that fixture's endpoint directly because each needs its own spawn function, its own
failure line or its own directory.
Every case takes "the in-process server" to be a server built from the same bytes by `27`'s path,
with `workers` absent. Every case that builds a pool gives that endpoint a context directory and a
budget. A case that drives a pool repeats `serve.step` with a 2 ms wait until it returns 0, up to
5000 tries, and fails when the tries run out. `serve.stop` is called at the end of every case that
built a pool. FR-008's non-streamed branch is not asserted: a case cannot kill a worker while
`serve.answer` blocks inside it.

- T-001 (FR-002, FR-003, FR-004): `two workers answer at once and the parent holds no model`,
  proves that the helper's endpoint built with `workers` absent gives a server whose `pool_of` is
  nil and whose `model_of` is a model of the fixture's path, that `pool_of` stays nil after it
  answers the body `{"messages":[{"role":"user","content":"a"}],"max_tokens":2}` with `OK` and
  content `aa`; that `serve.build` on a copy of that fixture with `workers` 2 returns a server
  whose `model_of` and `session_of(server, "s")` are nil, whose `pool_of` holds two workers with
  different slots, different pids and no sessions, and after which `/proc/self/maps` holds no
  occurrence of that copy's path; that the two streamed bodies
  `{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"s"}` and
  the same body with `session_id` `t` are each resumed with a collector of their own and the steps
  run to 0, after which each collector's payloads are the payloads the in-process server's
  producer for that body emits, and each worker's `sessions` hold one of `s` and `t`; that the
  body naming `s` without `stream` then answers the same status and the same body as the
  in-process server answers for it; and that `/health` on that server answers `OK` and the body
  `18` gives, and a path other than the two answers `NotFound`, while a streamed turn is live.
- T-002 (FR-005, FR-006): `a session goes back to the worker that served it`, proves on a pool
  built as T-001's is that the streamed body naming `s`, run to its end, leaves a worker whose pid
  it records and
  whose `sessions` hold `s`; that the streamed body naming `t`, run to its end, leaves the first
  worker's `sessions` still holding `s`, so a worker holds every session it served, up to the
  endpoint's `sessions`, and not only the session it served last; that the body naming `s` and the
  body holding `a`, an assistant `a` and `a` with `"session_id":"s"` are then resumed with a
  collector each, and that after the step that dispatches the first of them the second's collector
  is empty while the other worker has no turn in flight, so a session waits for its own worker; that the steps run to 0 leave that collector
  holding the payloads the in-process server's producer for that body emits, both turns served by
  the recorded pid, and a streamed body with no `session_id`, run to its end, leaving the recorded
  worker's `sessions` still holding `s`; and that on a second pool of the same fixture built with
  `workers` 2 and `sessions` 2, six streamed bodies naming six different sessions, run to their
  ends, leave every worker's `sessions` holding no more than two ids.
- T-003 (FR-007): `a third request waits for a free worker`, proves on a pool built with `workers`
  2 that the streamed bodies naming `s`, `t` and `u` are each resumed with a collector, that the
  steps run while `s`'s and `t`'s collectors are short of `[DONE]` leave `u`'s collector empty, and
  that the steps run to 0 after them give `u`'s collector the payloads the in-process server's
  producer for that body emits.
- T-004 (FR-001): `the worker count comes from config`, proves on `04`'s config surface, each case
  passing an environment of its own and none reading the real one, that a file holding
  `{"workers":3,"models":{"m":{"path":"P"}}}` calls `start` once with an endpoint whose `workers`
  is 3, that the same file without `workers` calls it once with `workers` 1, and that a file whose
  `workers` is `0`, `-1`, `1.5`, `"2"` or `true` each returns 1 with one stderr line holding
  `workers` and calls no start function.
- T-005 (FR-008): `a worker that dies loses one turn and keeps the session`, proves on a pool built
  with `workers` 2 over a context directory and a budget that the streamed body naming `s` run to
  its end stores its conversation; that the streamed body holding `a`, an assistant `a` and `a`
  naming `s`, resumed and stepped until its collector holds the `role` event and one piece, then
  leaves the worker whose `sessions` hold `s`, whose pid the case kills with `SIGKILL`; that the
  steps run to 0 after that add no payload to that turn's collector; that `pool_of` then holds two
  workers neither of whose pids is the killed one and no worker whose `sessions` hold `s`; that the
  same three-message body naming `s` then answers `OK` with `cached_tokens` 2 and the text the
  in-process server gives for it, so the conversation came back from its file on another worker;
  that the worker that answered `s` the second time is killed, and the steps run until `pool_of`
  holds two workers again, none of them the killed pid, so a replacement that built its server earns
  a replacement of its own; and that, on a second pool of a fresh copy built with `workers` 2 over a
  context directory and a budget, after the case replaces that copy's file with text that is not a
  GGUF and kills one of its workers, `pool_of` holds one worker whose pid is the same a hundred
  steps later.
- T-006 (FR-009): `a stop and a loop that ends leave nothing behind`, proves on a pool built as
  T-001's is that the directory `pool_of(server).dir` names exists, and that after `serve.stop(server)` it does
  not, each pid `pool_of` gave before the stop has no `/proc` entry and `pool_of(server)` is nil;
  and that a pool started through `serve.start` with a `write_err` that records its line and, as
  its third argument, a spawn function that records every pid and the directory it was given, on an
  endpoint whose `host` is `192.0.2.1`, which no interface here holds, returns 1 with one line
  holding `BIND`, and leaves every recorded pid without a `/proc` entry and that directory gone.
- T-007 (FR-003, FR-010): `a pool that cannot start leaves nothing behind`, proves that
  `serve.build` on a copy of the fixture with `workers` 2 and no context path, given a spawn
  function that records every call it takes, returns nil and one line holding `workers` and
  `context`, that function having been called none; that `serve.build` on a copy whose
  `embedding_length` is missing, with `workers` 2 and a context directory, given a spawn function
  that calls `process.spawn` and records every pid it returns, returns nil and one line holding
  `slot 0` and `build`, and that every recorded pid has no `/proc` entry afterwards; and that the
  same build given a spawn function that returns nil returns nil and one line holding `slot 0` and
  the line that spawn function gave.
- T-008 (FR-011): `a pool the card cannot hold is refused at start`, proves, on a machine where
  `compute.load()` gives a backend whose `memory` returns free bytes `f` and a total, so the
  library found a device, that `serve.build` on a copy of the fixture whose `workers` is
  `floor(f / t) + 1`, where `t` is that file's tensor byte sum, returns nil and one line holding
  that number and a byte count no larger than `f`, with no worker process started; on a machine
  where `memory` gives nil and nil it skips with the line `no CUDA device`, as `21/T-001` does.

- T-009 (FR-009): `a gguf is mapped at build and a bad file is refused`, proves over a server that
  runs the model itself that its file's path appears in `/proc/self/maps` after `serve.build`, and
  that after `serve.stop` the server holds no model and the path is gone from the process's
  mappings.
- T-010 (FR-013): `a pool's scratch directory is private and made by this call`, proves that the
  directory `pool_of(server).dir` names has the mode 0700 after `serve.build`, so `fs.stat` reports
  no group or other permission, and that a second pool built in the same second gets a different
  directory.

## Open questions

- None.

## Amendments

- None.
