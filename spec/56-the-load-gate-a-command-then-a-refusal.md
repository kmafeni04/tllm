# 56: The load gate: a command first, then a refusal when the card is busy

**Status**: pending   **Depends on**: 04, 18, 21, 28, 29, 35, 53, 55

## Problem & context

On this machine the card is held by another program, the strata gateway, from the moment the desktop
comes up. So the first thing that happens when this engine is started for a real run of a real file
is an out-of-memory failure part way through the load, after the mapping is open and after the tier
has copied what it could, and that failure costs the whole run: the person gives the card back by
hand, starts again, and pays the load twice. There is today no place in the config that says "run
this command to free the card before you load", and no place that says "do not start unless the card
has room". Both are missing at the same point, and both are needed for the same reason, so this spec
puts them at that point as one rule with two halves.

The other engine on this machine already ships both. Strata has a `min_free_vram_mib` launch key and
ships 503 with it, the number its own gateway leaves behind, and it refuses rather than loading into
a card somebody else is using. What it cannot do is tell another program to hand the card over; that
is a command, and the command that matters here is the gateway's own unload call, which answers in
seconds and leaves the gateway running:
`curl -sS -X POST -H 'Content-Type: application/json' -d '{}' 'http://127.0.0.1:11100/v1/unload?wait=120'`.

Both halves sit at one point: the head of `serve.build`, which every model load in this tree already
passes through, whether it is `serve`, a pool's parent, `bench`, `reference` or a worker. That is
after `--port`, `--host` and `--threads` are refused and before the context directory is created, the
file's metadata is read, the mapping is opened and any weight is placed, so a refusal costs a line
and an exit code and nothing else. A worker is the one load that does not gate: the two values stay
in the process that starts the run, so N workers neither run the command N times nor each refuse on
the card the worker before them took.

The order inside that point is fixed. The command runs first and is waited on, whatever the free
number turns out to be, because a run that is refused wants the card back anyway (it is the card
owner's, not this process's) and because a command that failed is a different line than a card that
is still busy. Then the engine opens a backend of its own and reads what the card has free, the same
number `21/FR-008` answers and `35/FR-004` budgets by, forgets that backend, and below the configured
threshold refuses with the number that would have worked. With no card there is nothing to leave room
for, so there is no gate; the command still runs.

The two config keys are machine keys and belong to the file's top level, not to a `models` entry:
what another program holds is a fact about the machine, while what a model needs is already
`35`'s per-model `vram`. The two numbers are not the same kind of number and never reach each other:
`35`'s `vram` is a cap in bytes on what that model's tier may take, and `min_free_vram_mib` is a
floor in mebibytes on what the card must hold before any tier is asked for anything. A value read for
a run is printed with the run: what command ran, and what the card had, on one line before
`35/FR-010`'s placement line.

```teal
local record serve
  record Endpoint
    -- the endpoint's fields so far, unchanged
    before_load: string | nil
    min_free_vram_mib: integer | nil
  end
end
```

The `Config` record in `src/config.tl`, the `Settings` record in `src/bench.tl` and the `Settings`
record in `src/reference.tl` gain those same two fields, `before_load` as `string | nil` and
`min_free_vram_mib` as `integer | nil`. The command is the `process.exec` of `/bin/sh` with `-c` and
that text, its `stdout` and `stderr` inherited and its first answer the exit status, waited on: that
is what puts the person's shell quoting and the child's own reason for failing on the terminal, where
the `curl` above needs its quoting and its `?wait=120`, and what leaves no status at all for a child
the kernel ended, which `os.execute` reports as a number instead. `serve.build` has no backend where
the gate sits, the mapping being opened before `compute.load` today, so the gate opens a library
handle of its own to read the free number, then forgets it with `compute.unload` before the load
continues, the way `pool_build` already does: that call drops the backend entry, and as `28` records,
the library and any device context it made stay until the process exits, so forgetting it returns no
room to the card and the number the gate read is the number the card held at that read. A build that
cannot open a backend at all answers what `21` answers, before a directory or a mapping.

The lines, in the shape the tests match:

```
load: before_load 'CMD' exited S
load: before_load 'CMD' gave no status
load: card has F MiB free, M MiB needed
load: before_load 'CMD' exited 0
load: card has F MiB free, M MiB asked
load: before_load 'CMD' exited 0, card has F MiB free, M MiB asked
```

`M` is the endpoint's `min_free_vram_mib`, and `F` is the free bytes the backend reports divided by
1048576, rounded down. The first three lines refuse: a command with a non-zero status, a command the
kernel ended without a status, and a card with too little room. The last three are what a run that
passed the gate writes, one for each half the endpoint names and the gate answered: the command, the
card, both. Where the backend found no device the gate answers no card half, so the run writes the
command half alone. A refusing line says `needed`; a passing line says `asked`.

## User stories

- US-001 (P1) As a person whose card is held by another program, I want the engine to run my command
  that frees the card before it loads, so that a real run does not begin by failing.
- US-002 (P1) As that person, I want the engine to refuse before it touches the card when the card is
  still busy, so that a run which cannot have the card says so at once instead of dying part way
  through a load that cost minutes.
- US-003 (P1) As the person who set the threshold, I want the refusal to name what the card had and
  what would have worked, so that I know whether to wait, to lower the number, or to fix the command.
- US-004 (P2) As the person starting the run, I want the start line to say what command ran and what
  the card had, so that I can see the gate did its job rather than assume it.
- US-005 (P2) As the person running a pool of workers, I want the gate to run once, in the process
  that starts the pool, so that the command is not run per worker and no worker refuses on a card its
  neighbour already took.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL accept `before_load` and `min_free_vram_mib` as keys of
    `config.json` at its top level only, `before_load` holding text of one or more characters and
    `min_free_vram_mib` holding an integer of one or more, and refuse either one inside a `models`
    entry as `53/FR-005` refuses a top-level-only key.
- FR-002 (US-001) IF a read file sets `before_load` to a value that is not text of one or more
    characters, or `min_free_vram_mib` to a value that is not an integer of one or more, THEN THE
    SYSTEM SHALL write one line through the stderr writer naming that key and return exit code 1
    without calling a start function, `before_load` before `min_free_vram_mib` in the fault order
    `04/FR-007` holds, after its `threads`.
- FR-003 (US-001, US-004) WHEN a read file sets them, THE SYSTEM SHALL pass `start`, `bench` and
    `reference` the file's `before_load` and `min_free_vram_mib`, as each command's own record field
    and as the endpoint fields of the server each of the three builds, and nil for each the file does
    not set.
- FR-004 (US-001) WHEN `serve.build` is given an endpoint whose `before_load` is text, THE SYSTEM
    SHALL run that text as the `process.exec` of `/bin/sh` with `-c` and that text, with the `stdout`
    and `stderr` inheritance FR-011 states, wait for it to exit and take the exit status its first
    answer gives, a first answer of nil being a command that came back with no status, before it
    creates any context directory, reads any file's metadata, opens any mapping or places any weight.
- FR-005 (US-001) IF that command comes back with no status, or with a status that is not 0, THEN
    THE SYSTEM SHALL return nil and its `load: ` refusal line, holding the command text and that
    status, or the command text and `no status` when none came back, having created no context
    directory, opened no mapping and started no worker process.
- FR-006 (US-002) WHEN an endpoint names `before_load` and also names `min_free_vram_mib`, THE
    SYSTEM SHALL run the command whatever the gate then decides about the card.
- FR-007 (US-002) WHEN `serve.build`'s endpoint names `min_free_vram_mib`, THE SYSTEM SHALL open a
    backend with `compute.load`, ask its `memory` for the free byte count and divide that count by
    1048576 rounded down as the gate's free mebibytes; WHEN that backend answers no free count because
    it found no device, THE SYSTEM SHALL apply no gate whatever the endpoint names.
- FR-008 (US-002, US-003) IF the gate's free mebibytes are fewer than the endpoint's
    `min_free_vram_mib`, THEN THE SYSTEM SHALL return nil and the card's `load: ` refusal line, with
    that line holding the free mebibytes and that `min_free_vram_mib` value as `needed`, having
    created no context directory, opened no mapping, placed no weight and started no worker process.
- FR-009 (US-004) WHEN `serve.build` is given a writer, builds a server whose model runs in that
    process, and is given an endpoint naming `before_load` or `min_free_vram_mib`, THE SYSTEM SHALL
    write through that writer, before `35/FR-010`'s line, the passing `load: ` line the Problem &
    context shows, its command half when a command ran and its card half only where the gate read a
    free count; and WHEN the endpoint names neither, or WHEN it builds a pool, THE SYSTEM SHALL write
    no such line.
- FR-010 (US-005) WHEN `serve.build` starts worker processes, THE SYSTEM SHALL run the gate in its
    own process once and give each worker an endpoint json holding neither `before_load` nor
    `min_free_vram_mib`, so a worker runs no command and applies no gate.
- FR-011 (US-001, US-002) THE SYSTEM SHALL ask for the free count FR-007 divides only after any
    `before_load` command has exited; IF the `compute.load` that opens that backend answers nil, THEN
    THE SYSTEM SHALL return nil and the line it answers, having created no context directory and opened
    no mapping; THE SYSTEM SHALL forget that backend with `compute.unload` before the model is built;
    and THE SYSTEM SHALL run the command with its `stdout` and `stderr` inherited, so the command's own
    output reaches the terminal.
    Not testable: those four. No key points `compute.load` at a library that is not there, so its nil
    answer is confirmed by hand, `libggml-cuda` moved aside, as `21`'s refusal line printed before any
    directory or mapping exists; nothing a case can read is the test process's own terminal, so a case
    that runs a command observes that it ran and what it returned, never where its output went, and the
    inheritance is one pair of `process.exec` `Options`, read in place; the read's place after the
    command is one statement in the gate, confirmed by reading it, since no case can move the number
    the card reports, and its place before any mapping is opened is driven by the card refusal's case,
    which fails once that read sits below the mapping, as FR-008 promises; and the forgetting is one
    call at the gate, read in place, because `compute` publishes `load` and `unload` and keeps no
    registry a case can read, and `28` records that the library and any device context it made outlive
    the entry, so a dropped probe and a kept one answer alike.

## Non-goals

- No flags for these two keys: `54`'s five model-key flags stay the only run flags, and a machine key
  never becomes a model key, so a model named as a path word takes the file's top level.
- No timeout, no retry and no wait-for-room loop: the command is run once and waited on as long as it
  takes, the free number is read once, and one refusal ends the run for the person to re-run.
- No measuring what the model needs against what the card has, and no reading another program's
  numbers: the threshold is the person's, the free number is `21/FR-008`'s, and `35`'s tier still
  spends whatever is free once the gate has passed.

## Tests

Fixtures are the ones the build cases already use: `18/T-001`'s fixture for `serve.build`, `29`'s
fixture for `bench`, and a recording in `55`'s shape for that same fixture, written to a temp file the
way `55`'s own cases write one, for `reference`; `04`'s `run` with a start stub covers the config. A
temp path is a file or directory under the case's own directory. Halves that need a card are gated the
way `35/T-005` gates, skipping with the line `no CUDA device`, while the clauses that ask `memory` to
answer nil and nil run on every machine inside the same case, the way `35/T-004`'s `none` half runs in
`tests/tier.test.tl`.

- T-001 (FR-004, FR-005): `the command runs before the model is opened and a failing one stops the
  load`, proves that `serve.build` on `18/T-001`'s fixture with `before_load` `touch P` returns a
  server and P exists once it has returned; that the same endpoint with `before_load` `exit 3`
  returns nil and one line holding `exit 3` and `3`, writes no `device: ` line, and leaves
  `/proc/self/maps` holding no mapping of the fixture path, the way `tests/serve.test.tl` reads it
  after a build and a `stop`; that with `before_load` `kill -9 $$` it returns nil and one line
  holding `kill -9 $$` and `no status`; that an endpoint whose `path` names a file that does not exist
  and whose `before_load` is `exit 3` returns nil with the line holding `exit 3` rather than a line
  naming that path, so the command ran before the metadata was asked for; that `18/T-001`'s fixture
  with `before_load` `exit 3` and a `context_path` naming an absent temp path returns nil with that line
  and leaves that path absent; and that an endpoint with `workers` 2, a `context_path` naming an absent
  temp path and that failing command returns nil with that line, leaves that path absent, and reaches
  the spawn stub no times.
- T-002 (FR-007, FR-008): `a busy card is refused and a card with room loads`, proves, where
  `compute.load()`'s `memory` answers free and total byte counts, F is the free count divided by
  1048576 rounded down and T is the total count so divided, that an endpoint with `min_free_vram_mib`
  T+1 returns nil and one line holding `MiB free`, `needed` and `T+1`, with the free mebibytes that
  line holds within 64 of F, writes no `device: ` line, and leaves `/proc/self/maps` holding no
  mapping of the fixture path, the way T-001 reads it; that that endpoint with `workers` 2 and an
  absent `context_path` leaves that path absent and reaches the spawn stub no times, and with
  `workers` nil and a `context_path` naming an absent temp path returns nil with that line and leaves
  that path absent; that `min_free_vram_mib` 1 returns a server; and, where `memory` answers nil and
  nil, that `min_free_vram_mib` 1000000000 returns a server, so no card means no gate.
- T-003 (FR-006): `the command runs whatever the gate decides`, proves, where `memory` answers a
  total byte count and T is that count divided by 1048576 rounded down, that an endpoint whose
  `before_load` appends one line to an absent temp file and whose `min_free_vram_mib` is T+1 returns
  nil with the line T-002 proves and leaves that file holding exactly one line.
- T-004 (FR-009): `the start line says what ran and what the card had`, proves that `serve.build`
  on `18/T-001`'s fixture with a collector and `before_load` `touch P` and no `min_free_vram_mib`
  writes as its first line `load: before_load 'touch P' exited 0` and as its second a line matching
  `35/T-004`'s `^device: ` pattern; that, where `memory` answers a free count, the same build with
  `min_free_vram_mib` 1 and no `before_load` writes as its first line
  `load: card has F MiB free, 1 MiB asked`, F being within 64 of the free mebibytes the case read and
  the line holding no `needed`, and one with both keys writes that command fact and that card fact on
  that one line; that an endpoint naming neither key writes no line starting `load: `; that a pool
  endpoint naming `before_load` writes none; and that, where `memory` answers nil and nil, a build with
  `before_load` `touch P` and `min_free_vram_mib` 1 writes as its first line
  `load: before_load 'touch P' exited 0` and writes no line holding `card has`.
- T-005 (FR-010): `a worker runs no command and applies no gate`, proves, on `28/T-001`'s fixture
  with `workers` 2, a context path, a spawn stub that records the text it hands to the child,
  `before_load` `printf x >> P` and `min_free_vram_mib` 1, that the build gives two workers, that
  the recorded text decodes to an object holding neither key, and that P exists and holds one
  `x`, so the command ran once in the parent; and that `serve.stop` leaves no worker behind.
- T-006 (FR-001, FR-002, FR-003): `the two keys reach the run and a bad value is refused`, proves
  with `04`'s `run` that a file naming `before_load` `echo x` and `min_free_vram_mib` 503 reaches
  `start` with both, that a file naming neither reaches it with nil and nil, that files naming
  `before_load` 7, `before_load` ``, `min_free_vram_mib` 0, `min_free_vram_mib` `"512"` and
  `min_free_vram_mib` 1.5 each return 1 with one stderr line holding that key and call no start
  function, that a file whose `before_load` is 7 and whose `min_free_vram_mib` is 0 writes the
  `before_load` line, that a file whose `threads` is 0 and whose `before_load` is 7 writes the
  `threads` line rather than the `before_load` one and one whose `port` is 0 and whose
  `min_free_vram_mib` is 0 writes the `port` line, and that a file naming `before_load` `echo x` and
  one naming `min_free_vram_mib` 503 in entry `a` each return 1 with one line holding that key and
  `a`.
- T-007 (FR-003): `bench and reference carry the gate`, proves on `29`'s fixture and a file naming
  that fixture and a `before_load` writing an absent temp path that
  `bench m --config F --prefill-tokens 4 --decode-tokens 4` writes its `prompt` line and leaves that
  path in place, and that `reference m --config F --reference R` with the same file returns 0 having
  written its `reference: PASS` line and written that path too.

## Open questions

- None.

## Amendments

- `53/FR-001`: the level table gains `before_load` and `min_free_vram_mib`, held at the file's top
  level only, so an entry naming either is refused by that requirement's entry check, and `53/T-004`,
  which asserts the table holds no other name at either level, grows by the two.
- `35/FR-010`: that line is no longer the only line the writer is given for a build whose model runs
  in its own process; the gate's `load: ` line comes before it. The pool clause, which writes nothing,
  is unchanged.
- `28/FR-003`: `endpoint_json` is the endpoint's fields with `workers` nil as JSON except
  `before_load` and `min_free_vram_mib`, which stay with the process that started the pool.
