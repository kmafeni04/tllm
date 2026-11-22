# 44: The thread count

**Status**: done   **Depends on**: 04, 43

## Problem & context

`43` gave a model one threadpool and kept it, but the pool is made with four threads, ggml's
`GGML_DEFAULT_N_THREADS`. That count is a library default, not this machine's: it has eight
physical cores, so a pool of four leaves half of them idle, and the CPU bounds `39` records for
strata say the count a matvec wants is a number to measure rather than assume. A user with a
different machine has no way to say so, and `bench` has no way to compare counts.

The count alone is not enough. The pinned ggml plans each graph from the CPU backend's own
`n_threads`, which `ggml_backend_cpu_init` leaves at four and `ggml_backend_cpu_set_threadpool`
does not change. A pool of seven on a backend left at four plans four tasks and leaves three pool
threads asleep, so this spec gives the backend the count too, through the registry entry
`ggml_backend_set_n_threads`, before its first graph.

This spec makes the count a setting: a top-level `threads` key in `config.json` and a `--threads N`
flag on both commands. The number is ggml's `n_threads`, the calling thread included, so a model at
`threads` 7 holds six worker threads and leaves the eighth core to the rest of the turn. It is
carried to `compute.model`, where the pool `43` made is built, into the endpoint json a worker
decodes the way `33` carries `ubatch`, and into `bench.Settings`. A file or flag that is not a whole
number from 1 to 512 is refused with `04`'s one line naming `threads`. The upper bound is
`GGML_MAX_N_THREADS`, the size of the thread parameters' `cpumask`: a larger count makes ggml
allocate and spawn that many threads without a check, which kills the process instead of answering
with one line.

The default is `cores.default`, a function over a `/proc/cpuinfo` text: the distinct `physical id`
and `core id` pairs the text reports less one, never below 1, or its `processor` count less one when
it reports no such pair, or 1 when it reports neither. On this machine that is seven, not the
sixteen logical processors, which share cores. When a config file runs more than one worker the
default is divided among them, rounded down and never below 1, because each worker builds a model of
its own; an explicit file or flag count is per model and is not divided.

Because the default is no longer four, the process gains a different number of threads than `43`'s
`T-001` expected: a pool of `n` threads is `n - 1` worker threads, so at seven the process gains six
rather than three. `43`'s requirements and test wording are amended below. The count itself is
chosen when a model is built and then fixed; nothing changes it under a running model.

```teal
-- src/cores.tl
local record cores
  -- the model's default thread count, from a /proc/cpuinfo text
  default: function(cpuinfo: string): integer
  -- this machine's default thread count
  machine: function(): integer
  -- whether a count is one ggml can be given, 1 to 512
  fits: function(count: number): boolean
end

local record compute
  record Model
    -- 15's, 33's and 38's fields, unchanged
    threads: integer
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
    prefetch?: boolean,
    threads?: integer
  ): Model | nil, string | nil
end

local record serve
  record Endpoint
    -- 38's fields, unchanged
    threads: integer | nil
  end
end

local record bench
  record Settings
    -- 38's fields, unchanged
    threads: integer | nil
  end
end
```

The `Config` record in `src/config.tl` gains `threads: integer | nil`. This snippet supersedes the
`model` declaration and the `Endpoint` and `Settings` records `38` leaves.

## User stories

- US-001 (P1) As a person running a model, I want to set how many threads it computes with in my config or on the command line, so that it uses my machine instead of a library default.
- US-002 (P2) As the engine author, I want serve, every worker and bench to give the model and its graph planner the same count, so that what I measure is what serves.
- US-003 (P1) As a user whose config is wrong, I want one line naming `threads` and exit code 1, so that I can fix it without reading source.
- US-004 (P2) As the person choosing the default, I want bench to print a rate at 1, 2, 4 and 8 threads on the real model, so that the default is a measured number and not a claim.

## Functional requirements

- FR-001 (US-001) WHEN a config file is read, THE SYSTEM SHALL accept `threads` as a top-level key holding an integer from 1 to 512.
- FR-002 (US-003) IF `threads` is present and is not an integer from 1 to 512, THEN THE SYSTEM SHALL write one line through the stderr writer naming `threads` and the file, and return exit code 1 without calling a start function.
- FR-003 (US-001) WHEN a config file sets `threads` and no `--threads` flag is given, THE SYSTEM SHALL pass that number to `start` as the endpoint's `threads`.
- FR-004 (US-001) WHEN neither the config file nor a `--threads` flag names a count, THE SYSTEM SHALL pass `cores.default` for this machine's `/proc/cpuinfo` text to `start` as the endpoint's `threads`; `cores.default` SHALL answer the distinct `physical id` and `core id` pairs the text reports less one and never below 1, its `processor` count less one when it reports no such pair, and 1 when it reports neither.
- FR-005 (US-002) WHEN the config file names `workers` more than one and neither the file nor a flag names `threads`, THE SYSTEM SHALL divide the default among those workers, rounded down and never below 1, and pass that count to `start`.
- FR-006 (US-001) WHEN `serve` or `bench` is given `--threads N`, THE SYSTEM SHALL use N in place of the config file's `threads`; IF N is not an integer from 1 to 512, THEN THE SYSTEM SHALL write one line through the stderr writer naming `--threads` and return exit code 1 without calling a start function or building a model.
- FR-007 (US-002) WHEN `serve.build` builds its model, THE SYSTEM SHALL give `compute.model` the endpoint's `threads`, `cores.default` for this machine when the endpoint names none.
- FR-008 (US-002) WHEN a worker decodes its endpoint json, THE SYSTEM SHALL carry the json's `threads` into the endpoint it builds and build its model with that count.
- FR-009 (US-002) WHEN `bench` builds its server, THE SYSTEM SHALL give the endpoint the `threads` the flag or the config file names, `cores.default` for this machine when neither names one.
  Not testable: bench publishes no thread count and `29` forbids a case that pins a rate; the four runs of FR-011 are the visible check.
- FR-010 (US-002) WHEN `compute.model` is given `threads`, THE SYSTEM SHALL build the model's one threadpool with that count, set the same count on the CPU backend that computes its graphs through the registry's `ggml_backend_set_n_threads` entry before that backend's first graph, and expose it as the model's `threads`; WHEN given none, THE SYSTEM SHALL use `cores.default` for this machine; IF given a count below 1 or above 512, THEN THE SYSTEM SHALL return nil and one line naming `model` and `threads`.
- FR-011 (US-004) WHEN `bench` runs with `--threads` 1, 2, 4 and 8 on the real model, THE SYSTEM SHALL print a prompt rate line and a decode rate line for each count.
  Not testable: the suite runs fixtures, not the real model, and no case pins a token rate (`29`'s non-goal); a person runs the four commands and records the four line pairs in the commit message.

## Non-goals

- No CPU affinity: no thread is pinned to a core and no `cpumask` is set.
- No per-graph count: every graph of a model's passes uses the one pool's count.
- No change to a running model and no `threads_batch` behaviour: the count is chosen when the model is built, and `33`'s chunks share the one pool.

## Tests

- T-001 (FR-001, FR-003, FR-004, FR-005, FR-006): `the count reaches start from the file, the flag and the default`, proves with `04`'s `run`, an `environment` and a start function that a file naming `threads` 3 for the entry `m` reaches `start` with 3, that the same file and `--threads 5` reaches it with 5, that a file naming no `threads` reaches it with the count `cores.default` gives for this machine's `/proc/cpuinfo`, that a file naming `workers` 2 and no `threads` reaches it with that default divided by two, rounded down and at least 1, that a file naming `workers` one more than the default and no `threads` reaches it with 1, that `workers` 1 reaches it with the default itself, and that `cores.default` gives 2 for a text holding three distinct pairs, 3 for a text of four processors and no pair, and 1 for a text of one pair and for an empty text.
- T-002 (FR-007, FR-010): `the count reaches the model and its graph planner`, proves on `18/T-001`'s fixture that `serve.build` on an endpoint with `threads` 3 returns a server whose `serve.model_of(server).generation.threads` is 3 and that an endpoint naming none gives the default; and that with the `ggml_backend_set_n_threads` entry `compute.load` resolves into the table it keeps beside the library replaced by a counting wrapper before `compute.model` is called, `compute.model(backend, handle, 32, nil, 1, nil, nil, nil, nil, 3)` exposes 3 and the first `pass` of it calls that wrapper once with 3 before that pass's first graph is computed, and a model built with no `threads` calls it with the default.
- T-003 (FR-002, FR-006, FR-010): `a count outside one to 512 is refused`, proves that files naming `threads` `"8"`, 0, 1.5, -1 and 513 each return 1 with one stderr line holding `threads` and the file's path and call no start function, that `serve --threads 0`, `serve --threads 1.5`, `serve --threads 513` and `bench --threads 513` each return 1 with one stderr line holding `--threads`, and that `compute.model` given `threads` 0 and given 513 returns nil and one line holding `model` and `threads`.
- T-004 (FR-008): `a worker decodes the count from its endpoint json`, proves on `28/T-001`'s fixture with a context path, `workers` 2 and `vram` 1, so that a worker's model is built on the CPU and holds a threadpool rather than copying its weights to the card as `43/FR-006` leaves it, that `serve.build` on an endpoint with `threads` 2 gives two workers, that a second build on an endpoint whose `threads` is 4 gives two workers each holding exactly two more threads in its `/proc` status entry's `Threads` line than the workers of the 2 build, and that both servers' `stop` leaves no worker process behind.

## Open questions

- None.

## Amendments

- None.
