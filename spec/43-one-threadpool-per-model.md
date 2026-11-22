# 43: One threadpool per model, not one per graph

**Status**: done   **Depends on**: 11, 15, 16, 21, 32, 33

## Problem & context

The pinned ggml's `ggml_backend_cpu_init` leaves a backend's `n_threads` at `GGML_DEFAULT_N_THREADS`,
which is 4, and its `threadpool` at NULL. `ggml_backend_cpu_graph_compute` hands those to
`ggml_graph_compute`, which finds no pool and builds a disposable one: `ggml_threadpool_new_impl`
creates three pthreads for four threads, the graph runs, and the pool is freed, joining them. Every
graph a model computes pays that. `32` makes a pass `3 * N + 2` graphs, so a forty-block model
builds about 360 pools of four threads a token, some 1,080 pthreads, and the 11 us a pool's create
and join measured here is 4 ms a token of thread churn serialised with the arithmetic it should
overlap.

This spec makes the model build one threadpool and keep it. `pool_new` in `src/compute/engine.tl`
already allocates the arena a model's graphs are placed in, and `pool_backend` already keeps the one
CPU backend the model's graphs are computed on, so the pool is held by the record that holds both,
the model's host pool and no other: it is made beside the arena `compute.model` makes for the run,
set on that backend before the first graph it computes, and freed with the arena. The arena `32/FR-006`
gives a block handle built outside a model carries none, and the maths handle of `11` keeps the
disposable pool ggml makes for each of its graphs.

`32/FR-005` is the only other place an arena is given back: a pass holding more rows than the arena
was measured at replaces it. The model's arena is measured at its `sequences` rows and `33`'s chunked
prefill sends one sequence's chunk in one pass, so that path is reached, and a pool goes with it: the
old pool's threadpool is freed after its backend, and the grown pool makes its own. One pool per
model and one graph at a time, which is `32`'s rule, so no two graphs ever share a pool.

The count the pool is made with is `44`'s: the configured or default number a model is given, not
ggml's four.

Reaching the calls takes care, because the library the engine holds does not export them. It is
`ffi.load("target/tllm/libggml.so")`, and `ggml_threadpool_new`, `ggml_threadpool_free` and
`ggml_backend_cpu_set_threadpool` are built into the CPU backend module ggml loads at run time
(`libggml-cpu-icelake.so` on this machine), so a lookup of any of them on that handle fails. The
route, checked here against the pinned build, is libggml-base.so's
`ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CPU"), name)`, which answers all three
and answers NULL for `ggml_threadpool_get_n_threads`; no `struct ggml_backend_reg` is mirrored, only
`struct ggml_threadpool_params`, byte-exact at 528 bytes: `bool cpumask[512]`, `int n_threads`,
`int prio`, `uint32_t poll`, `bool strict_cpu`, `bool paused`. The parameters are filled by
`ggml_threadpool_params_init(&p, 4)` and passed by pointer, so no 528-byte struct crosses the FFI by
value, where a field order taken from a different ggml version would write past the return buffer.
The three entries are resolved once, when `compute.load` loads the library, and kept in the table it
holds beside the library, so a case can replace them to count what the model does with them.

The device path of `21` is untouched: a model whose weights are in device memory or whose placement
marks a tensor computes on the device backend and holds no threadpool at all.

The numbers must not move. `11` to `15` pin them and every case of theirs runs unchanged. The suite's
largest model fixture measures 66 us per graph today, of which the churn this removes is about 11 us,
so no case here asserts a time: the pool is proved by what makes and frees it, and `29` reports the
speed on a real model.

## User stories

- US-001 (P1) As the engine author, I want a model to build one threadpool and compute every graph of its passes with it, so that a token stops paying a create and a join for each of about 360 graphs.
- US-002 (P1) As the engine author, I want every number and every reply the suite pins to stay as it is, so that this is a speed change and not a maths change.
- US-003 (P2) As the engine author, I want a model whose weights are on a device to hold no CPU threadpool, so that the device path is what `21` left it.

## Functional requirements

- FR-001 (US-001) WHEN a model whose weights are in host memory is built, THE SYSTEM SHALL create
  one ggml threadpool of the model's `threads`, the configured or default count `44` defines, and
  hold it with the model, so that building the model creates exactly one and the passes it runs
  create none.
- FR-002 (US-001) WHEN a graph of that model's pass is computed, THE SYSTEM SHALL compute it on the
  CPU backend the model holds with that threadpool set on that backend before the first graph it
  computes, so that every graph of the model's passes runs on that pool's threads.
- FR-003 (US-001) IF a pass holds more rows than the arena was measured at, THEN THE SYSTEM SHALL
  replace the pool with one sized for that pass, freeing the threadpool the replaced pool held, the
  replacement's threadpool and the backend it is set on using the same count as the model's, so that
  the model holds one threadpool after the replacement as it did before it.
- FR-004 (US-001) WHEN a model's `close` is called, THE SYSTEM SHALL free the threadpool it holds,
  after the CPU backend that held it, so that a closed model leaves no thread behind and a second
  `close` frees nothing further.
- FR-005 (US-002) WHEN a model computes a pass on the threadpool it holds, THE SYSTEM SHALL return
  the score rows and the reply `15/T-001`, `15/T-002` and `16/T-001` pin, as `32/FR-010` states for
  a pass's numbers.
  Not testable: the cases of `11`, `12`, `13`, `14`, `15` and `16` assert exactly these and run
  unchanged in the same suite, so a case here would be those assertions twice.
- FR-006 (US-003) WHEN a model's weights are in device memory or its placement marks at least one
  tensor, THE SYSTEM SHALL hold no threadpool and compute every graph of its passes on the device
  backend.
  Not testable: nothing a model publishes reports a threadpool, and only a model's host arena is
  handed the entries that make one; a reader confirms that `on_the_cpu` is the one place they enter
  `compute.model` and that `pass_arena` only carries the entries of the arena it replaces, so a
  model whose weights are in device memory, or whose placement marks a tensor, holds none.

## Non-goals

- No pool for the maths handle of `11` or for a block handle built outside a model, whose arena
  `32/FR-006` gives it: each graph they compute keeps the threadpool ggml makes for it and frees
  after.
- No thread count choice here: the number is `44`'s.
- No change to the routes, the sampler or `29`'s report lines, and no shared pool between worker
  processes: `28`'s workers each build a model and so each hold a pool of their own.

## Tests

- T-001 (FR-001, FR-002): `a model makes one threadpool and keeps it`, proves that with the
  threadpool entries `compute.load` resolves into the table it keeps beside the library replaced by
  counting wrappers before `compute.model` is called, building `15/T-001`'s fixture with
  `context_length` 32 by `compute.model(backend, handle, 32, nil)` makes one threadpool and frees
  none, that the first `forward(0, p)` sets that pool on the backend its graph is computed on, that
  twenty of them, p from 0 to 19 and 5 graphs each, make and free none more, and that the `Threads`
  line of `/proc/self/status` is exactly `n - 1` larger after the build than before it, `n` being the
  count `44/FR-004` gives this machine, and the same after those twenty.
- T-002 (FR-003): `a wider pass takes a new pool and frees the old`, proves that with the same
  wrappers, on `15/T-002`'s mixed fixture built by `compute.model(backend, handle, 4, nil, 1)` with
  the sequence `""` reserved at 4 positions, a `pass` of that sequence's three rows at positions 0, 1
  and 2 returns three score rows and nil, its 8 graphs needing the arena `32/FR-005` replaces, and
  that it makes one more threadpool, frees one, and leaves the `Threads` line where the build left
  it.
- T-003 (FR-004): `no pool survives close`, proves that `close` on T-001's model frees one threadpool
  and returns the `Threads` line of `/proc/self/status` to what it was before the build, and that a
  second `close` frees none more.

## Open questions

- None.

## Amendments

- None.
