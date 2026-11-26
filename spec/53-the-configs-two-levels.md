# 53: The config's two levels: what a model owns

**Status**: done   **Depends on**: 04, 24, 27, 28, 29, 33, 35, 36, 38, 44

## Problem & context

`04` gave `config.json` one flat level plus a `models` map whose every entry holds only a `path`,
and every key since has landed at that top level: `kv_cache` (24), `ubatch` (33), `vram` (35),
`profile` (36), `prefetch` (38), `threads` (44). Six specs, six locally correct additions, and not
one asked which model the value belongs to. Five of the six belong to a model: the cache type is a
quality call `24` already bounds per model by its `head_length`, `ubatch` is the group of rows
`33/FR-010` asks that model's run for, `vram` is that model's cap below what the card has left,
`profile` is that model's activation counts, and `prefetch` is that model's state, since `38` holds
it for the model's life. A file naming two models starts one with the other's numbers and says
nothing about it. `04`'s own sentence, "`path` is the only per-model setting for now", is where it
started, and "for now" never arrived. It was not wrong in
`04`, where one process served one model; it went wrong when `models` grew past one entry and no
later spec noticed.

The shape is the one the two config formats this one was meant to copy already use. llama.cpp's
model preset file (the ini its `--models-preset` takes) is a `[*]` defaults block and one block per
model, and it keeps `cache-type-k`, `ubatch-size` and `n-cpu-moe` per model, with its own block for
a vision variant precisely because those numbers change per model. Strata's launch file is one file
per model with every switch inside it: `--kv`, `--max-context`, `--expert-profile`,
`--vram-reserve-mib`, the port.

So a model entry holds `path` and may hold `kv_cache`, `ubatch`, `vram`, `profile` and `prefetch`,
and the top level keeps what one process owns: `host`, `port`, `sessions`, `workers`, `context` and
`threads`, and may hold any of the five model keys as the default for every entry. `threads` stays
top level because `44/FR-005` divides it across workers, which is a process fact and not a model
one.

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "threads": 8,
  "ubatch": 512,
  "models": {
    "qwen3.8-27b": {
      "path": "/home/me/models/qwen.gguf",
      "kv_cache": "q8_0",
      "vram": 3072,
      "profile": "/home/me/models/qwen.jsonl",
      "prefetch": true
    },
    "flash-next": {
      "path": "/home/me/models/flash.gguf",
      "kv_cache": "f32"
    }
  }
}
```

The block shows the shape and is printed for reading. A model-level value resolves entry, then the
file's top level, then the default. The top level is a default for every entry, so one key named at
both file levels leaves the file saying two things at once and is refused naming both rather than
resolved. `path` is required in an entry and is never a top-level key.

The worker's endpoint json (`28/FR-003`) and `bench.Settings` (`29/FR-011`) take the resolved values,
not the file's top level. A `27` context file already records the cache type in its own header field and
`27/FR-006` refuses a file written under another, so a per-model type needs no new check there.

The rule, so this is not done a seventh time: one table in `config.tl` declares every key the file
may hold, the level whose value wins for it and the levels it may appear at, both key checks read
their accepted set from that table, and a test asserts the table names exactly the keys the file
accepts, each with the level that owns its value. A key that skips the decision fails a test
instead of landing wherever the spec that added it happened to
write it. The same rule now sits in `spec/README.md`'s Constraints.

## User stories

- US-001 (P1) As someone whose file names two models, I want each model to carry its own cache type, ubatch, VRAM reserve, profile and prefetch, so that one model does not run on the other's numbers.
- US-002 (P1) As someone whose file names one key at both levels or puts a process key inside an entry, I want one line naming the key and every model at fault and exit code 1, so that I am never left guessing which number the engine used or fixing one entry at a time.
- US-003 (P1) As the next spec that adds a config key, I want one table saying which level owns every key and a test that fails when a key skips it, so that the level is decided once and not by whoever lands first.
- US-004 (P2) As a bench run on a configured model, I want the run to take that model's entry values, so that a measurement uses the numbers the model will be served with.

## Functional requirements

- FR-001 (US-003) THE SYSTEM SHALL expose from the config module one table naming every key `config.json` may hold, the level whose value wins for it, and the levels it may appear at: `host`, `port`, `models`, `sessions`, `context`, `workers` and `threads` at the top level only, `path` in a `models` entry only, and `kv_cache`, `ubatch`, `vram`, `profile` and `prefetch` owned by a `models` entry and also allowed at the file's top level as the default `FR-003` falls back to. The top-level check accepts the names the table allows there and the entry check the names it allows in an entry, so `path` at the top level, a top-level-only key inside an entry, and a name the table holds at neither level are each refused where they appear.
- FR-002 (US-001) WHEN a `models` entry holds `kv_cache`, `ubatch`, `vram`, `profile` or `prefetch`, THE SYSTEM SHALL check that value by the rule the same key is checked by at the top level and hold it as that entry's value.
- FR-003 (US-001) WHEN the engine passes `kv_cache`, `ubatch`, `vram`, `profile` or `prefetch` to the start function, to `bench.Settings` or to the endpoint json a worker is started from, THE SYSTEM SHALL use the resolved model's entry value when that entry names one, the file's top-level value when the entry names none and the file names one, and the default the file leaves (`f16`, `512`, nil, nil, nil) when neither names one.
- FR-004 (US-001) WHEN the model word names a file path instead of a `models` key, THE SYSTEM SHALL resolve every model-level value from the file's top level and the defaults, since no entry names it.
- FR-005 (US-002) IF a `models` entry holds a key the table allows at the top level only, THEN THE SYSTEM SHALL write one line through the stderr writer naming that key and every entry holding it in ascending byte order and return exit code 1 without calling a start function, at the position `04/FR-006` holds and before any value check, that position reporting `04/FR-006`'s unknown top-level key first, then `FR-006`'s both-levels key, then this one.
- FR-006 (US-002) IF a read file names one of `kv_cache`, `ubatch`, `vram`, `profile`, `prefetch` at the top level and in any `models` entry, THEN THE SYSTEM SHALL write one line through the stderr writer naming that key and every entry naming it in ascending byte order and return exit code 1 without calling a start function, whether or not any of those entries names the resolved model, at the same position and before any value check, so a file that trips this and a value fault writes only this line.
- FR-007 (US-002) IF an entry's value for one of those five keys breaks the rule `FR-002` checks it by, THEN THE SYSTEM SHALL write one line through the stderr writer naming that key and every entry holding a value for it that breaks that rule in ascending byte order and return exit code 1 without calling a start function, reported at the `models` position in the fault order `04/FR-007` sets, that position reporting an entry that is not a table holding a `path` string first, then the keys in the order `kv_cache`, `ubatch`, `vram`, `profile`, `prefetch` across every entry.
- FR-008 (US-004) WHEN `bench` resolves a model, THE SYSTEM SHALL give its `bench.Settings` and the endpoint it builds its server from that model's values resolved as `FR-003` says.

## Non-goals

- No flags for the model keys: `--kv-cache`, `--ubatch`, `--vram`, `--profile` and `--prefetch` do not appear, so a model named as a path word keeps the top level and the defaults, and setting numbers for a file that is not in the config stays its own problem.
- No model-level process keys: `host`, `port`, `sessions`, `workers`, `context` and `threads` stay top level, and `44/FR-005` still divides `threads` across the workers.
- No new check on `27`'s on-disk context files: `27/FR-006` already refuses a file written under another cache type.

## Tests

- T-001 (FR-002, FR-003, FR-004): `an entry's values reach its model and not its neighbour's`, proves with one file holding entry `a` (a `kv_cache`, `vram`, `profile`, `prefetch`) and entry `b` (only a `path`) and a top-level `ubatch` 128, that `serve a` reaches the start function with the entry's four values and `ubatch` 128, that `serve b` reaches it with `f16`, nil, nil, nil and `ubatch` 128, and that `serve` with `a`'s path as the model word reaches it the way `b` does, and that a file whose only entry is keyed by `a`'s path and names another `path` beside `kv_cache` `q8_0` and `vram` 3072 still reaches `start` with `f16`, nil and `ubatch` 512 when that path is the model word, since a key spelled like a path names no entry.
- T-002 (FR-005, FR-006): `a both-levels key is refused and the key checks run in order`, proves a file naming `kv_cache` at the top level and in entry `a` returns 1 with one stderr line holding `kv_cache` and `a` and no start call when `a` is the resolved model, and the same when `b` is the model word, that a file naming `kv_cache` at the top level and in entries `b` and `a` writes one line holding `kv_cache`, `a` and `b`, that a file naming `kv_cache` `bogus` at the top level and `kv_cache` `f32` in entry `a` writes one line holding `kv_cache` and `a` rather than the value fault's line, that a file holding `prot` at the top level and `port` in entry `a` writes one line holding `prot`, that a file holding `prot` and `kv_cache` at the top level and `kv_cache` in entry `a` writes one line holding `prot`, and that a file naming `kv_cache` at the top level and in entry `a` and holding `port` in entry `a` writes one line holding `kv_cache` and `a` rather than the `port` line.
- T-003 (FR-003): `a flag run still takes the entry's value`, proves `--threads 3` against a file naming top-level `threads` 2 and an entry naming `kv_cache` reaches the start function with 3 and the entry's cache type.
- T-004 (FR-001, FR-005): `the level table names every key the file accepts and nothing else`, proves the table the config module exposes names `host`, `port`, `models`, `sessions`, `context`, `workers` and `threads` at the top level only, `path` in an entry only, and `kv_cache`, `ubatch`, `vram`, `profile` and `prefetch` in an entry with the file's top level allowed as their default, and no other name at either level, that each of `host`, `port`, `sessions`, `workers`, `context`, `threads` inside an entry returns 1 with one stderr line holding that key and the model, that entries `b` and `a` each holding `port` write one line holding `port`, `a` and `b`, and that `path` at the top level returns 1 with one stderr line holding `path`, and that an entry `a` holding `prot` returns 1 with one stderr line holding `prot` and `a`.
- T-005 (FR-002, FR-007): `an entry's bad value names the key and every entry at fault`, proves an entry `a` holding `kv_cache` `q8_0x`, `ubatch` 0, `vram` 0, `profile` 1 and `prefetch` `"yes"` each return 1 with one stderr line holding `a` and that key and no start call, that entries `b` and `a` each holding `ubatch` 0 write one line holding `ubatch`, `a` and `b`, and that a file whose entry `a` holds `kv_cache` `f32` and no `path` and whose entry `b` holds `ubatch` 0 writes one line holding `a` rather than the `ubatch` line, and that a file whose entry `b` holds `ubatch` 0 and whose entry `a` holds `kv_cache` `q8_0x` writes one line holding `kv_cache` and `a` rather than the `ubatch` line.
- T-006 (FR-003): `the endpoint json a worker starts from carries the entry's values`, proves `serve.start` on an endpoint holding an entry's resolved values with `workers` 2 and a spawn stub that records the text it is given as stdin and starts no child, that the recorded text decodes to an endpoint holding those values.
- T-007 (FR-008): `bench takes the entry's numbers`, proves on `29`'s fixture and the machine `36/T-003` gates on that `bench m --config F --prefill-tokens 4 --decode-tokens 4`, with `vram` and `profile` in `m`'s entry rather than at the top level, writes the `experts 4:` line `36/T-003` proves for those same two values at the top level.

## Open questions

- None.

## Amendments

- None.
