# 54: The five model keys on the command line

**Status**: done   **Depends on**: 04, 24, 29, 33, 36, 38, 44, 53

## Problem & context

`53` gave `kv_cache`, `ubatch`, `vram`, `profile` and `prefetch` a home in a `models` entry, but an
entry exists only for a model the file names. `serve` and `bench` take a `.gguf` path as the model
word, and `53/FR-004` resolves every model-level value for such a word from the file's top level and
the defaults. So the one thing a person still cannot do is run a file that is not in the config with
numbers of their own: a cache type for a file they downloaded an hour ago, or a `vram` cap for a card
that is already busy, means opening `config.json`, adding an entry for a model they will use once, and
deleting it afterwards. It also blocks the comparisons these keys exist for: one cache type against
another, or read-ahead open against closed, needs a config file per run.

`44` already solved this for `threads`: `--threads N` on both commands, refused with one line naming
the flag, winning over the file's value (`44/FR-006`). The five model keys never got the same half of
the shape `53` copied: llama.cpp keeps those numbers in its per-model preset block and carries them
as run flags (`--cache-type-k`, `--ubatch-size`) as well.

The shape: `serve` and `bench` gain `--kv-cache TYPE`, `--ubatch N`, `--vram N`, `--profile FILE` and
`--prefetch` / `--no-prefetch`. Each value is checked by the rule the config already checks that key
by, and sits above both file levels in the order `53` sets: flag, entry, top level, default. Those rules are the config module's own, the ones `04/FR-007` holds for a value the file names and
`53/FR-002` reuses for an entry's value, and the flag check calls them rather than holding its own
copy: `53`'s whole point was that a key's rule is written once, and a command line with a second list
of accepted cache types and a second whole-number test drifts from it with nothing to catch the drift.
In `src/config.tl` they are two functions, `cache_type` for the four cache types and `whole_count` for
a whole number of one or more, and `value_ok`, the rule both file levels already used, now calls them
too. A flag that
breaks its rule is refused with one line naming the flag, before the model word is resolved and before
the file is read, where `--port`, `--host` and `--threads` are refused today, and several flags at
fault write the one line of the first of them. `prefetch` is a boolean and takes no value:
`--prefetch` forces it on and `--no-prefetch` forces it off, as llama.cpp's `--mmap` and `--no-mmap`
and git's `--no-ff` do, and giving both in one run is refused naming both. A value that is not a
number at all (`--ubatch abc`) is refused by the parser's own line naming the flag, as `--threads` is
today, so it never reaches the key's rule. `--profile` has no refusal case of its own, because a
flag's value is always a string and that is all the config's rule for `profile` asks.

## User stories

- US-001 (P1) As someone running a `.gguf` that is not in the config, I want to set the cache type, ubatch, VRAM reserve, profile and prefetch on the command line, so that a one-off run needs no config edit.
- US-002 (P1) As someone who mistypes a flag value, I want one line naming the flag and exit code 1, so that I am never handed a run on the file's numbers instead.
- US-003 (P2) As someone comparing two cache types, or read-ahead open against closed, I want the same flags on `bench` as on `serve`, so that the comparison is two command lines rather than two config files.

## Functional requirements

- FR-001 (US-001) WHEN `serve` or `bench` is given `--kv-cache V`, THE SYSTEM SHALL resolve that run's `kv_cache` to V, in place of the resolved model's `models` entry value and the file's top-level value.
- FR-002 (US-001, US-003) WHEN `serve` or `bench` is given `--ubatch N` or `--vram N`, THE SYSTEM SHALL resolve that run's `ubatch` or `vram` to N, in place of the resolved model's entry value and the file's top-level value.
- FR-003 (US-001, US-003) WHEN `serve` or `bench` is given `--profile P`, THE SYSTEM SHALL resolve that run's `profile` to P, in place of the resolved model's entry value and the file's top-level value.
- FR-004 (US-001) WHEN `serve` or `bench` is given `--prefetch`, THE SYSTEM SHALL resolve that run's `prefetch` to true, in place of the resolved model's entry value and the file's top-level value.
- FR-005 (US-001) WHEN `serve` or `bench` is given `--no-prefetch`, THE SYSTEM SHALL resolve that run's `prefetch` to false, in place of the resolved model's entry value and the file's top-level value.
- FR-006 (US-002) IF `--kv-cache V` holds a V that the rule `04/FR-007` checks `kv_cache` by refuses, V being none of `f32`, `f16`, `q8_0` and `q4_0` compared case-sensitively, THEN THE SYSTEM SHALL reach that refusal by calling the cache-type rule the config module exposes rather
  than a list the command line holds, a clause no case separates from a copy and so one a reader
  confirms in `src/cli.tl`, and SHALL write one line through the stderr writer naming `--kv-cache` and
  V and return exit code 1 without calling a start function or building a model.
- FR-007 (US-002) IF `--ubatch N` or `--vram N` holds an N that the rule `04/FR-007` checks that key by refuses, N not being a whole number of one or more, THEN THE SYSTEM SHALL reach that refusal by calling the whole-number rule that same module exposes rather than a whole-number test of its own, and SHALL write one line through the stderr writer naming that flag and N and return exit code 1 without calling a start function or building a model.
- FR-008 (US-002) IF `serve` or `bench` is given both `--prefetch` and `--no-prefetch`, THEN THE SYSTEM SHALL write one line through the stderr writer naming both flags and return exit code 1 without calling a start function or building a model.
- FR-009 (US-002) WHEN a flag is at fault under FR-006, FR-007 or FR-008, THE SYSTEM SHALL report it before the model word is resolved and before the config file is read, so a run whose flag and config file are both at fault writes only the flag's line.
- FR-010 (US-002) WHEN more than one of the flags named in this requirement is at fault, THE SYSTEM SHALL write the one line of the first of them at fault, the prefetch pair counting as one fault naming both, in the order `--port`, `--host`, `--threads`, `--kv-cache`, `--ubatch`, `--vram`, `--prefetch` and `--no-prefetch`, and write no other flag's line.

## Non-goals

- No flags for the process keys: `sessions`, `workers` and `context` stay file-only, and `host`, `port` and `threads` keep the flags `04` and `44` already give them.
- No flag for `path`: the model word stays the only way to name the model a run serves.
- No flag writes or edits `config.json`: `04`'s starter-file behaviour is untouched, so a flag run on a name no file holds still writes the starter.

## Tests

- T-001 (FR-001, FR-002, FR-003, FR-004, FR-005): `a flag on a path word reaches the endpoint and the keys it does not name keep their defaults`, proves with no config file and `F` a readable `.gguf` path word that `serve F --kv-cache q8_0` reaches `start` with `kv_cache` `q8_0`, `ubatch` 512 and nil `vram`, `profile` and `prefetch`; that `serve F --ubatch 8 --vram 4096 --profile P --prefetch` reaches it with `ubatch` 8, `vram` 4096, `profile` P, `prefetch` true and `kv_cache` `f16`; and that `serve F --config G --no-prefetch` against a file `G` whose top level holds `prefetch` true reaches it with `prefetch` false.
- T-002 (FR-001, FR-002, FR-003, FR-004, FR-005): `a flag beats an entry that names the key`, proves of a file whose entry `m` names all five keys that `serve m --kv-cache f32` reaches `start` with `f32` and the entry's other four values, and that `serve m --ubatch 4 --vram 1 --profile Q --no-prefetch` reaches it with those four and the entry's `kv_cache`.
- T-003 (FR-006, FR-007, FR-008, FR-010): `a flag value that breaks its key's rule is refused naming the flag`, proves `serve F --kv-cache q8_0x`, `serve F --kv-cache F16`, `serve F --ubatch 0`, `serve F --ubatch abc`, `serve F --vram 0` and `serve F --prefetch --no-prefetch` each return 1 with exactly one stderr line naming the flag at fault, both flags for the prefetch pair, and the value as well for the cases that give one, and no `start` call; and that `serve F --kv-cache q8_0x --ubatch 0` writes one line naming `--kv-cache` and not `--ubatch`, that `serve F --vram 0 --prefetch --no-prefetch` writes one line naming `--vram`, and that `serve F --port 0 --ubatch 0` writes one line naming `--port`.
- T-004 (FR-009): `the flag fault is reported before the model word and the file`, proves `serve bogus --ubatch 0` writes the `--ubatch` line and not the unknown-model line, and that `serve m --config F --kv-cache q4x` against a file naming `kv_cache` at its top level and in entry `m` writes only the flag line rather than `53/FR-006`'s.
- T-005 (FR-001, FR-002, FR-003, FR-004, FR-005, FR-007): `bench takes the flags`, proves on `29`'s fixture and the machine `36/T-003` gates on that `bench <fixture path> --vram N --profile P --no-prefetch --prefill-tokens 4 --decode-tokens 4` writes the `experts 4:` line `53/T-007` proves for those same two values held in an entry, holding `0.0% hit; 0 ahead, 0.0% landed; read ahead closed` where `36/T-003` matches its hit and ahead text, since `38/FR-009` gives a model built with `prefetch` false no queue at all; that the same run with `--prefetch` writes that line with `read ahead open` in that place, its ahead counts and landed rate whatever the open queue reports as `38/T-005` leaves them; that `bench <fixture path> --kv-cache q8_0 --prefill-tokens 4 --decode-tokens 4` returns 1 with one line naming `q8_0` and the head length `24/FR-004` reports for that fixture, so the flag's type reached the model `bench` built; and that `bench <fixture path> --vram 0` returns 1 with one line naming `--vram`.

## Open questions

- None.

## Amendments

- None.
