# 04: Read config.json and give serve its settings

**Status**: done   **Depends on**: 03

## Problem & context

`tllm serve` today has no settings but its flags. `--config` is parsed and dropped, no file is
read, and the only values the server ever sees are the two flags, with `127.0.0.1` and `8080`
standing in for the ones the user did not give. So a model can only be named on the command
line, there is nowhere to keep the list of models a machine has, and a typo in a flag is the whole
of the configuration surface.

This spec gives serve a file. It looks for `config.json` under the XDG config directory, falls
back to `~/.config`, takes a path from `--config` instead when one is given, and refuses every
file it cannot make sense of with one line and exit code 1 rather than starting on a guess. The
file names the models this machine has, keyed by the name a client will later ask for, which is
how `llama-server`'s preset file works.

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "models": {
    "qwen3.8-27b": {
      "path": "/home/me/models/qwen3.8-27b.gguf"
    }
  }
}
```

The block shows the shape and is printed for reading; `FR-004` writes the compact text.

`host` and `port` are what serve binds to, and a flag beats the file. `models` is keyed by name
because the name is what a request will ask for later, so it is an identity rather than a field to
be duplicated and kept unique by a rule. `path` is the only per-model setting for now.

The model word is `MODEL` from `02/FR-003`. It resolves one way or the other: a word ending in the
literal `.gguf` or holding a `/` is a file path, and anything else is a key of `models`. A key
spelled like a path can therefore never be selected by name, which is left as it is: keys are for
names, and nothing forbids the spelling. With no word given, a `models` holding one entry
resolves to that entry, and one holding several is refused, because nothing picks between them.

The model is opened in `build`, not here: `08` maps it there, after this spec has checked the
path is a file. An entry's `path` is checked for existence here
because a config that points nowhere should fail at the point a person wrote it, not at the point
a request arrives. The resolved model reaches `build` and `start` through the endpoint, and
`/health` names it. A first run that writes the starter file exits 1 because nothing is
listening: the file was created, but the server did not start, so a script must not read that run
as a success.

The starter file (`US-001`) and the path word (`US-004`) are here rather than in two specs: both
decide the same question, whether a file exists and which model it names, and neither has a
consumer of its own until the engine arrives.

Every file question goes through `fs` (lde registry, `^0.3.0`), which is what makes the rest of
this spec small: it gives `exists`, `isfile`, `isdir`, `mkdirAll`, `read` and `write` on LuaJIT,
so neither `ffi` for a directory nor the trap below is needed. It brings `path` and `utf8` with it
as its own dependencies, and it ships LuaCATS annotations and no `.d.tl`, so it needs a declaration
the way `lde-test` does. `tltp.json_decode` stays the decoder, as in `03`, so `json` is still not imported
directly, and the starter file's text is written literally, because `tltp.json_encode` treats an
empty table as an array and would write `"models":[]`.

The trap that `fs` avoids: `io.open` succeeds on a directory on Linux and fails on the first
read, so a hand-rolled existence check has to read, not just open, or a `path` pointing at a
directory passes.

Every rule below is proved under `lde run test`, in-process. A test that read the real
`~/.config` would write the starter file into the person's home and pass or fail on what happens
to be there, so `run` takes the two lookup variables as a table (`FR-001`) rather than reading them
itself, the way `03` takes a start function in place of binding a socket.

The checks run in this order: `FR-003` (no file at a lookup path, so the starter file is
written), `FR-002` (the chosen file reads), `FR-005` (it decodes), `FR-006` (unknown top-level
keys), `FR-007` (values), `FR-008` (entry paths), `FR-009` (the path word's file), `FR-010` (model
resolution). The first fault found is the one line written.

## User stories

- US-001 (P1) As a first-time user, I want `tllm serve` with no config file to leave me a file to fill in, so that I know where it belongs and what it looks like.
- US-002 (P1) As the engine author, I want serve to bind the host and port my file sets unless a flag overrides it, so that my settings live in one file.
- US-003 (P1) As a user whose config is wrong, I want one line naming the key at fault and exit code 1, so that I can fix it without reading source or watching a server bind on a guess.
- US-004 (P2) As a user trying a model, I want to point serve at a `.gguf` file that is not in my config, so that it is accepted and its path checked before the server starts.

## Functional requirements

- FR-001 (US-002) WHEN `serve` runs without `--config`, THE SYSTEM SHALL read the first file that exists among `$XDG_CONFIG_HOME/tllm/config.json`, when `XDG_CONFIG_HOME` holds a non-empty value, and `$HOME/.config/tllm/config.json` in that order, taking those variables from `run`'s optional fifth parameter `environment`, a table of strings that defaults to a table holding `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `HOME` read from the process environment, and skipping a lookup path whose variable is absent or empty.
- FR-002 (US-002) WHEN `--config PATH` is given, THE SYSTEM SHALL read that file and no other, and IF the file chosen by `FR-001` or by `--config` cannot be opened or cannot be read, THEN THE SYSTEM SHALL write one line naming the path through the stderr writer and return exit code 1 without calling a start function, without trying the next lookup path and without writing a starter file.
- FR-003 (US-001) IF no file exists at either lookup path and the model word is not a path, THEN THE SYSTEM SHALL create the missing directories, write the starter file at the first lookup path, write one line naming that path through the stderr writer, and return exit code 1 without calling a start function, and IF those directories or that file cannot be created, THEN THE SYSTEM SHALL write one line naming the path through the stderr writer and return exit code 1 without calling a start function, and IF no `--config` was given and the model word is not a path and no lookup path exists because both variables are absent or empty, THEN THE SYSTEM SHALL write one line saying so through the stderr writer and return exit code 1 without calling a start function.
- FR-004 (US-001) THE SYSTEM SHALL write the starter file with the text `{"host":"127.0.0.1","port":8080,"models":{}}` and nothing else, with `models` written as an empty object.
- FR-005 (US-003) IF a read file does not decode to a table, THEN THE SYSTEM SHALL write one line naming the path through the stderr writer and return exit code 1 without calling a start function.
- FR-006 (US-003) IF a read file holds a top-level key other than `host`, `port`, `models`, `sessions`, `kv_cache`, `context`, `workers`, `ubatch`, `vram` and `profile`, THEN THE SYSTEM SHALL write one line naming that key through the stderr writer and return exit code 1 without calling a start function.
- FR-007 (US-003) IF `host` is set and is not four dot-separated decimal parts each from 0 to 255 with no leading zero on a part of more than one digit, or `port` is set and is not an integer from 1 to 65535, or `models` is set and is not a table whose every value is a table holding a `path` string and no key other than `path`, or `sessions` is set and is not an integer of one or more, or `kv_cache` is set and is not `f32`, `f16`, `q8_0` or `q4_0`, compared case-sensitively, or `context` is set and is neither `false` nor a table holding a `path` string of one or more characters and a `bytes` integer of one or more and no key other than those two, or `ubatch` is set and is not an integer of one or more, or `vram` is set and is not an integer of one or more, or `profile` is set and is not text, checked after `vram`, THEN THE SYSTEM SHALL write one line through the stderr writer naming what is at fault, `host` then `port` then `models` then `sessions` then `kv_cache` then `context` then `ubatch` then `vram` then `profile` when several are, and the model name for a fault inside an entry, whether or not the same key was given as a flag, and return exit code 1 without calling a start function.
- FR-008 (US-003) IF a model entry's `path` does not name an existing readable file, or names a directory, THEN THE SYSTEM SHALL write one line through the stderr writer naming the model and its path and return exit code 1 without calling a start function.
- FR-009 (US-004) WHEN the model word ends in the literal `.gguf`, compared case-sensitively, or holds a `/`, THE SYSTEM SHALL treat it as a file path and start with it whether or not a config file was read, and IF that path does not name an existing readable file, or names a directory, THEN THE SYSTEM SHALL write one line through the stderr writer naming it and return exit code 1 without calling a start function.
- FR-010 (US-003) IF no model can be resolved, because the model word is given, is not a path and is not a key of `models`, or because no word is given and `models` holds no entry, THEN THE SYSTEM SHALL write one line through the stderr writer naming the word when one was given and saying no model is configured when none was, and return exit code 1 without calling a start function, and IF no word is given and `models` holds more than one entry, THEN THE SYSTEM SHALL write one line through the stderr writer naming every configured model in ascending byte order and return exit code 1 without calling a start function.
- FR-011 (US-002) WHEN a file is read, THE SYSTEM SHALL pass `start` the flag's value for each of `host` and `port` that was given, the file's value for each that was not given and the file sets, and `127.0.0.1` and `8080` for each that neither sets, pass the file's `sessions`, 4 when the file sets none, pass 4 when no file is read, pass the file's `kv_cache`, `f16` when the file sets none and `f16` when no file is read, pass the `context` object's `path` and `bytes` as `context_path` and `context_bytes`, `$XDG_DATA_HOME/tllm/contexts` and 1073741824 when the file sets no `context` or no file is read and `XDG_DATA_HOME` holds a non-empty value, `$HOME/.local/share/tllm/contexts` and 1073741824 when `XDG_DATA_HOME` is absent or empty and `HOME` holds a non-empty value, and nil for both when the file sets `"context":false` or when neither variable holds a value, pass the file's `workers`, 1 when the file sets none and 1 when no file is read, pass the file's `ubatch`, 512 when the file sets none and 512 when no file is read, pass the file's `vram`, nil when the file sets none and nil when no file is read, pass the file's `profile`, nil when the file sets none and nil when no file is read, then return the integer `start` returns.
- FR-012 (US-001) THE SYSTEM SHALL declare `fs` at `^0.3.0` in `lde.json`, hold its declaration at `types/fs.d.tl` as `02` and `03` hold theirs, and take every existence check, the directory creation and the starter file's write through it, so that nothing in the config path needs `ffi` or `os.execute`.
  Not testable: `lde.json` and the imports are read in the tree, and `lde run check` exits 0.

## Non-goals

- No engine: `serve` maps the model through `weights.load` under `08` and reads no weight, and the 404 answer and the 500 answer are unchanged.
- No per-model settings beyond `path`, no `TLLM_CONFIG` variable, no router behaviour (loading on request, a resident-model limit, idle sleep) and no `/v1/models` route.

## Tests

- T-001 (FR-001, FR-002): `config lookup reads xdg then home, --config replaces it`, proves with `environment` naming an XDG directory and a HOME directory, one file in each holding the entry `m` whose `path` names a temp file, that `run({"serve", "m"}, ...)` lets the XDG file's port reach `start`, that an XDG directory holding no file lets the HOME file's port reach it, and that `--config` naming a third file holding `m` with an existing `path` and its own port lets that port reach it while the XDG file is ignored.
- T-002 (FR-002): `an unreadable config path is refused`, proves `--config` naming a file that does not exist and `--config` naming a directory each return 1, write one stderr line holding that path, call no start function and write no file.
- T-003 (FR-003, FR-004): `first run writes the starter config and exits one`, proves with both directories empty and no model word that `run({"serve"}, ...)` returns 1, calls no start function, writes one stderr line holding the XDG path, and that the file appearing there reads exactly `{"host":"127.0.0.1","port":8080,"models":{}}`, that an empty `XDG_CONFIG_HOME` puts the file at the HOME path instead, and that a lookup path whose parent is a regular file, so its directory cannot be created, returns 1 with one stderr line and no start call, and that an `environment` holding neither variable returns 1 with one stderr line and no start call.
- T-004 (FR-005): `text that is not json is refused`, proves a file holding `{` returns 1, writes one stderr line holding the path and calls no start function.
- T-005 (FR-006): `an unknown key is refused`, proves a file holding `prot` beside a valid `port` returns 1, writes one stderr line holding `prot` and calls no start function.
- T-006 (FR-006, FR-007): `bad config values are refused`, proves `host` `localhost`, `port` `0`, `port` `"8080"`, `models` as a string, `sessions` `0` and `sessions` `"2"` each return 1 with one stderr line and no start call, that an entry `alpha` holding no `path` and an entry `alpha` holding `path` beside an extra key each return 1 with one stderr line holding `alpha` and no start call, that a file holding both `prot` and `port` `0` writes one stderr line holding `prot`, and that a file holding both `host` `localhost` and `port` `0` writes one stderr line holding `host`.
- T-007 (FR-008): `a configured model with no file is refused`, proves an entry whose `path` does not exist and an entry whose `path` names a directory each return 1, write one stderr line holding the model name and the path, and call no start function.
- T-008 (FR-009): `a model path is served without a config`, proves `run({"serve", path}, ...)` with `path` naming a temp file, and an `environment` naming two empty directories, calls `start` with `127.0.0.1`, `8080` and `sessions` 4 and writes no config file, and that the same call with a path naming a missing file returns 1 with one stderr line and no start call, and that a word naming a temp file called `up.GGUF` is treated as a name, so it writes one stderr line holding that word and calls no start function, and that a run with `--config` naming a valid file holding `m` with an existing `path`, an `environment` holding neither variable and `m` as the model word still calls `start`.
- T-009 (FR-010): `a model name must be configured`, proves with a file holding `m` that `run({"serve", "typo"}, ...)` returns 1 with one stderr line holding `typo` and no start call, that `run({"serve", "m"}, ...)` calls `start`, and that a file whose `models` holds no entry and no word returns 1 with one stderr line and no start call.
- T-010 (FR-011): `flags beat the file and the file beats the defaults`, proves `run({"serve", "m"}, ...)` with a file setting `host` and `port` and an `m` entry whose `path` exists reaches `start` with the file's values including `sessions` 2, that `--port 9090` overrides only the port while the file's host still applies, that a file setting neither passes `127.0.0.1`, `8080` and `sessions` 4 for the same call, and that a file whose `host` is `localhost` still returns 1 with one stderr line when `--host 127.0.0.1` is given.

## Open questions

- None.

## Amendments

- None.
