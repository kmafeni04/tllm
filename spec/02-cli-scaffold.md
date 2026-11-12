# 02: CLI scaffold with serve and bench

**Status**: done   **Depends on**: 01

## Problem & context

The package today is an entry point that returns an empty table; there is no command line at all. Two
commands are planned, `serve` and `bench`, and every later spec (config loading, engine, HTTP surface,
benchmarking) needs a stable place to hang its options. Building that place now, while there is
nothing to execute, keeps the engine specs about the engine: this spec adds the command surface, the
dispatcher and its exit codes, and stops `bench` at a single "not implemented" line that echoes
the values it was given. It reads no file: the config file, its schema and the JSON dependency that
decoding it needs arrive in a later spec, so `--config` is parsed and never opened.

The command surface, with targs (registry, 0.5.0) used only to turn the argument list into a parsed
command. Its `pparse` is the entry point: it returns errors instead of printing or exiting, with a
`usage` kind for bad input and an `exit` kind for `--help`. The CLI therefore calls `pparse` and never
`parse`, which writes to the real streams and exits. For both kinds the CLI writes `err.message` and
never `Parser:format_error`, whose `usage` rendering is two lines.

```
tllm serve [MODEL] [--config FILE] [--host HOST] [--port PORT]
tllm bench MODEL [--config FILE] [--prefill-tokens N] [--decode-tokens N]
```

`--port` takes an integer from 1 to 65535 and the two token counts take positive integers. `bench`
applies no default: a value the user did not give is simply absent, and its defaults belong to the
config spec. `serve` falls back to `127.0.0.1` and `8080` for `--host` and `--port`, which
`03/FR-007` requires.

The refusal line names the command, then, when at least one value was given, a parenthesised list
holding each given value as its label, a space and the value, in this order: `model`, `config`,
`host`, `port`, `prefill-tokens`, `decode-tokens`, joined by `, `. `tllm bench --prefill-tokens 512
qwen3.8-27b` therefore prints `bench: not implemented (model qwen3.8-27b, prefill-tokens 512)`.

`src/init.tl` is the entry point and the only one: it builds the dispatcher's three arguments, passes
them to `run` in `src/cli.tl`, then calls `os.exit(code, true)` with the integer it returns. Those
arguments are the global `arg`, which targs reads from index 1, so the program name in `arg[0]` is
never a command, and two writers that put one line each on `io.stdout` and `io.stderr`. lde ignores
what a script returns, so `os.exit` is what sets the process status.
Since the package is an application rather than a library, `src/init.tl` returns no table and
`require("tllm")` runs the CLI; `01/FR-008` was amended to say so.

Exit codes: `1` for a usage error with one line on stderr; `0` for `--help`, with the usage text on
stdout; `2` for the refusal, with one line on stdout.

Tests live in `tests/*.test.tl` and run under `lde run test`. They need a Teal declaration for
`lde-test`, which ships LuaCATS types and no `.d.tl`: add the dev dependency with
`lde add --dev gh:lde-test@lde-org/lde` for the types, write `types/lde-test.d.tl` by hand after the
one in `../targs/types/lde-test.d.tl` or generate it with tlcat, and put `types` on `include_dir` in
`tlconfig.lua`. `targs` ships Teal sources and needs no declaration file.

## User stories

- US-001 (P1) As a person trying `tllm`, I want `serve` and `bench` to accept their options and tell me what they were given, so that the command line exists and the surface is settled before the engine arrives.
- US-002 (P1) As a person getting the flags wrong, I want one line saying what is wrong and a non-zero code, so that I can fix the call without reading source.
- US-003 (P2) As a contributor, I want commands to return an exit code and write through supplied writers instead of exiting or printing directly, so that every rule here is testable without spawning a process.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL declare in `lde.json` the dependency `targs` at `^0.5.0`, hold the dispatcher in `src/cli.tl` exporting `run`, and keep `src/init.tl` as the entry point that calls `run` with the argument list and the two writers described above, then calls `os.exit` with the returned code, so that `lde run -- bench qwen3.8-27b` runs the `bench` command on the process arguments and exits with its code. `serve`'s process check is `03/FR-006`.
  Not testable: `lde.json`, the file layout and the process status are checked by hand, running `lde run -- bench qwen3.8-27b` and reading the line it prints and the shell's exit status.
- FR-002 (US-003) WHEN `require("tllm.cli").run(args, write_out, write_err)` is called, where `args` is the command line without the program name (`{"serve", "qwen3.8-27b"}`) and the writers take one string ending in a newline, THE SYSTEM SHALL return an integer exit code, send every message through `write_out` or `write_err`, and never terminate the process.
- FR-003 (US-001) THE SYSTEM SHALL accept an optional `MODEL` and the options `--config`, `--host` and `--port` for `serve`, and a required `MODEL` and the options `--config`, `--prefill-tokens` and `--decode-tokens` for `bench`.
- FR-004 (US-001) WHEN `bench` is invoked with valid arguments, THE SYSTEM SHALL write one line to stdout holding the command name, then `: not implemented`, then, when at least one value was given, a parenthesised list of the given values in the order and shape described above, and return exit code 2.
- FR-005 (US-002) IF no command is given or the command is neither `serve` nor `bench`, THEN THE SYSTEM SHALL write one line saying what is wrong to stderr and return exit code 1.
- FR-006 (US-002) IF targs rejects the argument list, THEN THE SYSTEM SHALL write the returned error's `message` to stderr as one line and return exit code 1. IF an argument is otherwise invalid for the chosen command, THEN THE SYSTEM SHALL write one line naming the offending flag or value to stderr and return exit code 1.
- FR-007 (US-001) WHEN `--help` is passed to `tllm` or to either command, THE SYSTEM SHALL write the usage text through the stdout writer and return exit code 0.

## Non-goals

- No config reading: the JSON dependency, the config file, its schema and its default location are a later spec, so `--config` is never opened and no file is read or checked.
- No engine work: no command measures a token rate, and `bench` opens no port, answers no request and loads nothing. `serve` loads and maps a model under `08`, and `bench` stops at the refusal line.
- No shell completions, no `--version`, and no logging beyond the lines above.

## Tests

- T-001 (FR-002): `dispatcher returns a code through supplied writers`, proves requiring `tllm.cli` and calling `run` returns an integer for both a failing and a refusing argument list, puts each message in the supplied writer, and leaves the process streams untouched.
- T-002 (FR-003, FR-004): `bench echoes the values it was given`, proves a model with `--config`, `--prefill-tokens` and `--decode-tokens` is accepted and each given value appears in the stdout line, exiting 2.
- T-003 (FR-005): `missing and unknown commands are usage errors`, proves no command and an unknown command each give exit 1, one stderr line and nothing on stdout.
- T-004 (FR-006): `bad arguments exit one`, proves `bench` without a model, `--port 0` and `--prefill-tokens 0` each give exit 1, one stderr line and nothing on stdout.
- T-005 (FR-007): `help returns zero`, proves `--help` and `serve --help` each return 0, put the usage text in the stdout writer, and write nothing to stderr.

## Open questions

- None.

## Amendments

- None.
