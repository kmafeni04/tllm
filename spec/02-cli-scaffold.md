# 02: CLI scaffold with serve and bench

**Status**: done   **Depends on**: 01

## Problem & context

The package today is an entry point that returns an empty table; there is no command line at all. Two
commands are planned, `serve` and `bench`, and every later spec (config loading, engine, HTTP surface,
benchmarking) needs a stable place to hang its options. Building that place now, while there is
nothing to execute, keeps the engine specs about the engine: this spec adds the command surface and
the dispatcher with its exit codes. `bench`'s report and the files it reads arrive in `29`, and the
config file, its schema and the JSON dependency that decoding it needs arrive in `04`.

The command surface, with targs (registry, 0.5.0) used only to turn the argument list into a parsed
command. Its `pparse` is the entry point: it returns errors instead of printing or exiting, with a
`usage` kind for bad input and an `exit` kind for `--help`. The CLI therefore calls `pparse` and never
`parse`, which writes to the real streams and exits. For both kinds the CLI writes `err.message` and
never `Parser:format_error`, whose `usage` rendering is two lines.

```
tllm serve [MODEL] [--config FILE] [--host HOST] [--port PORT] [--threads N]
tllm bench MODEL [--config FILE] [--prefill-tokens N] [--decode-tokens N] [--dump-routing FILE] [--threads N]
```

`--port` takes an integer from 1 to 65535 and the two token counts take positive integers. `serve`
falls back to `127.0.0.1` and `8080` for `--host` and `--port`, which `03/FR-007` requires, and
`bench`'s defaults are `29`'s.

`src/init.tl` is the entry point and the only one: it builds the dispatcher's three arguments, passes
them to `run` in `src/cli.tl`, then calls `os.exit(code, true)` with the integer it returns. Those
arguments are the global `arg`, which targs reads from index 1, so the program name in `arg[0]` is
never a command, and two writers that put one line each on `io.stdout` and `io.stderr`. lde ignores
what a script returns, so `os.exit` is what sets the process status.
Since the package is an application rather than a library, `src/init.tl` returns no table and
`require("tllm")` runs the CLI; `01/FR-008` was amended to say so.

Exit codes: `1` for a usage error with one line on stderr, and for a bare `tllm` with no command, which
puts the same usage text `--help` prints on stdout and still exits `1` rather than `0`; `0` for
`--help`, with the usage text on stdout. `bench`'s own exit codes are `29`'s.

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
- FR-003 (US-001) THE SYSTEM SHALL accept an optional `MODEL` and the options `--config`, `--host`, `--port` and `--threads` for `serve`, a required `MODEL` and the options `--config`, `--prefill-tokens`, `--decode-tokens`, `--dump-routing` and `--threads` for `bench`, and a required `MODEL` and the options `--config` and `--reference FILE` for `reference`.
- FR-004 (US-001) WHEN `bench` is invoked, THE SYSTEM SHALL write the report and return the exit codes `29/FR-004` to `29/FR-008` describe, replacing the refusal line and exit code `2` this spec first shipped.
  Not testable: the report is `29`'s, and `29/T-001` to `29/T-004` prove every line and exit code of it.
- FR-005 (US-002) IF no command is given, THEN THE SYSTEM SHALL write the usage text `--help` prints to stdout and return exit code 1. IF the command is neither `serve`, `bench` nor `reference`, THEN THE SYSTEM SHALL write one line saying what is wrong to stderr and return exit code 1.
- FR-006 (US-002) IF targs rejects the argument list, THEN THE SYSTEM SHALL write the returned error's `message` to stderr as one line and return exit code 1. IF an argument is otherwise invalid for the chosen command, THEN THE SYSTEM SHALL write one line naming the offending flag or value to stderr and return exit code 1.
- FR-007 (US-001) WHEN `--help` is passed to `tllm` or to either command, THE SYSTEM SHALL write the usage text through the stdout writer and return exit code 0.

## Non-goals

- No config reading in this spec: the JSON dependency, the config file, its schema and its default location arrive in `04`.
- No engine work in this spec: `serve` loads and maps a model under `08`, and `bench`'s measurements arrive in `29`.
- No shell completions, no `--version`, and no logging beyond the lines above.

## Tests

- T-001 (FR-002): `dispatcher returns a code through supplied writers`, proves requiring `tllm.cli` and calling `run` returns an integer for both a failing and a refusing argument list, puts each message in the supplied writer, and leaves the process streams untouched.
- T-003 (FR-005): `a bare invocation prints help and exits one`, proves no command gives exit 1, the usage text on stdout and nothing on stderr; `an unknown command is a usage error`, proves an unknown command gives exit 1, one stderr line and nothing on stdout.
- T-004 (FR-006): `bad arguments exit one`, proves `bench` without a model, `--port 0` and `--prefill-tokens 0` each give exit 1, one stderr line and nothing on stdout.
- T-005 (FR-007): `help returns zero`, proves `--help` and `serve --help` each return 0, put the usage text in the stdout writer, and write nothing to stderr.
- T-006 (FR-003): `every command accepts its options`, proves `serve m --config F --host 127.0.0.1 --port 9090` and `bench m --config F --prefill-tokens 8 --decode-tokens 8 --dump-routing R`, with F a file that does not exist, each return 1 with one stderr line holding F and no stdout line, so parsing accepted every option and the call reached the file, and that no start function ran for `serve`.

## Open questions

- None.

## Amendments

- None.
