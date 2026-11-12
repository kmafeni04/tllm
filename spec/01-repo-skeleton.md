# 01: Repository skeleton and toolchain

**Status**: done   **Depends on**: -

## Problem & context

The repository holds a hello-world placeholder, a single typecheck script, a `.gitignore` entry for
`target/`, and a `spec/README.md` that states the project's constraints. There is no test wiring and
no formatter. Every later spec in this engine will claim something works, and nothing in the
repository currently lets a person confirm or refute that claim.

This spec makes the repository verifiable: one command to typecheck, one to format, a test command
wired so that the first test needs no further setup, an entry point that starts the same CLI a user
runs, and the binding rules written down once so that contributors and agents stop guessing
them. The engine itself is not part of this spec; it arrives in later ones. Writing those rules
down is documentation work rather than toolchain work, and it rides along because the rules are what
every later spec is measured against.

This spec adds no test files, so `lde run test` has nothing to run until a later spec adds one: lde's
runner reports `No tests found` and exits non-zero when no `*.test.lua` file exists, and that is the
runner's behavior rather than something the script can work around. The script exists so that later
specs only write the test files.

The commands are the deliverable: `lde run check`, `lde run test` and `lde run format`, defined
in `lde.json` as package scripts.

## User stories

- US-001 (P1) As the engine author, I want one command to typecheck and one to format the code, plus a test script ready for the first test, so that every later change is verified the same way.
- US-002 (P1) As a contributor, I want the package to start the CLI when it runs, so that the command line is the application's single entry point.
- US-003 (P1) As a contributor, I want the project's binding rules written in the spec folder, so that what I build matches the project without asking.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL declare in `lde.json` the name `tllm`, the version `0.1.0`, the description `An LLM inference engine written in Teal for LuaJIT`, the authors list `["Leonard Mafeni <komemafeni944@gmail.com>"]`, and the scripts `check`, `test` and `format`, where `test` runs lde's built-in test runner with `--jobs 1`.
  Not testable: `lde.json` is the artifact; read the file.
- FR-002 (US-001) THE SYSTEM SHALL put `target` on the Teal include path in `tlconfig.lua`.
  Not testable: `tlconfig.lua` is the artifact; read the file.
- FR-003 (US-001) THE SYSTEM SHALL ignore `target/` in `.gitignore`.
  Not testable: `git check-ignore target/` reports it, and `git ls-files` never lists it.
- FR-004 (US-001) WHEN the check script runs, THE SYSTEM SHALL typecheck every `.tl` file in the project except those under `target/`, chosen at run time, and exit with status 0 when every one of them typechecks.
  Not testable: `lde run check` exits 0 after checking every file.
- FR-005 (US-001) IF any `.tl` file in that set holds a type error, THEN THE SYSTEM SHALL exit the check script with a non-zero status.
  Not testable: a deliberate type error makes `lde run check` exit non-zero.
- FR-006 (US-001) THE SYSTEM SHALL declare a `format` script that runs the cerulean formatter, `lde x rocks:cerulean --indent 2 --line-length 80`, over every `.tl` file in the project except those under `target/`, with the file list chosen at run time.
  Not testable: `lde.json` holds the script, and `lde run format` formats every `.tl` file.
- FR-007 (US-001) WHEN the format script runs on an already formatted tree, THE SYSTEM SHALL exit with status 0 and leave every `.tl` file in the project except those under `target/` byte-identical.
  Not testable: two `lde run format` runs leave `git diff` empty.
- FR-008 (US-002) THE SYSTEM SHALL make `src/init.tl` the application entry point, not a module: it runs the CLI and exits with its code, returns no table, and `require("tllm")` is no longer a contract.
  Not testable: `lde run -- serve` runs the CLI, and requiring the package runs the same entry point.
- FR-009 (US-003) THE SYSTEM SHALL hold in `spec/README.md` a `## Constraints` section stating that declarations are local with globals only for external library type definitions, that dependency sources are unrestricted, that source is Teal on LuaJIT, that the core stays small with features carried by plugins, and that every Teal file typechecks and is formatted.
  Not testable: `spec/README.md` is the artifact; read the file.

## Non-goals

- No engine behavior: no model reading, no kernels, no tokenizer, no sampling, no sessions, no storage tiering.
- No plugin API, and no core module beyond the CLI dispatcher and the entry point that runs it.
- No per-file check or format invocation: both gates cover the whole project.
- No type definitions for external libraries and no documentation generation.

## Tests

- None: every requirement here is scaffolding (manifest fields, script declarations, file content), proved by reading the artifact or by running the command by hand; this spec writes no test files.

## Open questions

- None.

## Amendments

- None.
