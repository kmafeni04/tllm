# Contributing

Thanks for wanting to help. A few rules are here so that the work stays
reviewable.

## Before you start

Open an issue or a pull request that says what you want to change and why,
before doing larger work. If you are driving an agent, it reads
[`AGENTS.md`](AGENTS.md) and this file first.

## What needs a spec

- Bug fixes, refactors, performance improvements and dependency bumps need no
  spec. They go in as they come.
- New features always need one. A feature arrives as a spec in
  [`spec/`](spec/README.md) that says what it does, and the code follows it. A
  pull request that adds a feature with no spec behind it gets asked for the
  spec first.
- Changing what a spec already says is a change to that spec, made in the same
  pull request. If the change is a feature of its own, it gets its own spec.

Read [`spec/README.md`](spec/README.md) for the format and for the constraints
that bind every spec.

## Writing a spec

- Name it `NN-slug.md`, where `NN` is the next free number (`ls spec`).
- Line 3 holds `**Status**: draft   **Depends on**: NN, NN`. Leave it at
  `draft`: the maintainer signs a spec off to `pending`, and the implementation
  moves it to `in-progress` and then `done`.
- Keep the sections in the order the folder uses: `## Problem & context`,
  `## User stories`, `## Functional requirements`, `## Non-goals`, `## Tests`,
  `## Open questions`, `## Amendments`.
- Every functional requirement says what must be true and which story it
  serves, and is written so it can be tested. When a test cannot prove it, the
  requirement carries a `Not testable:` line naming the artifact to read or the
  command to run by hand instead.

## Size of a change

There is no legal cap on the number of lines, but there is a practical one:
every pull request is read by a human. A change too large to get through in a
sitting can be closed without a full review, so keep pull requests small and
verifiable. Split a feature into the spec and the implementation, and keep
unrelated fixes in their own pull requests. One pull request, one thing.

## Commits

The log shows the shape:

- `spec(NN): what the spec is about` for a spec
- `impl(NN): what the implementation does` for the code behind a spec
- an ordinary subject line for everything else, lowercase, saying what
  changed

One topic per commit.

## Before you push

```sh
lde run check   # typecheck every .tl file
lde run test    # run the tests
lde run format  # format every .tl file with cerulean
```

All three pass. Tests are written the way the project wants them: unique and
important, no test that can never fail, no coverage for coverage's own sake. A
function already covered by another test that calls it does not need its own.

The code rules live in [`AGENTS.md`](AGENTS.md). Every declaration is `local`,
every `.tl` file typechecks and is formatted, and `any` is not used.

## Review

The maintainer reviews every change and decides what fits the project. Expect
questions about the spec, the tests and the size of the change rather than
about style, which `lde run format` settles.

## License

The project is MIT licensed. By contributing you agree that your work is
released under the same license.
