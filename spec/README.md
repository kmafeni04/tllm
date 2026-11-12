# tllm

An LLM inference engine written in Teal and run on LuaJIT through lde: a small core, with
features carried by plugins.

## Reading a spec

Each `NN-slug.md` file is one problem. `## Problem & context` says what is wrong and for whom,
the user stories and functional requirements say what must be true once it is built, `## Tests`
is the exact list of test cases the implementer writes, and `## Non-goals` bounds the work.

## Constraints

These rules bind every spec in this folder.

- Every declaration is `local`. A global is allowed only as a type definition for an external
  library.
- Dependencies may come from anywhere: the lde registry, luarocks, git, or a local path.
- Source is Teal only, run on LuaJIT through lde.
- The core stays small: a feature a plugin can carry does not belong in the core.
- Every `.tl` file typechecks and is formatted.
