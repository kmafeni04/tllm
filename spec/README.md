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
- Source is Teal only, run on LuaJIT through lde, except the lde build script at the package root, which lde runs only when it is named `build.lua`.
- The core stays small: a feature a plugin can carry does not belong in the core.
- A plugin adds knowledge, never policy about resources. Knowledge is what a model family looks
  like, how to sample, how to tokenize, how to render a template, which route answers, which backend
  to pick. Resource policy is the memory plan, the expert tier, the cache and its snapshot format,
  the scheduler, the graph and the refusals: those are the engine. A spec that hands any of them to
  a plugin is refused.
- A refusal names its seam. When the engine cannot load or run something, the message names the keys
  or shapes it matched on and the plugin kind that could claim it, so a refusal is a boundary a
  reader can act on and not a wall.
- The plugin api is a budget, not a surface. A spec that adds a hook to it says why the existing
  hooks cannot carry the feature.
- Refusing is allowed, foreclosing is a defect. The engine may refuse a capability it has not built,
  but it must not build a shape that makes that capability unbuildable later. Where a file declares a
  capability in its own metadata, the core passes the general thing rather than the narrow one, and
  refuses by name rather than dropping.
- Every `.tl` file typechecks and is formatted.
- The project must be memory safe, no memory leaks, memory corruption bugs, use after frees, etc.
- A spec that adds a config key names the level that owns it before it names what the key means.
- A spec cites only specs that already exist in this folder, in its `**Depends on**` line, in a
  requirement, in an amendment or in prose. A spec that has not been written has no number yet, so a
  forward reference to one is a guess about ordering that breaks when the plan changes: name the thing
  in plain words instead.
