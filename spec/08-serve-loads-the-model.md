# 08: Open the model at serve start, name it in /health, and resolve one configured entry

**Status**: done   **Depends on**: 04, 06

## Problem & context

`04` stops at a file check: `serve` confirms the model path exists and binds a port. `05` and `06`
gave the project a GGUF reader and a mapped handle, and `serve` calls neither. So `tllm serve
model.gguf` answers `/health` for a file it never opened. A file that is not GGUF, a file truncated
halfway, a tensor table that does not add up: all pass the existence check, and the client learns
about it from a failed request long after the port opened. Nothing anywhere says which model is
live, and no route can reach a weight.

This spec makes `serve` open the model before it binds, name it, and hold it where a later route can
find it.

The open is one call. `weights.load(path)` reads the header itself, maps the file and returns a
handle holding every tensor's pointer (`06`), so `serve` re-reads nothing, dequantises nothing and
looks at no tensor. A failure is the loader's to report: every reason it returns names what is
wrong.

Two decisions the code did not make. First, where the open sits: it is in `build`, not in `cli`.
`04` proves that the model word resolves to a path and that every refusal before a start happens in
the dispatcher; its cases inject a start function over a file that is not GGUF, and they keep
passing because the file is opened past the seam they test. It also means the server's own
constructor is the one thing that maps a file, so a route built in a test is built over a real
model. Second, `04/FR-010` leaves one hole, and this spec closes it: no word given and `models`
holding an entry starts the server, and nothing says which entry. One entry is the model, more than
one is refused with a line naming them.

The endpoint record that `03` passes to `build` and `start`, and that `04` fills with host and port,
grows two more members, `name` and `path`. The name is the configured key when the model word
selected a configured entry, and the word itself when the word was a path. `build` gains a failure:
nil and the loader's reason. `start` writes that reason as one line and returns 1 before a socket is
opened, which is the path `03/FR-009` already takes for a bind failure, so a bad file costs a line
and an exit code instead of a bound port.

```teal
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")
local weights = require("tllm.weights")

local record serve
  record Model
    name: string
    path: string
    handle: weights.Handle
    tokenizer: tokenizer.Tokenizer
    backend: compute.Backend
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
  end

  model_of: function(tltp.Server): Model | nil
  build: function(Endpoint): tltp.Server | nil, string | nil
  start: function(Endpoint, function(string)): integer
end
```

`tltp.Server` is a closed record with no field to put a model in, so the model lives in a table
declared in `tllm.serve` whose keys are the servers `build` returned, and `model_of` reads it back.
That table is how a later endpoint reaches a tensor, and it is what tells a kept handle from a
dropped one: the mapping outlives the handle on its own, because `06/FR-007` makes `release` an
explicit call and nothing collects the mapping for us. No release is ever called, so the model stays
mapped until the process ends.

## User stories

- US-001 (P1) As the engine author, I want `serve` to hold a mapped model before it binds, so that later endpoints have weights to work from.
- US-002 (P1) As a user pointing serve at a file that is not a model, I want one line and exit code 1 before the port opens, so that I hear it at start and not from a failed request.
- US-003 (P1) As a client, I want `GET /health` to name the model, so that I know what the server is serving without asking.
- US-004 (P2) As a user with one model configured, I want plain `tllm serve` to serve it, so that I do not repeat a name the file already holds.

## Functional requirements

- FR-001 (US-001) WHEN `require("tllm.serve").build({host = "127.0.0.1", port = 9090, name = "m", path = P, sessions = 4})` is called with `P` naming a GGUF file, THE SYSTEM SHALL return a `tltp.Server` whose config holds port 9090 and `bind_host` `127.0.0.1` and nil as its second return value.
- FR-002 (US-002) IF `weights.load(P)` returns a failure, THEN THE SYSTEM SHALL return nil and that failure as the reason.
- FR-003 (US-002) WHEN `start` is called with an endpoint whose `path` the load refuses, THE SYSTEM SHALL write one line through the stderr writer holding the loader's reason, with its trailing whitespace stripped and one newline appended, and return exit code 1.
- FR-004 (US-003) WHEN a GET request for `/health` is answered by a server `build` returned, THE SYSTEM SHALL answer status 200, content type `application/json`, and a body that decodes to an object whose members are `status` with the value `ok` and `model` with the value the endpoint's `name`.
- FR-005 (US-003) WHEN the model word names a configured entry, THE SYSTEM SHALL call the start function with an endpoint whose `name` is that word and whose `path` is that entry's `path`.
- FR-006 (US-003) WHEN the model word is a path, THE SYSTEM SHALL call the start function with an endpoint whose `name` and `path` are both that word.
- FR-007 (US-004) WHEN no model word is given and the config holds exactly one entry, THE SYSTEM SHALL call the start function with an endpoint whose `name` is that entry's key and whose `path` is that entry's `path`.
- FR-008 (US-002) IF no model word is given and the config holds more than one entry, THEN THE SYSTEM SHALL refuse, as `04/FR-010` states.
- FR-009 (US-001) WHEN `build` returns a server, THE SYSTEM SHALL hold that server's model in a table declared in `tllm.serve` whose keys are the servers `build` returned, and call no release on its mapping.
- FR-010 (US-001) WHEN `model_of(server)` is called with a server `build` returned, THE SYSTEM SHALL return the model it holds, whose `name` is that endpoint's `name`, whose `path` is that endpoint's `path`, whose `handle` is the handle `weights.load` returned for that path, and whose `tokenizer` and `backend` are the tokenizer and the compute backend `build` loaded from that file.
- FR-011 (US-001) IF `model_of` is called with a server `build` did not return, THEN THE SYSTEM SHALL return nil.

## Non-goals

- No maths, tokenizer, sampler or generation route: nothing reads a weight, and `serve` only maps the file, names it and holds it.
- No unload, reload or release, no second resident model, and no `bench` change: the mapping lives until the process ends, and `bench` still prints not implemented and loads nothing.
- No examination beyond what `weights.load` already refuses: no architecture, no metadata key and no tensor is read by `serve`.

## Tests

- T-001 (FR-001, FR-002, FR-004, FR-009, FR-010, FR-011): `a gguf is mapped at build and a bad file is refused`, proves with a one-tensor GGUF fixture and `name = "m"` that `build` returns a server and nil, that `model_of` on that server returns a model whose `path` is the fixture's path, whose `name` is `m` and whose `handle.tensors` holds the fixture's single tensor by name, that `model_of` on a `tltp.Server` the test built itself returns nil, that the server's `/health` answers 200 JSON decoding to `{status = "ok", model = "m"}`, that the fixture's path appears in `/proc/self/maps`, that a file holding `GGML` bytes returns nil with a reason holding its path, and that a path naming no file returns nil with a reason holding that path.
- T-002 (FR-005, FR-006, FR-007, FR-008): `serve resolves one model and names it`, proves with an injected start that a word naming a configured entry reaches it with that word as `name` and the entry's path as `path`, that a word that is a path reaches it with that word as both, that a config holding one entry and no word reaches it with that entry's key and path, and that a config holding the two entries `Zeta` and `alpha` and no word returns 1, calls no start function, and writes one stderr line holding both keys with `Zeta` before `alpha`.
- T-003 (FR-003): `a model that cannot be opened is refused`, proves that `start` with an endpoint whose `path` names a file that is not GGUF returns 1 and writes one line through the stderr writer that holds that path and ends in a single newline.

## Open questions

- None.

## Amendments

- None.
