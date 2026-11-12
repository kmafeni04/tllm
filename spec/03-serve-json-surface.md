# 03: Serve over HTTP with JSON answers

**Status**: done   **Depends on**: 02

## Problem & context

`tllm serve` parses its flags, prints `serve: not implemented (...)` and exits 2. Nothing listens,
so there is no HTTP surface for later specs to add endpoints to, and no agreed shape for a JSON
answer.

This spec turns `serve` into a server: it opens a socket on the host and port the call named,
answers `GET /health` with a JSON body, and answers every failure with one error envelope. The
engine is absent and stays absent: no model is loaded and no completion route exists.

`tltp` 0.3.5 (registry) is the server. Its `mock_request` method answers a request out of the
routing table with no socket, so every answer here is tested in-process under `lde run test`.
Its `self:json(code, body)` helper sets `application/json` and names 200 `OK` in that in-memory
response, while a real answer carries `HTTP/1.1 200 OK`. Bodies are encoded with `tltp.json_encode` and decoded with
`tltp.json_decode`, so this project does not import `json` directly. `tltp` already depends on
`json` 0.1.1 (yyjson-backed), and its first build fetches and compiles yyjson, so the machine that
runs `lde` needs a network connection once.

Three of the package's own answers are not the JSON envelope, and the built server replaces each:
`handle_404` answers a text 404, `default_route` redirects a path ending in `/` with `Found`
before reaching it, and `on_error` answers a text 500 carrying the traceback. The first two are
overridden so that no unrouted request is redirected; `on_error` is overridden so that a route
that throws still answers the JSON envelope, and the traceback it would have logged is recorded
nowhere, which a later spec that adds error logging owns.

Two facts about the pinned release that this spec depends on:

1. `inet_addr` reads a dotted quad and resolves no name, so `--host` takes a dotted IPv4 address
   and nothing else. An invalid quad yields `INADDR_NONE` and binds nothing, hence FR-008.
2. `json` ships no Teal declaration of its own, and lde does not link `tltp`'s `types/` into
   `target/`, so `types/json.d.tl` is copied from the pinned `tltp` source checkout. The
   declaration is needed here because the typechecker follows `tltp`'s own Teal sources.

## User stories

- US-001 (P1) As the engine author, I want `tllm serve` to open a socket and answer requests, so that later endpoints have a real server to be added to.
- US-002 (P1) As a client, I want `GET /health` to answer 200 with a JSON body, so that I can tell the server is up.
- US-003 (P1) As a client sending anything the server cannot answer, I want a JSON error body with one fixed envelope, so that one parser handles every failure.
- US-004 (P2) As a contributor, I want the routes built without a socket, so that the surface is tested in-process like the dispatcher is.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL declare the dependency `tltp` at `^0.3.5` in `lde.json` and hold in `types/` the `json` declaration copied from `tltp`'s `types/`, so that every `.tl` file typechecks without the project importing `json` directly.
  Not testable: read `lde.json` and `types/`, and `lde run check` exits 0.
- FR-002 (US-004) WHEN `require("tllm.serve").build({host = "127.0.0.1", port = 9090})` is called, THE SYSTEM SHALL return a `tltp.Server` whose config holds port 9090 and `bind_host` `127.0.0.1`, answering requests through `mock_request` without opening a socket.
- FR-003 (US-002) WHEN a GET request for `/health` is answered, THE SYSTEM SHALL answer status 200, content type `application/json`, and a body that decodes to an object whose single member is `status` with the value `ok`.
- FR-004 (US-003) WHEN a request matches no route, THE SYSTEM SHALL answer status 404, content type `application/json`, and a body that decodes to an object whose single member is `error`, an object whose members are `message` with the value `not found` and `type` with the value `not_found`, with no redirect answered in its place.
- FR-005 (US-003) WHEN a request for `/health` carries a method other than GET, THE SYSTEM SHALL answer status 404, content type `application/json`, and the body `FR-004` describes.
- FR-006 (US-001) WHEN `tllm serve` runs with valid arguments, THE SYSTEM SHALL build the server with `build`, bind the host and port it was given, answer requests over that socket until the process ends, and hold the process open while doing so, so that the start function returns only when the server could not start.
  Not testable: `lde run -- serve --host 127.0.0.1 --port 9090` then `curl -i 127.0.0.1:9090/health` answers `200 OK`, `curl -i 127.0.0.1:9090/nope` answers `404 NotFound`, and the command does not return until it is stopped.
- FR-007 (US-001) WHEN `run(args, write_out, write_err, start, environment)` dispatches the `serve` command, THE SYSTEM SHALL call `start({host, port}, write_err)` with the given value for each of `host` and `port` that the flag gave, the config file's value for each that the flag did not give and the file sets, and `127.0.0.1` and `8080` for each that neither sets, write nothing to either writer itself except one line through the stderr writer for a config or model failure, and return the integer `start` returns, where `start` is an optional fourth parameter of type `function({host: string, port: integer}, function(string)): integer` that defaults to `require("tllm.serve").start`, and `environment` is an optional fifth parameter, a table of strings that defaults to the process environment.
- FR-008 (US-001) IF `--host` is given and is not four dot-separated decimal numbers each from 0 to 255, with no leading zero on a part of more than one digit, THEN THE SYSTEM SHALL write one line naming the host to stderr, return exit code 1, and call no start function.
- FR-009 (US-001) IF the socket cannot be bound, THEN THE SYSTEM SHALL write the error the server raised as one line through the stderr writer and return exit code 1.
  Not testable: a second `lde run -- serve` on the same port prints the reason and exits 1.
- FR-010 (US-003) WHEN a route raises an error, THE SYSTEM SHALL answer status 500, content type `application/json`, and a body that decodes to an object whose single member is `error`, an object whose members are `message` with the value `internal error` and `type` with the value `internal_error`.

## Non-goals

- No config for `bench`: `bench` reads no file and keeps ignoring `--config` under `02/FR-004`; `serve`'s file, its schema and its model paths arrive in `04`.
- No engine: no model is loaded or opened beyond the existence check `04/FR-008` and `04/FR-009` make, the resolved model is not passed to `start`, and there is no completion route, no streaming, no request body decoding and no error logging beyond the JSON answer.
- No hostname resolution, no TLS, no graceful shutdown and no signal handling: the process ends when it is killed.

## Tests

- T-001 (FR-002): `builds a server for the given host and port`, proves `build({host = "127.0.0.1", port = 9090})` returns a server whose `config.port` is 9090 and whose `config.bind_host` is `127.0.0.1`, and whose routes answer `mock_request` with no socket open.
- T-002 (FR-003): `health answers json`, proves a GET for `/health` answers status `OK`, content type `application/json`, and a body decoding to `{status = "ok"}`.
- T-003 (FR-004): `unrouted paths answer json 404`, proves a GET for an unregistered path and a GET for an unregistered path ending in `/` each answer status `NotFound`, content type `application/json`, and a body decoding to `{error = {message = "not found", type = "not_found"}}`.
- T-004 (FR-005): `wrong method answers json 404`, proves a POST for `/health` answers status `NotFound` with the same decoded body as T-003.
- T-005 (FR-007): `serve starts with the given host and port`, proves `run({"serve", "--host", "127.0.0.1", "--port", "9090"}, out, err, stub, environment)` calls `stub` with `{host = "127.0.0.1", port = 9090}` and `err`, writes nothing through either writer and returns the stub's code, and that `run({"serve", "m"}, out, err, stub, environment)` passes `127.0.0.1` and `8080`, where both calls pass an `environment` naming temporary directories and a config file whose `models` holds the entry `m` with an existing `path`, the second call passing `m` as the model word while the first gives no word and resolves that entry under `04/FR-010`.
- T-006 (FR-008): `bad host is a usage error`, proves `--host localhost`, `--host 127.0.0.999` and `--host 127.0.0.01` each return 1, write one stderr line and call no start function.
- T-007 (FR-010): `route error answers json 500`, proves a registered route that calls `error("boom")` answers status `InternalServerError`, content type `application/json`, and a body decoding to `{error = {message = "internal error", type = "internal_error"}}`.

## Open questions

- None.

## Amendments

- None.
