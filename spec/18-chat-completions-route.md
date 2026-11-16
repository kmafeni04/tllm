# 18: Answer a chat conversation over HTTP

**Status**: done   **Depends on**: 03, 16, 17

## Problem & context

`serve` answers `/health` and holds a mapped file, `17` turns a messages list into the model's
prompt, and `16` turns prompt text into reply text, but nothing joins them. There is no route a chat
client can call, so the engine still cannot be talked to over the network, and the only way to get
an answer is to write a script.

This spec adds one route to `serve`: `POST /v1/chat/completions`, which takes a JSON body, renders
the model's template over the messages list, generates with the sampling numbers the body carries,
and answers the reply text with usage counts in the envelope `03` fixed. Streaming is spec 19 and is
not here.

Two seams are new. `serve.Model` grows the tokenizer and the compute backend `build` loads from the
file, and `serve.answer(server, body)` is the whole of one request: it takes the raw request body as
text and returns a status name and a JSON text, `serve.stream` handing the body to it when no stream
is asked for. That seam is where the cases do their work, because `tltp`'s `mock_request` writes its `params` argument as a
form body (`k=v&…`) with no way to send raw JSON, so it can prove a request reaches the route but
not carry a chat body. `mock_request` therefore carries the wiring cases and `answer` carries the
rest.

A reply's positions are sized from the budget, not the file's window. `16` takes an optional
`start` so a call can resume a model another call reached, and makes the caller size the model it
passes, and `15` makes `positions` an argument so that a caller not asked for a long context says
so. Taking the file's whole window would reserve about 10.7 GB of cache for one short reply, the
local file's context being 262144, so a body with no `max_tokens` gets a 256-token budget and the
model a request uses takes `positions` of the prompt's ids plus that budget minus one, or the
sizing `23/FR-004` gives a conversation. Checking the request against the file's window then needs
the window as a number, so the route builds a model over the file's window, reads the `positions`
it was built with and drops it, before taking the one it generates from; that reading is the
amendment to `15`. Answering `usage` needs the number of ids the generation sampled, which is the
amendment to `16`.

The calls run in one order: the shape checks of FR-008, `chat.render`, `tokenizer:encode` on the
rendered prompt, `compute.model` over the file's window to read the `positions` it was built with,
the budget check of FR-009, `compute.model` over the positions the request needs, then
`generate.text`. No shape fault reaches the engine; the budget check does, so a file whose window
model is refused answers 500 before any 400, and an engine fault is the line of the first call that
failed.

`build` also loads the tokenizer and the backend, so a file this engine cannot answer from is
refused at start by the line the loader gives, the shape `08/FR-002` already has. The serve tests'
existing fixtures gain the tokenizer metadata that load needs and their assertions stand.

```teal
local compute = require("tllm.compute")
local tokenizer = require("tllm.tokenizer")
local tltp = require("tltp")
local weights = require("tllm.weights")

local record serve
  record Model
    name: string
    path: string
    handle: weights.Handle
    tokenizer: tokenizer.Tokenizer
    backend: compute.Backend
    kv_cache: compute.CacheType | nil
  end

  record Endpoint
    host: string
    port: integer
    name: string
    path: string
    sessions: integer
    kv_cache: compute.CacheType | nil
  end

  record Reply
    status: tltp.Status
    body: string
  end

  model_of: function(tltp.Server): Model | nil
  build: function(Endpoint): tltp.Server | nil, string | nil
  start: function(Endpoint, function(string)): integer
  answer: function(tltp.Server, string): Reply
end
```

## User stories

- US-001 (P1) As a chat client, I want to POST a messages list and get the assistant's reply text, so that I can talk to the model over HTTP.
- US-002 (P1) As a chat client, I want the reply to report how many tokens the prompt and the reply used, so that I can see what the request cost.
- US-003 (P1) As a chat client, I want the body's temperature, top-k, seed and length to steer the reply, so that I control how it is generated.
- US-004 (P1) As a chat client sending a body the route cannot use, I want 400 in the existing error envelope naming what is at fault, so that I can fix my request.
- US-005 (P2) As the person running the server, I want a bad file named at start and an engine failure answered 500 with the reason, so that a fault is never silent and the process stays up.

## Functional requirements

- FR-001 (US-001) THE SYSTEM SHALL register the path `/v1/chat/completions` for POST on every
  server `build` returns, and leave that path unmatched for every other method.
- FR-002 (US-001) WHEN a POST for that path arrives, THE SYSTEM SHALL answer the response
  `serve.stream(server, the raw request body)` gives, the request body being the raw text the
  request carried.
- FR-003 (US-001) WHEN `answer` is given a body FR-008's checks accept, THE SYSTEM SHALL take the
  server's one model, its tokenizer and its backend, render the prompt from `messages` with
  `chat.render`, reserve a sequence where `23/FR-001` reuses a stored conversation's and otherwise
  reserve one sized by `23/FR-004`, generate from that prompt and budget, and answer `OK` with a
  body that decodes to an object whose `choices` is an array holding one object whose `index` is 0
  and whose `message` is an object whose `role` is `assistant` and whose `content` is the generated
  text.
- FR-004 (US-002) WHEN a successful reply is answered, THE SYSTEM SHALL hold in it a `usage` object
  whose `prompt_tokens` is the number of ids the rendered prompt encodes to, whose
  `completion_tokens` is the number of ids the generation sampled, whose `total_tokens` is their
  sum, and whose `prompt_tokens_details` is an object whose `cached_tokens` is the number of the
  prompt's ids the turn did not forward.
- FR-005 (US-001) WHEN the body holds a member other than `messages`, `temperature`, `top_k`,
  `max_tokens`, `seed`, `stream` and `session_id`, THE SYSTEM SHALL ignore it, and `stream` and
  `session_id` are honoured rather than ignored.
- FR-006 (US-003) WHEN the body holds `temperature`, `top_k` or `seed`, THE SYSTEM SHALL pass it to
  the generation as given, an absent `temperature` leaving the sampler greedy and an absent `seed`
  being 0.
- FR-007 (US-003) WHEN the body holds `max_tokens`, THE SYSTEM SHALL take it as the token budget,
  and 256 when it holds none, and the generation SHALL stop at that budget.
- FR-008 (US-004) IF the body is empty, is not JSON, or decodes to anything but a JSON object, or
  `messages` is absent, is not an array, holds no entry, or holds an entry that is not an object
  whose `role` and `content` are strings, or `temperature` is present and is not a number or is
  below zero, or `top_k` is present and is not a whole number or is below one, or `max_tokens` is
  present and is not a whole number or is below one, or `seed` is present and is not a whole number,
  or `stream` is present and is not a boolean (that check coming after `seed`), or `session_id` is
  present and is not a string (that check coming after `stream`), THEN THE SYSTEM
  SHALL answer `BadRequest` with a body decoding to an object whose single member is
  `error`, an object whose `type` is `invalid_request_error` and whose `message` names the first
  fault in the order this requirement names them: `body`; `messages` and, for an entry, its position
  counting from one; the member at fault.
- FR-009 (US-004) WHEN `chat.render` and `tokenizer:encode` have answered, THE SYSTEM SHALL take
  the server's one model over the file's window, its `positions` being the file's context length,
  and, IF the prompt's ids plus the budget minus one are more than the `positions` that model was
  built with, THEN THE SYSTEM SHALL answer `BadRequest` as FR-008 describes it, the message naming
  `messages`, `max_tokens`, the prompt's id count, the budget and that `positions`. Its tests set
  `f32` wherever they pin a reply.
- FR-010 (US-005) WHEN `build` is called, THE SYSTEM SHALL load the file's tokenizer with
  `tokenizer.load` and hold it on the model it keeps, returning nil and that loader's line as its
  second value when it refuses; for an endpoint whose `workers` is more than one, `build` loads no
  tokenizer into that process and holds none on the server it returns.
- FR-011 (US-005) WHEN `build` is called, THE SYSTEM SHALL load the compute backend with
  `compute.load` and hold it on the model it keeps, returning nil and that loader's line when it
  refuses; for an endpoint whose `workers` is more than one, `build` loads no compute backend into
  that process and holds none on the server it returns, except the one `28/FR-011` reads `memory`
  from, which it drops before returning.
  Not testable: `build` calls `compute.load` with no argument, so whether it refuses depends on the
  host's libraries; the field it fills is proven by every case that generates.
- FR-012 (US-005) IF `tokenizer:encode` for the rendered prompt, `chat.render`, a reservation
  `reserve` refuses or `generate.text` returns a failure, THEN THE SYSTEM SHALL answer
  `InternalServerError` with a body decoding to an object whose single member is `error`, an object
  whose `message` is that line and whose `type` is `internal_error`. `compute.model` leaves the
  per-request faults: a model the server cannot build is `build`'s refusal, as `26/T-007` has it,
  and a reservation `reserve` refuses is the turn's own fault, its line the body's.

## Non-goals

- No streamed reply of its own: the route answers whatever `serve.stream` gives, and the events of
  a streamed reply, their names and their order, are spec 19's.
- No reply member beyond `choices` and `usage`: no `id`, `object`, `created`, `model` or
  `finish_reason`, and one choice only.
- No trimming or reloading: a request without a `session_id` reserves its own sequence over the
  positions it needs, starting from position zero; `23` keeps a conversation's sequence between
  requests, so a cache does outlive a request.

## Tests

- T-001 (FR-003, FR-004): `a chat request answers the model's text with usage counts`, proves on a
  fixture carrying `16/T-001`'s metadata and tensors unchanged except `context_length` 260, the
  three pieces `a`, `b` and `c`, and a `tokenizer.chat_template` that writes each message's
  `content` inside a `{% for message in messages %}` block, that `build` returns a server and nil,
  that `answer(server, '{"messages":[{"role":"user","content":"a"}]}')` answers `OK` with a body
  decoding to `choices` holding one object whose `index` is 0 and whose `message` is
  `{role = "assistant", content = "a" written 256 times}` and `usage` of `{prompt_tokens = 1,
  completion_tokens = 256, total_tokens = 257, prompt_tokens_details = {cached_tokens = 0}}`; that a
  body also holding `"max_tokens":2` answers
  `aa` with `completion_tokens` 2 and `total_tokens` 3; and that on a copy whose
  `tokenizer.ggml.eos_token_id` is 0 the first body answers `content` `""` with `completion_tokens`
  0 and `total_tokens` 1.
- T-002 (FR-005, FR-006, FR-007): `the sampling numbers in the body steer the reply`, proves on
  T-001's fixture that a body holding `"temperature":1`, `"top_k":2`, `"seed":1000` and
  `"max_tokens":3` answers `ccc`; that the same body without `seed` answers `aac`, the seed
  defaulting to 0; that a body holding `"max_tokens":10` answers `a` written ten times with
  `completion_tokens` 10 and `total_tokens` 11; that T-001's second body carrying also `"model"` and
  `"stream":false` answers exactly as T-001's second call does; and that a body whose message
  content is `aaaaa` and holds no `max_tokens` answers `OK`, the five prompt ids plus 256 minus one
  being the 260 positions the file allows.
- T-003 (FR-008, FR-009): `a body the route cannot use is a bad request`, proves on T-001's
  fixture that each of these answers `BadRequest` and a body decoding to `error` with type
  `invalid_request_error`: the empty string, `nope` and `[]`, each with a message naming `body`;
  `{}`, `{"messages":5}` and `{"messages":[]}`, each with a message naming `messages`; and
  `{"messages":[{"role":"user"}]}` and `{"messages":[{"role":"user","content":["x"]}]}`, each with a
  message naming `messages` and `1`; a body whose only message is valid with
  `"temperature":"hot"`, with `"temperature":-1`, with `"top_k":2.5`, with `"max_tokens":0`,
  with `"seed":1.5` and with `"session_id":5`, the messages naming in turn `temperature`,
  `temperature`, `top_k`, `max_tokens`, `seed` and `session_id`; a body holding an empty `messages`
  and `"temperature":-1`, its message naming `messages`;
  a body holding both `"temperature":-1` and `"top_k":0`, its message naming `temperature`; and a
  body whose message content is six copies of `a`, its message naming `messages`, `max_tokens`, `6`,
  `256` and `260`, six ids plus the 256-token budget minus one being past the positions the file
  allows.
- T-004 (FR-001, FR-002, FR-012): `the route is wired and engine failures answer json 500`, proves
  on T-001's fixture that `mock_request("/v1/chat/completions")` answers `NotFound` with
  `03/FR-004`'s body, that `mock_request("/v1/chat/completions", {method = "post"})` answers
  `BadRequest` with content type `application/json` and a message naming `body`, that a copy whose
  `tokenizer.chat_template` is removed answers `InternalServerError` with type `internal_error` and
  a message naming `tokenizer.chat_template`, that a copy whose template is the single byte `0xFF`
  answers `InternalServerError` with the line `tokenizer:encode` gives for that same text, that a
  copy whose template is `{% for message in messages %}{% endfor %}` answers `InternalServerError`
  with a message naming `prompt`, and that a copy whose metadata holds the tokenizer keys but no
  `embedding_length` answers `InternalServerError` with a message naming `model` and
  `embedding_length`.
- T-005 (FR-010): `build loads the tokenizer and holds it`, proves on T-001's fixture that the model
  `model_of` gives holds a `tokenizer` whose `size` is 3, and that a copy whose metadata holds no
  tokenizer key at all makes `build` return nil and a line naming `tokenizer.ggml.model`.

## Open questions

- None.

## Amendments

- None.
