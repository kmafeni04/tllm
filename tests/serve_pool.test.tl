-- the transport, worker and pool cases, split out of tests/serve.test.tl so
-- that file stays under the size the project asks of one file. The names and
-- bodies are the ones that file already ran.
local fs = require("fs")
local test = require("lde-test")
local process = require("process")
local fixtures = require("tests.serve_fixtures")
local serve = require("tllm.serve")
local tltp = require("tltp")
local type HealthBody = fixtures.HealthBody
local type ErrorBody = fixtures.ErrorBody
local type ChatReply = fixtures.ChatReply
local type StreamDelta = fixtures.StreamDelta
local type Watcher = fixtures.Watcher
local ROOT, fixture, meta_bytes, meta_file =
  fixtures.ROOT, fixtures.fixture, fixtures.meta_bytes, fixtures.meta_file
local t001_meta, t001_entries, DEVICE_LINE, answered =
  fixtures.t001_meta,
  fixtures.t001_entries,
  fixtures.DEVICE_LINE,
  fixtures.answered
local build, temp_dir, begun, events_of =
  fixtures.build, fixtures.temp_dir, fixtures.begun, fixtures.events_of
local ROLE_EVENT, token_event, finish_event, FIRST_TURN =
  fixtures.ROLE_EVENT,
  fixtures.token_event,
  fixtures.finish_event,
  fixtures.FIRST_TURN
local SECOND_TURN, unreadable_fixture, streaming, C =
  fixtures.SECOND_TURN,
  fixtures.unreadable_fixture,
  fixtures.streaming,
  fixtures.C
local held, named, streamed = fixtures.held, fixtures.named, fixtures.streamed
local POOL_BUDGET, workers_of = fixtures.POOL_BUDGET, fixtures.workers_of

local function at(file_path: string, host: string, port: integer): tltp.Server
  local server, failure = serve.build(
    {
      host = host,
      port = port,
      name = "m",
      path = file_path,
      sessions = 4,
      kv_cache = "f32",
    }
  )
  test.equal(failure, nil, file_path)
  return server as tltp.Server
end

test.it(
  "a round steps the transport and the turns",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local server = at(file_path, "127.0.0.1", 9405)
    local _, round_run = serve.stream(server, streamed("s1"))
    test.truthy(round_run)
    local watcher = begun(round_run as tltp.StreamFn, nil, "A")

    test.equal(serve.round(server), nil)
    test.equal(serve.round(server), nil)
    test.equal(#watcher.payloads, 5)
    local events = events_of(watcher.payloads)
    test.deepEqual(events[1], ROLE_EVENT)
    test.deepEqual(events[2], token_event("a"))
    test.deepEqual(events[3], token_event("a"))
    test.deepEqual(events[4], finish_event("length"))
    test.equal(watcher.payloads[5], "[DONE]")

    local unreachable = at(file_path, "192.0.2.1", 9090)
    test.includes(serve.round(unreachable) as string, "BIND")

    local written: {string} = {}
    local code = serve.start(
      {
        host = "192.0.2.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
      },
      function(line: string) written[#written + 1] = line end
    )
    test.equal(code, 1)
    test.equal(#written, 2)
    test.equal((written[1]:gsub("\n$", "")):match(DEVICE_LINE) ~= nil, true)
    test.includes(written[2], "BIND")
    test.equal(#(written[2]:match("(\n*)$") as string), 1)
  end
)

test.it(
  "a turn registered while others are live is stepped next",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m")
    local order: {string} = {}
    local _, first_fn = serve.stream(server, streamed("s1"))
    local first = begun(first_fn as tltp.StreamFn, order, "A")
    test.equal(serve.step(server), 1)
    test.equal(#first.payloads, 2)
    test.deepEqual(events_of(first.payloads)[2], token_event("a"))

    local _, second_fn = serve.stream(server, streamed("s2"))
    local second = begun(second_fn as tltp.StreamFn, order, "B")
    test.deepEqual(events_of(second.payloads), {ROLE_EVENT})

    test.equal(serve.step(server), 1)
    test.equal(#first.payloads, 5)
    test.deepEqual(events_of(first.payloads)[4], finish_event("length"))
    test.equal(#second.payloads, 2)
    test.deepEqual(events_of(second.payloads)[2], token_event("a"))

    test.equal(serve.step(server), 0)
    test.equal(#second.payloads, 5)
  end
)

test.it(
  "a second turn on a live session takes its own sequence",
  function()
    local server = build(meta_file(t001_meta(), t001_entries()), "m", 4)

    local first = answered(server, FIRST_TURN)
    test.equal(first.choices[1].message.content, "aa")
    test.equal(first.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").length, 2)
    test.equal(held(server, "s").positions, 4)

    local _, live_fn = serve.stream(
      server, (SECOND_TURN:gsub("}$", ',"stream":true}'))
    )
    test.truthy(live_fn)
    local watcher = begun(live_fn as tltp.StreamFn, nil, "A")
    test.equal(held(server, "s").length, 2)

    local second = answered(server, SECOND_TURN)
    test.equal(second.choices[1].message.content, "aa")
    test.equal(second.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(held(server, "s").positions, 8)

    test.equal(serve.step(server), 1)
    test.equal(#watcher.payloads, 2)
    test.deepEqual(events_of(watcher.payloads)[2], token_event("a"))
    test.equal(serve.step(server), 0)
    test.equal(#watcher.payloads, 5)
  end
)

local function stream_text(watcher: Watcher): string
  local parts: {string} = {}
  for _, event in ipairs(events_of(watcher.payloads)) do
    local delta = event.choices[1].delta
    if delta is StreamDelta then
      local content = delta.content
      if content is string then
        parts[#parts + 1] = content
      end
    end
  end
  return table.concat(parts)
end

test.it(
  "two turns advance in one pass and keep their own streams",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local server = build(file_path, "m", 2)
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local body_b = '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true}'

    local order: {string} = {}
    local _, a_fn = serve.stream(server, body_a)
    local _, b_fn = serve.stream(server, body_b)
    test.truthy(a_fn)
    test.truthy(b_fn)
    local a = begun(a_fn as tltp.StreamFn, order, "A")
    local b = begun(b_fn as tltp.StreamFn, order, "B")
    test.equal(coroutine.status(a.co), "suspended")
    test.equal(coroutine.status(b.co), "suspended")
    test.deepEqual(events_of(a.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})
    test.deepEqual(order, {"A", "B"})

    local check = build(file_path, "check", 2)
    local wanted_a = answered(check, body_a).choices[1].message.content
    local wanted_b = answered(check, body_b).choices[1].message.content
    local first_a = wanted_a:sub(1, 1)
    local first_b = wanted_b:sub(1, 1)

    test.equal(serve.step(server), 2)
    test.deepEqual(order, {"A", "B", "A", "B"})
    test.deepEqual(events_of(a.payloads), {ROLE_EVENT, token_event(first_a)})
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT, token_event(first_b)})

    test.equal(serve.step(server), 0)
    test.deepEqual(order, {"A", "B", "A", "B", "A", "A", "A", "B", "B", "B"})
    test.deepEqual(
      events_of(a.payloads),
      {
        ROLE_EVENT,
        token_event(first_a),
        token_event(wanted_a:sub(2, 2)),
        finish_event("length"),
      }
    )
    test.equal(a.payloads[5], "[DONE]")
    test.deepEqual(
      events_of(b.payloads),
      {
        ROLE_EVENT,
        token_event(first_b),
        token_event(wanted_b:sub(2, 2)),
        finish_event("length"),
      }
    )
    test.equal(b.payloads[5], "[DONE]")
    test.equal(stream_text(a), wanted_a)
    test.equal(stream_text(b), wanted_b)
  end
)

test.it(
  "a turn waits for a sequence and starts when one frees",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local body_b = '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true}'

    local server = build(file_path, "m", 1)
    local model = serve.model_of(server) as serve.Model
    test.equal(model.generation.sequences, 1)

    local _, a_fn = serve.stream(server, body_a)
    local a = begun(a_fn as tltp.StreamFn, nil, "A")
    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(a.payloads)[2], token_event("a"))

    local _, b_fn = serve.stream(server, body_a)
    local b = begun(b_fn as tltp.StreamFn, nil, "B")
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})

    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(b.payloads), {ROLE_EVENT})
    test.equal(#a.payloads, 5)

    test.equal(serve.step(server), 1)
    test.deepEqual(events_of(b.payloads)[2], token_event("a"))
    test.equal(serve.step(server), 0)
    test.equal(#b.payloads, 5)

    local stored = build(file_path, "stored", 1)
    local _, s_fn = serve.stream(stored, streamed("s"))
    begun(s_fn as tltp.StreamFn, nil, "S")
    test.equal(serve.step(stored), 1)
    test.equal(serve.step(stored), 0)
    test.truthy(serve.session_of(stored, "s"))

    local _, evict_fn = serve.stream(stored, body_a)
    test.truthy(evict_fn)
    test.equal(serve.session_of(stored, "s"), nil)
    local evict = begun(evict_fn as tltp.StreamFn, nil, "E")
    test.equal(serve.step(stored), 1)
    test.deepEqual(events_of(evict.payloads)[2], token_event("a"))

    local third = build(file_path, "third", 2)
    local _, third_a = serve.stream(third, body_a)
    local _, third_b_fn = serve.stream(third, body_b)
    local a_live = begun(third_a as tltp.StreamFn, nil, "A")
    local b_live = begun(third_b_fn as tltp.StreamFn, nil, "B")
    test.equal(serve.step(third), 2)
    test.equal(#a_live.payloads, 2)
    test.equal(#b_live.payloads, 2)

    local _, third_c = serve.stream(third, body_a)
    local _, third_d = serve.stream(third, body_b)
    local c = begun(third_c as tltp.StreamFn, nil, "C")
    local d = begun(third_d as tltp.StreamFn, nil, "D")
    test.deepEqual(events_of(c.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(d.payloads), {ROLE_EVENT})

    test.equal(serve.step(third), 2)
    test.deepEqual(events_of(c.payloads), {ROLE_EVENT})
    test.deepEqual(events_of(d.payloads), {ROLE_EVENT})
    test.equal(#a_live.payloads, 5)
    test.equal(#b_live.payloads, 5)

    test.equal(serve.step(third), 2)
    test.deepEqual(events_of(c.payloads)[2], token_event("a"))
    test.deepEqual(events_of(d.payloads)[2], token_event("c"))

    local full = build(file_path, "full", 1)
    local _, storing_fn = serve.stream(full, streamed("s"))
    local storing = begun(storing_fn as tltp.StreamFn, nil, "S")
    test.equal(serve.step(full), 1)

    local _, waiting_fn = serve.stream(full, body_a)
    local waiting = begun(waiting_fn as tltp.StreamFn, nil, "W")
    test.deepEqual(events_of(waiting.payloads), {ROLE_EVENT})
    test.equal(serve.step(full), 1)
    test.equal(#storing.payloads, 5)
    test.truthy(serve.session_of(full, "s"))
    test.deepEqual(events_of(waiting.payloads), {ROLE_EVENT})

    test.equal(serve.step(full), 1)
    test.deepEqual(events_of(waiting.payloads)[2], token_event("a"))
    test.equal(serve.session_of(full, "s"), nil)
  end
)

test.it(
  "a reply that cannot wait is told so, a stored conversation keeps its sequence and a discarded one frees it",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local body_a = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'
    local plain = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2}'

    local live = build(file_path, "live", 1)
    local _, live_fn = serve.stream(live, body_a)
    local watcher = begun(live_fn as tltp.StreamFn, nil, "A")
    local before = #watcher.payloads
    local reply = serve.answer(live, plain)
    test.equal(reply.status, "TooManyRequests")
    local decoded = tltp.json_decode(reply.body) as ErrorBody
    test.equal(decoded.error.type, "too_many_requests")
    test.includes(decoded.error.message, "sessions")
    test.equal(serve.session_of(live, "s"), nil)
    test.equal(#watcher.payloads, before)

    local one = build(file_path, "one", 4)
    local first = answered(one, FIRST_TURN)
    test.equal(first.usage.prompt_tokens_details.cached_tokens, 0)
    local kept = held(one, "s")
    test.equal(kept.length, 2)
    test.equal(kept.positions, 4)
    test.equal(kept.sequence, "s")

    local second = answered(one, SECOND_TURN)
    test.equal(second.usage.prompt_tokens_details.cached_tokens, 2)
    test.equal(held(one, "s").length, 4)
    test.equal(held(one, "s").sequence, "s")

    local two = build(file_path, "two", 2)
    answered(two, FIRST_TURN)
    local _, copy_fn = serve.stream(two, streaming(FIRST_TURN))
    begun(copy_fn as tltp.StreamFn, nil, "A")
    test.equal(serve.step(two), 1)

    local fresh = answered(two, FIRST_TURN)
    test.equal(fresh.usage.prompt_tokens_details.cached_tokens, 0)
    local other = held(two, "s").sequence
    test.notEqual(other, "s")

    test.equal(serve.step(two), 0)
    test.equal(held(two, "s").sequence, "s")
    local generation = (serve.model_of(two) as serve.Model).generation
    test.equal(generation:cache_bytes(other, "blk.0."), nil)
    test.truthy(generation:cache_bytes("s", "blk.0."))

    local bad = build(unreadable_fixture(), "bad", 1)
    answered(bad, FIRST_TURN)
    test.equal(held(bad, "s").length, 2)
    local failing = '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"a"},'
      .. '{"role":"user","content":"a"}],"max_tokens":2,'
      .. '"temperature":1,"top_k":2,"seed":1000,"session_id":"s"}'
    local lost_reply = serve.answer(bad, failing)
    test.equal(lost_reply.status, "InternalServerError")
    test.equal(serve.session_of(bad, "s"), nil)
    local bad_generation = (serve.model_of(bad) as serve.Model).generation
    test.equal(bad_generation:cache_bytes("s", "blk.0."), nil)
    test.equal(serve.answer(bad, plain).status, "OK")
  end
)

test.it(
  "a model the server cannot build is a start failure",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local endpoint: serve.Endpoint = {
      host = "127.0.0.1",
      port = 9090,
      name = "m",
      path = file_path,
      sessions = 4,
      kv_cache = "q8_0",
    }
    local server, line = serve.build(endpoint)
    test.equal(server, nil)
    test.includes(line as string, "q8_0")
    test.includes(line as string, "2")

    local written: {string} = {}
    local code = serve.start(
      endpoint, function(one: string) written[#written + 1] = one end
    )
    test.equal(code, 1)
    test.equal(#written, 1)
    test.includes(written[1], "q8_0")
    test.includes(written[1], "2")
    test.equal(written[1]:sub(-1), "\n")
    test.equal(written[1]:find("\n"), #written[1])
  end
)

local function context_names(dir: string): {string}
  local names: {string} = {}
  local next_entry = fs.readdir(dir)
  if next_entry is nil then
    return names
  end
  while true do
    local entry = next_entry()
    if entry is nil then
      break
    end
    names[#names + 1] = (entry as fs.DirEntry).name
  end
  table.sort(names)
  return names
end

local function context_size(dir: string, name: string): integer
  local listed = fs.stat(dir .. "/" .. name)
  test.truthy(listed, name)
  return math.floor(tonumber((listed as fs.Stat).size) as number)
end

local function extended_with(id: string): string
  return named(
    '{"messages":[{"role":"user","content":"a"},'
      .. '{"role":"assistant","content":"a"},'
      .. '{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}',
    id
  )
end

test.it(
  "a restart continues a conversation from its file",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local first_dir = ROOT .. "/contexts1"
    local server = build(file_path, "m", 4, "f32", first_dir, 1000000)
    test.truthy(fs.isdir(first_dir))

    local response, stream_fn = serve.stream(
      server,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"stream":true,"session_id":"s"}'
    )
    test.equal(response.status, "OK")
    test.truthy(stream_fn)
    begun(stream_fn as tltp.StreamFn, nil, "A")
    while serve.step(server) > 0 do
    end
    local stored_names = context_names(first_dir)
    test.equal(#stored_names, 1)
    test.includes(stored_names[1], ".context")
    local stored_size = context_size(first_dir, stored_names[1])

    local second_dir = temp_dir()
    fs.write(
      second_dir .. "/" .. stored_names[1],
      fs.read(first_dir .. "/" .. stored_names[1]) as string
    )
    local copied = context_names(second_dir)
    test.equal(#copied, 1)
    test.equal(context_size(second_dir, copied[1]), stored_size)

    local restarted = build(file_path, "m", 4, "f32", first_dir, 1000000)
    local revived = answered(restarted, extended_with("s"))
    test.deepEqual(
      revived.usage,
      {
        prompt_tokens = 3,
        completion_tokens = 2,
        total_tokens = 5,
        prompt_tokens_details = {cached_tokens = 2},
      }
    )

    local memory = build(file_path, "m", 4, "f32")
    answered(
      memory,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    local wanted = answered(memory, extended_with("s"))
    test.equal(
      revived.choices[1].message.content, wanted.choices[1].message.content
    )

    local other = answered(restarted, extended_with("t"))
    test.equal(other.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(#context_names(first_dir), 2)

    local loose = (extended_with("s"):gsub(',"session_id":"s"', ""))
    local plain = answered(restarted, loose)
    test.equal(plain.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(#context_names(first_dir), 2)

    local halved = build(file_path, "m", 4, "f16", second_dir, 1000000)
    local half = answered(halved, extended_with("s"))
    test.equal(half.usage.prompt_tokens_details.cached_tokens, 0)
    local half_names = context_names(second_dir)
    test.equal(#half_names, 1)
    test.notEqual(context_size(second_dir, half_names[1]), stored_size)

    local blocker = ROOT .. "/blocker"
    fs.write(blocker, "not a directory")
    local refused_path = blocker .. "/contexts"
    local uncreated, uncreated_line = serve.build(
      {
        host = "127.0.0.1",
        port = 9090,
        name = "m",
        path = file_path,
        sessions = 4,
        kv_cache = "f32",
        context_path = refused_path,
        context_bytes = 1000,
      }
    )
    test.equal(uncreated, nil)
    test.includes(uncreated_line as string, refused_path)

    local lost_dir = temp_dir()
    local writing = build(file_path, "m", 4, "f32", lost_dir, 1000000)
    fs.rmdir(lost_dir)
    fs.write(lost_dir, "not a directory")
    local unpaired = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
      .. '"session_id":"f"}'
    local unwritten = answered(writing, unpaired)
    local contextless = answered(build(file_path, "m", 4, "f32"), unpaired)
    test.deepEqual(unwritten.usage, contextless.usage)
    test.equal(
      unwritten.choices[1].message.content,
      contextless.choices[1].message.content
    )
    local followed = answered(writing, extended_with("f"))
    test.equal(followed.usage.prompt_tokens_details.cached_tokens, 2)

    local fourth_dir = ROOT .. "/contexts4"
    local kept = build(file_path, "m", 4, "f32", fourth_dir, 1000000)
    answered(
      kept,
      '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    local rebuilt = build(file_path, "m", 4, "f32", fourth_dir, 1000000)
    local moved = answered(
      rebuilt,
      '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,'
        .. '"session_id":"s"}'
    )
    test.equal(moved.usage.prompt_tokens_details.cached_tokens, 0)
    for _, id in ipairs({"t", "u", "v"}) do
      local next_reply = serve.answer(
        rebuilt,
        '{"messages":[{"role":"user","content":"b"}],"max_tokens":2,'
          .. '"session_id":"'
          .. id
          .. '"}'
      )
      test.equal(next_reply.status, "OK", id)
    end
  end
)

local function long_body(count: integer, id: string): string
  return '{"messages":[{"role":"user","content":"'
    .. string.rep("a", count)
    .. '"}],"max_tokens":2,"session_id":"'
    .. id
    .. '"}'
end

local function long_extended(count: integer, id: string): string
  return '{"messages":[{"role":"user","content":"'
    .. string.rep("a", count)
    .. '"},{"role":"assistant","content":"a"},'
    .. '{"role":"user","content":"a"}],"max_tokens":2,"session_id":"'
    .. id
    .. '"}'
end

test.it(
  "the files stay inside the configured budget",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local wide_dir = temp_dir()
    local wide = build(file_path, "m", 4, "f32", wide_dir, 10000000)
    answered(wide, long_body(130, "s1"))
    local wide_names = context_names(wide_dir)
    test.equal(#wide_names, 1)
    local budget = context_size(wide_dir, wide_names[1])

    local kept_dir = temp_dir()
    local kept = build(file_path, "m", 4, "f32", kept_dir, budget)
    for _, step in ipairs({{130, "s1"}, {120, "s2"}, {110, "s3"}}) do
      answered(kept, long_body(step[1] as integer, step[2] as string))
      local names = context_names(kept_dir)
      test.equal(#names, 1, step[2] as string)
      test.lessEqual(
        context_size(kept_dir, names[1]), budget, step[2] as string
      )
    end

    local again = build(file_path, "m", 4, "f32", kept_dir, budget)
    local newest = answered(again, long_extended(110, "s3"))
    test.equal(newest.usage.prompt_tokens_details.cached_tokens, 111)
    test.equal(newest.usage.prompt_tokens, 112)

    local gone = answered(again, long_extended(130, "s1"))
    test.equal(gone.usage.prompt_tokens_details.cached_tokens, 0)
    test.equal(gone.usage.prompt_tokens, 132)

    local tight_dir = temp_dir()
    local tight = build(file_path, "m", 4, "f32", tight_dir, 1)
    answered(tight, long_body(130, "s1"))
    test.equal(#context_names(tight_dir), 0)
  end
)

test.it(
  "a session id cannot name a file outside the directory",
  function()
    local file_path = meta_file(t001_meta(), t001_entries())
    local dir = temp_dir()
    local server = build(file_path, "m", 4, "f32", dir, 10000000)

    local ids = {"../escaped", "x/y", "x_y", "a/b", "a%2Fb", "."}
    for _, id in ipairs(ids) do
      answered(
        server,
        '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
          .. '"session_id":"'
          .. id
          .. '"}'
      )
    end

    local names = context_names(dir)
    test.equal(#names, 6)
    for _, name in ipairs(names) do
      test.equal(name:sub(-8), ".context", name)
      test.falsy(fs.isdir(dir .. "/" .. name), name)
    end
    local parent = dir:match("^(.*)/") as string
    test.falsy(fs.exists(parent .. "/escaped.context"))

    local again = build(file_path, "m", 4, "f32", dir, 10000000)
    for _, id in ipairs({"../escaped", "x/y", "x_y"}) do
      local reply = answered(again, extended_with(id))
      test.equal(reply.usage.prompt_tokens_details.cached_tokens, 2, id)
    end
  end
)

--- pools of worker processes

local QUICK_BODY = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2}'

local SESSION_S = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"s"}'
local SESSION_T = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"t"}'
local SESSION_U = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"u"}'
local PLAIN_S = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}'
local THREE_S = '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},{"role":"user","content":"a"}],"max_tokens":2,"stream":true,"session_id":"s"}'
local THREE_PLAIN_S = '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"a"},{"role":"user","content":"a"}],"max_tokens":2,"session_id":"s"}'
local FREE_STREAM = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}'

local function pause()
  C.usleep(2000)
end

local function pool_build(file_path: string, context: string): tltp.Server
  return build(file_path, "m", 4, "f32", context, POOL_BUDGET, 2)
end

local function holder(server: tltp.Server, id: string): serve.Worker | nil
  for _, worker in ipairs(workers_of(server)) do
    for _, held_id in ipairs(worker.sessions) do
      if held_id == id then
        return worker
      end
    end
  end
  return nil
end

local function driven_pool(server: tltp.Server)
  local tries = 0
  while serve.step(server) > 0 do
    tries = tries + 1
    if tries > 5000 then
      test.equal(tries, 0, "the pool still held live turns")
      return
    end
    pause()
  end
end

local function streamed_on(server: tltp.Server, body: string): {string}
  local _, stream_fn = serve.stream(server, body)
  local watcher = begun(stream_fn as tltp.StreamFn, nil, "X")
  while serve.step(server) > 0 do
  end
  return watcher.payloads
end

local function in_process_payloads(bytes: string, body: string): {string}
  return streamed_on(build(fixture(bytes), "m"), body)
end

local function finished_watch(watch: Watcher): boolean
  return #watch.payloads > 0 and watch.payloads[#watch.payloads] == "[DONE]"
end

-- the same events, however the encoder ordered the keys of each one
local function same_payloads(got: {string}, expected: {string})
  test.equal(#got, #expected)
  test.deepEqual(events_of(got), events_of(expected))
  test.equal(got[#got], expected[#expected])
end

local function gone(pid: integer): boolean
  local line = fs.read("/proc/" .. tostring(pid) .. "/stat")
  if line is nil then
    return true
  end
  return line:match("%) (%a)") == "Z"
end

local function killed(pid: integer)
  local tries = 0
  while not gone(pid) do
    tries = tries + 1
    if tries > 5000 then
      test.equal(tries, 0, "the process did not die")
      return
    end
    pause()
  end
end

test.it(
  "two workers answer at once and the parent holds no model",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local single = fixture(bytes)
    local plain = build(single, "m")
    test.equal(serve.pool_of(plain), nil)
    local model = serve.model_of(plain) as serve.Model
    test.equal(model.path, single)
    local answer = serve.answer(plain, QUICK_BODY)
    test.equal(answer.status, "OK")
    test.equal(
      (tltp.json_decode(answer.body) as ChatReply).choices[1].message.content,
      "aa"
    )
    test.equal(serve.pool_of(plain), nil)

    local copy = fixture(bytes)
    local server = pool_build(copy, temp_dir())
    test.equal(serve.model_of(server), nil)
    test.equal(serve.session_of(server, "s"), nil)
    local pool = serve.pool_of(server) as serve.Pool
    test.equal(#pool.workers, 2)
    test.notEqual(pool.workers[1].slot, pool.workers[2].slot)
    test.notEqual(pool.workers[1].pid, pool.workers[2].pid)
    test.equal(#pool.workers[1].sessions, 0)
    test.equal(#pool.workers[2].sessions, 0)
    local maps = fs.read("/proc/self/maps") as string
    test.equal(maps:find(copy, 1, true), nil)

    local reference = build(fixture(bytes), "m")
    local s_expected = streamed_on(reference, SESSION_S)
    local t_expected = streamed_on(reference, SESSION_T)

    local _, s_fn = serve.stream(server, SESSION_S)
    local s_watch = begun(s_fn as tltp.StreamFn, nil, "S")
    local _, t_fn = serve.stream(server, SESSION_T)
    local t_watch = begun(t_fn as tltp.StreamFn, nil, "T")
    driven_pool(server)
    same_payloads(s_watch.payloads, s_expected)
    same_payloads(t_watch.payloads, t_expected)
    for _, worker in ipairs(workers_of(server)) do
      test.equal(#worker.sessions, 1)
      test.truthy(worker.sessions[1] == "s" or worker.sessions[1] == "t")
    end

    local expected = serve.answer(reference, PLAIN_S)
    local got = serve.answer(server, PLAIN_S)
    test.equal(got.status, expected.status)
    test.deepEqual(
      tltp.json_decode(got.body) as ChatReply,
      tltp.json_decode(expected.body) as ChatReply
    )

    local _, live_fn = serve.stream(server, FREE_STREAM)
    begun(live_fn as tltp.StreamFn, nil, "L")
    local health = server:mock_request("/health")
    test.equal(health.status, "OK")
    test.deepEqual(
      tltp.json_decode(health.body) as HealthBody, {status = "ok", model = "m"}
    )
    local missing = server:mock_request("/other")
    test.equal(missing.status, "NotFound")
    test.deepEqual(
      tltp.json_decode(missing.body) as ErrorBody,
      {error = {message = "not found", type = "not_found"}}
    )
    driven_pool(server)
    serve.stop(server)
  end
)

test.it(
  "a session goes back to the worker that served it",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local server = pool_build(fixture(bytes), temp_dir())

    local _, first_fn = serve.stream(server, SESSION_S)
    begun(first_fn as tltp.StreamFn, nil, "S")
    driven_pool(server)
    local recorded = holder(server, "s")
    test.truthy(recorded)
    local pid = (recorded as serve.Worker).pid

    local _, second_fn = serve.stream(server, SESSION_T)
    begun(second_fn as tltp.StreamFn, nil, "T")
    driven_pool(server)
    local still = holder(server, "s")
    test.truthy(still)
    test.equal((still as serve.Worker).pid, pid)

    local reference = build(fixture(bytes), "m")
    streamed_on(reference, SESSION_S)
    local expected = streamed_on(reference, THREE_S)

    local _, again_fn = serve.stream(server, SESSION_S)
    local again = begun(again_fn as tltp.StreamFn, nil, "A")
    local _, three_fn = serve.stream(server, THREE_S)
    local three_watch = begun(three_fn as tltp.StreamFn, nil, "B")
    test.equal(serve.step(server), 2)
    test.equal(#three_watch.payloads, 0)
    local tries = 0
    while not finished_watch(again) do
      serve.step(server)
      test.equal(
        #three_watch.payloads, 0, "a session waited for its own worker"
      )
      tries = tries + 1
      if tries > 5000 then
        test.equal(tries, 0, "the first turn did not end")
        break
      end
      pause()
    end
    driven_pool(server)
    same_payloads(three_watch.payloads, expected)
    local served_by = holder(server, "s")
    test.truthy(served_by)
    test.equal((served_by as serve.Worker).pid, pid)

    local _, free_fn = serve.stream(server, FREE_STREAM)
    begun(free_fn as tltp.StreamFn, nil, "F")
    driven_pool(server)
    local kept = holder(server, "s")
    test.truthy(kept)
    test.equal((kept as serve.Worker).pid, pid)

    local many = build(
      fixture(bytes), "m", 2, "f32", temp_dir(), POOL_BUDGET, 2
    )
    for _, id in ipairs({"a", "b", "c", "d", "e", "f"}) do
      local body = '{"messages":[{"role":"user","content":"a"}],"max_tokens":2,'
        .. '"stream":true,"session_id":"'
        .. id
        .. '"}'
      local _, fn = serve.stream(many, body)
      begun(fn as tltp.StreamFn, nil, id)
    end
    driven_pool(many)
    for _, worker in ipairs(workers_of(many)) do
      test.truthy(#worker.sessions <= 2)
    end
    serve.stop(many)
    serve.stop(server)
  end
)

test.it(
  "a third request waits for a free worker",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local server = pool_build(fixture(bytes), temp_dir())

    local _, s_fn = serve.stream(server, SESSION_S)
    local s_watch = begun(s_fn as tltp.StreamFn, nil, "S")
    local _, t_fn = serve.stream(server, SESSION_T)
    local t_watch = begun(t_fn as tltp.StreamFn, nil, "T")
    local _, u_fn = serve.stream(server, SESSION_U)
    local u_watch = begun(u_fn as tltp.StreamFn, nil, "U")

    local tries = 0
    while not (finished_watch(s_watch) or finished_watch(t_watch)) do
      serve.step(server)
      test.equal(#u_watch.payloads, 0, "the third request emitted while held")
      tries = tries + 1
      if tries > 5000 then
        test.equal(tries, 0, "the first two turns did not end")
        break
      end
      pause()
    end
    driven_pool(server)
    same_payloads(u_watch.payloads, in_process_payloads(bytes, SESSION_U))
    serve.stop(server)
  end
)

test.it(
  "a worker that dies loses one turn and keeps the session",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local context = temp_dir()
    local server = pool_build(fixture(bytes), context)

    local _, s_fn = serve.stream(server, SESSION_S)
    begun(s_fn as tltp.StreamFn, nil, "S")
    driven_pool(server)

    local _, three_fn = serve.stream(server, THREE_S)
    local watch = begun(three_fn as tltp.StreamFn, nil, "T")
    local tries = 0
    while #watch.payloads < 2 do
      serve.step(server)
      tries = tries + 1
      if tries > 5000 then
        test.equal(tries, 0, "the turn never started")
        break
      end
      pause()
    end
    test.truthy(#watch.payloads >= 2)

    local victim = holder(server, "s")
    test.truthy(victim)
    local killed_pid = (victim as serve.Worker).pid
    C.kill(killed_pid, 9)
    killed(killed_pid)
    local before = #watch.payloads
    driven_pool(server)
    test.equal(#watch.payloads, before)

    local after = workers_of(server)
    test.equal(#after, 2)
    for _, worker in ipairs(after) do
      test.notEqual(worker.pid, killed_pid)
    end
    test.equal(holder(server, "s"), nil)

    local recovered = serve.answer(server, THREE_PLAIN_S)
    test.equal(recovered.status, "OK")
    local decoded = tltp.json_decode(recovered.body) as ChatReply
    test.equal(decoded.usage.prompt_tokens_details.cached_tokens, 2)
    local expected = serve.answer(build(fixture(bytes), "m"), THREE_PLAIN_S)
    test.equal(
      decoded.choices[1].message.content,
      (tltp.json_decode(expected.body) as ChatReply).choices[1].message.content
    )

    -- the replacement built its server, so its death is answered like any other
    local replacement = holder(server, "s")
    test.truthy(replacement)
    local replacement_pid = (replacement as serve.Worker).pid
    C.kill(replacement_pid, 9)
    killed(replacement_pid)
    tries = 0
    while true do
      serve.step(server)
      tries = tries + 1
      local victim_gone = true
      local live = workers_of(server)
      for _, worker in ipairs(live) do
        if worker.pid == replacement_pid then
          victim_gone = false
        end
      end
      if victim_gone and #live == 2 then
        break
      end
      if tries > 5000 then
        test.equal(tries, 0, "the replacement's death was not answered")
        break
      end
      pause()
    end
    for _, worker in ipairs(workers_of(server)) do
      test.notEqual(worker.pid, replacement_pid)
    end

    local copy = fixture(bytes)
    local second = pool_build(copy, temp_dir())
    local doomed = workers_of(second)[1].pid
    fs.write(copy, "this is not a gguf file")
    C.kill(doomed, 9)
    killed(doomed)
    tries = 0
    while #workers_of(second) > 1 do
      serve.step(second)
      tries = tries + 1
      if tries > 5000 then
        test.equal(tries, 0, "the replacement never failed")
        break
      end
      pause()
    end
    test.equal(#workers_of(second), 1)
    local surviving = workers_of(second)[1].pid
    for _ = 1, 100 do
      serve.step(second)
      pause()
    end
    test.equal(#workers_of(second), 1)
    test.equal(workers_of(second)[1].pid, surviving)

    serve.stop(second)
    serve.stop(server)
  end
)

test.it(
  "a pool's scratch directory is private and made by this call",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local server = pool_build(fixture(bytes), temp_dir())
    local first = (serve.pool_of(server) as serve.Pool).dir
    local details = fs.stat(first) as fs.Stat
    test.equal((details.mode as number) % 512, 448)

    local second = pool_build(fixture(bytes), temp_dir())
    test.notEqual((serve.pool_of(second) as serve.Pool).dir, first)

    serve.stop(second)
    serve.stop(server)
  end
)

test.it(
  "a stop and a loop that ends leave nothing behind",
  function()
    local bytes = meta_bytes(t001_meta(), t001_entries())
    local server = pool_build(fixture(bytes), temp_dir())
    local pool = serve.pool_of(server) as serve.Pool
    test.truthy(fs.isdir(pool.dir))
    local pids: {integer} = {}
    for _, worker in ipairs(pool.workers) do
      pids[#pids + 1] = worker.pid
    end
    serve.stop(server)
    test.equal(fs.isdir(pool.dir), false)
    for _, pid in ipairs(pids) do
      test.equal(fs.exists("/proc/" .. tostring(pid)), false)
    end
    test.equal(serve.pool_of(server), nil)

    local lines: {string} = {}
    local recorded: {integer} = {}
    local dirs: {string} = {}
    local function watching(
      name: string, args: {string} | nil, opts: process.Options | nil
    ): process.Child | nil, string | nil
      local child, failure = process.spawn(name, args, opts)
      if child is nil then
        return nil, failure
      end
      recorded[#recorded + 1] = child.pid
      dirs[#dirs + 1] = (args as {string})[3]
      return child, nil
    end
    local code = serve.start(
      {
        host = "192.0.2.1",
        port = 9599,
        name = "m",
        path = fixture(bytes),
        sessions = 4,
        kv_cache = "f32",
        context_path = temp_dir(),
        context_bytes = POOL_BUDGET,
        workers = 2,
      },
      function(line: string) lines[#lines + 1] = line end,
      watching
    )
    test.equal(code, 1)
    test.equal(#lines, 1)
    test.includes(lines[1], "BIND")
    test.equal(#recorded, 2)
    for _, pid in ipairs(recorded) do
      test.equal(fs.exists("/proc/" .. tostring(pid)), false)
    end
    test.equal(fs.isdir(dirs[1]), false)
  end
)
