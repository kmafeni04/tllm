-- one process of a pool: it builds the endpoint's server, reports that it is
-- ready and then answers the bodies the parent writes into its directory

local fs = require("fs")
local compute = require("tllm.compute")
local pause = require("tllm.pause")
local serve = require("tllm.serve")
local tltp = require("tltp")

local record EndpointJson
  host: string
  port: integer
  name: string
  path: string
  sessions: integer
  kv_cache: compute.CacheType | nil
  context_path: string | nil
  context_bytes: integer | nil
  ubatch: integer | nil
  vram: integer | nil
  profile: string | nil
  prefetch: boolean | nil
  threads: integer | nil
end

local function write_frame(path: string, text: string)
  local file = io.open(path, "a")
  if file is nil then
    return
  end
  file:write(tostring(#text) .. ":" .. text)
  file:flush()
  file:close()
end

local function endpoint_of(text: string): serve.Endpoint | nil
  local decoded = tltp.json_decode(text) as EndpointJson | nil
  if decoded is nil then
    return nil
  end
  return {
    host = decoded.host,
    port = decoded.port,
    name = decoded.name,
    path = decoded.path,
    sessions = decoded.sessions,
    kv_cache = decoded.kv_cache,
    context_path = decoded.context_path,
    context_bytes = decoded.context_bytes,
    ubatch = decoded.ubatch,
    vram = decoded.vram,
    profile = decoded.profile,
    prefetch = decoded.prefetch,
    threads = decoded.threads,
  }
end

local function waited(watcher: fs.Watcher | nil, path: string)
  while not fs.isfile(path) do
    if watcher is nil then
      pause.milliseconds(2)
    else
      watcher:wait()
    end
  end
end

local function answered(
  server: tltp.Server,
  dir: string,
  base: string,
  body: string,
  watcher: fs.Watcher | nil
)
  local response, stream_fn = serve.stream(server, body)
  local out = dir .. "/" .. base .. ".out"
  if stream_fn is nil then
    write_frame(
      out,
      tltp.json_encode(
        {kind = "json", status = response.status, body = response.body}
      ) as string
    )
    return
  end

  write_frame(out, tltp.json_encode({kind = "stream"}) as string)
  local run = stream_fn
  local co = coroutine.create(
    function(): boolean | nil
      return run(
        function(payload: string | tltp.SSEMessage): string | nil, string | nil
          write_frame(out, payload as string)
          return nil, nil
        end
      )
    end
  )
  local ok = coroutine.resume(co)
  if not ok then
    return
  end
  local tick = dir .. "/" .. base .. ".go"
  while true do
    waited(watcher, tick)
    if not fs.isfile(tick) then
      return
    end
    local live = serve.step(server)
    fs.delete(tick)
    if live == 0 then
      break
    end
  end
  coroutine.resume(co)
end

local function served(
  server: tltp.Server, dir: string, name: string, watcher: fs.Watcher | nil
)
  local body = fs.read(dir .. "/" .. name)
  if body is nil then
    return
  end
  local base = name:sub(1, #name - 3)
  fs.delete(dir .. "/" .. name)
  answered(server, dir, base, body, watcher)
  fs.write(dir .. "/" .. base .. ".done", "")
end

local function next_jobs(dir: string): {string} | nil
  local batch: {string} = {}
  local next_entry = fs.readdir(dir)
  if next_entry is nil then
    return nil
  end
  while true do
    local found = next_entry()
    if found is nil then
      break
    end
    local name = (found as fs.DirEntry).name
    if name:sub(-3) == ".in" then
      batch[#batch + 1] = name
    end
  end
  table.sort(batch)
  return batch
end

local function main()
  local dir = arg[1]
  local slot = arg[2]
  if dir is nil or slot is nil then
    return
  end
  local workerdir = dir .. "/w" .. slot
  local text = io.read("*a") as string | nil
  local endpoint = endpoint_of(text or "")
  if endpoint is nil then
    fs.write(workerdir .. "/failed", "endpoint: not a json object\n")
    return
  end

  local server, failure = serve.build(endpoint)
  if server is nil then
    fs.write(workerdir .. "/failed", (failure as string) .. "\n")
    return
  end

  local watcher = fs.watch(
    workerdir, function(_event: fs.WatchEvent, _name: string) end, nil
  )
  fs.write(workerdir .. "/ready", "")

  while true do
    local batch = next_jobs(workerdir)
    if batch is nil then
      break
    end
    local jobs = batch as {string}
    if #jobs == 0 then
      if watcher is nil then
        pause.milliseconds(2)
      else
        watcher:wait()
      end
    else
      for _, name in ipairs(jobs) do
        served(server, workerdir, name, watcher)
      end
    end
  end
end

main()
