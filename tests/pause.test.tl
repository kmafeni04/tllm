local ffi = require("ffi")
local test = require("lde-test")
local pause = require("tllm.pause")

ffi.cdef(
  [[
struct tllm_pause_wall { long seconds; long nanos; };
int clock_gettime(int clock_id, struct tllm_pause_wall *at);
]]
)

local record PauseWall
  seconds: integer
  nanos: integer
end

local record Clocks
  clock_gettime: function(integer, ffi.CData): integer
end

local clocks = ffi.C as Clocks

local function now(): number
  local at = ffi.new("struct tllm_pause_wall") as PauseWall
  clocks.clock_gettime(1, at as ffi.CData)
  return tonumber(at.seconds) + tonumber(at.nanos) / 1e9
end

test.it(
  "a pause waits at least the time it is given",
  function()
    local started = now()
    pause.milliseconds(50)
    test.greaterEqual(now() - started, 0.025)
  end
)
