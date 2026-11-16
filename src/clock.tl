local ffi = require("ffi")

local record clock
  now: function(): number
end

-- the tag is this module's own because fs's stat already declares timespec,
-- and the pointer is void so a test that declares clock_gettime with its own
-- struct keeps linking to the same function
ffi.cdef(
  [[
struct tllm_clock_time { long tv_sec; long tv_nsec; };
int clock_gettime(int clock_id, void *at);
]]
)

local record Wall
  tv_sec: integer
  tv_nsec: integer
end

local record Clocks
  clock_gettime: function(integer, ffi.CData): integer
end

local MONOTONIC: integer = 1

local held = ffi.new("struct tllm_clock_time")

function clock.now(): number
  (ffi.C as Clocks).clock_gettime(MONOTONIC, held as ffi.CData)
  local wall = held as Wall
  return tonumber(wall.tv_sec) + tonumber(wall.tv_nsec) / 1000000000
end

return clock
