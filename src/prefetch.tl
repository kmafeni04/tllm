-- the gate read-ahead opens and closes by: the room the page cache has, and
-- how many of the experts a window named landed whole

local fs = require("fs")

local record prefetch
  -- the bytes the machine has free for a page cache, or nil where it does not
  -- say, from MemAvailable in /proc/meminfo
  memory: function(): integer | nil
  -- open read-ahead, close it, or leave it as it is, from the room a window had
  decide: function(
    room: number | nil, named: integer, landed: integer, open: boolean
  ): boolean
end

-- a kibibyte is what /proc/meminfo counts in
local KIB: integer = 1024

function prefetch.memory(): integer | nil
  local text = fs.read("/proc/meminfo")
  if text is nil then
    return nil
  end
  for line in (text as string):gmatch("[^\n]+") do
    local found = line:match("^MemAvailable:%s+(%d+)")
    if found is string then
      return (tonumber(found) as integer) * KIB
    end
  end
  return nil
end

function prefetch.decide(
  room: number | nil, named: integer, landed: integer, open: boolean
): boolean
  if room is nil then
    return false
  end
  local half = named == 0 or landed * 2 >= named
  if room >= 0.65 and half then
    return true
  end
  if room < 0.55 or not half then
    return false
  end
  return open
end

return prefetch
