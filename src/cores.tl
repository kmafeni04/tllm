-- the thread count a model computes with when neither a flag nor a file names
-- one, read from the machine's /proc/cpuinfo text, and the range ggml takes

local fs = require("fs")

local record cores
  -- the model's default thread count, from a /proc/cpuinfo text
  default: function(cpuinfo: string): integer
  -- this machine's default thread count
  machine: function(): integer
  -- whether a count is one ggml can be given
  fits: function(count: number): boolean
end

-- GGML_MAX_N_THREADS in the pinned ggml source: the size of the thread
-- parameters' cpumask, past which ggml spawns threads without a check
local MAX: integer = 512

function cores.fits(count: number): boolean
  return count % 1 == 0 and count >= 1 and count <= MAX
end

function cores.machine(): integer
  return cores.default(fs.read("/proc/cpuinfo") or "")
end

-- a text naming no core at all is one thread
function cores.default(cpuinfo: string): integer
  local pairs_held: {string: boolean} = {}
  local held = 0
  local processors = 0
  local physical: string | nil = nil
  -- the key stops before the padding a line puts around its colon, which is a
  -- tab in the kernel's own text and a space in a hand-written one
  for line in cpuinfo:gmatch("[^\n]+") do
    local key, value = line:match("^([%w_ ]-)%s*:%s*(.-)%s*$")
    if key == "processor" then
      processors = processors + 1
    elseif key == "physical id" then
      physical = value
    elseif key == "core id" and physical is string then
      local pair = physical .. ":" .. value
      if not pairs_held[pair] then
        pairs_held[pair] = true
        held = held + 1
      end
    end
  end

  -- the cores the text reports, or the processors when it reports no core,
  -- less the calling thread, which is one of them
  if held == 0 then
    held = processors
  end
  if held < 2 then
    return 1
  end
  return held - 1
end

return cores
