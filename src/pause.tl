local ffi = require("ffi")
local process = require("process")

local pause = {}

ffi.cdef(
  [[
  int usleep(unsigned int usec);
  void Sleep(unsigned int milliseconds);
]]
)

local names: ffi.CNamespace | nil = nil

local function namespace(): ffi.CNamespace
  if names is nil then
    if process.platform == "win32" then
      names = ffi.load("kernel32") as ffi.CNamespace
    else
      names = ffi.C
    end
  end
  return names
end

-- a bounded wait: the file watchers only wake on a change, and the pool must
-- also give up on a worker that dies without writing one
function pause.milliseconds(amount: integer)
  if amount < 1 then
    return
  end
  local space = namespace()
  if process.platform == "win32" then
    local sleep = space.Sleep as ffi.CFunction
    sleep(amount)
  else
    local sleep = space.usleep as ffi.CFunction
    sleep(amount * 1000)
  end
end

return pause
