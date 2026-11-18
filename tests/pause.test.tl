local test = require("lde-test")
local clock = require("tllm.clock")
local pause = require("tllm.pause")

test.it(
  "a pause waits at least the time it is given",
  function()
    local started = clock.now()
    pause.milliseconds(50)
    test.greaterEqual(clock.now() - started, 0.025)
  end
)
