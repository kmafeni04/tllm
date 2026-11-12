local cli = require("tllm.cli")

local function write_out(text: string)
  io.stdout:write(text)
end

local function write_err(text: string)
  io.stderr:write(text)
end

os.exit(cli.run(arg, write_out, write_err), true)
