local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")

test.it(
  "the placed ggml loads, resolves its sibling and reports its version",
  function()
    local backend = compute.load() as compute.Backend
    test.equal(backend.path, "target/tllm/libggml.so")

    local maps = fs.read("/proc/self/maps") as string
    test.includes(maps, "target/tllm/libggml-base")

    local again = compute.load(backend.path) as compute.Backend
    test.equal(again.path, backend.path)
  end
)

test.it(
  "a library that cannot be loaded is refused",
  function()
    local dir = os.tmpname()
    os.remove(dir)
    fs.mkdirAll(dir)
    local fake = dir .. "/not-a-library.so"
    fs.write(fake, "this is not a library")

    for _, lib_path in ipairs({dir .. "/missing.so", fake}) do
      local backend, message = compute.load(lib_path)
      test.equal(backend, nil)
      local refusal = message as string
      test.includes(refusal, lib_path)
      test.equal(refusal:find("\n"), nil)
    end

    fs.rmdir(dir)
  end
)
