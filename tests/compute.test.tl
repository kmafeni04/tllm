local fs = require("fs")
local test = require("lde-test")
local compute = require("tllm.compute")

test.it(
  "the placed ggml loads, resolves its sibling and reports its version",
  function()
    local backend = compute.load() as compute.Backend
    test.equal(backend.path, "target/tllm/libggml.so")
    test.equal(backend.version, "0.29.0")

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

test.it(
  "a matrix times a vector comes back in single precision",
  function()
    local backend = compute.load() as compute.Backend

    test.deepEqual(backend:matvec({{1, 2, 3}, {4, 5, 6}}, {1, 1, 1}), {6, 15})
    test.deepEqual(backend:matvec({{1, -2, 0.5}}, {4, 2, 1}), {0.5})
    test.deepEqual(backend:matvec({{0.1, 0.2}}, {1, 1}), {0.30000001192092896})
  end
)

test.it(
  "a matrix of the wrong shape is refused",
  function()
    local backend = compute.load() as compute.Backend

    local empty_matrix, matrix_message = backend:matvec({}, {1})
    test.equal(empty_matrix, nil)
    test.includes(matrix_message as string, "empty")

    local empty_row, row_message = backend:matvec({{}}, {1})
    test.equal(empty_row, nil)
    test.includes(row_message as string, "empty")

    local ragged, ragged_message = backend:matvec({{1, 2}, {3}}, {1, 1})
    test.equal(ragged, nil)
    local refusal = ragged_message as string
    test.includes(refusal, "1")
    test.includes(refusal, "2")

    test.deepEqual(backend:matvec({{1, 2, 3}, {4, 5, 6}}, {1, 1, 1}), {6, 15})
  end
)
