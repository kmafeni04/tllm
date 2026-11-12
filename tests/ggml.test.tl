local test = require("lde-test")
local ggml = require("tllm.ggml")

local record Row
  code: integer
  name: string
  block_elements: integer
  block_bytes: integer
end

local ROWS: {Row} = {
  {code = 0, name = "f32", block_elements = 1, block_bytes = 4},
  {code = 1, name = "f16", block_elements = 1, block_bytes = 2},
  {code = 2, name = "q4_0", block_elements = 32, block_bytes = 18},
  {code = 3, name = "q4_1", block_elements = 32, block_bytes = 20},
  {code = 6, name = "q5_0", block_elements = 32, block_bytes = 22},
  {code = 7, name = "q5_1", block_elements = 32, block_bytes = 24},
  {code = 8, name = "q8_0", block_elements = 32, block_bytes = 34},
  {code = 9, name = "q8_1", block_elements = 32, block_bytes = 36},
  {code = 10, name = "q2_K", block_elements = 256, block_bytes = 84},
  {code = 11, name = "q3_K", block_elements = 256, block_bytes = 110},
  {code = 12, name = "q4_K", block_elements = 256, block_bytes = 144},
  {code = 13, name = "q5_K", block_elements = 256, block_bytes = 176},
  {code = 14, name = "q6_K", block_elements = 256, block_bytes = 210},
  {code = 15, name = "q8_K", block_elements = 256, block_bytes = 292},
  {code = 16, name = "iq2_xxs", block_elements = 256, block_bytes = 66},
  {code = 17, name = "iq2_xs", block_elements = 256, block_bytes = 74},
  {code = 18, name = "iq3_xxs", block_elements = 256, block_bytes = 98},
  {code = 19, name = "iq1_s", block_elements = 256, block_bytes = 50},
  {code = 20, name = "iq4_nl", block_elements = 32, block_bytes = 18},
  {code = 21, name = "iq3_s", block_elements = 256, block_bytes = 110},
  {code = 22, name = "iq2_s", block_elements = 256, block_bytes = 82},
  {code = 23, name = "iq4_xs", block_elements = 256, block_bytes = 136},
  {code = 24, name = "i8", block_elements = 1, block_bytes = 1},
  {code = 25, name = "i16", block_elements = 1, block_bytes = 2},
  {code = 26, name = "i32", block_elements = 1, block_bytes = 4},
  {code = 27, name = "i64", block_elements = 1, block_bytes = 8},
  {code = 28, name = "f64", block_elements = 1, block_bytes = 8},
  {code = 29, name = "iq1_m", block_elements = 256, block_bytes = 56},
  {code = 30, name = "bf16", block_elements = 1, block_bytes = 2},
  {code = 34, name = "tq1_0", block_elements = 256, block_bytes = 54},
  {code = 35, name = "tq2_0", block_elements = 256, block_bytes = 66},
  {code = 39, name = "mxfp4", block_elements = 32, block_bytes = 17},
  {code = 40, name = "nvfp4", block_elements = 64, block_bytes = 36},
  {code = 41, name = "q1_0", block_elements = 128, block_bytes = 18},
  {code = 42, name = "q2_0", block_elements = 64, block_bytes = 18},
  {code = 43, name = "f8_e4m3", block_elements = 1, block_bytes = 1},
}

test.it(
  "every ggml type code keeps its block shape",
  function()
    for _, row in ipairs(ROWS) do
      local kind = ggml.type_of(row.code) as ggml.Type
      test.truthy(kind, tostring(row.code))
      test.equal(kind.code, row.code, row.name)
      test.equal(kind.name, row.name, row.name)
      test.equal(kind.block_elements, row.block_elements, row.name)
      test.equal(kind.block_bytes, row.block_bytes, row.name)
    end
  end
)

test.it(
  "a code with no type has no entry",
  function()
    local missing: {integer} = {4, 5, 31, 32, 33, 36, 37, 38, 99, -1}
    for _, code in ipairs(missing) do
      test.equal(ggml.type_of(code), nil, tostring(code))
    end
  end
)
