local record ggml
  record Type
    code: integer
    name: string
    block_elements: integer
    block_bytes: integer
  end

  type_of: function(code: integer): Type | nil
end

local type Kind = ggml.Type

local record Row
  name: string
  block_elements: integer
  block_bytes: integer
end

local ROWS: {integer: Row} = {
  [0] = {name = "f32", block_elements = 1, block_bytes = 4},
  [1] = {name = "f16", block_elements = 1, block_bytes = 2},
  [2] = {name = "q4_0", block_elements = 32, block_bytes = 18},
  [3] = {name = "q4_1", block_elements = 32, block_bytes = 20},
  [6] = {name = "q5_0", block_elements = 32, block_bytes = 22},
  [7] = {name = "q5_1", block_elements = 32, block_bytes = 24},
  [8] = {name = "q8_0", block_elements = 32, block_bytes = 34},
  [9] = {name = "q8_1", block_elements = 32, block_bytes = 36},
  [10] = {name = "q2_K", block_elements = 256, block_bytes = 84},
  [11] = {name = "q3_K", block_elements = 256, block_bytes = 110},
  [12] = {name = "q4_K", block_elements = 256, block_bytes = 144},
  [13] = {name = "q5_K", block_elements = 256, block_bytes = 176},
  [14] = {name = "q6_K", block_elements = 256, block_bytes = 210},
  [15] = {name = "q8_K", block_elements = 256, block_bytes = 292},
  [16] = {name = "iq2_xxs", block_elements = 256, block_bytes = 66},
  [17] = {name = "iq2_xs", block_elements = 256, block_bytes = 74},
  [18] = {name = "iq3_xxs", block_elements = 256, block_bytes = 98},
  [19] = {name = "iq1_s", block_elements = 256, block_bytes = 50},
  [20] = {name = "iq4_nl", block_elements = 32, block_bytes = 18},
  [21] = {name = "iq3_s", block_elements = 256, block_bytes = 110},
  [22] = {name = "iq2_s", block_elements = 256, block_bytes = 82},
  [23] = {name = "iq4_xs", block_elements = 256, block_bytes = 136},
  [24] = {name = "i8", block_elements = 1, block_bytes = 1},
  [25] = {name = "i16", block_elements = 1, block_bytes = 2},
  [26] = {name = "i32", block_elements = 1, block_bytes = 4},
  [27] = {name = "i64", block_elements = 1, block_bytes = 8},
  [28] = {name = "f64", block_elements = 1, block_bytes = 8},
  [29] = {name = "iq1_m", block_elements = 256, block_bytes = 56},
  [30] = {name = "bf16", block_elements = 1, block_bytes = 2},
  [34] = {name = "tq1_0", block_elements = 256, block_bytes = 54},
  [35] = {name = "tq2_0", block_elements = 256, block_bytes = 66},
  [39] = {name = "mxfp4", block_elements = 32, block_bytes = 17},
  [40] = {name = "nvfp4", block_elements = 64, block_bytes = 36},
  [41] = {name = "q1_0", block_elements = 128, block_bytes = 18},
  [42] = {name = "q2_0", block_elements = 64, block_bytes = 18},
  [43] = {name = "f8_e4m3", block_elements = 1, block_bytes = 1},
}

local TYPES: {integer: Kind} = {}

for code, row in pairs(ROWS) do
  TYPES[code] = {
    code = code,
    name = row.name,
    block_elements = row.block_elements,
    block_bytes = row.block_bytes,
  }
end

function ggml.type_of(code: integer): Kind | nil
  return TYPES[code]
end

return ggml
