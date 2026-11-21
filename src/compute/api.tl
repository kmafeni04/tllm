local weights = require("tllm.weights")

-- the published surface: the records and the function types every other
-- module reads, and the table compute/init.tl fills in

local record compute
  enum CacheType
    "f32"
    "f16"
    "q8_0"
    "q4_0"
  end

  record Backend
    path: string
    version: string
    device: string | nil
    memory: function(Backend): integer | nil, integer | nil
  end

  record Maths
    matvec: function(
      Maths, string, {number}, index?: integer
    ): {number} | nil, string | nil
    row: function(Maths, string, integer): {number} | nil, string | nil
    norm: function(
      Maths, {number}, string, number
    ): {number} | nil, string | nil
    rope: function(
      Maths, {number}, integer, integer, number
    ): {number} | nil, string | nil
    softmax: function(Maths, {number}): {number} | nil, string | nil
    add: function(Maths, {number}, {number}): {number} | nil, string | nil
    swiglu: function(Maths, {number}, {number}): {number} | nil, string | nil
    sigmoid_gate: function(
      Maths, {number}, {number}
    ): {number} | nil, string | nil
    close: function(Maths)
  end

  record Geometry
    heads: integer
    kv_heads: integer
    head_length: integer
    rope_dimensions: integer
    rope_base: number
    eps: number
    positions: integer
    experts_used: integer
    linear_key_heads: integer | nil
    linear_key_length: integer | nil
    linear_value_heads: integer | nil
    linear_value_length: integer | nil
    linear_conv_kernel: integer | nil
    cache_type: CacheType
  end

  record Row
    sequence: string
    hidden: {number}
    position: integer
  end

  record Input
    sequence: string
    token: integer
    position: integer
  end

  record Attention
    block: function(Attention, {Row}, string): {{number}} | nil, string | nil
    cache_bytes: function(Attention, string, string): integer | nil
    reserve: function(Attention, string, integer, {string}): string | nil
    release: function(Attention, string)
    snapshot: function(
      Attention, string, string, integer
    ): string | nil, string | nil
    restore: function(Attention, string, string, integer, string): string | nil
    close: function(Attention)
  end

  record Measure
    passes: integer
    tokens: integer
    compute: number
    sampling: number
    ready: number
    routed: integer
    routed_bytes: integer
    expert_hits: integer
    blob_reads: integer
    blob_bytes: integer
    blob_seconds: number
    graphs: integer
  end

  record Routing
    layer: integer
    token: integer
    experts: {integer}
  end

  record Watch
    layer: integer
    token: integer
    measure: Measure
    sink: (function(Routing) | nil)
  end

  record FeedForward
    block: function(
      FeedForward, {Row}, string, watch?: {Watch}
    ): {{number}} | nil, string | nil
    scores: function(FeedForward, {{number}}): {{number}} | nil, string | nil
    close: function(FeedForward)
  end

  record LinearAttention
    block: function(
      LinearAttention, {Row}, string
    ): {{number}} | nil, string | nil
    release: function(LinearAttention, string)
    snapshot: function(
      LinearAttention, string, string, integer
    ): string | nil, string | nil
    restore: function(
      LinearAttention, string, string, integer, string
    ): string | nil
    close: function(LinearAttention)
  end

  -- what one model holds on the card: the tensors it copies, the bytes it
  -- charged for them, the experts it keeps there per block, and the rest
  record Placement
    budget: integer
    reserve: integer
    resident: {string: boolean}
    resident_bytes: integer
    slots: {integer}
    slot_bytes: integer
    unplaced_bytes: integer
  end

  record Model
    positions: integer
    sequences: integer
    ubatch: integer
    measure: Measure
    placement: Placement
    routing_sink: (function(Routing) | nil)
    forward: function(Model, integer, integer): {number} | nil, string | nil
    reserve: function(Model, string, integer): string | nil
    release: function(Model, string)
    cache_bytes: function(Model, string, string): integer | nil
    capacity: function(Model, string): integer | nil
    pass: function(Model, {Input}): {{number}} | nil, string | nil
    save: function(Model, string, {integer}, string): string | nil
    load: function(Model, string, string): {integer} | nil, string | nil
    close: function(Model)
  end

  load: function(path?: string, offload?: boolean): Backend | nil, string | nil
  unload: function(Backend)
  attach: function(Backend, weights.Handle): Maths | nil, string | nil
  attend: function(
    Backend, weights.Handle, Geometry
  ): Attention | nil, string | nil
  feed_forward: function(
    Backend, weights.Handle, Geometry
  ): FeedForward | nil, string | nil
  linear_attention: function(
    Backend, weights.Handle, Geometry
  ): LinearAttention | nil, string | nil
  place: function(weights.Handle, integer, integer): Placement
  model: function(
    Backend,
    weights.Handle,
    integer | nil,
    CacheType | nil,
    sequences?: integer,
    ubatch?: integer,
    vram?: integer,
    profile?: string
  ): Model | nil, string | nil
end

return compute
