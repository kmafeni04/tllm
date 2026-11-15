# 24: A cache type the user chooses, held as a ggml tensor

**Status**: done   **Depends on**: 04, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23

## Problem & context

`12` keeps each block's keys and values in two `float[?]` arrays per prefix (`cache_of`,
`src/compute.tl`) and scores them and weights the values in a hand-written triple loop per query
head. Two things are wrong with that. Nothing a person runs can trade memory for precision: f32 is
the only option, 4 bytes per cached number, so a position costs 40 KB across the local file's 10
attention blocks and `23`'s 910-position conversation holds 37.3 MB. And no ggml kernel touches the
cache, so a token's attention is a Lua loop over every cached position.

This spec does both. A prefix's keys and values become two ggml tensors in host memory, of a type
the geometry names: `f32`, `f16`, `q8_0` or `q4_0`, at 4, 2, `34/32` and `18/32` bytes per number,
and `config.json`'s `kv_cache` chooses it, `f16` when the key is absent. The score and value
products become `ggml_mul_mat` calls, the scores over a view of the key tensor and the attention
numbers over a transposed copy of the value view, because ggml contracts along a tensor's first
dimension, so the hottest loop in the project becomes a few graph nodes and the position being
computed is written into the tensors before them and read back out of them, over positions 0 to
`position`. `f16` is the default because it halves
a conversation's memory with a rounding of one part in two thousand, which is what llama.cpp ships
for the same reason; at 910 positions the local file's conversation goes from 37.3 MB to 18.6 MB,
to 9.9 MB at `q8_0` and to 5.2 MB at `q4_0`.

One type for both keys and values. llama.cpp splits them because a key feeds a softmax, where a
small error shifts weights, while a value only adds up, so its value cache tolerates less precision
than its key cache; a split is a later spec if measurement on this engine says the same, and nothing
here forecloses it.

`q8_0` and `q4_0` hold whole 32-number blocks, so a head of 80 numbers cannot be held in either. A
type that cannot hold a model's head length is refused with one line naming the type and the length,
and nothing is silently substituted: the head length is in the GGUF, which serve opens when it
builds a model rather than when it reads `config.json`, so that line arrives then and not at
startup.

The cache stays host memory. With `21`'s offload a block's graph tensors live in the device buffer,
so the cache is copied into the device's memory for the length of a call and never kept there: this
saves RAM, never VRAM. No case here covers that copy, because the suite runs on the CPU backend and
`21`'s device cases skip when no card is present; the copy is what `21`'s amended text states and
what a reader of the block call sees.

Precision is pinned two ways. At block level, a type's numbers must stay within a stated fraction of
the largest f32 number for that call: 1e-3 for `f16`, 1e-2 for `q8_0` and 1e-1 for `q4_0`. At
whole-model level, the sampled reply must be the reply an f32 cache gave for all four types: a type
that changes what the model says is not a memory option, it is a different model. The cost of a long
read is pinned as a ratio against a cache of one position, so it does not depend on the machine.

Because the cache is no longer always f32, every test in the suite that pins a number or a reply
recorded against an f32 cache names `f32` in the geometry or the model build. Each such spec was
amended; nothing else about those specs changes.

The four names are one enum, declared once in `compute` and used by `04`'s `Settings`, `12`'s
`Geometry` and `18`'s `Endpoint`, so no consumer can be handed a name the engine does not hold:
`04`'s check turns the file's text into the enum where it accepts it, and a `Geometry` built by
hand says the same four names. A runtime check stays behind it, because a caller reaching the
engine from plain Lua can pass anything.

Refusals keep the shape `12` gave them: nil and one line.

```teal
local weights = require("tllm.weights")

local enum CacheType
  "f32"
  "f16"
  "q8_0"
  "q4_0"
end

local record Settings
  -- 04's other fields, unchanged
  kv_cache: CacheType | nil
end

local record compute
  record Geometry
    -- 12's other fields, unchanged
    cache_type: CacheType
  end

  record Attention
    block: function(
      Attention, {Row}, string
    ): {{number}} | nil, string | nil
    cache_bytes: function(Attention, string, string): integer | nil
    release: function(Attention, string)
  end

  model: function(
    Backend,
    weights.Handle,
    integer | nil,
    CacheType | nil,
    sequences?: integer
  ): Model | nil, string | nil
end

local record serve
  record Endpoint
    kv_cache: CacheType | nil
  end
end
```

## User stories

- US-001 (P1) As the engine author, I want the cache's type chosen in `config.json` and the bytes
  it costs readable from the handle, so that memory and precision are the caller's call.
- US-002 (P1) As the person running the server, I want f16 when I set nothing, so that a conversation costs half of today's memory without configuring anything.
- US-003 (P2) As the person running the server, I want a type my model cannot hold refused with one line, so that I am told rather than quietly given something else.
- US-004 (P1) As the engine author, I want the cache held as a ggml tensor whose products ggml computes, so that a token does not spend its time in a Lua loop over cached positions.
- US-005 (P2) As a chat client, I want the reply to be the reply the float cache gave whatever type the server is set to, so that the memory win does not change what the model says.

## Functional requirements

- FR-001 (US-001) WHEN `config.json` sets `kv_cache` to `f32`, `f16`, `q8_0` or `q4_0`, THE SYSTEM
  SHALL hold every prefix's keys and values in that type for every model the server builds.
- FR-002 (US-002) WHEN `config.json` sets no `kv_cache`, THE SYSTEM SHALL hold every prefix's keys
  and values in f16.
- FR-003 (US-003) IF `kv_cache` is set and is not one of `f32`, `f16`, `q8_0` and `q4_0`, compared
  case-sensitively, THEN THE SYSTEM SHALL write one line through the stderr writer naming
  `kv_cache` and return exit code 1 without calling a start function.
- FR-004 (US-003) IF the cache type is `q8_0` or `q4_0` and the model's run holds a full-attention
  block whose `head_length` is not a multiple of 32, THEN THE SYSTEM SHALL return nil and one line
  naming the type and the head length, so that no cache of that type is made.
- FR-005 (US-001) WHEN `cache_bytes(sequence, prefix)` is called, THE SYSTEM SHALL return the bytes
  allocated for that sequence at every block of the run, whether `reserve` gave it those positions
  or a block call named it first, and nil when neither has, a sequence holding no positions taken
  being an answer and not a refusal.
- FR-006 (US-001) WHEN a sequence's cache is allocated, THE SYSTEM SHALL have `cache_bytes` report
  the bytes of that sequence's positions, the reserved ones or the geometry's `positions` when a
  block call named it first, being
  `2 * positions * kv_heads * head_length` numbers at 4 bytes each for f32, 2 for f16,
  `34 / 32` for q8_0 and `18 / 32` for q4_0, whatever the number of positions filled since.
- FR-007 (US-004) WHEN a block call runs at position 1024 over a prefix's cache, THE SYSTEM SHALL
  return within ten times the time a call at position 1 takes, both over the backend
  `compute.load(nil, false)` returns.
- FR-008 (US-005) WHEN a block call runs over a cache that holds an earlier position, THE SYSTEM
  SHALL return numbers of which each is within `1e-3 * m` for f16, `1e-2 * m` for q8_0 and
  `1e-1 * m` for q4_0 of the number an f32 cache at the same position gives, where `m` is the
  largest absolute number that same call returned for the f32 cache, and the key and value numbers
  of the position being computed SHALL be written into the cache before that call's products and
  read back out of it.
- FR-009 (US-005) WHEN the whole model forwards a prompt and samples its ids with a cache of any of
  the four types, THE SYSTEM SHALL return the ids it returns with an f32 cache.
- FR-010 (US-003) IF `compute.model` is given a cache type that is not one of `f32`, `f16`, `q8_0`
  and `q4_0`, THEN THE SYSTEM SHALL return nil and one line naming `model` and the type.

## Non-goals

- No measurement against the local 23 GB file: FR-008 pins a fixture's numbers, FR-009 its reply and
  FR-007 its cost, and the real file's replies and tokens per second are left to a later spec.
- No split between the key and value types, and no type this spec does not name: bf16, the K-quants
  and every other row of `src/ggml.tl` stay out.
- No cache on the device and no case for the copy: the keys and values stay host memory, the device
  holds only a copy for the length of a call, and no conversation's VRAM grows with its context.

## Tests

All seven cases run on the CPU backend, `compute.load(nil, false)`. T-005 and T-006 build a second
handle or model with `cache_type` `"f32"` in the same case and compare against what it returns, so
no number is recorded by hand.

- T-001 (FR-002, FR-005, FR-006): `a cache reports what its type costs`, proves with a fixture
  holding, under `blk.0.` and `blk.1.`, f32 tensors of 32 numbers a row: `attn_norm.weight` 32
  numbers of 1; `attn_q.weight` 128 rows of 32 numbers, row `r` holding zeros with a 1 in column
  `(r - 1) % 32 + 1` for rows 1 to 96 and zeros in rows 97 to 128; `attn_q_norm.weight` and
  `attn_k_norm.weight` 32 numbers of 1; `attn_k.weight` and `attn_v.weight` 32 rows of 32 numbers,
  row `j` holding zeros with `j / 32` in column 1; `attn_output.weight` 32 rows of 64 numbers, row
  `r` holding zeros with a 1 in column `(r - 1) % 32 + 1`, and with a geometry of 2 heads, 1
  key/value head, heads of 32 numbers, rope over 2 numbers at base 10000, `eps` 0, 4 positions and
  2 experts used, that `cache_bytes` gives nil for `blk.0.` before any call and for `blk.2.` after
  one, and that after a block call on a hidden of 32 numbers of 1 at position 0 of `blk.0.` it
  gives 1024 for `f32`, 512 for `f16`, 272 for `q8_0` and 144 for `q4_0`, each being 4 positions of
  1 head of 32 numbers in each of keys and values at that type's bytes per number.
- T-002 (FR-001, FR-002, FR-003): `the configured type reaches the server, and one it does not name
  is refused`, proves on `04`'s config surface that a file holding
  `{"host":"127.0.0.1","port":8080,"kv_cache":"q8_0","models":{}}` passes `start` an endpoint whose
  `kv_cache` is `q8_0`, that the same file without `kv_cache` passes `f16`, and that a file holding
  `"kv_cache":"q8_0x"` writes one line naming `kv_cache` through the stderr writer and returns
  exit code 1 without calling `start`.
- T-003 (FR-004, FR-010): `a type the model cannot hold is refused`, proves on the model fixture
  built over `12/T-001`'s block, whose full-attention block has 2-number heads, that
  `compute.model(backend, handle, 4, "q8_0")` returns nil and one line holding `q8_0` and 2, that
  `compute.model(backend, handle, 4, "q4_0")` returns nil and one line holding `q4_0` and 2, that
  `compute.model(backend, handle, 4, "q4_0x")` returns nil and one line holding `model` and
  `q4_0x`, and that `compute.model(backend, handle, 4, "f16")` and
  `compute.model(backend, handle, 4, "f32")` each return a model and nil.
- T-004 (FR-007): `a thousand cached positions cost a token under ten short ones`, proves on T-001's
  fixture with a geometry holding 1025 positions and `cache_type` `"f16"` that a handle that has
  taken positions 0 to 1023 answers the call at position 1024 in no more than ten times the
  shortest of five timings, those five being the wall times of the call at position 1, each taken
  on its own fresh handle that has taken position 0, after one throwaway handle of the same shape
  has been run.
- T-005 (FR-008): `each type's numbers stay inside their tolerance`, proves on T-001's fixture
  remade so that `attn_k.weight` and `attn_v.weight` rows hold `j / 32`, that after a block call on
  a hidden of 32 numbers of 1 at position 0 of `blk.0.`, a block call on a hidden of 32 numbers
  holding 2 then 31 ones at position 1 returns, for `f16` within `1e-3 * m`, for `q8_0` within
  `1e-2 * m` and for `q4_0` within `1e-1 * m`, the 32 numbers the `f32` handle returned for that
  call, `m` being the largest absolute of those 32 numbers, and that for each of the three at
  least one of those numbers differs from the `f32` number by more than 1e-5, so a cache that
  stayed f32 fails the case; a worst deviation above twice the tolerance is a fault in the
  cache and not in the fixture.
- T-006 (FR-009): `every type gives the reply the float cache gave`, proves with a fixture holding
  `16/T-001`'s metadata and keys with `embedding_length` 32 in place of 2, `attention.head_count` 1
  and `rope.dimension_count` 2 unchanged, so the head holds 32 numbers, `context_length` 4,
  `15/T-001`'s tensors with every dimension of 2 replaced by 32 and `attn_q.weight` over 64 rows in
  place of 4, every number the draw of `16/FR-006`'s generator seeded 0, doubled and reduced by 1
  so that it lies between -1 and 1, taken in tensor order and row after row, that order being
  `blk.0.`'s tensors as `13`'s block reads them, the seven of attention then
  `post_attention_norm.weight`, `ffn_gate_inp.weight`, `ffn_gate_exps.weight`, `ffn_up_exps.weight`,
  `ffn_down_exps.weight`, `ffn_gate_inp_shexp.weight`, `ffn_up_shexp.weight`,
  `ffn_gate_shexp.weight` and `ffn_down_shexp.weight`, except `output.weight`, whose three rows of
  32 numbers hold 1, 1/2 and 1/4 in column 1 and zeros elsewhere, `output_norm.weight` 32 numbers
  of 1, and `token_embd.weight`, whose three rows of 32 numbers are ones, minus ones and ones, so
  no token's row holds zeros, that `compute.model(backend, handle, 4, cache_type)` returns a model
  for each of the four types and that `generate.text(model, tokenizer, "a", {max_tokens = 2, seed
  = 0, end_id = 3})` returns, for each of the four, the text the model built with `cache_type`
  `"f32"` returns for the same call, that text holding two pieces because `end_id` 3 is an id the
  three-piece vocabulary does not hold.
- T-007 (FR-001, FR-004): `the configured type reaches the model the server builds`, proves on
  `18/T-001`'s fixture built from an endpoint whose `kv_cache` is `"q8_0"` and whose `sessions` is
  4 that `answer(server, '{"messages":[{"role":"user","content":"a"}]}')` answers
  `InternalServerError` with a body holding `q8_0` and `2`, that the same call on a server built
  from an endpoint whose `kv_cache` is `"f32"` answers `OK`, and that a server whose `kv_cache` is
  `"f16"` answers `OK` with the body the `f32` server answers.

## Open questions

- None.

## Amendments

- None.
