# 35: The expert tier, a VRAM budget and what sits in it

**Status**: done   **Depends on**: 04, 06, 11, 18, 21, 24, 28, 29, 32, 33, 34

## Problem & context

`21/FR-002` copies every tensor of a model into device memory or computes the whole model on the CPU.
For the local 23 GB file on a 16 GB card that is neither: the copy cannot be allocated, `21/FR-007`
falls back to the CPU, and the card sits idle while every weight is read out of the mapping. So the
largest model this engine is meant for cannot use the machine's fast path at all, and the only
question a person can ask about the card is whether the whole file fits on it.

The fix is to stop asking for the whole file. A model is a few dense weights every token reads and a
huge number of expert matrices a token touches a handful of. The dense weights go on the card
because they are read on every pass; the bytes left over buy whole experts, sized per layer rather
than in units of the largest expert in the model, which is what strata measures as +13% decode at a
1K context for the same VRAM. Whatever does not fit is not lost: it stays in the mapping, read the
way `34` reads a chosen expert whole, or copied into the card for the length of one call.

The order of value comes from strata and is adapted to this engine's tensor names. It is one list:

1. `token_embd.weight`, the row every pass gathers from.
2. Every full-attention block's dense weights, in block order: `attn_norm.weight`, `attn_q.weight`,
   `attn_k.weight`, `attn_v.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`,
   `attn_output.weight`, `post_attention_norm.weight`.
3. Every linear block's (mixer's) dense weights, in block order: `attn_norm.weight`,
   `attn_qkv.weight`, `attn_gate.weight`, `ssm_beta.weight`, `ssm_alpha.weight`,
   `ssm_conv1d.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_norm.weight`, `ssm_out.weight`,
   `post_attention_norm.weight`.
4. Every block's routers, in block order: `ffn_gate_inp.weight`, `ffn_gate_inp_shexp.weight`.
5. Every block's shared experts, in block order: `ffn_up_shexp.weight`, `ffn_gate_shexp.weight`,
   `ffn_down_shexp.weight`.
6. The head: `output_norm.weight`, `output.weight`.
7. A reservation for the bytes the model's own graphs need on the card, which is what `32/FR-008`
   sizes for a pass at the geometry's positions: those buffers hold every row's scores and the
   cache rows a block's graph copies in for the length of a call.
8. Every byte still free, spent on whole experts.

Nothing here changes where the attention cache lives. `21/FR-003` and `24` keep each prefix's keys
and values in host memory and copy the rows a call needs into the card, and that copy is part of the
buffers tier 7 reserves, so the reservation is the cache's VRAM cost and not a second cache.

The tier is a plan and a copy. The plan is arithmetic over the handle's tensor bytes and a budget,
so it is testable without a card, and it is published as `compute.place` for that reason. The copy
is the model's own: the tensors the plan marks and the experts it places are copied into one device
buffer sized to the plan's `resident_bytes` plus its `slot_bytes`, that buffer belongs to the model
and its `close` frees it, and every other tensor keeps the mapping's pointer, which a graph that
computes on the card is handed with `host_tensor` for the length of a call, exactly as a graph
already takes the attention history. A maths handle a block builds on its own, the `attach`,
`attend`, `feed_forward` and `linear_attention` of `11` and `21`, takes no budget and keeps the
whole-model copy `21/FR-002` gives it, so a placement is never shared between two models over one
handle. A pass over a partially resident model is therefore a pass whose graph is assembled from two
sources of bytes and multiplies each by the type that weight already holds. The numbers do not move:
the bytes are the same bytes, so the product is the same product.

An expert the plan marks is a device descriptor, and an expert it does not mark is one the graph
copies in from the mapping each pass, which is where `34`'s whole read goes: `34/FR-001` reads no
matrix whole when the graph reads a device copy, and that is now wrong for exactly these experts, so
it is amended. `34/FR-002`'s record of what a handle has read still bounds the reads, and an expert
that later becomes resident simply stops being read.

The budget is the card's own number. `21/FR-008`'s `memory` reports what the device has free, and
that is what the tier may take; `config.json`'s `vram`, when the file sets it, caps it below that, so
a person can leave the card room for something else or hold a run to a number they chose. With no
card the budget is zero, the plan is empty and the model computes as it does today.

```teal
local record compute
  -- what one model holds on the card: the tensors it copies, the bytes it
  -- charged for them, the experts it keeps there per block, and the rest
  record Placement
    budget: integer               -- bytes the tier may use
    reserve: integer              -- what the graphs' own buffers need
    resident: {string: boolean}   -- the tensors copied to the card
    resident_bytes: integer       -- their bytes, each rounded to 128
    slots: {integer}              -- resident experts per block, in block order
    slot_bytes: integer           -- what those experts hold, each matrix
                                  -- rounded to 128
    unplaced_bytes: integer       -- tensor bytes that stay in the mapping
  end

  record Model
    -- 33's fields, unchanged
    placement: Placement
  end

  -- the plan for a budget and a reservation, over a handle's tensors
  place: function(weights.Handle, integer, integer): Placement

  model: function(
    backend: Backend,
    handle: weights.Handle,
    positions: integer | nil,
    cache_type: CacheType | nil,
    sequences: integer | nil,
    ubatch: integer | nil,
    vram: integer | nil
  ): Model | nil, string | nil
end

local record serve
  -- 18's, 24's, 28's and 33's fields, unchanged
  record Endpoint
    vram: integer | nil
  end

  build: function(
    Endpoint, spawn?: process.Spawn, write_out?: function(string)
  ): tltp.Server | nil, string | nil
end
```

This snippet supersedes the `Endpoint` record `18` declares and the fields `24`, `28` and `33` add to
it, and the `Model` and `model` declarations of `33`. `place` takes no backend because it allocates
nothing: it answers which tensors a budget would hold. The 128 is the alignment the pinned CUDA
backend's buffer type reports, so the bytes a plan charges are never fewer than the bytes the card
will want.

`serve.build`'s third argument is the writer the tier's line goes through, and nil means write
nothing: `bench` calls `build` with no writer (`29/FR-011`), so no run of `bench` grows a line.
`serve.start` passes its stderr writer so a person starting a server reads what the card took, and
`src/worker.tl` calls `build` with no writer, so a worker's line never reaches the parent's pipe and
a pool, whose models are the workers' own, writes no line at all.

## User stories

- US-001 (P1) As a person with a model larger than my card, I want the weights that fit to sit there and the rest to be read, so that the model loads and runs at all.
- US-002 (P1) As the engine author, I want the dense weights placed before any expert, so that no token streams the attention weights it reads every pass.
- US-003 (P1) As the engine author, I want the leftover bytes given to whole experts, layer by layer, so that a partly resident model answers what a fully resident one answers.
- US-004 (P1) As the person running the server, I want to cap what the tier may take, so that I can leave the card room for anything else.
- US-005 (P2) As the person running the server, I want to read what the card ended up holding, so that I know how much of a token's time is the disk.

## Functional requirements

- FR-001 (US-002) WHEN `place` is given a handle, a budget and a reservation, THE SYSTEM SHALL take
  the handle's tensors in the order of the eight tiers the Problem & context lists and mark each
  resident whose bytes, rounded up to 128, fit in what the budget holds after the reservation,
  skipping a tensor that does not fit and trying the next one, so `token_embd.weight` is taken first,
  `output.weight` last among the fixed tensors, the full-attention blocks' dense weights before the
  linear blocks', and the routers before the shared experts; a budget not above the reservation SHALL
  mark no tensor and place no expert.
- FR-002 (US-003) WHEN the fixed tiers of FR-001 have been taken and the budget holds bytes beyond
  them and the reservation, THE SYSTEM SHALL place whole experts in rounds, a round taking the blocks
  in block order and each block taking the lowest-numbered expert it has not placed whose three
  matrices, each rounded up to 128, fit what remains, SHALL stop at the end of the first round that
  placed no expert, and SHALL report in `slots` the experts each block holds and in `slot_bytes` the
  bytes charged for them, an expert's three matrices being placed together or not at all.
- FR-003 (US-003) WHEN `place` returns, THE SYSTEM SHALL report in `resident_bytes` the bytes it
  charged for the tensors it marked and in `unplaced_bytes` the handle's tensor bytes it did not
  mark, never below 0 when the charges for the tensors it marked are more than those bytes hold, and
  `resident_bytes` plus `slot_bytes` SHALL be at most the budget less the reservation and both
  SHALL be 0 when the reservation is at least the budget.
- FR-004 (US-004) WHEN `compute.model` is given a backend whose library found a device and which was
  not asked to stay on the CPU, THE SYSTEM SHALL take the tier's budget as the free bytes `memory`
  reports at that moment, or as the `vram` it is given when one is given and is the smaller, SHALL
  set the model's reservation to the bytes the tensors of the graph buffers `32/FR-008` gives a
  block or the pass hold at the geometry's positions, measured for a pass of one row per sequence,
  the rows `33/FR-003` sizes the model's arena at, one sum for each of those buffers, each sum
  rounded up to 16 rather than charged with the padded allocation each tensor lands in, SHALL copy
  the tensors its `place` marks and each resident
  expert's three matrices into device memory in one buffer of the model's own sized to
  `resident_bytes` plus `slot_bytes`, the experts it copies into a block's slots being those that
  block's profile seeded, or the lowest-numbered ones when `compute.model` is given no profile,
  while `place` reports the same `slots` counts and the same `slot_bytes` it reports today, and
  SHALL set the model's `placement` to that plan; WHEN the
  library found no device, or the backend was asked to stay on the CPU, THE SYSTEM SHALL set a
  `placement` whose budget, `reserve`, `resident_bytes`, `slots` and `slot_bytes` are 0, whose
  `resident` is empty and whose `unplaced_bytes` is the handle's tensor bytes, and compute as
  `21/FR-005` and `21/FR-006` do.
- FR-005 (US-001, US-003) WHEN a pass runs over a model whose `placement` marks some of its tensors,
  THE SYSTEM SHALL return the score rows, and choose the experts, that the pass of a model holding
  every tensor on the card returns and chooses, each number equal, so the placement changes where a
  weight is read from and nothing the pass answers.
- FR-006 (US-003) WHEN a pass chooses an expert whose three matrices its model's `placement` does not
  mark, THE SYSTEM SHALL read those matrices whole from the mapping as `34/FR-001` reads them and
  count the reads and their bytes in `measure` as `34/FR-003` counts them, and SHALL read no matrix
  whole for an expert its `placement` marks. The matrices it reads whole are those `34/FR-001` still
  reads whole after `42`'s page-cache exception, so a resident matrix its `placement` does not mark
  is not read whole.
- FR-007 (US-004) WHEN `serve.build` builds a server whose model runs in its own process, THE SYSTEM
  SHALL build that model with the endpoint's `vram`.
- FR-008 (US-004) WHEN a worker builds its model, THE SYSTEM SHALL build it with the `vram` the
  endpoint json it was given holds.
  Not testable: the model lives in a worker process and nothing a worker publishes carries its
  plan; a reader confirms the field in the decoded json and the argument in `src/worker.tl`'s
  `compute.model` call.
- FR-009 (US-004) IF `compute.model` is given a `vram` below one, THEN THE SYSTEM SHALL return nil
  and one line naming `model` and that number.
- FR-010 (US-005) WHEN `serve.build` is given a writer and builds a server whose model runs in that
  process, THE SYSTEM SHALL write through it the one line `device: NAME, B bytes budget, R bytes
  held, S expert slots in T bytes, U bytes from the mapping` and nothing else, where NAME is the
  `device` `21/FR-009` gives the model's backend, or `none` when it is nil, B is the placement's
  `budget`, R its `resident_bytes`, S the sum of its `slots`, T its `slot_bytes` and U its
  `unplaced_bytes`; WHEN it builds a pool, THE SYSTEM SHALL write nothing through that writer, a
  pool's placement being the workers' own and published by none of them; and the line SHALL be
  written once the model is in place and before the server's socket is bound, so a start that cannot
  bind has still reported what the card took.
- FR-011 (US-001) IF the card refuses room for the tensors a plan marks, THEN THE SYSTEM SHALL copy
  nothing to the card, set the model's `placement`'s `resident` empty, its `resident_bytes`, `slots`
  and `slot_bytes` to 0 and its `unplaced_bytes` to the handle's tensor bytes, and compute as
  `21/FR-007` does; IF it refuses room for the bytes a plan's expert slots hold, THEN THE SYSTEM
  SHALL place no expert slot, keep every tensor it already copied, and set the model's
  `placement`'s `slots` to one 0 per block and its `slot_bytes` to 0, so the placement reports what
  the card holds and `FR-010`'s line does too.
  Not testable: forcing either allocation to fail needs the free bytes to shrink between the plan and
  the copy, at a size that depends on the card, as `21/FR-007` states for the whole-model copy; both
  branches are one allocation check each, confirmed by inspection.
- FR-012 (US-001) WHEN a model's `close` is called, THE SYSTEM SHALL free the device buffer its
  placement's tensors and its expert slots were copied into before the call returns, and a second
  `close` SHALL free nothing further; IF `compute.model` returns nil after copying tensors or an
  expert slot to the card, THEN THE SYSTEM SHALL free that buffer before it returns, so a model that
  cannot be built leaves no bytes on the card.
- FR-013 (US-005) WHEN `serve.start` builds its server, THE SYSTEM SHALL pass its stderr writer to
  `serve.build` as the writer FR-010 writes through.

## Non-goals

- No rerouting to a resident expert, no skipping the experts that are not there, and no lower
  precision for an unplaced weight: the router's choice and every matrix's type are the run's, and a
  byte the card does not hold is read from the mapping.
- No cache on the card and no share of the budget between workers: keys and values stay in host
  memory as `21` and `24` have them, and each worker's tier takes what the card reports free when it
  builds, so a pool of several workers divides the card by arrival.
- No new line in `bench`'s report and no change to the lines `29` writes: the plan is read from the
  model and `bench` prints the lines it prints today, though the numbers in them move with the tier,
  as the amendment to `29/FR-011` says.

## Tests

**The plan fixture** is a GGUF built through the in-process builder the weights cases use, holding
`token_embd.weight`, `output_norm.weight` and `output.weight` at the root; under `blk.0.` the seven
attention names, `post_attention_norm.weight`, a router, a shared-expert router and the three shared
experts; under `blk.1.` the ten linear names, `post_attention_norm.weight`, a router, a
shared-expert router, the three shared experts and `ffn_up_exps.weight`, `ffn_gate_exps.weight` and
`ffn_down_exps.weight` of two experts; and under `blk.2.` the seven attention names,
`post_attention_norm.weight`, a router and those three expert tensors of two experts whose every
expert holds twice the bytes `blk.1.`'s hold. Every tensor is f32, the case gives every one of the
fixture's 41 fixed tensors the same element count, so `u` is what each of them is charged, and it
chooses the experts' element counts, so a plan's figures are numbers the case computed. `place`
needs no backend, so T-001 runs on any machine.

**The large fixture** is `24/T-006`'s fixture with `embedding_length` 1024 in place of its 32, the
row width of its tensors 1024 in place of 32, `attn_q.weight` over 2048 rows, `attn_k.weight`,
`attn_v.weight`, `attn_output.weight`, `ffn_up_shexp.weight`, `ffn_gate_shexp.weight` and
`ffn_down_shexp.weight` over 1024 rows, and its expert tensors' dims `{1024, 32, 32}`, keeping that
case's `rope.dimension_count` 2, `attention.head_count` 1, `context_length` 4, its 32 experts and
its three tokens, so its tensors hold about 44 MiB, which is what T-002 measures the card by.

- T-001 (FR-001, FR-002, FR-003): `the budget fills stratum by stratum`, proves on the plan fixture,
  for some reservation `R` the case picks, with `u` the bytes one fixed tensor is charged, `f` the
  sum of those bytes, being 41 times `u`, `t` the bytes `token_embd.weight` is charged, being `u`, `e`
  the bytes `blk.1.`'s one expert is charged and `g` the bytes `blk.2.`'s one expert is charged, being
  twice `e`, that `place(handle, t + R, R)` marks `token_embd.weight` and no other tensor; that
  `place(handle, t + 3 * u + R, R)` marks
  `token_embd.weight` and `blk.0.`'s `attn_norm.weight`, `attn_q.weight` and `attn_k.weight` and
  nothing else, so no later name of the block is taken; that `place(handle, t + 16 * u + R, R)` marks
  `token_embd.weight`, `blk.0.`'s eight dense tensors and `blk.2.`'s eight dense tensors and nothing
  under `blk.1.`, so the attention blocks' dense weights come before the linear block's; that
  `place(handle, t + 32 * u + R, R)` marks every tensor of tiers 2, 3 and 4 and no shared-expert
  tensor, so the routers come before the shared experts; that `place(handle, f - u + R, R)` marks
  every fixed tensor except `output.weight`, and that `place(handle, f + R, R)` marks all 41,
  reports `slots` `{0, 0, 0}` and `slot_bytes` 0 and takes no expert; that `place(handle, f + e + R,
  R)` gives `{0, 1, 0}`, the bytes buying `blk.1.`'s first expert and leaving none for `blk.2.`'s,
  which holds twice as much; that `place(handle, f + 3 * e + R, R)` gives `{0, 1, 1}`, where a plan
  charging every slot at `g` gives `{0, 1, 0}`; that `place(handle, f + 5 * e + R, R)` gives
  `{0, 2, 1}`, one round placing an expert of each block and the next placing `blk.1.`'s second and
  stopping because `blk.2.`'s does not fit; that `place(handle, R, R)` marks no tensor and gives
  `slots` `{0, 0, 0}`; and that each of those calls reports `resident_bytes` plus `slot_bytes` at
  most the budget less R and `unplaced_bytes` the handle's tensor bytes less those two.
- T-002 (FR-004, FR-009, FR-012): `a model takes the budget it is given`, proves on `15/T-001`'s
  fixture that `compute.model(backend, handle, 4, "f32", 1, 1, 0)` returns nil and one line holding
  `model` and `0`, and that on `compute.load(nil, false)` a model of that fixture has a `placement`
  whose budget, `resident_bytes`, `slot_bytes` and `slots` are 0 or empty, whose `unplaced_bytes` is
  the handle's tensor bytes, and whose `forward(0, 0)` returns `{1.4142136, 0, 1.4142136}` to within
  1e-5; and, on a machine where `compute.load()`'s `memory` gives free bytes, that a model of the
  large fixture built with no `vram` has a budget within 16 MiB of the free bytes `memory` reported
  before the build, that two models of `15/T-001`'s fixture built at 2 and at 4 positions have
  reservations of which the second is the larger, and that on a model of the large fixture the free
  bytes `memory` reports have risen by at least nine tenths of its placement's `resident_bytes` plus
  `slot_bytes` once `close` has returned and stay so after a second `close`, and that on `15/T-002`'s
  fixture with `blk.0.ssm_out.weight` removed `compute.model` returns nil and one line holding
  `blk.0.ssm_out.weight` and the free bytes `memory` reports afterwards are within 16 MiB of what
  they were before the call.
- T-003 (FR-005, FR-006): `one expert on the card and one off gives the same reply`, proves, on a
  machine whose library found a device, on `15/T-001`'s fixture whose `ffn_gate_inp.weight` holds
  `{1, 0}` and `{0, 1}` and whose three expert tensors hold 1 for the first expert and 2 for the
  second, so token 0 routes to expert 0 and token 1 to expert 1: that a model of it built with no
  `vram` routes both, its `routing_sink` holding `{0}` for the first pass and `{1}` for the second,
  that a model of it built with a `vram` of the first model's `resident_bytes` plus its `reserve`
  plus half its `slot_bytes`, being every fixed tensor and one expert, has `slots` `{1}`, that the
  two models return the same two score rows for those ids with every number equal, and that
  `measure.blob_reads` is 0 over the first model's two passes and above 0 over the capped model's.
- T-004 (FR-010): `build reports what the card holds`, proves that `serve.build` on `18/T-001`'s
  fixture given nil for its spawn argument and a collector as its third writes one line matching
  `^device: (.+), (%d+) bytes budget, (%d+) bytes held, (%d+) expert slots in (%d+) bytes, (%d+) bytes from the mapping$`,
  that on a machine where `memory` gives nil and nil that line holds `none`, a budget of 0, 0 bytes
  held, 0 slots and the handle's tensor bytes as the bytes from the mapping, that on a machine where
  it gives free bytes the line's name is neither `none` nor empty, its budget is above 0 and its
  bytes-held figure is at most the budget, that a `serve.build` on the same fixture with `workers` 2,
  a context directory and no spawn argument, built the way `28/T-001` builds its pool, returns a
  server and nil and writes no line through the collector it is given, so a pool's build reports no
  placement of its own, that `serve.stop` on that server returns, and that a `serve.build` on the
  same endpoint with no third argument writes nothing to stderr.
- T-005 (FR-007): `the cap reaches the model`, proves, on a machine where `memory` gives free bytes,
  that `serve.build` on `18/T-001`'s fixture whose endpoint names `vram` 1048576 gives a server whose
  `serve.model_of(server).generation.placement` has a budget of the smaller of 1048576 and the free
  bytes `memory` reported before the build, and that the same build on an endpoint naming no `vram`
  gives a budget within 16 MiB of those free bytes; on a machine where `memory` gives nil and nil it
  skips with the line `no CUDA device`.
- T-006 (FR-013): `start hands its writer to build`, proves on `18/T-001`'s fixture that `serve.start`
  with `host` `192.0.2.1`, a collector and no spawn argument writes the line of T-004's pattern as
  its first line and the bind failure as its second, and returns 1, so `tests/serve.test.tl`'s case
  `a round steps the transport and the turns`, which counts one written line, counts two.

## Open questions

- None.

## Amendments

- The reservation is measured for a pass of one row per sequence rather than the widest pass
  `ubatch` allows. Measured at the widest pass, a file whose window is 262,144 positions reserves
  what one row's copy of that whole window costs it, on every block that reads attention: on the
  Qwen3.6 MoE file on this machine that is 566,900,457,984 bytes, more than the card holds, so
  `FR-001` marked no tensor and the tier never placed anything. The rows a reservation is measured
  over are now the rows the arena is sized at, so it grows only when a wide pass grows the arena,
  as `33/FR-003` states.
