# 39: The miss policy: compute on the CPU or DMA to the card

**Status**: done   **Depends on**: 21, 29, 34, 35, 36, 38

## Problem & context

A partially resident model currently gives every non-resident expert to the device graph. The
`host_tensor` path copies its matrices over the link, even when the CPU could finish that expert
sooner and leave the link to another one. On a laptop with a narrow link that leaves a useful part
of the machine idle; on a fast link with a weak CPU it leaves the link idle instead.

A miss is one distinct expert a block's pass chooses whose slot copy has not landed. The policy
splits those misses between arithmetic on the CPU and DMA to the card. It does not split the three
matrices of one expert, reroute anything, or count a resident expert as a miss.

The split balances the two measured rates. `L` is the host-to-device link rate and `R` is the CPU
expert rate, both in decimal GB/s of expert matrix bytes. The DMA share is:

```text
share = L / (L + k R)
```

`k` accounts for the CPU pool and the link competing for RAM: 1.85 at `R >= 30`, 0.8 at `R <= 12`,
and linearly interpolated between those points. The raw share is rounded to the nearest 0.05 and
clamped to 0.10 through 0.95. Strata's
[`pcie_share_rule.hpp`](https://github.com/Niko1221/Strata/blob/main/engine/include/strata/core/pcie_share_rule.hpp)
states this rule. Its
[paper](https://raw.githubusercontent.com/Niko1221/Strata/master/docs/paper/Strata-Paper.pdf)
reports fixed shares of 55% for i-quants and 20% for Q2_0, with the i-quant CPU arithmetic-bound
at about 25 GB/s on six cores and Q2_0 already consuming its RAM bandwidth. Those examples explain
why the rate is measured here rather than replaced by either fixed share.

The first two passes containing misses are probes: the first computes every miss on the CPU, the
second sends every miss by DMA. After each of the first 16 such passes the cumulative measurements
refine the share and its published rates. Later passes keep that share and those published rates,
so ordinary traffic does not make the policy flap. A pass with no miss is not a calibration window.

```teal
-- src/miss.tl
local record miss
  -- the rounded, clamped DMA share for two positive rates in GB/s
  share: function(link: number, pool: number): number
  -- the distinct expert ids to send by DMA, in ascending order
  take: function(experts: {integer}, share: number): {integer}
end

local record compute
  record MissPolicy
    share: number          -- the share the next miss window uses
    pool: number           -- measured CPU expert GB/s
    link: number           -- measured host-to-device GB/s
    windows: integer       -- completed calibration windows, at most 16
    cpu: integer           -- misses assigned to the CPU over the model's life
    dma: integer           -- misses assigned to DMA over the model's life
    cpu_bytes: integer     -- calibration bytes whose CPU arithmetic was timed
    cpu_seconds: number    -- their CPU arithmetic seconds
    dma_bytes: integer     -- calibration bytes whose upload was timed
    dma_seconds: number    -- their host-to-device seconds
  end

  record Model
    -- 38's fields, unchanged
    miss: MissPolicy
  end
end
```

The byte and second fields are calibration totals, not per-turn measurements, so `bench` zeroing
`measure` does not reset the policy. `cpu` and `dma` count distinct misses per block, not rows that
share one expert. A fixed weight and a shared expert are never misses, so no policy applies to
them.

## User stories

- US-001 (P1) As a person running experts outside the card, I want each miss computed where the measured machine handles it best, so that the CPU and the link both contribute to a pass.
- US-002 (P1) As the engine author, I want the share learned from real work and then held steady, so that the policy reflects this model and machine without changing every pass.
- US-003 (P1) As the engine author, I want either path to produce the same next token, so that moving an expert changes only its cost.
- US-004 (P2) As the person benchmarking a model, I want the chosen share and both measured rates in the report, so that I can explain why the pass was split that way.

## Functional requirements

- FR-001 (US-001) WHEN `miss.share` is given positive `link` and `pool` rates in GB/s, THE SYSTEM
  SHALL return `link / (link + k * pool)`, with `k` 1.85 at a pool rate of 30 or more, 0.8 at 12 or
  less and `0.8 + (pool - 12) / 18 * 1.05` between them, rounded to the nearest 0.05 and clamped to
  at least 0.10 and at most 0.95.
- FR-002 (US-002) WHEN a model with a usable device completes its first pass containing a miss,
  THE SYSTEM SHALL have computed every miss of that pass on the CPU; WHEN it completes its second
  such pass, THE SYSTEM SHALL have sent every miss of that pass by DMA.
- FR-003 (US-002) WHEN each of the first 16 passes containing a miss completes, THE SYSTEM SHALL
  add that pass's measured bytes and seconds to the calibration totals, set `MissPolicy.pool` to
  `cpu_bytes / cpu_seconds / 1000000000` from the first window onward and, from the second window
  onward, also set `MissPolicy.link` to `dma_bytes / dma_seconds / 1000000000` and
  `MissPolicy.share` to `miss.share(MissPolicy.link, MissPolicy.pool)`; after 16 windows THE SYSTEM
  SHALL keep those three fields and the calibration totals unchanged and SHALL increment
  `MissPolicy.windows` no further.
- FR-004 (US-001) WHEN a pass has chosen a block's misses and the model has completed two passes
  containing a miss, THE SYSTEM SHALL send `floor(M * share + 0.5)` of its M distinct misses by
  DMA, choosing the lowest expert ids and computing every other miss on the CPU, with `miss.take`
  returning that DMA set in ascending order; the first pass containing a miss SHALL compute every
  miss on the CPU and the second SHALL send every miss by DMA, whatever `MissPolicy.share` holds;
  WHERE the model has no usable device, THE SYSTEM SHALL compute every miss on the CPU, keep
  `MissPolicy.share`, `MissPolicy.pool`, `MissPolicy.link` and `MissPolicy.windows` at 0, and count
  neither `MissPolicy.cpu` nor `MissPolicy.dma`.
- FR-005 (US-001) WHEN an expert is assigned to the CPU, THE SYSTEM SHALL compute its three matrix
  products from the mapping; WHEN it is assigned to DMA, THE SYSTEM SHALL copy its three matrices
  into tensors of the graph's own arena in the device buffer its existing pool owns, held for the
  length of that graph with no separate model-owned buffer, and compute its three matrix products
  there, with `34/FR-001`'s whole-matrix demand read made before either path that needs it, which is
  the read `34/FR-001` asks for after `42`'s page-cache exception: a resident matrix needs no demand
  read before it is computed.
- FR-006 (US-001) WHEN one block has misses on both paths, THE SYSTEM SHALL issue the DMA copies
  and device arithmetic without waiting for the CPU subset, run the CPU subset while that device
  work proceeds, and combine the block's result only after both subsets have completed.
  Not testable: returned numbers cannot establish overlap; a reader confirms that device work is
  issued before the CPU graph runs and that the result is read only after both finish.
- FR-007 (US-003) WHEN the same expert is computed on the CPU path or the DMA path with the same
  model inputs, THE SYSTEM SHALL return the same greedy next token on either path.
- FR-008 (US-002, US-004) WHEN a model with a usable device assigns a miss to a path, THE SYSTEM
  SHALL increment `MissPolicy.cpu` or `MissPolicy.dma` once for that assignment and, during a
  calibration window, SHALL time only its
  CPU expert arithmetic or its host-to-device copies, excluding demand reads and device arithmetic,
  charge the three matrices once per distinct missed expert to the corresponding byte total, and
  publish the fields of `MissPolicy` as the policy stands after the pass.
- FR-009 (US-004) WHEN `bench` writes a length's `experts` line, THE SYSTEM SHALL append
  `; S% over PCIe, pool R GB/s, link L GB/s` after the ahead text `38/FR-011` appends, where S is
  `MissPolicy.share` times 100, R is `MissPolicy.pool` and L is `MissPolicy.link`, each printed to
  one decimal place, with all three figures 0.0 before measurement and on a CPU-only model.

## Non-goals

- No fixed 0.55 or 0.20 preset and no assertion that the calculated share is the fastest one:
  `29`'s before-and-after rates are the evidence for that, while these cases pin the rule and its
  arithmetic.
- No config key or command-line flag: the measured policy is always used for misses on a device
  model, and CPU-only and fully resident models need no switch.
- No change to routing, resident slots, cache admission, read-ahead or demand-read accounting, and
  no special prompt-only offload path: this spec concerns only non-resident routed experts.

## Tests

**The miss fixture** is `36`'s cache fixture built with a `vram` of a first model's `resident_bytes`
plus its `reserve`, so every fixed tensor that fits is resident but there are no expert slots. Its
zero `attn_output.weight` makes its feed-forward result independent of the attention history, so a
pass of token 0 at position 0 and another at position 1 route the same expert with the same hidden
row. T-002 and T-003 use that fixture; T-005 uses the same base fixture with one slot. Every
device case gates on `on_device` as `35`'s cases gate.

- T-001 (FR-001, FR-004): `the measured rates choose a bounded share and a stable set`, proves
  without a device that `miss.share(56.1, 61.2)` is 0.35, `miss.share(56.3, 10.3)` is 0.85,
  `miss.share(56.2, 8.0)` is 0.90, `miss.share(56.0, 22.3)` is 0.65,
  `miss.share(56.0, 13.0)` is 0.85, `miss.share(56.0, 29.0)` is 0.50,
  `miss.share(56.0, 0.5)` is 0.95 and `miss.share(1.0, 200.0)` is 0.10, each to within 1e-9,
  so both k branches, the interpolation, rounding and both clamps hold; and that
  `miss.take({2, 0, 1}, 0.55)` returns
  `{0, 1}`, `miss.take({2, 0, 1}, 0.20)` returns `{0}`, `miss.take({2}, 0.10)` returns `{}` and
  `miss.take({2}, 0.95)` returns `{2}`, so assignment is by ascending id to the nearest whole
  expert rather than a random fraction.
- T-002 (FR-002, FR-003, FR-004, FR-008): `two probes teach the policy for sixteen windows`, proves
  on the miss fixture that its first pass of token 0 has `MissPolicy.cpu` 1,
  `MissPolicy.dma` 0, `MissPolicy.windows` 1, a positive pool rate, a zero link rate and a zero
  share, and `MissPolicy.cpu_seconds` below that pass's `measure.compute` delta; that its second
  pass has totals `MissPolicy.cpu` 1 and `MissPolicy.dma` 1, `MissPolicy.windows` 2, both rates
  positive and `MissPolicy.share` equal to `miss.share(MissPolicy.link, MissPolicy.pool)`, leaves
  `MissPolicy.cpu_bytes` and `MissPolicy.cpu_seconds` exactly where the first pass left them and
  has `MissPolicy.dma_seconds` below that pass's `measure.compute` delta; that the third pass's
  `MissPolicy.dma` delta is the length of `miss.take` over its routed expert ids with the
  `MissPolicy.share` read before that pass and its `MissPolicy.cpu` delta is the routed count less
  that length, so a pass assigns by the published share and not by a fixed rule; that after every
  pass from the second through the 16th the window count is the pass count, `MissPolicy.pool`
  equals
  `MissPolicy.cpu_bytes / MissPolicy.cpu_seconds / 1000000000`, `MissPolicy.link` equals
  `MissPolicy.dma_bytes / MissPolicy.dma_seconds / 1000000000`, `MissPolicy.share` equals the pure
  function of those rates and the sum of calibration bytes has increased; and that the 17th pass
  leaves `MissPolicy.windows`, `MissPolicy.share`, both rates and all four calibration totals
  exactly as the 16th left them while `MissPolicy.cpu` plus `MissPolicy.dma` is 17, so later work
  does not refine the policy.
- T-003 (FR-005, FR-007): `a missed expert gives the same token on either path`, proves on the miss
  fixture that the first pass, whose miss is counted on the CPU, and the second pass, whose miss is
  counted by DMA, choose the same greedy token from their score rows, with `MissPolicy.dma` 0 after
  the first and 1 after the second; and that a model of the same fixture on
  `compute.load(nil, false)` also
  chooses that token with `MissPolicy.share`, `MissPolicy.pool`, `MissPolicy.link`,
  `MissPolicy.windows`, `MissPolicy.cpu` and `MissPolicy.dma` all 0, so the CPU-only fallback is
  unchanged.
- T-004 (FR-009): `bench reports the split and its evidence`, proves that `bench m --config F
  --prefill-tokens 8 --decode-tokens 8` on a config naming `29`'s fixture and a `vram` of 1048576
  writes an `experts 8:` line ending `; S% over PCIe, pool R GB/s, link L GB/s`, with all three
  numbers at zero on a machine without a device, and on a device with a `vram` of a first model's
  `resident_bytes` plus its `reserve`, S between 10.0 and 95.0 and both rates printed as numbers at
  or above 0.0, so the report carries the chosen policy and the rates measured from real misses
  rather than a quantization preset. The fixture's expert matrices are tens of bytes, so the two
  rates it measures print as `0.0`: what the case proves of them is that the line carries a measured
  figure and the chosen share, not that the link is fast.
- T-005 (FR-002, FR-003): `a held expert is not a miss and a hit is not a window`, proves on
  `36`'s cache fixture built with that case's one-slot cap and a profile whose lines route expert
  0, on a device, that a pass of token 0 at position 0 leaves `MissPolicy.windows`,
  `MissPolicy.cpu`, `MissPolicy.dma` and all four calibration totals at 0, and that the following
  pass of token 1 at position 1 has `MissPolicy.windows` 1, `MissPolicy.cpu` 1 and
  `MissPolicy.dma` 0, so a resident expert is not a miss and a pass with no miss is not a window.

No case asserts that the chosen share gives the highest token rate; `29`'s measured token rates
are the comparison for that claim.

## Open questions

- None.

## Amendments

- None.
