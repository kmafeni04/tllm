# 36: The adaptive expert cache

**Status**: done   **Depends on**: 04, 15, 29, 34, 35

## Problem & context

`35` gives a model a fixed number of whole experts per block and fills them with the
lowest-numbered experts it finds, then never changes them. On a model with a few thousand slots
and a real conversation most of those slots hold experts the conversation never routes: strata
measures a static lowest-id profile serving about 50% of the routed experts from 4,500 slots, and a
profile that follows the workload about 72%. The engine already knows the workload, because
`29 --dump-routing FILE` writes one JSON line per token per layer holding that token's expert ids.
So the slots can start from a dump, and then follow the conversation that is running.

Three things make that a cache rather than a guess. The seed: the slots start from a profile of
what a workload routed, so the first tokens are not all misses. The adaptation: every eight rounds
the model ranks the experts it holds beside the experts those rounds routed and keeps the top of
each block, so a conversation that moves teaches the slots where it moved. The landing rule: an
expert chosen for admission is copied after the pass that chose it and is marked held only once its
matrices are in place, so a slot is never read while its bytes are moving; strata measures
landed-only admission at 91.7 to 94.4 tok/s, against a pass that waits on the copy.

A block holds an expert when the bytes in its slot are that expert's three matrices, and a copy has
landed once the last of those three has been copied. A pass is a round. The policy lives in the
model, so a bench run and a served conversation use the same one and a worker adapts its own model.
The only new input is a file named by `config.json`, and the only new output is the hit rate
`bench` prints, so a profile is judged by a number instead of a feeling. Nothing here changes the
expert the router chooses, the weight that expert multiplies, or the answer a pass returns; only
which copy of the chosen expert's bytes the graph reads.

```teal
local record compute
  record Measure
    -- 34's fields, unchanged
    expert_hits: integer   -- routes a held expert served since measure was zeroed
  end

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

local record serve
  record Endpoint
    -- 35's fields, unchanged
    profile: string | nil
  end
end

local record bench
  record Settings
    -- 29's fields, unchanged
    profile: string | nil
  end
end
```

This snippet supersedes the `Measure` record as `34` and `35` leave it, the `model` declaration of
`35`, the `Endpoint` record `35` declares and the `Settings` record `29` declares. `profile` is a
path to a file `29/FR-008` writes: one JSON object per line, holding `layer`, `token` and
`experts`.

## User stories

- US-001 (P1) As a person running a model larger than my card, I want the slots seeded from a profile
  of a real workload, so that the first tokens are not all misses.
- US-002 (P1) As a person running a conversation, I want the slots to follow the experts this
  conversation routes, so that the hit rate rises as it runs.
- US-003 (P1) As the engine author, I want an expert used from the card only after its copy has
  landed, so that no answer is computed from bytes still being copied.
- US-004 (P2) As the person benchmarking a change, I want the hit rate in the report, so that I can
  see what the cache bought.

## Functional requirements

- FR-001 (US-001) WHEN `compute.model` places at least one expert slot and is given a profile, THE
  SYSTEM SHALL fill each block's slots with the `slots` experts that profile routed most often, a tie
  going to the lower id, fill any slot left with the lowest-numbered expert not already held, and
  count no record naming a layer the model lacks or an expert a block lacks.
- FR-002 (US-001) WHEN `compute.model` is given a profile, THE SYSTEM SHALL read it whether or not
  the model places an expert slot, before copying anything to the card, and IF it cannot be read, or
  holds a non-empty line that is not a JSON object whose `layer` is an integer and whose `experts` is
  a list of integers, THEN THE SYSTEM SHALL return nil and one line naming `model`, `profile` and the
  file, after every fault `15/FR-009`, `15/FR-015` and `35/FR-009` name.
- FR-003 (US-002) WHEN a pass begins after eight completed rounds since the model's build or its
  last adaptation, the pass that adapted counting as the first of the next eight, THE SYSTEM SHALL
  choose each block's held experts from the union of the experts it holds and those the eight rounds
  routed, ranked by routes in those rounds descending, then by the round of their last route
  descending with an expert those rounds did not route ranking below every expert they did, then by
  id ascending, taking the first `slots`, treat a chosen expert it does not hold as admitted, and
  drop a held expert the choice leaves out.
- FR-004 (US-003) WHEN a pass begins, THE SYSTEM SHALL serve every route of that pass from the
  experts each block held when that pass began, so an expert the same pass admits serves none of its
  routes.
- FR-005 (US-003) WHEN a pass that admitted experts finishes its arithmetic, THE SYSTEM SHALL copy
  each admitted expert's three matrices from the mapping into its slot, and SHALL mark the admitted
  expert held when its copies have completed, and never before the next pass began, so a slot holds
  one expert at a time.
- FR-006 (US-002) WHEN a route of a pass selects an expert that pass's block held when the pass
  began, THE SYSTEM SHALL add one to `measure.expert_hits`, and SHALL add none for a route to any
  other expert.
- FR-007 (US-001) WHEN `serve.build` builds a model in its own process, THE SYSTEM SHALL give
  `compute.model` the endpoint's `profile`.
- FR-008 (US-004) WHEN `bench` builds its server, THE SYSTEM SHALL set the endpoint's `profile` to
  the `profile` resolved for the model `bench` resolved as `53/FR-003` says, the `--profile` flag of
  `54/FR-003` first, nil when none of its entry, the file and the flag names one.
- FR-009 (US-001) WHEN a worker decodes its endpoint json, THE SYSTEM SHALL carry the `profile` the
  json holds into the endpoint it builds, so `serve.build` gives it to `compute.model` as FR-007
  says.
  Not testable: the model lives in a worker process and nothing a worker publishes reports its slots
  or its hits; a reader confirms `profile` in `src/worker.tl`'s `EndpointJson` record and its
  `endpoint_of` table.
- FR-010 (US-004) WHEN `bench` writes a length's `experts` line, THE SYSTEM SHALL end it with
  `; H% hit`, H being the turn's `measure.expert_hits` over its `measure.routed` times 100 to one
  decimal, printed `0.0` when `routed` is 0, followed by the ahead text `38/FR-011` appends and
  then by the miss-policy text `39/FR-009` appends, and SHALL zero `measure.expert_hits` beside
  the other `measure` fields before a turn.
- FR-011 (US-003) THE SYSTEM SHALL keep a slot out of every graph that a pass builds before the
  pass that follows its copy, so no graph reads bytes still moving.
  Not testable: a copy in flight cannot be produced on demand, because a case would have to hold the
  device mid-copy; T-002 proves the admitting pass serves none of the admitted expert and the next
  pass serves its landed copy, and a reader confirms that a slot is written only after one pass's
  arithmetic has returned and read only by graphs built after the next pass began.
- FR-012 (US-001) THE SYSTEM SHALL pass the start function and the `bench.Settings` a bench run
  builds the `profile` resolved for the chosen model as `53/FR-003` says, nil when neither its
  entry nor the file's top level names one; IF a `profile` is set and is not text, at either
  level, THEN THE SYSTEM SHALL write one line naming `profile` and, for the entries holding one
  that is not text, every such entry in ascending byte order, and return exit code 1 without
  calling a start function.

## Non-goals

- No replay tool and no offline policy runner: the dump is the format the cache reads, and replaying
  a policy against it outside a run waits for a spec of its own.
- No rerouting, no skipping a held expert for a cheaper one, and no change to the expert the router
  chooses or the answer a pass returns: the cache changes where a weight is read from.
- No new card buffer and no reallocation: the slots are the ones `35` sized at build, and adaptation
  overwrites them in place, so a model's card footprint is what it was.

## Tests

**The cache fixture** is `15/T-001`'s fixture with its `qwen35moe.context_length` raised to 32, its
`ffn_gate_inp.weight` `{2, 3}` holding `{1, 0}`, `{0, 1}` and `{-1, -1}`, its three expert tensors
`{2, 2, 3}` holding 1 for the first expert, 2 for the second and 3 for the third, and its
`token_embd.weight` a third row `{-1, -1}`, so a pass of token 0 routes expert 0, of token 1 expert
1 and of token 2 expert 2. Every case that needs a cap builds a first model of the fixture with no
`vram`, takes its `resident_bytes`, `reserve` and `slot_bytes`, closes it, and uses a `vram` of
those first two plus a third of the third as a one-slot cap and plus two thirds of it as a two-slot
cap.

- T-001 (FR-001, FR-003, FR-006): `a wrong profile converges on the conversation's experts`, proves
  on a machine whose library found a device, gated on `on_device` as `35`'s cases gate, that a model
  of the cache fixture built with the one-slot cap and a profile whose lines route expert 0 counts no
  hit over eight passes of token 1 at positions 0 to 7 and has `measure.routed` 8; that its ninth
  pass, at position 8, counts no hit; that passes at positions 9 to 16 count 8 hits, so the slot
  moved to the expert the conversation routes; that a model of the fixture built with the same cap
  and a profile whose lines route expert 1 counts one hit over one pass of token 1 at position 0, so
  the profile chose the slot; and that a third model of the fixture, built with the same cap and a
  profile holding only a line naming layer 9 that routes expert 1 and a line naming layer 0 that
  routes expert 9, counts one hit over one pass of token 0 at position 0, so both records are counted
  nowhere and the unfilled slot takes the lowest-numbered expert; and that a fourth model of the
  fixture, built with the same cap and a profile holding one line that routes expert 0 and one line
  that routes expert 1, counts one hit over one pass of token 0 at position 0, so the seed's tie went
  to the lower id.
- T-002 (FR-004, FR-005, FR-006): `an admitted expert is not computed before it lands`, proves on
  the same machine and fixture, with a model whose profile routes expert 0 and a second model whose
  profile routes expert 1, both built with the one-slot cap and both driven by passes of token 1 at
  positions 0 to 9, that the first model counts no hit over its first nine passes and one hit after
  its tenth, that the second counts ten hits, that the ninth pass's score row of the first model
  equals the second model's ninth pass row number for number, so the pass that chose the admission
  served none of the admitted expert from the card, and that the tenth pass's rows are equal too, so
  the landed copy holds the mapping's bytes.
- T-003 (FR-001, FR-007, FR-008, FR-010): `bench reports the hit rate and the profile reaches the
  model`, proves on `29`'s fixture and a machine where `compute.load()`'s `memory` gives free bytes,
  gated as `35`'s device cases gate, with a `vram` of a model of that fixture built with no `vram`'s
  `resident_bytes` plus its `reserve` plus half its `slot_bytes`, that fixture holding two experts,
  that `bench m --config F --prefill-tokens 4 --decode-tokens 4` with a config naming the fixture,
  that `vram` and a `profile` whose lines route expert 1 returns 0 and writes an `experts 4:` line
  ending `; 0.0% hit; 0 ahead, 0.0% landed; read ahead`, with `open` or `closed` after it, so the
  profile seeded the one slot and the routed expert 0 is not in it; that
  the same run on a config naming the fixture and that `vram` but no `profile` writes an `experts
  4:` line ending `; 100.0% hit; 0 ahead, 0.0% landed; read ahead`, with `open` or `closed` after
  it, so the default seed is expert 0; and that on a 4096-position copy of
  that fixture with a `vram` computed the same way for that copy and no `profile`, a run of the
  default lengths writes an `experts 2048:` line ending `; 100.0% hit; 0 ahead, 0.0% landed; read
  ahead`, with `open` or `closed` after it, so the hit count is the second
  turn's own.
- T-004 (FR-003, FR-006): `a contested slot keeps what the rounds chose`, proves on the same machine
  and fixture, with the two-slot cap and a profile whose lines route expert 0 and expert 1 once each,
  so the seed's tie goes to those two ids, and driven by passes at positions 0 to 7 of tokens 0, 0,
  1, 2, 2, 2, 2, 1 and a pass of token 2 at position 8, that the passes at positions 9, 10 and 11
  route tokens 0, 1 and 2 and count 0, 1 and 1 hits, so expert 2 replaced expert 0 because experts 0
  and 1 were each routed twice and expert 1 was routed later; and that a second model of the fixture
  with the same cap and the same profile, driven by eight passes of token 2 at positions 0 to 7 and a
  pass of token 2 at position 8, then a pass of token 1 at position 9 and a pass of token 0 at
  position 10, counts no hit at position 9 and one at position 10, so expert 1 was dropped and expert
  0 kept because those rounds routed neither and the tie went to the lower id.
- T-005 (FR-002, FR-012): `a profile is refused or carried as the run names it`, proves on the
  cache fixture and `compute.load(nil, false)`, on any machine, that `compute.model` given a profile
  path that does not exist returns nil and one line holding `model`, `profile` and that path, and
  that a profile file holding the line `{not json` returns nil and one line holding `model`,
  `profile` and that file; that `bench m --config F` on a config naming the fixture whose `profile`
  is the number 1 returns 1, writes one stderr line holding `profile` and writes no `prompt` line;
  and that `cli.run` with a settings record whose `args` are `{"serve", "m", "--config", F}` and
  whose `start` collects its endpoint, on a config naming a fixture entry and a `profile`, calls
  `start` with an endpoint whose `profile` is that path, and with nil when the file sets none.
- T-006 (FR-003): `the policy adapts again after another eight rounds`, proves on the same machine
  and fixture, with a model built with the one-slot cap and a profile whose lines route expert 0,
  driven by passes of token 1 at positions 0 to 8 and then passes of token 2 at positions 9 to 24,
  that its pass at position 16 counts no hit and its passes at positions 17 to 24 count 8 hits, so
  the second window ranked expert 2 above the expert the first window left in the slot: an
  implementation that adapts only once counts no hit at any of those positions, and one that adapts
  a pass late counts none at position 17.

## Open questions

- None.

## Amendments

- None.
