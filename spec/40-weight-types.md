# 40: The weight types a block multiplies from

**Status**: done   **Depends on**: 05, 06, 11, 12, 13, 14, 15

## Problem & context

Every tensor a block, a linear block or the scores read is checked against one table, `WEIGHTS` in
`src/compute/maths.tl`, which holds four names: `f32`, `q8_0`, `q4_K` and `q5_K`. A block refuses a
tensor of any other type at build (`blocks.read_tensors` checks the seven attention tensors,
`feed_forward` and `linear` their own through `first_fault`) and a `matvec` or `row` refuses one at
call time (`looked_up`). The table is a hand-kept list of what ggml can multiply, and it is much
smaller than the list ggml has.

What that costs is every model on this machine. `Qwen3.6-35B-A3B` stops at
`block: tensor blk.34.ffn_down_exps.weight is q6_K`, and only 5 of its 753 tensors are outside the
four: those 3 `q6_K` expert matrices, in blocks 34, 38 and 39, and 2 `bf16` ones in the MTP block it
skips, the other 748 being `f32`, `q8_0`, `q4_K` and `q5_K` already. `Qwen3.8-Flash-Next` mixes
`iq2_s`, `iq3_s`, `iq3_xxs`, `iq4_xs`, `iq4_nl`, `q2_0` and `q6_K`. With `q6_K` added to the table
by hand, the 35B loads and answers a chat completion on this machine, so the four names are the
whole of what stands between the engine and the files it is for.

The boundary is ggml's, and this is where it can be read. Asked of the CPU module this build
produces (`libggml-cpu-icelake.so`, at the pinned commit `57e2c2c`), `ggml_get_type_traits_cpu`
reports a vector dot for 29 codes: `f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`,
`q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`,
`iq1_m`, `iq4_nl`, `iq4_xs`, `tq1_0`, `tq2_0`, `q1_0`, `q2_0`, `mxfp4`, `nvfp4` and `f8_e4m3`,
and none for `q8_1`, `q8_K`, `i8`, `i16`, `i32`, `i64` and `f64`, which are an intermediate format
or a scalar type and never a weight matrix.

That call is not reachable from this tree, which is worth writing down so nobody looks for it
again. The engine holds a handle to `libggml.so` only, and `libggml.so` does not export the symbol.
ggml's own loader picks the best-scoring CPU variant of the several beside it and `dlopen`s that
one with local visibility, so the symbol is not in the process's global namespace either, and no
device or registry function hands out the module's path or handle. Opening a variant file by guess
would `dlopen` one ggml did not load, whose constructor registers a second CPU backend, so that is
not a path this spec takes. The boundary therefore lands in Lua as a table of the type names
`ggml.type_of` already names codes for, complete for the pinned commit, and it ages the same way
`ggml.type_of`'s own code table does: a type from a later ggml is a row added there before a file
holding it can map at all.

Two things stay exactly as they are. The tensors whose numbers are read on the host rather than
multiplied from stay `f32`: the block and linear norms, `ffn_gate_inp.weight`,
`ffn_gate_inp_shexp.weight`, `output_norm.weight`, and the linear block's `ssm_a`, `ssm_alpha`,
`ssm_beta`, `ssm_conv1d`, `ssm_dt.bias` and `ssm_norm`. And `06` still plans a tensor's bytes from
its type, refusing a first dimension that is not a whole number of blocks, so a 256-wide type whose
rows are not whole blocks never reaches a multiply; nothing here reads quantized bytes as floats.

## User stories

- US-001 (P1) As the engine author, I want a mapped weight of any type ggml can multiply to be
  multiplied from, so that the models I have on disk load and answer instead of being refused on
  their first `q6_K` or `iq3_xxs` tensor.
- US-002 (P1) As the engine author, I want a type ggml cannot multiply, and a tensor whose numbers
  are read on the host, to still be refused with one line, so that a wrong type is a return value
  and not wrong numbers.

## Functional requirements

- FR-001 (US-001) WHEN `maths:matvec(name, vector, index)` or `maths:row(name, id)` is called on a
  tensor whose type is one of the 29 types listed in `## Problem & context`, THE SYSTEM
  SHALL multiply or gather it through ggml's kernel for that type and answer the numbers ggml
  decodes from its stored bytes, which a non-`f32` tensor reaches by ggml converting the vector to
  the format that kernel uses, for every one of those types and not only the four accepted today.
- FR-002 (US-001) WHEN a model, a block, a linear block or the scores are built from a handle whose
  multiplied tensors are each of a type listed in `## Problem & context` and whose host-read
  tensors, the ones FR-004 names, are each `f32`, THE SYSTEM SHALL build them and answer nil as the
  second value, whatever those multiplied tensors' types are called.
- FR-003 (US-002) IF a tensor the engine multiplies from is of a type not in that list, THEN THE
  SYSTEM SHALL answer nil and one line naming the operation, the tensor and that type's name, as
  `11/FR-002`, `12/FR-009`, `13/FR-012` and `14/FR-012` name a tensor whose type is at fault today,
  at build for a block, linear or scores tensor and at the call for `matvec` or `row`.
- FR-004 (US-002) IF a tensor whose numbers are read on the host, that is a block or linear norm,
  `ffn_gate_inp.weight`, `ffn_gate_inp_shexp.weight`, `output_norm.weight`, `ssm_a`, `ssm_alpha`,
  `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias` or `ssm_norm.weight`, is of a type other than `f32`, THEN
  THE SYSTEM SHALL answer nil and one line naming the operation, the tensor and that type's name.

## Non-goals

- No new type code and no change to how a file is mapped or a tensor's bytes are planned:
  `ggml.type_of`'s table, `05`'s tensor reading and `06`'s load-time refusal of a code it does not
  name or a first dimension that is not a whole number of blocks all stay as they are, so the two
  tables grow together.
- No per-backend policy. This is the CPU backend's list; whether a type can be multiplied on the
  card is placement's question, which `ggml_backend_dev_supports_op` can answer when a spec needs
  it, and nothing here places or refuses anything for a device.
- No change to `24`'s four cache types, and none to the tensors the host reads as numbers, which
  stay `f32` through FR-004.

## Tests

- T-001 (FR-001): `a q6_K tensor multiplies and gathers like its f32 twin`, proves the call-time
  half on `11/T-001`'s fixture grown by one tensor `six` of 256 numbers per row over 2 rows whose
  every number is 1, stated as `q6_K`, whose blocks `ggml_quantize_chunk` builds from those 1s:
  `row("six", 0)` returns 256 ones and `matvec("six", 256 ones)` returns `{256, 256}`, exactly what
  `11` asserts for its `q4_K` twin `quart`, so a type outside the four is decoded and multiplied
  and not read as bytes.
- T-002 (FR-002): `a model of q6_K matrices gives the scores its f32 twin gives`, proves the build
  half for both kinds of block on `15/T-002`'s fixture, whose metadata keys and tensor names it
  keeps, raised to `embedding_length` 256, `attention.head_count` 128, `ssm.time_step_rank` 128 and
  `ssm.inner_size` 256, with every tensor's numbers tiled as its dimensions grow, so each shape is
  the one `12/FR-010`, `13/FR-013` and `14/FR-013` asks of that model. Every hidden input is 256
  numbers a row. The 128 attention heads keep 2 numbers each, so `attn_output.weight` has a
  256-number input; the expert and shared-expert lengths are 256, so both down projections have a
  256-number input; and the 128 SSM value heads keep 2 numbers each, so `ssm_out.weight` does too.
  The query projection has 512 rows, the linear query/key/value projection 264 and the linear gate
  256, while the routers keep 2 rows and the token and score projections keep their original rows.
  `ssm_a` and `ssm_dt.bias` grow to 128 numbers, the alpha and beta projections to 128 rows and the
  convolution to 264 rows with its 2 taps unchanged. Every norm whose length is the hidden width,
  that is `attn_norm.weight`,
  `post_attention_norm.weight` and `output_norm.weight`, is 256 ones;
  `attn_q_norm.weight` and `attn_k_norm.weight` keep the 2 numbers `12/FR-010` asks of this
  fixture's head length, and `ssm_norm.weight` the 2 numbers `14/FR-013` asks of its value length.
  Every multiplied matrix is stated twice, once as `f32` and once as `q6_K` blocks
  `ggml_quantize_chunk` builds from the same numbers, the host-read tensors of FR-004 staying `f32`
  in both. Both `compute.model(backend, handle, 2)` calls answer a model and nil, and
  `forward(2, 0)` on each answers one number per `output.weight` row, the two lists differing at
  every component by no more than `1e-2` of the largest number in the `f32` model's list, because a
  `q6_K` block holds its numbers in 6-bit steps and every `q6_K` product converts its input vector
  to `q8_K` first, so the quantized twin's numbers carry that rounding; the exact product is
  T-001's.
- T-003 (FR-003): `a type ggml cannot multiply is refused by its name`, proves the boundary is
  ggml's and not "accept everything": T-002's fixture with `output.weight` stated as `q8_K`, which
  ggml has no vector dot for, answers nil and one line naming `scores`, that tensor and `q8_K`.
- T-004 (FR-004): `a tensor read on the host stays f32`, proves the host-read set was not widened
  with the table: T-002's fixture with `ffn_gate_inp.weight` stated as `q6_K` answers nil and one
  line naming `block`, that tensor and `q6_K`.

## Open questions

- None.

## Amendments

- None.
