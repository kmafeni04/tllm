# 09: Turn token ids back into text from the model's vocabulary

**Status**: done   **Depends on**: 05

## Problem & context

Nothing in the project turns a token id into text. `05` reads a GGUF file's metadata, and a model
file carries its vocabulary there: `tokenizer.ggml.model` names the scheme,
`tokenizer.ggml.tokens` holds one stored piece per id, and `tokenizer.ggml.token_type` classifies
each piece. Every model on this machine but Gemma states `gpt2`; Gemma states `gemma4`, which is a
later spec's business. No ids exist yet either, but a sampler and a generation route are the
next specs, and neither can show a user anything without this.

This spec adds one module, `tllm.tokenizer`, exporting `load` and a tokenizer whose `decode` turns
an ordered list of ids into text. It reads the vocabulary through `gguf.read`, keeps the pieces
and their types, and holds nothing else: no tensor is touched, no file stays open, and no route or
command changes.

The pieces are not text yet. A `gpt2` vocabulary stores bytes as codepoints through GPT-2's byte
map: 188 bytes (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF) keep their own codepoint, and the other 68 take
codepoints 256 to 323 in increasing byte order, so `Ġworld` is the six bytes ` world`. Decoding is
therefore codepoint to byte, in order (llama.cpp, `src/unicode.cpp`, `unicode_utf8_to_byte_map`;
confirmed on the local Qwen3.6 file, whose ids `9419, 1814, 11, 50203` come back as
`Hello world, café`).

What one id contributes follows its type (llama.cpp, `src/llama-vocab.cpp`, `token_to_piece` and
`llama_detokenize`): a normal piece is byte-decoded, a user-defined piece is copied as stored, and
a control, unknown or unused piece contributes nothing, which is llama.cpp's default detokenize
with special tokens removed. Byte-fallback tokens (type 6) are refused, as `FR-011` states,
because no `gpt2` vocabulary stores them. Every scheme other than `gpt2` is refused as `FR-003`
states.

Two shapes this spec fixes. `load` takes a path and calls `gguf.read` itself, so the
tokenizer needs no `weights` handle and the format stays in one module; the reader's failure is
passed through unchanged, so a caller prints one reason and not two. A vocabulary that cannot be
used is refused with nil and one line, the shape `05` and `weights.load` already use, rather than
an error, so the caller decides how loudly to say it.

```teal
local record tokenizer
  record Tokenizer
    size: integer
    decode: function(Tokenizer, {integer}): string | nil, string | nil
  end

  load: function(path: string): Tokenizer | nil, string | nil
end
```

## User stories

- US-001 (P1) As the engine author, I want a tokenizer built from a model's own vocabulary, so that a generated id can become text without a second copy of the vocabulary.
- US-002 (P1) As a person reading the engine's output, I want ids to come back as the bytes the model meant, so that the text is legible and no marker tokens leak into it.
- US-003 (P1) As a person with a file the tokenizer cannot use, I want one line naming what is at fault, so that I can tell a bad model from a bad engine.

## Functional requirements

- FR-001 (US-001) WHEN `require("tllm.tokenizer").load(P)` is called with `P` naming a GGUF file
  whose `tokenizer.ggml.model` is `gpt2`, THE SYSTEM SHALL return a tokenizer whose `size` is
  the number of entries in `tokenizer.ggml.tokens`, and nil as the second value.
- FR-002 (US-003) IF `gguf.read(P)` returns a failure, THEN THE SYSTEM SHALL return nil and that
  failure as the second value.
- FR-003 (US-003) IF the file states no `tokenizer.ggml.model`, or its text is not `gpt2`, THEN
  THE SYSTEM SHALL return nil and one line naming `tokenizer.ggml.model` and either the value
  found or that the key is missing.
- FR-004 (US-003) IF `tokenizer.ggml.tokens` is absent or is not an array whose every entry is a
  string, or `tokenizer.ggml.token_type` is present but is not an array of as many integers as
  `tokens` has entries, THEN THE SYSTEM SHALL return nil and one line naming the key at fault.
- FR-005 (US-001) WHEN `tokenizer.ggml.token_type` is absent, THE SYSTEM SHALL count every token
  as type 1.
- FR-006 (US-002) WHEN `decode` is given an id whose type is 1, THE SYSTEM SHALL contribute that
  token's stored text with every UTF-8 codepoint replaced by the byte GPT-2's map assigns it, an
  empty stored text contributing nothing.
- FR-007 (US-002) WHEN `decode` is given an id whose type is 2, 3 or 5, THE SYSTEM SHALL
  contribute nothing.
- FR-008 (US-002) WHEN `decode` is given an id whose type is 4, THE SYSTEM SHALL contribute that
  token's stored text unchanged.
- FR-009 (US-002) WHEN `decode` is given a list of ids whose every id decodes, THE SYSTEM SHALL
  return the concatenation of their contributions in list order and nil, the empty list giving
  the empty string.
- FR-010 (US-003) IF an id is less than 0 or not less than `size`, THEN THE SYSTEM SHALL return
  nil and one line naming the id and the vocabulary size as a number.
- FR-011 (US-003) IF an id's type is not one of 1 to 5, THEN THE SYSTEM SHALL return nil and one
  line naming the id and its type.
- FR-012 (US-003) IF a type-1 token's stored text is not valid UTF-8, or holds a codepoint that
  GPT-2's map gives no byte, THEN THE SYSTEM SHALL return nil and one line naming the id.

## Non-goals

- No encoding: text to ids, splitting, merges, scores, pre-tokenizer schemes and byte
  fallback are a later spec, and `decode` never produces ids.
- No scheme other than `gpt2`: `gemma4`, `llama`, `bert`, `t5` and the rest are refused as
  `FR-003` states.
- No route, command, weight or mapping: `serve` and `bench` are unchanged, and no file is left
  open.

## Tests

- T-001 (FR-001, FR-005, FR-006, FR-007, FR-008, FR-009): `a gpt2 vocabulary decodes ids to
  text`, proves on a fixture whose `tokenizer.ggml.tokens` holds `Hello` (type 1), `Ġworld`
  (type 1), `ĠcafÃ©` (type 1), `<|im_end|>` (type 3), `<|probe|>` (type 2), `Ġpad` (type 4) and
  `spare` (type 5) that `size` is 7, that `decode` of `{0, 1, 2}` returns `Hello world café`
  and nil, that `decode` of `{3, 4, 6}` returns the empty string and nil, that `decode` of
  `{5}` returns `Ġpad` and nil, that `decode` of `{}` returns the empty string and nil, that
  `load` returns nil as its second value for both fixtures, and on a second fixture carrying
  `tokenizer.ggml.tokens` holding `Ġend` and no `tokenizer.ggml.token_type` that `decode` of
  `{0}` returns ` end` and nil.
- T-002 (FR-002, FR-003, FR-004): `a vocabulary that cannot be used is refused`, proves that a
  file holding `GGML` bytes returns nil and a line equal to the one `gguf.read` returns for
  that same path, that a GGUF without `tokenizer.ggml.model` returns nil and a line naming that
  key, that a GGUF stating `gemma4` returns nil and a line holding `gemma4`, that a GGUF
  stating `gpt2` with no `tokenizer.ggml.tokens` returns nil and a line naming that key, that a
  GGUF whose `tokens` holds an integer returns nil and a line naming that key, and that a GGUF
  whose `tokens` holds two strings and whose `token_type` holds one integer returns nil and a
  line naming `tokenizer.ggml.token_type`.
- T-003 (FR-010, FR-011, FR-012): `an id that cannot be decoded is refused`, proves on a fixture
  with five tokens holding `Hello` (type 1), `π` (type 1), `<0x0A>` (type 6), `spare` (type 5)
  and `tail` (type 5) that `decode` of `{0, 9}` returns nil and a line holding `9` and `5`,
  that `decode` of `{-1}` returns nil and a line holding `-1`, that `decode` of `{2}` returns
  nil and a line holding `2` and `6`, that `decode` of `{1}` returns nil and a line holding
  `1`, and on a third fixture whose fourth entry stores the single byte `0xFF` with type 1 that
  `decode` of `{3}` returns nil and a line holding `3`.

## Open questions

- None.

## Amendments

- None.
