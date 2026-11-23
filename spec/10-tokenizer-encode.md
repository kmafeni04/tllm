# 10: Turn text into the model's token ids

**Status**: done   **Depends on**: 09

## Problem & context

`09` turns ids into text. Nothing turns text into ids, so a prompt cannot reach a model: `serve`
takes text and the next specs prefill from it, and neither can do that with `decode` alone.

This spec adds one field to the tokenizer record, `encode`. `load` keeps reading the file through
`gguf.read` and gains two things: it keeps the raw `tokenizer.ggml.pre` and `tokenizer.ggml.merges`
entries and validates neither, so it still fails on exactly the files `09/FR-002` to `09/FR-004`
refuse and every fault in those two keys belongs to `encode`; and it maps each stored piece to the
first id holding it, so a surviving symbol is resolved without scanning `tokenizer.ggml.tokens`.
Text and pieces are validated and read through the `utf8` package, required as `utf8lib` so that
`types/utf8lib.d.tl` states the API this code uses (`valid`, `codes`, `char`) instead of leaning on
Teal's declaration of Lua's own `utf8`. No UTF-8 codec is written here. No tensor touched, no
route, command or weight changed.

```teal
local record tokenizer
  record Tokenizer
    size: integer
    decode: function(Tokenizer, {integer}): string | nil, string | nil
    encode: function(Tokenizer, string): {integer} | nil, string | nil
  end

  load: function(
    path: string, info?: gguf.Info
  ): Tokenizer | nil, string | nil
end
```

Encoding is three steps, and each has to match the model (llama.cpp, `src/llama-vocab.cpp`,
`llm_tokenizer_bpe_session::tokenize`, and `src/unicode.cpp`,
`unicode_regex_split_custom_gpt2`, `unicode_regex_split_custom_qwen35`):

1. The text is cut into words by the rule its file names in `tokenizer.ggml.pre`.
2. Each word is byte-encoded into the alphabet `decode` reads back: one codepoint per byte, so a
   U+0020 becomes `Ġ`, and the merges and pieces, which are stored in that alphabet, compare
   directly.
3. The word's codepoints are merged by the rank of `tokenizer.ggml.merges`, lowest first, and each
   survivor becomes the id of the piece with that text; a survivor no entry stores falls back to
   its own codepoints, one piece per codepoint (`FR-009`).

The word rules classify codepoints as letters (Unicode general category L), numbers (category N),
marks (category M) and whitespace (the Unicode whitespace set), which llama.cpp takes from a
generated table (`src/unicode-data.cpp`). The implementation carries those ranges in `tllm.unicode`,
each class as sorted low and high pairs searched by halving; Unicode 16.0.0 is the version the
shipped table holds, and `scripts/gen_unicode.tl` is what produces it, so this spec fixes the
classes, and the data is regenerated rather than edited by hand.

llama.cpp is looser than this spec in four places, on purpose. It looks a missing symbol up one
byte at a time and drops every byte whose piece it cannot find; here each of the symbol's
codepoints is looked up as a piece, and a codepoint with no piece is refused with one line. It
ignores a merge entry with no space after its first character; here that entry is refused. It
turns invalid UTF-8 into replacement characters instead of refusing it. It also gives a file with
no `tokenizer.ggml.pre` a default scheme; here that file is refused. In every case `09` set the
shape for a fault, and silence hides a broken file.

The Qwen3.6 file on this machine,
`/home/kome/.local/share/llm-models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, states
`tokenizer.ggml.model` as `gpt2`, `tokenizer.ggml.pre` as `qwen35` and carries 247587 merges, so
those are the two rules this spec builds. The tests still use hand-made fixtures, because a 23 GB
model cannot be a test input.

## User stories

- US-001 (P1) As the engine author, I want prompt text turned into ids built from the model's own merges and pieces, so that a prompt can be prefilled without a second copy of the vocabulary.
- US-002 (P1) As a person asking the model something, I want the ids the model was trained to see for that text, so that its answers are the ones it was trained to give.
- US-003 (P1) As a person with a file or a prompt the encoder cannot handle, I want one line naming what is at fault, so that I can tell a bad model from a bad engine.

## Functional requirements

- FR-001 (US-001) WHEN `encode` is called with valid UTF-8 text on a tokenizer whose file states
  `tokenizer.ggml.pre` as `gpt2` or `qwen35`, holds well-formed merges, and stores an entry for
  every byte-map codepoint the text encodes to, THE SYSTEM SHALL return one list of ids and nil,
  the empty text giving the empty list.
- FR-002 (US-002) WHEN a word is encoded, THE SYSTEM SHALL replace each of the word's bytes with
  the codepoint GPT-2's byte map gives that byte, the map `09/FR-006` reads back, so that a U+0020
  character becomes `Ġ`.
- FR-003 (US-003) IF `tokenizer.ggml.pre` is absent or its text is neither `gpt2` nor `qwen35`,
  THEN THE SYSTEM SHALL return nil and one line naming `tokenizer.ggml.pre` and either the value
  found or that the key is missing.
- FR-004 (US-003) IF `tokenizer.ggml.merges` is absent, is not an array of strings, or holds an
  entry with no space after its first character, THEN THE SYSTEM SHALL return nil and one line
  naming `tokenizer.ggml.merges`.
- FR-005 (US-002) WHEN the file states `tokenizer.ggml.pre` as `gpt2`, THE SYSTEM SHALL split the
  text into words, taking at each codepoint the first of these that applies: an apostrophe and then
  `s`, `t`, `m` or `d`, or then `re`, `ve` or `ll`, every one of the letters in lowercase ASCII; a
  U+0020 character and then a run of letters, or a run of letters; a U+0020 character and then a
  run of digits, or a run of digits; a U+0020 character and then a run of codepoints that are none
  of a whitespace, a letter or a digit, or such a run; all but the last of a run of two or more
  whitespace codepoints that more text follows; the whole run of whitespace codepoints otherwise.
- FR-006 (US-002) WHEN the file states `tokenizer.ggml.pre` as `qwen35`, THE SYSTEM SHALL split
  the text into words, taking at each codepoint the first of these that applies: an apostrophe and
  then `s`, `t`, `m` or `d`, or then `re`, `ve` or `ll`, every one of the letters in either case;
  one codepoint that is not CR, LF, a letter, a digit or a mark and then a run of letters and
  marks, or a run of letters and marks; one digit; a U+0020 character and then a run of codepoints
  that are none of a whitespace, a letter, a mark or a digit, or such a run, with every CR and LF
  that directly follows it; a run of whitespace codepoints holding a CR or LF, cut just after its
  last CR or LF; all but the last of a run of two or more whitespace codepoints that more text
  follows; the whole run of whitespace codepoints otherwise.
- FR-007 (US-001) WHEN a word's bytes have become codepoints, THE SYSTEM SHALL join the adjacent
  pair whose left and right texts appear as an entry of `tokenizer.ggml.merges`, taking the pair
  whose entry comes first in that array at each step, until no adjacent pair appears in it.
- FR-013 (US-001) THE SYSTEM SHALL cost one word's joining no more than its merges times the
  logarithm of the word's length, so that a word with no separator in it, a run of Han characters
  or a long identifier, costs what its merges cost and not a scan of the word per merge.
- FR-008 (US-001) WHEN the joining ends, THE SYSTEM SHALL append, for each surviving symbol from
  left to right, the id of the first entry of `tokenizer.ggml.tokens` whose stored text equals that
  symbol.
- FR-009 (US-001) IF no entry of `tokenizer.ggml.tokens` has stored text equal to a surviving
  symbol, THEN THE SYSTEM SHALL append, in order, the id of the first entry whose stored text
  equals each of that symbol's byte-map codepoints in turn, `Ġ` counting as one codepoint.
- FR-010 (US-003) IF the text is not valid UTF-8, THEN THE SYSTEM SHALL return nil and one line
  saying so.
- FR-011 (US-003) IF no entry stores one of the codepoints `FR-009` looks up, THEN THE SYSTEM SHALL
  return nil and one line naming the first such codepoint as a decimal number.
- FR-012 (US-003) WHEN `load` reads a file, THE SYSTEM SHALL keep its `tokenizer.ggml.pre` and
  `tokenizer.ggml.merges` entries as read and refuse neither, so that `FR-003` and `FR-004` are
  answered by `encode` alone.
- FR-014 (US-002) WHEN the text holds the stored text of an entry of `tokenizer.ggml.tokens` whose
  `tokenizer.ggml.token_type` is CONTROL, THE SYSTEM SHALL return that entry's id in place of the
  ids the text around it encodes to, the longest such text winning where two start at the same
  character, and SHALL encode the text before and after it by `FR-001` to `FR-011`, so that the
  turn markers a chat template writes reach the model as the tokens it was trained to see instead
  of the characters they are spelled with.

## Non-goals

- No special-token text beyond `FR-014`: a stored control token's text is the only text that becomes
  an id of its own, a user-defined token's text still encodes as ordinary characters, and no BOS or
  EOS is inserted.
- No scheme other than the two rules above: `qwen2`, `llama3`, `gemma4`, `deepseek`, `spm` and the
  rest are refused as `FR-003` states, and the text is never normalised.
- No ignore-merges flag: `tokenizer.ggml.ignore_merges` is not read, so a word is always cut into
  codepoints and merged, never taken whole as one piece.
- No route, command, weight or sampling change: `serve` and `bench` are untouched, and no file
  stays open.

## Tests

- T-001 (FR-001, FR-002, FR-007, FR-008, FR-009): `text becomes the ids its merges give`, proves
  on a `gpt2` fixture whose pieces are `a`, `b`, `c`, `ab`, `bc` and `Ġ`, all type 1, and whose
  merges are `a b` then `b c`, that `encode("abc")` returns `{3, 2}` and nil, that a second file
  holding the same pieces with the merges reversed gives `{0, 4}`, that `encode("a b")` returns
  `{0, 5, 1}` and that `decode` of `{0, 5, 1}` returns `a b`, that a third file whose pieces are
  `a` and `b`, all type 1, and whose only merge is `a b`, so that no entry stores the symbol `ab`,
  gives `{0, 1}` for `encode("ab")`, and that `encode("")` returns the empty list and nil.
- T-002 (FR-005, FR-006): `each scheme splits text into its own words`, proves on two files
  holding the same pieces `A`, `'`, `S`, `'S`, `1`, `2`, `12`, `a`, `Ċ` and `ĊĊ` and the same merges
  `' S`, `1 2` and `Ċ Ċ`, one stating `tokenizer.ggml.pre` as `gpt2` and the other as `qwen35`,
  that with `gpt2` `encode("A'S")` returns `{0, 1, 2}`, `encode("12")` returns `{6}` and
  `encode("a\n\na")` returns `{7, 8, 8, 7}`, and that with `qwen35` the same text returns
  `{0, 3}`, `{4, 5}` and `{7, 9, 7}`.
- T-003 (FR-003, FR-004, FR-010, FR-011, FR-012): `text that cannot be encoded is refused`,
  proves that a tokenizer built from a file stating `gpt2` with no `tokenizer.ggml.merges` loads
  and that `encode("a")` on it returns nil and a line naming that key, that a file whose merges
  hold an integer loads and whose `encode("a")` returns nil and a line naming that key, that a file
  whose only merge entry is a newline then `ab` loads and whose `encode("a")` returns nil and a
  line naming that key and holding no newline, that a file with no `tokenizer.ggml.pre` loads and
  whose `encode("a")` returns nil and a line naming that key, that a file stating `llama3` loads
  and whose `encode("a")` returns nil and a line holding `llama3`, that a file whose pieces lack
  `b` and whose merges are empty returns nil for `encode("ab")` and a line holding `98`, and that
  `encode` of a string holding the single byte `0xFF` returns nil and a line saying the text is
  not valid UTF-8.
- T-004 (FR-006): `a combining mark stays with the letters beside it`, proves on a `qwen35` file
  whose pieces are `e`, `Ì`, `ģ`, `eÌ` and `5`, all type 1, and whose only merge is `e Ì`, that
  `encode("e\204\1295")` returns `{3, 2, 4}` and nil, the mark joining the letter word instead of
  forming one of its own.
- T-006 (FR-014): `a control token's text becomes its id`, proves on a `gpt2` fixture whose pieces
  are `a`, `b`, `X` at type 1 and `aXb` at type 3, with no merges, that `encode("aXb")` returns
  `{3}` rather than the three ids of the characters, that `encode("Xa")` returns `{2, 0}` and
  `encode("ab")` `{0, 1}`, and that the same file with `aXb` at type 4 encodes `"aXb"` to `{0, 2, 1}`.
- T-005 (FR-013): `one long word costs its merges, not a scan per merge`, proves on a file whose
  pieces and merges chain `a` and `b` into `ab`, `abab` and so on that a word of 8192 codepoints
  encodes to the one id of its whole text in under four tenths of a second, which a scan of the
  word per merge does not.

## Open questions

- None.

## Amendments

- A control token's text now encodes to its own id (`FR-014`, and the non-goal this supersedes).
  This spec was written with `parse_special` false in mind, which read a chat template's
  `<|im_start|>` and `<|im_end|>` as the characters they are spelled with: the Qwen3.6 file states
  `tokenizer.ggml.eos_token_id` as 248046, `<|im_end|>`, so a prompt written that way made the model
  spell the marker out as text and never sample the id `16/FR-007` stops on. A user-defined token's
  text still encodes as ordinary characters, which `T-006` pins.
