-- The `utf8` package, required as utf8lib because Teal answers
-- require("utf8") with its own declaration of Lua's utf8. See the
-- `package` field in lde.json.
local record utf8lib
  char: function(codepoint: number...): string
  codes: function(text: string): (function(
    text: string, position: integer
  ): (integer, integer), string, integer)
  codepoint: function(text: string, first?: integer, last?: integer): integer...
  decode: function(text: string, at?: integer): integer, integer
  one: function(char: string): integer, integer
  back: function(text: string, at: integer): integer
  forward: function(text: string, at: integer): integer
  len: function(
    text: string, first?: integer, last?: integer
  ): integer | nil, integer | nil
  valid: function(
    text: string, first?: integer, last?: integer
  ): boolean, integer | nil
  fromLatin1: function(text: string): string
  fromUTF16: function(text: string, bigEndian?: boolean): string
  toUTF16: function(text: string, bigEndian?: boolean): string
end

return utf8lib
