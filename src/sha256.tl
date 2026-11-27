local ffi = require("ffi")

-- the EVP digest interface of libcrypto, and the read that feeds it a shard at
-- a time: the names shared with `tllm.weights` and `tltp.utils.crypto` are
-- spelled exactly as those modules spell them, so the declarations agree
ffi.cdef(
  [[
typedef long off_t;
long pread(int fd, void *buf, size_t count, off_t offset);
typedef struct EVP_MD EVP_MD;
EVP_MD *EVP_sha256(void);
typedef struct EVP_MD_CTX EVP_MD_CTX;
EVP_MD_CTX *EVP_MD_CTX_new(void);
void EVP_MD_CTX_free(EVP_MD_CTX *ctx);
int EVP_DigestInit_ex(EVP_MD_CTX *ctx, const EVP_MD *type, void *impl);
int EVP_DigestUpdate(EVP_MD_CTX *ctx, const void *data, size_t len);
int EVP_DigestFinal_ex(EVP_MD_CTX *ctx, unsigned char *md, unsigned int *s);
]]
)

local record sha256
  -- the sha256 of the `bytes` bytes a descriptor holds, as 64 hex digits, or
  -- nil when the descriptor or libcrypto refused
  descriptor: function(descriptor: integer, bytes: integer): string | nil
end

local record LibC
  pread: function(integer, ffi.CData, integer, integer): integer
end

local record Crypto
  EVP_sha256: function(): ffi.CData
  EVP_MD_CTX_new: function(): ffi.CData
  EVP_MD_CTX_free: function(ffi.CData)
  EVP_DigestInit_ex: function(ffi.CData, ffi.CData, ffi.CData | nil): integer
  EVP_DigestUpdate: function(ffi.CData, ffi.CData, integer): integer
  EVP_DigestFinal_ex: function(ffi.CData, ffi.CData, ffi.CData): integer
end

local C = ffi.C as LibC
local crypto = ffi.load("crypto") as Crypto
local CHUNK: integer = 1048576
local DIGEST: integer = 32

local type Bytes = {integer: integer}

function sha256.descriptor(descriptor: integer, bytes: integer): string | nil
  local context = crypto.EVP_MD_CTX_new()
  if context == nil then
    return nil
  end
  local started = crypto.EVP_DigestInit_ex(context, crypto.EVP_sha256(), nil)
  if started ~= 1 then
    crypto.EVP_MD_CTX_free(context)
    return nil
  end

  local chunk = ffi.new("unsigned char[?]", CHUNK)
  local left = bytes
  local offset: integer = 0
  while left > 0 do
    local asked = CHUNK
    if left < asked then
      asked = left
    end
    local got = C.pread(descriptor, chunk, asked, offset)
    if got <= 0 then
      crypto.EVP_MD_CTX_free(context)
      return nil
    end
    crypto.EVP_DigestUpdate(context, chunk, got as integer)
    offset = offset + (got as integer)
    left = left - (got as integer)
  end

  local digest = ffi.new("unsigned char[?]", DIGEST)
  local size = ffi.new("unsigned int[1]")
  local ended = crypto.EVP_DigestFinal_ex(context, digest, size)
  crypto.EVP_MD_CTX_free(context)
  if ended ~= 1 then
    return nil
  end

  local cells = digest as Bytes
  local count = (size as Bytes)[0] as integer
  local parts: {string} = {}
  for index = 0, count - 1 do
    parts[index + 1] = string.format("%02x", cells[index] % 256)
  end
  return table.concat(parts)
end

return sha256
