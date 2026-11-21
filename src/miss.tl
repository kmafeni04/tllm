-- the miss policy's rule: the share of a block's missed experts that goes by
-- DMA over the link, and the experts that share takes

local record miss
  -- the rounded, clamped DMA share for two positive rates in GB/s
  share: function(link: number, pool: number): number
  -- the distinct expert ids to send by DMA, in ascending order
  take: function(experts: {integer}, share: number): {integer}
end

local STEP: number = 0.05
local LOW: number = 0.10
local HIGH: number = 0.95

-- what the CPU pool and the link competing for RAM costs, at a CPU expert rate
-- of 30 GB/s and above and of 12 and below, linearly between the two
local function weight(pool: number): number
  if pool >= 30 then
    return 1.85
  end
  if pool <= 12 then
    return 0.8
  end
  return 0.8 + (pool - 12) / 18 * 1.05
end

function miss.share(link: number, pool: number): number
  local raw = link / (link + weight(pool) * pool)
  local rounded = math.floor(raw / STEP + 0.5) * STEP
  if rounded < LOW then
    return LOW
  end
  if rounded > HIGH then
    return HIGH
  end
  return rounded
end

function miss.take(experts: {integer}, share: number): {integer}
  local ids: {integer} = {}
  for index, expert in ipairs(experts) do
    ids[index] = expert
  end
  table.sort(ids)
  local taken: {integer} = {}
  local want = math.floor(#ids * share + 0.5)
  for index = 1, want do
    taken[index] = ids[index]
  end
  return taken
end

return miss
