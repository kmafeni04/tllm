local record process
  enum Stdio
    "pipe"
    "inherit"
    "null"
  end

  record Options
    cwd: string
    env: {string: string}
    stdin: string
    stdout: Stdio
    stderr: Stdio
    unsafe: boolean
  end

  record Child
    pid: integer
    kill: function(Child, force?: boolean)
    wait: function(Child): integer | nil, string | nil, string | nil
    poll: function(Child): integer | nil
  end

  -- the shape of spawn, so a caller can hand another one in its place
  type Spawn = function(
    string, {string} | nil, Options | nil
  ): Child | nil, string | nil

  platform: string
  spawn: function(
    string, {string} | nil, Options | nil
  ): Child | nil, string | nil
  exec: function(
    string, {string} | nil, Options | nil
  ): integer | nil, string | nil, string | nil
end

return process
