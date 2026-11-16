local record fs is raw
  type ScanOpts = record
    absolute: boolean | nil
    followSymlinks: boolean | nil
    ignore: nil | function(name: string): boolean | nil
  end

  type RawWatchOpts = record
    recursive: boolean | nil
  end

  record Stat
    --- Size in bytes
    size: number
    accessTime: number
    modifyTime: number
    type: Stat.Type | nil
    --- Permission bits (Unix only)
    mode: number | nil
    type Type = DirEntry.Type
  end

  record DirEntry
    name: string
    type: DirEntry.Type
    enum Type
      "file"
      "dir"
      "symlink"
      "unknown"
    end
  end

  enum WatchEvent
    "create"
    "modify"
    "delete"
    "rename"
  end
  record Watcher
    poll: function(Watcher)
    wait: function(Watcher)
    close: function(Watcher)
  end

  interface raw
    exists: function(p: string): boolean
    isdir: function(p: string): boolean
    islink: function(p: string): boolean
    isfile: function(p: string): boolean
    readdir: function(p: string): nil | function(): DirEntry | nil
    mkdir: function(p: string): boolean
    mklink: function(src: string, dest: string): boolean
    rmlink: function(p: string): boolean
    stat: function(p: string): Stat | nil
    lstat: function(p: string): Stat | nil
    copyFile: function(src: string, dest: string): boolean
    chmod: function(p: string, mode: number): boolean
    watch: function(
      p: string,
      callback: function(event: WatchEvent, name: string),
      opts: RawWatchOpts | nil
    ): Watcher | nil
  end

  read: function(p: string): string | nil
  write: function(p: string, content: string): boolean
  copy: function(src: string, dest: string): boolean
  copyAtomic: function(src: string, dest: string)
  move: function(old: string, new: string)
  delete: function(p: string)
  rmdir: function(dir: string): boolean
  globToPattern: function(glob: string)
  globSuffix: function(glob: string): string | nil
  scan: function(
    cwd: string, glob: string, opts: ScanOpts | nil
  ): function(): string | nil
  mkdirAll: function(dir: string)
end

return fs
