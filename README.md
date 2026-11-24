<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo-light.svg" width="220" alt="tllm">
</picture>

# tllm

An LLM inference engine written in Teal and run on LuaJIT through
[lde](https://lde.sh). It serves a GGUF model over an OpenAI-style chat
completions route and benchmarks its own numbers from the command line.

The project is early. What it does is written down in [`spec/`](spec/README.md),
one file per feature.

## Requirements

- [lde](https://lde.sh/docs/general/getting-started/installation)
- `cmake` and a C/C++ toolchain
- The CUDA toolkit, plus an NVIDIA GPU (the build enables the CUDA backend for
  `sm_86`)
- A network connection on the first build, to fetch ggml and yyjson

## Install

From a checkout:

```sh
lde sync
```

The first run compiles ggml and the CUDA backend, so it takes a while.

## Use

```sh
lde run -- serve [MODEL] [--config FILE] [--host HOST] [--port PORT]
lde run -- bench MODEL [--config FILE] [--prefill-tokens N] [--decode-tokens N] [--dump-routing FILE]
```

`lde run --` runs this package, and everything after it goes to `tllm`.

`serve` listens on `127.0.0.1:8080` unless the flags or the config file say
otherwise, and answers `GET /health` and `POST /v1/chat/completions`.

`MODEL` is either a path to a `.gguf` file or the name of an entry in the
config's `models` map. The config file is `tllm/config.json` under
`$XDG_CONFIG_HOME`, falling back to `~/.config/tllm/config.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "models": {
    "qwen3.8-27b": {
      "path": "/home/me/models/qwen3.8-27b.gguf"
    }
  }
}
```

`bench` prints prefill and decode rates for a model, and `--dump-routing FILE`
writes the expert routing of the run.

## Development

Run these before opening a pull request; together they cover every `.tl` file
in the project.

```sh
lde run check   # typecheck every .tl file
lde run test    # run the tests
lde run format  # format every .tl file with cerulean
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. New features are backed by a
spec in [`spec/`](spec/README.md).

## License

MIT, see [LICENSE](LICENSE).
