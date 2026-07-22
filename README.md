# Corrode

A swarming coding agent built on the [hipfire](../hipfire) inference daemon, with
an embedded [HelixDB](https://github.com/HelixDB/helix-db) graph+vector store.

Corrode spawns many *prioritized* subagents and lets hipfire's continuous, aging,
priority-banded scheduler decide when each runs — foreground work preempts,
speculative work fills idle GPU. It presents your repo through a graph<->git
**virtual file system** whose source of truth is a HelixDB graph, and answers
documentation questions with HelixDB's GraphRAG.

## Pieces

- **corrode-daemon** — installed on a host; owns the swarm, the VFS, and the
  embedded HelixDB store. HelixDB is linked in-process — no separate service.
- **corrode-web** — a separate web server that serves the wasm webui and bridges
  it to the daemon.
- **webui/** — wasm front-end: virtual terminal, filesystem/repo/graph explorer,
  agent interface. (Not yet built — framework undecided.)

## Status

Scaffold. Workspace compiles; the daemon runs one smoke swarm. Everything past
the seams is `ponytail:`-marked and unwritten.

## Build & run

```bash
cargo run -p corrode-daemon                       # smoke swarm (needs `hipfire serve`)
cargo build -p corrode-daemon --features helix    # + the real in-process HelixDB (heavy compile)
```

Env: `HIPFIRE_BASE_URL`, `HIPFIRE_API_KEY`, `CORRODE_MODEL`, `CORRODE_GRAPH_DIR`.

## Licensing

**corrode-daemon is AGPL-3.0** (it links AGPL HelixDB in-process). `corrode-core`
and `corrode-web` are Apache-2.0. See [CLAUDE.md](CLAUDE.md).
