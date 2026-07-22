# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Corrode is

A swarming coding agent, in Rust, backed by the **hipfire** inference daemon
(`~/hipfire`). "Swarming" = it spawns many *prioritized* subagents and lets
hipfire's scheduler decide when each runs. It ships as two deployable pieces plus
a browser front-end:

- **corrode-daemon** — installed on a host. Owns everything stateful: the hipfire
  client, the swarm, the graph<->git VFS, and an **embedded HelixDB** store (graph
  + vectors + GraphRAG). It *is* the database — HelixDB is linked in-process, not
  run as a separate service.
- **corrode-web** — a separate, stateless web server. Serves the wasm webui and
  bridges browser <-> daemon. Links no agent logic.
- **webui/** — the wasm front-end: virtual terminal, filesystem/repo/graph
  explorer, agent interface. Not yet scaffolded (framework undecided).

Status: scaffold. The workspace compiles; the daemon runs one smoke swarm. The
daemon loop, the VFS impl, HelixDB queries, the web bridge, and the webui are not
written yet. Grep `ponytail:` for every deliberate seam and its upgrade trigger.

## Layout

```
crates/corrode-core     # shared wire types (Priority, AgentCommand/Event, node DTOs). Links nothing heavy; wasm-safe.
crates/corrode-daemon   # the agent (AGPL-3.0 — see below). modules: hipfire, swarm, vfs, graph
crates/corrode-web      # web server stub (Apache-2.0)
webui/                  # wasm front-end seam (out of the cargo workspace; its own trunk/wasm-pack build)
third_party/helix-db    # git submodule: HelixDB pinned at v2.3.5 (AGPL-3.0), linked in-process behind the `helix` feature
third_party/helix-skills# vendored HelixDB agent skills (MIT); Rust-relevant ones symlinked into .claude/skills/
```

## Commands

```bash
cargo build                                  # base workspace (no HelixDB compile)
cargo test                                   # unit tests
cargo test -p corrode-daemon <name>          # single test
cargo run -p corrode-daemon                  # smoke swarm against hipfire
cargo build -p corrode-daemon --features helix   # HEAVY: compiles vendored HelixDB (mimalloc/LMDB/HelixQL). Enables the real in-process store.
cargo run  -p corrode-web                    # web server stub
```

Env: `HIPFIRE_BASE_URL` (default `http://127.0.0.1:11435`), `HIPFIRE_API_KEY`,
`CORRODE_MODEL` (default `qwen3.5:9b`), `CORRODE_GRAPH_DIR` (HelixDB path, once
wired). A running `hipfire serve` is needed for `cargo run` to do anything real.

## Licensing — read before touching the daemon

**corrode-daemon is AGPL-3.0**, because the `helix` feature links HelixDB's
`helix_engine` in-process and HelixDB is AGPL-3.0. In-process linking makes the
daemon a derivative work; AGPL's network-use clause applies since the daemon is
served to the web UI. `corrode-core` and `corrode-web` link nothing GPL and stay
Apache-2.0 — keep it that way (don't add helix-db to them). If AGPL is
unacceptable, the options are a HelixDB commercial license or dropping to the
supervised-loopback deployment (helix as a child process over localhost, no
in-process link).

## HelixDB embedding

A git submodule at `third_party/helix-db`, **pinned to tag v2.3.5** (commit
`17e7ecf`) — the tag whose `helix_engine` is usable in-process; newer published
crates are HTTP-client-only. Clone with `git clone --recurse-submodules`, or run
`git submodule update --init third_party/helix-db` after a plain clone. The real
embed is `graph::embedded::HelixStore::open(path)`, which calls:

```rust
HelixGraphStorage::new(path, Config::default(), VersionInfo::default())
```

from `helix_db::helix_engine::storage_core`. HelixDB is one store for graph
traversal + vector search + GraphRAG: the graph side is the VFS's source of
truth; the vector side backs `AgentCommand::DocQuery` (documentation GraphRAG).

When writing HelixQL/Rust-DSL queries against it, the vendored **helix skills**
are symlinked into `.claude/skills/` (helix-query-rust, helix-query-optimize,
helix-query-json-dynamic, helix-cli, helix-memory-system). Use them.

**`--features helix` needs system OpenSSL + pkg-config** at build time. Upstream
helix-db (via its always-on `helix-metrics` crate) uses `native-tls`, so the
build links openssl regardless of features. This matches HelixDB's own build
requirements and works out of the box on hosts with `libssl-dev`/`pkg-config`
installed. The base workspace build needs none of this — helix-db is
feature-gated and `exclude`d from the workspace, so it's untouched until you pass
`--features helix`. (If you need an openssl-free pinned build, fork helix-db off
v2.3.5, switch its and `metrics`' reqwest to `rustls-tls`, and point the submodule
at the fork.)

## How hipfire's design constrains this codebase

Load-bearing, not stylistic. Read `~/hipfire/crates/hipfire-scheduler/src/lib.rs`
before changing swarm behavior.

1. **Priority is the only steering wheel.** Scheduler is banded u8 (0 realtime /
   64 default / 255 opportunistic), continuous batching with aging
   (anti-starvation). The swarm expresses intent by *band*, never by throttling
   locally. Speculative subagents go Opportunistic (idle GPU only). Bands are
   pinned to hipfire's `SCHED_PRIORITY_*`; `priority_bands_match_hipfire` guards it.
2. **Shared prompt prefix = shared KV cache** (`sessions_compatible_for_prefill`).
   Build subagent prompts as `[common repo/context prefix] + [short task tail]` so
   a wide fan-out collapses into one batched, prefix-shared run.
3. **Admission control is the daemon's**, against a VRAM/memory budget with
   per-owner fairness keys. Don't build a local scheduler or hard cap; enqueue and
   let hipfire queue/shed. The swarm's `inflight` semaphore is a socket courtesy.
4. **Embeddings + rerank are first-class** — code retrieval is a hipfire call, not
   a local index. (Doc retrieval instead uses HelixDB's own vectors, via GraphRAG.)
5. **Local single binary → requests are cheap, GPU-seconds aren't.** Optimize for
   batching and KV reuse, not request count.
