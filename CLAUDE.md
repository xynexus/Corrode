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
crates/corrode-daemon   # the agent (AGPL-3.0 — see below). modules: daemon (command loop), planner, plan_graph (reactive scheduler), swarm, roles, hipfire, skills, toolcall (Needle shim), tools (tool-exec loop), vfs, graph
crates/corrode-web      # web server stub (Apache-2.0)
crates/needle-toolcall-shim  # vendored Needle tool-call model (Apache-2.0, CPU/candle). Workspace-EXCLUDED; corrode-daemon links it behind `--features needle`. Weights committed under assets/needle.
third_party/needle      # git submodule: upstream Needle (Cactus) — training/finetuning code, kept for finetuning Needle on Corrode's real tool set.
webui/                  # wasm front-end seam (out of the cargo workspace; its own trunk/wasm-pack build)
third_party/helix-db    # git submodule: HelixDB pinned at v2.3.5 (AGPL-3.0), linked in-process behind the `helix` feature
third_party/helix-skills# vendored HelixDB agent skills (MIT); Rust-relevant ones symlinked into .claude/skills/
```

## Commands

```bash
cargo build                                  # base workspace (no HelixDB compile)
cargo test                                   # unit tests
cargo test -p corrode-daemon <name>          # single test
cargo run -p corrode-daemon                  # serve the daemon ws at ws://127.0.0.1:7878/agent
cargo build -p corrode-daemon --features helix   # HEAVY: compiles vendored HelixDB (mimalloc/LMDB/HelixQL). Enables the real in-process store.
cargo build -p corrode-daemon --features needle  # compiles the Needle tool-call shim (CPU/candle). Enables reliable tool-calling for small models.
cargo run  -p corrode-web                    # serve UI on http://127.0.0.1:8787, proxy /agent -> daemon
```

Run the pair: start `corrode-daemon` (needs `hipfire serve` up for role resolution),
then `corrode-web`, then open http://127.0.0.1:8787 — the dev page drives the
swarm over the bridge.

Env: `HIPFIRE_BASE_URL` (default `http://127.0.0.1:11435`), `HIPFIRE_API_KEY`,
`CORRODE_MODEL` (offline fallback model for all roles), `CORRODE_ROLES` (path to a
JSON `role -> model-id` override map), `CORRODE_REPO` (VFS root, default `.`),
`CORRODE_GRAPH_DIR` (HelixDB path under `--features helix`),
`CORRODE_NEEDLE_ASSETS` (Needle asset dir under `--features needle`; defaults to the
vendored `crates/needle-toolcall-shim/assets/needle`, resolved at build time, so it
works out of the box; absent -> tool-caller disabled, swarm falls back to
model-emitted calls), `CORRODE_SMALL_MODELS` (comma-separated substrings that
force-classify a model as "small" -> uses the Needle tool loop) and
`CORRODE_SMALL_MODEL_MAX_B` (billions-param cutoff below which a model counts as small,
default 32), `CORRODE_DAEMON_ADDR` (daemon ws bind, default `127.0.0.1:7878`),
`CORRODE_WEB_ADDR` (web bind, default `127.0.0.1:8787`), `CORRODE_DAEMON_URL`
(daemon ws the web proxies to), `CORRODE_MAX_TOKENS` (per-call output cap,
default 1024). The hipfire background daemon must be up (`hipfire start`, not just
`serve` — `serve` is only the HTTP frontend) for the daemon to resolve roles and
generate.

## Command loop, transport & daemon state

`daemon.rs` is the transport-agnostic loop: drain `AgentCommand` off an mpsc
channel, dispatch, stream `AgentEvent` back. `server.rs` puts it on a WebSocket —
the daemon serves `/agent`, bridging each connection's frames to a per-connection
channel pair over the shared `Daemon`. `corrode-web` serves the UI and *proxies*
`/agent` to the daemon (browser → web → daemon), keeping the daemon private; the
same loop serves both. Frames are the serde-JSON encoding of the enums (externally
tagged, e.g. `{"Prompt":{"text":"...","priority":0}}`). The `Daemon` owns the
host-side state handlers reach via `&self`: the `Swarm`, the `RoleModels`
assignments, an `Option<Box<dyn GraphStore>>` (HelixDB; `None` without
`--features helix`), an `Arc<dyn Vfs>`, and the `ApprovalGate`. Dispatch: `Prompt`→swarm
(spawned concurrently — it can be long-lived and block on approval, so the loop keeps
receiving), `ListDir`→vfs (real), `DocQuery`→graph (real when helix built),
`TerminalInput`→pty, `ApprovalResponse`→resolves a pending mutating-tool approval.

## Roles

`roles.rs` maps swarm roles (research/orchestration/architect/coder/review) to
models. At startup the daemon calls `list_models` on hipfire and resolves
assignments: a `CORRODE_ROLES` override wins if it names a served model, else a
default pick (first served non-embedding/non-image model). If hipfire is
unreachable, all roles fall back to `CORRODE_MODEL`.

## Planner

`planner.rs` is the two-phase decomposition, driven by `Daemon::plan`: phase 1 asks
the orchestration model for a JSON plan; phase 2 (`parse_plan`) turns it into
role-tagged `PlannedSubtask`s. Empty/unparseable plan degrades to one coder task on
the raw prompt. `plan` returns those subtasks plus the shared prefix; the daemon
seeds a `plan_graph::PlanGraph` with them and drives it via `run_reactive`.

`plan_graph.rs` is the **reactive scheduler** — the daemon's answer to "HelixDB has
no triggers" (confirmed: none at any layer, so reactivity lives here, like Leptos
builds it over state not in the DB). The graph is a dependency graph a *running*
agent can grow: `run_reactive` launches every ready task, and on each completion
marks it, folds in the tasks it emitted, and reschedules — until nothing is ready
or in flight. A subagent emits follow-up work (a test contract, a research
spin-off) by ending its reply with a fenced ` ```tasks ` JSON block, which
`parse_emitted` folds back in (`after: true` depends on the emitter). Tasks left
unschedulable after the run settles (`stuck`) surface as an Error. `band_for` maps
role→band (orchestration→Realtime, architect/coder/review→Default,
research→Opportunistic).

**Provenance.** A `PlanGraph` is rooted at a `plan` node (one per Prompt turn). Every
task is `part_of` the plan; an emitted task is a *contract* and also links
`emitted_from` its emitter; a file a task writes becomes a `code` node `produced_by`
that task (paths collected from `write_file` calls in the tool loop). `graph.provenance()`
exports this as nodes+edges, and `Daemon::persist_provenance` writes it to the graph
store via the `GraphStore::upsert_node` / `add_edge` seam so the code↔task↔plan lineage
is queryable. ponytail: the HelixDB write path is still stubbed (bails) — the in-memory
provenance is correct and tested; persistence logs "unavailable" until those writes land.

Every prompt in a turn — the orchestration call and each subagent
(`subagent_prompt`) — begins with a byte-identical **context prefix**
(`Daemon::context_prefix`), so hipfire batches them prefix-shared and reuses KV when
they land on the same model. The divergent role/task goes in the tail; nothing
role-specific precedes the prefix. The `subagent_prompt` test guards this invariant.
Remaining ponytail: the prefix is a shallow VFS root listing (plus AGENTS.md rules
and ranked skills) — the graph-backed VFS will supply richer, relevance-ranked
context without changing the sharing shape.

## Tool-calling (Needle shim)

`toolcall.rs` gives the swarm's small models reliable tool-calling. Small chat models
botch tool-call JSON; **Needle** (a tiny CPU/candle encoder-decoder) takes a plain
plain-English instruction plus a tool schema and picks ONE tool per turn, formatting
the call for them. Its native tool schema is *flat* — `parameters` is a `name -> {type,
description, required}` map, NOT the OpenAI `type:object/properties/required` nesting
(feeding the nested form is out-of-distribution and degrades output). The daemon
depends on the `ToolCaller` *trait* (always defined); the Needle backend is
feature-gated (`--features needle`), mirroring `graph::GraphStore` — base build never
compiles candle. `Daemon` holds `Option<Arc<dyn ToolCaller>>`, loaded from
`CORRODE_NEEDLE_ASSETS` (default: the vendored crate's weights). The shim was merged in
from a throwaway experiment and lives at `crates/needle-toolcall-shim` — a
workspace-EXCLUDED crate, weights committed under `assets/needle`.

**First use — reactive-planner task emission.** A subagent proposes a follow-up as a
plain-English `NEXT:` line (easy for a small model; no JSON). `emit_followups` feeds
that one line to Needle against `plan_graph::ROLE_TASK_TOOLS` (one tool per role —
`research_task`/`coding_task`/`architecture_task`/`review_task`); Needle picks the tool,
which gives the task's role (band/model). The task text stays verbatim from the `NEXT:`
line (Needle's `task` arg truncates pre-finetune). One instruction → one task; the
reactive graph chains the rest. No caller / Needle error -> the task still queues as
Coder; no `NEXT:` line -> no emission. ponytail: Needle will be finetuned on Corrode's
actual tools (incl. the real tool-execution set: read_file, run_command, ...) so the
small coders' tools are picked up reliably; then its structured args become trustworthy
too. Note: the guide's enum/literal token-forcing had a bug (the merged `":"` token
bypassed it) — fixed in `needle-toolcall-shim/src/guide.rs`.

**Second use — the tool-execution loop (`tools.rs`), small models only.** When a
subagent's role model is *small* (`roles::is_small_model` — a param-size heuristic; see
`CORRODE_SMALL_MODELS` / `CORRODE_SMALL_MODEL_MAX_B`) and a Needle caller is present,
`run_tool_loop` runs it: each turn the model writes a plain-English `TOOL:` line, Needle
builds the call against `tools::TOOL_SCHEMAS`, `ToolBox` executes it and feeds the
observation back; the loop ends on a turn with no `TOOL:` line (the final answer) or
after `MAX_TOOL_STEPS`. Larger models (or a build without Needle) take the single-shot
path unchanged. Path args come through Needle cleanly (paths tokenize well); tool
*selection* sharpens with the planned finetune. `Daemon`'s `vfs` is `Arc<dyn Vfs>` so the
loop's `'static` future owns a clone.

Tools: `read_file`, `list_dir` (read-only) run straight through; `write_file`,
`run_command` are **mutating** (`tools::is_mutating`) and pass through a human
**approval gate** first (`approval.rs`). The loop emits `AgentEvent::ApprovalRequest`
and blocks that one call until an `AgentCommand::ApprovalResponse` arrives; it fails
CLOSED (denied) if the client is gone. For the response to be received while a Prompt
handler waits, `Daemon::run` now dispatches `Prompt` handling on a spawned task (other
commands stay inline/ordered). `run_command`'s cwd is the daemon's `repo_root`.

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
