Bound what the swarm's commands can take from the host, then run the deployed Corrode daemon with its graph store (`--features helix`) and make the swarm's context come from it. Follows docs/todo/2026-10-06-trust-and-speed-goal.md. Review: https://claude.ai/artifact/EcBAQEpbBsMHBrHtRUSHVx (#N = finding).

How to work each item:
- Read the code end to end, and make the smallest root-cause fix.
- Add a test that fails without the fix.
- Verify it live against hipfire. Store changes are measured on a CAE turn, before and after.
- Branch, push, PR, and merge on green CI. CI builds `--features helix,docling` (#57), so store tests run on every PR.

Throughout: keep planner thinking on, the ROCr fragment allocator off, and the memory watchdog running. Ask me only about user-facing behaviour changes (defaults, routing, refusals).

Host resources (#30)
1. Measure build contention. Decode tok/s for both swarm models, measured three ways:
   - idle;
   - during a cold CAE `cargo build` at `-j32` and at `-j8`;
   - at `-j8` under `nice`/`ionice`.

   The build must still be running when the sample ends. The last attempt finished early and measured nothing.
2. A spawn policy at `Sandbox::wrap`. Today `wrap` does nothing when the sandbox is off, and `CARGO_BUILD_JOBS` is unset, so every cargo call runs 32 jobs. Apply the policy whether or not the sandbox is on:
   - lower CPU and IO priority, or a `systemd-run --user` scope with `MemoryMax`/`TasksMax`;
   - `CARGO_BUILD_JOBS` and `RUST_TEST_THREADS` defaults in the spawned env;
   - a doctor check.

   Ask me before fixing the default caps. A/B against item 1's numbers.

Graph store (#43, #44, #46, #47)
3. Embeddings and rerank. No embedding model is served today, so skill ranking is off and doc search is BM25-only.
   - The models are in the local store, `~/.hipfire/models`. Never load them from `/srv`, an NFS mount from `carbon`. Copied there:
     - Qwen3-Embedding 0.6B, 4B and 8B (`--bf16.hfq`) and EmbeddingGemma-300M;
     - Qwen3-Reranker-0.6B (`--oq8.hfq`);
     - the 0.6B, 4B and 8B reranker sources (`.hfa`, for quantizing the larger ones).
   - Serve Qwen3-Embedding-0.6B, and set `CORRODE_RERANK_MODEL` to the 0.6B reranker.
   - Check that `/v1/embeddings` and `/v1/rerank` answer and that the daemon reports "ranked retrieval: true".
   - Then #47: read `CORRODE_EMBED_MODEL`; record `{model, dim}` in a `meta:embedding` node; on a mismatch, fall back to BM25 and log it instead of failing every DocQuery.
4. Node kinds (#44). `file_nodes` reads stored kinds back as `item`/`trivia`, so reconcile pairs nodes by position: inserting a `use` before `fn a; fn b` re-keys both functions. Return the stored kind. Test: every key still maps to the same text after an insert.
5. Batched `record_trace` (#46). It runs inline on tokio workers, one fsync'd LMDB transaction per node and edge, and each note supersedes every prior note on every file it touches.
   - One transaction per task, in `spawn_blocking`.
   - Dedupe; supersede only the newest prior note; skip `about` edges for read-only calls and empty paths.
   - Batch `persist_provenance` the same way.
   - Measure LMDB commits and turn time on a CAE turn, before and after.
6. Fresh from disk (#43). With `CORRODE_VFS_GRAPH`, ingest reads back through the graph, so a file freezes at its first ingest, and there is no delete.
   - Ingest from disk; store the blob oid and re-ingest paths whose oid changed; add `drop_file`.
   - Refuse GraphVfs without a freshness check.
   - Test: a file changed on disk behind the store is served fresh, and a deleted one is gone.

Run on helix
7. Deploy it. Build the daemon with `--features helix,docling`, run doctor, and run one unattended CAE turn with the store on. Compare it with the base-build turn (2026-10-07: 910 s, 86% prefix reuse, MemAvailable low 76 GB) on turn time, memory, store size and errors. Restart the standing daemon on it.
8. Context from the graph. The shared prefix is a shallow tree listing (the `ponytail:` on `context_prefix`). Replace it with relevance-ranked context from the store: embedding search over code nodes for the turn's request.
   - Keep it byte-identical within a turn, and budgeted like today's tree.
   - A/B on CAE: crate-map accuracy against the real crates, tool steps per task, and turn time.

Small
9. Web UI: with auth on, a reload replays nothing until the user signs in and acts. Send `ListTurns` after a successful `Authenticate` when no repo is selected.
10. hipfire: a non-streamed batched request's `timings` carry token counts but no rates or TTFT, because the batch runner's `DoneEvent` sets them to `None`. Fill them in.

Done:
- Everything merged with tests.
- The standing daemon runs `--features helix,docling` on merged code, with an embedding model served.
- One unattended CAE turn on it (sandbox and auto-approve on) completes:
  - files written, nothing hanging, memory flat;
  - the store agrees with disk afterwards: every written file's oid matches.
- Reported:
  - decode tok/s under a build, before and after the spawn policy;
  - LMDB commits per turn, before and after;
  - the context-prefix A/B;
  - changes, measurements and deferrals.

## Status (2026-10-07)

Every item is merged with tests:
- Corrode #63, #64 and #66–#70;
- hipfire #455–#457;
- #62 corrected item 3 to load models from the local store, not `/srv`.

Also done, at the user's request:
- **ROCm.** The host moved from a June ROCm 7.14 nightly to stable **ROCm 10.1.0** (the TheRock gfx1151 tarball); 7.14 is kept in `~/rocm-backup`.
  - The GPU gates gave identical results on 7.14 and 10.1. The full agentic gate fails one cell on both: Qwen3.5-35B-A3B `pi_clamped`, a model that doesn't serve the swarm.
  - Performance is equal or slightly better: A3B pp512 588 → 604 tok/s, 27B pp512 344 → 349, decode unchanged.
- **AMD agent skills** vendored (#65): `third_party/amd-skills`, all 12 in `.agents/skills` and `.claude/skills`.
- **Models.** The embedding and reranker models were copied into `~/.hipfire/models`.

### Done check

The standing daemon (7878) runs merged main (`fe8857e`) as a release `--features helix,docling` build, with the NPU embedder and the reranker served. One unattended CAE crate-map turn ran on it, sandbox and auto-approve on:

| | base build (previous goal) | helix deploy |
|---|---|---|
| turn time | 910 s | 1643 s |
| tool results | | 127 |
| errors | 0 | 0 |
| files written | `docs/crate-map.md` | `docs/crate-map.md` (15 crates), nothing else |
| prefix reuse | 86% | 73.6% over 58 sessions |
| MemAvailable | low 76 GB | 117.8 idle → low 74.1 → 74.5 GB after (flat once both models resident) |
| store | none | 101 MB, the whole repository (911 files ingested at session open in 2 s) |

Turn times on this prompt varied from 953 to 1643 s across today's runs.

The store agrees with disk: the recorded oids matched `git hash-object` for the written file and 20 random tracked files (21/21).

### Measurements

**Items 1–2, build contention.** Three cold CAE builds ran beside both swarm models. Decode tok/s; idle is 27B 14.8, 35B 58.5.

| | 27B | 35B |
|---|---|---|
| `-j32` (before) | 12.2 (−18%) | 46.9 (−20%) |
| spawn policy | 13.0 (−12%) | 57.7 (−1%) |

- The policy is `CARGO_BUILD_JOBS=8`, plus a 16 GB systemd scope with no swap; the defaults are the user's.
- One build alone costs 6% at `-j32` and 3% at `-j8`.
- nice/ionice changed nothing. The APU's GPU clock falls as the CPU draws power.

**Item 3, embeddings.**
- hipfire serves embeddings only on the NPU. The Qwen3-Embedding-0.6B source was downloaded (the user's choice), calibrated from activation statistics, and quantized `--npu-embedding oq8+`.
- Two hipfire fixes made it serve correctly: the LUT3-coded table loads (#455), and inputs end with the `<|endoftext|>` the model pools (#456). Cosine to the fp32 model is 0.9997.
- The 0.6B reranker serves on the GPU.

**Item 5, store write transactions per turn** (LMDB's last transaction id):

| turn | main | after #67/#68 |
|---|---|---|
| crate-map | 46 | 8 |
| audit | 1,278 | 9 |

On main, one audit task wrote 65 notes with 1,170 edges, each committed separately. After the change, each task's trace is one transaction. The audit run on the new build was stopped at about 80 minutes, longer than main's whole 57-minute turn.

**Item 8, context from the graph.** CAE crate-map turn, B (without) vs C (with):
- Both named 15 of 16 crates and invented none.
- Requests per task fell from **11.5 to 4.7**.
- The summaries followed each crate's own docs: B called `cae-thermal` "on the constraint graph", which its doc contradicts ("graph-free"). C quoted its modules.

**Item 10.** A non-streamed batched reply reports `tok_s`, `decode_tok_s` and `ttft_ms` (before: token counts only).

### Deferred
- **Item 8 used BM25 plus the cross-encoder, not embedding search.** Code nodes carry no vectors, and the NPU embedder (~300 tok/s) would take over an hour per CAE-sized repository. The section is cached per turn, so adding vectors later changes nothing else.
- **The sweep reads every held tracked file each turn** (`ponytail:`). Drive it from `git status` once repositories get large.
- **The initial ingest is not batched** (two transactions per file, one-time).
- **The EmbeddingGemma path still embeds unframed inputs.** There is no local source to verify it against.
- **hipfire's reported `usage.prompt_tokens` undercounts** embedding inputs.
- **The full agentic gate's `pi_clamped` cell** fails for Qwen3.5-35B-A3B, on 7.14 and 10.1 alike.
- **The `rocm` CLI** (which drives the vendored `rocm-doctor` skill) is not installed; it needs consent for a remote installer.
- **Item 9 was checked with the page's frame sequence, not in a browser.** Typing an auth token into the LAN-served page is outside what browser automation may do.
