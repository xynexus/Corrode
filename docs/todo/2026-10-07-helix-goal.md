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
3. Embeddings. No embedding model is served today, so skill ranking is off and doc search is BM25-only.
   - Quantize Qwen3-Embedding-0.6B from the local weights in `/srv/huggingface` (no download), following hipfire `docs/QUANTIZE.md`. Use whichever path hipfire serves it on this host (GPU or NPU).
   - Check that `/v1/embeddings` answers and that the daemon reports "ranked retrieval: true".
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
