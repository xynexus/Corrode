Make Corrode (~/Corrode) and hipfire (~/hipfire) trustworthy to change and faster to run, working down this list in order. Follows docs/todo/2026-10-06-unattended-swarm-goal.md. Review: https://claude.ai/artifact/EcBAQEpbBsMHBrHtRUSHVx (#N = finding). Per item: read the code end to end; make the smallest root-cause fix; add a test that fails without it; verify it, live against hipfire for runtime behaviour and with an end-to-end tok/s A/B for GPU changes; then branch, push, PR, and merge on green CI. Keep planner thinking on, the ROCr fragment allocator off, and the memory watchdog running during GPU and swarm runs. Ask me only about user-facing behaviour changes (defaults, routing, refusals).

hipfire
1. GPU gates that run here. On this host the coherence battery skips every case and the agentic gate skips itself, because their model files are absent under the names they expect. Every hipfire commit therefore lands without GPU regression cover. Point the gates at models that are present: Qwen3.8-27B and Qwen3.6-35B-A3B oq4.25++, and the 9B. A gate that can run nothing must fail, not pass. Verify by injecting a known kernel bug and watching the gate catch it.
2. Warm vs cold determinism. A temperature-0 request to the 27B returns a different answer after server history (earlier sessions, checkpoints, speculation) than from a cold start; the A3B did not. Find the cause: forked checkpoint vs fresh prefill, batch composition, or n-gram speculation. Then fix it, or make the non-identical path an explicit, documented opt-out. Test: identical output warm and cold, both models.
3. A3B prefill (goal item 8, remainder). The grouped MoE f32 GEMM is about 55% of A3B prefill. Write a faster kernel behind a bit-exact parity gate, and A/B prefill tok/s at 1, 8 and 64 sessions. Current figure: 451 tok/s.
4. Control tokens in content (#16, quick win 9). Encode content specials literally, so `<|im_end|>`, `<tool_call>` or `<think>` inside a file or tool result never becomes a control id. Stop on token ids only, and never inside an open `<tool_call>`. End the last parameter at the last `</parameter>`. Test: write and read files containing each string through a live tool loop.
5. A batch runner testable without a GPU (#35, #17). An Engine trait plus a fake. A typed generation policy in SessionSpec, with the think budget enforced in batch decode (or routed to the legacy path). Port at least the park/resume, OOM retry and wedge cases to fake-engine tests.
6. One bad session fails only itself (#3, remainder). A non-overlong prefill failure still fails every request of its fused group. This needs per-session errors from the worker. Also: a length precheck when the prefix cache and row budget are both off, and a test for `reject_overlong`.
7. Port the deprecated daemon-socket tools to the HTTP API (docs/todo/2026-10-06-cli-tools-to-http-api.md in hipfire): `hipfire chat`, `hipfire bench` and the eval daemon executor. Then delete their socket paths and keep `refuse_unscoped`.
8. Residency and checkpoints (#28, #27, #34). Real recency in residency planning. Demote superseded PrefixIndex tails. A sealed-page floor. Measure GTT and prefix-hit rate over a CAE turn before and after.
9. Decode width 9-16 rows (goal item 9; hipfire docs/todo/2026-10-01-decode-width-opus-gemm.md). A decode-width kernel, A/B at 8-16 sessions.

Corrode
10. Neutralize protocol strings at the tool seam (#16, Corrode half). Add a reversible U+2060 neutralizer on observations, AGENTS.md rules, skills, digests and DocQuery excerpts. Strip it from `write_file`, `run_command` and `search_files` inputs. Log when an answer with unparsed `<tool_call` markup is accepted.
11. Clean streamed replies (goal item 6 remainder): the stray `<|im_start|>` in streamed subagent output.
12. Web UI reconnect, verified in a browser. It is only type-checked so far. Drop the socket mid-turn and confirm the console rebuilds from the replay.
13. Remote backend completeness. Count streamed requests' tokens (from `response.completed` or the final usage chunk). Send a reasoning effort to remote models that accept one, behind a per-endpoint knob.
14. Scheduling bands from graph position (#25, Corrode half). A task on the critical path of a plan should not wait behind speculative work of the same role.
15. A `--features helix,docling` CI job (§8), so the graph-store and ingestion code compiles and tests on every PR.
16. Graph store (#43, #44, #46, #47). These matter once the daemon runs with `--features helix`; ask before starting:
   - batched `record_trace`;
   - disk-sourced ingest with oid freshness;
   - the node-kind fix;
   - embedding metadata.

Done: all merged with tests; both daemons restarted on merged code; GPU gates proven to catch an injected bug; one full unattended CAE swarm turn (sandbox + AUTO_APPROVE on) completing with files written, nothing hanging, memory flat, and its temperature-0 planner output identical warm and cold. Report changes, measurements and deferrals.

## Status (2026-10-07)

Items 1-8 and 10-15 are merged with tests: hipfire #444-#453, Corrode #53-#58. Corrode #59 is also merged, found while driving item 12 live. Asked to "explain this crate, do not change any file", the swarm rewrote `is_prime`: only the planner had seen the request. Every task now sees it, and a `NEXT: none` no longer queues a task.

Final check: both daemons were restarted on merged code (hipfire `4d2e18f73`, Corrode `af66ac6`). One unattended CAE crate-map turn then ran with the sandbox on and auto-approve on:
- 910 s, 52 subagent outputs, 76 tool calls, no errors, nothing left running.
- `docs/crate-map.md` written with all 15 crates; no other file changed.
- 748k prompt tokens over 53 sessions, 86.2% of them from a cached prefix. The prefix cache had 51 hits in 58 lookups, demoted 45 superseded tails and made 31 evictions.
- MemAvailable 118 GB idle. Both models (34.7 GB) loaded by 400 s, and GTT rose 37.5 → 42.9 GiB as KV and checkpoints filled. It then held level for the last 4 minutes: a low of 76.0 GB, 76.3 GB after. The watchdog never fired.
- Planner, temperature 0, run alone (`hipfire_deterministic`): the turn's cold plan and two warm replays after the turn are identical (2446 chars).

Measurements:
- 1: an injected overlay-correction bug fails the serving-shape gate and the coherence battery, and the pre-commit hook blocks the commit. Clean master passes.
- 2: the cause is batch composition, not prefix reuse. Deterministic requests run alone, and the planner's calls are deterministic.
- 3: A3B prefill tok/s, master → #446:

  | sessions | master | #446 | change |
  |---|---|---|---|
  | 1 | 900 | 948 | +5% |
  | 8 | 451 | 469 | +4% |
  | 64 | 642 | 736 | +15% |

- 7: `hipfire bench` through serve measures pp512 at 512.8 t/s and tg128 at 59.7 t/s. The eval smoke and speed batteries pass 5/5.
- 8: over a CAE turn, before → after:
  - prompt tokens reused: 87.0% → 86.2%
  - GTT max: 41.7 → 43.0 GiB
  - MemAvailable min: 76.1 → 74.9 GB

  Neutral on this workload, which has fewer than 4 live tails per worker.
- 11: at effort `none` the 27B's streamed answer used to arrive entirely as reasoning; now it arrives as the answer.
- 12: in Chrome, restarting corrode-web mid-turn rebuilt the console from the replay and live events resumed (after #58; before it, nothing replayed).

Deferred:
- 9: decode width 9-16 rows is still open. Four lanes per group (64 weights per lane) halves the epilogue per MAC but was slower on 3 of 4 shapes, because the doubled activation tile halves occupancy. The table is in hipfire's `docs/todo/2026-10-01-decode-width-opus-gemm.md` (#454).
- 16: the graph store, deferred by decision until the daemon runs with `--features helix`.
- Read-only turns enforced by the harness: decided against; #59's prompt-level fix stands.
- hipfire: on the batched path a non-streamed request's `timings` carry token counts only, no rates. The tools stream to get them.
- Web UI: a reload with auth on replays nothing until the user signs in and acts. Reconnects, and reloads with auth off, replay at once.
