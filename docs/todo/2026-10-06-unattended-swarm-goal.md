Make Corrode (~/Corrode) and hipfire (~/hipfire) fit for multi-hour unattended swarms, working down this list in order. Review: https://claude.ai/artifact/EcBAQEpbBsMHBrHtRUSHVx (#N = finding). Per item: read the code end to end, smallest root-cause fix, a test that fails without it, verify (live against hipfire for runtime behaviour; end-to-end tok/s A/B for GPU changes), then branch, push, PR, merge on green CI. Keep planner thinking on, ROCr fragment allocator off, memory watchdog running during GPU/swarm runs. Ask me only about user-facing behaviour changes (defaults, routing, refusals).

hipfire
1. Wedged worker (#2): adapter progress deadlines + heartbeat, supervised batch runner, "wedged" on /health, respawn. Verify with SIGSTOP mid-request.
2. Typed errors (#3,#17): 400 context_length_exceeded (precheck), 404 unknown model, 503+Retry-After; a bad request fails only itself in a fused batch. Corrode: no retry on 4xx/context, jittered backoff on 503/429, retry a failed task once.
3. Fairness (#25,#35): aging in the batch scheduler; the other resident model preempts after a quantum. Test: two models x three bands all progress.
4. Host memory (#8-#11): admit on host MemAvailable as well as GTT, evict prefix checkpoints before refusing, OOM by error code, free GPU memory on every error path (Drop for session state, transactional load).
5. Auth (#21): /steer and /train behind the admin gate, api_gate deny-by-default, router-walking test.
6. Template/effort (#14, docs/todo/2026-10-02-stray-im-start-and-chat-template-check.md): 27B template override with a non-xhigh default effort, reasoning_effort in the render context; Corrode sends effort per role (planner bounded, still thinking); finish the stray <|im_start|> check for both models.
7. Serving-shape gate (#31): parity over the routes actually served (multicol <=16, BN=32, BN=64, default tile, MoE oq4.25++) without stopping the daemon.
8. A3B prefill: ~320 tok/s flat at any concurrency (64 x 640-token prompts took 135 s before decode began) -- profile the MoE prefill path and fix it.
9. Decode width 9-16 rows (docs/todo/2026-10-01-decode-width-opus-gemm.md): a decode-width kernel; A/B at 8-16 sessions.

Corrode
10. Sandbox that builds (#22,#23): read-only ~/.cargo and ~/.rustup binds, writable CARGO_HOME/target, explicit PATH; doctor checks cargo through wrap(); refuse malformed knobs; warn on AUTO_APPROVE without SANDBOX. Then ask me about sandbox default-on.
11. Turn liveness (#1,#24): watchdog, CORRODE_TASK_TIMEOUT_S, planning inside the budget, TurnComplete from a drop guard, CancelTurn/TurnStarted/plan_id on events.
12. Survive disconnects (#4): per-session event broadcast with replay, turn journal, ListTurns, web UI reattach.
13. Lost updates (#13): write_file refuses when the file changed since the task's last read.
14. Context budget (#36,#26): guard every generation incl. step 0 and the final answer, count tools, shrink old observations, prefix size budget, explicit per-request prefix instead of the 8-entry registry.
15. Observability (#41,#42): keep usage incl. cached tokens, one request id end to end, telemetry that reconstructs a run, per-worker /health gauges.
16. One tool loop (#37,#39,#40): one TaskCtx/step policy, tool capability metadata, hermetic fake-hipfire turn tests.
17. OpenAI client (#48, review s.6): backend seam (respond_turns, embeddings) with hipfire + OpenAI; route the hardest tasks by role/difficulty; per-run cost/rate/budget caps; fall back to local; secrets from env; hipfire KV-reuse request shape unchanged.

Done: all merged with tests, both daemons restarted on merged code, and one full unattended swarm turn on the CAE fixture (sandbox + AUTO_APPROVE on) completing with files written, nothing hanging, memory flat. Report changes, measurements, deferrals.

## Status (2026-10-06)

Items 1-7 and 10-17 are merged with tests: hipfire #428-#436, Corrode #31-#41.

Final check: both daemons were restarted on merged code (hipfire `c9c7eb16e`, Corrode `1b31e37`). One unattended CAE turn then ran with the sandbox on and auto-approve on, using the prompt "write docs/crate-map.md, one heading per crate":
- 1338 s, 9 tasks, 0 failed, 104 tool calls, no errors, no approval prompts.
- `docs/crate-map.md` written with all 15 crates; no other file changed.
- 994k prompt tokens over 68 requests, 800k (80%) of them from a cached prefix.
- MemAvailable 121 GB idle → about 78 GB with both models resident; low of 73 GB mid-turn, back to 77.5 GB after. The watchdog never fired.

Deferred:
- 8 (rest): the grouped MoE f32 GEMM is about 55% of A3B prefill and needs a faster bit-exact kernel. Prefill is 328 → 451 tok/s after #434.
- 9: a decode-width kernel for 9-16 rows (R&D).
- #31 extras: the fused-vs-serial cell, the spec oracle, and a KVarN smoke in the serving-shape gate.
- The web UI reconnect is type-checked only, not driven in a browser.
- The stray `<|im_start|>` in streamed replies (cleaning the stream) is still open.
- The remote backend counts streamed requests without their tokens, and never sends reasoning effort.

18. Fixed in hipfire #437; [system, user] on `qwen3.5:9b` went from 162 to 850 prompt tokens, and a 4-turn history from 162 to 1361. The 9B's plan now names the crate "mathkit", which it can only have read in the system turn. Not covered: the hook's GPU gates. On this host the coherence battery skips every case (none of its model files are present under the names it expects), and the agentic gate skips itself (no `mq4` A3B). The original finding: on hipfire's plain-scaffold path, every turn of `messages` is dropped except the last user message. That path is the default for Qwen3.5-family models without `jinja_chat: on`, and the fallback whenever a template fails to render; the separate `system` field still works. A [system, user] request counts the same tokens as user-only: on `qwen3.5:9b` (Corrode's offline `CORRODE_MODEL` default), 162 tokens versus 845 with the text merged. So system prompts, history, tool calls and tool results are lost silently. The configured swarm models carry `jinja_chat: on` and are unaffected. Fix: a messages-aware `ChatFrame::build_messages`, used at every plain render site in serving-core (about 12; see `generate.rs`, `generate_arch.rs`, `qwen35_prefill.rs`). The test: [system, user] must render like `system` + user.
