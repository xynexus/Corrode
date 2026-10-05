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
