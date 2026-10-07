Make a swarm turn take less time without making it worse. Follows docs/todo/2026-10-07-helix-goal.md.

How to work each item:
- Read the code end to end, and make the smallest root-cause fix.
- Add a test that fails without the fix.
- Verify it live: an unattended CAE turn before and after, and a tok/s A/B for any hipfire GPU change.
- Branch, push, PR, and merge on green CI.

Throughout: keep planner thinking on, the ROCr fragment allocator off, and the memory watchdog running. Ask me only about user-facing behaviour changes (defaults, routing, refusals).

Baseline: the deployed CAE crate-map turn on 2026-10-07. Standing helix daemon, sandbox and auto-approve on, telemetry `scratchpad/cae-deploy-telemetry.jsonl`.
- 1640 s for one short document.
- 948k input tokens against 14k output; 75% of the input came from cache.
- The two review tasks were 17 requests each (their whole 16-step budget plus the answer), 607k input tokens between them (64% of the turn). One review ran 567 s, a third of the wall time.
- Tasks summed to 1966 s against 1640 s of wall time, so little ran in parallel.
- An audit turn on the same daemon ran 57 to 80+ minutes, nearly all of it in tool loops.

Measure
1. A turn profile. For one CAE turn, the time per phase (planning, research, coding, review), and within each request the time to first token, decode time and cached/fresh input. Use the telemetry plus hipfire's `timings` (non-streamed requests now carry `ttft_ms` and `decode_tok_s`, hipfire #457). Write it as a table that later items compare against; add any field that is missing.

Review
2. Review from the diff. The plan review's digest lists each task's prompt, output and written files; the reviewer then re-reads them through tools, re-sending its growing conversation every step. Give it the diff of what the turn changed (and the outputs), so it checks rather than re-reads, and lower its step budget to match. Measure review input tokens and time, and check that it still catches a seeded defect (a task that writes a wrong fact).
3. Where review runs, and whether it runs. Ask me before changing either; both are defaults.
   - Review on the A3B model, which decodes 58 tok/s against the 27B's 15.
   - Skip review for a turn that changed no code (an answer, or docs only).

Tool loop
4. Cache reuse. Reuse fell from 86% to 74% this goal and the cause is not known. Within a tool loop each request should extend the previous one byte for byte, so only the new step is prefilled. Find where it doesn't (the prefix, the replayed steps, the tool declarations) and fix it. Target: 90% or more cached for every step after a loop's first.
5. Context growth. Every step replays every earlier tool output in full; `fit_context` trims only when the window overflows. Bound what a long loop re-sends: older outputs elided or summarised once the task has moved past them. Measure input tokens per step before and after on the audit turn.

hipfire
6. Batched decode. One request decoded at 43 tok/s through the batch runner against 58 through the streaming path (a single 64-token sample, A3B). Measure 1, 4 and 8 concurrent sessions batched vs streamed on both swarm models. If the gap holds, find the cause (per-step host work, scheduler cycle, sampling) and fix it, with a tok/s A/B and the GPU gates.
7. Parallelism. From item 1's profile, find what serialises a turn: the planner's dependency chains, a review that waits on everything, or one long task. Fix what is the swarm's to fix; report what is the model's.

Done:
- Everything merged with tests; both daemons restarted on merged code.
- The CAE crate-map turn completes in half the baseline time or less (820 s), with the same quality: 15/16 crates named, none invented, summaries that match each crate's docs, no other file changed.
- The audit turn completes, unattended, in under 30 minutes.
- A seeded defect is still caught by review.
- Reported: the turn profile before and after, per-item measurements, and deferrals.
