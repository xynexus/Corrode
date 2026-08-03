# TODO: Tool-call judgement (arguments, not syntax)

> **Status (2026-08-03): items 1, 2, and 4 shipped and measured; item 3 partially.**
> Live e2e with value constraints active: `.git/revisions` went from 10+ mentions to
> **zero** (grammar-unreachable, not corrected), 53 repeats absorbed by suppression,
> approvals 9 → 1, 0 errors. Item 5 remains deferred. One new bound landed alongside:
> emitted-task chains are capped per plan (`plan_graph::MAX_PLAN_TASKS`) after a
> measured runaway — once calls stopped dead-ending, every task emitted a follow-up
> and the chain never dried (65 emissions / 80 min on a one-line prompt).

**Goal:** make the swarm's tool calls *sensible*, not just well-formed. Syntax is now
solved — the model emits its native dialect and hipfire's grammar makes a malformed call
unreachable. What remains is that a structurally perfect call can still target a file
that doesn't exist or pass a Rust expression to a shell.

This is a harness problem, not a model problem. The model will not get better; the loop
around it can stop believing it.

---

## Evidence (observed, not assumed)

From a full-loop run against `fixtures/demo-repo` (MiniCPM5-1B.oq8++.coarse, native
dialect, 42 subagent outputs / 9 approvals / 0 errors / 360s):

| # | Observed | What it means |
|---|---|---|
| 1 | `list_dir(".git/revisions")` | invented a path that is not in the repo listing it was given |
| 2 | `run_command("is_prime(9999999)")` | passed a Rust expression to a shell |
| 3 | **the same failing `run_command` re-issued 3×** | the observation was fed back and *ignored* |
| 4 | several subagents each read `src/lib.rs` | no shared memory of what the swarm already learned |
| 5 | `NEXT: list_dir({"path":"…"})` | mixed the follow-up protocol with a call |

(3) is the load-bearing one. The loop already feeds the result back; the model does not
condition on it. Anything that relies on the model noticing its own failure is dead on
arrival — the fix has to be enforced by the harness.

## What NOT to do

- **More prompt engineering.** Measured brittle this session: identical wording changes
  flipped compliance, and `low`/`medium`/`high` reasoning produced byte-identical output.
  Do not spend here again.
- **The Needle finetune** (`finetune-needle-toolset.md`). Its premise — small models
  can't format calls — is falsified for models with a native dialect. It remains relevant
  only for models that have none.
- **A judge pass.** A 1B checking a 1B's judgement is not obviously better than the call
  it is checking, at double the cost.

---

## Scope, cheapest first

### 1. Repeat-call suppression — *small, deterministic, fixes (3)*

Keep a per-task set of `(tool, canonical args)` already executed. On a repeat, skip
execution and return the previous observation plus a note that it was already tried.

- Kills the 3× identical `run_command`, and the 3 approval prompts it burned.
- Also caps the runaway: that run spent 360s largely re-treading.
- Where: `run_native_tool_loop` / `run_tool_loop` in `daemon.rs`, next to
  `gate_and_execute`. ~20 lines plus a test.
- Risk: a legitimately repeated call (re-read a file after writing it) must not be
  suppressed — so key the set on args *and* invalidate on any successful mutating call.

### 2. Validate arguments before executing — *small, fixes (1)*

Check the call against repo state before running it, and return a corrective observation
instead of an exec failure:

- `read_file` / `list_dir` with a path not in the VFS → `error: no such path 'X'. Did you
  mean 'Y'?` with near-matches from the directory listing.
- `run_skill_script` already resolves by name; extend the same courtesy to paths.

Turns a hallucinated path into a *useful* observation rather than a raw errno. Where:
`tools.rs`, in front of `ToolBox::execute`. ~40 lines plus tests.

### 3. Route "verify" intents to skills, not raw shell — *small, fixes (2)*

`run_command` is the widest, sharpest tool we expose and the one it misuses. The fixture
ships a `run-tests` skill precisely so nothing has to invent `cargo test`.

- Prefer the skill in the tool set when one covers the intent; consider not exposing
  `run_command` at all to roles that have a skill for the job.
- Cheapest version: reorder/reword the tool descriptions so `run_skill_script` is the
  obvious choice for "check it works". No new machinery.

### 4. Constrain argument *values* in the grammar — *the real fix, medium*

The natural extension of the grammar work just landed. `ToolSchema` gains per-param
`allowed_values: Option<Vec<String>>`; `MiniCpmXmlGrammar` constrains a param body to
that set when present, exactly as it already constrains tool and param *names*.

Populate it per request from real state:
- path params ← the VFS listing already in the context prefix
- skill/script params ← `SkillContext::script_dirs`

Then `list_dir(".git/revisions")` is not merely corrected after the fact, it is
**unreachable** — the same guarantee we now have for syntax, applied to arguments.

- Where: `hipfire-runtime::tool_grammar` (schema + constraint), and the `tools` array
  Corrode sends must carry the value sets. hipfire currently derives `ToolSchema` from
  the OpenAI `parameters` block — `enum` is the natural carrier and is already standard
  JSON Schema, so no wire extension is needed.
- Risk: free-text params (`contents`, `command`) must stay unconstrained; only params
  with a closed, known set get this. Getting that wrong wedges generation.
- **Measured** (`cargo test -p hipfire-runtime --release -- --ignored --nocapture
  grammar_cost`, 129k vocab, after the per-position hoist):

  | candidates | per constrained token | per 20-token call |
  |---:|---:|---:|
  | 1 | 0.32 ms | 6 ms |
  | 4 | ~1.2 ms | 24 ms |
  | 16 | 4.38 ms | 88 ms |
  | 64 | ~20 ms | 400 ms |
  | 256 | 81 ms | 1.6 s |
  | 1024 | 320 ms | 6.4 s |

  Free positions cost 0.0008 ms, and 55% of a real call's tokens are constrained.

  **So the value set must be capped at ~64 candidates** (≈400 ms per call, tolerable);
  256+ is not viable. A repo listing longer than the cap should fall back to
  unconstrained and lean on items 2–3 instead — correction after the fact rather than
  a 1.6 s stall per call.

  The scan is still O(vocab × candidates). Bucketing candidates by first byte, or a trie
  over continuations, would raise the cap; do that only if a real value set exceeds 64.

### 5. Shared observation memory across the swarm — *larger, fixes (4)*

Subagents re-derive what siblings already learned because each has only its own
scratchpad. The plan graph already tracks task lineage and provenance; the same store
could expose "what has been read/run in this turn" so a task starts from the swarm's
knowledge, not its own blank scratchpad.

Defer until 1–4 land: it is the largest change and partly subsumed by (1) once the
suppression set is per-turn rather than per-task.

---

## Acceptance

Re-run the full-loop e2e against `fixtures/demo-repo` and require:

- [x] no tool call repeated identically within a task — suppression notes now visible
      as `ToolResult` events (53 in the measured run); approvals 9 → 1
- [x] no call executed against a path absent from the VFS — grammar-enforced: the
      invented path is unreachable at the token level (0 occurrences, 0 corrective
      observations needed)
- [ ] `cargo test` reached via the `run-tests` skill — the `target` enum now forces
      resolver-canonical `skill/script` pairs, but the model still often prefers
      `run_command` (free text by design); per-role tool subsets remain the lever
- [x] the turn settles — and is now *bounded*: `TurnComplete` ends the drain
      explicitly and `MAX_PLAN_TASKS` bounds the emission chain; raw step count
      trades against fanout/review depth by configuration

## References

- `crates/corrode-daemon/src/daemon.rs` — `run_native_tool_loop`, `gate_and_execute`
- `crates/corrode-daemon/src/tools.rs` — `ToolBox::execute`, `EXEC_TOOLS`
- `~/hipfire/crates/hipfire-runtime/src/tool_grammar.rs` — `ToolSchema`,
  `MiniCpmXmlGrammar`, and the `ToolGrammar` trait the value constraint extends
- `docs/todo/finetune-needle-toolset.md` — superseded for native-dialect models
