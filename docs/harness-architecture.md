# Corrode harness architecture

**Status: aspirational.** This is a target, not a description. What exists today is
the swarm (`plan_graph` reactive scheduling, role→model routing, priority bands,
`CORRODE_FANOUT` ensembles, the plan-review pass), the tool loop behind an approval
gate, and a filesystem `PassthroughVfs`; project scoping (`project.rs`) is on
`feat/project-awareness` and not yet on `main`. What does not exist: the graph
write path (`graph.rs` stubs bail), telemetry, sandboxing, cancellation, and
budgets. Sections below mark which is which.

Companion documents: `graph-model.md` (the graph the VFS projects from) and
`todo/`. This one is about the harness *around* the models.

---

## 1. Thesis

A coding harness is not a prompt with tools bolted on. It is a system that owns
**cognition state** — what is being attempted, what has been tried, what is
believed, and on what evidence — and treats model invocations as stateless compute
against that state. The token window becomes a cache over durable working state
rather than the place the agent lives.

That much is common ground in every harness design. The part that is not common
ground, and the reason this document exists, is **ordering**. Harness designs tend
to enumerate capabilities as coequal aspirations: semantic code graphs, learned
tool-use policies, cross-project memory, speculative execution. Every one of those
sits on a foundation, and when the foundation is missing they do not degrade
gracefully — they fail *confidently*, which is worse.

The ordering in §6 is not a roadmap of convenience. It is derived from measured
failures, recorded in §2.

## 2. What we measured (2026-08-30)

Every claim in this document traces to an observation from one session driving the
swarm against three repositories. They are recorded here because a design argument
without evidence is a preference.

**The swarm explained the wrong repository.** Pointed at `~/stitch` (a C++
lock-free threading library), six subagents across two turns described *hipfire* —
a Rust GPU inference engine — including MoE expert paging and HIP kernels. Zero
mentioned C++. The cause was not the model. Dumping the literal shared prefix
showed 4,106 bytes of which ~3,700 were skill descriptions from `~/.agents/skills`,
installed for other projects, presented under a header asserting they were
*"Relevant skills for this task"*. The repository itself was twelve filenames with
byte counts. The model summarized its input correctly.

**Relevance ranking did not rank.** The eight "relevant" skills scored
`0.318, 0.301, 0.301, 0.287, 0.286, 0.281, 0.270, 0.261` — a 0.057 spread, which is
noise. The top scorer was *below* the 0.35 bar that gates full-body injection: the
system judged nothing relevant enough to activate, then listed all eight as relevant
anyway, because stage 1 had no threshold at all.

**Independent review caught it and could not fix it.** The review-role subagent
correctly wrote: *"the explanation for the reactive plan graph and role-to-model
mapping is fabricated… I cannot verify these concepts from the available context, so
stating them as fact is a defect."* It then emitted a corrective task, which
produced the same wrong answer — because it inherited the same prefix.

**Served configurations are not interchangeable.** Six served models, one Python
task (`merge_intervals`), generated code executed against six cases:

| served artifact | decode t/s | code |
|---|---|---|
| `Qwen3.5--0.8b-oq4++` | 67.3 | ✗ repetition loop, never emits code |
| `MiniCPM5--1B.oq4.25++` | 52.5 | ✗ wrong (`[[1,3]] -> []`) |
| `ZAYA1--8b.oq4++` | 28.1 | ✗ broke off mid-expression |
| `Qwen3.6--35B-A3B.oq4.25++` | 23.6 | ✓ |
| `Qwen3.5-9B--oq4.25++` | 10.5 | ✓ |
| `Qwen3.8-27B--oq4.25++` | 4.7 | ✓ |

**These are measurements of the serving stack, not verdicts on the models.** Each row
is a (weights, quantization, decode path) triple, and a failure can originate
anywhere in it. Attributing the left column to model capability would be a category
error, and in at least one case demonstrably wrong: ZAYA1 stopped mid-expression at
`if not intervals(` and then emitted its closing fence, which reads as a decode or
tokenizer defect rather than weak coding — and it is precisely the artifact whose
quantization is unverified, because its tiny-quant KLD cells were left unrecorded
(hipfire `BUGS.md`). Separately, the serving layer reported `finish_reason: "stop"`
on runs truncated exactly at `max_tokens`, so a model that never terminated was
scored as finishing cleanly until its output was compiled.

What the table does establish: **throughput does not predict usefulness, and the
ordering is not guessable.** The three fastest configurations all fail; the fastest
usable one is the largest artifact on disk, because only 3.4B of its 34.7B params
are active. That is an argument for measuring what you deploy — not for ranking
model families.

**Tool access is decided by a regex on the model name.** `roles::is_small_model`
reads the largest `<n>b` marker in the id; its own test asserts
`is_small_model("Gemma-3-27B") == true`. A 27B gets tools; the 35B that passes the
benchmark does not, and was observed emitting `<tool_call>` blocks into a void that
nothing reads.

**Nothing is sandboxed.** `run_command` and `run_skill_script` execute
model-generated shell directly on the host with the user's privileges. `bwrap` is
installed and invoked nowhere — not in corrode, not in hipfire, not in any unit or
wrapper. The `Vfs` escape guard covers `read_file`/`write_file`; `run_command`
bypasses the VFS entirely, and its `repo_root` cwd is not a boundary.

**No telemetry existed to notice any of this.** The benchmark above and the
correctness harness were written by hand for one session and thrown away.

## 3. Principles

These are load-bearing. Each one is the generalization of a failure in §2.

### 3.1 Context carries provenance and authority

Every object entering a prompt is tagged with where it came from and how far it may
be trusted, and that tag is *visible in the prompt*:

```
repo file > README / AGENTS.md > project config > general capability > model prior
```

The stitch failure was entirely a failure of this rule. A home-directory skill and a
repository fact were rendered identically, so the model weighted them by size and
confidence instead of authority. "Relevant skills for this task" is an authority
claim the harness had no basis to make.

Corollary: a header may never assert relevance the ranker has not established. If
the score does not clear a bar, the section is labelled *available capabilities*, or
it is omitted.

### 3.2 Context is layered most-stable first

hipfire's prefix cache is **not** an exact-match cache on the whole prompt. It
matches the longest common prefix: `prefix1`, `prefix1+prefix2` and
`prefix1+prefix3` all reuse `prefix1`'s prefill. The server-side implementation is
`generate_arch.rs`'s LCP detection, and the intended client pattern is stated there
outright — construct prompts as `immutable_prefix + append_only_log`. There is also
an explicit path: `prefix_hash_preflight` lets a caller declare a prefix for reuse
rather than relying on incidental matching.

This dissolves the prefix-versus-tail dichotomy. There is no binary choice between
"shared" and "per-agent" content; there is an **ordering** of layers from most to
least stable, and every layer reuses everything above it:

```
L0  harness preamble        stable across every project and turn
L1  project identity        README digest, repo tree, AGENTS.md rules
L2  turn                    the plan, the objective, the shared digest
L3  role / branch           reviewer vs builder vs investigator
L4  task tail               retrieved context, working memory, observations
```

A builder and a critic that diverge at L3 still share the L0–L2 prefill. Two turns
on the same project share L0–L1. The design rule is therefore not "keep the prefix
small" but **order context by volatility and never put a volatile fact above a
stable one** — a single early-varying byte forfeits every layer below it.

**Layers are cheap; be generous.** A prefix is prefilled once per model and reused
by every prompt that shares it. For the workloads Corrode targets — one repository,
the same project facts, hundreds of turns — an L1 layer is amortized across hundreds
of prompts, not merely across one fan-out. Trimming the README digest to save prompt
budget optimizes the cheapest thing in the system. The scarce resource is L4.

**But the number of boundaries is bounded, and this is the real constraint.**
Qwen3.5/3.6 is 48 of 64 layers *linear* attention, whose recurrent state is not
positionally truncatable the way KV is. Reuse across a boundary requires a
checkpoint captured *at* that boundary, and checkpoints cost VRAM
(`resident_checkpoint_max = 4` today). So the layer stack above is not free to
extend indefinitely: a handful of well-chosen boundaries is the budget. Choose them
where branching actually happens — project, turn, role — not at every semantic
seam that looks tidy.

**Status: this does not work yet.** The cross-session prefix state cache — the one
the swarm's fan-out needs — is fully built and never engages, on two hardcoded
values (hipfire `docs/bugs/2026-08-30-prefix-state-cache-never-engages.md`, filed
2026-08-30, ~1–2 days, the wiring being the easy half). Measured today: every
request reports `cached_tokens: 0`. The multi-turn LCP path within one session does
work. Corrode should be built for the layered model regardless — the fix is
hipfire-side and tracked — but no measurement taken before it lands says anything
about the strategy, and no claim of amortization should be made until
`cached_tokens` is non-zero.

**Implication for `context_prefix`.** It currently builds one monolithic `String`.
To exploit any of this it must become an ordered list of layers with declared
boundaries, so the harness can branch at a chosen level and, once
`prefix_hash_preflight` has a caller, declare each layer explicitly rather than
hoping the tokenizer produces a byte-identical match.

### 3.3 Independence is a property of context, not of agent count

Three critics reading one prefix are one critic with three samples. Six subagents
converged on "this is hipfire" because they shared 4,106 bytes of wrong context;
that is prefix sharing working as designed, producing perfectly correlated failure.

Therefore: **every verification round branches above the builder's reasoning.**
Layering (§3.2) makes this precise and nearly free — independence is a choice of
*branch point*, not a decision to discard reuse. A critic that forks at L3 still
shares the harness preamble, project facts and objective (L0–L2, already prefilled)
while inheriting none of the builder's plan, intermediate conclusions or retrieved
evidence. It gets the requirement and the diff. Cheap enough to do on every round.

The corollary runs the other way and is the sharper half: **whatever is shared is
shared by everyone below it, including errors.** The stitch failure was not agents
being individually wrong; it was one bad L1 layer inherited identically by six
agents, none of which could see out of it — and the reviewer, which correctly
identified the output as fabricated, could not repair it because it read the same
layer. Layers high in the stack must therefore be *verified* facts (§3.1), and the
higher a fact sits, the higher the bar to put it there. Cheap reuse of an unverified
fact is cheap propagation of a defect.

Ensembles that share a prefix (`run_fanout`) are a latency optimization for
*generation*; they are not evidence for *verification*.

### 3.4 Capability is measured, not parsed

Routing decides which model gets which role and which tools. That decision is made
from a recorded capability profile — pass rate on a task battery, tokens/sec,
termination behaviour, tool-call validity — refreshed when the model set changes.
Never from a substring of the model id.

The profile describes a **served configuration**, not a model: the same weights at a
different quantization, or through a different decode path, is a different entry.
This is the deeper reason a name cannot be parsed for capability — the name does not
mention the two things most likely to have broken it. It also means a failing profile
is a bug report against the serving stack first and the model second, and should be
triaged that way rather than written off as "that model is weak" (§2).

Until a profile exists, the safe default is inverted from today's: a model with no
recorded profile gets tools and a small budget, not tools withheld. An agent that
can read is recoverable; an agent that must guess is not.

### 3.5 Observations are structured at the source

A tool result is a typed observation, not a transcript. The structure comes from the
tool where the tool offers it — `cargo --message-format=json` already yields the
diagnostic tree, so "17 failures, 14 descendants of one root, 3 independent" is
*parsing*, not summarization. Never spend a model call compressing output that the
producing program will hand over structured.

Full output remains addressable for drill-down; the digest is what enters context.

### 3.6 Every turn has a budget

A turn declares a ceiling — tokens, wall-clock, or both — before it starts. The
orchestrator schedules against it and sheds speculative branches when it tightens.
One observed turn on the 35B took 375s with no bound and no way to ask for a cheaper
answer. hipfire's priority bands schedule the GPU; they do not cap the swarm's
appetite, and admission control cannot see intent the harness never expressed.

### 3.7 Reversibility precedes autonomy

Authority is capability-based and tiered — read the repo, write a worktree, run a
build, install a package, touch credentials, push, deploy. Each tier is a distinct
grant, not one approval prompt. Speculative execution (§5.3) is only sound when
abandoning a branch is free, which means worktrees and snapshots come *before* the
orchestrator is allowed to fan out writes.

## 4. Structure

Three planes. The split matters because they have different persistence and
different trust.

**Knowledge plane** — durable, addressable, provenanced. Repository graph, README
and doc digests, task state, agent working memory, experiment history, capability
profiles. Owned by the harness, survives model replacement and context compression.
`graph-model.md` describes the code half; the rest shares the store so that a task,
the code it touched, and the evidence it saw are one traversal apart.

**Execution plane** — the real engineering loop: filesystem, shell, build,
formatters, linters, tests, debugger, profiler, git, CI. Every action sandboxed and
capability-gated (§3.7), every result a structured observation (§3.5).

**Supervision plane** — what a human sees and steers: the live task graph, active
agents, hypotheses, proposed diffs, test status, spend against budget, and
unresolved uncertainty. Autonomy without this is opacity.

Work units, context objects, and agent state are **immutable and addressable**. An
agent trajectory is then itself a graph: forking an investigation is copy-on-write,
a reviewer can be handed exactly the evidence a builder saw (or deliberately denied
it, per §3.3), identical prefixes are cacheable by construction, failed branches
stay searchable, and successful ones are extractable as training data.

## 5. Roles

### 5.1 Orchestration

Decompose into a DAG, launch ready work, absorb dynamically discovered tasks,
**cancel obsolete branches**, merge results. The first three exist today in
`plan_graph::run_reactive`; cancellation does not, and is the real gap — `run_reactive`
launches and waits, so a branch invalidated by a sibling's result runs to completion
and spends its budget anyway.

### 5.2 Build and verify

`Builder → Reviewer → Tester → Fixer`, with §3.3 binding: the reviewer is given the
requirement and the diff, not the builder's reasoning. Verification attempts
*falsification* — compile, existing tests, generated tests, type check, static
analysis, sanitizers where they apply, benchmark regression — and a claim is
accepted on evidence, not on agreement.

### 5.3 Speculation

Competing solutions, not merely parallel subtasks: three independent attempts at a
hard change and a fourth agent comparing conclusions beats one long trajectory.
Requires §3.7 (cheap abandonment) and §3.6 (a budget to spend), and the attempts
must diverge in more than sampling — an identical prompt at temperature 0 returns an
identical answer, which was observed when an emitted retry reproduced its
predecessor byte for byte.

## 6. Order of work

Derived from §2. Each rung is cheap relative to the one after it, and each removes a
class of confident failure.

1. **Ground the prefix, and layer it.** README digest, a repository tree deeper than
   one level, provenance/authority labels (§3.1) — emitted as ordered layers with
   declared boundaries rather than one string (§3.2). Hours. Prevents every §2
   grounding failure, and the content is amortized across hundreds of turns once the
   prefix cache engages. Be generous with L0–L2; frugal with L4.
2. **Give capable models tools.** Decouple the tool loop from `is_small_model`
   (`daemon.rs`). Hours. A model that can read stops guessing; this dominates any
   amount of static-context tuning.
3. **Structured observations** from `cargo --message-format=json` (§3.5). A day.
4. **Telemetry.** Model, context composition, retrieved objects, tool calls, tokens,
   wall-clock, patch, test result, outcome. Nothing after this point can be tuned
   without it, and it makes every later claim falsifiable.
5. **Sandbox and capability tiers** (§3.7) — before autonomy widens, not after.
6. **Cancellation and budgets** (§3.6, §5.1).
7. **The graph** — and first measure whether the embedder can discriminate at all.
   Given the 0.057 spread in §2, graph retrieval may inherit exactly the failure that
   sank skill ranking, with more machinery in the way. That measurement is a day and
   decides whether step 7 is retrieval work or embedding work.

The temptation is to start at 7. Step 7 is the most interesting and the most
defensible on paper. It is also the one whose payoff is bounded by every step above
it, and the system currently cannot name the repository it is working in.

## 7. Non-goals

- **A local scheduler.** Priority bands express intent; hipfire owns admission and
  batching. Do not build queueing or throttling here.
- **Conversational memory.** The knowledge plane stores task state and evidence, not
  dialogue history.
- **Model-agnostic prompting.** Dialects already exist because models differ; §3.4
  extends that to capability. Pretending otherwise is how a 0.8B model gets a coding
  task.
- **Autonomy ahead of auditability.** Any capability that cannot report what it did
  and why is not shipped, regardless of how well it demos.

## 8. Open questions

- **Does the embedding model discriminate?** Everything retrieval-shaped depends on
  it and it has failed once. Measure before building on it.
- **Where do the boundaries go?** Checkpoint residency is bounded
  (`resident_checkpoint_max = 4`), so the layer stack in §3.2 spends most of its
  budget immediately. Whether project/turn/role are the right three, and what the
  marginal boundary buys, is unmeasured.
- **How large can a layer usefully get?** Prefill amortizes, so the binding limit is
  context window and attention dilution, not cost. Where added grounding stops
  helping — or starts hurting — is unmeasured.
- **What belongs in the tail versus a tool call?** Retrieved context is paid per
  call whether the harness pushes it or the agent pulls it, but pulling costs a round
  trip and pushing costs relevance guesses. Untested.
- **What is the unit of provenance for a partial edit?** A file node produced by a
  task is clear; a hunk authored by one agent and revised by another is not.
- **How much context is too much?** No measurement exists relating context size to
  success rate. Telemetry (step 4) is what turns this from taste into data.
