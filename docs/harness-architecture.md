# Corrode harness architecture

**Status: aspirational.** This is a target, not a description. What exists today is
the swarm (`plan_graph` reactive scheduling, role→model routing, priority bands,
`CORRODE_FANOUT` ensembles, the plan-review pass), the tool loop behind an approval
gate, a filesystem `PassthroughVfs`, **bubblewrap sandboxing** (`sandbox.rs`), and a
**live provenance write path** — `upsert_node`/`add_edge` are real LMDB write
transactions under `--features helix`, not stubs. Project scoping (`project.rs`),
telemetry, structured observations, budgets and cancellation are in review. Sections
below mark which is which.

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

**Nothing is sandboxed** *(fixed since — see the note at the end of this section)*. `run_command` and `run_skill_script` execute
model-generated shell directly on the host with the user's privileges. `bwrap` is
installed and invoked nowhere — not in corrode, not in hipfire, not in any unit or
wrapper. The `Vfs` escape guard covers `read_file`/`write_file`; `run_command`
bypasses the VFS entirely, and its `repo_root` cwd is not a boundary.

**The embedder discriminates; the ranker was right.** The 0.057 band above looks
like an embedding failure, and was recorded as one hypothesis. It is not.
Embedding stitch's 16 real headers as `filename: \brief` and querying with known
answers gives **4/6 top-1** — 5/6 counting `hazard_pointers.cpp` beating its own
`.h` by 0.008 — and a **mean top-8 spread of 0.250**, 4.4x the skill band. The one
real miss is inside the queue family, whose variants differ by three letters
(`queue_spsc_waitfree.h` vs `queue_mpmc_lockfree.h`).

So the tight skill band was the ranker *correctly* reporting "nothing here matches"
a repo-identity query, and the harness listed the results anyway. The defect was
never retrieval quality; it was a header asserting a relevance the ranker had not
established. This is the single most important correction in this document, because
the opposite conclusion would have sent the next month into embedding work.

**No telemetry existed to notice any of this.** The benchmark above and the
correctness harness were written by hand for one session and thrown away.

**What has since changed.** This section is a dated record and is left as measured,
but two of its findings no longer hold: `sandbox.rs` now wraps every spawned process
in an unprivileged `bwrap` namespace — repo read-write, graph store read-only, rest
of the filesystem read-only, no network by default — and fails closed rather than
dropping silently to unsandboxed when `bwrap` is unavailable. Telemetry is in review.
The rest of the section stands.

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
it is omitted. A low spread across candidates is itself the signal — it is the
ranker reporting that nothing matches — and overriding it is how an unrelated
project's skills came to be presented as this repository's context.

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

Half of this exists. `sandbox.rs` provides *confinement* — a `bwrap` namespace per
spawned process, failing closed — but not *tiering*: approval is still one yes/no per
mutating call, and `CORRODE_AUTO_APPROVE` removes even that for unattended runs. The
two interact in the direction that matters: with auto-approve on, the sandbox is the
only boundary left, so it stops being defence in depth and becomes the whole defence.
Tiered grants are what would let an unattended swarm keep *some* authority without
being handed all of it.

## 4. Structure

Three planes. The split matters because they have different persistence and
different trust.

**Knowledge plane** — durable, addressable, provenanced. Repository graph, README
and doc digests, task state, agent working memory, experiment history, capability
profiles. Owned by the harness, survives model replacement and context compression.
`graph-model.md` describes the code half; the rest shares the store so that a task,
the code it touched, and the evidence it saw are one traversal apart.

**Execution plane** — the real engineering loop: filesystem, shell, build,
formatters, linters, tests, debugger, profiler, git, CI. Every action sandboxed
(`sandbox.rs`, real) and capability-gated (§3.7, still one yes/no), every result a
structured observation (§3.5).

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

**Status, 2026-09-02.** Steps 1–4 and 6 are implemented; step 5 shipped its sandbox
half and not its capability-tier half; step 7 has been the whole of the work since
2026-08-30 and has turned out to be a sub-project rather than a rung — it is broken out
below. Marked per step.

One caveat that the earlier status hid: **step 1 is shipped and its stated payoff is
not.** Its justification is that layered grounding "is amortized across hundreds of
turns once the prefix cache engages", and §8 records that assumption as **FALSE today** —
`cached_tokens` has never been observed above zero. The layering is built and correct;
what it was supposed to buy is still unavailable, and that is a hipfire investigation,
not a corrode one.

1. **[shipped]** **Ground the prefix, and layer it.** README digest, a repository tree deeper than
   one level, provenance/authority labels (§3.1) — emitted as ordered layers with
   declared boundaries rather than one string (§3.2). Hours. Prevents every §2
   grounding failure, and the content is amortized across hundreds of turns once the
   prefix cache engages. Be generous with L0–L2; frugal with L4.
2. **[shipped]** **Give capable models tools.** Decouple the tool loop from `is_small_model`
   (`daemon.rs`). Hours. A model that can read stops guessing; this dominates any
   amount of static-context tuning.
3. **[shipped]** **Structured observations** from `cargo --message-format=json` (§3.5). A day.
4. **[shipped]** **Telemetry.** Model, context composition, retrieved objects, tool calls, tokens,
   wall-clock, patch, test result, outcome. Nothing after this point can be tuned
   without it, and it makes every later claim falsifiable.
5. **[shipped, partially]** **Sandbox and capability tiers** (§3.7). `sandbox.rs`
   confines every spawned process — the agent's `run_command`/`run_skill_script` and
   the web terminal alike — in an unprivileged `bwrap` user namespace: repo
   read-write, graph store read-only, everything else read-only, no network unless
   `CORRODE_SANDBOX_NET` says so. Off by default (`CORRODE_SANDBOX`) so existing
   behaviour is unchanged, and it fails CLOSED: if `bwrap` cannot run, the command
   does not either. The *capability tiers* half is still open — approval remains one
   yes/no per mutating call, not a graded grant, and `CORRODE_AUTO_APPROVE` can
   disable even that for unattended runs, which is precisely the configuration that
   makes the sandbox load-bearing rather than belt-and-braces.
6. **[shipped, partially]** **Cancellation and budgets** (§3.6, §5.1). A turn
   declares a wall-clock ceiling; past it nothing new launches, emissions are
   dropped, and a tool loop stops at its next step boundary. Not preemption — a task
   inside one long model call still finishes, so a turn can overrun by one call.
7. **[in progress]** **The graph.** The gating measurement has been taken (§2): the embedder separates
   real matches by 0.250 on average, so graph retrieval does **not** inherit the
   failure that sank skill ranking. Step 7 is retrieval-structure work, not embedding
   work. What remains unproven is representation — the one real miss was between
   near-identical variants in a family, which is precisely the shape a code graph is
   full of (`foo` vs `foo_batched`, `spsc` vs `mpmc`). Structure is what disambiguates
   those; a description alone does not.

   Written as one rung it read as one session's work. It is not, and pretending
   otherwise is how the plan stopped describing the work. Its actual parts:

   - **7a [shipped] Ingest.** Source becomes nodes, language-agnostically: a
     `Language` seam with Rust (`syn`), C/C++ (lexer) and a marker-family text
     fallback, comments extracted as a separate pass and bound to what they describe
     by edge, a sparse `u64` order key, and tar-stream ingest that never unpacks. The
     kernel: 94,750 files, 1.6 GB, 7.2 s, byte-exact.
   - **7b [shipped] Reconcile.** A re-ingest diffs against stored nodes so survivors
     keep their keys, rather than renumbering the file. Measured over 5,000 curl
     commits: 19% of mutations are inserts, 0 rebalances. Wired into the daemon's
     ingest path via `ingest::file_against`, which reads the stored nodes back before
     writing — without that read-back `reconcile` was correct, tested and never
     called, and the sparse key bought nothing in production.
   - **7c [shipped] Persist.** `replace_file` writes the file/code/comment nodes and
     their edges atomically, with pruning. Until this landed the whole pipeline wrote
     to nothing.
   - **7d [first cut] Retrieve.** `code_search` (BM25 over source) with line numbers
     *derived* from the node cover, appended to `search_files`' literal scan.
   - **7e [measured, negative] Represent.** Graph structure does **not** separate
     near-identical siblings. Measured below: every representation drawn from the
     file's own bytes scores 1/4 — chance. What works, partially, is a generated
     description, which is a generation problem at ingest time rather than a graph
     one.
   - **7f [first cut] Project.** `graphvfs::GraphVfs` composes reads from graph nodes
     and falls through to the filesystem for anything the graph does not hold, behind
     `CORRODE_VFS_GRAPH` (off by default). Enumeration deliberately stays on the inner
     VFS. The staleness risk is pinned by a test rather than papered over.

8. **[diagnosed]** **Store throughput.** Not in the original ordering because nothing
   had measured it. Profiled below: **the cost is text, not nodes** — writing the same
   node count with the text removed is 7.4x faster. The cause is that helix
   BM25-indexes every property on every node write, with no field selection, and our
   `label` is the verbatim text. Fine for a repo (curl, 58k nodes, ~30 s); infeasible
   for a kernel-sized tree (13.8M nodes, ~1.8 h, ~23 GB). Also unfixed: a single token
   over LMDB's 511-byte max key (base64, minified JS) rejects a whole node write.

What steps 1-4 and 6 cost, for calibration: roughly one working session each,
several of them under an hour, and every one of them removed a failure that had
already been observed. That is the argument for the ordering restated as a
measurement — the foundation was not expensive, it was merely unglamorous.

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

## 8. Load-bearing assumptions

An open question costs a decision. A load-bearing assumption costs a *rewrite* — it is
a claim that, if false, invalidates work already built on it. They are tracked
separately for that reason, and each one names the cheapest experiment that would
falsify it.

**The rule: dependent work does not start while its assumption is UNTESTED.**

| assumption | status | falsifying test | blocks |
|---|---|---|---|
| A shared prefix is prefilled once and reused | **FALSE today** | `cached_tokens > 0` on a repeated prefix | §3.2 entirely; all layering |
| Item decomposition is total (necessary for composition) | **TRUE** — 99.6-99.9% item bytes, remainder pure whitespace, 92 files / 4 crates | `roundtrip.rs` census | graph-backed VFS; derived line numbers |
| Regenerating an item by PRINTING its AST is byte-exact | **FALSE** — 0 of 91 files | `roundtrip::regen` census | rules out a printer-based composer |
| Composing a repo from verbatim item nodes is byte-exact | **TRUE** — 1515 files, 31 MB, 3 repos | `roundtrip::compose` scan + regenerate | graph-backed VFS; derived line numbers |
| A canonical-form repo would remove the need for verbatim text | **REJECTED on cost, not feasibility** — converges in 2 passes, but destroys 35k body comments | `roundtrip::canonical` viability | — |
| Normalising with the language's own formatter removes the corner cases | **FALSE** — census identical before and after `rustfmt` (0/34 exact either way) | `normalize::normalising_shrinks_the_divergence_census` | `fidelity: normalized` paying off |
| The embedder discriminates well enough to retrieve | **TRUE**, with alias text | done (§2) | step 7 |
| Near-identical siblings are separable | **TRUE**, needs alias expansion | done — 4/4 with expansion, 1/4 without | code retrieval |
| **Graph structure** is what separates them | **FALSE** — comments 1/4, code 1/4, filename 1/4, all at chance | `bench_siblings::structure_versus_description` | step 7e; the case for graph retrieval |
| Generated notes fix sibling separation | **FALSE** — isolated 2/4, contrastive 1/4 and factually wrong | same benchmark | note-generation pass |
| A better DESCRIPTION can separate siblings | **FALSE** — 9 representations, none above 2/4; failures predicted by attribute uniqueness, not by text | `bench_siblings` full table | rules out description work; points at reranking |
| Decomposed matching beats a blended vector | **TRUE** — 3/4 vs a 2/4 ceiling, same embedder and documents, gaining exactly the files with no unique attribute | `bench_siblings` decomposed section | retrieval design; no cross-encoder needed |
| A cross-encoder beats decomposition | **FALSE — it ties** at 3/4, at 168 ms per candidate against ~0 | `bench_siblings::rerank_versus_decomposition` | whether reranking is worth its latency |
| The graph is the source of truth, files a projection | **aspiration** — ingest built, projection direction unwired | — | bijective line numbers |
| A sparse order key beats a dense index on real edits | **TRUE** — 19% of mutations are inserts; 0 rebalances in 28,881 re-ingests | `bench_history::replay_history` over 5,000 curl commits | node identity; provenance stability |
| Ingest holds up on unfamiliar languages at scale | **UNTESTED** — predictions recorded below | `CORRODE_SCAN_REPO=<repo>` round trip | absorbing arbitrary codebases |
| Re-ingest on write keeps the code index fresh | **TRUE** — edit prunes stale nodes, store still composes byte-exactly | `graph::reingest_after_an_edit_leaves_no_stale_nodes` | trusting index-backed search |
| The store takes an ingested repo at usable rate/size | **FALSE at kernel scale** — 2,847 nodes/s warm, 14.3x on disk | `bench_ingest::store_scale` | ingesting large trees |

Why this section exists, from the record of one session: every miss was a claim nobody
executed. "Two of the five survivors fall to property tests" — `needle.vocab` is plain
text and does not. "Shared prefix means shared KV" — the cross-session cache has never
engaged, and `cached_tokens: 0` sat in a benchmark result hours before §3.2 was written
asserting the economics of it. "Metadata latency will dominate a FUSE build" — metadata
is 1.5x, bulk reads are 16x, so the hypothesis was not merely unproven but inverted.

Two habits follow, and both are cheap:

**A composed file stores its text; it does not reprint it — and then it is exact.**
Printing an AST back is byte-exact for **0 of 91 files**, for two independent reasons:
plain comments have no node in the AST and are simply gone, and the printer
canonicalises (it adds a trailing comma when it breaks a parameter list — semantically
null, textually divergent).

Storing verbatim spans instead is byte-exact for **1515 of 1515 files, ~31 MB, across
three repositories** — corrode, its demo-repo fixture, and hipfire, the last including
macros, `unsafe`, and GPU dispatch. `syn` supplies the item boundaries and their kinds;
each node keeps its own bytes; regeneration concatenates. Byte-exactness is a property
of the decomposition rather than of a printer's manners.

So `ProjectionMode::Composed` is reachable, and the route matters: structure from the
parser, content from the source. The `FallbackReason` variants describe failure modes
of the printer-based approach this rules out — a span-based composer has none of them,
so they want revisiting rather than implementing.

**The canonical-form alternative was measured, not argued away.** Rewriting the repo
once into the printer's own form would remove the need for verbatim text entirely:
nodes would hold structure, trivia nodes would vanish (half of all nodes today), and
item order would become a property of the graph rather than the file. On feasibility it
holds up — the printer is idempotent for 1510 of 1514 files, and the four exceptions
converge to a fixed point in two passes (width oscillation: re-parsing changes the
nesting context, so a call that fitted on one line no longer does). No cycles.

It fails on cost. Canonicalisation deletes every plain comment, because `syn`'s AST has
no node for one: **42,309 lines, 8.5% of hipfire's source.** Doc comments survive as
attributes, and 5,827 of the losses sit between items so could be rewritten as `///`.
The remaining **35,477 are inside function bodies, where `///` is not legal** — they are
unrecoverable by any migration.

That is the wrong 8.5% to lose. This codebase's comments carry the reasoning that the
code cannot: why a gate exists, what a measurement cost, which approach was tried and
abandoned. Trading them for a simpler VFS buys implementation convenience with
institutional memory. The verbatim-span composer already delivers byte-exactness, so
the simplification is not needed to make projection work — only to make it tidier.

**Spans are an ingest-time artifact and must not survive into the store.** A byte
offset is a fact about one source text. A dynamically generated VFS has no such text:
it projects files from nodes that get reordered, inserted, edited, or drawn from a
branch never materialised — so a stored offset or line number is stale or meaningless
the moment the graph moves. Nodes carry content and order; **positions are an OUTPUT of
projection**, recomputed per materialisation and never persisted. `compose::project`
returns the text and where each node landed, and a test asserts that inserting a node
shifts the reported line while the value captured at scan time goes wrong — which is
what a stored position would have done silently.

The same rule fixes comment binding. A comment attaches to `(node, line-within-node)`,
which survives reordering, rather than to an absolute line, which does not.

**Comments are lost in the lexer, not in `syn`.** Rust's grammar treats a plain
comment as trivia and emits no token for it; a doc comment is desugared to
`#[doc = "..."]`. Parsing `// plain\n/// doc\nfn f() { /* inner */ let x = 1; }` yields
the token stream `# [doc = " doc"] fn f () { let x = 1 ; }` — both plain comments are
simply absent. So `syn` cannot be asked to attach them: nothing built on the
proc-macro token model can see one.

It does not need to. `Span::byte_range()` locates every item in the ORIGINAL source, so
the text between items is recoverable and attachment is a post-pass over positions we
already hold: 2,187 comment blocks in hipfire bind to the item they introduce, and the
file still regenerates byte-exactly because attachment is a view over the nodes rather
than a transformation of them.

**Position is not the useful relation; the edge is.** Graph search asks what a comment
is *about*, and what commentary applies to a *region* — neither of which a coordinate
answers. `describes` binds each comment to the syntax element it annotates, with the
relation typed: `Precedes` (introduces the element below it), `Trailing` (annotates the
code to its left, on the same line), `Encloses` (nothing follows in scope, so it belongs
to its container).

Sub-item granularity turned out not to need a CST after all, correcting an earlier note
here: `syn` gives a byte range for *any* syntax node, so statements, match arms and
struct fields are anchorable exactly like items. Measured over hipfire, **44,562 of
44,574 plain comments bind to an element — 12 unbound**:

| relation | | target kind | |
|---|---|---|---|
| precedes | 41,086 | stmt | 34,627 |
| trailing | 3,421 | use | 3,205 |
| encloses | 55 | match_arm | 1,568 |
| | | fn | 1,564 |

Statements dominate, which is the point: the 35,477 body comments that looked
unreachable are the bulk of the corpus's reasoning, and they resolve to the statement
they describe. `ra_ap_syntax` remains the tool if a comment ever has to bind *below*
expression level, which nothing here needs.

Line numbers fall out of the same mechanism. A node knows where it lands, so
`path:line` is derived at projection time; an edit above a node shifts it, which is
exactly why deriving beats persisting.

The first two tiers of this measurement each tested a *proxy* — a hand-rolled byte
census, then a printer nobody proposed — and the second produced a conclusion stated
so badly it read as "composition is unachievable". Only building the composer settled
it. That is the section's own rule applied late.

**Measure the artifact, not a proxy.** The prefix defect was invisible until someone
printed the literal prefix. A sibling-discrimination run produced a dramatic false
negative because it measured `\brief` text that three of the four files did not have.

**Record predictions so being wrong is cheap.** When a measurement contradicts a
document, correct the document with a dated note rather than quietly working around it.
Three lines of diff, and the next reader inherits the correction instead of the error.

The counter-example worth imitating is already in this codebase:
`FallbackReason::MacroExpansion` was written before any projector existed. Someone at
design time asked what would break when composing Rust, and put the answer in the type
system. That is the same failure — macros absent from an AST — that costs a comparable
published system its worst score (0.58 versus 1.00; §10).

## 9. Open questions

- ~~**Does the embedding model discriminate?**~~ **Answered** (§2): yes — 0.250 mean
  top-8 spread on real code, versus 0.057 on an unmatched query. Retrieval was never
  the problem.
- **Can retrieval separate near-identical siblings?** The one real miss was
  `spsc` vs `mpmc` — same family, three letters apart. Descriptions alone did not
  do it, and a codebase is full of that shape. Whether graph structure fixes it, or
  whether it needs a reranker, is untested and is the actual risk in step 7.
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

## 10. Related work

**Codebase-Memory: Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via
MCP** (Vogel, Meyer-Eschenbach, Kohler, Grünewald, Balzer, arXiv 2603.27277). Parses
66 languages with Tree-Sitter into a SQLite knowledge graph served over MCP. Linux
kernel in ~3 minutes (2.1M nodes, 4.9M edges); sub-millisecond queries against 10-30s
for a file-exploration agent; **10x fewer tokens and 2.1x fewer tool calls**.

It independently reaches the architecture §6 step 7 is heading for: *"the optimal
architecture is a hybrid: graph-based retrieval for structural queries, with fallback
to file exploration for source-level tasks."* Their failure analysis names the same two
categories the hybrid exists to cover — full source context (16/31 languages) and
exhaustive call-site grep (10/31), *"queries requiring line-level code that the graph
intentionally does not store."*

Three divergences matter here, and each is a decision rather than an oversight:

**They store no line numbers; we can derive them.** Their graph is an index *derived
from* files, so line information is lost at parse time and they pay for it in exactly
those two categories. Corrode inverts the direction — files are a projection *of* the
graph — so a line number is a function of the projection rather than a fact to store.
Deriving on a hit also costs nothing until something is found, and storing absolute
lines would reintroduce staleness: one insertion at the top of a file invalidates every
node below it. Store position relative to the node; resolve absolute lines at query
time.

**They use no embeddings at all** — Table 7 lists "No embed. model" as a design choice.
That silently bounds what their benchmark can ask: their 12 categories are structural
(who calls this, what does it return), and an AST index cannot serve an intent-shaped
query. Measured here, embedding retrieval answered 4/4 queries with *zero* literal
overlap where substring search returns nothing. So their 83% vs 92% is not a ceiling
for a system with both; it is the score on a question set that excludes the capability.

**Macros.** Their worst case is macro-heavy C at 0.58 versus 1.00, because *"macros are
not represented in Tree-Sitter ASTs"*. Rust's `macro_rules!` and proc-macros are
equally invisible, and this codebase uses them — but `FallbackReason::MacroExpansion`
already exists to mark a node whose expansion the projector cannot reproduce, so the
degradation is tracked per node instead of silently lowering answer quality.

Two caveats on their evidence: answer quality was scored by the paper's first author
against their own reference answers, not blind; and the hybrid they advocate has **no
experimental evaluation** — it is the one part of the paper nobody has measured, which
is precisely the part being adopted.

## 11. Predictions for the multi-language ingest test

Recorded BEFORE running against Python, C, C++ and the Linux kernel, so being wrong is
cheap and visible. Byte-exact projection should hold everywhere — it depends only on
the span cover being total, which the fallback guarantees by using one node per file.
What follows is where COMMENT recovery is expected to degrade.

**Python.** `#` comments are found. **Docstrings are not**, and that is the big one:
`"""..."""` is a string expression rather than a comment, so Python's primary
documentation mechanism is invisible to a marker-based backend. Triple quotes also
stress the coarse string skip, which understands only single-character quoting — a `#`
inside a docstring may be misread as a comment. Expect recall to look fine and
precision to be the problem.

**C.** `//` and `/* */` are found. A **line-continued comment** (`// text \` then a
newline) legally continues in C and this backend ends it at the newline, so the
continuation is treated as code. `#if 0 … #endif` is not a comment and will not be
reported as one, which is correct but means commented-out code is invisible.

**C++.** As C, plus **raw string literals** (`R"delim(...)delim"`), which the coarse
skip does not understand — a `//` inside one may be reported as a comment.

**Linux kernel.** Scale first: tens of thousands of files, with **non-UTF-8 sources**
that `read_to_string` rejects. Those are counted as `unreadable` rather than silently
dropped, precisely so the totals cannot flatter themselves. `.S` assembly mixes C-style
and `#`-style comments and will only get the former. Extensionless `Makefile`/`Kconfig`
files are now routed by filename; before that fix they fell to the C family and would
have found nothing in thousands of files.

### Ingest performance, measured

172 MB / 4,923 files (hipfire, mixed Rust + C/HIP + markdown + config):

| | before | after |
|---|---|---|
| wall | 36.3 s | **22.9 s** |
| throughput | 4.8 MB/s | **7.5 MB/s** |
| phases | parse 43%, bind 43% | parse 67%, comments 23%, **bind 9%** |

Storage amplification is **1.06x** — 172 MB of source becomes 172 MB of verbatim node
text plus 10 MB of ids. Verbatim storage costs the source once more by design; the
overhead above that is ids, at ~3.4 KB per node and 10 nodes per file.

Two optimisations, and the order they were tried is the lesson. `bind` was 43% of
ingest, so the anchor scan looked like the culprit — it is O(comments x anchors) and
binary search is the obvious fix. It bought **1.3 seconds of 36**. The actual cost was
line arithmetic: computing a comment's line by scanning from the start of the file,
per comment, which is O(comments x filesize). A line index computed once per file took
bind from 43% to 9%. The first fix was correct and nearly irrelevant.

A **third** instance of the same defect was found later, by a branch sweep rather than a
profile: `bind`'s owner lookup scanned every node extent per comment, O(comments x
nodes). Measured on synthetic files of 2k/4k/8k commented items it was 1/5/21 ms —
quadratic — and a binary search makes it 0/1/1. It had never bitten because the files
with the most nodes happen to have the fewest comments, which is the reason a profile
would not have found it. Three times now, in one file, the defect has been a lookup that
walks from the beginning; that is a pattern, not three accidents.

Extrapolated to a kernel-sized tree (~80k files, ~1.3 GB, overwhelmingly C and
therefore on the fast fallback path) this is single-digit minutes, not hours. Rust is
the slow backend at ~1.5 MB/s against ~25 MB/s for the fallback, because `syn` parses;
a repo that is mostly Rust will be bound by that.

### Ordering: a sparse key, not a dense index

Node order is a **sparse u64**, assigned `(i+1) << 32` on first ingest.

Dense indices make an insert renumber every node below it. The cost is not the writes
— it is the churn: on the measured worst-case file (1,821 nodes) adding one item marks
1,820 nodes modified, so a one-item diff reads as 1,821 changes and "which task
produced this node" becomes noise. With a sparse key an insert takes the midpoint of
its neighbours and touches exactly one key; every other node keeps its own, and
therefore keeps its id.

Deterministic, not random. A random key would be equally sparse and would make the
same file ingest to a different graph each time, breaking diffing, caching and
content-addressing — the reproducibility the scheme exists to protect.

Keys start at one stride rather than zero, so a file can gain a leading import or
licence header without a rebalance. That was found by a test, not by design: the first
version keyed from zero and left no room before the first node.

Exhaustion is recoverable. `order_between` returns `None` when a gap is spent and
`rebalance` restores full stride for that file — bounded, local, and the reason the key
is documented as overwriteable. At p99 = 121 nodes a rebalance is 121 writes and rare.

Rejected: **file-order edges** (`file -> n1 -> n2 -> …`). They make insert O(1) and
projection O(n) *dependent traversals*, which is the wrong trade — projection is the
VFS read path and runs on every materialisation, while inserts are occasional. A broken
link also truncates a file silently and a cycle hangs it. Containment stays an edge;
order stays a property.

#### Result: replaying 5,000 real commits

Until this was measured the sparse key was decoration, because nothing reconciled
against what was already stored: `ingest::file` assigned every node a fresh
`initial_order(i)`, so re-ingesting an edited file renumbered it end to end and — since
ids derive from the key — re-addressed the whole file for a one-line change. That is
precisely the churn the key exists to avoid. `projection::update::reconcile` is the
missing half: a sequence diff over node fingerprints (common prefix/suffix trimmed,
LCS on the remainder) that lets surviving nodes keep their keys, treats a rewritten
body as the *same slot with new text* rather than a death and a birth, and mints keys
only for genuinely new nodes.

Replayed over the last 5,000 first-parent commits of curl (`bench_history.rs`), which
is a real C edit stream rather than a synthetic one:

| | |
|---|---|
| re-ingests | 28,881 |
| nodes kept | 1,604,569 |
| updated in place | 42,184 (81.2% of mutations) |
| inserted | 9,743 (18.8% of mutations) |
| deleted | 12,614 |
| **rebalances** | **0** |

**Inserts are 19% of mutations — not the rounding error that would have justified a
dense index, and not the majority either.** A dense index would have renumbered a file
9,743 times over this history where the sparse key renumbered it zero times. That
settles the row in favour of the sparse key on evidence rather than on the argument
from churn.

The gap never ran out. Runs of inserts are spaced evenly across their gap rather than
bisected toward the upper bound, so k inserts at one point spend the gap once instead of
k times; across 9,743 inserts nothing exhausted 2^32. `rebalance` stays in the code as
the recovery path, but it is now known to be cold.

Two guards make the numbers trustworthy rather than merely printed. Every one of the
28,881 reconciles asserts that the resulting nodes still project **byte-exactly** back
to the file git holds and that keys stay strictly increasing — so a key-assignment bug
fails the replay instead of quietly reporting a nice ratio. And the LCS has a cell
budget with a cheap positional fallback; the replay counts how often a file could have
hit it, and the answer is **0 of 28,881** (max 1,283 nodes/file), so every alignment
reported here was a true diff and not a budget artifact.

**What this does not settle.** These are human commits — a proxy for agent edits, not
the same distribution. If Corrode's agents turn out to rewrite whole function bodies
almost exclusively, the ratio moves toward update and the argument weakens. What the
replay does settle, and what no argument could, is that a real edit stream at this
volume never exhausts the gap.

**Incidental finding: the node grain is coarse.** A median re-ingest touches **16%** of
its file's nodes, because a node is a top-level item — changing one function in a
six-function file is a sixth of it. That is fine for projection and poor for
provenance: "which task produced this node" is answered at item granularity, so a task
that edited one line claims credit for the whole function. Sub-item nodes would sharpen
it at the cost of many more nodes per file. Not acted on; recorded because the number
was there.

### The pipeline lands: store, freshness, and derived line numbers

For most of this work `replace_file` had no implementation. `HelixStore` implemented
`upsert_node`, `add_edge` and `replace_doc`, so `replace_file` fell through to the trait
default, which bails — and `ingest_written` handled that by logging and returning. Every
number above it, including the kernel's 11.5M nodes, had been measured on a pipeline that
wrote to nothing.

It now writes. A file becomes a `source_file` node owning its code nodes (`has_code`) and
comment nodes (`has_comment`), with `in_node` binding a comment to the item it sits in —
so "what does this comment describe" is an edge traversal. Replacement is atomic and
prunes, for the same reason `replace_doc` does: an index that answers with deleted code
is worse than one that answers nothing.

Two consequences are now measured rather than asserted:

- **Re-ingest keeps the index fresh.** Edit a file so one item changes and one is
  removed, re-ingest: the store serves the new text, the deleted item and its comment are
  gone, and what remains still composes byte-exactly.
- **A search hit carries a line number the graph derived.** Nothing stores a line.
  `file_nodes` returns the file's code nodes in order, `project` replays them and reports
  where each landed, and the hit's order key selects its placement. A line written at
  ingest time is wrong after the next edit above it; this is right by construction — the
  reason the sparse key and byte-exact composition were worth building. `search_files`
  now appends these BM25 hits to its literal scan rather than replacing it: grep answers
  "where is this exact string", which BM25 does not.

**And the store is now the bottleneck.** Ingesting curl into a live store:

```
2,995 files, 34,428 code + 24,193 comment nodes, 13.7 MB source
  cold  80.6s   (727 nodes/s)
  warm  20.6s (2,847 nodes/s)
  store on disk 195.9 MB — 14.3x the source
```

The in-memory pipeline does 225 MB/s. The store does roughly a hundredth of that, and
amplification rose with store size (9.8x at 400 files, 14.3x at 3,000). Extrapolated to
the kernel's 13.8M nodes that is over an hour and ~23 GB, so **large-tree ingest is not
viable against this store as written** — the next real piece of work, and worth knowing
before anything is built on top of it.

**A precise failure found on the way.** 5 of 2,995 curl files failed to write with
`MDB_BAD_VALSIZE`. Not a size limit on the node — the file is 1,934 bytes. It is a
**1,011-byte single token** (a base64 blob on one line): LMDB's max key is 511 bytes and
BM25 indexes every term as a key, so one long token rejects the whole node write. Any
tree with base64, minified JS, long hex or data URIs hits this.

Worse than the failure was the handling: `ingest_written` logged and `return`ed, so one
such file cost the entire turn's code ingest. That was correct when the only possible
error was "not implemented" and wrong the moment real per-file errors existed. It now
counts and continues. The underlying limit is unfixed and needs either a helix-side
change or the verbatim text moving to a property BM25 does not index.

### 7e: structure does not separate near-identical siblings

§6 asserted "Structure is what disambiguates those; a description alone does not." That
was the load-bearing claim under step 7 and it is wrong, at least for the structure the
graph actually holds.

The corpus is stitch's queue family — four headers differing by three letters, the one
real miss from §2. Six representations of the same four files, embedded with
`EmbeddingGemma-300M`, ranked against four queries that deliberately contain **no
acronym** (a query saying "spsc" makes every representation win and measures nothing):

| representation | bytes/doc | top-1 | what it does |
|---|---|---|---|
| filename + `\brief` | 119 | 1/4 | always answers `spsc` |
| **graph structure** (bound comments) | 1,198 | **1/4** | always answers `mpmc_lockfree` |
| filename only | 21 | 1/4 | always answers `mpmc_lockfree` |
| code only (verbatim nodes) | 3,363 | 1/4 | mostly answers `mpsc` |
| alias-expanded (hand-written table) | 179 | **2/4** | varies |
| model-written summary (9B) | 386 | **2/4** | varies |

**1/4 is chance for four items, and four of the six representations sit exactly there** —
each collapsing to one constant answer regardless of the query. The sharpest reading is
that **filename alone (21 bytes) scores the same as the full verbatim code (3,363
bytes)**: 160x more of the file's own content changes nothing. The discriminator is the
acronym, the embedder cannot decode it, and no quantity of surrounding text supplies it.

Only the two representations that inject knowledge **not present in the corpus** beat
chance, and both do it the same way — by expanding `spsc` into "single producer single
consumer". One does it from a table someone maintains; the other from a 9B model at
ingest time. That is the actionable finding: **the remedy is generated description, and
it does not need a hand-written alias table or a large model.**

Held against it: 2/4 is not "solved". Both remedies get the producer-count dimension
right and both fail the progress-guarantee one (`lockfree` vs `waitfree` — the summary
answers `mpsc` for both), and the summary's winning margins are tiny (+0.0015 to
+0.0185) where the failing representations' margins are larger. It separates barely.
n = 4, one family, one corpus, one embedder.

Also worth stating plainly: §2 recorded 4/4 with alias expansion and this run gets 2/4.
That is not a contradiction of the earlier measurement — these queries are harder by
construction (no acronyms at all, and they ask for the lockfree/waitfree distinction,
which the earlier set did not). The comparison that matters is within this run, where
every representation faced the same queries.

**Consequence for the plan.** The case for graph retrieval cannot rest on structure
disambiguating siblings, because it does not. It has to rest on generated per-node
description, and the ingest pipeline is where that would be produced — which makes 7d/7e
a summarisation pass, not a traversal problem. `search_files`' BM25 half is unaffected;
it answers a different question and answers it exactly.

### Commit messages bound to the text that changed

"Why is this line like this" is answered by the commit that wrote the line. Binding at
file granularity throws that away — a file accumulates hundreds of messages and none of
them point at anything. `reconcile` already knows exactly which nodes an edit touched, so
`Update` now reports them (`changed`), and a binding is `upsert_node` for the commit plus
`add_edge` per node: no new store method.

Two rules are built in. A first ingest attributes nothing — every node is new by
definition, and crediting a whole file to whatever commit added it is noise. And a
**whitespace-only edit attributes nothing**, because binding a commit's rationale to a
reformat is worse than not binding it: one formatter run would otherwise produce
thousands of false attachments.

That second rule reintroduced a bug this repo had already fixed and documented. The first
version compared `split_whitespace()` token streams, which is not normalisation —
`x(){1}` is one token and `x ( ) { 1 }` is seven, so a pure reflow reads as changed
content. `roundtrip::regen::formatting_only` carries a comment saying exactly this cost
76 of 78 files a false result. A test caught it; the rule now has **one** implementation,
in production, and `roundtrip` delegates to it.

Measured over 2,000 first-parent curl commits:

| | |
|---|---|
| commits binding at least one node | 1,878 of 2,000 |
| bindings | 16,094 across 9,187 distinct nodes |
| **carrying a reason** | **6,112 (38%)** |
| cosmetic, excluded | 1,035 (6% of would-be bindings) |
| notes per node | median 1, p99 10, **max 133** |

**Binding to changed nodes concentrates the signal**: 19% of all curl commits contain a
rationale word, but 38% of *bindings* do — because commits that touch code are richer
than version bumps and doc sweeps, and those never bind. The cosmetic filter fires on 1
in 16. And accumulation is mild where it was expected to be the problem: the median node
carries exactly one note, so decay is a tail concern (p99 = 10, max = 133), not a general
one. Selection — keeping the 38% that explain something — matters more than aging.

**It does not solve 7e.** Added to the sibling benchmark as a seventh representation,
commit notes score **1/4 — chance**, alongside structure, filename and code. The reason
is visible in the corpus: the messages are largely *shared* across siblings ("Add
lockfree MPMC queue; Rename queue classes and files" touches all four), and where they do
discriminate they use the same acronyms the embedder cannot decode. Rich in general does
not mean discriminating here.

So commit notes are a **gotcha index, not a disambiguator** — they answer "what should I
know before touching this" and not "which of these four is the one I want". Those are
different jobs and the earlier finding stands: only generated description separates
siblings.

### The note-generation pass, measured

7e ended pointing at generated description as the only thing that beat chance. Running
that as an actual pass, with the same 9B and the same benchmark:

| representation | bytes/doc | top-1 | mean winner margin |
|---|---|---|---|
| isolated note | 386 | **2/4** | +0.0084 |
| **contrastive note** (shown its siblings' names) | 431 | **1/4** | +0.0277 |

**Adding context made it worse.** The contrastive prompt was the direct fix for 7e's
observed failure — the summaries got producer-count right and progress-guarantee wrong,
so the model was shown its near-identical siblings and asked what distinguishes *this*
file. It lost a point, and the margins went **up** while accuracy went **down**: it
became confidently wrong, which is the worst failure shape retrieval can have.

The mechanism is visible in the output rather than inferred. For
`queue_mpsc_waitfree.h` — whose source it was given, and whose class is literally
`Waitfree_MPSC_Queue` — the contrastive note reads:

> This file implements a **Multi-Producer Multi-Consumer (MPMC)** queue…

It copied a sibling's classification onto the file. Naming the neighbours contaminated
the description of the thing itself.

**And the deeper result is that the isolated notes are wrong too, on precisely the axis
that matters.** `queue_mpmc_waitfree.h` is `Waitfree_MPMC_Queue`, and its isolated note
calls it "lock-free". So the lockfree/waitfree confusion 7e measured as a *retrieval*
failure is upstream of retrieval: **the embedder is separating notes that are factually
incorrect.** No amount of retrieval work fixes that, and it explains why every
description-based representation plateaus at 2/4 — two of the four notes assert the
wrong progress guarantee.

That redirects the remedy. Free-form generation is not reliable enough at this size to
be the disambiguator, and enriching its context made it less reliable, not more. What
the model *did* consistently get right is the identifier it was reading —
`Waitfree_MPSC_Queue`, `Waitfree_MPMC_Queue` — which suggests extracting and expanding
declared names rather than asking for prose about them. That is closer to the
alias-expansion that scored 2/4 without a model at all, and it is mechanical rather than
generative. Untested; recorded as the next thing to try rather than claimed.

**Caveat unchanged:** n = 4, one family, one corpus, one embedder, one model size. What
this rules out is "generate notes and the problem goes away", which was the standing
assumption after 7e.

### 7e closed: it is not a description problem

Identifier extraction was the last description-shaped idea, and it worked mechanically —
no model, no hand-written acronym table. Take the type a file DECLARES
(`Waitfree_MPSC_Queue`), then find where the repository's own prose explains that
identifier. stitch's `doc/pages/main.md` spells out every variant — "Wait-free
multi-producer-single-consumer bounded-size queue" — keyed by **class name**, which is
why nothing keyed on paths ever found it.

(An earlier note in this document said stitch's docs describe the library and not the
variants. That was wrong, and wrong in a specific way worth recording: it came from a
`grep … | head -8` whose output was filled by README hits before reaching `doc/pages/`.
A truncated command, again.)

It scores **2/4 at 192 bytes** — matching the 9B's isolated note (2/4, 386 bytes) and the
hand-written alias table (2/4, 179 bytes), for free. But the interesting thing is the
plateau, not the tie. Nine representations now, and **none exceeds 2/4**:

| representation | bytes/doc | top-1 |
|---|---|---|
| filename only | 21 | 1/4 |
| filename + `\brief` | 119 | 1/4 |
| commit notes | 168 | 1/4 |
| alias-expanded (hand table) | 179 | 2/4 |
| **identifier gloss** (mechanical) | 192 | **2/4** |
| model note, isolated | 386 | 2/4 |
| model note, contrastive | 431 | 1/4 |
| graph structure (comments) | 1,198 | 1/4 |
| code only | 3,363 | 1/4 |

**Which file fails is predicted by attribute uniqueness, not by representation.**

| target | fails |
|---|---|
| `queue_mpmc_waitfree.h` | **9 of 9** |
| `queue_mpsc_waitfree.h` | 7 of 9 |
| `queue_spsc_waitfree.h` | 4 of 9 |
| `queue_mpmc_lockfree.h` | 4 of 9 |

`spsc_waitfree` is the only single-producer file; `mpmc_lockfree` is the only lock-free
one. Each has an attribute no sibling shares, and each is found about half the time. The
two that fail — `mpsc_waitfree` and `mpmc_waitfree` — have **no unique attribute
whatever**: every property they have is shared with some sibling, and identifying them
requires matching two axes *at once*. `mpmc_waitfree` is found by nothing, in nine tries.

That is a mechanical prediction rather than a statistic, and it closes the question. A
single embedding is a blended bag of features, so a document that matches one axis
strongly outranks the document that matches two axes weakly — no description, however
accurate, changes that. **7e is not a description problem; it is a composition problem in
single-vector retrieval.**

The remedy therefore is not more text. It is matching on decomposed attributes — a
reranker scoring axes separately, or structured filtering on extracted properties. §9
listed "or whether it needs a reranker" as the alternative hypothesis; this measurement
discriminates between them, and hipfire already serves `/v1/rerank`.

**What still holds:** identifier→gloss is worth building regardless. It is mechanical,
192 bytes, ties the model at half the size and none of the cost, and it links prose to
code by identifier — the same derivation `docmap` does for directories, on the key that
actually works.

### The remedy: decompose the query, rank-combine the axes

7e's conclusion predicted that the ceiling was single-vector composition, not
description. That prediction is testable, and it holds.

**A reranker exists on this host but cannot be served.** `/srv/hipfire/models/` holds
`Qwen3-Reranker-0.6B/-4B/-8B` as `.hfa`, while everything the daemon serves is `.hfq`;
and `reranker` matches nothing in `hipfire-model` or `hipfire-serving-core` beyond the
scoring primitive itself. So the models are downloaded, the scorer
(`pooling::rerank_yes_no`) is written and unit-tested with zero production callers, and
the loading path is the missing piece. Filed as hipfire PR #407.

**There is no reranker to test with.** hipfire's `/v1/rerank` is
`rank_by_cosine(query_embedding, doc_embeddings)` over the same bi-encoder
(`hipfire-daemon/src/lib.rs`), so calling it would reproduce the numbers above *by
construction* — "the reranker did not help" would have been a false result about a
cross-encoder that was never involved. Worth knowing before trusting the endpoint's
name.

The hypothesis is testable without one. Score each axis of the query separately against
the same documents, then combine:

| scoring | identifier gloss | model summary |
|---|---|---|
| blended single vector | 2/4 | 2/4 |
| decomposed, combined by `min` | 2/4 | 2/4 |
| **decomposed, rank-combined** | **3/4** | **3/4** |

**3/4 beats a ceiling nine representations could not, using the same embedder and the
same documents.** Nothing was added — the query was taken apart.

`min` was the obvious combiner and the wrong one: axes are not on a common scale, so
comparing raw similarities across them penalises whichever axis happens to sit lower.
Ranking within each axis first removes the scale, then summing ranks asks "is this near
the top for *every* axis" without requiring the numbers to be comparable. That is the
whole difference between 2/4 and 3/4.

The gain lands exactly where the mechanism predicted. Under blended scoring
`mpsc_waitfree` failed 7 of 9 and `mpmc_waitfree` 9 of 9, because neither has an
attribute no sibling shares. Decomposed + rank-combined over the identifier gloss finds
**both** — and misses `spsc_waitfree`, which blending found easily. The two approaches
fail on opposite files, which is what "different failure mode" looks like rather than
"one is better".

**What this does not settle.** n = 4, and 3/4 versus 2/4 is a single file — the weight is
in the mechanism being predicted in advance, not in the score. The axis decomposition was
written by hand for this test; extracting axes from a real query is unsolved and is now
the load-bearing unknown, having replaced "write better notes". And Borda over four
documents is coarse.

What it does establish is a direction that needs no cross-encoder, no larger model and no
generated prose: **retrieval decomposes the query, scores axes independently, and
rank-combines.** Every piece of that is available today.

#### Shipped in `code_search`

`projection::query_axes` splits a query on clause separators and `rank_combine` sums
per-axis ranks; `code_search` runs one BM25 pass per axis and combines them. BM25 sums
per-term contributions, which *is* blending — a document matching one clause emphatically
outranks one matching every clause moderately — so the same fix applies to it as to
cosine.

Four decisions worth keeping:

- **A query with nothing to split comes back whole**, so single-clause search runs one
  pass and behaves exactly as before. The change can only affect multi-clause queries.
- **The splitter is deliberately dumb** — clause separators, no parsing, no model.
  Extracting real axes from arbitrary prose is the load-bearing unknown here; a
  heuristic that over-splits costs a little precision, while one that invents structure
  would put a wrong constraint on every search.
- **Rank-combine, not score-combine.** Axes are not on a common scale, and the obvious
  `min` combiner measured no better than blending. Ranking within each axis first is the
  whole difference between 2/4 and 3/4.
- **Absence from an axis is charged, not ignored.** A document missing from one axis's
  results ranked below everything that axis returned, which is information.

Two things a test pins because they look like bugs and are not: a perfect reversal
between two axes ties every document (Borda being right, with first-seen order keeping it
deterministic), and a multi-clause query can be won by a *comment* node — the text lives
where it lives, and requiring a `code:` node would be the wrong assertion rather than the
right result.

### Wiring retrieval to a real cross-encoder

The reranker that 7e wanted now exists: hipfire serves `Qwen3-Reranker-0.6B--oq8`
through `/v1/rerank`, and `code_search`'s BM25 shortlist can be rescored jointly instead
of term-by-term. `ToolBox::with_reranker` enables it when `CORRODE_RERANK_MODEL` names a
served reranker; absent, search is exactly what it was.

Two guards. The client **errors if the server answers in `cosine` mode** — hipfire picks
the scorer from the loaded model, so naming an embedding model would silently return
bi-encoder similarity under the name of reranking, which is the thing this was built to
get away from. And a reranker that is down falls through to the BM25 order rather than
emptying the results: a worse answer, not no answer.

**It ties decomposition, and costs real time.**

| approach | top-1 | cost |
|---|---|---|
| blended single vector (nine representations) | 2/4 | ~0 |
| decomposed query, rank-combined | 3/4 | one BM25 pass per axis |
| **cross-encoder rerank** | **3/4** | **168 ms per candidate** |

Both reach 3/4 and both recover `queue_mpmc_waitfree.h`, the file no blended
representation ever retrieved. They are two routes to the same place: decomposition
splits the query so a single vector never has to compose, while the cross-encoder reads
the pair jointly so composition never has to be factored out. The cross-encoder's margins
are far wider (+0.107 against +0.010), which suggests it is more robust than the tie
implies — but on this corpus it does not retrieve anything decomposition misses.

So the recommendation is decomposition first, reranking as an opt-in over a small
shortlist. A 24-candidate rerank is ~4 s, which is affordable for a deliberate search and
not for an incidental one.

**A performance defect found by measuring rather than by profiling.** The first wiring
measured **1,912 ms per pair** — 46 s for a 24-candidate shortlist, unusable. The cause
was in the code I had just written into hipfire: it drove `ChunkScoredForward::forward_chunk_scored`,
whose teacher-forced walk prefills one token and then decodes the rest one at a time, so
scoring a single final position cost a decode step per prompt token. `SimpleAr::prefill`
already takes the whole prompt and leaves exactly the logits wanted. One prefill instead
of the walk: **168 ms per pair, 11.4x**, with identical accuracy — the same logits,
obtained the cheap way (hipfire #409).

### A Qwen tool dialect

`tools.rs` records a 35B "emitting `<tool_call>` blocks nothing read" while the swarm
answered repository questions by guessing. That was still true: every Qwen model fell to
the Needle default, and its own calls went unparsed.

`ParseFormat::QwenToolCall` reads them, and `*qwen*` routes natively by default. Two
shapes are accepted, because **the artifact decides which, not the model family**:

- `<tool_call>{"name":"f","arguments":{…}}</tool_call>` — the Hermes JSON upstream
  documents, and what this was written for first.
- `<tool_call><invoke name="f"><parameter name="p">v</parameter></invoke></tool_call>` —
  what `Qwen3.5-9B--oq4.25++` actually emits, found by running it.

Assuming the documented shape produced an **empty** parse on the first live run: the model
sent `invoke` XML, the parser wanted JSON, the block yielded no call, and the loop read
that as a final answer — the identical failure the dialect was added to fix, one layer
further in. Both shapes are parsed now, and the malformed-block case is a test rather than
a discovery.

Measured on the demo repo, same prompt and model, before and after:

```
before   settled: 3 outputs, 0 tool results
after    settled: 5 outputs, 1 tool result
```

The parser is deliberately forgiving — a malformed block does not discard the well-formed
ones beside it, and a reply truncated at the token cap still yields its last complete call
— because every strictness here costs a tool call the model made correctly.

#### Withdrawn: the shape table was an artifact of a hipfire flag

Everything below this heading, up to the `SELF_TALK` section, was measured against a
hipfire in which **Jinja chat rendering was off for the qwen35 architecture** — so the
template's `{% if tools %}` branch never fired, the models were never told a tool existed,
and they improvised call syntax as free text. That is where the "eight shapes" came from.
One model produced different syntax under different prompt wordings because it was
guessing, not because the artifact has a shape.

With rendering on, all three Qwen versions return **structured tool calls** and hipfire
parses them itself. `code_search`'s companion dialects for those shapes
(`QwenToolCall`, `YamlToolCall`, the invoke/function/JSON payload matching) are removed:
they parsed text that should never have been text.

What survives, and why it is not the same mistake:

- **`MiniCpmXml` and `ZyphraXml` stay.** Measured: llama (arch 0) and zaya (arch 16)
  render their templates unconditionally — the flag never gated their paths — and hipfire
  does **not** parse their calls into structured form. Zaya still returns
  `<zyphra_tool_call>` text with rendering on. Those dialects read genuinely-templated
  output and always did, which is why MiniCPM scored 12/12 while Qwen produced noise.
- **The client now prefers the server's `function_call` output items** and falls back to
  the dialect when there are none. Empty is not "the model called nothing": an arch whose
  calls hipfire does not parse, or an older hipfire, returns no item.

Measured after the change, same prompt and repo: **1 tool result -> 10**.

Upstream this needed three hipfire fixes — the log claimed a template was adopted when it
could not reach the prompt (#411), the flag was env-only so it could not be set per model
(#413), and `/v1/responses` dropped the tool calls it had already parsed (#414). The
original table is kept below rather than deleted, because the wrong conclusions are the
reusable part.

#### The shape is a property of the artifact, not the model family

Routing `*qwen*` natively was wrong, and testing the family rather than the one model
showed it. Same prompt, same repo, one served artifact each:

| artifact | emits | tool results |
|---|---|---|
| `Qwen3.5-9B--oq4.25++` | `<tool_call><invoke name=…>` XML | 1 |
| `Qwen3.6--35B-A3B.oq4.25++` | `tool_name:` / `tool_args:` YAML | 0 |
| `Qwen3.8-27B--oq4.25++` | — | 0 |
| `Qwen3.5--0.8b-oq4++` | prose, no call at all | 0 |

Three different shapes across four builds of one model line. The family glob routed the
other three natively and **cost them their tools**: the 0.8b answered by inventing the
contents of `src/lib.rs` rather than reading it, which is the precise failure native
routing was added to fix, reintroduced one layer along. Narrowed to `*qwen3.5-9b*`, the
0.8b goes back through Needle and gets a tool result again.

So the rule is per artifact and never per family, and **Needle is the right default
because it is shape-agnostic** — it builds the call from a plain-English line, so a new
artifact's private format costs nothing. A native dialect is an optimisation that has to
be measured for the exact build it names.

For comparison, `MiniCPM5--1B.oq4.25++` on its own native dialect produced **12 tool
results in 12 outputs** — when the shape is right, native routing is very good, which is
what makes assuming it tempting.

#### Can the Needle shim be cut?

The question is worth asking: it is 2,813 lines plus **51 MB of committed weights**, a
candle dependency, a feature flag, and a second tool loop — and that last cost was paid
during this session, when trace recording was added to one loop and not the other, so
every model on the native path silently wrote no notes.

The deciding distinction is that **Needle is a generator and a dialect is a parser**. A
parser can only read a call the model emitted; Needle builds one from a plain-English
line. So the question is not "can we write more dialects" but "does every model we care
about emit a call at all".

Measured across all six served artifacts, with a YAML dialect added for the third shape:

| artifact | emits | routed |
|---|---|---|
| `MiniCPM5--1B.oq4.25++` | MiniCPM XML — 12 calls in 12 turns | native |
| `Qwen3.5-9B--oq4.25++` | `<tool_call><invoke>` XML | native |
| `Qwen3.6--35B-A3B.oq4.25++` | `tool_name:`/`tool_args:` YAML | native |
| `ZAYA1--8b.oq4++` | Zyphra XML | native |
| `Qwen3.8-27B--oq4.25++` | `<function_calls><invoke>` / `<function=…>` | native |
| **`Qwen3.5--0.8b-oq4++`** | **prose, invents file contents** | **Needle** |

**The 27B row above was wrong, and finding out why mattered more than the table.** It was
measured as "emits prose, no call" and that was an artefact of how corrode asked. Three
separate faults, each of which alone hid the model's calls:

1. **The parser keyed on the wrapper.** `Qwen3.5-9B` wraps its payload in `<tool_call>`,
   the 27B in `<function_calls>` — the same `<invoke>` body. Scanning for `<invoke>`
   regardless of envelope fixes it.
2. **One artifact emits more than one shape**, depending on wording: `<invoke>` when asked
   plainly, `<function=f><parameter=p>` — the shape zaya uses — when the prompt actively
   invites a call. So the parser tries every known payload rather than one per model.
3. **Corrode's own prompt suppressed the call.** `native_tool_prompt` ended with "reply
   with your final answer and **no tool call**", and bisecting it against the live model
   showed that clause alone turning a call into narration:

   | prompt | result |
   |---|---|
   | bare task | emits a call |
   | task + `[role: coder]` | *"I don't have access to your file system"* |
   | task + "You have tools available. Call one when you need it" | emits a call |
   | task + "reply with your final answer and no tool call" | **no call** |

Reworded to "Once the tool results answer the task, give your final answer", the 27B goes
from **0 to 1 tool result** in corrode.

#### The shape is not stable, so parsers cannot win

The version keys the shape — 3.5 wraps `<invoke>` in `<tool_call>`, 3.6 emits YAML, 3.8
wraps `<invoke>` in `<function_calls>` — but that is not the whole story either. Probing
the same two builds with four wordings of the same instruction produced **eight distinct
shapes**:

```
<tool_call><invoke name=…>          <function_calls><invoke name=…>
<tool_call><function=…>             tool_name: / tool_args: YAML
```json {"name":…,"arguments":…}    ```json {"tool_call":{…}}
<tool_use> {…}                      ```bash read_file src/lib.rs
```

One model, one task, different wording, different syntax. So the parser now matches the
**payload** rather than the wrapper — any JSON object carrying a `name` wherever it sits,
any `<invoke>` block whatever encloses it — which is strictly more robust than a tag per
build.

**Prompt tuning was tried and abandoned.** Three wordings, each trading one model for
another:

| wording | 9B | 27B |
|---|---|---|
| "…and no tool call" (original) | **1 call** | 0 |
| "Once the tool results answer the task…" | 0 | **1 call** |
| "…not a description of it, and not a code block" | 0 | 0 |

No wording served both, and an isolated probe did not predict corrode's behaviour — the
full prefix and role framing change the answer. The prompt is back to the original, since
churning it on failed tuning leaves the system worse than it was found.

**This reverses the shim conclusion above.** The argument for cutting Needle was that
enough dialects make it redundant; the evidence is that the target moves under prompt
wording, so a parser set is never finished. Needle is invariant to all of it: the 9B
saying "I need to read the file src/lib.rs" is a perfectly good input to a generator, in
every one of the eight cases. **Keep the shim.**

The design this argues for, unbuilt: parse natively first, and **fall back to Needle when
no call parses**. That costs a wasted turn only when the native path fails, and it is the
one arrangement where a new shape degrades instead of breaking.

#### Forcing a shared chat template does not fix it

hipfire honours `HIPFIRE_CHAT_TEMPLATE_FILE` (global, `hipfire-prompt/src/lib.rs`), so the
hypothesis is directly testable: if the shape is a template property, forcing one template
across versions should unify it. Two templates were tried, with the override verified
active in the daemon's environment and in the serve log each time:

- **3.5's own embedded template**, extracted from `tokenizer_config.chat_template` in the
  `.hfq`. It is explicit — *"If you choose to call a function ONLY reply in the following
  format"* — and the format it specifies is `<tool_call><function=…><parameter=…>`, which
  is **not** the `<invoke>` shape the 9B was observed emitting. The model drifts from its
  own template.
- **`froggeric/Qwen-Fixed-Chat-Templates`**, a 28 KB template covering 3.5 through 3.8.

**Neither changed anything.** Outputs were identical with and without the override, for
3.5, 3.6 and 3.8. So the shape is not template-determined in the way the version
correlation suggested, and a shared template does not unify these artifacts.

**A caveat that weakens several results above.** The 3.8 emitted a well-formed
`<function_calls><invoke>` call earlier in this session on essentially the request used
here, and now returns empty output consistently — 5 of 5, and unchanged by tool-schema
detail. That earlier success could not be reproduced and the cause is unknown. Single-run
probes are therefore weaker evidence than they were treated as: the eight-shape table is a
record of things observed, not a stable characterisation of what each artifact does. The
conclusion it supports — that shape-chasing with parsers is a losing game and a generator
is invariant — survives that caveat, because it only needs the shapes to be unstable,
which is exactly what could not be pinned down.

Four assumptions died here in sequence — that a model family shares a call shape, that the
shape correlates with capability, that the 27B shared the 35B's YAML, and that a model
narrating instead of calling is a model that cannot call. Each was plausible; each was one
measurement from being disproved. The last one held only because someone said "the 27B
definitely can" and I went and checked instead of trusting my own table.

### Notes from traces, and what happens to the wrong ones

Cold-generated documentation measured badly (a 9B calling a `Waitfree_MPMC_Queue`
"lock-free"), so `trace.rs` takes a different source. The expensive part of agent work is
not writing prose — it is **search and verification**. Establishing that a function is
never called, that a path is unwired, that a test fails for a particular reason costs real
effort and nothing to record, and a later agent would otherwise pay to rediscover it.
Those are also mostly *negative* facts, which no static extraction can produce.

**Observed and asserted, split mechanically.** The tool loop already separates the two:
text the model emitted is its own claim, and the string `ToolBox` returned is what the
system reported. So the split needs no judgement and no classifier — and `Observed` is not
a quality rating, it means "a tool produced this". Observations are kept **verbatim**,
never summarised, because a summary of a fact is a claim about a fact and the point of the
split is that one side introduces nothing.

**Wrong notes are expected; the design is about what happens next.** Prevention was never
available — this session alone produced three confident falsehoods worth storing as
warnings: that stitch's docs do not describe the queue variants (a truncated `head -8`),
that hipfire had no reranker work (a two-crate grep), that the sparse key was working in
production (nothing called `reconcile`). A filter that rejected implausible claims would
have kept all three, since each read perfectly plausible. So instead:

- **Provenance.** Every note says whether a tool produced it or an agent claimed it, so
  storing a guess cannot launder it into a fact.
- **Supersession.** Append-only. A correction is a new note plus a `supersedes` edge,
  never an edit — the wrong version stays readable and stays attributed, so a contested
  claim is visible as contested.
- **Staleness.** `reconcile` already reports which order keys an edit changed, so a note
  about a changed node is marked stale mechanically. This is what the earlier "decay"
  idea lacked: something to hang on.
- **Reading order.** Observed before asserted. Not a truth ranking — it is the one thing
  the provenance licenses, and it stops a confident sentence outranking the command output
  that contradicts it.

The filter keeps lines that state an outcome and drops narration, the same shape that made
commit-message binding concentrate signal (19% of commits carry a rationale word, 38% of
bindings do).

**Run against a working swarm.** Once hipfire rendered the qwen35 template and returned
structured calls (#413, #414), the same demo-repo turn went from **0 tool results to 11**,
and the trace path was exercised for the first time on real activity: **22 notes, 37
edges, 4 tasks**.

That surfaced two defects the earlier runs were too quiet to reach:

- **Task nodes were keyed by their prompt text.** A long task description exceeds LMDB's
  511-byte `key` index limit, so every such task failed to record — `cannot record task
  node (add_n Review the work this plan just completed…`. The key is now the task id and
  the text is the label, which is the same rule the code nodes already follow. Before:
  4 notes with two tasks silently failing. After: 22/22.
- **Notes hung off a task node the plan did not own.** `record_trace` keyed tasks
  `task:{id}` while `PlanGraph::provenance` writes `{plan}:task:{id}`, so every turn wrote
  a *second*, disconnected task node — and task 0 of every turn collided on one of them.
  The join back to the plan that `Note::task` documents did not exist. The turn's plan id
  now rides on `ToolBox` (which the loops already carry; `run_task` takes 16 arguments
  without it) and `ToolBox::task_ref` is the single place that shape is spelled.
- **Every note's `noted_by` edge named the wrong node.** The edge was built from the
  task's prompt *text* while the node had just been written under its id, so all 22 notes
  were attributed to a node that does not exist: stored, and unreachable from the task
  that produced them. `add_edge`'s error was `.is_ok()`-swallowed, so the write counter
  reported a clean run either way — which is why this survived the key fix above and read
  as "the readback query is too narrow" for two runs. The e2e now walks the real lineage — plan
  -> tasks -> notes — and asserts reachability, since a count of successful writes says
  nothing about whether the edges naming them resolve, and an id-range walk would still
  pass with the notes hanging off nodes disconnected from the plan.
- **The e2e's approval invariant passed by vacuity.** It asserts that no mutating call
  executes without an approval event, and `CORRODE_AUTO_APPROVE` approves without emitting
  one — so the assertion only held while the swarm was calling no tools at all. It now
  accounts for auto-approve, which is the gate being open by policy rather than bypassed.

**Persisted.** A note is `upsert_node` and each edge is `add_edge` — no new store method.
The note's `kind` *is* its node kind (`observed` / `asserted`), so provenance is not a
property a query has to remember to ask for. Edges: `noted_by` to the task that wrote it,
`about` to each file the task touched, and `supersedes` to prior notes on those files.

Two decisions worth stating. Paths come from the **structured tool call**, not from the
model's prose — a note bound to a path guessed out of English would attach real findings to
the wrong file. And notes bind to the **file**, not to a node inside it: "this loader is
never called" is about the file's role, and binding it to whichever node happened to be
read would claim a precision the trace does not have.

**Measured on a real trace, and the filter is loose.** Run over a full session
transcript — 1,761 turns of actual tool calls and results:

```
notes        845   (466 observed, 379 asserted)
yield        48% of steps produced a note
compression  766 KB of tool output -> 214 KB of notes (28%)
```

48% and 28% is a trim, not a summary, and the samples show why: a file that merely
*contains* the word "error" became a note about its own contents. So `extract` now
distinguishes tools that produce an **outcome** from tools that return **content** —
`read_file`, `list_dir` and `search_files` hand back what is already there, while running
a command makes something happen, and only the second can yield an observed finding.
(Unknown tools count as outcome-producing, so a new mutating tool is not silently dropped
by an out-of-date list.)

**Run against a real swarm turn.** The demo repo, a 9B, a live HelixDB store, both tool
loops instrumented (only the Needle loop was, so a model on a native dialect wrote no
notes at all — silently, since nothing reports notes it never tried to make):

```
trace: 1 note(s) from 2 step(s) — 0 observed, 1 asserted
trace: 1 note(s) from 6 step(s) — 1 observed, 0 asserted
persisted 2/2 notes, 4 edges; both readable back from file:src/lib.rs
```

The first run of that produced **4** notes, and two were the agent narrating its own
tool-call formatting errors — "the previous tool call failed because I tried to use JSON
syntax". Those pass the finding filter honestly (they contain "failed" and "because") and
say nothing about the code. A `SELF_TALK` filter drops them, taking yield from 50% to 25%
and leaving the two that are about the repository, including the one worth having:
`is_prime` is O(n) with a TODO to make it O(sqrt n).

The first store read returned **0 notes from a write that had happened** — the query asked
for neighbours of provenance `code:` nodes, and notes attach to `file:{path}`. Nearly
recorded as "persistence is broken". `record_trace` now reports what it wrote, so the next
such gap is visible from the log rather than from a guess.

**Note yield tracks trouble, not work.** The 25% above came from a run where the agent
was fighting its own tool-call formatting. Re-run after Qwen was given a native dialect —
so its calls parsed and executed cleanly — the same prompt produced **zero** notes: the
tool that ran was `read_file`, which the outcome/content split excludes by design, and
nothing failed. That is the filter behaving correctly and it bounds what notes are for.
They capture what went wrong or was discovered, not what a task did. A store of them will
be a gotcha index, which is what the commit-message measurement already suggested, and it
will be sparse on repositories where the swarm has an easy time.

**The transcript fix is unvalidated, and the corpus is why.** The session used a shell for
everything — 1,139 of 1,761 steps map to `run_command`, including greps and file reads —
so the split has almost nothing to separate here and the numbers are unchanged. In
corrode's own toolset `read_file` and `search_files` are distinct tools and it would bite,
but that is an argument, not a measurement. Tuning the filter harder against a corpus that
misrepresents the tool mix would be fitting to the wrong distribution, so the numbers stand
as recorded and the real yield question waits for a trace from the swarm's own loop.

The test that carries the design asserts the uncomfortable half. After a correction, the
**wrong note is still there** — still saying what it said, still naming the task that wrote
it, with a `supersedes` edge pointing at it and the correction reachable from it. Deleting
it would destroy the evidence that a claim was ever contested, which is the one thing an
append-only store buys over an editable one.

### Per-directory prose: what is actually there

`AGENTS.md` is read from the repo root only (`repo_root.join("AGENTS.md")`) — there is no
per-subdirectory stack, and composing one down the tree is unbuilt.

The kernel is the corpus worth sizing this against, and its per-directory prose is not
where it is assumed to be:

| | files |
|---|---|
| `Documentation/` | 12,037 |
| `Kconfig` (help blocks) | 1,916 |
| `README*` | **89** (of 6,204 directories — 1.4%) |

READMEs are not the kernel's mechanism. `Documentation/` is, and Kconfig help is
per-directory prose describing precisely what the code in that directory does.

#### Mapping them, without guessing

The obvious mapping is directory-name matching — `Documentation/networking` → `net/`,
`filesystems` → `fs/`. Measured, it is thin: nine top-level names match directly
(`arch`, `block`, `mm`, `security`, `sound`…), three need aliases, and the rest are
guides that map to nothing. It is also a *guess*, and a wrong edge points an agent at the
wrong subsystem, which is worse than no edge.

Two exact rules do better, both checked against the kernel before being written:

- **A config/build/readme file describes its own directory.** 1,912 of the kernel's 1,916
  `Kconfig` files live outside `Documentation/` — in the directory they document. No
  inference at all.
- **A prose file describes every source directory it NAMES**, confirmed against the
  repo's real directory set so a plausible-looking path yields nothing.

`projection::docmap` implements exactly those two and deliberately omits name matching.
Over the whole kernel:

```
6,204 directories
  prose + config files scanned              10,394
  linked to at least one directory           6,610  (64%)
  describes edges                           10,224
  … config/build files describing own dir    5,412
  linked by backend: hash 5,330  rst 1,164  none 47  c-family 61  markup 8
```

**10,224 derived edges, no table and no model.** The guard that makes this safe is that a
citation must resolve against the tree's actual directories: `net/imaginary/thing.c`
produces nothing, and `subnet/core` does not match the `net` root. Source files are not
scanned at all — a C file's `#include` is not a claim to document that directory.

**The graphs are now joined.** `GraphStore::place_file` writes the directory node, the
`in_dir` edge putting a file in its directory, and a `describes` edge per link, and
`ingest_written` calls it after every `replace_file` — computing the repo's directory set
once per turn so a doc naming twenty subsystems does not cost twenty walks. A described
directory is upserted rather than looked up, because prose can name a subsystem before
that subsystem's code has been walked, and an edge should not be dropped for arriving
early.

The path that now exists, and is tested end to end: **code file → its directory →
everything describing that directory.** A design note a task writes becomes reachable
from the code it explains, which is the point of ingesting prose into the same store as
source. The same test asserts the guard still holds — a C file that merely *names*
`drivers/pci` acquires no `describes` edge.

### Agent traces: summarise for retrieval, keep the reasoning

Recorded before it is built, because the shape is already decided by what exists.

An agent trace attached to the code it generated is **the same relation as a commit
message attached to the nodes it changed** — a change, its rationale, and the text that
moved. That relation is built (`Update::changed`), and `produced_by` edges from code
nodes to tasks already exist in provenance. So this is not new machinery; it is the
existing machinery pointed at the swarm's own output.

The one design commitment worth making now: **a summary is an index, not a replacement.**
Reasoning is the part that cannot be reconstructed — the alternatives rejected, the
constraint discovered, the thing that looked right and was not — and it is exactly what a
summary drops. The doc side already has the right pattern: `doc → has_chunk → chunk`,
where chunks are embedded and the doc is intact. Traces take the same shape — trace node
holds the full text, a summary node is what retrieval matches, an edge joins them — so a
hit on the summary is one traversal from the reasoning that produced it.

Two things this must not do, both already observed elsewhere in this document.
Summarising at every-node granularity does not scale (the kernel has 11.5M code nodes;
even at file granularity, note generation is an Opportunistic-band job, not a synchronous
one). And a summary that is embedded while the trace is discarded is unrecoverable —
unlike every other lossy step here, there is no verbatim copy to fall back to.

### Step 8: the store's cost is text, and identity was leaking into search

Four variants over the same 1,500 curl files isolate the cause instead of guessing at it:

| variant | nodes | text | time | nodes/s | store |
|---|---|---|---|---|---|
| baseline | 36,015 | 7.9 MB | 17.0 s | 2,121 | 144.6 MB |
| **one-char labels** | 36,015 | 0.0 MB | **2.3 s** | **15,816** | 63.4 MB |
| no trivia nodes | 28,184 | 7.3 MB | 11.6 s | 2,424 | 127.9 MB |
| no trivia, one-char | 28,184 | 0.0 MB | 4.0 s | 6,971 | 50.5 MB |

**Text is the cost, not node count.** Removing the text at identical node count is 7.4x
faster; removing 22% of the nodes is 1.5x, and even that is proportional to the text
those nodes carried. The cause is in the vendored engine:
`bm25::term_counts_for_node` iterates the **whole property map** and tokenises every
value — there is no field selection and no opt-out — and our `label` holds the verbatim
node text.

Run-to-run variance is large (the same workload measured 727 and 2,847 nodes/s cold
versus warm), so nothing under ~2x in this table should be read as a difference. The 7.4x
is well clear of it.

**The same design flaw is also a correctness bug**, which is the part worth acting on.
Every property is indexed, and our node key is `code:{path}#{order}` — so **identity is
searchable as though it were content**. Measured: querying `frobnicator`, a word that
appears nowhere in `drivers/frobnicator/widget.c`, returned both of its code nodes. On a
kernel-sized tree a query for `drm` would match millions of nodes on their paths alone,
and BM25's document lengths and IDF are skewed by path terms throughout.

`code_search` now drops hits whose query terms appear only in the key, pinned by a test
that also checks real content search still works.

**Two limits found while writing that filter, both pinned by the same test.** The first
version collected query terms longer than two characters and required one to appear in
the node text — and `.any()` over an empty iterator is `false`, so a query made *only* of
short tokens silently returned nothing. It now passes the hit through when no term is
long enough to check, because a filter that cannot distinguish must not discard.

The second is upstream and cannot be fixed here: helix's BM25 tokeniser drops any token
of **two characters or fewer**, at index time and query time alike (`bm25.rs`,
`if SHOULD_FILTER && token.len() <= 2`). So `code_search` structurally cannot find `fd`,
`mm`, `sk`, `nr`, `id`, `rq` — a large share of C identifiers — however plainly they
appear in the source. That is a good argument for the shape `search_files` already has:
BM25 is **appended to** the literal scan rather than replacing it, and grep finds exactly
the identifiers BM25 is blind to. That removes the false hits and costs
nothing; it recovers none of the write time or storage those path terms cost. The real
fix is field-selective indexing, which means forking the pinned engine — worth doing when
kernel-scale ingest is actually needed, and not before.

**Scoping the finding honestly.** At 2,121 nodes/s a normal repository ingests in tens of
seconds, which is fine. Only whole-tree ingest of something kernel-sized is out of reach,
and that was a stress test rather than a use case. Step 8 is therefore diagnosed rather
than urgent — but it is diagnosed, and the number that matters (7.4x, in the text) points
at one specific upstream behaviour rather than at "the store is slow".

### 7f: serving files from the graph

Everything until now ran one direction — source into nodes. `graph-model.md` makes files
a projection of the graph, and `graphvfs::GraphVfs` is the return trip: `file_nodes`
hands back a file's code nodes in order, `project` composes them, and the bytes are the
file.

Fidelity was never the risk here — composition is already exact in both directions
(94,750 of 94,750 kernel entries, 28,881 reconciles across 5,000 curl commits). **The
risk is staleness.** A graph that has not seen an edit serves confidently wrong bytes,
and an agent editing against them produces a patch that does not apply — worse than any
error this replaces, and invisible unless someone looks.

So the wrapper is built to be honest about not knowing:

- It serves the graph **only** for files the graph actually holds, and falls through
  otherwise. No nodes means "not ingested", not "empty file".
- `stat` reports the size of the **composed** bytes, not the file on disk. A stat that
  disagrees with the following read is worse than either answer alone, and FUSE will
  truncate a read to the size stat promised.
- `list` and `tracked_files` stay on the inner VFS. The graph knows only what has been
  ingested, so answering enumeration from it would make directories look emptier than
  they are — and `tracked_files` defines the search corpus, where under-reporting
  silently loses results. Reading is per-path and can fall through honestly;
  enumeration cannot.
- `CORRODE_VFS_VERIFY` compares every graph-served read against the filesystem and
  **reports** divergence rather than resolving it. Silently preferring either side is
  how an agent ends up editing text that does not exist, and which side is right
  depends on why they differ.
- Setting `CORRODE_VFS_GRAPH` with no store open says so instead of quietly staying a
  passthrough.

The end-to-end test pins the failure mode rather than hiding it: ingest a file (doc
comment, raw string, nested braces), read it back byte-exactly through the VFS, then
edit it on disk **without** re-ingesting and assert the VFS still serves the old bytes.
That is the staleness risk, written down as a test so it cannot be forgotten, and it is
the argument for `ingest_written` running on every write.

Off by default, like `CORRODE_SANDBOX`, so the passthrough remains the behaviour nobody
opted out of.

### Fidelity as project policy

Verbatim storage is exact today and its bill grows: every increase in node specificity
adds corner cases that keep projection byte-exact — `rustfmt::skip`, raw strings, macro
bodies, attribute placement, and whatever the next language brings. `.corrode/project.json`
therefore carries a `fidelity` field, `verbatim` (default) or `normalized`, with
`corrode-daemon normalize [--write]` to bring a repo to normal form in one reviewable
commit and to check that a repo claiming `normalized` actually is one.

Two things are deliberate. The formatter contract is stdin -> stdout with `{path}`
substitution, so `--check` never touches a file; and `--write` refuses on a dirty tree,
because rewriting every tracked file is only safe when `git checkout .` can undo it
without taking the user's own work.

**The measured result is that this does not yet buy what it was meant to buy.** The
premise was that regular source has fewer quirks to special-case, so a normalised repo
could drop verbatim text and generate its own. Running every tracked Rust file through
`rustfmt` and re-running the AST-regeneration census on the output:

```
34 Rust files
  as committed: 0 regenerate exactly, {UnknownDivergence: 32, MacroExpansion: 1, RustfmtSkip: 1}
  normalised:   0 regenerate exactly, {UnknownDivergence: 32, MacroExpansion: 1, RustfmtSkip: 1}
```

Identical. Not smaller — *identical*. The divergence was never caused by irregular
source; it is caused by two printers disagreeing about style. Normalising to `rustfmt`
cannot make `prettyplease` output match, because `rustfmt` is not the printer we would
generate from.

That sharpens what normalisation has to mean. A normal form only removes the corner
cases if it is **our own projection's output** — then source equals what we generate by
construction. Normalising to `prettyplease` would do it and is already rejected: it
destroys 35,477 body comments, because a `syn` AST has no node for them. So the trade
is not verbatim-versus-formatter; it is verbatim-versus-*a comment-aware printer of our
own*, which the comment nodes and their binding edges now make reachable and which does
not exist yet.

`fidelity` stays, because the seam is right and the check is useful now. What it does
not do is let anyone delete the fidelity machinery — and it would have been easy to
ship it believing otherwise.

**And it can only ever cover the languages that have a formatter.** Per-backend census
of curl's 4,472 tracked files:

| backend | files | formatter |
|---|---|---|
| c | 1,026 | clang-format |
| c-family (fallback) | 2,227 | none |
| markup (930 `.md`) | 930 | none, deliberately |
| hash (Makefiles, automake) | 271 | none, deliberately |
| none / semicolon | 18 | none |

**23% of the repo has a formatter.** The kernel's mix is the opposite way round (~78%
C/Rust), so "normalise the repo" means very different amounts of coverage per project —
but in both cases the verbatim path is what the remainder runs on, and cannot be
removed. Prose and make fragments are excluded *by construction*, not by convention: a
Makefile's recipe lines are distinguished by a leading TAB, so a formatter that
normalises indentation breaks the build while the tree still compiles for anyone who
has not re-run make. `tab_significant_and_prose_backends_have_no_default_formatter`
pins it.

Running the census found two wrong file-type mappings, in the same class as the four the
kernel sweep found — and this time with a formatter behind them:

- **`.inc` was claimed by the C backend.** All 12 `.inc` files in curl are `Makefile.inc`
  and all 3 in the kernel are shell fragments: **0 of 15 is C**. Under `normalize` that
  guess hands a tab-significant file to clang-format.
- **Make fragments were matched by exact filename**, so `Makefile.inc`, `Makefile.am`
  and `Kconfig.debug` all missed the hash backend and fell through to an extension
  lookup. The kernel has 82 more `Makefile.*`. Matching is now on the dotted stem, and
  `.am`/`.ac`/`.in`/`.inc` route to `hash`.

Worth noting how they surfaced: not from reading the routing table, but from a
per-backend breakdown printed by a tool built for something else. A bare total said
"1,038 C files" and looked right.

#### C: normalisation changes nothing either, for a different reason

Rust's census was unmoved because two printers disagree about style. C has no printer at
all — the backend is a lexer emitting spans and nothing regenerates C — so that
explanation cannot apply, and the question becomes whether normalised C is regular
enough to GENERATE the parts we currently store. Measured over curl's 1,026 C files with
`clang-format` 18.1.3:

```
1026 C files — 5 not idempotent, 0 not byte-exact after normalising
  as committed   whitespace-only trivia: 12,591 in 16 distinct forms; top 5 cover 99.8%
  normalised     whitespace-only trivia: 12,586 in 13 distinct forms; top 5 cover 99.9%
  trivia carrying a comment: 13.2% -> 13.1%
```

**The whitespace was already regular.** 87% of trivia nodes are pure whitespace drawn
from 16 forms, five of which cover 99.8% — before normalising. Normalisation removes
three rare forms and adds a tenth of a percentage point. Whatever makes generated trivia
possible for C, it is available today on unmodified source; `fidelity: normalized` is
not what unlocks it. The other 13% carries a comment, which no formatter removes and
which is already stored a second time as a comment node.

Two facts that bear on adopting it anyway. **5 of 1,026 files are not idempotent under
clang-format**, so a repo containing them can never make `normalize --check` go green —
the `normalized` claim would flap. And curl has no `.clang-format`, so normalising it
means rewriting **959 of 1,026 files** into LLVM style: adopting a formatter's defaults
wholesale is the real cost, not the projection change.

**What this does not settle.** The premise was about *finer* nodes, and every number
above is at today's item granularity. A proxy — distinct line-indent forms, standing in
for the sub-item separators finer nodes would need — was also unmoved (63 -> 65 forms,
top 8 covering 91.4% -> 92.1%). But that metric measures whether separators are
*enumerable*, and a formatter's indent is a function of nesting depth and paren
alignment: computable without being enumerable. So it is evidence against enumerating
them and no evidence either way about deriving them. Settling that means building the
printer, which is the same conclusion the Rust census reached from the other direction.

### Result: the Linux kernel, streamed from its tarball

94,750 files / 1,615 MB, ingested in **107.9 s** (15.0 MB/s, 878 files/s) **without
unpacking the archive** — the projection takes a path string and a content string, so a
tar entry feeds it exactly as a file does.

| backend | files | nodes | comments | bound |
|---|---|---|---|---|
| c-family | 81,386 | 81,355 | 2,494,696 | **0** |
| hash | 12,761 | 12,752 | 114,648 | **0** |
| markup | 130 | 130 | 2,007 | 0 |
| rust | 473 | 18,474 | 53,674 | 53,669 |

**Byte-exact: 94,750 of 94,750. Zero mismatches.** The prediction held — fidelity does
not depend on having a grammar, because the fallback's single-node cover is total by
construction. Non-UTF-8 files surfaced as predicted, though only 7 of them rather than
the thousands expected.

The finding that matters is the `bound` column. **C is 86% of the kernel and binds zero
comments** — 2.5 million of them are captured, positioned, and attached to nothing,
because the fallback has no grammar and reports no anchors rather than guessing. Across
the whole tree 97.9% of comments are unbound. Rust binds 53,669 of 53,674 because it
has a real backend.

So a C backend is the single highest-value addition, and the benchmark says what it is
worth: it would move 2.5 million comments from "stored" to "queryable". Throughput is
not the constraint — 15 MB/s on the fallback against 1.5 MB/s for `syn` means a
grammar-based C backend would be slower, and at under two minutes for the kernel there
is room to spend it.

### File-type sweep: what the kernel is made of

| type | files | MB | backend | comments | bound |
|---|---|---|---|---|---|
| `.c` | 36,922 | 728 | c-family | 1,259,966 | **0** |
| `.h` | 26,871 | 704 | c-family | 1,097,444 | **0** |
| `.yaml` | 5,665 | 17 | hash | 47,683 | 0 |
| `.rst` | 4,011 | 31 | rst | 14,842 | 0 |
| `.dts`/`.dtsi`/`.dtso` | 6,703 | 49 | c-family | 89,501 | 0 |
| `Makefile` | 3,194 | 3 | hash | 13,306 | 0 |
| `Kconfig` | 1,828 | 7 | hash | 5,062 | 0 |
| `.S` | 1,337 | 9 | c-family | 28,764 | 0 |
| `.rs` | 473 | 6 | rust | 53,674 | **53,669** |

**`.c` + `.h` is 63,793 files and 2.36 million unbound comments** — 88% of the tree's
commentary, captured and attached to nothing. That is the C backend's value stated as a
number, and it dwarfs everything else on the list.

The sweep also reports files where a backend found *no* comments at all, which
distinguishes a wrong marker guess from a genuinely undocumented corpus. It found four,
all now fixed:

- **`.rst`** was on C-family markers, so its `..` comments were invisible and `//` inside
  prose produced accidental hits. On the right markers it goes from 6,203 wrong hits to
  **14,842 real ones**.
- **`Kbuild`**, **`config`**, **`defconfig`** are Makefile- and config-shaped and were
  landing on C-family — 440 files whose `#` comments were silently missed.
- **`.json`**, **`.txt`** have no comment syntax at all. On C-family markers every `//`
  in a string or URL was a false comment; they now map to a `none` family, because "this
  format has no comments" is an answer rather than a guess to be improved later.

Worth noting how those were found: not by reading the mapping table and thinking
harder, but by measuring which types produced suspiciously zero comments. The signal
came from the corpus.

### C backend, and the ingest that got 15x faster

`projection/c.rs` is a lexer, not a parser — the projection needs byte ranges and never
regenerates, so a preprocessor is not required. Lexing C correctly is, and the failure
mode of getting it wrong is silent: a miscounted brace shifts every later boundary
without an error.

The gotchas were checked against the kernel rather than assumed. **Directives with
unbalanced braces are the critical one and are common** — `# define
randomized_struct_fields_start struct {` opens a brace that never closes, so directives
are lexed as opaque regions and never depth-counted. Block comments do not nest in C
(the first `*/` closes). Backslash-continued `//` comments are legal, handled, and
occurred **zero** times in the sample — a predicted gotcha that turned out not to
matter. Apostrophes inside comments are not character literals.

Result on the kernel: **70,506 C files, 2,257,988 of 2,259,710 comments bound to a
syntax element** — from zero. That is 88% of the tree's commentary moving from stored
to queryable, and it is what the sweep predicted a C backend was worth.

| | before | after |
|---|---|---|
| kernel ingest | 106 s (fallback, no C) | **7.2 s** |
| throughput | 15 MB/s | **225.6 MB/s** |
| comments bound | 53,669 | **2,311,657** |

Three things got it there, and only one was the obvious one.

**Parallelism was not the fix.** Ingest is per-file independent, so a bounded channel
feeding N workers is the right shape — the channel must be bounded, or the reader
buffers 1.6 GB. But threading alone changed nothing, because a single 24 MB generated
header was a single work item taking minutes. No thread count divides one file.

**`project` was quadratic.** It recomputed a node's line by counting newlines across
the whole accumulated text, per node: 2 MB cost 5.0 s, 4 MB 18.4 s, 8 MB 72.6 s.
Carrying the count forward made the same file 2 ms, 4 ms and 10 ms. This is the **same
defect fixed in `bind` an hour earlier** — a line number derived by scanning from the
beginning — reintroduced in the function beside it. Projection is the VFS read path, so
it would have been worse than slow.

**`bind`'s container search** scanned a whole anchor prefix per comment. Anchors nest,
so the innermost container is the last one in start order; a backward scan stops
immediately.

The lesson repeats the one from the first benchmark: the fix that looked obvious
(threads) was worth nothing, and the fix that mattered was a line-counting loop nobody
would look at twice. Measuring found it; reasoning had already missed it once.

**Not expected to break:** byte-exactness, in any language. If a mismatch appears, the
span cover is wrong and that is a real defect rather than a missing backend.