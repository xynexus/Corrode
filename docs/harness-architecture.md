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
     commits: 19% of mutations are inserts, 0 rebalances.
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
   - **7f [unwired] Project.** The VFS reading files *from* the graph rather than the
     disk. Composition is proven byte-exact in both directions; nothing calls it yet.

8. **[blocking 7 at scale]** **Store throughput.** Not in the original ordering because
   nothing had measured it. curl ingests at 2,847 nodes/s warm with 14.3x on-disk
   amplification, rising with store size — the in-memory pipeline is ~100x faster.
   Extrapolated to the kernel that is over an hour and ~23 GB, so large-tree ingest is
   not viable as written. Also unfixed: a single token over LMDB's 511-byte max key
   (base64, minified JS) rejects a whole node write.

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