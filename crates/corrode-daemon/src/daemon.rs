//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change.
//!
//! The daemon owns the host-side state the handlers reach through `&self`: the
//! swarm, the role->model assignments, the embedded graph store (HelixDB, when
//! built), and the VFS.

use crate::approval::ApprovalGate;
use crate::dialect::Dialects;
use crate::graph::GraphStore;
use crate::hipfire::Client;
use crate::plan_graph;
use crate::planner;
use crate::roles::{self, Role, RoleModels};
use crate::skills::SkillContext;
use crate::swarm::{Swarm, Task};
use crate::terminal::Terminals;
use crate::toolcall::ToolCaller;
use crate::tools::ToolBox;
use crate::vfs::Vfs;
use corrode_core::{AgentCommand, AgentEvent, Priority};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// How many relevance-ranked skills to surface in the shared prefix per turn.
const TOP_K_SKILLS: usize = 8;

pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    /// Embedded HelixDB. `None` unless built with `--features helix` and opened.
    graph: Option<Box<dyn GraphStore>>,
    /// Repo VFS. `Arc` so the tool-execution loop's `'static` future can own a clone.
    vfs: Arc<dyn Vfs>,
    /// Live pty-backed terminal sessions.
    terminals: Terminals,
    /// Agent Skills + AGENTS.md + the embedded index (fed into the shared prefix).
    skills: SkillContext,
    /// Reliable tool-calling for small models (the Needle shim). `None` unless built
    /// with `--features needle` and the assets were found; drives task emission.
    tool_caller: Option<Arc<dyn ToolCaller>>,
    /// Human-in-the-loop gate for mutating tool calls (write_file / run_command).
    /// Shared with the command loop, which resolves `ApprovalResponse`s.
    approvals: Arc<ApprovalGate>,
    /// Repo root — the working directory for `run_command` in the tool loop.
    repo_root: PathBuf,
    /// Monotonic id source for plan (provenance root) nodes, one per Prompt turn.
    next_plan_id: std::sync::atomic::AtomicU64,
    /// Skill name -> skill dir, for `run_skill_script` in the tool loop. Derived from
    /// `skills` at construction; `Arc` so the tool-loop future owns a clone.
    skill_scripts: Arc<std::collections::HashMap<String, PathBuf>>,
    /// Per-model tool dialects (schema/names/parse), matched to the tool-call model.
    dialects: Arc<Dialects>,
}

impl Daemon {
    pub fn new(
        swarm: Swarm,
        roles: RoleModels,
        graph: Option<Box<dyn GraphStore>>,
        vfs: Arc<dyn Vfs>,
        skills: SkillContext,
        tool_caller: Option<Arc<dyn ToolCaller>>,
        repo_root: PathBuf,
        dialects: Arc<Dialects>,
    ) -> Self {
        let skill_scripts = Arc::new(skills.script_dirs());
        Self {
            swarm,
            roles,
            graph,
            vfs,
            terminals: Terminals::new(),
            skills,
            tool_caller,
            approvals: Arc::new(ApprovalGate::default()),
            repo_root,
            next_plan_id: std::sync::atomic::AtomicU64::new(0),
            skill_scripts,
            dialects,
        }
    }

    /// Run until the command channel closes. Dropping the sender ends the loop,
    /// which drops `events` and unblocks the consumer.
    ///
    /// `Prompt` handling is dispatched concurrently (it can be long-lived and may block
    /// on a human approval), so the loop keeps receiving — crucially, the
    /// `ApprovalResponse` that unblocks a waiting tool call. Other commands (terminal
    /// I/O, approvals) are handled inline to preserve their ordering.
    pub async fn run(
        self: Arc<Self>,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        while let Some(cmd) = commands.recv().await {
            if matches!(cmd, AgentCommand::Prompt { .. }) {
                let this = Arc::clone(&self);
                let events = events.clone();
                tokio::spawn(async move { this.handle(cmd, &events).await });
            } else {
                self.handle(cmd, &events).await;
            }
        }
    }

    async fn handle(&self, cmd: AgentCommand, events: &mpsc::Sender<AgentEvent>) {
        match cmd {
            AgentCommand::Prompt { text, priority } => {
                let (subtasks, prefix) = match self.plan(&text, priority).await {
                    Ok(planned) => planned,
                    Err(e) => {
                        let _ = events
                            .send(AgentEvent::Error {
                                message: format!("planning failed: {e}"),
                            })
                            .await;
                        return;
                    }
                };

                // Seed the reactive plan graph with the decomposition, rooted at a plan
                // node. The scheduler fans ready tasks to the swarm and grows the graph
                // as agents emit follow-up work (a test contract, a research spin-off) —
                // dataflow, not a fixed fan-out. Concurrency is bounded by the swarm's
                // inflight semaphore inside `execute`, not here.
                let plan_id = format!(
                    "plan-{}",
                    self.next_plan_id
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                let mut graph = plan_graph::PlanGraph::new(&plan_id);
                for s in subtasks {
                    graph.add(s.role, s.prompt, Vec::new());
                }

                plan_graph::run_reactive(&mut graph, |task| {
                    let client = self.swarm.client();
                    let model = self
                        .roles
                        .model_for(task.role)
                        .unwrap_or_default()
                        .to_string();
                    let band = planner::band_for(task.role);
                    let prefix = prefix.clone();
                    let events = events.clone();
                    let tool_caller = self.tool_caller.clone();
                    let vfs = Arc::clone(&self.vfs);
                    let approvals = Arc::clone(&self.approvals);
                    let root = self.repo_root.clone();
                    let skill_scripts = Arc::clone(&self.skill_scripts);
                    let dialects = Arc::clone(&self.dialects);
                    let id = task.id;
                    let role = task.role;
                    let prompt = task.prompt.clone();
                    async move {
                        // Three paths, most capable first:
                        //  1. the model emits its own tool calls (its dialect says so) —
                        //     declare tools on the request and parse the reply directly.
                        //  2. otherwise a small model runs the Needle-mediated loop,
                        //     which reconstructs a call from a plain-English line.
                        //  3. anything else answers single-shot, as before.
                        // Size alone used to decide this, which sent a 1B through Needle
                        // even when it could call tools better itself.
                        // `artifacts` collects the files a tool-loop task wrote (its code
                        // nodes in provenance).
                        let mut artifacts = Vec::new();
                        let role_dialect = dialects.resolve(&model);
                        let output = if role_dialect.emits_own_calls() {
                            run_native_tool_loop(
                                &client,
                                &model,
                                band,
                                role_dialect,
                                ToolBox::new(vfs, root, skill_scripts),
                                &approvals,
                                &prefix,
                                role,
                                &prompt,
                                &events,
                                id,
                                &mut artifacts,
                            )
                            .await
                        } else if let (true, Some(caller)) =
                            (roles::is_small_model(&model), tool_caller.clone())
                        {
                            run_tool_loop(
                                &client,
                                &model,
                                band,
                                caller,
                                ToolBox::new(vfs, root, skill_scripts),
                                &approvals,
                                &dialects,
                                &prefix,
                                role,
                                &prompt,
                                &events,
                                id,
                                &mut artifacts,
                            )
                            .await
                        } else {
                            let full = planner::subagent_prompt(&prefix, role, &prompt);
                            let out = client.respond(&model, &full, band).await;
                            if let Ok(text) = &out {
                                let _ = events
                                    .send(AgentEvent::SubagentOutput {
                                        id,
                                        text: text.clone(),
                                    })
                                    .await;
                            }
                            out
                        };

                        let emitted = match &output {
                            Ok(text) => emit_followups(tool_caller, &dialects, text).await,
                            Err(e) => {
                                let _ = events
                                    .send(AgentEvent::Error {
                                        message: e.to_string(),
                                    })
                                    .await;
                                Vec::new()
                            }
                        };
                        plan_graph::Outcome {
                            output,
                            emitted,
                            artifacts,
                        }
                    }
                })
                .await;

                // Tasks left pending after the scheduler settled had a failed or
                // unmet dependency (a failed emitter) — surface them rather than
                // dropping them silently.
                let stuck = graph.stuck();
                if !stuck.is_empty() {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!(
                                "{} task(s) could not be scheduled (a dependency failed)",
                                stuck.len()
                            ),
                        })
                        .await;
                }

                // Persist the plan's provenance (plan <- task/contract <- code) to the
                // graph store, so the code<->task<->plan lineage is queryable.
                self.persist_provenance(&graph);
            }
            AgentCommand::DocQuery { question } => {
                let ev = match &self.graph {
                    Some(g) => match g.doc_search(&question, 8) {
                        // ponytail: grounding ids only for now; the GraphRAG answer
                        // (retrieve -> synthesize via hipfire) fills `text` next.
                        Ok(ids) => AgentEvent::DocAnswer {
                            text: String::new(),
                            grounded_on: ids,
                        },
                        Err(e) => AgentEvent::Error { message: e.to_string() },
                    },
                    None => AgentEvent::Error {
                        message: "DocQuery unavailable: build with --features helix and open a graph store".into(),
                    },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::ListDir { path } => {
                let ev = match self.vfs.list(&path).await {
                    Ok(entries) => AgentEvent::DirListing { path, entries },
                    Err(e) => AgentEvent::Error { message: e.to_string() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::TerminalInput { session, data } => {
                // Write keystrokes to the session's real pty; its output streams back
                // as TerminalOutput from the session's reader thread.
                if let Err(e) = self.terminals.input(&session, &data, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal input: {e}"),
                        })
                        .await;
                }
            }
            AgentCommand::TerminalResize {
                session,
                cols,
                rows,
            } => {
                if let Err(e) = self.terminals.resize(&session, cols, rows, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal resize: {e}"),
                        })
                        .await;
                }
            }
            AgentCommand::ApprovalResponse { id, approved } => {
                // Unblock the tool call waiting on this decision (no-op if it already
                // gave up). Handled inline so it lands while a Prompt handler waits.
                self.approvals.resolve(id, approved);
            }
        }
    }

    /// Decompose a prompt into role-tagged subtasks, with the shared context prefix.
    ///
    /// Phase 1: the orchestration model produces a plan (at the request's band).
    /// Phase 2: [`planner::parse_plan`] turns it into role-tagged [`PlannedSubtask`]s.
    /// If the model returns nothing parseable, degrade to a single coder task on the
    /// raw prompt. Returns the subtasks plus the prefix — the caller seeds a
    /// [`plan_graph::PlanGraph`] with the subtasks and prepends the prefix to every
    /// subagent prompt (KV reuse), and the reactive scheduler grows the graph as
    /// agents emit follow-up work.
    async fn plan(
        &self,
        text: &str,
        priority: Priority,
    ) -> anyhow::Result<(Vec<planner::PlannedSubtask>, String)> {
        // Built once and shared, byte-identical, by the planning call and every
        // subagent, so hipfire batches them prefix-shared and reuses KV.
        let prefix = self.context_prefix(text).await;

        let orch_model = self
            .roles
            .model_for(Role::Orchestration)
            .unwrap_or_default()
            .to_string();
        let plan_task = Task {
            prompt: planner::orchestration_prompt(&prefix, text),
            priority,
            model: orch_model,
        };
        let plan_text = self
            .swarm
            .run(vec![plan_task])
            .next()
            .await
            .map(|(_, r)| r)
            .transpose()?
            .unwrap_or_default();

        let mut plan = planner::parse_plan(&plan_text);
        if plan.is_empty() {
            // Degrade to one coder task on the raw prompt (still behind the shared
            // prefix) so a plan the model couldn't structure still gets attempted
            // rather than dropped.
            plan = vec![planner::PlannedSubtask {
                role: Role::Coder,
                prompt: text.to_string(),
            }];
        }
        Ok((plan, prefix))
    }

    /// Persist a plan's provenance subgraph to the graph store, if one is open.
    /// Best-effort: without `--features helix` there's no store (a no-op), and the
    /// HelixDB write path is still stubbed — so on the first write error we log once and
    /// stop rather than spamming. The in-memory provenance is already correct; this is
    /// the durability seam.
    fn persist_provenance(&self, graph: &plan_graph::PlanGraph) {
        let Some(store) = &self.graph else {
            return;
        };
        let prov = graph.provenance();
        for node in &prov.nodes {
            if let Err(e) = store.upsert_node(&node.id, node.kind.as_str(), &node.label) {
                eprintln!("provenance persistence unavailable ({e}); skipping");
                return;
            }
        }
        for edge in &prov.edges {
            if let Err(e) = store.add_edge(&edge.from, &edge.rel, &edge.to) {
                eprintln!("provenance edge persistence unavailable ({e}); skipping");
                return;
            }
        }
    }

    /// The shared context prefix prepended to every prompt in a Prompt turn.
    ///
    /// ponytail: a shallow repo digest (VFS root listing) plus a fixed preamble.
    /// The graph-backed VFS will supply richer, relevance-ranked context here
    /// (hipfire embeddings/rerank picking which nodes) — but the KV-sharing shape
    /// is already right: identical bytes across the whole swarm, task in the tail.
    async fn context_prefix(&self, task: &str) -> String {
        let mut s = String::from(
            "You are a subagent in the Corrode coding-agent swarm working on a shared \
repository.\n",
        );
        // Project rules (AGENTS.md) + skills relevant to this task. Byte-identical
        // across the turn's subagents (same task), so they share the KV prefill.
        let rules = self.skills.agents_rules();
        if !rules.trim().is_empty() {
            s.push_str("\nProject instructions (AGENTS.md):\n");
            s.push_str(rules.trim_end());
            s.push('\n');
        }
        let manifest = self
            .skills
            .prefix_section(task, &self.swarm.client(), TOP_K_SKILLS)
            .await;
        if !manifest.is_empty() {
            s.push('\n');
            s.push_str(&manifest);
        }
        s.push_str("\nRepository root:\n");
        match self.vfs.list("").await {
            Ok(entries) => {
                for e in entries {
                    s.push_str(&format!("  {} ({} bytes)\n", e.path, e.bytes));
                }
            }
            Err(_) => s.push_str("  (listing unavailable)\n"),
        }
        s
    }
}

/// Max tool calls a small model may make before it must answer — a bound on GPU spend
/// and runaway loops.
const MAX_TOOL_STEPS: usize = 6;

/// The Needle-mediated tool-execution loop for a small model.
///
/// Each turn the model responds (streamed as `SubagentOutput`). If it wrote a `TOOL:`
/// line, Needle structures that plain-English intent into a call — the small model
/// never writes JSON — `toolbox` executes it against the repo, and the observation is
/// appended to the scratchpad for the next turn. The loop ends when a turn has no
/// `TOOL:` line (that text is the final answer) or the step budget is spent. Tool and
/// Needle errors come back as observations (the model can recover), not hard failures;
/// only a model-generation error aborts the loop.
#[allow(clippy::too_many_arguments)]
/// Run one tool call: mutating ones (write_file / run_command / run_skill_script) pass
/// through the human approval gate first, read-only ones go straight through. A
/// successful write authors a code node, so its path is recorded for provenance.
///
/// Shared by both loops — the gate must not depend on how the call was produced.
async fn gate_and_execute(
    call: &crate::toolcall::ToolCall,
    toolbox: &ToolBox,
    approvals: &ApprovalGate,
    events: &mpsc::Sender<AgentEvent>,
    written: &mut Vec<String>,
) -> String {
    if crate::tools::is_mutating(call)
        && !approvals
            .request(events, crate::tools::describe(call))
            .await
    {
        return format!("denied: {} was not approved", crate::tools::describe(call));
    }
    let observation = toolbox.execute(call).await;
    if call.name == "write_file" && observation.starts_with("wrote") {
        if let Some(path) = call.arguments.get("path").and_then(|p| p.as_str()) {
            written.push(path.to_string());
        }
    }
    observation
}

/// The tool loop for models that emit their own calls.
///
/// The tools are declared on the request, so hipfire's chat template renders the block
/// the model was trained to read and it answers in its own syntax — which the dialect
/// parses directly. No Needle: nothing has to reconstruct the call from prose, so the
/// multi-param and truncation failures that motivated the Needle finetune cannot occur.
///
/// Thinking defaults to off (`CORRODE_REASONING_EFFORT` overrides): with reasoning on,
/// these models talk themselves out of calling — measured on MiniCPM5-1B, which
/// deliberated past its budget instead of emitting a call it had already chosen.
#[allow(clippy::too_many_arguments)]
async fn run_native_tool_loop(
    client: &Client,
    model: &str,
    band: Priority,
    dialect: &crate::dialect::ToolDialect,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
) -> anyhow::Result<String> {
    let tools = dialect.request_tools(crate::tools::EXEC_TOOLS);
    let effort = std::env::var("CORRODE_REASONING_EFFORT").unwrap_or_else(|_| "none".to_string());
    let mut scratchpad = String::new();
    let mut last = String::new();
    for _ in 0..MAX_TOOL_STEPS {
        let prompt = planner::native_tool_prompt(prefix, role, task, &scratchpad);
        let (text, _reasoning) = client
            .respond_full(model, &prompt, band, Some(&tools), Some(&effort))
            .await?;
        let _ = events
            .send(AgentEvent::SubagentOutput {
                id,
                text: text.clone(),
            })
            .await;
        last = text.clone();

        let calls = dialect.parse(&text).unwrap_or_default();
        let Some(call) = calls.first() else {
            return Ok(text); // no call -> this turn is the final answer
        };
        let observation = gate_and_execute(call, &toolbox, approvals, events, written).await;
        scratchpad.push_str(&format!(
            "\nCALLED: {}\nRESULT: {observation}\n",
            crate::tools::describe(call)
        ));
    }
    Ok(last)
}

async fn run_tool_loop(
    client: &Client,
    model: &str,
    band: Priority,
    caller: Arc<dyn ToolCaller>,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    dialects: &Dialects,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
) -> anyhow::Result<String> {
    // Render the exec toolset in the tool-call model's dialect once; parse each reply
    // with the same dialect (which maps its tool names back to canonical).
    let dialect = dialects.resolve(caller.model_id());
    let schema = dialect.render(crate::tools::EXEC_TOOLS);
    let mut scratchpad = String::new();
    let mut last = String::new();
    for _ in 0..MAX_TOOL_STEPS {
        let prompt = planner::tool_loop_prompt(prefix, role, task, &scratchpad);
        let text = client.respond(model, &prompt, band).await?;
        let _ = events
            .send(AgentEvent::SubagentOutput {
                id,
                text: text.clone(),
            })
            .await;
        last = text.clone();

        let Some(intent) = crate::tools::parse_tool_intent(&text) else {
            return Ok(text); // no TOOL: line -> this turn is the final answer
        };

        // The tool-caller turns the plain-English intent into a structured call
        // (spawn_blocking: Needle inference is synchronous and CPU-bound); the dialect
        // parses the raw reply.
        let query = intent.clone();
        let toolcaller = caller.clone();
        let schema = schema.clone();
        let raw = tokio::task::spawn_blocking(move || toolcaller.generate(&query, &schema)).await;
        let observation = match raw.map(|r| r.and_then(|raw| dialect.parse(&raw))) {
            Ok(Ok(calls)) => match calls.first() {
                Some(c) => gate_and_execute(c, &toolbox, approvals, events, written).await,
                None => "error: no tool call produced".to_string(),
            },
            Ok(Err(e)) => format!("error: tool-call construction failed: {e}"),
            Err(e) => format!("error: tool-call thread panicked: {e}"),
        };
        scratchpad.push_str(&format!("\nTOOL: {intent}\nRESULT: {observation}\n"));
    }
    // Step budget spent: hand back the last turn as the answer.
    Ok(last)
}

/// Extract the follow-up task a subagent proposed in its reply.
///
/// The agent writes a plain-English `NEXT:` line (see [`planner::subagent_prompt`]) —
/// easy even for a small model, which can't hand-write a tool call. We then need the
/// task's *role* (which band/model runs it). With a Needle tool-caller present, Needle
/// classifies the instruction into one of the role tools ([`plan_graph::ROLE_TASK_TOOLS`]);
/// tool selection is what it's trained for, so this is reliable — unlike trusting a
/// small model to emit structured JSON. The task text stays verbatim from the `NEXT:`
/// line (Needle's own `task` argument truncates). One instruction per reply → one task;
/// the reactive graph chains the rest.
///
/// Without a caller (base build, or assets absent) or on a Needle error, the task still
/// queues, defaulted to the Coder role. No `NEXT:` line → no emission. Needle inference
/// is synchronous and CPU-bound, so it runs on a blocking thread.
async fn emit_followups(
    tool_caller: Option<Arc<dyn ToolCaller>>,
    dialects: &Dialects,
    output: &str,
) -> Vec<plan_graph::Emit> {
    let Some(instruction) = plan_graph::parse_next_instruction(output) else {
        return Vec::new(); // no follow-up proposed
    };

    let role = match tool_caller {
        Some(caller) => {
            // Render the role tools in the caller's dialect, classify, parse back.
            let dialect = dialects.resolve(caller.model_id());
            let schema = dialect.render(plan_graph::ROLE_TOOLS);
            let query = instruction.clone();
            let raw =
                tokio::task::spawn_blocking(move || caller.generate(&query, &schema)).await;
            match raw.map(|r| r.and_then(|raw| dialect.parse(&raw))) {
                Ok(Ok(calls)) => plan_graph::role_from_tool_calls(&calls).unwrap_or(Role::Coder),
                Ok(Err(e)) => {
                    eprintln!("role classification failed ({e}); defaulting to coder");
                    Role::Coder
                }
                Err(e) => {
                    eprintln!("role classification thread panicked ({e}); defaulting to coder");
                    Role::Coder
                }
            }
        }
        None => Role::Coder, // no classifier: queue it as a coder task
    };

    vec![plan_graph::Emit {
        role,
        prompt: instruction,
        after_emitter: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::Client;
    use crate::vfs::PassthroughVfs;

    fn test_daemon() -> Daemon {
        Daemon::new(
            Swarm::new(Client::new("http://127.0.0.1:1", None), 1),
            RoleModels::uniform("test-model"),
            None,
            Arc::new(PassthroughVfs::new(std::env::temp_dir())),
            SkillContext::default(),
            None,
            std::env::temp_dir(),
            Arc::new(Dialects::default()),
        )
    }

    /// The `fixtures/demo-repo` submodule (`xynexus/corrode-demo`) — the deterministic
    /// repo the e2e tests run against: real files (`src/lib.rs`), real rules
    /// (`AGENTS.md`), a real skill (`.agents/skills/run-tests`). `None` when it isn't
    /// checked out (`git submodule update --init fixtures/demo-repo`), so the tests skip
    /// rather than fail on a missing fixture.
    fn demo_repo() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/demo-repo");
        dir.join("src/lib.rs").exists().then_some(dir)
    }

    /// The fixture is wired up, through the same `SkillContext` the daemon builds:
    /// discovery finds the bundled `run-tests` skill, it reaches the shared prefix, its
    /// scripts are resolvable, and `AGENTS.md` is read. Runs on the base build (no
    /// embedding model -> no hipfire call), so a moved path or an uninitialized submodule
    /// fails here rather than inside an ignored test.
    #[tokio::test]
    async fn demo_repo_fixture_is_discoverable() {
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let client = Client::new("http://127.0.0.1:1", None);
        let skills = SkillContext::build(&repo, &client, None).await;
        let section = skills
            .prefix_section("run the tests", &client, TOP_K_SKILLS)
            .await;
        assert!(section.contains("run-tests"), "got: {section}");
        assert!(skills.script_dirs().contains_key("run-tests"));
        let rules = skills.agents_rules();
        assert!(rules.contains("cargo test"), "got: {rules}");
    }

    // Real Needle emission, end-to-end through `emit_followups` (query wrapper +
    // grammar-guided decode + mapping). Ignored by default (needs the weights):
    //   cargo test -p corrode-daemon --features needle -- --ignored --nocapture
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_classifies_the_next_line_role_and_keeps_the_text() {
        use crate::toolcall::needle::NeedleToolCaller;
        let caller: Arc<dyn ToolCaller> = Arc::new(
            NeedleToolCaller::load_from_env()
                .expect("load Needle")
                .expect("Needle assets present"),
        );
        // A subagent's reply against the fixture repo, ending with a plain-English NEXT:
        // line describing a review follow-up (mathkit's standing is_prime task).
        let reply = "Rewrote is_prime in src/lib.rs to trial-divide up to sqrt(n).\n\
            NEXT: review the new is_prime implementation in src/lib.rs for correctness";
        let emits = emit_followups(Some(caller), &Dialects::default(), reply).await;
        let summary: Vec<_> = emits
            .iter()
            .map(|e| (e.role, e.prompt.as_str()))
            .collect();
        eprintln!("emitted: {summary:?}");
        assert_eq!(emits.len(), 1, "one NEXT: line -> one task");
        // Task text is verbatim from the NEXT: line (not Needle's truncated arg).
        assert_eq!(
            emits[0].prompt,
            "review the new is_prime implementation in src/lib.rs for correctness"
        );
        // Needle classifies the role from the instruction (tool selection).
        assert_eq!(emits[0].role, Role::Review);
    }

    // No NEXT: line -> no emission (and Needle isn't even called).
    #[tokio::test]
    async fn no_next_line_emits_nothing() {
        let emits = emit_followups(
            None,
            &Dialects::default(),
            "All done. The tests pass and nothing remains.",
        )
        .await;
        assert!(emits.is_empty());
    }

    // The tool-execution path a small model takes: a plain-English intent -> Needle
    // builds the call -> the ToolBox runs it against the repo. (The full loop also
    // needs hipfire for the model turns; this covers the Needle+tool half.)
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_builds_a_tool_call_that_the_toolbox_executes() {
        use crate::dialect::ToolDialect;
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(std::collections::HashMap::new()),
        );

        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let dialect = ToolDialect::default();
        let raw = caller
            .generate("read the file src/lib.rs", &dialect.render(EXEC_TOOLS))
            .expect("generation");
        let calls = dialect.parse(&raw).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("a tool call");
        assert_eq!(call.name, "read_file");

        // mathkit's source, straight out of the fixture repo.
        let observation = toolbox.execute(call).await;
        assert!(observation.contains("pub fn is_prime"), "got: {observation}");
    }

    // Stage 3 end-to-end against the fixture repo: discovery finds its bundled
    // `run-tests` skill -> a plain-English ask -> Needle builds a run_skill_script call
    // -> the ToolBox resolves the skill and runs `scripts/test.sh` (a real `cargo test`
    // over mathkit, so the first run compiles the fixture).
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_runs_a_skill_script_end_to_end() {
        use crate::dialect::ToolDialect;
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let client = Client::new("http://127.0.0.1:1", None);
        let skills = SkillContext::build(&repo, &client, None).await.script_dirs();
        assert!(skills.contains_key("run-tests"), "got: {skills:?}");
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(skills),
        );

        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let dialect = ToolDialect::default();
        let raw = caller
            .generate(
                "run the test.sh script from the run-tests skill",
                &dialect.render(EXEC_TOOLS),
            )
            .expect("generation");
        let calls = dialect.parse(&raw).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("a tool call");
        assert_eq!(call.name, "run_skill_script");

        // The call is mutating (executes code) -> would gate on approval in the loop.
        assert!(crate::tools::is_mutating(call));
        let observation = toolbox.execute(call).await;
        assert!(observation.starts_with("exit 0:"), "got: {observation}");
        assert!(observation.contains("test result: ok"), "got: {observation}");
    }

    // The whole loop, for real: a Prompt goes into the daemon, hipfire plans it, the
    // swarm runs the subtasks against the demo repo, and events stream back — no mocks
    // on any leg. Needs a live hipfire (HIPFIRE_BASE_URL, CORRODE_MODEL) plus the Needle
    // assets and the submodule:
    //   HIPFIRE_BASE_URL=http://host:11435 CORRODE_MODEL=<id> \
    //     cargo test -p corrode-daemon --features needle -- --ignored --nocapture
    //
    // ponytail: the turn is "done" when events go quiet — there's no turn-complete event
    // to wait on. Add one (AgentEvent::TurnComplete) and this drops the idle timeout.
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires a live hipfire + Needle assets + the demo-repo submodule"]
    async fn a_prompt_runs_the_swarm_against_the_demo_repo() {
        use std::time::Duration;
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let base_url = std::env::var("HIPFIRE_BASE_URL")
            .unwrap_or_else(|_| crate::hipfire::DEFAULT_BASE_URL.to_string());
        let client = Client::new(&base_url, std::env::var("HIPFIRE_API_KEY").ok());
        let Ok(models) = client.list_models().await else {
            eprintln!("skipped: no hipfire at {base_url}");
            return;
        };
        // The model under test: CORRODE_MODEL, else whatever hipfire serves first.
        let model = std::env::var("CORRODE_MODEL")
            .ok()
            .or_else(|| models.first().cloned())
            .expect("hipfire serves at least one model");
        eprintln!("model: {model} (small: {})", roles::is_small_model(&model));

        let embed = roles::default_embedding_model(&models).map(str::to_string);
        let skills = SkillContext::build(&repo, &client, embed).await;
        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let daemon = Arc::new(Daemon::new(
            Swarm::new(client, 4),
            RoleModels::uniform(&model),
            None,
            Arc::new(PassthroughVfs::new(&repo)),
            skills,
            Some(Arc::new(caller)),
            repo.clone(),
            // load() so CORRODE_TOOL_DIALECTS can put a model on its native dialect
            Arc::new(Dialects::load()),
        ));

        let (ctx, crx) = mpsc::channel(16);
        let (etx, mut erx) = mpsc::channel(64);
        tokio::spawn(Arc::clone(&daemon).run(crx, etx));
        ctx.send(AgentCommand::Prompt {
            text: "Report which functions src/lib.rs defines.".into(),
            priority: Priority::Default,
        })
        .await
        .unwrap();

        // Drain until the turn settles, approving any mutating call the swarm proposes
        // (unattended: nothing here would answer the gate otherwise, and it fails closed).
        let (mut outputs, mut approvals, mut errors) = (Vec::new(), Vec::new(), Vec::new());
        while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_secs(120), erx.recv()).await {
            match ev {
                AgentEvent::SubagentOutput { id, text } => {
                    eprintln!("--- subagent {id} ---\n{text}\n");
                    outputs.push(text);
                }
                AgentEvent::ApprovalRequest { id, action } => {
                    eprintln!("--- approval {id}: {action} -> approving");
                    approvals.push(action);
                    ctx.send(AgentCommand::ApprovalResponse { id, approved: true })
                        .await
                        .unwrap();
                }
                AgentEvent::Error { message } => {
                    eprintln!("--- error: {message}");
                    errors.push(message);
                }
                other => eprintln!("--- {other:?}"),
            }
        }
        drop(ctx);

        eprintln!(
            "settled: {} outputs, {} approvals, {} errors",
            outputs.len(),
            approvals.len(),
            errors.len()
        );
        assert!(errors.is_empty(), "daemon reported errors: {errors:?}");
        assert!(!outputs.is_empty(), "the swarm produced no subagent output");
        // Whether the small model actually drove a tool is a property of the model +
        // the tool-loop prompt, not of this wiring — reported above, asserted in the
        // Needle tests, deliberately not asserted here.
    }

    // The native path, end to end and Needle-free: Corrode declares its own tools on the
    // request, hipfire's template renders the model's `<tools>` block, the model emits
    // its own XML call, and the dialect parses it straight into a ToolCall the ToolBox
    // runs. This is the loop the Needle shim exists to substitute for — here it isn't
    // in it at all.
    //   HIPFIRE_BASE_URL=http://127.0.0.1:11435 CORRODE_MODEL=<id> \
    //     cargo test -p corrode-daemon -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "requires a live hipfire + the demo-repo submodule"]
    async fn a_model_drives_its_own_tool_call_without_needle() {
        use crate::dialect::{ParseFormat, SchemaFormat, ToolDialect};
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let base_url = std::env::var("HIPFIRE_BASE_URL")
            .unwrap_or_else(|_| crate::hipfire::DEFAULT_BASE_URL.to_string());
        let client = Client::new(&base_url, std::env::var("HIPFIRE_API_KEY").ok());
        let Ok(models) = client.list_models().await else {
            eprintln!("skipped: no hipfire at {base_url}");
            return;
        };
        let model = std::env::var("CORRODE_MODEL")
            .ok()
            .or_else(|| models.first().cloned())
            .expect("hipfire serves a model");

        // Corrode's canonical tools, rendered for a chat model. The `type`/`function`
        // envelope is what the chat templates serialize with `tool | tojson`.
        let rendered: serde_json::Value =
            serde_json::from_str(&ToolDialect::new(
                SchemaFormat::OpenAiNested,
                ParseFormat::MiniCpmXml,
                std::collections::HashMap::new(),
            )
            .render(EXEC_TOOLS))
            .expect("rendered tools are JSON");
        let tools = serde_json::Value::Array(
            rendered
                .as_array()
                .unwrap()
                .iter()
                .map(|t| serde_json::json!({"type": "function", "function": t}))
                .collect(),
        );

        // Thinking off: the model deliberates itself out of calling when it is on, and
        // a direct imperative is what the tool loop should be issuing anyway.
        let (answer, reasoning) = client
            .respond_full(
                &model,
                "Read the file src/lib.rs.",
                Priority::Default,
                Some(&tools),
                Some("none"),
            )
            .await
            .expect("generation");
        eprintln!("answer: {answer}\nreasoning: {} chars", reasoning.len());

        let dialect = ToolDialect::new(
            SchemaFormat::OpenAiNested,
            ParseFormat::MiniCpmXml,
            std::collections::HashMap::new(),
        );
        let calls = dialect.parse(&answer).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("the model emitted a tool call");
        assert_eq!(call.name, "read_file");

        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(std::collections::HashMap::new()),
        );
        let observation = toolbox.execute(call).await;
        assert!(observation.contains("pub fn is_prime"), "got: {observation}");
    }

    // The hipfire-free dispatch path: DocQuery without a graph store reports itself
    // unavailable (Error) rather than hanging or panicking. Guards the match, not the
    // network. (The real pty terminal path is covered in `terminal.rs`.)
    #[tokio::test]
    async fn loop_reports_docquery_unavailable_without_graph() {
        let daemon = test_daemon();
        let (ctx, crx) = mpsc::channel(8);
        let (etx, mut erx) = mpsc::channel(8);

        ctx.send(AgentCommand::DocQuery {
            question: "q".into(),
        })
        .await
        .unwrap();
        drop(ctx);

        Arc::new(daemon).run(crx, etx).await;

        assert!(matches!(
            erx.recv().await.unwrap(),
            AgentEvent::Error { .. }
        ));
    }
}
