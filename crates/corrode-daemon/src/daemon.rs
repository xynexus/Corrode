//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change.
//!
//! The daemon owns the host-side state the handlers reach through `&self`: the
//! swarm, the role->model assignments, the embedded graph store (HelixDB, when
//! built), and the VFS.

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
}

impl Daemon {
    pub fn new(
        swarm: Swarm,
        roles: RoleModels,
        graph: Option<Box<dyn GraphStore>>,
        vfs: Arc<dyn Vfs>,
        skills: SkillContext,
        tool_caller: Option<Arc<dyn ToolCaller>>,
    ) -> Self {
        Self {
            swarm,
            roles,
            graph,
            vfs,
            terminals: Terminals::new(),
            skills,
            tool_caller,
        }
    }

    /// Run until the command channel closes. Dropping the sender ends the loop,
    /// which drops `events` and unblocks the consumer.
    pub async fn run(
        &self,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        while let Some(cmd) = commands.recv().await {
            self.handle(cmd, &events).await;
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

                // Seed the reactive plan graph with the decomposition. The scheduler
                // fans ready tasks to the swarm and grows the graph as agents emit
                // follow-up work (a test contract, a research spin-off) — dataflow,
                // not a fixed fan-out. Concurrency is bounded by the swarm's inflight
                // semaphore inside `execute`, not here.
                let mut graph = plan_graph::PlanGraph::default();
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
                    let id = task.id;
                    let role = task.role;
                    let prompt = task.prompt.clone();
                    async move {
                        // Small models can't construct tool-call JSON, so they run the
                        // Needle-mediated tool-execution loop. Larger models (or a build
                        // without a Needle caller) do a single-shot response as before.
                        let output = if let (true, Some(caller)) =
                            (roles::is_small_model(&model), tool_caller.clone())
                        {
                            run_tool_loop(
                                &client,
                                &model,
                                band,
                                caller,
                                ToolBox::new(vfs),
                                &prefix,
                                role,
                                &prompt,
                                &events,
                                id,
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
                            Ok(text) => emit_followups(tool_caller, text).await,
                            Err(e) => {
                                let _ = events
                                    .send(AgentEvent::Error {
                                        message: e.to_string(),
                                    })
                                    .await;
                                Vec::new()
                            }
                        };
                        plan_graph::Outcome { output, emitted }
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
async fn run_tool_loop(
    client: &Client,
    model: &str,
    band: Priority,
    caller: Arc<dyn ToolCaller>,
    toolbox: ToolBox,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
) -> anyhow::Result<String> {
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

        // Needle turns the plain-English intent into a structured call (spawn_blocking:
        // Needle inference is synchronous and CPU-bound).
        let query = intent.clone();
        let caller = caller.clone();
        let observation = match tokio::task::spawn_blocking(move || {
            caller.call(&query, crate::tools::TOOL_SCHEMAS)
        })
        .await
        {
            Ok(Ok(calls)) => match calls.first() {
                Some(c) => toolbox.execute(c).await,
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
    output: &str,
) -> Vec<plan_graph::Emit> {
    let Some(instruction) = plan_graph::parse_next_instruction(output) else {
        return Vec::new(); // no follow-up proposed
    };

    let role = match tool_caller {
        Some(caller) => {
            let query = instruction.clone();
            match tokio::task::spawn_blocking(move || {
                caller.call(&query, plan_graph::ROLE_TASK_TOOLS)
            })
            .await
            {
                Ok(Ok(calls)) => plan_graph::role_from_tool_calls(&calls).unwrap_or(Role::Coder),
                Ok(Err(e)) => {
                    eprintln!("Needle role classification failed ({e}); defaulting to coder");
                    Role::Coder
                }
                Err(e) => {
                    eprintln!("Needle role classification thread panicked ({e}); defaulting to coder");
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
        )
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
        // Reply ends with a plain-English NEXT: line describing a review follow-up.
        let reply = "Added the auth middleware in auth.rs.\n\
            NEXT: review the token-expiry logic in the auth middleware for correctness";
        let emits = emit_followups(Some(caller), reply).await;
        let summary: Vec<_> = emits
            .iter()
            .map(|e| (e.role, e.prompt.as_str()))
            .collect();
        eprintln!("emitted: {summary:?}");
        assert_eq!(emits.len(), 1, "one NEXT: line -> one task");
        // Task text is verbatim from the NEXT: line (not Needle's truncated arg).
        assert_eq!(
            emits[0].prompt,
            "review the token-expiry logic in the auth middleware for correctness"
        );
        // Needle classifies the role from the instruction (tool selection).
        assert_eq!(emits[0].role, Role::Review);
    }

    // No NEXT: line -> no emission (and Needle isn't even called).
    #[tokio::test]
    async fn no_next_line_emits_nothing() {
        let emits = emit_followups(None, "All done. The tests pass and nothing remains.").await;
        assert!(emits.is_empty());
    }

    // The tool-execution path a small model takes: a plain-English intent -> Needle
    // builds the call -> the ToolBox runs it against the repo. (The full loop also
    // needs hipfire for the model turns; this covers the Needle+tool half.)
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_builds_a_tool_call_that_the_toolbox_executes() {
        use crate::tools::{ToolBox, TOOL_SCHEMAS};
        let dir = std::env::temp_dir().join(format!("corrode-toolloop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("greeting.txt"), b"hello from the tool loop").unwrap();
        let toolbox = ToolBox::new(Arc::new(PassthroughVfs::new(&dir)));

        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let calls = caller
            .call("read the file greeting.txt", TOOL_SCHEMAS)
            .expect("tool-call construction");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("a tool call");
        assert_eq!(call.name, "read_file");

        let observation = toolbox.execute(call).await;
        assert!(
            observation.contains("hello from the tool loop"),
            "got: {observation}"
        );
        std::fs::remove_dir_all(&dir).ok();
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

        daemon.run(crx, etx).await;

        assert!(matches!(
            erx.recv().await.unwrap(),
            AgentEvent::Error { .. }
        ));
    }
}
