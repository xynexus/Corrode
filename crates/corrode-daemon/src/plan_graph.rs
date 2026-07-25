//! Reactive task planning — the daemon's answer to "HelixDB has no triggers".
//!
//! HelixDB v2.3.5 has no trigger/reactive/changefeed at any layer (confirmed), so
//! reactivity lives here, where it belongs — same as Leptos builds reactivity over
//! state, not in the DB. A [`PlanGraph`] is a dependency graph of tasks that a
//! *running* agent can GROW: as a coder codes it emits new tasks (a test **contract**,
//! a research question), and [`run_reactive`] re-derives the runnable set on every
//! completion and fans it to the executor. Dataflow, not a fixed fan-out: state
//! (task nodes) → effects (runnable tasks) → run → new state.
//!
//! This is the engine. The live wiring — executor = the hipfire swarm, and parsing
//! emitted tasks out of agent output — is the integration step on top.

use crate::roles::Role;
use crate::toolcall::ToolCall;
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use std::future::Future;

pub type TaskId = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Pending,
    Running,
    Done,
    Failed,
}

#[derive(Clone)]
pub struct PlanTask {
    pub id: TaskId,
    pub role: Role,
    pub prompt: String,
    /// Tasks that must be `Done` before this one may run.
    pub deps: Vec<TaskId>,
}

/// A new task a running task emits — a contract or a spun-off line of work.
pub struct Emit {
    pub role: Role,
    pub prompt: String,
    /// If true, the emitted task depends on its emitter (e.g. a test contract waits
    /// for the code). If false, it's runnable immediately (e.g. parallel research).
    pub after_emitter: bool,
}

/// What executing one task yields: its output, and any tasks it emitted.
pub struct Outcome {
    pub output: anyhow::Result<String>,
    pub emitted: Vec<Emit>,
}

struct Node {
    task: PlanTask,
    status: Status,
}

#[derive(Default)]
pub struct PlanGraph {
    nodes: Vec<Node>,
    next_id: TaskId,
}

impl PlanGraph {
    /// Add a task; returns its id (usable as a dependency for later tasks).
    pub fn add(&mut self, role: Role, prompt: impl Into<String>, deps: Vec<TaskId>) -> TaskId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(Node {
            task: PlanTask {
                id,
                role,
                prompt: prompt.into(),
                deps,
            },
            status: Status::Pending,
        });
        id
    }

    fn status(&self, id: TaskId) -> Option<&Status> {
        self.nodes.iter().find(|n| n.task.id == id).map(|n| &n.status)
    }

    fn set_status(&mut self, id: TaskId, status: Status) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.task.id == id) {
            n.status = status;
        }
    }

    /// Pending tasks whose every dependency is `Done`.
    fn ready(&self) -> Vec<PlanTask> {
        self.nodes
            .iter()
            .filter(|n| {
                n.status == Status::Pending
                    && n.task
                        .deps
                        .iter()
                        .all(|d| self.status(*d) == Some(&Status::Done))
            })
            .map(|n| n.task.clone())
            .collect()
    }

    /// Tasks left `Pending` with no path to run (a failed/unmet dependency or a
    /// cycle) after the scheduler drains — the "couldn't schedule" signal.
    pub fn stuck(&self) -> Vec<TaskId> {
        self.nodes
            .iter()
            .filter(|n| n.status == Status::Pending)
            .map(|n| n.task.id)
            .collect()
    }
}

/// Drive the graph to completion reactively: launch every ready task, and each time
/// one finishes, mark it, fold in the tasks it emitted, and launch whatever that made
/// ready — until nothing is ready and nothing is in flight. `execute` runs a task
/// (in the daemon: fan it to the swarm and stream its output); it bounds real
/// concurrency itself (the swarm's `inflight` semaphore).
pub async fn run_reactive<E, Fut>(graph: &mut PlanGraph, execute: E)
where
    E: Fn(PlanTask) -> Fut,
    Fut: Future<Output = Outcome>,
{
    let mut inflight = FuturesUnordered::new();
    loop {
        for task in graph.ready() {
            let id = task.id;
            graph.set_status(id, Status::Running);
            let fut = execute(task); // borrows `execute`; only the future is moved
            inflight.push(async move { (id, fut.await) });
        }
        let Some((id, outcome)) = inflight.next().await else {
            break; // nothing running and nothing ready -> settled (or all remaining are stuck)
        };
        graph.set_status(
            id,
            if outcome.output.is_ok() {
                Status::Done
            } else {
                Status::Failed
            },
        );
        for emit in outcome.emitted {
            let deps = if emit.after_emitter { vec![id] } else { vec![] };
            graph.add(emit.role, emit.prompt, deps);
        }
    }
}

/// Extract the single follow-up instruction an agent proposed, from a `NEXT:` line
/// (see [`crate::planner::subagent_prompt`]). Plain English — a small model can write
/// this even though it can't hand-write a tool call. Returns the first such line's
/// text, trimmed; `None` (no follow-up) if absent or empty.
pub fn parse_next_instruction(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("NEXT:")?.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    })
}

/// The tools handed to Needle to *classify* a follow-up instruction's role: one tool
/// per role, in Needle's native flat schema (`parameters` = name -> spec). Role lives
/// in the tool *name* — Needle is trained to pick one tool per turn, which it does
/// reliably, unlike filling an enum argument. We take only the chosen tool (the role);
/// the instruction text comes verbatim from the agent's `NEXT:` line, not from Needle's
/// (often-truncated) `task` argument. Paired with [`role_from_tool_calls`].
///
/// ponytail: Needle will be finetuned on Corrode's actual tool set so the small coder
/// models' tools are picked up reliably. Once that lands, its `task` argument becomes
/// trustworthy too and the verbatim-text shortcut can drop — and this same schema
/// shape extends to the real tool-execution tools (read_file, run_command, ...).
pub const ROLE_TASK_TOOLS: &str = r#"[{"name":"research_task","description":"Investigate, read specs or docs, or survey prior art.","parameters":{"task":{"type":"string","description":"What to investigate.","required":true}}},{"name":"coding_task","description":"Write or modify code or tests.","parameters":{"task":{"type":"string","description":"What to build.","required":true}}},{"name":"architecture_task","description":"Make a design or structural decision.","parameters":{"task":{"type":"string","description":"What to design.","required":true}}},{"name":"review_task","description":"Check the correctness or quality of existing code.","parameters":{"task":{"type":"string","description":"What to review.","required":true}}}]"#;

/// The role Needle picked, from the first call's tool name (see [`ROLE_TASK_TOOLS`]).
/// Unknown/absent -> None (the caller defaults to Coder).
pub fn role_from_tool_calls(calls: &[ToolCall]) -> Option<Role> {
    let name = &calls.first()?.name;
    match name.as_str() {
        "research_task" => Some(Role::Research),
        "coding_task" => Some(Role::Coder),
        "architecture_task" => Some(Role::Architect),
        "review_task" => Some(Role::Review),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn parse_next_instruction_reads_the_next_line() {
        let out = "I implemented add() in math.rs.\nNEXT: write unit tests for add() covering overflow\n";
        assert_eq!(
            parse_next_instruction(out).as_deref(),
            Some("write unit tests for add() covering overflow")
        );
        // no line -> None; empty NEXT -> None
        assert_eq!(parse_next_instruction("all done, nothing to do"), None);
        assert_eq!(parse_next_instruction("NEXT:   "), None);
    }

    #[test]
    fn role_from_tool_calls_maps_the_chosen_tool_name() {
        let call = |n: &str| ToolCall {
            name: n.into(),
            arguments: serde_json::json!({"task":"x"}),
        };
        assert_eq!(
            role_from_tool_calls(&[call("research_task")]),
            Some(Role::Research)
        );
        assert_eq!(role_from_tool_calls(&[call("coding_task")]), Some(Role::Coder));
        assert_eq!(
            role_from_tool_calls(&[call("architecture_task")]),
            Some(Role::Architect)
        );
        assert_eq!(role_from_tool_calls(&[call("review_task")]), Some(Role::Review));
        // one call per turn: only the first is consulted
        assert_eq!(
            role_from_tool_calls(&[call("review_task"), call("coding_task")]),
            Some(Role::Review)
        );
        // unknown / empty -> None (caller defaults to Coder)
        assert_eq!(role_from_tool_calls(&[call("frobnicate")]), None);
        assert_eq!(role_from_tool_calls(&[]), None);
    }

    // A coder task emits a test contract mid-run; a review task depends on the coder.
    // Assert dependency order (coder before review) AND that the emitted test runs
    // after the coder that spawned it.
    #[tokio::test]
    async fn reactive_scheduler_orders_deps_and_runs_emitted_tasks() {
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut g = PlanGraph::default();
        let coder = g.add(Role::Coder, "write add()", vec![]);
        g.add(Role::Review, "review add()", vec![coder]);

        let rec = order.clone();
        run_reactive(&mut g, move |task: PlanTask| {
            let rec = rec.clone();
            async move {
                rec.lock().unwrap().push(task.prompt.clone());
                let emitted = if task.prompt.contains("write add") {
                    vec![Emit {
                        role: Role::Coder,
                        prompt: "test add()".into(),
                        after_emitter: true,
                    }]
                } else {
                    vec![]
                };
                Outcome {
                    output: Ok(format!("done: {}", task.prompt)),
                    emitted,
                }
            }
        })
        .await;

        let ord = order.lock().unwrap().clone();
        let pos = |needle: &str| ord.iter().position(|p| p.contains(needle));
        assert!(pos("write add") < pos("review add"), "dep order: coder before review");
        assert!(pos("test add").is_some(), "emitted test task ran");
        assert!(pos("write add") < pos("test add"), "emitted test runs after its emitter");
        assert!(g.stuck().is_empty(), "everything scheduled");
    }
}
