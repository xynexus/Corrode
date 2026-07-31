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

use crate::dialect::{Param, Tool};
use crate::roles::Role;
use crate::toolcall::ToolCall;
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use std::future::Future;

pub type TaskId = u64;

/// Byte cap on a provenance node's label (the task prompt can be arbitrarily long).
const LABEL_CAP: usize = 200;

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

/// What executing one task yields: its output, any tasks it emitted, and the repo paths
/// it produced (code nodes it authored via `write_file`).
pub struct Outcome {
    pub output: anyhow::Result<String>,
    pub emitted: Vec<Emit>,
    pub artifacts: Vec<String>,
}

/// A node in the provenance subgraph a plan produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
}

/// The kind of a provenance node. `Task` is a planned subtask; `Contract` is a task an
/// agent emitted mid-run (a test/research spin-off); `Code` is a produced file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Plan,
    Task,
    Contract,
    Code,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Plan => "plan",
            NodeKind::Task => "task",
            NodeKind::Contract => "contract",
            NodeKind::Code => "code",
        }
    }
}

/// A directed, labelled provenance edge (`from -rel-> to`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvEdge {
    pub from: String,
    pub rel: String,
    pub to: String,
}

impl ProvEdge {
    fn new(from: &str, rel: &str, to: &str) -> Self {
        Self {
            from: from.to_string(),
            rel: rel.to_string(),
            to: to.to_string(),
        }
    }
}

/// The plan's provenance subgraph: `nodes` (Plan + Task/Contract + Code) and the `edges`
/// connecting them (`part_of`, `emitted_from`, `produced_by`).
pub struct Provenance {
    pub nodes: Vec<ProvNode>,
    pub edges: Vec<ProvEdge>,
}

struct Node {
    task: PlanTask,
    status: Status,
    /// The task that emitted this one, if any — a task with an emitter is a *contract*
    /// (a test/research spin-off), and this is its provenance back to its emitter.
    emitted_by: Option<TaskId>,
    /// Repo paths this task produced (via `write_file`) — the code nodes it authored.
    artifacts: Vec<String>,
    /// The task's final answer, once `Done` — what the plan-level review pass reads.
    output: Option<String>,
}

#[derive(Default)]
pub struct PlanGraph {
    /// Stable id of the plan this graph *is* — the root every task hangs off in the
    /// provenance graph. Empty for graphs built without one (tests).
    plan_id: String,
    nodes: Vec<Node>,
    next_id: TaskId,
}

impl PlanGraph {
    /// A graph rooted at a named plan (its provenance root node).
    pub fn new(plan_id: impl Into<String>) -> Self {
        Self {
            plan_id: plan_id.into(),
            ..Self::default()
        }
    }

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
            emitted_by: None,
            artifacts: Vec::new(),
            output: None,
        });
        id
    }

    /// Record that `id` was emitted by `emitter` — makes `id` a contract in provenance.
    fn set_emitted_by(&mut self, id: TaskId, emitter: TaskId) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.task.id == id) {
            n.emitted_by = Some(emitter);
        }
    }

    /// Record the code artifacts (repo paths) a task produced.
    fn set_artifacts(&mut self, id: TaskId, artifacts: Vec<String>) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.task.id == id) {
            n.artifacts = artifacts;
        }
    }

    /// Record a task's final answer.
    fn set_output(&mut self, id: TaskId, output: &str) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.task.id == id) {
            n.output = Some(output.to_string());
        }
    }

    /// A compact digest of the settled plan for the review pass: every `Done` task's
    /// role, prompt, output (capped at `per_task_cap` bytes), and written paths, plus
    /// a line per `Failed` task — the reviewer must see the gaps, not only the wins.
    /// `None` when nothing completed — there is nothing to review.
    pub fn review_digest(&self, per_task_cap: usize) -> Option<String> {
        let done: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| n.status == Status::Done)
            .collect();
        if done.is_empty() {
            return None;
        }
        let mut digest = String::new();
        for n in done {
            digest.push_str(&format!(
                "task {} [{}]: {}\n",
                n.task.id,
                n.task.role.as_str(),
                n.task.prompt
            ));
            if let Some(out) = &n.output {
                let end = crate::tools::floor_char_boundary(out, per_task_cap);
                digest.push_str(&format!("output: {}\n", &out[..end]));
            }
            if !n.artifacts.is_empty() {
                digest.push_str(&format!("wrote: {}\n", n.artifacts.join(", ")));
            }
            digest.push('\n');
        }
        for n in self.nodes.iter().filter(|n| n.status == Status::Failed) {
            digest.push_str(&format!(
                "task {} [{}] FAILED: {}\n",
                n.task.id,
                n.task.role.as_str(),
                n.task.prompt
            ));
        }
        Some(digest)
    }

    fn node_ref(&self, id: TaskId) -> String {
        format!("{}:task:{}", self.plan_id, id)
    }

    /// The provenance subgraph this plan produced: a Plan root, a Task/Contract node per
    /// task (`part_of` the plan, and `emitted_from` its emitter for contracts), and a
    /// Code node per produced file (`produced_by` its task). This is what a graph store
    /// persists so the code<->task<->plan lineage is queryable.
    pub fn provenance(&self) -> Provenance {
        let plan = self.plan_id.clone();
        let mut nodes = vec![ProvNode {
            id: plan.clone(),
            kind: NodeKind::Plan,
            label: plan.clone(),
        }];
        let mut edges = Vec::new();
        for n in &self.nodes {
            let task_ref = self.node_ref(n.task.id);
            // Labels are for display/query — cap them so a digest-bearing prompt (the
            // plan-review task) doesn't duplicate every task's output into the store.
            let label_end = crate::tools::floor_char_boundary(&n.task.prompt, LABEL_CAP);
            nodes.push(ProvNode {
                id: task_ref.clone(),
                kind: if n.emitted_by.is_some() {
                    NodeKind::Contract
                } else {
                    NodeKind::Task
                },
                label: n.task.prompt[..label_end].to_string(),
            });
            edges.push(ProvEdge::new(&task_ref, "part_of", &plan));
            if let Some(emitter) = n.emitted_by {
                edges.push(ProvEdge::new(&task_ref, "emitted_from", &self.node_ref(emitter)));
            }
            for path in &n.artifacts {
                let code_ref = format!("{plan}:code:{path}");
                nodes.push(ProvNode {
                    id: code_ref.clone(),
                    kind: NodeKind::Code,
                    label: path.clone(),
                });
                edges.push(ProvEdge::new(&code_ref, "produced_by", &task_ref));
            }
        }
        Provenance { nodes, edges }
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
        if let Ok(text) = &outcome.output {
            graph.set_output(id, text); // what the plan-level review pass reads
        }
        graph.set_artifacts(id, outcome.artifacts); // code nodes this task produced
        for emit in outcome.emitted {
            let deps = if emit.after_emitter { vec![id] } else { vec![] };
            let emitted_id = graph.add(emit.role, emit.prompt, deps);
            graph.set_emitted_by(emitted_id, id); // the emitted task is a contract of `id`
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

/// The tools that *classify* a follow-up instruction's role: one tool per role, as
/// canonical [`Tool`] data (a model's [`crate::dialect::ToolDialect`] renders them). Role
/// lives in the tool *name* — the tool-caller is trained to pick one tool per turn,
/// which it does reliably, unlike filling an enum argument. We take only the chosen tool
/// (the role); the instruction text comes verbatim from the agent's `NEXT:` line, not
/// from the model's (often-truncated) `task` argument. Paired with [`role_from_tool_calls`].
pub const ROLE_TOOLS: &[Tool] = &[
    Tool {
        name: "research_task",
        description: "Investigate, read specs or docs, or survey prior art.",
        params: &[Param {
            name: "task",
            ty: "string",
            description: "What to investigate.",
            required: true,
        }],
    },
    Tool {
        name: "coding_task",
        description: "Write or modify code or tests.",
        params: &[Param {
            name: "task",
            ty: "string",
            description: "What to build.",
            required: true,
        }],
    },
    Tool {
        name: "architecture_task",
        description: "Make a design or structural decision.",
        params: &[Param {
            name: "task",
            ty: "string",
            description: "What to design.",
            required: true,
        }],
    },
    Tool {
        name: "review_task",
        description: "Check the correctness or quality of existing code.",
        params: &[Param {
            name: "task",
            ty: "string",
            description: "What to review.",
            required: true,
        }],
    },
];

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
                    artifacts: Vec::new(),
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

    // The plan-level review round: after the first drive settles, the digest carries the
    // done work (prompt, output, artifacts), a review task appended to the SAME graph is
    // driven by a second `run_reactive` with the SAME executor, and the fix it emits
    // runs too — provenance then links fix -emitted_from-> review.
    #[tokio::test]
    async fn review_round_reads_the_digest_and_runs_emitted_fixes() {
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut g = PlanGraph::new("plan-9");
        g.add(Role::Coder, "write add()", vec![]);

        let rec = order.clone();
        let execute = move |task: PlanTask| {
            let rec = rec.clone();
            async move {
                rec.lock().unwrap().push(task.prompt.clone());
                let emitted = if task.prompt.contains("review the plan") {
                    vec![Emit {
                        role: Role::Coder,
                        prompt: "fix the overflow in add()".into(),
                        after_emitter: true,
                    }]
                } else {
                    vec![]
                };
                let artifacts = if task.prompt.contains("write add") {
                    vec!["src/math.rs".to_string()]
                } else {
                    vec![]
                };
                Outcome {
                    output: Ok(format!("done: {}", task.prompt)),
                    emitted,
                    artifacts,
                }
            }
        };

        run_reactive(&mut g, &execute).await;
        let digest = g.review_digest(4096).expect("done work yields a digest");
        assert!(digest.contains("write add()"), "digest: {digest}");
        assert!(digest.contains("output: done: write add()"));
        assert!(digest.contains("wrote: src/math.rs"));

        let review = g.add(Role::Review, "review the plan's work", vec![]);
        run_reactive(&mut g, &execute).await;

        let ord = order.lock().unwrap().clone();
        let pos = |needle: &str| ord.iter().position(|p| p.contains(needle));
        assert!(pos("review the plan") < pos("fix the overflow"), "fix runs after review");
        // coder + review + fix, exactly once each — the review drive must never
        // re-execute settled tasks (that would mean duplicate real writes).
        assert_eq!(ord.len(), 3, "settled tasks re-ran: {ord:?}");
        assert!(g.stuck().is_empty());

        // The emitted fix is a contract of the review task in provenance.
        let prov = g.provenance();
        assert!(prov.edges.iter().any(|e| e.rel == "emitted_from"
            && e.to == format!("plan-9:task:{review}")));

        // Nothing done -> nothing to review.
        assert!(PlanGraph::default().review_digest(4096).is_none());
    }

    // The digest caps outputs on a char boundary (no mid-multibyte panic) and names
    // failed tasks — the reviewer must see the gaps, not only the wins.
    #[tokio::test]
    async fn review_digest_caps_output_and_reports_failures() {
        let mut g = PlanGraph::default();
        g.add(Role::Coder, "big output", vec![]);
        g.add(Role::Coder, "doomed", vec![]);
        run_reactive(&mut g, |task: PlanTask| async move {
            Outcome {
                output: if task.prompt.contains("doomed") {
                    Err(anyhow::anyhow!("boom"))
                } else {
                    Ok("€".repeat(50)) // 3-byte chars: cap 10 falls mid-char
                },
                emitted: vec![],
                artifacts: vec![],
            }
        })
        .await;

        let digest = g.review_digest(10).expect("one done task");
        let out_line = digest.lines().find(|l| l.starts_with("output:")).unwrap();
        assert_eq!(out_line, "output: €€€", "capped on a char boundary");
        assert!(digest.contains("task 1 [coder] FAILED: doomed"), "digest: {digest}");
    }

    // The provenance graph a plan produces: a coder task emits a contract and writes a
    // file; assert code -produced_by-> task, contract -emitted_from-> task, and every
    // task/contract -part_of-> the plan node.
    #[tokio::test]
    async fn provenance_links_code_to_task_and_contract_to_plan() {
        let mut g = PlanGraph::new("plan-7");
        g.add(Role::Coder, "write add()", vec![]);

        run_reactive(&mut g, |task: PlanTask| async move {
            let emits = if task.prompt.contains("write add") {
                vec![Emit {
                    role: Role::Coder,
                    prompt: "test add()".into(),
                    after_emitter: true,
                }]
            } else {
                vec![]
            };
            let artifacts = if task.prompt.contains("write add") {
                vec!["src/math.rs".to_string()]
            } else {
                vec![]
            };
            Outcome {
                output: Ok("done".into()),
                emitted: emits,
                artifacts,
            }
        })
        .await;

        let prov = g.provenance();
        let has_node = |id: &str, kind: NodeKind| {
            prov.nodes.iter().any(|n| n.id == id && n.kind == kind)
        };
        let has_edge = |from: &str, rel: &str, to: &str| {
            prov.edges
                .iter()
                .any(|e| e.from == from && e.rel == rel && e.to == to)
        };

        assert!(has_node("plan-7", NodeKind::Plan));
        assert!(has_node("plan-7:task:0", NodeKind::Task)); // original subtask
        assert!(has_node("plan-7:task:1", NodeKind::Contract)); // emitted test
        assert!(has_node("plan-7:code:src/math.rs", NodeKind::Code));

        assert!(has_edge("plan-7:task:0", "part_of", "plan-7"));
        assert!(has_edge("plan-7:task:1", "part_of", "plan-7"));
        assert!(has_edge("plan-7:task:1", "emitted_from", "plan-7:task:0"));
        assert!(has_edge("plan-7:code:src/math.rs", "produced_by", "plan-7:task:0"));
    }
}
