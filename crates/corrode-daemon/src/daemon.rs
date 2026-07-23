//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change.
//!
//! The daemon owns the host-side state the handlers reach through `&self`: the
//! swarm, the role->model assignments, the embedded graph store (HelixDB, when
//! built), and the VFS.

use crate::graph::GraphStore;
use crate::planner;
use crate::roles::{Role, RoleModels};
use crate::swarm::{Swarm, Task};
use crate::terminal::Terminals;
use crate::vfs::Vfs;
use corrode_core::{AgentCommand, AgentEvent, Priority};
use tokio::sync::mpsc;

pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    /// Embedded HelixDB. `None` unless built with `--features helix` and opened.
    graph: Option<Box<dyn GraphStore>>,
    vfs: Box<dyn Vfs>,
    /// Live pty-backed terminal sessions.
    terminals: Terminals,
}

impl Daemon {
    pub fn new(
        swarm: Swarm,
        roles: RoleModels,
        graph: Option<Box<dyn GraphStore>>,
        vfs: Box<dyn Vfs>,
    ) -> Self {
        Self {
            swarm,
            roles,
            graph,
            vfs,
            terminals: Terminals::new(),
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
                let tasks = match self.plan(&text, priority).await {
                    Ok(tasks) => tasks,
                    Err(e) => {
                        let _ = events
                            .send(AgentEvent::Error {
                                message: format!("planning failed: {e}"),
                            })
                            .await;
                        return;
                    }
                };
                for (id, result) in self.swarm.run(tasks).await {
                    let ev = match result {
                        Ok(text) => AgentEvent::SubagentOutput { id: id as u64, text },
                        Err(e) => AgentEvent::Error { message: e.to_string() },
                    };
                    if events.send(ev).await.is_err() {
                        return; // consumer gone
                    }
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

    /// Decompose a prompt into role-tagged swarm tasks.
    ///
    /// Phase 1: the orchestration model produces a plan (at the request's band).
    /// Phase 2: [`planner::parse_plan`] + [`planner::to_tasks`] turn it into tasks,
    /// each on its role's model and band. If the model returns nothing parseable,
    /// degrade to a single coder task on the raw prompt.
    async fn plan(&self, text: &str, priority: Priority) -> anyhow::Result<Vec<Task>> {
        // Built once and shared, byte-identical, by the planning call and every
        // subagent, so hipfire batches them prefix-shared and reuses KV.
        let prefix = self.context_prefix().await;

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
            .await
            .into_iter()
            .next()
            .map(|(_, r)| r)
            .transpose()?
            .unwrap_or_default();

        let plan = planner::parse_plan(&plan_text);
        if plan.is_empty() {
            // ponytail: degrade to one coder task on the raw prompt (still behind
            // the shared prefix) so a plan the model couldn't structure still gets
            // attempted rather than dropped.
            Ok(planner::to_tasks(
                vec![planner::PlannedSubtask {
                    role: Role::Coder,
                    prompt: text.to_string(),
                }],
                &self.roles,
                &prefix,
            ))
        } else {
            Ok(planner::to_tasks(plan, &self.roles, &prefix))
        }
    }

    /// The shared context prefix prepended to every prompt in a Prompt turn.
    ///
    /// ponytail: a shallow repo digest (VFS root listing) plus a fixed preamble.
    /// The graph-backed VFS will supply richer, relevance-ranked context here
    /// (hipfire embeddings/rerank picking which nodes) — but the KV-sharing shape
    /// is already right: identical bytes across the whole swarm, task in the tail.
    async fn context_prefix(&self) -> String {
        let mut s = String::from(
            "You are a subagent in the Corrode coding-agent swarm working on a shared \
repository. Repository root:\n",
        );
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
            Box::new(PassthroughVfs::new(std::env::temp_dir())),
        )
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
