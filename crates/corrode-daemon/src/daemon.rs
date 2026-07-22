//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change.
//!
//! The daemon owns the host-side state the handlers reach through `&self`: the
//! swarm, the role->model assignments, the embedded graph store (HelixDB, when
//! built), and the VFS.

use crate::graph::GraphStore;
use crate::roles::{Role, RoleModels};
use crate::swarm::{Swarm, Task};
use crate::vfs::Vfs;
use corrode_core::{AgentCommand, AgentEvent, Priority};
use tokio::sync::mpsc;

pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    /// Embedded HelixDB. `None` unless built with `--features helix` and opened.
    graph: Option<Box<dyn GraphStore>>,
    vfs: Box<dyn Vfs>,
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
                for (id, result) in self.swarm.run(self.plan_prompt(&text, priority)).await {
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
                let ev = match self.vfs.list(&path) {
                    Ok(entries) => AgentEvent::DirListing { path, entries },
                    Err(e) => AgentEvent::Error { message: e.to_string() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::TerminalInput { session, data } => {
                // ponytail: echo, so the wasm terminal has a live byte-pipe to build
                // against. Swap for a portable-pty shell per session (real terminal).
                let _ = events
                    .send(AgentEvent::TerminalOutput { session, data })
                    .await;
            }
        }
    }

    /// Turn a prompt into swarm tasks.
    ///
    /// ponytail: one task on the orchestration model. The real planner — the reason
    /// this is a *swarm* — has the orchestration model decompose the prompt into
    /// many role-tagged subagents (research/architect/coder/review), each on its
    /// role's model and priority band, sharing a context prefix for KV reuse. The
    /// loop above already fans out whatever this returns.
    fn plan_prompt(&self, text: &str, priority: Priority) -> Vec<Task> {
        let model = self
            .roles
            .model_for(Role::Orchestration)
            .unwrap_or_default()
            .to_string();
        vec![Task {
            prompt: text.to_string(),
            priority,
            model,
        }]
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

    // Exercises the real loop for the variants that don't touch hipfire: terminal
    // echoes its bytes, and DocQuery reports itself unavailable (no graph) rather
    // than hang or panic. Guards the dispatch/match, not the network.
    #[tokio::test]
    async fn loop_routes_terminal_and_docquery_without_hipfire() {
        let daemon = test_daemon();
        let (ctx, crx) = mpsc::channel(8);
        let (etx, mut erx) = mpsc::channel(8);

        ctx.send(AgentCommand::TerminalInput {
            session: "s".into(),
            data: b"hi".to_vec(),
        })
        .await
        .unwrap();
        ctx.send(AgentCommand::DocQuery {
            question: "q".into(),
        })
        .await
        .unwrap();
        drop(ctx);

        daemon.run(crx, etx).await;

        assert!(matches!(
            erx.recv().await.unwrap(),
            AgentEvent::TerminalOutput { data, .. } if data == b"hi"
        ));
        assert!(matches!(
            erx.recv().await.unwrap(),
            AgentEvent::Error { .. }
        ));
    }
}
