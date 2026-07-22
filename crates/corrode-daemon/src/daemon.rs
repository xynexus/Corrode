//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change. Wiring an HTTP framework is `corrode-web`'s
//! job, once it has this loop to bridge to.

use crate::swarm::{Swarm, Task};
use corrode_core::{AgentCommand, AgentEvent, Priority};
use tokio::sync::mpsc;

pub struct Daemon {
    swarm: Swarm,
    // ponytail: HelixStore (graph/GraphRAG) and the VFS land here next, behind the
    // same &self so handlers can reach them. DocQuery/terminal stay stubbed until then.
}

impl Daemon {
    pub fn new(swarm: Swarm) -> Self {
        Self { swarm }
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
                for (id, result) in self.swarm.run(plan_prompt(&text, priority)).await {
                    let ev = match result {
                        Ok(text) => AgentEvent::SubagentOutput { id: id as u64, text },
                        Err(e) => AgentEvent::Error { message: e.to_string() },
                    };
                    // Consumer gone = nobody's listening; stop pushing at it.
                    if events.send(ev).await.is_err() {
                        return;
                    }
                }
            }
            AgentCommand::DocQuery { question } => {
                // ponytail: GraphRAG over HelixDB — needs `--features helix` and an
                // open HelixStore on `self`. Report honestly until that's wired.
                let _ = events
                    .send(AgentEvent::Error {
                        message: format!("DocQuery not wired yet (needs HelixDB): {question}"),
                    })
                    .await;
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
}

/// Turn a prompt into swarm tasks.
///
/// ponytail: one task at the requested band. The real planner — the reason this is
/// a *swarm* — decomposes a prompt into many prioritized subagents (foreground
/// Realtime, speculative Opportunistic) sharing a context prefix for KV reuse.
/// That's the next thing to grow here; the loop above already fans out whatever
/// this returns.
fn plan_prompt(text: &str, priority: Priority) -> Vec<Task> {
    vec![Task {
        prompt: text.to_string(),
        priority,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::Client;

    // Exercises the real loop for the two variants that don't touch hipfire: the
    // terminal must echo its bytes, and DocQuery must report itself unwired rather
    // than hang or panic. Guards the dispatch/match, not the network.
    #[tokio::test]
    async fn loop_routes_terminal_and_docquery_without_hipfire() {
        let daemon = Daemon::new(Swarm::new(Client::new("http://127.0.0.1:1", None), "m", 1));
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
        drop(ctx); // close the loop

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
