//! Human-in-the-loop approval for mutating tool calls.
//!
//! Before a subagent may write a file or run a command, the daemon emits an
//! [`AgentEvent::ApprovalRequest`] and blocks that one tool call until a human answers
//! with [`AgentCommand::ApprovalResponse`]. The rest of the swarm keeps running — the
//! command loop dispatches Prompt handling concurrently (see [`crate::daemon::Daemon::run`]),
//! so the response is received while the requesting agent waits.
//!
//! Fail closed: if the event can't be sent or the response channel is dropped (client
//! gone), the action is DENIED. Read-only tools never pass through here.

use corrode_core::AgentEvent;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// Registry of in-flight approval requests, shared (`Arc`) between the command loop
/// (which resolves responses) and the tool loops (which await them).
#[derive(Default)]
pub struct ApprovalGate {
    pending: Mutex<HashMap<u64, oneshot::Sender<bool>>>,
    next_id: AtomicU64,
}

impl ApprovalGate {
    /// Ask a human to approve `action`. Emits an `ApprovalRequest` and awaits the
    /// matching `ApprovalResponse`. Returns `false` (denied) if the event can't be sent
    /// or the response channel is dropped.
    pub async fn request(&self, events: &mpsc::Sender<AgentEvent>, action: String) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let rx = {
            let (tx, rx) = oneshot::channel();
            self.pending.lock().unwrap().insert(id, tx);
            rx
        };
        if events
            .send(AgentEvent::ApprovalRequest { id, action })
            .await
            .is_err()
        {
            self.pending.lock().unwrap().remove(&id);
            return false; // client gone -> deny
        }
        rx.await.unwrap_or(false) // dropped without answering -> deny
    }

    /// Resolve a pending request with a human's decision. Unknown/duplicate ids are a
    /// no-op (the request already timed out on its side or was answered).
    pub fn resolve(&self, id: u64, approved: bool) {
        if let Some(tx) = self.pending.lock().unwrap().remove(&id) {
            let _ = tx.send(approved);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn approve_and_deny_round_trip() {
        let gate = std::sync::Arc::new(ApprovalGate::default());
        let (etx, mut erx) = mpsc::channel(8);

        // Approve: a responder resolves the first request as approved.
        let g = gate.clone();
        let responder = tokio::spawn(async move {
            let AgentEvent::ApprovalRequest { id, action } = erx.recv().await.unwrap() else {
                panic!("expected ApprovalRequest");
            };
            assert!(action.contains("write"));
            g.resolve(id, true);
            erx // hand the receiver back for the second case
        });
        assert!(gate.request(&etx, "write file foo".into()).await);
        let mut erx = responder.await.unwrap();

        // Deny: resolve as false.
        let g = gate.clone();
        tokio::spawn(async move {
            let AgentEvent::ApprovalRequest { id, .. } = erx.recv().await.unwrap() else {
                panic!()
            };
            g.resolve(id, false);
        });
        assert!(!gate.request(&etx, "run rm -rf".into()).await);
    }

    #[tokio::test]
    async fn dropped_client_denies() {
        let gate = ApprovalGate::default();
        let (etx, erx) = mpsc::channel(1);
        drop(erx); // client gone
        assert!(!gate.request(&etx, "write file".into()).await);
    }
}
