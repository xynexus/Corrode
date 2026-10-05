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
//!
//! Opt-in auto-approve (`CORRODE_AUTO_APPROVE`): for unattended operation (a cron/
//! headless swarm with no human to answer), the gate can auto-approve every mutating
//! call. This is what lets the swarm actually write code on its own — otherwise every
//! write blocks forever and fails closed. It is OFF by default and intended to be
//! paired with the sandbox (`CORRODE_SANDBOX`), which confines those writes/commands
//! to the repo; each auto-approval is logged, and the executed call still streams back
//! as a `ToolResult` so the operator sees everything that ran.

use corrode_core::AgentEvent;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

fn approval_timeout() -> std::time::Duration {
    let s = std::env::var("CORRODE_APPROVAL_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&s: &u64| s > 0)
        .unwrap_or(3600);
    std::time::Duration::from_secs(s)
}

/// Registry of in-flight approval requests, shared (`Arc`) between the command loop
/// (which resolves responses) and the tool loops (which await them).
#[derive(Default)]
pub struct ApprovalGate {
    pending: Mutex<HashMap<u64, oneshot::Sender<bool>>>,
    next_id: AtomicU64,
    /// When set, every request is auto-approved without a human (unattended mode).
    auto_approve: bool,
}

impl ApprovalGate {
    /// A gate whose auto-approve is read from `CORRODE_AUTO_APPROVE`. Off unless set
    /// on — the human-in-the-loop default.
    pub fn from_env() -> Self {
        let auto_approve = crate::knobs::flag("CORRODE_AUTO_APPROVE", false);
        Self {
            auto_approve,
            ..Default::default()
        }
    }

    /// Ask a human to approve `action`. Emits an `ApprovalRequest` and awaits the
    /// matching `ApprovalResponse`. Returns `false` (denied) if the event can't be sent
    /// or the response channel is dropped. With auto-approve on, returns `true`
    /// immediately (the call still streams back as a `ToolResult` after it runs).
    pub async fn request(&self, events: &mpsc::Sender<AgentEvent>, action: String) -> bool {
        if self.auto_approve {
            eprintln!("auto-approved (CORRODE_AUTO_APPROVE): {action}");
            return true;
        }
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
        // Deny when the client goes away after the request was sent (nothing would
        // ever answer it) or nobody answers within CORRODE_APPROVAL_TIMEOUT_S
        // (default 3600): a swarm is often left alone, and a turn must end.
        let approved = tokio::select! {
            r = rx => r.unwrap_or(false), // dropped without answering -> deny
            _ = events.closed() => false,
            _ = tokio::time::sleep(approval_timeout()) => false,
        };
        self.pending.lock().unwrap().remove(&id);
        approved
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

    // The client disconnects after the request went out: nobody can answer, so the
    // call is denied at once instead of waiting forever.
    #[tokio::test]
    async fn a_client_gone_mid_request_denies() {
        let gate = std::sync::Arc::new(ApprovalGate::default());
        let (etx, mut erx) = mpsc::channel(1);
        tokio::spawn(async move {
            let _ = erx.recv().await; // receives the request, then disconnects
        });
        let r = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            gate.request(&etx, "write file".into()),
        )
        .await;
        assert_eq!(r, Ok(false));
        assert!(gate.pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn auto_approve_grants_without_a_human() {
        // Unattended mode: no responder, and the client channel is even dropped —
        // fail-closed would deny, but auto-approve grants immediately.
        let gate = ApprovalGate {
            auto_approve: true,
            ..Default::default()
        };
        let (etx, erx) = mpsc::channel(1);
        drop(erx);
        assert!(gate.request(&etx, "write file foo".into()).await);
    }
}
