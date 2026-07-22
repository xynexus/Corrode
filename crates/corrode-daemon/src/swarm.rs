//! The swarm: many prioritized subagents, one shared hipfire backend.
//!
//! Corrode leans directly on hipfire's design instead of reimplementing it:
//!
//! - **Priority is the steering wheel.** Subagents are cheap to spawn; hipfire's
//!   aging scheduler decides *when* each runs. Foreground turns go Realtime,
//!   ordinary work Default, and speculative "explore this hunch" agents go
//!   Opportunistic so they only burn idle GPU cycles and never delay the user.
//! - **Shared prompt prefix = shared KV.** hipfire batches sessions that share a
//!   prompt prefix and reuses their KV cache (`sessions_compatible_for_prefill`).
//!   So subagents should be built from a common context prefix (repo digest +
//!   system prompt) with only the task tail differing — that turns a wide fan-out
//!   into one batched, prefix-shared run instead of N cold prompts.
//! - **Admission is the daemon's job.** hipfire admission-controls on a VRAM /
//!   system-memory budget with per-owner fairness keys. Corrode can enqueue a
//!   large swarm without a local concurrency cap and let the daemon shed/queue —
//!   the local bound is a courtesy, not the real limit. ponytail: fixed local
//!   semaphore for now; drop it once we trust the daemon's backpressure.

use crate::hipfire::Client;
use corrode_core::Priority;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// One unit of work handed to the swarm. Carries its own model, because different
/// subagent roles run on different models (see [`crate::roles`]).
pub struct Task {
    pub prompt: String,
    pub priority: Priority,
    pub model: String,
}

pub struct Swarm {
    client: Arc<Client>,
    // ponytail: courtesy local cap so a runaway fan-out doesn't flood the daemon
    // socket. The real admission control is hipfire's resource budget — raise or
    // remove this once we measure the daemon handling the backpressure cleanly.
    inflight: Arc<Semaphore>,
}

impl Swarm {
    pub fn new(client: Client, max_inflight: usize) -> Self {
        Self {
            client: Arc::new(client),
            inflight: Arc::new(Semaphore::new(max_inflight)),
        }
    }

    /// Fan out every task concurrently. hipfire's scheduler orders them by band;
    /// results come back in completion order paired with their originating index.
    pub async fn run(&self, tasks: Vec<Task>) -> Vec<(usize, anyhow::Result<String>)> {
        let mut handles = Vec::with_capacity(tasks.len());
        for (i, task) in tasks.into_iter().enumerate() {
            let client = Arc::clone(&self.client);
            let permit = Arc::clone(&self.inflight);
            handles.push(tokio::spawn(async move {
                let _guard = permit.acquire_owned().await.expect("semaphore not closed");
                let out = client.respond(&task.model, &task.prompt, task.priority).await;
                (i, out)
            }));
        }
        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(pair) => results.push(pair),
                Err(join) => results.push((usize::MAX, Err(anyhow::anyhow!(join)))),
            }
        }
        results
    }
}
