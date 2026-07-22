//! corrode-daemon — the agent, installed on a host.
//!
//! One process owns everything host-side: the hipfire client, the prioritized
//! swarm, the graph<->git VFS, and the embedded HelixDB store (graph + vectors +
//! GraphRAG). It exposes an API (websocket/HTTP, `corrode_core` messages) that the
//! separate `corrode-web` server drives on behalf of the wasm webui.
//!
//! This entry point is still the smoke path: it fans out one three-band task set.
//! The real daemon loop (accept `AgentCommand`, plan, swarm, stream `AgentEvent`)
//! is the next build.

mod graph;
mod hipfire;
mod swarm;
mod vfs;

use hipfire::{Client, DEFAULT_BASE_URL};
use corrode_core::Priority;
use swarm::{Swarm, Task};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        std::env::var("HIPFIRE_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("HIPFIRE_API_KEY").ok();
    let model = std::env::var("CORRODE_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    // ponytail: HelixDB opens here once wired — `graph::embedded::HelixStore::open(path)`
    // behind `--features helix`. Store path from CORRODE_GRAPH_DIR. Left out of the
    // smoke path so the default build stays light.

    let client = Client::new(base_url, api_key);
    let swarm = Swarm::new(client, model, 32);

    let tasks = vec![
        Task {
            prompt: "Say READY.".into(),
            priority: Priority::Realtime,
        },
        Task {
            prompt: "Summarize this repo in one line.".into(),
            priority: Priority::Default,
        },
        Task {
            prompt: "Speculatively list refactors worth exploring.".into(),
            priority: Priority::Opportunistic,
        },
    ];

    for (i, result) in swarm.run(tasks).await {
        match result {
            Ok(text) => println!("[{i}] {text}"),
            Err(e) => eprintln!("[{i}] error: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use corrode_core::Priority;

    // Bands must stay pinned to hipfire-scheduler's SCHED_PRIORITY_* (0/64/255) and
    // ordered, or the swarm mis-orders against the daemon.
    #[test]
    fn priority_bands_match_hipfire() {
        assert_eq!(Priority::Realtime.as_u8(), 0);
        assert_eq!(Priority::Default.as_u8(), 64);
        assert_eq!(Priority::Opportunistic.as_u8(), 255);
        assert!(Priority::Realtime.as_u8() < Priority::Default.as_u8());
        assert!(Priority::Default.as_u8() < Priority::Opportunistic.as_u8());
    }
}
