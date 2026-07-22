//! corrode-daemon — the agent, installed on a host.
//!
//! One process owns everything host-side: the hipfire client, the prioritized
//! swarm, the graph<->git VFS, and the embedded HelixDB store (graph + vectors +
//! GraphRAG). It exposes an API (websocket/HTTP, `corrode_core` messages) that the
//! separate `corrode-web` server drives on behalf of the wasm webui.
//!
//! This entry point drives the command loop over an in-process channel pair — a
//! stand-in for the `corrode-web` websocket bridge. It feeds a few sample
//! `AgentCommand`s, then prints the `AgentEvent`s the daemon streams back.

mod daemon;
mod graph;
mod hipfire;
mod swarm;
mod vfs;

use corrode_core::{AgentCommand, Priority};
use daemon::Daemon;
use hipfire::{Client, DEFAULT_BASE_URL};
use swarm::Swarm;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        std::env::var("HIPFIRE_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("HIPFIRE_API_KEY").ok();
    let model = std::env::var("CORRODE_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    // ponytail: HelixDB opens here once wired — `graph::embedded::HelixStore::open(path)`
    // behind `--features helix` (path from CORRODE_GRAPH_DIR) — and gets handed to
    // `Daemon::new` so DocQuery/VFS handlers can reach it.

    let daemon = Daemon::new(Swarm::new(Client::new(base_url, api_key), model, 32));

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);

    // Stand-in for the web bridge: enqueue sample commands, then close the channel
    // so the loop drains and exits.
    for cmd in [
        AgentCommand::Prompt {
            text: "Say READY.".into(),
            priority: Priority::Realtime,
        },
        AgentCommand::TerminalInput {
            session: "demo".into(),
            data: b"echo hi\n".to_vec(),
        },
        AgentCommand::DocQuery {
            question: "What is Corrode?".into(),
        },
    ] {
        cmd_tx.send(cmd).await?;
    }
    drop(cmd_tx);

    let loop_handle = tokio::spawn(async move { daemon.run(cmd_rx, ev_tx).await });

    while let Some(event) = ev_rx.recv().await {
        println!("{event:?}");
    }
    loop_handle.await?;
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
