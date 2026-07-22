//! corrode-daemon — the agent, installed on a host.
//!
//! One process owns everything host-side: the hipfire client, the prioritized
//! swarm, the role->model assignments, the embedded HelixDB store, and the VFS. It
//! exposes an API (websocket/HTTP, `corrode_core` messages) that `corrode-web`
//! drives on behalf of the wasm webui.
//!
//! This entry point drives the command loop over an in-process channel pair — a
//! stand-in for the `corrode-web` websocket bridge. It resolves roles against the
//! live model list, then feeds a few sample commands and prints the events back.

mod daemon;
mod graph;
mod hipfire;
mod roles;
mod swarm;
mod vfs;

use corrode_core::{AgentCommand, Priority};
use daemon::Daemon;
use hipfire::{Client, DEFAULT_BASE_URL};
use roles::RoleModels;
use swarm::Swarm;
use tokio::sync::mpsc;
use vfs::PassthroughVfs;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        std::env::var("HIPFIRE_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("HIPFIRE_API_KEY").ok();
    let fallback_model = std::env::var("CORRODE_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());
    let repo_root = std::env::var("CORRODE_REPO").unwrap_or_else(|_| ".".to_string());

    let client = Client::new(base_url, api_key.clone());

    // Resolve role -> model from hipfire's live model list + optional CORRODE_ROLES
    // overrides. If hipfire is unreachable, fall back to CORRODE_MODEL for all roles.
    let overrides = RoleModels::overrides_from_env()?;
    let roles = match client.list_models().await {
        Ok(models) => {
            eprintln!("hipfire models: {}", models.join(", "));
            RoleModels::resolve(&models, &overrides)
                .unwrap_or_else(|_| RoleModels::uniform(&fallback_model))
        }
        Err(e) => {
            eprintln!("hipfire model list unavailable ({e}); using CORRODE_MODEL for all roles");
            RoleModels::uniform(&fallback_model)
        }
    };
    let summary: Vec<String> = roles
        .0
        .iter()
        .map(|(role, model)| format!("{}={}", role.as_str(), model))
        .collect();
    eprintln!("role assignments: {}", summary.join("  "));

    let graph = open_graph();
    let vfs = Box::new(PassthroughVfs::new(&repo_root));
    let daemon = Daemon::new(Swarm::new(client, 32), roles, graph, vfs);

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);

    // Stand-in for the web bridge: enqueue sample commands, then close the channel
    // so the loop drains and exits.
    for cmd in [
        AgentCommand::Prompt {
            text: "Reply with exactly: READY".into(),
            priority: Priority::Realtime,
        },
        AgentCommand::ListDir { path: "".into() },
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

/// Open the embedded HelixDB store when built with `--features helix`.
#[cfg(feature = "helix")]
fn open_graph() -> Option<Box<dyn graph::GraphStore>> {
    let path = std::env::var("CORRODE_GRAPH_DIR").unwrap_or_else(|_| ".corrode/graph".to_string());
    match graph::embedded::HelixStore::open(&path) {
        Ok(store) => Some(Box::new(store)),
        Err(e) => {
            eprintln!("HelixDB open failed at {path}: {e}");
            None
        }
    }
}

#[cfg(not(feature = "helix"))]
fn open_graph() -> Option<Box<dyn graph::GraphStore>> {
    None
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
