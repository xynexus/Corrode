//! The daemon's network interface: a WebSocket endpoint at `/agent` speaking
//! `corrode_core::{AgentCommand, AgentEvent}` as JSON frames. Each connection gets
//! its own channel pair bridged to the shared `Daemon` command loop, so many
//! clients (or `corrode-web` proxying for the browser) can drive it concurrently
//! over the same host-side state.

use crate::daemon::Daemon;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use corrode_core::{AgentCommand, AgentEvent};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;

pub async fn serve(daemon: Daemon, addr: &str) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/agent", get(agent_ws))
        .with_state(Arc::new(daemon));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "corrode-daemon listening on ws://{}/agent",
        listener.local_addr()?
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn agent_ws(ws: WebSocketUpgrade, State(daemon): State<Arc<Daemon>>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, daemon))
}

/// Bridge one WebSocket to the command loop: inbound text frames deserialize into
/// `AgentCommand`, outbound `AgentEvent`s serialize back. The loop ends when the
/// socket closes (its command sender drops), which tears down both pumps.
async fn handle_socket(socket: WebSocket, daemon: Arc<Daemon>) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCommand>(64);
    let (ev_tx, mut ev_rx) = mpsc::channel::<AgentEvent>(64);

    let recv = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(t) => {
                    // ignore malformed frames; stop if the loop's receiver is gone
                    if let Ok(cmd) = serde_json::from_str::<AgentCommand>(t.as_str()) {
                        if cmd_tx.send(cmd).await.is_err() {
                            break;
                        }
                    }
                }
                Message::Close(_) => break,
                _ => continue,
            }
        }
    });

    let send = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            let Ok(json) = serde_json::to_string(&ev) else {
                continue;
            };
            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    daemon.run(cmd_rx, ev_tx).await;
    send.abort();
    recv.abort();
}
