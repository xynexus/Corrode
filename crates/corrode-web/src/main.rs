//! corrode-web — the web server, deployed separately from the daemon.
//!
//! Two jobs: serve the webui, and bridge browser <-> daemon. The browser opens one
//! WebSocket to `/agent` here; this proxies it to the daemon's `/agent` socket, so
//! the daemon stays private (one public origin, no CORS, no direct daemon exposure)
//! and all `AgentCommand`/`AgentEvent` frames pass through unchanged.
//!
//! Today `/` serves a dev placeholder page (see `index.html`); once the wasm webui
//! is built it serves that bundle instead. This crate links only `corrode-core` for
//! types plus the HTTP/ws plumbing — no agent logic, no hipfire, no HelixDB.

use axum::extract::ws::{Message as AxMsg, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use rust_embed::RustEmbed;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMsg;

/// The trunk-built webui bundle. Empty until `trunk build` runs in `webui/` (the
/// dir holds a `.gitkeep` so this compiles on a fresh clone); `static_handler`
/// falls back to the dev placeholder below when a requested asset is absent.
#[derive(RustEmbed)]
#[folder = "../../webui/dist"]
struct WebUi;

/// Dev placeholder served at `/` when no webui bundle is embedded yet.
const INDEX: &str = include_str!("../index.html");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("CORRODE_WEB_ADDR").unwrap_or_else(|_| "127.0.0.1:8787".to_string());
    let daemon_url = Arc::new(
        std::env::var("CORRODE_DAEMON_URL")
            .unwrap_or_else(|_| "ws://127.0.0.1:7878/agent".to_string()),
    );

    let app = Router::new()
        .route("/agent", get(agent_proxy))
        .fallback(static_handler)
        .with_state(daemon_url);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("corrode-web on http://{}  (proxying /agent -> daemon)", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Serve the embedded webui bundle; `/` -> index.html. Unknown asset falls back to
/// the dev placeholder at the root, or 404 for a sub-path.
async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match WebUi::get(path) {
        Some(content) => {
            let mime = content.metadata.mimetype().to_string();
            ([(header::CONTENT_TYPE, mime.as_str())], content.data.into_owned()).into_response()
        }
        None if path == "index.html" => Html(INDEX).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn agent_proxy(ws: WebSocketUpgrade, State(url): State<Arc<String>>) -> Response {
    ws.on_upgrade(move |socket| proxy_socket(socket, url))
}

/// Pump text frames both ways between the browser socket and the daemon socket.
async fn proxy_socket(browser: WebSocket, daemon_url: Arc<String>) {
    let upstream = match connect_async(daemon_url.as_str()).await {
        Ok((stream, _)) => stream,
        Err(e) => {
            eprintln!("daemon connect failed ({daemon_url}): {e}");
            return;
        }
    };
    let (mut daemon_tx, mut daemon_rx) = upstream.split();
    let (mut browser_tx, mut browser_rx) = browser.split();

    // browser -> daemon
    let b2d = tokio::spawn(async move {
        while let Some(Ok(msg)) = browser_rx.next().await {
            match msg {
                AxMsg::Text(t) => {
                    if daemon_tx.send(WsMsg::Text(t.as_str().into())).await.is_err() {
                        break;
                    }
                }
                AxMsg::Close(_) => break,
                _ => {}
            }
        }
    });

    // daemon -> browser
    let d2b = tokio::spawn(async move {
        while let Some(Ok(msg)) = daemon_rx.next().await {
            match msg {
                WsMsg::Text(t) => {
                    if browser_tx.send(AxMsg::Text(t.as_str().into())).await.is_err() {
                        break;
                    }
                }
                WsMsg::Close(_) => break,
                _ => {}
            }
        }
    });

    let _ = tokio::join!(b2d, d2b);
}
