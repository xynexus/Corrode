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
use axum::http::{header, HeaderMap, StatusCode, Uri};
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
    let addr = std::env::var("CORRODE_WEB_ADDR").unwrap_or_else(|_| "0.0.0.0:8787".to_string());
    let proxy = Arc::new(Proxy {
        daemon_url: std::env::var("CORRODE_DAEMON_URL")
            .unwrap_or_else(|_| "ws://127.0.0.1:7878/agent".to_string()),
        origins: std::env::var("CORRODE_WEB_ORIGINS")
            .map(|v| v.split(',').map(|o| o.trim().to_string()).filter(|o| !o.is_empty()).collect())
            .unwrap_or_default(),
    });

    let app = Router::new()
        .route("/agent", get(agent_proxy))
        .fallback(static_handler)
        .with_state(proxy);

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

struct Proxy {
    daemon_url: String,
    /// `CORRODE_WEB_ORIGINS`: extra page origins (`scheme://host:port`, comma-separated)
    /// allowed to open `/agent` -- for a UI reached through a reverse proxy that
    /// rewrites `Host`.
    origins: Vec<String>,
}

async fn agent_proxy(ws: WebSocketUpgrade, headers: HeaderMap, State(proxy): State<Arc<Proxy>>) -> Response {
    if !origin_ok(&headers, &proxy.origins) {
        return (StatusCode::FORBIDDEN, "cross-origin /agent refused").into_response();
    }
    ws.on_upgrade(move |socket| proxy_socket(socket, proxy))
}

/// Whether a WebSocket upgrade may proceed. A browser always sends `Origin`, and lets
/// any page open a socket to any host: without this, a site the user visits could
/// drive their swarm through their own browser (cross-site WebSocket hijacking) --
/// the web server binds 0.0.0.0 and auth is off by default. An origin naming the same
/// host:port the request was sent to, or one listed in `CORRODE_WEB_ORIGINS`, passes;
/// a request with no `Origin` (a CLI or script, not a page) passes.
fn origin_ok(headers: &HeaderMap, allowed: &[String]) -> bool {
    let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) else {
        return true;
    };
    if allowed.iter().any(|a| a.eq_ignore_ascii_case(origin)) {
        return true;
    }
    let authority = origin.split_once("://").map(|(_, rest)| rest.trim_end_matches('/'));
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok());
    matches!((authority, host), (Some(a), Some(h)) if a.eq_ignore_ascii_case(h))
}

/// Pump text frames both ways between the browser socket and the daemon socket.
async fn proxy_socket(browser: WebSocket, proxy: Arc<Proxy>) {
    let daemon_url = &proxy.daemon_url;
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

    // daemon -> browser, pinging the browser every 30 s: a laptop that slept or a
    // dropped proxy then surfaces as a failed send instead of a half-open socket
    // held for ~15 minutes.
    let d2b = tokio::spawn(async move {
        let mut ping = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            tokio::select! {
                msg = daemon_rx.next() => match msg {
                    Some(Ok(WsMsg::Text(t))) => {
                        if browser_tx.send(AxMsg::Text(t.as_str().into())).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(WsMsg::Close(_))) | Some(Err(_)) | None => break,
                    Some(Ok(_)) => {}
                },
                _ = ping.tick() => {
                    if browser_tx.send(AxMsg::Ping(Default::default())).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Either side ending ends both. `join!` waited for the other pump, so a closed
    // browser kept the daemon connection -- and its session binding -- open until
    // the daemon next wrote to it. The daemon's turns no longer depend on this
    // socket: a reconnecting client is replayed what it missed.
    let (mut b2d, mut d2b) = (b2d, d2b);
    tokio::select! {
        _ = &mut b2d => {}
        _ = &mut d2b => {}
    }
    b2d.abort();
    d2b.abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(origin: Option<&str>, host: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::HOST, host.parse().unwrap());
        if let Some(o) = origin {
            h.insert(header::ORIGIN, o.parse().unwrap());
        }
        h
    }

    #[test]
    fn only_the_ui_own_origin_may_open_the_agent_socket() {
        let none: &[String] = &[];
        assert!(origin_ok(&headers(Some("http://box:8787"), "box:8787"), none), "the UI itself");
        assert!(origin_ok(&headers(None, "box:8787"), none), "a CLI client sends no Origin");
        assert!(!origin_ok(&headers(Some("https://evil.example"), "box:8787"), none), "another site");
        assert!(!origin_ok(&headers(Some("http://box:9999"), "box:8787"), none), "another port");
        assert!(!origin_ok(&headers(Some("null"), "box:8787"), none), "a sandboxed page");
        let proxied = ["https://corrode.example".to_string()];
        assert!(origin_ok(&headers(Some("https://corrode.example"), "127.0.0.1:8787"), &proxied));
    }
}
