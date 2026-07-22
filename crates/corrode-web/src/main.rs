//! corrode-web — the web server, deployed separately from the daemon.
//!
//! Two jobs, no more:
//!   1. Serve the built wasm webui bundle (see `webui/`) as static assets.
//!   2. Bridge browser <-> daemon: forward `AgentCommand` to the daemon and stream
//!      `AgentEvent` back (websocket in practice), including terminal frames for the
//!      wasm virtual terminal and graph/repo/fs explorer queries.
//!
//! It links `corrode-core` for the wire types and nothing else — no hipfire, no
//! HelixDB. All agent logic and all state live in the daemon; this stays a thin,
//! stateless edge that can run on a different host from the daemon.
//!
//! ponytail: no HTTP framework yet — axum + tower-http static serving + a websocket
//! upgrade is ~40 lines, but it's speculative until the daemon exposes its socket.
//! Add axum here the moment the daemon has an endpoint to bridge to. Wiring a server
//! now would be scaffolding for a contract that doesn't exist.

fn main() {
    // Prove the shared contract compiles across the boundary; replace with the
    // real server once the daemon's socket lands.
    let ping = corrode_core::AgentCommand::DocQuery {
        question: "what is Corrode?".into(),
    };
    println!(
        "corrode-web: stub. daemon bridge not wired yet. sample command: {}",
        serde_json::to_string(&ping).unwrap()
    );
}
