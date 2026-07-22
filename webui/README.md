# corrode-webui (wasm)

The browser front-end: a **wasm virtual terminal**, a **filesystem / repo / graph
explorer**, and the **agent interface**. Served by `corrode-web`, driven over a
websocket carrying `corrode_core::{AgentCommand, AgentEvent}`.

## Status: seam only — framework not chosen yet

This is deliberately not scaffolded. Building it commits three decisions that
shouldn't be guessed before the UI is real:

- **Framework:** Leptos vs Dioxus vs Yew (or plain `wasm-bindgen`).
- **Terminal:** `xterm.js` via JS interop vs a pure-Rust terminal widget. The
  terminal is a byte pipe over the websocket (`AgentEvent::TerminalOutput` /
  `AgentCommand::TerminalInput`) regardless of choice.
- **Build tool:** `trunk` vs `wasm-pack`. Whichever, this crate stays **out of the
  root Cargo workspace** so `cargo build` never pulls in a wasm target — it builds
  on its own (`trunk build` here), and `corrode-web` serves the output.

It will depend on `corrode-core` (which links nothing native, so it compiles to
wasm cleanly) for the wire types, and nothing from the daemon.

Pick the stack when the first screen gets built, not before.
