# corrode-webui (wasm)

The browser front-end: a **virtual terminal**, a **filesystem / repo / graph
explorer**, and the **agent interface**. Served by `corrode-web`, driven over a
websocket carrying `corrode_core::{AgentCommand, AgentEvent}`.

## Stack

- **Leptos (CSR)** — the DOM shell: explorer, agent log, prompt input. Client-side
  only; all state lives in the daemon and is reached over `/agent`.
- **egui / eframe** — the virtual terminal + graph explorer, drawn immediate-mode
  on a `<canvas>` the Leptos shell owns (`egui_panel.rs`). The two panels that are
  custom-rendered rather than DOM.
- **trunk** — the build tool. This crate stays **out of the root Cargo workspace**
  (own `[workspace]`) so `cargo build` at the repo root never pulls a wasm target.
- **corrode-core** — shared wire types (links nothing native, compiles to wasm).

The two halves share state per their nature: the DOM uses Leptos signals; the egui
canvas reads an `Rc<RefCell<UiModel>>` the websocket task writes into (wasm is
single-threaded, so no locking). See `model.rs` / `ws.rs`.

## Layout

```
src/main.rs        mount the Leptos app
src/app.rs         the shell: explorer + agent panels + the egui <canvas>
src/ws.rs          /agent websocket: AgentCommand out, AgentEvent in
src/egui_panel.rs  eframe app: terminal (top) + graph explorer (below)
src/model.rs       shared UiModel for the egui side
index.html         trunk entry + layout CSS
Trunk.toml         build config + dev proxy of /agent -> corrode-web
```

## Build / run

```bash
trunk build                 # -> dist/  (corrode-web serves this; --release for a slim wasm)
trunk serve                 # dev server on :8080, proxies /agent -> corrode-web on :8787
```

Full stack: `hipfire start`, then `corrode-daemon`, then `corrode-web`, then either
open `corrode-web` (http://127.0.0.1:8787 — it serves the built `dist/`) or run
`trunk serve` and open http://127.0.0.1:8080 for hot-reload dev.

`corrode-web` embeds `dist/` via `rust-embed`: in debug it reads the dir live (no
recompile after `trunk build`); in release it bakes the bundle into the binary.
Until you run `trunk build`, `dist/` holds only `.gitkeep` and `corrode-web` serves
its built-in placeholder page.
