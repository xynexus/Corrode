//! corrode-webui — the wasm front-end. Leptos DOM shell with an xterm.js terminal
//! and an egui/WebGL graph canvas, over the `/agent` websocket (`corrode_core`).
//!
//! Built by trunk (`trunk build` / `trunk serve`), served by `corrode-web`. This
//! crate is intentionally outside the native cargo workspace.

mod app;
mod egui_panel;
mod model;
mod term;
mod ws;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::App);
}
