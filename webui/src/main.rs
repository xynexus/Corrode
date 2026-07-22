//! corrode-webui — the wasm front-end. CSR Leptos shell + an egui canvas panel,
//! talking to the daemon over the `/agent` websocket via `corrode_core` types.
//!
//! Built by trunk (`trunk build` / `trunk serve`), served by `corrode-web`. This
//! crate is intentionally outside the native cargo workspace.

mod app;
mod egui_panel;
mod model;
mod ws;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::App);
}
