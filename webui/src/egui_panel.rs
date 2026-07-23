//! The egui/WebGL graph-explorer canvas. (The terminal moved to xterm.js; the
//! Markdown/LaTeX agent console is DOM — this canvas is just the node graph, the
//! one surface where GPU rendering earns its keep.)
//!
//! eframe's `WebRunner` drives the render loop on a `<canvas>` the Leptos shell
//! owns; we hand it a clone of the [`Shared`] model that the websocket task writes.

use eframe::CreationContext;
use web_sys::HtmlCanvasElement;

use crate::model::Shared;

/// Start the egui app on `canvas`. Runs asynchronously (eframe's start is async);
/// stores the egui `Context` into the shared model so async pushes can repaint.
pub fn start(canvas: HtmlCanvasElement, shared: Shared) {
    let runner = eframe::WebRunner::new();
    wasm_bindgen_futures::spawn_local(async move {
        let result = runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc: &CreationContext<'_>| {
                    shared.borrow_mut().egui_ctx = Some(cc.egui_ctx.clone());
                    Ok(Box::new(GraphApp { shared }) as Box<dyn eframe::App>)
                }),
            )
            .await;
        if let Err(e) = result {
            web_sys::console::error_1(&format!("egui runner failed: {e:?}").into());
        }
    });
}

struct GraphApp {
    shared: Shared,
}

impl eframe::App for GraphApp {
    // egui 0.35 is Ui-first: we get a `&mut Ui` (the central area), not a Context.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let m = self.shared.borrow();
        ui.heading("graph explorer");
        ui.separator();
        if m.entries.is_empty() {
            ui.weak("(list a directory to populate)");
        }
        // ponytail: a real node-link graph render goes here; for now the listing is
        // drawn as a flat set of nodes.
        for (path, is_dir) in &m.entries {
            ui.label(format!("{} {}", if *is_dir { "▸" } else { "•" }, path));
        }
    }
}
