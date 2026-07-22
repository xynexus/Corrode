//! The egui canvas panel: the virtual terminal + graph explorer, drawn with
//! immediate-mode egui on a `<canvas>` the Leptos shell owns.
//!
//! eframe's `WebRunner` drives its own render loop on the canvas; we hand it a
//! clone of the [`Shared`] model so it renders whatever the websocket task has
//! pushed. Kept deliberately small — the DOM shell owns layout/input, this owns the
//! two panels that are custom-drawn rather than DOM.

use eframe::CreationContext;
use web_sys::HtmlCanvasElement;

use crate::model::Shared;

/// Start the egui app on `canvas`. Runs asynchronously (eframe's start is async);
/// stores the egui `Context` into the shared model so async pushes can repaint.
pub fn start(canvas: HtmlCanvasElement, shared: Shared) {
    let runner = eframe::WebRunner::new();
    wasm_bindgen_futures::spawn_local(async move {
        let creator_model = shared.clone();
        let result = runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc: &CreationContext<'_>| {
                    creator_model.borrow_mut().egui_ctx = Some(cc.egui_ctx.clone());
                    Ok(Box::new(TermGraphApp {
                        shared: creator_model.clone(),
                    }) as Box<dyn eframe::App>)
                }),
            )
            .await;
        if let Err(e) = result {
            web_sys::console::error_1(&format!("egui runner failed: {e:?}").into());
        }
    });
}

struct TermGraphApp {
    shared: Shared,
}

impl eframe::App for TermGraphApp {
    // egui 0.35 is Ui-first: we get a `&mut Ui` (already the central area), not a
    // Context. Lay the terminal (top) and graph explorer (below) onto it directly.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let m = self.shared.borrow();

        ui.heading("virtual terminal");
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .max_height(ui.available_height() * 0.6)
            .show(ui, |ui| {
                let text = if m.terminal.is_empty() {
                    "(no terminal output yet)"
                } else {
                    m.terminal.as_str()
                };
                ui.monospace(text);
            });

        ui.separator();
        ui.heading("graph explorer");
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
