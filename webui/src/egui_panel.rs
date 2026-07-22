//! The egui canvas panel: the virtual terminal + graph explorer, drawn with
//! immediate-mode egui on a `<canvas>` the Leptos shell owns.
//!
//! eframe's `WebRunner` drives its own render loop on the canvas; we hand it a
//! clone of the [`Shared`] model (what the websocket task pushes) and the command
//! sender (so terminal keystrokes flow back out as `TerminalInput`). Kept small —
//! the DOM shell owns layout/input for everything else.

use corrode_core::AgentCommand;
use eframe::CreationContext;
use futures::channel::mpsc::UnboundedSender;
use web_sys::HtmlCanvasElement;

use crate::model::Shared;

/// Terminal session id for `TerminalInput`. One web terminal for now.
const SESSION: &str = "web";

/// Start the egui app on `canvas`. Runs asynchronously (eframe's start is async);
/// stores the egui `Context` into the shared model so async pushes can repaint.
pub fn start(canvas: HtmlCanvasElement, shared: Shared, cmd_tx: UnboundedSender<AgentCommand>) {
    let runner = eframe::WebRunner::new();
    wasm_bindgen_futures::spawn_local(async move {
        let result = runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc: &CreationContext<'_>| {
                    shared.borrow_mut().egui_ctx = Some(cc.egui_ctx.clone());
                    Ok(Box::new(TermGraphApp { shared, cmd_tx }) as Box<dyn eframe::App>)
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
    cmd_tx: UnboundedSender<AgentCommand>,
}

impl eframe::App for TermGraphApp {
    // egui 0.35 is Ui-first: we get a `&mut Ui` (the central area), not a Context.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("virtual terminal");

        // Clone the scrollback out so we don't hold the model borrow across the UI.
        let term_text = {
            let m = self.shared.borrow();
            if m.terminal.is_empty() {
                "(click here, then type — keystrokes go to the daemon)".to_owned()
            } else {
                m.terminal.clone()
            }
        };

        let resp = egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .max_height(ui.available_height() * 0.6)
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(term_text).monospace())
                        .sense(egui::Sense::click()),
                )
            })
            .inner;

        if resp.clicked() {
            resp.request_focus();
        }
        if resp.has_focus() {
            // Only capture keys while the terminal is focused (input events reach
            // egui only when the canvas has browser focus, which the click grants).
            let events = ui.input(|i| i.events.clone());
            let bytes = keystrokes_to_bytes(&events);
            if !bytes.is_empty() {
                let _ = self.cmd_tx.unbounded_send(AgentCommand::TerminalInput {
                    session: SESSION.to_owned(),
                    data: bytes,
                });
            }
        } else {
            ui.weak("click the terminal to type");
        }

        ui.separator();
        ui.heading("graph explorer");
        let m = self.shared.borrow();
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

/// Translate one frame's egui input events into a terminal byte stream.
/// Printable input arrives as `Text`/`Paste`; the rest we map to the usual control
/// bytes (Enter -> CR, Backspace -> DEL, Ctrl-<letter> -> 0x01..0x1a, ...).
// ponytail: egui may swallow Tab for focus navigation before we see it; a real
// terminal would consume Tab/arrows first via `input_mut`.
fn keystrokes_to_bytes(events: &[egui::Event]) -> Vec<u8> {
    let mut out = Vec::new();
    for ev in events {
        match ev {
            egui::Event::Text(t) | egui::Event::Paste(t) => out.extend_from_slice(t.as_bytes()),
            egui::Event::Key {
                key,
                pressed: true,
                modifiers,
                ..
            } => {
                // Ctrl-<letter> -> control byte (Ctrl-C = 0x03, Ctrl-D = 0x04, ...).
                if modifiers.ctrl && !modifiers.shift {
                    let name = key.name();
                    let b = name.as_bytes();
                    if b.len() == 1 && b[0].is_ascii_uppercase() {
                        out.push(b[0] & 0x1f);
                        continue;
                    }
                }
                match key {
                    egui::Key::Enter => out.push(b'\r'),
                    egui::Key::Backspace => out.push(0x7f),
                    egui::Key::Tab => out.push(b'\t'),
                    egui::Key::Escape => out.push(0x1b),
                    egui::Key::Delete => out.extend_from_slice(b"\x1b[3~"),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    out
}
