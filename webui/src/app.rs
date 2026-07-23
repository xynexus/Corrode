//! The Leptos shell: a filesystem/repo explorer, the xterm.js terminal + egui graph
//! canvas (center), and the agent console (streamed output + prompt). CSR only —
//! state lives in the daemon, reached over the `/agent` websocket.

use corrode_core::{AgentCommand, Priority};
use leptos::html;
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::model;
use crate::{egui_panel, term, ws};

const SESSION: &str = "web";

/// Same-origin `/agent` websocket URL, `ws://` or `wss://` per the page scheme.
fn agent_ws_url() -> String {
    let loc = web_sys::window().expect("window").location();
    let scheme = match loc.protocol().as_deref() {
        Ok("https:") => "wss",
        _ => "ws",
    };
    let host = loc.host().unwrap_or_else(|_| "127.0.0.1:8787".to_string());
    format!("{scheme}://{host}/agent")
}

#[component]
pub fn App() -> impl IntoView {
    let shared = model::shared();
    let log = RwSignal::new(Vec::<String>::new());
    let entries = RwSignal::new(Vec::<(String, bool)>::new());

    let cmd_tx = ws::spawn_agent(agent_ws_url(), shared.clone(), log, entries);

    // xterm.js terminal: mount on the div once Leptos renders it. Keystrokes ->
    // TerminalInput, geometry -> TerminalResize; pty output arrives via ws::write.
    let term_ref = NodeRef::<html::Div>::new();
    {
        let cmd_tx = cmd_tx.clone();
        Effect::new(move |_| {
            if let Some(div) = term_ref.get() {
                let el: web_sys::HtmlElement = div.unchecked_into();
                let tx_data = cmd_tx.clone();
                let tx_resize = cmd_tx.clone();
                term::init(
                    el,
                    move |s: String| {
                        let _ = tx_data.unbounded_send(AgentCommand::TerminalInput {
                            session: SESSION.into(),
                            data: s.into_bytes(),
                        });
                    },
                    move |cols: u32, rows: u32| {
                        let _ = tx_resize.unbounded_send(AgentCommand::TerminalResize {
                            session: SESSION.into(),
                            cols: cols as u16,
                            rows: rows as u16,
                        });
                    },
                );
            }
        });
    }

    // egui/WebGL graph canvas.
    let canvas_ref = NodeRef::<html::Canvas>::new();
    Effect::new(move |_| {
        if let Some(canvas) = canvas_ref.get() {
            egui_panel::start(canvas, shared.clone());
        }
    });

    let prompt = RwSignal::new(String::new());
    let send_prompt = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let text = prompt.get();
            if !text.trim().is_empty() {
                let _ = cmd_tx.unbounded_send(AgentCommand::Prompt {
                    text,
                    priority: Priority::Default,
                });
                prompt.set(String::new());
            }
        }
    };
    let list_root = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let _ = cmd_tx.unbounded_send(AgentCommand::ListDir {
                path: String::new(),
            });
        }
    };

    view! {
        <header class="topbar"><span class="brand">"Corrode"</span>" swarm console"</header>
        <div class="cols">
            <section class="explorer">
                <div class="bar">
                    <span>"explorer"</span>
                    <button on:click=list_root>"list root"</button>
                </div>
                <ul class="tree">
                    {move || entries.get().into_iter().map(|(path, is_dir)| view! {
                        <li class:dir=is_dir>{if is_dir { "📁 " } else { "📄 " }}{path}</li>
                    }).collect_view()}
                </ul>
            </section>

            <section class="center">
                <div node_ref=term_ref class="terminal"></div>
                <canvas node_ref=canvas_ref class="graph-canvas"></canvas>
            </section>

            <section class="agent">
                <div class="log">
                    {move || log.get().into_iter().map(|line| view! {
                        <div class="line">{line}</div>
                    }).collect_view()}
                </div>
                <div class="prompt">
                    <input
                        prop:value=move || prompt.get()
                        on:input=move |e| prompt.set(event_target_value(&e))
                        placeholder="prompt the swarm..."
                    />
                    <button on:click=send_prompt>"send"</button>
                </div>
            </section>
        </div>
    }
}
