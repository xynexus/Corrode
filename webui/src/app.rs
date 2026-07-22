//! The Leptos shell: header, a filesystem/repo explorer, the egui canvas, and the
//! agent interface (prompt in, streamed output). CSR only — state lives in the
//! daemon, reached over the `/agent` websocket.

use corrode_core::{AgentCommand, Priority};
use leptos::html;
use leptos::prelude::*;

use crate::model;
use crate::{egui_panel, ws};

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

    // Hand the egui panels the canvas (once Leptos mounts it) plus the command
    // sender, so terminal keystrokes flow back out as `TerminalInput`.
    let canvas_ref = NodeRef::<html::Canvas>::new();
    {
        let egui_tx = cmd_tx.clone();
        Effect::new(move |_| {
            if let Some(canvas) = canvas_ref.get() {
                egui_panel::start(canvas, shared.clone(), egui_tx.clone());
            }
        });
    }

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

            <section class="canvas-wrap">
                <canvas node_ref=canvas_ref class="egui-canvas"></canvas>
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
