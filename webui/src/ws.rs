//! The `/agent` websocket client.
//!
//! One socket to `corrode-web` (which proxies to the daemon). UI commands flow in
//! through an mpsc channel and out as JSON `AgentCommand`; incoming JSON
//! `AgentEvent`s fan out to the Leptos signals (DOM) and the shared model (egui).
//! The frame encoding is exactly `corrode_core`'s serde-JSON, unchanged end to end.

use corrode_core::{AgentCommand, AgentEvent};
use futures::channel::mpsc::{unbounded, UnboundedSender};
use futures::{SinkExt, StreamExt};
use gloo_net::websocket::{futures::WebSocket, Message};
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::model::Shared;

/// Open the socket, wire both pump loops, and return the sender UI callbacks push
/// `AgentCommand`s into. Failures surface in `log` rather than panicking.
pub fn spawn_agent(
    url: String,
    shared: Shared,
    log: RwSignal<Vec<String>>,
    entries: RwSignal<Vec<(String, bool)>>,
) -> UnboundedSender<AgentCommand> {
    let (cmd_tx, mut cmd_rx) = unbounded::<AgentCommand>();

    let ws = match WebSocket::open(&url) {
        Ok(ws) => ws,
        Err(e) => {
            log.update(|l| l.push(format!("[ws] open failed: {e:?}")));
            return cmd_tx;
        }
    };
    let (mut sink, mut stream) = ws.split();

    // UI commands -> daemon
    spawn_local(async move {
        while let Some(cmd) = cmd_rx.next().await {
            if let Ok(txt) = serde_json::to_string(&cmd) {
                if sink.send(Message::Text(txt)).await.is_err() {
                    break;
                }
            }
        }
    });

    // daemon events -> UI
    spawn_local(async move {
        while let Some(Ok(msg)) = stream.next().await {
            let txt = match msg {
                Message::Text(t) => t,
                Message::Bytes(b) => String::from_utf8_lossy(&b).into_owned(),
            };
            match serde_json::from_str::<AgentEvent>(&txt) {
                Ok(ev) => apply_event(ev, &shared, log, entries),
                Err(e) => log.update(|l| l.push(format!("[ws] undecodable event: {e}"))),
            }
        }
        log.update(|l| l.push("[ws] agent socket closed".into()));
    });

    cmd_tx
}

fn apply_event(
    ev: AgentEvent,
    shared: &Shared,
    log: RwSignal<Vec<String>>,
    entries: RwSignal<Vec<(String, bool)>>,
) {
    match ev {
        // Terminal bytes -> the xterm.js terminal.
        AgentEvent::TerminalOutput { data, .. } => {
            crate::term::write(&data);
        }
        // Explorer listing -> both the DOM tree and the egui graph panel.
        AgentEvent::DirListing { entries: es, .. } => {
            let rows: Vec<(String, bool)> = es.into_iter().map(|e| (e.path, e.is_dir)).collect();
            {
                let mut m = shared.borrow_mut();
                m.entries = rows.clone();
                if let Some(ctx) = &m.egui_ctx {
                    ctx.request_repaint();
                }
            }
            entries.set(rows);
        }
        AgentEvent::SubagentOutput { id, text } => {
            log.update(|l| l.push(format!("[agent {id}] {text}")))
        }
        AgentEvent::DocAnswer { text, grounded_on } => {
            log.update(|l| l.push(format!("[doc] {text}  (grounded: {})", grounded_on.join(", "))))
        }
        AgentEvent::Error { message } => log.update(|l| l.push(format!("[error] {message}"))),
    }
}
