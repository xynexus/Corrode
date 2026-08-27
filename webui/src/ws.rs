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

/// One agent-console entry, typed at receive time so the view styles each event
/// kind instead of pattern-matching strings back apart.
#[derive(Clone)]
pub enum LogEntry {
    Agent { id: u64, text: String },
    Tool { call: String, observation: String },
    Turn { plan_id: String },
    Doc { text: String, grounded_on: Vec<String> },
    Error(String),
    /// Socket-level notices (open failure, undecodable frame, close).
    Ws(String),
}

/// Open the socket, wire both pump loops, and return the sender UI callbacks push
/// `AgentCommand`s into. Failures surface in `log` rather than panicking.
pub fn spawn_agent(
    url: String,
    shared: Shared,
    log: RwSignal<Vec<LogEntry>>,
    entries: RwSignal<Vec<(String, bool)>>,
    approvals: RwSignal<Vec<(u64, String)>>,
    busy: RwSignal<bool>,
) -> UnboundedSender<AgentCommand> {
    let (cmd_tx, mut cmd_rx) = unbounded::<AgentCommand>();

    let ws = match WebSocket::open(&url) {
        Ok(ws) => ws,
        Err(e) => {
            log.update(|l| l.push(LogEntry::Ws(format!("open failed: {e:?}"))));
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
                Ok(ev) => apply_event(ev, &shared, log, entries, approvals, busy),
                Err(e) => log.update(|l| l.push(LogEntry::Ws(format!("undecodable event: {e}")))),
            }
        }
        log.update(|l| l.push(LogEntry::Ws("agent socket closed".into())));
    });

    cmd_tx
}

fn apply_event(
    ev: AgentEvent,
    shared: &Shared,
    log: RwSignal<Vec<LogEntry>>,
    entries: RwSignal<Vec<(String, bool)>>,
    approvals: RwSignal<Vec<(u64, String)>>,
    busy: RwSignal<bool>,
) {
    match ev {
        // Terminal bytes -> the xterm.js terminal.
        AgentEvent::TerminalOutput { data, .. } => {
            crate::term::write(&data);
        }
        // Session/auth notices -> the console. (The repo tree still refreshes via a
        // ListDir the app fires after selecting a repo.)
        AgentEvent::AuthOk { user } => {
            log.update(|l| l.push(LogEntry::Ws(format!("authenticated as {user}"))))
        }
        AgentEvent::AuthRequired => log.update(|l| {
            l.push(LogEntry::Ws("authentication required — sign in first".into()))
        }),
        AgentEvent::RepoSelected { path, user } => log.update(|l| {
            let who = if user.is_empty() { String::new() } else { format!(" ({user})") };
            l.push(LogEntry::Ws(format!("repo selected: {path}{who}")))
        }),
        // File view (explorer click) -> a collapsible code block in the console,
        // reusing the tool-result rendering.
        AgentEvent::FileContent { path, content, truncated } => log.update(|l| {
            let call = if truncated { format!("{path} (truncated)") } else { path };
            l.push(LogEntry::Tool { call, observation: content })
        }),
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
        // Incremental streamed output: append to this id's entry, or start one.
        AgentEvent::SubagentDelta { id, text } => log.update(|l| {
            match l.iter_mut().rev().find(|e| matches!(e, LogEntry::Agent { id: i, .. } if *i == id)) {
                Some(LogEntry::Agent { text: t, .. }) => t.push_str(&text),
                _ => l.push(LogEntry::Agent { id, text }),
            }
        }),
        // Authoritative full text: finalize this id's entry (reconciling any streamed
        // deltas), or start one when nothing streamed (non-streaming mode).
        AgentEvent::SubagentOutput { id, text } => log.update(|l| {
            match l.iter_mut().rev().find(|e| matches!(e, LogEntry::Agent { id: i, .. } if *i == id)) {
                Some(LogEntry::Agent { text: t, .. }) => *t = text,
                _ => l.push(LogEntry::Agent { id, text }),
            }
        }),
        // A mutating tool call blocked on a human; the console renders the queue
        // with approve/deny buttons that reply `ApprovalResponse`.
        AgentEvent::ApprovalRequest { id, action } => {
            approvals.update(|a| a.push((id, action)))
        }
        AgentEvent::DocAnswer { text, grounded_on } => {
            log.update(|l| l.push(LogEntry::Doc { text, grounded_on }))
        }
        AgentEvent::ToolResult { call, observation, .. } => {
            log.update(|l| l.push(LogEntry::Tool { call, observation }))
        }
        // The turn's provenance graph -> the egui canvas.
        AgentEvent::PlanGraph { nodes, .. } => {
            let mut m = shared.borrow_mut();
            m.plan_nodes = nodes;
            if let Some(ctx) = &m.egui_ctx {
                ctx.request_repaint();
            }
        }
        // Click-to-expand: fold a node's persisted neighborhood into the canvas.
        AgentEvent::Neighbors { nodes, .. } => {
            let mut m = shared.borrow_mut();
            m.merge_nodes(nodes);
            if let Some(ctx) = &m.egui_ctx {
                ctx.request_repaint();
            }
        }
        AgentEvent::DocList { docs } => log.update(|l| {
            let line = if docs.is_empty() {
                "no documents ingested yet".to_string()
            } else {
                let list = docs
                    .iter()
                    .map(|d| format!("{} ({})", d.title, d.id))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!("{} doc(s): {list}", docs.len())
            };
            l.push(LogEntry::Ws(line));
        }),
        AgentEvent::DocIngested { path, doc_id, chunks, persisted } => {
            let note = if persisted { "stored" } else { "parsed (store unavailable)" };
            log.update(|l| {
                l.push(LogEntry::Ws(format!("ingested {path} -> {doc_id}: {chunks} chunks {note}")))
            });
        }
        AgentEvent::TurnComplete { plan_id } => {
            busy.set(false);
            log.update(|l| l.push(LogEntry::Turn { plan_id }));
        }
        AgentEvent::Error { message } => log.update(|l| l.push(LogEntry::Error(message))),
    }
}
