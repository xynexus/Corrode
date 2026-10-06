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
    /// `plan` is the turn: per-task ids restart at 0 every turn.
    Agent { plan: String, id: u64, text: String },
    Tool { call: String, observation: String },
    Turn { plan_id: String },
    Doc { text: String, grounded_on: Vec<String> },
    Error(String),
    /// Socket-level notices (open failure, undecodable frame, close).
    Ws(String),
}

/// Open the socket and keep it open: return the sender UI callbacks push
/// `AgentCommand`s into, and reconnect whenever the socket closes. The commands that
/// bind a connection (`Authenticate`, `SelectRepo`) are resent on every new socket
/// (`corrode_core::reattach`, which binds the default repo when none was selected),
/// so it lands on the same session, and the daemon replays that session's recent
/// turn events -- a reload, a sleep or a proxy restart no longer loses a turn's
/// answers. The console is cleared on reconnect and rebuilt from that replay.
pub fn spawn_agent(
    url: String,
    shared: Shared,
    log: RwSignal<Vec<LogEntry>>,
    entries: RwSignal<Vec<(String, bool, Option<String>)>>,
    approvals: RwSignal<Vec<(u64, String)>>,
    busy: RwSignal<bool>,
) -> UnboundedSender<AgentCommand> {
    use futures::future::{select, Either};
    let (cmd_tx, mut cmd_rx) = unbounded::<AgentCommand>();
    spawn_local(async move {
        let mut binding: Vec<AgentCommand> = Vec::new();
        let mut unsent: Vec<AgentCommand> = Vec::new();
        let mut first = true;
        loop {
            let ws = match WebSocket::open(&url) {
                Ok(ws) => ws,
                Err(e) => {
                    log.update(|l| l.push(LogEntry::Ws(format!("open failed: {e:?}"))));
                    sleep_ms(2000).await;
                    continue;
                }
            };
            if !first {
                log.set(vec![LogEntry::Ws("reconnected; replaying this session's recent turns".into())]);
                approvals.set(Vec::new());
            }
            first = false;
            let (mut sink, mut stream) = ws.split();
            let resend: Vec<AgentCommand> =
                corrode_core::reattach(&binding).into_iter().chain(unsent.drain(..)).collect();
            let mut alive = true;
            for cmd in resend {
                if alive && send(&mut sink, &cmd).await.is_err() {
                    alive = false;
                }
                // `reattach` adds a ListTurns to every resend; don't queue a second.
                if !alive && !is_binding(&cmd) && !matches!(cmd, AgentCommand::ListTurns) {
                    unsent.push(cmd);
                }
            }
            while alive {
                match select(cmd_rx.next(), stream.next()).await {
                    Either::Left((None, _)) => return, // the UI is gone
                    Either::Left((Some(cmd), _)) => {
                        if is_binding(&cmd) {
                            let same = std::mem::discriminant(&cmd);
                            binding.retain(|c| std::mem::discriminant(c) != same);
                            binding.push(cmd.clone());
                        }
                        if send(&mut sink, &cmd).await.is_err() {
                            if !is_binding(&cmd) {
                                unsent.push(cmd);
                            }
                            alive = false;
                        }
                    }
                    Either::Right((Some(Ok(msg)), _)) => {
                        let txt = match msg {
                            Message::Text(t) => t,
                            Message::Bytes(b) => String::from_utf8_lossy(&b).into_owned(),
                        };
                        match serde_json::from_str::<AgentEvent>(&txt) {
                            Ok(ev) => apply_event(ev, None, &shared, log, entries, approvals, busy),
                            Err(e) => log.update(|l| {
                                l.push(LogEntry::Ws(format!("undecodable event: {e}")))
                            }),
                        }
                    }
                    Either::Right(_) => alive = false,
                }
            }
            log.update(|l| l.push(LogEntry::Ws("agent socket closed; reconnecting".into())));
            sleep_ms(2000).await;
        }
    });
    cmd_tx
}

/// Commands that bind a connection to its session, resent on every new socket.
fn is_binding(cmd: &AgentCommand) -> bool {
    matches!(cmd, AgentCommand::Authenticate { .. } | AgentCommand::SelectRepo { .. })
}

async fn send(
    sink: &mut futures::stream::SplitSink<WebSocket, Message>,
    cmd: &AgentCommand,
) -> Result<(), ()> {
    let txt = serde_json::to_string(cmd).map_err(|_| ())?;
    sink.send(Message::Text(txt)).await.map_err(|_| ())
}

async fn sleep_ms(ms: i32) {
    let wait = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(wait).await;
}

fn apply_event(
    ev: AgentEvent,
    plan: Option<&str>,
    shared: &Shared,
    log: RwSignal<Vec<LogEntry>>,
    entries: RwSignal<Vec<(String, bool, Option<String>)>>,
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
            // egui graph panel wants just (path, is_dir); the DOM tree also wants the
            // graph node_id (to mark tracked files + pivot to provenance on click).
            let egui_rows: Vec<(String, bool)> = es.iter().map(|e| (e.path.clone(), e.is_dir)).collect();
            {
                let mut m = shared.borrow_mut();
                m.entries = egui_rows;
                if let Some(ctx) = &m.egui_ctx {
                    ctx.request_repaint();
                }
            }
            let rows: Vec<(String, bool, Option<String>)> =
                es.into_iter().map(|e| (e.path, e.is_dir, e.node_id)).collect();
            entries.set(rows);
        }
        // Incremental streamed output: append to this id's entry, or start one.
        AgentEvent::SubagentDelta { id, text } => log.update(|l| {
            let plan = plan.unwrap_or_default();
            match l.iter_mut().rev().find(
                |e| matches!(e, LogEntry::Agent { plan: p, id: i, .. } if *i == id && p == plan),
            ) {
                Some(LogEntry::Agent { text: t, .. }) => t.push_str(&text),
                _ => l.push(LogEntry::Agent { plan: plan.to_string(), id, text }),
            }
        }),
        // Authoritative full text: finalize this id's entry (reconciling any streamed
        // deltas), or start one when nothing streamed (non-streaming mode).
        AgentEvent::SubagentOutput { id, text } => log.update(|l| {
            let plan = plan.unwrap_or_default();
            match l.iter_mut().rev().find(
                |e| matches!(e, LogEntry::Agent { plan: p, id: i, .. } if *i == id && p == plan),
            ) {
                Some(LogEntry::Agent { text: t, .. }) => *t = text,
                _ => l.push(LogEntry::Agent { plan: plan.to_string(), id, text }),
            }
        }),
        // A mutating tool call blocked on a human; the console renders the queue
        // with approve/deny buttons that reply `ApprovalResponse`.
        AgentEvent::ApprovalRequest { id, action } => approvals.update(|a| {
            if !a.iter().any(|(i, _)| *i == id) {
                a.push((id, action))
            }
        }),
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
        // A turn's events, tagged with the turn.
        AgentEvent::Turn { plan_id, event } => {
            apply_event(*event, Some(&plan_id), shared, log, entries, approvals, busy)
        }
        AgentEvent::TurnList { turns } => log.update(|l| {
            for t in turns {
                l.push(LogEntry::Ws(format!("{} [{}] {}", t.plan_id, t.status, t.prompt)));
            }
        }),
        AgentEvent::TurnStarted { .. } => busy.set(true),
        AgentEvent::TurnComplete { plan_id } => {
            busy.set(false);
            log.update(|l| l.push(LogEntry::Turn { plan_id }));
        }
        AgentEvent::Error { message } => log.update(|l| l.push(LogEntry::Error(message))),
    }
}
