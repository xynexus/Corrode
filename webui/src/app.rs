//! The Leptos shell: a filesystem/repo explorer, the xterm.js terminal + egui graph
//! canvas (center), and the agent console (streamed output + prompt). CSR only —
//! state lives in the daemon, reached over the `/agent` websocket.

use corrode_core::{AgentCommand, Priority};
use leptos::html;
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::model;
use crate::{egui_panel, term, ws};

/// A per-browser-tab terminal session id, stable across reloads (so a reload
/// adopts the running shell) but unique per tab (so two tabs don't share one pty).
/// Backed by `sessionStorage`, which is exactly per-tab-and-survives-reload; a
/// random fallback keeps things working if storage is unavailable.
fn terminal_session_id() -> String {
    let store = web_sys::window().and_then(|w| w.session_storage().ok().flatten());
    if let Some(s) = &store {
        if let Ok(Some(id)) = s.get_item("corrode-term-id") {
            return id;
        }
    }
    let id = format!("tab-{:x}", (js_sys::Math::random() * 1e12) as u64);
    if let Some(s) = &store {
        let _ = s.set_item("corrode-term-id", &id);
    }
    id
}

/// Render one agent message (Markdown, possibly with `$…$` LaTeX) to HTML. KaTeX
/// renders the math afterward, over the mounted element. Raw HTML is dropped
/// (this feeds inner_html, so pass-through would be script injection), and link/
/// image URLs with a non-http(s) scheme are neutralized — DocAnswer now carries
/// verbatim ingested-document text, so `[x](javascript:…)` is attacker-reachable.
fn md_to_html(md: &str) -> String {
    use pulldown_cmark::{html, Event, Parser, Tag};
    let mut out = String::new();
    let events = Parser::new(md)
        .filter(|e| !matches!(e, Event::Html(_) | Event::InlineHtml(_)))
        .map(|e| match e {
            Event::Start(Tag::Link { link_type, dest_url, title, id }) => {
                Event::Start(Tag::Link { link_type, dest_url: safe_url(dest_url), title, id })
            }
            Event::Start(Tag::Image { link_type, dest_url, title, id }) => {
                Event::Start(Tag::Image { link_type, dest_url: safe_url(dest_url), title, id })
            }
            other => other,
        });
    html::push_html(&mut out, events);
    out
}

/// Allow only http/https/mailto or scheme-less (relative/anchor) URLs; replace
/// anything else (javascript:, data:, vbscript:, …) with a harmless anchor.
fn safe_url(url: pulldown_cmark::CowStr) -> pulldown_cmark::CowStr {
    let scheme_end = url.bytes().take_while(|&b| b != b':' && b != b'/' && b != b'#').count();
    let has_scheme = url.as_bytes().get(scheme_end) == Some(&b':');
    if !has_scheme {
        return url; // relative path or #anchor
    }
    let scheme = url[..scheme_end].to_ascii_lowercase();
    if matches!(scheme.as_str(), "http" | "https" | "mailto") {
        url
    } else {
        "#".into()
    }
}

/// HTML-escape untrusted text bound for `inner_html` (tool observations echo repo
/// content; markdown filtering doesn't apply to these non-markdown entries).
fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// One console entry -> its HTML. Typed rendering: agent output stays markdown
/// (badge-colored per subagent via a golden-angle hue), tool results collapse into
/// a native `<details>` (no JS), turns render as dividers, errors in red.
fn entry_html(e: &ws::LogEntry) -> String {
    use ws::LogEntry::*;
    match e {
        Agent { id, text } => {
            let hue = (*id * 137) % 360;
            format!(
                "<div class=\"msg agent\"><span class=\"agent-badge\" \
                 style=\"background:hsl({hue} 45% 28%)\">a{id}</span>\
                 <div class=\"agent-body\">{}</div></div>",
                md_to_html(text)
            )
        }
        Tool { call, observation } => format!(
            "<details class=\"tool\"><summary>{}</summary><pre>{}</pre></details>",
            esc(call),
            esc(observation)
        ),
        Turn { plan_id } => format!("<div class=\"turn\">turn {} settled</div>", esc(plan_id)),
        Doc { text, grounded_on } => format!(
            "<div class=\"msg doc\">{}<div class=\"grounded\">grounded: {}</div></div>",
            md_to_html(text),
            esc(&grounded_on.join(", "))
        ),
        Error(m) => format!("<div class=\"msg error\">{}</div>", esc(m)),
        Ws(m) => format!("<div class=\"msg ws\">{}</div>", esc(m)),
    }
}

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
    let log = RwSignal::new(Vec::<ws::LogEntry>::new());
    // (path, is_dir, graph node_id) — node_id set for files the graph tracks.
    let entries = RwSignal::new(Vec::<(String, bool, Option<String>)>::new());
    let approvals = RwSignal::new(Vec::<(u64, String)>::new());
    // In-flight turn indicator: set on Prompt, cleared by TurnComplete. ponytail:
    // one flag, not per-plan tracking — a second concurrent prompt's completion
    // reads as idle early; track plan ids when concurrent turns matter.
    let busy = RwSignal::new(false);

    let cmd_tx = ws::spawn_agent(agent_ws_url(), shared.clone(), log, entries, approvals, busy);

    // xterm.js terminal: mount on the div once Leptos renders it. Keystrokes ->
    // TerminalInput, geometry -> TerminalResize; pty output arrives via ws::write.
    let term_ref = NodeRef::<html::Div>::new();
    {
        let cmd_tx = cmd_tx.clone();
        let term_session = terminal_session_id();
        Effect::new(move |_| {
            if let Some(div) = term_ref.get() {
                let el: web_sys::HtmlElement = div.unchecked_into();
                let tx_data = cmd_tx.clone();
                let tx_resize = cmd_tx.clone();
                let sid_data = term_session.clone();
                let sid_resize = term_session.clone();
                term::init(
                    el,
                    move |s: String| {
                        let _ = tx_data.unbounded_send(AgentCommand::TerminalInput {
                            session: sid_data.clone(),
                            data: s.into_bytes(),
                        });
                    },
                    move |cols: u32, rows: u32| {
                        let _ = tx_resize.unbounded_send(AgentCommand::TerminalResize {
                            session: sid_resize.clone(),
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
    let canvas_tx = cmd_tx.clone();
    Effect::new(move |_| {
        if let Some(canvas) = canvas_ref.get() {
            egui_panel::start(canvas, shared.clone(), canvas_tx.clone());
        }
    });

    // Agent console: render each message as Markdown -> HTML, then KaTeX over it.
    // One effect owns both steps so innerHTML is set before math renders (no
    // two-effect ordering race), and it auto-scrolls to the newest message.
    let console_ref = NodeRef::<html::Div>::new();
    Effect::new(move |_| {
        let html = log.get().iter().map(entry_html).collect::<String>();
        if let Some(div) = console_ref.get() {
            let el: web_sys::HtmlElement = div.unchecked_into();
            el.set_inner_html(&html);
            term::render_math(&el);
            el.set_scroll_top(el.scroll_height());
        }
    });

    // Pane geometry — dragged via the splitter divs (pointer capture + buttons
    // check, so no drag flag; a divider keeps receiving moves outside its box).
    let explorer_w = RwSignal::new(240.0f64);
    let agent_w = RwSignal::new(340.0f64);
    let term_frac = RwSignal::new(0.6f64);
    // xterm's fit addon listens on window resize; fire it once a drag ends.
    let refit = || {
        if let Some(w) = web_sys::window() {
            if let Ok(e) = web_sys::Event::new("resize") {
                let _ = w.dispatch_event(&e);
            }
        }
    };
    let grab = |ev: &web_sys::PointerEvent| {
        let el = event_target::<web_sys::HtmlElement>(ev);
        let _ = el.set_pointer_capture(ev.pointer_id());
    };

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
                busy.set(true);
            }
        }
    };
    // The explorer's current directory ("" = repo root). Navigating a dir sets this
    // and re-lists; DirListing replaces the visible entries.
    let cwd = RwSignal::new(String::new());
    let list_root = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            cwd.set(String::new());
            let _ = cmd_tx.unbounded_send(AgentCommand::ListDir { path: String::new() });
        }
    };
    // Go up one directory from the current cwd.
    let go_up = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let here = cwd.get();
            let parent = here.rsplit_once('/').map(|(p, _)| p.to_string()).unwrap_or_default();
            cwd.set(parent.clone());
            let _ = cmd_tx.unbounded_send(AgentCommand::ListDir { path: parent });
        }
    };

    // Repo selection (Phase 2): bind this connection to a repo, then refresh the tree.
    let repo_path = RwSignal::new(String::new());
    let select_repo = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let path = repo_path.get();
            if !path.trim().is_empty() {
                let _ = cmd_tx.unbounded_send(AgentCommand::SelectRepo { path });
                let _ = cmd_tx.unbounded_send(AgentCommand::ListDir { path: String::new() });
            }
        }
    };

    // Docs (GraphRAG): one input, two actions — ask a question of the ingested docs
    // (DocQuery -> a synthesized, grounded answer in the console) or ingest a file
    // by path (DocIngest). Answers/notices render in the same console as the swarm.
    let doc_input = RwSignal::new(String::new());
    let ask_docs = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let question = doc_input.get();
            if !question.trim().is_empty() {
                let _ = cmd_tx.unbounded_send(AgentCommand::DocQuery { question });
                doc_input.set(String::new());
            }
        }
    };
    let ingest_doc = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let path = doc_input.get();
            if !path.trim().is_empty() {
                let _ = cmd_tx.unbounded_send(AgentCommand::DocIngest { path });
                doc_input.set(String::new());
            }
        }
    };
    let list_docs = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let _ = cmd_tx.unbounded_send(AgentCommand::ListDocs);
        }
    };

    // Authentication (Phase 3): sign in when the daemon has a user table configured.
    let auth_user = RwSignal::new(String::new());
    let auth_token = RwSignal::new(String::new());
    let sign_in = {
        let cmd_tx = cmd_tx.clone();
        move |_| {
            let user = auth_user.get();
            if !user.trim().is_empty() {
                let _ = cmd_tx.unbounded_send(AgentCommand::Authenticate {
                    user,
                    token: auth_token.get(),
                });
                auth_token.set(String::new());
            }
        }
    };

    // cmd sender for the explorer tree's per-file click (read the file into the console).
    let tree_tx = cmd_tx.clone();

    view! {
        <header class="topbar">
            <span class="brand">"Corrode"</span>" swarm console"
            <span class="auth">
                <input
                    class="auth-user"
                    placeholder="user"
                    prop:value=move || auth_user.get()
                    on:input=move |ev| auth_user.set(event_target_value(&ev))
                />
                <input
                    class="auth-token"
                    type="password"
                    placeholder="token"
                    prop:value=move || auth_token.get()
                    on:input=move |ev| auth_token.set(event_target_value(&ev))
                />
                <button on:click=sign_in>"sign in"</button>
            </span>
            <span class="status" class:busy=move || busy.get()>
                {move || if busy.get() { "working…" } else { "idle" }}
            </span>
        </header>
        <div
            class="cols"
            style=move || format!(
                "grid-template-columns:{}px 5px 1fr 5px {}px",
                explorer_w.get(),
                agent_w.get()
            )
        >
            <section class="explorer">
                <div class="bar">
                    <span class="cwd" title="current directory">
                        {move || { let d = cwd.get(); if d.is_empty() { "/".to_string() } else { format!("/{d}") } }}
                    </span>
                    <span class="explorer-actions">
                        <button on:click=go_up title="up one directory">"↑"</button>
                        <button on:click=list_root>"root"</button>
                    </span>
                </div>
                <div class="bar repo-bar">
                    <input
                        class="repo-input"
                        placeholder="repo path…"
                        prop:value=move || repo_path.get()
                        on:input=move |ev| repo_path.set(event_target_value(&ev))
                    />
                    <button on:click=select_repo>"select"</button>
                </div>
                <ul class="tree">
                    {move || {
                        let tx = tree_tx.clone();
                        entries.get().into_iter().map(move |(path, is_dir, node_id)| {
                            let tx = tx.clone();
                            let p = path.clone();
                            let nid = node_id.clone();
                            // Dir -> descend (set cwd + re-list). File -> read into the
                            // console; if the graph tracks it, also pivot the graph view
                            // to its provenance (ListNeighbors on its code node).
                            let open = move |_| {
                                if is_dir {
                                    cwd.set(p.clone());
                                    let _ = tx.unbounded_send(AgentCommand::ListDir { path: p.clone() });
                                } else {
                                    let _ = tx.unbounded_send(AgentCommand::ReadFile { path: p.clone() });
                                    if let Some(id) = &nid {
                                        let _ = tx.unbounded_send(AgentCommand::ListNeighbors { node_id: id.clone() });
                                    }
                                }
                            };
                            // Show just the basename — the bar shows the full cwd. A ●
                            // marks a file the provenance graph tracks (click to pivot).
                            let name = path.rsplit('/').next().unwrap_or(&path).to_string();
                            let tracked = !is_dir && node_id.is_some();
                            view! {
                                <li class:dir=is_dir class:file=!is_dir class:tracked=tracked on:click=open>
                                    {if is_dir { "📁 " } else { "📄 " }}{name}
                                    {if tracked { " ●" } else { "" }}
                                </li>
                            }
                        }).collect_view()
                    }}
                </ul>
            </section>

            <div
                class="divider-v"
                on:pointerdown=move |ev| grab(&ev)
                on:pointermove=move |ev| {
                    if ev.buttons() & 1 == 1 {
                        explorer_w.set((ev.client_x() as f64).clamp(140.0, 480.0));
                    }
                }
                on:pointerup=move |_| refit()
            ></div>

            <section class="center">
                <div
                    node_ref=term_ref
                    class="terminal"
                    style=move || format!("flex:0 0 {}%", term_frac.get() * 100.0)
                ></div>
                <div
                    class="divider-h"
                    on:pointerdown=move |ev| grab(&ev)
                    on:pointermove=move |ev| {
                        if ev.buttons() & 1 == 1 {
                            let el = event_target::<web_sys::HtmlElement>(&ev);
                            if let Some(parent) = el.parent_element() {
                                let r = parent.get_bounding_client_rect();
                                let frac = (ev.client_y() as f64 - r.top()) / r.height();
                                term_frac.set(frac.clamp(0.15, 0.85));
                            }
                        }
                    }
                    on:pointerup=move |_| refit()
                ></div>
                <canvas node_ref=canvas_ref class="graph-canvas"></canvas>
            </section>

            <div
                class="divider-v"
                on:pointerdown=move |ev| grab(&ev)
                on:pointermove=move |ev| {
                    if ev.buttons() & 1 == 1 {
                        let win = web_sys::window()
                            .and_then(|w| w.inner_width().ok())
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1200.0);
                        agent_w.set((win - ev.client_x() as f64).clamp(240.0, 640.0));
                    }
                }
                on:pointerup=move |_| refit()
            ></div>

            <section class="agent">
                <div node_ref=console_ref class="log"></div>
                <ul class="approvals">
                    {move || approvals.get().into_iter().map(|(id, action)| {
                        let decide = {
                            let cmd_tx = cmd_tx.clone();
                            move |approved: bool| {
                                let _ = cmd_tx.unbounded_send(
                                    AgentCommand::ApprovalResponse { id, approved },
                                );
                                approvals.update(|a| a.retain(|(i, _)| *i != id));
                            }
                        };
                        let deny = decide.clone();
                        view! {
                            <li class="approval">
                                <span>{action}</span>
                                <button on:click=move |_| decide(true)>"approve"</button>
                                <button on:click=move |_| deny(false)>"deny"</button>
                            </li>
                        }
                    }).collect_view()}
                </ul>
                <div class="prompt docs">
                    <input
                        prop:value=move || doc_input.get()
                        on:input=move |e| doc_input.set(event_target_value(&e))
                        placeholder="ask the docs, or a file path to ingest…"
                    />
                    <button on:click=ask_docs title="GraphRAG query over ingested docs">"ask"</button>
                    <button on:click=ingest_doc title="ingest the file at this path">"ingest"</button>
                    <button on:click=list_docs title="list ingested documents">"docs"</button>
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
