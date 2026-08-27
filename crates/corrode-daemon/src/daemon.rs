//! The daemon command loop: drain `AgentCommand`s, dispatch each, stream
//! `AgentEvent`s back. Transport-agnostic on purpose — it speaks mpsc channels, so
//! the same loop serves the in-process demo in `main` today and the `corrode-web`
//! websocket bridge later, without change.
//!
//! The daemon owns the host-side state the handlers reach through `&self`: the
//! swarm, the role->model assignments, the embedded graph store (HelixDB, when
//! built), and the VFS.

use crate::approval::ApprovalGate;
use crate::dialect::Dialects;
use crate::graph::GraphStore;
use crate::hipfire::Client;
use crate::plan_graph;
use crate::planner;
use crate::roles::{self, Role, RoleModels};
use crate::skills::SkillContext;
use crate::swarm::{Swarm, Task};
use crate::terminal::Terminals;
use crate::toolcall::ToolCaller;
use crate::tools::ToolBox;
use crate::vfs::Vfs;
use corrode_core::{AgentCommand, AgentEvent, Priority};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// How many relevance-ranked skills to surface in the shared prefix per turn.
const TOP_K_SKILLS: usize = 8;

pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    /// Embedded HelixDB. `None` unless built with `--features helix` and opened.
    /// `Arc` so a clone can move into `spawn_blocking` (LMDB writes are sync and
    /// fsync — never run them on a tokio worker inline).
    graph: Option<Arc<dyn GraphStore>>,
    /// Repo VFS. `Arc` so the tool-execution loop's `'static` future can own a clone.
    vfs: Arc<dyn Vfs>,
    /// Live pty-backed terminal sessions.
    terminals: Terminals,
    /// Agent Skills + AGENTS.md + the embedded index (fed into the shared prefix).
    skills: SkillContext,
    /// Reliable tool-calling for small models (the Needle shim). `None` unless built
    /// with `--features needle` and the assets were found; drives task emission.
    tool_caller: Option<Arc<dyn ToolCaller>>,
    /// Human-in-the-loop gate for mutating tool calls (write_file / run_command).
    /// Shared with the command loop, which resolves `ApprovalResponse`s.
    approvals: Arc<ApprovalGate>,
    /// Repo root — the working directory for `run_command` in the tool loop.
    repo_root: PathBuf,
    /// Monotonic id source for plan (provenance root) nodes, one per Prompt turn.
    next_plan_id: std::sync::atomic::AtomicU64,
    /// Skill name -> skill dir, for `run_skill_script` in the tool loop. Derived from
    /// `skills` at construction; `Arc` so the tool-loop future owns a clone.
    skill_scripts: Arc<std::collections::HashMap<String, PathBuf>>,
    /// Per-model tool dialects (schema/names/parse), matched to the tool-call model.
    dialects: Arc<Dialects>,
}

impl Daemon {
    pub fn new(
        swarm: Swarm,
        roles: RoleModels,
        graph: Option<Arc<dyn GraphStore>>,
        vfs: Arc<dyn Vfs>,
        skills: SkillContext,
        tool_caller: Option<Arc<dyn ToolCaller>>,
        repo_root: PathBuf,
        dialects: Arc<Dialects>,
    ) -> Self {
        let skill_scripts = Arc::new(skills.script_dirs());
        Self {
            swarm,
            roles,
            graph,
            vfs,
            terminals: Terminals::new(repo_root.clone()),
            skills,
            tool_caller,
            approvals: Arc::new(ApprovalGate::default()),
            repo_root,
            next_plan_id: std::sync::atomic::AtomicU64::new(0),
            skill_scripts,
            dialects,
        }
    }

    /// Run until the command channel closes. Dropping the sender ends the loop,
    /// which drops `events` and unblocks the consumer.
    ///
    /// `Prompt` handling is dispatched concurrently (it can be long-lived and may block
    /// on a human approval), so the loop keeps receiving — crucially, the
    /// `ApprovalResponse` that unblocks a waiting tool call. Other commands (terminal
    /// I/O, approvals) are handled inline to preserve their ordering.
    pub async fn run(
        self: Arc<Self>,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        while let Some(cmd) = commands.recv().await {
            // DocIngest/DocQuery spawn for the same reason Prompt does: convert +
            // embed can take seconds, and the loop must keep serving terminal I/O
            // + approvals.
            if matches!(
                cmd,
                AgentCommand::Prompt { .. }
                    | AgentCommand::DocIngest { .. }
                    | AgentCommand::DocQuery { .. }
            ) {
                let this = Arc::clone(&self);
                let events = events.clone();
                tokio::spawn(async move { this.handle(cmd, &events).await });
            } else {
                self.handle(cmd, &events).await;
            }
        }
    }

    async fn handle(&self, cmd: AgentCommand, events: &mpsc::Sender<AgentEvent>) {
        match cmd {
            AgentCommand::Prompt { text, priority } => {
                // The plan id exists before planning so TurnComplete is unconditional:
                // clients wait on it as the turn's terminal signal on every exit path.
                let plan_id = format!(
                    "plan-{}",
                    self.next_plan_id
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
                let (subtasks, prefix) = match self.plan(&text, priority).await {
                    Ok(planned) => planned,
                    Err(e) => {
                        let _ = events
                            .send(AgentEvent::Error {
                                message: format!("planning failed: {e}"),
                            })
                            .await;
                        let _ = events.send(AgentEvent::TurnComplete { plan_id }).await;
                        return;
                    }
                };

                // Seed the reactive plan graph with the decomposition, rooted at a plan
                // node. The scheduler fans ready tasks to the swarm and grows the graph
                // as agents emit follow-up work (a test contract, a research spin-off) —
                // dataflow, not a fixed fan-out. Real concurrency is hipfire's to bound
                // (admission control against its VRAM budget); nothing is capped here.
                let mut graph = plan_graph::PlanGraph::new(&plan_id);
                for s in subtasks {
                    graph.add(s.role, s.prompt, Vec::new());
                }

                let fanout = fanout_k();
                let review_model = self
                    .roles
                    .model_for(Role::Review)
                    .unwrap_or_default()
                    .to_string();
                // One observation memory for the whole turn (TODO item 5): every
                // task shares it, so siblings get cached results instead of
                // re-executing, and each launching task's tail carries a digest of
                // what the swarm already did.
                let turn_seen = Arc::new(std::sync::Mutex::new(SeenCalls::default()));
                let execute = |task: plan_graph::PlanTask| {
                    let client = self.swarm.client();
                    let model = self
                        .roles
                        .model_for(task.role)
                        .unwrap_or_default()
                        .to_string();
                    let review_model = review_model.clone();
                    let band = planner::band_for(task.role);
                    let prefix = prefix.clone();
                    let events = events.clone();
                    let tool_caller = self.tool_caller.clone();
                    let vfs = Arc::clone(&self.vfs);
                    let approvals = Arc::clone(&self.approvals);
                    let root = self.repo_root.clone();
                    let skill_scripts = Arc::clone(&self.skill_scripts);
                    let dialects = Arc::clone(&self.dialects);
                    let id = task.id;
                    let role = task.role;
                    // The swarm-knowledge digest rides the divergent tail — the
                    // shared prefix stays byte-identical (KV reuse).
                    let seen = Arc::clone(&turn_seen);
                    let prompt = match seen.lock().unwrap().digest(TURN_DIGEST_LINES) {
                        Some(d) => format!("{}\n\n{d}", task.prompt),
                        None => task.prompt.clone(),
                    };
                    async move {
                        // `artifacts` collects the files a tool-loop task wrote (its code
                        // nodes in provenance). Coder tasks fan out K read-only proposal
                        // attempts first when CORRODE_FANOUT > 1; everything else runs
                        // the capability paths directly (see `run_task`).
                        let mut artifacts = Vec::new();
                        let toolbox = ToolBox::new(vfs, root, skill_scripts);
                        let output = if role == Role::Coder && fanout > 1 {
                            run_fanout(
                                fanout,
                                &client,
                                &model,
                                &review_model,
                                band,
                                &dialects,
                                tool_caller.clone(),
                                toolbox,
                                &approvals,
                                &prefix,
                                role,
                                &prompt,
                                &events,
                                id,
                                &mut artifacts,
                                &seen,
                            )
                            .await
                        } else {
                            run_task(
                                &client,
                                &model,
                                band,
                                &dialects,
                                tool_caller.clone(),
                                toolbox,
                                &approvals,
                                &prefix,
                                role,
                                &prompt,
                                &events,
                                id,
                                &mut artifacts,
                                false,
                                &seen,
                            )
                            .await
                        };

                        let emitted = match &output {
                            Ok(text) => emit_followups(tool_caller, &dialects, text).await,
                            Err(e) => {
                                let _ = events
                                    .send(AgentEvent::Error {
                                        message: e.to_string(),
                                    })
                                    .await;
                                Vec::new()
                            }
                        };
                        plan_graph::Outcome {
                            output,
                            emitted,
                            artifacts,
                        }
                    }
                };
                plan_graph::run_reactive(&mut graph, &execute).await;

                // One plan-level review pass over the settled work: the review role
                // reads the digest (and, through its tools, the written files) and
                // routes fixes through the normal follow-up channel; the second
                // reactive drive runs the review task and whatever it emits.
                // ponytail: one round — loop-until-clean is the upgrade once fix
                // quality is measured.
                if plan_review_enabled() {
                    if let Some(digest) = graph.review_digest(REVIEW_OUTPUT_CAP) {
                        graph.add(Role::Review, planner::plan_review_task(&digest), Vec::new());
                        plan_graph::run_reactive(&mut graph, &execute).await;
                    }
                }

                // Tasks left pending after the scheduler settled had a failed or
                // unmet dependency (a failed emitter) — surface them rather than
                // dropping them silently.
                let stuck = graph.stuck();
                if !stuck.is_empty() {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!(
                                "{} task(s) could not be scheduled (a dependency failed)",
                                stuck.len()
                            ),
                        })
                        .await;
                }

                // Persist the plan's provenance (plan <- task/contract <- code) to the
                // graph store, so the code<->task<->plan lineage is queryable — and
                // ship it to the graph explorer.
                self.persist_provenance(&graph);
                let prov = graph.provenance();
                let nodes = prov
                    .nodes
                    .iter()
                    .map(|n| corrode_core::GraphNodeView {
                        id: n.id.clone(),
                        label: n.label.clone(),
                        kind: n.kind.as_str().to_string(),
                        edges_out: prov
                            .edges
                            .iter()
                            .filter(|e| e.from == n.id)
                            .map(|e| e.to.clone())
                            .collect(),
                    })
                    .collect();
                let _ = events
                    .send(AgentEvent::PlanGraph {
                        plan_id: plan_id.clone(),
                        nodes,
                    })
                    .await;

                // The turn's end is explicit: clients (and the e2e) wait on this
                // event, not on the stream going quiet.
                let _ = events.send(AgentEvent::TurnComplete { plan_id }).await;
            }
            AgentCommand::DocQuery { question } => {
                let ev = self.doc_query(&question).await;
                let _ = events.send(ev).await;
            }
            AgentCommand::DocIngest { path } => {
                let ev = self.ingest_doc(path).await;
                let _ = events.send(ev).await;
            }
            AgentCommand::ListDir { path } => {
                let ev = match self.vfs.list(&path).await {
                    Ok(entries) => AgentEvent::DirListing { path, entries },
                    Err(e) => AgentEvent::Error { message: e.to_string() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::TerminalInput { session, data } => {
                // Write keystrokes to the session's real pty; its output streams back
                // as TerminalOutput from the session's reader thread.
                if let Err(e) = self.terminals.input(&session, &data, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal input: {e}"),
                        })
                        .await;
                }
            }
            AgentCommand::TerminalResize {
                session,
                cols,
                rows,
            } => {
                if let Err(e) = self.terminals.resize(&session, cols, rows, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal resize: {e}"),
                        })
                        .await;
                }
            }
            AgentCommand::ApprovalResponse { id, approved } => {
                // Unblock the tool call waiting on this decision (no-op if it already
                // gave up). Handled inline so it lands while a Prompt handler waits.
                self.approvals.resolve(id, approved);
            }
        }
    }

    /// Decompose a prompt into role-tagged subtasks, with the shared context prefix.
    ///
    /// Phase 1: the orchestration model produces a plan (at the request's band).
    /// Phase 2: [`planner::parse_plan`] turns it into role-tagged [`PlannedSubtask`]s.
    /// If the model returns nothing parseable, degrade to a single coder task on the
    /// raw prompt. Returns the subtasks plus the prefix — the caller seeds a
    /// [`plan_graph::PlanGraph`] with the subtasks and prepends the prefix to every
    /// subagent prompt (KV reuse), and the reactive scheduler grows the graph as
    /// agents emit follow-up work.
    async fn plan(
        &self,
        text: &str,
        priority: Priority,
    ) -> anyhow::Result<(Vec<planner::PlannedSubtask>, String)> {
        // Built once and shared, byte-identical, by the planning call and every
        // subagent, so hipfire batches them prefix-shared and reuses KV.
        let prefix = self.context_prefix(text).await;

        let orch_model = self
            .roles
            .model_for(Role::Orchestration)
            .unwrap_or_default()
            .to_string();
        let plan_task = Task {
            prompt: planner::orchestration_prompt(&prefix, text),
            priority,
            model: orch_model,
        };
        let plan_text = self
            .swarm
            .run(vec![plan_task])
            .next()
            .await
            .map(|(_, r)| r)
            .transpose()?
            .unwrap_or_default();

        let mut plan = planner::parse_plan(&plan_text);
        if plan.is_empty() {
            // Degrade to one coder task on the raw prompt (still behind the shared
            // prefix) so a plan the model couldn't structure still gets attempted
            // rather than dropped.
            plan = vec![planner::PlannedSubtask {
                role: Role::Coder,
                prompt: text.to_string(),
            }];
        }
        Ok((plan, prefix))
    }

    /// Persist a plan's provenance subgraph to the graph store, if one is open.
    /// Best-effort: without `--features helix` there's no store (a no-op), and the
    /// HelixDB write path is still stubbed — so on the first write error we log once and
    /// stop rather than spamming. The in-memory provenance is already correct; this is
    /// the durability seam.
    fn persist_provenance(&self, graph: &plan_graph::PlanGraph) {
        let Some(store) = &self.graph else {
            return;
        };
        let prov = graph.provenance();
        for node in &prov.nodes {
            if let Err(e) = store.upsert_node(&node.id, node.kind.as_str(), &node.label) {
                eprintln!("provenance persistence unavailable ({e}); skipping");
                return;
            }
        }
        for edge in &prov.edges {
            if let Err(e) = store.add_edge(&edge.from, &edge.rel, &edge.to) {
                eprintln!("provenance edge persistence unavailable ({e}); skipping");
                return;
            }
        }
    }

    /// GraphRAG doc retrieval: embed the question via hipfire (query-side task
    /// prompt) and similarity-search the chunk vectors; without an embedding
    /// model, fall back to the store's BM25 text search. The answer text is the
    /// retrieved chunks verbatim.
    /// ponytail: retrieval-only — the synthesis pass (feed chunks + question to a
    /// hipfire chat model) is the next layer.
    async fn doc_query(&self, question: &str) -> AgentEvent {
        let Some(g) = &self.graph else {
            return AgentEvent::Error {
                message: "DocQuery unavailable: build with --features helix and open a graph store"
                    .into(),
            };
        };
        let query_vec = match self.skills.embed_model() {
            Some(model) => self.swarm.client().embed_query(model, question).await.ok(),
            None => None,
        };
        // LMDB read txn + HNSW search are sync — off the tokio worker.
        let store = Arc::clone(g);
        let q = question.to_string();
        let searched = tokio::task::spawn_blocking(move || {
            store.doc_search(&q, query_vec.as_deref(), 8)
        })
        .await;
        let result = match searched {
            Ok(r) => r,
            Err(e) => return AgentEvent::Error { message: format!("doc search task: {e}") },
        };
        match result {
            Ok(hits) => AgentEvent::DocAnswer {
                text: hits
                    .iter()
                    .map(|(id, text)| format!("> [{id}]\n{text}"))
                    .collect::<Vec<_>>()
                    .join("\n\n"),
                grounded_on: hits.into_iter().map(|(id, _)| id).collect(),
            },
            Err(e) => AgentEvent::Error {
                message: format!("doc search: {e}"),
            },
        }
    }

    /// Convert + chunk a reference doc via docling, then persist doc/chunk nodes
    /// (and their embeddings) into the graph store for `DocQuery`. Conversion is
    /// sync CPU work, so it runs on the blocking pool.
    ///
    /// The path is confined to the configured doc roots first: `DocIngest` reads a
    /// host file and `DocQuery` can read the content back, and the ws is reachable
    /// from the (LAN-exposed, unauthenticated) webui — so an unconfined ingest is
    /// arbitrary file read. Roots come from `CORRODE_DOC_ROOTS` (`:`-separated),
    /// defaulting to the repo root.
    #[cfg(feature = "docling")]
    async fn ingest_doc(&self, path: String) -> AgentEvent {
        let canonical = match self.confine_doc_path(&path) {
            Ok(p) => p,
            Err(e) => return AgentEvent::Error { message: format!("doc ingest: {e}") },
        };
        let converted = tokio::task::spawn_blocking(move || {
            crate::ingest::ingest(&canonical).map(|d| (canonical, d))
        })
        .await;
        let (path, doc) = match converted {
            Ok(Ok(ok)) => ok,
            Ok(Err(e)) => return AgentEvent::Error { message: format!("doc ingest: {e}") },
            Err(e) => return AgentEvent::Error { message: format!("doc ingest task: {e}") },
        };
        let persisted = self.persist_doc(&doc).await;
        AgentEvent::DocIngested {
            path,
            doc_id: doc.doc_id,
            chunks: doc.chunks.len(),
            persisted,
        }
    }

    /// Canonicalize `path` (also collapsing `..` and resolving symlinks — the file
    /// must exist) and require it under one of the doc roots. Returns the canonical
    /// path string, which also gives stable doc/chunk ids across path spellings.
    #[cfg(feature = "docling")]
    fn confine_doc_path(&self, path: &str) -> anyhow::Result<String> {
        let roots: Vec<PathBuf> = std::env::var("CORRODE_DOC_ROOTS")
            .ok()
            .map(|v| v.split(':').map(PathBuf::from).collect())
            .unwrap_or_else(|| vec![self.repo_root.clone()]);
        let canonical = std::fs::canonicalize(path)
            .map_err(|e| anyhow::anyhow!("resolve {path}: {e}"))?;
        let allowed = roots.iter().any(|r| {
            std::fs::canonicalize(r)
                .map(|cr| canonical.starts_with(cr))
                .unwrap_or(false)
        });
        if !allowed {
            anyhow::bail!(
                "{} is outside the allowed doc roots (set CORRODE_DOC_ROOTS)",
                canonical.display()
            );
        }
        Ok(canonical.to_string_lossy().into_owned())
    }

    #[cfg(not(feature = "docling"))]
    async fn ingest_doc(&self, _path: String) -> AgentEvent {
        AgentEvent::Error {
            message: "DocIngest unavailable: build with --features docling".into(),
        }
    }

    /// Write one converted doc into the graph store as one atomic `replace_doc`:
    /// the `doc` node, every chunk node (+ its hipfire embedding, batch-computed)
    /// and `has_chunk` edge, and pruning of chunks the doc no longer has. A chunk
    /// whose embedding failed is stored text-only (still BM25-searchable). The LMDB
    /// write runs on the blocking pool (sync fsync-per-commit). Returns whether it
    /// landed.
    #[cfg(feature = "docling")]
    async fn persist_doc(&self, doc: &crate::ingest::IngestedDoc) -> bool {
        let Some(store) = &self.graph else {
            return false;
        };
        let embeddings = self.embed_chunks(&doc.chunks).await;
        let chunks = doc
            .chunks
            .iter()
            .enumerate()
            .map(|(i, (id, text))| {
                let emb = embeddings.as_ref().and_then(|v| v[i].clone());
                (id.clone(), text.clone(), emb)
            })
            .collect();
        let write = crate::graph::DocWrite {
            doc_id: doc.doc_id.clone(),
            title: doc.title.clone(),
            chunks,
        };
        let store = Arc::clone(store);
        match tokio::task::spawn_blocking(move || store.replace_doc(&write)).await {
            Ok(Ok(())) => true,
            Ok(Err(e)) => {
                eprintln!("doc persistence unavailable ({e}); skipping");
                false
            }
            Err(e) => {
                eprintln!("doc persistence task failed ({e}); skipping");
                false
            }
        }
    }

    /// Batch-embed chunk texts (document side), aligned 1:1 with `chunks`. A batch
    /// that fails (e.g. one entry over hipfire's 2048-token cap 400s the whole
    /// batch) yields `None` for just that batch's chunks — they degrade to
    /// text-only rather than sinking the rest of the doc. Entries are clipped as a
    /// token-cap guard; the stored chunk text stays full-length. Returns `None`
    /// only when no embedding model is served at all.
    #[cfg(feature = "docling")]
    async fn embed_chunks(&self, chunks: &[(String, String)]) -> Option<Vec<Option<Vec<f32>>>> {
        // ponytail: chars as a token proxy — ~6000 chars sits safely under the
        // 2048-token embedding cap for prose; a real tokenizer clip can come with
        // the docling `chunking` feature.
        const EMBED_CLIP_CHARS: usize = 6000;
        const BATCH: usize = 64;
        let model = self.skills.embed_model()?;
        let client = self.swarm.client();
        let texts: Vec<String> = chunks
            .iter()
            .map(|(_, t)| {
                let mut end = t.len().min(EMBED_CLIP_CHARS);
                while !t.is_char_boundary(end) {
                    end -= 1;
                }
                t[..end].to_string()
            })
            .collect();
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            match client.embed_batch(model, batch, false).await {
                Ok(vecs) => out.extend(vecs.into_iter().map(Some)),
                Err(e) => {
                    eprintln!("chunk embedding batch failed ({e}); those chunks stored text-only");
                    out.extend(std::iter::repeat_with(|| None).take(batch.len()));
                }
            }
        }
        Some(out)
    }

    /// The shared context prefix prepended to every prompt in a Prompt turn.
    ///
    /// ponytail: a shallow repo digest (VFS root listing) plus a fixed preamble.
    /// The graph-backed VFS will supply richer, relevance-ranked context here
    /// (hipfire embeddings/rerank picking which nodes) — but the KV-sharing shape
    /// is already right: identical bytes across the whole swarm, task in the tail.
    async fn context_prefix(&self, task: &str) -> String {
        let mut s = String::from(
            "You are a subagent in the Corrode coding-agent swarm working on a shared \
repository.\n",
        );
        // Project rules (AGENTS.md) + skills relevant to this task. Byte-identical
        // across the turn's subagents (same task), so they share the KV prefill.
        let rules = self.skills.agents_rules();
        if !rules.trim().is_empty() {
            s.push_str("\nProject instructions (AGENTS.md):\n");
            s.push_str(rules.trim_end());
            s.push('\n');
        }
        let manifest = self
            .skills
            .prefix_section(task, &self.swarm.client(), TOP_K_SKILLS)
            .await;
        if !manifest.is_empty() {
            s.push('\n');
            s.push_str(&manifest);
        }
        s.push_str("\nRepository root:\n");
        match self.vfs.list("").await {
            Ok(entries) => {
                for e in entries {
                    s.push_str(&format!("  {} ({} bytes)\n", e.path, e.bytes));
                }
            }
            Err(_) => s.push_str("  (listing unavailable)\n"),
        }
        s
    }
}

/// Max tool calls a small model may make before it must answer — a bound on GPU spend
/// and runaway loops.
const MAX_TOOL_STEPS: usize = 6;

/// Byte cap per proposal fed to the fan-out judge — bounds the judge tail while the
/// shared prefix stays byte-identical (KV reuse).
const PROPOSAL_CAP: usize = 4096;

/// How long a speculative extra attempt (Opportunistic band) may keep the ensemble
/// waiting. Attempt 1 runs at the role's band and is never timed out; without this,
/// a starved straggler holds the whole Default-band task hostage (priority
/// inversion). ponytail: fixed grace — race extras against attempt 1 + margin if
/// this measurably discards useful proposals.
const FANOUT_EXTRA_GRACE: std::time::Duration = std::time::Duration::from_secs(120);

/// Byte cap per task output in the plan-review digest.
const REVIEW_OUTPUT_CAP: usize = 2048;

/// Byte cap on a `ToolResult` event's observation — bounds the event stream only;
/// the model's scratchpad still gets the full text.
const TOOL_RESULT_CAP: usize = 2048;

/// `CORRODE_FANOUT`: how many read-only proposal attempts a coder task fans out
/// before executing (1 = off, today's single-shot behavior).
/// ponytail: clamped to 8 — hipfire's admission control is the real limit; raise
/// the cap when a wider ensemble measurably helps.
fn fanout_k() -> usize {
    std::env::var("CORRODE_FANOUT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|k| k.clamp(1, 8))
        .unwrap_or(1)
}

/// `CORRODE_PLAN_REVIEW`: the plan-level review pass, on unless set to `0`/`false`.
fn plan_review_enabled() -> bool {
    !matches!(
        std::env::var("CORRODE_PLAN_REVIEW").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Turn-wide memory of tool calls already made, keyed on (tool, canonical args) —
/// TODO item 5: one map is shared by every task in a Prompt turn, so a sibling
/// re-asking what the swarm already learned gets the cached observation instead of
/// re-executing (and a mutating call is gated — or denied — once per turn, not once
/// per task). The model provably ignores fed-back errors (the same failing
/// `run_command` was observed re-issued 3×), so repeats are suppressed by the
/// harness. A successful mutating call clears everything — repo state changed, so a
/// legitimate re-read after a write is never suppressed. `log` keeps execution
/// order; its digest rides each launching task's *tail* (the shared prefix stays
/// byte-identical), so tasks start from the swarm's knowledge, not a blank slate.
/// Fan-out proposal attempts keep PRIVATE maps: their "read-only pass" notes must
/// never suppress the real, writable execution of the same call.
#[derive(Default)]
struct SeenCalls {
    seen: std::collections::HashMap<String, String>,
    log: Vec<String>,
}

impl SeenCalls {
    /// Key with object keys sorted at every level, so semantically identical calls
    /// collide regardless of the order the model emitted the arguments in.
    fn key(call: &crate::toolcall::ToolCall) -> String {
        fn canon(v: &serde_json::Value) -> serde_json::Value {
            match v {
                serde_json::Value::Object(m) => {
                    let mut entries: Vec<_> = m.iter().map(|(k, v)| (k.clone(), canon(v))).collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    serde_json::Value::Object(entries.into_iter().collect())
                }
                serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(canon).collect()),
                scalar => scalar.clone(),
            }
        }
        format!("{} {}", call.name, canon(&call.arguments))
    }

    /// The prior observation for an exact repeat, wrapped in the already-made note.
    fn repeat(&self, call: &crate::toolcall::ToolCall) -> Option<String> {
        self.seen.get(&Self::key(call)).map(|obs| {
            format!(
                "note: this exact call was already made this turn (by you or a \
                 sibling task); its result is repeated below unchanged. Do not \
                 repeat it — take a different next step.\n{obs}"
            )
        })
    }

    /// Record a call's observation. A mutating call that actually ran (`wrote …` /
    /// `exit 0:`) invalidates everything first — including the advertised log, which
    /// would otherwise sell stale knowledge; failed or denied ones stay recorded so
    /// their repeats are suppressed too.
    fn record(&mut self, call: &crate::toolcall::ToolCall, observation: &str) {
        if crate::tools::is_mutating(call)
            && (observation.starts_with("wrote") || observation.starts_with("exit 0:"))
        {
            self.seen.clear();
            self.log.clear();
        }
        let first = observation.lines().next().unwrap_or("");
        let end = crate::tools::floor_char_boundary(first, LOG_LINE_CAP);
        self.log
            .push(format!("{} -> {}", crate::tools::describe(call), &first[..end]));
        self.seen.insert(Self::key(call), observation.to_string());
    }

    /// The turn's activity so far (newest `max` lines) for a launching task's tail.
    fn digest(&self, max: usize) -> Option<String> {
        if self.log.is_empty() {
            return None;
        }
        let start = self.log.len().saturating_sub(max);
        let mut d = String::from(
            "Already done this turn by the swarm (an identical call returns its cached \
             result instantly):\n",
        );
        for line in &self.log[start..] {
            d.push_str(line);
            d.push('\n');
        }
        Some(d)
    }
}

/// Newest activity lines a launching task sees, and the byte cap per line.
const TURN_DIGEST_LINES: usize = 20;
const LOG_LINE_CAP: usize = 96;

#[allow(clippy::too_many_arguments)]
/// Run one tool call: an exact repeat short-circuits to its prior observation (see
/// [`SeenCalls`]); mutating ones (write_file / run_command / run_skill_script) pass
/// through the human approval gate first, read-only ones go straight through. A
/// successful write authors a code node, so its path is recorded for provenance.
/// `read_only` (a fan-out proposal pass) turns mutating calls into a no-op
/// observation instead — no execution and, crucially, no approval prompt, so K
/// speculative attempts never spam the human.
///
/// Shared by both loops — the gate must not depend on how the call was produced.
/// Every outcome (executed, suppressed repeat, denied, read-only note) is streamed
/// as a `ToolResult` for subagent `id` before it returns, capped for the wire.
async fn gate_and_execute(
    call: &crate::toolcall::ToolCall,
    toolbox: &ToolBox,
    approvals: &ApprovalGate,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
    seen: &std::sync::Mutex<SeenCalls>,
    read_only: bool,
) -> String {
    // The map is turn-shared across concurrent tasks: lock only around map ops,
    // never across an await (the approval gate can block for minutes). Two tasks
    // racing the same fresh call may both execute — benign, same as pre-sharing.
    let prior = seen.lock().unwrap().repeat(call);
    let observation = if let Some(prior) = prior {
        prior
    } else if read_only && crate::tools::is_mutating(call) {
        let note = format!(
            "read-only pass: {} was not executed. Describe the change in your final \
             answer instead — a reviewer picks what gets implemented.",
            crate::tools::describe(call)
        );
        seen.lock().unwrap().record(call, &note);
        note
    } else if crate::tools::is_mutating(call)
        && !approvals
            .request(events, crate::tools::describe(call))
            .await
    {
        let denied = format!("denied: {} was not approved", crate::tools::describe(call));
        seen.lock().unwrap().record(call, &denied);
        denied
    } else {
        let observation = toolbox.execute(call).await;
        if call.name == "write_file" && observation.starts_with("wrote") {
            if let Some(path) = call.arguments.get("path").and_then(|p| p.as_str()) {
                written.push(path.to_string());
            }
        }
        seen.lock().unwrap().record(call, &observation);
        observation
    };
    let mut shown = observation.clone();
    if shown.len() > TOOL_RESULT_CAP {
        shown.truncate(crate::tools::floor_char_boundary(&shown, TOOL_RESULT_CAP));
    }
    let _ = events
        .send(AgentEvent::ToolResult {
            id,
            call: crate::tools::describe(call),
            observation: shown,
        })
        .await;
    observation
}

/// The tool loop for models that emit their own calls.
///
/// The tools are declared on the request, so hipfire's chat template renders the block
/// the model was trained to read and it answers in its own syntax — which the dialect
/// parses directly. No Needle: nothing has to reconstruct the call from prose, so the
/// multi-param and truncation failures that motivated the Needle finetune cannot occur.
///
/// Thinking defaults to off (`CORRODE_REASONING_EFFORT` overrides): with reasoning on,
/// these models talk themselves out of calling — measured on MiniCPM5-1B, which
/// deliberated past its budget instead of emitting a call it had already chosen.
#[allow(clippy::too_many_arguments)]
async fn run_native_tool_loop(
    client: &Client,
    model: &str,
    band: Priority,
    dialect: &crate::dialect::ToolDialect,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
    read_only: bool,
    seen: &std::sync::Mutex<SeenCalls>,
) -> anyhow::Result<String> {
    // Per-task values overlay: params with a closed, known set (read/list paths,
    // skill targets) carry a JSON-Schema `enum`, which hipfire's grammar turns into
    // a hard constraint — an invented path becomes unreachable, not merely corrected
    // after the fact. ponytail: computed once per task — a within-task write can add
    // a path the enum lacks (the model then can't read it back this task); recompute
    // per step after a successful mutating call if that measurably bites. Also: the
    // tools JSON renders ahead of the shared prefix, so once a mid-turn write lands,
    // later tasks' tools bytes diverge and KV prefix-sharing splits for the rest of
    // the turn (CLAUDE.md constraint 2) — accepted; a per-turn overlay would make
    // fanout attempts blind to each other's era instead.
    let values = toolbox.param_values().await;
    let tools = dialect.request_tools(crate::tools::role_tools(role), Some(&values));
    let effort = std::env::var("CORRODE_REASONING_EFFORT").unwrap_or_else(|_| "none".to_string());
    let mut scratchpad = String::new();
    let mut last = String::new();
    for _ in 0..MAX_TOOL_STEPS {
        let prompt = planner::native_tool_prompt(prefix, role, task, &scratchpad);
        let (text, _reasoning) = client
            .respond_full(model, &prompt, band, Some(&tools), Some(&effort))
            .await?;
        let _ = events
            .send(AgentEvent::SubagentOutput {
                id,
                text: text.clone(),
            })
            .await;
        last = text.clone();

        let calls = dialect.parse(&text).unwrap_or_default();
        let Some(call) = calls.first() else {
            return Ok(text); // no call -> this turn is the final answer
        };
        let observation =
            gate_and_execute(call, &toolbox, approvals, events, id, written, seen, read_only)
                .await;
        scratchpad.push_str(&format!(
            "\nCALLED: {}\nRESULT: {observation}\n",
            crate::tools::describe(call)
        ));
    }
    Ok(last)
}

/// The Needle-mediated tool-execution loop for a small model.
///
/// Each turn the model responds (streamed as `SubagentOutput`). If it wrote a `TOOL:`
/// line, Needle structures that plain-English intent into a call — the small model
/// never writes JSON — `toolbox` executes it against the repo, and the observation is
/// appended to the scratchpad for the next turn. The loop ends when a turn has no
/// `TOOL:` line (that text is the final answer) or the step budget is spent. Tool and
/// Needle errors come back as observations (the model can recover), not hard failures;
/// only a model-generation error aborts the loop.
async fn run_tool_loop(
    client: &Client,
    model: &str,
    band: Priority,
    caller: Arc<dyn ToolCaller>,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    dialects: &Dialects,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
    read_only: bool,
    seen: &std::sync::Mutex<SeenCalls>,
) -> anyhow::Result<String> {
    // Render the exec toolset in the tool-call model's dialect once; parse each reply
    // with the same dialect (which maps its tool names back to canonical).
    let dialect = dialects.resolve(caller.model_id());
    let schema = dialect.render(crate::tools::role_tools(role), None);
    let mut scratchpad = String::new();
    let mut last = String::new();
    for _ in 0..MAX_TOOL_STEPS {
        let prompt = planner::tool_loop_prompt(prefix, role, task, &scratchpad);
        let text = client.respond(model, &prompt, band).await?;
        let _ = events
            .send(AgentEvent::SubagentOutput {
                id,
                text: text.clone(),
            })
            .await;
        last = text.clone();

        let Some(intent) = crate::tools::parse_tool_intent(&text) else {
            return Ok(text); // no TOOL: line -> this turn is the final answer
        };

        // The tool-caller turns the plain-English intent into a structured call
        // (spawn_blocking: Needle inference is synchronous and CPU-bound); the dialect
        // parses the raw reply.
        let query = intent.clone();
        let toolcaller = caller.clone();
        let schema = schema.clone();
        let raw = tokio::task::spawn_blocking(move || toolcaller.generate(&query, &schema)).await;
        let observation = match raw.map(|r| r.and_then(|raw| dialect.parse(&raw))) {
            Ok(Ok(calls)) => match calls.first() {
                Some(c) => {
                    gate_and_execute(
                        c, &toolbox, approvals, events, id, written, seen, read_only,
                    )
                    .await
                }
                None => "error: no tool call produced".to_string(),
            },
            Ok(Err(e)) => format!("error: tool-call construction failed: {e}"),
            Err(e) => format!("error: tool-call thread panicked: {e}"),
        };
        scratchpad.push_str(&format!("\nTOOL: {intent}\nRESULT: {observation}\n"));
    }
    // Step budget spent: hand back the last turn as the answer.
    Ok(last)
}

/// One full execution of a task: pick the capability path and run it.
///  1. the model emits its own tool calls (its dialect says so) — declare tools on
///     the request and parse the reply directly.
///  2. otherwise a small model runs the Needle-mediated loop, which reconstructs a
///     call from a plain-English line.
///  3. anything else answers single-shot.
/// `read_only` marks a fan-out proposal pass: mutating tool calls become no-op
/// observations (see [`gate_and_execute`]) without touching the paths themselves.
#[allow(clippy::too_many_arguments)]
async fn run_task(
    client: &Client,
    model: &str,
    band: Priority,
    dialects: &Dialects,
    tool_caller: Option<Arc<dyn ToolCaller>>,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
    read_only: bool,
    seen: &std::sync::Mutex<SeenCalls>,
) -> anyhow::Result<String> {
    let role_dialect = dialects.resolve(model);
    if role_dialect.emits_own_calls() {
        run_native_tool_loop(
            client, model, band, role_dialect, toolbox, approvals, prefix, role, task, events,
            id, written, read_only, seen,
        )
        .await
    } else if let (true, Some(caller)) = (roles::is_small_model(model), tool_caller) {
        run_tool_loop(
            client, model, band, caller, toolbox, approvals, dialects, prefix, role, task,
            events, id, written, read_only, seen,
        )
        .await
    } else {
        let full = planner::subagent_prompt(prefix, role, task);
        let out = client.respond(model, &full, band).await;
        if let Ok(text) = &out {
            let _ = events
                .send(AgentEvent::SubagentOutput {
                    id,
                    text: text.clone(),
                })
                .await;
        }
        out
    }
}

/// Fan a coder task out as `k` read-only proposal attempts, judge them, execute once.
///
/// The attempts run concurrently on the same shared prefix (hipfire batches them
/// prefix-shared); the first keeps the role's band, the rest are speculative and go
/// Opportunistic — idle GPU only, per the scheduler contract. The review model then
/// judges the surviving proposals into one directive, and the task executes once,
/// writable, steered by it. Scaffolding failures degrade to plain execution: the
/// ensemble may improve the task, never fail it.
#[allow(clippy::too_many_arguments)]
async fn run_fanout(
    k: usize,
    client: &Client,
    model: &str,
    review_model: &str,
    band: Priority,
    dialects: &Dialects,
    tool_caller: Option<Arc<dyn ToolCaller>>,
    toolbox: ToolBox,
    approvals: &ApprovalGate,
    prefix: &str,
    role: Role,
    task: &str,
    events: &mpsc::Sender<AgentEvent>,
    id: u64,
    written: &mut Vec<String>,
    seen: &std::sync::Mutex<SeenCalls>,
) -> anyhow::Result<String> {
    let attempts = (0..k).map(|i| {
        let attempt_task = planner::fanout_attempt_task(task, i + 1, k);
        let attempt_band = if i == 0 { band } else { Priority::Opportunistic };
        let toolbox = toolbox.clone();
        let tool_caller = tool_caller.clone();
        async move {
            let mut sink = Vec::new(); // read-only: no artifacts can land
            // Attempts get a PRIVATE map: their "read-only pass" notes must never
            // suppress the turn map's real, writable execution of the same call.
            let attempt_seen = std::sync::Mutex::new(SeenCalls::default());
            let fut = run_task(
                client, model, attempt_band, dialects, tool_caller, toolbox, approvals,
                prefix, role, &attempt_task, events, id, &mut sink, true, &attempt_seen,
            );
            if i == 0 {
                fut.await
            } else {
                match tokio::time::timeout(FANOUT_EXTRA_GRACE, fut).await {
                    Ok(res) => res,
                    Err(_) => Err(anyhow::anyhow!("fanout attempt {} timed out", i + 1)),
                }
            }
        }
    });
    let results = futures_util::future::join_all(attempts).await;
    let failed = results.iter().filter(|r| r.is_err()).count();
    if failed > 0 {
        // The ensemble degrades silently by design; the operator still gets a trace.
        eprintln!("fanout: {failed}/{k} attempts failed or timed out");
    }
    let mut proposals: Vec<String> = results.into_iter().filter_map(Result::ok).collect();
    for p in &mut proposals {
        if p.len() > PROPOSAL_CAP {
            p.truncate(crate::tools::floor_char_boundary(p, PROPOSAL_CAP));
            p.push_str("\n… (truncated)");
        }
    }

    let mut steered = task.to_string();
    if proposals.len() >= 2 {
        let judge_prompt = planner::fanout_judge_prompt(prefix, task, &proposals);
        match client
            .respond(review_model, &judge_prompt, planner::band_for(Role::Review))
            .await
        {
            Ok(directive) => {
                let _ = events
                    .send(AgentEvent::SubagentOutput {
                        id,
                        text: format!("[fanout judge] {directive}"),
                    })
                    .await;
                steered = format!(
                    "{task}\n\nA reviewer judged {} independent proposals and synthesized \
                     this directive — follow it:\n{directive}",
                    proposals.len()
                );
            }
            Err(e) => eprintln!("fanout: judge failed, executing unsteered: {e}"),
        }
    } else if let [only] = proposals.as_slice() {
        // A lone survivor skips the judge but still informs the implementer — the
        // exploration is already paid for.
        steered = format!(
            "{task}\n\nOne read-only exploration proposed this — weigh it before \
             implementing:\n{only}"
        );
    }
    run_task(
        client, model, band, dialects, tool_caller, toolbox, approvals, prefix, role,
        &steered, events, id, written, false, seen,
    )
    .await
}

/// Extract the follow-up task a subagent proposed in its reply.
///
/// The agent writes a plain-English `NEXT:` line (see [`planner::subagent_prompt`]) —
/// easy even for a small model, which can't hand-write a tool call. We then need the
/// task's *role* (which band/model runs it). With a Needle tool-caller present, Needle
/// classifies the instruction into one of the role tools ([`plan_graph::ROLE_TASK_TOOLS`]);
/// tool selection is what it's trained for, so this is reliable — unlike trusting a
/// small model to emit structured JSON. The task text stays verbatim from the `NEXT:`
/// line (Needle's own `task` argument truncates). One instruction per reply → one task;
/// the reactive graph chains the rest.
///
/// Without a caller (base build, or assets absent) or on a Needle error, the task still
/// queues, defaulted to the Coder role. No `NEXT:` line → no emission. Needle inference
/// is synchronous and CPU-bound, so it runs on a blocking thread.
async fn emit_followups(
    tool_caller: Option<Arc<dyn ToolCaller>>,
    dialects: &Dialects,
    output: &str,
) -> Vec<plan_graph::Emit> {
    let Some(instruction) = plan_graph::parse_next_instruction(output) else {
        return Vec::new(); // no follow-up proposed
    };

    let role = match tool_caller {
        Some(caller) => {
            // Render the role tools in the caller's dialect, classify, parse back.
            let dialect = dialects.resolve(caller.model_id());
            let schema = dialect.render(plan_graph::ROLE_TOOLS, None);
            let query = instruction.clone();
            let raw =
                tokio::task::spawn_blocking(move || caller.generate(&query, &schema)).await;
            match raw.map(|r| r.and_then(|raw| dialect.parse(&raw))) {
                Ok(Ok(calls)) => plan_graph::role_from_tool_calls(&calls).unwrap_or(Role::Coder),
                Ok(Err(e)) => {
                    eprintln!("role classification failed ({e}); defaulting to coder");
                    Role::Coder
                }
                Err(e) => {
                    eprintln!("role classification thread panicked ({e}); defaulting to coder");
                    Role::Coder
                }
            }
        }
        None => Role::Coder, // no classifier: queue it as a coder task
    };

    vec![plan_graph::Emit {
        role,
        prompt: instruction,
        after_emitter: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::Client;
    use crate::vfs::PassthroughVfs;

    fn test_daemon() -> Daemon {
        Daemon::new(
            Swarm::new(Client::new("http://127.0.0.1:1", None), 1),
            RoleModels::uniform("test-model"),
            None,
            Arc::new(PassthroughVfs::new(std::env::temp_dir())),
            SkillContext::default(),
            None,
            std::env::temp_dir(),
            Arc::new(Dialects::default()),
        )
    }

    /// The `fixtures/demo-repo` submodule (`xynexus/corrode-demo`) — the deterministic
    /// repo the e2e tests run against: real files (`src/lib.rs`), real rules
    /// (`AGENTS.md`), a real skill (`.agents/skills/run-tests`). `None` when it isn't
    /// checked out (`git submodule update --init fixtures/demo-repo`), so the tests skip
    /// rather than fail on a missing fixture.
    fn demo_repo() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/demo-repo");
        dir.join("src/lib.rs").exists().then_some(dir)
    }

    /// The fixture is wired up, through the same `SkillContext` the daemon builds:
    /// discovery finds the bundled `run-tests` skill, it reaches the shared prefix, its
    /// scripts are resolvable, and `AGENTS.md` is read. Runs on the base build (no
    /// embedding model -> no hipfire call), so a moved path or an uninitialized submodule
    /// fails here rather than inside an ignored test.
    #[tokio::test]
    async fn demo_repo_fixture_is_discoverable() {
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let client = Client::new("http://127.0.0.1:1", None);
        let skills = SkillContext::build(&repo, &client, None).await;
        let section = skills
            .prefix_section("run the tests", &client, TOP_K_SKILLS)
            .await;
        assert!(section.contains("run-tests"), "got: {section}");
        assert!(skills.script_dirs().contains_key("run-tests"));
        let rules = skills.agents_rules();
        assert!(rules.contains("cargo test"), "got: {rules}");
    }

    // Real Needle emission, end-to-end through `emit_followups` (query wrapper +
    // grammar-guided decode + mapping). Ignored by default (needs the weights):
    //   cargo test -p corrode-daemon --features needle -- --ignored --nocapture
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_classifies_the_next_line_role_and_keeps_the_text() {
        use crate::toolcall::needle::NeedleToolCaller;
        let caller: Arc<dyn ToolCaller> = Arc::new(
            NeedleToolCaller::load_from_env()
                .expect("load Needle")
                .expect("Needle assets present"),
        );
        // A subagent's reply against the fixture repo, ending with a plain-English NEXT:
        // line describing a review follow-up (mathkit's standing is_prime task).
        let reply = "Rewrote is_prime in src/lib.rs to trial-divide up to sqrt(n).\n\
            NEXT: review the new is_prime implementation in src/lib.rs for correctness";
        let emits = emit_followups(Some(caller), &Dialects::default(), reply).await;
        let summary: Vec<_> = emits
            .iter()
            .map(|e| (e.role, e.prompt.as_str()))
            .collect();
        eprintln!("emitted: {summary:?}");
        assert_eq!(emits.len(), 1, "one NEXT: line -> one task");
        // Task text is verbatim from the NEXT: line (not Needle's truncated arg).
        assert_eq!(
            emits[0].prompt,
            "review the new is_prime implementation in src/lib.rs for correctness"
        );
        // Needle classifies the role from the instruction (tool selection).
        assert_eq!(emits[0].role, Role::Review);
    }

    // The harness-enforced repeat suppression: an exact repeat comes back as the prior
    // observation plus the already-tried note, args collide regardless of key order, a
    // successful mutating call clears the memory (a re-read after a write must run for
    // real), and a *failing* mutating call does not (its 3×-observed retry stays dead).
    #[test]
    fn repeated_calls_are_suppressed_until_a_mutating_call_lands() {
        let call = |name: &str, args: serde_json::Value| crate::toolcall::ToolCall {
            name: name.to_string(),
            arguments: args,
        };
        let mut seen = SeenCalls::default();

        let read = call("read_file", serde_json::json!({"path": "src/lib.rs"}));
        assert!(seen.repeat(&read).is_none(), "first call is not a repeat");
        seen.record(&read, "contents of src/lib.rs:\nfn f() {}");
        let suppressed = seen.repeat(&read).expect("exact repeat is suppressed");
        assert!(suppressed.starts_with("note: this exact call was already made this turn"));
        assert!(suppressed.ends_with("contents of src/lib.rs:\nfn f() {}"));

        // Key order is not identity: semantically identical args collide.
        let a = call("write_file", serde_json::json!({"path": "a.rs", "contents": "x"}));
        let b = call("write_file", serde_json::json!({"contents": "x", "path": "a.rs"}));
        seen.record(&a, "denied: write_file a.rs was not approved");
        assert!(seen.repeat(&b).is_some(), "reordered args must collide");

        // A failing mutating call does NOT invalidate — its repeat stays suppressed.
        let bad = call("run_command", serde_json::json!({"command": "carg test"}));
        seen.record(&bad, "exit 127:\ncarg: command not found");
        assert!(seen.repeat(&bad).is_some());
        assert!(seen.repeat(&read).is_some(), "reads survive a failed command");

        // A successful mutating call clears everything: the re-read runs for real.
        seen.record(&a, "wrote 1 bytes to a.rs");
        assert!(seen.repeat(&read).is_none(), "read after write must not be suppressed");
        assert!(seen.repeat(&bad).is_none());
        assert!(seen.repeat(&a).is_some(), "the write itself stays recorded");
    }

    // A read-only pass (a fan-out proposal attempt) neither executes a mutating call
    // nor prompts for approval — K speculative attempts must not spam the human —
    // while plain reads still run for real.
    #[tokio::test]
    async fn read_only_pass_blocks_mutations_without_asking() {
        let dir = std::env::temp_dir().join(format!("corrode-fanout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.rs"), b"fn f() {}").unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(std::collections::HashMap::new()),
        );
        let approvals = ApprovalGate::default();
        let (tx, mut rx) = mpsc::channel(4);
        let mut written = Vec::new();
        let seen = std::sync::Mutex::new(SeenCalls::default());

        let write = crate::toolcall::ToolCall {
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "lib.rs", "contents": "boom"}),
        };
        // timeout, not await: if the read-only check regressed to after the approval
        // request, this would block on a oneshot nobody resolves — fail, don't hang.
        let obs = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            gate_and_execute(&write, &toolbox, &approvals, &tx, 0, &mut written, &seen, true),
        )
        .await
        .expect("read-only gate must not block on approval");
        assert!(obs.starts_with("read-only pass:"), "got: {obs}");
        assert!(written.is_empty(), "no artifact from a blocked write");
        // The only event is the ToolResult trace — never an ApprovalRequest.
        match rx.try_recv().expect("a ToolResult event") {
            AgentEvent::ToolResult { call, observation, .. } => {
                assert!(call.starts_with("write_file"), "got: {call}");
                assert!(observation.starts_with("read-only pass:"), "got: {observation}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "no approval request was emitted");
        assert_eq!(std::fs::read(dir.join("lib.rs")).unwrap(), b"fn f() {}");

        let read = crate::toolcall::ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "lib.rs"}),
        };
        let obs =
            gate_and_execute(&read, &toolbox, &approvals, &tx, 0, &mut written, &seen, true)
                .await;
        assert!(obs.contains("fn f() {}"), "got: {obs}");
        assert!(
            matches!(rx.try_recv(), Ok(AgentEvent::ToolResult { .. })),
            "the executed read is traced too"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // The env knobs parse defensively: junk/0/unset mean off, the width is clamped.
    // (No other test reads these vars, so set_var is race-free here.)
    #[test]
    fn fanout_and_review_knobs_parse_defensively() {
        std::env::remove_var("CORRODE_FANOUT");
        assert_eq!(fanout_k(), 1);
        std::env::set_var("CORRODE_FANOUT", "0");
        assert_eq!(fanout_k(), 1);
        std::env::set_var("CORRODE_FANOUT", "99");
        assert_eq!(fanout_k(), 8);
        std::env::set_var("CORRODE_FANOUT", "abc");
        assert_eq!(fanout_k(), 1);
        std::env::remove_var("CORRODE_FANOUT");

        std::env::remove_var("CORRODE_PLAN_REVIEW");
        assert!(plan_review_enabled());
        std::env::set_var("CORRODE_PLAN_REVIEW", "0");
        assert!(!plan_review_enabled());
        std::env::set_var("CORRODE_PLAN_REVIEW", "false");
        assert!(!plan_review_enabled());
        std::env::remove_var("CORRODE_PLAN_REVIEW");
    }

    // Item 5: the turn-shared memory serves a sibling's identical call from cache
    // (with the already-made note), the activity digest advertises what the swarm
    // did for a launching task's tail, and a successful mutating call drops both —
    // stale knowledge must be neither served nor advertised.
    #[tokio::test]
    async fn turn_memory_serves_siblings_and_digests_activity() {
        let dir = std::env::temp_dir().join(format!("corrode-turnmem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.rs"), b"fn f() {}").unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(std::collections::HashMap::new()),
        );
        let approvals = ApprovalGate::default();
        let (tx, _rx) = mpsc::channel(16);
        let mut written = Vec::new();
        let seen = std::sync::Mutex::new(SeenCalls::default());
        let read = crate::toolcall::ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "lib.rs"}),
        };

        // "Task A" reads; "task B" (same shared map) asks again and gets the cache.
        let a = gate_and_execute(&read, &toolbox, &approvals, &tx, 0, &mut written, &seen, false)
            .await;
        assert!(a.contains("fn f() {}"));
        let b = gate_and_execute(&read, &toolbox, &approvals, &tx, 1, &mut written, &seen, false)
            .await;
        assert!(b.starts_with("note: this exact call was already made this turn"), "got: {b}");
        assert!(b.contains("fn f() {}"), "the sibling still gets the knowledge");

        // The digest advertises the activity, once, for a launching task's tail.
        let digest = seen.lock().unwrap().digest(TURN_DIGEST_LINES).expect("activity");
        assert!(digest.contains("read_file lib.rs ->"), "got: {digest}");
        assert_eq!(digest.matches("read_file lib.rs").count(), 1, "cache hits are not re-logged");

        // A successful mutating call invalidates cache AND digest.
        let write = crate::toolcall::ToolCall {
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "lib.rs", "contents": "fn g() {}"}),
        };
        seen.lock().unwrap().record(&write, "wrote 9 bytes to lib.rs");
        assert!(seen.lock().unwrap().repeat(&read).is_none(), "read re-executes after a write");
        let digest = seen.lock().unwrap().digest(TURN_DIGEST_LINES).unwrap();
        assert!(!digest.contains("read_file"), "stale reads are not advertised: {digest}");
        assert!(digest.contains("write_file lib.rs ->"), "the write itself is: {digest}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // No NEXT: line -> no emission (and Needle isn't even called).
    #[tokio::test]
    async fn no_next_line_emits_nothing() {
        let emits = emit_followups(
            None,
            &Dialects::default(),
            "All done. The tests pass and nothing remains.",
        )
        .await;
        assert!(emits.is_empty());
    }

    // The tool-execution path a small model takes: a plain-English intent -> Needle
    // builds the call -> the ToolBox runs it against the repo. (The full loop also
    // needs hipfire for the model turns; this covers the Needle+tool half.)
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_builds_a_tool_call_that_the_toolbox_executes() {
        use crate::dialect::ToolDialect;
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(std::collections::HashMap::new()),
        );

        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let dialect = ToolDialect::default();
        let raw = caller
            .generate("read the file src/lib.rs", &dialect.render(EXEC_TOOLS, None))
            .expect("generation");
        let calls = dialect.parse(&raw).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("a tool call");
        assert_eq!(call.name, "read_file");

        // mathkit's source, straight out of the fixture repo.
        let observation = toolbox.execute(call).await;
        assert!(observation.contains("pub fn is_prime"), "got: {observation}");
    }

    // Stage 3 end-to-end against the fixture repo: discovery finds its bundled
    // `run-tests` skill -> a plain-English ask -> Needle builds a run_skill_script call
    // -> the ToolBox resolves the skill and runs `scripts/test.sh` (a real `cargo test`
    // over mathkit, so the first run compiles the fixture).
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires Needle assets (CORRODE_NEEDLE_ASSETS or the vendored default)"]
    async fn needle_runs_a_skill_script_end_to_end() {
        use crate::dialect::ToolDialect;
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let client = Client::new("http://127.0.0.1:1", None);
        let skills = SkillContext::build(&repo, &client, None).await.script_dirs();
        assert!(skills.contains_key("run-tests"), "got: {skills:?}");
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(skills),
        );

        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let dialect = ToolDialect::default();
        let raw = caller
            .generate(
                "run the test.sh script from the run-tests skill",
                &dialect.render(EXEC_TOOLS, None),
            )
            .expect("generation");
        let calls = dialect.parse(&raw).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("a tool call");
        assert_eq!(call.name, "run_skill_script");

        // The call is mutating (executes code) -> would gate on approval in the loop.
        assert!(crate::tools::is_mutating(call));
        let observation = toolbox.execute(call).await;
        assert!(observation.starts_with("exit 0:"), "got: {observation}");
        assert!(observation.contains("test result: ok"), "got: {observation}");
    }

    // The whole loop, for real: a Prompt goes into the daemon, hipfire plans it, the
    // swarm runs the subtasks against the demo repo, and events stream back — no mocks
    // on any leg. Needs a live hipfire (HIPFIRE_BASE_URL, CORRODE_MODEL) plus the Needle
    // assets and the submodule:
    //   HIPFIRE_BASE_URL=http://host:11435 CORRODE_MODEL=<id> \
    //     cargo test -p corrode-daemon --features needle -- --ignored --nocapture
    //
    // The turn's end is explicit (AgentEvent::TurnComplete): the drain exits on it,
    // and the per-event timeout is only a guard against a wedged run.
    #[cfg(feature = "needle")]
    #[tokio::test]
    #[ignore = "requires a live hipfire + Needle assets + the demo-repo submodule"]
    async fn a_prompt_runs_the_swarm_against_the_demo_repo() {
        use std::time::Duration;
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let base_url = std::env::var("HIPFIRE_BASE_URL")
            .unwrap_or_else(|_| crate::hipfire::DEFAULT_BASE_URL.to_string());
        let client = Client::new(&base_url, std::env::var("HIPFIRE_API_KEY").ok());
        let Ok(models) = client.list_models().await else {
            eprintln!("skipped: no hipfire at {base_url}");
            return;
        };
        // The model under test: CORRODE_MODEL, else whatever hipfire serves first.
        let model = std::env::var("CORRODE_MODEL")
            .ok()
            .or_else(|| models.first().cloned())
            .expect("hipfire serves at least one model");
        eprintln!("model: {model} (small: {})", roles::is_small_model(&model));

        let embed = roles::default_embedding_model(&models).map(str::to_string);
        let skills = SkillContext::build(&repo, &client, embed).await;
        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let daemon = Arc::new(Daemon::new(
            Swarm::new(client, 4),
            RoleModels::uniform(&model),
            None,
            Arc::new(PassthroughVfs::new(&repo)),
            skills,
            Some(Arc::new(caller)),
            repo.clone(),
            // load() so CORRODE_TOOL_DIALECTS can put a model on its native dialect
            Arc::new(Dialects::load()),
        ));

        let (ctx, crx) = mpsc::channel(16);
        let (etx, mut erx) = mpsc::channel(64);
        tokio::spawn(Arc::clone(&daemon).run(crx, etx));
        ctx.send(AgentCommand::Prompt {
            text: "Report which functions src/lib.rs defines.".into(),
            priority: Priority::Default,
        })
        .await
        .unwrap();

        // Drain until TurnComplete, approving any mutating call the swarm proposes
        // (unattended: nothing here would answer the gate otherwise, and it fails closed).
        let (mut outputs, mut approvals, mut errors) = (Vec::new(), Vec::new(), Vec::new());
        let mut tool_results: Vec<(String, String)> = Vec::new();
        while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_secs(120), erx.recv()).await {
            match ev {
                AgentEvent::SubagentOutput { id, text } => {
                    eprintln!("--- subagent {id} ---\n{text}\n");
                    outputs.push(text);
                }
                AgentEvent::ApprovalRequest { id, action } => {
                    eprintln!("--- approval {id}: {action} -> approving");
                    approvals.push(action);
                    ctx.send(AgentCommand::ApprovalResponse { id, approved: true })
                        .await
                        .unwrap();
                }
                AgentEvent::ToolResult { id, call, observation } => {
                    eprintln!("--- tool [{id}] {call} -> {observation}");
                    tool_results.push((call, observation));
                }
                AgentEvent::TurnComplete { plan_id } => {
                    eprintln!("--- turn {plan_id} complete");
                    break;
                }
                AgentEvent::Error { message } => {
                    eprintln!("--- error: {message}");
                    errors.push(message);
                }
                other => eprintln!("--- {other:?}"),
            }
        }
        drop(ctx);

        eprintln!(
            "settled: {} outputs, {} tool results, {} approvals, {} errors",
            outputs.len(),
            tool_results.len(),
            approvals.len(),
            errors.len()
        );
        assert!(errors.is_empty(), "daemon reported errors: {errors:?}");
        assert!(!outputs.is_empty(), "the swarm produced no subagent output");
        // Whether the small model actually drove a tool is a property of the model +
        // the tool-loop prompt, not of this wiring — reported above, asserted in the
        // Needle tests, deliberately not asserted here.

        // Structural guarantees the event stream carries regardless of what the model
        // chose to do:
        // (1) A hallucinated read/list path comes back as the corrective observation
        //     ("no such path …"), never the raw errno. Scoped to read_file/list_dir —
        //     a shell command's own output may legitimately contain the errno text.
        for (call, obs) in &tool_results {
            if call.starts_with("read_file") || call.starts_with("list_dir") {
                assert!(
                    !obs.contains("No such file or directory"),
                    "raw errno leaked past the corrective path: {call} -> {obs}"
                );
            }
        }
        // (2) Repeat suppression: an exact repeat within a task is answered from
        //     memory (the "already tried" note) without re-executing — so every
        //     *executed* mutating result burned exactly one approval. Events carry no
        //     task id, so "at most one executed observation per duplicate call" can't
        //     be asserted soundly across tasks (two tasks may legitimately run the
        //     same call, each with its own approval); the sound global form is that
        //     executed mutating results never exceed the approvals granted.
        let executed_mutating = tool_results
            .iter()
            .filter(|(call, obs)| {
                ["write_file", "run_command", "run_skill_script"]
                    .iter()
                    .any(|m| call.starts_with(m))
                    && !obs.starts_with("note: this exact call was already tried")
                    && !obs.starts_with("read-only pass:")
                    && !obs.starts_with("denied:")
            })
            .count();
        assert!(
            executed_mutating <= approvals.len(),
            "{executed_mutating} executed mutating tool results but only {} approvals — \
             a suppressed repeat must not have executed",
            approvals.len()
        );
    }

    // The native path, end to end and Needle-free: Corrode declares its own tools on the
    // request, hipfire's template renders the model's `<tools>` block, the model emits
    // its own XML call, and the dialect parses it straight into a ToolCall the ToolBox
    // runs. This is the loop the Needle shim exists to substitute for — here it isn't
    // in it at all.
    //   HIPFIRE_BASE_URL=http://127.0.0.1:11435 CORRODE_MODEL=<id> \
    //     cargo test -p corrode-daemon -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "requires a live hipfire + the demo-repo submodule"]
    async fn a_model_drives_its_own_tool_call_without_needle() {
        use crate::dialect::{ParseFormat, SchemaFormat, ToolDialect};
        use crate::tools::{ToolBox, EXEC_TOOLS};
        let Some(repo) = demo_repo() else {
            eprintln!("skipped: fixtures/demo-repo not checked out");
            return;
        };
        let base_url = std::env::var("HIPFIRE_BASE_URL")
            .unwrap_or_else(|_| crate::hipfire::DEFAULT_BASE_URL.to_string());
        let client = Client::new(&base_url, std::env::var("HIPFIRE_API_KEY").ok());
        let Ok(models) = client.list_models().await else {
            eprintln!("skipped: no hipfire at {base_url}");
            return;
        };
        let model = std::env::var("CORRODE_MODEL")
            .ok()
            .or_else(|| models.first().cloned())
            .expect("hipfire serves a model");

        // Corrode's canonical tools, rendered for a chat model. The `type`/`function`
        // envelope is what the chat templates serialize with `tool | tojson`.
        let rendered: serde_json::Value =
            serde_json::from_str(&ToolDialect::new(
                SchemaFormat::OpenAiNested,
                ParseFormat::MiniCpmXml,
                std::collections::HashMap::new(),
            )
            .render(EXEC_TOOLS, None))
            .expect("rendered tools are JSON");
        let tools = serde_json::Value::Array(
            rendered
                .as_array()
                .unwrap()
                .iter()
                .map(|t| serde_json::json!({"type": "function", "function": t}))
                .collect(),
        );

        // Thinking off: the model deliberates itself out of calling when it is on, and
        // a direct imperative is what the tool loop should be issuing anyway.
        let (answer, reasoning) = client
            .respond_full(
                &model,
                "Read the file src/lib.rs.",
                Priority::Default,
                Some(&tools),
                Some("none"),
            )
            .await
            .expect("generation");
        eprintln!("answer: {answer}\nreasoning: {} chars", reasoning.len());

        let dialect = ToolDialect::new(
            SchemaFormat::OpenAiNested,
            ParseFormat::MiniCpmXml,
            std::collections::HashMap::new(),
        );
        let calls = dialect.parse(&answer).expect("parse");
        eprintln!("calls: {calls:?}");
        let call = calls.first().expect("the model emitted a tool call");
        assert_eq!(call.name, "read_file");

        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&repo)),
            repo.clone(),
            Arc::new(std::collections::HashMap::new()),
        );
        let observation = toolbox.execute(call).await;
        assert!(observation.contains("pub fn is_prime"), "got: {observation}");
    }

    // The hipfire-free dispatch path: DocQuery without a graph store reports itself
    // unavailable (Error) rather than hanging or panicking. Guards the match, not the
    // network. (The real pty terminal path is covered in `terminal.rs`.)
    #[tokio::test]
    async fn loop_reports_docquery_unavailable_without_graph() {
        let daemon = test_daemon();
        let (ctx, crx) = mpsc::channel(8);
        let (etx, mut erx) = mpsc::channel(8);

        ctx.send(AgentCommand::DocQuery {
            question: "q".into(),
        })
        .await
        .unwrap();
        drop(ctx);

        Arc::new(daemon).run(crx, etx).await;

        assert!(matches!(
            erx.recv().await.unwrap(),
            AgentEvent::Error { .. }
        ));
    }
}
