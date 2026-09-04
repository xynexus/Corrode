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
use crate::hipfire::Client;
use crate::plan_graph;
use crate::project::Project;
use crate::planner;
use crate::roles::{Role, RoleModels};
use crate::session::{RepoResources, Session, SessionKey};
use crate::skills::SkillContext;
use crate::swarm::{Swarm, Task};
use crate::telemetry::Telemetry;
use crate::toolcall::ToolCaller;
use crate::tools::ToolBox;
use crate::vfs::{PassthroughVfs, Vfs};
use corrode_core::{AgentCommand, AgentEvent, Priority};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// How many relevance-ranked skills to surface in the shared prefix per turn.
const TOP_K_SKILLS: usize = 8;

/// Canonicalize a path, falling back to the raw path if it doesn't resolve — so a
/// repo dir that doesn't exist yet still yields a stable session key.
fn canonical(path: &str) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path))
}

/// One entry in the `CORRODE_USERS` table: the token that authenticates to the
/// daemon, plus (optionally) a per-user hipfire bearer for fairness attribution.
/// JSON: `{"alice": {"token": "…", "hipfire_token": "…"}}` (hipfire_token optional).
#[derive(serde::Deserialize, Clone)]
struct UserEntry {
    token: String,
    #[serde(default)]
    hipfire_token: Option<String>,
}

/// Load the auth table from `CORRODE_USERS` (a JSON file path). Absent or
/// unreadable => `None` (auth off, connections anonymous).
fn load_users() -> Option<HashMap<String, UserEntry>> {
    let path = std::env::var("CORRODE_USERS").ok()?;
    match std::fs::read_to_string(&path) {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(map) => Some(map),
            Err(e) => {
                eprintln!("CORRODE_USERS parse failed ({e}); auth disabled");
                None
            }
        },
        Err(e) => {
            eprintln!("CORRODE_USERS read failed at {path} ({e}); auth disabled");
            None
        }
    }
}
/// Cap on the README digest folded into the shared prefix. Generous on purpose: this
/// is prefix content, prefilled once per model and reused across the turn's fan-out
/// and every later turn on the same project.
const README_CAP: usize = 4096;
/// Entries listed per directory in the second level of the repo tree.
const TREE_BREADTH: usize = 24;
/// Directories never descended into — noise that would crowd out real source.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "dist", "build"];

pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    /// Reliable tool-calling for small models (the Needle shim). `None` unless built
    /// with `--features needle` and the assets were found; drives task emission.
    tool_caller: Option<Arc<dyn ToolCaller>>,
    /// Per-model tool dialects (schema/names/parse), matched to the tool-call model.
    dialects: Arc<Dialects>,
    /// Per-task JSONL record. Disabled unless `CORRODE_TELEMETRY` names a path.
    telemetry: Arc<Telemetry>,
    /// Optional bubblewrap confinement for every process the daemon spawns. Off
    /// unless `CORRODE_SANDBOX` is set; each session's terminals bind its own repo.
    sandbox: crate::sandbox::Sandbox,
    /// Monotonic id source for plan (provenance root) nodes, one per Prompt turn.
    next_plan_id: std::sync::atomic::AtomicU64,
    /// Per-repo resources (graph/vfs/skills), shared across users working a repo —
    /// the LMDB store can't be opened twice. Keyed by canonical repo path.
    repos: Mutex<HashMap<PathBuf, RepoResources>>,
    /// Per-(user,repo) live sessions (terminals + approval gate). A connection binds
    /// one of these in `run`; a user's tabs on the same repo share it.
    sessions: Mutex<HashMap<SessionKey, Arc<Session>>>,
    /// `user -> {token, hipfire_token}` for auth + fairness. `None` (or empty) =>
    /// auth off, connections anonymous.
    users: Option<HashMap<String, UserEntry>>,
    /// Default repo (`CORRODE_REPO`), for lazy binding when no `SelectRepo` arrives.
    default_repo: PathBuf,
    /// Embedding model id for building a repo's skill index (None if none served).
    embed_model: Option<String>,
}

impl Daemon {
    pub fn new(
        swarm: Swarm,
        roles: RoleModels,
        graph: Option<Arc<dyn crate::graph::GraphStore>>,
        vfs: Arc<dyn Vfs>,
        skills: SkillContext,
        embed_model: Option<String>,
        tool_caller: Option<Arc<dyn ToolCaller>>,
        repo_root: PathBuf,
        project: Project,
        dialects: Arc<Dialects>,
    ) -> Self {
        let telemetry = Arc::new(Telemetry::from_env());
        if telemetry.enabled() {
            eprintln!("telemetry: recording to $CORRODE_TELEMETRY");
        }
        let sandbox = crate::sandbox::Sandbox::from_env();
        let default_repo = canonical(&repo_root.to_string_lossy());
        // The default repo's resources come pre-built from `main` (or a test); seed
        // the registry so anonymous/default connections reuse them without reopening.
        // Step 7f: with a store open and `CORRODE_VFS_GRAPH` set, reads are composed
        // from graph nodes and fall through to the filesystem for anything the graph
        // does not hold. Off by default, so the passthrough stays the behaviour nobody
        // opted out of.
        let vfs: Arc<dyn Vfs> = match (&graph, crate::graphvfs::enabled()) {
            (Some(store), true) => {
                eprintln!("vfs: graph-backed reads enabled (CORRODE_VFS_GRAPH)");
                Arc::new(crate::graphvfs::GraphVfs::new(Arc::clone(store), vfs))
            }
            (None, true) => {
                // Asking for it and silently not getting it is the failure mode worth
                // avoiding: without a store there is nothing to compose from.
                eprintln!("vfs: CORRODE_VFS_GRAPH set but no graph store is open; using the filesystem");
                vfs
            }
            _ => vfs,
        };
        let default_res = RepoResources {
            repo_root: default_repo.clone(),
            project: Arc::new(project),
            graph,
            vfs,
            skill_scripts: Arc::new(skills.script_dirs()),
            skills: Arc::new(skills),
        };
        let mut repos = HashMap::new();
        repos.insert(default_repo.clone(), default_res);
        Self {
            swarm,
            roles,
            tool_caller,
            dialects,
            telemetry,
            sandbox,
            next_plan_id: std::sync::atomic::AtomicU64::new(0),
            repos: Mutex::new(repos),
            sessions: Mutex::new(HashMap::new()),
            users: load_users(),
            default_repo,
            embed_model,
        }
    }

    /// Whether a user table is configured (auth on). Empty table = off.
    fn auth_on(&self) -> bool {
        self.users.as_ref().is_some_and(|u| !u.is_empty())
    }

    /// Validate a user/token. Auth off => always accepts.
    fn authenticate(&self, user: &str, token: &str) -> bool {
        match &self.users {
            Some(map) if !map.is_empty() => map.get(user).is_some_and(|e| e.token == token),
            _ => true,
        }
    }

    /// The per-user hipfire bearer for fairness, if the user table carries one.
    fn owner_token_for(&self, user: &str) -> Option<String> {
        self.users.as_ref()?.get(user)?.hipfire_token.clone()
    }

    /// Get-or-open the shared resources for `repo` (graph/vfs/skills). Async because
    /// building the skill index embeds descriptions via hipfire; the lock is never
    /// held across the await.
    async fn repo_resources(&self, repo: &PathBuf) -> anyhow::Result<RepoResources> {
        if let Some(r) = self.repos.lock().unwrap().get(repo) {
            return Ok(r.clone());
        }
        let graph = crate::graph::open(repo);
        let vfs: Arc<dyn Vfs> = Arc::new(PassthroughVfs::new(repo));
        // Each repo carries its own identity and global-skill policy: a daemon serving
        // several projects must not hand one project's skills to another, which is the
        // whole point of the config.
        let project = Project::load(repo);
        let skills = SkillContext::build(
            repo,
            &self.swarm.client(),
            self.embed_model.clone(),
            &project.global_skills,
        )
        .await;
        let res = RepoResources {
            repo_root: repo.clone(),
            project: Arc::new(project),
            graph,
            vfs,
            skill_scripts: Arc::new(skills.script_dirs()),
            skills: Arc::new(skills),
        };
        Ok(self.repos.lock().unwrap().entry(repo.clone()).or_insert(res).clone())
    }

    /// Get-or-create the `(user, repo)` session for a connection. `path` empty =>
    /// the default repo. Sessions are shared across a user's tabs on the same repo.
    async fn bind_session(&self, user: Option<String>, path: &str) -> anyhow::Result<Arc<Session>> {
        let repo = if path.is_empty() { self.default_repo.clone() } else { canonical(path) };
        let key = SessionKey { user: user.unwrap_or_default(), repo: repo.clone() };
        if let Some(s) = self.sessions.lock().unwrap().get(&key) {
            return Ok(Arc::clone(s));
        }
        let res = self.repo_resources(&repo).await?;
        let owner_token = self.owner_token_for(&key.user);
        let session = Arc::new(Session::new(key.clone(), res, self.sandbox.clone(), owner_token));
        Ok(Arc::clone(
            self.sessions.lock().unwrap().entry(key).or_insert(session),
        ))
    }

    /// Run until the command channel closes. State that used to be process-global
    /// is now per-connection: the bound `session` (a `(user, repo)` tenant) and the
    /// authenticated `user`. Repo-scoped commands run against the session; a lazy
    /// default binding (to `CORRODE_REPO`) preserves single-tenant behaviour when no
    /// `SelectRepo` is sent.
    ///
    /// `Prompt`/`DocIngest`/`DocQuery` dispatch concurrently (long-lived, may block on
    /// approval), so the loop keeps receiving — crucially the `ApprovalResponse` that
    /// unblocks a waiting tool call. Everything else is inline to preserve ordering.
    pub async fn run(
        self: Arc<Self>,
        mut commands: mpsc::Receiver<AgentCommand>,
        events: mpsc::Sender<AgentEvent>,
    ) {
        let mut user: Option<String> = None;
        let mut session: Option<Arc<Session>> = None;
        while let Some(cmd) = commands.recv().await {
            match cmd {
                AgentCommand::Authenticate { user: u, token } => {
                    if self.authenticate(&u, &token) {
                        user = Some(u.clone());
                        session = None; // re-auth drops the old binding
                        let _ = events.send(AgentEvent::AuthOk { user: u }).await;
                    } else {
                        let _ = events.send(AgentEvent::AuthRequired).await;
                    }
                }
                AgentCommand::SelectRepo { path } => {
                    if self.auth_on() && user.is_none() {
                        let _ = events.send(AgentEvent::AuthRequired).await;
                        continue;
                    }
                    match self.bind_session(user.clone(), &path).await {
                        Ok(s) => {
                            let (p, u) = (s.repo_root.to_string_lossy().into_owned(), s.key.user.clone());
                            session = Some(s);
                            let _ = events.send(AgentEvent::RepoSelected { path: p, user: u }).await;
                        }
                        Err(e) => {
                            let _ = events
                                .send(AgentEvent::Error { message: format!("select repo: {e}") })
                                .await;
                        }
                    }
                }
                AgentCommand::ApprovalResponse { id, approved } => {
                    // Resolve on the connection's own session gate — a tenant can only
                    // answer its own approvals.
                    if let Some(s) = &session {
                        s.approvals.resolve(id, approved);
                    }
                }
                other => {
                    if self.auth_on() && user.is_none() {
                        let _ = events.send(AgentEvent::AuthRequired).await;
                        continue;
                    }
                    if session.is_none() {
                        match self.bind_session(user.clone(), "").await {
                            Ok(s) => session = Some(s),
                            Err(e) => {
                                let _ = events
                                    .send(AgentEvent::Error {
                                        message: format!("open default repo: {e}"),
                                    })
                                    .await;
                                continue;
                            }
                        }
                    }
                    let s = Arc::clone(session.as_ref().unwrap());
                    if matches!(
                        other,
                        AgentCommand::Prompt { .. }
                            | AgentCommand::DocIngest { .. }
                            | AgentCommand::DocQuery { .. }
                    ) {
                        let this = Arc::clone(&self);
                        let events = events.clone();
                        tokio::spawn(async move { this.handle(&s, other, &events).await });
                    } else {
                        self.handle(&s, other, &events).await;
                    }
                }
            }
        }
    }

    async fn handle(&self, session: &Arc<Session>, cmd: AgentCommand, events: &mpsc::Sender<AgentEvent>) {
        match cmd {
            AgentCommand::Prompt { text, priority } => {
                // The plan id exists before planning so TurnComplete is unconditional:
                // clients wait on it as the turn's terminal signal on every exit path.
                // Namespaced by project: a bare counter makes two repositories sharing
                // one graph store both write `plan-0`, with no way to tell their
                // lineage apart.
                let plan_id = session.project.scope(&format!(
                    "plan-{}",
                    self.next_plan_id
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ));
                let (subtasks, prefix) = match self.plan(session, &text, priority).await {
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
                // The turn declares its ceiling before it starts. hipfire's bands
                // schedule the GPU; nothing else bounds how much work the swarm
                // decides to create for itself.
                let deadline = turn_budget().map(|d| std::time::Instant::now() + d);
                // Cap concurrent generations: a wide plan otherwise fires every ready
                // task's request at once, and a memory-tight or fragile backend can
                // CRASH (not just shed) under that burst — observed with a DeltaNet
                // model on a 30 GiB APU. Default is effectively unlimited (hipfire's
                // admission control stays the real bound); set CORRODE_MAX_CONCURRENCY=N
                // to serialize on a constrained host.
                let gen_sem = Arc::new(tokio::sync::Semaphore::new(max_concurrency()));
                let execute = |task: plan_graph::PlanTask| {
                    let client = self.swarm.client();
                    let gen_sem = Arc::clone(&gen_sem);
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
                    let vfs = Arc::clone(&session.vfs);
                    let approvals = Arc::clone(&session.approvals);
                    let root = session.repo_root.clone();
                    let skill_scripts = Arc::clone(&session.skill_scripts);
                    let owner_token = session.owner_token.clone();
                    let sandbox = self.sandbox.clone();
                    // The store gives `search_files` its soft half. `None` in the base
                    // build, where search stays literal-only.
                    let graph = session.graph.clone();
                    // Cross-encoder reranking for graph hits, when CORRODE_RERANK_MODEL
                    // names a served reranker. Same client the swarm generates through.
                    let reranker = Some(Arc::clone(&client));
                    let dialects = Arc::clone(&self.dialects);
                    let telemetry = Arc::clone(&self.telemetry);
                    let telemetry_plan = plan_id.clone();
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
                        // Hold a generation permit for the whole task (released on drop
                        // when the task finishes), bounding how many hit the backend at once.
                        let _gen_permit = gen_sem.acquire_owned().await.ok();
                        // `artifacts` collects the files a tool-loop task wrote (its code
                        // nodes in provenance). Coder tasks fan out K read-only proposal
                        // attempts first when CORRODE_FANOUT > 1; everything else runs
                        // the capability paths directly (see `run_task`).
                        let mut artifacts = Vec::new();
                        let toolbox = ToolBox::new(vfs, root, skill_scripts)
                            .with_sandbox(sandbox)
                            .with_graph(graph)
                            .with_reranker(reranker)
                            .with_owner_token(owner_token);
                        let started = std::time::Instant::now();
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
                                deadline,
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
                                deadline,
                            )
                            .await
                        };

                        // One line per execution, before follow-up emission so a task
                        // that fails is still recorded (see `telemetry.rs`).
                        telemetry.record(&crate::telemetry::TaskRecord {
                            at: crate::telemetry::now_secs(),
                            plan: &telemetry_plan,
                            task: id,
                            role: role.as_str(),
                            model: &model,
                            band: band.as_u8(),
                            fanout: if role == Role::Coder { fanout } else { 1 },
                            prefix_bytes: prefix.len(),
                            task_bytes: prompt.len(),
                            output_bytes: output.as_ref().map(|t| t.len()).unwrap_or(0),
                            duration_ms: started.elapsed().as_millis(),
                            artifacts: artifacts.len(),
                            ok: output.is_ok(),
                            error: output.as_ref().err().map(|e| e.to_string()),
                        });

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
                let mut budget = plan_graph::run_reactive_until(&mut graph, &execute, deadline).await;

                // One plan-level review pass over the settled work: the review role
                // reads the digest (and, through its tools, the written files) and
                // routes fixes through the normal follow-up channel; the second
                // reactive drive runs the review task and whatever it emits.
                // ponytail: one round — loop-until-clean is the upgrade once fix
                // quality is measured.
                if plan_review_enabled() {
                    if let Some(digest) = graph.review_digest(REVIEW_OUTPUT_CAP) {
                        graph.add(Role::Review, planner::plan_review_task(&digest), Vec::new());
                        let second =
                            plan_graph::run_reactive_until(&mut graph, &execute, deadline).await;
                        budget.expired |= second.expired;
                        budget.shed += second.shed;
                        budget.unlaunched = second.unlaunched;
                    }
                }

                // Tasks left pending after the scheduler settled had a failed or
                // unmet dependency (a failed emitter) — surface them rather than
                // dropping them silently.
                // A turn cut short by its budget is reported as such: "could not be
                // scheduled (a dependency failed)" would be a lie, and the two have
                // opposite fixes — raise the budget vs debug the failure.
                if budget.expired {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!(
                                "turn budget exhausted: {} task(s) not launched, \
                                 {} emission(s) dropped (CORRODE_TURN_BUDGET_S)",
                                budget.unlaunched, budget.shed
                            ),
                        })
                        .await;
                } else {
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
                }

                // Persist the plan's provenance (plan <- task/contract <- code) to the
                // graph store, so the code<->task<->plan lineage is queryable — and
                // ship it to the graph explorer.
                self.persist_provenance(session, &graph);
                // Re-ingest what the turn wrote. Staleness is the one real cost of
                // indexing code — a stale index is confidently wrong where reading
                // files is merely slow — so the index is refreshed at the moment the
                // truth changes, using the written paths provenance already collects.
                self.ingest_written(session, &graph).await;
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
                let ev = self.doc_query(session, &question).await;
                let _ = events.send(ev).await;
            }
            AgentCommand::DocIngest { path } => {
                let ev = self.ingest_doc(session, path).await;
                let _ = events.send(ev).await;
            }
            AgentCommand::ListDir { path } => {
                let ev = match session.vfs.list(&path).await {
                    Ok(mut entries) => {
                        // Compose graph provenance onto the FS listing: tag each file
                        // the graph tracks with its code-node id, so the explorer can
                        // pivot a file to its provenance (ListNeighbors). One scan per
                        // listing; no store -> node_id stays None (plain passthrough).
                        if let Some(store) = &session.graph {
                            let store = store.clone();
                            if let Ok(Ok(code)) =
                                tokio::task::spawn_blocking(move || store.code_nodes()).await
                            {
                                let by_path: std::collections::HashMap<String, String> =
                                    code.into_iter().collect(); // last id per path wins
                                for e in &mut entries {
                                    if let Some(node_id) = by_path.get(&e.path) {
                                        e.node_id = Some(node_id.clone());
                                    }
                                }
                            }
                        }
                        AgentEvent::DirListing { path, entries }
                    }
                    Err(e) => AgentEvent::Error { message: e.to_string() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::ReadFile { path } => {
                // Read-only: no approval gate. Cap the returned text so a huge file
                // can't blow the ws frame budget; the model/tools path reads full
                // files itself, this is just the explorer viewer.
                const READ_CAP: usize = 256 * 1024;
                let ev = match session.vfs.read(&path).await {
                    Ok(bytes) => {
                        let full = String::from_utf8_lossy(&bytes).into_owned();
                        let truncated = full.len() > READ_CAP;
                        let content = if truncated {
                            let end = crate::tools::floor_char_boundary(&full, READ_CAP);
                            full[..end].to_string()
                        } else {
                            full
                        };
                        AgentEvent::FileContent { path, content, truncated }
                    }
                    Err(e) => AgentEvent::Error { message: format!("read {path}: {e}") },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::ListNeighbors { node_id } => {
                // Read-only graph expansion. No store (base build / open failed) ->
                // empty neighborhood, not an error, so the explorer just shows no
                // expansion. neighbors() is a blocking LMDB read -> spawn_blocking.
                let ev = match &session.graph {
                    Some(store) => {
                        let store = store.clone();
                        let id = node_id.clone();
                        match tokio::task::spawn_blocking(move || store.neighbors(&id)).await {
                            Ok(Ok(nodes)) => AgentEvent::Neighbors { node_id, nodes },
                            Ok(Err(e)) => AgentEvent::Error {
                                message: format!("neighbors {node_id}: {e}"),
                            },
                            Err(e) => AgentEvent::Error {
                                message: format!("neighbors {node_id}: {e}"),
                            },
                        }
                    }
                    None => AgentEvent::Neighbors { node_id, nodes: Vec::new() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::ListDocs => {
                // Read-only. No store -> empty list (nothing ingested/queryable).
                let ev = match &session.graph {
                    Some(store) => {
                        let store = store.clone();
                        match tokio::task::spawn_blocking(move || store.list_docs()).await {
                            Ok(Ok(rows)) => AgentEvent::DocList {
                                docs: rows
                                    .into_iter()
                                    .map(|(id, title)| corrode_core::DocEntry { id, title })
                                    .collect(),
                            },
                            Ok(Err(e)) => AgentEvent::Error { message: format!("list docs: {e}") },
                            Err(e) => AgentEvent::Error { message: format!("list docs: {e}") },
                        }
                    }
                    None => AgentEvent::DocList { docs: Vec::new() },
                };
                let _ = events.send(ev).await;
            }
            AgentCommand::TerminalInput { session: term, data } => {
                // Write keystrokes to the (per-tenant) pty; its output streams back as
                // TerminalOutput from the session's reader thread. `term` is the
                // client-chosen id, unique per browser tab.
                if let Err(e) = session.terminals.input(&term, &data, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal input: {e}"),
                        })
                        .await;
                }
            }
            AgentCommand::TerminalResize {
                session: term,
                cols,
                rows,
            } => {
                if let Err(e) = session.terminals.resize(&term, cols, rows, events) {
                    let _ = events
                        .send(AgentEvent::Error {
                            message: format!("terminal resize: {e}"),
                        })
                        .await;
                }
            }
            // Bound in `run` (they need the per-connection session/auth locals), never
            // dispatched here.
            AgentCommand::Authenticate { .. }
            | AgentCommand::SelectRepo { .. }
            | AgentCommand::ApprovalResponse { .. } => unreachable!("handled in run"),
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
        session: &Session,
        text: &str,
        priority: Priority,
    ) -> anyhow::Result<(Vec<planner::PlannedSubtask>, String)> {
        // Built once and shared, byte-identical, by the planning call and every
        // subagent, so hipfire batches them prefix-shared and reuses KV.
        let prefix = self.context_prefix(session, text).await;

        let orch_model = self
            .roles
            .model_for(Role::Orchestration)
            .unwrap_or_default()
            .to_string();
        let plan_task = Task {
            prompt: planner::orchestration_prompt(&prefix, text),
            priority,
            model: orch_model,
            owner_token: session.owner_token.clone(),
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

    /// Re-ingest every file this turn wrote into the code graph.
    ///
    /// Only files the swarm actually changed: a full re-scan would be wasteful and,
    /// worse, would hide which nodes a turn is responsible for. Best-effort like
    /// provenance — a store that cannot take the write logs once and stops, because
    /// losing an index refresh must never fail the work that produced it.
    ///
    /// Every written path is ingested, whatever the language: a backend exists for Rust
    /// and a byte-exact fallback for everything else, so an unfamiliar codebase is
    /// absorbed with less structure rather than not at all.
    async fn ingest_written(&self, session: &Session, graph: &plan_graph::PlanGraph) {
        let Some(store) = &session.graph else {
            return;
        };
        let mut seen: std::collections::BTreeSet<&str> = Default::default();
        let mut failed = 0usize;
        // The repo's real directory set, so `docmap` can confirm a cited path exists
        // rather than inventing an edge to a directory nobody has. Computed once per
        // turn: a doc naming twenty subsystems must not cost twenty walks.
        let known_dirs: std::collections::BTreeSet<String> = session
            .vfs
            .tracked_files()
            .await
            .unwrap_or_default()
            .iter()
            .flat_map(|p| {
                let mut acc = Vec::new();
                let mut cur = p.as_str();
                while let Some((d, _)) = cur.rsplit_once('/') {
                    acc.push(d.to_string());
                    cur = d;
                }
                acc
            })
            .collect();
        for node in &graph.provenance().nodes {
            // Code nodes are `{plan}:code:{path}`; match on the kind rather than
            // reaching for the plan id, which the graph keeps private.
            if node.kind != plan_graph::NodeKind::Code {
                continue;
            }
            let Some(path) = node.id.rsplit(":code:").next() else {
                continue;
            };
            if !seen.insert(path) {
                continue;
            }
            let Ok(bytes) = session.vfs.read(path).await else {
                continue; // written then removed, or outside the VFS
            };
            let Ok(src) = String::from_utf8(bytes) else {
                continue;
            };
            // Backend chosen per path: Rust gets syn, anything else falls back to the
            // plain-text projector, which still ingests and projects byte-exactly with
            // less structure. Absorbing an unfamiliar codebase is never blocked.
            let lang = crate::projection::for_path(path);
            // Reconcile against what the store already holds, rather than re-scanning
            // from scratch. A fresh scan renumbers the file end to end, and since ids
            // derive from the order key that re-addresses every node for a one-line
            // change — the churn the sparse key exists to prevent. `stored` is empty for
            // a file the graph has not seen, which is exactly a first ingest.
            let stored = store.file_nodes(path).unwrap_or_default();
            let Ok((fw, update)) = crate::projection::ingest::file_against(lang.as_ref(), path, &src, &stored)
            else {
                continue; // unparseable mid-edit: leave the previous nodes in place
            };
            if !update.changed.is_empty() || update.rebalanced {
                eprintln!(
                    "code ingest {path}: {} changed, {} cosmetic{}",
                    update.changed.len(),
                    update.cosmetic,
                    if update.rebalanced { ", rebalanced" } else { "" }
                );
            }
            if let Err(e) = store.replace_file(&fw) {
                // Continue, do not abort the turn. One unwritable file must not stop
                // every later file from being indexed: measured on curl, 5 of 2,995
                // files contain a token longer than LMDB's 511-byte max key, which the
                // BM25 index rejects (`MDB_BAD_VALSIZE`) — a base64 blob on one line is
                // enough. Returning here let one such file silently cost the whole
                // turn's code ingest.
                failed += 1;
                if failed <= 3 {
                    eprintln!("code ingest failed for {path}: {e}");
                }
                continue;
            }
            // Join the graphs: place the file in its directory, and link it to whatever
            // it documents. A README or design note written by a task becomes reachable
            // from the code it describes, which is the whole point of ingesting prose
            // into the same store as source.
            let describes = crate::projection::docmap::describes(path, &src, &known_dirs);
            if let Err(e) = store.place_file(path, &describes) {
                if failed < 3 {
                    eprintln!("placing {path} failed: {e}");
                }
            }
        }
        if failed > 3 {
            eprintln!("code ingest: {failed} files failed in total");
        }
    }

    /// Persist a plan's provenance subgraph to the graph store, if one is open.
    /// Best-effort: without `--features helix` there's no store (a no-op), and the
    /// HelixDB write path is still stubbed — so on the first write error we log once and
    /// stop rather than spamming. The in-memory provenance is already correct; this is
    /// the durability seam.
    fn persist_provenance(&self, session: &Session, graph: &plan_graph::PlanGraph) {
        let Some(store) = &session.graph else {
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

    /// GraphRAG: retrieve, then synthesize. Embed the question (query-side prompt)
    /// and vector-search the chunks (BM25 fallback with no embedding model), then
    /// feed the retrieved chunks + question to a hipfire chat model for a grounded
    /// answer that cites the chunk ids. Synthesis failure (no model, or the call
    /// errors) degrades to the raw chunks — retrieval still stands.
    async fn doc_query(&self, session: &Session, question: &str) -> AgentEvent {
        let Some(g) = &session.graph else {
            return AgentEvent::Error {
                message: "DocQuery unavailable: build with --features helix and open a graph store"
                    .into(),
            };
        };
        let query_vec = match session.skills.embed_model() {
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
        let hits = match searched {
            Ok(Ok(hits)) => hits,
            Ok(Err(e)) => return AgentEvent::Error { message: format!("doc search: {e}") },
            Err(e) => return AgentEvent::Error { message: format!("doc search task: {e}") },
        };
        let grounded_on: Vec<String> = hits.iter().map(|(id, _)| id.clone()).collect();
        let raw = || {
            hits.iter()
                .map(|(id, text)| format!("> [{id}]\n{text}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        // Nothing retrieved: no point synthesizing.
        if hits.is_empty() {
            return AgentEvent::DocAnswer {
                text: "No matching documentation.".into(),
                grounded_on,
            };
        }
        // Synthesis pass: a read/summarize model answers grounded on the chunks.
        let text = match self.roles.model_for(Role::Research) {
            Some(model) => {
                let prompt = planner::doc_synthesis_prompt(question, &hits);
                match self
                    .swarm
                    .client()
                    .respond(model, &prompt, Priority::Default, session.owner_token.as_deref())
                    .await
                {
                    Ok(answer) if !answer.trim().is_empty() => answer,
                    _ => raw(), // model absent/failed/empty -> hand back the chunks
                }
            }
            None => raw(),
        };
        AgentEvent::DocAnswer { text, grounded_on }
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
    async fn ingest_doc(&self, session: &Session, path: String) -> AgentEvent {
        let canonical = match self.confine_doc_path(session, &path) {
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
        let persisted = self.persist_doc(session, &doc).await;
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
    fn confine_doc_path(&self, session: &Session, path: &str) -> anyhow::Result<String> {
        let roots: Vec<PathBuf> = std::env::var("CORRODE_DOC_ROOTS")
            .ok()
            .map(|v| v.split(':').map(PathBuf::from).collect())
            .unwrap_or_else(|| vec![session.repo_root.clone()]);
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
    async fn ingest_doc(&self, _session: &Session, _path: String) -> AgentEvent {
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
    async fn persist_doc(&self, session: &Session, doc: &crate::ingest::IngestedDoc) -> bool {
        let Some(store) = &session.graph else {
            return false;
        };
        let embeddings = self.embed_chunks(session, &doc.chunks).await;
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
    async fn embed_chunks(
        &self,
        session: &Session,
        chunks: &[(String, String)],
    ) -> Option<Vec<Option<Vec<f32>>>> {
        // ponytail: chars as a token proxy — ~6000 chars sits safely under the
        // 2048-token embedding cap for prose; a real tokenizer clip can come with
        // the docling `chunking` feature.
        const EMBED_CLIP_CHARS: usize = 6000;
        const BATCH: usize = 64;
        let model = session.skills.embed_model()?;
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
    async fn context_prefix(&self, session: &Session, task: &str) -> String {
        // Name the repository. Without this the prefix said only "a shared repository",
        // leaving the skill manifest as the strongest identity signal in the prompt —
        // which is how a C++ project got explained as if it were hipfire.
        let mut s = format!(
            "You are a subagent in the Corrode coding-agent swarm working on the \
`{}` repository at {}.\n",
            session.project.name,
            session.project.root.display(),
        );
        // Project rules (AGENTS.md) + skills relevant to this task. Byte-identical
        // across the turn's subagents (same task), so they share the KV prefill.
        let rules = session.skills.agents_rules();
        if !rules.trim().is_empty() {
            s.push_str("\nProject instructions (AGENTS.md):\n");
            s.push_str(rules.trim_end());
            s.push('\n');
        }
        let manifest = session
            .skills
            .prefix_section(task, &self.swarm.client(), TOP_K_SKILLS)
            .await;
        if !manifest.is_empty() {
            s.push('\n');
            s.push_str(&manifest);
        }
        // The README is the canonical human-written answer to "what is this
        // repository". Omitting it is why the swarm, told only a list of filenames,
        // described a lock-free threading library as an image-stitching tool. It is
        // project-stable, so it rides the shared prefix and is prefilled once.
        if let Some(readme) = self.readme_digest(session).await {
            s.push_str(&readme);
        }
        s.push_str("\nRepository tree:\n");
        s.push_str(&self.repo_tree(session).await);
        s
    }

    /// `README.md` (or a close variant), truncated to [`README_CAP`] on a line
    /// boundary. `None` when the repo has none.
    async fn readme_digest(&self, session: &Session) -> Option<String> {
        for name in ["README.md", "README", "README.txt", "readme.md"] {
            let Ok(bytes) = session.vfs.read(name).await else {
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let end = crate::tools::floor_char_boundary(trimmed, README_CAP);
            let body = &trimmed[..end];
            // Cut at the last newline so a truncated digest never ends mid-sentence.
            let body = if end < trimmed.len() {
                body.rfind('\n').map(|i| &body[..i]).unwrap_or(body)
            } else {
                body
            };
            let mut out = format!("\nProject README ({name}):\n");
            out.push_str(body);
            out.push('\n');
            if end < trimmed.len() {
                out.push_str("  [...truncated; read the file for the rest]\n");
            }
            return Some(out);
        }
        None
    }

    /// Root listing plus one level into each directory. A single level showed no
    /// source files at all for any project that keeps them in a subdirectory, which
    /// left the model inferring a file list instead of reading one.
    ///
    /// ponytail: two levels, breadth-capped, no recursion into the tail — the
    /// graph-backed VFS is where a real relevance-ranked tree comes from.
    async fn repo_tree(&self, session: &Session) -> String {
        let Ok(entries) = session.vfs.list("").await else {
            return "  (listing unavailable)\n".to_string();
        };
        let mut out = String::new();
        for e in &entries {
            // Mark directories: `bytes` is 0 for a dir (vfs.rs), and an empty file is
            // 0 too, so size alone reads as "repo is empty" — which is exactly what a
            // subagent concluded on a repo whose sources all live one level down.
            if !e.is_dir {
                out.push_str(&format!("  {} ({} bytes)\n", e.path, e.bytes));
                continue;
            }
            out.push_str(&format!("  {}/\n", e.path));
            if SKIP_DIRS.contains(&e.path.as_str()) {
                continue;
            }
            let Ok(children) = session.vfs.list(&e.path).await else {
                continue;
            };
            for c in children.iter().take(TREE_BREADTH) {
                if c.is_dir {
                    out.push_str(&format!("    {}/\n", c.path));
                } else {
                    out.push_str(&format!("    {} ({} bytes)\n", c.path, c.bytes));
                }
            }
            if children.len() > TREE_BREADTH {
                out.push_str(&format!(
                    "    ... {} more\n",
                    children.len() - TREE_BREADTH
                ));
            }
        }
        out
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

/// `CORRODE_TURN_BUDGET_S`: wall-clock ceiling for one Prompt turn. Absent, zero or
/// unparseable -> unbounded, which is today's behaviour and stays the default: a ceiling
/// that silently truncates work is worse than a slow turn for anyone who has not asked
/// for one.
fn turn_budget() -> Option<std::time::Duration> {
    std::env::var("CORRODE_TURN_BUDGET_S")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(std::time::Duration::from_secs)
}

/// `CORRODE_MAX_CONCURRENCY`: cap on subagent generations in flight at once. Default
/// 1024 (effectively unlimited — the swarm is bounded by MAX_PLAN_TASKS and hipfire's
/// own admission control). Set to a small N to serialize on a backend that crashes
/// under a concurrent burst. A parsed 0 is treated as 1 (never a zero-permit deadlock).
fn max_concurrency() -> usize {
    std::env::var("CORRODE_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| n.max(1))
        .unwrap_or(1024)
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
    deadline: Option<std::time::Instant>,
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
    // Traces are recorded on BOTH capability paths. Instrumenting only the Needle loop
    // meant a model that emits its own calls (the `*minicpm*` / `*zaya*` dialects) wrote
    // no notes at all — silently, since nothing reports notes it never tried to make.
    let mut steps: Vec<crate::trace::Step> = Vec::new();
    let mut touched: Vec<String> = Vec::new();
    for _ in 0..MAX_TOOL_STEPS {
        // Cooperative cancellation at a STEP boundary — never mid-call. A mutating
        // tool call that is half-applied is worse than a turn that runs long, and
        // there is no way to un-run one. Reported, not silent: a truncated answer
        // that looks complete is how a budget turns into a wrong result.
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            let _ = events
                .send(AgentEvent::Error {
                    message: format!("task {id}: stopped at a tool-step boundary (turn budget)"),
                })
                .await;
            return Ok(format!("{last}\n[stopped: turn budget reached]"));
        }
        let prompt = planner::native_tool_prompt(prefix, role, task, &scratchpad);
        let (text, _reasoning) = client
            .respond_full(model, &prompt, band, toolbox.owner_token(), Some(&tools), Some(&effort))
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
            steps.push(crate::trace::Step {
                said: text.clone(),
                intent: None,
                tool: None,
                observation: None,
            });
            record_trace(&toolbox, task, &steps, &touched);
            return Ok(text); // no call -> this turn is the final answer
        };
        if let Some(p) = crate::tools::arg_str(call, "path") {
            if !touched.iter().any(|t| t == p) {
                touched.push(p.to_string());
            }
        }
        let observation =
            gate_and_execute(call, &toolbox, approvals, events, id, written, seen, read_only)
                .await;
        steps.push(crate::trace::Step {
            said: text.clone(),
            intent: Some(crate::tools::describe(call)),
            tool: Some(call.name.clone()),
            observation: Some(observation.clone()),
        });
        scratchpad.push_str(&format!(
            "\nCALLED: {}\nRESULT: {observation}\n",
            crate::tools::describe(call)
        ));
    }
    record_trace(&toolbox, task, &steps, &touched);
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
    deadline: Option<std::time::Instant>,
) -> anyhow::Result<String> {
    // Render the exec toolset in the tool-call model's dialect once; parse each reply
    // with the same dialect (which maps its tool names back to canonical).
    let dialect = dialects.resolve(caller.model_id());
    let schema = dialect.render(crate::tools::role_tools(role), None);
    let mut scratchpad = String::new();
    // The trace, kept as the loop already separates it: what the model said, and what a
    // tool returned. `trace::extract` needs no parsing of the scratchpad because the two
    // are never merged here in the first place.
    let mut steps: Vec<crate::trace::Step> = Vec::new();
    // Paths the task touched, taken from the STRUCTURED call rather than parsed out of
    // the model's prose — a note bound to a path guessed from English would attach real
    // findings to the wrong file.
    let mut touched: Vec<String> = Vec::new();
    // Canonical tool name of the turn's call, so extraction can tell an outcome from
    // content without re-reading the model's prose.
    let mut called: Option<String> = None;
    let mut last = String::new();
    for _ in 0..MAX_TOOL_STEPS {
        // Cooperative cancellation at a STEP boundary — never mid-call. A mutating
        // tool call that is half-applied is worse than a turn that runs long, and
        // there is no way to un-run one. Reported, not silent: a truncated answer
        // that looks complete is how a budget turns into a wrong result.
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            let _ = events
                .send(AgentEvent::Error {
                    message: format!("task {id}: stopped at a tool-step boundary (turn budget)"),
                })
                .await;
            return Ok(format!("{last}\n[stopped: turn budget reached]"));
        }
        let prompt = planner::tool_loop_prompt(prefix, role, task, &scratchpad);
        let text = client.respond(model, &prompt, band, toolbox.owner_token()).await?;
        let _ = events
            .send(AgentEvent::SubagentOutput {
                id,
                text: text.clone(),
            })
            .await;
        last = text.clone();

        let Some(intent) = crate::tools::parse_tool_intent(&text) else {
            // Final turn: it called nothing, so it contributes claims only.
            steps.push(crate::trace::Step {
                said: text.clone(),
                intent: None,
                tool: None,
                observation: None,
            });
            record_trace(&toolbox, task, &steps, &touched);
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
                    called = Some(c.name.clone());
                    if let Some(p) = crate::tools::arg_str(c, "path") {
                        if !touched.iter().any(|t| t == p) {
                            touched.push(p.to_string());
                        }
                    }
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
        steps.push(crate::trace::Step {
            said: text.clone(),
            intent: Some(intent.clone()),
            tool: called.clone(),
            observation: Some(observation.clone()),
        });
        scratchpad.push_str(&format!("\nTOOL: {intent}\nRESULT: {observation}\n"));
    }
    // Step budget spent: hand back the last turn as the answer.
    record_trace(&toolbox, task, &steps, &touched);
    Ok(last)
}

/// Extract a task's notes and persist them.
///
/// No new store method: a note is `upsert_node` and each edge is `add_edge`, both already
/// on `GraphStore`. Notes are append-only — a correction arrives as a new note plus a
/// `supersedes` edge, never as an edit — so nothing here removes or rewrites what an
/// earlier task recorded, however wrong it turns out to be.
fn record_trace(toolbox: &ToolBox, task: &str, steps: &[crate::trace::Step], touched: &[String]) {
    use crate::trace::NoteKind;
    let notes = crate::trace::extract(task, steps);
    if notes.is_empty() {
        return;
    }
    let observed = notes.iter().filter(|n| n.kind == NoteKind::Observed).count();
    eprintln!(
        "trace: {} note(s) from {} step(s) — {observed} observed, {} asserted, {} path(s)",
        notes.len(),
        steps.len(),
        notes.len() - observed,
        touched.len()
    );

    let Some(store) = toolbox.graph() else {
        eprintln!("trace: no store on this session; notes extracted but not persisted");
        return;
    };
    // The task must exist as a node before an edge can name it.
    if let Err(e) = store.upsert_node(task, "task", task) {
        eprintln!("trace: cannot record task node ({e}); notes not persisted");
        return;
    }
    let (mut wrote, mut edges) = (0usize, 0usize);
    for n in &notes {
        // `kind` carries the provenance, so a reader can weigh a tool result against a
        // claim rather than the store flattening both into "a note".
        if let Err(e) = store.upsert_node(&n.id(), n.kind.as_str(), &n.text) {
            eprintln!("trace: note {} failed ({e})", n.id());
            continue;
        }
        wrote += 1;
        if store.add_edge(&n.id(), "noted_by", task).is_ok() {
            edges += 1;
        }
        for path in touched {
            // Bound to the file, not to a node inside it: a finding like "this loader is
            // never called" is about the file's role, and binding it to whichever node
            // happened to be read would be a precision the trace does not have.
            // The file node may not exist yet — a task can read a file the ingest pass
            // has not walked — so create it rather than dropping the edge.
            let file_id = format!("file:{path}");
            let _ = store.upsert_node(&file_id, "source_file", path);
            match store.add_edge(&n.id(), "about", &file_id) {
                Ok(()) => edges += 1,
                Err(e) => eprintln!("trace: binding {} to {file_id} failed ({e})", n.id()),
            }
        }
    }
    eprintln!("trace: persisted {wrote}/{} note(s), {edges} edge(s)", notes.len());

    // Supersede prior notes on the same files. The claim is ordering — this note was
    // written with more of the trace behind it — not correctness.
    let prior: Vec<String> = touched
        .iter()
        .filter_map(|p| store.neighbors(&format!("file:{p}")).ok())
        .flatten()
        .filter(|n| n.kind == "observed" || n.kind == "asserted")
        .map(|n| n.id)
        .filter(|id| !id.starts_with(&format!("note:{task}#")))
        .collect();
    for (from, rel, to) in crate::trace::note_edges(&notes, task, &prior) {
        if rel == "supersedes" {
            let _ = store.add_edge(&from, rel, &to);
        }
    }
}

/// One full execution of a task: pick the capability path and run it.
///  1. the model emits its own tool calls (its dialect says so) — declare tools on
///     the request and parse the reply directly.
///  2. otherwise the Needle-mediated loop, which reconstructs a call from a
///     plain-English line — for ANY model, not just small ones.
///  3. single-shot only when there is no tool caller at all.
///
/// Path 2 was once gated on `roles::is_small_model`, on the theory that a large
/// model does not need help formatting a call. The consequence was that it got no
/// tools at all: the most capable model in the roster could not read a file. A 35B
/// was observed emitting `<tool_call>` blocks into a void, and the swarm answered
/// repository questions by guessing. Withholding tools from a model whose
/// capability is unknown is the wrong default — an agent that can read is
/// recoverable, one that must guess is not — and the gate was a substring match on
/// the model id anyway (`is_small_model("Gemma-3-27B") == true`).
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
    deadline: Option<std::time::Instant>,
) -> anyhow::Result<String> {
    let role_dialect = dialects.resolve(model);
    if role_dialect.emits_own_calls() {
        run_native_tool_loop(
            client, model, band, role_dialect, toolbox, approvals, prefix, role, task, events,
            id, written, read_only, seen, deadline,
        )
        .await
    } else if let Some(caller) = tool_caller {
        run_tool_loop(
            client, model, band, caller, toolbox, approvals, dialects, prefix, role, task,
            events, id, written, read_only, seen, deadline,
        )
        .await
    } else {
        let full = planner::subagent_prompt(prefix, role, task);
        let out = if client.streaming() {
            // Relay each delta to the UI as it arrives (best-effort: try_send drops
            // under backpressure, the final SubagentOutput below reconciles).
            let ev = events.clone();
            let (text, _reasoning) = client
                .respond_streaming(model, &full, band, toolbox.owner_token(), |delta| {
                    let _ = ev.try_send(AgentEvent::SubagentDelta {
                        id,
                        text: delta.to_string(),
                    });
                })
                .await?;
            Ok(text)
        } else {
            client.respond(model, &full, band, toolbox.owner_token()).await
        };
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
    deadline: Option<std::time::Instant>,
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
                deadline,
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
            .respond(
                review_model,
                &judge_prompt,
                planner::band_for(Role::Review),
                toolbox.owner_token(),
            )
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
        &steered, events, id, written, false, seen, deadline,
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
    use crate::project::GlobalSkills;
    use crate::vfs::PassthroughVfs;

    fn test_daemon() -> Daemon {
        Daemon::new(
            Swarm::new(Client::new("http://127.0.0.1:1", None), 1),
            RoleModels::uniform("test-model"),
            None,
            Arc::new(PassthroughVfs::new(std::env::temp_dir())),
            SkillContext::default(),
            None, // embed_model
            None, // tool_caller
            std::env::temp_dir(),
            Project::load(&std::env::temp_dir()),
            Arc::new(Dialects::default()),
        )
    }

    /// A tool caller that panics if used — the deadline check must return before any
    /// model or tool call happens, so reaching it is the failure.
    struct NeverCalled;
    impl crate::toolcall::ToolCaller for NeverCalled {
        fn generate(&self, _q: &str, _t: &str) -> anyhow::Result<String> {
            panic!("tool caller invoked past the deadline");
        }
        fn model_id(&self) -> &str {
            "never"
        }
    }

    /// Cooperative cancellation: an already-expired deadline stops the tool loop at the
    /// top of its first step, before touching the client or the tool caller. The client
    /// here points at a closed port and the caller panics, so anything other than an
    /// immediate return fails loudly rather than hanging.
    ///
    /// Live runs could not reach this branch reliably — it needs a task to enter step
    /// two while past the deadline, and the model kept answering in one turn — so the
    /// guarantee is pinned here instead.
    #[tokio::test]
    async fn an_expired_deadline_stops_the_tool_loop_before_any_call() {
        let dir = std::env::temp_dir().join(format!("corrode-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let client = Client::new("http://127.0.0.1:1", None);
        let dialects = Dialects::default();
        let approvals = ApprovalGate::default();
        let (tx, mut rx) = mpsc::channel(8);
        let seen = std::sync::Mutex::new(SeenCalls::default());
        let mut written = Vec::new();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(std::collections::HashMap::new()),
        );
        let expired = std::time::Instant::now() - std::time::Duration::from_secs(1);

        let out = run_tool_loop(
            &client,
            "test-model",
            Priority::Default,
            Arc::new(NeverCalled),
            toolbox,
            &approvals,
            &dialects,
            "prefix",
            Role::Coder,
            "task",
            &tx,
            7,
            &mut written,
            false,
            &seen,
            Some(expired),
        )
        .await
        .expect("returns rather than erroring");

        assert!(out.contains("stopped: turn budget reached"), "got: {out}");
        // Cutting a task short is reported, not silent: a truncated answer that looks
        // complete is how a budget turns into a wrong result.
        let ev = rx.try_recv().expect("an event was emitted");
        match ev {
            AgentEvent::Error { message } => {
                assert!(message.contains("task 7"), "{message}");
                assert!(message.contains("tool-step boundary"), "{message}");
            }
            other => panic!("expected Error, got {other:?}"),
        }
        assert!(written.is_empty(), "nothing was executed");

        std::fs::remove_dir_all(&dir).ok();
    }

    // Multi-tenancy keying: a (user, repo) session is created once and reused across
    // that user's connections; different users on the same repo get separate sessions
    // (separate terminals + approval gate). Uses the default repo (path "") so it hits
    // the pre-seeded resources — no network, no skill rebuild. Auth is off by default
    // (no CORRODE_USERS), so authenticate accepts anyone.
    #[tokio::test]
    async fn sessions_are_keyed_by_user_and_repo_and_reused() {
        let d = test_daemon();
        assert!(!d.auth_on(), "no CORRODE_USERS -> auth off");
        assert!(d.authenticate("anyone", "whatever"), "auth off accepts any token");

        let a1 = d.bind_session(Some("alice".into()), "").await.unwrap();
        let a2 = d.bind_session(Some("alice".into()), "").await.unwrap();
        let bob = d.bind_session(Some("bob".into()), "").await.unwrap();
        assert!(Arc::ptr_eq(&a1, &a2), "same (user,repo) reuses one session");
        assert!(!Arc::ptr_eq(&a1, &bob), "different users get different sessions");
        assert_eq!(a1.key.user, "alice");
        // The gate is per-session, so alice's and bob's are distinct instances.
        assert!(!Arc::ptr_eq(&a1.approvals, &bob.approvals));
    }

    /// The prefix must carry the repository's own account of itself and enough tree
    /// to see source files. A one-level listing showed none at all for a project
    /// that keeps sources in a subdirectory, and the swarm invented file names
    /// instead — so both halves are asserted here against a synthetic repo shaped
    /// like that (`stitch/atom.h`, not `atom.h`).
    #[tokio::test]
    async fn prefix_carries_readme_and_a_second_tree_level() {
        let root = std::env::temp_dir().join(format!("corrode-prefix-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("README.md"), "Widget is a lock-free queue library.").unwrap();
        std::fs::write(root.join("src/atom.h"), "// atom").unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main").unwrap();

        let daemon = Daemon::new(
            Swarm::new(Client::new("http://127.0.0.1:1", None), 1),
            RoleModels::uniform("test-model"),
            None,
            Arc::new(PassthroughVfs::new(&root)),
            SkillContext::default(),
            None, // embed_model
            None, // tool_caller
            root.clone(),
            Project::load(&root),
            Arc::new(Dialects::default()),
        );
        // "" binds the DEFAULT repo, whose resources were seeded from the args above.
        let session = daemon.bind_session(None, "").await.unwrap();
        let prefix = daemon.context_prefix(&session, "what is this").await;

        // The repo's own words, not a guess from filenames.
        assert!(prefix.contains("Project README (README.md)"), "{prefix}");
        assert!(prefix.contains("lock-free queue library"), "{prefix}");
        // Second level reached: the source file is one directory down.
        assert!(prefix.contains("src/atom.h"), "{prefix}");
        // Directories are distinguishable from empty files.
        assert!(prefix.contains("src/\n"), "{prefix}");
        // `.git` never appears: the VFS prunes it at the source (vfs.rs), so the tree
        // cannot descend into it and cannot mention what is inside.
        assert!(!prefix.contains(".git"), "{prefix}");
        assert!(!prefix.contains("HEAD"), "{prefix}");

        std::fs::remove_dir_all(&root).ok();
    }

    /// A repo with no README still produces a prefix, and says nothing about one.
    #[tokio::test]
    async fn prefix_without_a_readme_omits_the_section() {
        let root = std::env::temp_dir().join(format!("corrode-prefix-bare-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]").unwrap();

        let daemon = Daemon::new(
            Swarm::new(Client::new("http://127.0.0.1:1", None), 1),
            RoleModels::uniform("test-model"),
            None,
            Arc::new(PassthroughVfs::new(&root)),
            SkillContext::default(),
            None, // embed_model
            None, // tool_caller
            root.clone(),
            Project::load(&root),
            Arc::new(Dialects::default()),
        );
        // "" binds the DEFAULT repo, whose resources were seeded from the args above.
        let session = daemon.bind_session(None, "").await.unwrap();
        let prefix = daemon.context_prefix(&session, "what is this").await;
        assert!(!prefix.contains("Project README"), "{prefix}");
        assert!(prefix.contains("Cargo.toml"), "{prefix}");

        std::fs::remove_dir_all(&root).ok();
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
        let skills = SkillContext::build(&repo, &client, None, &GlobalSkills::default()).await;
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
        let skills = SkillContext::build(&repo, &client, None, &GlobalSkills::default()).await.script_dirs();
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
    /// The note store, end to end: run a real turn against a real store and check what
    /// landed.
    ///
    /// Extraction was measured on transcripts, but persistence never ran against anything
    /// — the e2e above passes `None` for the graph, so every note it reported went
    /// nowhere. This is the claim that matters: after a turn, the notes an agent produced
    /// are queryable, carry their provenance, and are attached to what the task touched.
    #[cfg(all(feature = "needle", feature = "helix"))]
    #[tokio::test]
    #[ignore = "requires a live hipfire + Needle assets + the demo-repo submodule"]
    async fn a_turn_persists_its_notes_to_the_store() {
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
        let model = std::env::var("CORRODE_MODEL")
            .ok()
            .or_else(|| models.first().cloned())
            .expect("hipfire serves at least one model");

        let dir = std::env::temp_dir().join(format!("corrode-notes-e2e-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let store: Arc<dyn crate::graph::GraphStore> =
            Arc::new(crate::graph::embedded::HelixStore::open(dir.to_str().unwrap()).unwrap());

        let embed = crate::roles::default_embedding_model(&models).map(str::to_string);
        let skills = SkillContext::build(&repo, &client, embed.clone(), &GlobalSkills::default()).await;
        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let daemon = Arc::new(Daemon::new(
            Swarm::new(client, 4),
            RoleModels::uniform(&model),
            Some(Arc::clone(&store)),
            Arc::new(PassthroughVfs::new(&repo)),
            skills,
            embed,
            Some(Arc::new(caller)),
            repo.clone(),
            Project::load(&repo),
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
        while let Ok(Some(ev)) = tokio::time::timeout(Duration::from_secs(180), erx.recv()).await {
            if let AgentEvent::ApprovalRequest { id, .. } = ev {
                let _ = ctx.send(AgentCommand::ApprovalResponse { id, approved: true }).await;
            } else if matches!(ev, AgentEvent::TurnComplete { .. }) {
                break;
            }
        }

        // What landed. `code_nodes` is the provenance view; notes are their own kinds, so
        // walk the tasks the turn created and read their notes back.
        // Notes attach to `file:{path}`, not to the provenance `code:` nodes — reading
        // the wrong node was this test's first result (0 notes, from a write that had
        // happened).
        let mut notes = Vec::new();
        for rel in ["src/lib.rs", "Cargo.toml", "README.md", "AGENTS.md"] {
            for n in store.neighbors(&format!("file:{rel}")).unwrap_or_default() {
                if n.kind == "observed" || n.kind == "asserted" {
                    notes.push((n.kind.clone(), n.label.clone()));
                }
            }
        }
        eprintln!("\npersisted notes reachable from provenance: {}", notes.len());
        for (k, t) in notes.iter().take(6) {
            eprintln!("  [{k}] {}", t.lines().next().unwrap_or("").chars().take(120).collect::<String>());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

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

        // Fully qualified: the `roles::` self-import was dropped when the
        // `is_small_model` gate went, and re-adding it would be unused in the base
        // build (this call site is feature-gated).
        let embed = crate::roles::default_embedding_model(&models).map(str::to_string);
        let skills = SkillContext::build(&repo, &client, embed.clone(), &GlobalSkills::default()).await;
        let caller = crate::toolcall::needle::NeedleToolCaller::load_from_env()
            .expect("load Needle")
            .expect("Needle assets present");
        let daemon = Arc::new(Daemon::new(
            Swarm::new(client, 4),
            RoleModels::uniform(&model),
            None,
            Arc::new(PassthroughVfs::new(&repo)),
            skills,
            embed,
            Some(Arc::new(caller)),
            repo.clone(),
            Project::load(&repo),
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
                None, // owner_token
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
