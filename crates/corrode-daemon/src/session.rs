//! Per-tenant session state — the multi-tenancy seam (docs/sessions-and-sandbox.md).
//!
//! Two tiers, because two kinds of "per-repo" state have different sharing rules:
//!
//! - [`RepoResources`] is keyed by canonical repo path and shared across *every*
//!   user working that repo. The HelixDB store is LMDB, which can't open the same
//!   path twice in one process, so the graph (and the VFS + skill index, which are
//!   just repo-derived) must be shared here, not duplicated per user.
//! - [`Session`] is keyed by `(user, repo)` and owns the *live, private* state: the
//!   pty terminals and the approval gate. A user's tabs on the same repo share one
//!   Session (so a reload adopts the running shell); different users get their own.
//!
//! The connection binds to a `Session` in the command loop; the shared `Daemon`
//! keeps a registry of both tiers and hands out `Arc<Session>`.

use corrode_core::AgentEvent;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::approval::ApprovalGate;
use crate::graph::GraphStore;
use crate::skills::SkillContext;
use crate::terminal::Terminals;
use crate::project::Project;
use crate::vfs::Vfs;

/// Identity of a tenant session: the authenticated user (`""` when auth is off)
/// and the canonical repo path. Two tabs of the same user on the same repo share
/// a session; different users, or the same user on a different repo, do not.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SessionKey {
    pub user: String,
    pub repo: PathBuf,
}

/// Repo-derived resources shared across all sessions on a repo (see module docs).
/// Cheap to clone — everything is behind `Arc`.
#[derive(Clone)]
pub struct RepoResources {
    pub repo_root: PathBuf,
    /// The repository's identity and policy (`.corrode/project.json`). Repo-derived,
    /// so it lives here rather than on the daemon: one daemon serves many repos, and
    /// "which project is this" is exactly a per-repo answer.
    pub project: Arc<Project>,
    /// Embedded HelixDB for this repo (`None` without `--features helix`).
    pub graph: Option<Arc<dyn GraphStore>>,
    pub vfs: Arc<dyn Vfs>,
    pub skills: Arc<SkillContext>,
    /// Skill name -> dir, derived from `skills`, for `run_skill_script`.
    pub skill_scripts: Arc<HashMap<String, PathBuf>>,
}

/// One tenant's working context. Holds Arc clones of the repo's shared resources
/// plus its own live terminals + approval gate.
pub struct Session {
    pub key: SessionKey,
    pub repo_root: PathBuf,
    /// This repo's identity + global-skill policy (shared with `RepoResources`).
    pub project: Arc<Project>,
    pub graph: Option<Arc<dyn GraphStore>>,
    pub vfs: Arc<dyn Vfs>,
    pub skills: Arc<SkillContext>,
    pub skill_scripts: Arc<HashMap<String, PathBuf>>,
    /// Live pty sessions, private to this (user, repo). The shell cwd is the repo,
    /// and (when enabled) bwrap confines it to the repo.
    pub terminals: Terminals,
    /// Human-in-the-loop gate for this session's mutating tool calls. Per-session
    /// so one tenant's `ApprovalResponse` can't resolve another's pending call.
    pub approvals: Arc<ApprovalGate>,
    /// Per-user hipfire bearer token for this session's generation calls (fairness).
    /// `None` => the daemon's shared key (all tenants share one fair share).
    pub owner_token: Option<String>,
    /// Running Prompt turns, by plan id: their cancel switches (per-session, so a
    /// tenant can cancel only its own) and what `ListTurns` reports.
    pub turns: std::sync::Mutex<HashMap<String, TurnHandle>>,
    /// Every event this session's turns emit, for whoever is attached (see [`TurnFeed`]).
    pub feed: Arc<TurnFeed>,
}

/// A running Prompt turn.
pub struct TurnHandle {
    pub cancel: tokio::sync::watch::Sender<bool>,
    pub prompt: String,
    /// Unix seconds.
    pub started: u64,
}

/// Recent events kept for a connection that (re)attaches. Streamed deltas are left
/// out: the `SubagentOutput` that follows carries the full text.
const FEED_RING: usize = 4096;

/// The session's turn events. Each turn publishes through [`TurnFeed::turn_sender`];
/// every connection bound to the session [`TurnFeed::attach`]es, which hands it the
/// recent events and then the live ones. Publishing and attaching take the same
/// lock, so a connection sees each event exactly once, in order: everything before
/// its attach from the ring, everything after from the broadcast.
///
/// A turn used to send to the socket that started it, and when that socket closed
/// -- a tab reload, a laptop asleep, a proxy restart -- every later answer and the
/// review verdict were dropped while the GPU work carried on.
pub struct TurnFeed {
    live: tokio::sync::broadcast::Sender<AgentEvent>,
    ring: std::sync::Mutex<std::collections::VecDeque<AgentEvent>>,
}

impl Default for TurnFeed {
    fn default() -> Self {
        Self {
            live: tokio::sync::broadcast::channel(1024).0,
            ring: Default::default(),
        }
    }
}

impl TurnFeed {
    pub fn publish(&self, ev: AgentEvent) {
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let delta = matches!(&ev, AgentEvent::Turn { event, .. }
            if matches!(**event, AgentEvent::SubagentDelta { .. }));
        if !delta {
            ring.push_back(ev.clone());
            if ring.len() > FEED_RING {
                ring.pop_front();
            }
        }
        let _ = self.live.send(ev); // no receivers: nobody attached right now
    }

    /// The recent events, then a receiver for everything published after them.
    pub fn attach(&self) -> (Vec<AgentEvent>, tokio::sync::broadcast::Receiver<AgentEvent>) {
        let ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        (ring.iter().cloned().collect(), self.live.subscribe())
    }

    /// A sender for one turn: what goes in comes out of the feed wrapped as
    /// `AgentEvent::Turn { plan_id, .. }`, in order. The pump ends with the last
    /// sender.
    pub fn turn_sender(self: &Arc<Self>, plan_id: &str) -> mpsc::Sender<AgentEvent> {
        let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
        let (feed, plan_id) = (Arc::clone(self), plan_id.to_string());
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                feed.publish(AgentEvent::Turn { plan_id: plan_id.clone(), event: Box::new(ev) });
            }
        });
        tx
    }
}

/// The repo's turn journal: one JSON line per finished Prompt turn (its prompt,
/// outcome, and each task's output and the files it wrote), so a turn's results
/// outlive every connection and the daemon itself.
pub fn journal_path(repo_root: &std::path::Path) -> PathBuf {
    repo_root.join(".corrode").join("turns.jsonl")
}

/// Append `record` to the repo's turn journal. Best-effort: a turn's results must
/// not fail because the journal could not be written.
pub fn journal_append(repo_root: &std::path::Path, record: &serde_json::Value) {
    use std::io::Write;
    let path = journal_path(repo_root);
    let written = path.parent().map(std::fs::create_dir_all).transpose().and_then(|_| {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{record}")
    });
    if let Err(e) = written {
        eprintln!("turn journal: could not append to {}: {e}", path.display());
    }
}

/// The last `n` journal records, newest first. Unreadable lines are skipped.
pub fn journal_tail(repo_root: &std::path::Path, n: usize) -> Vec<serde_json::Value> {
    let text = std::fs::read_to_string(journal_path(repo_root)).unwrap_or_default();
    text.lines()
        .rev()
        .filter_map(|l| serde_json::from_str(l).ok())
        .take(n)
        .collect()
}

impl Session {
    /// Build a session for `key` over already-opened repo resources, with fresh
    /// live state (terminals sandboxed by `sandbox`, a private approval gate) and
    /// the user's hipfire token for fairness attribution.
    pub fn new(
        key: SessionKey,
        repo: RepoResources,
        sandbox: crate::sandbox::Sandbox,
        owner_token: Option<String>,
    ) -> Self {
        Self {
            terminals: Terminals::new(repo.repo_root.clone()).with_sandbox(sandbox),
            approvals: Arc::new(ApprovalGate::from_env()),
            repo_root: repo.repo_root,
            project: repo.project,
            graph: repo.graph,
            vfs: repo.vfs,
            skills: repo.skills,
            skill_scripts: repo.skill_scripts,
            owner_token,
            key,
            turns: Default::default(),
            feed: Default::default(),
        }
    }
}
