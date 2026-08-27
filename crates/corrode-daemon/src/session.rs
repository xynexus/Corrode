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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::approval::ApprovalGate;
use crate::graph::GraphStore;
use crate::skills::SkillContext;
use crate::terminal::Terminals;
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
            graph: repo.graph,
            vfs: repo.vfs,
            skills: repo.skills,
            skill_scripts: repo.skill_scripts,
            owner_token,
            key,
        }
    }
}
