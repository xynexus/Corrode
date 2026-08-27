# Corrode: Sessions & the Sandbox

**Design note — for decision.** Making the Corrode daemon multi-tenant with
per-session repositories, and confining every shell it spawns (the agent's and
the human's) inside a bubblewrap namespace.

- **Status:** all three phases landed (Phase 1 sandbox; Phase 2 sessions +
  `SelectRepo` + per-tab terminals; Phase 3 auth + per-user keying + hipfire
  owner token)
- **Date:** 2026-08-27
- **Scope:** tenancy model + process sandbox
- **Decided:** tenancy key = **per-authenticated-user** (Option B). Ship
  per-connection first to prove the `Session` extraction, then re-key by user.
- **Web copy:** an artifact version of this note exists (published from the
  authoring session).

---

## 1. Where we are — one daemon, one repo, no walls

Corrode is **single-tenant by construction**. Every WebSocket lands on the same
shared `Arc<Daemon>` (`server.rs`, `handle_socket`), and the working directory is
fixed for the life of the process by `CORRODE_REPO`, read once at boot
(`main.rs:44`). There is no command to change it — `AgentCommand` has no
`SelectRepo`.

One thing is already isolated correctly: **event routing**. Each connection gets
its own command/event channel pair, so a prompt's output streams back only to its
originating socket. Everything *else* is global:

| Already per-connection | Global — and it shouldn't be |
|---|---|
| `cmd_rx` / `ev_tx` (one pair per socket) | `repo_root`, `graph`, `vfs`, `skills` — bound once at `Daemon::new` |
| | the terminal session id: the webui hardcodes `SESSION = "web"` (`app.rs:13`), so two tabs drive the **same pty** |

> ⚠ **Prerequisite, not an add-on.** There is **no identity or auth** anywhere.
> `corrode-web` proxies `/agent` unauthenticated over `0.0.0.0:8787` — anyone on
> the LAN has full access today. "Multiple users" has no meaning without
> something to isolate *by*, so auth is a hard precondition for the tenancy work
> (§3), though not for the sandbox (§5).

---

## 2. The tenancy decision — what keys a session?

"Switch the repo" and "support multiple users" touch the same code but are not the
same feature. A single *global* repo switch is actively worse for multi-user — one
person changing the repo yanks it out from under everyone. The real question is
the **tenancy key**: what owns a repo context.

**Option A — per-connection.** Every WebSocket is its own world; a repo is bound
to the socket. Simplest, no auth required to prototype, but a reload drops the
session and one human's tabs become confusing duplicate worlds.

**Option B — per-authenticated-user (chosen).** A session is keyed by identity. A
user's tabs share their repo, graph, and terminals; the key doubles as the
fairness token handed to hipfire and as the sandbox's filesystem boundary. Needs
auth up front, but it's the only model where the owner key, the graph store, and
the sandbox mount all agree on "who."

These aren't mutually exclusive in time: **ship per-connection first** to prove the
`Session` extraction, then add auth and re-key by user. The struct work in §3 is
identical either way — only the map's key type changes.

---

## 3. Extracting a `Session` from the `Daemon`

The shared, stateless machinery stays on `Daemon`; everything bound to a repo
moves into a `Session`, and the daemon holds a registry of them. Command handlers
stop reaching into `&self` for world-state and take a `&Session` instead.

**Moves into `Session`:**

- `repo_root: PathBuf` — cwd for shells, `run_command`, doc-ingest roots
- `graph: Option<Arc<dyn GraphStore>>` — the per-repo LMDB store at `<repo>/.corrode/graph`
- `vfs: Arc<dyn Vfs>` — `PassthroughVfs::new(repo_root)`
- `skills` + `skill_scripts` — discovered from the repo's `.agents`/`.corrode` dirs
- `terminals: Terminals` — ptys already cwd into the repo; ids must become unique per user
- `approvals: ApprovalGate` — route a mutating-call prompt back to the session that raised it

**Stays shared on `Daemon`:**

- `swarm: Swarm` — the hipfire client + inflight semaphore; one pool, hipfire orders the work
- `roles: RoleModels` — role→model map, resolved once at boot
- `tool_caller`, `dialects` — stateless, model-scoped, not repo-scoped
- `next_plan_id` — a process-wide counter is fine
- **NEW** `sessions: Mutex<HashMap<Key, Arc<Session>>>` — the registry

```rust
// Daemon keeps only the shared, stateless machinery + a session registry.
pub struct Daemon {
    swarm: Swarm,
    roles: RoleModels,
    tool_caller: Option<Arc<dyn ToolCaller>>,
    dialects: Arc<Dialects>,
    sessions: Mutex<HashMap<SessionKey, Arc<Session>>>,   // NEW
}

// Everything a repo owns lives here, one per tenant.
pub struct Session {
    key: SessionKey,               // == the owner key sent to hipfire
    repo_root: PathBuf,
    graph: Option<Arc<dyn GraphStore>>,
    vfs: Arc<dyn Vfs>,
    skills: SkillContext,
    terminals: Terminals,
    approvals: ApprovalGate,
    sandbox: SandboxProfile,       // NEW — §5
}
```

The dispatch loop gains one step. A `SelectRepo { path }` (Option A) or the auth
handshake (Option B) resolves — or lazily opens — the connection's session;
repo-scoped commands run against it, shared commands stay on `&self`:

```rust
match cmd {
    AgentCommand::SelectRepo { path } => self.bind_session(conn, path)?,   // NEW
    // repo-scoped -> the connection's Session
    AgentCommand::Prompt { .. }
    | AgentCommand::DocIngest { .. }
    | AgentCommand::TerminalInput { .. } => session.handle(cmd, events).await,
    // shared -> stays on the Daemon
    AgentCommand::ApprovalResponse { .. } => self.approvals_for(conn).resolve(..),
}
```

> ◆ **Why this is the load-bearing decision.** Opening a HelixDB store, building a
> VFS, discovering skills, and spawning ptys are all done once in `Daemon::new`
> against plain (non-`Arc`, non-lock) fields. Making them per-session is the moment
> they need interior mutability and a lifecycle (open on bind, close on
> last-connection-drop). That plumbing is the bulk of the work — the field list is
> small, but every `&self.graph` / `&self.vfs` call site moves to `&session.*`.

---

## 4. Fairness to hipfire — via the bearer token, not metadata

The scheduling side of multi-tenancy is already built downstream: hipfire does
admission control with **per-owner fairness keys**. But the earlier assumption that
Corrode could set a `hipfire_owner` *metadata* field was wrong — reading the hipfire
source (`hipfire-server/src/routes/responses.rs`, `chat.rs::scheduler_owner_from_principal`)
shows the fairness owner is derived **entirely from the authenticated principal**
(`principal.user_id` + `token_id`), i.e. from the **bearer token**. There is no
request-metadata field for it; only `hipfire_priority` is read from metadata.

So per-user fairness needs a **per-user hipfire token**, not a metadata key. What
landed:

- `hipfire::Client::respond`/`respond_full` gained an `owner_token: Option<&str>`
  that overrides the shared API key as the bearer for that one call.
- The token is carried per-session (`Session.owner_token`), sourced from the
  `hipfire_token` field of the user's `CORRODE_USERS` entry, and threaded to the
  generation calls via `swarm::Task.owner_token` (planning) and `ToolBox` (the tool
  loops, which already hold it — no long-signature churn).
- With a per-user `hipfire_token`, that user's swarm authenticates to hipfire as a
  distinct principal → its own fair share. Without one (or with auth off), all
  requests use the daemon's shared key exactly as before.

Embeddings (`/v1/embeddings`) carry no scheduler metadata and are left on the shared
key. ponytail: activating per-user fairness requires the operator to provision
per-user hipfire tokens; until then the threading is a clean no-op.

---

## 5. The bubblewrap sandbox

Corrode spawns real OS processes from two sources: the agent's tools and a human
at the web terminal. Both currently run with the daemon's full privileges, cwd'd
into the repo, over an unauthenticated socket. **bubblewrap** (`bwrap`) wraps each
spawn in an unprivileged user namespace — a per-process chroot with its own mount,
PID, and network view. It is defense-in-depth today and, once sessions exist, the
mechanism that makes one user's files unreachable from another's shell.

### Host status (checked 2026-08-27)

- `bubblewrap` **installed** (0.9.0).
- `kernel.unprivileged_userns_clone = 1` (userns enabled), `max_user_namespaces`
  ample.
- `kernel.apparmor_restrict_unprivileged_userns = 1` (Ubuntu 24.04) initially
  blocked unprivileged bwrap (`setting up uid map: Permission denied`). **Resolved**
  by loading an AppArmor profile granting bwrap the `userns` permission — the
  preferred fix, which keeps the host restriction on for everything else:

  ```
  # /etc/apparmor.d/bwrap
  abi <abi/4.0>,
  include <tunables/global>
  profile bwrap /usr/bin/bwrap flags=(unconfined) {
    userns,
    include if exists <local/bwrap>
  }
  ```
  then `sudo apparmor_parser -r /etc/apparmor.d/bwrap`. Verified: an unprivileged
  bwrap now runs, the repo is writable, `.corrode` is read-only, and the network is
  denied. This is a host security-policy change; `corrode doctor` detects its
  absence (see `docs/corrode-doctor.md`).

### Three spawn sites, one wrapper

Every process the daemon launches goes through exactly three call sites. Each
prepends the same `bwrap` argv built from the session's `SandboxProfile`:

- `ToolBox::run_command` — `sh -c <cmd>`, cwd `self.root` (`tools.rs:279`)
- `ToolBox::run_skill_script` — interpreter or script, cwd `self.root` (`tools.rs:335`)
- `Terminals::ensure` — the interactive `bash -i` pty (`terminal.rs`). The pty
  master stays with the daemon; bwrap runs the shell inside the namespace and
  passes the tty through.

```
bwrap \
  --unshare-all \                 # no network by default; --share-net is opt-in per session
  --die-with-parent \             # the sandbox dies with the daemon
  --new-session \                 # pty only: blocks TIOCSTI keystroke injection
  --ro-bind /usr /usr  --ro-bind-try /bin /bin  --ro-bind-try /lib /lib  --ro-bind-try /lib64 /lib64 \
  --ro-bind /etc /etc \
  --proc /proc  --dev /dev  --tmpfs /tmp \
  --bind        <repo> <repo> \                    # the working tree: read-write
  --ro-bind-try <repo>/.corrode <repo>/.corrode \  # graph store: read-only to shells
  --chdir <repo> \
  -- <argv…>                                       # sh -c … / the interpreter / bash -i
```

> ◆ **Protecting the store from the shell.** The HelixDB store lives *inside* the
> repo at `<repo>/.corrode/graph`. Bind the working tree read-write, then re-bind
> `.corrode` read-only on top — a later bwrap bind wins for that path. The daemon
> writes the graph directly (it is never sandboxed), so it keeps full access; a
> shell can edit code but cannot corrupt provenance or vectors.

### Configuration & failure mode

- `CORRODE_SANDBOX` = `off` (default) | `on`, plus a per-session profile (extra
  binds, net policy). Default off preserves current behavior; a real deployment
  sets it on in the service unit.
- **Network is deny-by-default.** That breaks tools that fetch — `cargo`, `pip`,
  `git clone`. Make `--share-net` an explicit per-session opt-in, or gate it
  through the approval prompt.
- **Env is a choice.** Shells inherit the daemon's env today (PATH, venv). Inside
  bwrap decide between passthrough and `--clearenv` + an allow-list.
- **Fail closed in service mode.** If `bwrap` is missing or userns is disabled:
  hard-refuse to spawn when running as a service; warn-and-degrade only in an
  explicit dev mode.

> ✓ **It also does the multi-user isolation.** Once sessions carry their own
> `repo_root`, each session's bwrap binds only *its* repo. Filesystem isolation
> between users falls out of the sandbox for free.

---

## 6. Rollout, in phases

Ordered by value-per-risk; each is independently deployable.

1. **Sandbox the shells.** ✅ *Landed.* `sandbox.rs` wraps the three spawn sites,
   gated by `CORRODE_SANDBOX` (default off). No tenancy change, no auth — pure
   hardening of the LAN-exposed surface. Verified end-to-end: the web terminal runs
   confined (repo rw, `.corrode` ro, no network, job control intact).
2. **Extract `Session`, per-connection.** ✅ *Landed.* Repo-scoped state moved into
   `Session`/`RepoResources`, a registry + `SelectRepo` added, the hardcoded `"web"`
   terminal id replaced by a per-tab `sessionStorage` id (stable across reloads, so
   adoption still works). Lazy default binding to `CORRODE_REPO` keeps single-tenant
   behaviour. Delivers "run as a service, pick the repo at connect".
3. **Auth + per-user keying.** ✅ *Landed.* Auth is an `Authenticate` first-frame
   validated by the daemon against `CORRODE_USERS` (corrode-web stays a dumb proxy);
   sessions are keyed by `(user, repo)`; the per-user hipfire token threads to the
   generation path (§4). Verified end-to-end: pre-auth commands get `AuthRequired`,
   a valid token unlocks the session, `SelectRepo` switches repos.

Orthogonal: a `systemd` unit (`After=hipfire`, sandbox env set) turns "runs as a
service" into config rather than code — worth doing alongside Phase 1.

---

## 7. Risks & open questions

| Severity | Item |
|---|---|
| **Blocker** | **AppArmor blocks unprivileged bwrap** here (§5 host status). Load a bwrap `userns` profile before Phase 1 can run end-to-end. |
| **Blocker** | **No auth exists.** Phase 3 needs an identity layer that isn't there yet; `corrode-web` (it terminates the browser socket) is the natural home. |
| Friction | **Net-deny breaks builds.** A default-no-network sandbox surprises anyone whose agent runs `cargo build` / `pip install`. Ship the `--share-net` opt-in in the same change. |
| Friction | **Approval routing under concurrency.** The approval gate is keyed by a process-global id; with concurrent sessions, confirm a prompt and its response can't cross wires. Per-session gates are the clean fix. |
| Confirm | **hipfire owner-key field.** §4 assumes a `hipfire_owner` metadata key; verify the exact name against hipfire's admission API. |
| Confirm | **Session lifecycle.** When does a session's HelixDB store close — last connection drops, idle timeout, LRU cap? LMDB opens a ~10 GB map per store, so a per-user fleet needs a bound. |
