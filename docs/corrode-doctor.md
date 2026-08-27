# `corrode doctor` — host readiness checks

**Implemented** as `corrode-daemon doctor` (see `doctor.rs`): runs the runtime
checks below and exits non-zero on any fatal. This doc is the spec + the fuller
catalog (build-time checks a future version can add).

The subcommand inspects the host and reports
what's missing for building and running Corrode, each check with a detection
command and a remediation. Ordered by concern. A check is **fatal** (Corrode
won't run / won't build in that mode), **degraded** (works but a capability is
off), or **info**.

Exit non-zero if any fatal check fails. Print remediation inline.

---

## Changes already made to this host (2026-08-27)

These were applied while bringing the sandbox up; `doctor` should treat them as the
expected baseline and detect their absence.

| Change | Why | How to reproduce |
|---|---|---|
| `apt install pkg-config` | `--features helix` links OpenSSL via `native-tls`, needs pkg-config for discovery | `sudo apt install pkg-config` |
| `apt install bubblewrap` | the process sandbox (`CORRODE_SANDBOX=on`) shells out to `bwrap` | `sudo apt install bubblewrap` |
| AppArmor profile `/etc/apparmor.d/bwrap` | Ubuntu 24.04 sets `kernel.apparmor_restrict_unprivileged_userns=1`; without a profile granting `userns`, unprivileged `bwrap` fails at `setting up uid map: Permission denied` | see [§ Sandbox](#3-sandbox-corrode_sandbonon) below |

---

## 1. Build prerequisites

### `cargo` / Rust toolchain — fatal
- **Detect:** `cargo --version`; MSRV for `--features helix` is rustc ≥ 1.88 (the `ort` floor).
- **Fix:** `rustup update`.

### OpenSSL + pkg-config (only for `--features helix`) — fatal-in-mode
- **Detect:** `pkg-config --modversion openssl` succeeds.
- **Fix:** `sudo apt install libssl-dev pkg-config`. Fallback on a host with
  `libssl-dev` but no pkg-config: build with
  `OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include`.

### Submodules present — fatal-in-mode
- **Detect:** `third_party/helix-db/helix-db/Cargo.toml` and
  `third_party/docling.rs/crates/docling/Cargo.toml` exist.
- **Fix:** `git submodule update --init third_party/helix-db third_party/docling.rs`.

---

## 2. Runtime

### hipfire reachable — fatal
- **Detect:** `GET {HIPFIRE_BASE_URL:-http://127.0.0.1:11435}/v1/models` returns 200
  with ≥1 model. (Corrode falls back to `CORRODE_MODEL` for all roles if it can't
  reach hipfire, so treat unreachable as **degraded** if `CORRODE_MODEL` is set,
  else fatal.)
- **Fix:** `hipfire start` (not just `serve` — `serve` is only the HTTP frontend).

### An embedding model is served — degraded
- **Detect:** any `/v1/models` id contains `embed` (matches Corrode's
  `roles::default_embedding_model` heuristic).
- **Impact if missing:** skill ranking and doc-GraphRAG fall back to non-vector
  paths (BM25 for `DocQuery`).

### Repo writable & graph dir — info
- **Detect:** `CORRODE_REPO` (default `.`) exists and is writable; the store dir
  `<repo>/.corrode/graph` (or `CORRODE_GRAPH_DIR`) is creatable. Note LMDB reserves
  a ~10 GB map per store.

---

## 3. Sandbox (`CORRODE_SANDBOX=on`)

Only checked when the sandbox is enabled. All fatal **in that mode** — Corrode
fails closed (a spawn that can't be sandboxed doesn't run unsandboxed).

### `bwrap` present — fatal-in-mode
- **Detect:** `bwrap --version`.
- **Fix:** `sudo apt install bubblewrap`.

### Unprivileged user namespaces usable — fatal-in-mode
This is the subtle one. Two layers must both allow it:
- **userns enabled:** `sysctl kernel.unprivileged_userns_clone` is `1` (or the knob
  is absent, meaning always-on), and `/proc/sys/user/max_user_namespaces` > 0.
- **AppArmor not blocking it:** on Ubuntu 23.10+ `kernel.apparmor_restrict_unprivileged_userns=1`
  requires the calling binary to have an AppArmor profile granting `userns`.
- **The real test — just try it:** the reliable check is to *run* bwrap, because the
  failure modes above are hard to enumerate:
  ```sh
  bwrap --unshare-all --die-with-parent --ro-bind /usr /usr --proc /proc \
        --dev /dev --tmpfs /tmp -- /bin/true
  ```
  Exit 0 = usable. `setting up uid map: Permission denied` = AppArmor/userns
  restriction. `bwrap: ... loopback: ...Operation not permitted` = same root cause,
  surfaced during net-namespace setup.
- **Fix (preferred — keeps the host restriction on for everything else):** install a
  profile that grants `bwrap` the `userns` permission:
  ```
  # /etc/apparmor.d/bwrap
  abi <abi/4.0>,
  include <tunables/global>
  profile bwrap /usr/bin/bwrap flags=(unconfined) {
    userns,
    include if exists <local/bwrap>
  }
  ```
  then `sudo apparmor_parser -r /etc/apparmor.d/bwrap`.
- **Fix (blunt, not recommended):** `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`
  drops the restriction host-wide.

### Confinement actually confines — info
A deeper self-test `doctor` can run when bwrap is usable: spawn a probe in a temp
repo and assert the shape holds.
- repo path is writable inside the sandbox;
- `<repo>/.corrode` is read-only (`touch` fails "Read-only file system");
- no network interfaces are visible under `--unshare-all` (unless `CORRODE_SANDBOX_NET=on`);
- cwd is the repo, uid is the invoking user.

### Job control note — info
The interactive terminal deliberately does **not** pass `bwrap --new-session`: it
would break shell job control (`bash: cannot set terminal process group`). The
TIOCSTI injection it guards against is already disallowed by modern kernels
(`CONFIG_LEGACY_TIOCSTI=n`; the knob `dev.tty.legacy_tiocsti_restore` is absent).
`doctor` can note if the kernel *does* expose that knob set to `1` (legacy TIOCSTI
re-enabled), which would be the only case where dropping `--new-session` weakens
the guard.

---

## 4b. Auth table (`CORRODE_USERS`) — info / warn

Only when set. `CORRODE_USERS` points to a JSON `user -> {token, hipfire_token?}`
table; its presence turns on auth (connections must `Authenticate` before any
repo-scoped command).
- **Detect:** the file exists and parses as the expected shape.
- **Warn if unparseable:** the daemon logs `CORRODE_USERS parse failed … auth
  disabled` and runs *anonymous* — a silent downgrade from "locked" to "open",
  worth flagging loudly.
- **Info:** per-user fairness only activates for users whose entry has a
  `hipfire_token`; without one, that user shares the daemon's hipfire principal.

## Environment knobs `doctor` should echo

`CORRODE_SANDBOX` (off), `CORRODE_SANDBOX_NET` (off), `CORRODE_USERS` (unset =
auth off), `HIPFIRE_BASE_URL`, `CORRODE_MODEL`, `CORRODE_REPO`,
`CORRODE_GRAPH_DIR`, `CORRODE_DOC_ROOTS`, `CORRODE_DAEMON_ADDR`, `CORRODE_WEB_ADDR`,
`CORRODE_DAEMON_URL`. Full list and defaults in `CLAUDE.md` § Commands.
