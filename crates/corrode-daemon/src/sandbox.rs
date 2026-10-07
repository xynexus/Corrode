//! Optional bubblewrap confinement for the processes the daemon spawns.
//!
//! Corrode launches real OS processes from two places: the agent's tools
//! (`tools::ToolBox::run_command` / `run_skill_script`) and a human at the web
//! terminal (`terminal::Terminals`). Both run with the daemon's privileges, cwd'd
//! into the repo, reachable over an unauthenticated socket. [`Sandbox`] wraps each
//! spawn in an unprivileged `bwrap` user namespace: a per-process view with the
//! repo bound read-write, the graph store read-only, the rest of the filesystem
//! read-only, and (by default) no network.
//!
//! On by default (the user's call, 2026-10-06, once builds worked inside it);
//! `CORRODE_SANDBOX=off` opts out and `wrap` then returns the argv untouched. When on
//! but `bwrap` can't run (absent, or an unprivileged-userns restriction like Ubuntu's
//! AppArmor default), the spawn fails and the command never runs — fail closed, never
//! a silent drop to unsandboxed; `doctor` reports it.
//!
//! Phase 1 (see docs/sessions-and-sandbox.md): one process-wide `Sandbox` from
//! env. When sessions land, it becomes a per-session `SandboxProfile` bound to the
//! session's own repo, which is also how per-user filesystem isolation falls out.

use std::path::{Path, PathBuf};

/// Credential stores under the daemon user's home that no tool may reach: not the
/// file tools (the VFS, which runs in the daemon, outside any sandbox) and not a
/// sandboxed command or terminal. Matters because a client may bind `~` itself as its
/// repo. ponytail: a fixed list -- make it configurable if a deployment keeps secrets
/// somewhere else under home.
pub const PROTECTED_HOME_PATHS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".config/gh",
    ".config/gcloud",
    ".password-store",
    ".local/share/keyrings",
    ".netrc",
    ".git-credentials",
];

/// The daemon user's home directory (its real path), if `HOME` is set.
pub fn home() -> Option<PathBuf> {
    let h = PathBuf::from(std::env::var_os("HOME")?);
    Some(std::fs::canonicalize(&h).unwrap_or(h))
}

/// [`PROTECTED_HOME_PATHS`] under `home`, as real paths where they exist (a `~/.ssh`
/// that is a link elsewhere protects its target).
pub fn protected_paths(home: &Path) -> Vec<PathBuf> {
    PROTECTED_HOME_PATHS
        .iter()
        .map(|p| {
            let p = home.join(p);
            std::fs::canonicalize(&p).unwrap_or(p)
        })
        .collect()
}

/// bwrap arguments hiding every protected path that the bind of `repo` would expose:
/// an empty tmpfs over a directory, `/dev/null` over a file. Outside the repo nothing
/// is mounted, so nothing needs hiding.
fn credential_masks(repo: &Path, home: &Path) -> Vec<String> {
    let mut a = Vec::new();
    for p in protected_paths(home) {
        if !p.starts_with(repo) {
            continue;
        }
        let s = p.to_string_lossy().into_owned();
        if p.is_dir() {
            a.extend(["--tmpfs".to_string(), s]);
        } else if p.exists() {
            a.extend(["--ro-bind".to_string(), "/dev/null".to_string(), s]);
        }
    }
    a
}

/// bwrap arguments giving the sandbox the Rust toolchain, so the swarm can build and
/// test what it writes -- without them every `cargo` exited 127 while turns completed
/// as if fine. Rustup's toolchains read-only; CARGO_HOME (its bin proxies and crate
/// cache) under a throwaway overlay, because cargo writes there (its package-cache
/// lock, fetches when the net is shared) but nothing the swarm runs may alter the
/// host's cache. Absent dirs are skipped. ponytail: Rust only -- give other
/// toolchains (node, a python venv) the same when a repo wants them.
fn toolchain_binds(cargo_home: Option<PathBuf>, rustup_home: Option<PathBuf>) -> Vec<String> {
    let mut a = Vec::new();
    if let Some(p) = rustup_home.filter(|p| p.is_dir()) {
        let p = p.to_string_lossy().into_owned();
        a.extend(["--ro-bind".to_string(), p.clone(), p]);
    }
    if let Some(p) = cargo_home.filter(|p| p.is_dir()) {
        let p = p.to_string_lossy().into_owned();
        a.extend(["--overlay-src".to_string(), p.clone(), "--tmp-overlay".to_string(), p]);
    }
    a
}

/// What every spawned process gets, confined or not (review #30): build-job defaults
/// and a memory cap. Measured 2026-10-07 with three cold CAE builds beside both swarm
/// models: at cargo's default `-j32` the 27B lost 18% of its decode rate and the 35B
/// 20% -- CPU and GPU share the APU's power budget, and the GPU clock fell to ~2.35
/// GHz; at `-j8`, 11% and 3%. `nice`/`ionice` changed nothing: CPU time was never
/// short. The cap keeps a runaway build or test from taking the host's RAM, which
/// is also hipfire's.
#[derive(Clone, Default)]
pub struct SpawnPolicy {
    /// `KEY=VALUE` defaults, each only where the daemon's own env leaves it unset.
    env: Vec<String>,
    /// `MemoryMax` of each command's systemd user scope (no swap); `None` = uncapped.
    memory_max: Option<String>,
    /// Why a cap that was asked for is not applied.
    pub uncapped: Option<String>,
}

/// Build parallelism a spawned command gets unless the daemon's env sets its own.
const JOB_DEFAULTS: &[(&str, &str)] = &[("CARGO_BUILD_JOBS", "8"), ("RUST_TEST_THREADS", "8")];
/// `CORRODE_COMMAND_MEMORY_MAX` when unset: room for about three cold CAE builds.
const MEMORY_MAX_DEFAULT: &str = "16G";

impl SpawnPolicy {
    /// From the env, with the cap dropped (and the reason kept) when a systemd user
    /// scope cannot start here -- a resource cap is not a security boundary, so the
    /// command runs uncapped rather than not at all; `doctor` warns.
    pub fn from_env() -> Self {
        let mut p = Self::with(|k| std::env::var(k).ok().filter(|v| !v.is_empty()));
        if let Some(m) = p.memory_max.clone() {
            let probe = std::process::Command::new("systemd-run")
                .args(scope_argv(&m))
                .arg("true")
                .output();
            let err = match probe {
                Ok(o) if o.status.success() => None,
                Ok(o) => Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
                Err(e) => Some(e.to_string()),
            };
            if let Some(e) = err {
                p.memory_max = None;
                p.uncapped = Some(format!("a systemd user scope cannot start: {e}"));
            }
        }
        p
    }

    fn with(get: impl Fn(&str) -> Option<String>) -> Self {
        let env = JOB_DEFAULTS
            .iter()
            .filter(|(k, _)| get(k).is_none())
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let memory_max = match get("CORRODE_COMMAND_MEMORY_MAX") {
            None => Some(MEMORY_MAX_DEFAULT.to_string()),
            Some(v) if crate::knobs::parse_flag(&v) == Some(false) => None,
            Some(v) => Some(v),
        };
        Self { env, memory_max, uncapped: None }
    }

    /// One line for the daemon's log and `doctor`.
    pub fn describe(&self) -> String {
        let env = if self.env.is_empty() { "none".to_string() } else { self.env.join(" ") };
        let cap = match (&self.memory_max, &self.uncapped) {
            (Some(m), _) => format!("{m} per command (systemd user scope, no swap)"),
            (None, Some(why)) => format!("none -- {why}"),
            (None, None) => "off (CORRODE_COMMAND_MEMORY_MAX)".to_string(),
        };
        format!("env defaults {env}; memory cap {cap}")
    }
}

/// `systemd-run` arguments that start `--`-separated command in its own capped scope.
/// `--scope` execs the command in place (same pid, so the tool loop's process-group
/// kill still reaches it), and `--expand-environment=no` keeps systemd-run from
/// rewriting `$` in it -- an agent's `sh -c 'echo $$'` arrived as `echo $`.
fn scope_argv(memory_max: &str) -> Vec<String> {
    [
        "--user",
        "--scope",
        "--quiet",
        "--expand-environment=no",
        "-p",
        &format!("MemoryMax={memory_max}"),
        "-p",
        "MemorySwapMax=0",
        "--",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[derive(Clone)]
pub struct Sandbox {
    enabled: bool,
    /// Share the host network into the sandbox. Off by default; needed for tools
    /// that fetch (cargo/pip/git clone). `CORRODE_SANDBOX_NET=on`.
    share_net: bool,
    /// Applied to every spawn, confined or not.
    policy: SpawnPolicy,
}

impl Sandbox {
    /// On unless `CORRODE_SANDBOX` is set off; `CORRODE_SANDBOX_NET` on shares the host
    /// network (both through [`crate::knobs::flag`]).
    pub fn from_env() -> Self {
        let s = Self {
            enabled: crate::knobs::flag("CORRODE_SANDBOX", true),
            share_net: crate::knobs::flag("CORRODE_SANDBOX_NET", false),
            policy: SpawnPolicy::from_env(),
        };
        if s.enabled {
            eprintln!(
                "sandbox: bubblewrap confinement ON (network {})",
                if s.share_net { "shared" } else { "denied" }
            );
        }
        eprintln!("spawn policy: {}", s.policy.describe());
        s
    }

    pub fn disabled() -> Self {
        Self { enabled: false, share_net: false, policy: SpawnPolicy::default() }
    }

    pub fn policy(&self) -> &SpawnPolicy {
        &self.policy
    }

    /// Turn `argv` (program + args) into the argv actually spawned: the spawn policy's
    /// env defaults just before the command, bwrap confinement to `repo` around it when
    /// enabled, and the policy's capped scope outermost. Returns `(program, args)` so
    /// both `tokio::process::Command` and portable-pty's `CommandBuilder` can consume it.
    pub fn wrap(&self, repo: &Path, argv: &[&str]) -> (String, Vec<String>) {
        debug_assert!(!argv.is_empty(), "wrap needs at least a program");
        let mut cmd: Vec<String> = Vec::new();
        if !self.policy.env.is_empty() {
            cmd.push("env".to_string());
            cmd.extend(self.policy.env.iter().cloned());
        }
        cmd.extend(argv.iter().map(|s| s.to_string()));
        if self.enabled {
            let mut b = vec!["bwrap".to_string()];
            b.extend(self.bwrap_args(repo));
            b.push("--".to_string());
            b.extend(cmd);
            cmd = b;
        }
        if let Some(m) = &self.policy.memory_max {
            let mut s = vec!["systemd-run".to_string()];
            s.extend(scope_argv(m));
            s.extend(cmd);
            cmd = s;
        }
        let prog = cmd.remove(0);
        (prog, cmd)
    }

    /// bwrap's arguments, up to (not including) the `--` before the command.
    ///
    /// Note: we deliberately do NOT `--new-session`. It would guard against TIOCSTI
    /// keystroke injection, but it detaches the controlling tty and breaks the
    /// interactive shell's job control (`bash: cannot set terminal process group`).
    /// Modern kernels disallow TIOCSTI by default (`CONFIG_LEGACY_TIOCSTI=n`), so
    /// the guard is redundant; a paranoid deployment on an old kernel can revisit.
    fn bwrap_args(&self, repo: &Path) -> Vec<String> {
        let repo = repo.to_string_lossy().into_owned();
        let mut a: Vec<String> = Vec::new();
        let mut push = |parts: &[&str]| a.extend(parts.iter().map(|s| s.to_string()));

        // Fresh namespaces: user (the unprivileged chroot), pid, ipc, uts, cgroup,
        // and net — then optionally re-share the host net.
        push(&["--unshare-all"]);
        if self.share_net {
            push(&["--share-net"]);
        }
        push(&["--die-with-parent"]);

        // Read-only system. `-try` tolerates merged-usr layouts where /bin, /lib,
        // /lib64, /sbin are symlinks (or absent).
        push(&["--ro-bind", "/usr", "/usr"]);
        push(&["--ro-bind-try", "/bin", "/bin"]);
        push(&["--ro-bind-try", "/lib", "/lib"]);
        push(&["--ro-bind-try", "/lib64", "/lib64"]);
        push(&["--ro-bind-try", "/sbin", "/sbin"]);
        push(&["--ro-bind", "/etc", "/etc"]);
        // No /dev/dri: a GPU in the sandbox would let a command reach the hipfire
        // daemon's device.
        push(&["--proc", "/proc"]);
        push(&["--dev", "/dev"]);
        push(&["--tmpfs", "/tmp"]);

        // The working tree: read-write. The graph store lives inside it at
        // <repo>/.corrode — re-bind that read-only on top (later bind wins) so a
        // shell can edit code but not corrupt provenance/vectors. The daemon writes
        // the store directly and is never sandboxed, so it keeps full access.
        push(&["--bind", &repo, &repo]);
        let corrode = format!("{repo}/.corrode");
        push(&["--ro-bind-try", &corrode, &corrode]);
        // .git read-only too (the user's call, 2026-10-06): a writable one let a
        // sandboxed command set `core.hooksPath` or `core.fsmonitor` -- code that runs
        // at the human's next `git status`, outside any sandbox. The file tools
        // already refuse .git (`vfs::confine_write`); this closes the shell route.
        // Commands can still read history (status, diff, log) but not commit.
        let git = format!("{repo}/.git");
        push(&["--ro-bind-try", &git, &git]);
        // After the repo bind, so a repo at `~` cannot hide them (a later bind of a
        // parent covers the mounts beneath it).
        let toolchain_dir = |var: &str, dir: &str| {
            std::env::var_os(var).map(PathBuf::from).or_else(|| home().map(|h| h.join(dir)))
        };
        a.extend(toolchain_binds(
            toolchain_dir("CARGO_HOME", ".cargo"),
            toolchain_dir("RUSTUP_HOME", ".rustup"),
        ));
        // A repo that contains home's credential stores (the repo is `~` itself) must
        // not hand them to the command: mount over them, after the bind.
        if let Some(home) = home() {
            a.extend(credential_masks(Path::new(&repo), &home));
        }
        let mut push = |parts: &[&str]| a.extend(parts.iter().map(|s| s.to_string()));
        push(&["--chdir", &repo]);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // The policy reaches a command whether or not it is confined: the scope outermost,
    // bwrap inside it, the env defaults just before the command (inside bwrap, so the
    // confined process sees them).
    #[test]
    fn the_spawn_policy_wraps_every_command() {
        let policy = SpawnPolicy::with(|_| None);
        let joined = |sb: &Sandbox| {
            let (prog, args) = sb.wrap(&PathBuf::from("/repo"), &["sh", "-c", "cargo test"]);
            format!("{prog} {}", args.join(" "))
        };
        let tail = "-- env CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8 sh -c cargo test";
        let bare = joined(&Sandbox { enabled: false, share_net: false, policy: policy.clone() });
        assert!(bare.starts_with("systemd-run --user --scope"), "{bare}");
        assert!(bare.contains("--expand-environment=no"), "{bare}");
        assert!(bare.contains("-p MemoryMax=16G -p MemorySwapMax=0"), "{bare}");
        assert!(bare.ends_with(tail), "{bare}");
        let boxed = joined(&Sandbox { enabled: true, share_net: false, policy });
        let bwrap = boxed.find(" -- bwrap ").expect(&boxed);
        assert!(boxed.starts_with("systemd-run "), "{boxed}");
        assert!(boxed[bwrap..].ends_with(tail), "{boxed}");
    }

    // The daemon's own env wins over a default, and the cap can be turned off.
    #[test]
    fn the_operator_overrides_the_spawn_defaults() {
        let set = |pairs: &'static [(&'static str, &'static str)]| {
            SpawnPolicy::with(move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()))
        };
        let p = set(&[("CARGO_BUILD_JOBS", "32"), ("CORRODE_COMMAND_MEMORY_MAX", "off")]);
        assert_eq!(p.env, vec!["RUST_TEST_THREADS=8".to_string()]);
        assert_eq!(p.memory_max, None);
        let (prog, args) = Sandbox { enabled: false, share_net: false, policy: p }
            .wrap(&PathBuf::from("/repo"), &["sh"]);
        assert_eq!((prog.as_str(), args), ("env", vec!["RUST_TEST_THREADS=8".to_string(), "sh".into()]));
        assert_eq!(set(&[("CORRODE_COMMAND_MEMORY_MAX", "24G")]).memory_max.as_deref(), Some("24G"));
    }

    #[test]
    fn disabled_is_a_passthrough() {
        let sb = Sandbox::disabled();
        let (prog, args) = sb.wrap(&PathBuf::from("/repo"), &["sh", "-c", "echo hi"]);
        assert_eq!(prog, "sh");
        assert_eq!(args, vec!["-c", "echo hi"]);
    }

    #[test]
    fn enabled_binds_repo_and_appends_argv_after_dashdash() {
        let sb = Sandbox { enabled: true, share_net: false, policy: SpawnPolicy::default() };
        let (prog, args) = sb.wrap(&PathBuf::from("/home/u/proj"), &["sh", "-c", "ls"]);
        assert_eq!(prog, "bwrap");
        // repo bound read-write, its .corrode re-bound read-only, no net.
        let joined = args.join(" ");
        assert!(joined.contains("--bind /home/u/proj /home/u/proj"), "{joined}");
        assert!(joined.contains("--ro-bind-try /home/u/proj/.corrode /home/u/proj/.corrode"), "{joined}");
        // .git re-bound read-only after the repo bind (a later bind wins).
        let git = joined.find("--ro-bind-try /home/u/proj/.git /home/u/proj/.git").expect(&joined);
        assert!(joined.find("--bind /home/u/proj /home/u/proj").unwrap() < git, "{joined}");
        assert!(joined.contains("--unshare-all"), "{joined}");
        assert!(!joined.contains("--share-net"), "net denied by default: {joined}");
        // --new-session is intentionally never emitted (breaks job control).
        assert!(!joined.contains("--new-session"), "{joined}");
        // the real command survives, verbatim, after `--`.
        let dd = args.iter().position(|s| s == "--").expect("has --");
        assert_eq!(&args[dd + 1..], &["sh", "-c", "ls"]);
    }

    // With the repo at home itself, the bind would expose the credential stores, so
    // each existing one is mounted over; a repo below home exposes none.
    #[test]
    fn credential_stores_inside_the_repo_are_masked() {
        let home = std::env::temp_dir().join(format!("corrode-masks-{}", std::process::id()));
        std::fs::create_dir_all(home.join(".ssh")).unwrap();
        std::fs::create_dir_all(home.join(".config/gh")).unwrap();
        std::fs::write(home.join(".netrc"), "machine x").unwrap();
        std::fs::create_dir_all(home.join("proj")).unwrap();
        let home = std::fs::canonicalize(&home).unwrap();
        let s = |p: &str| home.join(p).to_string_lossy().into_owned();

        let a = credential_masks(&home, &home).join(" ");
        assert!(a.contains(&format!("--tmpfs {}", s(".ssh"))), "{a}");
        assert!(a.contains(&format!("--tmpfs {}", s(".config/gh"))), "{a}");
        assert!(a.contains(&format!("--ro-bind /dev/null {}", s(".netrc"))), "{a}");
        assert!(!a.contains(".aws"), "absent stores need no mask: {a}");
        assert!(credential_masks(&home.join("proj"), &home).is_empty());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn rust_toolchain_is_reachable_but_not_writable() {
        let d = std::env::temp_dir().join(format!("corrode-tc-{}", std::process::id()));
        let (cargo, rustup) = (d.join("cargo"), d.join("rustup"));
        std::fs::create_dir_all(&cargo).unwrap();
        std::fs::create_dir_all(&rustup).unwrap();
        let (c, r) = (cargo.to_string_lossy(), rustup.to_string_lossy());
        let a = toolchain_binds(Some(cargo.clone()), Some(rustup.clone())).join(" ");
        assert!(a.contains(&format!("--ro-bind {r} {r}")), "{a}");
        assert!(a.contains(&format!("--overlay-src {c} --tmp-overlay {c}")), "{a}");
        assert!(!a.contains("--bind "), "nothing writable through to the host: {a}");
        assert!(toolchain_binds(Some(d.join("absent")), None).is_empty());
        std::fs::remove_dir_all(&d).ok();
    }

    // On unless set off: the default the user chose once builds worked inside it.
    #[test]
    fn the_sandbox_is_on_unless_set_off() {
        std::env::remove_var("CORRODE_SANDBOX");
        assert!(Sandbox::from_env().enabled);
        std::env::set_var("CORRODE_SANDBOX", "off");
        assert!(!Sandbox::from_env().enabled);
        std::env::remove_var("CORRODE_SANDBOX");
    }

    #[test]
    fn net_flag_opts_in() {
        let sb = Sandbox { enabled: true, share_net: true, policy: SpawnPolicy::default() };
        let (_, args) = sb.wrap(&PathBuf::from("/r"), &["/bin/bash", "-i"]);
        assert!(args.join(" ").contains("--share-net"));
    }

    // For real, where bwrap can run: inside the sandbox git can read the repo but
    // cannot write its config (a hooksPath would run at the human's next `git status`).
    #[test]
    fn git_metadata_is_read_only_inside_the_sandbox() {
        let repo = std::env::temp_dir().join(format!("corrode-sbgit-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        let ok = |cmd: &mut std::process::Command| cmd.output().is_ok_and(|o| o.status.success());
        if !ok(std::process::Command::new("git").arg("init").arg("-q").arg(&repo)) {
            eprintln!("skipped: no git");
            return;
        }
        let sb = Sandbox { enabled: true, share_net: false, policy: SpawnPolicy::default() };
        let run = |script: &str| {
            let (prog, args) = sb.wrap(&repo, &["sh", "-c", script]);
            std::process::Command::new(prog).args(args).output()
        };
        if !run("true").is_ok_and(|o| o.status.success()) {
            eprintln!("skipped: bwrap cannot run here");
            std::fs::remove_dir_all(&repo).ok();
            return;
        }
        let out = run("git status --short >/dev/null && echo READ_OK; \
                       git config core.hooksPath /tmp/x 2>/dev/null && echo WROTE || echo REFUSED")
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("READ_OK"), "git must still read: {text}");
        assert!(text.contains("REFUSED") && !text.contains("WROTE"), "{text}");
        let config = std::fs::read_to_string(repo.join(".git/config")).unwrap();
        assert!(!config.contains("hooksPath"), "{config}");
        std::fs::remove_dir_all(&repo).ok();
    }
}
