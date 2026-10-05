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
//! Off by default (`CORRODE_SANDBOX` unset) so existing behaviour is unchanged —
//! `wrap` returns the argv untouched. Turn it on in a real deployment (the service
//! unit). When on but `bwrap` can't run (absent, or an unprivileged-userns
//! restriction like Ubuntu's AppArmor default), the spawn fails and the command
//! never runs — fail closed, never a silent drop to unsandboxed.
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

#[derive(Clone)]
pub struct Sandbox {
    enabled: bool,
    /// Share the host network into the sandbox. Off by default; needed for tools
    /// that fetch (cargo/pip/git clone). `CORRODE_SANDBOX_NET=on`.
    share_net: bool,
}

impl Sandbox {
    /// `CORRODE_SANDBOX` on enables; `CORRODE_SANDBOX_NET` on shares the host network
    /// (both through [`crate::knobs::flag`]).
    pub fn from_env() -> Self {
        let on = |k: &str| crate::knobs::flag(k, false);
        let s = Self { enabled: on("CORRODE_SANDBOX"), share_net: on("CORRODE_SANDBOX_NET") };
        if s.enabled {
            eprintln!(
                "sandbox: bubblewrap confinement ON (network {})",
                if s.share_net { "shared" } else { "denied" }
            );
        }
        s
    }

    pub fn disabled() -> Self {
        Self { enabled: false, share_net: false }
    }

    /// Turn `argv` (program + args) into the argv actually spawned, confined to
    /// `repo`. Disabled -> `argv` unchanged. Returns `(program, args)` so both
    /// `tokio::process::Command` and portable-pty's `CommandBuilder` can consume it.
    ///
    /// Note: we deliberately do NOT `--new-session`. It would guard against TIOCSTI
    /// keystroke injection, but it detaches the controlling tty and breaks the
    /// interactive shell's job control (`bash: cannot set terminal process group`).
    /// Modern kernels disallow TIOCSTI by default (`CONFIG_LEGACY_TIOCSTI=n`), so
    /// the guard is redundant; a paranoid deployment on an old kernel can revisit.
    pub fn wrap(&self, repo: &Path, argv: &[&str]) -> (String, Vec<String>) {
        debug_assert!(!argv.is_empty(), "wrap needs at least a program");
        if !self.enabled {
            return (argv[0].to_string(), argv[1..].iter().map(|s| s.to_string()).collect());
        }

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

        push(&["--"]);
        a.extend(argv.iter().map(|s| s.to_string()));
        ("bwrap".to_string(), a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn disabled_is_a_passthrough() {
        let sb = Sandbox::disabled();
        let (prog, args) = sb.wrap(&PathBuf::from("/repo"), &["sh", "-c", "echo hi"]);
        assert_eq!(prog, "sh");
        assert_eq!(args, vec!["-c", "echo hi"]);
    }

    #[test]
    fn enabled_binds_repo_and_appends_argv_after_dashdash() {
        let sb = Sandbox { enabled: true, share_net: false };
        let (prog, args) = sb.wrap(&PathBuf::from("/home/u/proj"), &["sh", "-c", "ls"]);
        assert_eq!(prog, "bwrap");
        // repo bound read-write, its .corrode re-bound read-only, no net.
        let joined = args.join(" ");
        assert!(joined.contains("--bind /home/u/proj /home/u/proj"), "{joined}");
        assert!(joined.contains("--ro-bind-try /home/u/proj/.corrode /home/u/proj/.corrode"), "{joined}");
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

    #[test]
    fn net_flag_opts_in() {
        let sb = Sandbox { enabled: true, share_net: true };
        let (_, args) = sb.wrap(&PathBuf::from("/r"), &["/bin/bash", "-i"]);
        assert!(args.join(" ").contains("--share-net"));
    }
}
