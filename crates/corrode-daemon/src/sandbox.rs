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

use std::path::Path;

#[derive(Clone)]
pub struct Sandbox {
    enabled: bool,
    /// Share the host network into the sandbox. Off by default; needed for tools
    /// that fetch (cargo/pip/git clone). `CORRODE_SANDBOX_NET=on`.
    share_net: bool,
}

impl Sandbox {
    /// `CORRODE_SANDBOX` = `on`/`1`/`true` enables (anything else, or unset, is off).
    /// `CORRODE_SANDBOX_NET` = `on`/`1`/`true` shares the host network.
    pub fn from_env() -> Self {
        let on = |k: &str| {
            matches!(
                std::env::var(k).unwrap_or_default().to_ascii_lowercase().as_str(),
                "on" | "1" | "true" | "yes"
            )
        };
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

    #[test]
    fn net_flag_opts_in() {
        let sb = Sandbox { enabled: true, share_net: true };
        let (_, args) = sb.wrap(&PathBuf::from("/r"), &["/bin/bash", "-i"]);
        assert!(args.join(" ").contains("--share-net"));
    }
}
