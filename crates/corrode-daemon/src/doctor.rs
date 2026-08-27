//! `corrode-daemon doctor` — host readiness checks (spec: docs/corrode-doctor.md).
//!
//! Runtime diagnostics only: is hipfire reachable, can bwrap actually sandbox, is
//! the auth table valid, is the repo present, are the feature submodules there. It
//! never generates — safe to run anytime — and returns whether all FATAL checks
//! passed so the CLI can set an exit code.

use crate::hipfire::{Client, DEFAULT_BASE_URL};
use crate::roles;
use std::path::Path;
use std::process::Command;

fn ok(s: &str) {
    println!("  [ok]   {s}");
}
fn info(s: &str) {
    println!("  [info] {s}");
}
fn warn(s: &str) {
    println!("  [warn] {s}");
}
fn fail(s: &str, fix: &str) {
    println!("  [FAIL] {s}\n         fix: {fix}");
}

/// Run every check, print a report, and return `true` iff no FATAL check failed.
pub async fn run() -> bool {
    println!("corrode doctor\n");
    let mut fatal = 0u32;
    let has_fallback = std::env::var("CORRODE_MODEL").is_ok();

    // --- hipfire ---
    let base = std::env::var("HIPFIRE_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let client = Client::new(base.clone(), std::env::var("HIPFIRE_API_KEY").ok());
    match client.list_models().await {
        Ok(models) if !models.is_empty() => {
            ok(&format!("hipfire: {} model(s) at {base}", models.len()));
            match roles::default_embedding_model(&models) {
                Some(m) => ok(&format!("embedding model served: {m}")),
                None => warn(
                    "no embedding model served — DocQuery/skills fall back to BM25/manifest",
                ),
            }
        }
        Ok(_) if has_fallback => warn("hipfire served 0 models; using CORRODE_MODEL fallback"),
        Ok(_) => {
            fatal += 1;
            fail("hipfire served 0 models and no CORRODE_MODEL set", "hipfire start");
        }
        Err(_) if has_fallback => warn(&format!(
            "hipfire unreachable at {base}; CORRODE_MODEL fallback is set"
        )),
        Err(e) => {
            fatal += 1;
            fail(
                &format!("hipfire unreachable at {base} ({e})"),
                "hipfire start (not just `serve`)",
            );
        }
    }

    // --- sandbox (only meaningful when enabled) ---
    let sb = std::env::var("CORRODE_SANDBOX").unwrap_or_default().to_ascii_lowercase();
    if matches!(sb.as_str(), "on" | "1" | "true" | "yes") {
        match bwrap_usable() {
            Ok(()) => ok("sandbox: bwrap confines (repo rw, .corrode ro, no net)"),
            Err(e) => {
                fatal += 1;
                fail(
                    &format!("CORRODE_SANDBOX=on but bwrap is unusable: {e}"),
                    "apt install bubblewrap; on Ubuntu load an AppArmor userns profile \
                     for /usr/bin/bwrap (docs/corrode-doctor.md §3)",
                );
            }
        }
    } else {
        info("sandbox: off (set CORRODE_SANDBOX=on to confine spawned processes)");
    }

    // --- auth table ---
    match std::env::var("CORRODE_USERS") {
        Ok(path) => match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                Ok(v) if v.is_object() => {
                    ok(&format!("auth: on, {} user(s) configured", v.as_object().unwrap().len()))
                }
                _ => {
                    fatal += 1;
                    fail(
                        &format!("CORRODE_USERS at {path} is not a JSON object"),
                        "expected {\"alice\": {\"token\": \"…\", \"hipfire_token\": \"…\"}}",
                    );
                }
            },
            Err(e) => {
                fatal += 1;
                fail(
                    &format!("CORRODE_USERS unreadable at {path}: {e}"),
                    "the daemon degrades to ANONYMOUS (auth silently off) on this error",
                );
            }
        },
        Err(_) => info("auth: off (no CORRODE_USERS) — connections are anonymous"),
    }

    // --- repo ---
    let repo = std::env::var("CORRODE_REPO").unwrap_or_else(|_| ".".into());
    if Path::new(&repo).is_dir() {
        ok(&format!("repo: {repo}"));
    } else {
        fatal += 1;
        fail(&format!("CORRODE_REPO is not a directory: {repo}"), "point CORRODE_REPO at a repo");
    }

    // --- feature submodules (for building --features helix / docling) ---
    for (feat, path) in [
        ("helix", "third_party/helix-db/helix-db/Cargo.toml"),
        ("docling", "third_party/docling.rs/crates/docling/Cargo.toml"),
    ] {
        if Path::new(path).exists() {
            ok(&format!("submodule for --features {feat}: present"));
        } else {
            info(&format!(
                "submodule for --features {feat}: absent (git submodule update --init)"
            ));
        }
    }

    // --- env echo ---
    println!("\nenv:");
    for k in [
        "CORRODE_SANDBOX",
        "CORRODE_SANDBOX_NET",
        "CORRODE_USERS",
        "CORRODE_REPO",
        "CORRODE_GRAPH_DIR",
        "CORRODE_DOC_ROOTS",
        "CORRODE_MODEL",
        "CORRODE_ROLES",
        "HIPFIRE_BASE_URL",
        "CORRODE_DAEMON_ADDR",
        "CORRODE_WEB_ADDR",
    ] {
        println!("  {k}={}", std::env::var(k).unwrap_or_else(|_| "(unset)".into()));
    }

    if fatal > 0 {
        println!("\n{fatal} fatal issue(s) — the daemon may not work as configured.");
        false
    } else {
        println!("\nall clear.");
        true
    }
}

/// The real usability test for the sandbox: actually run an unprivileged bwrap.
/// Its failure modes (missing binary, AppArmor/userns restriction) are hard to
/// enumerate from config alone, so we just try it.
fn bwrap_usable() -> anyhow::Result<()> {
    let out = Command::new("bwrap")
        .args([
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind",
            "/usr",
            "/usr",
            // Mirror the real sandbox's bind set: /bin (merged-usr symlink) plus the
            // lib dirs, or a dynamically-linked probe binary can't find its ELF
            // interpreter and execvp reports a misleading ENOENT.
            "--ro-bind-try",
            "/bin",
            "/bin",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--",
            "/usr/bin/true",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("cannot exec bwrap ({e}) — is bubblewrap installed?"))?;
    if out.status.success() {
        Ok(())
    } else {
        anyhow::bail!("{}", String::from_utf8_lossy(&out.stderr).trim())
    }
}
