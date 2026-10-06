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

    // --- knobs: the daemon refuses to start on any of these ---
    let bad = crate::knobs::check();
    for b in &bad {
        fatal += 1;
        fail(b, "fix or unset it — the daemon refuses to start with it");
    }
    if bad.is_empty() {
        ok("knobs: every set CORRODE_* flag and bound parses");
    }

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
            role_assignments(&models);
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

    // --- remote endpoint (optional; work falls back to hipfire without it) ---
    if let Some(r) = crate::remote::Remote::from_env() {
        info(&r.describe());
        match r.client.list_models().await {
            Ok(models) if models.contains(&r.model) => {
                ok(&format!("remote serves {}", r.model))
            }
            Ok(models) => warn(&format!(
                "remote at {} does not list {} ({} model(s) listed); its work will fall back to hipfire if it refuses",
                r.client.base_url(),
                r.model,
                models.len()
            )),
            Err(e) => warn(&format!(
                "remote at {} unreachable ({e}); its work will fall back to hipfire",
                r.client.base_url()
            )),
        }
    }

    // --- sandbox (only meaningful when enabled) ---
    let repo = std::env::var("CORRODE_REPO").unwrap_or_else(|_| ".".into());
    let sandbox_on = crate::knobs::flag("CORRODE_SANDBOX", true);
    if sandbox_on {
        let root = std::fs::canonicalize(&repo).unwrap_or_else(|_| repo.clone().into());
        match sandboxed(&root, &["true"]) {
            Ok(_) => {
                ok("sandbox: bwrap confines (repo rw, .corrode ro, home credential stores masked, no net)");
                match sandboxed(&root, &["sh", "-c", "command -v cargo >/dev/null && cargo --version"]) {
                    Ok(v) => ok(&format!("sandbox: builds can run ({})", v.trim())),
                    Err(e) => warn(&format!(
                        "sandbox: no cargo inside it ({e}) — every build and test the swarm runs \
                         exits 127; put ~/.cargo/bin on the daemon's PATH"
                    )),
                }
            }
            Err(e) => {
                fatal += 1;
                fail(
                    &format!("the sandbox is on but bwrap is unusable, so every command fails: {e}"),
                    "apt install bubblewrap; on Ubuntu load an AppArmor userns profile \
                     for /usr/bin/bwrap (docs/corrode-doctor.md §3)",
                );
            }
        }
    } else {
        // The file tools refuse ~/.ssh and the other credential stores either way; a
        // command or the terminal outside the sandbox can still read them.
        warn(
            "sandbox: off (CORRODE_SANDBOX) — run_command, skill scripts and the web terminal \
             can read ~/.ssh and other credential stores",
        );
    }

    if crate::knobs::flag("CORRODE_AUTO_APPROVE", false) {
        if sandbox_on {
            info("auto-approve: on — writes and commands run without a human, inside the sandbox");
        } else {
            warn(
                "auto-approve: on WITHOUT the sandbox — every write and command the swarm \
                 proposes runs unconfined with the daemon's privileges (CORRODE_SANDBOX is off)",
            );
        }
    }

    // --- auth table ---
    match std::env::var("CORRODE_USERS") {
        Ok(path) => match crate::daemon::parse_users(&path) {
            Ok(users) => ok(&format!("auth: on, {} user(s) configured", users.len())),
            Err(e) => {
                fatal += 1;
                fail(
                    &format!("CORRODE_USERS at {path} is unusable: {e}"),
                    "expected {\"alice\": {\"token\": \"…\", \"hipfire_token\": \"…\"}} with at \
                     least one user; until it is fixed auth stays on and no connection can authenticate",
                );
            }
        },
        Err(_) => info("auth: off (no CORRODE_USERS) — connections are anonymous"),
    }
    match std::env::var("CORRODE_REPO_ALLOW") {
        Ok(list) => info(&format!("SelectRepo: confined to {list} (entries inside $HOME only)")),
        Err(_) => info("SelectRepo: any directory inside $HOME (CORRODE_REPO_ALLOW unset)"),
    }

    // --- repo ---
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

    // --- effective policy: what the daemon will do, defaults applied, read through the
    // same functions the daemon uses -- the raw env echo showed a hand-kept subset and
    // nothing for an unset knob ---
    use crate::knobs::flag;
    let on = |b: bool| if b { "on" } else { "off" };
    let secs = |d: Option<std::time::Duration>| {
        d.map_or_else(|| "unbounded".to_string(), |d| format!("{}s", d.as_secs()))
    };
    println!("\npolicy (effective):");
    println!(
        "  sandbox {} (network {}), auto-approve {}",
        on(flag("CORRODE_SANDBOX", true)),
        on(flag("CORRODE_SANDBOX_NET", false)),
        on(flag("CORRODE_AUTO_APPROVE", false))
    );
    println!(
        "  turn budget {}, task timeout {}, command timeout {}s, request timeout {}s, approval timeout {}s, retry window {}s",
        secs(crate::daemon::turn_budget()),
        secs(crate::daemon::task_timeout()),
        crate::tools::command_timeout().as_secs(),
        crate::hipfire::request_timeout_s(),
        crate::approval::approval_timeout().as_secs(),
        crate::hipfire::retry_window().as_secs()
    );
    println!(
        "  tool steps {} (research {}), follow-ups {} per drive, fan-out {}, plan review {}, concurrency {}",
        crate::daemon::max_tool_steps(),
        crate::daemon::max_tool_steps_for(roles::Role::Research),
        crate::plan_graph::max_followups(),
        crate::daemon::fanout_k(),
        on(crate::daemon::plan_review_enabled()),
        crate::daemon::max_concurrency()
    );
    println!(
        "  context {} tokens, output cap {} tokens, streaming {}, graph-backed vfs {}",
        crate::daemon::context_tokens(),
        crate::hipfire::max_output_tokens(),
        on(flag("CORRODE_STREAM", false)),
        on(flag("CORRODE_VFS_GRAPH", false))
    );
    let efforts: Vec<String> = roles::Role::ALL
        .iter()
        .map(|&r| format!("{}={}", r.as_str(), roles::effort_for(r)))
        .collect();
    println!("  reasoning effort: {}", efforts.join(", "));

    // --- env echo (where things are, not how they behave) ---
    println!("\nenv:");
    for k in [
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

/// Print the model each role will run on. `RoleModels::resolve` drops an override
/// naming a model hipfire does not serve, so a typo in `CORRODE_ROLES` would put
/// that role on the default pick without a word — warn about each one here.
fn role_assignments(served: &[String]) {
    let overrides = match roles::RoleModels::overrides_from_env() {
        Ok(o) => o,
        Err(e) => {
            warn(&format!("CORRODE_ROLES unreadable ({e}); every role uses the default pick"));
            roles::RoleModels::default()
        }
    };
    for (role, model) in &overrides.0 {
        if !served.iter().any(|s| s == model) {
            warn(&format!(
                "CORRODE_ROLES {}: '{model}' is not served; using the default pick",
                role.as_str()
            ));
        }
    }
    if let Ok(resolved) = roles::RoleModels::resolve(served, &overrides) {
        for role in roles::Role::ALL {
            let model = resolved.model_for(role).unwrap_or("?");
            info(&format!("role {:<13} -> {model}", role.as_str()));
        }
    }
}

/// Run `argv` through the daemon's real [`crate::sandbox::Sandbox::wrap`] for `repo`,
/// returning its stdout. A probe with its own bwrap arguments said "ok" while the
/// real bind set left cargo unreachable.
fn sandboxed(repo: &Path, argv: &[&str]) -> anyhow::Result<String> {
    let (prog, args) = crate::sandbox::Sandbox::from_env().wrap(repo, argv);
    let out = Command::new(&prog)
        .args(&args)
        .output()
        .map_err(|e| anyhow::anyhow!("cannot exec {prog} ({e}) — is bubblewrap installed?"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("exit {} {}", out.status.code().unwrap_or(-1), err.trim())
    }
}
