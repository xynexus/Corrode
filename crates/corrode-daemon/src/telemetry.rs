//! Per-task telemetry: one JSON line per subagent execution.
//!
//! A harness that routes work across models cannot tune that routing without a record
//! of what happened. "Which model should the coder role use?" was answered once in this
//! project by hand-building a benchmark and a correctness checker for a single session
//! and throwing both away — the data to answer it should fall out of ordinary operation
//! instead.
//!
//! JSONL on purpose: append-only, greppable, no schema migration, and readable with
//! `jq` on a box with nothing installed. The architecture doc puts telemetry in the
//! knowledge plane alongside task state; this is the stopgap until the graph write path
//! lands, and the fields are deliberately the ones that survive that move.
//!
//! **Off unless `CORRODE_TELEMETRY` names a path.** Recording is best-effort: a
//! telemetry failure must never fail a turn, so every error here is swallowed.

use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;

/// One subagent execution.
#[derive(Serialize)]
pub struct TaskRecord<'a> {
    /// Seconds since the epoch. Coarse on purpose — ordering matters, precision doesn't.
    pub at: u64,
    /// Provenance root for the turn (e.g. `plan-0`, project-scoped where the daemon
    /// knows its project).
    pub plan: &'a str,
    pub task: u64,
    pub role: &'a str,
    pub model: &'a str,
    /// hipfire scheduler band the request was enqueued at (0/64/255).
    pub band: u8,
    /// K for a fan-out coder task, 1 otherwise.
    pub fanout: usize,
    /// Shared-prefix size. Prefilled once per model and reused, so this is the
    /// amortized part; `task_bytes` is what is paid per call.
    pub prefix_bytes: usize,
    pub task_bytes: usize,
    pub output_bytes: usize,
    pub duration_ms: u128,
    /// Files the task wrote (its `produced_by` code nodes).
    pub artifacts: usize,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Requests and tokens the task's calls cost (retries included).
    #[serde(flatten)]
    pub usage: crate::hipfire::Usage,
    /// Each of those calls: wall time, tokens, and hipfire's time to first token and
    /// decode rate -- what a turn profile is built from.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<crate::hipfire::CallStat>,
    /// How an `ok` task ended when it was not a plain answer (`turn budget`), so a
    /// budget stop does not read like a clean finish.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<&'a str>,
    /// What the task spent on the remote endpoint (`remote.rs`), when it spent any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_usd: Option<f64>,
}

/// One line per Prompt turn, written as it ends: with the task rows, enough to
/// reconstruct an unattended run -- how it ended, what it cost, how much prefix reuse
/// it got. Tagged `"kind":"turn"` to tell it from the task rows.
#[derive(Serialize)]
#[serde(tag = "kind", rename = "turn")]
pub struct TurnRecord<'a> {
    pub at: u64,
    pub plan: &'a str,
    pub status: &'a str,
    pub tasks: usize,
    pub failed: usize,
    pub duration_s: u64,
    /// The planner's calls.
    pub planner: crate::hipfire::Usage,
    /// Each of them, as `TaskRecord::calls`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub planner_calls: Vec<crate::hipfire::CallStat>,
    /// Every task's calls, summed.
    pub usage: crate::hipfire::Usage,
    /// Spent on the remote endpoint, the planner included.
    pub remote_usd: f64,
}

/// Append-only sink. `None` path = disabled, and every method is then a no-op.
#[derive(Debug, Default)]
pub struct Telemetry {
    path: Option<PathBuf>,
}

impl Telemetry {
    /// Read `CORRODE_TELEMETRY`. Absent or empty -> disabled.
    pub fn from_env() -> Self {
        let path = std::env::var("CORRODE_TELEMETRY")
            .ok()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .map(PathBuf::from);
        Telemetry { path }
    }

    pub fn enabled(&self) -> bool {
        self.path.is_some()
    }

    pub fn record_turn(&self, rec: &TurnRecord<'_>) {
        self.append(rec);
    }

    pub fn record(&self, rec: &TaskRecord<'_>) {
        self.append(rec);
    }

    /// Append one record. Silently does nothing when disabled, and swallows IO and
    /// serialization errors — losing a telemetry line is always preferable to failing
    /// the work it describes.
    fn append(&self, rec: &impl Serialize) {
        let Some(path) = &self.path else {
            return;
        };
        let Ok(mut line) = serde_json::to_string(rec) else {
            return;
        };
        line.push('\n');
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Seconds since the epoch, 0 if the clock is before it.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec<'a>(plan: &'a str, ok: bool) -> TaskRecord<'a> {
        TaskRecord {
            at: 1_700_000_000,
            plan,
            task: 3,
            role: "coder",
            model: "Qwen3.6--35B-A3B.oq4.25++",
            band: 64,
            fanout: 1,
            prefix_bytes: 4106,
            task_bytes: 210,
            output_bytes: 989,
            duration_ms: 42_000,
            artifacts: 2,
            ok,
            error: (!ok).then(|| "boom".to_string()),
            usage: crate::hipfire::Usage {
                requests: 3,
                input_tokens: 9000,
                output_tokens: 400,
                cached_tokens: 8100,
            },
            calls: Vec::new(),
            stop: None,
            remote_usd: None,
        }
    }

    #[test]
    fn disabled_by_default_and_writes_nothing() {
        let t = Telemetry::default();
        assert!(!t.enabled());
        t.record(&rec("p/plan-0", true)); // must not panic or create anything
    }

    #[test]
    fn appends_one_json_line_per_record() {
        let dir = std::env::temp_dir().join(format!("corrode-telem-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        // A nested path also exercises parent creation.
        let path = dir.join("nested/telemetry.jsonl");
        let t = Telemetry {
            path: Some(path.clone()),
        };
        assert!(t.enabled());
        t.record(&rec("stitch/plan-0", true));
        t.record(&rec("stitch/plan-1", false));

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "one line per record");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["plan"], "stitch/plan-0");
        assert_eq!(first["model"], "Qwen3.6--35B-A3B.oq4.25++");
        assert_eq!(first["ok"], true);
        // A successful record carries no error key at all, so `jq 'select(.error)'`
        // finds exactly the failures.
        assert!(first.get("error").is_none(), "{first}");

        // Usage sits flat on the row, so `jq '.cached_tokens / .input_tokens'` works.
        assert_eq!(first["requests"], 3);
        assert_eq!(first["cached_tokens"], 8100);
        assert!(first.get("stop").is_none(), "{first}");

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["ok"], false);
        assert_eq!(second["error"], "boom");

        // A turn row is tagged, so it can be told from the task rows.
        t.record_turn(&TurnRecord {
            at: 1_700_000_100,
            plan: "stitch/plan-1",
            status: "complete",
            tasks: 2,
            failed: 1,
            duration_s: 100,
            planner: Default::default(),
            planner_calls: Vec::new(),
            usage: rec("", true).usage,
            remote_usd: 0.25,
        });
        let body = std::fs::read_to_string(&path).unwrap();
        let turn: serde_json::Value = serde_json::from_str(body.lines().nth(2).unwrap()).unwrap();
        assert_eq!(turn["kind"], "turn");
        assert_eq!(turn["usage"]["input_tokens"], 9000);
        assert_eq!(turn["planner"]["requests"], 0);
        assert_eq!(turn["remote_usd"], 0.25);
        assert!(
            first.get("remote_usd").is_none(),
            "no remote spend, no key: {first}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_bad_path_is_swallowed_not_propagated() {
        // A path under a file (not a directory) cannot be created.
        let file = std::env::temp_dir().join(format!("corrode-telem-file-{}", std::process::id()));
        std::fs::write(&file, b"x").unwrap();
        let t = Telemetry {
            path: Some(file.join("cannot/exist.jsonl")),
        };
        t.record(&rec("p/plan-0", true)); // must not panic
        std::fs::remove_file(&file).ok();
    }
}
