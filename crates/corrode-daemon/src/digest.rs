//! Structured observations: turn raw command output into something a model can act on.
//!
//! A tool result is an observation, not a transcript. The previous behaviour truncated
//! captured output to the first 4 KiB, which for a build log is the *compile progress*
//! — the diagnostics and the `N previous errors` summary are at the tail, so a failing
//! build reported its least useful bytes and dropped the rest.
//!
//! Two recognizers and a fallback, all pure string functions over already-captured
//! output. We deliberately parse what the command already printed rather than re-running
//! it with `--message-format=json`: rewriting a model-authored command is a surprising
//! thing for a tool layer to do, and the human format carries what we need.
//!
//! ponytail: rustc and libtest only, since that is what this repo builds. A JSON path
//! (`cargo --message-format=json`) is strictly better structured and worth taking when a
//! caller can opt in per command — see `docs/harness-architecture.md` §3.5.

/// Bytes of raw output kept when nothing is recognized. Matches the cap `tools.rs`
/// used before this module existed, so unrecognized output behaves exactly as before.
const FALLBACK_CAP: usize = 8192;
/// Diagnostics listed before the rest are elided.
const MAX_LISTED: usize = 12;
/// Failing test names listed before the rest are elided.
const MAX_FAILED_TESTS: usize = 20;

/// One rustc diagnostic: severity, optional `[E0061]` code, message, first location.
#[derive(Debug, PartialEq)]
struct Diagnostic {
    severity: Severity,
    code: Option<String>,
    message: String,
    location: Option<String>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Severity {
    Error,
    Warning,
}

/// Build the observation for a finished command.
///
/// `exit` is the process status; `raw` is stdout and stderr already concatenated in the
/// order the caller wants them read.
pub fn command_observation(exit: i32, raw: &str) -> String {
    if let Some(tests) = test_digest(exit, raw) {
        return tests;
    }
    if let Some(diags) = diagnostic_digest(exit, raw) {
        return diags;
    }
    format!("exit {exit}:\n{}", head_tail(raw.trim(), FALLBACK_CAP))
}

/// libtest summary: the counts line plus which tests actually failed. The counts line is
/// the last thing printed, so head-only truncation lost precisely the answer.
fn test_digest(exit: i32, raw: &str) -> Option<String> {
    let results: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("test result:"))
        .collect();
    if results.is_empty() {
        return None;
    }
    let failed: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter_map(|l| {
            // `test path::to::name ... FAILED`
            let rest = l.strip_prefix("test ")?;
            let (name, tail) = rest.split_once(" ... ")?;
            tail.starts_with("FAILED").then_some(name)
        })
        .collect();

    let mut out = format!("exit {exit} — test run\n");
    for r in &results {
        out.push_str(&format!("  {r}\n"));
    }
    if failed.is_empty() {
        return Some(out.trim_end().to_string());
    }
    out.push_str(&format!("\n{} failing:\n", failed.len()));
    for name in failed.iter().take(MAX_FAILED_TESTS) {
        out.push_str(&format!("  {name}\n"));
    }
    if failed.len() > MAX_FAILED_TESTS {
        out.push_str(&format!(
            "  … {} more\n",
            failed.len() - MAX_FAILED_TESTS
        ));
    }
    Some(out.trim_end().to_string())
}

/// rustc/cargo diagnostics, grouped so one root cause repeated across call sites reads
/// as one finding with a count rather than as N separate problems.
fn diagnostic_digest(exit: i32, raw: &str) -> Option<String> {
    let diags = parse_diagnostics(raw);
    if diags.is_empty() {
        return None;
    }
    let errors: Vec<&Diagnostic> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    let warnings = diags.len() - errors.len();

    let mut out = format!(
        "exit {exit} — {} error{}, {} warning{}\n",
        errors.len(),
        plural(errors.len()),
        warnings,
        plural(warnings)
    );

    // Group identical (code, message) pairs: the same error at nine call sites is one
    // thing to fix, and listing it nine times crowds out the others.
    let mut groups: Vec<(&Diagnostic, Vec<&str>)> = Vec::new();
    for d in &errors {
        let key = (d.code.as_deref(), d.message.as_str());
        match groups
            .iter_mut()
            .find(|(g, _)| (g.code.as_deref(), g.message.as_str()) == key)
        {
            Some((_, locs)) => locs.extend(d.location.as_deref()),
            None => groups.push((d, d.location.as_deref().into_iter().collect())),
        }
    }

    for (d, locs) in groups.iter().take(MAX_LISTED) {
        // Mirror rustc's own `error[E0061]: msg` shape — the model has seen it before.
        let code = d.code.as_deref().map(|c| format!("[{c}]")).unwrap_or_default();
        out.push_str(&format!("  error{code}: {}\n", d.message));
        match locs.split_first() {
            Some((first, rest)) if rest.is_empty() => out.push_str(&format!("    at {first}\n")),
            Some((first, rest)) => out.push_str(&format!(
                "    at {first} (+{} more site{})\n",
                rest.len(),
                plural(rest.len())
            )),
            None => {}
        }
    }
    if groups.len() > MAX_LISTED {
        out.push_str(&format!("  … {} more\n", groups.len() - MAX_LISTED));
    }
    if warnings > 0 && errors.is_empty() {
        // Warnings only: show them, since they are the whole result.
        for d in diags.iter().filter(|d| d.severity == Severity::Warning).take(MAX_LISTED) {
            out.push_str(&format!("  warning: {}\n", d.message));
            if let Some(loc) = &d.location {
                out.push_str(&format!("    at {loc}\n"));
            }
        }
    }
    Some(out.trim_end().to_string())
}

/// Scan for `error[E0061]: msg` / `error: msg` / `warning: msg`, attaching the `-->`
/// location that rustc prints on the following line.
fn parse_diagnostics(raw: &str) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("--> ") {
            // Belongs to the diagnostic opened most recently.
            if let Some(last) = out.last_mut() {
                if last.location.is_none() {
                    last.location = Some(rest.trim().to_string());
                }
            }
            continue;
        }
        let (severity, rest) = if let Some(r) = trimmed.strip_prefix("error") {
            (Severity::Error, r)
        } else if let Some(r) = trimmed.strip_prefix("warning") {
            (Severity::Warning, r)
        } else {
            continue;
        };
        // `[E0061]: msg` or `: msg`; anything else is prose that merely starts with the word.
        let (code, message) = if let Some(r) = rest.strip_prefix('[') {
            let Some((code, r)) = r.split_once(']') else {
                continue;
            };
            let Some(m) = r.strip_prefix(": ") else {
                continue;
            };
            (Some(code.to_string()), m)
        } else {
            let Some(m) = rest.strip_prefix(": ") else {
                continue;
            };
            (None, m)
        };
        if message.trim().is_empty() {
            continue;
        }
        // Cargo's own tail summaries duplicate counts we compute ourselves, and would
        // otherwise inflate them: `error: could not compile ... due to 4 previous
        // errors` and ``warning: `probe` (bin "probe") generated 1 warning``.
        if message.starts_with("could not compile") || message.starts_with("build failed") {
            continue;
        }
        if message.starts_with('`') && message.contains(") generated ") {
            continue;
        }
        out.push(Diagnostic {
            severity,
            code,
            message: message.trim().to_string(),
            location: None,
        });
    }
    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// Keep the head *and* the tail. A build log puts progress first and the answer last, so
/// head-only truncation drops the part worth reading.
fn head_tail(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let half = cap / 2;
    let head_len = crate::tools::floor_char_boundary(s, half);
    let tail_start = s.len() - half;
    let tail_start = (tail_start..s.len())
        .find(|i| s.is_char_boundary(*i))
        .unwrap_or(s.len());
    format!(
        "{}\n… ({} bytes elided) …\n{}",
        &s[..head_len],
        tail_start - head_len,
        &s[tail_start..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The failure this module exists for: a build log's verdict is at the END, so the
    /// old head-only cut kept compile progress and dropped the errors entirely.
    #[test]
    fn build_errors_are_grouped_with_locations_not_truncated_away() {
        let mut raw = String::new();
        for i in 0..400 {
            raw.push_str(&format!("   Compiling crate-number-{i} v0.1.0\n"));
        }
        raw.push_str(
            "error[E0061]: this function takes 9 arguments but 8 were supplied\n \
             --> crates/corrode-daemon/src/daemon.rs:1033:9\n\
             error[E0061]: this function takes 9 arguments but 8 were supplied\n \
             --> crates/corrode-daemon/src/daemon.rs:1067:22\n\
             error[E0425]: cannot find value `MAX_CMD_BYTES` in this scope\n \
             --> crates/corrode-daemon/src/tools.rs:612:29\n\
             warning: unused import: `self`\n \
             --> crates/corrode-daemon/src/daemon.rs:17:20\n\
             error: could not compile `corrode-daemon` due to 3 previous errors\n",
        );
        let obs = command_observation(101, &raw);

        assert!(obs.starts_with("exit 101 — 3 errors, 1 warning"), "{obs}");
        // The repeated E0061 is ONE thing to fix, reported once with a site count.
        assert!(obs.contains("error[E0061]: this function takes 9 arguments"), "{obs}");
        assert!(obs.contains("(+1 more site)"), "{obs}");
        assert!(obs.contains("error[E0425]: cannot find value"), "{obs}");
        assert!(obs.contains("crates/corrode-daemon/src/tools.rs:612:29"), "{obs}");
        // 400 lines of progress do not reach the model.
        assert!(!obs.contains("Compiling crate-number-200"), "{obs}");
        // Cargo's own tail summary is dropped: we report the count ourselves.
        assert!(!obs.contains("could not compile"), "{obs}");
    }

    #[test]
    fn test_runs_report_counts_and_which_tests_failed() {
        let raw = "\
running 3 tests
test daemon::tests::alpha ... ok
test daemon::tests::beta ... FAILED
test daemon::tests::gamma ... FAILED

failures:
    daemon::tests::beta

test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out
";
        let obs = command_observation(101, raw);
        assert!(obs.contains("test result: FAILED. 1 passed; 2 failed"), "{obs}");
        assert!(obs.contains("2 failing:"), "{obs}");
        assert!(obs.contains("daemon::tests::beta"), "{obs}");
        assert!(obs.contains("daemon::tests::gamma"), "{obs}");
    }

    #[test]
    fn a_passing_run_says_so_without_listing_failures() {
        let raw = "test result: ok. 59 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n";
        let obs = command_observation(0, raw);
        assert!(obs.contains("59 passed"), "{obs}");
        assert!(!obs.contains("failing:"), "{obs}");
    }

    /// Prose that merely begins with "error" is not a diagnostic, and unrecognized
    /// output keeps both ends.
    #[test]
    fn unrecognized_output_keeps_head_and_tail() {
        let mut raw = String::from("FIRST-LINE\n");
        raw.push_str(&"filler filler filler\n".repeat(500));
        raw.push_str("LAST-LINE\n");
        let obs = command_observation(0, &raw);
        assert!(obs.contains("FIRST-LINE"), "{obs}");
        assert!(obs.contains("LAST-LINE"), "head-only truncation would lose this: {obs}");
        assert!(obs.contains("elided"), "{obs}");
    }

    #[test]
    fn word_error_in_prose_is_not_a_diagnostic() {
        let raw = "error handling is described in the README\nerrors happen\n";
        let obs = command_observation(0, raw);
        assert!(obs.starts_with("exit 0:"), "should fall through to raw: {obs}");
    }

    #[test]
    fn elision_never_splits_a_multibyte_char() {
        let raw = "€".repeat(8000);
        let obs = command_observation(0, &raw);
        assert!(obs.contains("elided"), "{obs}");
    }
    /// Verbatim `cargo build` output from a crate broken on purpose — the parser is fed
    /// real rustc formatting, including the `note:`/`help:` blocks that carry their own
    /// `-->` lines and cargo's per-crate `generated N warning` summary, both of which
    /// would corrupt the counts if treated as diagnostics.
    #[test]
    fn parses_real_cargo_output() {
        let raw = r#"   Compiling probe v0.0.0 (/tmp/claude-1000/-home-sadara-corrode/86b0eea0-6ee4-409f-b8da-b208950435df/scratchpad/probe)
error[E0425]: cannot find value `MISSING_CONST` in this scope
 --> src/main.rs:7:20
  |
7 |     println!("{}", MISSING_CONST);
  |                    ^^^^^^^^^^^^^ not found in this scope

warning: unused import: `std::collections::HashMap`
 --> src/main.rs:2:5
  |
2 | use std::collections::HashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

error[E0061]: this function takes 3 arguments but 2 arguments were supplied
 --> src/main.rs:4:5
  |
4 |     takes_three(1, 2);
  |     ^^^^^^^^^^^------ argument #3 of type `u8` is missing
  |
note: function defined here
 --> src/main.rs:1:4
  |
1 | fn takes_three(_a: u8, _b: u8, _c: u8) {}
  |    ^^^^^^^^^^^                 ------
help: provide the argument
  |
4 |     takes_three(1, 2, /* u8 */);
  |                     ++++++++++

error[E0061]: this function takes 3 arguments but 2 arguments were supplied
 --> src/main.rs:5:5
  |
5 |     takes_three(3, 4);
  |     ^^^^^^^^^^^------ argument #3 of type `u8` is missing
  |
note: function defined here
 --> src/main.rs:1:4
  |
1 | fn takes_three(_a: u8, _b: u8, _c: u8) {}
  |    ^^^^^^^^^^^                 ------
help: provide the argument
  |
5 |     takes_three(3, 4, /* u8 */);
  |                     ++++++++++

error[E0308]: mismatched types
 --> src/main.rs:6:19
  |
6 |     let _x: u32 = "not a number";
  |             ---   ^^^^^^^^^^^^^^ expected `u32`, found `&str`
  |             |
  |             expected due to this

Some errors have detailed explanations: E0061, E0308, E0425.
For more information about an error, try `rustc --explain E0061`.
warning: `probe` (bin "probe") generated 1 warning
error: could not compile `probe` (bin "probe") due to 4 previous errors; 1 warning emitted
"#;
        let obs = command_observation(101, raw);

        // 4 errors / 1 warning is what cargo itself reports on its last line.
        assert!(obs.starts_with("exit 101 — 4 errors, 1 warning"), "{obs}");
        // The duplicated E0061 collapses to one finding across two call sites.
        assert!(obs.contains("(+1 more site)"), "{obs}");
        assert!(obs.contains("error[E0425]: cannot find value `MISSING_CONST` in this scope"), "{obs}");
        assert!(obs.contains("error[E0308]: mismatched types"), "{obs}");
        // Locations come from the diagnostic, never from the trailing `note:` block.
        assert!(obs.contains("src/main.rs:4:5"), "{obs}");
        assert!(!obs.contains("src/main.rs:1:4"), "note location must not win: {obs}");
        // Neither cargo summary line becomes a diagnostic.
        assert!(!obs.contains("could not compile"), "{obs}");
        assert!(!obs.contains("generated 1 warning"), "{obs}");
    }

}
