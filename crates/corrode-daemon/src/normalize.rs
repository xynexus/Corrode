//! Trading bijectivity away deliberately: normalise a repo to its formatters.
//!
//! Verbatim storage buys exactness cheaply *today*, but the bill grows: every time a
//! node gets more specific, the corner cases that keep projection byte-exact get more
//! numerous — `rustfmt::skip`, raw strings, macro bodies, attribute placement, and
//! whatever the next language brings. A project that would rather pay once can
//! normalise its source and stop paying at all.
//!
//! The key point is that this needs no second ingest path. **Normalised source is
//! byte-exact under the existing verbatim pipeline** — the quirks stop mattering because
//! they are no longer in the file. So `fidelity: normalized` is a claim about the repo,
//! enforced by this check, not a switch that makes ingest lossy.
//!
//! Normalising is done by the language's REAL formatter (`rustfmt`, `clang-format`), not
//! by printing our own nodes back out. That is the difference between this and the
//! canonical-form experiment the design doc rejected: printing a `syn` AST destroyed
//! 35,477 body comments because the AST has no node for them, while `rustfmt` keeps
//! every one. Comment loss was the whole objection, and using the real formatter
//! answers it.
//!
//! The contract for a formatter is stdin -> stdout, which both defaults satisfy and
//! which keeps `--check` from having to touch a single file.

use crate::project::Project;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Default)]
pub struct Report {
    pub unchanged: usize,
    pub changed: usize,
    pub skipped: usize,
    pub unreadable: usize,
    pub failed: usize,
    /// A few names, so a failure is actionable rather than a count.
    pub examples: Vec<String>,
    /// Per backend: (files, files with no formatter). A bare total hides that most of
    /// a real tree — Makefiles, Kconfig, docs — has no formatter and never will.
    pub by_language: std::collections::BTreeMap<&'static str, (usize, usize)>,
}

impl Report {
    /// Is the repo actually in the normal form it may be claiming?
    pub fn normalized(&self) -> bool {
        self.changed == 0 && self.failed == 0
    }
}

/// Run `cmd` over `src` as stdin, returning its stdout.
fn format_with(cmd: &[String], path: &str, src: &str) -> anyhow::Result<String> {
    let args: Vec<String> = cmd[1..].iter().map(|a| a.replace("{path}", path)).collect();
    let mut child = Command::new(&cmd[0])
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("{} not available: {e}", cmd[0]))?;

    // Write stdin on its own thread: a formatter that starts emitting before it has
    // consumed the input will deadlock against a single-threaded write-then-read.
    let mut stdin = child.stdin.take().expect("piped");
    let owned = src.to_string();
    let writer = std::thread::spawn(move || stdin.write_all(owned.as_bytes()));

    let out = child.wait_with_output()?;
    writer.join().map_err(|_| anyhow::anyhow!("stdin writer panicked"))??;
    if !out.status.success() {
        anyhow::bail!("{} failed ({}): {}", cmd[0], out.status, String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8(out.stdout)?)
}

/// Files git tracks. Using git rather than a directory walk is deliberate: it honours
/// `.gitignore`, skips `.git`, ignores untracked scratch files, and is the same set the
/// one-off commit will contain.
fn tracked_files(root: &Path) -> anyhow::Result<Vec<String>> {
    let out = Command::new("git").arg("-C").arg(root).args(["ls-files", "-z"]).output()?;
    if !out.status.success() {
        anyhow::bail!("not a git repository: {}", root.display());
    }
    Ok(String::from_utf8(out.stdout)?
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect())
}

/// Does the working tree have uncommitted changes?
fn dirty(root: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(true)
}

/// Normalise (or check) every tracked file.
///
/// `write: false` touches nothing and reports what would change — the default, because
/// the writing form rewrites every file in the repository.
pub fn run(project: &Project, write: bool) -> anyhow::Result<Report> {
    let root = &project.root;
    let files = tracked_files(root)?;

    // Rewriting a whole repo is only safe if it is revertible and reviewable. A dirty
    // tree means `git checkout .` would take the user's own work with it.
    if write && dirty(root) {
        anyhow::bail!(
            "working tree has uncommitted changes — commit or stash first, so the \
             normalisation is one reviewable commit that `git checkout .` can undo"
        );
    }

    let mut r = Report::default();
    for rel in &files {
        let lang = crate::projection::for_path(rel);
        let entry = r.by_language.entry(lang.name()).or_default();
        entry.0 += 1;
        let Some(cmd) = project.formatters.get(lang.name()) else {
            entry.1 += 1;
            r.skipped += 1;
            continue;
        };
        let Ok(src) = std::fs::read_to_string(root.join(rel)) else {
            r.unreadable += 1;
            continue;
        };
        match format_with(cmd, rel, &src) {
            Ok(out) if out == src => r.unchanged += 1,
            Ok(out) => {
                r.changed += 1;
                if r.examples.len() < 5 {
                    r.examples.push(rel.clone());
                }
                if write {
                    std::fs::write(root.join(rel), out)?;
                }
            }
            Err(e) => {
                r.failed += 1;
                if r.examples.len() < 5 {
                    r.examples.push(format!("{rel}: {e}"));
                }
            }
        }
    }
    Ok(r)
}

/// `corrode-daemon normalize [--write]`.
pub fn main(project: &Project, write: bool) -> bool {
    let r = match run(project, write) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("normalize: {e}");
            return false;
        }
    };
    let verb = if write { "rewrote" } else { "would rewrite" };
    eprintln!("normalize {} ({:?} fidelity)", project.name, project.fidelity);
    eprintln!("  already normal  {}", r.unchanged);
    eprintln!("  {verb:<15} {}", r.changed);
    eprintln!("  no formatter    {}", r.skipped);
    eprintln!("  unreadable      {}", r.unreadable);
    eprintln!("  failed          {}", r.failed);
    if !r.examples.is_empty() {
        eprintln!("  e.g. {}", r.examples.join(", "));
    }
    eprintln!("\n  {:<10} {:>7} {:>14}", "backend", "files", "no formatter");
    for (lang, (total, none)) in &r.by_language {
        eprintln!("  {lang:<10} {total:>7} {none:>14}");
    }
    if write && r.changed > 0 {
        eprintln!("\nreview and commit as one change; ingest stays byte-exact afterwards.");
    }
    // A repo that CLAIMS to be normalised and is not has a false statement in its
    // config, which is worth an exit code. A verbatim repo is just being surveyed.
    if !write && project.fidelity == crate::project::Fidelity::Normalized && !r.normalized() {
        eprintln!("\nfidelity is `normalized` but {} file(s) are not — run with --write", r.changed);
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing that is whitespace-significant or prose may acquire a default formatter.
    ///
    /// A Makefile's recipe lines are distinguished from continuations by a LEADING TAB;
    /// a formatter that normalises indentation silently breaks the build, and the repo
    /// still compiles for everyone who has not re-run make. Prose is worse in a quieter
    /// way — reflowing a README is a diff on every line of every document for no gain.
    /// These languages stay verbatim by construction, not by anyone remembering.
    #[test]
    fn tab_significant_and_prose_backends_have_no_default_formatter() {
        let p = crate::project::Project::load(std::path::Path::new("/nonexistent"));
        for lang in ["hash", "markup", "rst", "none", "dashdash", "semicolon"] {
            assert!(
                !p.formatters.contains_key(lang),
                "{lang} must not have a default formatter: Makefiles are tab-significant \
                 and prose reflow is a whole-repo diff for nothing"
            );
        }
    }

    /// The C-side version of the same question: does normalising make the parts we would
    /// have to GENERATE regular enough to stop storing?
    ///
    /// Verbatim trivia (the bytes between items) is what makes projection byte-exact. If
    /// normalised source draws its trivia from a handful of distinct strings, a
    /// normalised project could generate them and drop the stored text — that, and not
    /// printer agreement, is what would actually delete the fidelity machinery for C.
    /// Also checks idempotence, because `fidelity: normalized` is a claim that has to
    /// stay true across a second run.
    #[test]
    #[ignore = "probe: needs clang-format and a C repo"]
    fn c_trivia_regularity_before_and_after() {
        use std::collections::HashMap;
        let root = std::path::PathBuf::from(std::env::var("CORRODE_REPO").unwrap());
        let p = crate::project::Project::load(&root);
        let cmd = &p.formatters["c"];
        let mut census: Vec<(&str, HashMap<String, usize>, usize)> =
            vec![("as committed", HashMap::new(), 0), ("normalised", HashMap::new(), 0)];
        let (mut files, mut non_idempotent, mut not_exact) = (0, 0, 0);
        let mut indents: Vec<HashMap<String, usize>> = vec![HashMap::new(), HashMap::new()];

        for rel in tracked_files(&root).unwrap() {
            let lang = crate::projection::for_path(&rel);
            if lang.name() != "c" { continue }
            let Ok(src) = std::fs::read_to_string(root.join(&rel)) else { continue };
            let Ok(fmt) = format_with(cmd, &rel, &src) else { continue };
            files += 1;
            // A normal form that moves on a second pass is not a normal form.
            if format_with(cmd, &rel, &fmt).map(|f| f != fmt).unwrap_or(true) {
                non_idempotent += 1;
            }
            // Finer nodes need finer separators, so the premise is really about
            // sub-item trivia: line indentation. Measured as a proxy for the deep
            // granularity we do not have nodes for yet.
            for (i, text) in [&src, &fmt].into_iter().enumerate() {
                for line in text.lines() {
                    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                    if !indent.is_empty() {
                        *indents[i].entry(indent).or_default() += 1;
                    }
                }
            }
            for (i, text) in [&src, &fmt].into_iter().enumerate() {
                let Ok((items, _)) = lang.spans(text) else { continue };
                let nodes = crate::projection::nodes_from_items(&rel, text, &items);
                // Normalised source must still round-trip, or nothing above matters.
                if i == 1 && crate::projection::project(&nodes).0 != *text { not_exact += 1 }
                for n in nodes.iter().filter(|n| n.kind == "trivia") {
                    census[i].2 += n.text.len();
                    // Split whitespace trivia from trivia carrying a comment. They are
                    // different problems: whitespace could be GENERATED if it is
                    // regular, while comment text is unique by nature and no formatter
                    // makes it otherwise. Lumping them hides which one is irregular.
                    let key = if n.text.trim().is_empty() { n.text.clone() } else { "<comment>".into() };
                    *census[i].1.entry(key).or_default() += 1;
                }
            }
        }

        eprintln!("\n{files} C files — {non_idempotent} not idempotent, {not_exact} not byte-exact after normalising");
        for (label, counts, bytes) in &census {
            let mut top: Vec<_> = counts.iter().collect();
            top.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            let total: usize = counts.values().sum();
            let top5: usize = top.iter().take(5).map(|(_, c)| **c).sum();
            let with_comment = counts.get("<comment>").copied().unwrap_or(0);
            eprintln!(
                "  {label:<14} {total} trivia nodes, {bytes} bytes",
            );
            eprintln!(
                "      whitespace-only {} in {} distinct forms; top 5 cover {:.1}% of all trivia",
                total - with_comment, counts.len() - 1, 100.0 * top5 as f64 / total.max(1) as f64
            );
            eprintln!("      carrying a comment {with_comment} ({:.1}%)", 100.0 * with_comment as f64 / total.max(1) as f64);
            for (text, n) in top.iter().take(3) {
                eprintln!("      {n:>7}x {:?}", text.chars().take(24).collect::<String>());
            }
        }
        for (label, m) in ["as committed", "normalised"].iter().zip(&indents) {
            let total: usize = m.values().sum();
            let tabs: usize = m.iter().filter(|(k, _)| k.contains('\t')).map(|(_, v)| v).sum();
            let mut top: Vec<_> = m.iter().collect();
            top.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
            let top8: usize = top.iter().take(8).map(|(_, c)| **c).sum();
            eprintln!(
                "  {label:<14} line indents: {total} indented lines, {} distinct forms; \
                 top 8 cover {:.1}%; tab-containing {:.1}%",
                m.len(), 100.0 * top8 as f64 / total.max(1) as f64,
                100.0 * tabs as f64 / total.max(1) as f64
            );
        }
    }

    #[test]
    fn formatter_runs_over_stdin_and_substitutes_the_path() {
        let cmd: Vec<String> = ["sh", "-c", "printf '%s' \"$(cat)\"; printf '|{path}'"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = format_with(&cmd, "src/a.rs", "hello").unwrap();
        assert_eq!(out, "hello|src/a.rs");
    }

    #[test]
    fn a_failing_formatter_is_reported_not_swallowed() {
        let cmd: Vec<String> = ["sh", "-c", "exit 3"].iter().map(|s| s.to_string()).collect();
        assert!(format_with(&cmd, "x.rs", "hi").is_err());
    }

    /// Does normalising actually shrink the corner-case tail, or only move it?
    ///
    /// The premise of `fidelity: normalized` is that regular source has fewer quirks to
    /// special-case. That is a claim about a real repo, so it is measured against one
    /// rather than assumed: run every tracked Rust file through `rustfmt`, then re-run
    /// the AST-regeneration census on the result and compare it to the same census on
    /// the source as committed.
    #[test]
    #[ignore = "probe: needs rustfmt and a git repo"]
    fn normalising_shrinks_the_divergence_census() {
        use crate::roundtrip::regen::{diagnose, Regen};
        let root = std::path::PathBuf::from(
            std::env::var("CORRODE_REPO").unwrap_or_else(|_| ".".into()),
        );
        let p = crate::project::Project::load(&root);
        let cmd = &p.formatters["rust"];
        let (mut raw_exact, mut fmt_exact, mut n, mut unformattable) = (0, 0, 0, 0);
        let mut raw_reasons: std::collections::HashMap<String, usize> = Default::default();
        let mut fmt_reasons: std::collections::HashMap<String, usize> = Default::default();

        for rel in tracked_files(&root).unwrap() {
            if !rel.ends_with(".rs") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(root.join(&rel)) else { continue };
            let Ok(formatted) = format_with(cmd, &rel, &src) else {
                unformattable += 1;
                continue;
            };
            n += 1;
            for (text, exact, reasons) in [
                (&src, &mut raw_exact, &mut raw_reasons),
                (&formatted, &mut fmt_exact, &mut fmt_reasons),
            ] {
                match diagnose(text).0 {
                    Regen::Exact => *exact += 1,
                    Regen::Diverged(r) => *reasons.entry(format!("{r:?}").split(' ').next().unwrap().to_string()).or_default() += 1,
                    Regen::Unparseable(_) => *reasons.entry("Unparseable".to_string()).or_default() += 1,
                }
            }
        }
        eprintln!("\n{n} Rust files ({unformattable} unformattable)");
        eprintln!("  as committed: {raw_exact} regenerate exactly, reasons {raw_reasons:?}");
        eprintln!("  normalised:   {fmt_exact} regenerate exactly, reasons {fmt_reasons:?}");
    }

    #[test]
    fn large_input_does_not_deadlock() {
        // A formatter that echoes while we are still writing: the failure this guards
        // is a write-then-read that blocks once the child's stdout buffer fills.
        let cmd: Vec<String> = ["cat"].iter().map(|s| s.to_string()).collect();
        let big = "x".repeat(4 << 20);
        assert_eq!(format_with(&cmd, "x.rs", &big).unwrap().len(), big.len());
    }
}
