//! Replaying a real edit stream to settle the insert-versus-update ratio.
//!
//! `harness-architecture.md` records the sparse order key's one open evidence gap:
//! whether real edits mostly rewrite nodes in place (a dense index would have done) or
//! genuinely insert between them (the sparse key earns its keep). The doc proposed
//! settling it with live-store telemetry; git history settles it sooner, at far higher
//! volume, and reproducibly — a commit is exactly the edit the graph would have taken.
//!
//! The replay keeps a live node list per path and drives [`projection::update::reconcile`]
//! over every commit, which is also the first exercise of that function against anything
//! but hand-written fixtures.
//!
//! Caveat worth keeping attached to any number this produces: these are human commits,
//! which are a PROXY for agent edits, not the same distribution. What it can settle is
//! whether a real edit stream ever exhausts a 2^32 gap; what it cannot settle is the
//! ratio Corrode's own agents will produce.

#![cfg(test)]

use crate::projection::{self, update::Update, Node};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let out = Command::new("git").arg("-C").arg(repo).args(args).output()?;
    if !out.status.success() {
        anyhow::bail!(
            "git {:?} failed ({}): {}",
            args,
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(out.stdout)
}

/// Running totals across the whole replay.
#[derive(Default)]
struct Totals {
    commits: usize,
    first_ingests: usize,
    first_nodes: usize,
    reingests: usize,
    kept: usize,
    updated: usize,
    inserted: usize,
    deleted: usize,
    rebalanced: usize,
    /// Re-ingests that changed nothing at all — a commit touching a file's bytes but
    /// not any node boundary is possible, and counting it as work would flatter nothing.
    no_ops: usize,
    deletes: usize,
    /// Per re-ingest: what fraction of the file's nodes the edit touched.
    touched_frac: Vec<f64>,
    unreadable: usize,
    /// Upper bound on re-ingests that could have taken `reconcile`'s cheap positional
    /// path instead of a real LCS. Trimming only shrinks the product, so a zero here
    /// proves every reported alignment was a true diff rather than a budget artifact.
    lcs_budget_risk: usize,
    max_nodes: usize,
}

impl Totals {
    fn fold(&mut self, st: &Update, node_count: usize) {
        self.reingests += 1;
        self.kept += st.kept;
        self.updated += st.updated;
        self.inserted += st.inserted;
        self.deleted += st.deleted;
        if st.rebalanced {
            self.rebalanced += 1;
        }
        if st.touched() == 0 {
            self.no_ops += 1;
        }
        if node_count > 0 {
            self.touched_frac.push(st.touched() as f64 / node_count as f64);
        }
    }
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

#[test]
#[ignore = "needs a local git clone; set HISTORY_REPO"]
fn replay_history() -> anyhow::Result<()> {
    let repo = std::env::var("HISTORY_REPO")
        .unwrap_or_else(|_| format!("{}/.cache/corrode-fixtures/curl", std::env::var("HOME").unwrap()));
    let repo = Path::new(&repo);
    let limit: usize = std::env::var("HISTORY_COMMITS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);

    // Fail loudly rather than reporting a clean zero-commit replay.
    if !repo.join(".git").exists() && !repo.join("HEAD").exists() {
        anyhow::bail!("not a git repo: {}", repo.display());
    }

    // Newest `limit` commits, replayed oldest-first so each reconcile sees the state the
    // previous commit left. First-parent only: merges re-report changes already seen on
    // the branch, which would double-count edits that never happened twice.
    let out = git(repo, &["rev-list", "--first-parent", "-n", &limit.to_string(), "HEAD"])?;
    let commits: Vec<String> =
        String::from_utf8(out)?.lines().rev().map(str::to_owned).collect();
    anyhow::ensure!(!commits.is_empty(), "no commits in {}", repo.display());

    let mut live: HashMap<String, Vec<Node>> = HashMap::new();
    let mut t = Totals::default();
    let start = std::time::Instant::now();

    for sha in &commits {
        t.commits += 1;
        // -M resolves renames; a rename arrives as R<score> old\tnew and is replayed as
        // a delete plus an add, because the path IS the node id prefix.
        let out = git(repo, &["diff-tree", "--no-commit-id", "--name-status", "-r", "-M", sha])?;
        for line in String::from_utf8_lossy(&out).lines() {
            let mut f = line.split('\t');
            let status = f.next().unwrap_or("");
            let (old, new) = (f.next(), f.next());
            let Some(first) = old else { continue };

            if status.starts_with('R') {
                live.remove(first);
                t.deletes += 1;
            }
            let path = if status.starts_with('R') { new.unwrap_or(first) } else { first };

            if status.starts_with('D') {
                if live.remove(path).is_some() {
                    t.deletes += 1;
                }
                continue;
            }

            let Ok(blob) = git(repo, &[ "show", &format!("{sha}:{path}") ]) else {
                t.unreadable += 1;
                continue;
            };
            let Ok(src) = String::from_utf8(blob) else {
                t.unreadable += 1;
                continue;
            };
            let lang = projection::for_path(path);
            let Ok((items, _)) = lang.spans(&src) else {
                t.unreadable += 1;
                continue;
            };
            let fresh = projection::nodes_from_items(path, &src, &items);

            match live.get(path) {
                None => {
                    t.first_ingests += 1;
                    t.first_nodes += fresh.len();
                    let (nodes, _) = projection::update::reconcile(&[], &fresh);
                    live.insert(path.to_string(), nodes);
                }
                Some(stored) => {
                    if stored.len().saturating_mul(fresh.len()) > 4_000_000 {
                        t.lcs_budget_risk += 1;
                    }
                    t.max_nodes = t.max_nodes.max(fresh.len());
                    let (nodes, st) = projection::update::reconcile(stored, &fresh);
                    // The projection must survive the update, or the key assignment is
                    // wrong in a way no counter would show.
                    assert_eq!(
                        projection::project(&nodes).0,
                        src,
                        "{path} at {sha}: reconciled nodes no longer project to the file"
                    );
                    assert!(
                        nodes.windows(2).all(|w| w[0].order < w[1].order),
                        "{path} at {sha}: order keys are not strictly increasing"
                    );
                    t.fold(&st, nodes.len());
                    live.insert(path.to_string(), nodes);
                }
            }
        }
    }

    t.touched_frac.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = t.touched_frac.get(t.touched_frac.len() / 2).copied().unwrap_or(0.0);
    let p99 = t.touched_frac.get(t.touched_frac.len() * 99 / 100).copied().unwrap_or(0.0);
    let mutations = t.updated + t.inserted;

    eprintln!("\nrepo {} — {} commits in {:.1}s", repo.display(), t.commits, start.elapsed().as_secs_f64());
    eprintln!("live files {}, first ingests {} ({} nodes), deletes {}, unreadable {}",
        live.len(), t.first_ingests, t.first_nodes, t.deletes, t.unreadable);
    eprintln!("\nre-ingests {}", t.reingests);
    eprintln!("  kept       {:>9}", t.kept);
    eprintln!("  updated    {:>9}  ({:.1}% of mutations)", t.updated, pct(t.updated, mutations));
    eprintln!("  inserted   {:>9}  ({:.1}% of mutations)", t.inserted, pct(t.inserted, mutations));
    eprintln!("  deleted    {:>9}", t.deleted);
    eprintln!("  no-ops     {:>9}  ({:.1}% of re-ingests)", t.no_ops, pct(t.no_ops, t.reingests));
    eprintln!("  rebalanced {:>9}  ({:.3}% of re-ingests)", t.rebalanced, pct(t.rebalanced, t.reingests));
    eprintln!("  max nodes/file {:>5}, LCS-budget risk {}", t.max_nodes, t.lcs_budget_risk);
    eprintln!("\nnodes touched per re-ingest: median {:.2}%, p99 {:.2}%", median * 100.0, p99 * 100.0);
    eprintln!("insert:update = 1:{:.1}", t.updated as f64 / t.inserted.max(1) as f64);
    Ok(())
}
