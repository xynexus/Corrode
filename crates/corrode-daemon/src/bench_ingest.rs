//! Ingest metrics: throughput, storage amplification, and where the time goes.
//!
//! Absorbing a large tree is the case that decides whether the projection is usable.
//! A kernel-sized repository is tens of thousands of files, and three questions matter
//! before anything depends on it:
//!
//! 1. **Throughput** — can a tree be ingested in a sane wall-clock time?
//! 2. **Amplification** — nodes store text VERBATIM, so the graph holds at least the
//!    source again. How much *more* than that, in ids and per-node overhead?
//! 3. **Where the time goes** — parsing, comment scanning, or binding. Optimising the
//!    wrong phase is the usual outcome of not measuring this.
//!
//! Run explicitly, because it walks whole repositories:
//!
//! ```text
//! CORRODE_SCAN_REPO=/path/to/linux \
//!   cargo test -p corrode-daemon ingest_benchmark -- --ignored --nocapture
//! ```

#[cfg(test)]
use std::time::{Duration, Instant};

/// Per-backend accumulation.
#[cfg(test)]
#[derive(Default, Clone)]
struct Bench {
    files: usize,
    src_bytes: usize,
    node_text_bytes: usize,
    id_bytes: usize,
    nodes: usize,
    comments: usize,
    bound: usize,
    read: Duration,
    items: Duration,
    anchors: Duration,
    comment_scan: Duration,
    bind: Duration,
    /// Slowest files seen, for spotting pathological inputs rather than averages.
    slowest: Vec<(Duration, String)>,
}

#[cfg(test)]
impl Bench {
    fn note_slow(&mut self, d: Duration, path: &str) {
        self.slowest.push((d, path.to_string()));
        self.slowest.sort_by_key(|(d, _)| std::cmp::Reverse(*d));
        self.slowest.truncate(3);
    }
    fn total(&self) -> Duration {
        self.read + self.items + self.anchors + self.comment_scan + self.bind
    }
}

#[cfg(test)]
fn pct(part: Duration, whole: Duration) -> f64 {
    if whole.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / whole.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{self, bind, ingest};
    use std::collections::BTreeMap;

    #[test]
    #[ignore = "walks a whole repository; run explicitly with CORRODE_SCAN_REPO"]
    fn ingest_benchmark() {
        let repo = std::env::var("CORRODE_SCAN_REPO").unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_string_lossy()
                .into_owned()
        });
        let listed = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["ls-files", "-s", "-z"])
            .output()
            .expect("git ls-files");
        let files: Vec<String> = String::from_utf8_lossy(&listed.stdout)
            .split('\0')
            .filter_map(|r| {
                let (m, p) = r.split_once('\t')?;
                matches!(m.split_whitespace().next()?, "100644" | "100755").then(|| p.to_string())
            })
            .collect();

        let mut by: BTreeMap<&'static str, Bench> = BTreeMap::new();
        let mut skipped = 0usize;
        let wall = Instant::now();

        for rel in &files {
            let t0 = Instant::now();
            let Ok(src) = std::fs::read_to_string(std::path::Path::new(&repo).join(rel)) else {
                skipped += 1;
                continue;
            };
            let read = t0.elapsed();
            let lang = projection::for_path(rel);
            let b = by.entry(lang.name()).or_default();

            // Timed as the ingest path actually runs it: one `spans` call, so a
            // backend that parses once is measured parsing once.
            let t = Instant::now();
            let Ok((item_spans, anchors)) = lang.spans(&src) else {
                skipped += 1;
                continue;
            };
            let items = t.elapsed();
            let nodes = projection::nodes_from_items(rel, &src, &item_spans);
            let anchors_t = Duration::ZERO;

            let t = Instant::now();
            let cs = lang.comments(&src);
            let comment_scan = t.elapsed();

            let t = Instant::now();
            let edges = bind(&src, &cs, &anchors, &nodes);
            let bind_t = t.elapsed();

            // What the store would hold: verbatim text plus the ids addressing it.
            let node_text: usize = nodes.iter().map(|n| n.text.len()).sum();
            let ids: usize = nodes
                .iter()
                .map(|n| format!("code:{rel}#{}", n.ordinal).len())
                .sum::<usize>()
                + (0..edges.len()).map(|i| format!("comment:{rel}#{i}").len()).sum::<usize>();

            b.files += 1;
            b.src_bytes += src.len();
            b.node_text_bytes += node_text;
            b.id_bytes += ids;
            b.nodes += nodes.len();
            b.comments += edges.len();
            b.bound += edges.iter().filter(|e| e.target.is_some()).count();
            b.read += read;
            b.items += items;
            b.anchors += anchors_t;
            b.comment_scan += comment_scan;
            b.bind += bind_t;
            b.note_slow(read + items + anchors_t + comment_scan + bind_t, rel);
        }

        let elapsed = wall.elapsed();
        let mut agg = Bench::default();
        eprintln!("=== ingest benchmark: {repo} ===");
        eprintln!(
            "{:<12} {:>7} {:>9} {:>8} {:>7} {:>8}",
            "backend", "files", "MB", "MB/s", "files/s", "amp"
        );
        for (name, b) in &by {
            let secs = b.total().as_secs_f64().max(1e-9);
            eprintln!(
                "{:<12} {:>7} {:>9.1} {:>8.1} {:>7.0} {:>7.2}x",
                name,
                b.files,
                b.src_bytes as f64 / 1e6,
                (b.src_bytes as f64 / 1e6) / secs,
                b.files as f64 / secs,
                (b.node_text_bytes + b.id_bytes) as f64 / b.src_bytes.max(1) as f64,
            );
            agg.files += b.files;
            agg.src_bytes += b.src_bytes;
            agg.node_text_bytes += b.node_text_bytes;
            agg.id_bytes += b.id_bytes;
            agg.nodes += b.nodes;
            agg.comments += b.comments;
            agg.bound += b.bound;
            agg.read += b.read;
            agg.items += b.items;
            agg.anchors += b.anchors;
            agg.comment_scan += b.comment_scan;
            agg.bind += b.bind;
        }

        let t = agg.total();
        eprintln!();
        eprintln!(
            "totals: {} files, {:.1} MB, {} nodes, {} comments ({} bound), {} skipped",
            agg.files,
            agg.src_bytes as f64 / 1e6,
            agg.nodes,
            agg.comments,
            agg.bound,
            skipped
        );
        eprintln!(
            "wall {:.2}s  ({:.1} MB/s, {:.0} files/s)",
            elapsed.as_secs_f64(),
            (agg.src_bytes as f64 / 1e6) / elapsed.as_secs_f64().max(1e-9),
            agg.files as f64 / elapsed.as_secs_f64().max(1e-9),
        );
        eprintln!(
            "phases: read {:.0}%  parse+anchors {:.0}%  comments {:.0}%  bind {:.0}%",
            pct(agg.read, t),
            pct(agg.items, t),
            pct(agg.comment_scan, t),
            pct(agg.bind, t),
        );
        eprintln!(
            "storage: source {:.1} MB -> nodes {:.1} MB + ids {:.1} MB = {:.2}x",
            agg.src_bytes as f64 / 1e6,
            agg.node_text_bytes as f64 / 1e6,
            agg.id_bytes as f64 / 1e6,
            (agg.node_text_bytes + agg.id_bytes) as f64 / agg.src_bytes.max(1) as f64,
        );
        // Verbatim storage means the text is held once more, by design. Anything far
        // above 1x is per-node overhead, which is the number to watch as node
        // granularity gets finer.
        eprintln!(
            "  {:.0} bytes/node average, {:.1} nodes per file",
            agg.node_text_bytes as f64 / agg.nodes.max(1) as f64,
            agg.nodes as f64 / agg.files.max(1) as f64,
        );
        for (name, b) in &by {
            if let Some((d, p)) = b.slowest.first() {
                eprintln!("  slowest [{name}] {:.0}ms {p}", d.as_secs_f64() * 1000.0);
            }
        }
        assert!(agg.files > 0, "nothing ingested");
    }
}
