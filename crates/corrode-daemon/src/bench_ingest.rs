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
                .map(|n| format!("code:{rel}#{}", n.order).len())
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
        // Node-count distribution decides whether ordering needs anything cleverer
        // than a dense index: renumbering on insert is O(nodes in file), which is
        // free at 10 and painful at 10,000.
        let mut counts: Vec<usize> = Vec::new();
        for rel in &files {
            let Ok(src) = std::fs::read_to_string(std::path::Path::new(&repo).join(rel)) else {
                continue;
            };
            let lang = projection::for_path(rel);
            if let Ok(fw) = ingest::file(lang.as_ref(), rel, &src) {
                counts.push(fw.code.len());
            }
        }
        counts.sort_unstable();
        let at = |q: f64| counts[((counts.len() as f64 - 1.0) * q) as usize];
        eprintln!(
            "nodes/file: p50 {} p90 {} p99 {} max {}",
            at(0.50),
            at(0.90),
            at(0.99),
            counts.last().copied().unwrap_or(0)
        );
        assert!(agg.files > 0, "nothing ingested");
    }
}

#[cfg(test)]
mod archive_tests {
    use crate::projection::{self, archive, ingest};
    use std::time::Instant;

    /// Ingest an archive without unpacking it.
    ///
    /// ```text
    /// CORRODE_SCAN_ARCHIVE=fixtures/linux-7.2.2.tar.xz \
    ///   cargo test -p corrode-daemon ingest_archive -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "streams a whole archive; run explicitly with CORRODE_SCAN_ARCHIVE"]
    fn ingest_archive() {
        let Ok(path) = std::env::var("CORRODE_SCAN_ARCHIVE") else {
            eprintln!("set CORRODE_SCAN_ARCHIVE=<path to .tar/.tar.xz/.tar.gz/.tar.zst>");
            return;
        };
        let mut by: std::collections::BTreeMap<&'static str, (usize, usize, usize, usize)> =
            Default::default();
        let (mut exact, mut mismatched, mut failed) = (0usize, 0usize, 0usize);
        let mut bytes = 0usize;
        let wall = Instant::now();

        let (seen, skipped) = archive::for_each_file(std::path::Path::new(&path), |e| {
            let lang = projection::for_path(e.path);
            let slot = by.entry(lang.name()).or_default();
            slot.0 += 1;
            bytes += e.text.len();
            match ingest::file(lang.as_ref(), e.path, e.text) {
                Err(_) => failed += 1,
                Ok(fw) => {
                    slot.1 += fw.code.len();
                    slot.2 += fw.comments.len();
                    slot.3 += fw.comments.iter().filter(|c| c.describes_kind.is_some()).count();
                    // The invariant that must survive a different transport.
                    if ingest::project(&fw) == e.text {
                        exact += 1;
                    } else {
                        mismatched += 1;
                        if mismatched <= 3 {
                            eprintln!("  MISMATCH [{}] {}", lang.name(), e.path);
                        }
                    }
                }
            }
        })
        .expect("archive readable");

        let secs = wall.elapsed().as_secs_f64();
        eprintln!("=== archive ingest: {path} ===");
        eprintln!("{:<12} {:>8} {:>9} {:>10} {:>9}", "backend", "files", "nodes", "comments", "bound");
        for (name, (f, n, c, b)) in &by {
            eprintln!("{name:<12} {f:>8} {n:>9} {c:>10} {b:>9}");
        }
        eprintln!(
            "entries {seen}, byte-exact {exact}, mismatched {mismatched}, ingest errors {failed}, non-UTF-8 {skipped}"
        );
        eprintln!(
            "{:.1} MB in {:.1}s ({:.1} MB/s, {:.0} files/s) — never unpacked",
            bytes as f64 / 1e6,
            secs,
            (bytes as f64 / 1e6) / secs.max(1e-9),
            seen as f64 / secs.max(1e-9),
        );
        assert_eq!(mismatched, 0, "projection was not byte-exact from the archive");
    }
}

#[cfg(test)]
mod sweep {
    use crate::projection::{self, archive, ingest};

    #[derive(Default, Clone)]
    struct Ext {
        files: usize,
        bytes: usize,
        comments: usize,
        bound: usize,
        backend: &'static str,
        /// Files where the backend found no comments at all — the signal that the
        /// marker guess is wrong for this type, as distinct from a file that genuinely
        /// has none.
        commentless: usize,
    }

    /// What file types is a tree actually made of, and what does each currently get?
    ///
    /// The point is the backend worklist: a type's weight is files and bytes, but its
    /// VALUE is comments that could become queryable if it had a grammar. Sorted by
    /// unbound comments, because that is what a backend would buy.
    ///
    /// ```text
    /// CORRODE_SCAN_ARCHIVE=fixtures/linux-7.2.2.tar.xz \
    ///   cargo test -p corrode-daemon file_type_sweep -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "streams a whole archive"]
    fn file_type_sweep() {
        let Ok(path) = std::env::var("CORRODE_SCAN_ARCHIVE") else {
            eprintln!("set CORRODE_SCAN_ARCHIVE=<archive>");
            return;
        };
        let mut by: std::collections::BTreeMap<String, Ext> = Default::default();
        let (mut files, mut bytes) = (0usize, 0usize);

        let (_, skipped) = archive::for_each_file(std::path::Path::new(&path), |e| {
            let base = e.path.rsplit('/').next().unwrap_or(e.path);
            // Group by extension, or by filename when there is none — `Makefile` and
            // `Kconfig` are file *types* in a kernel tree, not oddities.
            let key = match base.rsplit_once('.') {
                Some((stem, ext)) if !stem.is_empty() => format!(".{ext}"),
                _ => base.to_string(),
            };
            let lang = projection::for_path(e.path);
            let slot = by.entry(key).or_default();
            slot.backend = lang.name();
            slot.files += 1;
            slot.bytes += e.text.len();
            files += 1;
            bytes += e.text.len();
            if let Ok(fw) = ingest::file(lang.as_ref(), e.path, e.text) {
                slot.comments += fw.comments.len();
                slot.bound += fw.comments.iter().filter(|c| c.describes_kind.is_some()).count();
                if fw.comments.is_empty() {
                    slot.commentless += 1;
                }
            }
        })
        .expect("archive readable");

        let mut v: Vec<(&String, &Ext)> = by.iter().collect();

        eprintln!("=== file types: {path} ===");
        eprintln!("{files} files, {:.0} MB, {skipped} non-UTF-8\n", bytes as f64 / 1e6);

        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.files));
        eprintln!("-- by file count --");
        eprintln!("{:<16} {:>8} {:>8} {:>12} {:>10} {:>9}", "type", "files", "MB", "backend", "comments", "bound");
        for (k, e) in v.iter().take(20) {
            eprintln!(
                "{:<16} {:>8} {:>8.1} {:>12} {:>10} {:>9}",
                k, e.files, e.bytes as f64 / 1e6, e.backend, e.comments, e.bound
            );
        }

        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.comments - e.bound));
        eprintln!("\n-- by UNBOUND comments (what a backend would buy) --");
        eprintln!("{:<16} {:>8} {:>12} {:>12}", "type", "files", "backend", "unbound");
        for (k, e) in v.iter().take(12) {
            if e.comments == e.bound {
                continue;
            }
            eprintln!("{:<16} {:>8} {:>12} {:>12}", k, e.files, e.backend, e.comments - e.bound);
        }

        // A type where most files yield NO comments is usually a wrong marker guess
        // rather than an undocumented corpus.
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.commentless));
        eprintln!("\n-- types where the marker guess looks wrong (files with zero comments) --");
        eprintln!("{:<16} {:>8} {:>12} {:>12}", "type", "files", "backend", "commentless");
        for (k, e) in v.iter().take(10) {
            if e.files < 20 || e.commentless * 2 < e.files {
                continue;
            }
            eprintln!("{:<16} {:>8} {:>12} {:>12}", k, e.files, e.backend, e.commentless);
        }
        assert!(files > 0);
    }
}
