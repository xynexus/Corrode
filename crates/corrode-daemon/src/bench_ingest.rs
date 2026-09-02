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

        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        // One lock around the tallies rather than per-worker maps: the work is the
        // ingest, and the accumulate is a handful of adds.
        let acc = std::sync::Mutex::new((by, exact, mismatched, failed, bytes));
        let (seen, skipped) =
            archive::par_for_each_file(std::path::Path::new(&path), workers, |e| {
                let lang = projection::for_path(e.path);
                let result = ingest::file(lang.as_ref(), e.path, e.text);
                let exact_here = result
                    .as_ref()
                    .map(|fw| ingest::project(fw) == e.text)
                    .unwrap_or(false);
                let mut g = acc.lock().unwrap();
                let slot = g.0.entry(lang.name()).or_default();
                slot.0 += 1;
                g.4 += e.text.len();
                match result {
                    Err(_) => g.3 += 1,
                    Ok(fw) => {
                        let slot = g.0.entry(lang.name()).or_default();
                        slot.1 += fw.code.len();
                        slot.2 += fw.comments.len();
                        slot.3 += fw
                            .comments
                            .iter()
                            .filter(|c| c.describes_kind.is_some())
                            .count();
                        if exact_here {
                            g.1 += 1;
                        } else {
                            g.2 += 1;
                            if g.2 <= 3 {
                                eprintln!("  MISMATCH [{}] {}", lang.name(), e.path);
                            }
                        }
                    }
                }
            })
            .expect("archive readable");
        let (by, exact, mismatched, failed, bytes) = acc.into_inner().unwrap();
        eprintln!("({workers} workers)");

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

#[cfg(test)]
mod c_profile {
    use crate::projection::{self, Language};
    use std::time::Instant;

    /// Where does the C backend's time go on real kernel sources?
    #[test]
    #[ignore = "probe"]
    fn profile_c_backend() {
        let dir = std::env::var("CORRODE_C_DIR").unwrap_or_default();
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { walk(&p, out); }
                else if matches!(p.extension().and_then(|e| e.to_str()), Some("c") | Some("h")) {
                    out.push(p);
                }
            }
        }
        walk(std::path::Path::new(&dir), &mut files);
        files.truncate(300);
        let c = projection::c::C;
        let (mut lex_t, mut items_t, mut anchors_t, mut bytes) = (0u128, 0u128, 0u128, 0usize);
        let mut bind_t = 0u128;
        let mut worst = (0u128, String::new(), 0usize);
        for f in &files {
            let Ok(src) = std::fs::read_to_string(f) else { continue };
            bytes += src.len();
            let t = Instant::now(); let _ = c.comments(&src); lex_t += t.elapsed().as_micros();
            let t = Instant::now(); let it = c.items(&src).unwrap(); items_t += t.elapsed().as_micros();
            let t = Instant::now(); let an = c.anchors(&src).unwrap(); let a = t.elapsed().as_micros();
            anchors_t += a;
            // The phase the benchmark could not see: binding, with REAL anchor counts.
            let cs = c.comments(&src);
            let nodes = crate::projection::nodes_from_items("t.c", &src, &it);
            let t = Instant::now();
            let _edges = crate::projection::bind(&src, &cs, &an, &nodes);
            let bt = t.elapsed().as_micros();
            bind_t += bt;
            if bt > worst.0 { worst = (bt, f.display().to_string(), an.len()); }
        }
        eprintln!("{} files, {:.1} MB", files.len(), bytes as f64 / 1e6);
        eprintln!("comments {}ms  items {}ms  anchors {}ms  BIND {}ms", lex_t/1000, items_t/1000, anchors_t/1000, bind_t/1000);
        eprintln!("worst bind: {}ms {} ({} anchors)", worst.0/1000, worst.1, worst.2);
    }
}

#[cfg(test)]
mod one_file {
    use crate::projection::{self, ingest, Language};
    use std::time::Instant;
    #[test]
    #[ignore = "probe"]
    fn time_one_file() {
        let f = std::env::var("PROBE_FILE").unwrap();
        let src = std::fs::read_to_string(&f).unwrap();
        eprintln!("{:.1} MB", src.len() as f64 / 1e6);
        let c = projection::c::C;
        let t = Instant::now(); let items = c.items(&src).unwrap();
        eprintln!("items   {:>7}ms -> {} items", t.elapsed().as_millis(), items.len());
        let t = Instant::now(); let an = c.anchors(&src).unwrap();
        eprintln!("anchors {:>7}ms -> {} anchors", t.elapsed().as_millis(), an.len());
        let t = Instant::now(); let cs = c.comments(&src);
        eprintln!("comments{:>7}ms -> {} comments", t.elapsed().as_millis(), cs.len());
        let t = Instant::now(); let nodes = projection::nodes_from_items("x.h", &src, &items);
        eprintln!("nodes   {:>7}ms -> {} nodes", t.elapsed().as_millis(), nodes.len());
        let t = Instant::now(); let _ = projection::bind(&src, &cs, &an, &nodes);
        eprintln!("bind    {:>7}ms", t.elapsed().as_millis());
        let t = Instant::now(); let fw = ingest::file(&c, "x.h", &src).unwrap();
        eprintln!("ingest  {:>7}ms", t.elapsed().as_millis());
        let t = Instant::now(); let ok = ingest::project(&fw) == src;
        eprintln!("project {:>7}ms (exact: {ok})", t.elapsed().as_millis());
    }
}

/// Does the store actually take a real repo, and at what rate?
///
/// Everything measured before this landed in memory. LMDB write throughput, on-disk
/// amplification, and whether a 24 MB single-node file is even writable are the first
/// things likely to break, and none of them are visible from an in-memory census.
#[cfg(all(test, feature = "helix"))]
mod store_scale {
    use crate::graph::GraphStore;
    use crate::projection::{self, ingest};
    use std::time::Instant;

    #[test]
    #[ignore = "probe: needs a repo and --features helix"]
    fn ingest_a_repo_into_a_live_store() {
        let repo = std::path::PathBuf::from(std::env::var("CORRODE_REPO").unwrap());
        let dir = std::env::temp_dir().join(format!("corrode-scale-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let store = crate::graph::embedded::HelixStore::open(dir.to_str().unwrap()).unwrap();
        let limit: usize = std::env::var("SCALE_FILES").ok().and_then(|v| v.parse().ok()).unwrap_or(400);

        let out = std::process::Command::new("git")
            .arg("-C").arg(&repo).args(["ls-files", "-z"]).output().unwrap();
        let files: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .split('\0').filter(|s| !s.is_empty()).map(str::to_owned).collect();

        let (mut ok, mut failed, mut nodes, mut comments, mut bytes) = (0, 0, 0, 0, 0usize);
        let t = Instant::now();
        for rel in files.iter().take(limit) {
            let Ok(src) = std::fs::read_to_string(repo.join(rel)) else { continue };
            let lang = projection::for_path(rel);
            let Ok(fw) = ingest::file(lang.as_ref(), rel, &src) else { continue };
            nodes += fw.code.len();
            comments += fw.comments.len();
            bytes += src.len();
            match store.replace_file(&fw) {
                Ok(()) => ok += 1,
                Err(e) => {
                    if failed == 0 { eprintln!("first write failure on {rel}: {e}") }
                    failed += 1;
                }
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let on_disk: u64 = std::fs::read_dir(&dir).into_iter().flatten().flatten()
            .filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
        eprintln!(
            "\n{ok} files written ({failed} failed), {nodes} code + {comments} comment nodes, \
             {:.1} MB source in {secs:.1}s ({:.0} nodes/s)",
            bytes as f64 / 1e6, (nodes + comments) as f64 / secs.max(0.001)
        );
        eprintln!("  store on disk {:.1} MB — {:.1}x the source", on_disk as f64 / 1e6, on_disk as f64 / bytes.max(1) as f64);

        // Re-read one file back and check the graph still composes it byte-exactly.
        let sample = files.iter().take(limit)
            .find(|r| std::fs::read_to_string(repo.join(r)).is_ok_and(|s| s.len() > 200));
        if let Some(rel) = sample {
            let src = std::fs::read_to_string(repo.join(rel)).unwrap();
            let back = projection::project(&store.file_nodes(rel).unwrap()).0;
            assert_eq!(back, src, "{rel} did not compose back byte-exactly from the store");
            eprintln!("  round trip from store: {rel} byte-exact");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// How much prose does the doc mapping actually connect, on a tree that has plenty?
#[cfg(test)]
mod doc_mapping {
    use crate::projection::{archive, docmap, for_path};
    use std::collections::BTreeSet;

    #[test]
    #[ignore = "probe: needs the kernel archive"]
    fn map_kernel_docs_to_source_dirs() {
        // Absolute, from the manifest dir: cargo runs tests with the PACKAGE as cwd, so
        // a workspace-relative default resolves to nothing.
        let archive_path = std::env::var("INGEST_ARCHIVE").unwrap_or_else(|_| {
            format!("{}/../../fixtures/linux-7.2.2.tar.xz", env!("CARGO_MANIFEST_DIR"))
        });
        let path = std::path::Path::new(&archive_path);

        // Pass 1: the repo's real directory set. Links are confirmed against this, so a
        // path that merely looks plausible produces no edge.
        let mut dirs: BTreeSet<String> = BTreeSet::new();
        archive::for_each_file(path, |e| {
            let mut p = e.path;
            while let Some((d, _)) = p.rsplit_once('/') {
                if !dirs.insert(d.to_string()) {
                    break;
                }
                p = d;
            }
        })
        .unwrap();

        // Pass 2: the links themselves.
        let (mut scanned, mut linked, mut edges, mut own_dir) = (0usize, 0usize, 0usize, 0usize);
        let mut by_kind: std::collections::BTreeMap<&str, usize> = Default::default();
        archive::for_each_file(path, |e| {
            let name = e.path.rsplit('/').next().unwrap_or(e.path);
            let is_prose = matches!(
                name.rsplit_once('.').map(|(_, x)| x).unwrap_or(""),
                "rst" | "txt" | "md"
            );
            let stem = name.split_once('.').map(|(s, _)| s).unwrap_or(name);
            let is_cfg = matches!(stem, "Kconfig" | "Makefile" | "Kbuild" | "README");
            if !is_prose && !is_cfg {
                return;
            }
            scanned += 1;
            let d = docmap::describes(e.path, e.text, &dirs);
            if !d.is_empty() {
                linked += 1;
                edges += d.len();
                *by_kind.entry(for_path(e.path).name()).or_default() += 1;
                if is_cfg {
                    own_dir += 1;
                }
            }
        })
        .unwrap();

        eprintln!("\n{} directories in the tree", dirs.len());
        eprintln!("  prose+config files scanned {scanned}");
        eprintln!("  files linked to at least one dir {linked} ({:.0}%)", 100.0 * linked as f64 / scanned.max(1) as f64);
        eprintln!("  describes edges {edges}");
        eprintln!("  of which config/build files describing their own dir {own_dir}");
        eprintln!("  linked files by backend: {by_kind:?}");
        assert!(edges > 0, "the doc mapping produced no links at all");
    }
}
