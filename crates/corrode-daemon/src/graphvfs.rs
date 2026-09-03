//! Step 7f: serving files FROM the graph.
//!
//! Everything so far ran one direction — source into nodes. `graph-model.md` makes files
//! a projection of the graph, and this is the return trip: `file_nodes` hands back a
//! file's code nodes in order, `project` composes them, and the bytes are the file.
//!
//! Composition is already proven exact in both directions (94,750 of 94,750 kernel
//! entries, and every one of 28,881 reconciles across 5,000 curl commits), so the risk
//! here was never fidelity. It is **staleness**: a graph that has not seen an edit will
//! serve confidently wrong bytes, and an agent editing against them produces a patch
//! that does not apply. That is worse than any error this replaces.
//!
//! So the wrapper is honest about not knowing. It serves the graph only for files the
//! graph actually holds, falls through to the inner VFS for everything else, and — when
//! `CORRODE_VFS_VERIFY` is on — compares its own answer against the inner one and reports
//! every divergence instead of silently preferring itself. Off by default
//! (`CORRODE_VFS_GRAPH`), like `CORRODE_SANDBOX`, so existing behaviour is unchanged
//! until someone opts in.

use crate::graph::GraphStore;
use crate::vfs::Vfs;
use async_trait::async_trait;
use corrode_core::{FileNodeView, ProjectionMode};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Reads a file by composing its graph nodes, falling back to `inner`.
pub struct GraphVfs {
    store: Arc<dyn GraphStore>,
    inner: Arc<dyn Vfs>,
    /// Compare every graph-served read against the inner VFS and report divergence.
    /// Costs a second read per file, so it is a diagnostic rather than the default.
    verify: bool,
    served: AtomicUsize,
    fell_through: AtomicUsize,
    diverged: AtomicUsize,
}

impl GraphVfs {
    pub fn new(store: Arc<dyn GraphStore>, inner: Arc<dyn Vfs>) -> Self {
        Self {
            store,
            inner,
            verify: env_on("CORRODE_VFS_VERIFY"),
            served: AtomicUsize::new(0),
            fell_through: AtomicUsize::new(0),
            diverged: AtomicUsize::new(0),
        }
    }

    /// `(served from graph, fell through, diverged)` — so a caller can report how much
    /// of a session the graph actually backed rather than assuming it backed all of it.
    #[allow(dead_code)]
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.served.load(Ordering::Relaxed),
            self.fell_through.load(Ordering::Relaxed),
            self.diverged.load(Ordering::Relaxed),
        )
    }

    /// The file's bytes as the graph holds them, if it holds it at all.
    fn compose(&self, path: &str) -> Option<Vec<u8>> {
        let nodes = self.store.file_nodes(path).ok()?;
        // No nodes means "not ingested", which is a fall-through, not an empty file. A
        // genuinely empty file has no nodes either — and composing it yields the same
        // empty bytes the inner VFS would, so preferring the fall-through costs nothing
        // and avoids inventing an empty file for a path the graph has never seen.
        if nodes.is_empty() {
            return None;
        }
        Some(crate::projection::project(&nodes).0.into_bytes())
    }
}

fn env_on(key: &str) -> bool {
    matches!(
        std::env::var(key).unwrap_or_default().to_ascii_lowercase().as_str(),
        "1" | "true" | "on"
    )
}

/// Is the graph-backed VFS enabled?
pub fn enabled() -> bool {
    env_on("CORRODE_VFS_GRAPH")
}

#[async_trait]
impl Vfs for GraphVfs {
    async fn read(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        let Some(bytes) = self.compose(path) else {
            self.fell_through.fetch_add(1, Ordering::Relaxed);
            return self.inner.read(path).await;
        };
        if self.verify {
            // Divergence means the graph missed an edit. Report it rather than resolve
            // it: silently preferring either side is how an agent ends up editing text
            // that does not exist, and which side is right depends on why they differ.
            if let Ok(disk) = self.inner.read(path).await {
                if disk != bytes {
                    self.diverged.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "vfs: graph and disk disagree on {path} ({} vs {} bytes) — graph is stale",
                        bytes.len(),
                        disk.len()
                    );
                }
            }
        }
        self.served.fetch_add(1, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn stat(&self, path: &str) -> anyhow::Result<FileNodeView> {
        match self.compose(path) {
            // Size comes from the composed bytes, not from disk: a stat that disagrees
            // with the following read is worse than either answer alone, and FUSE will
            // truncate a read to the size stat promised.
            Some(bytes) => Ok(FileNodeView {
                path: path.to_string(),
                is_dir: false,
                bytes: bytes.len() as u64,
                node_id: Some(format!("file:{path}")),
                mode: Some(ProjectionMode::Composed),
            }),
            None => self.inner.stat(path).await,
        }
    }

    /// Listing and the corpus stay with the inner VFS.
    ///
    /// The graph knows only the files that have been ingested, so answering from it
    /// would make directories look emptier than they are — and `tracked_files` defines
    /// the search corpus, where under-reporting silently loses results. Reading is
    /// per-path and can fall through honestly; enumeration cannot.
    async fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>> {
        self.inner.list(dir).await
    }

    async fn tracked_files(&self) -> anyhow::Result<Vec<String>> {
        self.inner.tracked_files().await
    }

    /// Writes go to the inner VFS. The graph catches up through `ingest_written`, which
    /// reconciles against the stored nodes — so a write is not lost, it is absorbed on
    /// the ingest path that already exists rather than through a second one here.
    async fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()> {
        self.inner.write(path, contents).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::PassthroughVfs;

    /// A store that holds exactly the nodes it was handed.
    struct FakeStore(Vec<crate::projection::Node>);

    impl GraphStore for FakeStore {
        fn neighbors(&self, _: &str) -> anyhow::Result<Vec<corrode_core::GraphNodeView>> {
            Ok(Vec::new())
        }
        fn doc_search(&self, _: &str, _: Option<&[f32]>, _: usize) -> anyhow::Result<Vec<(String, String)>> {
            Ok(Vec::new())
        }
        fn upsert_node(&self, _: &str, _: &str, _: &str) -> anyhow::Result<()> {
            Ok(())
        }
        fn add_edge(&self, _: &str, _: &str, _: &str) -> anyhow::Result<()> {
            Ok(())
        }
        fn replace_doc(&self, _: &crate::graph::DocWrite) -> anyhow::Result<()> {
            Ok(())
        }
        fn list_docs(&self) -> anyhow::Result<Vec<(String, String)>> {
            Ok(Vec::new())
        }
        fn code_nodes(&self) -> anyhow::Result<Vec<(String, String)>> {
            Ok(Vec::new())
        }
        fn file_nodes(&self, path: &str) -> anyhow::Result<Vec<crate::projection::Node>> {
            Ok(self.0.iter().filter(|n| n.path == path).cloned().collect())
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("corrode-gvfs-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[tokio::test]
    async fn a_file_the_graph_holds_is_served_from_nodes() {
        let dir = scratch("served");
        // Disk holds something DIFFERENT, so a pass-through would be visible.
        std::fs::write(dir.join("a.rs"), b"from disk\n").unwrap();
        let src = "fn a() { 1 }\n\nfn b() { 2 }\n";
        let lang = crate::projection::for_path("a.rs");
        let (items, _) = lang.spans(src).unwrap();
        let nodes = crate::projection::nodes_from_items("a.rs", src, &items);

        let vfs = GraphVfs::new(
            Arc::new(FakeStore(nodes)),
            Arc::new(PassthroughVfs::new(&dir)),
        );
        assert_eq!(vfs.read("a.rs").await.unwrap(), src.as_bytes(), "must compose from the graph");
        // stat must agree with read, or FUSE truncates to a size that is not the content.
        assert_eq!(vfs.stat("a.rs").await.unwrap().bytes, src.len() as u64);
        assert_eq!(vfs.counts().0, 1);
    }

    #[tokio::test]
    async fn a_file_the_graph_lacks_falls_through_to_disk() {
        let dir = scratch("fallthrough");
        std::fs::write(dir.join("b.rs"), b"only on disk\n").unwrap();
        let vfs = GraphVfs::new(Arc::new(FakeStore(Vec::new())), Arc::new(PassthroughVfs::new(&dir)));
        assert_eq!(vfs.read("b.rs").await.unwrap(), b"only on disk\n");
        assert_eq!(vfs.counts(), (0, 1, 0), "should have fallen through, not served");
        // And a path in neither place still errors rather than returning empty bytes.
        assert!(vfs.read("nope.rs").await.is_err());
    }

    #[tokio::test]
    async fn listing_and_corpus_come_from_the_inner_vfs() {
        // The graph holds only ingested files, so answering enumeration from it would
        // make directories look emptier than they are and shrink the search corpus.
        let dir = scratch("listing");
        std::fs::write(dir.join("x.rs"), b"x\n").unwrap();
        std::fs::write(dir.join("y.txt"), b"y\n").unwrap();
        // `tracked_files` is `git ls-files`, so exercise the real path rather than a
        // weakened assertion: a scratch dir that is not a repo tests nothing.
        for args in [vec!["init", "-q"], vec!["add", "-A"]] {
            let ok = std::process::Command::new("git")
                .arg("-C").arg(&dir).args(&args).status().map(|s| s.success()).unwrap_or(false);
            assert!(ok, "git {args:?} failed in the scratch repo");
        }
        let vfs = GraphVfs::new(Arc::new(FakeStore(Vec::new())), Arc::new(PassthroughVfs::new(&dir)));
        assert_eq!(vfs.list("").await.unwrap().len(), 2);
        assert_eq!(vfs.tracked_files().await.unwrap().len(), 2);
    }
}

/// End-to-end against a real store: ingest a file, then read it back through the VFS.
#[cfg(all(test, feature = "helix"))]
mod live {
    use super::*;
    use crate::projection::{self, ingest};
    use crate::vfs::PassthroughVfs;

    #[tokio::test]
    async fn ingested_files_read_back_byte_exactly_and_edits_are_detected() {
        let dir = std::env::temp_dir().join(format!("corrode-gvfs-live-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let store = Arc::new(
            crate::graph::embedded::HelixStore::open(dir.join(".graph").to_str().unwrap()).unwrap(),
        );

        // A file with the awkward parts: a doc comment, a raw string, nested braces and
        // trailing whitespace — the things a naive composer loses.
        let path = "src/lib.rs";
        let src = "//! Crate docs.\n\n/// Doc.\npub fn f(x: u32) -> u32 {\n    if x > 0 { x - 1 } else { 0 }\n}\n\nconst S: &str = r#\"a \"quoted\" thing\"#;\n";
        std::fs::write(dir.join(path), src).unwrap();
        let lang = projection::for_path(path);
        store.replace_file(&ingest::file(lang.as_ref(), path, src).unwrap()).unwrap();

        let vfs = GraphVfs::new(store.clone(), Arc::new(PassthroughVfs::new(&dir)));
        let read = vfs.read(path).await.unwrap();
        assert_eq!(
            String::from_utf8(read).unwrap(),
            src,
            "the graph must compose the file back byte-exactly"
        );
        assert_eq!(vfs.stat(path).await.unwrap().bytes, src.len() as u64, "stat must match read");
        assert_eq!(vfs.counts().0, 1, "should have been served from the graph");

        // Now the failure mode this design exists to be honest about: edit the file on
        // disk WITHOUT re-ingesting. The graph is stale and keeps serving the old bytes.
        let edited = format!("{src}\npub fn g() {{ }}\n");
        std::fs::write(dir.join(path), &edited).unwrap();
        let stale = vfs.read(path).await.unwrap();
        assert_eq!(
            String::from_utf8(stale).unwrap(),
            src,
            "an un-ingested edit is invisible to the graph — this is the staleness risk, pinned"
        );

        // Re-ingest reconciled, and the VFS serves the edit.
        let stored = store.file_nodes(path).unwrap();
        let (fw, _) = ingest::file_against(lang.as_ref(), path, &edited, &stored).unwrap();
        store.replace_file(&fw).unwrap();
        assert_eq!(
            String::from_utf8(vfs.read(path).await.unwrap()).unwrap(),
            edited,
            "after re-ingest the graph must serve the new bytes"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
