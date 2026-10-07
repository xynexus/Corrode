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
//! So the wrapper is honest about not knowing. It serves the graph only for a file whose
//! CURRENT content it holds -- the git blob oid recorded at ingest must match the file's
//! on disk (#43; it used to serve whatever it had ingested last, and ingest itself read
//! back through this wrapper, so a file froze at its first ingest) -- falls through to
//! the inner VFS for everything else, and — when
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
            verify: crate::knobs::flag("CORRODE_VFS_VERIFY", false),
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

    /// The graph's composition of `path`, if it holds the content `disk` currently is.
    fn fresh(&self, path: &str, disk: &[u8]) -> Option<Vec<u8>> {
        let stored = self.store.node_label(&crate::graph::oid_key(path)).ok().flatten()?;
        if stored != crate::vfs::blob_oid(disk) {
            eprintln!("vfs: the graph's copy of {path} is stale; reading it from disk");
            return None;
        }
        self.compose(path)
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


/// Is the graph-backed VFS enabled?
pub fn enabled() -> bool {
    crate::knobs::flag("CORRODE_VFS_GRAPH", false)
}

#[async_trait]
impl Vfs for GraphVfs {
    async fn read(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        // The freshness check needs the disk's bytes, so read them first; a path that is
        // not on disk errors as the inner VFS would.
        let disk = self.inner.read(path).await?;
        let Some(bytes) = self.fresh(path, &disk) else {
            self.fell_through.fetch_add(1, Ordering::Relaxed);
            return Ok(disk);
        };
        if self.verify && disk != bytes {
            // Same content by oid, different bytes composed: the projection, not the
            // ingest, is wrong. Report it rather than resolve it.
            self.diverged.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "vfs: graph composes {path} differently from the content it ingested ({} vs {} bytes)",
                bytes.len(),
                disk.len()
            );
        }
        self.served.fetch_add(1, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn read_disk(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn stat(&self, path: &str) -> anyhow::Result<FileNodeView> {
        let composed = match self.inner.read(path).await {
            Ok(disk) => self.fresh(path, &disk),
            Err(_) => None,
        };
        match composed {
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

    /// A store that holds exactly the nodes it was handed, ingested from `oid`.
    struct FakeStore(Vec<crate::projection::Node>, Option<String>);

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
        fn node_label(&self, id: &str) -> anyhow::Result<Option<String>> {
            Ok(id.starts_with("oid:").then(|| self.1.clone()).flatten())
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
        let src = "fn a() { 1 }\n\nfn b() { 2 }\n";
        std::fs::write(dir.join("a.rs"), src).unwrap();
        let lang = crate::projection::for_path("a.rs");
        let (items, _) = lang.spans(src).unwrap();
        let nodes = crate::projection::nodes_from_items("a.rs", src, &items);
        let oid = Some(crate::vfs::blob_oid(src.as_bytes()));

        let vfs = GraphVfs::new(
            Arc::new(FakeStore(nodes, oid)),
            Arc::new(PassthroughVfs::new(&dir)),
        );
        assert_eq!(vfs.read("a.rs").await.unwrap(), src.as_bytes());
        // stat must agree with read, or FUSE truncates to a size that is not the content.
        assert_eq!(vfs.stat("a.rs").await.unwrap().bytes, src.len() as u64);
        assert_eq!(vfs.counts().0, 1, "served from the graph");
    }

    // #43: the graph used to serve whatever it ingested last, however the file had
    // changed since; an agent editing against those bytes wrote a patch that did not apply.
    #[tokio::test]
    async fn a_stale_graph_answers_with_the_disk() {
        let dir = scratch("stale");
        let ingested = "fn a() { 1 }\n";
        let lang = crate::projection::for_path("a.rs");
        let (items, _) = lang.spans(ingested).unwrap();
        let nodes = crate::projection::nodes_from_items("a.rs", ingested, &items);
        // Edited on disk after the ingest.
        std::fs::write(dir.join("a.rs"), b"fn a() { 2 }\n").unwrap();
        let vfs = GraphVfs::new(
            Arc::new(FakeStore(nodes, Some(crate::vfs::blob_oid(ingested.as_bytes())))),
            Arc::new(PassthroughVfs::new(&dir)),
        );
        assert_eq!(vfs.read("a.rs").await.unwrap(), b"fn a() { 2 }\n");
        assert_eq!(vfs.stat("a.rs").await.unwrap().bytes, 13);
        assert_eq!(vfs.counts().0, 0, "not served from the graph");
        // And ingest reads the disk, not the wrapper's answer.
        assert_eq!(vfs.read_disk("a.rs").await.unwrap(), b"fn a() { 2 }\n");
    }

    #[tokio::test]
    async fn a_file_the_graph_lacks_falls_through_to_disk() {
        let dir = scratch("fallthrough");
        std::fs::write(dir.join("b.rs"), b"only on disk\n").unwrap();
        let vfs = GraphVfs::new(Arc::new(FakeStore(Vec::new(), None)), Arc::new(PassthroughVfs::new(&dir)));
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
        let vfs = GraphVfs::new(Arc::new(FakeStore(Vec::new(), None)), Arc::new(PassthroughVfs::new(&dir)));
        assert_eq!(vfs.list("").await.unwrap().len(), 2);
        assert_eq!(vfs.tracked_files().await.unwrap().len(), 2);
    }
}

/// End-to-end against a real store: ingest a file, then read it back through the VFS.
#[cfg(all(test, feature = "helix"))]
mod live {
    use super::*;
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
        crate::graph::ingest_source(store.as_ref(), path, src).unwrap();

        let vfs = GraphVfs::new(store.clone(), Arc::new(PassthroughVfs::new(&dir)));
        let read = vfs.read(path).await.unwrap();
        assert_eq!(
            String::from_utf8(read).unwrap(),
            src,
            "the graph must compose the file back byte-exactly"
        );
        assert_eq!(vfs.stat(path).await.unwrap().bytes, src.len() as u64, "stat must match read");
        assert_eq!(vfs.counts().0, 1, "should have been served from the graph");

        // The failure mode this design must not have (#43): edit the file on disk WITHOUT
        // re-ingesting. The graph's copy is stale, so the disk answers.
        let edited = format!("{src}\npub fn g() {{ }}\n");
        std::fs::write(dir.join(path), &edited).unwrap();
        assert_eq!(String::from_utf8(vfs.read(path).await.unwrap()).unwrap(), edited);
        assert_eq!(vfs.counts().0, 1, "the stale copy was not served");

        // Re-ingest from disk, and the graph serves the edit; the same content again is a no-op.
        let disk = String::from_utf8(vfs.read_disk(path).await.unwrap()).unwrap();
        assert!(crate::graph::ingest_source(store.as_ref(), path, &disk).unwrap().is_some());
        assert!(crate::graph::ingest_source(store.as_ref(), path, &disk).unwrap().is_none());
        assert_eq!(String::from_utf8(vfs.read(path).await.unwrap()).unwrap(), edited);
        assert_eq!(vfs.counts().0, 2, "after re-ingest the graph serves the new bytes");

        // Deleted: dropped, and nothing of it answers a search.
        store.drop_file(path).unwrap();
        assert!(store.file_nodes(path).unwrap().is_empty());
        assert_eq!(store.node_label(&crate::graph::oid_key(path)).unwrap(), None);
        assert!(store.code_search("quoted", 5).unwrap().iter().all(|(k, _)| !k.contains(path)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
