//! The embedded graph+vector store: HelixDB, linked in-process — no separate service.
//!
//! HelixDB is vendored at `third_party/helix-db` (tag v2.3.5). We link its
//! `helix_engine` crate directly and open an LMDB-backed store at a local path, so
//! the daemon *is* the database — no `helix start`, no port 6969, no HTTP hop. This
//! is the piece that makes "embedded into the daemon" literally true rather than a
//! supervised child process.
//!
//! HelixDB gives us graph traversal, vector similarity, and GraphRAG in one store:
//! - the **graph** side is the VFS's source of truth (files/symbols/edges);
//! - the **vector** side backs GraphRAG over documentation (`AgentCommand::DocQuery`).
//!
//! License note: HelixDB is AGPL-3.0; see this crate's Cargo.toml.

use corrode_core::GraphNodeView;

/// The daemon's view of the embedded store. Kept as a trait so the swarm/VFS code
/// and its tests don't drag the (heavy, feature-gated) HelixDB compile in, and so
/// the daemon can hold it as `Option<Box<dyn GraphStore>>` (None until opened).
pub trait GraphStore: Send + Sync {
    /// Nodes directly reachable from `id` — one hop of the explorer's graph view.
    // ponytail: no caller in the base build yet; wired with a ListNeighbors command
    // when the webui graph explorer lands.
    #[allow(dead_code)]
    fn neighbors(&self, id: &str) -> anyhow::Result<Vec<GraphNodeView>>;

    /// GraphRAG: vector-search docs, then walk the graph for grounding. Returns the
    /// node ids the answer is grounded on.
    fn doc_search(&self, question: &str, k: usize) -> anyhow::Result<Vec<String>>;
}

/// In-process HelixDB. Only compiled with `--features helix`.
#[cfg(feature = "helix")]
pub mod embedded {
    use super::*;
    use helix_db::helix_engine::storage_core::version_info::VersionInfo;
    use helix_db::helix_engine::storage_core::HelixGraphStorage;
    use helix_db::helix_engine::traversal_core::config::Config;

    pub struct HelixStore {
        storage: HelixGraphStorage,
    }

    impl HelixStore {
        /// Open (creating if absent) the LMDB-backed store at `path`, in-process.
        pub fn open(path: &str) -> anyhow::Result<Self> {
            let storage = HelixGraphStorage::new(path, Config::default(), VersionInfo::default())
                .map_err(|e| anyhow::anyhow!("open HelixDB at {path}: {e:?}"))?;
            Ok(Self { storage })
        }
    }

    impl GraphStore for HelixStore {
        fn neighbors(&self, _id: &str) -> anyhow::Result<Vec<GraphNodeView>> {
            // ponytail: storage handle is open and typed; the actual traversal_core
            // query (out-edges -> node views) is the first real query to write.
            let _ = &self.storage;
            anyhow::bail!("neighbors: not implemented yet")
        }

        fn doc_search(&self, _question: &str, _k: usize) -> anyhow::Result<Vec<String>> {
            // ponytail: wire vector_core HNSW search + graph grounding here.
            anyhow::bail!("doc_search: not implemented yet")
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A fresh, process-unique store directory that doesn't yet exist —
        /// `HelixGraphStorage::new` does its own `create_dir_all`.
        fn scratch_dir(tag: &str) -> std::path::PathBuf {
            std::env::temp_dir().join(format!("corrode-helix-{}-{tag}", std::process::id()))
        }

        #[test]
        fn open_creates_store_persists_and_serves_as_graphstore() {
            let dir = scratch_dir("open");
            std::fs::remove_dir_all(&dir).ok(); // start clean if a prior run crashed
            let path = dir.to_str().unwrap();

            // Opening in-process actually initializes the LMDB env + tables — this is
            // the real proof that vendored helix_engine links and runs, not a stub.
            let store = HelixStore::open(path).expect("open fresh store");
            assert!(dir.is_dir(), "open should have created the store dir");

            // Reopening the same path must succeed against the existing env (persistence).
            drop(store);
            let reopened = HelixStore::open(path).expect("reopen existing store");

            // It satisfies the daemon's trait object, the way the Daemon actually holds it.
            let store: Box<dyn GraphStore> = Box::new(reopened);
            // ponytail: neighbors/doc_search are still stubs — assert the seam is wired
            // (real handle, error surfaces through the trait). Flip these to positive
            // assertions when the traversal_core / vector_core queries land.
            assert!(store.neighbors("does-not-exist").is_err());
            assert!(store.doc_search("anything", 4).is_err());

            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
