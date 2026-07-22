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
/// and its tests don't drag the (heavy, feature-gated) HelixDB compile in.
pub trait GraphStore {
    /// Nodes directly reachable from `id` — one hop of the explorer's graph view.
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
}
