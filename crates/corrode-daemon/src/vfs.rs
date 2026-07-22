//! Embedded virtual file system: the HelixDB graph presented as a git-compliant tree.
//!
//! The graph — held in the daemon's embedded HelixDB store ([`crate::graph`]) — is
//! the source of truth (nodes = files/symbols, edges = references, ownership,
//! call/import relations). The VFS projects a slice of that graph as an ordinary
//! directory tree so any git-aware tool — including the swarm's own subagents
//! shelling out — sees a normal working copy. Writes flow the other way: a file
//! edit is diffed back into node/edge mutations on the graph.
//!
//! Two retrieval paths feed context selection: HelixDB's own vector search
//! (GraphRAG, for docs) and hipfire's first-class `/v1/embeddings` + `/v1/rerank`
//! for code. The projected tree is what keeps all of it *git-compliant*: subagents
//! diff, stage, and blame against real paths while the graph stays the authority.
#![allow(dead_code)] // ponytail: unwired seam; remove once the first Vfs impl lands.

use std::path::Path;

/// A node materialized as a file in the projected tree.
#[derive(Clone, Debug)]
pub struct FileNode {
    /// git-compliant path, e.g. `src/swarm.rs`.
    pub path: String,
    pub contents: Vec<u8>,
}

/// Translates between the graph database and a git-compliant file tree.
///
/// ponytail: trait with a single concern and no impl yet — this is the seam the
/// scaffold exists to name. The first real impl picks a graph store (the "add
/// when" for that decision is: once we know the query shape retrieval needs).
/// Don't pull in a graph DB dependency before then.
pub trait Vfs {
    /// Project the graph slice reachable from `root` into concrete files.
    fn project(&self, root: &str) -> anyhow::Result<Vec<FileNode>>;

    /// Fold an edited file back into graph mutations (the reverse translation).
    fn absorb(&mut self, path: &Path, contents: &[u8]) -> anyhow::Result<()>;
}
