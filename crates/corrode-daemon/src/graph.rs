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
/// the daemon can hold it as `Option<Arc<dyn GraphStore>>` (None until opened).
pub trait GraphStore: Send + Sync {
    /// The one-hop neighborhood around `id` — the queried node plus every adjacent
    /// node in either direction — for the explorer's interactive graph browse
    /// (`AgentCommand::ListNeighbors`). Same node+`edges_out` shape as a `PlanGraph`
    /// event so the front-end merges it directly.
    fn neighbors(&self, id: &str) -> anyhow::Result<Vec<GraphNodeView>>;

    /// Doc retrieval for GraphRAG: HNSW similarity when `query_vec` is given
    /// (hipfire-embedded question), BM25 text search otherwise (no embedding
    /// model served). Returns `(id, text)` per hit, best first.
    fn doc_search(
        &self,
        question: &str,
        query_vec: Option<&[f32]>,
        k: usize,
    ) -> anyhow::Result<Vec<(String, String)>>;

    /// Create-or-update a provenance node (plan / task / contract / code) by id.
    fn upsert_node(&self, id: &str, kind: &str, label: &str) -> anyhow::Result<()>;

    /// Add a directed, labelled provenance edge (`from -rel-> to`), e.g. a code node
    /// `produced_by` its task, a task `part_of` its plan. Idempotent per (from,rel,to).
    fn add_edge(&self, from: &str, rel: &str, to: &str) -> anyhow::Result<()>;

    /// Replace a document and ALL its chunks atomically (one write txn): upsert the
    /// `doc` node, write each `(chunk_id, text, embedding?)` as a chunk node +
    /// `has_chunk` edge (+ HNSW vector when the embedding is present), and prune any
    /// chunk previously linked to this doc that isn't in `chunks` — so re-ingesting
    /// a shrunk/shifted doc never leaves stale chunks serving deleted content.
    fn replace_doc(&self, doc: &DocWrite) -> anyhow::Result<()>;
}

/// One document's full chunk set, for [`GraphStore::replace_doc`]. Owned strings:
/// it crosses a `spawn_blocking` boundary.
pub struct DocWrite {
    pub doc_id: String,
    pub title: String,
    /// `(chunk_id, text, embedding)` — embedding `None` means text/BM25-only.
    pub chunks: Vec<(String, String, Option<Vec<f32>>)>,
}

/// Open the embedded store for a repo at `<repo>/.corrode/graph`, per-repo — the
/// same opener the daemon uses for any session's repo. `None` without
/// `--features helix`, or if the open fails (logged).
#[cfg(feature = "helix")]
pub fn open(repo_root: &std::path::Path) -> Option<std::sync::Arc<dyn GraphStore>> {
    let path = repo_root.join(".corrode/graph");
    match embedded::HelixStore::open(&path.to_string_lossy()) {
        Ok(store) => Some(std::sync::Arc::new(store)),
        Err(e) => {
            eprintln!("HelixDB open failed at {}: {e}", path.display());
            None
        }
    }
}

#[cfg(not(feature = "helix"))]
pub fn open(_repo_root: &std::path::Path) -> Option<std::sync::Arc<dyn GraphStore>> {
    None
}

/// In-process HelixDB. Only compiled with `--features helix`.
///
/// Schema: every graph node gets the single label `"corrode"`; the caller's
/// string id lives in a `Unique` secondary index `"key"`, and `kind`/`label`
/// are plain properties (helix mints its own u128 node ids — `n_from_index`
/// is the bridge back to `"doc:path"`-style ids). Chunk embeddings are HNSW
/// vectors under one vector label, linked node -`embedding_of`-> vector; the
/// chunk's key + text ride on the vector's properties so a similarity hit
/// needs no extra hop. BM25 indexing is on by default (every `add_n` pays it),
/// which is what backs the no-embedding-model retrieval path.
#[cfg(feature = "helix")]
pub mod embedded {
    use super::*;
    use bumpalo::Bump;
    use helix_db::helix_engine::storage_core::storage_methods::StorageMethods;
    use helix_db::helix_engine::storage_core::version_info::VersionInfo;
    use helix_db::helix_engine::storage_core::HelixGraphStorage;
    use helix_db::helix_engine::traversal_core::config::Config;
    use helix_db::helix_engine::traversal_core::ops::bm25::search_bm25::SearchBM25Adapter;
    use helix_db::helix_engine::traversal_core::ops::g::G;
    use helix_db::helix_engine::traversal_core::ops::in_::in_::InAdapter;
    use helix_db::helix_engine::traversal_core::ops::out::out::OutAdapter;
    use helix_db::helix_engine::traversal_core::ops::out::out_e::OutEdgesAdapter;
    use helix_db::helix_engine::traversal_core::ops::source::add_e::AddEAdapter;
    use helix_db::helix_engine::traversal_core::ops::source::add_n::AddNAdapter;
    use helix_db::helix_engine::traversal_core::ops::source::n_from_id::NFromIdAdapter;
    use helix_db::helix_engine::traversal_core::ops::source::n_from_index::NFromIndexAdapter;
    use helix_db::helix_engine::traversal_core::ops::util::upsert::UpsertAdapter;
    use helix_db::helix_engine::traversal_core::ops::vectors::insert::InsertVAdapter;
    use helix_db::helix_engine::traversal_core::ops::vectors::search::SearchVAdapter;
    use helix_db::helix_engine::traversal_core::traversal_value::TraversalValue;
    use helix_db::helix_engine::types::SecondaryIndex;
    use helix_db::helix_engine::vector_core::vector::HVector;
    use helix_db::protocol::value::Value;
    use helix_db::utils::properties::ImmutablePropertiesMap;

    /// The one node label; `kind` is a property, so index lookups never need to
    /// know a node's kind. ponytail: label-level partitioning if scans ever hurt.
    const LABEL: &str = "corrode";
    /// The one vector label (one embedding space; helix's HNSW is global anyway).
    const VEC_LABEL: &str = "embedding";
    const REL_EMBEDDING: &str = "embedding_of";
    /// Relations `neighbors` walks (out-edge scans are per-label in helix).
    const NEIGHBOR_RELS: &[&str] = &["part_of", "emitted_from", "produced_by", "has_chunk"];

    /// insert_v/search_v want a filter type even when unused (helix's own tests
    /// pass this fn-pointer turbofish).
    type NoFilter = fn(&HVector, &heed3::RoTxn) -> bool;

    pub struct HelixStore {
        storage: HelixGraphStorage,
    }

    fn props<'a>(arena: &'a Bump, kv: Vec<(&'static str, Value)>) -> ImmutablePropertiesMap<'a> {
        let len = kv.len();
        ImmutablePropertiesMap::new(len, kv.into_iter().map(|(k, v)| (k, v)), arena)
    }

    fn prop_str(v: &TraversalValue, key: &str) -> String {
        match v.get_property(key) {
            Some(Value::String(s)) => s.to_string(),
            Some(other) => format!("{other:?}"),
            None => String::new(),
        }
    }

    /// Vectors are f64 in helix at this tag; embeddings arrive as f32.
    fn widen<'a>(arena: &'a Bump, v: &[f32]) -> &'a [f64] {
        arena.alloc_slice_fill_iter(v.iter().map(|&x| x as f64))
    }

    impl HelixStore {
        /// Open (creating if absent) the LMDB-backed store at `path`, in-process.
        /// The `"key"` secondary index MUST be declared here: `HelixGraphStorage::new`
        /// only creates index tables listed in the config, and `n_from_index` on an
        /// undeclared index panics rather than erroring.
        pub fn open(path: &str) -> anyhow::Result<Self> {
            let mut config = Config::default();
            if let Some(gc) = config.graph_config.as_mut() {
                gc.secondary_indices = Some(vec![SecondaryIndex::Unique("key".to_string())]);
            }
            let storage = HelixGraphStorage::new(path, config, VersionInfo::default())
                .map_err(|e| anyhow::anyhow!("open HelixDB at {path}: {e:?}"))?;
            Ok(Self { storage })
        }

        /// The engine-minted u128 id behind our string `key`, if the node exists.
        /// Works inside a write txn too (`RwTxn` derefs to `RoTxn`).
        fn find_id(
            &self,
            txn: &heed3::RoTxn,
            arena: &Bump,
            key: &str,
        ) -> Option<u128> {
            let hits: Vec<TraversalValue> = G::new(&self.storage, txn, arena)
                .n_from_index(LABEL, "key", &key.to_string())
                .take_and_collect_to(1);
            hits.first().map(|n| n.id())
        }

        /// Create-or-update the `"corrode"` node behind `key` in an open write txn;
        /// returns its engine id. The existing-node path goes through `upsert_n`
        /// (property + index + BM25 update); the create path uses `add_n` with the
        /// index written explicitly.
        fn upsert_in<'a>(
            &'a self,
            txn: &mut heed3::RwTxn<'a>,
            arena: &'a Bump,
            key: &str,
            kind: &str,
            label: &str,
        ) -> anyhow::Result<u128> {
            let existing: Vec<TraversalValue> = G::new(&self.storage, txn, arena)
                .n_from_index(LABEL, "key", &key.to_string())
                .take_and_collect_to(1);
            let node = if existing.is_empty() {
                G::new_mut(&self.storage, arena, txn)
                    .add_n(
                        LABEL,
                        Some(props(
                            arena,
                            vec![
                                ("key", Value::from(key.to_string())),
                                ("kind", Value::from(kind.to_string())),
                                ("label", Value::from(label.to_string())),
                            ],
                        )),
                        Some(&["key"]),
                    )
                    .collect_to_obj()
                    .map_err(|e| anyhow::anyhow!("add_n {key}: {e:?}"))?
            } else {
                G::new_mut_from_iter(&self.storage, txn, existing.into_iter(), arena)
                    .upsert_n(
                        LABEL,
                        &[
                            ("key", Value::from(key.to_string())),
                            ("kind", Value::from(kind.to_string())),
                            ("label", Value::from(label.to_string())),
                        ],
                    )
                    .collect_to_obj()
                    .map_err(|e| anyhow::anyhow!("upsert_n {key}: {e:?}"))?
            };
            Ok(node.id())
        }
    }

    impl GraphStore for HelixStore {
        fn neighbors(&self, id: &str) -> anyhow::Result<Vec<GraphNodeView>> {
            let arena = Bump::new(); // arena outlives the txn ('db: 'arena: 'txn)
            let txn = self.storage.graph_env.read_txn()?;
            let Some(node_id) = self.find_id(&txn, &arena, id) else {
                return Ok(Vec::new());
            };
            // One hop in BOTH directions, returned in the same shape as a `PlanGraph`
            // event (nodes carrying `edges_out`) so the webui merges it with zero new
            // logic. The queried node carries its outgoing targets; an *incoming*
            // neighbor (n -rel-> id) carries `edges_out=[id]` so that edge is drawn;
            // an outgoing neighbor's edge is already given by the queried node.
            // Both directions matter: expanding a `plan` must reveal the tasks that
            // point AT it (`part_of` is task->plan), not only what it points to.
            let mut nodes: std::collections::HashMap<String, GraphNodeView> =
                std::collections::HashMap::new();
            let mut out_targets: Vec<String> = Vec::new();
            for rel in NEIGHBOR_RELS {
                let outs: Vec<TraversalValue> = G::new(&self.storage, &txn, &arena)
                    .n_from_id(&node_id)
                    .out_node(rel)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("neighbors {id} out via {rel}: {e:?}"))?;
                for n in outs {
                    let key = prop_str(&n, "key");
                    out_targets.push(key.clone());
                    nodes.entry(key.clone()).or_insert_with(|| GraphNodeView {
                        id: key,
                        label: prop_str(&n, "label"),
                        kind: prop_str(&n, "kind"),
                        edges_out: Vec::new(),
                    });
                }
                let ins: Vec<TraversalValue> = G::new(&self.storage, &txn, &arena)
                    .n_from_id(&node_id)
                    .in_node(rel)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("neighbors {id} in via {rel}: {e:?}"))?;
                for n in ins {
                    let key = prop_str(&n, "key");
                    let entry = nodes.entry(key.clone()).or_insert_with(|| GraphNodeView {
                        id: key,
                        label: prop_str(&n, "label"),
                        kind: prop_str(&n, "kind"),
                        edges_out: Vec::new(),
                    });
                    if !entry.edges_out.iter().any(|t| t == id) {
                        entry.edges_out.push(id.to_string());
                    }
                }
            }
            // The queried node itself, carrying every outgoing edge to a neighbor.
            let self_hit: Vec<TraversalValue> = G::new(&self.storage, &txn, &arena)
                .n_from_id(&node_id)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("neighbors {id} self: {e:?}"))?;
            if let Some(s) = self_hit.first() {
                nodes.insert(
                    id.to_string(),
                    GraphNodeView {
                        id: id.to_string(),
                        label: prop_str(s, "label"),
                        kind: prop_str(s, "kind"),
                        edges_out: out_targets,
                    },
                );
            }
            Ok(nodes.into_values().collect())
        }

        fn doc_search(
            &self,
            question: &str,
            query_vec: Option<&[f32]>,
            k: usize,
        ) -> anyhow::Result<Vec<(String, String)>> {
            let arena = Bump::new();
            let txn = self.storage.graph_env.read_txn()?;
            let mut out: Vec<(String, String)> = Vec::new();
            let mut seen = std::collections::HashSet::new();

            // Vector path (when the question was embedded): chunk key + text ride
            // on the vector's own properties, so a hit needs no vector->node hop.
            if let Some(q) = query_vec {
                match G::new(&self.storage, &txn, &arena)
                    .search_v::<NoFilter, _>(widen(&arena, q), k, VEC_LABEL, None)
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(hits) => {
                        for h in &hits {
                            let key = prop_str(h, "key");
                            if seen.insert(key.clone()) {
                                out.push((key, prop_str(h, "text")));
                            }
                        }
                    }
                    // An empty HNSW has no entry point; that's "no docs", not an error.
                    Err(e) if format!("{e:?}").contains("no entry point") => {}
                    Err(e) => return Err(anyhow::anyhow!("vector search: {e:?}")),
                }
            }

            // BM25 always runs too: it's the sole path when no embedding model is
            // served, AND it backfills chunks that failed to embed (which the vector
            // path can't see). ponytail: provenance nodes share the corpus (one
            // label), so over-fetch and post-filter to doc/chunk kinds; a dedicated
            // doc label would remove the crowding — do that if recall degrades.
            if out.len() < k {
                let fetch = (k * 8).max(64);
                let bm: Vec<TraversalValue> = G::new(&self.storage, &txn, &arena)
                    .search_bm25(LABEL, question, fetch)
                    .map_err(|e| anyhow::anyhow!("bm25 search: {e:?}"))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("bm25 collect: {e:?}"))?;
                for h in &bm {
                    if out.len() >= k {
                        break;
                    }
                    if !matches!(prop_str(h, "kind").as_str(), "chunk" | "doc") {
                        continue;
                    }
                    let key = prop_str(h, "key");
                    if seen.insert(key.clone()) {
                        out.push((key, prop_str(h, "label")));
                    }
                }
            }
            out.truncate(k);
            Ok(out)
        }

        fn upsert_node(&self, id: &str, kind: &str, label: &str) -> anyhow::Result<()> {
            let arena = Bump::new();
            let mut txn = self.storage.graph_env.write_txn()?;
            self.upsert_in(&mut txn, &arena, id, kind, label)?;
            txn.commit()?;
            Ok(())
        }

        fn add_edge(&self, from: &str, rel: &str, to: &str) -> anyhow::Result<()> {
            let arena = Bump::new();
            let mut txn = self.storage.graph_env.write_txn()?;
            let from_id = self
                .find_id(&txn, &arena, from)
                .ok_or_else(|| anyhow::anyhow!("add_edge: unknown node {from}"))?;
            let to_id = self
                .find_id(&txn, &arena, to)
                .ok_or_else(|| anyhow::anyhow!("add_edge: unknown node {to}"))?;
            let result = G::new_mut(&self.storage, &arena, &mut txn)
                .add_edge(arena.alloc_str(rel), None, from_id, to_id, false, true)
                .collect_to_obj();
            match result {
                Ok(_) => {
                    txn.commit()?;
                    Ok(())
                }
                // is_unique=true refuses a duplicate (from, rel, to) — that's the
                // idempotence we want, not an error.
                Err(e) if format!("{e:?}").contains("DuplicateKey") => {
                    txn.commit()?;
                    Ok(())
                }
                Err(e) => Err(anyhow::anyhow!("add_edge {from} -{rel}-> {to}: {e:?}")),
            }
        }

        fn replace_doc(&self, doc: &DocWrite) -> anyhow::Result<()> {
            let arena = Bump::new();
            let mut txn = self.storage.graph_env.write_txn()?;

            // Which chunks does this doc currently have? (prune the ones that go away)
            let doc_node = self.find_id(&txn, &arena, &doc.doc_id);
            let mut existing: Vec<String> = Vec::new();
            if let Some(doc_id) = doc_node {
                let chunks: Vec<TraversalValue> = G::new(&self.storage, &txn, &arena)
                    .n_from_id(&doc_id)
                    .out_node("has_chunk")
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("list chunks of {}: {e:?}", doc.doc_id))?;
                existing.extend(chunks.iter().map(|c| prop_str(c, "key")));
            }

            self.upsert_in(&mut txn, &arena, &doc.doc_id, "doc", &doc.title)?;

            let new_ids: std::collections::HashSet<&str> =
                doc.chunks.iter().map(|(id, _, _)| id.as_str()).collect();
            for stale in existing.iter().filter(|k| !new_ids.contains(k.as_str())) {
                self.drop_chunk_in(&mut txn, &arena, stale)?;
            }

            for (chunk_id, text, embedding) in &doc.chunks {
                let node_id = self.upsert_in(&mut txn, &arena, chunk_id, "chunk", text)?;
                self.write_embedding_in(&mut txn, &arena, node_id, chunk_id, text, embedding.as_deref())?;
                // has_chunk is idempotent (is_unique); re-ingest doesn't duplicate.
                let doc_id = self
                    .find_id(&txn, &arena, &doc.doc_id)
                    .ok_or_else(|| anyhow::anyhow!("doc node vanished mid-write"))?;
                match G::new_mut(&self.storage, &arena, &mut txn)
                    .add_edge("has_chunk", None, doc_id, node_id, false, true)
                    .collect_to_obj()
                {
                    Ok(_) => {}
                    Err(e) if format!("{e:?}").contains("DuplicateKey") => {}
                    Err(e) => return Err(anyhow::anyhow!("link chunk {chunk_id}: {e:?}")),
                }
            }
            txn.commit()?;
            Ok(())
        }
    }

    impl HelixStore {
        /// Replace a chunk's embedding in an open write txn: tombstone the old
        /// vector(s) + link edge(s), then insert the new one (if any). Vector delete
        /// is a soft tombstone in helix, skipped at search time.
        fn write_embedding_in<'a>(
            &'a self,
            txn: &mut heed3::RwTxn<'a>,
            arena: &'a Bump,
            node_id: u128,
            key: &str,
            text: &str,
            embedding: Option<&[f32]>,
        ) -> anyhow::Result<()> {
            let old_edges: Vec<TraversalValue> = G::new(&self.storage, txn, arena)
                .n_from_id(&node_id)
                .out_e(REL_EMBEDDING)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("old embedding edges of {key}: {e:?}"))?;
            let old_vecs: Vec<TraversalValue> = G::new(&self.storage, txn, arena)
                .n_from_id(&node_id)
                .out_vec(REL_EMBEDDING, false)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("old embeddings of {key}: {e:?}"))?;
            for e in &old_edges {
                if let Err(e2) = self.storage.drop_edge(txn, &e.id()) {
                    if !format!("{e2:?}").contains("EdgeNotFound") {
                        anyhow::bail!("drop old embedding edge of {key}: {e2:?}");
                    }
                }
            }
            for v in &old_vecs {
                self.storage
                    .drop_vector(txn, &v.id())
                    .map_err(|e| anyhow::anyhow!("drop old embedding of {key}: {e:?}"))?;
            }

            let Some(vec) = embedding else { return Ok(()) };
            let vector = G::new_mut(&self.storage, arena, txn)
                .insert_v::<NoFilter>(
                    widen(arena, vec),
                    VEC_LABEL,
                    Some(props(
                        arena,
                        vec![
                            ("key", Value::from(key.to_string())),
                            // ponytail: text duplicated onto the vector so a hit
                            // needs no hop; dedupe if store size bites.
                            ("text", Value::from(text.to_string())),
                        ],
                    )),
                )
                .collect_to_obj()
                .map_err(|e| anyhow::anyhow!("insert embedding of {key}: {e:?}"))?;
            G::new_mut(&self.storage, arena, txn)
                .add_edge(REL_EMBEDDING, None, node_id, vector.id(), false, false)
                .collect_to_obj()
                .map_err(|e| anyhow::anyhow!("link embedding of {key}: {e:?}"))?;
            Ok(())
        }

        /// Drop a chunk fully in an open write txn: its embedding vector(s), then
        /// the node (helix's drop_node also removes both edge directions + the
        /// secondary index).
        /// ponytail: drop_node does NOT remove the node's BM25 doc at helix v2.3.5,
        /// so a pruned chunk lingers in BM25 text search until the index is rebuilt;
        /// the vector + node are gone, so vector search and graph walks are clean.
        fn drop_chunk_in<'a>(
            &'a self,
            txn: &mut heed3::RwTxn<'a>,
            arena: &'a Bump,
            key: &str,
        ) -> anyhow::Result<()> {
            let Some(node_id) = self.find_id(txn, arena, key) else {
                return Ok(());
            };
            let vecs: Vec<TraversalValue> = G::new(&self.storage, txn, arena)
                .n_from_id(&node_id)
                .out_vec(REL_EMBEDDING, false)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("embeddings of stale chunk {key}: {e:?}"))?;
            for v in &vecs {
                self.storage
                    .drop_vector(txn, &v.id())
                    .map_err(|e| anyhow::anyhow!("drop stale embedding {key}: {e:?}"))?;
            }
            self.storage
                .drop_node(txn, &node_id)
                .map_err(|e| anyhow::anyhow!("drop stale chunk {key}: {e:?}"))?;
            Ok(())
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
            // Empty store: lookups miss cleanly, empty vector index is "no hits",
            // not an error (helix's EntryPointNotFound is mapped away).
            assert!(store.neighbors("does-not-exist").unwrap().is_empty());
            assert!(store
                .doc_search("anything", Some(&[0.5; 8]), 4)
                .unwrap()
                .is_empty());

            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn provenance_round_trip_upserts_edges_and_neighbors() {
            let dir = scratch_dir("prov");
            std::fs::remove_dir_all(&dir).ok();
            let store = HelixStore::open(dir.to_str().unwrap()).expect("open");

            store.upsert_node("plan-0", "plan", "the plan").unwrap();
            store.upsert_node("task-1", "task", "write the parser").unwrap();
            store.add_edge("task-1", "part_of", "plan-0").unwrap();
            // Idempotence: same node + same edge again must not duplicate.
            store.upsert_node("task-1", "task", "write the parser v2").unwrap();
            store.add_edge("task-1", "part_of", "plan-0").unwrap();

            // From the task: the subgraph is the task itself (carrying its outgoing
            // edge to the plan) plus the plan, no duplicates from the re-upsert.
            let n = store.neighbors("task-1").unwrap();
            assert_eq!(n.len(), 2, "self + one plan neighbor: {n:?}");
            let task = n.iter().find(|v| v.id == "task-1").expect("self node present");
            assert_eq!(task.edges_out, vec!["plan-0".to_string()], "task -> plan edge");
            let plan = n.iter().find(|v| v.id == "plan-0").expect("plan neighbor present");
            assert_eq!(plan.kind, "plan");

            // From the plan: the incoming task is reachable too (part_of is task->plan),
            // and it carries the edge back to the plan so the explorer can draw it.
            let m = store.neighbors("plan-0").unwrap();
            assert_eq!(m.len(), 2, "self + the task pointing at it: {m:?}");
            let inbound = m.iter().find(|v| v.id == "task-1").expect("incoming task present");
            assert_eq!(inbound.edges_out, vec!["plan-0".to_string()], "task -> plan edge drawn");

            // Unknown node -> empty, never an error.
            assert!(store.neighbors("ghost").unwrap().is_empty());
            let missing = store.add_edge("task-1", "part_of", "ghost");
            assert!(missing.is_err(), "edges to unknown nodes must fail");

            std::fs::remove_dir_all(&dir).ok();
        }

        fn dw(doc_id: &str, chunks: Vec<(&str, &str, Option<Vec<f32>>)>) -> DocWrite {
            DocWrite {
                doc_id: doc_id.to_string(),
                title: "t".to_string(),
                chunks: chunks
                    .into_iter()
                    .map(|(id, t, e)| (id.to_string(), t.to_string(), e))
                    .collect(),
            }
        }

        #[test]
        fn chunks_round_trip_vector_and_bm25_search() {
            let dir = scratch_dir("chunks");
            std::fs::remove_dir_all(&dir).ok();
            let store = HelixStore::open(dir.to_str().unwrap()).expect("open");

            store
                .replace_doc(&dw(
                    "doc:1",
                    vec![
                        ("chunk:a", "the scheduler uses priority bands", Some(vec![1.0, 0.0, 0.0, 0.0])),
                        ("chunk:b", "interrupt vectors live at zero", Some(vec![0.0, 1.0, 0.0, 0.0])),
                        ("chunk:c", "registers are general purpose", Some(vec![0.0, 0.0, 1.0, 0.0])),
                    ],
                ))
                .unwrap();

            // Vector path: nearest to b's embedding is b.
            let hits = store.doc_search("", Some(&[0.05, 0.9, 0.05, 0.0]), 2).unwrap();
            assert!(!hits.is_empty());
            assert_eq!(hits[0].0, "chunk:b", "nearest chunk wins: {hits:?}");
            assert!(hits[0].1.contains("interrupt"), "text rides the hit: {hits:?}");

            // BM25 path (no query vector): finds by word.
            let hits = store.doc_search("scheduler priority", None, 4).unwrap();
            assert!(
                hits.iter().any(|(id, _)| id == "chunk:a"),
                "bm25 should find chunk:a: {hits:?}"
            );

            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn reingest_prunes_stale_chunks_and_replaces_embeddings() {
            let dir = scratch_dir("reingest");
            std::fs::remove_dir_all(&dir).ok();
            let store = HelixStore::open(dir.to_str().unwrap()).expect("open");

            // First ingest: 3 chunks.
            store
                .replace_doc(&dw(
                    "doc:1",
                    vec![
                        ("chunk:1#0", "alpha content", Some(vec![1.0, 0.0, 0.0, 0.0])),
                        ("chunk:1#1", "bravo content", Some(vec![0.0, 1.0, 0.0, 0.0])),
                        ("chunk:1#2", "charlie content", Some(vec![0.0, 0.0, 1.0, 0.0])),
                    ],
                ))
                .unwrap();
            // neighbors() includes the queried node itself; count only the chunks.
            let chunks = |id| store.neighbors(id).unwrap().into_iter().filter(|x| x.id != "doc:1").count();
            assert_eq!(chunks("doc:1"), 3);

            // Re-ingest the SHRUNK doc: only 2 chunks, #1 re-embedded elsewhere.
            store
                .replace_doc(&dw(
                    "doc:1",
                    vec![
                        ("chunk:1#0", "alpha content", Some(vec![1.0, 0.0, 0.0, 0.0])),
                        ("chunk:1#1", "bravo REVISED", Some(vec![0.0, 0.0, 0.0, 1.0])),
                    ],
                ))
                .unwrap();

            // The dropped chunk is gone from the graph and from vector search.
            let n = store.neighbors("doc:1").unwrap();
            assert_eq!(chunks("doc:1"), 2, "pruned to 2 chunks: {n:?}");
            assert!(!n.iter().any(|x| x.id == "chunk:1#2"), "stale chunk pruned: {n:?}");

            // charlie's old embedding must no longer match (its vector was dropped).
            let hits = store.doc_search("", Some(&[0.0, 0.0, 1.0, 0.0]), 3).unwrap();
            assert!(
                !hits.iter().any(|(id, _)| id == "chunk:1#2"),
                "stale embedding must not surface: {hits:?}"
            );

            // The replaced embedding serves the new vector, with the new text.
            let hits = store.doc_search("", Some(&[0.0, 0.0, 0.0, 1.0]), 1).unwrap();
            assert_eq!(hits[0].0, "chunk:1#1");
            assert!(hits[0].1.contains("REVISED"), "new text rides the hit: {hits:?}");

            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn text_only_chunk_is_bm25_reachable_alongside_embedded_ones() {
            let dir = scratch_dir("textonly");
            std::fs::remove_dir_all(&dir).ok();
            let store = HelixStore::open(dir.to_str().unwrap()).expect("open");

            // One chunk failed to embed (None); others embedded.
            store
                .replace_doc(&dw(
                    "doc:1",
                    vec![
                        ("chunk:0", "the widget frobnicator overheats", None),
                        ("chunk:1", "cooling fans spin up", Some(vec![1.0, 0.0])),
                    ],
                ))
                .unwrap();

            // Vector-path query still returns the embedded chunk AND backfills the
            // text-only one via BM25 (union), so it isn't invisible.
            let hits = store.doc_search("frobnicator overheats", Some(&[1.0, 0.0]), 4).unwrap();
            assert!(
                hits.iter().any(|(id, _)| id == "chunk:0"),
                "text-only chunk reachable via BM25 union: {hits:?}"
            );

            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
