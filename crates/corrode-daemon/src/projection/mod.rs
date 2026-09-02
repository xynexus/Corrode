//! Source <-> node decomposition: the projection half of the graph model.
//!
//! `graph-model.md` makes files a projection of the graph. This is the machinery that
//! makes that true, and it is deliberately **language-agnostic**: Corrode absorbs
//! whatever codebase it is pointed at, so nothing here may assume Rust.
//!
//! # The seam
//!
//! A [`Language`] backend supplies three things, all of them spans:
//!
//! - `items` — top-level boundaries, which become nodes
//! - `anchors` — every element a comment could describe, at any depth
//! - `comments` — where the comments are, since most lexers discard them
//!
//! Everything else is shared: the node model, verbatim storage, ordinal ordering,
//! projection back to text, and [`bind`] — the precedes/trailing/encloses algorithm
//! that decides what a comment describes. That algorithm contains no language detail
//! at all, which is why it is worth having once rather than per backend.
//!
//! # Why the parser bar is low
//!
//! Nodes store **verbatim text** and are never regenerated from a syntax tree
//! (measured: printing an AST back is byte-exact for 0 of 91 files, while verbatim
//! composition is exact for 1515 of 1515). A backend therefore only has to report byte
//! ranges — it never has to reproduce source. That makes a lossy parser perfectly
//! adequate here, including `tree-sitter`, whose ~66 grammars would each be a thin
//! backend. The lossiness that costs an AST-storing system its line-level queries
//! (`harness-architecture.md` §10) simply does not apply when the bytes are kept.
//!
//! # Degradation
//!
//! [`text::PlainText`] handles anything with no backend: one node for the whole file,
//! comments by configured markers, no anchors. Structure is lost; **projection stays
//! byte-exact**. Absorbing an unfamiliar codebase is therefore never blocked, it is
//! only less queryable.

pub mod archive;
pub mod c;
pub mod docmap;
pub mod rust;
pub mod text;
pub mod update;

/// A byte range in the source, with what kind of thing it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Backend-defined: `fn`, `class`, `def`, `rule`, … Free-form on purpose — the core
    /// never interprets it, and a taxonomy imposed here would fit one language.
    pub kind: &'static str,
    pub start: usize,
    pub end: usize,
}

/// How a comment relates to what it describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// On its own line(s), introducing the element below it.
    Precedes,
    /// After code on the same line — annotates that code, not the next line.
    Trailing,
    /// Nothing follows in scope, so it belongs to its container.
    Encloses,
}

/// Comment flavour. Doc comments are called out because many languages give them
/// special meaning, and consumers usually want them separable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentKind {
    Line,
    Block,
    Doc,
    InnerDoc,
}

/// A comment found by a backend, before it is bound to anything.
#[derive(Debug, Clone, PartialEq)]
pub struct CommentSpan {
    pub kind: CommentKind,
    pub start: usize,
    pub end: usize,
}

/// What a backend must provide. Three span-producing methods and its identity.
pub trait Language: Send + Sync {
    fn name(&self) -> &'static str;
    /// Extensions this backend claims, without the dot.
    fn extensions(&self) -> &'static [&'static str];
    /// Top-level item boundaries in source order. Gaps between them become trivia.
    fn items(&self, src: &str) -> anyhow::Result<Vec<Span>>;
    /// Everything a comment could describe, at any depth. Empty is legal — comments
    /// then bind to nothing, which is honest rather than wrong.
    fn anchors(&self, src: &str) -> anyhow::Result<Vec<Span>>;
    /// Comment locations. Separate from parsing because most lexers discard comments
    /// before a syntax tree exists.
    fn comments(&self, src: &str) -> Vec<CommentSpan>;

    /// Items and anchors together, so a backend can parse once.
    ///
    /// The default calls both, which is correct and parses twice. Benchmarking showed
    /// that costing 72% of ingest time on Rust (items 32% + anchors 40%, each doing a
    /// full `syn::parse_file`), so a backend with an expensive parser should override
    /// this. Kept as a default rather than a required method because a cheap backend
    /// gains nothing from the complexity.
    fn spans(&self, src: &str) -> anyhow::Result<(Vec<Span>, Vec<Span>)> {
        Ok((self.items(src)?, self.anchors(src).unwrap_or_default()))
    }
}

/// Pick a backend for a path. Unknown extensions fall back to [`text::PlainText`], so
/// every file can be ingested — with less structure, never with less fidelity.
pub fn for_path(path: &str) -> Box<dyn Language> {
    let base = path.rsplit('/').next().unwrap_or(path);
    // Filename first: `Makefile`, `Kconfig` and friends carry no extension, and an
    // extension-only lookup would hand thousands of `#`-commented files to the C-family
    // default and find nothing in them.
    if let Some(named) = text::PlainText::for_filename(base) {
        return Box::new(named);
    }
    // Only treat a trailing segment as an extension if the name actually has a dot;
    // otherwise `rsplit` hands back the whole filename.
    let ext = base.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if rust::Rust.extensions().contains(&ext) {
        return Box::new(rust::Rust);
    }
    if c::C.extensions().contains(&ext) {
        return Box::new(c::C);
    }
    Box::new(text::PlainText::for_extension(ext))
}

/// Gap left between adjacent nodes on first ingest.
///
/// Dense indices make an insert renumber every node below it. On a 1,821-node file
/// (the measured maximum) that is 1,820 rewrites for one added item — and in a
/// provenance graph the cost is not the writes but the CHURN: every node below the
/// edit is marked modified, a one-item diff reads as 1,821 changes, and "which task
/// produced this node" becomes noise.
///
/// A stride of 2^32 leaves room for ~32 successive midpoint insertions at any single
/// point before the gap is exhausted, without touching a neighbour.
pub const ORDER_STRIDE: u64 = 1 << 32;

/// The order key for the `i`-th node of a freshly ingested file.
///
/// Deterministic on purpose. A random key would also be sparse, but the same file
/// would ingest to a different graph every time, which breaks diffing, caching and
/// content-addressing — the reproducibility this is supposed to protect.
/// Keys start at one stride, not zero, so there is room to insert BEFORE the first
/// node — a file gaining a new leading import or licence header is common, and a
/// zero-based first key would force a rebalance for it.
pub fn initial_order(i: usize) -> u64 {
    (i as u64 + 1).saturating_mul(ORDER_STRIDE)
}

/// A key strictly between `a` and `b`, or `None` when the gap is exhausted and the
/// file needs [`rebalance`].
#[allow(dead_code)] // mutation API: used once the graph is edited in place
pub fn order_between(a: u64, b: u64) -> Option<u64> {
    (b.saturating_sub(a) >= 2).then(|| a + (b - a) / 2)
}

/// Reassign every key at full stride, restoring room to insert.
///
/// Rare, bounded to one file, and the reason the key is documented as overwriteable:
/// exhaustion is recoverable rather than terminal. Node identity derives from the key,
/// so this IS a re-addressing operation — callers holding ids must re-read.
#[allow(dead_code)] // mutation API: used once the graph is edited in place
pub fn rebalance(nodes: &mut [Node]) {
    nodes.sort_by_key(|n| n.order);
    for (i, n) in nodes.iter_mut().enumerate() {
        n.order = initial_order(i);
    }
}

/// One node: a slice of the file, stored verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub path: String,
    /// Sparse order key within the file. Projection sorts by it; nothing else reads it.
    /// Sparse rather than dense so an insert costs one write instead of renumbering
    /// every node below it — see [`ORDER_STRIDE`].
    pub order: u64,
    /// Backend-defined kind, or `trivia` for the bytes between items.
    pub kind: &'static str,
    /// Verbatim source. Never regenerated.
    pub text: String,
}

/// Decompose a file into nodes covering every byte.
#[allow(dead_code)] // `ingest::file` uses the single-parse `spans` path instead
pub fn scan(lang: &dyn Language, path: &str, src: &str) -> anyhow::Result<Vec<Node>> {
    Ok(nodes_from_items(path, src, &lang.items(src)?))
}

/// Build the node cover from already-computed item spans, so a caller that already has
/// them does not ask the backend to parse again.
pub fn nodes_from_items(path: &str, src: &str, items: &[Span]) -> Vec<Node> {
    let mut nodes = Vec::new();
    let (mut cursor, mut ordinal) = (0usize, 0usize);
    // `ordinal` is a local counter only; what is stored is the sparse key.
    for it in items {
        if it.start > cursor {
            nodes.push(Node {
                path: path.into(),
                order: initial_order(ordinal),
                kind: "trivia",
                text: src[cursor..it.start].into(),
            });
            ordinal += 1;
        }
        nodes.push(Node {
            path: path.into(),
            order: initial_order(ordinal),
            kind: it.kind,
            text: src[it.start..it.end].into(),
        });
        ordinal += 1;
        cursor = it.end;
    }
    if cursor < src.len() {
        nodes.push(Node {
            path: path.into(),
            order: initial_order(ordinal),
            kind: "trivia",
            text: src[cursor..].into(),
        });
    }
    nodes
}

/// Where a node landed. Produced BY projection, never stored: a byte offset is a fact
/// about one source text, and a generated VFS has none.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub order: u64,
    pub kind: &'static str,
    pub start_line: usize,
    pub start_byte: usize,
}

/// Project nodes into a file and report where each landed.
pub fn project(nodes: &[Node]) -> (String, Vec<Placement>) {
    let mut sorted: Vec<&Node> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.order);
    // Preallocate: growing a 24 MB string by doubling copies it repeatedly.
    let total: usize = sorted.iter().map(|n| n.text.len()).sum();
    let mut text = String::with_capacity(total);
    let mut places = Vec::with_capacity(sorted.len());
    // Carry the line count forward instead of recounting the accumulated text for
    // every node. That recount was O(nodes x text), which on a 24 MB generated header
    // with 400k nodes meant PROJECTION took minutes: 2 MB cost 5s, 4 MB 18s, 8 MB 73s.
    //
    // This is the same defect as the one fixed in `bind` — a line number derived by
    // scanning from the beginning — written a second time in the function next to it.
    // Projection is the VFS read path, so here it would have been worse than slow.
    let mut line = 1usize;
    for n in sorted {
        places.push(Placement {
            order: n.order,
            kind: n.kind,
            start_line: line,
            start_byte: text.len(),
        });
        line += n.text.bytes().filter(|b| *b == b'\n').count();
        text.push_str(&n.text);
    }
    (text, places)
}

/// Reassemble a file from its nodes.
#[allow(dead_code)] // read half: called when the VFS projects from the graph
pub fn regenerate(nodes: &[Node]) -> String {
    project(nodes).0
}

/// A comment bound to what it describes.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub kind: CommentKind,
    pub text: String,
    pub relation: Relation,
    /// The element it describes, if any.
    pub target: Option<Span>,
    /// Order key of the node containing it.
    pub node_order: Option<u64>,
    /// 1-based line WITHIN that node — stable under edits elsewhere in the file.
    pub line_in_node: usize,
}

/// Bind comments to the elements they describe.
///
/// Language-agnostic by construction: it sees byte ranges and nothing else. This is the
/// part worth sharing — the relation a comment has to code is the same question in
/// every language, even though finding the spans is not.
pub fn bind(src: &str, comments: &[CommentSpan], anchors: &[Span], nodes: &[Node]) -> Vec<Edge> {
    // Sorted by `start` so the search below is sound. Backends are expected to return
    // them sorted; re-sorting a sorted slice is cheap and removes the assumption.
    let mut anchors: Vec<Span> = anchors.to_vec();
    anchors.sort_by_key(|a| (a.start, std::cmp::Reverse(a.end)));
    let anchors = &anchors[..];
    // Node extents in projection order, so a comment can be located within one.
    let mut sorted: Vec<&Node> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.order);
    let mut extents: Vec<(usize, usize, u64)> = Vec::new();
    let mut at = 0usize;
    for n in &sorted {
        extents.push((at, at + n.text.len(), n.order));
        at += n.text.len();
    }

    // Byte offsets of every line start, computed ONCE. The obvious implementation
    // recomputes a comment's line by scanning from the file's beginning, which is
    // O(comments x filesize) and was the real cost of `bind` — not the anchor search,
    // which was the first guess and barely moved the number.
    let mut line_starts: Vec<usize> = vec![0];
    line_starts.extend(src.bytes().enumerate().filter(|(_, b)| *b == b'\n').map(|(i, _)| i + 1));
    // 1-based line containing `byte`.
    let line_of = |byte: usize| line_starts.partition_point(|s| *s <= byte);

    comments
        .iter()
        .map(|c| {
            let line_start = line_starts[line_of(c.start) - 1];
            let trailing = !src[line_start..c.start].trim().is_empty();
            // Anchors arrive sorted by `start`, so the candidates for every case are a
            // contiguous prefix or suffix. Scanning them all per comment made `bind`
            // 43% of ingest on a 172 MB tree — as costly as parsing — because it is
            // O(comments x anchors). Binary search bounds the work instead.
            let after = anchors.partition_point(|a| a.start < c.end);
            let containing_end = anchors.partition_point(|a| a.start <= c.start);
            // Anchors nest, so among those containing a point the one with the LARGEST
            // start is the innermost — and therefore the smallest. Scanning backwards
            // and stopping at the first hit finds it in a step or two, where the
            // forward `filter().min_by_key()` scanned the whole prefix: on a kernel
            // header with 6,108 anchors that was the real cost of ingest, not the
            // parser, which runs at ~77 MB/s.
            let innermost = |covering: usize| {
                anchors[..containing_end]
                    .iter()
                    .rev()
                    .find(|a| a.end >= covering)
                    .cloned()
            };
            let (relation, target) = if trailing {
                (Relation::Trailing, innermost(line_start))
            } else if after < anchors.len() {
                // `partition_point` gives the FIRST anchor starting at or after the
                // comment's end, which is exactly "the element it introduces".
                (Relation::Precedes, Some(anchors[after].clone()))
            } else {
                (Relation::Encloses, innermost(c.end))
            };
            // Binary search, not a scan. `extents` is built in projection order and
            // therefore ascending, so the owner is the last one starting at or before
            // the comment. Scanning was O(comments x nodes) — measured quadratic at
            // 1/5/21 ms for 2k/4k/8k commented items — and it is the THIRD instance of
            // that same defect in this file after `bind`'s line arithmetic and
            // `project`'s newline count. It had not bitten yet only because the files
            // with the most nodes happen to have the fewest comments.
            let owner = extents
                .partition_point(|(s, _, _)| *s <= c.start)
                .checked_sub(1)
                .map(|i| &extents[i])
                .filter(|(_, e, _)| c.start < *e);
            Edge {
                kind: c.kind,
                text: src[c.start..c.end].to_string(),
                relation,
                target,
                node_order: owner.map(|(_, _, o)| *o),
                // Lines are a subtraction now, not a scan.
                line_in_node: line_of(c.start)
                    - line_of(owner.map(|(s, _, _)| *s).unwrap_or(0))
                    + 1,
            }
        })
        .collect()
}

/// What commentary applies to a byte region — everything whose TARGET falls in it.
#[allow(dead_code)]
pub fn comments_for<'a>(edges: &'a [Edge], from: usize, to: usize) -> Vec<&'a Edge> {
    edges
        .iter()
        .filter(|e| matches!(&e.target, Some(a) if a.start < to && a.end > from))
        .collect()
}

pub mod ingest;

/// Split a query into the independent things it asks for.
///
/// Retrieval that blends a whole query into one score cannot express "matches on two
/// axes at once": a document matching one axis strongly outranks one matching two axes
/// weakly. Measured on four near-identical C++ queue headers, nine document
/// representations all plateaued at 2/4 under blended scoring — and which file failed
/// was predicted by attribute uniqueness, not by text. Scoring each axis separately and
/// rank-combining reached 3/4 with the same retrieval and the same documents.
///
/// The split is deliberately dumb: clause separators only, no parsing and no model.
/// Extracting real axes from arbitrary prose is unsolved, and a heuristic that
/// over-splits costs a little precision, whereas one that invents structure would put
/// the wrong constraint on every search. A query with nothing to split on comes back as
/// itself, so single-axis queries behave exactly as before.
pub fn query_axes(query: &str) -> Vec<&str> {
    const SEPARATORS: &[&str] = &[",", " and ", " but ", " while ", " with ", " that also "];
    let mut parts = vec![query.trim()];
    for sep in SEPARATORS {
        parts = parts
            .into_iter()
            .flat_map(|p| p.split(sep))
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
    }
    // A fragment too short to constrain anything is noise, not an axis — and if the
    // split leaves nothing usable, fall back to the whole query rather than searching
    // for scraps.
    let axes: Vec<&str> = parts.into_iter().filter(|p| p.split_whitespace().count() >= 2).collect();
    if axes.len() < 2 {
        vec![query.trim()]
    } else {
        axes
    }
}

/// Combine per-axis rankings by summing ranks, best first.
///
/// Rank-combining rather than score-combining, because axes are not on a common scale:
/// comparing raw scores across them penalises whichever axis happens to sit lower, and
/// the obvious `min` combiner measured no better than blending. Ranking within each axis
/// removes the scale first, so the question becomes "is this near the top for EVERY
/// axis" without requiring the numbers to be comparable. That distinction was the whole
/// difference between 2/4 and 3/4.
///
/// A document absent from an axis's results is charged `penalty` — it ranked below
/// everything that axis returned, which is information, not a missing value.
pub fn rank_combine(per_axis: &[Vec<String>], penalty: usize) -> Vec<String> {
    use std::collections::HashMap;
    let mut points: HashMap<&str, usize> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for axis in per_axis {
        for key in axis {
            if !points.contains_key(key.as_str()) {
                points.insert(key.as_str(), 0);
                order.push(key.as_str());
            }
        }
    }
    for key in &order {
        for axis in per_axis {
            let rank = axis.iter().position(|k| k == key).unwrap_or(penalty);
            *points.get_mut(key).unwrap() += rank;
        }
    }
    // Stable on ties: first-seen order, so an axis's own ordering breaks ties rather
    // than hash iteration order making results non-deterministic between runs.
    let mut out: Vec<&str> = order.clone();
    out.sort_by_key(|k| (points[k], order.iter().position(|o| o == k).unwrap()));
    out.into_iter().map(str::to_string).collect()
}

#[cfg(test)]
mod axis_tests {
    use super::*;

    #[test]
    fn a_single_axis_query_is_unchanged() {
        assert_eq!(query_axes("lock free queue"), vec!["lock free queue"]);
        // Splitting into fragments too short to constrain anything is worse than not
        // splitting, so it falls back to the whole query.
        assert_eq!(query_axes("a, b"), vec!["a, b"]);
    }

    #[test]
    fn clause_separators_become_axes() {
        assert_eq!(
            query_axes("many producer threads and exactly one consumer"),
            vec!["many producer threads", "exactly one consumer"]
        );
        assert_eq!(
            query_axes("bounded steps per operation, many consumer threads"),
            vec!["bounded steps per operation", "many consumer threads"]
        );
    }

    #[test]
    fn rank_combining_prefers_the_document_good_on_every_axis() {
        // `a` tops the first axis and comes last on the second — the winner-take-all
        // document a blended score rewards. `b` is second then first: never the best on
        // either axis, and the best overall. Rank-combining picks `b`.
        let axes = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["b".to_string(), "c".to_string(), "a".to_string()],
        ];
        assert_eq!(rank_combine(&axes, 8)[0], "b");
    }

    #[test]
    fn a_perfect_reversal_is_a_tie_and_stays_deterministic() {
        // Worth pinning because it looks like a bug: when one axis ranks the documents
        // in exactly the reverse order of another, every document scores identically —
        // that is Borda being correct, not the combiner failing. First-seen order breaks
        // the tie so results do not vary between runs.
        let axes = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["c".to_string(), "b".to_string(), "a".to_string()],
        ];
        assert_eq!(rank_combine(&axes, 8), vec!["a", "b", "c"]);
    }

    #[test]
    fn absence_from_an_axis_is_charged_not_ignored() {
        // `x` tops one axis and is absent from the other; `y` is present in both.
        let axes = vec![
            vec!["x".to_string(), "y".to_string()],
            vec!["y".to_string()],
        ];
        assert_eq!(rank_combine(&axes, 8)[0], "y", "a missing axis must cost, not be free");
    }
}
