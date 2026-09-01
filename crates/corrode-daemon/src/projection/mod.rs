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

pub mod rust;
pub mod text;

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
    Box::new(text::PlainText::for_extension(ext))
}

/// One node: a slice of the file, stored verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub path: String,
    /// Order within the file — the only thing projection needs.
    pub ordinal: usize,
    /// Backend-defined kind, or `trivia` for the bytes between items.
    pub kind: &'static str,
    /// Verbatim source. Never regenerated.
    pub text: String,
}

/// Decompose a file into nodes covering every byte.
pub fn scan(lang: &dyn Language, path: &str, src: &str) -> anyhow::Result<Vec<Node>> {
    let items = lang.items(src)?;
    let mut nodes = Vec::new();
    let (mut cursor, mut ordinal) = (0usize, 0usize);
    for it in &items {
        if it.start > cursor {
            nodes.push(Node {
                path: path.into(),
                ordinal,
                kind: "trivia",
                text: src[cursor..it.start].into(),
            });
            ordinal += 1;
        }
        nodes.push(Node {
            path: path.into(),
            ordinal,
            kind: it.kind,
            text: src[it.start..it.end].into(),
        });
        ordinal += 1;
        cursor = it.end;
    }
    if cursor < src.len() {
        nodes.push(Node {
            path: path.into(),
            ordinal,
            kind: "trivia",
            text: src[cursor..].into(),
        });
    }
    Ok(nodes)
}

/// Where a node landed. Produced BY projection, never stored: a byte offset is a fact
/// about one source text, and a generated VFS has none.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub ordinal: usize,
    pub kind: &'static str,
    pub start_line: usize,
    pub start_byte: usize,
}

/// Project nodes into a file and report where each landed.
pub fn project(nodes: &[Node]) -> (String, Vec<Placement>) {
    let mut sorted: Vec<&Node> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.ordinal);
    let (mut text, mut places) = (String::new(), Vec::with_capacity(sorted.len()));
    for n in sorted {
        places.push(Placement {
            ordinal: n.ordinal,
            kind: n.kind,
            start_line: text.bytes().filter(|b| *b == b'\n').count() + 1,
            start_byte: text.len(),
        });
        text.push_str(&n.text);
    }
    (text, places)
}

/// Reassemble a file from its nodes.
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
    /// Which node contains it.
    pub node_ordinal: Option<usize>,
    /// 1-based line WITHIN that node — stable under edits elsewhere in the file.
    pub line_in_node: usize,
}

/// Bind comments to the elements they describe.
///
/// Language-agnostic by construction: it sees byte ranges and nothing else. This is the
/// part worth sharing — the relation a comment has to code is the same question in
/// every language, even though finding the spans is not.
pub fn bind(src: &str, comments: &[CommentSpan], anchors: &[Span], nodes: &[Node]) -> Vec<Edge> {
    // Node extents in projection order, so a comment can be located within one.
    let mut sorted: Vec<&Node> = nodes.iter().collect();
    sorted.sort_by_key(|n| n.ordinal);
    let mut extents: Vec<(usize, usize, usize)> = Vec::new();
    let mut at = 0usize;
    for n in &sorted {
        extents.push((at, at + n.text.len(), n.ordinal));
        at += n.text.len();
    }

    comments
        .iter()
        .map(|c| {
            let line_start = src[..c.start].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let trailing = !src[line_start..c.start].trim().is_empty();
            let (relation, target) = if trailing {
                (
                    Relation::Trailing,
                    anchors
                        .iter()
                        .filter(|a| a.start <= c.start && a.end >= line_start)
                        .min_by_key(|a| a.end - a.start)
                        .cloned(),
                )
            } else {
                match anchors.iter().filter(|a| a.start >= c.end).min_by_key(|a| a.start) {
                    Some(a) => (Relation::Precedes, Some(a.clone())),
                    None => (
                        Relation::Encloses,
                        anchors
                            .iter()
                            .filter(|a| a.start <= c.start && a.end >= c.end)
                            .min_by_key(|a| a.end - a.start)
                            .cloned(),
                    ),
                }
            };
            let owner = extents.iter().find(|(s, e, _)| c.start >= *s && c.start < *e);
            Edge {
                kind: c.kind,
                text: src[c.start..c.end].to_string(),
                relation,
                target,
                node_ordinal: owner.map(|(_, _, o)| *o),
                line_in_node: src[owner.map(|(s, _, _)| *s).unwrap_or(0)..c.start]
                    .bytes()
                    .filter(|b| *b == b'\n')
                    .count()
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
