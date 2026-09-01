//! Assembling a file's nodes and edges for the store.
//!
//! Language-agnostic: it takes a [`Language`](super::Language) and does the same work
//! whatever the backend is. Ids derive from path + ordinal, so re-ingesting addresses
//! the same nodes rather than accumulating duplicates.

use super::{bind, project as project_nodes, CommentKind, Language, Relation};


/// One code node: an item, with its verbatim text.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeNode {
    pub id: String,
    pub kind: &'static str,
    pub ordinal: usize,
    pub text: String,
}

/// One comment, and what it describes.
#[derive(Debug, Clone, PartialEq)]
pub struct CommentNode {
    pub id: String,
    pub text: String,
    /// Id of the code node containing it, if any.
    pub in_node: Option<String>,
    /// 1-based line WITHIN that node. Stable under edits elsewhere in the file;
    /// an absolute line would not be, which is why none is stored.
    pub line_in_node: usize,
    /// How it relates to what it describes.
    pub relation: &'static str,
    /// Kind of the element it describes (`stmt`, `fn`, `match_arm`, …).
    pub describes_kind: Option<&'static str>,
    /// Doc comments reach many languages' ASTs as attributes; plain ones are why
    /// this module exists. Kept separable because consumers usually want one or
    /// the other.
    pub doc: bool,
}

/// Everything one file contributes to the graph, for an atomic replace.
#[derive(Debug, Clone, PartialEq)]
pub struct FileWrite {
    pub file_id: String,
    pub path: String,
    /// Which backend produced these nodes — so a consumer knows how much structure
    /// to expect, and a re-ingest after adding a backend is detectable.
    pub language: &'static str,
    pub code: Vec<CodeNode>,
    pub comments: Vec<CommentNode>,
}

/// Decompose `src` into the graph's view of it, using `lang` for the spans.
///
/// Trivia nodes are kept: they carry the whitespace that makes projection
/// byte-exact. They are marked as such so a query can ignore them.
pub fn file(lang: &dyn Language, path: &str, src: &str) -> anyhow::Result<FileWrite> {
    // One call so a backend with an expensive parser parses once (see `Language::spans`).
    // A backend with no grammar returns no anchors; comments then bind to nothing,
    // which is reported rather than guessed.
    let (items, anchors) = lang.spans(src)?;
    let nodes = super::nodes_from_items(path, src, &items);
    let edges = bind(src, &lang.comments(src), &anchors, &nodes);

    let code = nodes
        .iter()
        .map(|n| CodeNode {
            id: format!("code:{path}#{}", n.ordinal),
            kind: n.kind,
            ordinal: n.ordinal,
            text: n.text.clone(),
        })
        .collect();

    let comments = edges
        .iter()
        .enumerate()
        .map(|(i, e)| CommentNode {
            id: format!("comment:{path}#{i}"),
            text: e.text.clone(),
            in_node: e.node_ordinal.map(|o| format!("code:{path}#{o}")),
            line_in_node: e.line_in_node,
            relation: match e.relation {
                Relation::Precedes => "precedes",
                Relation::Trailing => "trailing",
                Relation::Encloses => "encloses",
            },
            describes_kind: e.target.as_ref().map(|a| a.kind),
            doc: matches!(e.kind, CommentKind::Doc | CommentKind::InnerDoc),
        })
        .collect();

    Ok(FileWrite {
        file_id: format!("file:{path}"),
        path: path.to_string(),
        language: lang.name(),
        code,
        comments,
    })
}

/// Project a file back from its code nodes — the inverse of [`file`].
#[allow(dead_code)]
pub fn project(fw: &FileWrite) -> String {
    let nodes: Vec<super::Node> = fw
        .code
        .iter()
        .map(|c| super::Node {
            path: fw.path.clone(),
            ordinal: c.ordinal,
            kind: c.kind,
            text: c.text.clone(),
        })
        .collect();
    project_nodes(&nodes).0
}
