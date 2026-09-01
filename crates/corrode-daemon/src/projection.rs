//! Source <-> node decomposition: the projection half of the graph model.
//!
//! `graph-model.md` makes files a projection of the graph. This is the machinery that
//! makes that true for Rust: decompose a file into item nodes carrying VERBATIM text,
//! recover the comments the lexer discarded, bind them to the elements they describe,
//! and project nodes back into a file.
//!
//! Measured before being relied on (see `harness-architecture.md` §8): composition is
//! byte-exact for 1515 files across three repositories, and 44,562 of 44,574 plain
//! comments bind to a syntax element. The measurements themselves live in
//! `roundtrip.rs`, which stays test-only.
//!
//! One rule holds the design together: **positions are an output, never state.** A byte
//! offset is a fact about one source text, and a dynamically generated VFS has none.

/// The composer itself: scan a repo into nodes, regenerate it, diff the tree.
///
/// Tiers 1 and 2 each measured a *proxy*. Tier 1 counted bytes inside spans found by a
/// hand-rolled scanner; tier 2 round-tripped whole files through a printer nobody
/// proposed using. Neither built the thing under test — a node model — so neither could
/// answer whether a repo survives decomposition and reassembly.
///
/// This does. `syn` supplies the item boundaries (real structure, not a brace count);
/// each node stores its VERBATIM bytes; regeneration concatenates. Byte-exactness is
/// then a property of the decomposition, not of a printer's manners.
pub mod compose {
    use proc_macro2::Span;
    use syn::spanned::Spanned;

    /// One node as the graph would hold it.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Node {
        pub path: String,
        /// Order within the file — the only thing regeneration needs.
        pub ordinal: usize,
        /// `fn`, `struct`, `impl`, `use`, `mod`, `macro`, … from syn, or `trivia` for
        /// the whitespace and free-standing comments between items.
        pub kind: &'static str,
        /// Verbatim source. NOT regenerated: tier 2 measured that regeneration loses
        /// comments and canonicalises punctuation, so content is stored, not printed.
        pub text: String,
        /// 1-based line at SCAN time only, and not authoritative.
        ///
        /// Kept because the ingest census reports it, but a projector must never read
        /// it: a byte offset or line number is a fact about one source text, and a
        /// dynamically generated VFS has no such text. Files are produced FROM nodes,
        /// possibly reordered, edited, or drawn from a branch that was never
        /// materialised — so a stored position is either stale or meaningless. Use
        /// [`compose::project`], which computes positions as an OUTPUT.
        pub scan_line: usize,
    }

    /// Public alias so the `describes` visitor can label item anchors identically.
    pub fn kind_of_public(item: &syn::Item) -> &'static str {
        kind_of(item)
    }

    fn kind_of(item: &syn::Item) -> &'static str {
        match item {
            syn::Item::Fn(_) => "fn",
            syn::Item::Struct(_) => "struct",
            syn::Item::Enum(_) => "enum",
            syn::Item::Impl(_) => "impl",
            syn::Item::Trait(_) => "trait",
            syn::Item::Mod(_) => "mod",
            syn::Item::Use(_) => "use",
            syn::Item::Const(_) => "const",
            syn::Item::Static(_) => "static",
            syn::Item::Type(_) => "type",
            syn::Item::Macro(_) => "macro",
            syn::Item::ExternCrate(_) => "extern_crate",
            syn::Item::ForeignMod(_) => "foreign_mod",
            syn::Item::TraitAlias(_) => "trait_alias",
            syn::Item::Union(_) => "union",
            _ => "item",
        }
    }

    /// Byte range of a span, using proc-macro2's `span-locations`.
    fn range(span: Span) -> std::ops::Range<usize> {
        span.byte_range()
    }

    /// Decompose one file into nodes covering every byte.
    ///
    /// An item's span does NOT include its outer attributes or doc comments, so those
    /// would land in trivia and lose their association with the item they document.
    /// Each item's start is pulled back over any immediately preceding attribute and
    /// doc-comment lines, which is where they belong.
    pub fn scan(path: &str, src: &str) -> anyhow::Result<Vec<Node>> {
        let file = syn::parse_file(src)?;
        let mut nodes = Vec::new();
        let mut cursor = 0usize;
        let mut ordinal = 0usize;

        for item in &file.items {
            let r = range(item.span());
            let start = pull_back_attrs(src, r.start, cursor);
            if start > cursor {
                push_trivia(&mut nodes, path, &mut ordinal, src, cursor, start);
            }
            nodes.push(Node {
                path: path.to_string(),
                ordinal,
                kind: kind_of(item),
                text: src[start..r.end].to_string(),
                scan_line: line_of(src, start),
            });
            ordinal += 1;
            cursor = r.end;
        }
        if cursor < src.len() {
            push_trivia(&mut nodes, path, &mut ordinal, src, cursor, src.len());
        }
        Ok(nodes)
    }

    fn push_trivia(
        nodes: &mut Vec<Node>,
        path: &str,
        ordinal: &mut usize,
        src: &str,
        from: usize,
        to: usize,
    ) {
        nodes.push(Node {
            path: path.to_string(),
            ordinal: *ordinal,
            kind: "trivia",
            text: src[from..to].to_string(),
            scan_line: line_of(src, from),
        });
        *ordinal += 1;
    }

    /// Walk back from an item's span over attribute and doc-comment lines directly
    /// above it, stopping at `floor` (the previous item's end) or a blank line.
    fn pull_back_attrs(src: &str, start: usize, floor: usize) -> usize {
        let mut at = start;
        loop {
            let Some(line_start) = src[floor..at].rfind('\n').map(|i| floor + i + 1) else {
                break;
            };
            let prev_end = line_start.saturating_sub(1);
            let Some(prev_start) = src[floor..prev_end].rfind('\n').map(|i| floor + i + 1) else {
                // First line of the region.
                let l = src[floor..prev_end].trim_start();
                if l.starts_with("#[") || l.starts_with("///") || l.starts_with("//!") {
                    return floor;
                }
                break;
            };
            let line = src[prev_start..prev_end].trim();
            if line.starts_with("#[") || line.starts_with("///") || line.starts_with("//!") {
                at = prev_start;
                continue;
            }
            break;
        }
        // `at` now points at the first attribute line, or the original start.
        src[floor..at].rfind('\n').map(|i| floor + i + 1).unwrap_or(at)
    }

    fn line_of(src: &str, byte: usize) -> usize {
        src[..byte].bytes().filter(|b| *b == b'\n').count() + 1
    }


    /// Where a node landed in a projection. Produced BY projection, never stored.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Placement {
        pub ordinal: usize,
        pub kind: &'static str,
        /// 1-based line where this node starts in the projected file.
        pub start_line: usize,
        pub start_byte: usize,
    }

    /// Project nodes into a file AND report where each one landed.
    ///
    /// The read half of the projection. Exercised by tests and by the census; the
    /// daemon calls it once the VFS materialises from the graph instead of passing
    /// through to disk (`harness-architecture.md` §8, "graph is the source of truth").
    #[allow(dead_code)]
    ///
    /// This is the operation a dynamic VFS actually performs. Positions come out of it;
    /// they are never inputs. Reorder the nodes, insert one, edit one — the next
    /// projection reports the new truth, and nothing had to be invalidated because
    /// nothing had been stored.
    pub fn project(nodes: &[Node]) -> (String, Vec<Placement>) {
        let mut sorted: Vec<&Node> = nodes.iter().collect();
        sorted.sort_by_key(|n| n.ordinal);
        let mut text = String::new();
        let mut places = Vec::with_capacity(sorted.len());
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

    /// Reassemble a file from its nodes. This is the whole composer.
    #[allow(dead_code)]
    pub fn regenerate(nodes: &[Node]) -> String {
        let mut out = String::new();
        let mut sorted: Vec<&Node> = nodes.iter().collect();
        sorted.sort_by_key(|n| n.ordinal);
        for n in sorted {
            out.push_str(&n.text);
        }
        out
    }
}


/// The comment-extraction pass.
///
/// Comments never reach the AST — Rust's lexer discards them — so recovering them is a
/// SEPARATE pass over the source, run alongside the parse rather than derived from it.
/// This is that pass: find every comment lexically, then bind it to the node that
/// contains it and to a line within that node.
///
/// Granularity worth being precise about. Byte offsets resolve a comment to a *line*
/// inside a node, which is what a reader needs and what `path:line` reporting wants.
/// They do not resolve it to a syntactic *element* — "the comment on this match arm" —
/// because item spans stop at item boundaries. That needs a lossless CST
/// (`ra_ap_syntax`). Line-level does not.
pub mod comments {
    use super::compose::Node;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Kind {
        /// `// ...`
        Line,
        /// `/* ... */`
        Block,
        /// `/// ...` or `/** ... */` — survives into the AST as `#[doc]`.
        Doc,
        /// `//! ...` — inner doc.
        InnerDoc,
    }

    /// One comment, bound to where it lives.
    #[derive(Debug, Clone, PartialEq)]
    pub struct CommentRef {
        pub path: String,
        /// Which node contains it. `None` only if it precedes every node.
        pub node_ordinal: Option<usize>,
        /// 1-based line WITHIN that node — stable under edits elsewhere in the file.
        pub line_in_node: usize,
        /// 1-based line in the file as it currently projects. Derived, not stored.
        pub abs_line: usize,
        pub kind: Kind,
        pub text: String,
    }

    /// Find every comment in `src`, skipping string, raw-string and char literals so a
    /// `//` inside a string is never mistaken for one.
    pub fn extract(path: &str, src: &str, nodes: &[Node]) -> Vec<CommentRef> {
        let b = src.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;

        // Byte ranges of each node, in ordinal order, so a comment can be located.
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, ordinal)
        let mut at = 0usize;
        let mut sorted: Vec<&Node> = nodes.iter().collect();
        sorted.sort_by_key(|n| n.ordinal);
        for n in &sorted {
            ranges.push((at, at + n.text.len(), n.ordinal));
            at += n.text.len();
        }

        while i < b.len() {
            if let Some((body, hashes)) = super::raw_string_start(b, i) {
                if let Some(e) = super::raw_string_end(b, body, hashes) {
                    i = e;
                    continue;
                }
            }
            if b[i] == b'"' {
                if let Some(e) = super::plain_string_end(b, i) {
                    i = e;
                    continue;
                }
            }
            if b[i] == b'\'' {
                i = super::char_or_lifetime_end(b, i);
                continue;
            }
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                let kind = match b.get(i + 2) {
                    Some(b'/') => Kind::Doc,
                    Some(b'!') => Kind::InnerDoc,
                    _ => Kind::Line,
                };
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                out.push(make_ref(path, src, &ranges, start, i, kind));
                continue;
            }
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                let kind = match b.get(i + 2) {
                    Some(b'*') => Kind::Doc,
                    Some(b'!') => Kind::InnerDoc,
                    _ => Kind::Block,
                };
                let start = i;
                let mut nest = 1;
                i += 2;
                while i + 1 < b.len() && nest > 0 {
                    if b[i] == b'/' && b[i + 1] == b'*' {
                        nest += 1;
                        i += 2;
                    } else if b[i] == b'*' && b[i + 1] == b'/' {
                        nest -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                out.push(make_ref(path, src, &ranges, start, i, kind));
                continue;
            }
            i += 1;
        }
        out
    }

    fn make_ref(
        path: &str,
        src: &str,
        ranges: &[(usize, usize, usize)],
        start: usize,
        end: usize,
        kind: Kind,
    ) -> CommentRef {
        let owner = ranges
            .iter()
            .find(|(s, e, _)| start >= *s && start < *e)
            .map(|(s, _, ord)| (*s, *ord));
        let (node_start, node_ordinal) = match owner {
            Some((s, o)) => (s, Some(o)),
            None => (0, None),
        };
        CommentRef {
            path: path.to_string(),
            node_ordinal,
            line_in_node: src[node_start..start].bytes().filter(|c| *c == b'\n').count() + 1,
            abs_line: src[..start].bytes().filter(|c| *c == b'\n').count() + 1,
            kind,
            text: src[start..end].to_string(),
        }
    }
}


/// What a comment DESCRIBES, as an edge rather than a position.
///
/// Graph search asks two things a coordinate cannot answer: what is this comment
/// about, and what commentary applies to this region of code. Both need an edge from
/// the comment to the syntax element it annotates.
///
/// `syn` gives spans for any syntax node, not just top-level items — a statement inside
/// a body has a byte range like anything else — so anchors go all the way down without
/// a lossless CST. That corrects an earlier note in this module.
pub mod describes {
    use super::comments::{CommentRef, Kind};
    use syn::spanned::Spanned;
    use syn::visit::Visit;

    /// A syntax element a comment can be about.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Anchor {
        pub kind: &'static str,
        pub start: usize,
        pub end: usize,
    }

    /// How the comment relates to what it describes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Relation {
        /// On its own line(s), immediately above the element.
        Precedes,
        /// After code on the same line — describes that line's element.
        Trailing,
        /// Nothing follows it in scope; it belongs to the element enclosing it.
        Encloses,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Edge {
        pub comment: CommentRef,
        pub relation: Relation,
        pub target: Option<Anchor>,
    }

    #[derive(Default)]
    struct Collect {
        anchors: Vec<Anchor>,
    }

    impl<'ast> Visit<'ast> for Collect {
        fn visit_item(&mut self, i: &'ast syn::Item) {
            let s = i.span().byte_range();
            self.anchors.push(Anchor {
                kind: super::compose::kind_of_public(i),
                start: s.start,
                end: s.end,
            });
            syn::visit::visit_item(self, i);
        }
        fn visit_stmt(&mut self, s: &'ast syn::Stmt) {
            let r = s.span().byte_range();
            self.anchors.push(Anchor {
                kind: "stmt",
                start: r.start,
                end: r.end,
            });
            syn::visit::visit_stmt(self, s);
        }
        fn visit_arm(&mut self, a: &'ast syn::Arm) {
            let r = a.span().byte_range();
            self.anchors.push(Anchor {
                kind: "match_arm",
                start: r.start,
                end: r.end,
            });
            syn::visit::visit_arm(self, a);
        }
        fn visit_field(&mut self, f: &'ast syn::Field) {
            let r = f.span().byte_range();
            self.anchors.push(Anchor {
                kind: "field",
                start: r.start,
                end: r.end,
            });
            syn::visit::visit_field(self, f);
        }
    }

    /// Every anchorable element in the file, innermost-last.
    pub fn anchors(src: &str) -> anyhow::Result<Vec<Anchor>> {
        let file = syn::parse_file(src)?;
        let mut c = Collect::default();
        c.visit_file(&file);
        c.anchors.sort_by_key(|a| (a.start, std::cmp::Reverse(a.end)));
        Ok(c.anchors)
    }

    /// Bind each comment to what it describes.
    pub fn bind(src: &str, comments: &[CommentRef], anchors: &[Anchor]) -> Vec<Edge> {
        comments
            .iter()
            .map(|c| {
                let start = byte_of(src, c);
                let end = start + c.text.len();
                // Is there code before it on this line? Then it trails that code.
                let line_start = src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
                let trailing = !src[line_start..start].trim().is_empty();

                let (relation, target) = if trailing {
                    // The innermost anchor whose span covers the code to its left.
                    let t = anchors
                        .iter()
                        .filter(|a| a.start <= start && a.end >= line_start)
                        .min_by_key(|a| a.end - a.start)
                        .cloned();
                    (Relation::Trailing, t)
                } else {
                    // The first element starting after it — what it introduces.
                    match anchors.iter().filter(|a| a.start >= end).min_by_key(|a| a.start) {
                        Some(a) => (Relation::Precedes, Some(a.clone())),
                        None => {
                            let t = anchors
                                .iter()
                                .filter(|a| a.start <= start && a.end >= end)
                                .min_by_key(|a| a.end - a.start)
                                .cloned();
                            (Relation::Encloses, t)
                        }
                    }
                };
                Edge {
                    comment: c.clone(),
                    relation,
                    target,
                }
            })
            .collect()
    }

    /// Query 2: what commentary applies to a byte region of the source?
    ///
    /// Read-side API for graph search; no caller in the daemon until search queries the
    /// graph rather than the filesystem.
    #[allow(dead_code)]
    ///
    /// Everything whose TARGET falls in the region — so a comment introducing a
    /// function is returned when asking about that function's body, not merely when
    /// asking about the line it sits on.
    pub fn comments_for(edges: &[Edge], from: usize, to: usize) -> Vec<&Edge> {
        edges
            .iter()
            .filter(|e| match &e.target {
                Some(a) => a.start < to && a.end > from,
                None => false,
            })
            .collect()
    }

    /// Doc comments already reach the AST as attributes; plain ones are the reason this
    /// module exists.
    #[allow(dead_code)]
    pub fn is_plain(c: &CommentRef) -> bool {
        matches!(c.kind, Kind::Line | Kind::Block)
    }

    fn byte_of(src: &str, c: &CommentRef) -> usize {
        // `abs_line` is 1-based; walk to that line then find the comment text on it.
        let mut at = 0usize;
        for _ in 1..c.abs_line {
            match src[at..].find('\n') {
                Some(i) => at += i + 1,
                None => break,
            }
        }
        let line_end = src[at..].find('\n').map(|i| at + i).unwrap_or(src.len());
        let first = c.text.lines().next().unwrap_or("");
        src[at..line_end].find(first).map(|i| at + i).unwrap_or(at)
    }
}


/// Lexer helpers shared by the scanners: skipping literals correctly is what keeps a
/// `//` inside a string from being read as a comment.
pub(super) fn raw_string_start(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let mut j = i;
    if b[j] == b'b' {
        j += 1;
    }
    if j >= b.len() || b[j] != b'r' {
        return None;
    }
    j += 1;
    let mut hashes = 0usize;
    while j < b.len() && b[j] == b'#' {
        hashes += 1;
        j += 1;
    }
    if j < b.len() && b[j] == b'"' {
        Some((j + 1, hashes))
    } else {
        None
    }
}

pub(super) fn raw_string_end(b: &[u8], body: usize, hashes: usize) -> Option<usize> {
    let mut j = body;
    while j < b.len() {
        if b[j] == b'"' && b[j + 1..].iter().take(hashes).all(|c| *c == b'#') && j + hashes < b.len()
        {
            return Some(j + 1 + hashes);
        }
        j += 1;
    }
    None
}

pub(super) fn plain_string_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

pub(crate) fn char_or_lifetime_end(b: &[u8], i: usize) -> usize {
    if i + 2 < b.len() && b[i + 1] == b'\\' {
        let mut j = i + 2;
        while j < b.len() && b[j] != b'\'' {
            j += 1;
        }
        return j + 1;
    }
    if i + 2 < b.len() && b[i + 2] == b'\'' {
        return i + 3; // 'x'
    }
    i + 1 // lifetime
}

/// Turning a source file into the node and edge set the graph stores.
///
/// This is the ingest half of the projection: `compose` decomposes, `comments`
/// recovers what the lexer dropped, `describes` binds it, and this assembles the
/// result into something a store can write atomically.
///
/// Ids are derived from the path and the node's ordinal, so re-ingesting a file
/// addresses the same nodes rather than accumulating duplicates — the same property
/// `replace_doc` relies on for documents.
pub mod ingest {
    use super::{comments, compose, describes};

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
    }

    /// Everything one file contributes to the graph, for an atomic replace.
    #[derive(Debug, Clone, PartialEq)]
    pub struct FileWrite {
        pub file_id: String,
        pub path: String,
        pub code: Vec<CodeNode>,
        pub comments: Vec<CommentNode>,
    }

    /// Decompose `src` into the graph's view of it.
    ///
    /// Trivia nodes are kept: they carry the whitespace that makes projection
    /// byte-exact. They are marked as such so a query can ignore them.
    pub fn file(path: &str, src: &str) -> anyhow::Result<FileWrite> {
        let nodes = compose::scan(path, src)?;
        let refs = comments::extract(path, src, &nodes);
        let anchors = describes::anchors(src)?;
        let edges = describes::bind(src, &refs, &anchors);

        let code: Vec<CodeNode> = nodes
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
                text: e.comment.text.clone(),
                in_node: e
                    .comment
                    .node_ordinal
                    .map(|o| format!("code:{path}#{o}")),
                line_in_node: e.comment.line_in_node,
                relation: match e.relation {
                    describes::Relation::Precedes => "precedes",
                    describes::Relation::Trailing => "trailing",
                    describes::Relation::Encloses => "encloses",
                },
                describes_kind: e.target.as_ref().map(|a| a.kind),
            })
            .collect();

        Ok(FileWrite {
            file_id: format!("file:{path}"),
            path: path.to_string(),
            code,
            comments,
        })
    }

    /// Project a file back from its code nodes — the inverse of [`file`].
    #[allow(dead_code)]
    pub fn project(fw: &FileWrite) -> String {
        let mut sorted: Vec<&CodeNode> = fw.code.iter().collect();
        sorted.sort_by_key(|c| c.ordinal);
        sorted.iter().map(|c| c.text.as_str()).collect()
    }
}

#[cfg(test)]
mod ingest_tests {
    use super::ingest::*;

    const SRC: &str = "\
use std::fmt;

/// documented
fn outer() {
    // sets the seed
    let seed = 7;
}
";

    #[test]
    fn a_file_becomes_nodes_that_project_back_exactly() {
        let fw = file("t.rs", SRC).unwrap();
        assert_eq!(fw.file_id, "file:t.rs");
        // The whole point: the graph's view of the file reproduces the file.
        assert_eq!(project(&fw), SRC, "projection from nodes is byte-exact");

        let kinds: Vec<&str> = fw.code.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&"use") && kinds.contains(&"fn"), "{kinds:?}");
        // Ids are derived from path + ordinal, so re-ingest addresses the same nodes
        // instead of accumulating duplicates.
        let again = file("t.rs", SRC).unwrap();
        assert_eq!(
            fw.code.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            again.code.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn comments_carry_their_binding_into_the_graph() {
        let fw = file("t.rs", SRC).unwrap();
        let seed = fw
            .comments
            .iter()
            .find(|c| c.text.contains("sets the seed"))
            .expect("comment ingested");
        assert_eq!(seed.relation, "precedes");
        assert_eq!(seed.describes_kind, Some("stmt"), "binds to the statement below it");
        assert!(seed.in_node.is_some(), "knows which code node contains it");
        assert!(seed.line_in_node >= 1);

        // A doc comment reaches the AST as an attribute, so it also lives inside its
        // item's verbatim text — it is ingested as a comment too, deliberately: the
        // graph should be able to answer "what documents this" without re-parsing.
        assert!(
            fw.comments.iter().any(|c| c.text.contains("documented")),
            "doc comments are ingested as well"
        );
    }

    /// Re-ingesting an edited file yields the same ids for surviving nodes and a
    /// shorter list when items are removed — which is what makes atomic
    /// replace-and-prune correct rather than merely convenient.
    #[test]
    fn re_ingest_is_addressable_and_shrinks() {
        let before = file("t.rs", SRC).unwrap();
        let shrunk = "use std::fmt;\n";
        let after = file("t.rs", shrunk).unwrap();
        assert!(after.code.len() < before.code.len(), "fewer nodes after removal");
        assert_eq!(after.code[0].id, before.code[0].id, "surviving node keeps its id");
        assert_eq!(project(&after), shrunk);
    }
}
