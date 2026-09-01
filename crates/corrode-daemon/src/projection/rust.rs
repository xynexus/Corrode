//! Rust backend: `syn` for spans, a hand-rolled lexer for comments.
//!
//! Everything language-specific about the projection lives here. The core needs byte
//! ranges and nothing else, so this file is the whole cost of supporting a language —
//! roughly 150 lines, most of it comment lexing.

use super::{CommentKind, CommentSpan, Language, Span};
use syn::spanned::Spanned;
use syn::visit::Visit;

pub struct Rust;

impl Language for Rust {
    fn name(&self) -> &'static str {
        "rust"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn items(&self, src: &str) -> anyhow::Result<Vec<Span>> {
        let file = syn::parse_file(src)?;
        let mut out = Vec::new();
        let mut cursor = 0usize;
        for item in &file.items {
            let r = item.span().byte_range();
            // An item's span excludes its outer attributes and doc comments, which
            // belong to it — pull the start back over them.
            let start = pull_back_attrs(src, r.start, cursor);
            out.push(Span {
                kind: kind_of(item),
                start,
                end: r.end,
            });
            cursor = r.end;
        }
        Ok(out)
    }

    fn anchors(&self, src: &str) -> anyhow::Result<Vec<Span>> {
        let file = syn::parse_file(src)?;
        let mut c = Collect::default();
        c.visit_file(&file);
        c.0.sort_by_key(|a| (a.start, std::cmp::Reverse(a.end)));
        Ok(c.0)
    }

    /// One parse for both: `syn::parse_file` dominates ingest cost for Rust, and doing
    /// it twice was measured at 72% of total time.
    fn spans(&self, src: &str) -> anyhow::Result<(Vec<Span>, Vec<Span>)> {
        let file = syn::parse_file(src)?;
        let mut items = Vec::new();
        let mut cursor = 0usize;
        for item in &file.items {
            let r = item.span().byte_range();
            let start = pull_back_attrs(src, r.start, cursor);
            items.push(Span { kind: kind_of(item), start, end: r.end });
            cursor = r.end;
        }
        let mut c = Collect::default();
        c.visit_file(&file);
        c.0.sort_by_key(|a| (a.start, std::cmp::Reverse(a.end)));
        Ok((items, c.0))
    }

    fn comments(&self, src: &str) -> Vec<CommentSpan> {
        let b = src.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < b.len() {
            // Skip literals so a `//` inside a string is never read as a comment.
            if let Some((body, hashes)) = raw_string_start(b, i) {
                if let Some(e) = raw_string_end(b, body, hashes) {
                    i = e;
                    continue;
                }
            }
            if b[i] == b'"' {
                if let Some(e) = plain_string_end(b, i) {
                    i = e;
                    continue;
                }
            }
            if b[i] == b'\'' {
                i = char_or_lifetime_end(b, i);
                continue;
            }
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                let kind = match b.get(i + 2) {
                    Some(b'/') => CommentKind::Doc,
                    Some(b'!') => CommentKind::InnerDoc,
                    _ => CommentKind::Line,
                };
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                out.push(CommentSpan { kind, start, end: i });
                continue;
            }
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                let kind = match b.get(i + 2) {
                    Some(b'*') => CommentKind::Doc,
                    Some(b'!') => CommentKind::InnerDoc,
                    _ => CommentKind::Block,
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
                out.push(CommentSpan { kind, start, end: i });
                continue;
            }
            i += 1;
        }
        out
    }
}

#[derive(Default)]
struct Collect(Vec<Span>);

impl<'ast> Visit<'ast> for Collect {
    fn visit_item(&mut self, i: &'ast syn::Item) {
        let r = i.span().byte_range();
        self.0.push(Span { kind: kind_of(i), start: r.start, end: r.end });
        syn::visit::visit_item(self, i);
    }
    fn visit_stmt(&mut self, s: &'ast syn::Stmt) {
        let r = s.span().byte_range();
        self.0.push(Span { kind: "stmt", start: r.start, end: r.end });
        syn::visit::visit_stmt(self, s);
    }
    fn visit_arm(&mut self, a: &'ast syn::Arm) {
        let r = a.span().byte_range();
        self.0.push(Span { kind: "match_arm", start: r.start, end: r.end });
        syn::visit::visit_arm(self, a);
    }
    fn visit_field(&mut self, f: &'ast syn::Field) {
        let r = f.span().byte_range();
        self.0.push(Span { kind: "field", start: r.start, end: r.end });
        syn::visit::visit_field(self, f);
    }
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

fn pull_back_attrs(src: &str, start: usize, floor: usize) -> usize {
    let mut at = start;
    loop {
        let Some(line_start) = src[floor..at].rfind('\n').map(|i| floor + i + 1) else {
            break;
        };
        let prev_end = line_start.saturating_sub(1);
        let Some(prev_start) = src[floor..prev_end].rfind('\n').map(|i| floor + i + 1) else {
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
    src[floor..at].rfind('\n').map(|i| floor + i + 1).unwrap_or(at)
}

pub(crate) fn raw_string_start(b: &[u8], i: usize) -> Option<(usize, usize)> {
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
    (j < b.len() && b[j] == b'"').then_some((j + 1, hashes))
}

pub(crate) fn raw_string_end(b: &[u8], body: usize, hashes: usize) -> Option<usize> {
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

pub(crate) fn plain_string_end(b: &[u8], i: usize) -> Option<usize> {
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
        return i + 3;
    }
    i + 1
}
