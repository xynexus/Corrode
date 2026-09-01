//! The fallback backend: any file, no parser.
//!
//! Corrode absorbs whatever it is pointed at, and most of that will be a language with
//! no backend written yet. This one keeps the guarantee that matters — the file is
//! ingested and projects back byte-exactly — and gives up only the guarantee that
//! costs a parser: structure. One node for the whole file, comments by marker, no
//! anchors, so comments bind to nothing and say so.
//!
//! Markers are chosen per extension because comment syntax clusters into a handful of
//! families across most of the ecosystem, and getting comments out of an unfamiliar
//! file is worth far more than getting them perfect.

use super::{CommentKind, CommentSpan, Language, Span};

/// Line- and block-comment markers for a file family.
pub struct PlainText {
    name: &'static str,
    line: &'static [&'static str],
    block: Option<(&'static str, &'static str)>,
}

impl PlainText {
    /// Marker set for an extension. Unknown extensions get the C-family defaults, which
    /// covers the majority of what a mixed repo contains; a file with no comments in
    /// that syntax simply yields none.
    pub fn for_extension(ext: &str) -> PlainText {
        match ext {
            "py" | "rb" | "sh" | "bash" | "toml" | "yaml" | "yml" | "cfg" | "conf" | "mk" => {
                PlainText { name: "hash", line: &["#"], block: None }
            }
            "sql" | "hs" | "lua" | "elm" => {
                PlainText { name: "dashdash", line: &["--"], block: None }
            }
            "lisp" | "clj" | "el" | "scm" => {
                PlainText { name: "semicolon", line: &[";"], block: None }
            }
            "html" | "xml" | "svg" | "md" => {
                PlainText { name: "markup", line: &[], block: Some(("<!--", "-->")) }
            }
            // C family: c, h, cpp, hpp, js, ts, go, java, cs, swift, kt, scala, rs…
            _ => PlainText { name: "c-family", line: &["//"], block: Some(("/*", "*/")) },
        }
    }
}

impl Language for PlainText {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extensions(&self) -> &'static [&'static str] {
        // Claims nothing: it is chosen as a fallback, never matched against.
        &[]
    }

    /// One node for the file. Byte-exact projection needs a total cover, not a smart
    /// one, and inventing item boundaries without a grammar would be a guess that
    /// silently misplaces code.
    fn items(&self, src: &str) -> anyhow::Result<Vec<Span>> {
        Ok(if src.is_empty() {
            Vec::new()
        } else {
            vec![Span { kind: "file", start: 0, end: src.len() }]
        })
    }

    /// None. A comment then binds to nothing, which is the honest answer without a
    /// grammar — better than attaching it to a line-range guess.
    fn anchors(&self, _src: &str) -> anyhow::Result<Vec<Span>> {
        Ok(Vec::new())
    }

    fn comments(&self, src: &str) -> Vec<CommentSpan> {
        let mut out = Vec::new();
        let b = src.as_bytes();
        let mut i = 0usize;
        // Advance by CHARACTER, not by byte. Slicing `src[i..]` at a non-boundary
        // panics, and a fallback that dies on an em-dash is worse than no fallback —
        // this is exactly what running it over a real repository caught.
        let step = |src: &str, i: usize| -> usize {
            src[i..].chars().next().map(|c| i + c.len_utf8()).unwrap_or(i + 1)
        };
        // Quoted-string skip, so a marker inside a string is not a comment. Coarse: no
        // raw strings or language-specific escapes, because this backend exists
        // precisely where the grammar is unknown.
        while i < b.len() {
            if b[i] == b'"' || b[i] == b'\'' {
                let q = b[i];
                i += 1;
                while i < b.len() && b[i] != q {
                    i = if b[i] == b'\\' { step(src, step(src, i)) } else { step(src, i) };
                }
                i = (i + 1).min(src.len());
                continue;
            }
            if let Some((open, close)) = self.block {
                if src[i..].starts_with(open) {
                    let end = src[i + open.len()..]
                        .find(close)
                        .map(|j| i + open.len() + j + close.len())
                        .unwrap_or(src.len());
                    out.push(CommentSpan { kind: CommentKind::Block, start: i, end });
                    i = end;
                    continue;
                }
            }
            if let Some(m) = self.line.iter().find(|m| src[i..].starts_with(**m)) {
                let start = i;
                i += m.len();
                while i < b.len() && b[i] != b'\n' {
                    i = step(src, i);
                }
                out.push(CommentSpan { kind: CommentKind::Line, start, end: i });
                continue;
            }
            i = step(src, i);
        }
        out
    }
}
