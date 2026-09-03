//! C and C++ backend: a lexer, not a parser.
//!
//! The projection needs byte ranges, never regeneration, so a full parse is not
//! required — which is fortunate, because parsing C properly needs a preprocessor.
//! What IS required is lexing C correctly, and the failure mode of getting it wrong is
//! silent: a miscounted brace shifts every subsequent item boundary without any error.
//!
//! # The gotchas, and which ones are real
//!
//! Checked against the kernel tree rather than assumed:
//!
//! - **Preprocessor directives with unbalanced braces.** The critical one, and common:
//!   `# define randomized_struct_fields_start struct {` opens a brace that never
//!   closes. A depth counter that sees it is wrong for the rest of the file. Directives
//!   are therefore lexed as OPAQUE regions — consumed whole, never depth-counted — and
//!   `#` may be separated from its keyword by whitespace.
//! - **Multi-line directives.** `#define X \` continues while lines end in a backslash,
//!   so the opaque region spans all of them.
//! - **Block comments do not nest in C**, unlike Rust: `/* /* */` ends at the first
//!   `*/`. Nesting them would swallow the rest of the file.
//! - **Backslash-continued `//` comments** legally span lines. Measured at zero
//!   occurrences in the kernel sample, but handled — it costs three lines and the
//!   failure is silent.
//! - **Apostrophes.** `'` starts a character literal in code and means nothing inside a
//!   comment (`don't`), so char-literal detection runs only in code regions.
//! - **Comment markers inside strings.** `"http://x"` is not a comment.
//!
//! Not handled, deliberately: trigraphs (`??/` for a backslash) and digraphs, removed
//! in C23 and absent from the corpus; and `#if 0` blocks, whose contents are lexed as
//! ordinary code — 2 files in the kernel sample, and treating them otherwise would mean
//! evaluating the preprocessor.

use super::{CommentKind, CommentSpan, Language, Span};

pub struct C;

/// What a byte belongs to. Depth is counted only in [`Region::Code`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum Region {
    Code,
    LineComment,
    BlockComment,
    /// A preprocessor line and its backslash continuations.
    Directive,
    Str,
    Char,
}

/// One lexical pass; everything else is derived from it.
fn lex(src: &str) -> Vec<(Region, usize, usize)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut code_start = 0usize;

    // Is `i` the first non-whitespace byte on its line? Directives only start there.
    let at_line_start = |b: &[u8], i: usize| {
        let mut j = i;
        while j > 0 && b[j - 1] != b'\n' {
            j -= 1;
            if !b[j].is_ascii_whitespace() {
                return false;
            }
        }
        true
    };

    while i < b.len() {
        let start = i;
        match b[i] {
            b'#' if at_line_start(b, i) => {
                // Opaque through the end of the line, following `\` continuations.
                while i < b.len() {
                    if b[i] == b'\n' {
                        let cont = i > 0 && b[i - 1] == b'\\';
                        i += 1;
                        if !cont {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
                push_code(&mut out, code_start, start);
                out.push((Region::Directive, start, i));
                code_start = i;
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                while i < b.len() {
                    if b[i] == b'\n' {
                        // A backslash before the newline continues the comment.
                        let cont = i > 0 && b[i - 1] == b'\\';
                        if !cont {
                            break;
                        }
                    }
                    i += 1;
                }
                push_code(&mut out, code_start, start);
                out.push((Region::LineComment, start, i));
                code_start = i;
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                i += 2;
                // NOT nested: the first `*/` closes it.
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
                push_code(&mut out, code_start, start);
                out.push((Region::BlockComment, start, i));
                code_start = i;
            }
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i = (i + 1).min(b.len());
                push_code(&mut out, code_start, start);
                out.push((Region::Str, start, i));
                code_start = i;
            }
            b'\'' => {
                i += 1;
                while i < b.len() && b[i] != b'\'' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i = (i + 1).min(b.len());
                push_code(&mut out, code_start, start);
                out.push((Region::Char, start, i));
                code_start = i;
            }
            _ => i += 1,
        }
    }
    push_code(&mut out, code_start, b.len());
    out
}

fn push_code(out: &mut Vec<(Region, usize, usize)>, from: usize, to: usize) {
    if to > from {
        out.push((Region::Code, from, to));
    }
}

/// A crude kind for an item, from its leading keyword. Free-form by contract — the core
/// never interprets it — so a wrong guess costs a label, not correctness.
fn kind_of(text: &str) -> &'static str {
    let t = text.trim_start();
    for (prefix, kind) in [
        ("typedef", "typedef"),
        ("struct", "struct"),
        ("union", "union"),
        ("enum", "enum"),
        ("class", "class"),
        ("namespace", "namespace"),
        ("template", "template"),
        ("extern \"C\"", "extern_c"),
    ] {
        if t.starts_with(prefix) {
            return kind;
        }
    }
    if t.ends_with('}') {
        "definition"
    } else {
        "decl"
    }
}

/// Walk the lexed regions, counting brace depth only in code, and emit spans.
///
/// `deep` controls whether statements inside bodies are emitted too — items want top
/// level only, anchors want everything a comment could describe.
fn spans_from(src: &str, deep: bool) -> Vec<Span> {
    let b = src.as_bytes();
    let regions = lex(src);
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut item_start: Option<usize> = None;
    let mut prev_end = 0usize;

    // Where the next item begins: the first non-whitespace byte after the previous one.
    let begin = |from: usize| {
        let mut j = from;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        j
    };

    for (region, rs, re) in &regions {
        match region {
            // A directive is an item in its own right and never affects depth.
            Region::Directive if depth == 0 => {
                out.push(Span { kind: "directive", start: *rs, end: *re });
                prev_end = *re;
                item_start = None;
                continue;
            }
            Region::Code => {}
            _ => continue, // comments and literals never move depth
        }

        for i in *rs..*re {
            match b[i] {
                b'{' => {
                    if depth == 0 && item_start.is_none() {
                        item_start = Some(begin(prev_end));
                    }
                    depth += 1;
                }
                b'}' => {
                    depth -= 1;
                    if depth < 0 {
                        // Unbalanced — a macro body or `#if` arm we cannot see through.
                        // Reset rather than emit nonsense from here on.
                        depth = 0;
                        item_start = None;
                        prev_end = i + 1;
                    } else if depth == 0 {
                        let s = item_start.take().unwrap_or_else(|| begin(prev_end));
                        out.push(Span { kind: kind_of(&src[s..i + 1]), start: s, end: i + 1 });
                        prev_end = i + 1;
                    } else if deep {
                        // Closing an inner block: an anchor for whatever it belongs to.
                        out.push(Span { kind: "block", start: begin(prev_end), end: i + 1 });
                    }
                }
                b';' if depth == 0 => {
                    let s = item_start.take().unwrap_or_else(|| begin(prev_end));
                    out.push(Span { kind: kind_of(&src[s..i + 1]), start: s, end: i + 1 });
                    prev_end = i + 1;
                }
                b';' if deep => {
                    // A statement inside a body — what most C comments describe.
                    let s = begin(prev_end.max(stmt_floor(&out, prev_end)));
                    if i + 1 > s {
                        out.push(Span { kind: "stmt", start: s, end: i + 1 });
                    }
                    prev_end = i + 1;
                }
                _ => {}
            }
        }
    }
    out.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
    out
}

/// Statements start after the previous emitted span, so they cannot overlap it.
fn stmt_floor(out: &[Span], prev_end: usize) -> usize {
    out.last().map(|s| s.end.max(prev_end)).unwrap_or(prev_end)
}

impl Language for C {
    fn name(&self) -> &'static str {
        "c"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "dts", "dtsi", "dtso"]
    }

    fn items(&self, src: &str) -> anyhow::Result<Vec<Span>> {
        // Top level only, and non-overlapping: the node cover must partition the file.
        let mut items = spans_from(src, false);
        items.retain(|s| s.end > s.start);
        let mut out: Vec<Span> = Vec::with_capacity(items.len());
        for s in items {
            if out.last().map(|p| s.start >= p.end).unwrap_or(true) {
                out.push(s);
            }
        }
        Ok(out)
    }

    fn anchors(&self, src: &str) -> anyhow::Result<Vec<Span>> {
        Ok(spans_from(src, true))
    }

    fn comments(&self, src: &str) -> Vec<CommentSpan> {
        lex(src)
            .into_iter()
            .filter_map(|(r, s, e)| match r {
                // `/** …` and `/*! …` are doc conventions in C; `///` likewise.
                Region::BlockComment => Some(CommentSpan {
                    kind: match src.as_bytes().get(s + 2) {
                        Some(b'*') | Some(b'!') => CommentKind::Doc,
                        _ => CommentKind::Block,
                    },
                    start: s,
                    end: e,
                }),
                Region::LineComment => Some(CommentSpan {
                    kind: match src.as_bytes().get(s + 2) {
                        Some(b'/') | Some(b'!') => CommentKind::Doc,
                        _ => CommentKind::Line,
                    },
                    start: s,
                    end: e,
                }),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{ingest, Language};

    fn rt(src: &str) -> ingest::FileWrite {
        let fw = ingest::file(&C, "t.c", src).unwrap();
        assert_eq!(ingest::project(&fw), src, "projection must be byte-exact");
        fw
    }

    /// The critical gotcha, measured as real in the kernel: a directive that opens a
    /// brace it never closes. Depth-counting it corrupts every boundary after it.
    /// Constructs the lexer does not model must still round-trip byte-exactly.
    ///
    /// The lexer is deliberately shallow — it tracks regions, not C++ grammar — so
    /// several real constructs mis-lex: a raw string `R"(…)"` is read as an ordinary
    /// string and ends at the first inner quote, and a C++14 digit separator (`1'000`)
    /// opens a char literal that runs to the next apostrophe. Both corrupt item
    /// BOUNDARIES.
    ///
    /// What must never happen is corrupted BYTES. Nodes store verbatim text and the node
    /// cover partitions the file, so a mis-lex costs structure and never fidelity — that
    /// is the property the whole projection rests on, and it is the one worth pinning
    /// against inputs chosen to break the lexer.
    #[test]
    fn pathological_c_still_projects_byte_exactly() {
        let cases: &[(&str, &str)] = &[
            ("raw string", "const char *s = R\"(he said \"hi\" and / * )\";\nvoid f(void) { }\n"),
            ("digit separator", "int big = 1'000'000;\nvoid g(void) { }\n"),
            ("url in a string", "const char *u = \"http://example.com/*not a comment*/\";\n"),
            ("escaped quote", "const char *q = \"say \\\" now\";\nvoid h(void) { }\n"),
            ("char literal quote", "char c = \'\\\'\';\nvoid i(void) { }\n"),
            ("unterminated string", "const char *bad = \"never closed\n"),
            ("unterminated block comment", "void j(void) { }\n/* never closed\n"),
            ("directive with brace", "#define OPEN struct {\nvoid k(void) { }\n"),
            ("backslash-continued line comment", "// keeps going \\\nstill comment\nvoid l(void) { }\n"),
            ("empty file", ""),
        ];
        for (name, src) in cases {
            let fw = crate::projection::ingest::file(&C, "t.cpp", src)
                .unwrap_or_else(|e| panic!("{name}: ingest failed: {e}"));
            assert_eq!(
                crate::projection::ingest::project(&fw),
                *src,
                "{name}: projection was not byte-exact"
            );
            // The node cover must also partition the file with no gap or overlap.
            let total: usize = fw.code.iter().map(|c| c.text.len()).sum();
            assert_eq!(total, src.len(), "{name}: nodes do not cover the file exactly");
        }
    }

    #[test]
    fn a_directive_with_an_unbalanced_brace_does_not_break_depth() {
        let src = "\
# define randomized_struct_fields_start struct {
int before;
void f(void) { g(); }
int after;
";
        let fw = rt(src);
        let kinds: Vec<&str> = fw.code.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&"directive"), "{kinds:?}");
        // `f` is still found as a definition, which only holds if the `{` in the
        // directive was never counted.
        assert!(
            fw.code.iter().any(|c| c.kind == "definition" && c.text.contains("void f")),
            "{:?}",
            fw.code.iter().map(|c| (c.kind, &c.text)).collect::<Vec<_>>()
        );
    }

    /// `#define X \` continues while lines end in a backslash.
    #[test]
    fn a_multiline_directive_is_one_opaque_region() {
        let src = "#define A(x) ({ \\\n    int y = (x); \\\n    y + 1; })\nint after;\n";
        let fw = rt(src);
        let d = fw.code.iter().find(|c| c.kind == "directive").expect("directive");
        assert!(d.text.contains("y + 1"), "continuation swallowed: {:?}", d.text);
        assert!(fw.code.iter().any(|c| c.text.contains("int after")));
    }

    /// C block comments do NOT nest — the first `*/` closes.
    #[test]
    fn block_comments_do_not_nest() {
        let src = "/* outer /* inner */ int x;\n";
        let cs = C.comments(src);
        assert_eq!(cs.len(), 1);
        assert_eq!(&src[cs[0].start..cs[0].end], "/* outer /* inner */");
        rt(src);
    }

    /// A `//` inside a string is not a comment; an apostrophe inside a comment is not a
    /// character literal.
    #[test]
    fn literals_and_apostrophes_do_not_confuse_the_lexer() {
        let src = "char *u = \"http://x\"; // don't be fooled\nint y;\n";
        let cs = C.comments(src);
        assert_eq!(cs.len(), 1, "{cs:?}");
        assert!(src[cs[0].start..cs[0].end].contains("don't"));
        rt(src);
    }

    /// Legal in C, absent from the kernel sample, and silent when wrong.
    #[test]
    fn a_backslash_continued_line_comment_spans_lines() {
        let src = "// first \\\nstill comment\nint x;\n";
        let cs = C.comments(src);
        assert_eq!(cs.len(), 1);
        assert!(src[cs[0].start..cs[0].end].contains("still comment"), "{cs:?}");
    }

    /// The payoff: comments inside a body bind to the statement they describe, which
    /// is what the fallback could never do.
    #[test]
    fn body_comments_bind_to_statements() {
        let src = "\
void f(void)
{
    // set the flag
    int flag = 1;
    g(flag);
}
";
        let fw = rt(src);
        let c = fw
            .comments
            .iter()
            .find(|c| c.text.contains("set the flag"))
            .expect("comment ingested");
        assert_eq!(c.relation, "precedes");
        assert_eq!(c.describes_kind, Some("stmt"), "binds to the statement below it");
    }

    #[test]
    fn headers_and_cpp_are_handled_by_the_same_backend() {
        for path in ["a.h", "a.cpp", "a.hpp", "a.cc", "a.dts"] {
            let lang = crate::projection::for_path(path);
            assert_eq!(lang.name(), "c", "{path}");
        }
    }
}
