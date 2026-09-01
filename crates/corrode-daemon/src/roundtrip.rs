//! Projection fidelity: can a file be reconstructed from the nodes that represent it?
//!
//! `graph-model.md` makes a load-bearing claim — files are a *projection* of the graph,
//! not the other way round — and `ProjectionMode::Composed` is what that claim looks
//! like when it holds. Everything above it depends on it: a graph-backed VFS, deriving
//! line numbers instead of storing them, and the argument that we avoid the line-level
//! failures a Tree-Sitter index pays for (`harness-architecture.md` §10).
//!
//! Nothing had ever tested it. This is the cheapest test that can falsify it.
//!
//! # What this tier measures, and what it does not
//!
//! Byte-exact composition needs two things to hold. This module tests the first:
//!
//! 1. **Decomposition is total** — the item spans account for every byte of the file,
//!    so reassembling them reproduces it exactly. If some bytes belong to no node, the
//!    node model cannot represent the file and no amount of clever regeneration fixes
//!    it.
//! 2. **Regeneration is exact** — an item rendered back from its structured form is
//!    byte-identical to its source. That is where `RustfmtSkip`, `MacroExpansion` and
//!    `RawStringMismatch` live, and it needs a real parser + printer (`syn` +
//!    `prettyplease`), which the workspace does not depend on. Tier 2, not here.
//!
//! Tier 1 is the *necessary* condition, so it is the one worth having first: if the
//! byte census shows unclaimed regions, tier 2 is moot.
//!
//! # Honesty rule
//!
//! The scanner is a brace/string/comment-aware splitter, not a parser. When it cannot
//! confidently find a boundary it reports [`Fidelity::Unscannable`] rather than
//! guessing. A check that says "I could not read this" is useful; one that silently
//! mis-splits is the failure mode this whole exercise exists to avoid.

/// Where a file's bytes went when split into top-level items.
#[derive(Debug, PartialEq)]
pub struct Census {
    /// Bytes inside a top-level item span.
    pub item_bytes: usize,
    /// Bytes between items — blank lines, free-standing comments, inner attributes.
    /// A node model made only of items cannot reproduce these.
    pub gap_bytes: usize,
    /// Number of top-level items found.
    pub items: usize,
    /// Reassembling spans + gaps reproduced the input exactly. This is the tier-1
    /// round trip; it fails only if the scanner loses or duplicates bytes.
    pub exact: bool,
    /// What the gap bytes actually are, so the schema knows what it must carry.
    pub gap_kinds: GapKinds,
}

#[derive(Debug, Default, PartialEq)]
pub struct GapKinds {
    pub blank: usize,
    pub comment: usize,
    pub attribute: usize,
    pub other: usize,
}

#[derive(Debug, PartialEq)]
pub enum Fidelity {
    Scanned(Census),
    /// The scanner hit something it cannot bound confidently.
    Unscannable(&'static str),
}

/// Split `src` at top-level item boundaries and account for every byte.
///
/// "Top-level item" is approximated as: a run of source at brace depth 0 ending at the
/// `}` that closes a depth-1 block, or at a `;` at depth 0. That covers `fn`, `struct`,
/// `impl`, `mod`, `use`, `const`, and macro invocations with braces or a trailing `;`.
pub fn census(src: &str) -> Fidelity {
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut depth: i32 = 0;
    let mut item_start: Option<usize> = None;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    // An item begins after the previous one ends. Deriving the start from the file
    // origin instead made consecutive items resolve to the same offset — every span
    // after the first then overlapped, which is what the first run of this check
    // actually found (a defect here, not a fact about Rust).
    let mut prev_end = 0usize;

    while i < b.len() {
        match b[i] {
            // Comments: skip whole, so braces and quotes inside cannot mislead.
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
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
                if nest > 0 {
                    return Fidelity::Unscannable("unterminated block comment");
                }
                continue;
            }
            // Raw strings: r"..", r#".."#, br#".."# — the hash count sets the terminator.
            b'r' | b'b' if raw_string_start(b, i).is_some() => {
                let Some((body, hashes)) = raw_string_start(b, i) else {
                    return Fidelity::Unscannable("raw string");
                };
                match raw_string_end(b, body, hashes) {
                    Some(e) => {
                        i = e;
                        continue;
                    }
                    None => return Fidelity::Unscannable("unterminated raw string"),
                }
            }
            b'"' => match plain_string_end(b, i) {
                Some(e) => {
                    i = e;
                    continue;
                }
                None => return Fidelity::Unscannable("unterminated string"),
            },
            // Char literal or lifetime — `'a` is not a string, `'x'` is.
            b'\'' => {
                i = char_or_lifetime_end(b, i);
                continue;
            }
            b'{' => {
                if depth == 0 && item_start.is_none() {
                    item_start = Some(item_start_after(b, prev_end));
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth < 0 {
                    return Fidelity::Unscannable("unbalanced closing brace");
                }
                if depth == 0 {
                    if let Some(s) = item_start.take() {
                        spans.push((s, i + 1));
                        prev_end = i + 1;
                    }
                }
            }
            b';' if depth == 0 => {
                // `use x;`, `const A: u8 = 1;`, `mac!();` — an item with no block.
                let s = item_start.take().unwrap_or_else(|| item_start_after(b, prev_end));
                spans.push((s, i + 1));
                prev_end = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return Fidelity::Unscannable("unbalanced braces at EOF");
    }

    // Account for every byte: spans, plus the gaps between them.
    let mut item_bytes = 0usize;
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    for (s, e) in &spans {
        if *s < cursor {
            return Fidelity::Unscannable("overlapping item spans");
        }
        if *s > cursor {
            gaps.push((cursor, *s));
        }
        item_bytes += e - s;
        cursor = *e;
    }
    if cursor < b.len() {
        gaps.push((cursor, b.len()));
    }

    // The tier-1 round trip: spans + gaps, in order, must rebuild the input.
    let mut rebuilt = String::with_capacity(src.len());
    let mut all: Vec<(usize, usize)> = spans.iter().copied().chain(gaps.iter().copied()).collect();
    all.sort();
    for (s, e) in &all {
        rebuilt.push_str(&src[*s..*e]);
    }

    let mut kinds = GapKinds::default();
    let mut gap_bytes = 0usize;
    for (s, e) in &gaps {
        let text = &src[*s..*e];
        gap_bytes += e - s;
        let t = text.trim();
        if t.is_empty() {
            kinds.blank += e - s;
        } else if t.lines().all(|l| {
            let l = l.trim();
            l.is_empty() || l.starts_with("//")
        }) {
            kinds.comment += e - s;
        } else if t.starts_with("#!") || t.starts_with("#[") {
            kinds.attribute += e - s;
        } else {
            kinds.other += e - s;
        }
    }

    Fidelity::Scanned(Census {
        item_bytes,
        gap_bytes,
        items: spans.len(),
        exact: rebuilt == src,
        gap_kinds: kinds,
    })
}

/// Where the next item begins, given where the previous one ended: the first
/// non-whitespace byte at or after `prev_end`. Monotonic by construction, so spans
/// cannot overlap. Doc comments and attributes preceding an item fall INSIDE its span,
/// which is correct — they belong to the item they document.
fn item_start_after(b: &[u8], prev_end: usize) -> usize {
    let mut j = prev_end;
    while j < b.len() && b[j].is_ascii_whitespace() {
        j += 1;
    }
    j
}

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

/// `'a` (lifetime) vs `'x'` / `'\n'` (char). Returns the index just past it.
fn char_or_lifetime_end(b: &[u8], i: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scanned(src: &str) -> Census {
        match census(src) {
            Fidelity::Scanned(c) => c,
            Fidelity::Unscannable(why) => panic!("unscannable: {why}"),
        }
    }

    #[test]
    fn every_byte_is_accounted_for() {
        let c = scanned("use a::b;\n\nfn f() {}\n");
        assert!(c.exact, "spans + gaps must rebuild the input");
        assert_eq!(c.items, 2, "`use ...;` and `fn f() {{}}`");
    }

    /// Braces and quotes inside strings, raw strings and comments must not move an
    /// item boundary — these are what a naive scanner gets wrong.
    #[test]
    fn braces_inside_literals_and_comments_do_not_split_items() {
        for src in [
            "fn f() { let s = \"}{\"; }\n",
            "fn f() { let s = r#\"}{ \"quoted\" \"#; }\n",
            "fn f() { /* } { */ }\n",
            "fn f() { let c = '}'; }\n",
            "fn f<'a>(x: &'a str) -> &'a str { x }\n",
        ] {
            let c = scanned(src);
            assert!(c.exact, "round trip failed: {src:?}");
            assert_eq!(c.items, 1, "one item expected in {src:?}");
        }
    }

    #[test]
    fn unterminated_input_is_reported_not_guessed() {
        assert!(matches!(census("fn f() {\n"), Fidelity::Unscannable(_)));
        assert!(matches!(census("let s = \"oops\n"), Fidelity::Unscannable(_)));
    }

    /// The load-bearing measurement: over this crate's real sources, how many bytes
    /// belong to no item? Those are bytes an item-only node model cannot reproduce,
    /// and they decide whether `ProjectionMode::Composed` is reachable at all.
    ///
    /// Asserts only the round trip. The census is PRINTED rather than thresholded —
    /// a number nobody has seen before should not become a pass/fail gate on its
    /// first run.
    #[test]
    fn census_over_this_crate() {
        // Defaults to this crate; `CORRODE_CENSUS_DIR` points it at a wider corpus
        // (one crate is not evidence about Rust in general).
        let dir = match std::env::var("CORRODE_CENSUS_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        };
        let mut files = 0usize;
        let mut unscannable: Vec<(String, &'static str)> = Vec::new();
        let (mut item, mut gap) = (0usize, 0usize);
        let mut kinds = GapKinds::default();

        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            match census(&src) {
                Fidelity::Scanned(c) => {
                    assert!(c.exact, "{name}: spans + gaps did not rebuild the file");
                    files += 1;
                    item += c.item_bytes;
                    gap += c.gap_bytes;
                    kinds.blank += c.gap_kinds.blank;
                    kinds.comment += c.gap_kinds.comment;
                    kinds.attribute += c.gap_kinds.attribute;
                    kinds.other += c.gap_kinds.other;
                }
                Fidelity::Unscannable(why) => unscannable.push((name, why)),
            }
        }

        let total = item + gap;
        eprintln!("--- projection census: {files} scanned, {} unscannable ---", unscannable.len());
        for (f, why) in &unscannable {
            eprintln!("  UNSCANNABLE {f}: {why}");
        }
        eprintln!(
            "  item bytes {item} ({:.1}%), gap bytes {gap} ({:.1}%)",
            100.0 * item as f64 / total as f64,
            100.0 * gap as f64 / total as f64
        );
        eprintln!(
            "  gaps: blank {} comment {} attribute {} other {}",
            kinds.blank, kinds.comment, kinds.attribute, kinds.other
        );
        assert!(files > 0, "no sources scanned");
    }
}

/// Tier 2: can an item be REGENERATED from its parsed form byte-for-byte?
///
/// Tier 1 showed decomposition is total — items plus whitespace account for every
/// byte. That makes verbatim composition possible. Tier 2 asks the harder question
/// `ProjectionMode::Composed` actually poses: if a node stores *structure* rather than
/// text, does rendering it back reproduce the source?
///
/// This is where [`crate::FallbackReason`]'s variants were always going to live.
pub mod regen {
    use corrode_core::FallbackReason;

    /// Outcome of regenerating one file through `syn` -> `prettyplease`.
    #[derive(Debug, PartialEq)]
    pub enum Regen {
        /// Byte-identical to the source.
        Exact,
        /// Parsed and printed, but diverged. Carries the classification and the first
        /// differing byte offset — the same information `UnknownDivergence` exists for.
        Diverged(FallbackReason),
        /// `syn` could not parse it at all.
        Unparseable(String),
    }

    /// Parse `src` as a Rust file, print it back, and classify any divergence.
    pub fn regenerate(src: &str) -> Regen {
        let file = match syn::parse_file(src) {
            Ok(f) => f,
            Err(e) => return Regen::Unparseable(e.to_string()),
        };
        let printed = prettyplease::unparse(&file);
        if printed == src {
            return Regen::Exact;
        }
        Regen::Diverged(classify(src, &printed))
    }

    /// Is the divergence only in whitespace?
    ///
    /// Deliberately coarse, and the limit is worth stating: this cannot separate
    /// "content was lost" from "the printer canonicalised". prettyplease adds a
    /// trailing comma when it breaks a parameter list — semantically null, but a real
    /// token, so no text-level comparison classifies it as formatting. A sound split
    /// needs token-level comparison, and even then the trailing comma is a token.
    ///
    /// Two attempts at a finer classification were wrong before this one: splitting on
    /// whitespace made cosmetic reflow look like content loss, and stripping whitespace
    /// still cannot see a canonicalised comma. The census reports what it can defend.
    pub fn formatting_only(src: &str, printed: &str) -> bool {
        // Remove whitespace ENTIRELY rather than normalising runs of it. Splitting on
        // whitespace and rejoining looks equivalent and is not: `lookup(&self,` is one
        // token and `lookup( &self,` is two, so a purely cosmetic reflow inside a
        // parameter list reads as changed content. That mistake turned 76 of 78 files
        // into false "content lost" results on the first run of this census.
        let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        strip(src) == strip(printed)
    }

    /// Regenerate and report both the classification and whether the loss was purely
    /// cosmetic — the pair a schema decision actually needs.
    pub fn diagnose(src: &str) -> (Regen, bool) {
        let file = match syn::parse_file(src) {
            Ok(f) => f,
            Err(e) => return (Regen::Unparseable(e.to_string()), false),
        };
        let printed = prettyplease::unparse(&file);
        if printed == src {
            return (Regen::Exact, true);
        }
        let fmt_only = formatting_only(src, &printed);
        (Regen::Diverged(classify(src, &printed)), fmt_only)
    }

    /// Strip plain (non-doc) comments, respecting strings and raw strings, so the
    /// census can ATTRIBUTE divergence rather than assume its cause. If a file
    /// regenerates exactly once its comments are gone, comment loss explains it.
    pub fn strip_plain_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out = String::with_capacity(src.len());
        let mut i = 0usize;
        while i < b.len() {
            // Doc comments are attributes in the AST and DO survive; keep them.
            if b[i] == b'/' && i + 2 < b.len() && b[i + 1] == b'/' {
                let doc = b[i + 2] == b'/' || b[i + 2] == b'!';
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                if doc {
                    out.push_str(&src[start..i]);
                }
                continue;
            }
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
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
                continue;
            }
            if let Some((body, hashes)) = super::raw_string_start(b, i) {
                if let Some(e) = super::raw_string_end(b, body, hashes) {
                    out.push_str(&src[i..e]);
                    i = e;
                    continue;
                }
            }
            if b[i] == b'"' {
                if let Some(e) = super::plain_string_end(b, i) {
                    out.push_str(&src[i..e]);
                    i = e;
                    continue;
                }
            }
            let ch_end = i + src[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            out.push_str(&src[i..ch_end]);
            i = ch_end;
        }
        out
    }

    /// Attribute a divergence to a cause, in the order the causes actually dominate.
    ///
    /// The order matters: a file can have several of these at once, and reporting the
    /// most explanatory one is the point. A `#[rustfmt::skip]` region is *intentionally*
    /// non-canonical, so it outranks a generic formatting difference.
    fn classify(src: &str, printed: &str) -> FallbackReason {
        if src.contains("rustfmt::skip") {
            return FallbackReason::RustfmtSkip;
        }
        // A non-doc comment cannot survive: `syn`'s AST has no node for it. Doc
        // comments become attributes and do survive, so only plain ones count.
        if has_plain_comment(src) && !has_plain_comment(printed) {
            return FallbackReason::MacroExpansion.pick_comment_loss();
        }
        if src.contains("r#\"") || src.contains("br#\"") {
            return FallbackReason::RawStringMismatch;
        }
        if src.contains("macro_rules!") || src.contains("!(") || src.contains("![") {
            return FallbackReason::MacroExpansion;
        }
        let first_diff = src
            .bytes()
            .zip(printed.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(src.len().min(printed.len())) as u64;
        FallbackReason::UnknownDivergence {
            first_diff_offset: first_diff,
        }
    }

    /// A `//` or `/* */` comment that is not a doc comment.
    fn has_plain_comment(s: &str) -> bool {
        s.lines().any(|l| {
            let t = l.trim_start();
            (t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!"))
                || t.starts_with("/*")
        })
    }

    /// `FallbackReason` has no comment-loss variant. Rather than invent one here, this
    /// names the gap explicitly: comment loss is the DOMINANT divergence for `syn`
    /// regeneration and the enum does not model it.
    trait CommentLoss {
        fn pick_comment_loss(self) -> FallbackReason;
    }
    impl CommentLoss for FallbackReason {
        fn pick_comment_loss(self) -> FallbackReason {
            // ponytail: reported as UnknownDivergence(0) until `FallbackReason` gains a
            // `CommentsDropped` variant — inventing one in a check module would put the
            // wire type's shape in the wrong crate.
            FallbackReason::UnknownDivergence {
                first_diff_offset: 0,
            }
        }
    }
}

#[cfg(test)]
mod regen_tests {
    use super::regen::*;

    /// Tier 2 over real Rust. Like the tier-1 census this PRINTS rather than
    /// thresholds: the point is to learn what regeneration actually costs before
    /// anything depends on `Composed`.
    #[test]
    fn regeneration_census() {
        let dir = match std::env::var("CORRODE_CENSUS_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        };
        let (mut exact, mut diverged, mut unparseable) = (0usize, 0usize, 0usize);
        let (mut fmt_recoverable, mut beyond_ws) = (0usize, 0usize);
        let mut comments_explain = 0usize;
        let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();

        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            match regenerate(&src) {
                Regen::Exact => exact += 1,
                Regen::Diverged(r) => {
                    diverged += 1;
                    let (_, fmt_only) = diagnose(&src);
                    if fmt_only {
                        fmt_recoverable += 1;
                    } else {
                        beyond_ws += 1;
                        // Attribution: does removing plain comments make it exact?
                        let stripped = strip_plain_comments(&src);
                        if let Ok(f) = syn::parse_file(&stripped) {
                            if formatting_only(&stripped, &prettyplease::unparse(&f)) {
                                comments_explain += 1;
                            }
                        }
                    }
                    let key = match r {
                        corrode_core::FallbackReason::UnknownDivergence { .. } => {
                            "UnknownDivergence".to_string()
                        }
                        other => format!("{other:?}"),
                    };
                    *reasons.entry(key).or_default() += 1;
                }
                Regen::Unparseable(_) => unparseable += 1,
            }
        }
        let total = exact + diverged + unparseable;
        eprintln!("--- regeneration census: {total} files ---");
        eprintln!("  exact {exact}, diverged {diverged}, unparseable {unparseable}");
        eprintln!("  of the diverged: {fmt_recoverable} whitespace-only, {beyond_ws} differ beyond whitespace");
        eprintln!("  of those: {comments_explain} become whitespace-only once plain comments are removed");
        eprintln!("  (remainder is canonicalisation — prettyplease adds trailing commas when it");
        eprintln!("   breaks a parameter list, which no text-level comparison can call formatting)");
        for (r, n) in &reasons {
            eprintln!("    {r}: {n}");
        }
        assert!(total > 0, "no sources scanned");
    }

    /// The specific thing that decides the schema: a plain comment has no node in
    /// syn's AST, so regeneration cannot reproduce it. Doc comments become attributes
    /// and do survive. This is why a node storing STRUCTURE cannot be byte-exact,
    /// while a node storing its verbatim span can.
    #[test]
    fn plain_comments_do_not_survive_regeneration_but_doc_comments_do() {
        let with_plain = "fn f() {\n    // a plain comment\n}\n";
        assert_ne!(regenerate(with_plain), Regen::Exact, "plain comment survived?");

        let with_doc = "/// documented\nfn f() {}\n";
        let printed = match syn::parse_file(with_doc) {
            Ok(f) => prettyplease::unparse(&f),
            Err(e) => panic!("parse: {e}"),
        };
        assert!(printed.contains("/// documented"), "doc comment lost: {printed}");
    }

    #[test]
    fn unparseable_input_is_reported_not_guessed() {
        assert!(matches!(regenerate("fn ("), Regen::Unparseable(_)));
    }
}


#[cfg(test)]
mod compose_tests {
    use crate::projection::compose::*;

    /// Scan a whole repository into nodes, regenerate every file, diff the tree.
    ///
    /// This is the end-to-end claim `graph-model.md` makes — files are a projection of
    /// the graph — reduced to something that either holds byte-for-byte or does not.
    /// `CORRODE_SCAN_REPO` points it at any repo; the default is this one.
    #[test]
    fn scan_and_regenerate_a_repository() {
        let repo = std::env::var("CORRODE_SCAN_REPO").unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_string_lossy()
                .into_owned()
        });
        let listed = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["ls-files", "-s", "-z"])
            .output()
            .expect("git ls-files");
        let files: Vec<String> = String::from_utf8_lossy(&listed.stdout)
            .split('\0')
            .filter_map(|rec| {
                let (meta, path) = rec.split_once('\t')?;
                let mode = meta.split_whitespace().next()?;
                (matches!(mode, "100644" | "100755") && path.ends_with(".rs"))
                    .then(|| path.to_string())
            })
            .collect();

        let (mut ok, mut mismatched, mut unparseable) = (0usize, 0usize, 0usize);
        let (mut nodes_total, mut bytes_total) = (0usize, 0usize);
        let mut kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut failures: Vec<String> = Vec::new();

        for rel in &files {
            let full = std::path::Path::new(&repo).join(rel);
            let Ok(src) = std::fs::read_to_string(&full) else {
                continue;
            };
            match scan(rel, &src) {
                Err(_) => unparseable += 1,
                Ok(nodes) => {
                    let rebuilt = regenerate(&nodes);
                    if rebuilt == src {
                        ok += 1;
                        nodes_total += nodes.len();
                        bytes_total += src.len();
                        for n in &nodes {
                            *kinds.entry(n.kind).or_default() += 1;
                        }
                    } else {
                        mismatched += 1;
                        if failures.len() < 5 {
                            let at = src
                                .bytes()
                                .zip(rebuilt.bytes())
                                .position(|(a, b)| a != b)
                                .unwrap_or(src.len().min(rebuilt.len()));
                            failures.push(format!(
                                "{rel}: first diff at byte {at} (src {} bytes, rebuilt {})",
                                src.len(),
                                rebuilt.len()
                            ));
                        }
                    }
                }
            }
        }

        eprintln!("--- scan + regenerate: {} .rs files in {repo} ---", files.len());
        eprintln!("  byte-exact {ok}, mismatched {mismatched}, unparseable {unparseable}");
        eprintln!("  {nodes_total} nodes over {bytes_total} bytes");
        let trivia = kinds.get("trivia").copied().unwrap_or(0);
        eprintln!(
            "  node kinds: {}",
            kinds
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        eprintln!(
            "  {:.1}% of nodes are real items (rest trivia)",
            100.0 * (nodes_total - trivia) as f64 / nodes_total.max(1) as f64
        );
        for f in &failures {
            eprintln!("  MISMATCH {f}");
        }

        assert!(!files.is_empty(), "no .rs files found in {repo}");
        assert_eq!(mismatched, 0, "regeneration was not byte-exact");
    }

    /// Positions are an OUTPUT of projection, not node state.
    ///
    /// The distinction is not academic: a dynamically generated VFS projects files
    /// that were never on disk, from nodes that get reordered, inserted and edited. A
    /// byte offset or line number stored on a node is a fact about one source text,
    /// so it is stale or meaningless the moment the graph moves. This asserts that
    /// `project` reports the truth after a mutation that would invalidate any stored
    /// position.
    #[test]
    fn positions_are_recomputed_not_stored() {
        let src = "use a::b;\n\nfn one() {}\n\nfn two() {}\n";
        let nodes = scan("t.rs", src).unwrap();
        let (text, places) = project(&nodes);
        assert_eq!(text, src, "projection reproduces the source");

        let line_of = |ord: usize| places.iter().find(|p| p.ordinal == ord).unwrap().start_line;
        let two = nodes.iter().find(|n| n.text.contains("fn two")).unwrap();
        assert_eq!(line_of(two.ordinal), 5, "fn two starts at line 5");
        let stored_before = two.scan_line;

        // Now do what a live graph does: insert a node ahead of it.
        let mut mutated: Vec<Node> = nodes
            .iter()
            .cloned()
            .map(|mut n| {
                n.ordinal += 2;
                n
            })
            .collect();
        mutated.push(Node {
            path: "t.rs".into(),
            ordinal: 0,
            kind: "use",
            text: "use inserted::thing;".into(),
            scan_line: 0,
        });
        mutated.push(Node {
            path: "t.rs".into(),
            ordinal: 1,
            kind: "trivia",
            text: "\n\n".into(),
            scan_line: 0,
        });

        let (text2, places2) = project(&mutated);
        assert!(text2.starts_with("use inserted::thing;"), "insert took effect");
        let two_after = places2
            .iter()
            .find(|p| p.ordinal == two.ordinal + 2)
            .expect("fn two still projected");
        assert_eq!(
            two_after.start_line,
            stored_before + 2,
            "two inserted lines shifted it; projection reports the new line"
        );
        // The node's own `scan_line` is now wrong — which is exactly why a projector
        // must never read it.
        assert_ne!(
            stored_before, two_after.start_line,
            "a stored position would have been stale here"
        );
    }

    /// Comments bind to (node, line-within-node), so they survive the same mutation.
    /// An absolute line would not.
    #[test]
    fn comment_attachment_survives_reordering() {
        use crate::projection::comments::*;
        let src = "fn f() {\n    // explains the loop\n    let x = 1;\n}\n";
        let nodes = scan("t.rs", src).unwrap();
        let refs = extract("t.rs", src, &nodes);
        let c = refs.iter().find(|c| c.text.contains("explains")).expect("found");
        assert_eq!(c.line_in_node, 2, "second line of its node");

        // Shift the whole file down; the relative position is unchanged, the absolute
        // one is not.
        let shifted = format!("// header\n\n{src}");
        let nodes2 = scan("t.rs", &shifted).unwrap();
        let refs2 = extract("t.rs", &shifted, &nodes2);
        let c2 = refs2.iter().find(|c| c.text.contains("explains")).expect("found");
        assert_eq!(c2.line_in_node, 2, "line WITHIN the node is stable");
        assert_ne!(c.abs_line, c2.abs_line, "the absolute line moved");
    }
}

/// The canonical-form alternative: instead of preserving arbitrary source, rewrite the
/// repo ONCE into the projector's own form, after which regeneration is exact because
/// the source is already what the printer emits.
///
/// Measured and REJECTED ON COST, not feasibility: the printer reaches a fixed point in
/// two passes, but canonicalising destroys 35,477 body comments that no migration can
/// recover. Kept so the measurement can be re-run rather than re-argued.
pub mod canonical {
    /// Does printing a canonical file reproduce it? If the printer is not idempotent
    /// there is no fixed point to converge on and the scheme has no foundation.
    pub fn is_idempotent(src: &str) -> Option<bool> {
        let once = prettyplease::unparse(&syn::parse_file(src).ok()?);
        let twice = prettyplease::unparse(&syn::parse_file(&once).ok()?);
        Some(once == twice)
    }

    /// What canonicalisation destroys. `syn`'s AST has no node for a plain comment, so
    /// rewriting a repo into printer output deletes every one of them permanently.
    /// Doc comments become attributes and survive.
    #[derive(Debug, Default, PartialEq)]
    pub struct Loss {
        pub plain_comment_lines: usize,
        pub plain_comment_bytes: usize,
        pub doc_comment_lines: usize,
        pub total_bytes: usize,
    }

    /// Splits the destroyed comments into the ones that could be SAVED by rewriting
    /// them as doc comments (they sit between items, so they attach to the next one)
    /// and the ones that cannot (inside a function body, where `///` is not legal).
    /// Uses the composer's own node decomposition, so the split is structural rather
    /// than an indentation guess.
    pub fn migratable(path: &str, src: &str) -> (usize, usize) {
        let Ok(nodes) = crate::projection::compose::scan(path, src) else {
            return (0, 0);
        };
        let (mut between, mut inside) = (0usize, 0usize);
        for n in &nodes {
            for line in n.text.lines() {
                let t = line.trim_start();
                let plain = (t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!"))
                    || t.starts_with("/*");
                if !plain {
                    continue;
                }
                if n.kind == "trivia" {
                    between += 1;
                } else {
                    inside += 1;
                }
            }
        }
        (between, inside)
    }

    pub fn loss(src: &str) -> Loss {
        let mut l = Loss {
            total_bytes: src.len(),
            ..Default::default()
        };
        let stripped = crate::roundtrip::regen::strip_plain_comments(src);
        l.plain_comment_bytes = src.len().saturating_sub(stripped.len());
        for line in src.lines() {
            let t = line.trim_start();
            if t.starts_with("///") || t.starts_with("//!") {
                l.doc_comment_lines += 1;
            } else if t.starts_with("//") || t.starts_with("/*") || t.starts_with("*") {
                l.plain_comment_lines += 1;
            }
        }
        l
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::canonical::*;

    /// Both halves of the canonical-form question, over a real repo.
    #[test]
    fn canonical_form_viability() {
        let repo = std::env::var("CORRODE_SCAN_REPO").unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_string_lossy()
                .into_owned()
        });
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["ls-files", "-s", "-z"])
            .output()
            .expect("git");
        let files: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .split('\0')
            .filter_map(|r| {
                let (m, p) = r.split_once('\t')?;
                (matches!(m.split_whitespace().next()?, "100644" | "100755") && p.ends_with(".rs"))
                    .then(|| p.to_string())
            })
            .collect();

        let (mut idem, mut not_idem, mut skipped) = (0usize, 0usize, 0usize);
        let mut agg = Loss::default();
        let (mut between_items, mut inside_bodies) = (0usize, 0usize);
        for rel in &files {
            let Ok(src) = std::fs::read_to_string(std::path::Path::new(&repo).join(rel)) else {
                continue;
            };
            match is_idempotent(&src) {
                Some(true) => idem += 1,
                Some(false) => {
                    not_idem += 1;
                    eprintln!("  NOT IDEMPOTENT: {rel}");
                }
                None => {
                    skipped += 1;
                    continue;
                }
            }
            let l = loss(&src);
            agg.plain_comment_lines += l.plain_comment_lines;
            agg.plain_comment_bytes += l.plain_comment_bytes;
            agg.doc_comment_lines += l.doc_comment_lines;
            agg.total_bytes += l.total_bytes;
            let (b, i) = migratable(rel, &src);
            between_items += b;
            inside_bodies += i;
        }

        eprintln!("--- canonical-form viability over {} .rs files ---", files.len());
        eprintln!("  printer idempotent: {idem} yes, {not_idem} no, {skipped} unparseable");
        eprintln!(
            "  destroyed by canonicalisation: {} plain-comment lines, {} bytes ({:.1}% of source)",
            agg.plain_comment_lines,
            agg.plain_comment_bytes,
            100.0 * agg.plain_comment_bytes as f64 / agg.total_bytes.max(1) as f64
        );
        eprintln!("  preserved (become attributes): {} doc-comment lines", agg.doc_comment_lines);
        eprintln!(
            "  of the destroyed: {between_items} between items (could migrate to ///), \
             {inside_bodies} inside bodies (unrecoverable — /// is not legal there)"
        );
        assert!(!files.is_empty());
    }
}

#[cfg(test)]
mod attach_tests {
    use crate::projection::compose::*;


}



#[cfg(test)]
mod describes_tests {
    use crate::projection::comments;
    use crate::projection::compose;
    use crate::projection::describes::*;

    const SRC: &str = "\
/// documented
fn outer() {
    // sets the seed
    let seed = 7;
    let x = compute(seed); // trailing note
    match x {
        // the happy path
        Ok(v) => use_it(v),
        Err(_) => {}
    }
}
";

    fn edges_for(src: &str) -> Vec<Edge> {
        let nodes = compose::scan("t.rs", src).unwrap();
        let cs = comments::extract("t.rs", src, &nodes);
        let anchors = anchors(src).unwrap();
        bind(src, &cs, &anchors)
    }

    /// Query 1: what is this comment describing?
    #[test]
    fn a_comment_resolves_to_the_element_it_describes() {
        let edges = edges_for(SRC);
        let find = |needle: &str| {
            edges
                .iter()
                .find(|e| e.comment.text.contains(needle))
                .unwrap_or_else(|| panic!("no comment matching {needle}"))
        };

        let seed = find("sets the seed");
        assert_eq!(seed.relation, Relation::Precedes);
        let t = seed.target.as_ref().expect("has a target");
        assert_eq!(t.kind, "stmt");
        assert!(SRC[t.start..t.end].starts_with("let seed"), "{:?}", &SRC[t.start..t.end]);

        // A trailing comment describes the code to its LEFT, not the next line.
        let trailing = find("trailing note");
        assert_eq!(trailing.relation, Relation::Trailing);
        let t = trailing.target.as_ref().expect("has a target");
        assert!(SRC[t.start..t.end].contains("compute(seed)"), "{:?}", &SRC[t.start..t.end]);

        // Sub-item granularity: a comment inside a match binds to the arm.
        let arm = find("happy path");
        let t = arm.target.as_ref().expect("has a target");
        assert_eq!(t.kind, "match_arm");
        assert!(SRC[t.start..t.end].starts_with("Ok(v)"), "{:?}", &SRC[t.start..t.end]);
    }

    /// Query 2: what commentary applies to this region of code?
    #[test]
    fn a_region_resolves_to_the_commentary_about_it() {
        let edges = edges_for(SRC);
        let match_start = SRC.find("match x").unwrap();
        let match_end = SRC[match_start..].find("\n    }").unwrap() + match_start;

        let found = comments_for(&edges, match_start, match_end);
        let texts: Vec<&str> = found.iter().map(|e| e.comment.text.trim()).collect();
        assert!(
            texts.iter().any(|t| t.contains("happy path")),
            "the arm's comment belongs to the match region: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.contains("sets the seed")),
            "a comment about an earlier statement does not: {texts:?}"
        );
    }

    /// Over a real repo: how much commentary actually binds to something?
    #[test]
    fn binding_census() {
        let repo = std::env::var("CORRODE_SCAN_REPO").unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_string_lossy()
                .into_owned()
        });
        let out = std::process::Command::new("git")
            .arg("-C").arg(&repo).args(["ls-files", "-s", "-z"])
            .output().expect("git");
        let files: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .split('\0')
            .filter_map(|r| {
                let (m, p) = r.split_once('\t')?;
                (matches!(m.split_whitespace().next()?, "100644" | "100755") && p.ends_with(".rs"))
                    .then(|| p.to_string())
            })
            .collect();

        let (mut bound, mut unbound) = (0usize, 0usize);
        let mut rel: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for f in &files {
            let Ok(src) = std::fs::read_to_string(std::path::Path::new(&repo).join(f)) else { continue };
            let Ok(nodes) = compose::scan(f, &src) else { continue };
            let Ok(a) = anchors(&src) else { continue };
            for e in bind(&src, &comments::extract(f, &src, &nodes), &a) {
                if !is_plain(&e.comment) { continue; }
                match &e.target {
                    Some(t) => {
                        bound += 1;
                        *kinds.entry(t.kind).or_default() += 1;
                        *rel.entry(match e.relation {
                            Relation::Precedes => "precedes",
                            Relation::Trailing => "trailing",
                            Relation::Encloses => "encloses",
                        }).or_default() += 1;
                    }
                    None => unbound += 1,
                }
            }
        }
        eprintln!("--- comment binding over {} files ---", files.len());
        eprintln!("  plain comments bound to an element: {bound}, unbound: {unbound}");
        eprintln!("  relation: {}", rel.iter().map(|(k,v)| format!("{k}={v}")).collect::<Vec<_>>().join(" "));
        eprintln!("  target kinds: {}", kinds.iter().map(|(k,v)| format!("{k}={v}")).collect::<Vec<_>>().join(" "));
        assert!(!files.is_empty());
    }
}
