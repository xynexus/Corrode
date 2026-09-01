//! Measurements of the projection, kept separate from the projection itself.
//!
//! These verify the load-bearing claims in `harness-architecture.md` §8 and print the
//! censuses behind them. They live here rather than in `projection/` because their
//! corpora are whole repositories and their job is to *measure* the projection rather
//! than perform it — `CORRODE_SCAN_REPO` points them at any repo.

use crate::projection::{self, ingest, regenerate, scan, Language};

/// The end-to-end claim: a repository decomposes into nodes and reassembles byte-for-
/// byte. Runs whatever backend each path selects, so it covers the fallback too.
/// Per-backend outcome for one repository.
#[cfg(test)]
#[derive(Default)]
struct Census {
    files: usize,
    exact: usize,
    mismatched: usize,
    nodes: usize,
    comments: usize,
    bound: usize,
}

/// Ingest every tracked file and report what happened, by language.
///
/// The breakdown matters more than the total on a mixed repo: a fallback that quietly
/// swallows half a tree looks identical to full coverage if you only count round trips.
#[cfg(test)]
fn scan_repo(repo: &str) -> (std::collections::BTreeMap<&'static str, Census>, usize, usize) {
    use std::collections::BTreeMap;
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["ls-files", "-s", "-z"])
        .output()
        .expect("git ls-files");
    let files: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .split('\0')
        .filter_map(|r| {
            let (m, p) = r.split_once('\t')?;
            matches!(m.split_whitespace().next()?, "100644" | "100755").then(|| p.to_string())
        })
        .collect();

    let mut by_lang: BTreeMap<&'static str, Census> = BTreeMap::new();
    // Files skipped for reasons that are NOT the projection's fault, counted rather
    // than silently dropped: a kernel tree has latin-1 sources and binaries, and
    // "skipped 9000" reads very differently from "byte-exact 100%".
    let (mut unreadable, mut ingest_failed) = (0usize, 0usize);

    for rel in &files {
        let path = std::path::Path::new(repo).join(rel);
        let Ok(src) = std::fs::read_to_string(&path) else {
            unreadable += 1; // binary, or not UTF-8
            continue;
        };
        let lang = projection::for_path(rel);
        let name = lang.name();
        let e = by_lang.entry(name).or_default();
        e.files += 1;
        match ingest::file(lang.as_ref(), rel, &src) {
            Err(_) => {
                ingest_failed += 1;
                e.files -= 1;
            }
            Ok(fw) => {
                if ingest::project(&fw) == src {
                    e.exact += 1;
                } else {
                    e.mismatched += 1;
                    if e.mismatched <= 3 {
                        eprintln!("  MISMATCH [{name}] {rel}");
                    }
                }
                e.nodes += fw.code.len();
                e.comments += fw.comments.len();
                e.bound += fw.comments.iter().filter(|c| c.describes_kind.is_some()).count();
            }
        }
    }
    (by_lang, unreadable, ingest_failed)
}

/// Which extensions are landing on the fallback, biggest first — the worklist for
/// deciding which backend to write next.
#[cfg(test)]
fn fallback_extensions(repo: &str) -> Vec<(String, usize)> {
    use std::collections::BTreeMap;
    let out = std::process::Command::new("git")
        .arg("-C").arg(repo).args(["ls-files", "-z"]).output().expect("git");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for rel in String::from_utf8_lossy(&out.stdout).split('\0').filter(|s| !s.is_empty()) {
        let lang = projection::for_path(rel);
        if lang.name() == "rust" {
            continue;
        }
        let ext = rel.rsplit('/').next().unwrap_or(rel);
        let ext = ext.rsplit_once('.').map(|(_, e)| e.to_string()).unwrap_or_else(|| "(none)".into());
        *counts.entry(ext).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tracked file in a repo, whatever its language, projects back exactly.
    ///
    /// `CORRODE_SCAN_REPO=/path/to/repo` points it at anything.
    #[test]
    fn a_repository_round_trips() {
        let repo = std::env::var("CORRODE_SCAN_REPO").unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_string_lossy()
                .into_owned()
        });
        let (by_lang, unreadable, failed) = scan_repo(&repo);

        eprintln!("--- ingest census: {repo} ---");
        eprintln!(
            "  {:<12} {:>7} {:>8} {:>7} {:>8} {:>9} {:>7}",
            "backend", "files", "exact", "MISM", "nodes", "comments", "bound"
        );
        let (mut tf, mut te, mut tm) = (0usize, 0usize, 0usize);
        for (name, c) in &by_lang {
            eprintln!(
                "  {:<12} {:>7} {:>8} {:>7} {:>8} {:>9} {:>7}",
                name, c.files, c.exact, c.mismatched, c.nodes, c.comments, c.bound
            );
            tf += c.files;
            te += c.exact;
            tm += c.mismatched;
        }
        eprintln!("  {:<12} {:>7} {:>8} {:>7}", "TOTAL", tf, te, tm);
        eprintln!("  skipped: {unreadable} unreadable/non-UTF-8, {failed} ingest errors");

        let fb = fallback_extensions(&repo);
        if !fb.is_empty() {
            eprintln!("  top extensions without a backend:");
            for (ext, n) in fb.iter().take(8) {
                eprintln!("    .{ext:<10} {n}");
            }
        }

        assert!(tf > 0, "nothing scanned");
        assert_eq!(tm, 0, "projection was not byte-exact");
    }

    /// Positions are an OUTPUT of projection, never node state: a byte offset is a fact
    /// about one source text, and a generated VFS has none.
    #[test]
    fn positions_are_recomputed_not_stored() {
        let lang = projection::rust::Rust;
        let src = "use a::b;\n\nfn one() {}\n\nfn two() {}\n";
        let nodes = scan(&lang, "t.rs", src).unwrap();
        let (text, places) = projection::project(&nodes);
        assert_eq!(text, src);

        let two = nodes.iter().find(|n| n.text.contains("fn two")).unwrap();
        let before = places.iter().find(|p| p.ordinal == two.ordinal).unwrap().start_line;

        let mut mutated: Vec<projection::Node> = nodes
            .iter()
            .cloned()
            .map(|mut n| {
                n.ordinal += 1;
                n
            })
            .collect();
        mutated.push(projection::Node {
            path: "t.rs".into(),
            ordinal: 0,
            kind: "use",
            text: "use inserted::thing;\n\n".into(),
        });
        let (_, after) = projection::project(&mutated);
        let now = after.iter().find(|p| p.ordinal == two.ordinal + 1).unwrap().start_line;
        assert_eq!(now, before + 2, "projection reports the shifted line");
    }

    /// Braces and quotes inside literals and comments must not move a boundary.
    #[test]
    fn literals_do_not_confuse_the_rust_backend() {
        let lang = projection::rust::Rust;
        for src in [
            "fn f() { let s = \"}{\"; }\n",
            "fn f() { let s = r#\"}{ \"q\" \"#; }\n",
            "fn f() { /* } { */ }\n",
            "fn f() { let c = '}'; }\n",
            "fn f<'a>(x: &'a str) -> &'a str { x }\n",
        ] {
            let nodes = scan(&lang, "t.rs", src).unwrap();
            assert_eq!(regenerate(&nodes), src, "round trip: {src:?}");
            assert_eq!(
                nodes.iter().filter(|n| n.kind == "fn").count(),
                1,
                "one item: {src:?}"
            );
        }
    }

    /// A language with no backend is still absorbed: byte-exact, less structure.
    #[test]
    fn an_unknown_language_degrades_without_losing_fidelity() {
        let src = "# a python comment\ndef f():\n    return 1  # trailing\n";
        let lang = projection::for_path("s.py");
        assert_eq!(lang.name(), "hash", "hash-comment family selected");
        let fw = ingest::file(lang.as_ref(), "s.py", src).unwrap();

        assert_eq!(ingest::project(&fw), src, "fidelity is not language-dependent");
        assert_eq!(fw.code.len(), 1, "no grammar -> one node for the file");
        assert_eq!(fw.comments.len(), 2, "comments still recovered");
        assert!(
            fw.comments.iter().all(|c| c.describes_kind.is_none()),
            "no anchors without a grammar — reported, not guessed"
        );
    }

    /// C-family markers cover most of what a mixed repo contains.
    #[test]
    fn the_fallback_covers_the_common_comment_families() {
        for (path, text, want) in [
            ("a.go", "// go\nfunc f() {}\n", 1),
            ("a.ts", "/* block */\nlet x = 1;\n", 1),
            ("a.sql", "-- sql\nSELECT 1;\n", 1),
            ("a.html", "<!-- markup -->\n<p/>\n", 1),
            ("a.yaml", "# yaml\nk: v\n", 1),
        ] {
            let lang = projection::for_path(path);
            let fw = ingest::file(lang.as_ref(), path, text).unwrap();
            assert_eq!(ingest::project(&fw), text, "{path} round trips");
            assert_eq!(fw.comments.len(), want, "{path}: {:?}", fw.comments);
        }
    }
}
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
            if let Some((body, hashes)) = crate::projection::rust::raw_string_start(b, i) {
                if let Some(e) = crate::projection::rust::raw_string_end(b, body, hashes) {
                    out.push_str(&src[i..e]);
                    i = e;
                    continue;
                }
            }
            if b[i] == b'"' {
                if let Some(e) = crate::projection::rust::plain_string_end(b, i) {
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
        let Ok(nodes) = crate::projection::scan(&crate::projection::rust::Rust, path, src)
        else {
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
