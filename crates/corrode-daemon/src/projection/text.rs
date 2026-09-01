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
    /// Marker set for a whole filename, for the many build and config files that
    /// carry no extension. Getting these wrong is not cosmetic: a kernel tree is full
    /// of `Makefile` and `Kconfig`, all of which comment with `#`, and defaulting them
    /// to the C family would silently find no comments in thousands of files.
    pub fn for_filename(name: &str) -> Option<PlainText> {
        let base = name.rsplit('/').next().unwrap_or(name);
        // `Kbuild`, `config` and `defconfig` were measured landing on the C family in a
        // kernel sweep — 440 files whose `#` comments were invisible because the marker
        // guess was wrong, reported as "commentless" rather than as a bug.
        matches!(
            base,
            "Makefile" | "makefile" | "GNUmakefile" | "Kbuild" | "Kconfig" | "Dockerfile"
                | "Containerfile" | "Vagrantfile" | "Rakefile" | "Gemfile" | "Justfile"
                | "justfile" | "CMakeLists.txt" | "config" | "defconfig" | "Doxyfile"
        )
        .then(|| PlainText { name: "hash", line: &["#"], block: None })
    }

    pub fn for_extension(ext: &str) -> PlainText {
        match ext {
            "py" | "rb" | "sh" | "bash" | "toml" | "yaml" | "yml" | "cfg" | "conf" | "mk"
            | "pl" | "pm" | "r" | "jl" | "tf" | "gitignore" | "dockerignore" | "service"
            | "ini" | "properties" | "env" | "config" | "defconfig" | "kconfig" => {
                PlainText { name: "hash", line: &["#"], block: None }
            }
            "sql" | "hs" | "lua" | "elm" | "vhd" | "vhdl" | "adb" | "ads" => {
                PlainText { name: "dashdash", line: &["--"], block: None }
            }
            "lisp" | "clj" | "el" | "scm" => {
                PlainText { name: "semicolon", line: &[";"], block: None }
            }
            "html" | "xml" | "svg" | "md" => {
                PlainText { name: "markup", line: &[], block: Some(("<!--", "-->")) }
            }
            // reStructuredText comments are `..` at the line start.
            "rst" => PlainText { name: "rst", line: &[".."], block: None },
            // Formats with NO comment syntax. Mapping them to the C family made every
            // `//` inside a string or URL a false comment; "no comments" is the correct
            // answer, not a guess to be improved.
            "json" | "txt" | "csv" | "tsv" | "lock" | "log" | "map" | "bin" | "dat" => {
                PlainText { name: "none", line: &[], block: None }
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

#[cfg(test)]
mod tests {
    use super::super::{for_path, ingest, Language};

    /// Extensionless build files are the common case in a kernel tree, and defaulting
    /// them to C-family markers finds no comments at all.
    #[test]
    fn extensionless_build_files_get_hash_comments() {
        for name in ["Makefile", "drivers/net/Kconfig", "Dockerfile", "CMakeLists.txt"] {
            let lang = for_path(name);
            assert_eq!(lang.name(), "hash", "{name} should use # comments");
            let src = "# a comment\nall:\n\techo hi\n";
            assert_eq!(lang.comments(src).len(), 1, "{name}");
        }
    }

    /// A dotted directory must not make the path look like an extension.
    #[test]
    fn a_dot_in_a_directory_is_not_an_extension() {
        assert_eq!(for_path("some.dir/Makefile").name(), "hash");
        assert_eq!(for_path("src/lib.rs").name(), "rust");
        assert_eq!(for_path("noextension").name(), "c-family", "unknown -> default");
    }
}
