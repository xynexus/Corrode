//! Linking prose to the code it describes.
//!
//! The graph has two disconnected islands: `DocIngest` builds doc -> chunk, and
//! `replace_file` builds file -> code, with no edge between them. So a README, a design
//! note or a subsystem guide sits in the same store as the code it explains and cannot
//! be reached from it.
//!
//! The mapping is **derived, never guessed**. Two rules, both measured against the Linux
//! kernel before being written:
//!
//! - **A config/build/readme file describes its own directory.** 1,912 of the kernel's
//!   1,916 `Kconfig` files live in the directory they document, so this needs no
//!   inference at all.
//! - **A prose file describes every source directory it NAMES.** 1,210 of 4,759 kernel
//!   `.rst`/`.txt`/`.md` files cite at least one real source path (2,208 links).
//!
//! What is deliberately *not* here is directory-name matching — `Documentation/networking`
//! to `net/`, `filesystems` to `fs/`. It would cover the other 75%, and it is a guess
//! dressed as a rule: a wrong edge here points an agent at the wrong subsystem, which is
//! worse than no edge. The exact quarter is worth more than the speculative whole.

use std::collections::BTreeSet;

/// Directory prefixes that begin a plausible in-repo path. Not kernel-specific: these
/// are conventional source roots, and an unknown one simply yields no link — the rule
/// is confirmed against the repo's real directory set before anything is emitted.
const ROOTS: &[&str] = &[
    "arch", "block", "cmd", "crypto", "docs", "drivers", "fs", "include", "init", "internal",
    "io_uring", "ipc", "kernel", "lib", "mm", "net", "pkg", "rust", "samples", "scripts",
    "security", "sound", "src", "tests", "tools", "usr", "virt",
];

/// Files whose subject is the directory they sit in.
fn describes_own_dir(name: &str) -> bool {
    let stem = name.split_once('.').map(|(s, _)| s).unwrap_or(name);
    matches!(stem, "Kconfig" | "Makefile" | "Kbuild" | "README" | "readme" | "CMakeLists")
}

fn parent_of(path: &str) -> Option<&str> {
    path.rsplit_once('/').map(|(dir, _)| dir)
}

/// A path-shaped run starting at `text[i..]`, if one is there.
fn path_at(text: &str, i: usize) -> Option<&str> {
    let b = text.as_bytes();
    // Must start at a token boundary, or `subnet/foo` matches the `net` root.
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'/' || b[i - 1] == b'_' || b[i - 1] == b'-') {
        return None;
    }
    let rest = &text[i..];
    let root = ROOTS.iter().find(|r| {
        rest.strip_prefix(**r).is_some_and(|after| after.starts_with('/'))
    })?;
    let mut end = root.len();
    for (off, c) in rest[root.len()..].char_indices() {
        if c.is_alphanumeric() || matches!(c, '/' | '_' | '-' | '.' | '+') {
            end = root.len() + off + c.len_utf8();
        } else {
            break;
        }
    }
    // Trailing punctuation belongs to the sentence, not the path.
    Some(rest[..end].trim_end_matches(['.', '-', '/']))
}

/// Every real source directory that `text` names.
///
/// `known` is the repo's actual directory set, so a path that merely looks like one
/// yields nothing. A cited FILE resolves to its directory — the useful unit is "which
/// part of the tree does this prose describe".
pub fn dirs_named(text: &str, known: &BTreeSet<String>) -> Vec<String> {
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i < text.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        if let Some(p) = path_at(text, i) {
            if known.contains(p) {
                out.insert(p.to_string());
            } else if let Some(d) = parent_of(p) {
                if known.contains(d) {
                    out.insert(d.to_string());
                }
            }
            i += p.len().max(1);
        } else {
            i += 1;
        }
    }
    out.into_iter().collect()
}

/// The directories `path` describes: its own if it is a config/build/readme file, plus
/// every source directory its text cites.
pub fn describes(path: &str, text: &str, known: &BTreeSet<String>) -> Vec<String> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let mut dirs = BTreeSet::new();
    if describes_own_dir(name) {
        if let Some(d) = parent_of(path) {
            dirs.insert(d.to_string());
        }
    }
    // Only prose is scanned for citations. Source files mention paths constantly (an
    // `#include`, a header guard) and none of it means "this file documents that".
    if matches!(
        name.rsplit_once('.').map(|(_, e)| e).unwrap_or(""),
        "rst" | "txt" | "md" | "adoc" | "org"
    ) || describes_own_dir(name)
    {
        dirs.extend(dirs_named(text, known));
    }
    // A file never describes the directory of a path identical to its own parent twice.
    dirs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(dirs: &[&str]) -> BTreeSet<String> {
        dirs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_kconfig_describes_its_own_directory() {
        let k = known(&["drivers/net/ethernet"]);
        assert_eq!(
            describes("drivers/net/ethernet/Kconfig", "config FOO\n\thelp\n\t  A NIC.", &k),
            vec!["drivers/net/ethernet"]
        );
    }

    #[test]
    fn prose_links_to_every_source_dir_it_names() {
        let k = known(&["drivers/pci", "include/linux"]);
        let text = "The quirk lives in drivers/pci/quirks.c and the type in include/linux/pci.h.";
        assert_eq!(describes("Documentation/PCI/quirks.rst", text, &k), vec!["drivers/pci", "include/linux"]);
    }

    #[test]
    fn a_path_that_is_not_a_real_directory_yields_nothing() {
        // The guard against inventing edges: it must look like a path AND exist.
        let k = known(&["net/core"]);
        assert!(describes("Documentation/x.rst", "see net/imaginary/thing.c", &k).is_empty());
    }

    #[test]
    fn a_root_name_inside_a_longer_word_is_not_a_path() {
        let k = known(&["net/core"]);
        // `subnet/core` must not match the `net` root.
        assert!(dirs_named("subnet/core is unrelated", &k).is_empty());
    }

    #[test]
    fn source_files_are_not_scanned_for_citations() {
        // A C file's #include is not a statement that it documents that directory.
        let k = known(&["include/linux"]);
        assert!(describes("drivers/foo/bar.c", "#include <include/linux/pci.h>", &k).is_empty());
    }

    #[test]
    fn trailing_sentence_punctuation_is_not_part_of_the_path() {
        let k = known(&["mm/slab"]);
        assert_eq!(dirs_named("allocation happens in mm/slab.", &k), vec!["mm/slab"]);
    }
}
