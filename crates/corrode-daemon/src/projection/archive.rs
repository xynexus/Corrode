//! Ingesting straight from a tar stream, without unpacking it.
//!
//! The projection is already source-agnostic — [`ingest::file`](super::ingest::file)
//! takes a path string and a content string and touches no filesystem — so absorbing an
//! archive needs a reader, not a change to the pipeline.
//!
//! Worth doing rather than extracting first: a kernel tarball is ~160 MB compressed and
//! well over a gigabyte unpacked, and extraction costs that in disk, in inodes, and in
//! a second full pass over every byte. Streaming reads each entry once.
//!
//! Decompression is delegated to a subprocess (`xz`, `gzip`, `zstd`, `bzip2`) rather
//! than a codec crate, so `.tar.xz` works without linking liblzma and a new compression
//! format is a table entry rather than a dependency.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

/// The decompressor for an archive's extension, if it needs one.
fn decompressor(path: &Path) -> Option<(&'static str, &'static [&'static str])> {
    let name = path.file_name()?.to_str()?;
    if name.ends_with(".tar.xz") || name.ends_with(".txz") {
        Some(("xz", &["-dc"]))
    } else if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Some(("gzip", &["-dc"]))
    } else if name.ends_with(".tar.zst") {
        Some(("zstd", &["-dc"]))
    } else if name.ends_with(".tar.bz2") {
        Some(("bzip2", &["-dc"]))
    } else {
        None // plain .tar
    }
}

/// One entry's worth of source, as the projection wants it.
pub struct Entry<'a> {
    /// Path inside the archive, with the leading component stripped — a release
    /// tarball wraps everything in `project-1.2.3/`, and keeping that in every node id
    /// would bake the version into the graph.
    pub path: &'a str,
    pub text: &'a str,
}

/// Stream `archive`, calling `f` for every regular file that is valid UTF-8.
///
/// Returns `(entries_seen, skipped_non_utf8)`. Binary and non-UTF-8 entries are counted
/// rather than dropped silently: on a kernel tree that number is thousands, and a
/// total that hides it flatters the result.
pub fn for_each_file(
    archive: &Path,
    mut f: impl FnMut(Entry<'_>),
) -> anyhow::Result<(usize, usize)> {
    // Fail loudly on a missing archive rather than handing the decompressor a path it
    // cannot open and reading its empty output as an empty repository.
    if !archive.is_file() {
        anyhow::bail!("archive not found: {}", archive.display());
    }
    let mut child = None;
    let reader: Box<dyn Read> = match decompressor(archive) {
        Some((cmd, args)) => {
            let mut c = Command::new(cmd)
                .args(args)
                .arg(archive)
                .stdout(Stdio::piped())
                // Inherited, not silenced: a decompressor's complaint is the only clue
                // when a stream yields nothing.
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| anyhow::anyhow!("{cmd} not available for {archive:?}: {e}"))?;
            let out = c.stdout.take().expect("piped");
            child = Some(c);
            Box::new(out)
        }
        None => Box::new(std::fs::File::open(archive)?),
    };

    let mut tar = tar::Archive::new(reader);
    let (mut seen, mut skipped) = (0usize, 0usize);
    let mut buf = String::new();

    for entry in tar.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let raw = entry.path()?.to_string_lossy().into_owned();
        // Strip the wrapper directory a release tarball adds.
        let path = raw.split_once('/').map(|(_, rest)| rest).unwrap_or(&raw);
        if path.is_empty() {
            continue;
        }
        buf.clear();
        // `read_to_string` fails on non-UTF-8, which is the same rule the filesystem
        // walker applies — the two paths must agree or their censuses are not
        // comparable.
        match entry.read_to_string(&mut buf) {
            Ok(_) => {
                seen += 1;
                f(Entry { path, text: &buf });
            }
            Err(_) => skipped += 1,
        }
    }

    if let Some(mut c) = child {
        // A decompressor that died mid-stream leaves a TRUNCATED archive that parses
        // as a short one. Reporting success for a partial read is the failure mode
        // this guards: an empty or half-read tree must not look like a clean ingest.
        let status = c.wait()?;
        if !status.success() {
            anyhow::bail!("decompressor failed ({status}) after {seen} entries");
        }
    }
    if seen == 0 {
        anyhow::bail!("no files in {} — wrong format, or not a tar", archive.display());
    }
    Ok((seen, skipped))
}
