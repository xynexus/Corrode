//! Virtual file system: the interface the explorer and subagents see the repo through.
//!
//! The target design is graph<->git: the HelixDB graph ([`crate::graph`]) is the
//! source of truth, and the VFS projects a slice of it as a git-compliant tree so
//! any git-aware tool sees a normal working copy, while writes fold back into graph
//! mutations. The trait below is that seam — `list`/`read`/`write` over
//! git-compliant paths.
//!
//! Methods take `&self`: the daemon shares one VFS across the command loop, and
//! writes land in the store/filesystem, not in per-instance state.

use corrode_core::FileNodeView;
use std::path::PathBuf;

pub trait Vfs: Send + Sync {
    /// Entries directly under `dir` (explorer one-level listing).
    fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>>;
    // ponytail: read/write have no loop caller yet; wired with ReadFile/WriteFile
    // commands when the explorer's file open/edit lands. Covered by the vfs test.
    /// Full contents of a file path.
    #[allow(dead_code)]
    fn read(&self, path: &str) -> anyhow::Result<Vec<u8>>;
    /// Write a file path (the edit/"absorb" direction).
    #[allow(dead_code)]
    fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()>;
}

/// Passthrough VFS over a real directory tree, rooted at `root`.
///
/// ponytail: a real-but-plain stand-in so the explorer and subagents have live
/// files today. It is NOT the graph projection — it reads/writes the host
/// filesystem directly. The HelixDB-backed `Vfs` (project graph nodes as files,
/// absorb edits as node/edge mutations) supersedes it; this exists so nothing
/// downstream has to wait for that. Paths are confined under `root`.
pub struct PassthroughVfs {
    root: PathBuf,
}

impl PassthroughVfs {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Join a VFS path onto the root, rejecting escapes (`..`, absolute paths).
    fn resolve(&self, path: &str) -> anyhow::Result<PathBuf> {
        let rel = path.trim_start_matches('/');
        if rel.split('/').any(|c| c == "..") {
            anyhow::bail!("path escapes VFS root: {path}");
        }
        Ok(self.root.join(rel))
    }
}

impl Vfs for PassthroughVfs {
    fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>> {
        let base = self.resolve(dir)?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&base)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel = if dir.is_empty() || dir == "/" {
                name
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), name)
            };
            entries.push(FileNodeView {
                path: rel,
                bytes: if meta.is_file() { meta.len() } else { 0 },
                node_id: None, // ponytail: set once entries are backed by graph nodes.
            });
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }

    fn read(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        Ok(std::fs::read(self.resolve(path)?)?)
    }

    fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()> {
        let full = self.resolve(path)?;
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(std::fs::write(full, contents)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_write_list_read_roundtrip_and_rejects_escape() {
        let root = std::env::temp_dir().join(format!("corrode-vfs-{}", std::process::id()));
        let vfs = PassthroughVfs::new(&root);

        vfs.write("sub/a.txt", b"hello").unwrap();
        assert_eq!(vfs.read("sub/a.txt").unwrap(), b"hello");

        let listing = vfs.list("sub").unwrap();
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].path, "sub/a.txt");
        assert_eq!(listing[0].bytes, 5);

        assert!(vfs.read("../etc/passwd").is_err());

        std::fs::remove_dir_all(&root).ok();
    }
}
