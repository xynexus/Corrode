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

use async_trait::async_trait;
use corrode_core::FileNodeView;
use std::path::{Path, PathBuf};

/// Async because the real backing store is I/O: the graph-backed impl hits
/// HelixDB/LMDB and hipfire, and a FUSE mount awaits these handlers per syscall.
/// An impl must never stall the executor — `PassthroughVfs` honors that by
/// offloading its blocking `std::fs` work to the blocking pool.
#[async_trait]
pub trait Vfs: Send + Sync {
    /// Entries directly under `dir` (explorer one-level listing).
    async fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>>;
    /// Attributes for a single path — what FUSE `getattr`/`lookup` needs, without
    /// listing (and scanning) the whole parent directory per call.
    async fn stat(&self, path: &str) -> anyhow::Result<FileNodeView>;
    /// Full contents of a file path. Used by the context prefix's README digest.
    async fn read(&self, path: &str) -> anyhow::Result<Vec<u8>>;
    // ponytail: `write` has no loop caller yet; wired with the WriteFile command when
    // the explorer's file open/edit lands. Covered by the vfs test.
    /// Write a file path (the edit/"absorb" direction).
    #[allow(dead_code)]
    async fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()>;

    /// Every regular file this VFS considers part of the project — the searchable
    /// corpus. Paths are repo-relative, in the VFS's own terms.
    ///
    /// This exists so `search_files` holds no policy: it asks what exists rather than
    /// deciding. A blacklist is always one new vendored directory behind, and the
    /// question "what is the corpus" already has an owner — the VFS. A graph-backed
    /// VFS answers from its file nodes at the same call site.
    ///
    /// It also removes a real hazard: when search and read derive from one definition
    /// of what exists they cannot disagree, whereas a subprocess grep reading the
    /// filesystem directly would, the moment the VFS stops being a passthrough.
    async fn tracked_files(&self) -> anyhow::Result<Vec<String>>;
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

/// The real path of `full` must stay under the real `root`. `resolve` rejects `..` only
/// lexically, while the I/O here runs in the daemon -- outside any sandbox -- and
/// follows symlinks: a link planted in the repo (by a sandboxed `ln -s ~/.bashrc x`, or
/// committed) made write_file and read_file reach outside it. The deepest existing part
/// of the path is checked, so a file about to be created is covered by its parent, and
/// a dangling link is refused (writing through it would create its outside target).
/// ponytail: check-then-open, not openat2(RESOLVE_BENEATH): a process racing to swap a
/// directory for a link between the check and the open can still escape.
fn confine(root: &Path, full: &Path) -> anyhow::Result<()> {
    if real_path(full)?.starts_with(real_path(root)?) {
        Ok(())
    } else {
        anyhow::bail!("path leaves the repository: {}", full.display())
    }
}

/// `p` with every existing part resolved (symlinks followed) and the part that does
/// not exist yet appended as written. A dangling link is an error.
fn real_path(p: &Path) -> anyhow::Result<PathBuf> {
    let mut probe = p;
    let mut rest: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        match std::fs::canonicalize(probe) {
            Ok(real) => return Ok(rest.iter().rev().fold(real, |acc, c| acc.join(c))),
            Err(_) if std::fs::symlink_metadata(probe).is_ok() => {
                anyhow::bail!("refusing a dangling symlink: {}", probe.display())
            }
            Err(_) => {
                rest.extend(probe.file_name());
                probe = match probe.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => parent,
                    _ => Path::new("."),
                };
            }
        }
    }
}

/// Directories the tool layer may read but never write: `.git` (a written
/// `core.hooksPath` or `fsmonitor` runs code at the human's next `git status`) and
/// `.corrode` (the graph store and skills, which the sandbox mounts read-only).
fn write_protected(path: &str) -> bool {
    matches!(
        path.trim_start_matches('/').split('/').next(),
        Some(".git" | ".corrode")
    )
}

#[async_trait]
impl Vfs for PassthroughVfs {
    async fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>> {
        let base = self.resolve(dir)?; // pure path check, no I/O — stays on the async side
        let dir = dir.to_string();
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<FileNodeView>> {
            confine(&root, &base)?;
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(&base)? {
                let entry = entry?;
                let meta = entry.metadata()?;
                let name = entry.file_name().to_string_lossy().into_owned();
                // Skip VCS/build noise at every level — matches the path-enum walk and
                // search_files prune, keeping both the explorer and agents' list_dir on
                // source (agents were observed inventing `.git/revisions` paths).
                if name == ".git" || name == "target" {
                    continue;
                }
                let rel = if dir.is_empty() || dir == "/" {
                    name
                } else {
                    format!("{}/{}", dir.trim_end_matches('/'), name)
                };
                entries.push(FileNodeView {
                    path: rel,
                    is_dir: meta.is_dir(),
                    bytes: if meta.is_file() { meta.len() } else { 0 },
                    node_id: None, // ponytail: set once entries are backed by graph nodes.
                    mode: None,    // ponytail: passthrough can't know; set by the graph-backed VFS.
                });
            }
            entries.sort_by(|a, b| a.path.cmp(&b.path));
            Ok(entries)
        })
        .await?
    }

    async fn stat(&self, path: &str) -> anyhow::Result<FileNodeView> {
        let full = self.resolve(path)?;
        let path = path.to_string();
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<FileNodeView> {
            confine(&root, &full)?;
            let meta = std::fs::metadata(&full)?;
            Ok(FileNodeView {
                path,
                is_dir: meta.is_dir(),
                bytes: if meta.is_file() { meta.len() } else { 0 },
                node_id: None,
                mode: None,
            })
        })
        .await?
    }

    async fn read(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        let full = self.resolve(path)?;
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
            confine(&root, &full)?;
            Ok(std::fs::read(full)?)
        })
        .await?
    }

    /// `git ls-files` restricted to regular blobs.
    ///
    /// Git already answers "what is this project's content": submodule *contents* are
    /// excluded automatically (a submodule is one gitlink entry, not its tree), and
    /// ignored build output like `webui/dist/` never appears. Measured on this repo:
    /// 165 tracked regular files versus a walk that scans up to 4000, and a search for
    /// "overview" went from 211 hits (~120 junk) to 5.
    ///
    /// The mode filter is load-bearing, not tidiness. A gitlink entry is a DIRECTORY
    /// path; handing one to a searcher makes it recurse straight back into the
    /// submodule, which is the failure this is meant to prevent. `ls-files -s` prints
    /// the mode, and only `100644`/`100755` (regular blobs) are kept — dropping
    /// `160000` gitlinks and `120000` symlinks, the latter because following them
    /// re-enters the tree by another name.
    async fn tracked_files(&self) -> anyhow::Result<Vec<String>> {
        let root = self.root.clone();
        let out = tokio::task::spawn_blocking(move || {
            std::process::Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["ls-files", "-s", "-z"])
                .output()
        })
        .await??;
        if !out.status.success() {
            anyhow::bail!(
                "git ls-files failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        // `-s` entries are `<mode> <oid> <stage>\t<path>`, NUL-separated by `-z` so a
        // path containing a newline can't split one record into two.
        Ok(String::from_utf8_lossy(&out.stdout)
            .split('\0')
            .filter_map(|rec| {
                let (meta, path) = rec.split_once('\t')?;
                let mode = meta.split_whitespace().next()?;
                matches!(mode, "100644" | "100755").then(|| path.to_string())
            })
            .collect())
    }

    async fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()> {
        if write_protected(path) {
            anyhow::bail!("refusing to write inside .git or .corrode: {path}");
        }
        let full = self.resolve(path)?;
        let contents = contents.to_vec();
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            confine(&root, &full)?;
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(full, contents)?;
            Ok(())
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Links planted in the repo must not carry reads or writes out of it: an existing
    // outside target, a dangling one (the write would create it), and a linked
    // directory. Writes into .git and .corrode are refused outright.
    #[tokio::test]
    async fn symlinks_cannot_carry_io_out_of_the_repo() {
        let base = std::env::temp_dir().join(format!("corrode-confine-{}", std::process::id()));
        let (root, outside) = (base.join("repo"), base.join("outside"));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret"), "s3cret").unwrap();
        std::os::unix::fs::symlink(outside.join("secret"), root.join("leak")).unwrap();
        std::os::unix::fs::symlink(outside.join("new"), root.join("dangling")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("dir")).unwrap();
        std::fs::write(root.join("inside.txt"), "ok").unwrap();
        std::os::unix::fs::symlink(root.join("inside.txt"), root.join("alias")).unwrap();
        let vfs = PassthroughVfs::new(&root);

        assert!(vfs.read("leak").await.is_err(), "read through a link to outside");
        assert!(vfs.write("leak", b"x").await.is_err(), "write through a link to outside");
        assert_eq!(std::fs::read_to_string(outside.join("secret")).unwrap(), "s3cret");
        assert!(vfs.write("dangling", b"x").await.is_err());
        assert!(!outside.join("new").exists(), "a dangling link must not create its target");
        assert!(vfs.write("dir/planted", b"x").await.is_err());
        assert!(!outside.join("planted").exists());
        assert!(vfs.list("dir").await.is_err());
        assert!(vfs.write(".git/config", b"x").await.is_err());
        assert!(vfs.write(".corrode/skills/x/SKILL.md", b"x").await.is_err());

        // Inside the repo everything still works, links included.
        assert_eq!(vfs.read("alias").await.unwrap(), b"ok");
        vfs.write("new/deep/file.txt", b"made").await.unwrap();
        assert_eq!(vfs.read("new/deep/file.txt").await.unwrap(), b"made");
        std::fs::remove_dir_all(&base).ok();
    }

    #[tokio::test]
    async fn passthrough_write_list_stat_read_roundtrip_and_rejects_escape() {
        let root = std::env::temp_dir().join(format!("corrode-vfs-{}", std::process::id()));
        let vfs = PassthroughVfs::new(&root);

        vfs.write("sub/a.txt", b"hello").await.unwrap();
        assert_eq!(vfs.read("sub/a.txt").await.unwrap(), b"hello");

        let stat = vfs.stat("sub/a.txt").await.unwrap();
        assert_eq!(stat.path, "sub/a.txt");
        assert_eq!(stat.bytes, 5);

        let listing = vfs.list("sub").await.unwrap();
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].path, "sub/a.txt");
        assert_eq!(listing[0].bytes, 5);

        // Escapes are rejected before any I/O, on both the read and stat paths.
        assert!(vfs.read("../etc/passwd").await.is_err());
        assert!(vfs.stat("../etc/passwd").await.is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn list_prunes_vcs_and_build_noise() {
        let root = std::env::temp_dir().join(format!("corrode-vfs-prune-{}", std::process::id()));
        let vfs = PassthroughVfs::new(&root);
        vfs.write("src/main.rs", b"fn main() {}").await.unwrap();
        // The tool layer may not write .git, so the fixture writes it directly.
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/HEAD"), b"ref: x").unwrap();
        vfs.write("target/debug/x", b"blob").await.unwrap();
        let names: Vec<String> = vfs.list("").await.unwrap().into_iter().map(|e| e.path).collect();
        assert!(names.contains(&"src".to_string()), "source kept: {names:?}");
        assert!(!names.iter().any(|n| n == ".git" || n == "target"), "noise pruned: {names:?}");
        std::fs::remove_dir_all(&root).ok();
    }
}
