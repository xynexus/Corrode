//! FUSE adapter: mount the [`Vfs`] as a real filesystem, so git and subagent
//! shells see the graph<->git projection as a normal working copy.
//!
//! Built on `fuse3`'s *path* API — it owns the inode<->path map, and our `Vfs` is
//! already path-based, so we skip inode bookkeeping. `fuse3` is a pure-protocol
//! crate (talks `/dev/fuse`, mounts via `fusermount3`); no libfuse link. Enabled
//! by `--features fuse`.
//!
//! Two mismatches drive the design:
//! - `Vfs` is async, so `fuse3` handlers just `.await` it — no sync bridge (the
//!   whole reason the trait went async).
//! - FUSE delivers writes as `(offset, chunk)` but `Vfs::write` is whole-file, so
//!   writes accumulate in a per-fh buffer and commit once at `release` — which is
//!   also the graph-backed VFS's "absorb" boundary (one mutation per edit, not per
//!   syscall).

use crate::vfs::Vfs;
use bytes::Bytes;
use corrode_core::FileNodeView;
use fuse3::path::prelude::*;
use fuse3::{Errno, MountOptions, Result as FuseResult};
use futures_util::stream::{self, Stream};
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Attribute/entry cache TTL handed to the kernel. Short: the graph is the truth,
/// and we don't yet push invalidations.
const TTL: Duration = Duration::from_secs(1);

/// A file open for writing: byte-range FUSE writes splice into `data`, flushed to
/// `Vfs::write(path, ..)` once at `release`.
struct WriteBuf {
    path: String,
    data: Vec<u8>,
}

pub struct CorrodeFs {
    vfs: Arc<dyn Vfs>,
    next_fh: AtomicU64,
    writes: Mutex<HashMap<u64, WriteBuf>>,
}

impl CorrodeFs {
    pub fn new(vfs: Arc<dyn Vfs>) -> Self {
        Self {
            vfs,
            next_fh: AtomicU64::new(0),
            writes: Mutex::new(HashMap::new()),
        }
    }
}

/// git-compliant VFS path from a fuse3 (parent, name) pair. Non-UTF8 -> EINVAL.
fn join(parent: &OsStr, name: &OsStr) -> FuseResult<String> {
    let p = os_str(parent)?.trim_start_matches('/');
    let n = os_str(name)?;
    Ok(if p.is_empty() {
        n.to_owned()
    } else {
        format!("{p}/{n}")
    })
}

fn os_str(s: &OsStr) -> FuseResult<&str> {
    s.to_str().ok_or_else(|| Errno::from(libc::EINVAL))
}

/// Synthesize FUSE attributes from a projected view. Times are epoch (the graph is
/// the source of truth, not mtime); perms are conventional.
// ponytail: `v.mode` (Composed/Overlay/Fallback) is a natural xattr, so `getfattr`
// / the explorer can see which files fell back — wire via getxattr/listxattr.
fn attr_for(v: &FileNodeView, uid: u32, gid: u32) -> FileAttr {
    FileAttr {
        size: v.bytes,
        blocks: v.bytes.div_ceil(512),
        atime: SystemTime::UNIX_EPOCH,
        mtime: SystemTime::UNIX_EPOCH,
        ctime: SystemTime::UNIX_EPOCH,
        kind: if v.is_dir {
            FileType::Directory
        } else {
            FileType::RegularFile
        },
        perm: if v.is_dir { 0o755 } else { 0o644 },
        nlink: if v.is_dir { 2 } else { 1 },
        uid,
        gid,
        rdev: 0,
        blksize: 512,
    }
}

/// Attributes for a synthetic directory (root, `.`, `..`).
fn dir_attr(uid: u32, gid: u32) -> FileAttr {
    attr_for(
        &FileNodeView {
            path: String::new(),
            is_dir: true,
            bytes: 0,
            node_id: None,
            mode: None,
        },
        uid,
        gid,
    )
}

impl PathFilesystem for CorrodeFs {
    async fn init(&self, _req: Request) -> FuseResult<ReplyInit> {
        Ok(ReplyInit {
            max_write: NonZeroU32::new(128 * 1024).unwrap(),
        })
    }

    async fn destroy(&self, _req: Request) {}

    async fn lookup(&self, req: Request, parent: &OsStr, name: &OsStr) -> FuseResult<ReplyEntry> {
        let path = join(parent, name)?;
        let view = self
            .vfs
            .stat(&path)
            .await
            .map_err(|_| Errno::new_not_exist())?;
        Ok(ReplyEntry {
            ttl: TTL,
            attr: attr_for(&view, req.uid, req.gid),
        })
    }

    async fn getattr(
        &self,
        req: Request,
        path: Option<&OsStr>,
        _fh: Option<u64>,
        _flags: u32,
    ) -> FuseResult<ReplyAttr> {
        // `None` means root (or a deleted path) — treat as the root directory.
        let attr = match path {
            None => dir_attr(req.uid, req.gid),
            Some(p) => {
                let view = self
                    .vfs
                    .stat(os_str(p)?)
                    .await
                    .map_err(|_| Errno::new_not_exist())?;
                attr_for(&view, req.uid, req.gid)
            }
        };
        Ok(ReplyAttr { ttl: TTL, attr })
    }

    async fn open(&self, _req: Request, _path: &OsStr, _flags: u32) -> FuseResult<ReplyOpen> {
        // Stateless handle: reads go straight to the Vfs; writes buffer under this fh.
        let fh = self.next_fh.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(ReplyOpen { fh, flags: 0 })
    }

    async fn read(
        &self,
        _req: Request,
        path: Option<&OsStr>,
        _fh: u64,
        offset: u64,
        size: u32,
    ) -> FuseResult<ReplyData> {
        let p = os_str(path.ok_or_else(Errno::new_not_exist)?)?;
        // ponytail: reads the whole file then slices — this is exactly where lazy
        // graph materialization would fault in only the requested byte range.
        let bytes = self.vfs.read(p).await.map_err(|_| Errno::new_not_exist())?;
        let start = (offset as usize).min(bytes.len());
        let end = (start + size as usize).min(bytes.len());
        Ok(ReplyData::from(Bytes::copy_from_slice(&bytes[start..end])))
    }

    #[allow(clippy::too_many_arguments)]
    async fn write(
        &self,
        _req: Request,
        path: Option<&OsStr>,
        fh: u64,
        offset: u64,
        data: &[u8],
        _write_flags: u32,
        _flags: u32,
    ) -> FuseResult<ReplyWrite> {
        let rel = path.map(os_str).transpose()?;
        // ponytail: no truncate/setattr yet, so an editor that truncates-then-writes
        // (setattr size=0) won't round-trip; offset-splice covers sequential writes.
        let mut writes = self.writes.lock().unwrap(); // no await while held
        let buf = writes.entry(fh).or_insert_with(|| WriteBuf {
            path: rel.unwrap_or_default().to_owned(),
            data: Vec::new(),
        });
        if buf.path.is_empty() {
            if let Some(r) = rel {
                buf.path = r.to_owned();
            }
        }
        let off = offset as usize;
        let end = off + data.len();
        if buf.data.len() < end {
            buf.data.resize(end, 0);
        }
        buf.data[off..end].copy_from_slice(data);
        Ok(ReplyWrite {
            written: data.len() as u32,
        })
    }

    async fn release(
        &self,
        _req: Request,
        path: Option<&OsStr>,
        fh: u64,
        _flags: u32,
        _lock_owner: u64,
        _flush: bool,
    ) -> FuseResult<()> {
        let taken = self.writes.lock().unwrap().remove(&fh); // drop lock before await
        if let Some(buf) = taken {
            let target = if buf.path.is_empty() {
                path.map(os_str).transpose()?.map(str::to_owned)
            } else {
                Some(buf.path)
            };
            // The "absorb" boundary: one whole-file write per edit, not per syscall.
            if let Some(p) = target {
                self.vfs
                    .write(&p, &buf.data)
                    .await
                    .map_err(|_| Errno::from(libc::EIO))?;
            }
        }
        Ok(())
    }

    async fn readdirplus<'a>(
        &'a self,
        req: Request,
        parent: &'a OsStr,
        _fh: u64,
        offset: u64,
        _lock_owner: u64,
    ) -> FuseResult<ReplyDirectoryPlus<impl Stream<Item = FuseResult<DirectoryEntryPlus>> + Send + 'a>>
    {
        let (uid, gid) = (req.uid, req.gid);
        let listing = self
            .vfs
            .list(os_str(parent)?)
            .await
            .map_err(|_| Errno::new_not_exist())?;

        // `.` and `..` first, then the real entries. Cookie = 1-based index; the
        // kernel resumes by passing the last cookie it consumed as `offset`.
        let mut rows: Vec<(OsString, FileAttr)> = Vec::with_capacity(listing.len() + 2);
        let d = dir_attr(uid, gid);
        rows.push((OsString::from("."), d));
        rows.push((OsString::from(".."), d));
        for e in &listing {
            let name = e.path.rsplit('/').next().unwrap_or(&e.path);
            rows.push((OsString::from(name), attr_for(e, uid, gid)));
        }

        let entries: Vec<FuseResult<DirectoryEntryPlus>> = rows
            .into_iter()
            .enumerate()
            .map(|(i, (name, attr))| (i as u64 + 1, name, attr))
            .filter(|(cookie, _, _)| *cookie > offset)
            .map(|(cookie, name, attr)| {
                Ok(DirectoryEntryPlus {
                    kind: attr.kind,
                    name,
                    offset: cookie as i64,
                    attr,
                    entry_ttl: TTL,
                    attr_ttl: TTL,
                })
            })
            .collect();

        Ok(ReplyDirectoryPlus {
            entries: stream::iter(entries),
        })
    }
}

/// Mount the VFS at `mountpoint` and serve until unmounted. Unprivileged mount
/// (via `fusermount3`), so the daemon needn't run as root.
pub async fn mount(vfs: Arc<dyn Vfs>, mountpoint: impl AsRef<Path>) -> anyhow::Result<()> {
    let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
    let mut opts = MountOptions::default();
    opts.fs_name("corrode")
        .force_readdir_plus(true)
        .uid(uid)
        .gid(gid);
    let handle = Session::new(opts)
        .mount_with_unprivileged(CorrodeFs::new(vfs), mountpoint.as_ref())
        .await?;
    handle.await?; // MountHandle is a Future that resolves on unmount
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::PassthroughVfs;
    use futures_util::StreamExt;

    fn req() -> Request {
        Request {
            unique: 1,
            uid: 1000,
            gid: 1000,
            pid: 1,
        }
    }

    // Drives the adapter's handlers directly (no real mount, no root needed):
    // lookup/read/readdirplus over a PassthroughVfs, plus the buffered write path
    // committing at release.
    #[tokio::test]
    async fn adapter_lookup_read_readdirplus_and_write_absorb() {
        let root = std::env::temp_dir().join(format!("corrode-fuse-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        PassthroughVfs::new(&root)
            .write("dir/a.txt", b"hello fuse")
            .await
            .unwrap();

        let fs = CorrodeFs::new(Arc::new(PassthroughVfs::new(&root)));

        // lookup: file kind + size, and directory kind
        let file = fs
            .lookup(req(), OsStr::new("dir"), OsStr::new("a.txt"))
            .await
            .unwrap();
        assert_eq!(file.attr.kind, FileType::RegularFile);
        assert_eq!(file.attr.size, 10);
        let dir = fs
            .lookup(req(), OsStr::new(""), OsStr::new("dir"))
            .await
            .unwrap();
        assert_eq!(dir.attr.kind, FileType::Directory);

        // read (with offset slicing)
        let data = fs
            .read(req(), Some(OsStr::new("dir/a.txt")), 0, 6, 100)
            .await
            .unwrap();
        assert_eq!(&data.data[..], b"fuse");

        // readdirplus lists a.txt alongside . and ..
        let reply = fs
            .readdirplus(req(), OsStr::new("dir"), 0, 0, 0)
            .await
            .unwrap();
        let names: Vec<String> = reply
            .entries
            .filter_map(|e| async move { e.ok().map(|d| d.name.to_string_lossy().into_owned()) })
            .collect()
            .await;
        assert!(names.contains(&"a.txt".to_string()));
        assert!(names.contains(&".".to_string()));
        assert!(names.contains(&"..".to_string()));

        // write path: buffer chunks under an fh, absorb once at release
        let fh = fs
            .open(req(), OsStr::new("dir/b.txt"), 0)
            .await
            .unwrap()
            .fh;
        fs.write(req(), Some(OsStr::new("dir/b.txt")), fh, 0, b"wr", 0, 0)
            .await
            .unwrap();
        fs.write(req(), Some(OsStr::new("dir/b.txt")), fh, 2, b"itten", 0, 0)
            .await
            .unwrap();
        fs.release(req(), Some(OsStr::new("dir/b.txt")), fh, 0, 0, false)
            .await
            .unwrap();
        assert_eq!(
            PassthroughVfs::new(&root).read("dir/b.txt").await.unwrap(),
            b"written"
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
