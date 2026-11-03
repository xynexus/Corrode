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

/// A file open for writing. Byte-range writes splice into `data`, which starts as the
/// file's contents -- empty after `O_TRUNC` -- and goes back to the VFS at every
/// `flush` (each `close`), so the next process to open the file sees it, and at
/// `release`. Starting from nothing made an append, or any write past offset 0,
/// replace the file with zeros and the new bytes.
struct WriteBuf {
    path: String,
    data: Vec<u8>,
    dirty: bool,
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

    fn new_fh(&self) -> u64 {
        self.next_fh.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Write `fh`'s buffer back to the VFS when it has changed since the last time.
    async fn commit(&self, fh: u64) -> FuseResult<()> {
        let pending = {
            let mut writes = self.writes.lock().unwrap(); // no await while held
            match writes.get_mut(&fh) {
                Some(b) if b.dirty => {
                    b.dirty = false;
                    Some((b.path.clone(), b.data.clone()))
                }
                _ => None,
            }
        };
        if let Some((path, data)) = pending {
            // The "absorb" boundary: one whole-file write per close, not per syscall.
            self.vfs.write(&path, &data).await.map_err(|e| errno(&e))?;
        }
        Ok(())
    }

    /// The VFS's attributes for `path`, with the size of a write still in a buffer: a
    /// `stat` between `write` and `close` must not report the old length.
    async fn stat(&self, path: &str) -> FuseResult<FileNodeView> {
        let mut view = self.vfs.stat(path).await.map_err(|e| errno(&e))?;
        let writes = self.writes.lock().unwrap();
        if let Some(b) = writes.values().find(|b| b.path == path && b.dirty) {
            view.bytes = b.data.len() as u64;
        }
        Ok(view)
    }
}

/// The errno for a VFS error: its `std::io::ErrorKind` when it carries one.
fn errno(e: &anyhow::Error) -> Errno {
    use std::io::ErrorKind::*;
    let code = match e.downcast_ref::<std::io::Error>().map(|e| e.kind()) {
        Some(NotFound) => libc::ENOENT,
        Some(AlreadyExists) => libc::EEXIST,
        Some(DirectoryNotEmpty) => libc::ENOTEMPTY,
        Some(IsADirectory) => libc::EISDIR,
        Some(NotADirectory) => libc::ENOTDIR,
        Some(PermissionDenied) => libc::EACCES,
        _ => libc::EIO,
    };
    Errno::from(code)
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
    // The VFS's time, not a constant: cargo rebuilds what is newer than its last
    // build, and through an epoch mtime an edit was never rebuilt (measured: a syntax
    // error appended through the mount still built "Finished").
    let t = SystemTime::UNIX_EPOCH + Duration::from_nanos(v.mtime_ns.unwrap_or(0));
    FileAttr {
        size: v.bytes,
        blocks: v.bytes.div_ceil(512),
        atime: t,
        mtime: t,
        ctime: t,
        kind: if v.is_dir {
            FileType::Directory
        } else {
            FileType::RegularFile
        },
        perm: if v.is_dir || v.executable { 0o755 } else { 0o644 },
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
            mtime_ns: None,
            executable: false,
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
        let view = self.stat(&join(parent, name)?).await?;
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
            Some(p) => attr_for(&self.stat(os_str(p)?).await?, req.uid, req.gid),
        };
        Ok(ReplyAttr { ttl: TTL, attr })
    }

    async fn setattr(
        &self,
        req: Request,
        path: Option<&OsStr>,
        fh: Option<u64>,
        set_attr: SetAttr,
    ) -> FuseResult<ReplyAttr> {
        let path = match path {
            Some(p) => os_str(p)?.to_owned(),
            None => {
                let writes = self.writes.lock().unwrap();
                let fh = fh.ok_or_else(Errno::new_not_exist)?;
                writes.get(&fh).map(|b| b.path.clone()).ok_or_else(Errno::new_not_exist)?
            }
        };
        if let Some(size) = set_attr.size {
            // Truncate (or extend with zeros): in the open buffer when there is one, so
            // a later write to it lands on the truncated file.
            let in_buffer = {
                let mut writes = self.writes.lock().unwrap();
                let buf = fh
                    .and_then(|fh| writes.get_mut(&fh).filter(|b| b.path == path))
                    .is_some();
                if buf {
                    let b = writes.get_mut(&fh.unwrap()).unwrap();
                    b.data.resize(size as usize, 0);
                    b.dirty = true;
                }
                buf
            };
            if !in_buffer {
                let mut data = self.vfs.read(&path).await.map_err(|e| errno(&e))?;
                data.resize(size as usize, 0);
                self.vfs.write(&path, &data).await.map_err(|e| errno(&e))?;
            }
        }
        if let Some(mode) = set_attr.mode {
            let executable = mode & 0o100 != 0;
            if self.stat(&path).await?.executable != executable {
                self.vfs.set_executable(&path, executable).await.map_err(|e| errno(&e))?;
            }
        }
        // ponytail: an explicit atime/mtime (`touch`) is ignored -- the VFS keeps its own
        // times. A `touch` meant to force a rebuild does not; add `Vfs::touch` if needed.
        Ok(ReplyAttr {
            ttl: TTL,
            attr: attr_for(&self.stat(&path).await?, req.uid, req.gid),
        })
    }

    async fn open(&self, _req: Request, path: &OsStr, flags: u32) -> FuseResult<ReplyOpen> {
        let fh = self.new_fh();
        let flags = flags as i32;
        // Opened to write and truncate: the buffer starts empty and is written back
        // even if nothing is written to it, since the truncation is the change.
        if flags & libc::O_ACCMODE != libc::O_RDONLY && flags & libc::O_TRUNC != 0 {
            let path = os_str(path)?.to_owned();
            self.writes
                .lock()
                .unwrap()
                .insert(fh, WriteBuf { path, data: Vec::new(), dirty: true });
        }
        Ok(ReplyOpen { fh, flags: 0 })
    }

    async fn create(
        &self,
        req: Request,
        parent: &OsStr,
        name: &OsStr,
        mode: u32,
        _flags: u32,
    ) -> FuseResult<ReplyCreated> {
        let path = join(parent, name)?;
        self.vfs.write(&path, b"").await.map_err(|e| errno(&e))?;
        if mode & 0o100 != 0 {
            self.vfs.set_executable(&path, true).await.map_err(|e| errno(&e))?;
        }
        let fh = self.new_fh();
        self.writes
            .lock()
            .unwrap()
            .insert(fh, WriteBuf { path: path.clone(), data: Vec::new(), dirty: false });
        Ok(ReplyCreated {
            ttl: TTL,
            attr: attr_for(&self.stat(&path).await?, req.uid, req.gid),
            generation: 0,
            fh,
            flags: 0,
        })
    }

    async fn mkdir(
        &self,
        req: Request,
        parent: &OsStr,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
    ) -> FuseResult<ReplyEntry> {
        let path = join(parent, name)?;
        self.vfs.mkdir(&path).await.map_err(|e| errno(&e))?;
        Ok(ReplyEntry {
            ttl: TTL,
            attr: attr_for(&self.stat(&path).await?, req.uid, req.gid),
        })
    }

    async fn unlink(&self, _req: Request, parent: &OsStr, name: &OsStr) -> FuseResult<()> {
        let path = join(parent, name)?;
        if self.stat(&path).await?.is_dir {
            return Err(Errno::from(libc::EISDIR));
        }
        self.vfs.remove(&path).await.map_err(|e| errno(&e))
    }

    async fn rmdir(&self, _req: Request, parent: &OsStr, name: &OsStr) -> FuseResult<()> {
        let path = join(parent, name)?;
        if !self.stat(&path).await?.is_dir {
            return Err(Errno::from(libc::ENOTDIR));
        }
        self.vfs.remove(&path).await.map_err(|e| errno(&e))
    }

    async fn rename(
        &self,
        _req: Request,
        origin_parent: &OsStr,
        origin_name: &OsStr,
        parent: &OsStr,
        name: &OsStr,
    ) -> FuseResult<()> {
        let (from, to) = (join(origin_parent, origin_name)?, join(parent, name)?);
        self.vfs.rename(&from, &to).await.map_err(|e| errno(&e))?;
        // A file still open under its old name keeps writing to it at its new one.
        let prefix = format!("{from}/");
        for b in self.writes.lock().unwrap().values_mut() {
            if b.path == from {
                b.path = to.clone();
            } else if let Some(rest) = b.path.strip_prefix(&prefix) {
                b.path = format!("{to}/{rest}");
            }
        }
        Ok(())
    }

    async fn read(
        &self,
        _req: Request,
        path: Option<&OsStr>,
        fh: u64,
        offset: u64,
        size: u32,
    ) -> FuseResult<ReplyData> {
        let slice = |bytes: &[u8]| {
            let start = (offset as usize).min(bytes.len());
            let end = (start + size as usize).min(bytes.len());
            ReplyData::from(Bytes::copy_from_slice(&bytes[start..end]))
        };
        // A handle being written reads its own writes.
        if let Some(b) = self.writes.lock().unwrap().get(&fh) {
            return Ok(slice(&b.data));
        }
        let p = os_str(path.ok_or_else(Errno::new_not_exist)?)?;
        // ponytail: reads the whole file then slices — this is exactly where lazy
        // graph materialization would fault in only the requested byte range.
        let bytes = self.vfs.read(p).await.map_err(|e| errno(&e))?;
        Ok(slice(&bytes))
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
        let known = self.writes.lock().unwrap().contains_key(&fh);
        if !known {
            // The first write through this handle: start from what the file holds.
            let p = os_str(path.ok_or_else(Errno::new_not_exist)?)?.to_owned();
            let current = match self.vfs.read(&p).await {
                Ok(bytes) => bytes,
                Err(e) if errno(&e) == Errno::from(libc::ENOENT) => Vec::new(),
                Err(e) => return Err(errno(&e)),
            };
            self.writes
                .lock()
                .unwrap()
                .entry(fh)
                .or_insert(WriteBuf { path: p, data: current, dirty: false });
        }
        let mut writes = self.writes.lock().unwrap(); // no await while held
        let buf = writes.get_mut(&fh).expect("inserted above");
        let off = offset as usize;
        let end = off + data.len();
        if buf.data.len() < end {
            buf.data.resize(end, 0);
        }
        buf.data[off..end].copy_from_slice(data);
        buf.dirty = true;
        Ok(ReplyWrite {
            written: data.len() as u32,
        })
    }

    async fn flush(&self, _req: Request, _path: Option<&OsStr>, fh: u64, _lock_owner: u64) -> FuseResult<()> {
        self.commit(fh).await
    }

    async fn fsync(&self, _req: Request, _path: Option<&OsStr>, fh: u64, _datasync: bool) -> FuseResult<()> {
        self.commit(fh).await
    }

    async fn release(
        &self,
        _req: Request,
        _path: Option<&OsStr>,
        fh: u64,
        _flags: u32,
        _lock_owner: u64,
        _flush: bool,
    ) -> FuseResult<()> {
        let committed = self.commit(fh).await;
        self.writes.lock().unwrap().remove(&fh);
        committed
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

    // What shells, editors and build tools do to a mounted VFS view: append, truncate,
    // save through a temp file and rename, make and remove directories, chmod.
    #[cfg(feature = "fuse")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_view_takes_the_writes_tools_make() {
        let dir = crate::view::tests::repo("fuse-ops");
        let view = Arc::new(crate::view::ViewVfs::open(&dir, "HEAD").unwrap());
        let fs = CorrodeFs::new(view.clone());
        let p = |s: &'static str| Some(OsStr::new(s));

        // Append to a base file: the existing bytes stay.
        let fh = fs.open(req(), OsStr::new("README.md"), libc::O_WRONLY as u32).await.unwrap().fh;
        fs.write(req(), p("README.md"), fh, 6, b"world\n", 0, 0).await.unwrap();
        // Before close, stat reports the new size and the handle reads its own write.
        assert_eq!(fs.getattr(req(), p("README.md"), None, 0).await.unwrap().attr.size, 12);
        fs.flush(req(), p("README.md"), fh, 0).await.unwrap();
        fs.release(req(), p("README.md"), fh, 0, 0, false).await.unwrap();
        assert_eq!(view.read("README.md").await.unwrap(), b"hello\nworld\n");

        // O_TRUNC with nothing written still empties the file.
        let fh = fs.open(req(), OsStr::new("README.md"), (libc::O_WRONLY | libc::O_TRUNC) as u32).await.unwrap().fh;
        fs.release(req(), p("README.md"), fh, 0, 0, true).await.unwrap();
        assert_eq!(view.read("README.md").await.unwrap(), b"");

        // An editor's save: create a temp file, write it, rename it over the original.
        let fh = fs.create(req(), OsStr::new("src"), OsStr::new(".lib.rs.swp"), 0o644, 0).await.unwrap().fh;
        fs.write(req(), p("src/.lib.rs.swp"), fh, 0, b"pub fn g() {}\n", 0, 0).await.unwrap();
        fs.release(req(), p("src/.lib.rs.swp"), fh, 0, 0, true).await.unwrap();
        fs.rename(req(), OsStr::new("src"), OsStr::new(".lib.rs.swp"), OsStr::new("src"), OsStr::new("lib.rs")).await.unwrap();
        assert_eq!(view.read("src/lib.rs").await.unwrap(), b"pub fn g() {}\n");
        assert!(fs.lookup(req(), OsStr::new("src"), OsStr::new(".lib.rs.swp")).await.is_err());

        // Truncate by setattr, chmod +x, mkdir/rmdir, unlink, and the errnos for misuse.
        let size = SetAttr { size: Some(3), ..Default::default() };
        assert_eq!(fs.setattr(req(), p("src/lib.rs"), None, size).await.unwrap().attr.size, 3);
        let mode = SetAttr { mode: Some(0o755), ..Default::default() };
        assert_eq!(fs.setattr(req(), p("src/lib.rs"), None, mode).await.unwrap().attr.perm, 0o755);
        fs.mkdir(req(), OsStr::new(""), OsStr::new("out"), 0o755, 0).await.unwrap();
        assert_eq!(fs.mkdir(req(), OsStr::new(""), OsStr::new("out"), 0o755, 0).await.unwrap_err(), Errno::from(libc::EEXIST));
        assert_eq!(fs.rmdir(req(), OsStr::new(""), OsStr::new("src")).await.unwrap_err(), Errno::from(libc::ENOTEMPTY));
        fs.rmdir(req(), OsStr::new(""), OsStr::new("out")).await.unwrap();
        fs.unlink(req(), OsStr::new(""), OsStr::new("run.sh")).await.unwrap();
        assert_eq!(fs.lookup(req(), OsStr::new(""), OsStr::new("run.sh")).await.unwrap_err(), Errno::from(libc::ENOENT));
        assert_eq!(view.tracked_files().await.unwrap(), ["README.md", "src/lib.rs"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
