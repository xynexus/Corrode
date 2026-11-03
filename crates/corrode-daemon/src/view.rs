//! A VFS view: one turn's picture of the repository -- a base commit plus the changes
//! the turn made, kept apart from the working tree and from every other view
//! (docs/vfs-branches-and-builds.md §8).
//!
//! Unchanged files are read from git's object store and report the base commit's time,
//! so the same file shows the same mtime in every view at that base (build reuse needs
//! that). A changed file reports when it was written, so a build after the write sees
//! it as newer. At the end of a turn [`ViewVfs::commit`] stores the changes as git
//! objects and points `refs/corrode/<name>` at a commit on top of the base: project
//! history only changes when someone promotes that commit.

use crate::vfs::{io_err, Vfs};
use std::io::ErrorKind::{AlreadyExists, DirectoryNotEmpty, IsADirectory, NotADirectory, NotFound, PermissionDenied};
use async_trait::async_trait;
use corrode_core::FileNodeView;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A file or directory of the base commit's tree.
#[derive(Clone)]
struct BaseEntry {
    /// Blob oid; empty for a directory.
    oid: String,
    size: u64,
    dir: bool,
    executable: bool,
}

/// What the view holds for a path the turn changed.
#[derive(Clone)]
enum Change {
    File { data: Arc<[u8]>, executable: bool, mtime_ns: u64 },
    Dir { mtime_ns: u64 },
    Gone,
}

pub struct ViewVfs {
    root: PathBuf,
    base: String,
    base_mtime_ns: u64,
    tree: HashMap<String, BaseEntry>,
    /// Base directory -> the names directly in it.
    children: HashMap<String, BTreeSet<String>>,
    changes: Mutex<BTreeMap<String, Change>>,
    /// The `.git` file a view serves at its root (`gitdir: ...`), making it a linked
    /// worktree of the repository to stock git (see [`TurnView::start`]). Unset: no `.git`.
    git_link: std::sync::OnceLock<String>,
}

fn git(root: &std::path::Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let out = std::process::Command::new("git").arg("-C").arg(root).args(args).output()?;
    if !out.status.success() {
        anyhow::bail!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(out.stdout)
}

fn git_stdin(root: &std::path::Path, args: &[&str], env: &[(&str, &std::ffi::OsStr)], input: &[u8]) -> anyhow::Result<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .envs(env.iter().copied())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().expect("piped");
    let input = input.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&input));
    let out = child.wait_with_output()?;
    writer.join().map_err(|_| anyhow::anyhow!("stdin writer panicked"))??;
    if !out.status.success() {
        anyhow::bail!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(out.stdout)
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(p, _)| p)
}

/// A view path: relative, no `..`, and not inside `.git` or `.corrode` -- a view has no
/// business there. The one exception is `.git` itself, the worktree link file.
fn clean(path: &str) -> anyhow::Result<String> {
    let p = path.trim_matches('/');
    if p.split('/').any(|c| c == ".." || c == ".") {
        return Err(io_err(PermissionDenied, format!("path escapes the view: {path}")));
    }
    if p != ".git" && matches!(p.split('/').next(), Some(".git" | ".corrode")) {
        return Err(io_err(NotFound, format!("no such path in the view: {path}")));
    }
    Ok(p.to_string())
}

/// [`clean`], for a change: the `.git` link is the daemon's, not the turn's.
fn writable(path: &str) -> anyhow::Result<String> {
    let p = clean(path)?;
    if p == ".git" {
        return Err(io_err(PermissionDenied, "the view's .git is read-only"));
    }
    Ok(p)
}

impl ViewVfs {
    /// A view of `rev` in the repository at `root`, with no changes yet.
    pub fn open(root: impl Into<PathBuf>, rev: &str) -> anyhow::Result<Self> {
        let root = root.into();
        let spec = format!("{rev}^{{commit}}");
        let base = String::from_utf8(git(&root, &["rev-parse", "--verify", &spec])?)?.trim().to_string();
        let secs: u64 = String::from_utf8(git(&root, &["show", "-s", "--format=%ct", &base])?)?.trim().parse()?;
        let mut tree = HashMap::new();
        let mut children: HashMap<String, BTreeSet<String>> = HashMap::new();
        children.entry(String::new()).or_default();
        // `<mode> <type> <oid> <size|->\t<path>`, NUL-terminated; `-t` lists the trees too.
        let listing = git(&root, &["ls-tree", "-r", "-t", "-l", "-z", &base])?;
        for rec in listing.split(|&b| b == 0).filter(|r| !r.is_empty()) {
            let rec = String::from_utf8_lossy(rec);
            let Some((meta, path)) = rec.split_once('\t') else { continue };
            let f: Vec<&str> = meta.split_whitespace().collect();
            let [mode, _kind, oid, size] = f[..] else { continue };
            // ponytail: symlinks (120000) and submodules (160000) are left out of the
            // view; serve them once a project needs them inside a turn.
            let entry = match mode {
                "040000" => BaseEntry { oid: String::new(), size: 0, dir: true, executable: false },
                "100644" | "100755" => BaseEntry {
                    oid: oid.to_string(),
                    size: size.parse().unwrap_or(0),
                    dir: false,
                    executable: mode == "100755",
                },
                _ => continue,
            };
            let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
            children.entry(dir.to_string()).or_default().insert(name.to_string());
            if entry.dir {
                children.entry(path.to_string()).or_default();
            }
            tree.insert(path.to_string(), entry);
        }
        Ok(Self {
            root,
            base,
            base_mtime_ns: secs * 1_000_000_000,
            tree,
            children,
            changes: Mutex::new(BTreeMap::new()),
            git_link: std::sync::OnceLock::new(),
        })
    }

    /// Serve `link` as the view's root `.git` file.
    pub fn set_git_link(&self, link: String) {
        let _ = self.git_link.set(link);
    }

    /// The commit this view starts from.
    #[allow(dead_code)]
    pub fn base(&self) -> &str {
        &self.base
    }

    fn view_of(&self, path: &str, changes: &BTreeMap<String, Change>) -> Option<FileNodeView> {
        let node = |dir, bytes, mtime_ns, executable| FileNodeView {
            path: path.to_string(),
            is_dir: dir,
            bytes,
            node_id: None,
            mode: None,
            mtime_ns: Some(mtime_ns),
            executable,
        };
        if path.is_empty() {
            return Some(node(true, 0, self.base_mtime_ns, false));
        }
        if path == ".git" {
            return self.git_link.get().map(|l| node(false, l.len() as u64, self.base_mtime_ns, false));
        }
        match changes.get(path) {
            Some(Change::Gone) => None,
            Some(Change::File { data, executable, mtime_ns }) => {
                Some(node(false, data.len() as u64, *mtime_ns, *executable))
            }
            Some(Change::Dir { mtime_ns }) => Some(node(true, 0, *mtime_ns, false)),
            None => self
                .tree
                .get(path)
                .map(|e| node(e.dir, e.size, self.base_mtime_ns, e.executable)),
        }
    }

    /// Names directly under `dir`, the turn's changes applied.
    fn names_in(&self, dir: &str, changes: &BTreeMap<String, Change>) -> BTreeSet<String> {
        let mut names = self.children.get(dir).cloned().unwrap_or_default();
        if dir.is_empty() && self.git_link.get().is_some() {
            names.insert(".git".to_string());
        }
        let prefix = if dir.is_empty() { String::new() } else { format!("{dir}/") };
        for (path, change) in changes.range(prefix.clone()..) {
            let Some(rest) = path.strip_prefix(&prefix) else { break };
            if rest.contains('/') {
                continue;
            }
            match change {
                Change::Gone => names.remove(rest),
                _ => names.insert(rest.to_string()),
            };
        }
        names
    }

    fn is_dir(&self, path: &str, changes: &BTreeMap<String, Change>) -> bool {
        self.view_of(path, changes).is_some_and(|v| v.is_dir)
    }

    /// Record `path`'s missing ancestors as directories, so a new file's parent lists.
    fn ensure_parents(&self, path: &str, changes: &mut BTreeMap<String, Change>) -> anyhow::Result<()> {
        let mut dirs = Vec::new();
        let mut cur = parent(path);
        while !cur.is_empty() {
            match self.view_of(cur, changes) {
                Some(v) if v.is_dir => break,
                Some(_) => return Err(io_err(NotADirectory, format!("{cur} is a file"))),
                None => dirs.push(cur.to_string()),
            }
            cur = parent(cur);
        }
        let t = now_ns();
        for d in dirs {
            changes.insert(d, Change::Dir { mtime_ns: t });
        }
        Ok(())
    }

    /// Delete a file, or an empty directory.
    fn remove_now(&self, path: &str) -> anyhow::Result<()> {
        let path = writable(path)?;
        let mut changes = self.changes.lock().unwrap();
        let Some(v) = self.view_of(&path, &changes) else { return Err(io_err(NotFound, format!("no such path: {path}"))) };
        if v.is_dir && !self.names_in(&path, &changes).is_empty() {
            return Err(io_err(DirectoryNotEmpty, format!("directory not empty: {path}")));
        }
        changes.insert(path, Change::Gone);
        Ok(())
    }

    /// Create an empty directory.
    fn mkdir_now(&self, path: &str) -> anyhow::Result<()> {
        let path = writable(path)?;
        let mut changes = self.changes.lock().unwrap();
        if self.view_of(&path, &changes).is_some() {
            return Err(io_err(AlreadyExists, format!("already exists: {path}")));
        }
        self.ensure_parents(&path, &mut changes)?;
        changes.insert(path, Change::Dir { mtime_ns: now_ns() });
        Ok(())
    }

    /// Set or clear a file's executable bit.
    fn set_executable_now(&self, path: &str, executable: bool) -> anyhow::Result<()> {
        let path = writable(path)?;
        let data = self.read_now(&path)?;
        let mut changes = self.changes.lock().unwrap();
        changes.insert(path, Change::File { data, executable, mtime_ns: now_ns() });
        Ok(())
    }

    /// Move a file or directory (with everything under it), replacing a file at `to`.
    fn rename_now(&self, from: &str, to: &str) -> anyhow::Result<()> {
        let (from, to) = (writable(from)?, writable(to)?);
        if from == to {
            return Ok(());
        }
        // Every file and directory at or under `from`, as it now is.
        let mut moved: Vec<(String, FileNodeView)> = Vec::new();
        let mut stack = vec![from.clone()];
        {
            let changes = self.changes.lock().unwrap();
            if self.view_of(&from, &changes).is_none() {
                return Err(io_err(NotFound, format!("no such path: {from}")));
            }
            if to.starts_with(&format!("{from}/")) {
                anyhow::bail!("cannot move {from} into itself");
            }
            if self.is_dir(&to, &changes) && !self.names_in(&to, &changes).is_empty() {
                return Err(io_err(DirectoryNotEmpty, format!("directory not empty: {to}")));
            }
            while let Some(p) = stack.pop() {
                let v = self.view_of(&p, &changes).expect("listed");
                if v.is_dir {
                    stack.extend(self.names_in(&p, &changes).into_iter().map(|n| format!("{p}/{n}")));
                }
                moved.push((p, v));
            }
        }
        // Contents read outside the lock: a base blob is a git call.
        let mut files = Vec::new();
        for (p, v) in &moved {
            if !v.is_dir {
                files.push((p.clone(), self.read_now(p)?, v.executable));
            }
        }
        let mut changes = self.changes.lock().unwrap();
        self.ensure_parents(&to, &mut changes)?;
        let t = now_ns();
        let target = |p: &str| format!("{to}{}", &p[from.len()..]);
        for (p, v) in &moved {
            if v.is_dir {
                changes.insert(target(p), Change::Dir { mtime_ns: t });
            }
        }
        for (p, data, executable) in files {
            changes.insert(target(&p), Change::File { data, executable, mtime_ns: t });
        }
        for (p, _) in moved {
            if !self.tree.contains_key(&p) {
                changes.remove(&p);
            } else {
                changes.insert(p, Change::Gone);
            }
        }
        Ok(())
    }

    /// The file's bytes now, blocking (a base blob is a `git cat-file`).
    fn read_now(&self, path: &str) -> anyhow::Result<Arc<[u8]>> {
        let oid = {
            let changes = self.changes.lock().unwrap();
            match changes.get(path) {
                Some(Change::File { data, .. }) => return Ok(Arc::clone(data)),
                Some(Change::Gone) | Some(Change::Dir { .. }) => anyhow::bail!("not a file: {path}"),
                None => match self.tree.get(path) {
                    Some(e) if !e.dir => e.oid.clone(),
                    _ => anyhow::bail!("not a file: {path}"),
                },
            }
        };
        // ponytail: one git process per base read; a long-lived `cat-file --batch`
        // once reads through a mounted view show up in build times.
        Ok(git(&self.root, &["cat-file", "blob", &oid])?.into())
    }

    /// Store the turn's changes as a commit on the base and point `refs/corrode/<name>`
    /// at it. `None` when the view is unchanged (empty directories are not content).
    pub fn commit(&self, name: &str, message: &str) -> anyhow::Result<Option<String>> {
        let mut changes = self.changes.lock().unwrap().clone();
        // A new file that .gitignore excludes is build output or scratch a command left
        // in the tree (`make` writes its objects beside the sources), not the turn's work.
        let new: Vec<&String> = changes
            .iter()
            .filter(|(p, c)| matches!(c, Change::File { .. }) && !self.tree.contains_key(*p))
            .map(|(p, _)| p)
            .collect();
        if !new.is_empty() {
            let list: String = new.iter().map(|p| format!("{p}\n")).collect();
            // Exit 1 = nothing ignored, which is not an error here.
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&self.root)
                .args(["check-ignore", "--stdin"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .and_then(|mut c| {
                    use std::io::Write;
                    c.stdin.take().expect("piped").write_all(list.as_bytes())?;
                    c.wait_with_output()
                })?;
            for ignored in String::from_utf8_lossy(&out.stdout).lines() {
                changes.remove(ignored);
            }
        }
        // Index lines for `update-index --index-info`: `<mode> <oid>\t<path>`, mode 0
        // removing the path.
        let mut info = Vec::new();
        for (path, change) in &changes {
            match change {
                Change::File { data, executable, .. } => {
                    let oid = git_stdin(&self.root, &["hash-object", "-w", "--stdin"], &[], data)?;
                    let mode = if *executable { "100755" } else { "100644" };
                    info.extend(format!("{mode} {}\t{path}\n", String::from_utf8(oid)?.trim()).bytes());
                }
                Change::Gone if self.tree.get(path).is_some_and(|e| !e.dir) => {
                    info.extend(format!("0 {}\t{path}\n", "0".repeat(40)).bytes());
                }
                // A removed directory went file by file; empty ones git cannot hold.
                Change::Gone | Change::Dir { .. } => {}
            }
        }
        let index = tempfile_path(&self.root)?;
        let env = [("GIT_INDEX_FILE", index.as_os_str())];
        let result = (|| {
            git_stdin(&self.root, &["read-tree", &self.base], &env, b"")?;
            git_stdin(&self.root, &["update-index", "--index-info"], &env, &info)?;
            let tree = String::from_utf8(git_stdin(&self.root, &["write-tree"], &env, b"")?)?.trim().to_string();
            let base_tree = String::from_utf8(git(&self.root, &["rev-parse", &format!("{}^{{tree}}", self.base)])?)?;
            if tree == base_tree.trim() {
                return Ok(None);
            }
            let commit = git_stdin(
                &self.root,
                &["-c", "user.name=Corrode", "-c", "user.email=corrode@localhost", "commit-tree", &tree, "-p", &self.base, "-F", "-"],
                &[],
                message.as_bytes(),
            )?;
            let commit = String::from_utf8(commit)?.trim().to_string();
            git(&self.root, &["update-ref", &format!("refs/corrode/{name}"), &commit])?;
            Ok(Some(commit))
        })();
        let _ = std::fs::remove_file(&index);
        result
    }
}

/// A turn's view, mounted where its sandboxed commands find it: the view's FUSE mount
/// under the user's runtime dir, and a target dir of its own for cargo (see
/// [`crate::sandbox::Sandbox::with_view`]). `finish` commits it and takes it down.
#[cfg(feature = "fuse")]
pub struct TurnView {
    pub vfs: Arc<ViewVfs>,
    pub mountpoint: PathBuf,
    pub target: PathBuf,
    /// The repository's common git dir, which the sandbox binds read-only at
    /// [`crate::sandbox::SANDBOX_GIT`] for the view's `.git` link to point into.
    pub git_dir: PathBuf,
    /// The view's linked-worktree admin dir, `<git dir>/worktrees/corrode-<plan>`.
    admin: PathBuf,
    name: String,
    mount: fuse3::raw::MountHandle,
}

#[cfg(feature = "fuse")]
impl TurnView {
    /// A view of the repository's `HEAD`, mounted.
    pub async fn start(repo: &std::path::Path, plan_id: &str) -> anyhow::Result<Self> {
        let name = crate::sandbox::safe_name(plan_id);
        let vfs = Arc::new(tokio::task::block_in_place(|| ViewVfs::open(repo, "HEAD"))?);
        let mountpoint = crate::sandbox::runtime_dir("corrode-views").join(&name);
        crate::sandbox::private_dir(&mountpoint)?;
        // Build output is gigabytes: on disk, not under the (RAM-backed) runtime dir.
        let target = crate::sandbox::home()
            .map(|h| h.join(".cache"))
            .unwrap_or_else(std::env::temp_dir)
            .join("corrode/targets")
            .join(&name);
        std::fs::create_dir_all(&target)?;
        let (git_dir, admin) = tokio::task::block_in_place(|| link_worktree(repo, &vfs, &name))?;
        let mount = match crate::fuse::mount_handle(vfs.clone(), &mountpoint).await {
            Ok(m) => m,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&admin);
                return Err(e);
            }
        };
        Ok(Self { vfs, mountpoint, target, git_dir, admin, name, mount })
    }

    /// Commit the view to `refs/corrode/<plan>` (`None`: nothing changed), unmount it
    /// and remove its target dir.
    pub async fn finish(self, message: &str) -> anyhow::Result<Option<String>> {
        let committed = tokio::task::block_in_place(|| self.vfs.commit(&self.name, message));
        if let Err(e) = self.mount.unmount().await {
            eprintln!("view: unmounting {}: {e}", self.mountpoint.display());
        }
        let (mountpoint, target, admin) = (self.mountpoint, self.target, self.admin);
        tokio::task::spawn_blocking(move || {
            let _ = std::fs::remove_dir(&mountpoint);
            let _ = std::fs::remove_dir_all(&target);
            let _ = std::fs::remove_dir_all(&admin);
        });
        Ok(committed?.map(|c| format!("refs/corrode/{} {c}", self.name)))
    }
}

/// Make `vfs` a linked worktree of `repo` to stock git (docs/vfs-branches-and-builds.md
/// §8): an admin dir under `<git dir>/worktrees/` -- `HEAD` detached at the view's base,
/// `commondir`, `gitdir`, an index of the base tree, and `locked`, so a
/// `git worktree prune` on the host leaves a view that exists only inside a sandbox
/// alone -- and the view's own `.git` file pointing at it through the sandbox's
/// read-only bind of the git dir. `git status`, `diff` and `log` then work inside a
/// turn; committing does not (the git dir is read-only there), which is the point: the
/// turn's commit is the daemon's, at the end.
///
/// The index carries no stat data, so `git status` hashes every file to compare it --
/// correct, and cheap at CAE's size; it cannot save a refreshed index (read-only).
/// Returns `(git dir, admin dir)`.
#[cfg_attr(not(feature = "fuse"), allow(dead_code))]
fn link_worktree(repo: &std::path::Path, vfs: &ViewVfs, name: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    let common = git(repo, &["rev-parse", "--path-format=absolute", "--git-common-dir"])?;
    let git_dir = PathBuf::from(String::from_utf8(common)?.trim());
    let wt = format!("corrode-{name}");
    let admin = git_dir.join("worktrees").join(&wt);
    std::fs::create_dir_all(&admin)?;
    let repo_abs = std::fs::canonicalize(repo)?;
    std::fs::write(admin.join("HEAD"), format!("{}\n", vfs.base))?;
    std::fs::write(admin.join("commondir"), "../..\n")?;
    // Where the worktree's `.git` sits as git inside the sandbox sees it: the view is
    // mounted at the repository's own path.
    std::fs::write(admin.join("gitdir"), format!("{}/.git\n", repo_abs.display()))?;
    std::fs::write(admin.join("locked"), "a Corrode turn's VFS view; removed when the turn ends\n")?;
    let index = admin.join("index");
    git_stdin(repo, &["read-tree", &vfs.base], &[("GIT_INDEX_FILE", index.as_os_str())], b"")?;
    vfs.set_git_link(format!("gitdir: {}/worktrees/{wt}\n", crate::sandbox::SANDBOX_GIT));
    Ok((git_dir, admin))
}

/// Without the `fuse` feature a view cannot be mounted, so a turn cannot work in one.
#[cfg(not(feature = "fuse"))]
pub struct TurnView {
    pub vfs: Arc<ViewVfs>,
    pub mountpoint: PathBuf,
    pub target: PathBuf,
    pub git_dir: PathBuf,
}

#[cfg(not(feature = "fuse"))]
impl TurnView {
    pub async fn start(_repo: &std::path::Path, _plan_id: &str) -> anyhow::Result<Self> {
        anyhow::bail!("CORRODE_VIEWS needs a daemon built with --features fuse")
    }

    pub async fn finish(self, _message: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// A fresh path for a throwaway index inside the repository's git dir, where git
/// itself keeps such files.
fn tempfile_path(root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let gitdir = String::from_utf8(git(root, &["rev-parse", "--absolute-git-dir"])?)?;
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(PathBuf::from(gitdir.trim()).join(format!("corrode-index-{}-{n}", std::process::id())))
}

#[async_trait]
impl Vfs for ViewVfs {
    async fn list(&self, dir: &str) -> anyhow::Result<Vec<FileNodeView>> {
        let dir = clean(dir)?;
        let changes = self.changes.lock().unwrap();
        if !self.is_dir(&dir, &changes) {
            return Err(io_err(NotADirectory, format!("not a directory: {dir}")));
        }
        Ok(self
            .names_in(&dir, &changes)
            .into_iter()
            .filter_map(|n| {
                let p = if dir.is_empty() { n } else { format!("{dir}/{n}") };
                self.view_of(&p, &changes)
            })
            .collect())
    }

    async fn stat(&self, path: &str) -> anyhow::Result<FileNodeView> {
        let path = clean(path)?;
        let changes = self.changes.lock().unwrap();
        self.view_of(&path, &changes).ok_or_else(|| io_err(NotFound, format!("no such path: {path}")))
    }

    async fn read(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        let path = clean(path)?;
        if path == ".git" {
            return self.git_link.get().map(|l| l.clone().into_bytes()).ok_or_else(|| io_err(NotFound, "no .git"));
        }
        let oid = {
            let changes = self.changes.lock().unwrap();
            match changes.get(&path) {
                Some(Change::File { data, .. }) => return Ok(data.to_vec()),
                Some(Change::Dir { .. }) => return Err(io_err(IsADirectory, format!("is a directory: {path}"))),
                Some(_) => return Err(io_err(NotFound, format!("no such file: {path}"))),
                None => match self.tree.get(&path) {
                    Some(e) if !e.dir => e.oid.clone(),
                    Some(_) => return Err(io_err(IsADirectory, format!("is a directory: {path}"))),
                    None => return Err(io_err(NotFound, format!("no such file: {path}"))),
                },
            }
        };
        // A base file is a git call: off the executor, since views serve FUSE.
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || git(&root, &["cat-file", "blob", &oid])).await?
    }

    async fn write(&self, path: &str, contents: &[u8]) -> anyhow::Result<()> {
        let path = writable(path)?;
        if path.is_empty() {
            anyhow::bail!("cannot write the view's root");
        }
        let mut changes = self.changes.lock().unwrap();
        let executable = match self.view_of(&path, &changes) {
            Some(v) if v.is_dir => return Err(io_err(IsADirectory, format!("is a directory: {path}"))),
            Some(v) => v.executable,
            None => false,
        };
        self.ensure_parents(&path, &mut changes)?;
        changes.insert(
            path,
            Change::File { data: contents.into(), executable, mtime_ns: now_ns() },
        );
        Ok(())
    }

    async fn tracked_files(&self) -> anyhow::Result<Vec<String>> {
        let changes = self.changes.lock().unwrap();
        let mut files: BTreeSet<String> = self
            .tree
            .iter()
            .filter(|(_, e)| !e.dir)
            .map(|(p, _)| p.clone())
            .collect();
        for (p, c) in changes.iter() {
            match c {
                Change::File { .. } => files.insert(p.clone()),
                _ => files.remove(p),
            };
        }
        Ok(files.into_iter().collect())
    }

    async fn remove(&self, path: &str) -> anyhow::Result<()> {
        self.remove_now(path)
    }

    async fn mkdir(&self, path: &str) -> anyhow::Result<()> {
        self.mkdir_now(path)
    }

    // These two read file contents, which for a base file is a git call: off the executor.
    async fn rename(&self, from: &str, to: &str) -> anyhow::Result<()> {
        tokio::task::block_in_place(|| self.rename_now(from, to))
    }

    async fn set_executable(&self, path: &str, executable: bool) -> anyhow::Result<()> {
        tokio::task::block_in_place(|| self.set_executable_now(path, executable))
    }

    async fn touch(&self, path: &str) -> anyhow::Result<()> {
        let path = writable(path)?;
        tokio::task::block_in_place(|| {
            let changes = self.changes.lock().unwrap();
            match self.view_of(&path, &changes) {
                None => return Err(io_err(NotFound, format!("no such path: {path}"))),
                Some(v) if v.is_dir => {
                    drop(changes);
                    self.changes.lock().unwrap().insert(path.clone(), Change::Dir { mtime_ns: now_ns() });
                    return Ok(());
                }
                Some(v) => {
                    drop(changes);
                    let data = self.read_now(&path)?;
                    self.changes
                        .lock()
                        .unwrap()
                        .insert(path.clone(), Change::File { data, executable: v.executable, mtime_ns: now_ns() });
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A repository with one commit: `README.md`, `src/lib.rs`, an executable script.
    pub(crate) fn repo(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("corrode-view-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("README.md"), "hello\n").unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn f() {}\n").unwrap();
        std::fs::write(dir.join("run.sh"), "#!/bin/sh\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.join("run.sh"), std::fs::Permissions::from_mode(0o755)).unwrap();
        for args in [
            &["init", "-q"][..],
            &["add", "-A"],
            &["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base"],
        ] {
            git(&dir, args).unwrap();
        }
        dir
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_view_reads_the_base_and_keeps_its_changes_to_itself() {
        let dir = repo("isolated");
        let a = ViewVfs::open(&dir, "HEAD").unwrap();
        let b = ViewVfs::open(&dir, "HEAD").unwrap();
        assert_eq!(a.read("src/lib.rs").await.unwrap(), b"pub fn f() {}\n");
        assert!(a.stat("run.sh").await.unwrap().executable);
        let names = |v: Vec<FileNodeView>| v.into_iter().map(|e| e.path).collect::<Vec<_>>();
        assert_eq!(names(a.list("").await.unwrap()), ["README.md", "run.sh", "src"]);

        // An unchanged file reports the base commit's time, the same in every view.
        let before = a.stat("src/lib.rs").await.unwrap().mtime_ns;
        assert_eq!(before, b.stat("src/lib.rs").await.unwrap().mtime_ns);
        a.write("src/lib.rs", b"pub fn g() {}\n").await.unwrap();
        a.write("src/new/mod.rs", b"// new\n").await.unwrap();
        // ...and a write is newer than the base, so a build after it rebuilds.
        assert!(a.stat("src/lib.rs").await.unwrap().mtime_ns > before);
        assert_eq!(names(a.list("src").await.unwrap()), ["src/lib.rs", "src/new"]);
        assert_eq!(a.tracked_files().await.unwrap(), ["README.md", "run.sh", "src/lib.rs", "src/new/mod.rs"]);

        // The other view and the working tree never see it.
        assert_eq!(b.read("src/lib.rs").await.unwrap(), b"pub fn f() {}\n");
        assert!(b.stat("src/new/mod.rs").await.is_err());
        assert_eq!(std::fs::read(dir.join("src/lib.rs")).unwrap(), b"pub fn f() {}\n");
        assert!(a.stat(".git/config").await.is_err(), "a view has no .git");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rename_and_remove_move_whole_trees() {
        let dir = repo("rename");
        let v = ViewVfs::open(&dir, "HEAD").unwrap();
        v.write("src/b.rs", b"b\n").await.unwrap();
        v.rename("src", "lib").await.unwrap();
        assert!(v.stat("src").await.is_err());
        assert_eq!(v.read("lib/lib.rs").await.unwrap(), b"pub fn f() {}\n");
        assert_eq!(v.read("lib/b.rs").await.unwrap(), b"b\n");
        assert!(v.remove("lib").await.is_err(), "not empty");
        v.remove("lib/b.rs").await.unwrap();
        v.remove("lib/lib.rs").await.unwrap();
        v.remove("lib").await.unwrap();
        // An editor's save: write a temp file, then move it over the original.
        v.write("README.md.tmp", b"saved\n").await.unwrap();
        v.rename("README.md.tmp", "README.md").await.unwrap();
        assert_eq!(v.read("README.md").await.unwrap(), b"saved\n");
        assert_eq!(v.tracked_files().await.unwrap(), ["README.md", "run.sh"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The whole loop: a turn's view, mounted, is a git worktree to a sandboxed shell --
    /// `status` and `diff` see the turn's change, `commit` is refused -- and finishing the
    /// turn commits it, unmounts, and leaves no worktree behind.
    #[cfg(feature = "fuse")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_mounted_view_is_a_git_worktree_inside_the_sandbox() {
        let dir = repo("worktree");
        let plan = format!("view-test-{}", std::process::id());
        let view = match TurnView::start(&dir, &plan).await {
            Ok(v) => v,
            Err(e) => return eprintln!("skipped: cannot mount a view here ({e})"),
        };
        view.vfs.write("src/lib.rs", b"pub fn g() {}\n").await.unwrap();
        let sb = crate::sandbox::Sandbox::from_env().with_view(view.mountpoint.clone(), view.target.clone(), view.git_dir.clone());
        let run = |script: &str| {
            let (prog, args) = sb.wrap(&std::fs::canonicalize(&dir).unwrap(), &["sh", "-c", script]);
            std::process::Command::new(prog).args(args).output().unwrap()
        };
        if !sb.enabled() || !run("true").status.success() {
            view.finish("skip").await.ok();
            return eprintln!("skipped: bwrap cannot run here");
        }
        let status = run("git status --porcelain");
        assert_eq!(String::from_utf8_lossy(&status.stdout), " M src/lib.rs\n", "{}", String::from_utf8_lossy(&status.stderr));
        let diff = String::from_utf8_lossy(&run("git diff").stdout).into_owned();
        assert!(diff.contains("-pub fn f() {}") && diff.contains("+pub fn g() {}"), "{diff}");
        assert!(!run("git -c user.name=t -c user.email=t@t commit -qam x").status.success(), "commit must be refused");
        // Seen from the host, the view is a locked worktree, gone once the turn ends.
        let list = String::from_utf8(git(&dir, &["worktree", "list", "--porcelain"]).unwrap()).unwrap();
        assert!(list.contains("locked"), "{list}");
        let commit = view.finish("turn").await.unwrap().expect("a commit");
        assert!(commit.starts_with(&format!("refs/corrode/{plan}")), "{commit}");
        let list = String::from_utf8(git(&dir, &["worktree", "list"]).unwrap()).unwrap();
        assert_eq!(list.lines().count(), 1, "{list}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn commit_stores_the_changes_under_a_corrode_ref_and_leaves_branches_alone() {
        let dir = repo("commit");
        let head = String::from_utf8(git(&dir, &["rev-parse", "HEAD"]).unwrap()).unwrap();
        let v = ViewVfs::open(&dir, "HEAD").unwrap();
        assert_eq!(v.commit("t", "nothing").unwrap(), None, "an unchanged view makes no commit");
        v.write("src/lib.rs", b"pub fn g() {}\n").await.unwrap();
        v.remove("README.md").await.unwrap();
        v.set_executable("src/lib.rs", true).await.unwrap();
        let c = v.commit("plan-1", "the turn").unwrap().expect("a commit");
        let show = |args: &[&str]| String::from_utf8(git(&dir, args).unwrap()).unwrap();
        assert_eq!(show(&["rev-parse", "refs/corrode/plan-1"]).trim(), c);
        assert_eq!(show(&["show", "refs/corrode/plan-1:src/lib.rs"]), "pub fn g() {}\n");
        assert!(show(&["ls-tree", "refs/corrode/plan-1"]).lines().all(|l| !l.ends_with("README.md")));
        assert!(show(&["ls-tree", "refs/corrode/plan-1", "src/lib.rs"]).starts_with("100755"));
        assert_eq!(show(&["rev-parse", "refs/corrode/plan-1^"]), head, "on top of the base");
        assert_eq!(show(&["rev-parse", "HEAD"]), head, "the branch did not move");
        assert_eq!(show(&["status", "--porcelain"]), "", "the working tree and index are untouched");
        std::fs::remove_dir_all(&dir).ok();
    }
}
