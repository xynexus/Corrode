# Corrode: per-branch VFS, overlays, and build reuse

**Design note.** How agents should see the repository when several of them
work at once, and what that implies for the graph, the VFS, and their builds.

- **Status:** §1 is **implemented** (`Vfs::tracked_files` + the two property
  filters). §2–§6 were the exploration; **§8 records the decision** (2026-11-03)
  and the measurement it rests on.
- **Date:** 2026-08-29
- **Scope:** VFS presentation, per-branch isolation, build artefact handling
- **Guiding principle:** *an agent must see a standard filesystem.* Corrode
  exists to make the agent's life smooth. Anything that forces agents (or the
  tools they run) through a bespoke API is the wrong shape, however elegant.

Measurements below were taken on nix2 (gfx1103, 30 GB UMA, XFS root) against
this repo and `~/hipfire`. Where something was reasoned rather than measured it
says so.

---

## 1. What started it: `search_files` returns garbage

A swarm turn looped and died. A subagent searched for `overview`, and the tool
returned tokenizer binaries (`needle.model`, `needle.vocab`), minified
`webui/dist/xterm.js`, vendored `third_party/minicpm` data files, and helix test
fixtures. The subagent re-ran the same search, got the same wall of noise, and
burned minutes of GPU on it before the turn errored out.

`tools.rs` walks the VFS, skips `.git` and `target/`, skips files over 1 MB, and
`String::from_utf8_lossy`-decodes everything else. So binaries match as text,
and 0.28 MB of minified JS passes the size filter as a single enormous "line".

### The corpus is the fix, not a blacklist

Hardcoded exclusions were rejected: they are endless and always one new vendored
directory behind. The question is *what is the searchable corpus*, and git
already answers it.

| corpus | hits for `overview` |
|---|---|
| naive walk (current) | 211, of which ~120 junk |
| `rg` with default settings | 211 — **defaults do not help** |
| `rg` with explicit `-g '!third_party/**'` etc. | 1 |
| **git-tracked regular files** | **5** |

`rg`'s defaults fail because `third_party/` is git *submodules* — tracked
content that gitignore has no reason to skip. 208 of the 211 hits came from
there.

`git ls-files` excludes submodule *contents* automatically (a submodule is one
gitlink entry), and excludes `webui/dist/` because it is gitignored. 165 regular
tracked files here, versus a walk that scans up to 4000. No hardcoded paths, and
it self-maintains.

**Trap:** submodule gitlinks are directory paths. Handing them to a searcher
makes it recurse straight back into the submodule. The corpus must be filtered
to regular files — this cost one wrong measurement (212 hits) that looked like
the idea had failed.

The five survivors are all vendored-but-committed: `xterm.js`, `xterm.css`,
`needle.model`, `needle.vocab`, and a helix-skills doc. Two fall to *property*
tests rather than path lists — a NUL sniff for binaries, a maximum line length
for minified files. The helix-skills doc is arguably a legitimate hit.

**Correction, measured after implementing (2026-09-01).** Only *one* of those two
falls to a property test. `needle.model` is binary and is caught; `needle.vocab` is
not — it is a token table in plain text, no NUL, longest line 22 characters, and it
matches "overview" because it literally contains the token `▁overview`. It is
indistinguishable from source by any property, and only a path rule would exclude
it, which is what this section rejects. It is left as residual noise: one file,
against ~120 junk hits before. Post-fix on this repo the corpus is 3824 files → 170,
the filters skip 22 binary + 7 minified, and 6 files remain.

### Where it belongs

On the `Vfs` trait. It currently has `list`, `stat`, `read`, `write` and no
notion of "the set of files I track".

- `PassthroughVfs` answers via `git ls-files` + regular-file filter — today.
- A graph-backed VFS answers from its file nodes — same call site.
- `search_files` then holds **no policy at all**; it asks the VFS what exists.

This also removes a real hazard: if search and read derive from one VFS-owned
definition of what exists, they cannot disagree. A subprocess `rg` reading the
filesystem directly *would* disagree once the VFS is graph-backed.

---

## 2. Per-branch agents

The larger idea: give each agent its own branch and its own view, so several can
work simultaneously without colliding. Attractive on a large tree (the Linux
kernel was the example) where N× checkout is untenable.

### What that requires of the graph

The load-bearing property is **structural sharing**, not branch modelling. Git's
trick is that two branches differing in one file share every other blob. A graph
that maps `branch → nodes → files` literally stores N branches × M files — ~80k
files per branch on a kernel-sized tree, ten agents, 800k nodes that are ~99%
duplicates.

So a "git-compatible graph" means **content addressing**: nodes keyed on blob
OID, not on `(branch, path)`. That is the hard requirement.

The split that follows: **git owns content and history; the graph owns semantics
keyed by git's OIDs.** A file unchanged across ten branches is indexed once.
Branch → commit → tree resolution stays in git, which is very good at it.
Cross-branch and historical search becomes: query the graph by blob OID, map
back to `(branch, path)` through git.

Scope caveat: index the semantics of what agents actually touch, not the whole
tree and certainly not the whole history. Git answers "what exists"; the graph
answers "what do we know about this blob."

---

## 3. Presentation: FUSE, and a correction

An earlier claim in this discussion — that a virtual filesystem cannot serve
standard build tools — was **wrong**, and the repo already refutes it:

- `vfs.rs:17`: "a FUSE mount awaits these handlers per syscall"
- `Vfs::stat` exists specifically to serve FUSE `getattr`/`lookup` without
  scanning the parent directory
- `fuse3` is already an optional dependency, commented: *"FUSE mount of the Vfs,
  so git and subagent shells see the graph<->git projection as a real
  filesystem"*

`fusermount3` is installed; kernel 6.17 has `fuse`. A mounted VFS serves
`cargo`, `make` and `rg` because they issue ordinary syscalls. This is the
guiding principle in action, and it is already the project's stated direction.

The real constraints are performance and POSIX detail, not capability:

- Builds are metadata storms; every `stat` is a userspace round-trip. Tunable
  with attribute/entry timeouts, `readdirplus`, page cache, `writeback_cache`.
- Cargo leans on advisory file locks (`.cargo-lock`), atomic `rename`, stable
  inode numbers, and `mmap` (rustc). Each is a place a naive implementation
  breaks a build confusingly.

---

## 4. Overlays: stack them, don't hand-roll COW

The proposal — mount an overlay over the VFS — is better than implementing
copy-on-write inside the FUSE daemon, for three concrete reasons:

1. **Writes bypass FUSE after copy-up.** Once overlayfs copies a file up, all
   further reads and writes go to the upper filesystem at native speed. The
   compiler's hot path never re-enters userspace. In-FUSE COW would route every
   write *and every later read* of that file through the daemon.
2. **Zero code.** The kernel implements copy-up, whiteouts, and merge semantics.
3. **The page cache then works normally on the upper layer**, which is where the
   "buffer writes in RAM, drain lazily" behaviour comes from — for free, applied
   to exactly the churny files, with kernel-managed eviction.

**The upperdir is the agent's diff** — a materialised changeset you can commit
to that agent's branch with no separate tracking.

### Requirements this puts on the design

**Stable mtimes — make or break.** Cargo decides what to rebuild by comparing
source mtimes against `.fingerprint/`. Overlayfs preserves the lower layer's
mtime for anything not copied up, which is what we want. The danger is upstream:
**git checkout sets mtime to checkout time.** If a branch view materialises
source with "now" mtimes, everything looks newer than the base build and cargo
rebuilds the entire tree — the scheme delivers nothing.

So the VFS must report mtimes derived from **content or commit**, not from
checkout or access time. A file unchanged between two branches must present the
same mtime in both. Natural for a graph/git-backed VFS (derive from the last
commit touching the blob, or fix per blob OID); a naive passthrough cannot fake
it.

**Identical mount paths.** Cargo's metadata hashes and rustc's embedded paths
include the workspace path. Agents mounted at the *same* path in separate mount
namespaces share fingerprints and can reuse a common lower layer; agents at
different paths each rebuild from scratch and produce non-interchangeable
artefacts. Per-agent mount namespaces (bubblewrap, already in `sandbox.rs`) make
identical paths trivial.

**Two overlays, not one.** Source and `target/` have opposite lifecycles. The
source upper is precious — it becomes a commit. The target upper is disposable.
Separating them means build output can be deleted without touching the diff, and
the source upper stays a clean patch instead of being buried under rebuilt
`.rlib` files.

**Incremental off for agent builds.** `CARGO_INCREMENTAL=0`. `incremental/` is
5.4 GB here, nondeterministic, and generates copy-up churn for state that exists
to speed up repeated *local* rebuilds — not the agent's pattern.

### Overlay mechanics, answered

**Copy-up is whole-file, not delta.** Writing one byte to a lower file copies the
entire file up first. Three things rescue it here:

- **This root filesystem is XFS with reflink** (verified: `cp --reflink=always`
  succeeds). Copy-up can clone extents rather than duplicate bytes, so only
  modified blocks diverge. That delivers "store only the difference" from the
  filesystem, not from overlayfs.
- **New files never copy up** — they are created directly in the upper layer.
  Decisive for builds, since rustc and cargo overwhelmingly create hash-named
  artefacts rather than modifying existing ones.
- `metacopy=on` makes metadata-only changes copy no data.

**There is no merge-down operation.** The upper is a directory tree; merging it
into the lower means doing it yourself and interpreting overlay bookkeeping:

- **whiteouts** — character devices 0:0 meaning "deleted"; a naive `rsync`
  copies the char device instead of deleting
- **opaque directories** — an xattr meaning "replaces lower entirely"
- **metacopy files** — metadata-only upper entries redirecting to lower data; a
  naive copy produces a broken, effectively empty file
- these live in `trusted.overlay.*`, which needs `CAP_SYS_ADMIN` to read;
  userns-mounted overlays on newer kernels use `user.overlay.*`, so tooling must
  handle both

**For the source overlay, merging down is the wrong operation anyway.** The
merge target is a git commit: read the upper as a changeset, map whiteouts to
deletions, commit to the agent's branch, and let git do the merging — with
conflict resolution, review, and history. The target overlay is never merged;
it is discarded, and a fresh full build is periodically promoted as the new
lower.

### Build reuse across agents

Stack multiple overlays over one complete base build. Each agent reads the base
(so cargo sees existing outputs and does not regenerate the tree) and writes only
its own delta. An agent touching one crate materialises that crate plus its
dependents rather than 15 GB.

This works only with the stable-mtime and identical-path requirements above, and
it wants agents branching from a common base that has been built once. Agents
branching from scattered commits degrade to full rebuilds.

---

## 5. Build artefacts: capture metadata, not bytes

`target/` for this repo is 15 GB (hipfire's is 35 GB), and it decomposes as:

| | size |
|---|---|
| `deps/` (regenerable intermediates) | 8.6 GB |
| `incremental/` (per-machine, nondeterministic scratch) | 5.4 GB |
| `build/` | 297 MB |
| **final binaries** | **429 MB — under 3%** |

So "capture the build artefacts" is mostly capturing compiler scratch.

**Diffing artefacts across runs does not work as stated.** Rust builds are not
bit-reproducible by default — rustc embeds absolute paths and metadata hashes,
and incremental compilation is deliberately nondeterministic in output layout. A
diff between two builds of identical source would be dominated by noise. Making
it meaningful is its own project (`--remap-path-prefix`, pinned toolchain,
`SOURCE_DATE_EPOCH`, non-incremental) — and once done, a content **hash** per
artefact carries the same information, so store hashes rather than bytes.

**Worth capturing:** a build-result node in the existing provenance graph —
branch/commit, toolchain version, profile, features, exit status, test results,
and a content hash per final binary. A few hundred bytes, queryable, extending
the current `plan → task → code` lineage in the same shape. Optionally the final
binaries themselves, the only bytes with independent value (bisect without a
rebuild). Content-addressing them means ten agents producing an identical binary
store it once.

**For build reuse specifically, the tool is a cache, not a capture** — `sccache`
keyed by input hash with per-agent target dirs. Note a single *shared* `target/`
does not work for concurrency: cargo locks the target directory, so agents
serialise, defeating the point of separate branches. The overlay approach above
avoids this, since each agent has its own upper.

### RAM-backing writes: the kernel already does it

Buffering writes in RAM and draining lazily is exactly what the page cache does:
`dirty_ratio=20`, `dirty_background_ratio=10`, `dirty_expire_centisecs=3000`.
Writes land in RAM, background writeback starts at 10% dirty, throttles at 20%,
and pages older than 30 s flush — with eviction under memory pressure that a
userspace buffer would have to reimplement while competing for the same RAM.

The idea also inverts on inspection: what motivates it (churny build output, 15–35 GB)
cannot fit in RAM on a 30 GB host, and what does fit (source edits, kilobytes)
gains nothing. `/dev/shm` is 16 GB here — viable for edits, not for a build.

Where a RAM layer genuinely wins is branches that will be discarded, and `tmpfs`
delivers that with kernel-managed eviction and no code.

---

## 6. Alternatives weighed

**git worktrees + existing bwrap sandbox + per-agent `CARGO_TARGET_DIR`.** The
cheap path that works today: real directories, every tool works unmodified,
isolation immediately. Costs N× checkout on disk (several GB each at kernel
scale) and gives no cross-branch querying. Days of work, not a project. This is
the pragmatic near-term option if the goal is only "agents don't collide".

**Docker / OCI containers.** Nothing is installed on this host. Docker's storage
driver *is* overlayfs, so it would replace the overlay *stacking*, not the VFS.
It brings real value — productionised layer management and GC, toolchain pinning
(which feeds the build-provenance story), a base build as a first-class
distributable layer, and stable mtimes in image layers. But image layers are not
git-branch projections, so it buys none of the per-branch-without-N×-disk or
cross-branch-query value that motivates the VFS.

Costs specific to Corrode: Docker classically wants a root daemon, whereas the
existing sandbox is deliberately *unprivileged* bwrap; GPU passthrough is doable
(`/dev/kfd`, `/dev/dri/renderD128` present) but fiddly and interacts with
hipfire's GPU flock; container startup is heavy for a swarm spawning many
subagents; and it adds a second isolation mechanism. If containers are wanted,
**podman rootless** preserves the unprivileged property.

---

## 7. Where this leaves things

**~~Ready to do now~~ Done (2026-09-01):** `tracked_files()` is on the `Vfs` trait
with the git-backed implementation, `search_files` is rewired onto it, and the binary
and long-line property filters are in. Root fix, no blacklist, and the first step of
the branch-aware design stands — `ls-files` becomes `ls-tree <rev>` later, same call
site, and a graph-backed VFS answers from its file nodes without `search_files`
changing at all.

**Decided in principle:** the graph should be git-compatible, keyed by content
(blob OID), with git owning content and history.

**Open:** whether the VFS itself becomes branch-scoped, and on what timeline.
The guiding principle argues for FUSE presentation regardless, since agents and
their tools must see a normal filesystem.

**Unvalidated, and worth measuring before committing:**

- build-over-FUSE cost on a real tree — time `cargo build` through the mount
  versus native, with attribute caching on. If metadata latency dominates, the
  answer may be FUSE for read/search views plus a materialised worktree for the
  build-and-test loop.
- the overlay mtime behaviour above could not be tested here: the tool sandbox
  blocks nested user namespaces and bubblewrap 0.9.0 predates `--overlay`.
- whether cargo actually reuses a base build through a stacked overlay, which is
  the whole premise of §4's reuse section.

---

## 8. Decision (2026-11-03)

### Where agent work lives

**In-flight work belongs to the VFS; project history only receives it on
promotion.**

- A running turn's changes live in a **VFS view**: a base tree OID plus a
  changeset (`path -> new blob OID | deleted`). The file tools already write
  through the VFS, so it already sees every edit. Views isolate concurrent
  agents from each other and from the human's working tree.
- The changeset's content is stored as **git objects** (blobs, trees) in the
  repo's own object store — no working tree, no branch. That gives every
  version an OID, which the graph is keyed on anyway (§2), and deduplicates
  across agents for free.
- At turn end the daemon commits the view to a private ref,
  `refs/corrode/<plan_id>`. **Promotion** — merge, cherry-pick, PR — is what
  turns it into project history, by a human or after review passes.

Keeping work *only* in the VFS was rejected: Corrode would become a second VCS
(history, merge, blame), and none of the human's tools — `git log`, PRs, CI —
would see it.

### Views live in the VFS; build output does not

§4 preferred kernel overlayfs to COW inside the FUSE daemon. That argument is
about write *volume*, and it splits by layer:

- **Source: the VFS owns the overlay.** A view is a tiny structure (base OID +
  changeset). Owning it means search, read and the graph see exactly what the
  agent sees; the diff is already in hand at commit time; and the VFS can
  present **content-derived mtimes**, which §4 says build reuse depends on and
  a checkout cannot provide.
- **`target/`: not the VFS.** Gigabytes of churn, no semantic value. Each view
  gets a native per-agent target directory, optionally over a shared base
  build (§4's second overlay).

### Views as git worktrees

A linked git worktree is just a directory whose `.git` is a *file*
(`gitdir: <repo>/.git/worktrees/<name>`) plus an admin directory holding
`HEAD`, `commondir`, `gitdir` and `index`. A VFS view can synthesize both, and
then every view **is** a worktree to stock git:

- `git status`, `git diff`, `git log` inside a view work unmodified — the
  guiding principle (a standard filesystem) extended to version control.
- The view's `HEAD` is `refs/corrode/<plan_id>`; the turn-end commit is an
  ordinary commit in that worktree, not a hand-built tree.
- `git worktree list` in the human's checkout shows every live agent view; a
  view mounted on the host (say `~/corrode-views/<plan_id>`) can be inspected
  or `cd`'d into like any worktree.

What it requires:

- **A valid index stat cache.** `git status` compares the index's
  `(mtime, size, ino, dev)` with `lstat`; a mismatch forces a rehash of the
  file (correct, slow at scale). Content-derived mtimes and stable inode
  numbers from the VFS keep the cache valid; better still, the VFS writes the
  view's index from the base tree with matching stat data.
- **`git worktree prune` must not reap views.** A view mounted only inside an
  agent's mount namespace does not exist on the host, so prune would delete its
  admin dir. Each view's admin dir carries a `locked` file
  (`git worktree lock`), which prune honours.
- **Shared objects stay read-only in the sandbox** (as `.git` is today). Commits
  are made by the daemon, outside the sandbox; inside, `git status` may fail to
  rewrite the index, which git tolerates.
- Views mounted at the *same* path in separate namespaces (for build reuse, §4)
  give their admin dirs identical back-pointers; `git worktree list` then shows
  one path several times. Cosmetic, but a host-side mount per view avoids it.

### Order of work

1. Measure builds through the FUSE mount (below) — gates everything after.
2. `ViewVfs`: base OID + changeset over the git object store; content-derived
   mtimes; `write_file`/read/search go through it.
3. Mount each view at the repo path inside its sandbox; bind `target/` to a
   per-view native directory. Needs FUSE create/mkdir/rename/unlink/setattr.
4. Synthesize the worktree `.git` file + locked admin dir; commit at turn end
   to `refs/corrode/<plan_id>`; promote by merge or PR.

**Done (2026-11-03): steps 2, 3 and the commit of step 4**, behind
`CORRODE_VIEWS` (off by default; needs `--features fuse` and the sandbox).
`view.rs` is the view; `fuse.rs` takes create/mkdir/unlink/rmdir/rename/
truncate/chmod; a turn mounts its view under `$XDG_RUNTIME_DIR/corrode-views`,
binds it at the repo path, gives cargo a fresh target dir under
`~/.cache/corrode/targets`, and commits to `refs/corrode/<plan>` at the end,
leaving out new files `.gitignore` excludes. A live wordfreq turn ran this way
in 410 s: tests passed in the view, the working tree and branch were untouched,
and the commit checked out and tested clean on its own.

Decided along the way:
- **Unchanged files report the base commit's time**, a written file its write
  time. Pure content-derived times were rejected: a file reverted to an older
  blob would report an older time than the build that used the newer one, and
  cargo would keep that build.
- **A target dir per turn, not shared.** Cargo judges a path dependency fresh
  by mtime alone, so two views building into one target dir would each get
  the other's artifacts. A turn's first build is therefore cold (CAE: 64 s).
  Reuse needs a seeded copy whose mtimes the view can vouch for -- §4's
  problem, still open.
- **fuse3 0.10**, not 0.9: 0.9's path API lost the moved inode on a rename over
  an existing file, so `sed -i` and editor saves left the file answering ENOENT.

**Done (2026-11-03): step 4's worktree half.** A turn's view is a linked
worktree: an admin dir `<git dir>/worktrees/corrode-<plan>` (`HEAD` detached at
the base, `commondir`, `gitdir`, an index of the base tree, `locked`) and a
read-only `.git` file the view serves, pointing into the git dir the sandbox
binds read-only at `/corrode/git`. Inside a turn `git status`, `diff` and `log`
work; `commit` is refused. The index has no stat data, so `status` hashes every
file -- fine at CAE's size, and it cannot write a refreshed index back. The
admin dir goes when the turn ends.

Since then (2026-11-03): `touch` marks a file modified now; symlinks are served
(readlink/symlink, followed within the view by the file tools, committed as
`120000`); a submodule is an empty read-only directory the sandbox binds the
host's checkout over; configured git filter drivers (git-lfs) are pass-through
inside a view, whose blobs are stored bytes, and submodules are `ignore = all`
for git there (their `.git` files point into a modules dir a view does not
have); commands no longer inherit the daemon's `CORRODE_*`/`HIPFIRE_*` env. A view of the Corrode repo itself --
9 submodules, 35 symlinks, an LFS filter -- runs `cargo check`, the whole
workspace's tests and `git status`.

Still to do: reusing a target dir across turns (a turn's first build is
cold); a FUSE handle reads its file once per open, but whole.

### Measurement: builds through the FUSE mount (2026-11-03)

Through a mounted **view** of the same commit (`CORRODE_MOUNT_REV=HEAD`): cold
64.0 s, no-op 0.1 s, after editing `cae-graph` 7.3 s; a syntax error appended
through the mount fails the build, and `sed -i` removing it builds again.

CAE (14 crates, ~119K lines) cloned to a scratch directory and mounted through
`fuse.rs` over `PassthroughVfs` (`--features fuse`, `CORRODE_MOUNT`). Halo host
(32 threads), `CARGO_BUILD_JOBS=8` (the spawn policy's default),
`CARGO_INCREMENTAL=0`, dependencies prefetched, `CARGO_TARGET_DIR` on native
disk outside the mount (step 3's layout). Native first, then the mount, so both
read source from a warm page cache.

| `cargo build --workspace` | native | through FUSE |
|---|---|---|
| cold | 60.7 s | 62.1 s (+2%) |
| no-op | 0.1 s | 0.1 s |
| after editing `cae-graph` (11 dependents) | 7.1 s | 7.3 s |
| after editing a leaf crate | 3.2 s | 3.2 s |

`rg fn` over the whole tree: 0.01 s native, 0.04 s through the mount.

**FUSE is not the bottleneck for a build-and-test loop**, so §7's fallback
(materialise a worktree for builds) is not needed. Not covered: several agents
building at once through one daemon, `cargo test`, and a cold page cache.

**Prerequisite found.** `fuse.rs` reports every timestamp as the Unix epoch.
Through the unpatched mount, appending a syntax error to a source file and
rebuilding gave `Finished` with nothing recompiled — cargo's freshness check
never sees an edit. The table above used a temporary patch that reported the
backing file's real mtime. A view must report content-derived mtimes (step 2)
before any command builds through it.
