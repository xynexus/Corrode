# Corrode: per-branch VFS, overlays, and build reuse

**Design note — exploratory, nothing decided.** How agents should see the
repository when several of them work at once, and what that implies for the
graph, the VFS, and their builds.

- **Status:** discussion only, except §1 — the `search_files` corpus fix is
  **implemented** (`Vfs::tracked_files` + the two property filters). Everything
  from §2 on remains undecided.
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
