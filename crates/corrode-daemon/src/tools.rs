//! The tools a subagent can execute, and how a small model reaches them.
//!
//! Small models can't be trusted to construct tool-call JSON, so in the tool-execution
//! loop ([`crate::daemon`]) a small model states its intent in plain English on a
//! `TOOL:` line, and Needle turns that into a structured call against [`EXEC_TOOLS`].
//! [`ToolBox`] then executes the call against the daemon's real capabilities and hands
//! back an observation the model reads on its next turn.
//!
//! Mutating tools (`write_file` / `run_command` / `run_skill_script`) sit behind the
//! daemon's human approval gate before [`ToolBox::execute`] runs them. ponytail: they
//! still execute unsandboxed on the host — sandboxing is the remaining gap before
//! approvals can be relaxed for unattended swarms.

use crate::dialect::{Effect, Param, Tool};
use crate::toolcall::ToolCall;
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// The tool-execution toolset, as canonical (model-agnostic) [`Tool`] data. A model's
/// [`crate::dialect::ToolDialect`] renders these into the schema it expects and maps its
/// call names back to these canonical names. A tool's [`Effect`] decides how it is
/// treated: anything but `Read` is gated behind human approval ([`is_mutating`])
/// before [`ToolBox::execute`] runs it.
pub const EXEC_TOOLS: &[Tool] = &[
    Tool {
        name: "read_file",
        effect: Effect::Read,
        description: "Read the contents of a file in the repository.",
        params: &[Param {
            name: "path",
            ty: "string",
            description: "Repository-relative file path.",
            required: true,
        }],
    },
    Tool {
        name: "list_dir",
        effect: Effect::Read,
        description: "List the entries of a directory in the repository.",
        params: &[Param {
            name: "path",
            ty: "string",
            description: "Repository-relative directory path.",
            required: true,
        }],
    },
    // Order is load-bearing: role_tools slices contiguous prefixes off this array
    // (observe | +skills | full), so the read-only tools lead, skills sit before the
    // sharp mutating pair.
    Tool {
        name: "search_files",
        effect: Effect::Read,
        description: "Find lines matching a substring across repository files. Use this \
to locate code without reading whole files.",
        params: &[
            Param {
                name: "query",
                ty: "string",
                description: "The text to search for (plain substring).",
                required: true,
            },
            Param {
                name: "path",
                ty: "string",
                description: "Optional repository-relative directory to scope the search.",
                required: false,
            },
        ],
    },
    Tool {
        name: "run_skill_script",
        effect: Effect::Exec,
        description: "Run a script bundled with an installed skill.",
        params: &[Param {
            name: "target",
            ty: "string",
            description: "The skill and script as skill/script (e.g. impeccable/hook.mjs), or just the script name.",
            required: true,
        }],
    },
    Tool {
        name: "write_file",
        effect: Effect::Mutate,
        description: "Create or overwrite a file with the given contents.",
        params: &[
            Param {
                name: "path",
                ty: "string",
                description: "Repository-relative file path.",
                required: true,
            },
            Param {
                name: "contents",
                ty: "string",
                description: "The full new contents of the file.",
                required: true,
            },
        ],
    },
    Tool {
        name: "run_command",
        effect: Effect::Exec,
        description: "Run a shell command in the repository and return its output.",
        params: &[Param {
            name: "command",
            ty: "string",
            description: "The shell command line to run.",
            required: true,
        }],
    },
];

/// The exec tools a role may use. Harness-enforced, not suggested: the declared set
/// is all the grammar (native path) or the rendered schema (Needle path) can ever
/// produce, so an out-of-role call is unreachable rather than discouraged.
/// Research/architect observe; review verifies through skills — `run_command` was
/// its measured misuse (docs/todo/tool-call-judgement.md item 3) — and only the
/// coder gets the full set. Costs cross-role KV sharing: the tools JSON renders
/// ahead of the shared prefix, so roles with different sets no longer prefix-share
/// on a common model (within-role sharing, including fan-out attempts, is intact).
pub fn role_tools(role: crate::roles::Role) -> &'static [Tool] {
    use crate::roles::Role;
    match role {
        Role::Coder => EXEC_TOOLS,
        Role::Review => &EXEC_TOOLS[..4],
        // read_file, list_dir, search_files — read-only observation for the roles
        // that must never mutate.
        Role::Research | Role::Architect | Role::Orchestration => &EXEC_TOOLS[..3],
    }
}

/// Hard cap on path-enum candidates for the grammar value constraint. hipfire's scan
/// is O(vocab × candidates): 64 ≈ 400 ms per call (tolerable), 256+ stalls — measured
/// in docs/todo/tool-call-judgement.md item 4. Over the cap sends NO path enum.
const MAX_PATH_VALUES: usize = 64;

/// Cap on how many bytes of a file a `read_file` observation carries back into the
/// model's context — enough to be useful without blowing the window.
const MAX_READ_BYTES: usize = 4096;

/// Whether a tool call mutates or executes and so must clear the human approval gate
/// before it runs: its tool's [`Effect`] is not `Read`. An unknown name counts as
/// mutating -- fail closed.
pub fn is_mutating(call: &ToolCall) -> bool {
    EXEC_TOOLS
        .iter()
        .find(|t| t.name == call.name)
        .is_none_or(|t| t.effect != Effect::Read)
}

/// A one-line, human-readable description of what a call will do — shown in the approval
/// prompt so a person knows exactly what they're authorizing.
pub fn describe(call: &ToolCall) -> String {
    match call.name.as_str() {
        "read_file" => format!(
            "read_file {}",
            arg_str(call, "path").unwrap_or("<missing path>")
        ),
        "list_dir" => format!(
            "list_dir {}",
            arg_str(call, "path").unwrap_or("<missing path>")
        ),
        "search_files" => format!(
            "search_files {:?}{}",
            arg_str(call, "query").unwrap_or("<missing query>"),
            arg_str(call, "path").map(|p| format!(" in {p}")).unwrap_or_default()
        ),
        "write_file" => format!(
            "write_file {}",
            arg_str(call, "path").unwrap_or("<missing path>")
        ),
        "run_command" => format!(
            "run_command: {}",
            arg_str(call, "command").unwrap_or("<missing command>")
        ),
        "run_skill_script" => format!(
            "run_skill_script {}",
            arg_str(call, "target").unwrap_or("<missing target>")
        ),
        other => format!("{other}({})", call.arguments),
    }
}

/// Executes tool calls against the daemon's VFS (and, for `run_command`/`run_skill_script`,
/// the repo root as the working directory). Holds shared/owned state so it can live in
/// the (`'static`) tool-loop future rather than borrowing the daemon.
#[derive(Clone)]
pub struct ToolBox {
    vfs: Arc<dyn Vfs>,
    root: PathBuf,
    /// Skill name -> skill directory, for resolving `run_skill_script` (stage 3).
    skill_scripts: Arc<HashMap<String, PathBuf>>,
    /// Optional bubblewrap confinement for `run_command`/`run_skill_script`.
    /// Disabled by default (see `with_sandbox`).
    sandbox: crate::sandbox::Sandbox,
    /// Per-user hipfire bearer for this session's model calls (fairness). `None`
    /// uses the daemon's shared key. Carried here so the tool loops (which already
    /// hold the ToolBox) can attribute their `respond` calls without extra params.
    owner_token: Option<String>,
    /// The session's graph store, when one is open. Gives `search_files` a SOFT half:
    /// BM25 over ingested code and comments, on top of the literal scan. `None` in the
    /// base build, where search stays literal-only.
    graph: Option<Arc<dyn crate::graph::GraphStore>>,
    /// The turn's plan id, so a trace note can name the SAME task node the plan graph
    /// writes (`{plan}:task:{id}`). Carried here rather than threaded through
    /// `run_task`/`run_tool_loop`/`run_native_tool_loop`, which already take 16 args.
    /// `None` outside a turn -> bare `task:{id}`.
    plan: Option<String>,
    /// hipfire client + model for cross-encoder reranking of graph hits. `None` unless
    /// `CORRODE_RERANK_MODEL` names a served reranker, so search is unchanged by default.
    reranker: Option<(Arc<crate::hipfire::Client>, String)>,
    /// The version (content hash) of each file this task last read or wrote, for
    /// `write_file`'s check (see [`ToolBox::stale_write`]).
    versions: Arc<std::sync::Mutex<HashMap<String, Option<u64>>>>,
}

/// A path's key in `ToolBox::versions`: `./src/lib.rs` and `src/lib.rs` are one file.
fn version_key(path: &str) -> String {
    Path::new(path.trim())
        .components()
        .filter(|c| !matches!(c, Component::CurDir))
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn content_version(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Chat-template markup that file contents, tool output, rule files and skills
/// may quote. A backend that matches these strings anywhere in a prompt (Hugging
/// Face tokenizers do, and vLLM and llama.cpp with them) reads the quote as
/// structure: a file quoting `<|im_end|>` ends the turn it is shown in, one
/// quoting `<tool_call>` opens a call. hipfire encodes them literally; Corrode
/// neutralizes them anyway, for the backends that do not.
const PROTOCOL_MARKERS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
];

/// WORD JOINER: invisible, and no tokenizer folds it into the markup around it.
const JOINER: char = '\u{2060}';

/// `text` with each protocol marker broken by a [`JOINER`] after its `<`, so it
/// reads as text to any backend. [`restore`] is its exact inverse.
pub fn neutralize(text: &str) -> std::borrow::Cow<'_, str> {
    if !PROTOCOL_MARKERS.iter().any(|m| text.contains(m)) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = text.to_string();
    for m in PROTOCOL_MARKERS {
        out = out.replace(m, &format!("<{JOINER}{}", &m[1..]));
    }
    std::borrow::Cow::Owned(out)
}

/// `text` with the joiners [`neutralize`] put in protocol markers taken out, and
/// no others: what a model copies back out of an observation (a file it rewrites,
/// a command it reruns) gets the original bytes.
pub fn restore(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains(JOINER) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = text.to_string();
    for m in PROTOCOL_MARKERS {
        out = out.replace(&format!("<{JOINER}{}", &m[1..]), m);
    }
    std::borrow::Cow::Owned(out)
}

impl ToolBox {
    pub fn new(
        vfs: Arc<dyn Vfs>,
        root: PathBuf,
        skill_scripts: Arc<HashMap<String, PathBuf>>,
    ) -> Self {
        Self {
            vfs,
            root,
            skill_scripts,
            sandbox: crate::sandbox::Sandbox::disabled(),
            owner_token: None,
            graph: None,
            reranker: None,
            plan: None,
            versions: Default::default(),
        }
    }

    /// Record the file's current version as what this task has seen. For a read the
    /// turn's cache answered: the cache is cleared by every write in the turn, so what
    /// it returned is what is on disk now.
    pub async fn note_read(&self, path: &str) {
        let version = self.vfs.read(path).await.ok().map(|b| content_version(&b));
        self.versions.lock().unwrap().insert(version_key(path), version);
    }

    /// Why a whole-file write of `path` would discard someone else's change, if it
    /// would: the file is not what this task last read or wrote. Two coders
    /// registering modules in one `lib.rs`, or a task rewriting from text a sibling
    /// had since changed, otherwise lost each other's work silently -- and the
    /// result usually still compiled. A file the task never read is not checked
    /// (creating one, or a deliberate full rewrite).
    async fn stale_write(&self, path: &str) -> Option<String> {
        let seen = *self.versions.lock().unwrap().get(&version_key(path))?;
        let now = self.vfs.read(path).await.ok().map(|b| content_version(&b));
        (now != seen).then(|| {
            format!(
                "error: {path} changed since you last read it -- another task or a command \
                 wrote it. Writing now would discard that change. Read it again and apply \
                 your edit to its current contents."
            )
        })
    }

    /// Confine spawned processes with this sandbox (builder; default is disabled).
    pub fn with_sandbox(mut self, sandbox: crate::sandbox::Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Attribute this session's hipfire calls to a per-user token (builder;
    /// default `None` = the daemon's shared key).
    pub fn with_graph(mut self, graph: Option<Arc<dyn crate::graph::GraphStore>>) -> Self {
        self.graph = graph;
        self
    }

    /// Attach a cross-encoder reranker for graph hits. Reads `CORRODE_RERANK_MODEL`;
    /// absent means no reranking, which is the behaviour that predates this.
    pub fn with_reranker(mut self, client: Option<Arc<crate::hipfire::Client>>) -> Self {
        self.reranker = match (client, std::env::var("CORRODE_RERANK_MODEL").ok()) {
            (Some(c), Some(m)) if !m.trim().is_empty() => Some((c, m)),
            _ => None,
        };
        self
    }

    pub fn with_owner_token(mut self, owner_token: Option<String>) -> Self {
        self.owner_token = owner_token;
        self
    }

    pub fn with_plan(mut self, plan: &str) -> Self {
        self.plan = Some(plan.to_string());
        self
    }

    /// The graph node id for a task in this turn. Must match `PlanGraph::node_ref`, or a
    /// note attaches to a second, disconnected task node instead of joining the plan.
    pub fn task_ref(&self, id: u64) -> String {
        match &self.plan {
            Some(plan) => format!("{plan}:task:{id}"),
            None => format!("task:{id}"),
        }
    }

    /// The per-user hipfire bearer for `respond` calls in the tool loops.
    /// The session's graph store, for callers that need to write alongside a tool run
    /// (trace notes) rather than through a tool.
    pub fn graph(&self) -> Option<&Arc<dyn crate::graph::GraphStore>> {
        self.graph.as_ref()
    }

    pub fn owner_token(&self) -> Option<&str> {
        self.owner_token.as_deref()
    }

    /// Per-task value sets for the grammar value constraint (item 4 of
    /// docs/todo/tool-call-judgement.md): `read_file`/`list_dir` paths from a VFS
    /// walk, `run_skill_script` targets from the installed skills. `write_file.path`
    /// and `run_command.command` are free text and must NEVER be constrained — a
    /// closed set there would wedge generation. Recomputed per task execution (repo
    /// state moves between tasks); the walk cap keeps it cheap.
    pub async fn param_values(&self) -> crate::dialect::ParamValues {
        let mut values = crate::dialect::ParamValues::new();
        if let Some(paths) = self.walk_paths().await {
            if !paths.is_empty() {
                values.insert(("read_file".into(), "path".into()), paths.clone());
                values.insert(("list_dir".into(), "path".into()), paths);
            }
        }
        // Real `skill/script` pairs — the resolver's canonical form. A bare skill
        // name is NOT resolvable (it would be read as a script filename), so a
        // name-only enum would grammar-force strings that always fail.
        let mut targets: Vec<String> = self
            .skill_scripts
            .iter()
            .flat_map(|(name, dir)| {
                std::fs::read_dir(dir.join("scripts"))
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| e.path().is_file())
                    .filter_map(move |e| Some(format!("{name}/{}", e.file_name().to_str()?)))
            })
            .collect();
        if !targets.is_empty() {
            targets.sort();
            values.insert(("run_skill_script".into(), "target".into()), targets);
        }
        values
    }

    /// Every repo path (files AND directories), breadth-first, `.git` and `target`
    /// pruned (noise — `.git/revisions` must simply be absent, not enumerated).
    /// `None` past [`MAX_PATH_VALUES`] or on any listing error: a partial enum would
    /// make real paths unreachable, so over-cap falls back to no constraint and leans
    /// on the corrective observations instead (items 2–3 of the TODO).
    async fn walk_paths(&self) -> Option<Vec<String>> {
        // "." seeds the set so the repo root itself stays a legal list_dir target.
        let mut paths = vec![".".to_string()];
        let mut queue = std::collections::VecDeque::from([String::new()]);
        while let Some(dir) = queue.pop_front() {
            for e in self.vfs.list(&dir).await.ok()? {
                let name = e.path.rsplit('/').next().unwrap_or("");
                if e.is_dir && matches!(name, ".git" | "target") {
                    continue;
                }
                if paths.len() >= MAX_PATH_VALUES {
                    return None;
                }
                if e.is_dir {
                    queue.push_back(e.path.clone());
                }
                paths.push(e.path);
            }
        }
        Some(paths)
    }

    /// Run one tool call and return an observation string (result or a readable error —
    /// errors go back to the model as text so it can recover, never as a hard failure).
    /// Callers must already have cleared [`is_mutating`] calls through the approval gate.
    pub async fn execute(&self, call: &ToolCall) -> String {
        neutralize(&self.execute_raw(call).await).into_owned()
    }

    /// [`Self::execute`] before its observation is neutralized; the inputs a model
    /// copied out of an earlier observation get their markup restored.
    async fn execute_raw(&self, call: &ToolCall) -> String {
        // Schema check before dispatch. Naming what is missing AND what did arrive gives
        // the model something to correct; the per-tool arms below stay as the
        // destructuring, but no longer carry the burden of being the only guard.
        // Belt and braces: `gate_and_execute` refuses these before the approval gate, but
        // `execute` is public and a caller reaching it directly must not skip the check.
        let missing = missing_required(call);
        if !missing.is_empty() {
            return missing_required_error(call, &missing);
        }
        match call.name.as_str() {
            "read_file" => match arg_str(call, "path") {
                Some(path) => self.read_file(path).await,
                None => "error: read_file needs a `path` argument".to_string(),
            },
            "list_dir" => match arg_str(call, "path") {
                Some(path) => self.list_dir(path).await,
                None => "error: list_dir needs a `path` argument".to_string(),
            },
            "search_files" => match arg_str(call, "query") {
                Some(query) => self.search_files(&restore(query), arg_str(call, "path")).await,
                None => "error: search_files needs a `query` argument".to_string(),
            },
            "write_file" => match (arg_str(call, "path"), arg_text(call, "contents")) {
                (Some(path), Some(contents)) => self.write_file(path, &restore(&contents)).await,
                _ => "error: write_file needs `path` and `contents` arguments".to_string(),
            },
            "run_command" => match arg_text(call, "command") {
                Some(command) => self.run_command(restore(&command).trim()).await,
                None => "error: run_command needs a `command` argument".to_string(),
            },
            "run_skill_script" => match arg_str(call, "target") {
                Some(target) => self.run_skill_script(target).await,
                None => "error: run_skill_script needs a `target` argument".to_string(),
            },
            other => format!("error: unknown tool `{other}`"),
        }
    }

    /// Read-only grep over the VFS's tracked corpus: `path:line: text` for every line
    /// containing `query` (plain, case-sensitive substring). Bounded on files scanned,
    /// matches returned, per-file size and line length, so a research/architect agent —
    /// which has no `run_command` — can still locate code cheaply.
    ///
    /// The corpus comes from [`Vfs::tracked_files`], not a directory walk. A walk
    /// returned tokenizer binaries, minified `xterm.js` and vendored submodule data to
    /// a subagent, which re-ran the same search, got the same wall of noise, and burned
    /// minutes of GPU before its turn errored out. Blacklisting those paths would
    /// always be one new vendored directory behind; asking what the project tracks is
    /// the root fix, and it self-maintains.
    ///
    /// Two filters remain, and they are properties rather than paths, because the
    /// survivors of the corpus fix are committed-but-not-source files that no path list
    /// predicts: a NUL sniff (binary) and a maximum line length (minified).
    ///
    /// Measured on this repo, searching "overview": corpus 3824 files -> 170, and the
    /// filters then skip 22 binary + 7 minified, leaving 6 files. One of those six is
    /// residual noise — `needle.vocab` is a token table that is genuinely plain text
    /// (no NUL, longest line 22 chars) and matches because it contains the token
    /// `overview`. No property distinguishes it from source; only a path rule would,
    /// which is what this design rejects. Recorded rather than special-cased.
    /// BM25 hits from the ingested graph, as `path:line: text`, excluding anything the
    /// literal scan already reported.
    ///
    /// The line number is DERIVED, not stored: the file's code nodes come back from the
    /// store in order, `project` replays them and reports where each landed, and the
    /// hit's order key selects its placement. A line written at ingest time would be
    /// wrong after the next edit above it; this is right by construction, which is what
    /// the sparse order key and byte-exact composition were for.
    async fn graph_matches(&self, query: &str, prefix: Option<&str>, already: &[String]) -> Vec<String> {
        const MAX_SOFT: usize = 12;
        /// Shortlist handed to the cross-encoder. It costs one forward per candidate, so
        /// this is the knob that decides whether reranking is affordable: BM25 narrows
        /// cheaply and the reranker only reorders what survived.
        const SHORTLIST: usize = 24;
        let Some(store) = &self.graph else {
            return Vec::new();
        };
        let fetch = if self.reranker.is_some() { SHORTLIST } else { MAX_SOFT * 4 };
        let Ok(mut hits) = store.code_search(query, fetch) else {
            return Vec::new();
        };

        // Rerank the shortlist jointly. BM25 (even decomposed) scores a document against
        // the query one term at a time; a cross-encoder reads the pair together, which is
        // what separates candidates that share every individual term.
        if let Some((client, model)) = &self.reranker {
            let docs: Vec<String> = hits.iter().map(|(_, text)| text.clone()).collect();
            match client.rerank(model, query, &docs).await {
                Ok(order) if !order.is_empty() => {
                    hits = order.into_iter().filter_map(|(i, _)| hits.get(i).cloned()).collect();
                }
                // A reranker that is down or misconfigured must not empty the results:
                // fall through to the BM25 order, which is a worse answer, not no answer.
                Ok(_) => {}
                Err(e) => eprintln!("rerank unavailable ({e}); using BM25 order"),
            }
        }

        let mut out = Vec::new();
        let mut placements: HashMap<String, Vec<crate::projection::Placement>> = HashMap::new();
        for (key, text) in hits {
            if out.len() >= MAX_SOFT {
                break;
            }
            // `code:{path}#{order}` / `comment:{path}#{n}`.
            let Some((kind, rest)) = key.split_once(':') else { continue };
            let Some((path, tail)) = rest.rsplit_once('#') else { continue };
            if prefix.is_some_and(|p| !path.starts_with(p)) {
                continue;
            }
            let places = placements.entry(path.to_string()).or_insert_with(|| {
                store
                    .file_nodes(path)
                    .map(|nodes| crate::projection::project(&nodes).1)
                    .unwrap_or_default()
            });
            // A comment's id counts comments, not order keys, so only a code hit can
            // select a placement directly. A comment reports its file rather than
            // guessing a line — the `in_node` edge is the honest way to place it, and
            // that is a traversal this does not yet do.
            let line = (kind == "code")
                .then(|| tail.parse::<u64>().ok())
                .flatten()
                .and_then(|order| places.iter().find(|p| p.order == order))
                .map(|p| p.start_line);
            let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
            let shown = &flat[..floor_char_boundary(&flat, 120)];
            let entry = match line {
                Some(l) => format!("{path}:{l}: {shown}"),
                None => format!("{path}: {shown}"),
            };
            // Don't repeat what the literal scan already found.
            if already.iter().any(|m| m.starts_with(&format!("{path}:"))) {
                continue;
            }
            out.push(entry);
        }
        out
    }

    async fn search_files(&self, query: &str, scope: Option<&str>) -> String {
        const MAX_FILES: usize = 4000;
        const MAX_MATCHES: usize = 60;
        const MAX_LINE: usize = 200;
        const MAX_FILE_BYTES: usize = 1_000_000; // skip blobs — not source
        /// A line longer than this means minified/generated content, not something a
        /// human reads — and one such "line" can be the entire file.
        const MAX_SOURCE_LINE: usize = 1000;
        /// Bytes sniffed for NUL before deciding a file is binary.
        const SNIFF: usize = 8192;

        if query.is_empty() {
            return "error: search_files needs a non-empty query".to_string();
        }
        let files = match self.vfs.tracked_files().await {
            Ok(f) => f,
            Err(e) => return format!("error: could not determine the searchable file set: {e}"),
        };
        // `scope` narrows to a subtree; paths are repo-relative, so a prefix match is
        // the whole of it. A trailing slash keeps `src` from matching `src-gen/`.
        let prefix = scope
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != ".")
            .map(|s| format!("{}/", s.trim_end_matches('/')));

        let mut files_scanned = 0usize;
        let mut matches: Vec<String> = Vec::new();
        let mut capped = false;
        let mut skipped_binary = 0usize;
        let mut skipped_minified = 0usize;

        'walk: for path in files {
            if let Some(p) = &prefix {
                if !path.starts_with(p.as_str()) {
                    continue;
                }
            }
            if files_scanned >= MAX_FILES {
                capped = true;
                break;
            }
            let Ok(bytes) = self.vfs.read(&path).await else {
                continue; // raced deletion, or tracked-but-absent
            };
            if bytes.len() > MAX_FILE_BYTES {
                continue;
            }
            // Binary: a NUL in the head. Cheaper and more reliable than extensions,
            // and it catches the tokenizer blobs that started this.
            if bytes.iter().take(SNIFF).any(|b| *b == 0) {
                skipped_binary += 1;
                continue;
            }
            files_scanned += 1;
            let text = String::from_utf8_lossy(&bytes);
            let mut minified = false;
            for (n, line) in text.lines().enumerate() {
                if line.len() > MAX_SOURCE_LINE {
                    minified = true;
                    break;
                }
                if line.contains(query) {
                    let l = line.trim();
                    let shown = &l[..floor_char_boundary(l, MAX_LINE)];
                    matches.push(format!("{path}:{}: {shown}", n + 1));
                    if matches.len() >= MAX_MATCHES {
                        capped = true;
                        break 'walk;
                    }
                }
            }
            if minified {
                skipped_minified += 1;
                // Drop this file's partial matches: a minified file's "lines" are not
                // lines, so reporting some of them is worse than reporting none.
                matches.retain(|m| !m.starts_with(&format!("{path}:")));
            }
        }

        // The graph's half. Appended, never substituted: a literal scan answers "where
        // is this exact string", which BM25 does not, and losing that would be a
        // regression dressed up as an upgrade.
        let soft = self.graph_matches(query, prefix.as_deref(), &matches).await;

        if matches.is_empty() && soft.is_empty() {
            let mut msg = format!("no matches for {query:?}");
            if let Some(p) = &prefix {
                msg.push_str(&format!(" under {p}"));
            }
            return msg;
        }
        let mut out = matches.join("\n");
        if !soft.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("related (from the code graph):\n");
            out.push_str(&soft.join("\n"));
        }
        if capped {
            out.push_str("\n… (results capped; refine the query or narrow the path)");
        }
        if skipped_binary + skipped_minified > 0 {
            out.push_str(&format!(
                "\n… (skipped {skipped_binary} binary, {skipped_minified} minified)"
            ));
        }
        out
    }

    async fn write_file(&self, path: &str, contents: &str) -> String {
        if let Some(refused) = self.stale_write(path).await {
            return refused;
        }
        match self.vfs.write(path, contents.as_bytes()).await {
            Ok(()) => {
                let version = Some(content_version(contents.as_bytes()));
                self.versions.lock().unwrap().insert(version_key(path), version);
                format!("wrote {} bytes to {path}", contents.len())
            }
            Err(e) => format!("error: could not write {path}: {e}"),
        }
    }

    async fn run_command(&self, command: &str) -> String {
        // sandbox.wrap is a no-op when disabled: plain `sh -c <command>`.
        let (prog, args) = self.sandbox.wrap(&self.root, &["sh", "-c", command]);
        let mut cmd = tokio::process::Command::new(prog);
        cmd.args(args).current_dir(&self.root);
        run_bounded(cmd, &format!("`{command}`"), command_timeout()).await
    }

    /// Stage-3 skill execution: run a script bundled with an installed skill, from the
    /// repo root. `target` is `skill/script` (e.g. `impeccable/hook.mjs`) or a bare
    /// script name — a bare name is resolved against every installed skill (so it works
    /// even when the model drops the skill name, which Needle tends to do pre-finetune).
    /// The interpreter is chosen by extension (`.mjs`/`.js`→node, `.py`→python3,
    /// `.sh`→bash, else direct exec). The script path is validated to stay inside the
    /// skill dir (no `..`/absolute escape).
    async fn run_skill_script(&self, target: &str) -> String {
        // Take the first whitespace token of each part: a skill/script name has no
        // spaces, and Needle pre-finetune tends to append a stray word (e.g. "hello.sh
        // script"). Robustness, not correctness — a finetuned model won't need it.
        let first_token = |s: &str| s.trim().split_whitespace().next().unwrap_or("").to_string();
        let (skill, script) = match target.split_once('/') {
            Some((s, sc)) => (Some(first_token(s)), first_token(sc)),
            None => (None, first_token(target)),
        };
        let skill = skill.as_deref();
        let script = script.as_str();
        let rel = Path::new(script);
        if script.is_empty()
            || rel.is_absolute()
            || rel.components().any(|c| c == Component::ParentDir)
        {
            return format!("error: invalid script `{target}`");
        }

        // Resolve to a concrete script path: a named skill, else the first installed
        // skill that actually has this script.
        let path = match skill {
            Some(name) => match self.skill_scripts.get(name) {
                Some(dir) => match script_path(dir, rel) {
                    Some(p) => p,
                    None => return format!("error: skill `{name}` has no script `{script}`"),
                },
                None => return format!("error: no installed skill named `{name}`"),
            },
            None => match self.skill_scripts.values().find_map(|d| script_path(d, rel)) {
                Some(p) => p,
                None => return format!("error: no installed skill has a script `{script}`"),
            },
        };

        let path_str = path.to_string_lossy();
        let argv: Vec<&str> = match interpreter_for(&path) {
            Some(interp) => vec![interp, &path_str],
            None => vec![&path_str],
        };
        let (prog, args) = self.sandbox.wrap(&self.root, &argv);
        let mut cmd = tokio::process::Command::new(prog);
        cmd.args(args).current_dir(&self.root);
        run_bounded(cmd, &format!("skill script `{target}`"), command_timeout()).await
    }

    async fn read_file(&self, path: &str) -> String {
        match self.vfs.read(path).await {
            Ok(bytes) => {
                let version = Some(content_version(&bytes));
                self.versions.lock().unwrap().insert(version_key(path), version);
                let text = String::from_utf8_lossy(&bytes);
                let (shown, truncated) = if text.len() > MAX_READ_BYTES {
                    (&text[..floor_char_boundary(&text, MAX_READ_BYTES)], true)
                } else {
                    (text.as_ref(), false)
                };
                let mut out = format!("contents of {path}:\n{shown}");
                if truncated {
                    out.push_str("\n… (truncated)");
                }
                out
            }
            Err(e) => self.path_error("read", path, &e).await,
        }
    }

    async fn list_dir(&self, path: &str) -> String {
        match self.vfs.list(path).await {
            Ok(entries) => {
                let mut out = format!("entries of {}:", if path.is_empty() { "." } else { path });
                // Capped: one listing of a generated or vendored directory could fill a
                // step's share of the context on its own.
                const LIST_CAP: usize = 500;
                let more = entries.len().saturating_sub(LIST_CAP);
                for e in entries.into_iter().take(LIST_CAP) {
                    out.push_str(&format!(
                        "\n  {}{}",
                        e.path,
                        if e.is_dir {
                            "/".to_string()
                        } else {
                            format!(" ({} bytes)", e.bytes)
                        }
                    ));
                }
                if more > 0 {
                    out.push_str(&format!("\n  ... {more} more (list a subdirectory)"));
                }
                out
            }
            Err(e) => self.path_error("list", path, &e).await,
        }
    }

    /// A readable error for a failed `read_file`/`list_dir`. A path that doesn't stat is
    /// a miss (likely hallucinated): name it and suggest near-matches so the model gets
    /// a corrective observation instead of a raw errno. A path that stats but still
    /// failed keeps the underlying error (permissions, is-a-directory, …).
    async fn path_error(&self, verb: &str, path: &str, e: &anyhow::Error) -> String {
        if self.vfs.stat(path).await.is_ok() {
            return format!("error: could not {verb} {path}: {e}");
        }
        let close = self.near_matches(path).await;
        let mut out = format!("error: no such path '{path}'");
        if !close.is_empty() {
            out.push_str(&format!(". Did you mean '{}'?", close.join("', '")));
        }
        out
    }

    /// Up to 3 entries near a missing path, from its parent's listing (the root listing
    /// when the parent is absent too), closest final component first.
    async fn near_matches(&self, path: &str) -> Vec<String> {
        let trimmed = path.trim_end_matches('/');
        let (parent, name) = trimmed.rsplit_once('/').unwrap_or(("", trimmed));
        if name.is_empty() {
            return Vec::new();
        }
        let entries = match self.vfs.list(parent).await {
            Ok(entries) => entries,
            Err(_) => self.vfs.list("").await.unwrap_or_default(),
        };
        let name = name.to_ascii_lowercase();
        let mut scored: Vec<(usize, String)> = entries
            .into_iter()
            .filter_map(|e| {
                let last = e.path.rsplit('/').next().unwrap_or_default().to_ascii_lowercase();
                closeness(&name, &last).map(|d| (d, e.path))
            })
            .collect();
        scored.sort();
        scored.truncate(3);
        scored.into_iter().map(|(_, p)| p).collect()
    }
}

/// The string value of a call argument, trimmed. `None` only if the key is missing or
/// not a string — an empty string is a valid path (the repo root for `list_dir`).
/// The observation for a call that is missing required arguments. Names what is missing
/// AND what did arrive, so the model can correct the call rather than guess at it.
pub(crate) fn missing_required_error(call: &ToolCall, missing: &[&'static str]) -> String {
    let mut got: Vec<&str> = call
        .arguments
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    got.sort_unstable();
    format!(
        "error: {} is missing required argument(s): {}. Received: {}. \
         Send every required argument in one call.",
        call.name,
        missing.join(", "),
        if got.is_empty() { "nothing".to_string() } else { got.join(", ") },
    )
}

/// Required arguments a call is missing, per the canonical tool schema.
///
/// Driven by `EXEC_TOOLS`' own `required` flags rather than a per-tool arm, so a tool
/// gains this the moment it is declared and no dispatch arm can forget it. Observed
/// live: a native emitter produced `write_file` with `path` and no `contents`, which
/// reached execution and came back as a hand-written per-tool error.
///
/// An argument counts as missing when the key is absent or null. A non-string value is
/// present: free text (`contents`, `command`) takes it back as its JSON text (see
/// [`arg_text`]). A present-but-EMPTY string is NOT missing: `write_file` with `contents: ""` is a
/// truncation, and rejecting it here would break a legitimate call to protect against
/// a malformed one.
pub(crate) fn missing_required(call: &ToolCall) -> Vec<&'static str> {
    let Some(tool) = EXEC_TOOLS.iter().find(|t| t.name == call.name) else {
        return Vec::new(); // unknown tool: the dispatch arm reports it
    };
    tool.params
        .iter()
        .filter(|p| p.required && call.arguments.get(p.name).is_none_or(|v| v.is_null()))
        .map(|p| p.name)
        .collect()
}

pub(crate) fn arg_str<'a>(call: &'a ToolCall, key: &str) -> Option<&'a str> {
    call.arguments.get(key)?.as_str().map(str::trim)
}

/// A free-text argument (`contents`, `command`) exactly as sent: never trimmed, and a
/// non-string value taken back as its JSON text. Trimming stripped every written
/// file's trailing newline and first-line indentation (`cargo fmt --check` then
/// failed on everything the swarm wrote), and a server that JSON-coerces tool
/// arguments hands over a file whose contents parse as JSON -- a `package.json`, a
/// bare number -- as an object or number, which read as "missing contents".
pub(crate) fn arg_text<'a>(call: &'a ToolCall, key: &str) -> Option<std::borrow::Cow<'a, str>> {
    match call.arguments.get(key)? {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(std::borrow::Cow::Borrowed(s)),
        other => Some(std::borrow::Cow::Owned(other.to_string())),
    }
}

/// Format a finished process's exit code + stdout/stderr into a bounded observation.
///
/// Delegates to [`crate::digest`], which recognizes rustc diagnostics and libtest
/// results and renders them structured. It matters that this is not a plain truncation:
/// a build log prints progress first and its verdict last, so the previous head-only cut
/// kept the least useful bytes and dropped the counts entirely.
/// Wall-clock limit for one `run_command` / `run_skill_script`:
/// `CORRODE_COMMAND_TIMEOUT_S`, default 1800. A swarm runs unattended, so a
/// command that never ends (a server, a deadlocked test, `cargo run` on a
/// binary that waits) must end the call, not the turn.
pub(crate) fn command_timeout() -> std::time::Duration {
    let s = std::env::var("CORRODE_COMMAND_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&s: &u64| s > 0)
        .unwrap_or(1800);
    std::time::Duration::from_secs(s)
}

/// Bytes kept from each of a command's stdout and stderr: this much of the head
/// and this much of the tail, the middle dropped. Output past it is still read
/// (so the child never blocks on a full pipe) but not kept -- `yes` used to grow
/// the daemon without bound, and on a UMA host that RAM is the GPU's.
const COMMAND_CAPTURE_HALF: usize = 512 * 1024;

/// Read `r` to EOF, keeping the head and tail `COMMAND_CAPTURE_HALF` bytes.
async fn read_capped(mut r: impl tokio::io::AsyncRead + Unpin) -> Vec<u8> {
    use tokio::io::AsyncReadExt;
    let (mut head, mut tail, mut dropped) = (Vec::new(), Vec::new(), 0usize);
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = match r.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut chunk = &buf[..n];
        if head.len() < COMMAND_CAPTURE_HALF {
            let take = chunk.len().min(COMMAND_CAPTURE_HALF - head.len());
            head.extend_from_slice(&chunk[..take]);
            chunk = &chunk[take..];
        }
        tail.extend_from_slice(chunk);
        if tail.len() > 2 * COMMAND_CAPTURE_HALF {
            let cut = tail.len() - COMMAND_CAPTURE_HALF;
            tail.drain(..cut);
            dropped += cut;
        }
    }
    if tail.len() > COMMAND_CAPTURE_HALF {
        let cut = tail.len() - COMMAND_CAPTURE_HALF;
        tail.drain(..cut);
        dropped += cut;
    }
    if dropped > 0 {
        head.extend_from_slice(format!("\n… [{dropped} bytes omitted] …\n").as_bytes());
    }
    head.extend_from_slice(&tail);
    head
}

/// Kills a command's whole process group when dropped, unless disarmed: on timeout,
/// and when the task running the command is itself dropped (its ceiling, the turn's
/// cut-off, CancelTurn) -- `kill_on_drop` reaches only the direct child.
struct GroupKill(Option<u32>);

impl Drop for GroupKill {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            // The group id is the child's pid (process_group(0)).
            let _ = std::process::Command::new("kill")
                .args(["-s", "KILL", "--", &format!("-{pid}")])
                .status();
        }
    }
}

/// Run `cmd` to completion under `limit`, capturing bounded output. The child
/// gets its own process group, and on timeout the whole group is killed --
/// killing only `sh` would leave what it started (test binaries, servers, a
/// backgrounded `&` that holds the pipe open) running.
async fn run_bounded(
    mut cmd: tokio::process::Command,
    what: &str,
    limit: std::time::Duration,
) -> String {
    use std::process::Stdio;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .process_group(0);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return format!("error: could not run {what}: {e}"),
    };
    let mut group = GroupKill(child.id());
    let started = std::time::Instant::now();
    let (so, se) = (child.stdout.take(), child.stderr.take());
    let run = async {
        let out = async {
            if let Some(r) = so {
                read_capped(r).await
            } else {
                Vec::new()
            }
        };
        let err = async {
            if let Some(r) = se {
                read_capped(r).await
            } else {
                Vec::new()
            }
        };
        tokio::join!(out, err, child.wait())
    };
    let done = tokio::time::timeout(limit, run).await;
    if done.is_ok() {
        group.0 = None; // finished: leave alone whatever it deliberately left running
    }
    drop(group);
    match done {
        // How long it took goes at the END, so the digest's first line (exit status,
        // test counts) is unchanged: a model choosing between a full build and a
        // focused test had no way to know one took 4 minutes and the other 2 seconds.
        Ok((stdout, stderr, Ok(status))) => format!(
            "{}\n(took {:.1}s)",
            format_command_output(std::process::Output {
                status,
                stdout,
                stderr,
            }),
            started.elapsed().as_secs_f32()
        ),
        Ok((_, _, Err(e))) => format!("error: {what}: {e}"),
        Err(_) => {
            format!(
                "exit timeout: {what} was still running after {}s and was killed \
                 (CORRODE_COMMAND_TIMEOUT_S). Run long or never-ending commands with a \
                 bound of their own (e.g. `timeout 60 ...`).",
                limit.as_secs()
            )
        }
    }
}

fn format_command_output(out: std::process::Output) -> String {
    let mut text = String::new();
    if !out.stdout.is_empty() {
        text.push_str(&String::from_utf8_lossy(&out.stdout));
    }
    if !out.stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&out.stderr));
    }
    crate::digest::command_observation(out.status.code().unwrap_or(-1), &text)
}

/// Largest index `<= max` on a char boundary — truncating mid-multibyte-char panics.
pub(crate) fn floor_char_boundary(s: &str, max: usize) -> usize {
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Whether a candidate final component is near a missing one — `Some(edit distance)`
/// when it's close (substring containment, a 3-char shared prefix, or an edit distance
/// within half the longer length), `None` otherwise. Lower is closer.
fn closeness(miss: &str, cand: &str) -> Option<usize> {
    let d = edit_distance(miss, cand);
    let prefix = miss
        .bytes()
        .zip(cand.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    (miss.contains(cand) || cand.contains(miss) || prefix >= 3 || d * 2 <= miss.len().max(cand.len()))
        .then_some(d)
}

/// Levenshtein distance, two-row DP — path components are short, so O(a·b) is nothing.
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let step = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
            cur.push(step);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Resolve a skill-relative script to a concrete path: `<skill>/scripts/<rel>` if it
/// exists, else `<skill>/<rel>`, else `None`.
fn script_path(skill_dir: &Path, rel: &Path) -> Option<PathBuf> {
    let in_scripts = skill_dir.join("scripts").join(rel);
    if in_scripts.exists() {
        return Some(in_scripts);
    }
    let at_root = skill_dir.join(rel);
    at_root.exists().then_some(at_root)
}

/// The interpreter to run a script with, by file extension. `None` -> execute the file
/// directly (relies on its shebang + exec bit).
fn interpreter_for(script: &Path) -> Option<&'static str> {
    match script.extension().and_then(|e| e.to_str()) {
        Some("mjs" | "js" | "cjs") => Some("node"),
        Some("py") => Some("python3"),
        Some("sh" | "bash") => Some("bash"),
        _ => None,
    }
}

/// The plain-English tool request a model wrote on a `TOOL:` line — the intent Needle
/// structures into a call. `None` (no `TOOL:` line) means the model's turn is its final
/// answer, not a tool step.
pub fn parse_tool_intent(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("TOOL:")?.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    })
}

#[cfg(test)]
mod tests {

    fn sh(script: &str) -> tokio::process::Command {
        let mut c = tokio::process::Command::new("sh");
        c.args(["-c", script]);
        c
    }

    // A command that never ends must end the call, and take what it started with
    // it: the backgrounded sleep holds stdout open, so without the group kill the
    // read never sees EOF.
    #[tokio::test]
    async fn a_hung_command_is_killed_with_its_process_group() {
        // A sleep length unique to this run marks the backgrounded grandchild.
        let secs = 100_000 + std::process::id() % 100_000;
        let t0 = std::time::Instant::now();
        let out = super::run_bounded(
            sh(&format!("sleep {secs} & sleep 300")),
            "`hang`",
            std::time::Duration::from_secs(1),
        )
        .await;
        assert!(out.starts_with("exit timeout"), "{out}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(10));
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let left = std::process::Command::new("pgrep")
            .args(["-f", &format!("^sleep {secs}$")])
            .output()
            .unwrap();
        assert!(
            left.stdout.is_empty(),
            "survivor: {}",
            String::from_utf8_lossy(&left.stdout)
        );
    }

    // A task dropped mid-command (its ceiling, the turn's cut-off, CancelTurn) must
    // take the command's whole group with it, not just `sh`.
    #[tokio::test]
    async fn dropping_a_running_command_kills_its_process_group() {
        let secs = 200_000 + std::process::id() % 100_000;
        let run = super::run_bounded(
            sh(&format!("sleep {secs} & sleep 300")),
            "`dropped`",
            std::time::Duration::from_secs(600),
        );
        assert!(tokio::time::timeout(std::time::Duration::from_millis(500), run).await.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let left = std::process::Command::new("pgrep")
            .args(["-f", &format!("^sleep {secs}$")])
            .output()
            .unwrap();
        assert!(
            left.stdout.is_empty(),
            "survivor: {}",
            String::from_utf8_lossy(&left.stdout)
        );
    }

    // Two tasks edit one file: the one that writes second, from text the first had
    // since changed, is refused instead of silently reverting that change -- and goes
    // through once it has read the current text. Creating a file needs no read.
    #[tokio::test]
    async fn a_write_from_stale_text_is_refused() {
        let dir = std::env::temp_dir().join(format!("corrode-occ-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.rs"), "mod a;\n").unwrap();
        let task = || {
            ToolBox::new(
                Arc::new(crate::vfs::PassthroughVfs::new(&dir)),
                dir.clone(),
                Arc::new(HashMap::new()),
            )
        };
        let call = |name: &str, args: serde_json::Value| ToolCall {
            name: name.to_string(),
            arguments: args,
        };
        let (a, b) = (task(), task());
        let read = call("read_file", serde_json::json!({"path": "lib.rs"}));
        a.execute(&read).await;
        b.execute(&read).await;
        let b_write = call("write_file", serde_json::json!({"path": "./lib.rs", "contents": "mod a;\nmod b;\n"}));
        assert!(b.execute(&b_write).await.starts_with("wrote"));
        let a_write = call("write_file", serde_json::json!({"path": "lib.rs", "contents": "mod a;\nmod c;\n"}));
        let refused = a.execute(&a_write).await;
        assert!(refused.contains("changed since you last read it"), "{refused}");
        assert_eq!(std::fs::read_to_string(dir.join("lib.rs")).unwrap(), "mod a;\nmod b;\n");
        a.execute(&read).await;
        assert!(a.execute(&a_write).await.starts_with("wrote"), "after re-reading");
        let fresh = call("write_file", serde_json::json!({"path": "new.rs", "contents": "x"}));
        assert!(b.execute(&fresh).await.starts_with("wrote"), "a new file needs no read");
        std::fs::remove_dir_all(&dir).ok();
    }

    // Unbounded output is drained but not kept: head and tail survive, the middle
    // is dropped with a count, and the exit status still comes through.
    #[tokio::test]
    async fn command_output_is_capped_head_and_tail() {
        let raw = super::read_capped(std::io::Cursor::new({
            let mut v = b"FIRST\n".to_vec();
            v.extend(std::iter::repeat_n(b'x', 5 * super::COMMAND_CAPTURE_HALF));
            v.extend_from_slice(b"\nLAST");
            v
        }))
        .await;
        let text = String::from_utf8_lossy(&raw);
        assert!(text.starts_with("FIRST"), "head lost");
        assert!(text.ends_with("LAST"), "tail lost");
        assert!(text.contains("bytes omitted"));
        assert!(raw.len() <= 2 * super::COMMAND_CAPTURE_HALF + 64);
        let out = super::run_bounded(
            sh("head -c 20000000 /dev/zero | tr '\\0' y; exit 3"),
            "`flood`",
            std::time::Duration::from_secs(60),
        )
        .await;
        assert!(out.starts_with("exit 3"), "{}", &out[..out.len().min(200)]);
    }

    // A native emitter was observed sending `write_file` with `path` and no `contents`;
    // it reached execution and came back as a per-tool error written by hand. The check
    // is schema-driven so every tool gets it, and it names what arrived so the model can
    // correct rather than guess.
    #[tokio::test]
    async fn a_call_missing_a_required_argument_is_refused_before_execution() {
        let dir = std::env::temp_dir().join(format!("corrode-req-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("keep.txt"), "original").unwrap();
        let toolbox = ToolBox::new(
            Arc::new(crate::vfs::PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let call = |name: &str, args: serde_json::Value| ToolCall {
            name: name.to_string(),
            arguments: args,
        };

        let out = toolbox
            .execute(&call("write_file", serde_json::json!({"path": "keep.txt"})))
            .await;
        assert!(out.starts_with("error:"), "{out}");
        assert!(out.contains("contents"), "must name the missing argument: {out}");
        assert!(out.contains("path"), "must name what did arrive: {out}");
        // Refused BEFORE execution: the file is untouched.
        assert_eq!(std::fs::read_to_string(dir.join("keep.txt")).unwrap(), "original");

        // A non-string value is missing too — the schema says string, and `arg_str`
        // cannot tell the difference, so the model would otherwise get "needs a path".
        let out = toolbox.execute(&call("read_file", serde_json::json!({"path": 3}))).await;
        assert!(out.contains("path"), "{out}");

        // An empty string is PRESENT: truncating a file is a legitimate call.
        let out = toolbox
            .execute(&call("write_file", serde_json::json!({"path": "keep.txt", "contents": ""})))
            .await;
        assert!(!out.starts_with("error:"), "empty contents is a truncation, not an error: {out}");
        assert_eq!(std::fs::read_to_string(dir.join("keep.txt")).unwrap(), "");
    }

    // Contents arrive exactly as sent. Trimming stripped the trailing newline and the
    // first line's indentation from every file the swarm wrote, and contents a server
    // had JSON-coerced (a file that parses as JSON) were refused as missing.
    // Through the tool seam: a file read shows its markup neutralized, and the
    // model writing back what it was shown stores the original bytes.
    #[tokio::test]
    async fn a_file_read_shows_neutralized_markup_and_writing_it_back_restores_it() {
        let dir = std::env::temp_dir().join(format!("corrode-markup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = "template ends with <|im_end|> and opens <tool_call>\n";
        std::fs::write(dir.join("t.txt"), original).unwrap();
        let toolbox = ToolBox::new(
            Arc::new(crate::vfs::PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let call = |name: &str, args: serde_json::Value| ToolCall {
            name: name.to_string(),
            arguments: args,
        };
        let shown = toolbox.execute(&call("read_file", serde_json::json!({"path": "t.txt"}))).await;
        assert!(!shown.contains("<|im_end|>") && !shown.contains("<tool_call>"), "{shown}");
        assert!(shown.contains(&*neutralize(original)), "{shown}");
        let copied = neutralize(original).into_owned();
        let out = toolbox
            .execute(&call("write_file", serde_json::json!({"path": "u.txt", "contents": copied})))
            .await;
        assert!(out.starts_with("wrote"), "{out}");
        assert_eq!(std::fs::read_to_string(dir.join("u.txt")).unwrap(), original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_file_contents_are_written_exactly_and_json_values_are_taken_as_text() {
        let dir = std::env::temp_dir().join(format!("corrode-exact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toolbox = ToolBox::new(
            Arc::new(crate::vfs::PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let call = |name: &str, args: serde_json::Value| ToolCall {
            name: name.to_string(),
            arguments: args,
        };
        let src = "    fn f() {}\n";
        let out = toolbox
            .execute(&call("write_file", serde_json::json!({"path": "a.rs", "contents": src})))
            .await;
        assert!(out.starts_with("wrote"), "{out}");
        assert_eq!(std::fs::read_to_string(dir.join("a.rs")).unwrap(), src);

        let call_json = call(
            "write_file",
            serde_json::json!({"path": "p.json", "contents": {"name": "x", "version": 1}}),
        );
        assert!(missing_required(&call_json).is_empty(), "coerced contents are present");
        let out = toolbox.execute(&call_json).await;
        assert!(out.starts_with("wrote"), "{out}");
        let back: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("p.json")).unwrap()).unwrap();
        assert_eq!(back, serde_json::json!({"name": "x", "version": 1}));

        let out = toolbox.execute(&call("run_command", serde_json::json!({"command": true}))).await;
        assert!(out.starts_with("exit 0"), "`true` coerced to a bool still runs: {out}");
        // The run's duration closes the observation, after the digest.
        let took = out.lines().last().unwrap_or_default();
        assert!(took.starts_with("(took ") && took.ends_with("s)"), "{out}");
        std::fs::remove_dir_all(&dir).ok();
    }
    use super::*;

    // Markup quoted in an observation reads as text, and comes back out exactly:
    // a model that rewrites a file it read writes the original bytes. A joiner
    // anywhere else is the file's own and is left alone.
    #[test]
    fn neutralized_markup_restores_to_the_original_bytes() {
        let file = "end: <|im_end|>\n<tool_call>{}</tool_call> <think>x</think>\n\
                    <function=f><parameter=p>v</parameter></function> keep\u{2060}me";
        let shown = neutralize(file);
        for m in PROTOCOL_MARKERS {
            assert!(!shown.contains(m), "{m} survived neutralize");
        }
        assert_eq!(restore(&shown), file);
        assert!(matches!(neutralize("plain"), std::borrow::Cow::Borrowed(_)));
        assert!(matches!(restore("plain"), std::borrow::Cow::Borrowed(_)));
    }
    use crate::vfs::PassthroughVfs;
    use serde_json::json;

    #[test]
    fn parse_tool_intent_reads_the_tool_line() {
        let out = "Let me look at the entry point.\nTOOL: read the file src/main.rs\n";
        assert_eq!(
            parse_tool_intent(out).as_deref(),
            Some("read the file src/main.rs")
        );
        // no TOOL line -> final answer, not a tool step
        assert_eq!(parse_tool_intent("Here is my final answer."), None);
        assert_eq!(parse_tool_intent("TOOL:   "), None);
    }

    #[tokio::test]
    async fn toolbox_reads_and_lists_and_reports_errors() {
        let dir = std::env::temp_dir().join(format!("corrode-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.txt"), b"hi there").unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );

        let read = toolbox
            .execute(&ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "hello.txt"}),
            })
            .await;
        assert!(read.contains("hi there"), "got: {read}");

        let list = toolbox
            .execute(&ToolCall {
                name: "list_dir".into(),
                arguments: json!({"path": ""}),
            })
            .await;
        assert!(list.contains("hello.txt"), "got: {list}");

        // unknown tool and a read miss both come back as readable errors.
        let unknown = toolbox
            .execute(&ToolCall {
                name: "delete_everything".into(),
                arguments: json!({}),
            })
            .await;
        assert!(unknown.starts_with("error: unknown tool"), "got: {unknown}");

        let miss = toolbox
            .execute(&ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "nope.txt"}),
            })
            .await;
        assert!(miss.starts_with("error:"), "got: {miss}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn truncation_never_splits_a_multibyte_char() {
        use std::os::unix::process::ExitStatusExt;
        let dir = std::env::temp_dir().join(format!("corrode-tools-mb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 3-byte chars: 4096 % 3 != 0, so a byte-index truncation lands mid-char.
        let big = "€".repeat(MAX_READ_BYTES / 3 + 2);
        std::fs::write(dir.join("mb.txt"), big.as_bytes()).unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let read = toolbox
            .execute(&ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "mb.txt"}),
            })
            .await;
        assert!(read.contains("truncated"), "got: {read}");

        // Command output over the cap keeps head AND tail (a unique marker at each
        // end must survive), elides the middle, and never splits a multibyte char.
        let big = format!("HEADMARK{}TAILMARK", "€".repeat(8192));
        let out = std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: big.into_bytes(),
            stderr: Vec::new(),
        };
        let formatted = format_command_output(out);
        assert!(formatted.contains("elided"), "got: {}", &formatted[..80.min(formatted.len())]);
        assert!(formatted.contains("HEADMARK"), "head lost");
        assert!(formatted.contains("TAILMARK"), "tail lost");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn search_files_finds_matches_and_reports_misses() {
        let dir = std::env::temp_dir().join(format!("corrode-search-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn frobnicate() {}\nlet x = 1;\n").unwrap();
        std::fs::write(dir.join("src/b.rs"), "// calls frobnicate here\n").unwrap();
        // Tracked but not source: a NUL-bearing blob and a minified one-liner. These
        // are the shapes that survived the corpus fix, and they fall to property tests
        // rather than to a path list.
        std::fs::write(dir.join("blob.bin"), b"frobnicate\0\0binary").unwrap();
        std::fs::write(
            dir.join("min.js"),
            format!("var a=1;{} frobnicate;\n", "x".repeat(2000)),
        )
        .unwrap();
        // Untracked: present on disk, absent from the corpus, must never be searched.
        std::fs::write(dir.join("untracked.rs"), "frobnicate untracked\n").unwrap();

        // The corpus is git's answer, so the fixture has to be a repo. Staging is
        // enough — `ls-files` reads the index, no commit (or identity) required.
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(&dir)
                .args(args)
                .output()
                .expect("git")
        };
        git(&["init", "-q"]);
        git(&["add", "src/a.rs", "src/b.rs", "blob.bin", "min.js"]);

        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let run = |q: &str, scope: Option<&str>| {
            let tb = toolbox.clone();
            let q = q.to_string();
            let scope = scope.map(str::to_string);
            async move {
                let args = match scope {
                    Some(s) => json!({ "query": q, "path": s }),
                    None => json!({ "query": q }),
                };
                tb.execute(&ToolCall {
                    name: "search_files".into(),
                    arguments: args,
                })
                .await
            }
        };

        let hit = run("frobnicate", None).await;
        assert!(
            hit.contains("src/a.rs:1:") && hit.contains("src/b.rs:1:"),
            "both tracked sources should match: {hit}"
        );
        assert!(!hit.contains("a.rs:2:"), "non-matching line excluded: {hit}");
        // The corpus fix: on disk and matching, but not tracked, so not searched.
        assert!(!hit.contains("untracked.rs"), "untracked file searched: {hit}");
        // The property filters: both are tracked and both contain the query.
        assert!(!hit.contains("blob.bin"), "binary searched: {hit}");
        assert!(!hit.contains("min.js"), "minified searched: {hit}");
        assert!(hit.contains("skipped 1 binary, 1 minified"), "counts reported: {hit}");

        // `path` narrows to a subtree.
        let scoped = run("frobnicate", Some("src")).await;
        assert!(scoped.contains("src/a.rs:1:"), "{scoped}");
        let elsewhere = run("frobnicate", Some("docs")).await;
        assert!(elsewhere.contains("no matches"), "{elsewhere}");

        let miss = run("nonexistent_zzz", None).await;
        assert!(miss.contains("no matches"), "got: {miss}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // The role subsets are the enforcement: observing roles get zero mutating tools
    // (they can never block on approval), review verifies through skills but has no
    // raw shell, and only the coder holds the full set. Guards the slice indices
    // against an EXEC_TOOLS reorder.
    #[test]
    fn role_tools_slice_by_privilege() {
        use crate::roles::Role;
        let names = |r| {
            role_tools(r)
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(Role::Coder).len(), EXEC_TOOLS.len());
        assert_eq!(
            names(Role::Review),
            vec!["read_file", "list_dir", "search_files", "run_skill_script"]
        );
        // Read-only roles get the observation trio (incl. search) and nothing mutating.
        assert_eq!(names(Role::Research), vec!["read_file", "list_dir", "search_files"]);
        // The sets are array slices, so a tool's position is its privilege. Pin the
        // slices to the tools' effects: a Mutate/Exec tool placed among the leading
        // reads would hand it to every observing role.
        let reads: Vec<&str> =
            EXEC_TOOLS.iter().filter(|t| t.effect == Effect::Read).map(|t| t.name).collect();
        for role in [Role::Research, Role::Architect, Role::Orchestration] {
            assert_eq!(names(role), reads, "{role:?} observes, and observes everything");
        }
        assert!(
            role_tools(Role::Review).iter().all(|t| t.effect != Effect::Mutate),
            "review verifies; it never writes"
        );
        for role in [Role::Research, Role::Architect, Role::Orchestration] {
            assert!(
                role_tools(role).iter().all(|t| {
                    !is_mutating(&ToolCall {
                        name: t.name.into(),
                        arguments: serde_json::json!({}),
                    })
                }),
                "{role:?} must hold no mutating tool"
            );
        }
    }

    #[test]
    fn is_mutating_classifies_write_and_run() {
        let call = |n: &str| ToolCall {
            name: n.into(),
            arguments: json!({}),
        };
        assert!(is_mutating(&call("write_file")));
        assert!(is_mutating(&call("run_command")));
        assert!(!is_mutating(&call("read_file")));
        assert!(!is_mutating(&call("list_dir")));
        assert!(!is_mutating(&call("search_files")));
        // A name no tool declares is gated, not waved through.
        assert!(is_mutating(&call("delete_everything")));
    }

    #[tokio::test]
    async fn toolbox_writes_files_and_runs_commands() {
        let dir = std::env::temp_dir().join(format!("corrode-mut-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );

        let wrote = toolbox
            .execute(&ToolCall {
                name: "write_file".into(),
                arguments: json!({"path": "new.txt", "contents": "made by a tool"}),
            })
            .await;
        assert!(wrote.starts_with("wrote"), "got: {wrote}");
        assert_eq!(
            std::fs::read_to_string(dir.join("new.txt")).unwrap(),
            "made by a tool"
        );

        let ran = toolbox
            .execute(&ToolCall {
                name: "run_command".into(),
                arguments: json!({"command": "echo hello && ls new.txt"}),
            })
            .await;
        assert!(ran.contains("hello") && ran.contains("new.txt"), "got: {ran}");
        assert!(ran.starts_with("exit 0"), "got: {ran}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn run_skill_script_resolves_by_target_and_rejects_escape() {
        let dir = std::env::temp_dir().join(format!("corrode-skillrun-{}", std::process::id()));
        let skill_dir = dir.join("skills/greeter");
        std::fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        std::fs::write(
            skill_dir.join("scripts/greet.sh"),
            "#!/bin/sh\necho hi from greeter\n",
        )
        .unwrap();

        let mut map = HashMap::new();
        map.insert("greeter".to_string(), skill_dir.clone());
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(map),
        );
        let run = |target: &str| {
            let tb = &toolbox;
            let target = target.to_string();
            async move {
                tb.execute(&ToolCall {
                    name: "run_skill_script".into(),
                    arguments: json!({ "target": target }),
                })
                .await
            }
        };

        // Explicit skill/script.
        assert!(run("greeter/greet.sh").await.contains("hi from greeter"));
        // Bare script name resolves against the only skill that has it (Needle often
        // drops the skill name pre-finetune).
        assert!(run("greet.sh").await.contains("hi from greeter"));
        // Unknown skill, missing script, and path escape are readable errors, never run.
        assert!(run("nope/greet.sh").await.contains("no installed skill named"));
        assert!(run("ghost.sh").await.contains("no installed skill has a script"));
        assert!(run("greeter/../../../bin/sh").await.contains("invalid script"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn missing_path_gets_a_corrective_suggestion() {
        let dir = std::env::temp_dir().join(format!("corrode-suggest-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("readme.md"), b"docs").unwrap();
        std::fs::write(dir.join("src/lib.rs"), b"pub fn x() {}").unwrap();
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        let read = |path: &str| {
            let tb = &toolbox;
            let path = path.to_string();
            async move {
                tb.execute(&ToolCall {
                    name: "read_file".into(),
                    arguments: json!({ "path": path }),
                })
                .await
            }
        };

        // A near-miss on the final component names the miss and suggests the neighbor.
        let close = read("src/lib.sr").await;
        assert!(close.contains("no such path 'src/lib.sr'"), "got: {close}");
        assert!(close.contains("Did you mean 'src/lib.rs'?"), "got: {close}");

        // An absent parent falls back to suggesting from the root listing.
        let orphan = read("nope/readme.md").await;
        assert!(orphan.contains("Did you mean 'readme.md'?"), "got: {orphan}");

        // Nothing close -> the miss is named without a bogus suggestion.
        let far = read("zzz.qqq").await;
        assert!(far.contains("no such path 'zzz.qqq'"), "got: {far}");
        assert!(!far.contains("Did you mean"), "got: {far}");

        // list_dir gets the same treatment.
        let listed = toolbox
            .execute(&ToolCall {
                name: "list_dir".into(),
                arguments: json!({"path": "srcs"}),
            })
            .await;
        assert!(listed.contains("Did you mean 'src'?"), "got: {listed}");

        // A path that exists but fails another way keeps the underlying error.
        let is_dir = read("src").await;
        assert!(is_dir.starts_with("error: could not read src:"), "got: {is_dir}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // The values overlay (grammar value constraint, TODO item 4): paths for the
    // read-only tools from a pruned recursive walk, skill names for run_skill_script,
    // free-text params never constrained, and the hard cap dropping the path enum
    // entirely (never partially) when the repo outgrows it.
    #[tokio::test]
    async fn param_values_walks_prunes_caps_and_never_constrains_free_text() {
        let dir = std::env::temp_dir().join(format!("corrode-values-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join(".git/objects")).unwrap();
        std::fs::create_dir_all(dir.join("target/debug")).unwrap();
        std::fs::write(dir.join("readme.md"), b"d").unwrap();
        std::fs::write(dir.join("src/lib.rs"), b"x").unwrap();
        std::fs::write(dir.join(".git/HEAD"), b"ref").unwrap();
        std::fs::create_dir_all(dir.join("skill/scripts")).unwrap();
        std::fs::write(dir.join("skill/scripts/test.sh"), b"#!/bin/sh").unwrap();
        let mut skills = HashMap::new();
        skills.insert("run-tests".to_string(), dir.join("skill"));
        let toolbox = ToolBox::new(
            Arc::new(PassthroughVfs::new(&dir)),
            dir.clone(),
            Arc::new(skills),
        );

        let v = toolbox.param_values().await;
        let paths = v.get(&("read_file".into(), "path".into())).unwrap();
        assert!(paths.contains(&".".to_string()), "root stays reachable: {paths:?}");
        assert!(paths.contains(&"src".to_string()), "dirs included: {paths:?}");
        assert!(paths.contains(&"src/lib.rs".to_string()), "recursive: {paths:?}");
        assert!(paths.contains(&"readme.md".to_string()));
        assert!(
            !paths.iter().any(|p| p.starts_with(".git") || p.starts_with("target")),
            ".git/target pruned, got: {paths:?}"
        );
        assert_eq!(v.get(&("list_dir".into(), "path".into())), Some(paths));
        // The target enum carries resolver-canonical `skill/script` pairs — a bare
        // skill name would be grammar-forced but never resolve.
        assert_eq!(
            v.get(&("run_skill_script".into(), "target".into())),
            Some(&vec!["run-tests/test.sh".to_string()])
        );
        // Free-text params must never carry a constraint (TODO item 4's risk note).
        assert!(v.get(&("write_file".into(), "path".into())).is_none());
        assert!(v.get(&("run_command".into(), "command".into())).is_none());

        // Over the cap: NO path enum at all (fallback to corrective observations),
        // while the skill enum — small and closed — stays.
        for i in 0..MAX_PATH_VALUES {
            std::fs::write(dir.join(format!("f{i}.txt")), b"x").unwrap();
        }
        let v = toolbox.param_values().await;
        assert!(v.get(&("read_file".into(), "path".into())).is_none());
        assert!(v.get(&("list_dir".into(), "path".into())).is_none());
        assert!(v.get(&("run_skill_script".into(), "target".into())).is_some());

        // A walk error (unlistable root) also means no path constraint.
        let broken = ToolBox::new(
            Arc::new(PassthroughVfs::new(dir.join("gone"))),
            dir.clone(),
            Arc::new(HashMap::new()),
        );
        assert!(broken.param_values().await.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn is_mutating_includes_run_skill_script() {
        let call = ToolCall {
            name: "run_skill_script".into(),
            arguments: json!({}),
        };
        assert!(is_mutating(&call));
    }
}
