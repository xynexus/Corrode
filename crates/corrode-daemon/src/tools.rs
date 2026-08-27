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

use crate::dialect::{Param, Tool};
use crate::toolcall::ToolCall;
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// The tool-execution toolset, as canonical (model-agnostic) [`Tool`] data. A model's
/// [`crate::dialect::ToolDialect`] renders these into the schema it expects and maps its
/// call names back to these canonical names. `write_file`/`run_command`/`run_skill_script`
/// are *mutating* — [`is_mutating`] gates them behind human approval before
/// [`ToolBox::execute`] runs.
pub const EXEC_TOOLS: &[Tool] = &[
    Tool {
        name: "read_file",
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
        description: "List the entries of a directory in the repository.",
        params: &[Param {
            name: "path",
            ty: "string",
            description: "Repository-relative directory path.",
            required: true,
        }],
    },
    // Order is load-bearing: role_tools slices contiguous prefixes off this array
    // (observe | +skills | full), so the read-only pair leads, skills sit before the
    // sharp mutating pair.
    Tool {
        name: "run_skill_script",
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
        Role::Review => &EXEC_TOOLS[..3],
        Role::Research | Role::Architect | Role::Orchestration => &EXEC_TOOLS[..2],
    }
}

/// Hard cap on path-enum candidates for the grammar value constraint. hipfire's scan
/// is O(vocab × candidates): 64 ≈ 400 ms per call (tolerable), 256+ stalls — measured
/// in docs/todo/tool-call-judgement.md item 4. Over the cap sends NO path enum.
const MAX_PATH_VALUES: usize = 64;

/// Cap on how many bytes of a file a `read_file` observation carries back into the
/// model's context — enough to be useful without blowing the window.
const MAX_READ_BYTES: usize = 4096;
/// Cap on captured command / script output.
const MAX_CMD_BYTES: usize = 4096;

/// Whether a tool call mutates or executes and so must clear the human approval gate
/// before it runs. Read-only tools (read_file, list_dir) return false.
pub fn is_mutating(call: &ToolCall) -> bool {
    matches!(
        call.name.as_str(),
        "write_file" | "run_command" | "run_skill_script"
    )
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
        }
    }

    /// Confine spawned processes with this sandbox (builder; default is disabled).
    pub fn with_sandbox(mut self, sandbox: crate::sandbox::Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Attribute this session's hipfire calls to a per-user token (builder;
    /// default `None` = the daemon's shared key).
    pub fn with_owner_token(mut self, owner_token: Option<String>) -> Self {
        self.owner_token = owner_token;
        self
    }

    /// The per-user hipfire bearer for `respond` calls in the tool loops.
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
        match call.name.as_str() {
            "read_file" => match arg_str(call, "path") {
                Some(path) => self.read_file(path).await,
                None => "error: read_file needs a `path` argument".to_string(),
            },
            "list_dir" => match arg_str(call, "path") {
                Some(path) => self.list_dir(path).await,
                None => "error: list_dir needs a `path` argument".to_string(),
            },
            "write_file" => match (arg_str(call, "path"), arg_str(call, "contents")) {
                (Some(path), Some(contents)) => self.write_file(path, contents).await,
                _ => "error: write_file needs `path` and `contents` arguments".to_string(),
            },
            "run_command" => match arg_str(call, "command") {
                Some(command) => self.run_command(command).await,
                None => "error: run_command needs a `command` argument".to_string(),
            },
            "run_skill_script" => match arg_str(call, "target") {
                Some(target) => self.run_skill_script(target).await,
                None => "error: run_skill_script needs a `target` argument".to_string(),
            },
            other => format!("error: unknown tool `{other}`"),
        }
    }

    async fn write_file(&self, path: &str, contents: &str) -> String {
        match self.vfs.write(path, contents.as_bytes()).await {
            Ok(()) => format!("wrote {} bytes to {path}", contents.len()),
            Err(e) => format!("error: could not write {path}: {e}"),
        }
    }

    async fn run_command(&self, command: &str) -> String {
        // sandbox.wrap is a no-op when disabled: plain `sh -c <command>`.
        let (prog, args) = self.sandbox.wrap(&self.root, &["sh", "-c", command]);
        let output = tokio::process::Command::new(prog)
            .args(args)
            .current_dir(&self.root)
            .output()
            .await;
        match output {
            Ok(out) => format_command_output(out),
            Err(e) => format!("error: could not run `{command}`: {e}"),
        }
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
        match cmd.output().await {
            Ok(out) => format_command_output(out),
            Err(e) => format!("error: could not run skill script `{target}`: {e}"),
        }
    }

    async fn read_file(&self, path: &str) -> String {
        match self.vfs.read(path).await {
            Ok(bytes) => {
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
                for e in entries {
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
fn arg_str<'a>(call: &'a ToolCall, key: &str) -> Option<&'a str> {
    call.arguments.get(key)?.as_str().map(str::trim)
}

/// Format a finished process's exit code + stdout/stderr into a bounded observation.
fn format_command_output(out: std::process::Output) -> String {
    let mut text = String::new();
    if !out.stdout.is_empty() {
        text.push_str(&String::from_utf8_lossy(&out.stdout));
    }
    if !out.stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&out.stderr));
    }
    if text.len() > MAX_CMD_BYTES {
        text.truncate(floor_char_boundary(&text, MAX_CMD_BYTES));
        text.push_str("\n… (truncated)");
    }
    format!("exit {}:\n{}", out.status.code().unwrap_or(-1), text.trim())
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
    use super::*;
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

        let out = std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: "€".repeat(MAX_CMD_BYTES / 3 + 2).into_bytes(),
            stderr: Vec::new(),
        };
        assert!(format_command_output(out).contains("truncated"));

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
        assert_eq!(names(Role::Review), vec!["read_file", "list_dir", "run_skill_script"]);
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
