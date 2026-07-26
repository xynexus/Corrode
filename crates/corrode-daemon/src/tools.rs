//! The tools a subagent can execute, and how a small model reaches them.
//!
//! Small models can't be trusted to construct tool-call JSON, so in the tool-execution
//! loop ([`crate::daemon`]) a small model states its intent in plain English on a
//! `TOOL:` line, and Needle turns that into a structured call against [`TOOL_SCHEMAS`].
//! [`ToolBox`] then executes the call against the daemon's real capabilities and hands
//! back an observation the model reads on its next turn.
//!
//! The initial toolset is deliberately READ-ONLY (`read_file`, `list_dir`) — a swarm of
//! small models acting on the repo should observe before it can mutate. ponytail:
//! `write_file` / `run_command` need sandboxing + the daemon's action-approval gate
//! before they join the set.

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
];

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
pub struct ToolBox {
    vfs: Arc<dyn Vfs>,
    root: PathBuf,
    /// Skill name -> skill directory, for resolving `run_skill_script` (stage 3).
    skill_scripts: Arc<HashMap<String, PathBuf>>,
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
        }
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
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
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

        let mut cmd = match interpreter_for(&path) {
            Some(interp) => {
                let mut c = tokio::process::Command::new(interp);
                c.arg(&path);
                c
            }
            None => tokio::process::Command::new(&path),
        };
        cmd.current_dir(&self.root);
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
                    (&text[..MAX_READ_BYTES], true)
                } else {
                    (text.as_ref(), false)
                };
                let mut out = format!("contents of {path}:\n{shown}");
                if truncated {
                    out.push_str("\n… (truncated)");
                }
                out
            }
            Err(e) => format!("error: could not read {path}: {e}"),
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
            Err(e) => format!("error: could not list {path}: {e}"),
        }
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
        text.truncate(MAX_CMD_BYTES);
        text.push_str("\n… (truncated)");
    }
    format!("exit {}:\n{}", out.status.code().unwrap_or(-1), text.trim())
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

    #[test]
    fn is_mutating_includes_run_skill_script() {
        let call = ToolCall {
            name: "run_skill_script".into(),
            arguments: json!({}),
        };
        assert!(is_mutating(&call));
    }
}
