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

use crate::toolcall::ToolCall;
use crate::vfs::Vfs;
use std::sync::Arc;

/// Needle-native (flat) schema for the read-only toolset: `parameters` is a
/// `name -> {type, description, required}` map (NOT OpenAI `type:object/properties`,
/// which is out-of-distribution for Needle). One tool per turn; role/args come back as
/// a single call.
pub const TOOL_SCHEMAS: &str = r#"[{"name":"read_file","description":"Read the contents of a file in the repository.","parameters":{"path":{"type":"string","description":"Repository-relative file path.","required":true}}},{"name":"list_dir","description":"List the entries of a directory in the repository.","parameters":{"path":{"type":"string","description":"Repository-relative directory path.","required":true}}}]"#;

/// Cap on how many bytes of a file a `read_file` observation carries back into the
/// model's context — enough to be useful without blowing the window.
const MAX_READ_BYTES: usize = 4096;

/// Executes tool calls against the daemon's VFS. Holds a shared `Arc<dyn Vfs>` so it can
/// be owned by the (`'static`) tool-loop future rather than borrowing the daemon.
pub struct ToolBox {
    vfs: Arc<dyn Vfs>,
}

impl ToolBox {
    pub fn new(vfs: Arc<dyn Vfs>) -> Self {
        Self { vfs }
    }

    /// Run one tool call and return an observation string (result or a readable error —
    /// errors go back to the model as text so it can recover, never as a hard failure).
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
            other => format!("error: unknown tool `{other}`"),
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
        let toolbox = ToolBox::new(Arc::new(PassthroughVfs::new(&dir)));

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
}
