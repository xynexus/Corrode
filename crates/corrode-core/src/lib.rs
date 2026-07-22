//! Wire types shared by the Corrode daemon, the web server, and the wasm webui.
//!
//! This crate is the contract at the seams. It links nothing heavy (no hipfire,
//! no HelixDB, no HTTP) so the wasm webui can depend on it too. Keep it that way:
//! transport and engine details belong in the daemon, not here.

use serde::{Deserialize, Serialize};

/// hipfire scheduler priority bands (mirrors hipfire-scheduler `SCHED_PRIORITY_*`).
///
/// The swarm expresses *all* intent through these; it never throttles locally.
/// Realtime preempts, Opportunistic fills idle GPU only. Serialized as the raw
/// u8 the daemon hands to hipfire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "u8", try_from = "u8")]
pub enum Priority {
    Realtime = 0,
    Default = 64,
    Opportunistic = 255,
}

impl Priority {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl From<Priority> for u8 {
    fn from(p: Priority) -> u8 {
        p.as_u8()
    }
}

impl TryFrom<u8> for Priority {
    type Error = &'static str;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Priority::Realtime),
            64 => Ok(Priority::Default),
            255 => Ok(Priority::Opportunistic),
            _ => Err("priority must be 0, 64, or 255"),
        }
    }
}

/// A file materialized by the VFS from a graph node — what the repo/graph explorer
/// renders and what a subagent shell sees on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileNodeView {
    /// git-compliant path, e.g. `src/swarm.rs`.
    pub path: String,
    pub bytes: u64,
    /// Backing HelixDB node id, so the explorer can pivot file -> graph.
    pub node_id: Option<String>,
}

/// A HelixDB graph node projected for the explorer (graph view side of the UI).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNodeView {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub edges_out: Vec<String>,
}

/// webui/web-server -> daemon. The single command channel into the agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AgentCommand {
    /// Free-form instruction; the daemon plans and fans out a swarm.
    Prompt { text: String, priority: Priority },
    /// A keystroke chunk for the wasm virtual terminal's active session.
    TerminalInput { session: String, data: Vec<u8> },
    /// GraphRAG documentation query over HelixDB's vector+graph store.
    DocQuery { question: String },
}

/// daemon -> webui/web-server. Streamed events (websocket in practice).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AgentEvent {
    /// A subagent produced output.
    SubagentOutput { id: u64, text: String },
    /// Terminal frame back to the wasm terminal.
    TerminalOutput { session: String, data: Vec<u8> },
    /// GraphRAG answer with the node ids it grounded on.
    DocAnswer { text: String, grounded_on: Vec<String> },
    Error { message: String },
}
