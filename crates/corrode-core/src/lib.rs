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

/// Why a Rust file that should have been *composed* (regenerated from graph nodes)
/// instead fell back to verbatim flat+overlay: its parse -> regenerate round-trip
/// wasn't byte-identical. Typed (not a free-text string) so fallbacks *aggregate* —
/// "30 of 37 fallbacks are `MacroExpansion`" is the weak-projector signal. A growing
/// fallback set is surfaced deliberately, never a silent degrade.
// ponytail: no producer yet — emitted by the composed-Rust ingest round-trip check
// (parse -> regenerate -> diff) when the graph-backed VFS lands. See the design memo.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackReason {
    /// `#[rustfmt::skip]` — the region is intentionally non-canonical.
    RustfmtSkip,
    /// A macro invocation whose expansion the projector can't reproduce verbatim.
    MacroExpansion,
    /// A raw/byte string literal that didn't survive regeneration byte-for-byte.
    RawStringMismatch,
    /// Attribute placement rustfmt won't canonicalize to the projector's form.
    AttributePlacement,
    /// Uncategorized divergence — carries the first byte offset where regeneration
    /// diverged from the original, i.e. the file to go read.
    UnknownDivergence { first_diff_offset: u64 },
}

/// How the VFS backs a given file. `Composed` is Rust regenerated from graph nodes
/// with a verified byte-identical round-trip; `Overlay` is the *native* flat-node +
/// function-overlay mode (C/C++/Python/Bash/Markdown), source of truth on disk;
/// `OverlayFallback` is Rust that attempted composition but failed its round-trip
/// and dropped to overlay — the reason is what makes the fallback set visible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    Composed,
    Overlay,
    OverlayFallback(FallbackReason),
}

/// A file materialized by the VFS from a graph node — what the repo/graph explorer
/// renders and what a subagent shell sees on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileNodeView {
    /// git-compliant path, e.g. `src/swarm.rs`.
    pub path: String,
    /// True for a directory entry. Files and dirs both appear in a listing, and a
    /// FUSE mount / the explorer must tell them apart (an empty file also has 0
    /// bytes, so size can't stand in for kind).
    pub is_dir: bool,
    pub bytes: u64,
    /// Backing HelixDB node id, so the explorer can pivot file -> graph.
    pub node_id: Option<String>,
    /// How the VFS backs this file, so the explorer can flag which Rust files
    /// silently dropped to overlay. `None` until the graph-backed VFS sets it
    /// (the filesystem passthrough can't know), mirroring `node_id`.
    pub mode: Option<ProjectionMode>,
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
    /// Explorer: list the VFS entries under a directory path.
    ListDir { path: String },
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
    /// Explorer: entries under a listed directory.
    DirListing { path: String, entries: Vec<FileNodeView> },
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_mode_round_trips_through_json() {
        // The explorer parses this off the wire, so every variant — including the
        // nested reason with its payload — must survive serialize -> deserialize.
        let cases = [
            ProjectionMode::Composed,
            ProjectionMode::Overlay,
            ProjectionMode::OverlayFallback(FallbackReason::RustfmtSkip),
            ProjectionMode::OverlayFallback(FallbackReason::MacroExpansion),
            ProjectionMode::OverlayFallback(FallbackReason::RawStringMismatch),
            ProjectionMode::OverlayFallback(FallbackReason::AttributePlacement),
            ProjectionMode::OverlayFallback(FallbackReason::UnknownDivergence {
                first_diff_offset: 4096,
            }),
        ];
        for mode in cases {
            let json = serde_json::to_string(&mode).unwrap();
            let back: ProjectionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back, "round-trip changed {json}");
        }
    }

    #[test]
    fn projection_mode_wire_shape_is_externally_tagged() {
        // Pin the exact JSON the webui contract depends on; a stray serde attribute
        // (rename/tagging change) would break the front-end silently otherwise.
        assert_eq!(
            serde_json::to_string(&ProjectionMode::Composed).unwrap(),
            r#""Composed""#
        );
        assert_eq!(
            serde_json::to_string(&ProjectionMode::OverlayFallback(
                FallbackReason::UnknownDivergence { first_diff_offset: 7 }
            ))
            .unwrap(),
            r#"{"OverlayFallback":{"UnknownDivergence":{"first_diff_offset":7}}}"#
        );
    }
}
