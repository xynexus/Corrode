//! Thin client for the hipfire inference daemon.
//!
//! hipfire serves an OpenAI-compatible API on `127.0.0.1:11435`. `/v1/responses`
//! is the primary first-class interface; `/v1/embeddings` and `/v1/rerank` are
//! first-class alongside it and back Corrode's code retrieval over the VFS graph.
//!
//! The scheduler is priority-banded (u8): 0 = realtime, 64 = default,
//! 255 = opportunistic. Every request Corrode makes carries a band so hipfire's
//! continuous, aging batcher can order the swarm without starving foreground work.

use corrode_core::Priority;
use serde::{Deserialize, Serialize};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435";

pub struct Client {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_output_tokens: u32,
}

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    input: &'a str,
    /// Hard cap on generated tokens. Without it a slow model generates until EOS
    /// and one subagent can hog the GPU for minutes, starving the rest of the swarm.
    max_output_tokens: u32,
    // ponytail: hipfire carries scheduler priority on the internal WorkloadSpec,
    // but the exact HTTP wire field for per-request priority isn't a stable
    // documented header yet — passing it in `metadata` and confirming against the
    // daemon's /v1/responses parser is the next step. Upgrade to whatever the
    // daemon actually reads (header vs body field) once pinned down.
    metadata: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ResponsesReply {
    #[serde(default)]
    pub output_text: String,
}

#[derive(Deserialize)]
struct ModelsReply {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

impl Client {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        // ponytail: one cap for every call. Split per-role (a planner wants more
        // than a research skim) once we tune it.
        let max_output_tokens = std::env::var("CORRODE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            max_output_tokens,
        }
    }

    /// The models hipfire currently serves (`GET /v1/models`), by id. Role
    /// assignment resolves against this list.
    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let mut rb = self.http.get(format!("{}/v1/models", self.base_url));
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let reply: ModelsReply = rb.send().await?.error_for_status()?.json().await?;
        Ok(reply.data.into_iter().map(|m| m.id).collect())
    }

    /// One completion over `/v1/responses` at the given priority band.
    pub async fn respond(
        &self,
        model: &str,
        input: &str,
        priority: Priority,
    ) -> anyhow::Result<String> {
        let req = ResponsesRequest {
            model,
            input,
            max_output_tokens: self.max_output_tokens,
            metadata: serde_json::json!({ "hipfire_priority": priority.as_u8() }),
        };
        let mut rb = self
            .http
            .post(format!("{}/v1/responses", self.base_url))
            .json(&req);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let reply: ResponsesReply = rb.send().await?.error_for_status()?.json().await?;
        Ok(reply.output_text)
    }
}
