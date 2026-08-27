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
    /// Scheduler priority band. hipfire's /v1/responses parser reads
    /// `metadata.hipfire_priority` (a top-level `priority` field, the chat-route
    /// convention, would also work); absent/malformed falls back to the default band.
    metadata: serde_json::Value,
    /// Tool declarations. hipfire feeds these to the model's chat template, which
    /// renders the `<tools>` block the model was trained to read — without it the
    /// template's `{% if tools %}` branch never fires and the model is never told a
    /// tool exists. Omitted entirely when we have no tools to declare.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a serde_json::Value>,
    /// Thinking mode. Absent means non-thinking, which is what every Corrode call was
    /// getting by default — a reasoning model run with reasoning switched off.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

#[derive(Deserialize)]
pub struct ResponsesReply {
    #[serde(default)]
    pub output_text: String,
    /// Output items. Reasoning rides here rather than inside `output_text`, so the
    /// answer stays clean and the reasoning is still recoverable.
    #[serde(default)]
    pub output: Vec<OutputItem>,
}

#[derive(Deserialize)]
pub struct OutputItem {
    #[serde(default)]
    pub reasoning_content: String,
}

impl ResponsesReply {
    /// The model's reasoning, if it thought and the server surfaced it.
    pub fn reasoning(&self) -> &str {
        self.output
            .iter()
            .map(|item| item.reasoning_content.as_str())
            .find(|r| !r.is_empty())
            .unwrap_or_default()
    }
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
    /// "query" for retrieval queries, "document" for corpus text — EmbeddingGemma
    /// prepends a different task prompt per type, so mixing them skews cosine.
    /// The server default is "document"; send it explicitly either way.
    input_type: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingsReply {
    #[serde(default)]
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: Vec<f32>,
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
    ///
    /// `owner_token`, when set, overrides the default API key as the bearer for
    /// this one request — hipfire derives its per-owner fairness key from the
    /// authenticated principal, so a per-user hipfire token is what actually
    /// separates one tenant's fair share from another's. `None` uses the daemon's
    /// shared key (all requests attributed to one owner, as before).
    pub async fn respond(
        &self,
        model: &str,
        input: &str,
        priority: Priority,
        owner_token: Option<&str>,
    ) -> anyhow::Result<String> {
        Ok(self
            .respond_full(model, input, priority, owner_token, None, None)
            .await?
            .0)
    }

    /// One completion, declaring tools and/or requesting thinking. Returns
    /// `(answer, reasoning)`.
    ///
    /// Declaring tools is what lets a model emit its OWN tool calls — MiniCPM5 answers
    /// in `<function name="…"><param name="…">` once its template renders the `<tools>`
    /// block, no Needle in the loop. `effort` selects the thinking mode
    /// (`none`/`minimal`/`low`/`medium`/`high`); absent means non-thinking.
    pub async fn respond_full(
        &self,
        model: &str,
        input: &str,
        priority: Priority,
        owner_token: Option<&str>,
        tools: Option<&serde_json::Value>,
        effort: Option<&str>,
    ) -> anyhow::Result<(String, String)> {
        let req = ResponsesRequest {
            model,
            input,
            max_output_tokens: self.max_output_tokens,
            metadata: serde_json::json!({ "hipfire_priority": priority.as_u8() }),
            tools,
            reasoning_effort: effort,
        };
        let mut rb = self
            .http
            .post(format!("{}/v1/responses", self.base_url))
            .json(&req);
        // Per-user hipfire token (fairness) overrides the shared key for this call.
        if let Some(token) = owner_token.or(self.api_key.as_deref()) {
            rb = rb.bearer_auth(token);
        }
        let reply: ResponsesReply = rb.send().await?.error_for_status()?.json().await?;
        let reasoning = reply.reasoning().to_string();
        Ok((reply.output_text, reasoning))
    }

    /// Embed one string (`/v1/embeddings`) — code/doc/skill retrieval is a hipfire
    /// call, not a local index. Wraps [`Self::embed_batch`]; `input_type` stays
    /// "document" so existing corpus-side callers (skill descriptions) are unchanged.
    pub async fn embed(&self, model: &str, input: &str) -> anyhow::Result<Vec<f32>> {
        self.embed_one(model, input, false).await
    }

    /// Embed one retrieval *query* — EmbeddingGemma's query/document task prompts
    /// are asymmetric, so search text must not be embedded as a document.
    pub async fn embed_query(&self, model: &str, input: &str) -> anyhow::Result<Vec<f32>> {
        self.embed_one(model, input, true).await
    }

    async fn embed_one(&self, model: &str, input: &str, query: bool) -> anyhow::Result<Vec<f32>> {
        let texts = [input.to_string()];
        Ok(self.embed_batch(model, &texts, query).await?.remove(0))
    }

    /// Batch-embed via `POST /v1/embeddings`: one request, N vectors, input order.
    /// Empty inputs are rejected server-side (400s the whole batch) — filter first.
    /// Entries over ~2048 tokens 400 too, so chunk before embedding.
    pub async fn embed_batch(
        &self,
        model: &str,
        texts: &[String],
        query: bool,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let req = EmbeddingsRequest {
            model,
            input: texts,
            input_type: if query { "query" } else { "document" },
        };
        let mut rb = self
            .http
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&req);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let reply: EmbeddingsReply = rb.send().await?.error_for_status()?.json().await?;
        let mut data = reply.data;
        data.sort_by_key(|d| d.index); // index is authoritative, not response order
        if data.len() != texts.len() || data.iter().any(|d| d.embedding.is_empty()) {
            anyhow::bail!(
                "embeddings: {} vectors for {} inputs (model {model})",
                data.len(),
                texts.len()
            );
        }
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}
