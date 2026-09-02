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
    /// Stream `/v1/responses` (SSE) so subagent output reaches the UI as it's
    /// generated, not in one block at the end. Off by default (`CORRODE_STREAM`).
    stream: bool,
}

/// One decoded SSE event from hipfire's `/v1/responses` stream. hipfire tags each
/// event both with an `event:` name and a `"type"` inside the JSON `data:`; we key
/// off the JSON type, so the `event:` line is ignored.
#[derive(Debug, PartialEq)]
enum SseDelta {
    /// An incremental chunk of the answer.
    Text(String),
    /// An incremental chunk of reasoning (streamed separately from the answer).
    Reasoning(String),
    /// The final, authoritative full answer text (`response.output_text.done`).
    TextDone(String),
}

/// Whether a response status is transient overload worth retrying (hipfire sheds
/// with 5xx under memory pressure; 429 is explicit rate-limit). A 4xx is our bug —
/// don't retry it.
fn is_transient(status: reqwest::StatusCode) -> bool {
    status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

/// The usable answer from a `/v1/responses` reply. Some hipfire model builds (the
/// think-native / think-filtered quants seen in practice) route the whole answer
/// into `reasoning_content` and leave `output_text` empty; an empty answer is
/// useless to the swarm, so fall back to the reasoning when the answer channel is
/// blank. When `output_text` is present it wins untouched.
fn answer_or_reasoning(output_text: String, reasoning: &str) -> String {
    if output_text.trim().is_empty() && !reasoning.trim().is_empty() {
        reasoning.to_string()
    } else {
        output_text
    }
}

/// Parse ONE complete SSE event block (the text between `\n\n` boundaries) into a
/// delta, if it carries one. Concatenates multiple `data:` lines per the SSE spec;
/// non-JSON payloads (e.g. `[DONE]`) and uninteresting events yield `None`.
///
/// Classification keys off the `event:` line, NOT the JSON `type`: hipfire builds
/// every delta's `data` with the same helper, so the JSON `type` is always
/// `response.output_text.delta` even for a reasoning delta — only the `event:` name
/// distinguishes them. Falls back to the JSON `type` if there's no `event:` line
/// (a spec-compliant server that omits it).
fn parse_sse_event(block: &str) -> Option<SseDelta> {
    let mut event_name = "";
    let mut data_parts: Vec<&str> = Vec::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            event_name = v.trim();
        } else if let Some(v) = line.strip_prefix("data:") {
            data_parts.push(v.trim());
        }
    }
    let data = data_parts.join("\n");
    if data.is_empty() {
        return None;
    }
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let kind = if event_name.is_empty() {
        json.get("type").and_then(|t| t.as_str())?
    } else {
        event_name
    };
    match kind {
        "response.output_text.delta" => {
            Some(SseDelta::Text(json.get("delta")?.as_str()?.to_string()))
        }
        "response.reasoning.delta" => {
            Some(SseDelta::Reasoning(json.get("delta")?.as_str()?.to_string()))
        }
        "response.output_text.done" => {
            Some(SseDelta::TextDone(json.get("text")?.as_str()?.to_string()))
        }
        _ => None,
    }
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

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
}

#[derive(Deserialize)]
struct RerankReply {
    results: Vec<RerankHit>,
    /// Which scorer ran: `cross-encoder` (one forward per pair, `yes` against `no`) or
    /// `cosine` (bi-encoder). hipfire picks by loaded model, so asking for reranking
    /// with an embedding model silently gets cosine — which is the same computation
    /// this crate could do itself, and cannot express "matches on two axes at once".
    /// Checked rather than assumed.
    #[serde(default)]
    mode: String,
}

#[derive(Deserialize)]
struct RerankHit {
    index: usize,
    relevance_score: f32,
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
        // A ceiling, not a target: models stop at EOS, so a higher cap costs nothing
        // for short outputs (a TOOL: line) and only spares long ones (a multi-task
        // plan, a synthesized doc answer) from truncation — 1024 tokens (~750 words)
        // was clipping real plans/answers mid-output.
        // ponytail: still one cap for every call. Split per-role (a planner wants more
        // than a research skim) once we tune it.
        let max_output_tokens = std::env::var("CORRODE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let stream = matches!(
            std::env::var("CORRODE_STREAM").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        );
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            max_output_tokens,
            stream,
        }
    }

    /// Whether `/v1/responses` should be streamed (SSE). Callers that want to relay
    /// deltas to the UI branch on this and use [`Self::respond_streaming`].
    pub fn streaming(&self) -> bool {
        self.stream
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
        let body = serde_json::to_value(&req)?;
        let reply = self.post_responses(&body, owner_token).await?;
        let reasoning = reply.reasoning().to_string();
        Ok((answer_or_reasoning(reply.output_text, &reasoning), reasoning))
    }

    /// POST a `/v1/responses` body, retrying on transient server-side shedding.
    ///
    /// hipfire sheds under memory pressure by returning 5xx (or 429), not by queuing,
    /// and the reactive scheduler fires many tasks at once — so a burst can shed even
    /// though each call fits when run sequentially (observed on a 30 GiB box: a lone
    /// call succeeds while the swarm's concurrent burst 500s). Back off and retry,
    /// letting earlier calls finish and free memory, rather than failing the task.
    /// 4xx (except 429) and other errors are returned immediately — only transient
    /// overload gets the backoff.
    async fn post_responses(
        &self,
        body: &serde_json::Value,
        owner_token: Option<&str>,
    ) -> anyhow::Result<ResponsesReply> {
        const MAX_ATTEMPTS: u32 = 4;
        let mut backoff = std::time::Duration::from_millis(400);
        let mut last: Option<anyhow::Error> = None;
        for attempt in 0..MAX_ATTEMPTS {
            let mut rb = self
                .http
                .post(format!("{}/v1/responses", self.base_url))
                .json(body);
            if let Some(token) = owner_token.or(self.api_key.as_deref()) {
                rb = rb.bearer_auth(token);
            }
            match rb.send().await {
                Ok(resp) if is_transient(resp.status()) => {
                    last = Some(anyhow::anyhow!("hipfire sched-shed {}", resp.status()));
                }
                Ok(resp) => return Ok(resp.error_for_status()?.json().await?),
                Err(e) if e.is_timeout() || e.is_connect() => last = Some(e.into()),
                Err(e) => return Err(e.into()), // non-transient -> don't retry
            }
            if attempt + 1 < MAX_ATTEMPTS {
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }
        }
        Err(last.unwrap_or_else(|| anyhow::anyhow!("hipfire: retries exhausted")))
    }

    /// Like [`Self::respond`], but streams: `on_delta` is called with each incremental
    /// chunk as it arrives — answer AND reasoning deltas both, so the UI shows
    /// progress even on models that emit only reasoning — and the accumulated
    /// `(answer, reasoning)` is returned at the end (answer falls back to reasoning
    /// when the model left `output_text` empty, same as [`Self::respond_full`]). A
    /// non-empty `response.output_text.done` wins over the concatenated answer
    /// deltas. `on_delta` is sync and best-effort (a UI relay uses `try_send`); it
    /// must not block.
    pub async fn respond_streaming(
        &self,
        model: &str,
        input: &str,
        priority: Priority,
        owner_token: Option<&str>,
        mut on_delta: impl FnMut(&str),
    ) -> anyhow::Result<(String, String)> {
        use futures_util::StreamExt;

        let req = ResponsesRequest {
            model,
            input,
            max_output_tokens: self.max_output_tokens,
            metadata: serde_json::json!({ "hipfire_priority": priority.as_u8() }),
            tools: None,
            reasoning_effort: None,
        };
        // `stream` is a top-level field on the responses request; ResponsesRequest
        // doesn't carry it, so merge it into the serialized object.
        let mut body = serde_json::to_value(&req)?;
        body["stream"] = serde_json::Value::Bool(true);

        let mut rb = self
            .http
            .post(format!("{}/v1/responses", self.base_url))
            .json(&body);
        if let Some(token) = owner_token.or(self.api_key.as_deref()) {
            rb = rb.bearer_auth(token);
        }
        let resp = rb.send().await?.error_for_status()?;

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut apply = |ev: Option<SseDelta>, text: &mut String, reasoning: &mut String| {
            match ev {
                Some(SseDelta::Text(d)) => {
                    on_delta(&d);
                    text.push_str(&d);
                }
                // Relay reasoning to the UI too (so streaming is visible even on models
                // that emit everything as reasoning), but keep it in the reasoning
                // channel — the final answer reconciles via SubagentOutput.
                Some(SseDelta::Reasoning(d)) => {
                    on_delta(&d);
                    reasoning.push_str(&d);
                }
                // Authoritative only when non-empty: some models send an empty `done`
                // and carry the real answer in the deltas we accumulated.
                Some(SseDelta::TextDone(full)) if !full.is_empty() => *text = full,
                Some(SseDelta::TextDone(_)) | None => {}
            }
        };
        // Buffer BYTES, not a lossy string: a chunk can split a multi-byte UTF-8 char
        // at its boundary, and decoding each chunk in isolation would corrupt it to
        // U+FFFD. SSE events end at ASCII "\n\n", so decode only complete blocks.
        let mut buf: Vec<u8> = Vec::new();
        let mut bytes = resp.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            buf.extend_from_slice(&chunk?);
            while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buf.drain(..pos + 2).collect();
                apply(parse_sse_event(&String::from_utf8_lossy(&block)), &mut text, &mut reasoning);
            }
        }
        // A final event without a trailing blank line.
        if !buf.is_empty() {
            let block = String::from_utf8_lossy(&buf);
            if !block.trim().is_empty() {
                apply(parse_sse_event(&block), &mut text, &mut reasoning);
            }
        }
        Ok((answer_or_reasoning(text, &reasoning), reasoning))
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
    /// Rerank `documents` against `query` (`/v1/rerank`), best first.
    ///
    /// Returns `(index, score)` into `documents`. This is a CROSS-encoder call: hipfire
    /// scores each pair jointly rather than embedding the two sides separately, which is
    /// the difference that matters for near-identical candidates — a blended vector
    /// cannot express "matches on two axes at once".
    ///
    /// Errors if the server answers in `cosine` mode. That is not a failure of the
    /// server; it means the named model is a bi-encoder, and silently accepting it would
    /// return embedding similarity under the name of reranking.
    pub async fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: &[String],
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let req = RerankRequest { model, query, documents };
        let mut rb = self
            .http
            .post(format!("{}/v1/rerank", self.base_url))
            .json(&req);
        if let Some(key) = &self.api_key {
            rb = rb.bearer_auth(key);
        }
        let reply: RerankReply = rb.send().await?.error_for_status()?.json().await?;
        if reply.mode == "cosine" {
            anyhow::bail!(
                "rerank: {model} scored in cosine mode — a bi-encoder, not a cross-encoder"
            );
        }
        Ok(reply
            .results
            .into_iter()
            .filter(|h| h.index < documents.len())
            .map(|h| (h.index, h.relevance_score))
            .collect())
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_output_reasoning_and_done_events_ignores_the_rest() {
        // Real hipfire format: the JSON `type` is ALWAYS output_text.delta; only the
        // `event:` line distinguishes an answer delta from a reasoning delta.
        assert_eq!(
            parse_sse_event(
                "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}"
            ),
            Some(SseDelta::Text("Hel".into()))
        );
        // Same JSON type, but the reasoning event name -> classified as reasoning.
        assert_eq!(
            parse_sse_event(
                "event: response.reasoning.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"think\"}"
            ),
            Some(SseDelta::Reasoning("think".into()))
        );
        // A `done` may be empty (the answer came via deltas) — parsed as-is; the
        // accumulator ignores an empty one so it can't clobber the deltas.
        assert_eq!(
            parse_sse_event(
                "event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"text\":\"\"}"
            ),
            Some(SseDelta::TextDone(String::new()))
        );
        // Fallback to JSON `type` when there's no event: line (spec-compliant server).
        assert_eq!(
            parse_sse_event("data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}"),
            Some(SseDelta::Text("x".into()))
        );
        // Uninteresting events and non-JSON payloads are skipped.
        assert_eq!(
            parse_sse_event(
                "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}"
            ),
            None
        );
        assert_eq!(parse_sse_event("data: [DONE]"), None);
        assert_eq!(parse_sse_event(": keep-alive comment"), None);
    }

    #[test]
    fn transient_statuses_retry_client_errors_dont() {
        use reqwest::StatusCode;
        assert!(is_transient(StatusCode::INTERNAL_SERVER_ERROR)); // 500 — hipfire shed
        assert!(is_transient(StatusCode::SERVICE_UNAVAILABLE)); // 503
        assert!(is_transient(StatusCode::TOO_MANY_REQUESTS)); // 429
        assert!(!is_transient(StatusCode::BAD_REQUEST)); // 400 — our bug
        assert!(!is_transient(StatusCode::UNAUTHORIZED)); // 401
        assert!(!is_transient(StatusCode::OK));
    }

    #[test]
    fn answer_falls_back_to_reasoning_only_when_output_is_blank() {
        // Normal case: a real answer is returned as-is.
        assert_eq!(answer_or_reasoning("the answer".into(), "some thinking"), "the answer");
        // Think-native model: empty output_text -> recover the answer from reasoning.
        assert_eq!(answer_or_reasoning("   ".into(), "4"), "4");
        // Nothing anywhere stays empty (no spurious fallback).
        assert_eq!(answer_or_reasoning(String::new(), "  "), "");
    }
}
