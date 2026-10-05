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
    /// Context prefixes the swarm's prompts start with, most recent last; see
    /// [`Client::input_items`].
    shared_prefixes: std::sync::Mutex<Vec<String>>,
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
    /// The reply was cut off (`response.incomplete`), with hipfire's reason.
    Incomplete(String),
}

/// Whether a status means "come back shortly": hipfire answers 503 while its
/// worker is respawned (with Retry-After) and 429 to rate-limit; 502/504 are a
/// proxy saying the same. A 500 is a fault, and a 4xx a request that will never
/// succeed: neither is retried here (a 500 gets one task-level retry instead).
fn is_transient(status: reqwest::StatusCode) -> bool {
    use reqwest::StatusCode as S;
    matches!(
        status,
        S::SERVICE_UNAVAILABLE | S::TOO_MANY_REQUESTS | S::BAD_GATEWAY | S::GATEWAY_TIMEOUT
    )
}

/// A request hipfire refused for good (4xx): the prompt does not fit
/// (`context_length_exceeded`), the model does not exist, or the request is
/// malformed. Retrying cannot help, so nothing does.
#[derive(Debug)]
pub struct Rejected {
    pub status: u16,
    pub code: Option<String>,
    pub message: String,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = self.code.as_deref().unwrap_or("rejected");
        write!(f, "hipfire {} {code}: {}", self.status, self.message)
    }
}

impl std::error::Error for Rejected {}

/// A call that ran the whole request timeout (`CORRODE_REQUEST_TIMEOUT_S`).
#[derive(Debug)]
pub struct TimedOut(pub String);

impl std::fmt::Display for TimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipfire request timed out (CORRODE_REQUEST_TIMEOUT_S): {}", self.0)
    }
}

impl std::error::Error for TimedOut {}

/// Whether a failed task is worth running once more: not when hipfire refused the
/// request, the reply was cut off at the output limit (it would be again), or the
/// call already ran the whole request timeout.
pub fn is_retryable(e: &anyhow::Error) -> bool {
    !(e.is::<Rejected>() || e.is::<Truncated>() || e.is::<TimedOut>())
}

/// How long transient failures (503/429, an unreachable server) are retried for:
/// `CORRODE_RETRY_WINDOW_S`, default 180 -- longer than hipfire takes to respawn
/// a worker and reload its model. Past it the call fails.
fn retry_window() -> std::time::Duration {
    std::time::Duration::from_secs(
        std::env::var("CORRODE_RETRY_WINDOW_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(180),
    )
}

/// The wait before retry `attempt` (0-based): the server's Retry-After if it
/// gave one, else 1 s doubling to a 30 s cap -- plus up to 25% jitter, so the
/// swarm's concurrent calls do not come back in lockstep and re-coalesce into
/// the same failure.
fn retry_delay(attempt: u32, retry_after: Option<u64>) -> std::time::Duration {
    let base = retry_after.unwrap_or_else(|| (1u64 << attempt.min(5)).min(30)) * 1000;
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
        % (base / 4 + 1);
    std::time::Duration::from_millis(base + jitter)
}

/// hipfire's error body: `{"error": {"message", "code"}}`, or the raw text.
fn error_body(text: &str) -> (Option<String>, String) {
    let v: serde_json::Value = serde_json::from_str(text).unwrap_or_default();
    let e = &v["error"];
    let message = e["message"].as_str().map(str::to_string).unwrap_or_else(|| text.to_string());
    (e["code"].as_str().map(str::to_string), message)
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

/// A reply hipfire cut off (`status: "incomplete"`): at `max_output_tokens`, or at
/// the KV capacity the session had left. An error, not a reply, so no caller can take
/// half an answer for a whole one -- a coder's `write_file` cut off mid-`contents` used
/// to end its task Done with no file written. Callers that can recover (the tool loop,
/// the planner) downcast to this; the rest fail the task, visibly.
#[derive(Debug)]
pub struct Truncated {
    /// What the model produced before it was cut off (answer, else reasoning).
    pub partial: String,
    pub reason: String,
}

impl std::fmt::Display for Truncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reply cut off before it finished ({}, {} bytes produced)",
            self.reason,
            self.partial.len()
        )
    }
}

impl std::error::Error for Truncated {}

/// Per-call output cap, `CORRODE_MAX_TOKENS` (default 8192). Shared with the tool
/// loop's context guard, which must keep this much of the context free: hipfire clamps
/// a reply to the KV capacity left, so a smaller reserve cut long answers short.
pub fn max_output_tokens() -> u32 {
    std::env::var("CORRODE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192)
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
        "response.reasoning.delta" => Some(SseDelta::Reasoning(
            json.get("delta")?.as_str()?.to_string(),
        )),
        "response.output_text.done" => {
            Some(SseDelta::TextDone(json.get("text")?.as_str()?.to_string()))
        }
        "response.incomplete" => Some(SseDelta::Incomplete(
            json.pointer("/incomplete_details/reason")
                .and_then(|r| r.as_str())
                .unwrap_or("max_output_tokens")
                .to_string(),
        )),
        _ => None,
    }
}

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    /// A string, or role-tagged items when the prompt leads with a shared prefix
    /// (see [`Client::input_items`]).
    input: serde_json::Value,
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
    /// `completed`, or `incomplete` when hipfire cut the reply off (see [`Truncated`]).
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub incomplete_details: Option<serde_json::Value>,
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
    /// `message` or `function_call`. A tool call is its OWN output item in the Responses
    /// shape, not a field on the message — so a reader that only looks at `output_text`
    /// sees an empty answer and no call at all.
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    /// The call's arguments, as a JSON *string* — that is what the API sends, and what
    /// the model actually emitted. Parsed once here rather than re-serialised.
    #[serde(default)]
    pub arguments: String,
}

impl ResponsesReply {
    /// Tool calls the SERVER parsed, as `function_call` output items.
    ///
    /// Preferred over parsing the model's text, because it is the shape hipfire produces
    /// once it has understood the call: no dialect, no per-model syntax, no guessing. An
    /// empty result is not "no call" — an older hipfire, or an architecture whose calls
    /// the server does not parse (measured: zaya still returns `<zyphra_tool_call>` text),
    /// returns nothing here and the caller must still fall back to its dialect.
    pub fn tool_calls(&self) -> Vec<crate::toolcall::ToolCall> {
        self.output
            .iter()
            .filter(|item| item.kind == "function_call" && !item.name.is_empty())
            .map(|item| crate::toolcall::ToolCall {
                name: item.name.clone(),
                arguments: serde_json::from_str(&item.arguments)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
            })
            .collect()
    }

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
        // was clipping real plans/answers mid-output, and 4096 clipped a coder's
        // write_file of a 14 KB document (~4K tokens plus JSON escaping).
        // ponytail: still one cap for every call. Split per-role (a planner wants more
        // than a research skim) once we tune it.
        let max_output_tokens = max_output_tokens();
        let stream = matches!(
            std::env::var("CORRODE_STREAM").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        );
        // Every call is bounded. Nothing else was: no timeout here or in hipfire,
        // so one wedged generation hung its turn forever. The bound covers the
        // whole generation (a non-streamed reply arrives only when it is done), so
        // it is sized for a full max_output_tokens at a busy server's per-session
        // rate, not for a typical call. `CORRODE_REQUEST_TIMEOUT_S`, default 3600.
        let timeout = std::env::var("CORRODE_REQUEST_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s: &u64| s > 0)
            .unwrap_or(3600);
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .expect("reqwest client");
        Self {
            shared_prefixes: std::sync::Mutex::new(Vec::new()),
            http,
            base_url: base_url.into(),
            api_key,
            max_output_tokens,
            stream,
        }
    }

    /// Record a context prefix the swarm's prompts will start with. A few are kept,
    /// since several repos or tenants can be mid-turn at once.
    pub fn register_shared_prefix(&self, prefix: &str) {
        const KEEP: usize = 8;
        let Ok(mut prefixes) = self.shared_prefixes.lock() else {
            return;
        };
        prefixes.retain(|p| p != prefix);
        prefixes.push(prefix.to_string());
        let over = prefixes.len().saturating_sub(KEEP);
        prefixes.drain(..over);
    }

    /// The `/v1/responses` `input` for a prompt. One that starts with a registered
    /// context prefix goes as a system turn (the prefix) plus a user turn (the rest);
    /// anything else as the plain string it is.
    ///
    /// hipfire can only checkpoint a prompt at a chat-turn boundary, and Qwen3.5's
    /// recurrent state cannot be rewound to an arbitrary shared prefix — so sent as
    /// one user message, a prefix byte-identical across the whole swarm was
    /// prefilled again by every request. As its own turn it is prefilled once.
    fn input_items(&self, input: &str) -> serde_json::Value {
        let prefixes = self
            .shared_prefixes
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        let split = prefixes
            .iter()
            .filter(|p| !p.is_empty())
            .filter_map(|p| {
                input
                    .strip_prefix(p.as_str())
                    .map(|rest| (p.as_str(), rest))
            })
            .max_by_key(|(p, _)| p.len());
        match split {
            Some((prefix, rest)) => serde_json::json!([
                {"role": "system", "content": prefix},
                {"role": "user", "content": rest.trim_start()},
            ]),
            None => serde_json::json!(input),
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
    /// `(answer, reasoning, tool_calls)`.
    ///
    /// `tool_calls` are the ones the SERVER parsed, from `function_call` output items.
    /// Empty means hipfire produced none — which is not the same as the model calling
    /// nothing, since an architecture whose calls hipfire does not parse (zaya) returns
    /// its call as text instead. The caller falls back to its dialect on empty.
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
    ) -> anyhow::Result<(String, String, Vec<crate::toolcall::ToolCall>)> {
        self.respond_turns(model, input, &[], priority, owner_token, tools, effort)
            .await
    }

    /// [`Self::respond_full`] continuing a conversation: `turns` are Responses input
    /// items after `prompt` — an assistant `message`, its `function_call`, and that
    /// call's `function_call_output`, per tool step.
    ///
    /// Replayed as turns, each step's input is the previous step's plus what is new,
    /// so hipfire forks the checkpoint the last step left at its end and prefills only
    /// the new turns. Folded into one growing user message instead, the whole
    /// transcript was prefilled again at every step.
    #[allow(clippy::too_many_arguments)]
    pub async fn respond_turns(
        &self,
        model: &str,
        prompt: &str,
        turns: &[serde_json::Value],
        priority: Priority,
        owner_token: Option<&str>,
        tools: Option<&serde_json::Value>,
        effort: Option<&str>,
    ) -> anyhow::Result<(String, String, Vec<crate::toolcall::ToolCall>)> {
        let mut input = self.input_items(prompt);
        if !turns.is_empty() {
            if let serde_json::Value::String(text) = &input {
                input = serde_json::json!([{"role": "user", "content": text}]);
            }
            if let Some(items) = input.as_array_mut() {
                items.extend(turns.iter().cloned());
            }
        }
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
        if reply.status == "incomplete" {
            let reason = reply
                .incomplete_details
                .as_ref()
                .and_then(|d| d.get("reason"))
                .and_then(|r| r.as_str())
                .unwrap_or("incomplete")
                .to_string();
            let partial = answer_or_reasoning(reply.output_text, &reasoning);
            return Err(Truncated { partial, reason }.into());
        }
        let calls = reply.tool_calls();
        Ok((
            answer_or_reasoning(reply.output_text, &reasoning),
            reasoning,
            calls,
        ))
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
        let started = std::time::Instant::now();
        let window = retry_window();
        let mut attempt = 0u32;
        loop {
            let mut rb = self
                .http
                .post(format!("{}/v1/responses", self.base_url))
                .json(body);
            if let Some(token) = owner_token.or(self.api_key.as_deref()) {
                rb = rb.bearer_auth(token);
            }
            let (failure, retry_after) = match rb.send().await {
                Ok(resp) if resp.status().is_success() => return Ok(resp.json().await?),
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = resp
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.trim().parse::<u64>().ok());
                    let (code, message) = error_body(&resp.text().await.unwrap_or_default());
                    if status.is_client_error() && status != reqwest::StatusCode::TOO_MANY_REQUESTS {
                        return Err(Rejected { status: status.as_u16(), code, message }.into());
                    }
                    let err = anyhow::anyhow!("hipfire {status}: {message}");
                    if !is_transient(status) {
                        return Err(err);
                    }
                    (err, retry_after)
                }
                // Unreachable (restarting, not up yet), or the connection dropped
                // before a reply: back off and retry. A call that ran the whole
                // request timeout is not shedding, it is stuck or far too slow --
                // retrying would multiply the wait, so fail it.
                Err(e) if e.is_timeout() => return Err(TimedOut(e.to_string()).into()),
                Err(e) if e.is_connect() || e.is_request() => (e.into(), None),
                Err(e) => return Err(e.into()), // non-transient -> don't retry
            };
            let wait = retry_delay(attempt, retry_after);
            if started.elapsed() + wait > window {
                return Err(failure.context(format!(
                    "hipfire still unavailable after retrying for {}s (CORRODE_RETRY_WINDOW_S)",
                    started.elapsed().as_secs()
                )));
            }
            eprintln!("hipfire: {failure}; retrying in {:.1}s", wait.as_secs_f32());
            tokio::time::sleep(wait).await;
            attempt += 1;
        }
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
            input: self.input_items(input),
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
        let mut cut_off: Option<String> = None;
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
                Some(SseDelta::Incomplete(reason)) => cut_off = Some(reason),
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
                apply(
                    parse_sse_event(&String::from_utf8_lossy(&block)),
                    &mut text,
                    &mut reasoning,
                );
            }
        }
        // A final event without a trailing blank line.
        if !buf.is_empty() {
            let block = String::from_utf8_lossy(&buf);
            if !block.trim().is_empty() {
                apply(parse_sse_event(&block), &mut text, &mut reasoning);
            }
        }
        let answer = answer_or_reasoning(text, &reasoning);
        if let Some(reason) = cut_off {
            return Err(Truncated { partial: answer, reason }.into());
        }
        Ok((answer, reasoning))
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
        let req = RerankRequest {
            model,
            query,
            documents,
        };
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

    /// A `function_call` output item is read without any dialect.
    #[test]
    fn server_parsed_tool_calls_are_read_from_output_items() {
        let reply: ResponsesReply = serde_json::from_str(
            r#"{"output_text":"","output":[
                {"type":"function_call","name":"read_file","arguments":"{\"path\":\"src/lib.rs\"}"}
            ]}"#,
        )
        .unwrap();
        let calls = reply.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "src/lib.rs");
    }

    /// An empty result is "the server parsed none", NOT "the model called nothing" —
    /// zaya returns its call as text, and an older hipfire returns no item at all. The
    /// caller must be free to fall back to its dialect, so this must not invent a call.
    #[test]
    fn a_message_only_reply_yields_no_calls() {
        let reply: ResponsesReply = serde_json::from_str(
            r#"{"output_text":"<zyphra_tool_call><function=read_file>","output":[
                {"type":"message"}
            ]}"#,
        )
        .unwrap();
        assert!(reply.tool_calls().is_empty());
        // And the text survives for the dialect to parse.
        assert!(reply.output_text.contains("zyphra_tool_call"));
    }

    /// Malformed arguments must not drop the call: the name is the part that selects the
    /// tool, and an empty argument map is recoverable where a lost call is not.
    #[test]
    fn a_call_with_unparseable_arguments_still_names_its_tool() {
        let reply: ResponsesReply = serde_json::from_str(
            r#"{"output_text":"","output":[
                {"type":"function_call","name":"list_dir","arguments":"not json"}
            ]}"#,
        )
        .unwrap();
        let calls = reply.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_dir");
    }
    use super::*;

    #[test]
    fn a_streamed_reply_cut_off_is_reported_incomplete() {
        assert_eq!(
            parse_sse_event(
                "event: response.incomplete\ndata: {\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}"
            ),
            Some(SseDelta::Incomplete("max_output_tokens".into()))
        );
    }

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
    fn a_prompt_leading_with_the_shared_prefix_is_sent_as_a_system_turn() {
        let client = Client::new("http://unused", None);
        // Unregistered: the prompt goes as the plain string it always was.
        assert_eq!(
            client.input_items("PREFIX\n\n[role: coder]\ndo it"),
            "PREFIX\n\n[role: coder]\ndo it"
        );

        client.register_shared_prefix("PRE");
        client.register_shared_prefix("PREFIX");
        let items = client.input_items("PREFIX\n\n[role: coder]\ndo it");
        // The longest registered prefix wins, so the split lands where the turn's
        // prefix actually ends.
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], "PREFIX");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "[role: coder]\ndo it");

        assert_eq!(client.input_items("unrelated"), "unrelated");
    }

    /// A fake hipfire whose replies are scripted in order, counting requests.
    async fn scripted(replies: Vec<(u16, Option<&'static str>, &'static str)>) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use axum::{response::IntoResponse, routing::post, Router};
        use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
        let n = Arc::new(AtomicUsize::new(0));
        let (seen, replies) = (n.clone(), Arc::new(replies));
        let app = Router::new().route(
            "/v1/responses",
            post(move || {
                let (seen, replies) = (seen.clone(), replies.clone());
                async move {
                    let i = seen.fetch_add(1, Ordering::SeqCst).min(replies.len() - 1);
                    let (code, retry_after, body) = replies[i];
                    let mut r = (axum::http::StatusCode::from_u16(code).unwrap(), body.to_string()).into_response();
                    if let Some(ra) = retry_after {
                        r.headers_mut().insert("retry-after", ra.parse().unwrap());
                    }
                    r
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), n)
    }

    // A prompt that cannot fit is refused once, typed, and never retried; a 503
    // with Retry-After is retried until it answers; a 500 fails at once (the task
    // gets its own single retry instead).
    #[tokio::test]
    async fn errors_are_classified_and_only_transient_ones_retried() {
        use std::sync::atomic::Ordering;
        const OK: &str = r#"{"status":"completed","output_text":"fine","output":[]}"#;
        let ctx = r#"{"error":{"message":"the prompt is 40000 tokens","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
        let (url, n) = scripted(vec![(400, None, ctx)]).await;
        let err = Client::new(url, None).respond("m", "p", Priority::Default, None).await.unwrap_err();
        let r = err.downcast_ref::<Rejected>().expect("typed rejection");
        assert_eq!(r.code.as_deref(), Some("context_length_exceeded"));
        assert!(r.message.contains("40000"));
        assert!(!is_retryable(&err));
        assert_eq!(n.load(Ordering::SeqCst), 1, "never retried");

        let busy = r#"{"error":{"message":"worker respawning","type":"service_unavailable"}}"#;
        let (url, n) = scripted(vec![(503, Some("0"), busy), (503, Some("0"), busy), (200, None, OK)]).await;
        let out = Client::new(url, None).respond("m", "p", Priority::Default, None).await.unwrap();
        assert_eq!(out, "fine");
        assert_eq!(n.load(Ordering::SeqCst), 3, "retried through the 503s");

        let (url, n) = scripted(vec![(500, None, r#"{"error":{"message":"boom"}}"#), (200, None, OK)]).await;
        let err = Client::new(url, None).respond("m", "p", Priority::Default, None).await.unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert!(is_retryable(&err), "a 500 is worth one task-level retry");
        assert_eq!(n.load(Ordering::SeqCst), 1, "not retried by the client");
    }

    // Against a live hipfire (ignored: needs the server and kills its worker). An
    // oversized prompt is refused at once, typed; a request whose worker is killed
    // mid-generation rides out the 503 and succeeds once the worker is respawned.
    #[tokio::test]
    #[ignore]
    async fn live_rejection_and_retry_through_a_worker_restart() {
        let model = std::env::var("CORRODE_LIVE_MODEL").unwrap_or("Qwen3.6-35B-A3B--oq4.25++".into());
        let c = Client::new("http://127.0.0.1:11435", None);
        let t0 = std::time::Instant::now();
        let err = c.respond(&model, &"word ".repeat(60_000), Priority::Default, None).await.unwrap_err();
        let r = err.downcast_ref::<Rejected>().expect("typed rejection");
        assert_eq!(r.code.as_deref(), Some("context_length_exceeded"), "{err}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "refused, not retried");

        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let _ = std::process::Command::new("sh")
                .args(["-c", "kill -9 $(ps -eo pid,args | awk '$2 ~ /hipfire$/ && $3==\"daemon\" {print $1}')"])
                .status();
        });
        let t0 = std::time::Instant::now();
        let out = c
            .respond(&model, "Write a 300-word essay about prime numbers.", Priority::Default, None)
            .await
            .expect("retried through the restart");
        eprintln!("answered {} bytes after {:.1}s", out.len(), t0.elapsed().as_secs_f32());
        assert!(!out.is_empty());
    }

    #[test]
    fn transient_statuses_retry_client_errors_dont() {
        use reqwest::StatusCode;
        assert!(!is_transient(StatusCode::INTERNAL_SERVER_ERROR)); // 500 — a fault, task-level retry
        assert!(is_transient(StatusCode::SERVICE_UNAVAILABLE)); // 503
        assert!(is_transient(StatusCode::TOO_MANY_REQUESTS)); // 429
        assert!(!is_transient(StatusCode::BAD_REQUEST)); // 400 — our bug
        assert!(!is_transient(StatusCode::UNAUTHORIZED)); // 401
        assert!(!is_transient(StatusCode::OK));
    }

    #[test]
    fn answer_falls_back_to_reasoning_only_when_output_is_blank() {
        // Normal case: a real answer is returned as-is.
        assert_eq!(
            answer_or_reasoning("the answer".into(), "some thinking"),
            "the answer"
        );
        // Think-native model: empty output_text -> recover the answer from reasoning.
        assert_eq!(answer_or_reasoning("   ".into(), "4"), "4");
        // Nothing anywhere stays empty (no spurious fallback).
        assert_eq!(answer_or_reasoning(String::new(), "  "), "");
    }
}
