//! Reliable tool-calling for small models, via the Needle shim.
//!
//! Small chat models (the swarm's fast tier, e.g. zaya) botch tool-call formatting:
//! malformed JSON, invented argument keys, hallucinated tool names. Needle is a
//! tiny encoder-decoder trained for exactly one contract —
//!
//! ```text
//! query + tools JSON -> JSON tool calls
//! ```
//!
//! — with *grammar-guided* decoding that constrains the output to valid tool names,
//! argument keys, enum/boolean/null values, and JSON structure. So rather than
//! trust a small chat model to emit a call, the daemon routes the tool-selection to
//! Needle and gets back structurally-valid calls.
//!
//! Kept behind a trait that is *always* defined, with the Needle implementation
//! feature-gated (`--features needle`) — mirroring [`crate::graph::GraphStore`]. The
//! daemon holds an `Option<Arc<dyn ToolCaller>>`, and the base build never compiles
//! candle. Enable the real backend with `--features needle` and point
//! `CORRODE_NEEDLE_ASSETS` at the Needle asset dir.
//!
//! ponytail: [`ToolCall`] lives here for now; it moves to `corrode-core` once the
//! tool-execution loop runs calls and the webui surfaces them over the wire.

use serde::{Deserialize, Serialize};

/// One tool invocation Needle decoded: a tool name and its JSON arguments.
// ponytail: dead in the base build until the tool-execution loop consumes calls;
// exercised by tests and the `needle` backend now.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Generates a model's raw tool-call reply for a request + a (dialect-rendered) tool
/// schema. The daemon depends on the trait, not the model, so the base build stays free
/// of candle and the backend can be swapped. Rendering the schema and *parsing* the
/// reply are the [`crate::dialect::ToolDialect`]'s job, keyed by [`ToolCaller::model_id`]
/// — so different tool-call models can use different names/formats.
pub trait ToolCaller: Send + Sync {
    /// The model's raw reply for `query` given `tools_json` (already in the model's
    /// dialect). The caller parses it with the matching dialect.
    fn generate(&self, query: &str, tools_json: &str) -> anyhow::Result<String>;

    /// Identifier used to resolve this caller's tool dialect (schema/names/parse).
    fn model_id(&self) -> &str;
}

/// The Needle backend. Only compiled with `--features needle`, which pulls in the
/// (CPU/candle) `needle-toolcall-shim` crate.
#[cfg(feature = "needle")]
pub mod needle {
    use super::*;
    use needle_toolcall_shim::model::{Assets, NeedleModel};
    use needle_toolcall_shim::tokenizer::NeedleTokenizer;
    use std::path::Path;
    use std::sync::Mutex;

    /// Fallback asset dir when `CORRODE_NEEDLE_ASSETS` is unset: the vendored crate's
    /// committed weights, resolved at build time relative to this crate so the default
    /// works regardless of the daemon's runtime CWD.
    const DEFAULT_ASSETS: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../needle-toolcall-shim/assets/needle");
    /// Generation cap — tool-call JSON is short; this is a runaway backstop.
    const MAX_GEN_LEN: usize = 512;

    /// Loaded Needle model + tokenizer behind a `Mutex`: inference is synchronous and
    /// CPU-bound, so calls serialize (Needle is small and fast, one-at-a-time is fine)
    /// and the `Mutex` makes the caller `Sync` regardless of the tokenizer's internals.
    pub struct NeedleToolCaller {
        inner: Mutex<Inner>,
        /// Dialect key (see `CORRODE_NEEDLE_MODEL_ID`, default `needle`). Lets different
        /// Needle finetunes resolve to different tool dialects.
        model_id: String,
    }

    struct Inner {
        model: NeedleModel,
        tokenizer: NeedleTokenizer,
    }

    impl NeedleToolCaller {
        /// Load model + tokenizer from an asset dir (see the shim's README layout).
        pub fn load(assets_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
            let assets = Assets::resolve(assets_dir)?;
            let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
            let model = NeedleModel::load(&assets)?;
            Ok(Self {
                inner: Mutex::new(Inner { model, tokenizer }),
                model_id: std::env::var("CORRODE_NEEDLE_MODEL_ID")
                    .unwrap_or_else(|_| "needle".to_string()),
            })
        }

        /// Load from `CORRODE_NEEDLE_ASSETS` (falling back to `assets/needle`).
        /// Returns `Ok(None)` when the asset dir is absent, so a daemon built with the
        /// feature but run without the weights degrades to "no tool-caller" rather
        /// than failing startup.
        pub fn load_from_env() -> anyhow::Result<Option<Self>> {
            let dir = std::env::var("CORRODE_NEEDLE_ASSETS")
                .unwrap_or_else(|_| DEFAULT_ASSETS.to_string());
            if !Path::new(&dir).join("config.json").exists() {
                return Ok(None);
            }
            Ok(Some(Self::load(dir)?))
        }
    }

    impl ToolCaller for NeedleToolCaller {
        fn generate(&self, query: &str, tools_json: &str) -> anyhow::Result<String> {
            let inner = self
                .inner
                .lock()
                .map_err(|_| anyhow::anyhow!("Needle tool-caller mutex poisoned"))?;
            // Guided + normalized: constrain to valid JSON/tool structure and accept
            // loosely-cased tool names (the shim maps them back). The dialect parses.
            inner.model.generate(
                &inner.tokenizer,
                query,
                tools_json,
                MAX_GEN_LEN,
                /* guided */ true,
                /* normalize */ true,
            )
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // End-to-end against the real Needle weights. Ignored by default (needs the
        // asset dir); run with the weights on hand:
        //   CORRODE_NEEDLE_ASSETS=~/build/needle-toolcall-shim/assets/needle \
        //     cargo test -p corrode-daemon --features needle -- --ignored --nocapture
        #[test]
        #[ignore = "requires Needle assets (set CORRODE_NEEDLE_ASSETS)"]
        fn decodes_a_real_tool_call() {
            let caller = NeedleToolCaller::load_from_env()
                .expect("load Needle")
                .expect("CORRODE_NEEDLE_ASSETS must point at the asset dir");
            // Needle-flat schema; the dialect parses the raw reply.
            let tools = r#"[{"name":"get_weather","description":"Look up current weather","parameters":{"location":{"type":"string","description":"City","required":true}}}]"#;
            let raw = caller
                .generate("What's the weather in San Francisco?", tools)
                .expect("inference");
            eprintln!("raw: {raw}");
            let calls = crate::dialect::ToolDialect::default().parse(&raw).unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "get_weather");
            assert!(calls[0].arguments.get("location").is_some());
            assert_eq!(caller.model_id(), "needle");
        }
    }
}
