//! Per-model tool dialects: different tool-call models expect different tool *names*,
//! *schema formats*, and *call syntaxes*. Corrode's tools are defined once as canonical
//! [`Tool`] data; a [`ToolDialect`] renders that into the schema a given model expects
//! and parses the model's reply back into canonical [`ToolCall`]s.
//!
//! Dialects are matched to the tool-call model by id via a glob config file
//! (`CORRODE_TOOL_DIALECTS`, a JSON map of `model-glob -> profile`, `"default"` for the
//! fallback). With no config, the built-in default is Needle's flat schema + JSON-array
//! calls + no renames — i.e. exactly today's behavior.

use crate::toolcall::ToolCall;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// One parameter of a canonical tool.
pub struct Param {
    pub name: &'static str,
    pub ty: &'static str,
    pub description: &'static str,
    pub required: bool,
}

/// A canonical, model-agnostic tool. Rendered per model by [`ToolDialect::render`];
/// executed by name (its canonical name) after [`ToolDialect::parse`] maps a model's
/// exposed name back.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub params: &'static [Param],
}

/// How a model wants the tools JSON shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaFormat {
    /// Needle-native flat: `parameters` is a `name -> {type,description,required}` map.
    NeedleFlat,
    /// OpenAI-style nested: `parameters` is `{type:object, properties:{…}, required:[…]}`.
    OpenAiNested,
}

/// How to read a model's tool-call reply back into [`ToolCall`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseFormat {
    /// A JSON array of `{"name","arguments"}` (optionally prefixed with `<tool_call>`).
    JsonArray,
}

/// A model's tool dialect: how to render tool schemas, how to parse its calls, and the
/// name mapping between Corrode's canonical tool names and the names it exposes.
pub struct ToolDialect {
    schema: SchemaFormat,
    parse: ParseFormat,
    /// canonical -> exposed (what the model sees).
    names: HashMap<String, String>,
    /// exposed -> canonical (reverse of `names`, for parsing).
    rev_names: HashMap<String, String>,
}

impl Default for ToolDialect {
    /// Today's behavior: Needle flat schema, JSON-array calls, no renames.
    fn default() -> Self {
        Self::new(SchemaFormat::NeedleFlat, ParseFormat::JsonArray, HashMap::new())
    }
}

impl ToolDialect {
    pub fn new(schema: SchemaFormat, parse: ParseFormat, names: HashMap<String, String>) -> Self {
        let rev_names = names.iter().map(|(c, e)| (e.clone(), c.clone())).collect();
        Self {
            schema,
            parse,
            names,
            rev_names,
        }
    }

    fn exposed(&self, canonical: &str) -> String {
        self.names
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| canonical.to_string())
    }

    fn canonical(&self, exposed: &str) -> String {
        self.rev_names
            .get(exposed)
            .cloned()
            .unwrap_or_else(|| exposed.to_string())
    }

    /// Render canonical tools into the schema JSON string this model expects.
    pub fn render(&self, tools: &[Tool]) -> String {
        let arr: Vec<Value> = tools.iter().map(|t| self.render_tool(t)).collect();
        Value::Array(arr).to_string()
    }

    fn render_tool(&self, t: &Tool) -> Value {
        let params = match self.schema {
            SchemaFormat::NeedleFlat => {
                let mut m = Map::new();
                for p in t.params {
                    m.insert(
                        p.name.to_string(),
                        json!({"type": p.ty, "description": p.description, "required": p.required}),
                    );
                }
                Value::Object(m)
            }
            SchemaFormat::OpenAiNested => {
                let mut props = Map::new();
                let mut required = Vec::new();
                for p in t.params {
                    props.insert(
                        p.name.to_string(),
                        json!({"type": p.ty, "description": p.description}),
                    );
                    if p.required {
                        required.push(Value::String(p.name.to_string()));
                    }
                }
                json!({"type": "object", "properties": Value::Object(props), "required": required})
            }
        };
        json!({"name": self.exposed(t.name), "description": t.description, "parameters": params})
    }

    /// Parse a model's raw reply into canonical [`ToolCall`]s (exposed names mapped back).
    pub fn parse(&self, raw: &str) -> anyhow::Result<Vec<ToolCall>> {
        let trimmed = raw.trim().trim_start_matches("<tool_call>").trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        let calls: Vec<ToolCall> = match self.parse {
            ParseFormat::JsonArray => serde_json::from_str(trimmed)
                .map_err(|e| anyhow::anyhow!("parsing tool calls from {trimmed:?}: {e}"))?,
        };
        Ok(calls
            .into_iter()
            .map(|mut c| {
                c.name = self.canonical(&c.name);
                c
            })
            .collect())
    }
}

/// The resolved set of dialects: glob rules (most specific first) plus a default.
pub struct Dialects {
    rules: Vec<(String, ToolDialect)>,
    default: ToolDialect,
}

impl Default for Dialects {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default: ToolDialect::default(),
        }
    }
}

impl Dialects {
    /// Load from `CORRODE_TOOL_DIALECTS` (a JSON `model-glob -> profile` file). Absent or
    /// unreadable -> the built-in default (Needle). A malformed file logs and falls back.
    pub fn load() -> Self {
        let Ok(path) = std::env::var("CORRODE_TOOL_DIALECTS") else {
            return Self::default();
        };
        match std::fs::read_to_string(&path).map_err(anyhow::Error::from).and_then(Self::parse_config) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("CORRODE_TOOL_DIALECTS unusable ({e}); using the default dialect");
                Self::default()
            }
        }
    }

    fn parse_config(text: String) -> anyhow::Result<Self> {
        let raw: HashMap<String, ProfileConfig> = serde_json::from_str(&text)?;
        let mut default = ToolDialect::default();
        let mut rules = Vec::new();
        for (glob, profile) in raw {
            let dialect = profile.into_dialect()?;
            if glob == "default" {
                default = dialect;
            } else {
                rules.push((glob, dialect));
            }
        }
        // Most specific first: exact patterns before wildcards, longer prefixes first.
        rules.sort_by(|(a, _), (b, _)| {
            let key = |g: &str| (g.ends_with('*'), std::cmp::Reverse(g.len()));
            key(a).cmp(&key(b))
        });
        Ok(Self { rules, default })
    }

    /// The dialect for a tool-call model id — the first matching glob rule, else default.
    pub fn resolve(&self, model_id: &str) -> &ToolDialect {
        self.rules
            .iter()
            .find(|(g, _)| glob_match(g, model_id))
            .map(|(_, d)| d)
            .unwrap_or(&self.default)
    }
}

/// Trailing-`*` prefix glob (or exact match). Enough for model-id patterns like
/// `needle*`, `zaya1-8b*`, or an exact id — no glob crate needed.
fn glob_match(pattern: &str, id: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => id.starts_with(prefix),
        None => pattern == id,
    }
}

#[derive(Deserialize)]
struct ProfileConfig {
    schema: String,
    parse: String,
    #[serde(default)]
    names: HashMap<String, String>,
}

impl ProfileConfig {
    fn into_dialect(self) -> anyhow::Result<ToolDialect> {
        let schema = match self.schema.as_str() {
            "needle-flat" => SchemaFormat::NeedleFlat,
            "openai-nested" => SchemaFormat::OpenAiNested,
            other => anyhow::bail!("unknown schema format `{other}`"),
        };
        let parse = match self.parse.as_str() {
            "json-array" => ParseFormat::JsonArray,
            other => anyhow::bail!("unknown parse format `{other}`"),
        };
        Ok(ToolDialect::new(schema, parse, self.names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS: &[Tool] = &[Tool {
        name: "run_command",
        description: "Run a shell command.",
        params: &[Param {
            name: "command",
            ty: "string",
            description: "The command.",
            required: true,
        }],
    }];

    #[test]
    fn needle_flat_render_matches_the_native_shape() {
        let d = ToolDialect::default();
        let s = d.render(TOOLS);
        assert!(s.contains(r#""name":"run_command""#));
        assert!(s.contains(r#""command":{"type":"string""#));
        assert!(s.contains(r#""required":true"#));
        assert!(!s.contains(r#""properties""#), "flat, not nested");
    }

    #[test]
    fn openai_nested_render_and_name_rename() {
        let names = HashMap::from([("run_command".to_string(), "bash".to_string())]);
        let d = ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::JsonArray, names);
        let s = d.render(TOOLS);
        assert!(s.contains(r#""name":"bash""#), "renamed to the exposed name");
        assert!(s.contains(r#""type":"object""#));
        assert!(s.contains(r#""properties""#));
        assert!(s.contains(r#""required":["command"]"#));
    }

    #[test]
    fn parse_maps_exposed_names_back_to_canonical() {
        let names = HashMap::from([("run_command".to_string(), "bash".to_string())]);
        let d = ToolDialect::new(SchemaFormat::NeedleFlat, ParseFormat::JsonArray, names);
        let calls = d
            .parse(r#"<tool_call>[{"name":"bash","arguments":{"command":"ls"}}]"#)
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run_command"); // mapped back to canonical
        assert_eq!(calls[0].arguments["command"], "ls");
        assert!(d.parse("   ").unwrap().is_empty());
        assert!(d.parse("[{not json").is_err());
    }

    #[test]
    fn config_globs_resolve_most_specific_first() {
        let cfg = r#"{
            "default": {"schema":"needle-flat","parse":"json-array"},
            "needle*": {"schema":"openai-nested","parse":"json-array"},
            "needle-corrode-v2": {"schema":"needle-flat","parse":"json-array","names":{"run_command":"sh"}}
        }"#;
        let d = Dialects::parse_config(cfg.to_string()).unwrap();
        // exact id wins over the wildcard
        assert_eq!(d.resolve("needle-corrode-v2").exposed("run_command"), "sh");
        // wildcard for other needle ids
        assert_eq!(d.resolve("needle-base").schema, SchemaFormat::OpenAiNested);
        // no match -> default
        assert_eq!(d.resolve("zaya1-8b").schema, SchemaFormat::NeedleFlat);
    }
}
