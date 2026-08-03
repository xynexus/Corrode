//! Per-model tool dialects: different tool-call models expect different tool *names*,
//! *schema formats*, and *call syntaxes*. Corrode's tools are defined once as canonical
//! [`Tool`] data; a [`ToolDialect`] renders that into the schema a given model expects
//! and parses the model's reply back into canonical [`ToolCall`]s.
//!
//! Dialects are matched to the tool-call model by id via a glob config file
//! (`CORRODE_TOOL_DIALECTS`, a JSON map of `model-glob -> profile`, `"default"` for the
//! fallback; the file replaces the built-ins wholesale). With no config, one built-in
//! rule routes MiniCPM models to their native XML dialect (the measured-better path);
//! every other model gets Needle's flat schema + JSON-array calls + no renames.

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

/// Per-request value sets for params with a closed, known set: `(tool, param) ->
/// allowed values`. Rendered as a standard JSON-Schema `enum` array — the carrier
/// hipfire's grammar derives value constraints from — in the OpenAI-nested schema
/// only; the Needle-flat rendering ignores it (no enum in Needle's training shape).
/// Absent (or a missing key) leaves the rendering byte-identical to today's.
pub type ParamValues = HashMap<(String, String), Vec<String>>;

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
    /// MiniCPM5's native XML: `<function name="f"><param name="p">v</param></function>`,
    /// with `<![CDATA[…]]>` around values containing `<`, `&` or newlines.
    ///
    /// This is what the model emits *by itself* once its chat template renders the
    /// `<tools>` block — no Needle in the loop. CDATA is why it matters: multi-param
    /// calls carrying file contents are structurally safe here, which is precisely the
    /// case `docs/todo/finetune-needle-toolset.md` calls the finetune's primary target
    /// (the base weights emit bare keys like `{"skill":"…","script"}` for it).
    MiniCpmXml,
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
    /// `values` is the optional per-request value overlay (see [`ParamValues`]).
    pub fn render(&self, tools: &[Tool], values: Option<&ParamValues>) -> String {
        let arr: Vec<Value> = tools.iter().map(|t| self.render_tool(t, values)).collect();
        Value::Array(arr).to_string()
    }

    fn render_tool(&self, t: &Tool, values: Option<&ParamValues>) -> Value {
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
                    let mut prop = Map::new();
                    prop.insert("type".to_string(), json!(p.ty));
                    prop.insert("description".to_string(), json!(p.description));
                    if let Some(vals) =
                        values.and_then(|v| v.get(&(t.name.to_string(), p.name.to_string())))
                    {
                        prop.insert("enum".to_string(), json!(vals));
                    }
                    props.insert(p.name.to_string(), Value::Object(prop));
                    if p.required {
                        required.push(Value::String(p.name.to_string()));
                    }
                }
                json!({"type": "object", "properties": Value::Object(props), "required": required})
            }
        };
        json!({"name": self.exposed(t.name), "description": t.description, "parameters": params})
    }

    /// Whether this model emits tool calls itself, so the request should declare tools
    /// and the reply is parsed directly — no Needle in the loop. False for dialects
    /// whose calls are constructed for the model (the Needle-flat / json-array default).
    pub fn emits_own_calls(&self) -> bool {
        matches!(self.parse, ParseFormat::MiniCpmXml)
    }

    /// Tools rendered for the `tools` field of a chat/responses request.
    ///
    /// Chat templates serialize each entry with `tool | tojson`, and the models are
    /// trained on the nested `{"type":"function","function":{…}}` envelope — so the
    /// envelope belongs here rather than at each call site. `values` as in [`render`].
    ///
    /// [`render`]: Self::render
    pub fn request_tools(&self, tools: &[Tool], values: Option<&ParamValues>) -> Value {
        Value::Array(
            tools
                .iter()
                .map(|t| json!({"type": "function", "function": self.render_tool(t, values)}))
                .collect(),
        )
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
            ParseFormat::MiniCpmXml => parse_minicpm_xml(raw),
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
    /// Built-in rules: MiniCPM models emit their own XML calls natively — the
    /// measured-better path (see `docs/todo/finetune-needle-toolset.md`) — so they skip
    /// the Needle shim out of the box. Everything else falls to the Needle default.
    fn default() -> Self {
        Self {
            rules: vec![(
                "*minicpm*".to_string(),
                ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::MiniCpmXml, HashMap::new()),
            )],
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

/// Trailing-`*` prefix glob, `*infix*` contains, or exact match — case-insensitive,
/// like every model-id match in `roles` (served ids are mixed-case, e.g. MiniCPM5-1B).
/// Enough for patterns like `needle*`, `*minicpm*`, or an exact id — no glob crate needed.
fn glob_match(pattern: &str, id: &str) -> bool {
    let (pattern, id) = (pattern.to_lowercase(), id.to_lowercase());
    match pattern.strip_suffix('*') {
        Some(rest) => match rest.strip_prefix('*') {
            Some(infix) => id.contains(infix),
            None => id.starts_with(rest),
        },
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
            "minicpm-xml" => ParseFormat::MiniCpmXml,
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
        let s = d.render(TOOLS, None);
        assert!(s.contains(r#""name":"run_command""#));
        assert!(s.contains(r#""command":{"type":"string""#));
        assert!(s.contains(r#""required":true"#));
        assert!(!s.contains(r#""properties""#), "flat, not nested");
    }

    #[test]
    fn openai_nested_render_and_name_rename() {
        let names = HashMap::from([("run_command".to_string(), "bash".to_string())]);
        let d = ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::JsonArray, names);
        let s = d.render(TOOLS, None);
        assert!(s.contains(r#""name":"bash""#), "renamed to the exposed name");
        assert!(s.contains(r#""type":"object""#));
        assert!(s.contains(r#""properties""#));
        assert!(s.contains(r#""required":["command"]"#));
    }

    // The grammar value constraint rides a standard JSON-Schema `enum`
    // (docs/todo/tool-call-judgement.md item 4). With no overlay the bytes are
    // pinned — hipfire derives its ToolSchema from this block, so a stray key
    // would change what every model sees.
    #[test]
    fn no_overlay_keeps_the_nested_render_byte_identical() {
        let d = ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::JsonArray, HashMap::new());
        assert_eq!(
            d.render(TOOLS, None),
            r#"[{"name":"run_command","description":"Run a shell command.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The command."}},"required":["command"]}}]"#
        );
    }

    #[test]
    fn values_overlay_adds_enum_only_where_named_and_only_nested() {
        let overlay: ParamValues = HashMap::from([(
            ("run_command".to_string(), "command".to_string()),
            vec!["cargo test".to_string(), "cargo build".to_string()],
        )]);
        let nested =
            ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::JsonArray, HashMap::new());
        let s = nested.render(TOOLS, Some(&overlay));
        assert!(s.contains(r#""enum":["cargo test","cargo build"]"#), "got: {s}");
        // The request envelope carries the same overlay through.
        let req = nested.request_tools(TOOLS, Some(&overlay)).to_string();
        assert!(req.contains(r#""enum":["cargo test","cargo build"]"#), "got: {req}");
        // A (tool, param) the overlay doesn't name is untouched.
        let other: ParamValues =
            HashMap::from([(("read_file".to_string(), "path".to_string()), vec!["x".into()])]);
        assert_eq!(nested.render(TOOLS, Some(&other)), nested.render(TOOLS, None));
        // The Needle-flat rendering ignores the overlay entirely.
        let flat = ToolDialect::default();
        assert_eq!(flat.render(TOOLS, Some(&overlay)), flat.render(TOOLS, None));
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
    fn builtin_default_routes_minicpm_natively_needle_flat_otherwise() {
        let d = Dialects::default();
        // MiniCPM ids (any case, any position) take the native path: nested schema,
        // XML parse, the model emits its own calls.
        let m = d.resolve("MiniCPM5-1B");
        assert_eq!(m.schema, SchemaFormat::OpenAiNested);
        assert!(m.emits_own_calls());
        // Everything else — including the Needle caller's own id — stays on today's
        // Needle-flat / json-array default.
        for id in ["needle", "zaya1-8b"] {
            let n = d.resolve(id);
            assert_eq!(n.schema, SchemaFormat::NeedleFlat);
            assert!(!n.emits_own_calls());
        }
    }

    #[test]
    fn user_config_overrides_the_builtin_minicpm_rule() {
        // A config file replaces the built-ins wholesale, so its minicpm rule wins.
        let cfg = r#"{"*minicpm*": {"schema":"needle-flat","parse":"json-array"}}"#;
        let d = Dialects::parse_config(cfg.to_string()).unwrap();
        assert!(!d.resolve("MiniCPM5-1B").emits_own_calls());
        assert_eq!(d.resolve("MiniCPM5-1B").schema, SchemaFormat::NeedleFlat);
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

/// Parse MiniCPM5's native XML tool calls out of a reply.
///
/// The model reasons in prose and then emits one or more `<function>` blocks, so this
/// scans rather than expecting the whole reply to be a call. A malformed or absent
/// block yields no calls (the reply was a final answer, not a tool step) — never an
/// error, matching how the tool loop treats a turn with no call.
///
/// ponytail: string scan, not an XML parser. The grammar in hipfire constrains this
/// shape at the token level, and values arrive either plain or CDATA-wrapped; a real
/// parser buys nothing until the dialect grows attributes or nesting.
fn parse_minicpm_xml(raw: &str) -> Vec<ToolCall> {
    const FN_OPEN: &str = "<function name=\"";
    const PARAM_OPEN: &str = "<param name=\"";
    let mut calls = Vec::new();
    let mut rest = raw;
    while let Some(at) = rest.find(FN_OPEN) {
        let after = &rest[at + FN_OPEN.len()..];
        let Some(name_end) = after.find("\">") else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        let body_start = &after[name_end + 2..];
        // A call ends at its close tag; without one, take the rest (truncated output).
        let (body, tail) = match body_start.find("</function>") {
            Some(end) => (&body_start[..end], &body_start[end + "</function>".len()..]),
            None => (body_start, ""),
        };

        let mut args = serde_json::Map::new();
        let mut param_rest = body;
        while let Some(p_at) = param_rest.find(PARAM_OPEN) {
            let p_after = &param_rest[p_at + PARAM_OPEN.len()..];
            let Some(p_name_end) = p_after.find("\">") else {
                break;
            };
            let p_name = p_after[..p_name_end].trim().to_string();
            let value_start = &p_after[p_name_end + 2..];
            let (value, next) = match value_start.find("</param>") {
                Some(end) => (
                    &value_start[..end],
                    &value_start[end + "</param>".len()..],
                ),
                None => (value_start, ""),
            };
            args.insert(p_name, serde_json::Value::String(unwrap_cdata(value)));
            param_rest = next;
        }

        if !name.is_empty() {
            calls.push(ToolCall {
                name,
                arguments: serde_json::Value::Object(args),
            });
        }
        rest = tail;
    }
    calls
}

/// Strip a `<![CDATA[…]]>` wrapper. Values only need it when they contain `<`, `&` or
/// newlines, so both forms turn up in the same reply and the content is verbatim either
/// way — no entity decoding, which is the point of CDATA for source code.
fn unwrap_cdata(value: &str) -> String {
    let trimmed = value.trim();
    match trimmed
        .strip_prefix("<![CDATA[")
        .and_then(|inner| inner.strip_suffix("]]>"))
    {
        Some(inner) => inner.to_string(),
        // Only trim when there was no CDATA: inside CDATA, whitespace is content.
        None => trimmed.to_string(),
    }
}

#[cfg(test)]
mod minicpm_xml_tests {
    use super::*;

    fn dialect() -> ToolDialect {
        ToolDialect::new(
            SchemaFormat::OpenAiNested,
            ParseFormat::MiniCpmXml,
            HashMap::new(),
        )
    }

    // The shape MiniCPM5 actually emitted on the live daemon once its template rendered
    // the <tools> block. Single-param, no CDATA.
    #[test]
    fn parses_a_native_single_param_call() {
        let calls = dialect()
            .parse(r#"<function name="read_file"><param name="path">src/lib.rs</param></function>"#)
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "src/lib.rs");
    }

    // The case the Needle finetune exists to fix: multi-param with a file body. In JSON
    // the base weights emit a bare key and it won't parse; in CDATA it is just content.
    #[test]
    fn parses_multi_param_with_cdata_verbatim() {
        let raw = "<function name=\"write_file\">\
             <param name=\"path\">src/util.rs</param>\
             <param name=\"contents\"><![CDATA[pub fn double(x: i64) -> i64 {\n    x * 2\n}]]></param>\
             </function>";
        let calls = dialect().parse(raw).unwrap();
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments["path"], "src/util.rs");
        // newlines, `<` and `>` survive untouched — no escaping, no entity decoding
        assert_eq!(
            calls[0].arguments["contents"],
            "pub fn double(x: i64) -> i64 {\n    x * 2\n}"
        );
    }

    // The model reasons in prose and then calls; the parser must find the block, and
    // must return nothing when the turn was a plain answer.
    #[test]
    fn finds_a_call_after_prose_and_ignores_a_reply_with_none() {
        let calls = dialect()
            .parse("I'll read it first.\n<function name=\"list_dir\"><param name=\"path\">src</param></function>")
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["path"], "src");

        assert!(dialect()
            .parse("The file defines add, factorial and is_prime.")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn parses_several_calls_in_one_reply() {
        let raw = "<function name=\"read_file\"><param name=\"path\">a.rs</param></function>\
                   <function name=\"read_file\"><param name=\"path\">b.rs</param></function>";
        let calls = dialect().parse(raw).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments["path"], "b.rs");
    }

    // Truncated output (hit the token cap mid-call) must degrade to no call rather than
    // a malformed one the tool loop would try to execute.
    #[test]
    fn a_truncated_call_does_not_panic() {
        let calls = dialect().parse("<function name=\"read_file\"><param name=\"pa").unwrap();
        assert_eq!(calls.len(), 1, "the name was complete");
        assert!(
            calls[0].arguments.get("pa").is_none(),
            "the half-written param is dropped, not invented"
        );
        assert!(dialect().parse("<function name=\"read_fi").unwrap().is_empty());
    }

    // Exposed->canonical renaming must apply to this format too, like json-array.
    #[test]
    fn exposed_names_map_back_to_canonical() {
        let names = HashMap::from([("run_command".to_string(), "sh".to_string())]);
        let d = ToolDialect::new(SchemaFormat::OpenAiNested, ParseFormat::MiniCpmXml, names);
        let calls = d
            .parse(r#"<function name="sh"><param name="command">cargo test</param></function>"#)
            .unwrap();
        assert_eq!(calls[0].name, "run_command");
    }
}
