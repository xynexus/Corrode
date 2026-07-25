use anyhow::{Context, Result};
use needle_toolcall_shim::guide::{normalize_tools, restore_tool_names, to_snake_case};
use needle_toolcall_shim::model::build_encoder_input;
use needle_toolcall_shim::tokenizer::{NeedleTokenizer, TOOLS_ID};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ToolSuite {
    suite: String,
    source: String,
    notes: String,
    tools: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct ToolCase {
    suite: String,
    tool_name: String,
    variant: String,
    query: String,
    tools: Vec<Value>,
    call: Vec<Value>,
}

fn suites() -> Result<Vec<ToolSuite>> {
    let raw = include_str!("fixtures/tool_suites.json");
    serde_json::from_str(raw).context("parsing tool_suites.json")
}

fn cases() -> Result<Vec<ToolCase>> {
    let raw = include_str!("fixtures/tool_suite_cases.json");
    serde_json::from_str(raw).context("parsing tool_suite_cases.json")
}

#[test]
fn fixture_has_broad_named_tool_suites() -> Result<()> {
    let suites = suites()?;
    assert_eq!(suites.len(), 4);
    let mut total = 0;
    for suite in suites {
        assert!(!suite.source.is_empty(), "{} has source", suite.suite);
        assert!(!suite.notes.is_empty(), "{} has notes", suite.suite);
        assert!(suite.tools.len() >= 10, "{} is too small", suite.suite);
        let mut names = HashSet::new();
        for tool in &suite.tools {
            let name = tool["name"].as_str().context("tool name")?;
            assert!(names.insert(name.to_string()), "duplicate {name}");
            assert!(
                tool["parameters"].is_object(),
                "{}:{name} parameters must be an object",
                suite.suite
            );
        }
        total += suite.tools.len();
    }
    assert!(total >= 70);
    Ok(())
}

#[test]
fn hermes_current_session_export_is_present() -> Result<()> {
    let suites = suites()?;
    let hermes = suites
        .iter()
        .find(|suite| suite.suite == "hermes_current_session")
        .context("missing hermes_current_session suite")?;
    assert_eq!(hermes.tools.len(), 33);

    let names = hermes
        .tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<HashSet<_>>();
    for expected in [
        "browser_navigate",
        "delegate_task",
        "execute_code",
        "patch",
        "terminal",
        "web_search",
        "multi_tool_use.parallel",
    ] {
        assert!(names.contains(expected), "missing {expected}");
    }

    let terminal = hermes
        .tools
        .iter()
        .find(|tool| tool["name"] == "terminal")
        .context("terminal tool")?;
    assert_eq!(
        terminal["parameters"]["required"],
        serde_json::json!(["command"])
    );
    assert!(terminal["parameters"]["properties"]["pty"]["default"] == false);
    Ok(())
}

#[test]
fn fixture_suites_normalize_and_restore_tool_names() -> Result<()> {
    for suite in suites()? {
        let tools_json = serde_json::to_string(&suite.tools)?;
        let (normalized, name_map) = normalize_tools(&tools_json)?;
        let normalized_tools: Vec<Value> = serde_json::from_str(&normalized)?;
        assert_eq!(normalized_tools.len(), suite.tools.len());

        for original_tool in suite.tools {
            let original = original_tool["name"].as_str().unwrap();
            let snake = to_snake_case(original);
            let call = format!(r#"[{{"name":"{snake}","arguments":{{}}}}]"#);
            let restored = restore_tool_names(&call, &name_map);
            assert!(
                restored.contains(&format!(r#""name":"{original}""#)),
                "{} did not restore {original}: {restored}",
                suite.suite
            );
        }
    }
    Ok(())
}

#[test]
fn fixture_suites_fit_encoder_input_with_tokenizer() -> Result<()> {
    let path = "needle-weights/tokenizer/needle.model";
    if !Path::new(path).exists() {
        eprintln!("skipping tool-suite encoder input test; local tokenizer is missing");
        return Ok(());
    }
    let tokenizer = NeedleTokenizer::load(path)?;
    for suite in suites()? {
        let tools_json = serde_json::to_string(&suite.tools)?;
        let (normalized, _) = normalize_tools(&tools_json)?;
        let enc = build_encoder_input(
            &tokenizer,
            "Inspect the repository and update the relevant file.",
            &normalized,
            1024,
        )?;
        assert!(
            enc.contains(&TOOLS_ID),
            "{} missing tools separator",
            suite.suite
        );
        assert!(
            enc.len() <= 1024,
            "{} exceeded max encoder length",
            suite.suite
        );
    }
    Ok(())
}

#[test]
fn generated_cases_have_three_samples_per_tool() -> Result<()> {
    let suites = suites()?;
    let cases = cases()?;
    let expected_tool_count = suites.iter().map(|suite| suite.tools.len()).sum::<usize>();
    assert_eq!(cases.len(), expected_tool_count * 3);

    let mut counts: BTreeMap<(String, String), HashSet<String>> = BTreeMap::new();
    for case in cases {
        assert!(!case.query.is_empty());
        counts
            .entry((case.suite, case.tool_name))
            .or_default()
            .insert(case.variant);
    }

    for ((suite, tool), variants) in counts {
        assert_eq!(
            variants,
            ["minimal_required", "full_arguments", "normalized_name"]
                .into_iter()
                .map(str::to_string)
                .collect::<HashSet<_>>(),
            "{suite}:{tool}"
        );
    }
    Ok(())
}

#[test]
fn generated_cases_match_tool_schemas() -> Result<()> {
    for case in cases()? {
        assert_eq!(case.tools.len(), 1, "{}:{}", case.suite, case.tool_name);
        assert_eq!(case.call.len(), 1, "{}:{}", case.suite, case.tool_name);
        let tool = &case.tools[0];
        let call = &case.call[0];
        assert_eq!(tool["name"], case.tool_name);

        let call_name = call["name"].as_str().context("call name")?;
        if case.variant == "normalized_name" {
            assert_eq!(call_name, to_snake_case(&case.tool_name));
        } else {
            assert_eq!(call_name, case.tool_name);
        }

        let args = call["arguments"].as_object().context("call arguments")?;
        let params = tool["parameters"].as_object().context("parameters")?;
        assert_required_present(params, args, &case)?;
        for (key, value) in args {
            let schema = params
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|props| props.get(key))
                .and_then(Value::as_object)
                .with_context(|| format!("schema for {}:{}:{key}", case.suite, case.tool_name))?;
            assert_value_matches_schema(
                value,
                schema,
                &format!("{}:{}:{key}", case.suite, case.tool_name),
            );
        }
    }
    Ok(())
}

#[test]
fn generated_cases_normalize_restore_and_pack() -> Result<()> {
    let tokenizer = if Path::new("needle-weights/tokenizer/needle.model").exists() {
        Some(NeedleTokenizer::load(
            "needle-weights/tokenizer/needle.model",
        )?)
    } else {
        None
    };

    for case in cases()? {
        let tools_json = serde_json::to_string(&case.tools)?;
        let call_json = serde_json::to_string(&case.call)?;
        let (normalized_tools, name_map) = normalize_tools(&tools_json)?;
        let restored = restore_tool_names(&call_json, &name_map);
        assert!(
            restored.contains(&format!(r#""name":"{}""#, case.tool_name)),
            "{}:{}:{} restore failed: {restored}",
            case.suite,
            case.tool_name,
            case.variant
        );

        if let Some(tokenizer) = &tokenizer {
            let enc = build_encoder_input(tokenizer, &case.query, &normalized_tools, 1024)?;
            assert!(enc.contains(&TOOLS_ID));
            assert!(enc.len() <= 1024);
        }
    }
    Ok(())
}

fn assert_required_present(
    params: &Map<String, Value>,
    args: &Map<String, Value>,
    case: &ToolCase,
) -> Result<()> {
    let required = params
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    for key in required {
        assert!(
            args.contains_key(key),
            "{}:{}:{} missing required {key}",
            case.suite,
            case.tool_name,
            case.variant
        );
    }
    Ok(())
}

fn assert_value_matches_schema(value: &Value, schema: &Map<String, Value>, label: &str) {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        assert!(values.contains(value), "{label} value {value} not in enum");
        return;
    }
    let typ = schema_type(schema);
    match typ {
        "string" => assert!(value.is_string(), "{label} expected string, got {value}"),
        "integer" => assert!(
            value.as_i64().is_some(),
            "{label} expected integer, got {value}"
        ),
        "number" => assert!(
            value.as_f64().is_some(),
            "{label} expected number, got {value}"
        ),
        "boolean" => assert!(value.is_boolean(), "{label} expected boolean, got {value}"),
        "null" => assert!(value.is_null(), "{label} expected null, got {value}"),
        "array" => {
            let arr = value
                .as_array()
                .unwrap_or_else(|| panic!("{label} expected array, got {value}"));
            if let Some(item_schema) = schema.get("items").and_then(Value::as_object) {
                for item in arr {
                    assert_value_matches_schema(item, item_schema, label);
                }
            }
        }
        "object" => {
            let obj = value
                .as_object()
                .unwrap_or_else(|| panic!("{label} expected object, got {value}"));
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                for (key, child) in obj {
                    if let Some(child_schema) = props.get(key).and_then(Value::as_object) {
                        assert_value_matches_schema(child, child_schema, &format!("{label}.{key}"));
                    }
                }
            }
        }
        _ => {}
    }
}

fn schema_type(schema: &Map<String, Value>) -> &str {
    match schema.get("type") {
        Some(Value::String(typ)) => typ.as_str(),
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(Value::as_str)
            .find(|typ| *typ != "null")
            .unwrap_or("null"),
        _ => "string",
    }
}
