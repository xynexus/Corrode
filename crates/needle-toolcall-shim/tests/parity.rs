use anyhow::{Context, Result};
use needle_toolcall_shim::model::{Assets, DebugProbe, NeedleModel, TensorProbe};
use needle_toolcall_shim::tokenizer::NeedleTokenizer;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct OutputCase {
    name: String,
    query: String,
    tools: String,
    unconstrained: String,
    constrained: String,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn assets_if_available() -> Option<Assets> {
    let path = root().join("assets/needle");
    Assets::resolve(path).ok()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?)
}

#[test]
fn python_fixture_outputs_match_rust_decode() -> Result<()> {
    if std::env::var_os("NEEDLE_RUN_PARITY").is_none() {
        eprintln!("skipping full decode parity; set NEEDLE_RUN_PARITY=1 to run");
        return Ok(());
    }
    let Some(assets) = assets_if_available() else {
        eprintln!("skipping full decode parity; assets/needle is missing");
        return Ok(());
    };
    let cases: Vec<OutputCase> = read_json(&root().join("tests/fixtures/python_outputs.json"))?;
    let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
    let model = NeedleModel::load(&assets)?;

    for case in cases {
        let constrained = model.generate(&tokenizer, &case.query, &case.tools, 64, true, true)?;
        assert_eq!(constrained, case.constrained, "constrained {}", case.name);

        let unconstrained =
            model.generate(&tokenizer, &case.query, &case.tools, 64, false, true)?;
        assert_eq!(
            unconstrained, case.unconstrained,
            "unconstrained {}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn python_intermediate_probes_match_rust() -> Result<()> {
    if std::env::var_os("NEEDLE_RUN_PARITY").is_none() {
        eprintln!("skipping numeric probe parity; set NEEDLE_RUN_PARITY=1 to run");
        return Ok(());
    }
    let Some(assets) = assets_if_available() else {
        eprintln!("skipping numeric probe parity; assets/needle is missing");
        return Ok(());
    };
    let expected: DebugProbe = read_json(&root().join("tests/fixtures/python_probes.json"))?;
    let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
    let model = NeedleModel::load(&assets)?;
    let actual = model.debug_probe(
        &tokenizer,
        "What's the weather in San Francisco?",
        r#"[{"name":"get_weather","parameters":{"location":"string"}}]"#,
    )?;

    assert_eq!(actual.encoder_tokens, expected.encoder_tokens);
    assert_probe_close(
        "embedding",
        &actual.embedding,
        &expected.embedding,
        2e-2,
        5e-2,
    );
    assert_probe_close(
        "encoder_layer0_attention",
        &actual.encoder_layer0_attention,
        &expected.encoder_layer0_attention,
        2e-2,
        5e-2,
    );
    assert_probe_close(
        "encoder_final",
        &actual.encoder_final,
        &expected.encoder_final,
        2e-2,
        5e-2,
    );
    assert_probe_close(
        "decoder_logits0",
        &actual.decoder_logits0,
        &expected.decoder_logits0,
        5e-2,
        1e-1,
    );
    Ok(())
}

fn assert_probe_close(
    name: &str,
    actual: &TensorProbe,
    expected: &TensorProbe,
    sample_tol: f32,
    sum_tol: f32,
) {
    assert_eq!(actual.shape, expected.shape, "{name} shape");
    for (idx, (a, e)) in actual.sample.iter().zip(expected.sample.iter()).enumerate() {
        assert!(
            (a - e).abs() <= sample_tol,
            "{name} sample[{idx}] actual={a} expected={e}"
        );
    }
    assert!(
        (actual.sum - expected.sum).abs() <= sum_tol.max(expected.sum.abs() * 1e-4),
        "{name} sum actual={} expected={}",
        actual.sum,
        expected.sum
    );
}
