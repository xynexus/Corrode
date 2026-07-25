use anyhow::Result;
use needle_toolcall_shim::model::{Assets, GenerateOptions, NeedleModel};
use needle_toolcall_shim::tokenizer::NeedleTokenizer;
use std::path::Path;

struct BenchCase {
    name: &'static str,
    guided: bool,
    guide_fast_forward: bool,
}

#[test]
#[ignore = "asset-backed benchmark; run with `cargo test --release --test benchmark -- --ignored --nocapture`"]
fn benchmark_weather_decode_matrix() -> Result<()> {
    if !Path::new("assets/needle/model.safetensors").exists() {
        eprintln!("skipping benchmark; assets/needle is missing");
        return Ok(());
    }

    let assets = Assets::resolve("assets/needle")?;
    let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
    let model = NeedleModel::load(&assets)?;
    let tools = r#"[{"name":"get_weather","parameters":{"location":"string"}}]"#;
    let query = "What's the weather in San Francisco?";
    let cases = [
        BenchCase {
            name: "guided",
            guided: true,
            guide_fast_forward: true,
        },
        BenchCase {
            name: "guided_no_fast_forward",
            guided: true,
            guide_fast_forward: false,
        },
        BenchCase {
            name: "unconstrained",
            guided: false,
            guide_fast_forward: false,
        },
    ];

    for case in cases {
        let result = model.generate_with_options(
            &tokenizer,
            query,
            tools,
            &GenerateOptions {
                max_gen_len: 64,
                max_enc_len: model.config.max_seq_len,
                guided: case.guided,
                normalize: true,
                guide_fast_forward: case.guide_fast_forward,
            },
        )?;
        let tps = result.generated_tokens as f64 / result.elapsed.as_secs_f64();
        println!(
            "{}: tokens={} seconds={:.3} tok/s={:.2} output={}",
            case.name,
            result.generated_tokens,
            result.elapsed.as_secs_f64(),
            tps,
            result.output
        );
        assert!(result.generated_tokens > 0, "{}", case.name);
        assert!(tps.is_finite() && tps > 0.0, "{}", case.name);
    }
    Ok(())
}
