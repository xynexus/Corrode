use anyhow::Result;
use clap::{Parser, Subcommand};
use needle_toolcall_shim::model::{inspect_assets, Assets, GenerateOptions, NeedleModel};
use needle_toolcall_shim::tokenizer::NeedleTokenizer;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "needle-toolcall-shim")]
#[command(about = "Native Rust/Candle inference for Needle tool calls")]
struct Cli {
    #[arg(long, global = true)]
    json_errors: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        #[arg(long, default_value = "assets/needle")]
        assets: String,
    },
    Infer {
        #[arg(long, default_value = "assets/needle")]
        assets: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "[]")]
        tools: String,
        #[arg(long, default_value_t = 512)]
        max_gen_len: usize,
        #[arg(long)]
        max_enc_len: Option<usize>,
        #[arg(long)]
        unconstrained: bool,
        #[arg(long)]
        no_guide_fast_forward: bool,
        #[arg(long)]
        no_normalize: bool,
    },
    Bench {
        #[arg(long, default_value = "assets/needle")]
        assets: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value = "[]")]
        tools: String,
        #[arg(long, default_value_t = 512)]
        max_gen_len: usize,
        #[arg(long)]
        max_enc_len: Option<usize>,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long)]
        unconstrained: bool,
        #[arg(long)]
        no_guide_fast_forward: bool,
        #[arg(long)]
        matrix: bool,
        #[arg(long)]
        no_normalize: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(&cli) {
        if cli.json_errors {
            eprintln!("{}", json!({ "error": err.to_string() }));
        } else {
            eprintln!("Error: {err:#}");
        }
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Command::Inspect { assets } => {
            let assets = Assets::resolve(assets)?;
            println!("{}", inspect_assets(&assets)?);
        }
        Command::Infer {
            assets,
            query,
            tools,
            max_gen_len,
            max_enc_len,
            unconstrained,
            no_guide_fast_forward,
            no_normalize,
        } => {
            let assets = Assets::resolve(assets)?;
            let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
            let model = NeedleModel::load(&assets)?;
            let options = GenerateOptions {
                max_gen_len: *max_gen_len,
                max_enc_len: max_enc_len.unwrap_or(model.config.max_seq_len),
                guided: !unconstrained,
                normalize: !no_normalize,
                guide_fast_forward: !no_guide_fast_forward,
            };
            let result = model.generate_with_options(&tokenizer, query, tools, &options)?;
            println!("{}", result.output);
        }
        Command::Bench {
            assets,
            query,
            tools,
            max_gen_len,
            max_enc_len,
            iterations,
            unconstrained,
            no_guide_fast_forward,
            matrix,
            no_normalize,
        } => {
            let assets = Assets::resolve(assets)?;
            let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
            let model = NeedleModel::load(&assets)?;
            let max_enc_len = max_enc_len.unwrap_or(model.config.max_seq_len);
            let cases = if *matrix {
                vec![
                    ("guided", true, true),
                    ("guided_no_fast_forward", true, false),
                    ("unconstrained", false, false),
                ]
            } else {
                vec![(
                    if *unconstrained {
                        "unconstrained"
                    } else if *no_guide_fast_forward {
                        "guided_no_fast_forward"
                    } else {
                        "guided"
                    },
                    !unconstrained,
                    !no_guide_fast_forward,
                )]
            };
            let mut results = Vec::with_capacity(cases.len());
            for (name, guided, guide_fast_forward) in cases {
                let options = GenerateOptions {
                    max_gen_len: *max_gen_len,
                    max_enc_len,
                    guided,
                    normalize: !no_normalize,
                    guide_fast_forward: guided && guide_fast_forward,
                };
                let mut runs = Vec::with_capacity(*iterations);
                for _ in 0..*iterations {
                    runs.push(model.generate_with_options(&tokenizer, query, tools, &options)?);
                }
                let total_tokens = runs.iter().map(|run| run.generated_tokens).sum::<usize>();
                let total_secs = runs
                    .iter()
                    .map(|run| run.elapsed.as_secs_f64())
                    .sum::<f64>();
                let avg_secs = if runs.is_empty() {
                    0.0
                } else {
                    total_secs / runs.len() as f64
                };
                results.push(json!({
                    "name": name,
                    "iterations": runs.len(),
                    "total_tokens": total_tokens,
                    "total_seconds": total_secs,
                    "avg_seconds": avg_secs,
                    "tokens_per_second": if total_secs > 0.0 { total_tokens as f64 / total_secs } else { 0.0 },
                    "last_output": runs.last().map(|run| run.output.as_str()).unwrap_or(""),
                    "guided": options.guided,
                    "guide_fast_forward": options.guide_fast_forward,
                    "max_enc_len": options.max_enc_len,
                    "max_gen_len": options.max_gen_len,
                }));
            }
            println!(
                "{}",
                if *matrix {
                    json!({ "results": results })
                } else {
                    results.into_iter().next().unwrap_or_else(|| json!({}))
                }
            );
        }
    }
    Ok(())
}
