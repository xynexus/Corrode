# needle-toolcall-shim

Standalone Rust/Candle CPU shim for the Needle encoder-decoder tool-call model.

The runtime contract is deliberately narrow:

```text
query + tools JSON -> JSON tool calls
```

It has no dependency on any agent framework. The only runtime inputs are Needle assets and request-time query/tool JSON.

## Assets

Needle runtime assets are committed under `assets/needle`. They are derived from
the MIT-licensed upstream Needle checkpoint and tokenizer.

Expected files:

```text
assets/needle/model.safetensors
assets/needle/config.json
assets/needle/needle.model
assets/needle/needle.vocab
assets/needle/SHA256SUMS
assets/needle/LICENSE
```

To verify the committed assets:

```bash
cd assets/needle && sha256sum -c SHA256SUMS
```

To regenerate the safetensors export from a local upstream pickle:

```bash
python3 scripts/export_needle_checkpoint.py \
  --checkpoint needle-weights/needle.pkl \
  --tokenizer needle-weights/tokenizer/needle.model \
  --out assets/needle
```

## Inspect

```bash
cargo run -- inspect --assets assets/needle
```

## Infer

```bash
cargo run -- infer \
  --assets assets/needle \
  --query "What's the weather in San Francisco?" \
  --tools '[{"name":"get_weather","parameters":{"location":"string"}}]'
```

`infer` writes only final JSON text to stdout. Useful switches:

```text
--max-gen-len N
--max-enc-len N
--unconstrained
--no-guide-fast-forward
--no-normalize
--json-errors
```

Guided decoding constrains Needle's compact JSON structure, tool names, argument keys, enum values, booleans, and nulls. Free-form strings remain model-driven.

## Benchmark

```bash
cargo run --release -- bench --matrix \
  --assets assets/needle \
  --query "What's the weather in San Francisco?" \
  --tools '[{"name":"get_weather","parameters":{"location":"string"}}]' \
  --iterations 3
```

The benchmark prints JSON with elapsed time, generated tokens, and tokens/sec.
Use `--release` for meaningful timings. `--matrix` compares guided, guided
without fast-forward, and unconstrained decoding.

The correctness suite does not enforce a performance threshold. To run the
asset-backed benchmark test explicitly:

```bash
cargo test --release --test benchmark -- --ignored --nocapture
# or
NEEDLE_RUN_BENCH=1 scripts/check.sh
```

## Checks

Fast checks:

```bash
scripts/check.sh
```

Full local parity checks use the committed assets:

```bash
NEEDLE_RUN_PARITY=1 scripts/check.sh
```

Refresh Python fixtures with an optional local upstream Needle checkout:

```bash
needle/.venv/bin/python scripts/generate_python_fixtures.py
needle/.venv/bin/python scripts/generate_python_probes.py
```

## Tool-Suite Corpus

The test corpus in `tests/fixtures/tool_suites.json` covers broad tool lists from:

- OpenCode built-ins
- SmallCode-style coding tools
- Hermes-style domain/API tools
- Current Hermes session tools imported from a local export

These are committed as compact OpenAI-style schemas so guide parsing, normalization, enum/boolean/null constraints, duplicate-key handling, and encoder packing can be tested without depending on another agent runtime at test time.

`tests/fixtures/tool_suite_cases.json` contains three generated sample calls per tool:

- `minimal_required`
- `full_arguments`
- `normalized_name`

Regenerate it after editing the suite corpus:

```bash
python3 scripts/generate_tool_suite_cases.py
```
