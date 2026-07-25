# Agent Notes

This repository is a standalone Rust/Candle tool-call shim for the Needle model.
It should stay decoupled from SmallCode, Hermes, OpenCode, or any other agent
runtime. Treat those projects only as schema/test-corpus sources.

## Assets

The default runtime assets are committed under `assets/needle`:

- `model.safetensors`
- `config.json`
- `needle.model`
- `needle.vocab`
- `SHA256SUMS`
- `LICENSE`

Verify asset integrity with:

```bash
cd assets/needle && sha256sum -c SHA256SUMS
```

Do not commit `needle.pkl`, local HF caches, Python virtualenvs, or local
upstream checkouts. The pickle exporter is kept only as a regeneration path.

## Checks

Before handing off changes, run:

```bash
scripts/check.sh
```

For full parity and benchmark passes:

```bash
NEEDLE_RUN_PARITY=1 scripts/check.sh
NEEDLE_RUN_BENCH=1 scripts/check.sh
```

The benchmark can also be run directly:

```bash
cargo run --release -- bench --matrix \
  --query "What's the weather in San Francisco?" \
  --tools '[{"name":"get_weather","parameters":{"location":"string"}}]' \
  --iterations 3
```

## Runtime Invariants

- `infer` stdout must contain only the final JSON tool-call text.
- Keep CPU inference correctness ahead of lower-level optimization.
- Keep guided decoding deterministic: hard masks should only allow valid schema
  continuations, and tool names must restore to their original spelling.
- The decoder uses a self-attention KV cache during generation; forced guide
  tokens must still advance the model cache.
- Do not introduce framework-specific request formats into the public CLI.
