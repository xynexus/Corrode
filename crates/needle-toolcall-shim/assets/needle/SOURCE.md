# Needle Assets

These files are derived from the MIT-licensed Needle model published by Cactus
Compute:

- Model: https://huggingface.co/Cactus-Compute/needle
- Source code/license: https://github.com/cactus-compute/needle

The tokenizer files are copied from the Hugging Face repository. The safetensors
file and runtime config were exported from the upstream `needle.pkl` checkpoint
with `scripts/export_needle_checkpoint.py`.

The upstream pickle is not committed here. For provenance, the source checkpoint
used for this export had SHA256:

```text
40a32e91d1d4197bf15ba559b74f6727c342dc8746918742fc7d8e2c1f18df40  needle.pkl
```
