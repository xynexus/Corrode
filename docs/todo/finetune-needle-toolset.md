# TODO: Finetune Needle on Corrode's tool set

**Goal:** make the Needle tool-caller reliably produce Corrode's tool calls — valid
JSON, correct tool selection, and full multi-field argument extraction — so the
single-param workarounds we built to cope with the base weights can be retired.

Upstream trainer: the `third_party/needle` submodule (Cactus's Needle). It ships the
data format, a Gemini-backed data generator, a local finetuner, and an evaluator
(`needle finetune`, `needle playground`, `needle eval`). See its `README.md` →
**Tool schema** / **Data format** / **Finetuning**.

---

## Why (the observed failure modes)

The base weights handle **one clean single-param intent** well but degrade on the
shapes Corrode actually needs. Observed this session:

- **Multi-param → invalid JSON.** `write_file(path, contents)` and the original
  `run_skill_script(skill, script)` emit a bare key with no value
  (`{"skill":"…","script"}`), which won't parse. Single-param tools are fine.
- **Argument truncation / stray words.** `"Write unit tests for add() covering
  overflow"` → `task:"covering integer overflow"`; `"run the hello.sh script"` →
  `target:"hello.sh script"`.
- **Dropped fields from prose.** `"run hook.mjs from the impeccable skill"` →
  `target:"hook.mjs"` (skill name dropped).
- **Imperfect tool selection.** Role classification via tool name is mostly right for
  clear intents but misses ambiguous ones (`"show me README.md"` → `list_dir`).
- **Enum args unreliable** — why role lives in the tool *name* (`coding_task`), not an
  enum arg.

## What to finetune for

Corrode's canonical tool sets (see `crates/corrode-daemon/src/tools.rs` `EXEC_TOOLS`
and `crates/corrode-daemon/src/plan_graph.rs` `ROLE_TOOLS`), in the **Needle-flat**
schema the `ToolDialect` renders:

| Tool | Params | Notes |
|---|---|---|
| `read_file` | `path` | single-param — already reliable |
| `list_dir` | `path` | single-param |
| `write_file` | `path`, `contents` | **multi-param — primary target** |
| `run_command` | `command` | single-param |
| `run_skill_script` | `target` (`skill/script`) | want reliable `skill`+`script`, ideally split |
| `research_task` / `coding_task` / `architecture_task` / `review_task` | `task` | role = tool name; want sharper selection |

Success = every tool above emits **valid JSON with all required args fully populated**,
tool selection ≥ ~95% on held-out phrasings, and no truncation of paths/contents.

---

## Tasks

- [ ] **Export the training schema from the code, not by hand.** Add a small exporter
      (or a `cargo test`/bin) that dumps `ToolDialect::default().render(EXEC_TOOLS)` and
      `…render(ROLE_TOOLS)` so the *training* schema is byte-identical to what the daemon
      sends at runtime. Prevents train/serve schema skew.
- [ ] **Generate a dataset** in Needle's JSONL format (`{query, tools, answers}` per
      line; `tools`/`answers` are JSON-encoded strings), using **`fixtures/demo-repo`**
      (the `xynexus/corrode-demo` submodule) as the realistic repo so queries reference
      real paths/files/skills. Cover, per the README's **≥120 examples per tool** (100
      train / 10 val / 10 test):
  - [ ] The **plain-English phrasings the tool loop actually produces** — `TOOL:` lines
        (imperatives like "read the file X", "write <contents> to <path>", "run <cmd>")
        and `NEXT:` lines for role classification.
  - [ ] **Multi-param** examples for `write_file` (path + full contents) and a
        `run_skill_script` that fills both `skill` and `script`.
  - [ ] **Adversarial phrasings** that currently break it: prose with extra words,
        dropped skill names, embedded paths, long content bodies.
  - [ ] Examples with **multiple tools available** so selection is trained under choice.
  - [ ] Role-classification examples across all four `*_task` tools, including ambiguous
        verbs ("show me", "look at", "make sure").
- [ ] **Finetune** locally: `needle finetune data.jsonl` (or `needle playground`), then
      `needle eval --checkpoint …` on the held-out split.
- [ ] **Export the checkpoint** to the shim's asset layout with
      `crates/needle-toolcall-shim/scripts/export_needle_checkpoint.py` → a new
      `assets/needle-corrode/` (keep the base weights alongside for comparison).
- [ ] **Wire the finetuned model in:** point `CORRODE_NEEDLE_ASSETS` at it and set
      `CORRODE_NEEDLE_MODEL_ID` (e.g. `needle-corrode-v1`). Add a
      `CORRODE_TOOL_DIALECTS` profile for that id if its expected schema/names differ
      from the default.
- [ ] **Validate** against the ignored e2e tests with the new weights:
      `cargo test -p corrode-daemon --features needle -- --ignored` — plus new cases for
      `write_file`/multi-param that currently can't pass.

## Workarounds to retire once reliable

These exist only because the base weights are weak; grep and remove/simplify after the
finetune clears the acceptance bar:

- [ ] `run_skill_script`'s single `target` param + `first_token` leniency + bare-name
      resolution (`tools.rs`) → go back to a real `skill` + `script` multi-param tool.
- [ ] `emit_followups` using the **verbatim `NEXT:` text** instead of Needle's `task`
      argument (`daemon.rs`) → trust the (now-untruncated) arg.
- [ ] The general **single-param bias** in tool design — `write_file` etc. can rely on
      multi-param extraction.
- [ ] `is_mutating`/approval unaffected; keep the approval gate regardless.

## Acceptance criteria

- [ ] 100% valid-JSON tool calls across the eval split (no bare keys).
- [ ] Multi-param args fully populated; no path/contents truncation.
- [ ] Tool selection ≥ ~95% on held-out phrasings; role classification ≥ ~90%.
- [ ] The retired-workaround changes above land with green base + `--features needle`.

## References

- `third_party/needle/README.md` — Tool schema, Data format, `needle finetune` /
  `playground` / `generate-data` / `eval`.
- `crates/needle-toolcall-shim/scripts/export_needle_checkpoint.py` — checkpoint →
  shim assets.
- `crates/corrode-daemon/src/dialect.rs` — canonical tools + schema rendering (the
  source of truth for the training schema).
- CLAUDE.md → "Tool dialects" / "Tool-calling (Needle shim)".
