//! The planner: the orchestration model decomposes a prompt into role-tagged
//! subagents, which the swarm then runs — each on its role's model and priority
//! band. This is what makes Corrode a *swarm* rather than a single agent.
//!
//! Flow (in the daemon's Prompt handler):
//!   1. ask the orchestration model for a plan (JSON subtasks),
//!   2. [`parse_plan`] extracts it,
//!   3. [`to_tasks`] maps each subtask to a `Task` (role -> model, role -> band),
//!   4. the swarm fans them out.
//!
//! Bands come from the role, not the model — foreground planning is Realtime,
//! build work is Default, and speculative research fills idle GPU
//! (Opportunistic). Keeping band assignment here (not asking the model to pick)
//! means the swarm stays predictable against hipfire's scheduler.

use crate::roles::Role;
use corrode_core::Priority;
use serde::Deserialize;

/// Upper bound on subtasks per prompt — a runaway-plan backstop.
/// ponytail: fixed cap; make it budget-aware once we track per-request cost.
const MAX_SUBTASKS: usize = 8;

/// Instruction handed to the orchestration model, behind the shared context
/// prefix. Placing `context_prefix` first — byte-identical to every subtask's
/// prefix (see [`to_tasks`]) — lets hipfire batch the planning call and the
/// subagents prefix-shared and reuse their KV cache, when they land on the same
/// model. Only the tail (instructions, then the user request) diverges.
pub fn orchestration_prompt(context_prefix: &str, user_prompt: &str) -> String {
    format!(
        "{context_prefix}\n\n\
You are the orchestrator of a coding-agent swarm. Decompose the user's request \
into a small set of subtasks, each assigned to one role from: research, architect, \
coder, review. Reply with ONLY a JSON array, no prose, each element \
{{\"role\": <role>, \"task\": <self-contained instruction>}}. Use at most {MAX_SUBTASKS} \
subtasks.\n\nUser request:\n{user_prompt}"
    )
}

/// Compose one subagent prompt: the shared prefix, then the divergent role+task
/// tail. The prefix must be byte-identical across the whole swarm for KV reuse, so
/// nothing role-specific goes before it. The tail also invites the agent to propose a
/// single follow-up (a test contract, a research spin-off) as a plain-English `NEXT:`
/// line — easy for a small model to write, unlike a JSON tool call. The reactive
/// planner ([`crate::plan_graph`]) classifies its role (via Needle) and schedules it;
/// more follow-ups emerge as that task runs and proposes its own.
pub fn subagent_prompt(context_prefix: &str, role: Role, task: &str) -> String {
    format!(
        "{context_prefix}\n\n[role: {}]\n{task}\n\n\
(Optional) If one clear follow-up is warranted, end your reply with a single line:\n\
NEXT: <one plain-English instruction for the next task>\n\
Write plain English, not JSON. Omit the line if no follow-up is needed.",
        role.as_str()
    )
}

/// Compose a turn for a model that emits its own tool calls.
///
/// Deliberately does NOT teach a call syntax: the tools are declared on the request and
/// the model's chat template renders the format it was trained on. Describing a second
/// syntax here would compete with that one. The shared prefix still leads, so hipfire
/// reuses the KV prefill across the swarm; only the scratchpad tail grows.
pub fn native_tool_prompt(context_prefix: &str, role: Role, task: &str, scratchpad: &str) -> String {
    format!(
        "{context_prefix}\n\n[role: {}]\n{task}\n{scratchpad}\n\
You have tools available. Call one when you need it — you will get the result and can \
continue. Never guess a file's contents: read it first. When you have enough to answer, \
reply with your final answer and no tool call. Optionally end with:\n\
NEXT: <one plain-English follow-up task>",
        role.as_str()
    )
}

/// Compose a tool-loop turn for a small model: the shared prefix, the role+task, the
/// scratchpad of tool calls/results so far, then instructions. The model acts by writing
/// a plain-English `TOOL:` line (Needle structures it — the small model never writes a
/// tool call); it finishes with a turn that has no `TOOL:` line. The shared prefix stays
/// byte-identical across turns and across the swarm, so hipfire reuses the KV prefill;
/// only the scratchpad tail grows.
pub fn tool_loop_prompt(context_prefix: &str, role: Role, task: &str, scratchpad: &str) -> String {
    format!(
        "{context_prefix}\n\n[role: {}]\n{task}\n{scratchpad}\n\
You can use tools. To use one, write a line:\n\
TOOL: <one plain-English request> (e.g. TOOL: read the file crates/corrode-core/src/lib.rs)\n\
Write plain English, not JSON — the tool call is constructed for you. You will get the \
result and can continue. When you have enough to answer, reply with your final answer and \
NO TOOL: line. Optionally end with:\n\
NEXT: <one plain-English follow-up task>",
        role.as_str()
    )
}

/// The task text for one read-only proposal attempt in a coder fan-out
/// (`CORRODE_FANOUT`). The attempt explores but cannot mutate; its deliverable is a
/// proposal the review judge weighs against its siblings'. Variation lives in this
/// tail — the shared prefix stays byte-identical, so hipfire batches the whole
/// ensemble prefix-shared.
pub fn fanout_attempt_task(task: &str, attempt: usize, of: usize) -> String {
    format!(
        "{task}\n[proposal {attempt}/{of}: this pass is read-only — explore as needed, then \
end with your proposed approach and the exact changes to make. A reviewer compares \
independent proposals and one implementer executes the winner.]"
    )
}

/// The full judge prompt for a fan-out: shared prefix first (byte-identical — KV
/// reuse), then the task and every surviving proposal in the tail.
pub fn fanout_judge_prompt(context_prefix: &str, task: &str, proposals: &[String]) -> String {
    let mut tail = format!(
        "[role: review]\nYou are judging {} independent proposals for this task:\n{task}\n",
        proposals.len()
    );
    for (i, p) in proposals.iter().enumerate() {
        tail.push_str(&format!("\n--- proposal {} ---\n{p}\n", i + 1));
    }
    tail.push_str(
        "\nPick the strongest approach, folding in anything clearly better from the others, \
and reply with ONE precise directive for the implementing agent — the concrete changes, \
files, and checks. Reply with only the directive.",
    );
    format!("{context_prefix}\n\n{tail}")
}

/// The GraphRAG synthesis prompt: answer `question` grounded ONLY in the retrieved
/// `chunks` (`(id, text)`), citing the chunk ids used. Kept strict — no outside
/// knowledge, and an explicit "not in the docs" escape — so the answer stays
/// attributable to the store, which is the whole point of retrieval-grounding.
pub fn doc_synthesis_prompt(question: &str, chunks: &[(String, String)]) -> String {
    let mut s = String::from(
        "Answer the question using ONLY the reference excerpts below. Do not use outside \
knowledge. Cite the excerpt ids you rely on in square brackets, e.g. [chunk:foo#0]. If the \
excerpts don't contain the answer, say so plainly.\n\n",
    );
    s.push_str("Excerpts:\n");
    for (id, text) in chunks {
        s.push_str(&format!("[{id}]\n{text}\n\n"));
    }
    s.push_str(&format!("Question: {question}\n\nAnswer:"));
    s
}

/// The task text for the plan-level review pass: the settled plan's digest, plus an
/// instruction to verify against the repo and route fixes through the normal
/// follow-up channel (`NEXT:` line -> emitted fix task).
pub fn plan_review_task(digest: &str) -> String {
    format!(
        "Review the work this plan just completed. The digest below lists each task, its \
output, and the files it wrote. Verify against the actual repo — read the written files \
rather than trusting the outputs. If you find a defect or a gap, name the single most \
important fix as your NEXT: line; if the work holds, say so and emit no follow-up.\n\n\
{digest}"
    )
}

#[derive(Deserialize)]
struct RawSubtask {
    role: String,
    task: String,
}

/// One decomposed unit of work: a role and its instruction.
#[derive(Debug, PartialEq, Eq)]
pub struct PlannedSubtask {
    pub role: Role,
    pub prompt: String,
}

/// Extract the subtask list from the orchestration model's reply. Tolerant of
/// surrounding prose: parses the whole text as JSON, else the first `[`..last `]`
/// slice. Unknown role names fall back to Coder. Returns empty if nothing parses.
///
/// ponytail: the bracket-slice fallback is naive (it ignores brackets inside JSON
/// string values). Fine for well-behaved plans; tighten if models start embedding
/// arrays in task text.
pub fn parse_plan(text: &str) -> Vec<PlannedSubtask> {
    let raw: Vec<RawSubtask> = serde_json::from_str(text)
        .ok()
        .or_else(|| {
            let start = text.find('[')?;
            let end = text.rfind(']')?;
            if end <= start {
                return None;
            }
            serde_json::from_str(&text[start..=end]).ok()
        })
        .unwrap_or_default();

    raw.into_iter()
        .take(MAX_SUBTASKS)
        .filter(|r| !r.task.trim().is_empty())
        .map(|r| PlannedSubtask {
            role: Role::from_str(&r.role).unwrap_or(Role::Coder),
            prompt: r.task,
        })
        .collect()
}

/// Default priority band for a subagent role.
pub fn band_for(role: Role) -> Priority {
    match role {
        Role::Orchestration => Priority::Realtime,
        Role::Architect | Role::Coder | Role::Review => Priority::Default,
        Role::Research => Priority::Opportunistic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_synthesis_prompt_grounds_and_carries_chunks() {
        let chunks = vec![
            ("chunk:cpu.md#0".to_string(), "R15 is the program counter.".to_string()),
            ("chunk:cpu.md#1".to_string(), "Interrupts vector at 0xFFFF0000.".to_string()),
        ];
        let p = doc_synthesis_prompt("where do interrupts vector?", &chunks);
        // the question, both chunk ids, and both texts must all reach the model,
        // with an explicit grounding instruction and citation format.
        assert!(p.contains("where do interrupts vector?"));
        assert!(p.contains("[chunk:cpu.md#0]") && p.contains("[chunk:cpu.md#1]"));
        assert!(p.contains("0xFFFF0000") && p.contains("program counter"));
        assert!(p.contains("ONLY") && p.to_lowercase().contains("cite"));
    }

    #[test]
    fn parse_plan_extracts_from_surrounding_prose_and_defaults_unknown_role() {
        let out = "Here is the plan:\n\
            [{\"role\":\"coder\",\"task\":\"write the parser\"},\
             {\"role\":\"research\",\"task\":\"survey prior art\"},\
             {\"role\":\"wizard\",\"task\":\"cast a spell\"}]\nDone.";
        let plan = parse_plan(out);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].role, Role::Coder);
        assert_eq!(plan[1].role, Role::Research);
        assert_eq!(plan[2].role, Role::Coder); // unknown "wizard" -> Coder
    }

    #[test]
    fn subagent_prompt_shares_prefix_and_bands_by_role() {
        let prefix = "SHARED-CONTEXT-DIGEST";
        let coder = subagent_prompt(prefix, Role::Coder, "write the parser");
        let research = subagent_prompt(prefix, Role::Research, "survey prior art");

        // KV-reuse invariant: every subagent prompt begins with the identical prefix;
        // only the tail (role/task) differs.
        assert!(coder.starts_with(prefix));
        assert!(research.starts_with(prefix));
        assert!(coder.contains("[role: coder]"));
        assert!(research.contains("[role: research]"));
        assert_ne!(coder, research);

        // bands come from the role: build work Default, research fills idle GPU.
        assert_eq!(band_for(Role::Coder), Priority::Default);
        assert_eq!(band_for(Role::Research), Priority::Opportunistic);
    }

    #[test]
    fn parse_plan_returns_empty_on_junk() {
        assert!(parse_plan("no json here").is_empty());
    }

    // Fan-out prompts obey the same KV-reuse invariant as subagent prompts: the shared
    // prefix leads the judge prompt, and attempt variation lives after the task text.
    #[test]
    fn fanout_and_review_prompts_keep_the_shared_prefix_leading() {
        let prefix = "SHARED-CONTEXT-DIGEST";
        let judge = fanout_judge_prompt(
            prefix,
            "write the parser",
            &["use nom".into(), "hand-roll it".into()],
        );
        assert!(judge.starts_with(prefix));
        assert!(judge.contains("write the parser"), "judge sees the task");
        assert!(judge.contains("--- proposal 1 ---\nuse nom"));
        assert!(judge.contains("--- proposal 2 ---\nhand-roll it"));

        let attempt = fanout_attempt_task("write the parser", 2, 3);
        assert!(attempt.starts_with("write the parser"));
        assert!(attempt.contains("[proposal 2/3"));

        let review = plan_review_task("task 0 [coder]: write the parser\nwrote: src/parser.rs");
        assert!(review.contains("NEXT:"));
        assert!(review.ends_with("wrote: src/parser.rs"));
    }
}
