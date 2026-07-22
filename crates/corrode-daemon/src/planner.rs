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

use crate::roles::{Role, RoleModels};
use crate::swarm::Task;
use corrode_core::Priority;
use serde::Deserialize;

/// Upper bound on subtasks per prompt — a runaway-plan backstop.
/// ponytail: fixed cap; make it budget-aware once we track per-request cost.
const MAX_SUBTASKS: usize = 8;

/// Instruction handed to the orchestration model. Asks for a strict JSON plan.
///
/// ponytail: no shared context prefix yet. The KV-reuse win (see CLAUDE.md) comes
/// from prepending the same repo/context digest to every subtask prompt so hipfire
/// batches them prefix-shared — add that when the VFS feeds real context in.
pub fn orchestration_prompt(user_prompt: &str) -> String {
    format!(
        "You are the orchestrator of a coding-agent swarm. Decompose the user's \
request into a small set of subtasks, each assigned to one role from: research, \
architect, coder, review. Reply with ONLY a JSON array, no prose, each element \
{{\"role\": <role>, \"task\": <self-contained instruction>}}. Use at most {MAX_SUBTASKS} \
subtasks.\n\nUser request:\n{user_prompt}"
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

/// Map planned subtasks to runnable swarm tasks (role -> model, role -> band).
pub fn to_tasks(plan: Vec<PlannedSubtask>, roles: &RoleModels) -> Vec<Task> {
    plan.into_iter()
        .map(|s| Task {
            model: roles.model_for(s.role).unwrap_or_default().to_string(),
            priority: band_for(s.role),
            prompt: s.prompt,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn to_tasks_assigns_role_model_and_band() {
        let roles = {
            let mut r = RoleModels::default();
            r.0.insert(Role::Coder, "coder-model".to_string());
            r.0.insert(Role::Research, "research-model".to_string());
            r
        };
        let plan = vec![
            PlannedSubtask { role: Role::Coder, prompt: "x".into() },
            PlannedSubtask { role: Role::Research, prompt: "y".into() },
        ];
        let tasks = to_tasks(plan, &roles);
        assert_eq!(tasks[0].model, "coder-model");
        assert_eq!(tasks[0].priority, Priority::Default);
        assert_eq!(tasks[1].model, "research-model");
        assert_eq!(tasks[1].priority, Priority::Opportunistic); // research fills idle
    }

    #[test]
    fn parse_plan_returns_empty_on_junk() {
        assert!(parse_plan("no json here").is_empty());
    }
}
