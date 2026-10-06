//! The one parser for `CORRODE_*` on/off flags, and the startup check that refuses
//! a malformed knob.
//!
//! Each knob used to parse itself, so they disagreed and failed open:
//! `CORRODE_AUTO_APPROVE=yes` left the human gate on (only `1`/`true`/`on` counted)
//! and an unattended swarm blocked on its first write; `CORRODE_TURN_BUDGET_S=2h`
//! quietly meant unbounded; `CORRODE_REASONING_EFFORT=off` meant thinking with no
//! budget. Now every flag reads through [`flag`], and a set value that does not parse
//! stops the daemon at startup (and fails `doctor`), naming the knob.

/// `1`/`true`/`on`/`yes` or `0`/`false`/`off`/`no`, any case.
pub fn parse_flag(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// An on/off knob; unset or empty is `default`.
pub fn flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| parse_flag(&v))
        .unwrap_or(default)
}

const FLAGS: &[&str] = &[
    "CORRODE_SANDBOX",
    "CORRODE_SANDBOX_NET",
    "CORRODE_AUTO_APPROVE",
    "CORRODE_STREAM",
    "CORRODE_PLAN_REVIEW",
    "CORRODE_VFS_GRAPH",
    "CORRODE_VFS_VERIFY",
    "CORRODE_PLANNER_DETERMINISTIC",
];

/// Counts and whole seconds.
const WHOLE: &[&str] = &[
    "CORRODE_TURN_BUDGET_S",
    "CORRODE_TASK_TIMEOUT_S",
    "CORRODE_REQUEST_TIMEOUT_S",
    "CORRODE_COMMAND_TIMEOUT_S",
    "CORRODE_APPROVAL_TIMEOUT_S",
    "CORRODE_RETRY_WINDOW_S",
    "CORRODE_MAX_CONCURRENCY",
    "CORRODE_MAX_FOLLOWUPS",
    "CORRODE_MAX_TOOL_STEPS",
    "CORRODE_RESEARCH_TOOL_STEPS",
    "CORRODE_MAX_TOKENS",
    "CORRODE_CONTEXT_TOKENS",
    "CORRODE_FANOUT",
    "CORRODE_OPENAI_MAX_INFLIGHT",
];

/// Dollar amounts: the remote's per-turn budget and its prices.
const USD: &[&str] = &[
    "CORRODE_OPENAI_BUDGET_USD",
    "CORRODE_OPENAI_PRICE_IN",
    "CORRODE_OPENAI_PRICE_OUT",
];

const FRACTIONS: &[&str] = &["CORRODE_SKILL_ACTIVATE_MIN", "CORRODE_SKILL_LIST_MIN"];

/// What hipfire budgets; anything else it takes as thinking with no budget.
const EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Every set knob whose value does not parse, as `NAME="value": expected ...`.
pub fn check() -> Vec<String> {
    check_with(|k| std::env::var(k).ok())
}

fn check_with(get: impl Fn(&str) -> Option<String>) -> Vec<String> {
    let set = |k: &str| get(k).filter(|v| !v.is_empty());
    let mut bad = Vec::new();
    for k in FLAGS {
        if let Some(v) = set(k).filter(|v| parse_flag(v).is_none()) {
            bad.push(format!("{k}={v:?}: expected on/off (1/0, true/false, yes/no)"));
        }
    }
    for k in WHOLE {
        if let Some(v) = set(k).filter(|v| v.parse::<u64>().is_err()) {
            bad.push(format!("{k}={v:?}: expected a whole number"));
        }
    }
    for k in FRACTIONS {
        if let Some(v) = set(k).filter(|v| v.parse::<f32>().is_err()) {
            bad.push(format!("{k}={v:?}: expected a number"));
        }
    }
    for k in USD {
        if let Some(v) =
            set(k).filter(|v| !v.parse::<f64>().is_ok_and(|x| x.is_finite() && x >= 0.0))
        {
            bad.push(format!(
                "{k}={v:?}: expected a non-negative number of dollars"
            ));
        }
    }
    if let Some(v) = set("CORRODE_OPENAI_ROLES") {
        let unknown: Vec<&str> = v
            .split(',')
            .map(str::trim)
            .filter(|r| crate::roles::Role::from_str(r).is_none())
            .collect();
        if !unknown.is_empty() {
            bad.push(format!(
                "CORRODE_OPENAI_ROLES={v:?}: unknown role(s) {}; expected research/orchestration/architect/coder/review",
                unknown.join(", ")
            ));
        }
    }
    if let Some(v) = set("CORRODE_OPENAI_BASE_URL")
        .filter(|v| !v.starts_with("http://") && !v.starts_with("https://"))
    {
        bad.push(format!(
            "CORRODE_OPENAI_BASE_URL={v:?}: expected an http(s) URL"
        ));
    }
    // Remote settings without a model would leave routing silently off.
    if set("CORRODE_OPENAI_MODEL").is_none() {
        for k in [
            "CORRODE_OPENAI_ROLES",
            "CORRODE_OPENAI_BASE_URL",
            "CORRODE_OPENAI_BUDGET_USD",
        ] {
            if set(k).is_some() {
                bad.push(format!(
                    "{k} is set but CORRODE_OPENAI_MODEL is not: name the remote model"
                ));
            }
        }
    }
    let per_role = crate::roles::Role::ALL
        .map(|r| format!("CORRODE_EFFORT_{}", r.as_str().to_ascii_uppercase()));
    for k in per_role.iter().map(String::as_str).chain(["CORRODE_REASONING_EFFORT"]) {
        if let Some(v) = set(k).filter(|v| !EFFORTS.contains(&v.as_str())) {
            bad.push(format!("{k}={v:?}: expected one of {}", EFFORTS.join("/")));
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_of(pairs: &[(&str, &str)]) -> Vec<String> {
        let m: std::collections::HashMap<_, _> = pairs.iter().copied().collect();
        check_with(|k| m.get(k).map(|v| v.to_string()))
    }

    #[test]
    fn flags_accept_every_spelling_and_nothing_else() {
        for v in ["1", "true", "ON", "Yes"] {
            assert_eq!(parse_flag(v), Some(true), "{v}");
        }
        for v in ["0", "false", "Off", "no"] {
            assert_eq!(parse_flag(v), Some(false), "{v}");
        }
        for v in ["y", "enable", " on", "2"] {
            assert_eq!(parse_flag(v), None, "{v}");
        }
    }

    #[test]
    fn malformed_bounds_are_refused_by_name() {
        let bad = check_of(&[
            ("CORRODE_AUTO_APPROVE", "always"),
            ("CORRODE_TURN_BUDGET_S", "2h"),
            ("CORRODE_MAX_CONCURRENCY", "8 "),
            ("CORRODE_SKILL_LIST_MIN", "high"),
            ("CORRODE_REASONING_EFFORT", "off"),
            ("CORRODE_EFFORT_ORCHESTRATION", "max-ish"),
        ]);
        assert_eq!(bad.len(), 6, "{bad:?}");
        for k in [
            "AUTO_APPROVE",
            "TURN_BUDGET_S",
            "MAX_CONCURRENCY",
            "SKILL_LIST_MIN",
            "REASONING_EFFORT",
            "EFFORT_ORCHESTRATION",
        ] {
            assert!(bad.iter().any(|b| b.contains(k)), "{k} not refused: {bad:?}");
        }
    }

    #[test]
    fn remote_knobs_are_checked_and_need_a_model() {
        let bad = check_of(&[
            ("CORRODE_OPENAI_ROLES", "review,boss"),
            ("CORRODE_OPENAI_BUDGET_USD", "-1"),
            ("CORRODE_OPENAI_PRICE_IN", "cheap"),
            ("CORRODE_OPENAI_BASE_URL", "api.openai.com/v1"),
        ]);
        for k in [
            "boss",
            "BUDGET_USD=",
            "PRICE_IN",
            "http(s) URL",
            "CORRODE_OPENAI_MODEL is not",
        ] {
            assert!(
                bad.iter().any(|b| b.contains(k)),
                "{k} not refused: {bad:?}"
            );
        }
        assert!(check_of(&[
            ("CORRODE_OPENAI_MODEL", "deepseek-chat"),
            ("CORRODE_OPENAI_ROLES", "orchestration, review"),
            ("CORRODE_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            ("CORRODE_OPENAI_BUDGET_USD", "2"),
            ("CORRODE_OPENAI_PRICE_IN", "0.27"),
        ])
        .is_empty());
    }

    // The VFS flags parsed themselves, so `yes` silently meant off.
    #[test]
    fn vfs_flags_go_through_the_shared_parser() {
        let bad = check_of(&[("CORRODE_VFS_GRAPH", "maybe"), ("CORRODE_VFS_VERIFY", "sure")]);
        assert_eq!(bad.len(), 2, "{bad:?}");
        assert!(check_of(&[("CORRODE_VFS_GRAPH", "yes"), ("CORRODE_VFS_VERIFY", "0")]).is_empty());
    }

    #[test]
    fn well_formed_and_unset_knobs_pass() {
        assert!(check_of(&[]).is_empty());
        assert!(check_of(&[
            ("CORRODE_AUTO_APPROVE", "yes"),
            ("CORRODE_SANDBOX", ""),
            ("CORRODE_TURN_BUDGET_S", "7200"),
            ("CORRODE_SKILL_ACTIVATE_MIN", "0.35"),
            ("CORRODE_REASONING_EFFORT", "low"),
        ])
        .is_empty());
    }
}
