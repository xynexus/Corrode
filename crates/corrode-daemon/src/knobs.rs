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
    let k = "CORRODE_REASONING_EFFORT";
    if let Some(v) = set(k).filter(|v| !EFFORTS.contains(&v.as_str())) {
        bad.push(format!("{k}={v:?}: expected one of {}", EFFORTS.join("/")));
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
        ]);
        assert_eq!(bad.len(), 5, "{bad:?}");
        for k in ["AUTO_APPROVE", "TURN_BUDGET_S", "MAX_CONCURRENCY", "SKILL_LIST_MIN", "REASONING_EFFORT"] {
            assert!(bad.iter().any(|b| b.contains(k)), "{k} not refused: {bad:?}");
        }
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
