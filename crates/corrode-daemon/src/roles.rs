//! Model -> role assignment.
//!
//! The swarm runs subagents in distinct roles; each role wants a different model
//! (a tiny fast model for research fan-out, a big one for architecture, a
//! code-tuned one for the coder, etc.). Assignments are resolved once at startup
//! from two inputs: the live model list hipfire reports (`Client::list_models`)
//! and optional user overrides (a JSON `role -> model-id` map at `CORRODE_ROLES`).
//!
//! An override naming a model hipfire doesn't serve is ignored (not an error) and
//! falls back to the default pick, so a stale config never wedges startup.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Research,
    Orchestration,
    Architect,
    Coder,
    Review,
}

impl Role {
    pub const ALL: [Role; 5] = [
        Role::Research,
        Role::Orchestration,
        Role::Architect,
        Role::Coder,
        Role::Review,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Research => "research",
            Role::Orchestration => "orchestration",
            Role::Architect => "architect",
            Role::Coder => "coder",
            Role::Review => "review",
        }
    }

    /// Parse a role name (case-insensitive). Unknown -> None; callers decide the
    /// fallback (the planner defaults unknown roles to Coder).
    pub fn from_str(s: &str) -> Option<Role> {
        match s.trim().to_lowercase().as_str() {
            "research" => Some(Role::Research),
            "orchestration" => Some(Role::Orchestration),
            "architect" => Some(Role::Architect),
            "coder" => Some(Role::Coder),
            "review" => Some(Role::Review),
            _ => None,
        }
    }
}

/// Reasoning effort a role's generations run at: `CORRODE_EFFORT_<ROLE>` (e.g.
/// `CORRODE_EFFORT_ORCHESTRATION`), else `CORRODE_REASONING_EFFORT`, else the role's
/// default -- the planner thinks within a bounded budget (`medium`, 1024 tokens: it
/// must keep thinking, and an unbounded think ran out its output cap), every other
/// role does not (`none`). Always sent: hipfire's own default for a request that
/// names none (a per-model `reasoning_effort`, else unbudgeted thinking) is not what
/// a swarm role wants. Values are validated at startup (`knobs::check`).
pub fn effort_for(role: Role) -> String {
    let var = format!("CORRODE_EFFORT_{}", role.as_str().to_ascii_uppercase());
    std::env::var(&var)
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var("CORRODE_REASONING_EFFORT").ok().filter(|v| !v.is_empty()))
        .unwrap_or_else(|| match role {
            Role::Orchestration => "medium",
            _ => "none",
        }
        .to_string())
}

/// Resolved `role -> model id`. Every role is populated after [`RoleModels::resolve`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoleModels(pub BTreeMap<Role, String>);

impl RoleModels {
    pub fn model_for(&self, role: Role) -> Option<&str> {
        self.0.get(&role).map(String::as_str)
    }

    /// Read a `role -> model` override map from a JSON file (the `CORRODE_ROLES`
    /// path). Absent env var -> empty overrides.
    pub fn overrides_from_env() -> anyhow::Result<RoleModels> {
        match std::env::var("CORRODE_ROLES") {
            Ok(path) => {
                let text = std::fs::read_to_string(&path)?;
                Ok(serde_json::from_str(&text)?)
            }
            Err(_) => Ok(RoleModels::default()),
        }
    }

    /// Assign every role: its override, else the default pick. An override naming a
    /// model hipfire does not serve is an error, not a silent swap to the default --
    /// a typo in the roles file used to run that role on whatever model sorted first.
    /// Also errors if hipfire serves nothing to assign.
    pub fn resolve(available: &[String], overrides: &RoleModels) -> anyhow::Result<RoleModels> {
        let unserved: Vec<String> = overrides
            .0
            .iter()
            .filter(|(_, m)| !available.iter().any(|a| a == *m))
            .map(|(r, m)| format!("{} -> {m}", r.as_str()))
            .collect();
        if !unserved.is_empty() {
            anyhow::bail!("CORRODE_ROLES names models hipfire does not serve: {}", unserved.join(", "));
        }
        let default = default_pick(available)
            .ok_or_else(|| anyhow::anyhow!("hipfire reports no usable models to assign"))?;
        let mut out = BTreeMap::new();
        for role in Role::ALL {
            let model = overrides.0.get(&role).cloned().unwrap_or_else(|| default.to_string());
            out.insert(role, model);
        }
        Ok(RoleModels(out))
    }

    /// Assign one model to every role — the offline fallback when hipfire's list
    /// is unreachable (e.g. `CORRODE_MODEL`).
    pub fn uniform(model: &str) -> RoleModels {
        RoleModels(Role::ALL.iter().map(|&r| (r, model.to_string())).collect())
    }
}

/// Substrings that mark a model id as unable to drive a chat/coder role —
/// embeddings and image/diffusion models. `list_models` yields only ids (no arch
/// metadata), so this is necessarily name-based.
// ponytail: name heuristic — a new image family with none of these markers would
// still slip through. The real fix reads per-model arch from hipfire; until then,
// pin exact models via `CORRODE_ROLES`.
const NON_CHAT_MARKERS: &[&str] = &[
    "embed",            // embedding models (EmbeddingGemma, ...)
    ".dit",             // diffusion transformer (e.g. Krea-2-Turbo.dit)
    "diffusion",
    "krea", "flux", "sdxl", "sd3", "stable-diffusion", "-sd", "pixart", "kolors", "imagen",
];

/// The embedding model for retrieval (skill ranking, doc and code search):
/// `CORRODE_EMBED_MODEL` when set -- it must be served, and `off` turns embedding
/// retrieval off -- else [`default_embedding_model`]. The default takes the first id
/// containing "embed", which on a host with several (a bf16 copy hipfire can only
/// serve as an NPU artifact, a second family) may be one that cannot embed at all.
pub fn embedding_model(available: &[String], wanted: Option<&str>) -> anyhow::Result<Option<String>> {
    match wanted {
        Some(w) if crate::knobs::parse_flag(w) == Some(false) => Ok(None),
        Some(w) if available.iter().any(|a| a == w) => Ok(Some(w.to_string())),
        Some(w) => anyhow::bail!("CORRODE_EMBED_MODEL={w} is not a model hipfire serves"),
        None => Ok(default_embedding_model(available).map(str::to_string)),
    }
}

/// `CORRODE_EMBED_MODEL`, if set.
pub fn embed_model_env() -> Option<String> {
    std::env::var("CORRODE_EMBED_MODEL").ok().filter(|v| !v.is_empty())
}

/// The embedding model to use for retrieval (skill/doc selection): the first served
/// model that looks like an embedding model. `None` if hipfire serves none.
pub fn default_embedding_model(available: &[String]) -> Option<&str> {
    available
        .iter()
        .find(|id| id.to_lowercase().contains("embed"))
        .map(String::as_str)
}


/// Default model for unassigned roles: the first served model that isn't an
/// embedding or image/diffusion model. No size/capability ranking yet.
fn default_pick(available: &[String]) -> Option<&str> {
    let chatty = |id: &&String| {
        let l = id.to_lowercase();
        !NON_CHAT_MARKERS.iter().any(|m| l.contains(m))
    };
    available
        .iter()
        .find(chatty)
        .or_else(|| available.first())
        .map(String::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The planner thinks within a bounded budget and nothing else thinks, unless
    // the environment says otherwise -- per role first, then for every role.
    #[test]
    fn effort_is_bounded_thinking_for_the_planner_and_none_elsewhere() {
        for k in ["CORRODE_EFFORT_REVIEW", "CORRODE_EFFORT_ORCHESTRATION", "CORRODE_REASONING_EFFORT"] {
            std::env::remove_var(k);
        }
        assert_eq!(effort_for(Role::Orchestration), "medium");
        for r in [Role::Research, Role::Architect, Role::Coder, Role::Review] {
            assert_eq!(effort_for(r), "none", "{}", r.as_str());
        }
        std::env::set_var("CORRODE_EFFORT_REVIEW", "low");
        assert_eq!(effort_for(Role::Review), "low");
        assert_eq!(effort_for(Role::Coder), "none", "a per-role knob is that role's only");
        std::env::remove_var("CORRODE_EFFORT_REVIEW");
    }

    #[test]
    fn resolve_honors_valid_overrides_and_fills_the_rest() {
        let available = vec![
            "EmbeddingGemma-300M".to_string(),
            "Gemma-3-27B".to_string(),
            "qwen3.5-9b".to_string(),
        ];
        let mut ov = RoleModels::default();
        ov.0.insert(Role::Coder, "qwen3.5-9b".to_string()); // valid

        let r = RoleModels::resolve(&available, &ov).unwrap();
        assert_eq!(r.model_for(Role::Coder), Some("qwen3.5-9b"));
        // default pick skips the embedding model
        assert_eq!(r.model_for(Role::Review), Some("Gemma-3-27B"));
        assert_eq!(r.model_for(Role::Architect), Some("Gemma-3-27B"));
        // every role assigned
        assert!(Role::ALL.iter().all(|&role| r.model_for(role).is_some()));
    }

    #[test]
    fn default_pick_skips_embedding_and_image_models() {
        // Real hipfire ids: embeddings + Krea diffusion models must be skipped so the
        // default lands on the text model (regression for the Krea-2-Turbo.dit bug).
        let available = vec![
            "EmbeddingGemma-300M.oq4++".to_string(),
            "Krea-2-Turbo.dit.oq4.25".to_string(),
            "Krea-2-Turbo.source".to_string(),
            "zaya1-8b-native.oq8++".to_string(),
        ];
        let r = RoleModels::resolve(&available, &RoleModels::default()).unwrap();
        assert_eq!(r.model_for(Role::Coder), Some("zaya1-8b-native.oq8++"));
    }

    // An override hipfire does not serve is refused by name, not swapped for the
    // default pick.
    #[test]
    fn the_embedding_model_is_the_operators_when_named() {
        let served: Vec<String> = ["EmbeddingGemma-300M.bf16", "Qwen3-Embedding-0.6B--npu.oq8+.gfx1151", "Qwen3.8-27B--oq4.25++"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let pick = |w: Option<&str>| embedding_model(&served, w);
        assert_eq!(pick(None).unwrap().as_deref(), Some("EmbeddingGemma-300M.bf16"), "first 'embed' id");
        assert_eq!(
            pick(Some("Qwen3-Embedding-0.6B--npu.oq8+.gfx1151")).unwrap().as_deref(),
            Some("Qwen3-Embedding-0.6B--npu.oq8+.gfx1151")
        );
        assert_eq!(pick(Some("off")).unwrap(), None);
        assert!(pick(Some("Qwen3-Embedding-8B")).is_err(), "a typo is an error, not a silent default");
    }

    #[test]
    fn resolve_refuses_an_unserved_override() {
        let available = vec!["Gemma-3-27B".to_string()];
        let mut ov = RoleModels::default();
        ov.0.insert(Role::Review, "ghost-model".to_string());
        let err = RoleModels::resolve(&available, &ov).unwrap_err().to_string();
        assert!(err.contains("review -> ghost-model"), "{err}");
    }

    #[test]
    fn resolve_errors_on_empty_model_list() {
        assert!(RoleModels::resolve(&[], &RoleModels::default()).is_err());
    }

}
