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

    /// Assign every role: use the override if it names a currently-served model,
    /// else the default pick. Errors only if hipfire serves nothing to assign.
    pub fn resolve(available: &[String], overrides: &RoleModels) -> anyhow::Result<RoleModels> {
        let default = default_pick(available)
            .ok_or_else(|| anyhow::anyhow!("hipfire reports no usable models to assign"))?;
        let mut out = BTreeMap::new();
        for role in Role::ALL {
            let model = overrides
                .0
                .get(&role)
                .filter(|m| available.iter().any(|a| a == *m))
                .cloned()
                .unwrap_or_else(|| default.to_string());
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

    #[test]
    fn resolve_honors_valid_overrides_and_fills_the_rest() {
        let available = vec![
            "EmbeddingGemma-300M".to_string(),
            "Gemma-3-27B".to_string(),
            "qwen3.5-9b".to_string(),
        ];
        let mut ov = RoleModels::default();
        ov.0.insert(Role::Coder, "qwen3.5-9b".to_string()); // valid
        ov.0.insert(Role::Review, "ghost-model".to_string()); // not served -> dropped

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

    #[test]
    fn resolve_errors_on_empty_model_list() {
        assert!(RoleModels::resolve(&[], &RoleModels::default()).is_err());
    }
}
