//! A second backend for the work the local models struggle with: any server that
//! speaks OpenAI's Chat Completions -- OpenAI itself, or vLLM, llama.cpp, a hosted
//! GLM or DeepSeek.
//!
//! **Off unless `CORRODE_OPENAI_MODEL` names a model** (there is no default model: a
//! baked-in id would go stale or silently pick a pricier one). Routing:
//! - the roles in `CORRODE_OPENAI_ROLES` (comma-separated) run there first, and
//!   fall back to hipfire when the remote fails;
//! - any other task that fails on hipfire, after its one retry, is escalated there
//!   once.
//!
//! Spend is capped per Prompt turn: `CORRODE_OPENAI_BUDGET_USD` (default 2), priced
//! from `CORRODE_OPENAI_PRICE_IN` / `CORRODE_OPENAI_PRICE_OUT` (USD per million
//! tokens). Past the cap, everything stays on hipfire. Unpriced (a local vLLM),
//! nothing is counted and the cap never binds. `CORRODE_OPENAI_MAX_INFLIGHT`
//! (default 4) caps concurrent requests; a 429 is retried like hipfire's.
//!
//! The endpoint is `CORRODE_OPENAI_BASE_URL` (default `https://api.openai.com/v1`),
//! the key `CORRODE_OPENAI_API_KEY`, else `OPENAI_API_KEY` (optional: a local server
//! may need none). hipfire's requests are untouched -- this is a second `Client`.
//!
//! `CORRODE_OPENAI_REASONING_EFFORT` sends a `reasoning_effort`, for a model that
//! takes one: `role` sends each role's own (`CORRODE_EFFORT_<ROLE>` and the
//! defaults; a role with none sends nothing), a level sends that level. Unset,
//! nothing is sent: many servers refuse a field they do not know.

use crate::hipfire::{Client, Usage};
use crate::roles::Role;

pub struct Remote {
    pub client: Client,
    pub model: String,
    roles: Vec<Role>,
    budget_usd: f64,
    /// USD per million input / output tokens.
    price_in: f64,
    price_out: f64,
}

fn env(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

impl Remote {
    /// From `CORRODE_OPENAI_*`; `None` when no model is named. Values are checked at
    /// startup (`knobs::check`), so a malformed one is never seen here.
    pub fn from_env() -> Option<Self> {
        let model = env("CORRODE_OPENAI_MODEL")?;
        let number = |k: &str, default: f64| env(k).and_then(|v| v.parse().ok()).unwrap_or(default);
        let base =
            env("CORRODE_OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com/v1".into());
        let key = env("CORRODE_OPENAI_API_KEY").or_else(|| env("OPENAI_API_KEY"));
        let inflight = env("CORRODE_OPENAI_MAX_INFLIGHT")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        Some(Self {
            client: Client::openai_compatible(&base, key, inflight)
                .with_reasoning_effort(env("CORRODE_OPENAI_REASONING_EFFORT").as_deref()),
            model,
            roles: env("CORRODE_OPENAI_ROLES").map_or_else(Vec::new, |v| parse_roles(&v)),
            budget_usd: number("CORRODE_OPENAI_BUDGET_USD", 2.0),
            price_in: number("CORRODE_OPENAI_PRICE_IN", 0.0),
            price_out: number("CORRODE_OPENAI_PRICE_OUT", 0.0),
        })
    }

    /// Whether `role` runs here first.
    pub fn routes(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }

    /// What `usage` cost. Cached input is charged at the full input price: an upper
    /// bound, so the cap errs toward stopping early.
    pub fn cost(&self, usage: Usage) -> f64 {
        (usage.input_tokens as f64 * self.price_in + usage.output_tokens as f64 * self.price_out)
            / 1e6
    }

    /// Whether a turn that has spent `spent` may send more work here.
    pub fn affordable(&self, spent: f64) -> bool {
        spent < self.budget_usd
    }

    /// One startup line, saying plainly when the cap cannot bind.
    pub fn describe(&self) -> String {
        let roles: Vec<&str> = self.roles.iter().map(|r| r.as_str()).collect();
        let priced = if self.price_in > 0.0 || self.price_out > 0.0 {
            format!(
                "${:.2}/turn at ${}/${} per 1M tokens in/out",
                self.budget_usd, self.price_in, self.price_out
            )
        } else {
            "unpriced -- set CORRODE_OPENAI_PRICE_IN/OUT for the spend cap to bind".into()
        };
        format!(
            "remote: {} at {} first for [{}], escalation for the rest; {priced}",
            self.model,
            self.client.base_url(),
            roles.join(", ")
        )
    }

    #[cfg(test)]
    pub fn for_test(
        client: Client,
        model: &str,
        roles: &[Role],
        budget_usd: f64,
        price: f64,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            roles: roles.to_vec(),
            budget_usd,
            price_in: price,
            price_out: price,
        }
    }
}

/// `research, coder` -> the roles named; unknown names are dropped here and refused
/// at startup by `knobs::check`.
pub fn parse_roles(v: &str) -> Vec<Role> {
    v.split(',')
        .filter_map(|r| Role::from_str(r.trim()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spend_is_priced_per_million_and_capped() {
        let r = Remote::for_test(
            Client::openai_compatible("http://x/v1", None, 1),
            "m",
            &[Role::Review],
            2.0,
            10.0,
        );
        let u = Usage {
            requests: 1,
            input_tokens: 150_000,
            output_tokens: 50_000,
            cached_tokens: 0,
        };
        assert!((r.cost(u) - 2.0).abs() < 1e-9, "{}", r.cost(u));
        assert!(r.affordable(1.99) && !r.affordable(2.0));
        assert!(r.routes(Role::Review) && !r.routes(Role::Coder));
        assert_eq!(
            parse_roles(" review,coder , nonsense"),
            vec![Role::Review, Role::Coder]
        );
    }
}
