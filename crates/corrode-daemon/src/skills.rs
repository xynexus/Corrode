//! Agent Skills discovery — Corrode consumes skills authored in the open Agent
//! Skills format (agentskills.io: a directory with a `SKILL.md` whose frontmatter
//! has `name` + `description`), and project rules from `AGENTS.md` (agents.md).
//!
//! Locations, deliberately narrow: the vendor-neutral **standard** `.agents/skills/`
//! plus Corrode's own **`.corrode/skills/`** — both project (under the repo root)
//! and global (`~/`). We do NOT read agent-specific dirs (`.claude`, `.opencode`,
//! `.codex`) or agent-specific rule files (CLAUDE.md, .cursorrules): those can carry
//! non-standard extensions and would drift Corrode off the standard. Corrode-custom
//! shadows standard; project shadows global (first-seen wins).
//!
//! Progressive disclosure: stage 1 (discovery) surfaces name+description manifests on
//! the swarm's shared `context_prefix`; stage 2 (activation) injects the single
//! most-relevant skill's full `SKILL.md` body when it clears the relevance bar (see
//! [`SkillContext::prefix_section`]). Stage 3 (execution — running a skill's `scripts/`
//! through the tool loop) is still ahead.

use crate::hipfire::Client;
use crate::project::GlobalSkills;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Where a skill came from. A project skill belongs to the repository under work; a
/// global one is installed in `~/` and belongs to no project until a project admits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillOrigin {
    Project,
    Global,
}

/// A discovered skill (name + description + its directory). The full `SKILL.md`
/// body is read lazily by [`SkillRegistry::body`] only when a task activates it.
pub struct Skill {
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

#[derive(Default)]
pub struct SkillRegistry {
    skills: Vec<Skill>,
    /// Project `AGENTS.md` (repo root), if present.
    agents_md: Option<PathBuf>,
}

impl SkillRegistry {
    /// Scan the project's skill locations — plus the global (`~/`) ones the project
    /// admits — and the project `AGENTS.md`.
    pub fn discover(repo_root: &Path, global: &GlobalSkills) -> Self {
        let home = std::env::var("HOME").ok().map(PathBuf::from);
        Self::discover_in(repo_root, home.as_deref(), global)
    }

    /// Discovery core with an injectable home dir (so tests stay hermetic — the
    /// real `discover` reads `$HOME`, whose global skill dirs would leak in).
    fn discover_in(repo_root: &Path, home: Option<&Path>, global: &GlobalSkills) -> Self {
        let mut skills = Vec::new();
        let mut seen = HashSet::new();
        for (dir, origin) in search_dirs(repo_root, home, global) {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue; // dir absent -> skip
            };
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(dir.join("SKILL.md")) else {
                    continue; // not a skill dir
                };
                let Some((mut name, description)) = parse_frontmatter(&text) else {
                    continue; // missing required fields
                };
                if name.is_empty() {
                    // spec: name must equal the dir; fall back to it.
                    name = dir
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string();
                }
                if name.is_empty() || !seen.insert(name.clone()) {
                    continue; // first-seen wins (precedence)
                }
                // A global skill belongs to no project until one says so. Project-local
                // skills are admitted unconditionally — they ARE this project.
                if origin == SkillOrigin::Global && !global.admits(&name) {
                    continue;
                }
                skills.push(Skill {
                    name,
                    description,
                    dir,
                });
            }
        }
        let agents_md = {
            let p = repo_root.join("AGENTS.md");
            p.is_file().then_some(p)
        };
        Self { skills, agents_md }
    }

    /// Progressive-disclosure stage 1: a compact `name: description` manifest for the
    /// shared `context_prefix`. Empty when no skills are present.
    pub fn manifest(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut s = String::from(
            "Available skills (load a skill's full instructions by name when a task matches):\n",
        );
        for sk in &self.skills {
            s.push_str(&Self::line(&sk.name, &sk.description));
        }
        s
    }

    /// Project rules from `AGENTS.md` (empty if none). Folded into `context_prefix`
    /// like a README-for-agents.
    pub fn agents_rules(&self) -> String {
        self.agents_md
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default()
    }

    /// Stage 2 (activation): the full `SKILL.md` body for `name`. Injected into the
    /// context prefix by [`SkillContext::prefix_section`] when a task activates the skill.
    pub fn body(&self, name: &str) -> anyhow::Result<String> {
        let sk = self
            .skills
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow::anyhow!("no skill named {name}"))?;
        Ok(std::fs::read_to_string(sk.dir.join("SKILL.md"))?)
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// The `name: description` line for one skill (manifest entry).
    fn line(name: &str, description: &str) -> String {
        format!("  - {name}: {description}\n")
    }
}

/// Embedding-ranked skill selection: each skill's description is embedded once, then
/// cosine-ranked against a query (the prompt) to pick the top-K most relevant. Brute
/// force is the right call at this scale (dozens of skills); HelixStore's HNSW is the
/// endgame, when skill descriptions join the code/doc corpus in one embedding space.
#[derive(Default)]
pub struct SkillIndex {
    entries: Vec<IndexedSkill>,
}

struct IndexedSkill {
    name: String,
    description: String,
    embedding: Vec<f32>,
}

impl SkillIndex {
    /// Embed every discovered skill's description via hipfire. Best-effort: a skill
    /// whose description fails to embed is skipped (so an empty index degrades to the
    /// full manifest rather than dropping skills silently).
    pub async fn build(registry: &SkillRegistry, client: &Client, embed_model: &str) -> Self {
        let mut entries = Vec::new();
        for sk in &registry.skills {
            let text = format!("{}: {}", sk.name, sk.description);
            if let Ok(embedding) = client.embed(embed_model, &text).await {
                entries.push(IndexedSkill {
                    name: sk.name.clone(),
                    description: sk.description.clone(),
                    embedding,
                });
            }
        }
        Self { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Skills ranked by cosine similarity to `query`, most-relevant first (with scores,
    /// so the caller can both list the top-k and decide whether the top skill is
    /// relevant enough to *activate*).
    pub fn rank(&self, query: &[f32]) -> Vec<Ranked<'_>> {
        let mut scored: Vec<Ranked> = self
            .entries
            .iter()
            .map(|e| Ranked {
                score: cosine(query, &e.embedding),
                name: &e.name,
                description: &e.description,
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

/// One skill scored against a query (a borrow into the index).
pub struct Ranked<'a> {
    pub score: f32,
    pub name: &'a str,
    pub description: &'a str,
}

/// Everything the daemon needs for skills: the discovered registry, the embedded
/// index (for ranked selection), and the embedding model. One value on the `Daemon`.
#[derive(Default)]
pub struct SkillContext {
    registry: SkillRegistry,
    index: SkillIndex,
    embed_model: Option<String>,
}

impl SkillContext {
    /// Discover skills + AGENTS.md, then embed the skill descriptions (if an
    /// embedding model is served) for relevance ranking.
    pub async fn build(
        repo_root: &Path,
        client: &Client,
        embed_model: Option<String>,
        global: &GlobalSkills,
    ) -> Self {
        let registry = SkillRegistry::discover(repo_root, global);
        let index = match &embed_model {
            Some(m) => SkillIndex::build(&registry, client, m).await,
            None => SkillIndex::default(),
        };
        Self {
            registry,
            index,
            embed_model,
        }
    }

    pub fn count(&self) -> usize {
        self.registry.len()
    }

    /// Map of skill name -> its directory, so the tool loop can resolve and run a
    /// skill's bundled `scripts/` (progressive-disclosure stage 3, execution).
    pub fn script_dirs(&self) -> HashMap<String, PathBuf> {
        self.registry
            .skills
            .iter()
            .map(|s| (s.name.clone(), s.dir.clone()))
            .collect()
    }

    pub fn ranked(&self) -> bool {
        !self.index.is_empty()
    }

    pub fn agents_rules(&self) -> String {
        self.registry.agents_rules()
    }

    /// The skills section for the shared `context_prefix`. With retrieval available:
    /// the relevance-ranked top-`k` descriptions (discovery) **plus** the full
    /// instructions of the single most-relevant skill when it clears the activation bar
    /// (progressive-disclosure activation). Without retrieval: the full manifest, so
    /// skills are still surfaced.
    pub async fn prefix_section(&self, task: &str, client: &Client, k: usize) -> String {
        if !self.index.is_empty() {
            if let Some(model) = &self.embed_model {
                if let Ok(q) = client.embed(model, task).await {
                    return self.render(&q, k);
                }
            }
        }
        self.registry.manifest()
    }

    /// Render the skills section for a task's query embedding: top-`k` ranked
    /// descriptions, then the top skill's `SKILL.md` body if it's relevant enough to
    /// activate. Split from `prefix_section` (which does the embedding) so it's a pure,
    /// testable function of the index + registry.
    fn render(&self, query: &[f32], k: usize) -> String {
        let ranked = self.index.rank(query);
        let mut s = String::from(
            "Relevant skills for this task (load a skill's full instructions by name):\n",
        );
        for r in ranked.iter().take(k) {
            s.push_str(&SkillRegistry::line(r.name, r.description));
        }
        // Activation: inject the single most-relevant skill's instructions, but only
        // when it clears the bar — otherwise every prompt would carry the least-
        // irrelevant skill's whole body. The body rides the shared prefix, so all the
        // turn's subagents get the same activated skill (KV-reuse preserved).
        if let Some(top) = ranked.first() {
            if top.score >= activate_min() {
                if let Ok(body) = self.registry.body(top.name) {
                    s.push_str(&activated_section(top.name, &body));
                }
            }
        }
        s
    }
}

/// Cosine bar the top skill must clear to be *activated* (full body injected), not just
/// listed. Conservative default; override with `CORRODE_SKILL_ACTIVATE_MIN`.
const DEFAULT_ACTIVATE_MIN: f32 = 0.35;
/// Cap on an injected `SKILL.md` body, so a large skill can't blow the shared prefix.
const MAX_BODY_BYTES: usize = 8192;

fn activate_min() -> f32 {
    std::env::var("CORRODE_SKILL_ACTIVATE_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ACTIVATE_MIN)
}

/// The activated-skill block: the skill's `SKILL.md` instructions, truncated at a char
/// boundary to `MAX_BODY_BYTES`.
fn activated_section(name: &str, body: &str) -> String {
    let mut end = body.len().min(MAX_BODY_BYTES);
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    let shown = &body[..end];
    let ellipsis = if end < body.len() { "\n… (truncated)" } else { "" };
    format!("\n--- Activated skill: {name} (follow these instructions) ---\n{shown}{ellipsis}\n")
}

/// Cosine similarity; 0 for mismatched/degenerate vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// The skill search dirs in precedence order (first-seen wins): project before
/// global, Corrode-custom before standard.
fn search_dirs(
    repo_root: &Path,
    home: Option<&Path>,
    global: &GlobalSkills,
) -> Vec<(PathBuf, SkillOrigin)> {
    let mut dirs = vec![
        (repo_root.join(".corrode/skills"), SkillOrigin::Project),
        (repo_root.join(".agents/skills"), SkillOrigin::Project),
    ];
    // Skip `~/` entirely when the project admits nothing — the common case, and it
    // keeps an unconfigured repo from even reading another codebase's skills.
    if let (Some(home), true) = (home, global.any()) {
        dirs.push((home.join(".corrode/skills"), SkillOrigin::Global));
        dirs.push((home.join(".agents/skills"), SkillOrigin::Global));
    }
    dirs
}

/// Minimal `SKILL.md` frontmatter parse: pull single-line `name` + `description`
/// from the leading `--- ... ---` block. Missing `description` -> None.
// ponytail: line-based, so YAML block scalars / multi-line descriptions aren't
// handled; swap for a real YAML parse if skills in the wild need it.
fn parse_frontmatter(md: &str) -> Option<(String, String)> {
    let rest = md.strip_prefix("---")?;
    let end = rest.find("\n---")?;
    let mut name = String::new();
    let mut description = None;
    for line in rest[..end].lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = unquote(v);
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(unquote(v));
        }
    }
    description.map(|d| (name, d))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if s.len() >= 2
        && ((bytes[0] == b'"' && bytes[s.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[s.len() - 1] == b'\''))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported bug: a project with no skills of its own was handed every skill
    /// installed in `~/` — a C++ library got 22 hipfire skills and its subagents
    /// concluded they were working on hipfire. A populated home is the whole point
    /// here, so this cannot reuse the empty-home fixture the other tests rely on.
    #[test]
    fn global_skills_enter_only_when_the_project_admits_them() {
        let root = std::env::temp_dir().join(format!("corrode-skills-global-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let repo = root.join("repo");
        let home = root.join("home");

        let write = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        write(
            repo.join(".agents/skills/repo-audit/SKILL.md"),
            "---\nname: repo-audit\ndescription: Audit this repo.\n---\nbody",
        );
        write(
            home.join(".agents/skills/hipfire-diag/SKILL.md"),
            "---\nname: hipfire-diag\ndescription: Diagnose hipfire GPUs.\n---\nbody",
        );
        write(
            home.join(".agents/skills/helix-query-rust/SKILL.md"),
            "---\nname: helix-query-rust\ndescription: HelixDB Rust queries.\n---\nbody",
        );

        let names = |reg: &SkillRegistry| {
            let mut v: Vec<String> = reg.skills.iter().map(|s| s.name.clone()).collect();
            v.sort();
            v
        };

        // Default: the project said nothing, so it gets only its own.
        let reg = SkillRegistry::discover_in(&repo, Some(&home), &GlobalSkills::default());
        assert_eq!(names(&reg), vec!["repo-audit"]);

        // Opt in to everything.
        let reg = SkillRegistry::discover_in(&repo, Some(&home), &GlobalSkills::All(true));
        assert_eq!(
            names(&reg),
            vec!["helix-query-rust", "hipfire-diag", "repo-audit"]
        );

        // Opt in to a named subset: the allow-list admits one, not the other.
        let reg = SkillRegistry::discover_in(
            &repo,
            Some(&home),
            &GlobalSkills::Named(vec!["helix-query-rust".to_string()]),
        );
        assert_eq!(names(&reg), vec!["helix-query-rust", "repo-audit"]);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn discovers_standard_and_corrode_skills_and_agents_md() {
        let root = std::env::temp_dir().join(format!("corrode-skills-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();

        let write = |rel: &str, body: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        // standard skill
        write(
            ".agents/skills/pdf-processing/SKILL.md",
            "---\nname: pdf-processing\ndescription: \"Extract PDF text. Use for PDFs.\"\n---\nbody",
        );
        // corrode-custom skill that shadows a standard one by name
        write(
            ".corrode/skills/pdf-processing/SKILL.md",
            "---\nname: pdf-processing\ndescription: Corrode override.\n---\ncustom body",
        );
        // a corrode-only skill
        write(
            ".corrode/skills/repo-audit/SKILL.md",
            "---\nname: repo-audit\ndescription: Audit the repo.\n---\naudit body",
        );
        write("AGENTS.md", "# Rules\nRun tests before committing.\n");

        // Isolated (empty) home so the real ~/.agents,~/.corrode skills don't leak in.
        let empty_home = root.join("home");
        let reg = SkillRegistry::discover_in(&root, Some(&empty_home), &GlobalSkills::default());
        assert_eq!(reg.len(), 2, "pdf-processing deduped, repo-audit kept");

        let manifest = reg.manifest();
        assert!(manifest.contains("pdf-processing:"));
        assert!(manifest.contains("repo-audit:"));

        // corrode-custom shadows standard (first-seen wins)
        assert!(reg.body("pdf-processing").unwrap().contains("custom body"));
        assert!(reg.agents_rules().contains("Run tests before committing"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rank_orders_by_cosine_similarity() {
        let mk = |name: &str, emb: Vec<f32>| IndexedSkill {
            name: name.to_string(),
            description: format!("desc of {name}"),
            embedding: emb,
        };
        let index = SkillIndex {
            entries: vec![
                mk("db-skill", vec![1.0, 0.0, 0.0]),
                mk("ui-skill", vec![0.0, 1.0, 0.0]),
                mk("test-skill", vec![0.0, 0.0, 1.0]),
            ],
        };
        // query closest to db-skill, then ui-skill, with test-skill last.
        let ranked = index.rank(&[0.9, 0.4, 0.0]);
        let order: Vec<&str> = ranked.iter().map(|r| r.name).collect();
        assert_eq!(order, vec!["db-skill", "ui-skill", "test-skill"]);
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn activation_injects_top_skill_body_only_when_relevant() {
        let root =
            std::env::temp_dir().join(format!("corrode-skill-activate-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let p = root.join(".agents/skills/db-skill/SKILL.md");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            "---\nname: db-skill\ndescription: Database helper.\n---\nDB INSTRUCTIONS: use the pool.",
        )
        .unwrap();

        let empty_home = root.join("home");
        let registry = SkillRegistry::discover_in(&root, Some(&empty_home), &GlobalSkills::default());
        let index = SkillIndex {
            entries: vec![
                IndexedSkill {
                    name: "db-skill".into(),
                    description: "Database helper.".into(),
                    embedding: vec![1.0, 0.0, 0.0],
                },
                IndexedSkill {
                    name: "ui-skill".into(),
                    description: "UI helper.".into(),
                    embedding: vec![0.0, 1.0, 0.0],
                },
            ],
        };
        let ctx = SkillContext {
            registry,
            index,
            embed_model: Some("m".into()),
        };

        // Query aligned with db-skill: listed AND activated (full body injected).
        let out = ctx.render(&[1.0, 0.05, 0.0], 5);
        assert!(out.contains("db-skill:"), "listed in manifest");
        assert!(out.contains("Activated skill: db-skill"), "activated");
        assert!(out.contains("DB INSTRUCTIONS"), "body injected");

        // Query orthogonal to every skill: nothing clears the bar -> no body injected.
        let out2 = ctx.render(&[0.0, 0.0, 1.0], 5);
        assert!(
            !out2.contains("DB INSTRUCTIONS"),
            "below the activation bar -> description only"
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
