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
//! This is progressive-disclosure stage 1 (discovery: name+description). The manifest
//! rides the swarm's shared `context_prefix`; `body()` loads a full SKILL.md on
//! activation; execution runs the skill's `scripts/` through the pty/VFS.

use crate::hipfire::Client;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

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
    /// Scan the standard + Corrode skill locations and the project `AGENTS.md`.
    pub fn discover(repo_root: &Path) -> Self {
        let home = std::env::var("HOME").ok().map(PathBuf::from);
        Self::discover_in(repo_root, home.as_deref())
    }

    /// Discovery core with an injectable home dir (so tests stay hermetic — the
    /// real `discover` reads `$HOME`, whose global skill dirs would leak in).
    fn discover_in(repo_root: &Path, home: Option<&Path>) -> Self {
        let mut skills = Vec::new();
        let mut seen = HashSet::new();
        for dir in search_dirs(repo_root, home) {
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

    /// Stage 2 (activation): the full `SKILL.md` body for `name`.
    // ponytail: no loop caller yet — wired when the planner routes a task to a skill.
    #[allow(dead_code)]
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

    /// A manifest of the top-`k` skills by cosine similarity to `query` — the same
    /// shape as `SkillRegistry::manifest`, but relevance-ranked and trimmed.
    pub fn top_k_manifest(&self, query: &[f32], k: usize) -> String {
        let mut scored: Vec<(f32, &IndexedSkill)> = self
            .entries
            .iter()
            .map(|e| (cosine(query, &e.embedding), e))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut s = String::from(
            "Relevant skills for this task (load a skill's full instructions by name):\n",
        );
        for (_, e) in scored.into_iter().take(k) {
            s.push_str(&SkillRegistry::line(&e.name, &e.description));
        }
        s
    }
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
    pub async fn build(repo_root: &Path, client: &Client, embed_model: Option<String>) -> Self {
        let registry = SkillRegistry::discover(repo_root);
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

    pub fn ranked(&self) -> bool {
        !self.index.is_empty()
    }

    pub fn agents_rules(&self) -> String {
        self.registry.agents_rules()
    }

    /// The skills section for the shared `context_prefix`: the relevance-ranked
    /// top-`k` when the task can be embedded, else the full manifest (so skills are
    /// still surfaced when retrieval is unavailable).
    pub async fn prefix_section(&self, task: &str, client: &Client, k: usize) -> String {
        if !self.index.is_empty() {
            if let Some(model) = &self.embed_model {
                if let Ok(q) = client.embed(model, task).await {
                    return self.index.top_k_manifest(&q, k);
                }
            }
        }
        self.registry.manifest()
    }
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
fn search_dirs(repo_root: &Path, home: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = vec![
        repo_root.join(".corrode/skills"),
        repo_root.join(".agents/skills"),
    ];
    if let Some(home) = home {
        dirs.push(home.join(".corrode/skills"));
        dirs.push(home.join(".agents/skills"));
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
        let reg = SkillRegistry::discover_in(&root, Some(&empty_home));
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
    fn top_k_manifest_ranks_by_cosine_similarity() {
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
        // query closest to db-skill, then ui-skill
        let out = index.top_k_manifest(&[0.9, 0.4, 0.0], 2);
        assert!(out.contains("db-skill"));
        assert!(out.contains("ui-skill"));
        assert!(!out.contains("test-skill"), "trimmed to top-2");
        // db-skill should rank above ui-skill (appears first)
        assert!(out.find("db-skill").unwrap() < out.find("ui-skill").unwrap());
    }
}
