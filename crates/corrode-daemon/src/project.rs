//! Project identity and per-project policy.
//!
//! Corrode serves one repository per daemon (`CORRODE_REPO`). The *project* is that
//! repository's identity plus the policy deciding what may enter its context. Without
//! it two things go wrong, and both were observed: a `~/`-installed skill authored for
//! another codebase is indistinguishable from one belonging to this repo (a C++ library
//! was handed 22 hipfire skills and every subagent concluded it was working on hipfire),
//! and two repositories' provenance collide in one graph store because plan ids are a
//! bare per-process counter.
//!
//! Config lives at `<repo>/.corrode/project.json` — Corrode's own namespace, never a
//! vendor-specific one. Every field is optional; absent file = defaults.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Which global (`~/`) skills a project admits.
///
/// Default is **none**: a project that has said nothing about another codebase's skills
/// has not asked for them. Opting in is explicit — `true` for all, or an allow-list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum GlobalSkills {
    /// `false` -> no global skills; `true` -> every global skill.
    All(bool),
    /// Allow-list of skill names.
    Named(Vec<String>),
}

impl Default for GlobalSkills {
    fn default() -> Self {
        GlobalSkills::All(false)
    }
}

impl GlobalSkills {
    /// Does this project admit the global skill `name`?
    pub fn admits(&self, name: &str) -> bool {
        match self {
            GlobalSkills::All(all) => *all,
            GlobalSkills::Named(names) => names.iter().any(|n| n == name),
        }
    }

    /// Could any global skill be admitted? Lets discovery skip the `~/` dirs entirely.
    pub fn any(&self) -> bool {
        match self {
            GlobalSkills::All(all) => *all,
            GlobalSkills::Named(names) => !names.is_empty(),
        }
    }
}

/// `<repo>/.corrode/project.json`.
#[derive(Debug, Default, Deserialize)]
struct ProjectFile {
    name: Option<String>,
    global_skills: Option<GlobalSkills>,
}

/// The repository this daemon serves.
#[derive(Debug, Clone)]
pub struct Project {
    /// Display + namespacing name. From config, else the root directory's name.
    pub name: String,
    pub root: PathBuf,
    pub global_skills: GlobalSkills,
}

impl Project {
    /// Load `<root>/.corrode/project.json`. A missing or unparseable file yields
    /// defaults (name from the directory, no global skills) rather than an error — an
    /// unconfigured repo must still work.
    pub fn load(root: &Path) -> Self {
        let file: ProjectFile = std::fs::read_to_string(root.join(".corrode/project.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Project {
            name: file
                .name
                .map(|n| n.trim().to_string())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| dir_name(root)),
            root: root.to_path_buf(),
            global_skills: file.global_skills.unwrap_or_default(),
        }
    }

    /// Namespace an id with the project, so repositories sharing one graph store keep
    /// distinct provenance (`stitch/plan-0` vs `corrode/plan-0`).
    pub fn scope(&self, id: &str) -> String {
        format!("{}/{}", self.name, id)
    }
}

/// The root directory's own name, resolved through symlinks/`.` so `CORRODE_REPO=.`
/// still yields a real name.
fn dir_name(root: &Path) -> String {
    let canonical = root.canonicalize();
    let path = canonical.as_deref().unwrap_or(root);
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch dir per test (same convention as `skills.rs` / `vfs.rs` — no
    /// dev-dependency for something this small).
    fn scratch(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("corrode-project-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_config(dir: &Path, body: &str) {
        std::fs::create_dir_all(dir.join(".corrode")).unwrap();
        std::fs::write(dir.join(".corrode/project.json"), body).unwrap();
    }

    #[test]
    fn unconfigured_project_names_itself_and_admits_no_global_skills() {
        let root = scratch("plain").join("stitch");
        std::fs::create_dir_all(&root).unwrap();
        let p = Project::load(&root);
        assert_eq!(p.name, "stitch");
        // The reported bug: no config must mean no foreign skills.
        assert!(!p.global_skills.any());
        assert!(!p.global_skills.admits("hipfire-diag"));
    }

    #[test]
    fn global_skills_opt_in_by_flag_or_allow_list() {
        let root = scratch("optin");
        write_config(&root, r#"{"name":"demo","global_skills":true}"#);
        let all = Project::load(&root);
        assert_eq!(all.name, "demo");
        assert!(all.global_skills.admits("anything"));

        write_config(&root, r#"{"global_skills":["helix-query-rust"]}"#);
        let listed = Project::load(&root);
        assert!(listed.global_skills.admits("helix-query-rust"));
        assert!(!listed.global_skills.admits("hipfire-diag"));
        assert!(listed.global_skills.any());
    }

    #[test]
    fn malformed_config_degrades_to_defaults_and_scope_namespaces_ids() {
        let root = scratch("bad");
        write_config(&root, "{ not json");
        let p = Project::load(&root);
        assert!(!p.global_skills.any());
        assert_eq!(p.scope("plan-0"), format!("{}/plan-0", p.name));
    }
}
