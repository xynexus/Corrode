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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// How exactly a project's source must survive the graph round trip.
///
/// `Verbatim` (the default) keeps every byte, which is what makes projection
/// byte-exact on source nobody has normalised — and what forces a growing tail of
/// corner cases as nodes get more specific.
///
/// `Normalized` is the deliberate trade: run the language's own formatter over the repo
/// once, commit that, and the quirks stop existing rather than being handled. It does
/// NOT make ingest lossy — normalised source round-trips byte-exactly through the same
/// verbatim pipeline. It is a claim about the repo, and `normalize --check` enforces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fidelity {
    #[default]
    Verbatim,
    Normalized,
}

/// Formatters by [`Language::name`](crate::projection::Language::name). Each is argv
/// with a stdin -> stdout contract; `{path}` is replaced with the file's repo-relative
/// path, which is how `clang-format` picks C from C++ and finds the right `.clang-format`.
///
/// Defaults cover the two backends that have a real parser. A language with no entry is
/// left alone rather than guessed at — normalising a file with the wrong tool is worse
/// than not normalising it.
fn default_formatters() -> HashMap<String, Vec<String>> {
    let mut m = HashMap::new();
    m.insert(
        "rust".to_string(),
        ["rustfmt", "--emit", "stdout", "--edition", "2021"].iter().map(|s| s.to_string()).collect(),
    );
    m.insert(
        "c".to_string(),
        ["clang-format", "--assume-filename={path}"].iter().map(|s| s.to_string()).collect(),
    );
    m
}

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
    fidelity: Option<Fidelity>,
    /// Merged over the defaults, so a project overrides one language without
    /// restating the rest. An empty argv removes a default.
    formatters: Option<HashMap<String, Vec<String>>>,
}

/// The repository this daemon serves.
#[derive(Debug, Clone)]
pub struct Project {
    /// Display + namespacing name. From config, else the root directory's name.
    pub name: String,
    pub root: PathBuf,
    pub global_skills: GlobalSkills,
    pub fidelity: Fidelity,
    pub formatters: HashMap<String, Vec<String>>,
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
            fidelity: file.fidelity.unwrap_or_default(),
            formatters: {
                let mut m = default_formatters();
                for (lang, argv) in file.formatters.unwrap_or_default() {
                    if argv.is_empty() {
                        m.remove(&lang);
                    } else {
                        m.insert(lang, argv);
                    }
                }
                m
            },
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
    fn fidelity_defaults_to_verbatim_and_formatters_merge_over_defaults() {
        let root = scratch("fidelity");
        let plain = Project::load(&root);
        // Absent config must not silently opt a repo into being rewritten.
        assert_eq!(plain.fidelity, Fidelity::Verbatim);
        assert!(plain.formatters.contains_key("rust"));

        write_config(
            &root,
            r#"{"fidelity":"normalized","formatters":{"c":["my-fmt","{path}"],"rust":[]}}"#,
        );
        let p = Project::load(&root);
        assert_eq!(p.fidelity, Fidelity::Normalized);
        assert_eq!(p.formatters["c"], vec!["my-fmt", "{path}"]);
        assert!(!p.formatters.contains_key("rust"), "empty argv removes a default");
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
