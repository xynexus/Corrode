//! Turning what an agent did into notes bound to the code it touched.
//!
//! Cold-generated documentation measured badly: asked to describe code it had just read,
//! a 9B was wrong on the axis that mattered (a `Waitfree_MPMC_Queue` called "lock-free"),
//! and richer context made it worse. A trace is a different source. The expensive part of
//! agent work is not writing prose, it is **search and verification** — establishing that
//! a function is unused, that a path is never called, that a test fails for a particular
//! reason. Those facts cost real effort to obtain and nothing to record, and a later agent
//! would otherwise pay to rediscover them.
//!
//! # Observed and asserted
//!
//! Notes carry which they are, and the split is **mechanical, not a judgement**. The tool
//! loop already separates the two: text the model emitted is its own claim; the string
//! `ToolBox` handed back is what the system reported. So [`NoteKind::Observed`] is not a
//! quality rating — it means "a tool produced this", and nothing more.
//!
//! # Wrong notes are expected
//!
//! They will arrive however careful the extraction is; the design question is what happens
//! to them afterwards. Three answers here, none of which is prevention:
//!
//! - **Provenance.** Every note says whether a tool produced it or an agent claimed it, so
//!   a reader can weigh them differently instead of a guess being laundered into a fact by
//!   the act of storing it.
//! - **Supersession.** Notes are append-only and ordered. A later note about the same node
//!   supersedes an earlier one rather than editing it, so a correction is visible as a
//!   correction and the wrong version stays auditable.
//! - **Staleness.** A note describes code as it was. `reconcile` already reports exactly
//!   which nodes an edit changed, so a note about a changed node is marked stale rather
//!   than quietly continuing to describe something that no longer exists.

use crate::projection::update::Update;

/// Where a note came from. Not a confidence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteKind {
    /// A tool returned this: a command's output, a file's contents, an error. The agent
    /// did not decide it. Still recorded verbatim rather than summarised, so nothing new
    /// can be invented on the way in.
    Observed,
    /// The agent's own words. May be perfectly true; carries no evidence either way.
    Asserted,
}

impl NoteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NoteKind::Observed => "observed",
            NoteKind::Asserted => "asserted",
        }
    }
}

/// One thing a task learned, bound to the code it concerns.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub kind: NoteKind,
    pub text: String,
    /// The task that produced it — provenance, and the join back to the plan graph.
    pub task: String,
    /// Monotonic within a task, so supersession has an order without needing clocks.
    pub seq: usize,
}

impl Note {
    /// Stable id: task + sequence. Re-running extraction over the same trace addresses
    /// the same notes rather than accumulating duplicates, which is the same property
    /// `ingest::file` gets from deriving ids rather than minting them.
    pub fn id(&self) -> String {
        format!("note:{}#{}", self.task, self.seq)
    }
}

/// One turn of the tool loop, as the loop already has it.
#[derive(Debug, Clone)]
pub struct Step {
    /// What the model wrote this turn.
    pub said: String,
    /// The plain-English intent it produced, if it called a tool.
    pub intent: Option<String>,
    /// The canonical tool name, from the structured call. Decides whether the result is
    /// an OUTCOME or just content — see [`produces_outcome`].
    pub tool: Option<String>,
    /// What the tool returned. `None` on the final turn, which calls nothing.
    pub observation: Option<String>,
}

/// Bytes of a tool result kept in a note. Long enough for an error and its context,
/// short enough that a file dump does not become the note.
const MAX_OBSERVED: usize = 600;

/// Lines that carry a finding rather than narration.
///
/// This is the same filter shape that the commit-message binding measured: 19% of all
/// commits carry a rationale word but 38% of BINDINGS do, because binding to changed
/// nodes concentrates the signal. Here the concentration comes from keeping outcomes and
/// dropping the running commentary around them.
const FINDING: &[&str] = &[
    "error", "failed", "failure", "panic", "cannot", "does not", "doesn't", "not found",
    "no such", "missing", "unused", "never", "unimplemented", "todo", "unwired", "stale",
    "because", "so that", "beware", "gotcha", "note that", "test result", "assert",
];

/// Does this tool produce an outcome, or return content?
///
/// `read_file`, `list_dir` and `search_files` hand back what is already there; running a
/// command or writing a file makes something happen. Only the second kind can yield an
/// observed FINDING — measured on a real session, treating reads as findings meant a
/// source file that merely CONTAINED the word "error" became a 600-byte note about its
/// own contents, which is not something the agent learned.
///
/// Unknown tools count as outcome-producing: a new mutating tool should not silently stop
/// being recorded because this list was not updated.
fn produces_outcome(tool: Option<&str>) -> bool {
    !matches!(tool, Some("read_file") | Some("list_dir") | Some("search_files"))
}

/// Talk about the harness rather than the repository.
///
/// Measured on a real swarm turn: two of four notes were the agent narrating its own
/// tool-call formatting errors ("the previous tool call failed because I tried to use
/// JSON syntax"). Those trip the finding filter honestly — they contain "failed" and
/// "because" — and they are facts about the agent's interaction with the harness, not
/// about the code. A store of code notes that fills with them teaches a later agent
/// nothing about the repository.
const SELF_TALK: &[&str] = &[
    "tool call", "tool-call", "json syntax", "the system expects", "plain english",
    "i need to construct", "my previous", "let me try again", "i tried to use",
];

fn is_self_talk(line: &str) -> bool {
    let l = line.to_lowercase();
    SELF_TALK.iter().any(|w| l.contains(w))
}

fn is_finding(line: &str) -> bool {
    let l = line.to_lowercase();
    FINDING.iter().any(|w| l.contains(w))
}

/// Extract notes from a task's trace.
///
/// Observations are kept **verbatim** (trimmed), never summarised: a summary of a fact is
/// a claim about a fact, and the whole point of the observed/asserted split is that one
/// side of it introduces nothing. What is model-written stays `Asserted` no matter how
/// confident it reads.
pub fn extract(task: &str, steps: &[Step]) -> Vec<Note> {
    let mut out = Vec::new();
    let mut seq = 0usize;
    for step in steps {
        if let Some(obs) = step.observation.as_deref().filter(|_| produces_outcome(step.tool.as_deref())) {
            // Keep the lines that report an outcome. A successful `list_dir` is not a
            // finding; an error, a failing assert or a "not found" is.
            let kept: Vec<&str> = obs
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && is_finding(l))
                .collect();
            if !kept.is_empty() {
                let mut text = kept.join("\n");
                text.truncate(floor_char_boundary(&text, MAX_OBSERVED));
                if let Some(intent) = step.intent.as_deref() {
                    text = format!("{intent}\n{text}");
                }
                out.push(Note { kind: NoteKind::Observed, text, task: task.to_string(), seq });
                seq += 1;
            }
        }
        for line in step.said.lines().map(str::trim) {
            // The model's `TOOL:` line is an instruction, not a finding, and its prose is
            // only worth keeping when it states something.
            if line.is_empty() || line.starts_with("TOOL:") || !is_finding(line) {
                continue;
            }
            // An agent's account of its own tool-call trouble is not repository
            // knowledge, however genuinely it reports a failure.
            if is_self_talk(line) {
                continue;
            }
            out.push(Note {
                kind: NoteKind::Asserted,
                text: line.to_string(),
                task: task.to_string(),
                seq,
            });
            seq += 1;
        }
    }
    out
}

fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Which of a node's existing notes an edit has made stale.
///
/// A note describes code as it was. `reconcile` reports exactly which order keys an edit
/// changed, so this is a lookup rather than a guess — and it is why staleness is
/// mechanical here where the earlier "decay" idea had nothing to hang on.
#[allow(dead_code)] // read side: a reader that surfaces notes has to skip the stale ones
pub fn stale_notes<'a>(update: &Update, notes: &'a [(String, u64)]) -> Vec<&'a str> {
    notes
        .iter()
        .filter(|(_, order)| update.changed.contains(order))
        .map(|(id, _)| id.as_str())
        .collect()
}

/// Edges a task's notes contribute to the graph.
///
/// Append-only by construction: a correction is a NEW note plus a `supersedes` edge, never
/// an edit to the note it corrects. The wrong version stays readable and stays attributed,
/// which is what makes a correction auditable rather than a silent rewrite — and what lets
/// a reader see that a claim was contested at all.
pub fn note_edges(notes: &[Note], task: &str, prior_about_same: &[String]) -> Vec<(String, &'static str, String)> {
    let mut edges = Vec::new();
    for n in notes {
        // Every note is attributed to the task that produced it.
        edges.push((n.id(), "noted_by", task.to_string()));
    }
    // A task that revisits ground an earlier note covered supersedes it. Ordering is the
    // only claim made here — that the later note was written with more of the trace behind
    // it — not that it is correct. An observed note superseding an asserted one is the
    // case worth having: it is how a tool result overrides a guess.
    if let (Some(latest), true) = (notes.last(), !prior_about_same.is_empty()) {
        for old in prior_about_same {
            edges.push((latest.id(), "supersedes", old.clone()));
        }
    }
    edges
}

/// Order notes for a reader: observed before asserted, newest first within each.
///
/// Not a truth ranking. It is the one thing the provenance actually licenses — that a
/// tool result was produced by the system and a claim was produced by an agent — and the
/// cheapest way to stop a confident sentence outranking the command output that
/// contradicts it.
#[allow(dead_code)] // read side: ordering matters when notes are surfaced, not when written
pub fn rank_for_reading(notes: &mut [Note]) {
    notes.sort_by_key(|n| (n.kind != NoteKind::Observed, std::cmp::Reverse(n.seq)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(said: &str, intent: Option<&str>, obs: Option<&str>) -> Step {
        // Fixtures default to an outcome-producing tool; `read_step` covers the other side.
        Step {
            said: said.to_string(),
            intent: intent.map(str::to_string),
            tool: Some("run_command".to_string()),
            observation: obs.map(str::to_string),
        }
    }

    fn read_step(obs: &str) -> Step {
        Step {
            said: String::new(),
            intent: Some("read the file".to_string()),
            tool: Some("read_file".to_string()),
            observation: Some(obs.to_string()),
        }
    }

    #[test]
    fn a_tool_result_is_observed_and_model_prose_is_asserted() {
        let notes = extract(
            "task-1",
            &[step(
                "TOOL: run the tests\nThe parser cannot handle nested groups.",
                Some("run cargo test"),
                Some("running 3 tests\ntest parse::nested ... FAILED\nassertion failed: got None"),
            )],
        );
        let kinds: Vec<NoteKind> = notes.iter().map(|n| n.kind).collect();
        assert_eq!(kinds, vec![NoteKind::Observed, NoteKind::Asserted]);
        // The observation keeps the failing lines and drops the "running 3 tests" noise.
        assert!(notes[0].text.contains("FAILED"));
        assert!(notes[0].text.contains("assertion failed"));
        assert!(!notes[0].text.contains("running 3 tests"));
        // The model's TOOL: line is an instruction, not a finding.
        assert!(!notes[1].text.starts_with("TOOL:"));
        assert!(notes[1].text.contains("cannot handle nested groups"));
    }

    #[test]
    fn a_confident_wrong_claim_is_recorded_as_asserted_not_rejected() {
        // The failure this design expects: an agent stating something false. It is stored,
        // because filtering on plausibility would be exactly the judgement the split
        // exists to avoid — but it is stored as a CLAIM, which is what lets a reader and
        // a later correction weigh it.
        let notes = extract("task-2", &[step("The store does not implement replace_file.", None, None)]);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].kind, NoteKind::Asserted);
    }

    #[test]
    fn reading_a_file_that_contains_trigger_words_is_not_a_finding() {
        // Measured on a real session: a source file containing the word "error" became a
        // note about its own contents. Content is not something the agent learned.
        let notes = extract("task-6", &[read_step("fn f() {\n  // error handling below\n}")]);
        assert!(notes.is_empty(), "a read must not yield an observed finding: {notes:?}");
        // The same text from a command IS an outcome.
        let ran = extract("task-7", &[step("", Some("run it"), Some("error handling below"))]);
        assert_eq!(ran.len(), 1);
        assert_eq!(ran[0].kind, NoteKind::Observed);
    }

    #[test]
    fn the_agent_narrating_its_own_tool_trouble_is_not_a_note() {
        // Both of these are real lines from a swarm turn. They contain "failed" and
        // "because", so the finding filter keeps them — and they say nothing about the
        // code, which is what the note store is for.
        let notes = extract("task-8", &[step(
            "The previous tool call failed because I tried to use JSON syntax inside a plain English tool call.",
            None,
            None,
        )]);
        assert!(notes.is_empty(), "harness self-talk must not become a code note: {notes:?}");

        // A real finding about the code still lands.
        let real = extract("task-8", &[step("is_prime is O(n) with a TODO to optimize", None, None)]);
        assert_eq!(real.len(), 1);
    }

    #[test]
    fn narration_without_a_finding_is_dropped() {
        let notes = extract(
            "task-3",
            &[step("I will look at the parser next.", Some("read parser.rs"), Some("fn parse() {}"))],
        );
        assert!(notes.is_empty(), "neither side stated an outcome: {notes:?}");
    }

    #[test]
    fn ids_are_stable_so_re_extraction_does_not_duplicate() {
        let steps = [step("cannot find the symbol", None, None)];
        let a = extract("task-4", &steps);
        let b = extract("task-4", &steps);
        assert_eq!(a[0].id(), b[0].id());
        assert_eq!(a[0].id(), "note:task-4#0");
    }

    #[test]
    fn an_edit_marks_notes_on_the_changed_nodes_stale() {
        let update = Update { changed: vec![100, 300], ..Default::default() };
        let notes = vec![
            ("note:t#0".to_string(), 100u64),
            ("note:t#1".to_string(), 200u64),
            ("note:t#2".to_string(), 300u64),
        ];
        // Only notes about nodes the edit actually touched go stale — a note about an
        // untouched node still describes the code that is there.
        assert_eq!(stale_notes(&update, &notes), vec!["note:t#0", "note:t#2"]);
    }

    #[test]
    fn a_correction_supersedes_rather_than_edits() {
        let notes = extract("task-9", &[step("the loader is never called", None, None)]);
        let edges = note_edges(&notes, "task-9", &["note:task-1#3".to_string()]);
        assert!(edges.contains(&(notes[0].id(), "noted_by", "task-9".to_string())));
        assert!(
            edges.contains(&(notes[0].id(), "supersedes", "note:task-1#3".to_string())),
            "a later note must supersede the earlier one it revisits: {edges:?}"
        );
        // The superseded note is referenced, never removed — the wrong version stays
        // auditable and stays attributed to whoever wrote it.
        assert!(edges.iter().all(|(_, rel, _)| *rel != "deletes"));
    }

    #[test]
    fn a_tool_result_outranks_a_claim_that_contradicts_it() {
        let mut notes = extract(
            "task-10",
            &[step(
                "The file cannot exist.",
                Some("list the directory"),
                Some("error: no such file"),
            )],
        );
        rank_for_reading(&mut notes);
        assert_eq!(notes[0].kind, NoteKind::Observed, "observed must read before asserted");
    }

    #[test]
    fn a_long_observation_is_truncated_on_a_char_boundary() {
        let obs = format!("error: {}", "é".repeat(2000));
        let notes = extract("task-5", &[step("", None, Some(&obs))]);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].text.len() <= MAX_OBSERVED);
        // Truncating mid-character would panic on the slice; this is the same UTF-8
        // hazard the text fallback hit on a real repository.
        assert!(notes[0].text.chars().count() > 0);
    }
}

#[cfg(test)]
mod real_trace {
    use super::*;

    /// Run the filter over a REAL agent trace, not a fixture.
    ///
    /// The open question when extraction was built was whether the finding filter keeps
    /// anything worth keeping at volume, or whether it is either so loose that every line
    /// becomes a note or so tight that nothing does. A session transcript is the honest
    /// corpus: thousands of turns of actual tool calls and results.
    #[test]
    #[ignore = "probe: needs TRACE_STEPS pointing at a converted transcript"]
    fn filter_yield_on_a_real_session() {
        let path = std::env::var("TRACE_STEPS").expect("set TRACE_STEPS");
        let raw = std::fs::read_to_string(path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let steps: Vec<Step> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|s| Step {
                said: s["said"].as_str().unwrap_or("").to_string(),
                intent: s["intent"].as_str().map(str::to_string),
                tool: s["tool"].as_str().map(str::to_string),
                observation: s["observation"].as_str().filter(|o| !o.is_empty()).map(str::to_string),
            })
            .collect();

        let notes = extract("session", &steps);
        let observed = notes.iter().filter(|n| n.kind == NoteKind::Observed).count();
        let obs_bytes: usize = steps.iter().filter_map(|s| s.observation.as_ref()).map(|o| o.len()).sum();
        let note_bytes: usize = notes.iter().map(|n| n.text.len()).sum();

        eprintln!("\n{} steps ({} with a tool result)", steps.len(),
            steps.iter().filter(|s| s.observation.is_some()).count());
        eprintln!("  notes           {}", notes.len());
        eprintln!("    observed      {observed}");
        eprintln!("    asserted      {}", notes.len() - observed);
        eprintln!("  yield           {:.1}% of steps produced a note",
            100.0 * notes.len() as f64 / steps.len().max(1) as f64);
        eprintln!("  compression     {} KB of tool output -> {} KB of notes ({:.1}%)",
            obs_bytes / 1024, note_bytes / 1024,
            100.0 * note_bytes as f64 / obs_bytes.max(1) as f64);
        for n in notes.iter().filter(|n| n.kind == NoteKind::Observed).take(3) {
            eprintln!("  [observed] {}", n.text.lines().take(2).collect::<Vec<_>>().join(" / "));
        }
        for n in notes.iter().filter(|n| n.kind == NoteKind::Asserted).take(3) {
            eprintln!("  [asserted] {}", n.text.chars().take(110).collect::<String>());
        }
    }
}
