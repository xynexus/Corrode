//! 7e: does graph structure separate near-identical siblings?
//!
//! §9 calls this the actual risk of step 7, and it is the last part of it nobody has
//! measured. The known failure is stitch's queue family — four headers differing by
//! three letters (`queue_spsc_waitfree.h` vs `queue_mpmc_lockfree.h`) — where embedding
//! `filename: \brief` scored 1/4 and hand-written alias expansion scored 4/4.
//!
//! Alias expansion works and does not scale: someone has to write down that `spsc` means
//! single-producer single-consumer, for every acronym in every codebase Corrode absorbs.
//! The question 7e asks is whether what the graph ALREADY holds — verbatim node text and
//! the comments bound to those nodes by the extraction pass — separates them for free.
//!
//! So three representations of the same four files are embedded and ranked against the
//! same queries, and the queries deliberately avoid the acronym: a query containing
//! "spsc" makes every representation win and measures nothing.

#![cfg(test)]

use crate::hipfire::Client;
use crate::projection::{self, ingest};

const EMBED_MODEL: &str = "EmbeddingGemma--300M.oq4.25++";
/// The summariser. Deliberately a small served model: summarising every node in a large
/// tree with a 35B is a different proposition from doing it with a 9B, so if the remedy
/// only works at 35B it is not much of a remedy.
const SUMMARY_MODEL: &str = "Qwen3.5-9B--oq4.25++";

/// Queries that identify exactly one sibling, phrased the way someone who does not
/// already know the filename would phrase them.
const QUERIES: &[(&str, &str)] = &[
    ("a queue for one producer thread and one consumer thread", "queue_spsc_waitfree.h"),
    ("a queue many threads push to and one thread drains", "queue_mpsc_waitfree.h"),
    (
        "a queue many threads push to and many drain, where a stalled thread cannot block the others but an individual operation may retry",
        "queue_mpmc_lockfree.h",
    ),
    (
        "a queue many threads push to and many drain, where every operation finishes in a bounded number of steps",
        "queue_mpmc_waitfree.h",
    ),
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Acronym table, written by hand — this is exactly the cost the structure
/// representation is trying to avoid paying.
fn alias_expand(name: &str) -> String {
    let mut s = name.to_string();
    for (short, long) in [
        ("spsc", "single producer single consumer"),
        ("mpsc", "multiple producer single consumer"),
        ("mpmc", "multiple producer multiple consumer"),
        ("waitfree", "wait free bounded steps per operation"),
        ("lockfree", "lock free system wide progress with retries"),
    ] {
        s = s.replace(short, long);
    }
    s
}

#[tokio::test]
#[ignore = "needs a served embedding model and ~/stitch"]
async fn structure_versus_description_on_near_identical_siblings() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var("SIBLING_REPO").unwrap_or_else(|_| format!("{}/stitch", std::env::var("HOME").unwrap())),
    );
    let client = Client::new(
        std::env::var("HIPFIRE_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11435".into()),
        std::env::var("HIPFIRE_API_KEY").ok(),
    );

    let names: Vec<&str> = QUERIES.iter().map(|(_, f)| *f).collect();
    let (mut description, mut aliased, mut structure) = (Vec::new(), Vec::new(), Vec::new());
    // Controls. `filename` alone tests whether all the prose is signal or dilution;
    // `code` tests whether the identifiers discriminate where the comments do not.
    let (mut filename_only, mut code_only) = (Vec::new(), Vec::new());
    // The remedy the negative result implies: if no amount of the file's own bytes
    // separates these, have a model say what the file IS. That is generation at ingest
    // time, not graph structure — and it is the difference between "structure does not
    // work" and knowing what does.
    let mut summarised: Vec<String> = Vec::new();
    // Commit messages bound to this file's changed nodes — the richest human-written
    // source of "why", per `bench_history::bind_commit_messages_to_changed_nodes`.
    // Whether rich in general means discriminating HERE is the question.
    let mut commit_notes: Vec<String> = Vec::new();
    // The note-generation pass proper. 7e's summaries were written in ISOLATION and
    // failed on exactly one axis — lockfree vs waitfree — so the hypothesis is that a
    // note written knowing its siblings separates where an isolated one cannot. Same
    // model, same file, one extra sentence of context: the cheapest thing that could
    // fix the observed failure.
    let mut contrastive: Vec<String> = Vec::new();
    // Identifier -> gloss. Mechanical, no model and no hand-written acronym table:
    // take the type the file DECLARES, then find where the repository's own prose
    // explains that identifier. stitch's `doc/pages/main.md` spells out every variant
    // ("Wait-free multi-producer-single-consumer bounded-size queue"), keyed by class
    // name rather than by filename — which is why nothing keyed on paths ever found it.
    let mut glossed: Vec<String> = Vec::new();
    let mut briefs_found = 0;

    for name in &names {
        let path = root.join("stitch").join(name);
        let src = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let rel = format!("stitch/{name}");
        let lang = projection::for_path(&rel);
        let fw = ingest::file(lang.as_ref(), &rel, &src)?;

        // (a) filename + the file's \brief. Counted, not assumed: a previous run of this
        // experiment reported a dramatic false negative because only 2 of 4 files had a
        // \brief and one of those belonged to a method.
        let brief = fw
            .comments
            .iter()
            .find(|c| c.text.contains("\\brief"))
            .map(|c| c.text.clone())
            .unwrap_or_default();
        if !brief.is_empty() {
            briefs_found += 1;
        }
        description.push(format!("{name}: {brief}"));

        // (b) the same thing with acronyms expanded by hand.
        aliased.push(format!("{}: {}", alias_expand(name), brief));

        // (c) what the graph holds: the comments bound to this file's nodes. No alias
        // table, no filename cleverness — the extraction pass's own output.
        let comments: Vec<&str> = fw.comments.iter().map(|c| c.text.as_str()).collect();
        structure.push(format!("{name}\n{}", comments.join("\n")));

        filename_only.push(name.to_string());

        // Declared types, in declaration order.
        let idents: Vec<String> = src
            .lines()
            .filter_map(|l| {
                let t = l.trim_start();
                let rest = t.strip_prefix("class ").or_else(|| t.strip_prefix("struct "))?;
                let id: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                (id.len() > 2).then_some(id)
            })
            .collect();
        // Every prose line in the repo that mentions one of them.
        let mut gloss: Vec<String> = Vec::new();
        for id in &idents {
            let out = std::process::Command::new("grep")
                .arg("-rhI").arg("--include=*.md").arg("--include=*.rst").arg("--include=*.txt")
                .arg(id.as_str()).arg(&root)
                .output()?;
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let l = line.trim();
                // A mention is not a gloss: keep lines that say something beyond the
                // name itself, and cap so one verbose page cannot dominate.
                if l.len() > id.len() + 20 && gloss.len() < 6 && !gloss.iter().any(|g| g == l) {
                    gloss.push(l.to_string());
                }
            }
        }
        eprintln!("  gloss[{name}] idents={idents:?} lines={}", gloss.len());
        glossed.push(format!("{name} {}\n{}", idents.join(" "), gloss.join("\n")));

        let log = std::process::Command::new("git")
            .arg("-C").arg(&root)
            .args(["log", "--format=%s%n%b", "--", &format!("stitch/{name}")])
            .output()?;
        commit_notes.push(format!("{name}\n{}", String::from_utf8_lossy(&log.stdout).trim()));
        // Verbatim code the graph holds, minus trivia. Caps at 4 KB: the embedder has a
        // window, and a truncated tail is a different experiment from a diluted one.
        let code: String = fw.code.iter().filter(|c| c.kind != "trivia").map(|c| c.text.as_str()).collect();
        let code = code.chars().take(4000).collect::<String>();
        code_only.push(code.clone());

        let prompt = format!(
            "Here is a C++ header file named {name}:\n\n{code}\n\n\
             In two sentences, state what this data structure provides and the exact \
             concurrency conditions it is for. Expand any acronym in the filename. \
             Answer with the description only.",
        );
        // Fail loudly. An `unwrap_or_default()` here silently embedded four empty
        // strings and reported them as a result.
        let summary = match client
            .respond(SUMMARY_MODEL, &prompt, corrode_core::Priority::Default, None)
            .await
        {
            Ok(t) => t,
            Err(e) => anyhow::bail!("summarise {name}: {e}"),
        };
        eprintln!("  summary[{name}] {}", summary.trim().chars().take(140).collect::<String>());
        summarised.push(format!("{name}: {}", summary.trim()));

        let siblings: Vec<&str> = names.iter().filter(|n| **n != *name).copied().collect();
        let cprompt = format!(
            "Here is a C++ header file named {name}:\n\n{code}\n\n\
             This file sits alongside these near-identical siblings: {}.\n\
             In two sentences, state what distinguishes THIS file from those specific \
             siblings — the exact concurrency conditions it is for and its progress \
             guarantee. Expand every acronym. Do not mention the siblings by filename. \
             Answer with the description only.",
            siblings.join(", ")
        );
        let cs = match client
            .respond(SUMMARY_MODEL, &cprompt, corrode_core::Priority::Default, None)
            .await
        {
            Ok(t) => t,
            Err(e) => anyhow::bail!("contrastive summarise {name}: {e}"),
        };
        eprintln!("  contrastive[{name}] {}", cs.trim().chars().take(150).collect::<String>());
        contrastive.push(format!("{name}: {}", cs.trim()));
    }

    eprintln!("\n{}/{} files have a \\brief", briefs_found, names.len());

    // Embed every query ONCE. Re-embedding per representation is both wasted work and
    // enough request volume to trip hipfire's rate limit mid-run.
    let query_texts: Vec<String> = QUERIES.iter().map(|(q, _)| q.to_string()).collect();
    let qvecs = client.embed_batch(EMBED_MODEL, &query_texts, true).await?;

    for (label, docs) in [
        ("description", &description),
        ("alias-expanded", &aliased),
        ("graph structure", &structure),
        ("filename only", &filename_only),
        ("code only", &code_only),
        ("model summary", &summarised),
        ("commit notes", &commit_notes),
        ("contrastive note", &contrastive),
        ("identifier gloss", &glossed),
    ] {
        let vecs = client.embed_batch(EMBED_MODEL, docs, false).await?;
        let mut correct = 0;
        let mut spreads = Vec::new();
        let mut rows = Vec::new();
        for ((q_i, (_, want)), qv) in QUERIES.iter().enumerate().zip(&qvecs) {
            let _ = q_i;
            let mut scored: Vec<(f32, &str)> =
                vecs.iter().zip(&names).map(|(v, n)| (cosine(qv, v), *n)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let hit = scored[0].1 == *want;
            if hit {
                correct += 1;
            }
            // Margin between winner and runner-up: 4/4 by a hair is a different result
            // from 4/4 decisively, and only one of them survives a bigger family.
            let margin = scored[0].0 - scored[1].0;
            spreads.push(margin);
            rows.push((hit, *want, scored[0].1, margin));
        }
        let mean: f32 = spreads.iter().sum::<f32>() / spreads.len() as f32;
        eprintln!(
            "  {label:<16} {correct}/{} top-1, mean winner margin {mean:+.4}, bytes/doc {}",
            QUERIES.len(),
            docs.iter().map(|d| d.len()).sum::<usize>() / docs.len()
        );
        for (hit, want, got, margin) in rows {
            let mark = if hit { "ok  " } else { "MISS" };
            eprintln!("      {mark} want {want:<26} got {got:<26} margin {margin:+.4}");
        }
    }
    // ---- Does DECOMPOSED matching beat a single blended vector? ----
    //
    // Nine representations plateaued at 2/4, and which file failed was predicted by
    // attribute uniqueness: the two with no unique property were never found, because
    // identifying them needs two axes matched at once and a single embedding is a
    // blended bag of features.
    //
    // A cross-encoder would test that directly, and none is served — hipfire's
    // `/v1/rerank` is `rank_by_cosine` over the SAME bi-encoder, so calling it would
    // reproduce the numbers above by construction rather than test anything. This tests
    // the hypothesis with what exists: score each axis of the query separately against
    // the same documents, and rank by the WEAKEST axis, which is "must match all of
    // them" rather than "matches something strongly".
    const AXES: &[(&str, &[&str])] = &[
        ("queue_spsc_waitfree.h", &["exactly one producer thread", "exactly one consumer thread"]),
        ("queue_mpsc_waitfree.h", &["many producer threads", "exactly one consumer thread"]),
        ("queue_mpmc_lockfree.h", &["many producer threads", "many consumer threads", "lock-free progress, an operation may retry"]),
        ("queue_mpmc_waitfree.h", &["many producer threads", "many consumer threads", "wait-free progress, bounded steps per operation"]),
    ];

    for (label, docs) in [("identifier gloss", &glossed), ("model summary", &summarised)] {
        let vecs = client.embed_batch(EMBED_MODEL, docs, false).await?;
        let (mut correct, mut borda) = (0, 0);
        for (want, axes) in AXES {
            let axis_texts: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
            let avecs = client.embed_batch(EMBED_MODEL, &axis_texts, true).await?;
            // Per document: the weakest axis score. A document that nails one axis and
            // misses another scores low, which is exactly what the blended vector cannot
            // express.
            let mut scored: Vec<(f32, &str)> = vecs
                .iter()
                .zip(&names)
                .map(|(v, n)| {
                    let worst = avecs
                        .iter()
                        .map(|a| cosine(a, v))
                        .fold(f32::INFINITY, f32::min);
                    (worst, *n)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let hit = scored[0].1 == *want;
            if hit {
                correct += 1;
            }
            eprintln!(
                "      min  {} want {want:<26} got {:<26}",
                if hit { "ok  " } else { "MISS" },
                scored[0].1,
            );

            // Rank-combine (Borda) instead of `min`. Axes are not on a common scale —
            // "many producer threads" and "wait-free progress" have different absolute
            // similarities — so comparing raw scores across axes penalises whichever
            // axis happens to sit lower. Ranking within each axis first removes the
            // scale, then summing ranks asks "is this document near the top for EVERY
            // axis" without requiring the numbers to be comparable.
            let mut points: Vec<(usize, &str)> = names.iter().map(|n| (0usize, *n)).collect();
            for a in &avecs {
                let mut per: Vec<(f32, usize)> =
                    vecs.iter().enumerate().map(|(i, v)| (cosine(a, v), i)).collect();
                per.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                for (rank, (_, i)) in per.iter().enumerate() {
                    points[*i].0 += rank;
                }
            }
            points.sort_by_key(|(p, _)| *p);
            let borda_hit = points[0].1 == *want;
            if borda_hit {
                borda += 1;
            }
            eprintln!(
                "      rank {} want {want:<26} got {:<26}",
                if borda_hit { "ok  " } else { "MISS" },
                points[0].1,
            );
        }
        eprintln!("  decomposed/{label:<18} min {correct}/{n} top-1, rank-combined {borda}/{n} top-1", n = AXES.len());
    }

    Ok(())
}
