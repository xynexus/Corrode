//! Reconciling a re-scanned file against the nodes already stored for it.
//!
//! [`ingest::file`](super::ingest::file) assigns every node a fresh key from
//! [`initial_order`], which is right for a first ingest and wrong for every one after:
//! re-running it on an edited file renumbers every node and — since ids derive from the
//! key — re-addresses the whole file for a one-line change. That is exactly the churn
//! the sparse key exists to avoid, so the key only pays off once something reconciles
//! against what is already stored. This is that something.
//!
//! The reconciliation is a sequence diff over node fingerprints: nodes that survive keep
//! their key, nodes that changed in place keep their key too (same slot, new text), and
//! only genuinely new nodes need a key minted between their neighbours.

use super::{initial_order, order_between, rebalance, Node, ORDER_STRIDE};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// What one re-ingest did to a file's stored nodes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Update {
    /// Byte-identical: key and text both untouched.
    pub kept: usize,
    /// Same slot, new text — the key survives, so ids and provenance edges survive.
    pub updated: usize,
    /// New nodes, each needing a key minted between its neighbours.
    pub inserted: usize,
    pub deleted: usize,
    /// A gap ran out and the file was renumbered. Correct, but it re-addresses every
    /// node in the file, so its frequency is the number that decides whether a sparse
    /// key was worth having.
    pub rebalanced: bool,
}

impl Update {
    pub fn touched(&self) -> usize {
        self.updated + self.inserted + self.deleted
    }
}

fn fingerprint(n: &Node) -> u64 {
    let mut h = DefaultHasher::new();
    n.kind.hash(&mut h);
    n.text.hash(&mut h);
    h.finish()
}

/// Fingerprints match AND the nodes really are equal — a 64-bit collision would
/// otherwise silently splice one node's text onto another's identity.
fn same(a: &Node, fa: u64, b: &Node, fb: u64) -> bool {
    fa == fb && a.kind == b.kind && a.text == b.text
}

/// Which stored node, if any, a slot in the new sequence inherits its key from.
enum Slot {
    /// Reuses `stored[i]`'s key; carries the fresh text (identical when `kept`).
    Keep(usize, usize),
    /// Genuinely new: index into `fresh`.
    New(usize),
}

/// ponytail: O(n*m) LCS after prefix/suffix trimming. A commit usually touches a few
/// nodes in a large file, so the trimmed middle is tiny; the guard below covers the
/// rewrite-the-world case rather than making the common one clever.
const LCS_CELL_BUDGET: usize = 4_000_000;

/// Reconcile `fresh` (freshly scanned, keys meaningless) against `stored` (sorted by
/// key), returning the file's new node list and what changed.
pub fn reconcile(stored: &[Node], fresh: &[Node]) -> (Vec<Node>, Update) {
    let mut st = Update::default();

    // A file that is new, or emptied, has nothing to reconcile against.
    if stored.is_empty() {
        st.inserted = fresh.len();
        let nodes: Vec<Node> = fresh
            .iter()
            .enumerate()
            .map(|(i, n)| Node { order: initial_order(i), ..n.clone() })
            .collect();
        return (nodes, st);
    }

    let fs: Vec<u64> = stored.iter().map(fingerprint).collect();
    let ff: Vec<u64> = fresh.iter().map(fingerprint).collect();

    // Trim the common head and tail first. This is what keeps the diff cheap: an edit
    // in the middle of a 2,000-node file leaves a handful of nodes to actually align.
    let mut p = 0;
    while p < stored.len() && p < fresh.len() && same(&stored[p], fs[p], &fresh[p], ff[p]) {
        p += 1;
    }
    let mut s = 0;
    while s < stored.len() - p
        && s < fresh.len() - p
        && same(
            &stored[stored.len() - 1 - s],
            fs[stored.len() - 1 - s],
            &fresh[fresh.len() - 1 - s],
            ff[fresh.len() - 1 - s],
        )
    {
        s += 1;
    }
    st.kept = p + s;

    let (o_lo, o_hi) = (p, stored.len() - s);
    let (n_lo, n_hi) = (p, fresh.len() - s);

    let mut slots: Vec<Slot> = (0..p).map(|i| Slot::Keep(i, i)).collect();
    align(
        &stored[o_lo..o_hi],
        &fs[o_lo..o_hi],
        &fresh[n_lo..n_hi],
        &ff[n_lo..n_hi],
        o_lo,
        n_lo,
        &mut slots,
        &mut st,
    );
    slots.extend((0..s).map(|k| Slot::Keep(o_hi + k, n_hi + k)));

    (assign_keys(stored, fresh, &slots, &mut st), st)
}

/// Align the trimmed middles, appending slots and counting kept/updated/inserted/deleted.
#[allow(clippy::too_many_arguments)]
fn align(
    old: &[Node],
    fo: &[u64],
    new: &[Node],
    fnew: &[u64],
    o_off: usize,
    n_off: usize,
    slots: &mut Vec<Slot>,
    st: &mut Update,
) {
    let (n, m) = (old.len(), new.len());
    if n == 0 || m == 0 || n.saturating_mul(m) > LCS_CELL_BUDGET {
        // Nothing to align against, or too big to be worth aligning: pair positionally
        // and let the remainder be pure insert/delete.
        let common = n.min(m);
        for k in 0..common {
            slots.push(Slot::Keep(o_off + k, n_off + k));
            st.updated += 1;
        }
        for k in common..m {
            slots.push(Slot::New(n_off + k));
            st.inserted += 1;
        }
        st.deleted += n.saturating_sub(m);
        return;
    }

    // LCS table over fingerprints.
    let mut dp = vec![0u32; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[at(i, j)] = if same(&old[i], fo[i], &new[j], fnew[j]) {
                dp[at(i + 1, j + 1)] + 1
            } else {
                dp[at(i + 1, j)].max(dp[at(i, j + 1)])
            };
        }
    }

    // Walk the table forward, collecting each unmatched region as a (delete-run,
    // insert-run) pair so it can be reported as in-place updates rather than as a
    // delete plus an unrelated insert — a rewritten function body is one node changing,
    // not one node dying and another being born.
    let (mut i, mut j) = (0, 0);
    let (mut del, mut ins) = (Vec::new(), Vec::new());
    let mut flush = |del: &mut Vec<usize>, ins: &mut Vec<usize>, slots: &mut Vec<Slot>, st: &mut Update| {
        let common = del.len().min(ins.len());
        for k in 0..common {
            slots.push(Slot::Keep(del[k], ins[k]));
            st.updated += 1;
        }
        for &k in &ins[common..] {
            slots.push(Slot::New(k));
            st.inserted += 1;
        }
        st.deleted += del.len().saturating_sub(ins.len());
        del.clear();
        ins.clear();
    };
    while i < n && j < m {
        if same(&old[i], fo[i], &new[j], fnew[j]) {
            flush(&mut del, &mut ins, slots, st);
            slots.push(Slot::Keep(o_off + i, n_off + j));
            st.kept += 1;
            i += 1;
            j += 1;
        } else if dp[at(i + 1, j)] >= dp[at(i, j + 1)] {
            del.push(o_off + i);
            i += 1;
        } else {
            ins.push(n_off + j);
            j += 1;
        }
    }
    del.extend((i..n).map(|k| o_off + k));
    ins.extend((j..m).map(|k| n_off + k));
    flush(&mut del, &mut ins, slots, st);
}

/// Give every slot a key: reused where a stored node backs it, minted between
/// neighbours where one does not.
fn assign_keys(stored: &[Node], fresh: &[Node], slots: &[Slot], st: &mut Update) -> Vec<Node> {
    let mut out: Vec<Node> = Vec::with_capacity(slots.len());
    let mut exhausted = false;
    let mut k = 0;
    while k < slots.len() {
        match slots[k] {
            Slot::Keep(si, ni) => {
                out.push(Node { order: stored[si].order, ..fresh[ni].clone() });
                k += 1;
            }
            Slot::New(_) => {
                // A whole run of new nodes shares one gap, so space them evenly across
                // it rather than repeatedly halving toward the upper bound — bisecting
                // burns the gap in ~32 inserts, spreading uses it once.
                let run_end = slots[k..]
                    .iter()
                    .position(|s| matches!(s, Slot::Keep(..)))
                    .map_or(slots.len(), |off| k + off);
                let count = run_end - k;
                let lo = out.last().map_or(0, |n| n.order);
                let hi = match slots.get(run_end) {
                    Some(Slot::Keep(si, _)) => Some(stored[*si].order),
                    // Appending past the last stored node: no upper bound, so extend at
                    // full stride instead of subdividing.
                    _ => None,
                };
                for (idx, slot) in slots[k..run_end].iter().enumerate() {
                    let ni = match slot {
                        Slot::New(ni) => *ni,
                        _ => unreachable!(),
                    };
                    let order = match hi {
                        None => lo.saturating_add(ORDER_STRIDE * (idx as u64 + 1)),
                        Some(hi) => {
                            let step = (hi - lo) / (count as u64 + 1);
                            match order_between(lo, hi).filter(|_| step > 0) {
                                Some(_) => lo + step * (idx as u64 + 1),
                                None => {
                                    exhausted = true;
                                    // Placeholder; the rebalance below overwrites it.
                                    lo
                                }
                            }
                        }
                    };
                    out.push(Node { order, ..fresh[ni].clone() });
                }
                k = run_end;
            }
        }
    }
    if exhausted {
        rebalance(&mut out);
        st.rebalanced = true;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(order: u64, text: &str) -> Node {
        Node { path: "f".into(), order, kind: "item", text: text.into() }
    }

    #[test]
    fn unchanged_file_touches_nothing() {
        let stored = vec![n(1 << 32, "a"), n(2 << 32, "b")];
        let fresh = vec![n(0, "a"), n(0, "b")];
        let (out, st) = reconcile(&stored, &fresh);
        assert_eq!(out, stored);
        assert_eq!(st.touched(), 0);
        assert_eq!(st.kept, 2);
    }

    #[test]
    fn insert_at_top_keeps_every_existing_key() {
        let stored = vec![n(1 << 32, "a"), n(2 << 32, "b")];
        let fresh = vec![n(0, "hdr"), n(0, "a"), n(0, "b")];
        let (out, st) = reconcile(&stored, &fresh);
        assert_eq!((st.inserted, st.updated, st.deleted, st.kept), (1, 0, 0, 2));
        assert!(!st.rebalanced);
        assert_eq!(out[1].order, 1 << 32, "existing nodes must not be renumbered");
        assert_eq!(out[2].order, 2 << 32);
        assert!(out[0].order > 0 && out[0].order < (1 << 32));
    }

    #[test]
    fn edited_body_is_an_update_not_a_delete_plus_insert() {
        let stored = vec![n(1 << 32, "a"), n(2 << 32, "fn x(){1}"), n(3 << 32, "c")];
        let fresh = vec![n(0, "a"), n(0, "fn x(){2}"), n(0, "c")];
        let (out, st) = reconcile(&stored, &fresh);
        assert_eq!((st.updated, st.inserted, st.deleted), (1, 0, 0));
        assert_eq!(out[1].order, 2 << 32, "an edited node keeps its identity");
        assert_eq!(out[1].text, "fn x(){2}");
    }

    #[test]
    fn keys_stay_ordered_and_project_in_sequence() {
        let stored = vec![n(1 << 32, "a"), n(2 << 32, "d")];
        let fresh = vec![n(0, "a"), n(0, "b"), n(0, "c"), n(0, "d")];
        let (out, st) = reconcile(&stored, &fresh);
        assert_eq!(st.inserted, 2);
        assert!(out.windows(2).all(|w| w[0].order < w[1].order));
        assert_eq!(super::super::project(&out).0, "abcd");
    }

    #[test]
    fn exhausted_gap_rebalances_instead_of_failing() {
        // Adjacent keys leave no room between them.
        let stored = vec![n(10, "a"), n(11, "b")];
        let fresh = vec![n(0, "a"), n(0, "mid"), n(0, "b")];
        let (out, st) = reconcile(&stored, &fresh);
        assert!(st.rebalanced);
        assert!(out.windows(2).all(|w| w[0].order < w[1].order));
        assert_eq!(super::super::project(&out).0, "amidb");
    }

    #[test]
    fn deletion_drops_only_the_removed_node() {
        let stored = vec![n(1 << 32, "a"), n(2 << 32, "b"), n(3 << 32, "c")];
        let fresh = vec![n(0, "a"), n(0, "c")];
        let (out, st) = reconcile(&stored, &fresh);
        assert_eq!((st.deleted, st.inserted, st.updated), (1, 0, 0));
        assert_eq!(out.iter().map(|n| n.order).collect::<Vec<_>>(), vec![1 << 32, 3 << 32]);
    }

    #[test]
    fn first_ingest_numbers_from_scratch() {
        let (out, st) = reconcile(&[], &[n(0, "a"), n(0, "b")]);
        assert_eq!(st.inserted, 2);
        assert_eq!(out[0].order, initial_order(0));
        assert_eq!(out[1].order, initial_order(1));
    }
}
