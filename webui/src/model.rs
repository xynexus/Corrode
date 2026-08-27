//! Shared UI state written by the websocket task and read by the egui graph canvas.
//!
//! wasm is single-threaded and every task here runs on that one thread
//! (`spawn_local`), so `Rc<RefCell<_>>` is sufficient — no `Send`/locking. (The DOM
//! panels use Leptos signals; the terminal lives in xterm.js; this backs the egui
//! graph canvas only.)

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub struct UiModel {
    /// Graph explorer nodes, `(path, is_dir)`, from `DirListing`.
    pub entries: Vec<(String, bool)>,
    /// The latest turn's provenance graph (plan -> task/contract -> code), from
    /// `PlanGraph` — when non-empty the canvas draws this instead of the listing.
    pub plan_nodes: Vec<corrode_core::GraphNodeView>,
    /// egui repaint handle, set once the canvas app starts, so an async push from
    /// the websocket can wake the render loop (egui only repaints on demand).
    pub egui_ctx: Option<egui::Context>,
}

impl UiModel {
    /// Merge a one-hop `Neighbors` subgraph into the drawn graph: union nodes by id
    /// and union their outgoing edges, so clicking a node *expands* the canvas
    /// (revealing provenance from earlier turns) instead of replacing it.
    pub fn merge_nodes(&mut self, incoming: Vec<corrode_core::GraphNodeView>) {
        for n in incoming {
            if let Some(existing) = self.plan_nodes.iter_mut().find(|e| e.id == n.id) {
                for t in n.edges_out {
                    if !existing.edges_out.contains(&t) {
                        existing.edges_out.push(t);
                    }
                }
            } else {
                self.plan_nodes.push(n);
            }
        }
    }
}

pub type Shared = Rc<RefCell<UiModel>>;

pub fn shared() -> Shared {
    Rc::new(RefCell::new(UiModel::default()))
}
