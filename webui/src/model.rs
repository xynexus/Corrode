//! Shared UI state written by the websocket task and read by the egui canvas.
//!
//! wasm is single-threaded and every task here runs on that one thread
//! (`spawn_local`), so `Rc<RefCell<_>>` is sufficient — no `Send`/locking. (The
//! DOM panels use Leptos signals instead; this model backs only the egui side.)

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub struct UiModel {
    /// Virtual-terminal scrollback: decoded `AgentEvent::TerminalOutput` bytes.
    pub terminal: String,
    /// Repo/graph explorer entries, `(path, is_dir)`, from `DirListing`.
    pub entries: Vec<(String, bool)>,
    /// egui repaint handle, set once the canvas app starts, so an async push from
    /// the websocket can wake the render loop (egui only repaints on demand).
    pub egui_ctx: Option<egui::Context>,
}

pub type Shared = Rc<RefCell<UiModel>>;

pub fn shared() -> Shared {
    Rc::new(RefCell::new(UiModel::default()))
}
