//! The egui/WebGL graph-explorer canvas. (The terminal moved to xterm.js; the
//! Markdown/LaTeX agent console is DOM — this canvas is just the node graph, the
//! one surface where GPU rendering earns its keep.)
//!
//! eframe's `WebRunner` drives the render loop on a `<canvas>` the Leptos shell
//! owns; we hand it a clone of the [`Shared`] model that the websocket task writes.
//! The provenance graph renders force-directed: pairwise repulsion, edge springs,
//! a mild centering pull, damped velocities — stepped per frame, repainting only
//! while the layout is still moving. Nodes are draggable; new nodes join the
//! simulation where their id hash seeds them, so layouts are stable per plan.

use std::collections::HashMap;

use eframe::CreationContext;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use web_sys::HtmlCanvasElement;

use crate::model::Shared;

/// Start the egui app on `canvas`. Runs asynchronously (eframe's start is async);
/// stores the egui `Context` into the shared model so async pushes can repaint.
pub fn start(canvas: HtmlCanvasElement, shared: Shared) {
    let runner = eframe::WebRunner::new();
    wasm_bindgen_futures::spawn_local(async move {
        let result = runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc: &CreationContext<'_>| {
                    shared.borrow_mut().egui_ctx = Some(cc.egui_ctx.clone());
                    Ok(Box::new(GraphApp {
                        shared,
                        sim: HashMap::new(),
                    }) as Box<dyn eframe::App>)
                }),
            )
            .await;
        if let Err(e) = result {
            web_sys::console::error_1(&format!("egui runner failed: {e:?}").into());
        }
    });
}

/// One simulated node: position and velocity, keyed by provenance id so the
/// layout survives graph growth (new nodes join, vanished nodes drop out).
struct Body {
    pos: Pos2,
    vel: Vec2,
}

struct GraphApp {
    shared: Shared,
    sim: HashMap<String, Body>,
}

// Force-layout tuning. Small graphs (MAX_PLAN_TASKS bounds them) converge in well
// under a second with these; repulsion is O(n²) per frame, fine at this scale.
const REPULSION: f32 = 2600.0;
const SPRING_K: f32 = 0.06;
const SPRING_REST: f32 = 70.0;
const CENTERING: f32 = 0.015;
const DAMPING: f32 = 0.82;
const SETTLED: f32 = 0.08;

impl eframe::App for GraphApp {
    // egui 0.35 is Ui-first: we get a `&mut Ui` (the central area), not a Context.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let nodes = self.shared.borrow().plan_nodes.clone();
        if !nodes.is_empty() {
            self.draw_force_graph(ui, &nodes);
            return;
        }
        let m = self.shared.borrow();
        ui.heading("graph explorer");
        ui.separator();
        if m.entries.is_empty() {
            ui.weak("(list a directory, or prompt the swarm, to populate)");
        }
        for (path, is_dir) in &m.entries {
            ui.label(format!("{} {}", if *is_dir { "▸" } else { "•" }, path));
        }
    }
}

impl GraphApp {
    fn draw_force_graph(&mut self, ui: &mut egui::Ui, nodes: &[corrode_core::GraphNodeView]) {
        let rect = ui.available_rect_before_wrap();
        let painter = ui.painter_at(rect);

        // Membership: seed newcomers (hash-scattered around a kind-biased row so
        // plan floats up and code sinks), drop the departed.
        self.sim
            .retain(|id, _| nodes.iter().any(|n| &n.id == id));
        for n in nodes {
            self.sim.entry(n.id.clone()).or_insert_with(|| {
                let h = fnv(&n.id);
                let fx = (h % 997) as f32 / 997.0;
                let fy = (h / 997 % 997) as f32 / 997.0;
                let row = match n.kind.as_str() {
                    "plan" => 0.2,
                    "task" | "contract" => 0.5,
                    _ => 0.8,
                };
                Body {
                    pos: Pos2::new(
                        rect.left() + rect.width() * (0.2 + 0.6 * fx),
                        rect.top() + rect.height() * (row + 0.12 * (fy - 0.5)),
                    ),
                    vel: Vec2::ZERO,
                }
            });
        }

        // Snapshot into index order for the O(n²) pass.
        let mut pos: Vec<Pos2> = nodes.iter().map(|n| self.sim[&n.id].pos).collect();
        let mut vel: Vec<Vec2> = nodes.iter().map(|n| self.sim[&n.id].vel).collect();
        let index: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();

        let mut force = vec![Vec2::ZERO; nodes.len()];
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let d = pos[i] - pos[j];
                let dist2 = d.length_sq().max(64.0);
                let push = d / dist2.sqrt() * (REPULSION / dist2);
                force[i] += push;
                force[j] -= push;
            }
            force[i] += (rect.center() - pos[i]) * CENTERING;
        }
        for (i, n) in nodes.iter().enumerate() {
            for target in &n.edges_out {
                if let Some(&j) = index.get(target.as_str()) {
                    let d = pos[j] - pos[i];
                    let len = d.length().max(1.0);
                    let pull = d / len * ((len - SPRING_REST) * SPRING_K);
                    force[i] += pull;
                    force[j] -= pull;
                }
            }
        }

        let mut kinetic = 0.0f32;
        for i in 0..nodes.len() {
            vel[i] = (vel[i] + force[i]) * DAMPING;
            pos[i] += vel[i];
            pos[i] = pos[i].clamp(rect.min + Vec2::splat(14.0), rect.max - Vec2::splat(20.0));
            kinetic += vel[i].length();
        }

        // Edges under nodes.
        for (i, n) in nodes.iter().enumerate() {
            for target in &n.edges_out {
                if let Some(&j) = index.get(target.as_str()) {
                    painter.line_segment(
                        [pos[i], pos[j]],
                        Stroke::new(1.0, Color32::from_gray(80)),
                    );
                }
            }
        }
        // Nodes: draw, drag (a dragged node is pinned to the pointer, velocity
        // reset so it doesn't fling on release), hover for the full label.
        for (i, n) in nodes.iter().enumerate() {
            let hit = Rect::from_center_size(pos[i], Vec2::splat(18.0));
            let resp = ui.interact(hit, ui.id().with(("node", i)), Sense::drag());
            if resp.dragged() {
                pos[i] += resp.drag_delta();
                vel[i] = Vec2::ZERO;
            }
            let r = if resp.hovered() { 9.0 } else { 7.0 };
            painter.circle_filled(pos[i], r, kind_color(&n.kind));
            let label: String = n.label.chars().take(26).collect();
            painter.text(
                pos[i] + Vec2::new(0.0, 11.0),
                Align2::CENTER_TOP,
                label,
                FontId::proportional(10.0),
                Color32::from_gray(205),
            );
            resp.on_hover_text(format!("[{}] {}", n.kind, n.label));
        }

        for (n, body) in nodes.iter().zip(pos.iter().zip(vel.iter())) {
            self.sim.insert(
                n.id.clone(),
                Body {
                    pos: *body.0,
                    vel: *body.1,
                },
            );
        }

        // Legend, top-left.
        let mut x = rect.left() + 8.0;
        for kind in ["plan", "task", "contract", "code"] {
            painter.circle_filled(Pos2::new(x, rect.top() + 10.0), 4.0, kind_color(kind));
            painter.text(
                Pos2::new(x + 8.0, rect.top() + 10.0),
                Align2::LEFT_CENTER,
                kind,
                FontId::proportional(10.0),
                Color32::from_gray(160),
            );
            x += 16.0 + 6.5 * kind.len() as f32;
        }

        if kinetic > SETTLED {
            ui.ctx().request_repaint();
        }
    }
}

/// Node fill per provenance kind (`NodeKind::as_str` on the wire).
fn kind_color(kind: &str) -> Color32 {
    match kind {
        "plan" => Color32::from_rgb(224, 138, 75),
        "task" => Color32::from_rgb(94, 140, 190),
        "contract" => Color32::from_rgb(150, 110, 190),
        _ => Color32::from_rgb(110, 170, 100), // code
    }
}

/// Tiny FNV-1a — deterministic seed scatter without a rand dependency.
fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
