# Corrode graph model

**Status: design.** The graph-backed VFS is not yet implemented — `vfs.rs` ships a
filesystem `PassthroughVfs` and `graph.rs`'s `HelixStore` methods are stubs. This
document is the intended design the implementation grows into. The wire types it
references that *do* exist (`FileNodeView.is_dir`/`.mode`, `ProjectionMode`,
`FallbackReason`) live in `corrode-core`; the store is the embedded HelixDB
(`graph::embedded::HelixStore`).

---

## 1. Why a graph

Corrode's source of truth is a **graph**, not a file tree. A file is a *projection*
of the graph — the VFS materializes a git-compliant working copy from graph nodes,
and edits absorb back into node/edge mutations. The graph earns this role by holding
what a file tree cannot: the **relationships** between pieces of code, including the
ones that are invisible in the source text (callbacks, events, reactive
state→render, dynamic dispatch, the Rust↔JS boundary). Those hidden relationships
are the payoff — they answer the runtime questions an agent must answer to edit
safely ("if I change this signal, what re-renders?", "what actually runs when this
trait method is called?"), and they are exactly what a static call graph misses.

The whole world — the project's code *and* its dependencies *and* the platform APIs
it calls — lives in **one uniform graph of functions-with-descriptions**, in one
HelixDB store (graph traversal + vector search + GraphRAG in a single embedding
space). Code, deps, and browser platform are the same kind of node.

## 2. Core model

- **Nodes are functions.** The function is the atomic unit.
- **Node metadata is the function's comments.** Doc/inline comments ride along as
  the node's metadata — which is also what the vector side embeds for retrieval.
- **The file↔graph mapping is a bijection — "assisted byte-identity".**
  `file → graph → file` reconstitutes the original; `graph → file → graph` preserves
  the graph. This invariant is the correctness contract that lets the VFS be a
  projection rather than a separate copy: writes can absorb losslessly, and
  git-aware tools trust the working copy.
- **Bodies exact, connective tissue tool-canonical.** Function bodies round-trip
  verbatim. Imports/connective tissue are **not** authored text — an import is the
  *projection of an edge* to a library node (see §5). Dependencies are ingested as a
  subgraph of API (signatures) + descriptions (doc comments), bodies elided. Adding
  a node that references a library symbol auto-resolves to the API node and creates
  the edge; regeneration materializes the `use` line from it.

## 3. Projection modes

Each file is backed one of two ways (`FileNodeView.mode: ProjectionMode`):

- **Composed** — the file is *derived* from graph nodes; byte-identity is *earned*
  by regeneration (canonical formatting + edge-derived imports). No verbatim safety
  net, so composition is **gated on a verified round-trip at ingest** (parse →
  regenerate → diff vs. original). On mismatch it **falls back to overlay**.
  "Composed by default, overlay by proof."
- **Overlay** — the file is a **verbatim flat node** (the source of truth); function
  nodes are spans/anchors *onto* it (anchors, never copied bytes) so they still
  participate in the graph. Byte-identity is by *storage*, so extraction is
  **best-effort and non-lossy** — a parser can recognize what it recognizes and
  leave the rest untouched, improving over time with zero identity risk. Overlay is
  the **universal floor**, including Composed's fallback.

A file that fails composition surfaces the reason, never degrades silently:
`ProjectionMode::OverlayFallback(FallbackReason)` with a typed reason
(`RustfmtSkip`, `MacroExpansion`, `RawStringMismatch`, `AttributePlacement`,
`UnknownDivergence { first_diff_offset }`) so fallbacks *aggregate* into a
weak-projector signal, and the explorer flags which files dropped to overlay.

## 4. Projector architecture — core vs. modular

A **projector/absorber** is the per-language plugin that (a) parses source into
nodes/edges, (b) composes/overlays files, and (c) extracts hidden edges (§6). The
architecture is deliberately split:

- **Core (compiled into the daemon): Rust + Leptos.** Corrode is a Rust tool whose
  own front-end is Leptos/wasm, so Rust is the reference *composed* projector and
  Leptos is a first-class, framework-aware extension of it (it understands `view!`,
  `signal()`, `Effect::new`, `RwSignal`, and cribs `reactive_graph`'s subscriber
  index — §7). These ship in-process; they are the substrate Corrode dogfoods on.
- **Modular (plugins conforming to the projector interface): everything else.**
  JS/TS, **React**, Python, C/C++, Bash, Markdown, and the browser-API subgraph are
  loadable modules, present only when a project needs them. React in particular is a
  *module*, not core — it implements the same reactive-edge *type* as Leptos through
  a different source (declared `useEffect` deps rather than a runtime subscriber
  index — §7).

The projector interface is the seam. Roughly: `parse(bytes) → (nodes, structural
edges, best-effort semantic edges)`, `compose(nodes, edges) → bytes` (optional; only
composed-mode languages implement it), and `hidden_edges(...) → semantic edges`. A
new language is a new plugin; the core never grows to know it. This mirrors the
"bootstrap-and-graduate" path: a language starts as an overlay-only plugin and earns
a composer as its projector matures.

## 5. Edges — two layers

- **Structural edges** — imports, module tree. Composed-mode edges that **derive
  text**: an edge regenerates the `use`/`import` line. The edge stores the import
  *kind*; **order is a per-language projector property, not a universal edge field**
  (noise in Rust/Python, load-bearing in C/C++/Bash).
- **Semantic edges** — the call graph **plus all hidden/dynamic edges (§6)**.
  **Knowledge-only: never projected back to text**, so they cannot threaten
  byte-identity. Pure enrichment for GraphRAG, navigation, and impact analysis.

Overlay-mode languages still carry semantic edges (for retrieval/navigation); they
just don't regenerate imports from structural edges until the language graduates to
composed.

## 6. Hidden / dynamic edges

The relationships a static call graph is blind to. Each is a semantic edge type,
extracted per-language by a framework-aware projector, best-effort (it rides the
overlay safety net — extract what you recognize, the verbatim node protects the
rest):

1. **Callback / higher-order** — a function passed as a value (`onData(cb)`,
   `.map(f)`, `Box<dyn Fn>`) linked to its later invocation site. *"Who eventually
   calls this closure?"*
2. **Event / pub-sub** — an event node with emitters ↔ listeners
   (`addEventListener`, `emit`, channels). Many-to-many.
3. **Reactive: state → render** — a signal/state node with **writers**
   (`signal.set`, `setState`) ↔ **readers/effects** that re-run when it changes
   (Leptos `Effect`/`view!`, React component/`useEffect` deps). *"If I change this
   signal, what re-renders?"* The highest-value edge for UI code. See §7.
4. **Dynamic dispatch** — an abstract method (Rust trait, TS interface, virtual) ↔
   its concrete impls; a call-site → the impl **set** it may resolve to (`dyn
   Trait`, monomorphization, JS prototype/duck-typing). *"What actually runs when
   this trait method is called?"*
5. **Cross-boundary (FFI / wasm-bindgen)** — a name-link *across* the
   language/runtime boundary. Corrode's own webui is the canonical example: Rust
   `#[wasm_bindgen] extern` `corrode_term_init` ↔ JS `window.corrodeTermInit`. The
   wasm module's import/export interface is the seam; this reuses the "add a node →
   auto-link via edge" mechanism, cross-language.

## 7. Reactive edges — crib the framework's own index

Statically, reactive edges look approximate: a signal read inside a branch/loop is
only a *possible* dependency. But the frameworks already compute the exact answer —
so crib it. Three levels, increasingly strong:

1. **Vocabulary (compile-time, static).** Reuse the framework's own definitions —
   Leptos's `Track`/`Get` traits, `RwSignal`/`Memo`/`Effect`, and the `view!` macro,
   which *brackets* each dynamic fragment in a reactive closure. The projector then
   knows exactly what a read and a reactive scope are, not a heuristic. Accurate up
   to control flow.
2. **Runtime index (dynamic, exact).** Leptos's `reactive_graph` maintains, at
   runtime, each signal's subscriber set + each effect's source set — that index
   **is** the reactive edge set, computed by the framework. Patch it to emit
   registrations, or snapshot after running the target's tests; those edges are
   ground truth, branches and all.
3. **Hybrid → principled confidence.** An edge is **confirmed** if the runtime
   registered it, **inferred** if only the static projector saw it. Corrode can
   *close the loop*: it has a terminal and a VFS, so it can run a target's tests,
   harvest the reactive graph, and upgrade inferred → confirmed — dogfooding on its
   own Leptos webui first. GraphRAG ranks confirmed edges above inferred.

**Per-framework, and why Leptos is core / React is modular:** Leptos/SolidJS
reactivity is *runtime-tracked and inspectable*, so the core Leptos projector cribs
the subscriber index for exact edges. **React is different** — `useEffect` deps are
a *declared* array a module can read statically (no runtime harvest), and
render-time reads aren't tracked the same way. Same edge *type*, different source;
hence React is a module implementing the reactive-edge contract its own way.

The confirmed/inferred split generalizes beyond reactivity: **dynamic dispatch** and
**callbacks** also have runtime ground truth (a trace of which impl ran, which
closure fired), so a Corrode test run can confirm those edge classes too. The static
projector proposes; execution confirms.

## 8. Language & platform targets

| Target | Tier | Projection | Notable edges |
|---|---|---|---|
| **Rust** | core | composed (overlay fallback) | reference composer; traits → dispatch |
| **Leptos** | core | (rides Rust) | reactive (crib `reactive_graph`), `view!`, components |
| Python | module | overlay → graduate | callbacks, dispatch (duck) |
| C / C++ | module | overlay (preprocessor hard) | dispatch (virtual), FFI |
| Bash | module | overlay | `source`, functions |
| Markdown | module | overlay (never graduates) | doc links |
| **JS / TS** | module | overlay | callbacks, events (event-driven) |
| **React** | module | (rides JS/TS) | reactive (declared `useEffect` deps) |
| **WASM** | module | boundary artifact | wasm-bindgen name-links; import/export interface |
| **Browser APIs** | module | external API subgraph | code → API; `addEventListener` → event |

**WASM** is not a source language you edit — the artifact is the wasm-bindgen
boundary (Rust↔JS name-links) plus the module's import/export interface. **Browser
APIs** are an external API+description subgraph (DOM, `fetch`, `WebSocket`, `Canvas`,
`HtmlElement`, …) exactly like the library-dependency subgraph — MDN-like
descriptions instead of doc comments — so code + deps + platform share one embedding
space and retrieval spans all three.

The through-line: a static call graph is nearly blind to a Leptos/wasm app (the real
structure is signals→views, DOM events→handlers, closures→async, Rust↔JS). So the
web-stack targets and the hidden edges are **one requirement**, and it's exactly the
code Corrode's own webui is made of.

## 9. Fallback & confidence visibility

Two "never present a guess as fact" surfaces, same spirit:

- **Projection fallback** — `ProjectionMode::OverlayFallback(FallbackReason)`,
  aggregated by reason, so a growing fallback set flags a weak composer (§3).
- **Edge confidence** — semantic edges carry confirmed (runtime-observed) vs.
  inferred (static) so the agent knows a certainty from a heuristic (§7). Direct
  calls are certain; possible reactive/dispatch/callback links are inferred until a
  run confirms them.

## 10. Materialization & retrieval

- **VFS / FUSE.** The `Vfs` trait (`list`/`stat`/`read`/`write`, async) projects the
  graph as a git-compliant tree. A feature-gated `fuse` adapter mounts it as a real
  filesystem (git and subagent shells see a normal working copy); writes buffer
  per-fh and commit once at `release` — the graph "absorb" boundary (one mutation
  per edit, not per syscall).
- **Retrieval (GraphRAG).** The vector side embeds node descriptions (comments, dep
  API descriptions, browser-API descriptions); the graph side grounds answers by
  walking edges — including hidden edges (all effects of a signal, all handlers of
  an event). `context_prefix` relevance-ranking and `AgentCommand::DocQuery` both
  run over this one store. The `node_id` on `FileNodeView` is the file→graph pivot.

## 11. Open questions

- **Vendoring `reactive_graph`.** Patch/fork it to emit dependency registrations
  (like the `helix-db` submodule), or upstream a small "registration hook" so
  Corrode isn't maintaining a fork?
- **Anchor maintenance in overlay mode.** Function-node spans must survive edits to
  other parts of the flat file (incremental re-extraction) — the running cost overlay
  pays that composed mode avoids.
- **Cross-boundary resolution.** wasm-bindgen name-links are resolvable statically;
  looser FFI (dlopen, dynamic `import()`) may need runtime confirmation.
- **Approximation budget.** How aggressively to infer dispatch/callback edges before
  a run confirms them, without drowning retrieval in low-confidence noise.
