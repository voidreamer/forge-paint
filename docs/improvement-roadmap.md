# forge-paint improvement roadmap

An organize / simplify / improve audit of this repository, and the
phased plan for acting on it. Produced 2026-07-10 from a full read of
the tree at `385efb7` (three parallel deep-dives: core architecture,
subsystem boundaries/duplication, and hygiene/CI/robustness), then
verified against the code before writing anything down.

Phase 1 was executed on the same branch that added this document; its
items are marked ✅ with the landing commits. Everything else is
sequenced but not started.

---

## 1. Current state

**Size** (at audit time): ~25.5k lines of Rust across 70 files, plus
~2.7k lines of WGSL in 18 shaders. One binary crate (`src/`) + one
vendored library crate (`crates/texture-baker`).

| Hotspot | Size | Note |
| --- | --- | --- |
| `src/app.rs` | 6,301 lines | 66-field `App` struct; ~49% egui panels, ~16% Hydra orchestration, ~18% USD/IO, ~17% state/types |
| `src/viewport.rs` | 2,285 lines | 66-field `Viewport`, 47 fields `pub`; `show()` alone is ~1,010 lines |
| `src/assets.rs` | 1,402 lines | asset browser + eight near-duplicate `apply_as_*` fns |
| `crates/texture-baker/src/baker.rs` | 852 lines | the crate's real API surface |
| `src/main.rs` | 792 lines | entry + CLI + logging + ~340 lines of bundle-env bootstrap |

**Tests**: 3 unit tests at audit time (OBJ/glTF converters), now 11
(Phase 1 added 8 to texture-baker). No GPU/integration harness.

### Strengths to preserve

The codebase is unusually clean for its size — the problems are size
and coupling, not rot:

- Zero `TODO`/`FIXME`/commented-out blocks; debt is written as honest
  prose ("for now", "v0") where it lives.
- Disciplined `anyhow` + `.with_context()` at I/O boundaries; only ~22
  `unwrap`/`expect` in 25k lines, mostly invariant-based.
- No `Rc`/`RefCell`/`Mutex` tangles — ownership is a clean tree
  (`App` → `Viewport` → GPU resources).
- Excellent inline comment density; the release pipeline
  ([release.md](release.md)) documents genuinely subtle Windows
  bundling behavior (USD double-load, pre-`main()` env capture).
- Tiny-assets policy with a well-annotated `.gitignore`.

---

## 2. Findings by severity

### HIGH — onboarding, CI, distribution (all fixed in Phase 1 ✅)

1. **Fresh clones could not build.** `[patch.crates-io]` pointed at the
   private `../rust-usd` sibling; Cargo hard-errors when patch paths
   are missing, so even `cargo metadata` failed without it. Both
   release workflows carried a strip-the-manifest regex that deleted
   from a comment marker to EOF — silently eating anything appended
   after the block. *Fixed:* the override moved to an untracked
   `.cargo/config.toml` (README documents the 3-line file); workflows
   dropped the strip steps; the committed `Cargo.lock` is now
   registry-flavored.
2. **No CI on pushes or PRs** — only dispatch/tag release builds; no
   `cargo test`/`clippy`/`fmt` anywhere. *Fixed:* `ci.yml` gates every
   push/PR with fmt + clippy (`-D warnings`) + texture-baker tests on
   ubuntu.
3. **Proprietary 3Delight SDK zip (11.8 MB) committed** to a public
   repo at `.github/3delight-windows.zip`. *Fixed:* removed from HEAD;
   packaging uses the `DELIGHT_WINDOWS_ARCHIVE_URL` secret (or
   `DELIGHT`/`DELIGHT_WINDOWS_ARCHIVE_PATH` on self-hosted runners).
   The blob is still in *history* — purge is a deferred opt-in (§5).
4. **No README, no LICENSE.** *Fixed:* both added (MIT).

### MEDIUM — correctness/robustness debt (Phases 2–6)

5. **Sidecar versioning is nominal.** `project.rs` writes
   `version: SCHEMA_VERSION` but `load_sidecar()` never reads it — no
   migration hook, and opening a newer sidecar silently drops unknown
   fields, destroying them on the next save.
6. **Undo holds up to 16 full-resolution GPU texture copies**
   (`undo.rs`, `DEFAULT_DEPTH = 16`, not configurable): ~256 MB VRAM at
   2k, ~1 GB at 4k, ~4 GB at 8k. Structural edits (layer add/remove)
   and the Displacement channel aren't undoable at all.
7. **Errors are log-only for most failure paths.** `rfd` is a
   dependency but `MessageDialog` is never used; `App.status` is set
   inconsistently; a failed texture import or sidecar parse is
   invisible unless the user reads the log. GPU readback panics on
   device loss (`texture-baker/src/gpu/common.rs`, `read_back_buffer`'s
   double `unwrap`); two render-path unwraps in `viewport.rs`
   (mask-channel view, stencil view) can crash the frame loop.
8. **Version lattice**: egui 0.31 ↔ egui-wgpu 0.31 ↔ wgpu 24 ↔
   egui-snarl 0.7 ↔ egui-phosphor 0.9, with texture-baker pinned down
   to wgpu 24 from upstream 25. Well-documented but a lockstep
   migration whenever egui moves (snarl 0.8+ requires egui 0.33).

### Code organization (Phases 4–5)

9. **`App` is a God object**: 66 fields in visible clusters — ~14
   Hydra fields, 9 `uv_*` fields, 6 material-binding, 6 dialog/modal.
   The borrow checker forced UI panels into free functions taking
   fields individually, producing `draw_hydra_central` with **19
   parameters** and `uv_view_body` with 13 — each signature is a
   sub-struct begging to exist.
10. **`Viewport` leaks 47 of 66 fields as `pub`**; `app.rs` reaches into
    them 381 times (`vp.layer_stack` ×39, `vp.brush` ×26,
    `vp.paint_target` ×17, `vp.mesh_maps` ×14…). There is no actual
    App↔Viewport API boundary.
11. **`viewport.show()` (~1,010 lines)** braids input handling (~450),
    the GPU pass sequence (~200), and the stencil egui overlay (~100)
    into one method.
12. **`main.rs` mixes concerns**: entry, `Convert` subcommand business
    logic, file/panic/native-crash logging, and ~340 lines of
    Windows/macOS bundle-env bootstrap (`relaunch_from_bundled_usd_lib`
    and friends).
13. **Near-circular module dependency**: `MaterialBindingInstance`
    lives in `app.rs` but `material_graph.rs` reaches back into
    `crate::app` for it.
14. **~350 lines of USD authoring inside `app.rs`**
    (`write_usd_preview_material`, `write_texture_shader`,
    `resolve_texture_reference`, …) that belong under `src/usd/`.

### Duplication (Phases 2–3, one item done ✅)

15. ✅ **`src/accel.rs` was fully dead** — a near-copy of
    texture-baker's BVH that `Viewport` *built on every mesh load and
    never queried* (picking is brute-force `pick::pick`). Deleted in
    Phase 1, along with root-crate `bvh`/`nalgebra` deps it alone used,
    plus three dead stubs (`to_baker_mesh`, a rollout placeholder
    const, `env::_unused`) and a vestigial AO bind group + params
    buffer in `texture-baker/src/gpu/ao_baker.rs`.
16. **PBR channel-set triplication**: `paint/target.rs`,
    `paint/layer.rs`, and `bake/mod.rs` each re-implement the same
    "4-channel texture array + per-tile views + neutral clear-fill"
    trio (`make_array` / `array_view` / `tile_view`). A shared
    `PbrChannelSet` would collapse a large slice of those 1,600 lines.
17. **`post.rs` / `fxaa.rs` are near-identical twins** (same BGL, same
    fullscreen-triangle pipeline, same uniform plumbing — only labels
    and the uniform struct differ); the same shape recurs in
    `background.rs` and `env/skybox.rs`. Across the app there are ~70
    pipeline-creation and ~90 resource-creation boilerplate sites and
    no shared GPU utility module; the premultiplied-alpha `BlendState`
    literal is pasted 3×.
18. **`BakeConfig` (~30 fields) is populated twice** with mostly
    identical values (`bake_cli.rs` vs `bake/integration.rs::make_config`),
    including file-output fields that are meaningless for the preview
    path; `curvature_intensity` still shadows
    `curvature_settings.intensity` for backward compat.
19. **UDIM reverse mapping** (tile id → offset) is re-derived inline in
    four places (`bake/mod.rs`, `bake/integration.rs`, `app.rs` ×2)
    while `paint/udim.rs` only exposes the forward mapping.
20. **Tangent generation duplicated ~90%** between `src/tangents.rs`
    and `texture-baker/src/tangent.rs` (both wrap `bevy_mikktspace`).
21. **Mesh ingestion is spread over four parsers**: tobj (baker),
    a hand-rolled OBJ parser (`obj_to_usd.rs`), and the `gltf` crate
    twice (baker + `gltf_to_usd.rs`); `compute_vertex_normals` exists
    twice. (`tools/obj_to_usd.py` is a deliberate dependency-free
    parity port — keep.)
22. **Small dedups**: `usd/loader.rs`'s two merge functions are ~90%
    identical; `Layer::new` vs `new_fill` ~95%; `assets.rs`'s eight
    `apply_as_*` fns are tile/non-tile pairs of the same body;
    `PaintTarget::new` takes an unused `_material_bgl` param.

### LOW / notes

23. `egui-snarl`'s `serde` feature is justified by a comment promising
    sidecar round-trips of graph layout, but `MaterialGraph` is never
    persisted — implement or drop (Phase 7).
24. `test_assets/cube.usda` is not used by any test — it's a bundled
    sample staged into release zips; rename or point a real test at it.
25. Module-level `//!` docs exist on only ~32/70 files; none on
    `app.rs`, `viewport.rs`, or anything in texture-baker.
26. "Unbounded" dilation (`dilate.rs`, `iterations == 0`) caps at
    `max(w, h)` passes, which cannot cross the full atlas diagonally
    from a corner seed (needs `w + h - 2`); fine for real bakes seeded
    from island edges — documented in its test.
27. Layer *reordering* is promised in a `layer.rs` comment but not
    implemented (no `move_up`/`move_down` on `LayerStack`).

---

## 3. Phase plan

Dependencies: P1 underpins everything (CI + resolvable workspace).
P2–P3 shrink the code before P4–P5 move it. P6 is independent after P1
and can jump the queue (the undo-VRAM item especially). P7–P8 are
opportunistic.

### ✅ Phase 1 — onboarding, hygiene, CI foundations (done on this branch)

Repo-wide `cargo fmt` + `.git-blame-ignore-revs`; patch-block migration
to untracked `.cargo/config.toml` (fresh clones resolve; strip-regex
steps deleted from both release workflows); `ci.yml` (fmt + clippy
`-D warnings` + tests on every push/PR); texture-baker's first
Linux-clean compile, 8 seed tests; dead-code deletions (§15); 3Delight
zip removal (§3); doc moves (Windows postmortem → `docs/`, baking
research → `crates/texture-baker/docs/`); MIT LICENSE + README; this
document. **Exit criteria met**: fresh clone resolves with no sibling
checkout; ubuntu CI green; macos-build dispatch green on the branch.

### Phase 2 — compile-verified small dedups (~0.5–1 day)

One small, independently revertable commit each:
`udim::tile_offset(id)` helper replacing the four inline derivations
(§19); single `BakeConfig` builder shared by `bake_cli` and
`bake/integration`, retiring the `curvature_intensity` shadow (§18);
`usd/loader.rs` merge-fn unification (§22); `Layer::new_sized` merging
`new`/`new_fill` (§22); drop `PaintTarget::new`'s unused param (§22);
implement-or-delete layer reordering (§27).
**Verify**: macos-build dispatch green; paint/bake smoke on the dev Mac.

### Phase 3 — shared GPU utils, PbrChannelSet, converter crate (~2–3 days)

(a) `src/gpu/` utility module: fullscreen-pass builder unifying
`post.rs`/`fxaa.rs` first (the twins), then `background.rs`/`skybox.rs`
opportunistically; shared premult-alpha `BlendState` const;
render-target/texture-array helpers (§16–17).
(b) `PbrChannelSet` owned by `PaintTarget`, `Layer`, and `MeshMaps`.
(c) Extract `obj_to_usd` / `gltf_to_usd` / the text half of `usd_out`
into a USD-free crate (e.g. `crates/usd-emit`) so its tests run in the
cheap Linux CI — the `.usd`/`.usdc` re-encode call stays app-side.
(d) Unify tangents with `texture_baker::tangent` (§20).
**Verify**: ubuntu CI now runs two crates' tests; macos dispatch green;
manual smoke (paint each channel, bake, post/FXAA toggle, skybox).

### Phase 4 — app.rs decomposition (~2–4 days, stepwise)

Step 1: field-cluster sub-structs on `App` — `HydraUiState`,
`UvViewState`, `MaterialBindingState`, `DialogState` — turning the
19-param `draw_hydra_central` and 13-param `uv_view_body` into methods
(§9). Step 2: file split to `src/app/{mod,menu,panels/{properties,
layers,mesh_maps,environment,uv_view,material_editor,asset_browser},
hydra,stage_io,persistence}.rs`; USD authoring helpers move to
`src/usd/authoring.rs` (§14); `MaterialBindingInstance` moves out of
`app.rs` to break the back-reference (§13). Step 3: split `main.rs`
into `main.rs` + `bootstrap.rs` (bundle-env machinery) +
`convert_cli.rs` (§12).
**Target**: `app/mod.rs` < ~800 lines; no function > ~100 lines except
the `update()` dispatch; behavior unchanged.

### Phase 5 — viewport decomposition + a real App↔Viewport API (~2–3 days)

Split into `viewport/{mod,setup,input,render,overlay,paint,sidecar}.rs`
(§11); then shrink the public surface: the 148 direct subsystem
reach-ins become intent-level methods or narrow accessor structs;
target < 15 pub fields (§10). Best after P4 so both sides of the
boundary are already organized.
**Verify**: reach-in count (grep-countable) drops; macos dispatch
green; paint/bake/undo smoke.

### Phase 6 — robustness (~1–2 days; items independent)

Read + validate the sidecar `version` with a migration hook and a
newer-than-me guard (§5); byte-budgeted undo ring (and/or dirty-rect
snapshots) replacing the hardcoded 16 full-res copies, plus undo for
layer add/remove and displacement (§6); consistent error surfacing —
status-bar channel everywhere + `rfd::MessageDialog` for fatal
load/save failures (§7); `read_back_buffer` returns `Result` instead
of panicking on device loss; defuse the two render-path unwraps (§7).

### Phase 7 — consolidation odds and ends

Mesh-ingestion unification toward one loader + one
`compute_vertex_normals` (§21); `assets.rs` `apply_as_*` collapse
(§22); egui-snarl serde decision — persist `MaterialGraph` in the
sidecar as the comment promises, or drop the feature flag (§23).

### Phase 8 — test buildout + tooling maturity (ongoing)

Golden-image CPU bake test on a tiny mesh (finally using
`test_assets/`); baker math tests (normals encode, curvature, height);
WGSL validation (naga CLI) in ci.yml so the 2.7k lines of shaders are
checked before runtime; workspace-wide clippy `-D warnings`; module
`//!` docs for the new `app/`/`viewport/` trees and texture-baker;
`cargo-deny` license/advisory audit; egui 0.31 → 0.33+ lattice bump
(snarl 0.8, wgpu 25 — undoes texture-baker's downgrade pin).

---

## 4. Verification playbook

| Layer | What it proves | How |
| --- | --- | --- |
| ubuntu `ci.yml` (every push/PR, ~2–4 min) | workspace resolves from crates.io; rustfmt; texture-baker clippy `-D warnings` + tests | automatic |
| `macos-build.yml` dispatch on a branch | the full app compiles and links against real OpenUSD; release staging works | Actions tab → run workflow → select branch (~10 min warm cache, ~45+ cold) |
| Dev Mac runtime smoke | painting, baking, Hydra, undo actually behave | open a stage; paint each channel; bake mesh maps; toggle Storm/hdNSI; save + reload sidecar |
| Windows dispatch (`include_hdnsi` as needed) | bundle layout, DLL staging, hdNSI packaging | occasional, before tagging |

Refactor phases (P2+) should end with one macos dispatch minimum;
UI-touching phases (P4/P5) also need the runtime smoke.

## 5. Deferred / opt-in items

- **History purge of the 3Delight zip** (`git filter-repo` + force
  push): invalidates existing clones; coordinate before doing it. Until
  then the blob remains in old commits.
- **`DELIGHT_WINDOWS_ARCHIVE_URL` secret**: must be set in repo
  settings for hosted Windows hdNSI builds now that the in-repo zip is
  gone (tag builds skip hdNSI gracefully without it).
- **Linux as a runtime target**: eframe/wgpu/OpenUSD all support it;
  needs a Linux OpenUSD CI job and someone to exercise it.
- **egui-lattice bump** (see Phase 8) — do as its own change, nothing
  else mixed in.
