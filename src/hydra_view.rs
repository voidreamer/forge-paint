//! Hydra (Storm) preview viewport.
//!
//! A second viewport panel that renders the active USD stage through
//! `hydra_rs::Renderer`, in parallel with the wgpu painter. The
//! painting pipeline (brush, composite, smart masks) doesn't change —
//! Hydra is the "production reference" view: how does this asset
//! actually look in a Storm-based viewport?
//!
//! Lifecycle: construct once per stage, mutate camera / size /
//! lights between frames, call `render()` to get a tightly-packed
//! RGBA8 buffer that gets uploaded into an egui texture for display.
//!
//! Caveats (per upstream notes):
//!   - Authored UsdLux scene lights aren't routed through hydra-rs
//!     yet — we mirror the wgpu studio rig from Rust via
//!     `add_distant_light`. Materials still flow through:
//!     `enableSceneMaterials = true` in the C++ bridge.
//!   - Color AOV only; no depth or primId, so picking has to fall
//!     back to forge-paint's existing CPU pick on the wgpu side.
//!   - Single-threaded; never share a `Renderer` across threads.
//!   - Storm requires a GPU; CI without one will fail to render.

use anyhow::Result;
use hydra_rs::Renderer;
use std::path::Path;

/// Snapshot of the wgpu side's three-point studio rig. Each frame the
/// caller copies these from `Viewport` and hands them to
/// `HydraView::set_rig` so Hydra mirrors the same key/fill/rim that
/// wgpu's shader uses. Convention matches `Viewport::light_dir`:
/// `key_dir` is the direction the key light TRAVELS, not the direction
/// TO the light — `set_rig` negates internally to fit Hydra's
/// `GlfSimpleLight` (position with w=0 is the direction from origin
/// to the light source).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudioRig {
    pub key_dir: [f32; 3],
    pub key_intensity: f32,
    pub fill_ratio: f32,
    pub rim_ratio: f32,
    /// Studio rig off → just the key light, no fill/rim. Matches the
    /// wgpu side's `studio_rig_enabled` toggle.
    pub enabled: bool,
}

/// Snapshot of the wgpu side's HDRI environment, packaged for the
/// Hydra dome. `path` is the on-disk HDRI file (or any URI the USD
/// resolver can chase); `intensity`, `exposure_stops`, and
/// `rotation_y_radians` mirror the wgpu shader's env state plus the
/// viewport's exposure slider. `None` → no dome (procedural sky on
/// the wgpu side, nothing on the Hydra side — clear-colour shows
/// through). UsdLux combines intensity and exposure multiplicatively
/// (`final = intensity * 2^exposure`), so handing the slider's stops
/// straight to the dome's `inputs:exposure` attr lines up.
#[derive(Debug, Clone, PartialEq)]
pub struct DomeEnv {
    pub path: std::path::PathBuf,
    pub intensity: f32,
    pub exposure_stops: f32,
    pub rotation_y_radians: f32,
}

/// Per-panel Hydra renderer. Held alongside the wgpu viewport — the
/// caller decides which panel(s) to surface in the UI.
pub struct HydraView {
    renderer: Renderer,
    pub width: u32,
    pub height: u32,
    /// Stage path that was passed to `HydraView::new`. The C++ Renderer
    /// holds an internal `UsdStageRefPtr` baked in at construction, so
    /// loading a different USD file means the consumer has to drop the
    /// whole `HydraView` and lazy-init a new one. Exposed so the
    /// caller can detect the mismatch without bookkeeping of its own.
    pub stage_path: std::path::PathBuf,

    // Dirty-tracking shadows. The bridge's `set_*` methods write into
    // the USD session layer (for dome / painted material) or push
    // into the legacy lighting context (for the rig); every push
    // triggers a scene-index notification, which the delegate burns
    // CPU on its next Sync. Idempotent in value but never free.
    // Comparing here against the last-applied snapshot keeps the
    // per-frame setters as no-ops when nothing actually changed,
    // which makes zoom / orbit feel smooth on the path-tracer path.
    last_rig: Option<StudioRig>,
    last_dome: Option<DomeEnv>,
    last_purposes: Option<(bool, bool, bool)>,
    /// Packed snapshot of the user-light array that was last pushed
    /// through `set_user_lights`. Compared against on each frame to
    /// skip the session-layer rewrite when nothing changed —
    /// without this, every frame would Define a fresh
    /// `UsdLuxDistantLight` / `UsdLuxSphereLight` and trigger a
    /// scene-index notification on the delegate. Hashed-equivalent
    /// (bytewise) because lights pack to plain `f32`s.
    last_user_lights: Vec<[f32; 16]>,
    /// Library material currently referenced into the session layer
    /// (source path + sub-prim path). `None` means no library
    /// material is bound. Compared against to skip bridge calls
    /// when the selection hasn't changed.
    last_external_material: Option<(std::path::PathBuf, String)>,
    /// Last target-prims list authored via
    /// `set_external_material_on_prims`. Empty == stage-wide, which
    /// is also the legacy `set_external_material` semantics, so a
    /// switch between the two re-authors correctly.
    last_target_prims: Vec<String>,

    /// Wall-clock timestamp of the most-recent successful
    /// `render(...)` call. The caller throttles re-render frequency
    /// against this so a path-tracer delegate doesn't peg the UI
    /// thread doing one sample pass per egui frame.
    pub last_render: Option<std::time::Instant>,
    /// Latest RGBA8 buffer returned by `render(...)`. Cached so the
    /// caller can re-display it on frames that don't trigger a new
    /// render (during the throttle window, while is_converged stays
    /// true, etc.). Tuple is `(pixels, width, height)`.
    pub last_pixels: Option<(Vec<u8>, u32, u32)>,
}

impl HydraView {
    /// Open a stage path through Hydra. Caller passes the same `&Path`
    /// they'd pass to `rust_usd::Stage::open` — `forge://…` URIs work
    /// because the resolver is already registered process-wide.
    ///
    /// Lighting is left empty here; the caller is expected to call
    /// `set_rig` before the first `render()` so the wgpu studio rig
    /// drives Hydra too.
    pub fn new(stage_path: &Path) -> Result<Self> {
        // Diagnostic override: setting HYDRA_TEST_STAGE points the
        // renderer at any USDA you supply (e.g. the bundled
        // hydra-rs/examples/hydra_test.usda — a red sphere with a
        // displayColor primvar). If the test asset renders correctly
        // but the user's stage doesn't, the issue is asset-side, not
        // renderer-side. Drop the env var to go back to the live
        // forge-paint stage path.
        let actual = std::env::var_os("HYDRA_TEST_STAGE").map(std::path::PathBuf::from);
        let path: &Path = actual.as_deref().unwrap_or(stage_path);
        log::info!("HydraView::new opening: {}", path.display());
        let mut renderer = Renderer::new(path)?;
        renderer.set_size(1280, 720);

        // Neutral dark backdrop for when no dome is bound — matches
        // the wgpu central panel's `from_rgb(18, 18, 22)`. Once a
        // `DomeEnv` is set via `set_environment`, the dome draws
        // behind the geometry and the clear colour is only visible
        // through any holes in coverage.
        renderer.set_clear_color([0.02, 0.02, 0.025, 1.0]);

        // Fall-back scene ambient for the no-dome case. With the dome
        // bound Storm gets full IBL out of the env-map importance
        // sample, so this ambient becomes a small residual; without
        // the dome it's the only thing keeping the shadow side off
        // pure black. Linear-space, sky-tinted to match the wgpu
        // shader's ambient term.
        renderer.set_scene_ambient([0.20, 0.24, 0.30, 1.0]);

        Ok(Self {
            renderer,
            width: 1280,
            height: 720,
            stage_path: path.to_path_buf(),
            last_rig: None,
            last_dome: None,
            last_purposes: None,
            last_user_lights: Vec::new(),
            last_external_material: None,
            last_target_prims: Vec::new(),
            last_render: None,
            last_pixels: None,
        })
    }

    /// Plugin IDs of every Hydra render delegate the USD plug
    /// registry found at startup. Storm is always there
    /// (`HdStormRendererPlugin`); production delegates show up only
    /// when their anvil packages compose into the run — `3delight`
    /// adds `HdNSiRendererPlugin`, etc.
    pub fn list_delegates() -> Vec<String> {
        hydra_rs::list_render_delegates()
    }

    /// Plugin ID currently driving `render()`. Empty string if the
    /// renderer hasn't picked one yet (shouldn't happen post-`new`).
    pub fn current_delegate(&self) -> String {
        self.renderer.current_renderer()
    }

    /// Swap the active Hydra render delegate. The engine keeps the
    /// stage, scene index, camera, light rig, and dome across the
    /// swap — only the rasteriser / path-tracer behind them changes.
    /// Returns `Err` when `plugin_id` isn't a registered delegate.
    pub fn set_delegate(&mut self, plugin_id: &str) -> Result<()> {
        if !self.renderer.set_renderer_plugin(plugin_id) {
            anyhow::bail!(
                "Hydra render delegate not registered: {plugin_id}. \
                 Available: {:?}",
                Self::list_delegates()
            );
        }
        Ok(())
    }

    /// Author a `UsdPreviewSurface` material into the stage's session
    /// layer pointing at on-disk PNGs for the four PBR channels, and
    /// bind it to every mesh. Each path can use the `<UDIM>` token so
    /// a single material spans multi-tile UDIM layouts. Empty paths
    /// skip that channel (the corresponding PBR input keeps its
    /// UsdPreviewSurface default).
    ///
    /// Forwards straight to `hydra_rs::Renderer::set_painted_material`.
    /// See the C++ bridge header for the source-colour-space split
    /// (sRGB for base colour, raw for roughness/metallic/normal).
    pub fn set_painted_material(
        &mut self,
        base_color: &Path,
        roughness: &Path,
        metallic: &Path,
        normal: &Path,
    ) -> Result<()> {
        self.renderer
            .set_painted_material(base_color, roughness, metallic, normal)
            .map_err(|e| anyhow::anyhow!("set_painted_material failed: {}", e.what()))
    }

    /// Drop the painted material authored by `set_painted_material`
    /// and unbind it from every mesh. The stage's originally-authored
    /// bindings (if any) drive shading again afterwards.
    pub fn clear_painted_material(&mut self) {
        self.renderer.clear_painted_material();
    }

    /// Reference a library material into the stage's session layer
    /// and bind it to every mesh. `source` is the USD file from the
    /// `Materials` library pane; `prim_path` is the path inside that
    /// file (empty = source's default prim, which is the convention
    /// the discovery walker expects). Hydra-only — wgpu mode keeps
    /// rendering paint targets.
    ///
    /// Idempotent: re-pointing at a different material is one bridge
    /// call. Dirty-tracking lives in the caller (`draw_hydra_central`
    /// only invokes this when the selected index changes).
    pub fn set_external_material(&mut self, source: &Path, prim_path: &str) -> Result<()> {
        let next = (source.to_path_buf(), prim_path.to_string());
        if self.last_external_material.as_ref() == Some(&next) {
            return Ok(());
        }
        self.renderer
            .set_external_material(source, prim_path)
            .map_err(|e| anyhow::anyhow!("set_external_material failed: {}", e.what()))?;
        self.last_external_material = Some(next);
        Ok(())
    }

    /// Per-prim variant of `set_external_material`. Empty
    /// `target_prims` ⇒ stage-wide (same as the legacy method).
    /// Non-empty ⇒ bind only to the listed SdfPaths; Xform / Scope
    /// entries cascade to descendant Mesh prims hydra-rs-side.
    pub fn set_external_material_on_prims(
        &mut self,
        source: &Path,
        prim_path: &str,
        target_prims: &[String],
    ) -> Result<()> {
        // Skip the redundant authoring if neither source nor scope
        // has moved. Tracked separately from last_external_material
        // so a switch between stage-wide and per-prim still re-fires.
        let src_next = (source.to_path_buf(), prim_path.to_string());
        let same_src = self.last_external_material.as_ref() == Some(&src_next);
        let same_scope = self.last_target_prims.as_slice() == target_prims;
        if same_src && same_scope {
            return Ok(());
        }
        self.renderer
            .set_external_material_on_prims(source, prim_path, target_prims)
            .map_err(|e| anyhow::anyhow!("set_external_material_on_prims failed: {}", e.what()))?;
        self.last_external_material = Some(src_next);
        self.last_target_prims = target_prims.to_vec();
        Ok(())
    }

    /// Drop the library material reference and unbind from meshes.
    pub fn clear_external_material(&mut self) {
        if self.last_external_material.is_none() {
            return;
        }
        self.renderer.clear_external_material();
        self.last_external_material = None;
    }

    /// Live-edit a scalar input on the currently-bound external
    /// material's surface shader (e.g. `metallic`, `roughness`,
    /// `opacity`). Forwards to `hydra-rs::set_external_material_
    /// input_f` which authors a session-layer override on top of
    /// the referenced source value.
    pub fn set_external_material_input_f(&mut self, input_name: &str, value: f32) {
        self.renderer
            .set_external_material_input_f(input_name, value);
    }

    /// Live-edit a colour input on the currently-bound external
    /// material's surface shader (e.g. `diffuseColor`, `base_color`,
    /// `emissiveColor`).
    pub fn set_external_material_input_color3(&mut self, input_name: &str, color: [f32; 3]) {
        self.renderer
            .set_external_material_input_color3(input_name, color);
    }

    /// Live-edit an int input on the currently-bound external
    /// material's surface shader. Used for OSL bool-flavoured
    /// toggles (`coating_on`, `sss_on`, …) which gate downstream
    /// layer contribution and can't be set through the float setter.
    pub fn set_external_material_input_i(&mut self, input_name: &str, value: i32) {
        self.renderer
            .set_external_material_input_i(input_name, value);
    }

    /// Push Storm's selection-highlight set. Storm bakes the
    /// outline into the color AOV via HdxColorizeSelectionTask
    /// (only active when `color` is registered as the viewport
    /// AOV, which hydra-rs does at construction).
    pub fn set_selection<S: AsRef<str>>(&mut self, paths: &[S]) {
        self.renderer.set_selection(paths);
    }

    pub fn set_selection_color(&mut self, color: [f32; 4]) {
        self.renderer.set_selection_color(color);
    }

    /// Concurrent multi-material binding (C2b) — apply or update
    /// the binding identified by `binding_id`. Idempotent; the bridge
    /// re-authors only the affected SdfPaths.
    pub fn apply_material_binding(
        &mut self,
        binding_id: u64,
        source: &Path,
        prim_path: &str,
        target_prims: &[String],
    ) -> Result<()> {
        self.renderer
            .apply_material_binding(binding_id, source, prim_path, target_prims)
            .map_err(|e| anyhow::anyhow!("apply_material_binding failed: {}", e.what()))
    }

    pub fn remove_material_binding(&mut self, binding_id: u64) {
        self.renderer.remove_material_binding(binding_id);
    }

    pub fn clear_all_material_bindings(&mut self) {
        self.renderer.clear_all_material_bindings();
    }

    pub fn set_binding_input_f(&mut self, binding_id: u64, input_name: &str, value: f32) {
        self.renderer
            .set_binding_input_f(binding_id, input_name, value);
    }

    pub fn set_binding_input_color3(&mut self, binding_id: u64, input_name: &str, color: [f32; 3]) {
        self.renderer
            .set_binding_input_color3(binding_id, input_name, color);
    }

    pub fn set_binding_input_i(&mut self, binding_id: u64, input_name: &str, value: i32) {
        self.renderer
            .set_binding_input_i(binding_id, input_name, value);
    }

    /// Push the consumer's analytic-light list into the Hydra
    /// session layer. Each `Light` becomes either a
    /// `UsdLuxDistantLight` (directional) or a `UsdLuxSphereLight`
    /// with `shaping:cone:*` (spot) at `/_hydraLight<i>`.
    ///
    /// Dirty-tracked — the bridge call is skipped when the packed
    /// payload matches the last one we sent. Each session-layer
    /// edit triggers a delegate Sync (path-tracer restart on
    /// hdNSI), so the early-out matters when the panel is idle.
    pub fn set_user_lights(&mut self, lights: &[crate::lights::Light]) -> Result<()> {
        let packed: Vec<[f32; 16]> = lights
            .iter()
            .map(|l| {
                let gl = crate::lights::GpuLight::from_light(l);
                let mut row = [0.0f32; 16];
                row[0..4].copy_from_slice(&gl.direction_type);
                row[4..8].copy_from_slice(&gl.position_enabled);
                row[8..12].copy_from_slice(&gl.color_intensity);
                row[12..16].copy_from_slice(&gl.cone);
                row
            })
            .collect();
        if packed == self.last_user_lights {
            return Ok(());
        }
        log::info!(
            "Hydra: pushing {} user light(s) to bridge ({} prev)",
            packed.len(),
            self.last_user_lights.len(),
        );
        self.renderer
            .set_user_lights(&packed)
            .map_err(|e| anyhow::anyhow!("set_user_lights failed: {}", e.what()))?;
        self.last_user_lights = packed;
        Ok(())
    }

    /// Apply `UsdGeomImageable::purpose` filters in one call. Cheap
    /// enough to send every frame from the consumer; the bridge just
    /// stores three bools and reads them inside `render_color`.
    pub fn set_purposes(&mut self, render: bool, proxy: bool, guides: bool) {
        // Dirty-check: cheaper than the dome auth (just three bool
        // writes inside the bridge struct), but it still triggers a
        // resync on the delegate side because purpose filters are
        // part of render params. No reason to do it every frame.
        let new = (render, proxy, guides);
        if self.last_purposes == Some(new) {
            return;
        }
        self.last_purposes = Some(new);
        self.renderer.set_show_render(render);
        self.renderer.set_show_proxy(proxy);
        self.renderer.set_show_guides(guides);
    }

    /// Author a `UsdLuxDomeLight` into the stage's session layer (or
    /// remove it, if `env` is `None`). Storm renders the dome as both
    /// the visible skybox and an IBL source, so calling this routes
    /// the wgpu HDRI into Hydra for free — no separate background
    /// task, no manual env-map importance sampling.
    ///
    /// Cheap enough to call every frame: the bridge re-uses the same
    /// session-layer prim path and just rewrites the attributes,
    /// rather than tearing it down and rebuilding.
    pub fn set_environment(&mut self, env: Option<&DomeEnv>) -> Result<()> {
        // Dirty-check: dome auth writes into the session layer, which
        // triggers a scene-index notification → delegate Sync → path
        // tracer restart. Doing that every frame even when intensity
        // / rotation / path haven't changed is what made zoom feel
        // laggy. Compare against last-applied snapshot first.
        if self.last_dome.as_ref() == env {
            return Ok(());
        }
        match env {
            Some(e) => {
                let degrees = e.rotation_y_radians.to_degrees();
                // env_intensity → UsdLux `inputs:intensity` (linear
                // multiplier). `exposure_stops` (the viewport's
                // exposure slider, in stops) → UsdLux
                // `inputs:exposure`. UsdLux multiplies them as
                // `final = intensity * 2^exposure`, which matches
                // the wgpu shader's `intensity * 2^exposure_stops`
                // — same value flows into both panels.
                self.renderer
                    .set_dome_light(&e.path, e.intensity, e.exposure_stops, degrees)?;
            }
            None => {
                self.renderer.clear_dome_light();
            }
        }
        self.last_dome = env.cloned();
        Ok(())
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        self.renderer.set_size(w, h);
    }

    /// Mirror the wgpu studio rig in Hydra. Cheap enough to call every
    /// frame: clears the C++-side light vector and re-adds 1-3 lights.
    /// `key_dir` is in wgpu convention (direction the light travels),
    /// negated here before being passed to `add_distant_light` because
    /// `GlfSimpleLight` reads position(w=0) as the direction from the
    /// shaded surface TO the light source.
    ///
    /// Mirrors the wgpu fill/rim derivation:
    ///   fill = (-key.x,  key.y,             -key.z)
    ///   rim  = (-key.x,  max(key.y, 0.2),   -key.z)
    /// and the studio color tints + intensity ratios — see
    /// `viewport::Viewport::show` for the canonical values.
    pub fn set_rig(&mut self, rig: StudioRig) {
        // Dirty-check: rig values get snapshotted from `Viewport`
        // every frame even when the user hasn't touched a slider, so
        // skip the push when the rig is identical to last frame.
        // Each `clear_lights` + `add_distant_light` triplet pushes
        // into the legacy `GlfSimpleLighting` context — non-zero work
        // + triggers a Sync, which on a path tracer means a fresh
        // sample cycle.
        if self.last_rig.as_ref() == Some(&rig) {
            return;
        }
        self.last_rig = Some(rig);

        // clear_lights() drops the explicit list AND turns off the
        // default-headlight gating bool. That's fine — Storm's
        // render path computes `enableLighting = !lights.empty() ||
        // use_default_lighting`, so the moment we push a single
        // explicit light below the rig becomes active.
        self.renderer.clear_lights();

        // wgpu's key/fill/rim derivation — kept verbatim so the two
        // viewports light the same surfaces from the same angles.
        let key = rig.key_dir;
        let key_color = [1.0, 0.98, 0.95];
        // Direction TO the light = -(direction of travel).
        let to_light = |d: [f32; 3]| [-d[0], -d[1], -d[2]];

        self.renderer
            .add_distant_light(to_light(key), key_color, rig.key_intensity);

        if rig.enabled {
            let fill = [-key[0], key[1], -key[2]];
            let rim = [-key[0], key[1].max(0.2), -key[2]];
            let fill_color = [0.78, 0.86, 1.00];
            let rim_color = [1.00, 0.92, 0.80];
            self.renderer.add_distant_light(
                to_light(fill),
                fill_color,
                rig.key_intensity * rig.fill_ratio,
            );
            self.renderer.add_distant_light(
                to_light(rim),
                rim_color,
                rig.key_intensity * rig.rim_ratio,
            );
        }
    }

    /// Render one frame. `view` and `proj` are 4×4 matrices in *row*-
    /// vector convention — Hydra's `GfMatrix4d` expects
    /// `vec_world * view = vec_camera`. forge-paint's `OrbitCamera`
    /// produces column-vector matrices, so we transpose at the
    /// caller's boundary before passing them in.
    ///
    /// Returns a tightly packed top-down RGBA8 buffer (origin at the
    /// top-left) of size `width * height * 4`. Hand it to whatever
    /// texture-upload path the UI uses (egui `ColorImage`, wgpu
    /// `write_texture`, …). hydra-rs 0.0.3+ reverses Storm's
    /// GL-bottom-up AOV inside the C++ bridge, so no flip needed here.
    pub fn render(&mut self, view: &[f32; 16], proj: &[f32; 16]) -> Result<Vec<u8>> {
        self.renderer.set_camera_matrices(view, proj);
        Ok(self.renderer.render()?)
    }

    /// Whether the active delegate considers the current frame fully
    /// resolved. Storm flips to true after the first pass; sampling
    /// delegates (hdNSI etc.) flip true once their sample budget is
    /// hit. The viewport loop uses this to decide whether to request
    /// another repaint after the throttle window expires — once
    /// converged, no point burning CPU.
    pub fn is_converged(&self) -> bool {
        self.renderer.is_converged()
    }
}

/// Map a Hydra delegate plugin ID to the short label we show in the
/// UI. Falls back to the raw ID when no mapping exists, so an
/// unrecognised delegate (custom in-house plugin, beta build) still
/// reads sensibly in the combo box rather than appearing as an empty
/// row.
pub fn delegate_label(plugin_id: &str) -> &str {
    match plugin_id {
        "HdStormRendererPlugin" => "Storm",
        // hdNSI registers as `HdNSIRendererPlugin` with three caps —
        // NSI is the acronym (Nodal Scene Interface). Easy typo.
        "HdNSIRendererPlugin" => "3Delight",
        "HdArnoldRendererPlugin" => "Arnold",
        "HdPrmanLoaderRendererPlugin" | "HdPrmanRendererPlugin" => "RenderMan",
        "HdCyclesRendererPlugin" => "Cycles",
        "HdEmbreeRendererPlugin" => "Embree",
        other => other,
    }
}

/// Convert a glam Mat4 into the flat row-major array Hydra expects.
///
/// glam: column-vector convention, column-major storage. cols[c][r] is
/// the matrix element at math row r, column c.
/// Hydra: row-vector convention, row-major storage. flat[r*4+c] is the
/// matrix element at math row r, column c.
///
/// To go from glam's column-vector to Hydra's row-vector convention we
/// need a math transpose (V_row = V_col.T). Combining with the storage
/// swap (column-major → row-major, also a transpose), the two cancel,
/// and we end up writing `out[r*4 + c] = cols[r][c]`. That gives Hydra
/// `V_col` interpreted as a row-vector matrix, which is `V_col.T` in
/// terms of math behavior — exactly the row-vector view matrix we want.
///
/// Tried the other flatten (`m[c][r]`) — that version made the model
/// invisible / wildly mis-sized, confirming this is the correct one.
pub fn glam_to_hydra(m: &[[f32; 4]; 4]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = m[r][c];
        }
    }
    out
}

/// Build an OpenGL-style perspective projection (clip Z range [-1, 1])
/// in the row-major form Hydra expects, directly from camera params.
/// `Mat4::perspective_rh` (used by `OrbitCamera::proj`) emits a wgpu/
/// Vulkan-style projection (clip Z [0, 1]) — Hydra rejects that as
/// "not a valid perspective matrix" because GfCamera assumes GL
/// conventions. We synthesise the GL form here so `OrbitCamera`
/// stays unchanged.
pub fn perspective_for_hydra(fovy_radians: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy_radians / 2.0).tan();
    let nf = 1.0 / (near - far);
    // Row-vector / row-major form. `m[r][c]` reads naturally below.
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][2] = (far + near) * nf;
    m[2][3] = -1.0;
    m[3][2] = 2.0 * far * near * nf;
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = m[r][c];
        }
    }
    out
}
