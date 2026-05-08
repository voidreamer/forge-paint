//! PARKED — not compiled. See `attic/README.md` for the full rationale.
//!
//! This module is the in-progress Hydra (Storm) preview viewport. It
//! was wired into the app once (View → "Hydra preview" toggle, side
//! panel rendering an RGBA8 frame from `hydra_rs::Renderer`) but the
//! integration was unreliable in practice and complicated app state,
//! so it was lifted out of `src/`. The previous call sites in
//! `src/app.rs` (fields, View-menu checkbox, lazy constructor,
//! `show_hydra_window`) and the `hydra-rs = "0.0.2"` dependency in
//! `Cargo.toml` were removed in the same change.
//!
//! Reviving it (rough recipe):
//!   1. Add `hydra-rs = "0.0.2"` back to `[dependencies]`.
//!   2. Move this file back to `src/hydra_view.rs` and re-add
//!      `mod hydra_view;` in `src/main.rs`.
//!   3. Re-introduce the App fields (`show_hydra_view`, `hydra`,
//!      `hydra_egui_tex`), the View-menu checkbox, and the
//!      `show_hydra_window` side panel.
//!   4. Re-evaluate the caveats below before exposing it again.
//!
//! Original module description follows.
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
//!   - Authored scene lights aren't routed yet — we mirror them from
//!     Rust via `add_distant_light` until upstream lands the bridge.
//!   - Color AOV only; no depth or primId, so picking has to fall
//!     back to forge-paint's existing CPU pick on the wgpu side.
//!   - Single-threaded; never share a `Renderer` across threads.
//!   - Storm requires a GPU; CI without one will fail to render.

use anyhow::Result;
use hydra_rs::Renderer;
use std::path::Path;

/// Per-panel Hydra renderer. Held alongside the wgpu viewport — the
/// caller decides which panel(s) to surface in the UI.
pub struct HydraView {
    renderer: Renderer,
    pub width: u32,
    pub height: u32,
}

impl HydraView {
    /// Open a stage path through Hydra and seed a sensible default
    /// 3-point lighting rig. Caller passes the same `&Path` they'd
    /// pass to `rust_usd::Stage::open` — `forge://…` URIs work
    /// because the resolver is already registered process-wide.
    pub fn new(stage_path: &Path) -> Result<Self> {
        // Diagnostic override: setting HYDRA_TEST_STAGE points the
        // renderer at any USDA you supply (e.g. the bundled
        // hydra-rs/examples/hydra_test.usda — a red sphere with a
        // displayColor primvar). If the test asset renders correctly
        // but the user's stage doesn't, the issue is asset-side, not
        // renderer-side. Drop the env var to go back to the live
        // forge-paint stage path.
        let actual = std::env::var_os("HYDRA_TEST_STAGE")
            .map(std::path::PathBuf::from);
        let path: &Path = actual.as_deref().unwrap_or(stage_path);
        log::info!("HydraView::new opening: {}", path.display());
        let mut renderer = Renderer::new(path)?;
        renderer.set_size(1280, 720);

        // hydra-rs lighting model (verified by reading the C++ bridge):
        //   - `clear_lights()` sets `enableLighting = false` AND wipes
        //     the light list. Anything you add afterwards lives on the
        //     Rust-side vector but the engine still renders unlit
        //     because the gating bool is off.
        //   - `use_default_light()` clears the list AND turns the
        //     bool back on, then injects the default headlight.
        //   - `add_distant_light` / `add_positional_light` populate
        //     the legacy GlfSimpleLight pipeline. Authored UsdLux
        //     lights in the stage are NOT routed (enableSceneLights
        //     is hard-coded false in the bridge).
        //
        // Default headlight = correct PBR lighting of the camera-
        // facing surfaces, no convention guessing required.
        renderer.use_default_light();
        // Diagnostic magenta clear so empty-frustum / black-shading
        // are visually distinguishable. Once we have lighting working
        // we'll go back to a neutral grey or pull from forge-paint's
        // background.
        renderer.set_clear_color([1.0, 0.0, 1.0, 1.0]);

        Ok(Self {
            renderer,
            width: 1280,
            height: 720,
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        self.renderer.set_size(w, h);
    }

    /// Render one frame. `view` and `proj` are 4×4 matrices in *row*-
    /// vector convention — Hydra's `GfMatrix4d` expects
    /// `vec_world * view = vec_camera`. forge-paint's `OrbitCamera`
    /// produces column-vector matrices, so we transpose at the
    /// caller's boundary before passing them in.
    ///
    /// Returns a tightly packed RGBA8 buffer of size
    /// `width * height * 4`. Hand it to whatever texture-upload path
    /// the UI uses (egui `ColorImage`, wgpu `write_texture`, …).
    pub fn render(&mut self, view: &[f32; 16], proj: &[f32; 16]) -> Result<Vec<u8>> {
        self.renderer.set_camera_matrices(view, proj);
        Ok(self.renderer.render()?)
    }

    /// Replace the lighting rig wholesale. Caller is expected to
    /// re-add lights afterwards via `add_distant_light` etc.
    #[allow(dead_code)]
    pub fn clear_lights(&mut self) {
        self.renderer.clear_lights();
    }

    #[allow(dead_code)]
    pub fn add_distant_light(&mut self, dir: [f32; 3], color: [f32; 3], intensity: f32) {
        self.renderer.add_distant_light(dir, color, intensity);
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
