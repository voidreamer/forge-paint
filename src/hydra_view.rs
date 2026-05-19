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
#[derive(Debug, Clone, Copy)]
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
/// resolver can chase); `intensity` and `rotation_y_radians` mirror
/// `EnvUniforms` directly. `None` → no dome (procedural sky on the
/// wgpu side, nothing on the Hydra side — clear-colour shows through).
#[derive(Debug, Clone)]
pub struct DomeEnv {
    pub path: std::path::PathBuf,
    pub intensity: f32,
    pub rotation_y_radians: f32,
}

/// Per-panel Hydra renderer. Held alongside the wgpu viewport — the
/// caller decides which panel(s) to surface in the UI.
pub struct HydraView {
    renderer: Renderer,
    pub width: u32,
    pub height: u32,
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
        let actual = std::env::var_os("HYDRA_TEST_STAGE")
            .map(std::path::PathBuf::from);
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
        })
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
        match env {
            Some(e) => {
                let degrees = e.rotation_y_radians.to_degrees();
                // forge-paint's env_intensity slider is a linear
                // multiplier in `[0, 4]`; map it straight to the
                // UsdLux `intensity` attr and leave exposure at 0,
                // since UsdLux combines them multiplicatively (final
                // = intensity * 2^exposure) and the user-facing
                // slider already lives in the intensity dimension.
                self.renderer
                    .set_dome_light(&e.path, e.intensity, 0.0, degrees)?;
            }
            None => {
                self.renderer.clear_dome_light();
            }
        }
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
