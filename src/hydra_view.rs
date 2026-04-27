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
        let mut renderer = Renderer::new(stage_path)?;
        renderer.set_size(1280, 720);

        // Default rig — replace via `clear_lights` + per-light add
        // calls once the user surfaces lighting controls. Mirrors the
        // 3-point studio rig forge-paint already uses on the wgpu
        // side (key + cool fill); rim is omitted because Hydra's
        // distant-light support is the immediate API.
        renderer.clear_lights();
        renderer.add_distant_light([1.0, 1.0, 0.5], [1.0, 1.0, 0.95], 1.5);
        renderer.add_distant_light([-1.0, -0.3, 1.0], [0.4, 0.5, 0.8], 0.8);

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

/// Transpose a column-major 4×4 (glam / OpenGL) into a row-major
/// flat array Hydra expects. Used when bridging forge-paint's
/// `OrbitCamera` matrices into the renderer.
pub fn transpose_for_hydra(m: &[[f32; 4]; 4]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            // Source m[col][row] — flatten as row-major.
            out[r * 4 + c] = m[c][r];
        }
    }
    out
}
