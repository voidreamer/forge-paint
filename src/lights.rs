//! Dynamic analytic-light list. Replaces the hard-coded 3-point
//! studio rig. The user adds/removes lights from the Lighting panel;
//! `pbr.wgsl` loops over `MAX_LIGHTS` entries each frame.
//!
//! Both renderers should eventually read from this list:
//!   * wgpu: serialised into `FrameUniforms.lights[]` and consumed
//!     by the PBR fragment shader's lighting loop.
//!   * Hydra (planned, second stage): each `Light` gets authored as
//!     a `UsdLuxDistantLight` (directional) or `UsdLuxSphereLight`
//!     with a shaping cone (spot) into the stage's session layer.
//!     Hydra delegates already consume UsdLux via the dome path, so
//!     this drops in cleanly.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Max simultaneous lights the wgpu shader's uniform buffer can carry.
/// Eight is plenty for a previewing context — a production asset
/// usually has a key + fill + rim plus an HDRI, well under the cap.
pub const MAX_LIGHTS: usize = 8;

/// Light type. Kept narrow — directional + spot covers most preview
/// needs and matches what we can route through UsdLux cleanly later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightKind {
    /// Infinitely-distant directional light. Defined by direction
    /// alone; position is ignored.
    Directional,
    /// Cone-shaped spot light from a world-space position pointed
    /// along a direction. Inner/outer cone angles taper falloff
    /// inside the cone.
    Spot,
}

impl LightKind {
    pub fn label(self) -> &'static str {
        match self {
            LightKind::Directional => "Directional",
            LightKind::Spot => "Spot",
        }
    }
}

/// One analytic light authored by the user.
///
/// Fields cover both light types — `position` / cone angles are
/// ignored for `Directional`, `direction` is the only directional
/// input for that type. Keeping a single struct (vs. per-variant
/// enum payload) simplifies serialisation, GPU packing, and the
/// per-light UI form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Light {
    pub kind: LightKind,
    /// Travel direction. Normalised before being sent to the GPU.
    /// For `Directional`: light comes from `-direction`, hits the
    /// model travelling along `direction`. For `Spot`: cone-axis
    /// direction from `position` outward.
    pub direction: [f32; 3],
    /// World-space position. `Spot` only — `Directional` ignores.
    pub position: [f32; 3],
    /// Linear-space RGB tint. Multiplied by `intensity` to get the
    /// radiometric energy. Default white.
    pub color: [f32; 3],
    /// Linear multiplier. The wgpu shader applies it on top of
    /// `color`; the eventual Hydra integration maps it to UsdLux's
    /// `inputs:intensity`.
    pub intensity: f32,
    /// `Spot` only — angle (in degrees) of the full-energy core. The
    /// spot's intensity stays full until the angle from the cone
    /// axis exceeds this, then falls off to `outer_cone_deg`.
    pub inner_cone_deg: f32,
    /// `Spot` only — angle (in degrees) beyond which the spot
    /// contributes nothing. Falloff between inner and outer is
    /// smoothstep.
    pub outer_cone_deg: f32,
    /// Per-light visibility toggle. Lets the user keep a light in
    /// the list while temporarily disabling it without losing its
    /// settings. Disabled lights skip the shader's BRDF eval.
    pub enabled: bool,
}

impl Light {
    /// Sensible directional default — points slightly down + forward,
    /// neutral white, intensity 3 (matches the old studio key's
    /// magnitude so the first light a user adds doesn't look dim).
    pub fn new_directional() -> Self {
        Self {
            kind: LightKind::Directional,
            direction: [-0.4, -1.0, -0.3],
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.98, 0.95],
            intensity: 3.0,
            inner_cone_deg: 15.0,
            outer_cone_deg: 30.0,
            enabled: true,
        }
    }

    /// Sensible spot default — origin-ish position pointing toward
    /// the origin, ~30°-wide cone with a soft 15°-30° falloff.
    pub fn new_spot() -> Self {
        Self {
            kind: LightKind::Spot,
            direction: [0.0, -1.0, 0.0],
            position: [0.0, 2.0, 2.0],
            color: [1.0, 1.0, 1.0],
            intensity: 5.0,
            inner_cone_deg: 15.0,
            outer_cone_deg: 30.0,
            enabled: true,
        }
    }
}

/// GPU layout — one row per shader-side `Light`. Four vec4s = 64
/// bytes per light, std140-aligned. Keep the field order in sync
/// with `pbr.wgsl`'s `struct Light`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuLight {
    /// `x,y,z` = travel direction (normalised). `w` = type tag:
    /// 0.0 = directional, 1.0 = spot.
    pub direction_type: [f32; 4],
    /// `x,y,z` = world position (spot only). `w` = `1.0` enabled /
    /// `0.0` disabled.
    pub position_enabled: [f32; 4],
    /// `x,y,z` = linear-space colour. `w` = intensity.
    pub color_intensity: [f32; 4],
    /// Spot cone parameters in cosine-of-half-angle form: shader
    /// compares `dot(spot_axis, -L)` against these directly without
    /// a per-fragment `acos`. `x` = inner (full energy below).
    /// `y` = outer (zero energy above). `z`,`w` = padding.
    pub cone: [f32; 4],
}

impl GpuLight {
    /// Pack a CPU-side `Light` into its GPU representation.
    pub fn from_light(l: &Light) -> Self {
        let dir = normalize(l.direction);
        let type_tag: f32 = match l.kind {
            LightKind::Directional => 0.0,
            LightKind::Spot => 1.0,
        };
        let enabled = if l.enabled { 1.0 } else { 0.0 };
        let inner_cos = (l.inner_cone_deg.to_radians() * 0.5).cos();
        let outer_cos = (l.outer_cone_deg.to_radians() * 0.5).cos();
        Self {
            direction_type: [dir[0], dir[1], dir[2], type_tag],
            position_enabled: [l.position[0], l.position[1], l.position[2], enabled],
            color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
            cone: [inner_cos, outer_cos, 0.0, 0.0],
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if m > 1e-6 {
        [v[0] / m, v[1] / m, v[2] / m]
    } else {
        [0.0, -1.0, 0.0]
    }
}
