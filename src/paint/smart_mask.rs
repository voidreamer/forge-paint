//! Smart-mask source + params. A "smart mask" is a per-layer mask
//! generated procedurally from one or more baked mesh maps (AO,
//! curvature, thickness, world position). When present on a `Mask`,
//! the existing manual paint affordances on that mask are replaced by
//! a regenerator: change a knob and the mask texture gets re-baked
//! from the baked maps.
//!
//! Phase A in this slice is data only — no shader, no UI yet. The
//! Mask struct gains `smart: Option<SmartMaskParams>`; the rest of the
//! pipeline (composite, brush, etc.) sees a regular R8 atlas and
//! doesn't care whether it was painted or generated.

use crate::bake::integration::MapKind;

/// Which baked map drives the mask. Each variant maps to one of the
/// `MeshMaps` slots; the regenerator pulls that slot's R8 / Rgba
/// atlas through a thresholding shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SmartMaskSource {
    /// `1 - smoothstep(low, high, ao)`. Bright in cavities — the
    /// classic dirt / grime accumulation source.
    AoCrevice,
    /// Curvature shading > 0.5 means convex edge. Useful for edge wear
    /// and scratch effects on the silhouette.
    CurvatureConvex,
    /// Curvature < 0.5 means concave / cavity. Paint chips, cracks,
    /// dust at the bottom of grooves.
    CurvatureConcave,
    /// Smoothstep over thickness — bright at thin features. Gives an
    /// SSS-like falloff, useful for fabric edges or wax glow hints.
    Thickness,
    /// World-space Y axis dominance. Bright where the surface faces
    /// up — drives "dust on top", "drips run downward".
    WorldYUp,
}

impl SmartMaskSource {
    pub fn label(self) -> &'static str {
        match self {
            SmartMaskSource::AoCrevice => "AO crevice",
            SmartMaskSource::CurvatureConvex => "Curvature edge (convex)",
            SmartMaskSource::CurvatureConcave => "Curvature cavity (concave)",
            SmartMaskSource::Thickness => "Thickness",
            SmartMaskSource::WorldYUp => "World Y up",
        }
    }

    /// Which `MeshMaps` slot this source needs baked. UI uses this to
    /// gray out sources whose source map isn't available yet.
    pub fn required_map(self) -> MapKind {
        match self {
            SmartMaskSource::AoCrevice => MapKind::AmbientOcclusion,
            SmartMaskSource::CurvatureConvex | SmartMaskSource::CurvatureConcave => {
                MapKind::Curvature
            }
            SmartMaskSource::Thickness => MapKind::Thickness,
            // World-Y dominance reads the world-normal map, which lives
            // outside the texture-baker slots — we'll route it through
            // the existing world_normal MRT bake at regeneration time.
            SmartMaskSource::WorldYUp => MapKind::WorldNormal,
        }
    }

    pub const ALL: &'static [SmartMaskSource] = &[
        SmartMaskSource::AoCrevice,
        SmartMaskSource::CurvatureConvex,
        SmartMaskSource::CurvatureConcave,
        SmartMaskSource::Thickness,
        SmartMaskSource::WorldYUp,
    ];
}

impl Default for SmartMaskSource {
    fn default() -> Self {
        SmartMaskSource::AoCrevice
    }
}

/// Knobs the regenerator uses to turn a baked map into an R8 mask.
///
/// `low` / `high` form a soft step: values below `low` map to 0,
/// above `high` map to 1, in-between get `smoothstep`-interpolated.
/// `contrast` post-multiplies the falloff. `invert` swaps 0/1 — easy
/// way to switch "AO crevice" between "dirt in crevices" and "edge
/// highlights on protrusions" without changing the source.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SmartMaskParams {
    pub source: SmartMaskSource,
    pub low: f32,
    pub high: f32,
    pub contrast: f32,
    pub invert: bool,
}

impl Default for SmartMaskParams {
    fn default() -> Self {
        Self {
            source: SmartMaskSource::default(),
            low: 0.30,
            high: 0.70,
            contrast: 1.0,
            invert: false,
        }
    }
}
