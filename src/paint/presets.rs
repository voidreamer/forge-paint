//! Smart-material presets — one-click stack of (fill layer + smart
//! mask) recipes that produce recognisable looks (edge wear, cavity
//! dirt, dust-on-top). Each preset:
//!
//!   1. Adds a fill layer with a preset-specific colour/roughness/metalness.
//!   2. Adds a mask to that layer.
//!   3. Stamps a `SmartMaskParams` config onto the mask (source +
//!      threshold + invert).
//!
//! Regen happens via the standard `Viewport::regenerate_smart_mask`
//! path — which means the required source bake (curvature for edge /
//! cavity, AO for grime, world_normal for dust) must already be
//! available, otherwise the layer renders as fully visible until the
//! user runs the bake. The UI flags this case in the panel.

use crate::paint::layer::FillParams;
use crate::paint::smart_mask::{SmartMaskParams, SmartMaskSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartMaterialPreset {
    /// Bright metal show-through on convex edges. Hits any high-
    /// curvature region — typical "well-worn" prop look.
    EdgeWear,
    /// Dark grime in concave / cavity regions. Pairs naturally with
    /// edge wear above it; together they read as old + handled.
    CavityDirt,
    /// Light dust accumulating on upward-facing surfaces. Good for
    /// horizontal panels, shelves, anything that's been sitting.
    DustOnTop,
}

impl SmartMaterialPreset {
    pub fn label(self) -> &'static str {
        match self {
            SmartMaterialPreset::EdgeWear => "Edge wear",
            SmartMaterialPreset::CavityDirt => "Cavity dirt",
            SmartMaterialPreset::DustOnTop => "Dust on top",
        }
    }

    /// Fill layer colour / roughness / metalness for this preset.
    pub fn fill_params(self) -> FillParams {
        match self {
            // Polished bright metal — silver / nickel-ish. Roughness
            // low enough to read as exposed metal, metalness 1.
            SmartMaterialPreset::EdgeWear => FillParams {
                base_color_srgb: [0.78, 0.80, 0.82],
                roughness: 0.18,
                metallic: 1.0,
                ..FillParams::default()
            },
            // Dark grime — warm brown, fully rough, dielectric.
            SmartMaterialPreset::CavityDirt => FillParams {
                base_color_srgb: [0.20, 0.13, 0.08],
                roughness: 0.95,
                metallic: 0.0,
                ..FillParams::default()
            },
            // Light dust — desaturated cool gray, very rough.
            SmartMaterialPreset::DustOnTop => FillParams {
                base_color_srgb: [0.85, 0.83, 0.78],
                roughness: 0.92,
                metallic: 0.0,
                ..FillParams::default()
            },
        }
    }

    /// Smart-mask config that drives the layer's reveal pattern.
    pub fn smart_mask(self) -> SmartMaskParams {
        match self {
            // Convex curvature, narrow band — only the actual edges
            // show through. low/high tuned so a well-shaped curvature
            // bake gives ~5–15% mask coverage on a typical asset.
            SmartMaterialPreset::EdgeWear => SmartMaskParams {
                source: SmartMaskSource::CurvatureConvex,
                low: 0.25,
                high: 0.55,
                contrast: 1.2,
                invert: false,
            },
            // Concave curvature — broader band so dirt accumulates in
            // every cavity, not just the deepest.
            SmartMaterialPreset::CavityDirt => SmartMaskParams {
                source: SmartMaskSource::CurvatureConcave,
                low: 0.10,
                high: 0.55,
                contrast: 1.0,
                invert: false,
            },
            // World-Y up — only horizontal-ish surfaces. low cuts off
            // anything too vertical; high feathers the band.
            SmartMaterialPreset::DustOnTop => SmartMaskParams {
                source: SmartMaskSource::WorldYUp,
                low: 0.40,
                high: 0.85,
                contrast: 1.1,
                invert: false,
            },
        }
    }

    pub const ALL: &'static [SmartMaterialPreset] = &[
        SmartMaterialPreset::EdgeWear,
        SmartMaterialPreset::CavityDirt,
        SmartMaterialPreset::DustOnTop,
    ];
}
