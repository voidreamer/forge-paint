//! Project sidecar — JSON metadata that round-trips between sessions.
//!
//! Lives alongside the painted-layer PNGs in the work-dir tree
//! (`forge-project.json`). The texture content of paint layers and
//! mesh-map bakes is *not* in the JSON — those stay as binary files
//! next to it. Persistence v1 captures everything *except* the heavy
//! baked GPU textures: HP / cage paths, bake settings, material
//! factors, and per-layer smart-mask params. Re-bake after load.
//!
//! Forward-compat: each version has a `version: u32`. When we add
//! fields we either default-them-on-load (`#[serde(default)]`) or
//! bump the version and write a migration.
//!
//! Files written:
//!   forge-project.json        — this struct
//!   basecolor.<udim>.png      — existing paint sidecars
//!   roughness.<udim>.png
//!   metallic.<udim>.png
//!   normal.<udim>.png

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::assets::MaterialInputs;
use crate::bake::integration::BakeSettings;
use crate::paint::smart_mask::SmartMaskParams;

const SIDECAR_FILENAME: &str = "forge-project.json";
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSidecar {
    pub version: u32,
    #[serde(default)]
    pub bake: BakeSection,
    #[serde(default)]
    pub material: MaterialSection,
    #[serde(default)]
    pub layers: Vec<LayerSection>,
    /// Library material the user bound through the Materials pane
    /// (gallery chip click), plus any per-input slider tweaks. `None`
    /// means no library material is bound — the stage's authored
    /// material (or painted-material fallback) takes over. Restored
    /// at stage load by looking up the matching `MaterialAsset` in
    /// the freshly-scanned library and replaying the inputs through
    /// hydra-rs's `set_external_material*` path on the next frame.
    #[serde(default)]
    pub bound_material: Option<BoundMaterialBinding>,
}

impl Default for ProjectSidecar {
    fn default() -> Self {
        Self {
            version: SCHEMA_VERSION,
            bake: BakeSection::default(),
            material: MaterialSection::default(),
            layers: Vec::new(),
            bound_material: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundMaterialBinding {
    /// Library .usd / .usda / .usdc the binding references. Absolute
    /// path so the binding survives running forge-paint from a
    /// different cwd. If the file has moved by the time the sidecar
    /// is reloaded, the binding is dropped (with a warning) rather
    /// than crashing.
    pub source: PathBuf,
    /// `defaultPrim` (or explicit prim path within `source`) — the
    /// `Material` prim hydra-rs's `set_external_material` references.
    pub prim_path: String,
    /// Snapshot of the live editor inputs at save time. Replayed
    /// through `set_external_material_input_*` on next-frame draw.
    pub inputs: MaterialInputs,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BakeSection {
    /// Absolute path to the high-poly source. Re-loaded on project
    /// open if the file still exists.
    #[serde(default)]
    pub high_poly_path: Option<PathBuf>,
    /// Absolute path to the cage source. Same lifetime as high-poly.
    #[serde(default)]
    pub cage_path: Option<PathBuf>,
    /// Bake-settings snapshot (ray counts, AA, GPU toggle, normal
    /// convention). Defaults applied on load if absent.
    #[serde(default)]
    pub settings: Option<BakeSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialSection {
    pub base_color_factor: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub displacement_scale: f32,
    pub baked_normal_blend: f32,
    pub ao_intensity: f32,
}

impl Default for MaterialSection {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0],
            metallic: 1.0,
            roughness: 1.0,
            normal_scale: 1.0,
            displacement_scale: 0.0,
            baked_normal_blend: 0.0,
            ao_intensity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerSection {
    /// Per-layer smart-mask params, when present. The mask texture
    /// content itself isn't persisted in v1 — caller calls
    /// `regenerate_smart_mask()` after load to repopulate it.
    #[serde(default)]
    pub smart_mask: Option<SmartMaskParams>,
}

/// Sidecar JSON path inside `work_dir`.
pub fn sidecar_path(work_dir: &Path) -> PathBuf {
    work_dir.join(SIDECAR_FILENAME)
}

/// Read `forge-project.json` if it exists. Returns `Ok(None)` when
/// the file is absent — a fresh project is the default state.
pub fn load_sidecar(work_dir: &Path) -> Result<Option<ProjectSidecar>> {
    let path = sidecar_path(work_dir);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let parsed: ProjectSidecar = serde_json::from_str(&text)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(parsed))
}

/// Write `forge-project.json`, creating the directory if needed.
pub fn save_sidecar(work_dir: &Path, sidecar: &ProjectSidecar) -> Result<()> {
    std::fs::create_dir_all(work_dir)
        .with_context(|| format!("create {}", work_dir.display()))?;
    let path = sidecar_path(work_dir);
    let text = serde_json::to_string_pretty(sidecar)
        .context("serialize sidecar")?;
    std::fs::write(&path, text)
        .with_context(|| format!("write {}", path.display()))?;
    log::info!("project sidecar saved: {}", path.display());
    Ok(())
}
