//! Sidecar save/load of painted texture layers.
//!
//! Default work dir resolution:
//!     $FORGE_PAINT_WORK_DIR/<usd_stem>/
//! or, if that env var isn't set:
//!     <usd_parent>/forge-paint/<usd_stem>/
//!
//! Files are standard PNGs with the same naming convention `export` writes:
//! `<channel>.<udim>.png`. Upload on load uses `queue.write_texture` directly
//! into the right layer of the paint target (GPU-side — no host buffer beyond
//! the decoded PNG).

use anyhow::{Context, Result};
use egui_wgpu::wgpu;
use std::path::{Path, PathBuf};

use crate::paint::{Layer, PaintTarget};

const LAYER_CHANNELS: &[(&str, fn(&Layer) -> &wgpu::Texture)] = &[
    ("basecolor", |l| &l.base_color),
    ("roughmetal", |l| &l.rough_metal),
    ("normal", |l| &l.normal),
];

/// Work directory for persistence, honoring `FORGE_PAINT_WORK_DIR`. If the
/// env var isn't set, colocate a `forge-paint/<stem>/` folder next to the USD.
pub fn default_work_dir(usd_path: &Path) -> PathBuf {
    let stem = usd_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");
    if let Some(root) = std::env::var_os("FORGE_PAINT_WORK_DIR") {
        return PathBuf::from(root).join(stem);
    }
    let parent = usd_path.parent().unwrap_or(Path::new("."));
    parent.join("forge-paint").join(stem)
}

/// Upload sidecar PNGs into the given paint `layer`'s textures (one per
/// (channel × UDIM tile)). `tiles` and `resolution` come from the display
/// paint target so we can address layers correctly. Caller should recomposite
/// the layer stack afterwards to make the uploads visible.
pub fn load_sidecars(
    queue: &wgpu::Queue,
    layer: &Layer,
    tiles: &[u32],
    resolution: u32,
    work_dir: &Path,
) -> usize {
    if !work_dir.is_dir() {
        return 0;
    }
    let mut loaded = 0;

    for (layer_idx, udim) in tiles.iter().enumerate() {
        for (channel, tex_getter) in LAYER_CHANNELS {
            let path = work_dir.join(format!("{channel}.{udim}.png"));
            if !path.exists() {
                continue;
            }
            match image::open(&path) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    if rgba.width() != resolution || rgba.height() != resolution {
                        log::warn!(
                            "sidecar {} is {}×{}, expected {}×{} — skipping",
                            path.display(),
                            rgba.width(),
                            rgba.height(),
                            resolution,
                            resolution
                        );
                        continue;
                    }
                    upload_layer(
                        queue,
                        tex_getter(layer),
                        resolution,
                        layer_idx as u32,
                        rgba.as_raw(),
                    );
                    loaded += 1;
                    log::info!(
                        "loaded sidecar {} -> layer {layer_idx} ({channel}.{udim})",
                        path.display()
                    );
                }
                Err(e) => {
                    log::warn!("failed to decode {}: {e}", path.display());
                }
            }
        }
    }
    loaded
}

/// Save current paint target state as PNGs into `work_dir` (created if absent).
/// Delegates to the export module — the file format and naming are identical.
pub fn save_sidecars(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    paint_target: &PaintTarget,
    work_dir: &Path,
) -> Result<Vec<crate::export::Export>> {
    std::fs::create_dir_all(work_dir)
        .with_context(|| format!("create work dir {}", work_dir.display()))?;
    crate::export::export_tiles(device, queue, paint_target, work_dir)
}

fn upload_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    resolution: u32,
    layer: u32,
    rgba: &[u8],
) {
    let bytes_per_row = resolution * 4;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(resolution),
        },
        wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
    );
}
