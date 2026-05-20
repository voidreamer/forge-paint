//! Bottom-panel asset browser — imported textures (and later meshes / stencils)
//! that can be dropped onto layers. Mirrors ArmorPaint's Browser/Meshes/Textures
//! tabs at the bottom of the viewport.

use anyhow::{anyhow, Context, Result};
use egui_wgpu::wgpu;
use std::path::{Path, PathBuf};

use crate::paint::Layer;

/// Cheap string-match against the `exr` crate's error message for the
/// DWAA/DWAB case. The crate prints something like
/// `"yet unimplemented compression method: dwaa compression"` so both
/// "dwa" and "not supported" are reasonable markers. Using string match
/// (instead of matching on the concrete error variant) keeps us robust
/// if the exr crate reshapes its error types in minor releases.
fn looks_like_unsupported_dwa(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("dwa")
        || (lower.contains("not supported") && lower.contains("compression"))
        || lower.contains("unimplemented compression")
}

/// Run an external tool (oiiotool first, ffmpeg as fallback) to transcode
/// a DWA-compressed EXR into a ZIP-compressed EXR the `exr` crate can
/// read. Returns the path of the produced temp file on success — the
/// caller is responsible for deleting it.
fn transcode_dwa_to_zip(src: &Path) -> Result<PathBuf> {
    use std::process::Command;

    // Derive a unique-ish temp path. std::env::temp_dir() + PID + file
    // stem keeps parallel imports from fighting over the same file.
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "asset".into());
    let tmp = std::env::temp_dir().join(format!(
        "forge-paint-dwa-{}-{}.exr",
        std::process::id(),
        stem
    ));

    // oiiotool: preserves all channels and layers; the preferred path.
    //   oiiotool <src> --compression zip -o <dst>
    let oiio = Command::new("oiiotool")
        .arg(src)
        .arg("--compression")
        .arg("zip")
        .arg("-o")
        .arg(&tmp)
        .output();
    if let Ok(out) = oiio {
        if out.status.success() && tmp.exists() {
            log::info!(
                "DWA EXR transcoded via oiiotool: {} -> {}",
                src.display(),
                tmp.display()
            );
            return Ok(tmp);
        }
        log::warn!(
            "oiiotool transcode failed ({}). stderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }

    // ffmpeg fallback: flattens multi-channel / multi-layer EXRs to RGBA,
    // which is fine for asset browser imports.
    //   ffmpeg -y -i <src> -compression zip1 <dst>
    let ff = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(src)
        .arg("-compression")
        .arg("zip1")
        .arg(&tmp)
        .output();
    if let Ok(out) = ff {
        if out.status.success() && tmp.exists() {
            log::info!(
                "DWA EXR transcoded via ffmpeg: {} -> {}",
                src.display(),
                tmp.display()
            );
            return Ok(tmp);
        }
        log::warn!(
            "ffmpeg transcode failed ({}). stderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }

    Err(anyhow!(
        "neither `oiiotool` (OpenImageIO) nor `ffmpeg` could transcode the file; \
         install either tool (macOS: brew install openimageio / ffmpeg)"
    ))
}

/// Read an EXR file into an sRGB-encoded RGBA8 buffer, bypassing the
/// `image` crate's stricter wrapper so we handle the files Poly Haven
/// actually ships: single-channel luminance (Y), 8K normal RGB, and
/// arbitrary channel layouts. DWAA/DWAB-compressed files still fail
/// (exr 1.74 can't decode them); we surface a clear error for those.
///
/// The result is linear float → sRGB 8-bit, clamped to [0, 1]. Float
/// HDR values > 1 get clipped — fine for asset-browser thumbnails and
/// for stamping as base color / roughness / metallic where channels
/// are 8-bit anyway.
fn load_exr_rgba8(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    use exr::prelude::*;

    // Fast path — stock exr read. On failure, check whether it tripped on
    // DWAA/DWAB compression (which exr 1.74 doesn't decode yet). If so,
    // transparently transcode to a ZIP-compressed temp file via oiiotool
    // or ffmpeg and retry. Keeps zero C/C++ build deps at the cost of a
    // subprocess per DWA file; Poly Haven-sized assets transcode in ~1s
    // and the temp file is deleted after the re-read.
    let image = match read_first_flat_layer_from_file(path) {
        Ok(img) => img,
        Err(e) => {
            let msg = e.to_string();
            if looks_like_unsupported_dwa(&msg) {
                match transcode_dwa_to_zip(path) {
                    Ok(tmp) => {
                        let result = read_first_flat_layer_from_file(&tmp);
                        let _ = std::fs::remove_file(&tmp);
                        result.map_err(|e2| {
                            anyhow!(
                                "EXR {}: transcoded via external tool but re-read still failed: {e2}",
                                path.display()
                            )
                        })?
                    }
                    Err(transcode_err) => {
                        return Err(anyhow!(
                            "EXR {}: uses DWAA/DWAB compression, which the exr 1.74 Rust crate \
                             doesn't support. Install `oiiotool` (brew install openimageio) or \
                             `ffmpeg` so forge-paint can auto-transcode to ZIP on import, or \
                             re-export the file as ZIP / PIZ (or PNG) manually. \n\
                             Transcode attempt: {transcode_err}\nExr error: {msg}",
                            path.display()
                        ));
                    }
                }
            } else {
                return Err(anyhow::Error::new(e)
                    .context(format!("read EXR {}", path.display())));
            }
        }
    };

    // Use the layer's own size (matches the actual channel sample count);
    // attributes.display_window can differ from data_window in exotic
    // crops and would mislead per-pixel iteration below.
    let size = image.layer_data.size;
    let w = size.width() as u32;
    let h = size.height() as u32;
    let total = (w as usize) * (h as usize);

    let channels = &image.layer_data.channel_data.list;
    if channels.is_empty() {
        return Err(anyhow!("EXR {}: no channels", path.display()));
    }

    // Pull each channel's samples into a contiguous Vec<f32> so the
    // per-pixel assembly loop below indexes by flat pixel index without
    // caring about f16 vs f32 vs u32 storage.
    let channel_f32: Vec<(String, Vec<f32>)> = channels
        .iter()
        .map(|ch| {
            let name = ch.name.to_string();
            let v: Vec<f32> = match &ch.sample_data {
                FlatSamples::F16(v) => v.iter().map(|s| s.to_f32()).collect(),
                FlatSamples::F32(v) => v.clone(),
                FlatSamples::U32(v) => v.iter().map(|&s| s as f32).collect(),
            };
            (name, v)
        })
        .collect();

    let find = |target: &str| -> Option<usize> {
        channel_f32
            .iter()
            .position(|(n, _)| n.eq_ignore_ascii_case(target))
    };
    let r_idx = find("R");
    let g_idx = find("G");
    let b_idx = find("B");
    let a_idx = find("A");
    let y_idx = find("Y").or_else(|| find("L")).or_else(|| find("Luminance"));

    // Single-channel fallback: if no R/G/B and no Y, replicate the first
    // channel as grayscale. Matches ArmorPaint's behavior for arbitrary
    // scalar maps (often what Poly Haven's metalness/AO exports look like).
    let scalar_only = r_idx.is_none() && g_idx.is_none() && b_idx.is_none();
    let fallback_idx = if scalar_only && y_idx.is_none() {
        Some(0usize)
    } else {
        None
    };

    let mut rgba = vec![0u8; total * 4];
    let to_srgb_byte = |linear: f32| -> u8 {
        let v = linear.clamp(0.0, 1.0);
        let c = if v < 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (c * 255.0 + 0.5) as u8
    };

    for i in 0..total {
        let (rf, gf, bf) = if let Some(yi) = y_idx.or(fallback_idx) {
            let v = channel_f32[yi].1.get(i).copied().unwrap_or(0.0);
            (v, v, v)
        } else {
            (
                r_idx.map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                g_idx.map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                b_idx.map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
            )
        };
        let af = a_idx
            .map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(1.0))
            .unwrap_or(1.0);

        rgba[i * 4] = to_srgb_byte(rf);
        rgba[i * 4 + 1] = to_srgb_byte(gf);
        rgba[i * 4 + 2] = to_srgb_byte(bf);
        rgba[i * 4 + 3] = (af.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }

    Ok((w, h, rgba))
}

fn decode_texture_rgba8(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    // EXRs go through our custom loader so Poly Haven's single-channel
    // and 8K-sized normals decode. PNG / JPG / HDR stay on the `image`
    // crate which handles them fine.
    let is_exr = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("exr"))
        .unwrap_or(false);
    if is_exr {
        load_exr_rgba8(path)
    } else {
        let img = image::open(path)
            .with_context(|| format!("open image {}", path.display()))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        Ok((w, h, img.into_raw()))
    }
}

/// A single imported texture — kept in CPU memory so we can re-upload to
/// Layer textures at any tile resolution, and mirrored to a GPU texture for
/// the thumbnail preview in the browser.
pub struct TextureAsset {
    pub name: String,
    pub source: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Raw sRGB-encoded RGBA8, PNG-order (V=0 at top). We flip on the way to
    /// a Layer's base_color to match our GPU convention (see persist.rs).
    pub pixels: Vec<u8>,
    /// Thumbnail GPU texture — native resolution, uploaded once.
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub thumb_id: egui::TextureId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Textures,
    Meshes,
    Stencils,
    Swatches,
    Materials,
}

impl Tab {
    pub fn label(self) -> &'static str {
        match self {
            Tab::Textures => "Textures",
            Tab::Meshes => "Meshes",
            Tab::Stencils => "Stencils",
            Tab::Swatches => "Swatches",
            Tab::Materials => "Materials",
        }
    }
    pub const ALL: &'static [Tab] = &[
        Tab::Textures,
        Tab::Meshes,
        Tab::Stencils,
        Tab::Swatches,
        Tab::Materials,
    ];
}

/// What flavour of shader network the material is built on. Used by
/// the Materials pane to filter the library, and to decide whether a
/// given material is safe to apply for the user's current delegate
/// (UsdPreviewSurface is universal; 3Delight OSL is hdNSI-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialKind {
    UsdPreviewSurface,
    MaterialX,
    /// 3Delight's native OSL shaders. Rendered correctly by hdNSI;
    /// other delegates may ignore or fall back.
    OslDelight,
    Other,
}

impl MaterialKind {
    pub fn label(self) -> &'static str {
        match self {
            MaterialKind::UsdPreviewSurface => "UsdPreviewSurface",
            MaterialKind::MaterialX => "MaterialX",
            MaterialKind::OslDelight => "3Delight (OSL)",
            MaterialKind::Other => "Other",
        }
    }
    pub const ALL: &'static [MaterialKind] = &[
        MaterialKind::UsdPreviewSurface,
        MaterialKind::MaterialX,
        MaterialKind::OslDelight,
        MaterialKind::Other,
    ];
}

/// One entry in the Materials pane. Sourced from a USD file on disk —
/// applying the material references the source into the loaded stage's
/// session layer (bridge's `set_external_material`).
#[derive(Debug, Clone)]
pub struct MaterialAsset {
    pub name: String,
    /// USD file containing the material network. Handed to the bridge
    /// as the reference source.
    pub source: PathBuf,
    /// Path of the `UsdShadeMaterial` prim inside `source`. Empty
    /// string means "use the source's default prim", which is the
    /// pipeline convention most material libraries follow.
    pub prim_path: String,
    pub kind: MaterialKind,
}

/// Reference to a USD file on disk — thumbnail comes later; for now we
/// render a Phosphor cube glyph in the Meshes tab.
#[derive(Debug, Clone)]
pub struct MeshAsset {
    pub name: String,
    pub path: PathBuf,
}

pub struct AssetBrowser {
    pub textures: Vec<TextureAsset>,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub active_tab: Tab,
    /// Filter chip set for the Materials pane — `None` entries are
    /// hidden. Tracks each `MaterialKind`'s visibility independently
    /// so the user can multi-select (UsdPreviewSurface + MaterialX
    /// but not 3Delight, say).
    pub material_kind_filter: [bool; 4],
}

impl Default for AssetBrowser {
    fn default() -> Self {
        Self {
            textures: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            active_tab: Tab::Textures,
            material_kind_filter: [true; 4],
        }
    }
}

impl AssetBrowser {
    pub fn import_texture(
        &mut self,
        path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut egui_wgpu::Renderer,
    ) -> Result<()> {
        let (w, h, pixels) = decode_texture_rgba8(path)?;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("asset.texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let thumb_id = renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear);

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "texture".into());

        self.textures.push(TextureAsset {
            name,
            source: path.to_path_buf(),
            width: w,
            height: h,
            pixels,
            texture,
            view,
            thumb_id,
        });
        Ok(())
    }
}

/// Resize sRGB RGBA8 pixels using a triangle filter — cheap, good enough for
/// asset application. Returns a buffer of `target_w * target_h * 4` bytes.
fn resize_rgba8(src: &[u8], src_w: u32, src_h: u32, target_w: u32, target_h: u32) -> Vec<u8> {
    if src_w == target_w && src_h == target_h {
        return src.to_vec();
    }
    let buf = image::RgbaImage::from_raw(src_w, src_h, src.to_vec())
        .expect("RgbaImage from_raw size mismatch");
    let resized =
        image::imageops::resize(&buf, target_w, target_h, image::imageops::FilterType::Triangle);
    resized.into_raw()
}

/// Flip rows of an RGBA8 image in place. Matches the convention used by
/// persist.rs / export.rs — GPU textures store V=0 at bottom.
fn flip_rows_rgba8(pixels: &mut [u8], w: u32, h: u32) {
    let stride = (w * 4) as usize;
    let h = h as usize;
    for y in 0..h / 2 {
        let top = y * stride;
        let bot = (h - 1 - y) * stride;
        for i in 0..stride {
            pixels.swap(top + i, bot + i);
        }
    }
}

/// Upload `asset` into `layer.base_color` across every tile. Flips rows to
/// match the project's V convention and resamples to `tile_resolution`.
pub fn apply_as_base_color(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    if tile_count == 0 {
        return Err(anyhow!("layer has no tiles"));
    }
    let mut pixels = resize_rgba8(&asset.pixels, asset.width, asset.height, tile_resolution, tile_resolution);
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    for tile_idx in 0..tile_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &layer.base_color,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: tile_idx,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_resolution * 4),
                rows_per_image: Some(tile_resolution),
            },
            wgpu::Extent3d {
                width: tile_resolution,
                height: tile_resolution,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

/// Upload `asset` into `layer.normal` across every tile. Normal maps are
/// RGBA8 direct — no channel conversion, just resize + V-flip.
pub fn apply_as_normal(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    if tile_count == 0 {
        return Err(anyhow!("layer has no tiles"));
    }
    let mut pixels = resize_rgba8(&asset.pixels, asset.width, asset.height, tile_resolution, tile_resolution);
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    for tile_idx in 0..tile_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &layer.normal,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: tile_idx },
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_resolution * 4),
                rows_per_image: Some(tile_resolution),
            },
            wgpu::Extent3d {
                width: tile_resolution,
                height: tile_resolution,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

fn apply_as_single_channel(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    texture: &wgpu::Texture,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    if tile_count == 0 {
        return Err(anyhow!("layer has no tiles"));
    }
    let mut pixels = resize_rgba8(&asset.pixels, asset.width, asset.height, tile_resolution, tile_resolution);
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    let mut r8 = vec![0u8; (tile_resolution * tile_resolution) as usize];
    for (i, chunk) in pixels.chunks_exact(4).enumerate() {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        r8[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8;
    }
    for tile_idx in 0..tile_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: tile_idx },
                aspect: wgpu::TextureAspect::All,
            },
            &r8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_resolution),
                rows_per_image: Some(tile_resolution),
            },
            wgpu::Extent3d {
                width: tile_resolution,
                height: tile_resolution,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

pub fn apply_as_roughness(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    apply_as_single_channel(queue, asset, &layer.roughness, tile_count, tile_resolution)
}

pub fn apply_as_metallic(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    apply_as_single_channel(queue, asset, &layer.metallic, tile_count, tile_resolution)
}

/// Upload `asset` into the active layer's mask (single-channel R8). Takes
/// the luminance of each pixel. Caller must ensure the layer has a mask.
pub fn apply_as_mask(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_count: u32,
    tile_resolution: u32,
) -> Result<()> {
    let Some(mask) = &layer.mask else {
        return Err(anyhow!("layer has no mask"));
    };
    if tile_count == 0 {
        return Err(anyhow!("layer has no tiles"));
    }
    let mut pixels = resize_rgba8(&asset.pixels, asset.width, asset.height, tile_resolution, tile_resolution);
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    // RGBA → luminance (Rec. 709). Cheap approximation — users rarely
    // care about precise colorimetry for mask input.
    let mut r8 = vec![0u8; (tile_resolution * tile_resolution) as usize];
    for (i, chunk) in pixels.chunks_exact(4).enumerate() {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        r8[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8;
    }
    for tile_idx in 0..tile_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &mask.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: tile_idx,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &r8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_resolution),
                rows_per_image: Some(tile_resolution),
            },
            wgpu::Extent3d {
                width: tile_resolution,
                height: tile_resolution,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

/// Walk `<root>/assets/materials/` (and a few fallback locations
/// matching the HDRI discovery's heuristic) for USD files that
/// look like material definitions, classify each by shader id, and
/// return the resulting list ready to drop into
/// `AssetBrowser::materials`.
///
/// Empty result is OK — `assets/materials/` doesn't exist by default;
/// the user (or pipeline) drops `*.usd` / `*.usda` / `*.usdc` files
/// with `def Material` prims into it to populate the pane.
pub fn discover_materials(root: &Path) -> Vec<MaterialAsset> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(root.join("assets").join("materials"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("assets").join("materials"));
            if let Some(p2) = parent.parent() {
                candidates.push(p2.join("assets").join("materials"));
            }
            if let Some(p3) = parent.parent().and_then(|p| p.parent()) {
                candidates.push(p3.join("assets").join("materials"));
            }
        }
    }
    if let Some(env_dir) = std::env::var_os("FORGE_PAINT_MATERIAL_DIR") {
        candidates.push(env_dir.into());
    }

    for dir in candidates {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let is_usd = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    let l = e.to_ascii_lowercase();
                    l == "usd" || l == "usda" || l == "usdc" || l == "usdz"
                })
                .unwrap_or(false);
            if !is_usd {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("material")
                .to_string();
            let kind = classify_material_file(&path);
            out.push(MaterialAsset {
                name,
                source: path,
                // Library convention: each file's default prim is the
                // material. Multi-material files extracted into one-
                // per-entry later if needed — keeps v1 shape simple.
                prim_path: String::new(),
                kind,
            });
        }
        if !out.is_empty() {
            out.sort_by(|a, b| a.name.cmp(&b.name));
            return out;
        }
    }
    Vec::new()
}

/// Cheap shader-id sniff of a USD material file. Filename hints
/// first (`*.materialx.usd`, `*.osl.usd`, …); falls back to
/// pattern-matching the first ~64 KB as text. Binary `.usdc` files
/// without matching identifiers land in `Other` — full
/// classification would mean opening the stage via rust-usd, which
/// is heavier than this pane warrants for v1.
fn classify_material_file(path: &Path) -> MaterialKind {
    let lower_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if lower_name.contains("materialx") || lower_name.contains(".mtlx") {
        return MaterialKind::MaterialX;
    }
    if lower_name.contains(".osl.") || lower_name.contains("delight") {
        return MaterialKind::OslDelight;
    }
    if lower_name.contains("preview") || lower_name.contains(".up.") {
        return MaterialKind::UsdPreviewSurface;
    }

    let Ok(mut f) = std::fs::File::open(path) else {
        return MaterialKind::Other;
    };
    use std::io::Read;
    let mut buf = vec![0u8; 64 * 1024];
    let n = f.read(&mut buf).unwrap_or(0);
    let head = std::str::from_utf8(&buf[..n]).unwrap_or_default();
    if head.contains("UsdPreviewSurface") {
        return MaterialKind::UsdPreviewSurface;
    }
    if head.contains("MaterialX") || head.contains("ND_") {
        // `ND_` is MaterialX's node-definition prefix — common in any
        // serialised MaterialX network even when embedded in USD.
        return MaterialKind::MaterialX;
    }
    if head.contains("dlPrincipled")
        || head.contains("dlSubstance")
        || head.contains("DelightSurface")
    {
        return MaterialKind::OslDelight;
    }
    MaterialKind::Other
}
