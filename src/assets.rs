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
                return Err(anyhow::Error::new(e).context(format!("read EXR {}", path.display())));
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
    let y_idx = find("Y")
        .or_else(|| find("L"))
        .or_else(|| find("Luminance"));

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
                r_idx
                    .map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0))
                    .unwrap_or(0.0),
                g_idx
                    .map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0))
                    .unwrap_or(0.0),
                b_idx
                    .map(|ci| channel_f32[ci].1.get(i).copied().unwrap_or(0.0))
                    .unwrap_or(0.0),
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
    /// 3Delight Disney-style shaders — `dlPrincipled` and its
    /// specialized siblings (`dlMetal`, `dlGlass`, `dlSkin`,
    /// `dlCarPaint`, …). Rendered correctly by hdNSI; other
    /// delegates may fall back to a default surface.
    DlPrincipled,
    Other,
}

impl MaterialKind {
    pub fn label(self) -> &'static str {
        match self {
            MaterialKind::UsdPreviewSurface => "UsdPreviewSurface",
            MaterialKind::MaterialX => "MaterialX",
            MaterialKind::DlPrincipled => "3Delight",
            MaterialKind::Other => "Other",
        }
    }
    pub const ALL: &'static [MaterialKind] = &[
        MaterialKind::UsdPreviewSurface,
        MaterialKind::MaterialX,
        MaterialKind::DlPrincipled,
        MaterialKind::Other,
    ];

    /// Surface-shader input names this kind uses for the editor's
    /// slider set. UsdPreviewSurface, MaterialX
    /// `ND_standard_surface_surfaceshader`, and 3Delight's
    /// `_3DelightMaterial` all expose conceptually-equivalent inputs
    /// under different attribute names. `None` for an entry means
    /// "this shader doesn't have a corresponding knob" — the editor
    /// hides that row and the bridge skips the override.
    pub fn input_names(self) -> ShaderInputNames {
        match self {
            MaterialKind::UsdPreviewSurface | MaterialKind::Other => ShaderInputNames {
                diffuse_color: Some("diffuseColor"),
                metallic: Some("metallic"),
                roughness: Some("roughness"),
                opacity: Some("opacity"),
                clearcoat: Some("clearcoat"),
                clearcoat_roughness: Some("clearcoatRoughness"),
                emission_color: Some("emissiveColor"),
                normal: Some("normal"),
                occlusion: Some("occlusion"),
                // UsdPreviewSurface folds emission magnitude into the
                // emissiveColor's components — no separate scalar.
                emission_intensity: None,
            },
            MaterialKind::MaterialX => ShaderInputNames {
                diffuse_color: Some("base_color"),
                metallic: Some("metalness"),
                roughness: Some("specular_roughness"),
                opacity: Some("opacity"),
                // MaterialX standard_surface uses `coat` for the
                // clearcoat layer toggle (0..1 weight) and
                // `coat_roughness` for its roughness.
                clearcoat: Some("coat"),
                clearcoat_roughness: Some("coat_roughness"),
                emission_color: Some("emission_color"),
                normal: Some("normal"),
                occlusion: None,
                emission_intensity: Some("emission"),
            },
            MaterialKind::DlPrincipled => ShaderInputNames {
                // dlPrincipled / dlMetal / dlCarPaint all use the
                // Disney-style naming below. The coat slot drives
                // `coating_thickness` directly — there is no
                // `coating_on` toggle, coat activates whenever
                // thickness > 0.
                diffuse_color: Some("i_color"),
                metallic: Some("metallic"),
                roughness: Some("roughness"),
                opacity: Some("opacity"),
                clearcoat: Some("coating_thickness"),
                clearcoat_roughness: Some("coating_roughness"),
                emission_color: Some("incandescence"),
                normal: None,
                occlusion: None,
                emission_intensity: Some("incandescence_intensity"),
            },
        }
    }
}

/// Maps the editor's standardised slider concepts onto shader-id-
/// specific `inputs:*` names. `None` = the shader doesn't expose
/// that concept under any name we know of — the editor hides the
/// row and the bridge skips the override.
#[derive(Debug, Clone, Copy)]
pub struct ShaderInputNames {
    pub diffuse_color: Option<&'static str>,
    pub metallic: Option<&'static str>,
    pub roughness: Option<&'static str>,
    pub opacity: Option<&'static str>,
    pub clearcoat: Option<&'static str>,
    pub clearcoat_roughness: Option<&'static str>,
    pub emission_color: Option<&'static str>,
    pub normal: Option<&'static str>,
    pub occlusion: Option<&'static str>,
    pub emission_intensity: Option<&'static str>,
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
    /// Cached preview inputs for the gallery chip and material-graph
    /// node preview. Read at discovery time from the material's
    /// authored diffuse / base / `i_color` input plus common scalar
    /// controls.
    pub preview_inputs: MaterialInputs,
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

    pub fn texture_index_for_source(&self, path: &Path) -> Option<usize> {
        let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.textures.iter().position(|asset| {
            asset.source == path
                || std::fs::canonicalize(&asset.source)
                    .map(|p| p == target)
                    .unwrap_or(false)
        })
    }

    pub fn import_texture_once(
        &mut self,
        path: &Path,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut egui_wgpu::Renderer,
    ) -> Result<usize> {
        if let Some(idx) = self.texture_index_for_source(path) {
            return Ok(idx);
        }
        self.import_texture(path, device, queue, renderer)?;
        Ok(self.textures.len() - 1)
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
    let resized = image::imageops::resize(
        &buf,
        target_w,
        target_h,
        image::imageops::FilterType::Triangle,
    );
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
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
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

pub fn apply_as_base_color_tile(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_layer: u32,
    tile_resolution: u32,
) -> Result<()> {
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &layer.base_color,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: tile_layer,
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
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    for tile_idx in 0..tile_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &layer.normal,
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

pub fn apply_as_normal_tile(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_layer: u32,
    tile_resolution: u32,
) -> Result<()> {
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &layer.normal,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: tile_layer,
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
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
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

fn apply_as_single_channel_tile(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    texture: &wgpu::Texture,
    tile_layer: u32,
    tile_resolution: u32,
) -> Result<()> {
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
    flip_rows_rgba8(&mut pixels, tile_resolution, tile_resolution);
    let mut r8 = vec![0u8; (tile_resolution * tile_resolution) as usize];
    for (i, chunk) in pixels.chunks_exact(4).enumerate() {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        r8[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8;
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: tile_layer,
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

pub fn apply_as_roughness_tile(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_layer: u32,
    tile_resolution: u32,
) -> Result<()> {
    apply_as_single_channel_tile(queue, asset, &layer.roughness, tile_layer, tile_resolution)
}

pub fn apply_as_metallic_tile(
    queue: &wgpu::Queue,
    asset: &TextureAsset,
    layer: &Layer,
    tile_layer: u32,
    tile_resolution: u32,
) -> Result<()> {
    apply_as_single_channel_tile(queue, asset, &layer.metallic, tile_layer, tile_resolution)
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
    let mut pixels = resize_rgba8(
        &asset.pixels,
        asset.width,
        asset.height,
        tile_resolution,
        tile_resolution,
    );
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
            let preview_inputs = read_material_inputs(&path);
            out.push(MaterialAsset {
                name,
                source: path,
                // Library convention: each file's default prim is the
                // material. Multi-material files extracted into one-
                // per-entry later if needed — keeps v1 shape simple.
                prim_path: String::new(),
                kind,
                preview_inputs,
            });
        }
        if !out.is_empty() {
            out.sort_by(|a, b| a.name.cmp(&b.name));
            return out;
        }
    }
    Vec::new()
}

/// Snapshot of a UsdPreviewSurface / MaterialX standard_surface /
/// 3Delight `_3DelightMaterial` shader's commonly-edited inputs.
/// Populated by reading the source USDA text — binary `.usdc` files
/// miss it (and the editor falls back to safe defaults).
///
/// Point is to seed the editor sliders with the material's authored
/// values so the user starts from what the library says rather than
/// mid-range guesses. Any change authors an override session-layer-
/// side via the bridge.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaterialInputs {
    pub diffuse_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub opacity: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub emission_color: [f32; 3],
    pub emission_intensity: f32,
}

impl Default for MaterialInputs {
    fn default() -> Self {
        Self {
            diffuse_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            opacity: 1.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.5,
            emission_color: [0.0, 0.0, 0.0],
            emission_intensity: 0.0,
        }
    }
}

/// Parse the first ~64KB of a `.usda` (or any UTF-8 USD text) for the
/// common surface-shader inputs. Recognises both UsdPreviewSurface
/// names (`diffuseColor`, `metallic`, `roughness`, `opacity`) and
/// MaterialX standard_surface aliases (`base_color`, `metalness`,
/// `specular_roughness`) — first match wins. Binary `.usdc` files
/// return defaults.
///
/// Cheap text-pattern match — not a full parser. Misses textured
/// inputs (`inputs:diffuseColor.connect = …`), which is fine: the
/// sliders only drive literal values.
pub fn read_material_inputs(path: &Path) -> MaterialInputs {
    let mut inputs = MaterialInputs::default();
    let Ok(mut f) = std::fs::File::open(path) else {
        return inputs;
    };
    use std::io::Read;
    let mut buf = vec![0u8; 64 * 1024];
    let n = f.read(&mut buf).unwrap_or(0);
    let Ok(text) = std::str::from_utf8(&buf[..n]) else {
        return inputs;
    };
    let scan_color = |needles: &[&str]| -> Option<[f32; 3]> {
        for needle in needles {
            for prefix in [
                format!("color3f inputs:{needle} = ("),
                format!("color3f inputs:{needle}= ("),
            ] {
                if let Some(start) = text.find(&prefix) {
                    let tail = &text[start + prefix.len()..];
                    if let Some(end) = tail.find(')') {
                        let parts: Vec<f32> = tail[..end]
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if parts.len() == 3 {
                            return Some([parts[0], parts[1], parts[2]]);
                        }
                    }
                }
            }
        }
        None
    };
    let scan_float = |needles: &[&str]| -> Option<f32> {
        for needle in needles {
            for prefix in [
                format!("float inputs:{needle} = "),
                format!("float inputs:{needle}= "),
            ] {
                if let Some(start) = text.find(&prefix) {
                    let tail = &text[start + prefix.len()..];
                    let end = tail
                        .find(|c: char| c == '\n' || c == ' ')
                        .unwrap_or(tail.len());
                    if let Ok(v) = tail[..end].trim().parse::<f32>() {
                        return Some(v);
                    }
                }
            }
        }
        None
    };

    let is_dl_glass = text.contains("info:id = \"dlGlass\"");

    if let Some(c) = scan_color(&[
        "diffuseColor",
        "base_color",
        "i_color",
        "reflect_color",
        "refract_color",
    ]) {
        inputs.diffuse_color = c;
    }
    if let Some(v) = scan_float(&["metallic", "metalness"]) {
        inputs.metallic = v;
    }
    if let Some(v) = scan_float(&[
        "roughness",
        "specular_roughness",
        "reflect_roughness",
        "refract_roughness",
    ]) {
        inputs.roughness = v;
    }
    if let Some(v) = scan_float(&["opacity"]) {
        inputs.opacity = v;
    }
    // dlPrincipled has no clearcoat weight — `coating_thickness > 0`
    // activates coat directly. Scan it so the slider reflects the
    // authored thickness for those materials.
    if let Some(v) = scan_float(&["clearcoat", "coat", "coating_thickness"]) {
        inputs.clearcoat = v;
    }
    if let Some(v) = scan_float(&["clearcoatRoughness", "coat_roughness", "coating_roughness"]) {
        inputs.clearcoat_roughness = v;
    }
    if let Some(c) = scan_color(&["emissiveColor", "emission_color", "incandescence"]) {
        inputs.emission_color = c;
    }
    if let Some(v) = scan_float(&["emission", "incandescence_intensity"]) {
        inputs.emission_intensity = v;
    }
    if is_dl_glass {
        inputs.opacity = inputs.opacity.min(0.28);
        inputs.metallic = 0.0;
    }
    inputs
}

/// Paint a small, self-contained material preview ball. It is not a
/// renderer thumbnail, but it does read the same scalar inputs the editor
/// mutates, so swatches and shader nodes show metal/roughness/opacity
/// differences instead of a flat colour glyph.
pub fn paint_material_preview_ball(ui: &egui::Ui, rect: egui::Rect, inputs: MaterialInputs) {
    let painter = ui.painter();
    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(24, 25, 28));

    let pad = rect.width().min(rect.height()) * 0.11;
    let side = (rect.width().min(rect.height()) - pad * 2.0).max(8.0);
    let ball_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
    let center = ball_rect.center();
    let radius = side * 0.5;

    if inputs.opacity < 0.98 {
        paint_checker(ui, ball_rect, radius);
    }

    let base = shade_base(inputs.diffuse_color, inputs.opacity);
    painter.circle_filled(center, radius, base);

    let mut mesh = egui::epaint::Mesh::default();
    let steps = 30;
    for y in 0..steps {
        for x in 0..steps {
            let x0 = -1.0 + 2.0 * x as f32 / steps as f32;
            let y0 = -1.0 + 2.0 * y as f32 / steps as f32;
            let x1 = -1.0 + 2.0 * (x + 1) as f32 / steps as f32;
            let y1 = -1.0 + 2.0 * (y + 1) as f32 / steps as f32;
            let cx = (x0 + x1) * 0.5;
            let cy = (y0 + y1) * 0.5;
            if cx * cx + cy * cy > 1.0 {
                continue;
            }

            let idx = mesh.vertices.len() as u32;
            for (sx, sy) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
                let r2 = (sx * sx + sy * sy).min(0.995);
                let z = (1.0 - r2).sqrt();
                let color = material_preview_color(sx, sy, z, inputs);
                mesh.colored_vertex(
                    egui::pos2(center.x + sx * radius, center.y + sy * radius),
                    color,
                );
            }
            mesh.add_triangle(idx, idx + 1, idx + 2);
            mesh.add_triangle(idx, idx + 2, idx + 3);
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    let gloss = (1.0 - inputs.roughness.clamp(0.0, 1.0)).powf(2.0);
    if gloss > 0.08 || inputs.clearcoat > 0.05 {
        let highlight = egui::pos2(center.x - radius * 0.33, center.y - radius * 0.42);
        let r = radius * (0.09 + 0.1 * inputs.roughness.clamp(0.0, 1.0));
        painter.circle_filled(
            highlight,
            r,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, (120.0 * gloss) as u8),
        );
    }
    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 90)),
    );
}

fn paint_checker(ui: &egui::Ui, rect: egui::Rect, radius: f32) {
    let painter = ui.painter();
    let cell = (rect.width() / 7.0).max(4.0);
    let center = rect.center();
    let mut y = rect.top();
    let mut row = 0;
    while y < rect.bottom() {
        let mut x = rect.left();
        let mut col = 0;
        while x < rect.right() {
            let mid = egui::pos2(
                (x + cell * 0.5).min(rect.right()),
                (y + cell * 0.5).min(rect.bottom()),
            );
            let d = mid - center;
            if d.length_sq() <= radius * radius {
                let color = if (row + col) % 2 == 0 {
                    egui::Color32::from_rgb(66, 68, 72)
                } else {
                    egui::Color32::from_rgb(42, 44, 48)
                };
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x, y),
                        egui::pos2((x + cell).min(rect.right()), (y + cell).min(rect.bottom())),
                    ),
                    0.0,
                    color,
                );
            }
            x += cell;
            col += 1;
        }
        y += cell;
        row += 1;
    }
}

fn material_preview_color(nx: f32, ny: f32, nz: f32, inputs: MaterialInputs) -> egui::Color32 {
    let base = inputs.diffuse_color.map(|v| v.clamp(0.0, 1.0));
    let metallic = inputs.metallic.clamp(0.0, 1.0);
    let roughness = inputs.roughness.clamp(0.02, 1.0);
    let opacity = inputs.opacity.clamp(0.08, 1.0);

    let n = glam::Vec3::new(nx, -ny, nz).normalize_or_zero();
    let light = glam::Vec3::new(-0.45, 0.62, 0.78).normalize();
    let view = glam::Vec3::Z;
    let half = (light + view).normalize();
    let ndotl = n.dot(light).max(0.0);
    let ndoth = n.dot(half).max(0.0);
    let rim = (1.0 - nz).clamp(0.0, 1.0).powf(2.0);

    let diffuse_energy = 1.0 - metallic * 0.72;
    let diffuse = 0.13 + 0.78 * ndotl;
    let spec_power = 3.0 + (1.0 - roughness).powf(2.2) * 180.0;
    let spec = ndoth.powf(spec_power) * (1.15 - roughness * 0.72);
    let coat = ndoth.powf(260.0) * inputs.clearcoat.clamp(0.0, 2.0) * 0.32;
    let emission = inputs.emission_intensity.clamp(0.0, 12.0) * 0.08;

    let mut rgb = [0.0; 3];
    for i in 0..3 {
        let f0 = 0.045 * (1.0 - metallic) + base[i] * metallic;
        let env = 0.08 + 0.13 * rim + 0.06 * (1.0 - roughness);
        rgb[i] = base[i] * diffuse * diffuse_energy
            + f0 * spec * (0.75 + 1.35 * metallic)
            + env * (0.7 + metallic * 0.4)
            + coat
            + inputs.emission_color[i].clamp(0.0, 1.0) * emission;
    }

    egui::Color32::from_rgba_unmultiplied(
        to_u8(rgb[0]),
        to_u8(rgb[1]),
        to_u8(rgb[2]),
        (opacity * 255.0) as u8,
    )
}

fn shade_base(rgb: [f32; 3], opacity: f32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        to_u8(rgb[0] * 0.28),
        to_u8(rgb[1] * 0.28),
        to_u8(rgb[2] * 0.28),
        (opacity.clamp(0.08, 1.0) * 255.0) as u8,
    )
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
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
    // `dl_` is our naming convention for any 3Delight Disney-style
    // material (dlPrincipled / dlMetal / dlSkin / dlGlass / dlCarPaint
    // / dlHairAndFur). `.osl.` / `delight` substrings catch the same
    // family under alternative naming.
    if lower_name.starts_with("dl_")
        || lower_name.contains(".osl.")
        || lower_name.contains("delight")
    {
        return MaterialKind::DlPrincipled;
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
        || head.contains("dlMetal")
        || head.contains("dlGlass")
        || head.contains("dlSkin")
        || head.contains("dlCarPaint")
        || head.contains("dlHairAndFur")
        || head.contains("dlSubstance")
    {
        return MaterialKind::DlPrincipled;
    }
    MaterialKind::Other
}
