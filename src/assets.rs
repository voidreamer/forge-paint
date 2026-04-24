//! Bottom-panel asset browser — imported textures (and later meshes / stencils)
//! that can be dropped onto layers. Mirrors ArmorPaint's Browser/Meshes/Textures
//! tabs at the bottom of the viewport.

use anyhow::{anyhow, Context, Result};
use egui_wgpu::wgpu;
use std::path::{Path, PathBuf};

use crate::paint::Layer;

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
}

impl Tab {
    pub fn label(self) -> &'static str {
        match self {
            Tab::Textures => "Textures",
            Tab::Meshes => "Meshes",
            Tab::Stencils => "Stencils",
            Tab::Swatches => "Swatches",
        }
    }
    pub const ALL: &'static [Tab] = &[Tab::Textures, Tab::Meshes, Tab::Stencils, Tab::Swatches];
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
    pub active_tab: Tab,
}

impl Default for AssetBrowser {
    fn default() -> Self {
        Self {
            textures: Vec::new(),
            meshes: Vec::new(),
            active_tab: Tab::Textures,
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
        let img = image::open(path)
            .with_context(|| format!("open image {}", path.display()))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        let pixels = img.into_raw();

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
