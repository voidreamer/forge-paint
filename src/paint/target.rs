use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::mesh::CpuMesh;
use crate::paint::udim;

/// Capacity for the UDIM tile table baked into the material uniform.
/// 32 tiles covers characters/props/sets in prototype scope; larger sets are
/// clipped and a warning is logged. Packed as 8 × vec4<u32> in WGSL.
pub const MAX_TILES: usize = 32;

/// Material uniform matches WGSL `struct Material` in pbr.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniforms {
    pub base_color_factor: [f32; 4],
    /// x=metallic, y=roughness, z=normal_scale, w=_
    pub params: [f32; 4],
    pub tile_count: u32,
    pub _pad0: [u32; 3],
    /// 32 tile ids packed into 8 vec4<u32> to satisfy uniform array stride.
    pub tile_ids: [[u32; 4]; 8],
}

impl Default for MaterialUniforms {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            params: [0.0, 0.5, 1.0, 0.0],
            tile_count: 0,
            _pad0: [0; 3],
            tile_ids: [[0; 4]; 8],
        }
    }
}

pub struct PaintTarget {
    pub tiles: Vec<u32>,
    pub resolution: u32,

    pub base_color: wgpu::Texture,
    pub base_color_view: wgpu::TextureView,
    /// Per-layer views for render-attachment stamping (one per UDIM tile).
    pub base_color_layer_views: Vec<wgpu::TextureView>,

    pub rough_metal: wgpu::Texture,
    pub rough_metal_view: wgpu::TextureView,
    pub rough_metal_layer_views: Vec<wgpu::TextureView>,

    pub normal: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub normal_layer_views: Vec<wgpu::TextureView>,

    /// Scalar height buffer — R16Float keeps the HDR range we need for
    /// projected displacement maps without quadrupling the per-tile VRAM.
    /// R = accumulated displacement value (premultiplied), G = coverage.
    /// Final sampled height = R / max(G, epsilon).
    pub displacement: wgpu::Texture,
    pub displacement_view: wgpu::TextureView,
    pub displacement_layer_views: Vec<wgpu::TextureView>,

    pub sampler: wgpu::Sampler,

    /// Shared fully-visible mask bound when the active layer has none. D2Array
    /// with the same tile count as the content textures so shaders can sample
    /// it with the UDIM layer index like any real mask.
    pub dummy_mask: wgpu::Texture,
    pub dummy_mask_view: wgpu::TextureView,
}

impl PaintTarget {
    /// Index of `tile_id` in the layers array, or None if the tile isn't present.
    pub fn layer_for_tile(&self, tile_id: u32) -> Option<u32> {
        self.tiles.iter().position(|&t| t == tile_id).map(|i| i as u32)
    }
}

impl PaintTarget {
    /// Build a paint target sized to the UDIM tiles used by `mesh`.
    /// Layers are initialised to neutral PBR defaults (gray albedo, 0.5
    /// roughness, flat normal).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _material_bgl: &wgpu::BindGroupLayout,
        mesh: &CpuMesh,
        resolution: u32,
    ) -> Self {
        let mut tiles = udim::tiles_for_mesh(mesh);
        if tiles.len() > MAX_TILES {
            log::warn!(
                "mesh uses {} UDIM tiles; truncating to first {MAX_TILES}",
                tiles.len()
            );
            tiles.truncate(MAX_TILES);
        }
        if tiles.is_empty() {
            tiles.push(1001);
        }
        let layer_count = tiles.len() as u32;

        let base_color = create_array_texture(
            device,
            "paint.base_color",
            resolution,
            layer_count,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let rough_metal = create_array_texture(
            device,
            "paint.rough_metal",
            resolution,
            layer_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let normal = create_array_texture(
            device,
            "paint.normal",
            resolution,
            layer_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        // Rg16Float = 4 bytes/texel. R = premultiplied height, G = coverage.
        // Stored separately so the brush can composite signed height
        // changes (Rgba8-style premultiplied alpha in 16-bit precision).
        let displacement = create_array_texture(
            device,
            "paint.displacement",
            resolution,
            layer_count,
            wgpu::TextureFormat::Rg16Float,
        );

        let base_color_view = make_array_view(&base_color);
        let rough_metal_view = make_array_view(&rough_metal);
        let normal_view = make_array_view(&normal);
        let displacement_view = make_array_view(&displacement);

        // Per-layer views for render-attachment stamping on base_color and rough_metal.
        let base_color_layer_views: Vec<wgpu::TextureView> = (0..layer_count)
            .map(|layer| {
                base_color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint.base_color.layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let rough_metal_layer_views: Vec<wgpu::TextureView> = (0..layer_count)
            .map(|layer| {
                rough_metal.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint.rough_metal.layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let normal_layer_views: Vec<wgpu::TextureView> = (0..layer_count)
            .map(|layer| {
                normal.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint.normal.layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let displacement_layer_views: Vec<wgpu::TextureView> = (0..layer_count)
            .map(|layer| {
                displacement.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint.displacement.layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Seed every layer with neutral defaults via GPU clear-fill render passes —
        // avoids allocating a full resolution²·4·3·N host buffer per asset load.
        //
        // Clear colours are authored so the stored pixel bytes match the intended
        // neutrals. For sRGB targets the hardware linear→sRGB-encodes on store, so
        // the clear value must be pre-linearized. For unorm targets it's stored
        // as-is (clear value and byte value / 255 coincide).
        let bc_clear = wgpu::Color {
            // Target bytes (180,180,180,255) in sRGB → linear 0.4485
            r: srgb_to_linear(180.0 / 255.0) as f64,
            g: srgb_to_linear(180.0 / 255.0) as f64,
            b: srgb_to_linear(180.0 / 255.0) as f64,
            a: 1.0,
        };
        let rm_clear = wgpu::Color {
            // glTF packing: R=AO(1), G=roughness(0.5), B=metallic(0)
            r: 1.0,
            g: 128.0 / 255.0,
            b: 0.0,
            a: 1.0,
        };
        let nm_clear = wgpu::Color {
            // Flat tangent-space normal (0.5, 0.5, 1.0) stored as bytes (128,128,255)
            r: 128.0 / 255.0,
            g: 128.0 / 255.0,
            b: 1.0,
            a: 1.0,
        };

        // Pre-allocate per-layer views for each channel; keep them alive through
        // encoder submission, then drop.
        let mut fills: Vec<(wgpu::TextureView, wgpu::Color)> =
            Vec::with_capacity((layer_count * 3) as usize);
        for layer in 0..layer_count {
            fills.push((make_layer_view(&base_color, layer), bc_clear));
            fills.push((make_layer_view(&rough_metal, layer), rm_clear));
            fills.push((make_layer_view(&normal, layer), nm_clear));
            // Displacement: R=height, G=coverage — both 0 = no displacement.
            fills.push((
                make_layer_view(&displacement, layer),
                wgpu::Color::TRANSPARENT,
            ));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("paint.default_fill_enc"),
        });
        for (view, clear) in &fills {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint.default_fill_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(*clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        queue.submit(Some(encoder.finish()));
        drop(fills);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("paint.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Dummy "all 1.0" mask D2Array sized to the same tile count so the
        // PBR shader can address it with the UDIM layer index when the active
        // layer has no mask of its own.
        let dummy_mask = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("paint.dummy_mask"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_mask_view = dummy_mask.create_view(&wgpu::TextureViewDescriptor {
            label: Some("paint.dummy_mask.array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // Clear-fill each layer of the dummy to 1.0
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("paint.dummy_mask.init"),
            });
            for layer in 0..layer_count {
                let view = dummy_mask.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("paint.dummy_mask.init_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("paint.dummy_mask.init_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }
            queue.submit(Some(encoder.finish()));
        }

        Self {
            tiles,
            resolution,
            base_color,
            base_color_view,
            base_color_layer_views,
            rough_metal,
            rough_metal_view,
            rough_metal_layer_views,
            normal,
            normal_view,
            normal_layer_views,
            displacement,
            displacement_view,
            displacement_layer_views,
            sampler,
            dummy_mask,
            dummy_mask_view,
        }
    }

    /// Produce a MaterialUniforms value reflecting the tile table + the
    /// artist's material factors. Viewport owns the buffer; call this to get
    /// a POD it can `queue.write_buffer` into its own `material_buf`.
    pub fn material_uniforms(
        &self,
        base_color_factor: [f32; 4],
        metallic: f32,
        roughness: f32,
        normal_scale: f32,
        displacement_scale: f32,
    ) -> MaterialUniforms {
        let mut u = MaterialUniforms::default();
        u.base_color_factor = base_color_factor;
        u.params = [metallic, roughness, normal_scale, displacement_scale];
        u.tile_count = self.tiles.len() as u32;
        for (i, &tid) in self.tiles.iter().enumerate() {
            u.tile_ids[i / 4][i % 4] = tid;
        }
        u
    }
}

fn create_array_texture(
    device: &wgpu::Device,
    label: &str,
    resolution: u32,
    layers: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // TEXTURE_BINDING for sampling; COPY_DST for queue.write_texture;
        // RENDER_ATTACHMENT for brush-stamp passes; COPY_SRC for export readback.
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn make_array_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("paint.array_view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn make_layer_view(tex: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("paint.clear_layer_view"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

/// Inverse of the sRGB-encode used on store for Rgba8UnormSrgb render targets.
/// Keeps clear colours in the space the hardware expects (linear).
fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Neutral defaults — reused by `layer::Layer::new` to seed new paint layers
/// so a single-layer stack looks identical to today's direct paint target.
pub mod defaults {
    use egui_wgpu::wgpu;

    pub fn base_color_clear() -> wgpu::Color {
        // Bytes (180,180,180,255) as sRGB → linear ≈ 0.4485 for the clear
        let linear = super::srgb_to_linear(180.0 / 255.0) as f64;
        wgpu::Color {
            r: linear,
            g: linear,
            b: linear,
            a: 1.0,
        }
    }

    pub fn rough_metal_clear() -> wgpu::Color {
        // glTF packing: R=AO(1), G=rough(0.5), B=metal(0)
        wgpu::Color {
            r: 1.0,
            g: 128.0 / 255.0,
            b: 0.0,
            a: 1.0,
        }
    }

    pub fn normal_clear() -> wgpu::Color {
        // Flat tangent-space normal stored as bytes (128,128,255)
        wgpu::Color {
            r: 128.0 / 255.0,
            g: 128.0 / 255.0,
            b: 1.0,
            a: 1.0,
        }
    }
}
