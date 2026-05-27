//! GPU pipeline for the smart-mask regenerator. One fullscreen render
//! per tile of the active layer's mask, reading a baked mesh-map and
//! writing the threshold / falloff result into the mask atlas.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::bake::integration::MapKind;
use crate::bake::MeshMaps;
use crate::paint::layer::{Layer, Mask};
use crate::paint::smart_mask::{SmartMaskParams, SmartMaskSource};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SmartUniforms {
    low: f32,
    high: f32,
    contrast: f32,
    invert: u32,
    source_kind: u32,
    tile_layer: u32,
    _pad: [u32; 2],
}

pub struct SmartMaskPipeline {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl SmartMaskPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smart_mask.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("smart_mask.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smart_mask.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Source baked map — bound as a D2Array so the same
                // layout works for AO / curvature / thickness / world
                // normal regardless of channel count (the shader picks
                // .r or .g per source kind).
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smart_mask.uniforms"),
            size: std::mem::size_of::<SmartUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smart_mask.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smart_mask.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("smart_mask.pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_smart"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_smart"),
                targets: &[Some(wgpu::ColorTargetState {
                    // Mask atlas is R8Unorm. Shader writes vec4 with
                    // value in .r — only the red channel survives.
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            uniform_buf,
            sampler,
        }
    }

    /// Regenerate every tile of `mask` from the baked map matching
    /// `params.source`. Returns Err with a user-readable hint if the
    /// required source bake is missing — the caller surfaces that in
    /// the UI rather than crashing the bind group.
    pub fn regenerate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mask: &Mask,
        mesh_maps: &MeshMaps,
        params: &SmartMaskParams,
    ) -> Result<(), String> {
        // Pick the source view per-kind. Matches the doc table in
        // smart_mask.rs / smart_mask.wgsl.
        let kind_idx: u32 = match params.source {
            SmartMaskSource::AoCrevice => 0,
            SmartMaskSource::CurvatureConvex => 1,
            SmartMaskSource::CurvatureConcave => 2,
            SmartMaskSource::Thickness => 3,
            SmartMaskSource::WorldYUp => 4,
        };

        let required = params.source.required_map();
        let source_view: &wgpu::TextureView = match required {
            MapKind::AmbientOcclusion => mesh_maps
                .ao
                .as_ref()
                .map(|b| &b.view)
                .ok_or_else(|| "bake AO first".to_string())?,
            MapKind::Curvature => mesh_maps
                .curvature
                .as_ref()
                .map(|b| &b.view)
                .ok_or_else(|| "bake curvature first".to_string())?,
            MapKind::Thickness => mesh_maps
                .thickness
                .as_ref()
                .map(|b| &b.view)
                .ok_or_else(|| "bake thickness first".to_string())?,
            MapKind::WorldNormal => &mesh_maps.world_normal_view,
            _ => return Err(format!("unsupported smart-mask source: {required:?}")),
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smart_mask.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // One submit per tile — the uniform buffer is shared across
        // tiles; if we batched all tile passes into one encoder + one
        // submit, every pass would see the same (last-written) uniform
        // value because queue.write_buffer copies are serialised before
        // command-buffer execution within a submission. Per-tile submit
        // is slow for thousands of tiles but smart-mask regen happens
        // only on user-driven parameter changes (rare), so the cost is
        // a non-issue for typical asset sizes.
        for (tile_idx, view) in mask.layer_views.iter().enumerate() {
            queue.write_buffer(
                &self.uniform_buf,
                0,
                bytemuck::bytes_of(&SmartUniforms {
                    low: params.low,
                    high: params.high,
                    contrast: params.contrast,
                    invert: if params.invert { 1 } else { 0 },
                    source_kind: kind_idx,
                    tile_layer: tile_idx as u32,
                    _pad: [0; 2],
                }),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart_mask.encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("smart_mask.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }
}

/// Convenience: regenerate the active layer's smart mask, no-op if it
/// isn't smart or has no mask. Returns the user-visible status string
/// so the UI can show success / "bake AO first" type errors.
pub fn regenerate_active_smart_mask(
    pipeline: &SmartMaskPipeline,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layer: &Layer,
    mesh_maps: &MeshMaps,
) -> Option<Result<(), String>> {
    let mask = layer.mask.as_ref()?;
    let params = mask.smart.as_ref()?;
    Some(pipeline.regenerate(device, queue, mask, mesh_maps, params))
}
