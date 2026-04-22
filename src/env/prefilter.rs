//! GGX-prefiltered specular environment: multiple mips of a small equirect
//! where mip 0 is a mirror (roughness=0) and the smallest mip corresponds to
//! the fully rough case (roughness=1). The PBR shader samples with
//! `lod = roughness * (mip_count - 1)` to pick the right blur level.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

pub const PREFILTER_W: u32 = 512;
pub const PREFILTER_H: u32 = 256;
pub const PREFILTER_MIPS: u32 = 6; // 512×256 → 256×128 → 128×64 → 64×32 → 32×16 → 16×8

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct PrefilterParams {
    pub roughness: f32,
    pub _pad: [f32; 3],
}

pub struct PrefilterBaker {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PrefilterBaker {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefilter.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("prefilter.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefilter.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefilter.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefilter.pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_prefilter"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("prefilter.src_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            sampler,
        }
    }

    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_env_view: &wgpu::TextureView,
    ) -> (wgpu::Texture, wgpu::TextureView, u32) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env.prefilter"),
            size: wgpu::Extent3d {
                width: PREFILTER_W,
                height: PREFILTER_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: PREFILTER_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("env.prefilter.view"),
            ..Default::default()
        });

        // One uniform buffer + bind group per mip so the per-pass roughness
        // write has already committed by the time the pass reads it at submit.
        for mip in 0..PREFILTER_MIPS {
            let roughness = mip as f32 / (PREFILTER_MIPS as f32 - 1.0).max(1.0);
            let mip_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("env.prefilter.mip_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prefilter.params"),
                size: std::mem::size_of::<PrefilterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &uniform_buf,
                0,
                bytemuck::bytes_of(&PrefilterParams {
                    roughness,
                    _pad: [0.0; 3],
                }),
            );

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prefilter.bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prefilter.bake_enc"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("prefilter.bake_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &mip_view,
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
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }

        (texture, array_view, PREFILTER_MIPS)
    }
}
