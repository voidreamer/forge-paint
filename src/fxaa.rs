//! FXAA post-process anti-aliasing pass. Runs after the post tonemap,
//! reads the LDR intermediate, writes the final viewport texture.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct FxaaUniforms {
    pub enabled: u32,
    /// CAS-style adaptive sharpen amount applied after FXAA. 0 = off,
    /// 0.4 ≈ FidelityFX default. Keeps tight specular highlights crisp
    /// without the ringing of a naive unsharp mask.
    pub sharpen: f32,
    /// 1-LSB dither amount applied before the sRGB encode in the
    /// swapchain. Kills 8-bit gradient banding which reads as "fog" or
    /// "washed" — set 1.0 for full dither, 0 to disable.
    pub dither: f32,
    pub _pad0: f32,
}

pub struct FxaaPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    pub enabled: bool,
    pub sharpen: f32,
    pub dither: f32,
}

impl FxaaPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fxaa.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa.bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            label: Some("fxaa.uniform_buf"),
            size: std::mem::size_of::<FxaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fxaa.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa.pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fxaa"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fxaa"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
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

        Self {
            pipeline,
            bgl,
            uniform_buf,
            sampler,
            enabled: true,
            sharpen: 0.40,
            dither: 1.0,
        }
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue) {
        let u = FxaaUniforms {
            enabled: if self.enabled { 1 } else { 0 },
            sharpen: self.sharpen,
            dither: self.dither,
            _pad0: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }
}
