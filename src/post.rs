//! Post-process chain — runs after the PBR pass, reads the HDR intermediate,
//! applies exposure + tonemap, writes to the final viewport color texture.
//! Future passes (SSAO, bloom) slot in here before the tonemap step.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct PostUniforms {
    pub exposure: f32,
    pub view_mode: u32,
    pub tonemap_mode: u32,
    /// Post-tonemap multiplier on the deviation from mid-gray (luminance
    /// pivot 0.5). 1.0 = identity. Drives the "punchy vs flat" knob — most
    /// of the perceived "Marmoset crispness" lives here.
    pub contrast: f32,
    /// Post-tonemap saturation around per-pixel luminance. 1.0 = identity,
    /// 0 = grayscale, > 1 boosts color. Cheap & high-impact.
    pub saturation: f32,
    /// "Clarity" / unsharp mask amount. 0 = off; small positive values
    /// (0.1..0.3) sharpen edges via a 5-tap box blur and add to the
    /// center sample. Reads as micro-contrast / "sharpness".
    pub clarity: f32,
    /// Pixel size (1/width, 1/height) — needed by the clarity unsharp
    /// kernel to sample neighbors at exact texel offsets.
    pub texel_size: [f32; 2],
}

pub struct PostPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
}

impl PostPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.bgl"),
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
            label: Some("post.uniform_buf"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post.pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_post"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_post"),
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
        }
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, u: &PostUniforms) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(u));
    }
}
