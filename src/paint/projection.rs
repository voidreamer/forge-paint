//! Projection brush — writes a stencil image projected through the
//! current camera into the active layer's base_color. Reads the baked
//! world-position mesh map to find where each tile texel sits in world
//! space, then projects back to screen to sample the stencil. Scissored
//! to the brush's UV footprint same as the regular radial brush.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct ProjBrushUniforms {
    pub view_proj: [[f32; 4]; 4],
    /// Brush center in clip-space NDC ([-1, 1] with +Y up).
    pub center_screen: [f32; 2],
    /// Brush radius in NDC units.
    pub radius_screen: f32,
    pub opacity: f32,
    pub hardness: f32,
    /// Viewport aspect (width / height) so the footprint stays circular.
    pub aspect: f32,
    /// Stencil transform — NDC offset, scale, rotation (pre-computed
    /// cos/sin to avoid redundant trig on every fragment).
    pub stencil_offset: [f32; 2],
    pub stencil_scale: f32,
    pub stencil_cos_rot: f32,
    pub stencil_sin_rot: f32,
    /// Stencil's own width/height ratio — used to correct its display
    /// aspect so a wide photo doesn't stretch vertically.
    pub stencil_aspect: f32,
    /// 0 = base color projection (RGBA). 1 = displacement projection
    /// (luminance → height × coverage, coverage packed into RG).
    pub mode: u32,
    /// Padding to 128 bytes. WGSL rounds the uniform struct's size up
    /// to the next 16-byte multiple, so `mode` alone (4 B after 112 B
    /// of real data) would leave the Rust side 8 B short.
    pub _pad: [f32; 3],
}

pub struct ProjectionBrushPipeline {
    /// Targets Rgba8UnormSrgb — projected base color.
    pub pipeline: wgpu::RenderPipeline,
    /// Targets Rg16Float — projected displacement (R=h·cov, G=cov).
    pub displacement_pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
}

impl ProjectionBrushPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projection.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("projection.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("projection.bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("projection.uniform_buf"),
            size: std::mem::size_of::<ProjBrushUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("projection.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("projection.pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let make_pipe = |format: wgpu::TextureFormat, label: &str| -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_project"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_project"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
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
            })
        };
        let pipeline = make_pipe(wgpu::TextureFormat::Rgba8UnormSrgb, "projection.pipe");
        let displacement_pipeline =
            make_pipe(wgpu::TextureFormat::Rg16Float, "projection.pipe.disp");

        Self {
            pipeline,
            displacement_pipeline,
            bgl,
            uniform_buf,
            sampler,
        }
    }

    /// Stamp a projection-paint dab. `tile_resolution` and the brush's
    /// UV bounds drive the scissor, same pattern as `BrushPipeline::stamp`.
    pub fn stamp(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        position_view: &wgpu::TextureView,
        stencil_view: &wgpu::TextureView,
        uniforms: &ProjBrushUniforms,
        tile_resolution: u32,
        brush_center_uv: [f32; 2],
        brush_radius_uv: f32,
    ) {
        let res = tile_resolution as f32;
        let cx = brush_center_uv[0] * res;
        let cy = brush_center_uv[1] * res;
        let r = brush_radius_uv * res + 1.0;
        let x0 = (cx - r).max(0.0).floor() as u32;
        let y0 = (cy - r).max(0.0).floor() as u32;
        let x1 = (cx + r).min(res).ceil() as u32;
        let y1 = (cy + r).min(res).ceil() as u32;
        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);
        if w == 0 || h == 0 {
            return;
        }

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("projection.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(stencil_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("projection_stamp_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_scissor_rect(x0, y0, w, h);
        // Pick the pipeline whose target format matches the bound view.
        let pipe = if uniforms.mode == 1 {
            &self.displacement_pipeline
        } else {
            &self.pipeline
        };
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
