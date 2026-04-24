use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BrushUniforms {
    pub color: [f32; 4],
    pub center_uv: [f32; 2],
    pub radius: f32,
    pub hardness: f32,
    /// When 1, the fragment shader skips the distance check — the stamp
    /// covers the whole UV tile at full opacity. Used by the Fill tool.
    pub uniform_fill: u32,
    pub _pad: [u32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintChannel {
    BaseColor,
    Roughness,
    Metallic,
    Mask,
}

pub struct BrushPipeline {
    /// Writes to base_color (Rgba8UnormSrgb), ColorWrites::ALL.
    pub base_color: wgpu::RenderPipeline,
    /// Writes the G channel of rough_metal (Rgba8Unorm) only.
    pub roughness: wgpu::RenderPipeline,
    /// Writes the B channel of rough_metal (Rgba8Unorm) only.
    pub metallic: wgpu::RenderPipeline,
    /// Writes an R8Unorm layer mask.
    pub mask: wgpu::RenderPipeline,

    pub bgl: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl BrushPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("brush.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_uniform_buf"),
            size: std::mem::size_of::<BrushUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let base_color = make_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::ColorWrites::ALL,
            "brush_pipe_base_color",
        );
        let roughness = make_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::ColorWrites::GREEN,
            "brush_pipe_roughness",
        );
        let metallic = make_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::ColorWrites::BLUE,
            "brush_pipe_metallic",
        );
        // Mask target is R8Unorm — only R exists, so ALL = R.
        let mask = make_pipeline(
            device,
            &shader,
            &pipeline_layout,
            wgpu::TextureFormat::R8Unorm,
            wgpu::ColorWrites::ALL,
            "brush_pipe_mask",
        );

        Self {
            base_color,
            roughness,
            metallic,
            mask,
            bgl,
            uniform_buf,
            bind_group,
        }
    }

    pub fn pipeline_for(&self, channel: PaintChannel) -> &wgpu::RenderPipeline {
        match channel {
            PaintChannel::BaseColor => &self.base_color,
            PaintChannel::Roughness => &self.roughness,
            PaintChannel::Metallic => &self.metallic,
            PaintChannel::Mask => &self.mask,
        }
    }

    /// Stamp a single brush dab onto `layer_view`. Caller is responsible for
    /// passing a view whose format matches the channel's pipeline.
    ///
    /// `tile_resolution` is used to compute a scissor rect tight around the
    /// brush footprint — without it the fragment shader runs over the
    /// entire tile (millions of fragments) and `discard`s 99%+ of them.
    /// This is the main reason 4K / 8K painting was slow.
    pub fn stamp(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer_view: &wgpu::TextureView,
        channel: PaintChannel,
        uniforms: &BrushUniforms,
        tile_resolution: u32,
    ) {
        // Fill tool covers the whole tile; radial stamp gets a tight box.
        let res = tile_resolution as f32;
        let scissor = if uniforms.uniform_fill == 1 {
            Some((0u32, 0u32, tile_resolution, tile_resolution))
        } else {
            let cx = uniforms.center_uv[0] * res;
            let cy = uniforms.center_uv[1] * res;
            // Add 1-pixel margin for rasterization coverage at the edges.
            let r = uniforms.radius * res + 1.0;
            let x0 = (cx - r).max(0.0).floor() as u32;
            let y0 = (cy - r).max(0.0).floor() as u32;
            let x1 = (cx + r).min(res).ceil() as u32;
            let y1 = (cy + r).min(res).ceil() as u32;
            let w = x1.saturating_sub(x0);
            let h = y1.saturating_sub(y0);
            if w == 0 || h == 0 {
                return; // brush footprint lies outside the tile
            }
            Some((x0, y0, w, h))
        };

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brush_stamp_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: layer_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        if let Some((x, y, w, h)) = scissor {
            pass.set_scissor_rect(x, y, w, h);
        }
        pass.set_pipeline(self.pipeline_for(channel));
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    write_mask: wgpu::ColorWrites,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_stamp"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
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
                }),
                write_mask,
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
}
