use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::mesh::Vertex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub ambient_sky: [f32; 4],
    pub ambient_ground: [f32; 4],
    /// Picks what the fragment shader visualises. See `ViewMode::as_u32`.
    pub view_mode: u32,
    pub _pad: [u32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Material,
    BaseColor,
    Roughness,
    Metallic,
    Normal,
    Mask,
}

impl ViewMode {
    pub fn as_u32(self) -> u32 {
        match self {
            ViewMode::Material => 0,
            ViewMode::BaseColor => 1,
            ViewMode::Roughness => 2,
            ViewMode::Metallic => 3,
            ViewMode::Normal => 4,
            ViewMode::Mask => 5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Material => "Material",
            ViewMode::BaseColor => "Base Color",
            ViewMode::Roughness => "Roughness",
            ViewMode::Metallic => "Metallic",
            ViewMode::Normal => "Normal",
            ViewMode::Mask => "Mask",
        }
    }

    pub const ALL: &'static [ViewMode] = &[
        ViewMode::Material,
        ViewMode::BaseColor,
        ViewMode::Roughness,
        ViewMode::Metallic,
        ViewMode::Normal,
        ViewMode::Mask,
    ];
}

pub struct Renderer {
    pub pipeline: wgpu::RenderPipeline,
    pub frame_bgl: wgpu::BindGroupLayout,
    /// Material bind group layout — used by `PaintTarget` to construct its bind group.
    pub material_bgl: wgpu::BindGroupLayout,
    pub env_bgl: wgpu::BindGroupLayout,
    pub frame_buf: wgpu::Buffer,
    pub frame_bg: wgpu::BindGroup,
    pub depth: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbr.wgsl").into()),
        });

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_bgl"),
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

        // Group 1: material uniforms + 3 texture_2d_array + sampler
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Active layer mask (R8 D2Array) — shown in Mask view mode,
                // falls back to a dummy all-1.0 array when the layer has no mask.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_buf"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });

        let env_bgl = crate::env::env_bgl(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_pl"),
            bind_group_layouts: &[&frame_bgl, &material_bgl, &env_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr_pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            frame_bgl,
            material_bgl,
            env_bgl,
            frame_buf,
            frame_bg,
            depth: None,
        }
    }

    pub fn ensure_depth(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let need = match &self.depth {
            Some((_, _, s)) => s[0] != w || s[1] != h,
            None => true,
        };
        if need {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.depth = Some((tex, view, [w, h]));
        }
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth.as_ref().expect("ensure_depth first").1
    }

    pub fn write_frame(&self, queue: &wgpu::Queue, u: &FrameUniforms) {
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(u));
    }
}
