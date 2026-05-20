use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::lights::{GpuLight, MAX_LIGHTS};
use crate::mesh::Vertex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub ambient_sky: [f32; 4],
    pub ambient_ground: [f32; 4],
    /// Picks what the fragment shader visualises. See `ViewMode::as_u32`.
    pub view_mode: u32,
    /// Which tonemap curve to run after lighting. See `TonemapMode::as_u32`.
    pub tonemap_mode: u32,
    /// Pre-tonemap linear multiplier (= 2^exposure_stops).
    pub exposure: f32,
    /// Multiplier on the IBL contribution. Stays at 1.0 now that the
    /// hard-coded studio rig is gone; kept as a uniform so the
    /// shader can still scale IBL if a downstream feature wants it
    /// (e.g. future ambient-occlusion modulation).
    pub ibl_scale: f32,
    /// Inverse of `view_proj` — used by the skybox vertex shader to
    /// reconstruct a world-space ray direction from clip-space NDC.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Number of valid entries in `lights`. Shader iterates
    /// `0..light_count` and skips the rest.
    pub light_count: u32,
    /// std140 pads `light_count` (a scalar) up to a vec4 before the
    /// `lights` array starts. The three extra u32s keep the CPU and
    /// shader views byte-for-byte aligned.
    pub _pad_lights: [u32; 3],
    /// Dynamic analytic-light array — directional + spot. Driven
    /// from `Viewport::lights`. Zero count by default; user adds via
    /// the Lighting panel's `+ Directional` / `+ Spot` buttons.
    pub lights: [GpuLight; MAX_LIGHTS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonemapMode {
    None,
    Reinhard,
    Aces,
    Filmic,
    /// Khronos PBR Neutral — true-to-color glTF 2024 standard. Best for
    /// material-look judgement (no saturated-blue → cyan shift like ACES).
    Neutral,
    /// AgX — Blender / Filament 2024 default. Filmic shoulder + log
    /// encoding, ships with a default-contrast sigmoid that reads as
    /// "crisp + saturated" without crushing highlights.
    AgX,
}

impl TonemapMode {
    pub fn as_u32(self) -> u32 {
        match self {
            TonemapMode::None => 0,
            TonemapMode::Reinhard => 1,
            TonemapMode::Aces => 2,
            TonemapMode::Filmic => 3,
            TonemapMode::Neutral => 4,
            TonemapMode::AgX => 5,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            TonemapMode::None => "None (clamp)",
            TonemapMode::Reinhard => "Reinhard",
            TonemapMode::Aces => "ACES filmic",
            TonemapMode::Filmic => "Filmic (UC2)",
            TonemapMode::Neutral => "Khronos PBR Neutral",
            TonemapMode::AgX => "AgX",
        }
    }
    pub const ALL: &'static [TonemapMode] = &[
        TonemapMode::None,
        TonemapMode::Reinhard,
        TonemapMode::Aces,
        TonemapMode::Filmic,
        TonemapMode::Neutral,
        TonemapMode::AgX,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Material,
    BaseColor,
    Roughness,
    Metallic,
    Normal,
    Mask,
    WorldNormalBaked,
    Height,
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
            ViewMode::WorldNormalBaked => 6,
            ViewMode::Height => 7,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Material => "Material",
            ViewMode::BaseColor => "Base Color",
            ViewMode::Roughness => "Roughness",
            ViewMode::Metallic => "Metallic",
            ViewMode::Normal => "Normal (tangent)",
            ViewMode::Mask => "Mask",
            ViewMode::WorldNormalBaked => "World Normal (baked)",
            ViewMode::Height => "Height",
        }
    }

    pub const ALL: &'static [ViewMode] = &[
        ViewMode::Material,
        ViewMode::BaseColor,
        ViewMode::Roughness,
        ViewMode::Metallic,
        ViewMode::Normal,
        ViewMode::Mask,
        ViewMode::WorldNormalBaked,
        ViewMode::Height,
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
    /// Rgba16Float HDR intermediate — PBR writes here, the post pass reads.
    /// Moving tonemap out of pbr.wgsl means we can also layer AO and bloom
    /// into the post pass later.
    pub hdr: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    /// Tonemapped LDR intermediate — the post pass writes here and FXAA
    /// reads it before producing the final viewport texture.
    pub ldr: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
}

/// Target format for the HDR intermediate. Must match what the post pass
/// samples from.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Target format for the LDR intermediate (post → FXAA chain) and the
/// final egui viewport texture.
pub const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

impl Renderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let color_format = HDR_FORMAT;
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
                    // Vertex shader reads material.params.w + tile_ids
                    // for displacement lookup, so this has to be visible
                    // from VERTEX too.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: base_color
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
                // 2: roughness (R8 D2Array)
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
                // 3: metallic (R8 D2Array)
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
                // 4: normal
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
                // 5: active layer mask (R8 D2Array) — dummy all-1.0 when
                // the layer has no mask.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6: shared sampler — VERTEX_FRAGMENT because the vertex
                // shader reads the displacement map.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 7: world_normal mesh map (Rgba16Float D2Array).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 8: displacement (Rg16Float D2Array).
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 9: AO mesh map (R8Unorm D2Array). 1×1 dummy of value
                // 1.0 when the user hasn't baked AO yet — multiplying by
                // 1.0 is a no-op so the shader doesn't need a branch.
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 10: baked tangent-space normal map (Rgba8Unorm D2Array).
                // 1×1 dummy encoded as flat (0.5, 0.5, 1.0, 1.0) so the
                // blend in the shader is a no-op until the user bakes it.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
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
            hdr: None,
            ldr: None,
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
                // TEXTURE_BINDING so the post pass (SSAO in E.4) can sample
                // depth without a separate depth-prepass.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.depth = Some((tex, view, [w, h]));
        }
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth.as_ref().expect("ensure_depth first").1
    }

    pub fn ensure_hdr(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let need = match &self.hdr {
            Some((_, _, s)) => s[0] != w || s[1] != h,
            None => true,
        };
        if need {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hdr_color"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.hdr = Some((tex, view, [w, h]));
        }
    }

    pub fn hdr_view(&self) -> &wgpu::TextureView {
        &self.hdr.as_ref().expect("ensure_hdr first").1
    }

    pub fn ensure_ldr(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let need = match &self.ldr {
            Some((_, _, s)) => s[0] != w || s[1] != h,
            None => true,
        };
        if need {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ldr_color"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.ldr = Some((tex, view, [w, h]));
        }
    }

    pub fn ldr_view(&self) -> &wgpu::TextureView {
        &self.ldr.as_ref().expect("ensure_ldr first").1
    }

    pub fn write_frame(&self, queue: &wgpu::Queue, u: &FrameUniforms) {
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(u));
    }
}
