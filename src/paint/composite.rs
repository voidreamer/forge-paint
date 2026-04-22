//! Compositor — renders the layer stack into the display PaintTarget.
//!
//! Per tile, an MRT render pass writes all 3 channels (base_color,
//! rough_metal, normal) in one draw per layer. Passes clear to the neutral
//! defaults, then OVER-blend each visible layer bottom-to-top.

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::paint::layer::{BlendMode, LayerStack};
use crate::paint::target::{defaults, PaintTarget};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct CompositeUniforms {
    pub opacity: f32,
    pub _pad: [f32; 3],
}

pub struct Compositor {
    /// One pipeline per `BlendMode` — all share the same bind group layout;
    /// only the per-target BlendState differs. Indexed by `BlendMode as usize`.
    pub pipelines: [wgpu::RenderPipeline; 4],
    pub bgl: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    /// 1×1 R8 texture initialised to 1.0, bound for layers without a mask.
    pub dummy_mask_view: wgpu::TextureView,
    // kept alive for the view
    _dummy_mask: wgpu::Texture,
}

impl Compositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                // params (opacity)
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
                // base_color
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
                // rough_metal
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // normal
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
                // mask (R8)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // Blend states — all assume the fragment outputs premultiplied alpha
        // (rgb*a, a). Each mode derives from a different (src_factor, dst_factor)
        // combo that wgpu's fixed-function blend can express without needing
        // ping-pong textures.
        //
        //   Normal   (OVER)     dst' = src + dst*(1-src.a)
        //   Multiply           dst' = dst*src.rgb + dst*(1-src.a)  (src is rgb*a)
        //                       → src_factor=Dst, dst_factor=OneMinusSrcAlpha
        //   Screen             dst' = src + dst*(1-src)
        //                       → src_factor=One, dst_factor=OneMinusSrc
        //   Add                 dst' = src + dst
        //                       → src_factor=One, dst_factor=One
        fn blend_state(mode: BlendMode) -> wgpu::BlendState {
            use wgpu::{BlendComponent, BlendFactor, BlendOperation};
            let (sf, df) = match mode {
                BlendMode::Normal => (BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
                BlendMode::Multiply => (BlendFactor::Dst, BlendFactor::OneMinusSrcAlpha),
                BlendMode::Screen => (BlendFactor::One, BlendFactor::OneMinusSrc),
                BlendMode::Add => (BlendFactor::One, BlendFactor::One),
            };
            wgpu::BlendState {
                color: BlendComponent {
                    src_factor: sf,
                    dst_factor: df,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }
        }

        fn make_pipeline(
            device: &wgpu::Device,
            layout: &wgpu::PipelineLayout,
            shader: &wgpu::ShaderModule,
            blend: wgpu::BlendState,
            label: &str,
        ) -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_composite"),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
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

        let pipelines = [
            make_pipeline(
                device,
                &pipeline_layout,
                &shader,
                blend_state(BlendMode::Normal),
                "composite.pipe.normal",
            ),
            make_pipeline(
                device,
                &pipeline_layout,
                &shader,
                blend_state(BlendMode::Multiply),
                "composite.pipe.multiply",
            ),
            make_pipeline(
                device,
                &pipeline_layout,
                &shader,
                blend_state(BlendMode::Screen),
                "composite.pipe.screen",
            ),
            make_pipeline(
                device,
                &pipeline_layout,
                &shader,
                blend_state(BlendMode::Add),
                "composite.pipe.add",
            ),
        ];

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Dummy 1×1 R8 mask initialised to 1.0 for layers without a mask.
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("composite.dummy_mask"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor {
            label: Some("composite.dummy_mask.view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        // Clear-fill to 1.0 once.
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite.dummy_mask.init"),
            });
            let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite.dummy_mask.init_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dummy_view,
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
            drop(_pass);
            queue.submit(Some(enc.finish()));
        }

        Self {
            pipelines,
            bgl,
            sampler,
            dummy_mask_view: dummy_view,
            _dummy_mask: dummy,
        }
    }

    fn pipeline_for(&self, mode: BlendMode) -> &wgpu::RenderPipeline {
        let idx = match mode {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            BlendMode::Add => 3,
        };
        &self.pipelines[idx]
    }

    /// Composite `stack` into `target`. One render pass per tile; one draw per
    /// visible layer. Appends commands into `encoder`; caller submits.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        stack: &LayerStack,
        target: &PaintTarget,
    ) {
        let tile_count = target.tiles.len();
        if tile_count == 0 {
            return;
        }

        // One uniform buffer per visible layer — written once, reused across
        // every tile in this composite.
        let visible: Vec<&crate::paint::layer::Layer> =
            stack.layers.iter().filter(|l| l.visible).collect();
        let uniform_bufs: Vec<wgpu::Buffer> = visible
            .iter()
            .map(|layer| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("composite.params_buf"),
                    size: std::mem::size_of::<CompositeUniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(
                    &buf,
                    0,
                    bytemuck::bytes_of(&CompositeUniforms {
                        opacity: layer.opacity,
                        _pad: [0.0; 3],
                    }),
                );
                buf
            })
            .collect();

        let base_clear = defaults::base_color_clear();
        let rm_clear = defaults::rough_metal_clear();
        let nm_clear = defaults::normal_clear();

        for t in 0..tile_count {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.base_color_layer_views[t],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(base_clear),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.rough_metal_layer_views[t],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(rm_clear),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.normal_layer_views[t],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(nm_clear),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            // Layers can have different blend modes, so we (re)bind a
            // pipeline per layer inside the same pass. Cheap.
            for (i, layer) in visible.iter().enumerate() {
                pass.set_pipeline(self.pipeline_for(layer.blend_mode));
                let mask_view = layer
                    .mask
                    .as_ref()
                    .map(|m| &m.layer_views[t])
                    .unwrap_or(&self.dummy_mask_view);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite.bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_bufs[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &layer.base_color_layer_views[t],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &layer.rough_metal_layer_views[t],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &layer.normal_layer_views[t],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(mask_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // (end of per-tile loop)
    }

    /// One-shot helper: builds its own encoder and submits. Use this when you
    /// just need a recomposite and aren't already inside a render pass.
    pub fn run_and_submit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        target: &PaintTarget,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite_one_shot"),
        });
        self.run(device, queue, &mut encoder, stack, target);
        queue.submit(Some(encoder.finish()));
    }
}
