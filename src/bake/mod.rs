//! Mesh-map baking. Phase D.1 starts with world-space normal; position /
//! curvature / AO / thickness follow in later slices. Output textures are
//! `texture_2d_array` D2Array layers matching the asset's UDIM tiles so the
//! PBR shader can index them with the same layer math as painted content.

pub mod integration;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::mesh::{GpuMesh, Vertex};

/// One-shot baked data. Textures live at full resolution once baked; prior
/// to the first bake they're 1×1 so the bind group stays valid.
///
/// world_normal + world_position are baked together by the in-tree
/// `worldnormal.wgsl` MRT pass — they're cheap and the projection brush
/// needs them to be in lockstep. Everything else (`ao`, `curvature`, …)
/// is baked on demand via the vendored texture-baker through
/// `integration::bake_map`, lives as `Option<BakedMap>`, and is `None`
/// until the user clicks the relevant button in the Mesh maps panel.
pub struct MeshMaps {
    pub world_normal: wgpu::Texture,
    pub world_normal_view: wgpu::TextureView,
    /// World-space position sampled at each UV texel. Baked in the same
    /// pass as world_normal via MRT. Consumed by the projection brush
    /// (world → screen → stencil) and future smart-mask generators.
    pub world_position: wgpu::Texture,
    pub world_position_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub baked: bool,

    pub ao: Option<integration::BakedMap>,
    pub curvature: Option<integration::BakedMap>,
    pub thickness: Option<integration::BakedMap>,
    pub height: Option<integration::BakedMap>,
    pub normal: Option<integration::BakedMap>,
    pub bent_normal: Option<integration::BakedMap>,
    pub id: Option<integration::BakedMap>,
    /// 1×1 R8 array of value 1.0, one layer per tile. Bound at slots
    /// expecting a scalar mesh-map (AO) when the user hasn't baked the
    /// real one yet — multiplying by 1.0 in the shader is a no-op so we
    /// avoid feature flags in the PBR pass.
    pub r8_ones_view: wgpu::TextureView,
    /// Keeps the texture alive for the lifetime of the view above.
    _r8_ones_tex: wgpu::Texture,
    /// Monotonic revision the mesh was at when these maps were baked.
    /// Bumped on the viewport whenever the mesh / subdivision / tile
    /// resolution changes; UI compares against this to flag stale maps.
    pub baked_at_revision: u64,
}

impl MeshMaps {
    /// Slot-aware setter — drop the previously baked texture (if any)
    /// and stash the new one. Used by the Mesh maps panel after a
    /// successful bake.
    pub fn set(&mut self, slot: integration::MapKind, baked: integration::BakedMap) {
        match slot {
            integration::MapKind::AmbientOcclusion => self.ao = Some(baked),
            integration::MapKind::Curvature => self.curvature = Some(baked),
            integration::MapKind::Thickness => self.thickness = Some(baked),
            integration::MapKind::Height => self.height = Some(baked),
            integration::MapKind::Normal => self.normal = Some(baked),
            integration::MapKind::BentNormal => self.bent_normal = Some(baked),
            integration::MapKind::Id => self.id = Some(baked),
            // World normal / world position go through the MRT bake
            // path, not the texture-baker integration. Silently ignore
            // here so the panel can use one slot enum across all maps.
            integration::MapKind::WorldNormal | integration::MapKind::Position => {}
        }
    }

    pub fn clear(&mut self, slot: integration::MapKind) {
        match slot {
            integration::MapKind::AmbientOcclusion => self.ao = None,
            integration::MapKind::Curvature => self.curvature = None,
            integration::MapKind::Thickness => self.thickness = None,
            integration::MapKind::Height => self.height = None,
            integration::MapKind::Normal => self.normal = None,
            integration::MapKind::BentNormal => self.bent_normal = None,
            integration::MapKind::Id => self.id = None,
            integration::MapKind::WorldNormal | integration::MapKind::Position => {}
        }
    }

    pub fn slot(&self, slot: integration::MapKind) -> Option<&integration::BakedMap> {
        match slot {
            integration::MapKind::AmbientOcclusion => self.ao.as_ref(),
            integration::MapKind::Curvature => self.curvature.as_ref(),
            integration::MapKind::Thickness => self.thickness.as_ref(),
            integration::MapKind::Height => self.height.as_ref(),
            integration::MapKind::Normal => self.normal.as_ref(),
            integration::MapKind::BentNormal => self.bent_normal.as_ref(),
            integration::MapKind::Id => self.id.as_ref(),
            integration::MapKind::WorldNormal | integration::MapKind::Position => None,
        }
    }
}

impl MeshMaps {
    /// Build a 1×1-per-tile placeholder (cheap dummy, keeps the bind group
    /// valid until the user explicitly bakes).
    pub fn new_empty(device: &wgpu::Device, queue: &wgpu::Queue, tile_count: u32) -> Self {
        let tile_count = tile_count.max(1);
        let world_normal =
            make_array(device, "mesh_maps.world_normal", 1, tile_count, FORMAT);
        let world_normal_view = array_view(&world_normal, "mesh_maps.world_normal.array_view");
        let world_position =
            make_array(device, "mesh_maps.world_position", 1, tile_count, FORMAT);
        let world_position_view =
            array_view(&world_position, "mesh_maps.world_position.array_view");

        // Flat-up (0.5, 0.5, 1.0, 1.0) normal; zeroed position. Preview looks
        // sensible before the first bake, and the projection brush early-outs
        // on (0,0,0) positions via its falloff check.
        let flat_normal = encode_normal(&[0.0, 0.0, 1.0]);
        let zero_pos: [half::f16; 4] = [half::f16::ZERO; 4];
        for layer in 0..tile_count {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &world_normal,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&flat_normal),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &world_position,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&zero_pos),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_maps.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Dummy R8 array of value 1.0 — bound by the PBR shader at slot 9
        // until the user bakes a real AO map. Tiny (tile_count bytes total).
        let r8_ones_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_maps.r8_ones"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: tile_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for layer in 0..tile_count {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &r8_ones_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
        let r8_ones_view = r8_ones_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("mesh_maps.r8_ones.array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        Self {
            world_normal,
            world_normal_view,
            world_position,
            world_position_view,
            sampler,
            baked: false,
            ao: None,
            curvature: None,
            thickness: None,
            height: None,
            normal: None,
            bent_normal: None,
            id: None,
            r8_ones_view,
            _r8_ones_tex: r8_ones_tex,
            baked_at_revision: 0,
        }
    }

    /// View to bind for AO at slot 9 — falls back to the all-ones dummy
    /// when the user hasn't baked it yet.
    pub fn ao_view(&self) -> &wgpu::TextureView {
        self.ao
            .as_ref()
            .map(|b| &b.view)
            .unwrap_or(&self.r8_ones_view)
    }

    /// Run the world-normal bake, producing a full-resolution texture per
    /// UDIM tile. Replaces the prior texture, so the bind group needs a
    /// rebuild from the caller after this.
    pub fn bake(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        baker: &Baker,
        gpu_mesh: &GpuMesh,
        tiles: &[u32],
        resolution: u32,
    ) {
        let tile_count = tiles.len().max(1) as u32;

        let world_normal = make_array(
            device,
            "mesh_maps.world_normal.baked",
            resolution,
            tile_count,
            FORMAT,
        );
        let world_normal_view =
            array_view(&world_normal, "mesh_maps.world_normal.baked.array_view");
        let world_position = make_array(
            device,
            "mesh_maps.world_position.baked",
            resolution,
            tile_count,
            FORMAT,
        );
        let world_position_view =
            array_view(&world_position, "mesh_maps.world_position.baked.array_view");

        // One pass per tile; inside the pass we MRT into normal + position.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh_maps.bake_enc"),
        });
        for (layer, &tile_id) in tiles.iter().enumerate() {
            let (tu, tv) = udim_offset(tile_id);
            let normal_view = world_normal.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mesh_maps.world_normal.baked.layer_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let position_view = world_position.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mesh_maps.world_position.baked.layer_view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_maps.bake.params_buf"),
                size: std::mem::size_of::<BakeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &params_buf,
                0,
                bytemuck::bytes_of(&BakeParams {
                    tile_u: tu as f32,
                    tile_v: tv as f32,
                    _pad: [0.0; 2],
                }),
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh_maps.bake.bg"),
                layout: &baker.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mesh_maps.bake_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.5,
                                    g: 0.5,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &position_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&baker.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(
                    gpu_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
            }
        }
        queue.submit(Some(encoder.finish()));

        self.world_normal = world_normal;
        self.world_normal_view = world_normal_view;
        self.world_position = world_position;
        self.world_position_view = world_position_view;
        self.baked = true;
    }
}

/// Pipeline state for the mesh-map bakes. Currently hosts the world-normal
/// pipeline; later bake types (position / curvature / AO) extend this struct.
pub struct Baker {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl Baker {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bake_worldnormal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("worldnormal.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_maps.bake_bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_maps.bake_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_maps.bake_pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bake"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bake"),
                // MRT: [0] = world normal, [1] = world position.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: None,
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
        });

        Self { pipeline, bgl }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct BakeParams {
    tile_u: f32,
    tile_v: f32,
    _pad: [f32; 2],
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn make_array(
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
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn array_view(tex: &wgpu::Texture, label: &str) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn udim_offset(udim: u32) -> (u32, u32) {
    let n = udim.saturating_sub(1001);
    (n % 10, n / 10)
}

fn encode_normal(n: &[f32; 3]) -> [half::f16; 4] {
    [
        half::f16::from_f32(n[0] * 0.5 + 0.5),
        half::f16::from_f32(n[1] * 0.5 + 0.5),
        half::f16::from_f32(n[2] * 0.5 + 0.5),
        half::f16::from_f32(1.0),
    ]
}
