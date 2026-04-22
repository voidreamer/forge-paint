use eframe::egui;
use egui_wgpu::wgpu;
use glam::Vec2;

use crate::accel::MeshAccel;
use crate::camera::OrbitCamera;
use crate::mesh::{CpuMesh, GpuMesh};
use crate::paint::{udim, BrushPipeline, BrushUniforms, PaintChannel, PaintTarget};
use crate::pick;
use crate::render::{FrameUniforms, Renderer, ViewMode};

pub struct Viewport {
    renderer: Renderer,
    brush_pipeline: BrushPipeline,
    color: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    egui_tex_id: Option<egui::TextureId>,

    mesh: GpuMesh,
    cpu_mesh: CpuMesh,
    accel: MeshAccel,
    paint_target: PaintTarget,

    pub camera: OrbitCamera,

    pub brush: BrushState,

    // Material factor tweakers (multiply sampled texture values)
    pub base_color_factor: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,

    // Lighting
    pub light_intensity: f32,
    pub light_dir: [f32; 3],

    pub view_mode: ViewMode,

    pub tile_resolution: u32,

    /// Last successful pick this frame, if any — surfaced so the UI can show
    /// the cursor UV / tile.
    pub last_hit_uv: Option<[f32; 2]>,
    pub last_hit_tile: Option<u32>,

    /// Screen position of the last successful paint stamp, used to interpolate
    /// stamps between frames so fast drags don't leave gaps.
    last_paint_pos: Option<egui::Pos2>,
}

pub struct BrushState {
    pub channel: PaintChannel,
    pub color_srgb: [f32; 3], // base_color
    pub value: f32,           // 0..1, used for Roughness / Metallic
    pub radius: f32,          // in UV units (local to a tile)
    pub hardness: f32,        // 0 soft, 1 hard
    pub opacity: f32,         // 0..1
}

impl Default for BrushState {
    fn default() -> Self {
        Self {
            channel: PaintChannel::BaseColor,
            color_srgb: [0.95, 0.2, 0.2],
            value: 0.5,
            radius: 0.04,
            hardness: 0.4,
            opacity: 1.0,
        }
    }
}

impl Viewport {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cpu: &CpuMesh) -> Self {
        let renderer = Renderer::new(device, wgpu::TextureFormat::Bgra8UnormSrgb);
        let brush_pipeline = BrushPipeline::new(device);
        let gpu = GpuMesh::from_cpu(device, cpu);
        let tile_resolution = std::env::var("FORGE_PAINT_RESOLUTION")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|r| [1024u32, 2048, 4096, 8192].contains(r))
            .unwrap_or(2048);
        let paint_target = PaintTarget::new(
            device,
            queue,
            &renderer.material_bgl,
            cpu,
            tile_resolution,
        );

        let mut camera = OrbitCamera::default();
        camera.target = gpu.center;
        camera.distance = (gpu.radius * 2.5).max(1.5);

        let accel = MeshAccel::build(cpu);
        Self {
            renderer,
            brush_pipeline,
            color: None,
            egui_tex_id: None,
            mesh: gpu,
            cpu_mesh: cpu.clone(),
            accel,
            paint_target,
            camera,
            brush: BrushState::default(),
            base_color_factor: [1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            light_intensity: 3.0,
            light_dir: [-0.4, -1.0, -0.3],
            view_mode: ViewMode::Material,
            tile_resolution,
            last_hit_uv: None,
            last_hit_tile: None,
            last_paint_pos: None,
        }
    }

    pub fn set_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cpu: &CpuMesh) {
        let gpu = GpuMesh::from_cpu(device, cpu);
        self.camera.target = gpu.center;
        self.camera.distance = (gpu.radius * 2.5).max(1.5);
        self.mesh = gpu;
        self.cpu_mesh = cpu.clone();
        self.accel = MeshAccel::build(cpu);
        self.paint_target = PaintTarget::new(
            device,
            queue,
            &self.renderer.material_bgl,
            cpu,
            self.tile_resolution,
        );
        self.last_hit_uv = None;
        self.last_hit_tile = None;
    }

    pub fn tiles(&self) -> &[u32] {
        &self.paint_target.tiles
    }

    pub fn tile_resolution(&self) -> u32 {
        self.paint_target.resolution
    }

    pub fn paint_target(&self) -> &PaintTarget {
        &self.paint_target
    }

    /// Rebuild the paint target at a new per-tile resolution. Painted content
    /// is discarded — callers should warn the user first.
    pub fn set_tile_resolution(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
    ) {
        self.tile_resolution = resolution;
        self.paint_target = PaintTarget::new(
            device,
            queue,
            &self.renderer.material_bgl,
            &self.cpu_mesh,
            resolution,
        );
    }

    /// Approximate VRAM used by the paint target in bytes.
    /// 3 channels × N tiles × res² × 4 bytes per texel.
    pub fn paint_target_vram_bytes(&self) -> u64 {
        let res = self.paint_target.resolution as u64;
        let tiles = self.paint_target.tiles.len() as u64;
        3 * tiles * res * res * 4
    }

    pub fn show(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let available = ui.available_size();
        let (rect, response) =
            ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let w = (rect.width() as u32).max(1);
        let h = (rect.height() as u32).max(1);

        let Some(render_state) = frame.wgpu_render_state() else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No GPU available",
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
            return;
        };

        // Paint input detection — plain LMB (no modifiers) drag or click.
        let modifiers = ui.ctx().input(|i| i.modifiers);
        let no_mods = !modifiers.ctrl
            && !modifiers.mac_cmd
            && !modifiers.command
            && !modifiers.shift
            && !modifiers.alt;
        let primary_active = response.dragged_by(egui::PointerButton::Primary)
            || response.clicked_by(egui::PointerButton::Primary);
        let paint_pos = if primary_active && no_mods {
            response.interact_pointer_pos()
        } else {
            None
        };

        let tex_id = {
            let mut egui_renderer = render_state.renderer.write();
            self.ensure_color(&render_state.device, &mut egui_renderer, w, h);
            self.renderer.ensure_depth(&render_state.device, w, h);

            let aspect = w as f32 / h as f32;
            let view_proj = self.camera.view_proj(aspect);
            let eye = self.camera.eye();

            self.renderer.write_frame(
                &render_state.queue,
                &FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    camera_pos: [eye.x, eye.y, eye.z, 1.0],
                    light_dir: [self.light_dir[0], self.light_dir[1], self.light_dir[2], 0.0],
                    light_color: [1.0, 0.98, 0.95, self.light_intensity],
                    ambient_sky: [0.35, 0.45, 0.55, 1.0],
                    ambient_ground: [0.08, 0.07, 0.06, 1.0],
                    view_mode: self.view_mode.as_u32(),
                    _pad: [0; 3],
                },
            );
            self.paint_target.update_material_factors(
                &render_state.queue,
                [
                    self.base_color_factor[0],
                    self.base_color_factor[1],
                    self.base_color_factor[2],
                    1.0,
                ],
                self.metallic_factor,
                self.roughness_factor,
                self.normal_scale,
            );

            // Try to pick under the cursor (even if not painting — drives UI readout).
            let hover_pos = response
                .hover_pos()
                .or_else(|| ui.ctx().input(|i| i.pointer.hover_pos()));
            let hover_hit = hover_pos.and_then(|p| {
                if !rect.contains(p) {
                    return None;
                }
                let (orig, dir) = pick::screen_to_ray(p, rect, view_proj, eye);
                pick::pick(&self.cpu_mesh, orig, dir)
            });
            if let Some(h) = hover_hit {
                self.last_hit_uv = Some(h.uv.to_array());
                self.last_hit_tile = Some(udim::tile_id(h.uv.to_array()));
            } else {
                self.last_hit_uv = None;
                self.last_hit_tile = None;
            }

            // Interpolate stamp positions in screen space between last frame's
            // cursor and this frame's, at ~2 pixel spacing, so fast drags don't
            // leave gaps. Cap step count as a runaway guard.
            let stamp_positions: Vec<egui::Pos2> = match (paint_pos, self.last_paint_pos) {
                (Some(cur), Some(prev)) => {
                    const STEP_PX: f32 = 2.0;
                    const MAX_STEPS: u32 = 128;
                    let delta = cur - prev;
                    let distance = delta.length();
                    let steps = ((distance / STEP_PX).ceil() as u32).clamp(1, MAX_STEPS);
                    (1..=steps)
                        .map(|i| prev + delta * (i as f32 / steps as f32))
                        .collect()
                }
                (Some(cur), None) => vec![cur],
                (None, _) => Vec::new(),
            };

            let strokes: Vec<(u32, Vec2)> = stamp_positions
                .iter()
                .filter_map(|p| {
                    let (orig, dir) = pick::screen_to_ray(*p, rect, view_proj, eye);
                    let hit = pick::pick(&self.cpu_mesh, orig, dir)?;
                    let tile = udim::tile_id(hit.uv.to_array());
                    let layer = self.paint_target.layer_for_tile(tile)?;
                    let local_uv =
                        Vec2::new(hit.uv.x - hit.uv.x.floor(), hit.uv.y - hit.uv.y.floor());
                    Some((layer, local_uv))
                })
                .collect();

            // Reset when paint_pos becomes None (button released or modifier
            // held) so the next stroke starts fresh instead of sweeping back.
            self.last_paint_pos = paint_pos;

            let mut encoder = render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forge_paint_enc"),
                });

            if !strokes.is_empty() {
                let channel = self.brush.channel;
                // Brush color components: either linear-sRGB base color or a
                // grayscale `value` triplet for roughness/metallic — the
                // pipeline's write-mask selects which channels land.
                let color_comp = match channel {
                    PaintChannel::BaseColor => {
                        let lin = self.brush.color_linear();
                        [lin[0], lin[1], lin[2]]
                    }
                    PaintChannel::Roughness | PaintChannel::Metallic => {
                        let v = self.brush.value;
                        [v, v, v]
                    }
                };
                for (layer, local_uv) in &strokes {
                    let uniforms = BrushUniforms {
                        color: [color_comp[0], color_comp[1], color_comp[2], self.brush.opacity],
                        center_uv: local_uv.to_array(),
                        radius: self.brush.radius,
                        hardness: self.brush.hardness,
                    };
                    let layer_view = match channel {
                        PaintChannel::BaseColor => {
                            &self.paint_target.base_color_layer_views[*layer as usize]
                        }
                        PaintChannel::Roughness | PaintChannel::Metallic => {
                            &self.paint_target.rough_metal_layer_views[*layer as usize]
                        }
                    };
                    self.brush_pipeline.stamp(
                        &render_state.queue,
                        &mut encoder,
                        layer_view,
                        channel,
                        &uniforms,
                    );
                }
            }

            // PBR render pass
            {
                let color_view = &self.color.as_ref().unwrap().1;
                let depth_view = self.renderer.depth_view();

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("forge_paint_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.07,
                                g: 0.07,
                                b: 0.09,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_pipeline(&self.renderer.pipeline);
                pass.set_bind_group(0, &self.renderer.frame_bg, &[]);
                pass.set_bind_group(1, &self.paint_target.material_bg, &[]);
                pass.set_vertex_buffer(0, self.mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(self.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.mesh.index_count, 0, 0..1);
            }
            render_state.queue.submit(Some(encoder.finish()));
            self.egui_tex_id.unwrap()
        };

        // Paint the offscreen color texture into the allocated rect.
        ui.painter().image(
            tex_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Camera nav
        let scroll_dy = if response.hovered() {
            ui.input(|i| i.smooth_scroll_delta.y)
        } else {
            0.0
        };
        if self.camera.handle_input(&response, scroll_dy) {
            ui.ctx().request_repaint();
        }

        // Keep the redraw going while painting so drags are smooth.
        if paint_pos.is_some() {
            ui.ctx().request_repaint();
        }
    }

    fn ensure_color(
        &mut self,
        device: &wgpu::Device,
        egui_renderer: &mut egui_wgpu::Renderer,
        w: u32,
        h: u32,
    ) {
        let need = match &self.color {
            Some((_, _, s)) => s[0] != w || s[1] != h,
            None => true,
        };
        if !need {
            return;
        }
        if let Some(id) = self.egui_tex_id.take() {
            egui_renderer.free_texture(&id);
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let id = egui_renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear);
        self.color = Some((tex, view, [w, h]));
        self.egui_tex_id = Some(id);
    }
}

impl BrushState {
    pub fn color_linear(&self) -> [f32; 3] {
        [
            srgb_to_linear(self.color_srgb[0]),
            srgb_to_linear(self.color_srgb[1]),
            srgb_to_linear(self.color_srgb[2]),
        ]
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c < 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
