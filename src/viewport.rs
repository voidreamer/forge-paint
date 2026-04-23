use eframe::egui;
use egui_wgpu::wgpu;
use glam::Vec2;

use crate::accel::MeshAccel;
use crate::bake::{Baker, MeshMaps};
use crate::camera::OrbitCamera;
use crate::env::{
    BrdfLut, Environment, EnvUniforms, IrradianceBaker, PrefilterBaker, SkyboxPipeline,
};
use crate::mesh::{CpuMesh, GpuMesh};
use crate::paint::{
    target::MaterialUniforms, udim, BrushPipeline, BrushUniforms, Compositor, Layer, LayerStack,
    PaintChannel, PaintTarget,
};
use crate::pick;
use crate::render::{FrameUniforms, Renderer, TonemapMode, ViewMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Paint,
    Erase,
    Fill,
    Eyedropper,
}

impl Tool {
    pub fn label(self) -> &'static str {
        match self {
            Tool::Paint => "Paint",
            Tool::Erase => "Erase",
            Tool::Fill => "Fill",
            Tool::Eyedropper => "Eyedropper",
        }
    }
    pub fn shortcut(self) -> &'static str {
        match self {
            Tool::Paint => "B",
            Tool::Erase => "E",
            Tool::Fill => "G",
            Tool::Eyedropper => "I",
        }
    }
}

pub struct Viewport {
    renderer: Renderer,
    brush_pipeline: BrushPipeline,
    compositor: Compositor,
    color: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    egui_tex_id: Option<egui::TextureId>,

    mesh: GpuMesh,
    cpu_mesh: CpuMesh,
    accel: MeshAccel,
    /// Composited display textures — this is what the PBR shader samples.
    paint_target: PaintTarget,
    /// The paint stack. Brushes stamp into `layers[active]`; the compositor
    /// flattens the stack into `paint_target` after each stamp batch.
    pub layer_stack: LayerStack,

    /// Material uniform buffer (factors + tile table). Rebuilt on factor
    /// changes via `queue.write_buffer`.
    material_buf: wgpu::Buffer,

    pub env: Environment,
    pub env_intensity: f32,
    pub env_rotation_y: f32,
    pub env_skybox_visible: bool,
    /// Baked once per device at startup; shared across all Environments.
    pub brdf_lut: BrdfLut,
    pub irradiance_baker: IrradianceBaker,
    pub prefilter_baker: PrefilterBaker,
    skybox: SkyboxPipeline,
    pub mesh_maps: MeshMaps,
    baker: Baker,

    pub camera: OrbitCamera,

    pub brush: BrushState,
    pub tool: Tool,

    // Material factor tweakers (multiply sampled texture values)
    pub base_color_factor: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,

    // Lighting
    pub light_intensity: f32,
    pub light_dir: [f32; 3],

    pub view_mode: ViewMode,

    pub tonemap_mode: TonemapMode,
    /// Exposure compensation in stops (−∞..∞ in principle; UI caps at ±4).
    /// Shader receives `2^stops` as a linear pre-tonemap multiplier.
    pub exposure_stops: f32,

    pub tile_resolution: u32,

    /// Last successful pick this frame, if any — surfaced so the UI can show
    /// the cursor UV / tile.
    pub last_hit_uv: Option<[f32; 2]>,
    pub last_hit_tile: Option<u32>,

    /// Screen position of the last successful paint stamp, used to interpolate
    /// stamps between frames so fast drags don't leave gaps.
    last_paint_pos: Option<egui::Pos2>,

    /// Stroke-level undo / redo.
    undo_stack: crate::undo::UndoStack,

    /// egui TextureIds for each layer's tile-0 base_color view, used as the
    /// layer-row thumbnail. Parallel to `layer_stack.layers`.
    pub layer_thumb_cache: Vec<Option<egui::TextureId>>,
}

pub struct BrushState {
    pub channel: PaintChannel,
    pub color_srgb: [f32; 3], // base_color
    pub value: f32,           // 0..1, used for Roughness / Metallic / Mask
    pub radius: f32,          // in UV units (local to a tile)
    pub hardness: f32,        // 0 soft, 1 hard
    pub opacity: f32,         // 0..1
    /// When true and the active layer has a mask, paint routes to the mask.
    pub mask_edit: bool,
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
            mask_edit: false,
        }
    }
}

impl Viewport {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cpu: &CpuMesh) -> Self {
        let renderer = Renderer::new(device, wgpu::TextureFormat::Bgra8UnormSrgb);
        let brush_pipeline = BrushPipeline::new(device);
        let compositor = Compositor::new(device, queue);
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
        let layer_stack = LayerStack::new_with_initial_layer(
            device,
            queue,
            tile_resolution,
            paint_target.tiles.len() as u32,
        );
        compositor.run_and_submit(device, queue, &layer_stack, &paint_target);

        let material_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport.material_buf"),
            size: std::mem::size_of::<MaterialUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let brdf_lut = BrdfLut::new(device, queue);
        let irradiance_baker = IrradianceBaker::new(device);
        let prefilter_baker = PrefilterBaker::new(device);
        let env = Environment::new_procedural(
            device,
            queue,
            &brdf_lut,
            &irradiance_baker,
            &prefilter_baker,
        );
        let skybox = SkyboxPipeline::new(
            device,
            &renderer.frame_bgl,
            &renderer.material_bgl,
            &renderer.env_bgl,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth32Float,
        );

        let mesh_maps = MeshMaps::new_empty(device, queue, paint_target.tiles.len() as u32);
        let baker = Baker::new(device);

        let mut camera = OrbitCamera::default();
        camera.target = gpu.center;
        camera.distance = (gpu.radius * 2.5).max(1.5);

        let accel = MeshAccel::build(cpu);
        Self {
            renderer,
            brush_pipeline,
            compositor,
            color: None,
            egui_tex_id: None,
            mesh: gpu,
            cpu_mesh: cpu.clone(),
            accel,
            paint_target,
            layer_stack,
            material_buf,
            env,
            env_intensity: 1.0,
            env_rotation_y: 0.0,
            env_skybox_visible: false,
            brdf_lut,
            irradiance_baker,
            prefilter_baker,
            skybox,
            mesh_maps,
            baker,
            camera,
            brush: BrushState::default(),
            tool: Tool::Paint,
            base_color_factor: [1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            light_intensity: 3.0,
            light_dir: [-0.4, -1.0, -0.3],
            view_mode: ViewMode::Material,
            // ArmorPaint defaults to Filmic (Hable UC2) — reads as less crushed
            // than ACES and matches the reference painter's look.
            tonemap_mode: TonemapMode::Filmic,
            exposure_stops: 0.0,
            tile_resolution,
            last_hit_uv: None,
            last_hit_tile: None,
            last_paint_pos: None,
            undo_stack: crate::undo::UndoStack::default(),
            layer_thumb_cache: Vec::new(),
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
        self.layer_stack = LayerStack::new_with_initial_layer(
            device,
            queue,
            self.tile_resolution,
            self.paint_target.tiles.len() as u32,
        );
        self.compositor
            .run_and_submit(device, queue, &self.layer_stack, &self.paint_target);
        self.last_hit_uv = None;
        self.last_hit_tile = None;
        // Prior undo history references textures from the old layer stack —
        // those are invalid now that we rebuilt it. Drop them.
        self.undo_stack.clear();
        // Reset mesh maps to a neutral placeholder for the new mesh/tile set.
        self.mesh_maps =
            MeshMaps::new_empty(device, queue, self.paint_target.tiles.len() as u32);
        // Thumbnail TextureIds from the old stack point to textures that have
        // been dropped — we leak the egui slots here (rare enough to ignore).
        self.layer_thumb_cache.clear();
    }

    /// Bake mesh maps (currently: world normal) for the loaded mesh.
    pub fn bake_mesh_maps(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.mesh_maps.bake(
            device,
            queue,
            &self.baker,
            &self.mesh,
            &self.paint_target.tiles,
            self.tile_resolution,
        );
    }

    pub fn active_layer(&self) -> &Layer {
        self.layer_stack.active_layer()
    }

    /// Request a recomposite — run after any external change that touched the
    /// active layer's textures (e.g. loading sidecars).
    pub fn recomposite(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.compositor
            .run_and_submit(device, queue, &self.layer_stack, &self.paint_target);
    }

    pub fn can_undo(&self) -> bool {
        self.undo_stack.can_undo()
    }
    pub fn can_redo(&self) -> bool {
        self.undo_stack.can_redo()
    }

    pub fn undo(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport.undo"),
        });
        let did = self.undo_stack.undo(device, &mut encoder, &mut self.layer_stack);
        queue.submit(Some(encoder.finish()));
        if did {
            self.recomposite(device, queue);
        }
        did
    }

    pub fn redo(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport.redo"),
        });
        let did = self.undo_stack.redo(device, &mut encoder, &mut self.layer_stack);
        queue.submit(Some(encoder.finish()));
        if did {
            self.recomposite(device, queue);
        }
        did
    }

    /// Append a new empty paint layer on top and recomposite.
    pub fn add_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.layer_stack.add_layer(device, queue);
        self.layer_thumb_cache.push(None);
        self.recomposite(device, queue);
    }

    pub fn add_fill_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.layer_stack.add_fill_layer(device, queue);
        self.layer_thumb_cache.push(None);
        self.recomposite(device, queue);
    }

    /// Delete a layer. Always keeps at least one.
    pub fn remove_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, idx: usize) {
        self.layer_stack.remove_at(idx);
        if idx < self.layer_thumb_cache.len() {
            self.layer_thumb_cache.remove(idx);
        }
        self.recomposite(device, queue);
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

    /// Register (or reuse) an egui TextureId pointing at `idx`'s tile-0
    /// base_color view, for use as a layer thumbnail. Writes through to the
    /// live texture, so stamps update the thumbnail automatically.
    pub fn ensure_layer_thumb(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
        idx: usize,
    ) -> Option<egui::TextureId> {
        if idx >= self.layer_stack.layers.len() {
            return None;
        }
        if self.layer_thumb_cache.len() != self.layer_stack.layers.len() {
            self.layer_thumb_cache
                .resize(self.layer_stack.layers.len(), None);
        }
        if self.layer_thumb_cache[idx].is_none() {
            if let Some(view) = self.layer_stack.layers[idx]
                .base_color_layer_views
                .first()
            {
                let id =
                    renderer.register_native_texture(device, view, wgpu::FilterMode::Linear);
                self.layer_thumb_cache[idx] = Some(id);
            }
        }
        self.layer_thumb_cache[idx]
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
        self.layer_stack = LayerStack::new_with_initial_layer(
            device,
            queue,
            resolution,
            self.paint_target.tiles.len() as u32,
        );
        self.compositor
            .run_and_submit(device, queue, &self.layer_stack, &self.paint_target);
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
            let inv_view_proj = view_proj.inverse();
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
                    tonemap_mode: self.tonemap_mode.as_u32(),
                    exposure: (2.0_f32).powf(self.exposure_stops),
                    _pad: 0,
                    inv_view_proj: inv_view_proj.to_cols_array_2d(),
                },
            );
            let material_uniforms = self.paint_target.material_uniforms(
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
            render_state.queue.write_buffer(
                &self.material_buf,
                0,
                bytemuck::bytes_of(&material_uniforms),
            );

            // Push env uniforms (intensity / rotation / skybox flag — mip_count
            // is baked by the environment at load time).
            self.env.write_uniforms(
                &render_state.queue,
                &EnvUniforms {
                    intensity: self.env_intensity,
                    rotation_y: self.env_rotation_y,
                    skybox_visible: if self.env_skybox_visible { 1 } else { 0 },
                    mip_count: self.env.mip_count as f32,
                },
            );

            // Build the material bind group for THIS frame so the active layer's
            // mask (or the dummy) goes into binding 4.
            let active_mask_view = match &self.layer_stack.active_layer().mask {
                Some(m) => &m.array_view,
                None => &self.paint_target.dummy_mask_view,
            };
            let material_bg =
                render_state
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("viewport.material_bg"),
                        layout: &self.renderer.material_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.material_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.base_color_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.rough_metal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.normal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(active_mask_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.paint_target.sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.mesh_maps.world_normal_view,
                                ),
                            },
                        ],
                    });

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

            // Detect stroke start BEFORE we overwrite last_paint_pos: a stroke
            // begins when this frame has a paint_pos but the previous one did
            // not.
            let stroke_starting = paint_pos.is_some() && self.last_paint_pos.is_none();

            // Reset when paint_pos becomes None (button released or modifier
            // held) so the next stroke starts fresh instead of sweeping back.
            self.last_paint_pos = paint_pos;

            let mut encoder = render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forge_paint_enc"),
                });

            // Eyedropper: sample the composited base_color at the hit UV and
            // write it into the brush color. Uses a synchronous readback —
            // acceptable for a one-shot click action.
            if self.tool == Tool::Eyedropper && stroke_starting && !strokes.is_empty() {
                let (layer, local_uv) = strokes[0];
                let res = self.paint_target.resolution;
                let px = ((local_uv.x * res as f32) as u32).min(res.saturating_sub(1));
                let py = ((local_uv.y * res as f32) as u32).min(res.saturating_sub(1));
                let rgba = sample_srgb_u8(
                    &render_state.device,
                    &render_state.queue,
                    &self.paint_target.base_color,
                    layer,
                    px,
                    py,
                );
                self.brush.color_srgb = [
                    rgba[0] as f32 / 255.0,
                    rgba[1] as f32 / 255.0,
                    rgba[2] as f32 / 255.0,
                ];
            }

            // Erase auto-provisions a mask on the active layer so there is
            // somewhere to paint black. The fresh mask view lands in next
            // frame's material_bg; this frame still stamps into it correctly.
            if self.tool == Tool::Erase
                && stroke_starting
                && !strokes.is_empty()
                && self.layer_stack.active_layer().mask.is_none()
            {
                let active_idx = self.layer_stack.active;
                self.layer_stack
                    .add_mask_to(active_idx, &render_state.device, &render_state.queue);
            }

            // Stamping tools. Fill layers are parameter-only, so we skip them.
            let active_is_fill = self.layer_stack.active_layer().is_fill();
            let is_stamping = matches!(self.tool, Tool::Paint | Tool::Erase | Tool::Fill);
            if is_stamping && !strokes.is_empty() && !active_is_fill {
                let active_idx = self.layer_stack.active;
                let active = self.layer_stack.active_layer();
                let has_mask = active.mask.is_some();

                // Erase always routes to the mask. Paint / Fill honor mask_edit.
                let channel = match self.tool {
                    Tool::Erase => PaintChannel::Mask,
                    _ => {
                        if self.brush.mask_edit && has_mask {
                            PaintChannel::Mask
                        } else {
                            self.brush.channel
                        }
                    }
                };

                // Skip if the channel is Mask but the layer still has no mask
                // (e.g. add_mask_to failed silently — defensive).
                let can_stamp = channel != PaintChannel::Mask || has_mask;

                if can_stamp {
                    // Fill is one-shot per stroke; Paint/Erase stamp every
                    // interpolated position along the drag.
                    let fill_stamp;
                    let stamps: &[(u32, Vec2)] = if self.tool == Tool::Fill {
                        if stroke_starting {
                            fill_stamp = [strokes[0]];
                            &fill_stamp
                        } else {
                            &[]
                        }
                    } else {
                        &strokes
                    };

                    if !stamps.is_empty() {
                        // Snapshot BEFORE the stamp lands so Cmd+Z rolls back.
                        // Only at stroke start.
                        if stroke_starting {
                            let kind = crate::undo::snapshot_kind_for_stamp(
                                channel,
                                matches!(channel, PaintChannel::Mask),
                                has_mask,
                            );
                            self.undo_stack.push_pre_stroke(
                                &render_state.device,
                                &mut encoder,
                                &self.layer_stack,
                                active_idx,
                                kind,
                            );
                        }

                        let value = if self.tool == Tool::Erase {
                            0.0
                        } else {
                            self.brush.value
                        };
                        let color_comp = match channel {
                            PaintChannel::BaseColor => {
                                let lin = self.brush.color_linear();
                                [lin[0], lin[1], lin[2]]
                            }
                            PaintChannel::Roughness | PaintChannel::Metallic => {
                                [value, value, value]
                            }
                            PaintChannel::Mask => {
                                // Pure white (reveal) or pure black (hide) —
                                // feathering comes from the brush's falloff.
                                let v = if value >= 0.5 { 1.0 } else { 0.0 };
                                [v, v, v]
                            }
                        };
                        let uniform_fill = if self.tool == Tool::Fill { 1u32 } else { 0u32 };

                        for (layer, local_uv) in stamps {
                            let uniforms = BrushUniforms {
                                color: [
                                    color_comp[0],
                                    color_comp[1],
                                    color_comp[2],
                                    self.brush.opacity,
                                ],
                                center_uv: local_uv.to_array(),
                                radius: self.brush.radius,
                                hardness: self.brush.hardness,
                                uniform_fill,
                                _pad: [0; 3],
                            };
                            let layer_view = match channel {
                                PaintChannel::BaseColor => {
                                    &active.base_color_layer_views[*layer as usize]
                                }
                                PaintChannel::Roughness | PaintChannel::Metallic => {
                                    &active.rough_metal_layer_views[*layer as usize]
                                }
                                PaintChannel::Mask => {
                                    &active.mask.as_ref().unwrap().layer_views[*layer as usize]
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
                        // Flatten the stack into the display target before PBR samples it.
                        self.compositor.run(
                            &render_state.device,
                            &render_state.queue,
                            &mut encoder,
                            &self.layer_stack,
                            &self.paint_target,
                        );
                    }
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

                pass.set_bind_group(0, &self.renderer.frame_bg, &[]);
                pass.set_bind_group(1, &material_bg, &[]);
                pass.set_bind_group(2, &self.env.bind_group, &[]);

                // Skybox background (optional) — draws at the far plane with
                // LessEqual depth so the mesh renders over it. Skipped in
                // channel-isolation view modes so they're easier to read.
                if self.env_skybox_visible && matches!(self.view_mode, ViewMode::Material) {
                    pass.set_pipeline(&self.skybox.pipeline);
                    pass.draw(0..3, 0..1);
                }

                pass.set_pipeline(&self.renderer.pipeline);
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

        // Brush cursor — ring over the native cursor when the pointer is
        // hovering the viewport in paint mode (plain LMB, no modifiers).
        // Size is an approximation of brush.radius in UV → screen space.
        if response.hovered() && no_mods {
            if let Some(pos) = response.hover_pos() {
                let on_mesh = self.last_hit_uv.is_some();
                let screen_radius = (self.brush.radius * rect.height() * 0.5)
                    .clamp(4.0, rect.height() * 0.3);
                let (color, stroke_width) = if on_mesh {
                    if self.brush.mask_edit {
                        (egui::Color32::from_rgb(120, 200, 255), 1.5)
                    } else {
                        (egui::Color32::from_rgb(255, 140, 120), 1.5)
                    }
                } else {
                    (egui::Color32::from_gray(120), 1.0)
                };
                let painter = ui.painter();
                painter.circle_stroke(pos, screen_radius, egui::Stroke::new(stroke_width, color));
                // Inner ring hints at brush hardness (softer → smaller core).
                let inner = screen_radius * self.brush.hardness.clamp(0.05, 0.95);
                if inner > 2.0 {
                    painter.circle_stroke(
                        pos,
                        inner,
                        egui::Stroke::new(1.0, color.linear_multiply(0.5)),
                    );
                }
            }
        }

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

/// Synchronous 1-pixel readback of an Rgba8UnormSrgb D2Array texture.
/// Returns the raw sRGB-encoded bytes — they match the on-wire color_srgb
/// representation directly (divide by 255 to get f32 in [0,1]).
fn sample_srgb_u8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    layer: u32,
    px: u32,
    py: u32,
) -> [u8; 4] {
    // COPY_BYTES_PER_ROW_ALIGNMENT is 256; a 1-pixel copy still needs a full row.
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("eyedropper_readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("eyedropper_enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: px, y: py, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    buf.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    let _ = device.poll(wgpu::Maintain::Wait);
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => return [0, 0, 0, 0xFF],
    }
    let view = buf.slice(..).get_mapped_range();
    let out = [view[0], view[1], view[2], view[3]];
    drop(view);
    buf.unmap();
    out
}
