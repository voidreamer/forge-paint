use eframe::egui;
use egui_wgpu::wgpu;
use glam::Vec2;

use crate::accel::MeshAccel;
use crate::background::BackgroundPipeline;
use crate::bake::{Baker, MeshMaps};
use crate::camera::OrbitCamera;
use crate::env::{
    BrdfLut, Environment, EnvUniforms, IrradianceBaker, PrefilterBaker, SkyboxPipeline,
};
use crate::mesh::{CpuMesh, GpuMesh};
use crate::paint::{
    target::MaterialUniforms, udim, BrushPipeline, BrushUniforms, Compositor, Layer, LayerStack,
    PaintChannel, PaintTarget, ProjBrushUniforms, ProjectionBrushPipeline,
};
use crate::pick;
use crate::fxaa::FxaaPipeline;
use crate::post::{PostPipeline, PostUniforms};
use crate::render::{FrameUniforms, Renderer, TonemapMode, ViewMode, HDR_FORMAT, LDR_FORMAT};
use crate::wireframe::WireframePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Paint,
    Erase,
    Fill,
    Eyedropper,
    /// Projection-paint a chosen stencil through the current camera.
    /// Selecting this tool is what activates an asset as the stencil;
    /// switching to any other tool cancels the stencil mode.
    Stencil,
}

/// Screen-space transform for the projected stencil. `offset` is in NDC,
/// `rotation` in radians (CCW), `scale` covers NDC [-scale, scale] in Y
/// with X scaled by the stencil's own aspect ratio.
#[derive(Debug, Clone, Copy)]
pub struct StencilTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
}

impl Default for StencilTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            rotation: 0.0,
            scale: 0.8,
        }
    }
}

impl Tool {
    pub fn label(self) -> &'static str {
        match self {
            Tool::Paint => "Paint",
            Tool::Erase => "Erase",
            Tool::Fill => "Fill",
            Tool::Eyedropper => "Eyedropper",
            Tool::Stencil => "Stencil",
        }
    }
    pub fn shortcut(self) -> &'static str {
        match self {
            Tool::Paint => "B",
            Tool::Erase => "E",
            Tool::Fill => "G",
            Tool::Eyedropper => "I",
            Tool::Stencil => "",
        }
    }
}

pub struct Viewport {
    renderer: Renderer,
    background: BackgroundPipeline,
    post: PostPipeline,
    pub fxaa: FxaaPipeline,
    pub wireframe: WireframePipeline,
    brush_pipeline: BrushPipeline,
    projection_brush: ProjectionBrushPipeline,
    pub smart_mask_pipeline: crate::paint::SmartMaskPipeline,

    /// Index into `AssetBrowser.textures` for the stencil currently
    /// routed through the projection brush. `None` = regular painting.
    pub active_stencil: Option<usize>,
    /// Screen-space transform for the stencil preview + projection sample.
    pub stencil_transform: StencilTransform,
    compositor: Compositor,
    color: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    egui_tex_id: Option<egui::TextureId>,

    mesh: GpuMesh,
    pub cpu_mesh: CpuMesh,
    accel: MeshAccel,
    /// Composited display textures — this is what the PBR shader samples.
    pub paint_target: PaintTarget,
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
    /// User-tunable bake settings forwarded to texture-baker. Persists
    /// across bakes so each kind doesn't reset its ray count after one
    /// click. Mutated by the Mesh maps panel.
    pub bake_settings: crate::bake::integration::BakeSettings,
    /// Monotonic mesh revision. Bumped whenever the mesh / subdivision
    /// changes; compared against `MeshMaps::baked_at_revision` so the
    /// panel can mark maps stale.
    pub mesh_revision: u64,
    /// Optional high-poly source for low→high projection bakes. Loaded
    /// from disk via the Mesh maps panel; pre-converted to texture-
    /// baker's `Mesh` once so the per-tile loop can share the BVH input.
    pub bake_high_poly: Option<texture_baker::mesh::Mesh>,
    /// Display name (file stem) of the high-poly currently loaded.
    pub bake_high_poly_label: Option<String>,
    /// Source path on disk — persisted into the project sidecar so the
    /// HP gets re-loaded automatically next session.
    pub bake_high_poly_path: Option<std::path::PathBuf>,
    /// Optional cage mesh — shares the low-poly's vertex count + index
    /// layout. The integration adapter slices it per-tile in lockstep
    /// with the low-poly extractor.
    pub bake_cage: Option<CpuMesh>,
    pub bake_cage_label: Option<String>,
    pub bake_cage_path: Option<std::path::PathBuf>,
    /// Most-recent bake outcome, surfaced as a status strip in the
    /// Bake tab. Persists across frames until the next bake replaces
    /// it. Threading the bake itself is a separate phase — for now
    /// the strip just reports duration / failure after the sync call
    /// returns. None = no bake has run since startup.
    pub last_bake_status: Option<BakeStatus>,
    baker: Baker,

    pub camera: OrbitCamera,

    pub brush: BrushState,
    pub tool: Tool,

    // Material factor tweakers (multiply sampled texture values)
    pub base_color_factor: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    /// Multiplier applied to the painted height when vertex-displacing
    /// the mesh. 0 = display disabled (paint data still persists).
    pub displacement_scale: f32,
    /// 0 = painted normal only (default). 1 = baked normal fully
    /// replaces the painted layer. Mid values mix tangent-space then
    /// renormalise — the standard "use HP detail" knob.
    pub baked_normal_blend: f32,
    /// Multiplier on the baked AO contribution. 0 = AO disabled (no
    /// occlusion at all), 1 = full strength (default), > 1 = stronger.
    /// Maps onto Marmoset's "AO strength" / Substance Painter's
    /// "Ambient Occlusion intensity".
    pub ao_intensity: f32,
    /// How many midpoint subdivisions to apply to the display mesh.
    /// Picking + paint UV still use the *base* mesh; this only changes
    /// the rendered geometry so vertex displacement has fine enough
    /// resolution to look smooth.
    pub subdivision_level: u32,

    // Lighting — dynamic analytic-light list. Default empty: HDRI is
    // the only contributor until the user adds a Directional / Spot
    // via the Lighting panel's `+` buttons. Replaces the old
    // light_dir / light_intensity / studio_* rig.
    pub lights: Vec<crate::lights::Light>,

    pub view_mode: ViewMode,

    pub tonemap_mode: TonemapMode,
    /// Exposure compensation in stops (−∞..∞ in principle; UI caps at ±4).
    /// Shader receives `2^stops` as a linear pre-tonemap multiplier.
    pub exposure_stops: f32,
    /// Post-tonemap contrast around mid-gray. 1.0 = identity. Defaults to
    /// a slight punch (1.1) so out-of-the-box renders read closer to a
    /// product-shot look than a flat-film look.
    pub grading_contrast: f32,
    /// Post-tonemap saturation around per-pixel luminance. 1.0 = identity.
    pub grading_saturation: f32,
    /// Unsharp-mask amount applied to the HDR input before tonemap (so
    /// sharpening tracks intensity). 0 = off; 0.1..0.3 reads as "crisp".
    pub grading_clarity: f32,

    pub tile_resolution: u32,

    /// Cached scene bounding-sphere radius from the most-recent mesh
    /// load. Drives the per-frame camera near/far recomputation so
    /// large models (SimReady-class assets, hi-res hero geometry)
    /// don't get clipped by the OrbitCamera's fixed 0.01 / 1000
    /// defaults. Updated whenever `load_*` rebuilds the GPU mesh.
    scene_radius: f32,

    /// Last successful pick this frame, if any — surfaced so the UI can show
    /// the cursor UV / tile.
    pub last_hit_uv: Option<[f32; 2]>,
    pub last_hit_tile: Option<u32>,

    /// Screen position of the last successful paint stamp, used to interpolate
    /// stamps between frames so fast drags don't leave gaps.
    last_paint_pos: Option<egui::Pos2>,

    /// Anchor position for the brush-adjust overlay — captured at the
    /// start of an S/D/F drag so the cursor ring stays put while the
    /// mouse moves to scrub.
    adjust_anchor: Option<egui::Pos2>,

    /// Stroke-level undo / redo.
    undo_stack: crate::undo::UndoStack,

    /// egui TextureIds for each layer's tile-0 base_color view, used as the
    /// layer-row thumbnail. Parallel to `layer_stack.layers`.
    pub layer_thumb_cache: Vec<Option<egui::TextureId>>,
}

/// One-shot record of the last completed bake. The Bake tab renders
/// this as a small strip so the user gets feedback after the bake
/// returns (the call is sync and freezes the UI for the duration —
/// this is the post-hoc receipt). `tile_count` reports how many
/// non-empty tiles actually got texels.
#[derive(Debug, Clone)]
pub struct BakeStatus {
    pub kind: crate::bake::integration::MapKind,
    pub duration_ms: u128,
    pub tile_count: u32,
    pub resolution: u32,
    pub error: Option<String>,
}

pub struct BrushState {
    pub channel: PaintChannel,
    /// Single-source color for the brush. Drives base color directly, and
    /// drives scalar channels (roughness / metallic / displacement / mask)
    /// via `luminance()` — one swatch picks every channel's write value.
    pub color_srgb: [f32; 3],
    /// Brush radius in **screen pixels**. Converted to UV at stamp time
    /// via the UV gradient at the hit — this way the ring size is
    /// consistent regardless of how the mesh is UV-unwrapped.
    pub radius: f32,
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
            radius: 40.0,
            hardness: 0.4,
            opacity: 1.0,
            mask_edit: false,
        }
    }
}

impl Viewport {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cpu: &CpuMesh) -> Self {
        let renderer = Renderer::new(device);
        // Post now writes to the LDR intermediate; FXAA writes to the
        // egui-facing color texture. Both use Bgra8UnormSrgb.
        let post = PostPipeline::new(device, LDR_FORMAT);
        let fxaa = FxaaPipeline::new(device, LDR_FORMAT);
        let background = BackgroundPipeline::new(device);
        let wireframe = WireframePipeline::new(device, &renderer.frame_bgl);
        let brush_pipeline = BrushPipeline::new(device);
        let smart_mask_pipeline = crate::paint::SmartMaskPipeline::new(device);
        let projection_brush = ProjectionBrushPipeline::new(device);
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
            HDR_FORMAT,
            wgpu::TextureFormat::Depth32Float,
        );

        let mesh_maps = MeshMaps::new_empty(device, queue, paint_target.tiles.len() as u32);
        let baker = Baker::new(device);

        let mut camera = OrbitCamera::default();
        camera.target = gpu.center;
        camera.distance = (gpu.radius * 2.5).max(1.5);
        let scene_radius = gpu.radius.max(0.001);

        let accel = MeshAccel::build(cpu);
        Self {
            renderer,
            background,
            post,
            fxaa,
            wireframe,
            brush_pipeline,
            smart_mask_pipeline,
            projection_brush,
            active_stencil: None,
            stencil_transform: StencilTransform::default(),
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
            bake_settings: crate::bake::integration::BakeSettings::default(),
            mesh_revision: 0,
            bake_high_poly: None,
            bake_high_poly_label: None,
            bake_high_poly_path: None,
            bake_cage: None,
            bake_cage_label: None,
            bake_cage_path: None,
            last_bake_status: None,
            baker,
            camera,
            brush: BrushState::default(),
            tool: Tool::Paint,
            base_color_factor: [1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            displacement_scale: 0.0,
            baked_normal_blend: 0.0,
            ao_intensity: 1.0,
            subdivision_level: 0,
            lights: Vec::new(),
            view_mode: ViewMode::Material,
            // ArmorPaint defaults to Filmic (Hable UC2) — reads as less crushed
            // than ACES and matches the reference painter's look.
            tonemap_mode: TonemapMode::Neutral,
            exposure_stops: 0.0,
            grading_contrast: 1.10,
            grading_saturation: 1.10,
            grading_clarity: 0.15,
            tile_resolution,
            // gpu is moved into `mesh` above; the radius is read
            // before then via `gpu.radius` in the camera-init block,
            // so we stash a Copy primitive of it here.
            scene_radius,
            last_hit_uv: None,
            last_hit_tile: None,
            last_paint_pos: None,
            adjust_anchor: None,
            undo_stack: crate::undo::UndoStack::default(),
            layer_thumb_cache: Vec::new(),
        }
    }

    /// Auto camera clipping — derive `z_near` / `z_far` from the
    /// camera's current `distance` and the scene's bounding-sphere
    /// `scene_radius`. Same idea usdview uses: near pulled in just
    /// shy of the nearest visible point on the model, far pushed out
    /// past the back. Called per-frame from both `vp.show` (wgpu) and
    /// the Hydra dispatch so the projection stays valid no matter
    /// which renderer is driving the central viewport.
    ///
    /// Without this, the fixed 0.01 / 1000 defaults break either
    /// way: large assets (SimReady props, the train) clip in front,
    /// and tiny meshes lose depth-buffer precision because the far
    /// plane is way too distant for what's actually visible.
    pub fn refresh_clip_planes(&mut self) {
        let r = self.scene_radius.max(1e-3);
        let dist = self.camera.distance;
        let near = ((dist - r) * 0.5).max(r * 0.001).max(0.001);
        let far = (dist + r) * 1.5 + r;
        self.camera.z_near = near;
        self.camera.z_far = far;
    }

    pub fn set_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cpu: &CpuMesh) {
        let gpu = GpuMesh::from_cpu(device, cpu);
        self.camera.target = gpu.center;
        self.camera.distance = (gpu.radius * 2.5).max(1.5);
        self.scene_radius = gpu.radius.max(0.001);
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
        // Bumping the revision invalidates any prior baked maps — the
        // Mesh maps panel will mark them stale until the user rebakes.
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
    }

    /// Rebuild the display GpuMesh by midpoint-subdividing the base
    /// cpu_mesh `level` times. Picking / paint UV still use the base
    /// mesh — this only adds geometry so vertex displacement has
    /// resolution to work with.
    pub fn set_subdivision(&mut self, device: &wgpu::Device, level: u32) {
        let level = level.min(5);
        if level == self.subdivision_level {
            return;
        }
        self.subdivision_level = level;
        let subdivided = if level == 0 {
            self.cpu_mesh.clone()
        } else {
            crate::mesh::subdivide(&self.cpu_mesh, level)
        };
        self.mesh = GpuMesh::from_cpu(device, &subdivided);
        // Subdivision changes the geometry the bakers see → maps go stale.
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
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

    /// Stamp the current brush at a tile's local UV — the UV-view
    /// entry point. Unlike the 3D-viewport path, there's no picker
    /// involved (the caller already knows which tile and where) so we
    /// bypass all the ray-cast plumbing and go straight to the brush
    /// pipeline. Recomposites just the touched tile afterward.
    pub fn stamp_at_uv(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_idx: u32,
        local_uv: [f32; 2],
    ) {
        let active_is_fill = self.layer_stack.active_layer().is_fill();
        if active_is_fill {
            return;
        }
        let active = self.layer_stack.active_layer();
        let has_mask = active.mask.is_some();
        let channel = if self.brush.mask_edit && has_mask {
            PaintChannel::Mask
        } else {
            self.brush.channel
        };
        // 400 px per UV is a reasonable default mapping for a square-ish
        // model view — matches how the main paint path picks
        // pixels-per-UV when the Jacobian probe fails. Clamp to half a
        // tile so extreme brush radii (500 px) don't overshoot.
        let uv_radius = (self.brush.radius / 400.0).min(0.5);
        let color_comp = brush_color_components(&self.brush, channel);
        let uniforms = BrushUniforms {
            color: [color_comp[0], color_comp[1], color_comp[2], self.brush.opacity],
            center_uv: local_uv,
            radius: uv_radius,
            hardness: self.brush.hardness,
            uniform_fill: 0,
            _pad: [0; 3],
        };
        let layer_view = match channel {
            PaintChannel::BaseColor => &active.base_color_layer_views[tile_idx as usize],
            PaintChannel::Roughness => &active.roughness_layer_views[tile_idx as usize],
            PaintChannel::Metallic => &active.metallic_layer_views[tile_idx as usize],
            PaintChannel::Mask => match &active.mask {
                Some(m) => &m.layer_views[tile_idx as usize],
                None => return,
            },
            PaintChannel::Displacement => {
                &self.paint_target.displacement_layer_views[tile_idx as usize]
            }
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("uv_view_stamp"),
        });
        self.brush_pipeline.stamp(
            queue,
            &mut encoder,
            layer_view,
            channel,
            &uniforms,
            self.paint_target.resolution,
        );
        if channel != PaintChannel::Displacement {
            self.compositor.run_sparse(
                device,
                queue,
                &mut encoder,
                &self.layer_stack,
                &self.paint_target,
                &[tile_idx as usize],
            );
        }
        queue.submit(Some(encoder.finish()));
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

    /// Snapshot the bake / material / smart-mask state into a project
    /// sidecar suitable for `forge-project.json` round-tripping. Heavy
    /// data (paint layers, baked GPU textures) lives next to the JSON
    /// as separate binaries — this struct is metadata only.
    pub fn build_sidecar(&self) -> crate::project::ProjectSidecar {
        let layers = self
            .layer_stack
            .layers
            .iter()
            .map(|l| crate::project::LayerSection {
                smart_mask: l.mask.as_ref().and_then(|m| m.smart),
            })
            .collect();
        crate::project::ProjectSidecar {
            version: 1,
            bake: crate::project::BakeSection {
                high_poly_path: self.bake_high_poly_path.clone(),
                cage_path: self.bake_cage_path.clone(),
                settings: Some(self.bake_settings),
            },
            material: crate::project::MaterialSection {
                base_color_factor: self.base_color_factor,
                metallic: self.metallic_factor,
                roughness: self.roughness_factor,
                normal_scale: self.normal_scale,
                displacement_scale: self.displacement_scale,
                baked_normal_blend: self.baked_normal_blend,
                ao_intensity: self.ao_intensity,
            },
            layers,
        }
    }

    /// Apply a smart-material preset: add a fill layer with the
    /// preset's colour / roughness / metalness, attach a mask with the
    /// preset's smart-mask config, and run a regen so the visible
    /// reveal pattern shows up immediately. Returns the new layer's
    /// index, or Err if the required source bake is missing — the
    /// caller surfaces that to the user.
    pub fn apply_smart_material_preset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        preset: crate::paint::SmartMaterialPreset,
    ) -> Result<usize, String> {
        // 1. Add the fill layer + push its params.
        self.add_fill_layer(device, queue);
        let idx = self.layer_stack.layers.len() - 1;
        self.layer_stack.layers[idx].set_fill_params(queue, preset.fill_params());

        // 2. Add a mask + stamp the smart-mask config.
        self.layer_stack.add_mask_to(idx, device, queue);
        if let Some(mask) = self.layer_stack.layers[idx].mask.as_mut() {
            mask.smart = Some(preset.smart_mask());
        }

        // 3. Activate the new layer so the user can immediately tweak
        //    the smart-mask params from the layer panel.
        self.layer_stack.active = idx;

        // 4. Best-effort regen. If the source bake is missing the
        //    error bubbles up; the layer is already added either way
        //    so the user can bake then refresh.
        let regen_result = self.regenerate_smart_mask(device, queue, idx);
        // Recomposite regardless — the new fill layer needs to land
        // even if the smart-mask regen errored out.
        self.recomposite(device, queue);
        regen_result.map(|_| idx)
    }

    /// Apply a previously-saved sidecar to this viewport. Bake textures
    /// aren't restored in v1 — caller re-bakes after load. Smart-mask
    /// params are stamped onto each layer's mask (creating one if the
    /// layer didn't have a mask), then `regenerate_smart_mask` runs to
    /// repopulate the texture from the freshly baked maps (when the
    /// required source map is available).
    pub fn apply_sidecar(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sidecar: &crate::project::ProjectSidecar,
    ) {
        // Material factors.
        self.base_color_factor = sidecar.material.base_color_factor;
        self.metallic_factor = sidecar.material.metallic;
        self.roughness_factor = sidecar.material.roughness;
        self.normal_scale = sidecar.material.normal_scale;
        self.displacement_scale = sidecar.material.displacement_scale;
        self.baked_normal_blend = sidecar.material.baked_normal_blend;
        self.ao_intensity = sidecar.material.ao_intensity;

        if let Some(s) = sidecar.bake.settings {
            self.bake_settings = s;
        }

        // Per-layer smart-mask params. Iterate up to the smaller of
        // the two — adding new layers from the sidecar is out of
        // scope for v1 (the layer stack itself isn't persisted yet).
        let n = self.layer_stack.layers.len().min(sidecar.layers.len());
        for i in 0..n {
            let Some(params) = sidecar.layers[i].smart_mask else {
                continue;
            };
            // Auto-create a mask if the layer doesn't have one — the
            // smart-mask sub-block needs a mask to live on.
            if self.layer_stack.layers[i].mask.is_none() {
                self.layer_stack.add_mask_to(i, device, queue);
            }
            if let Some(mask) = self.layer_stack.layers[i].mask.as_mut() {
                mask.smart = Some(params);
            }
            // Best-effort regen — silently skip if the source bake
            // isn't available (the user will see "(no bake)" in the
            // picker and can re-bake).
            let _ = self.regenerate_smart_mask(device, queue, i);
        }
    }

    /// Regenerate the smart mask on layer `idx` from the current
    /// `MeshMaps`. Returns Err with a user-readable hint when the
    /// required source bake is missing (e.g. "bake AO first"); the
    /// caller surfaces it to the layer panel as a status string.
    pub fn regenerate_smart_mask(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        idx: usize,
    ) -> Result<(), String> {
        let layer = self
            .layer_stack
            .layers
            .get(idx)
            .ok_or_else(|| "layer index out of range".to_string())?;
        let mask = layer
            .mask
            .as_ref()
            .ok_or_else(|| "layer has no mask".to_string())?;
        let params = mask
            .smart
            .as_ref()
            .ok_or_else(|| "mask isn't smart".to_string())?;
        self.smart_mask_pipeline
            .regenerate(device, queue, mask, &self.mesh_maps, params)?;
        // Mask atlas changed → recomposite the affected layer.
        self.recomposite(device, queue);
        Ok(())
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

    /// Read-only access to the base (pre-subdivision) CPU mesh —
    /// needed by the UV view to draw the mesh's UV wireframe.
    pub fn cpu_mesh(&self) -> &CpuMesh {
        &self.cpu_mesh
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

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
        stencil_view: Option<&wgpu::TextureView>,
        stencil_aspect: f32,
        stencil_egui_tex: Option<egui::TextureId>,
    ) {
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

        // Brush-adjust modifier drags — hold S/D/F and drag LMB to scrub
        // radius/hardness/opacity. Mari-ish workflow. Holding any of them
        // suppresses the paint path so we don't stroke while tweaking.
        let (adj_s, adj_d, adj_f) = ui.ctx().input(|i| {
            (
                i.key_down(egui::Key::S),
                i.key_down(egui::Key::D),
                i.key_down(egui::Key::F),
            )
        });
        let adjust_mode = adj_s || adj_d || adj_f;

        // Stencil transform manipulation — only meaningful when a stencil
        // is active, but the key-held detection is cheap enough to do
        // unconditionally (suppresses paint on accidental presses even
        // without a stencil, but that's fine).
        let (sxf_m, sxf_r, sxf_t) = ui.ctx().input(|i| {
            (
                i.key_down(egui::Key::M),
                i.key_down(egui::Key::R),
                i.key_down(egui::Key::T),
            )
        });
        let stencil_xf_mode = (sxf_m || sxf_r || sxf_t) && self.active_stencil.is_some();
        if primary_active && stencil_xf_mode {
            let delta = response.drag_delta();
            if sxf_m {
                // Pixels → NDC: 2 NDC units span the viewport height.
                self.stencil_transform.offset[0] += delta.x * 2.0 / rect.height();
                self.stencil_transform.offset[1] -= delta.y * 2.0 / rect.height();
            }
            if sxf_r {
                // ~180 px horizontal = half a turn.
                self.stencil_transform.rotation += delta.x * 0.01;
            }
            if sxf_t {
                // ~200 px horizontal = one unit scale.
                self.stencil_transform.scale =
                    (self.stencil_transform.scale + delta.x * 0.005).clamp(0.05, 4.0);
            }
        }

        if primary_active && adjust_mode {
            // Lock the cursor-ring anchor at wherever the drag started
            // so the preview stays put while the mouse moves to scrub.
            if self.adjust_anchor.is_none() {
                self.adjust_anchor = response
                    .interact_pointer_pos()
                    .or_else(|| response.hover_pos());
            }
            let dx = response.drag_delta().x;
            if adj_s {
                // brush.radius is screen pixels; 1 drag-pixel = 1 px of
                // brush-radius feels right (200 px of motion = 200 px of
                // size change).
                self.brush.radius = (self.brush.radius + dx).clamp(2.0, 500.0);
            }
            if adj_d {
                self.brush.hardness = (self.brush.hardness + dx * 0.003).clamp(0.0, 1.0);
            }
            if adj_f {
                self.brush.opacity = (self.brush.opacity + dx * 0.003).clamp(0.0, 1.0);
            }
        } else {
            self.adjust_anchor = None;
        }

        let paint_pos = if primary_active && no_mods && !adjust_mode && !stencil_xf_mode {
            response.interact_pointer_pos()
        } else {
            None
        };

        let tex_id = {
            let mut egui_renderer = render_state.renderer.write();
            self.ensure_color(&render_state.device, &mut egui_renderer, w, h);
            self.renderer.ensure_depth(&render_state.device, w, h);
            self.renderer.ensure_hdr(&render_state.device, w, h);
            self.renderer.ensure_ldr(&render_state.device, w, h);

            self.refresh_clip_planes();
            let aspect = w as f32 / h as f32;
            let view_proj = self.camera.view_proj(aspect);
            let inv_view_proj = view_proj.inverse();
            let eye = self.camera.eye();

            // Pack the user's analytic lights into the GPU array.
            // `Vec::iter().take(MAX_LIGHTS)` silently drops anything
            // past the cap; UI prevents the user from adding more so
            // this is just defensive.
            let mut gpu_lights = [crate::lights::GpuLight::default();
                crate::lights::MAX_LIGHTS];
            let light_count = self.lights.len().min(crate::lights::MAX_LIGHTS);
            for (i, light) in self.lights.iter().take(light_count).enumerate() {
                gpu_lights[i] = crate::lights::GpuLight::from_light(light);
            }

            self.renderer.write_frame(
                &render_state.queue,
                &FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    camera_pos: [eye.x, eye.y, eye.z, 1.0],
                    ambient_sky: [0.35, 0.45, 0.55, 1.0],
                    ambient_ground: [0.08, 0.07, 0.06, 1.0],
                    view_mode: self.view_mode.as_u32(),
                    tonemap_mode: self.tonemap_mode.as_u32(),
                    exposure: (2.0_f32).powf(self.exposure_stops),
                    ibl_scale: 1.0,
                    inv_view_proj: inv_view_proj.to_cols_array_2d(),
                    light_count: light_count as u32,
                    _pad_lights: [0; 3],
                    lights: gpu_lights,
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
                self.displacement_scale,
                self.baked_normal_blend,
                self.ao_intensity,
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
                                    &self.paint_target.roughness_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.metallic_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.normal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(active_mask_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.paint_target.sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.mesh_maps.world_normal_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.paint_target.displacement_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(
                                    self.mesh_maps.ao_view(),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    self.mesh_maps.baked_normal_view(),
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

            let strokes: Vec<(u32, Vec2, egui::Pos2)> = stamp_positions
                .iter()
                .filter_map(|p| {
                    let (orig, dir) = pick::screen_to_ray(*p, rect, view_proj, eye);
                    let hit = pick::pick(&self.cpu_mesh, orig, dir)?;
                    let tile = udim::tile_id(hit.uv.to_array());
                    let layer = self.paint_target.layer_for_tile(tile)?;
                    let local_uv =
                        Vec2::new(hit.uv.x - hit.uv.x.floor(), hit.uv.y - hit.uv.y.floor());
                    Some((layer, local_uv, *p))
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
                let (layer, local_uv, _) = strokes[0];
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
            let is_stamping =
                matches!(self.tool, Tool::Paint | Tool::Erase | Tool::Fill | Tool::Stencil);
            if is_stamping && !strokes.is_empty() && !active_is_fill {
                let active_idx = self.layer_stack.active;
                let active = self.layer_stack.active_layer();
                let has_mask = active.mask.is_some();

                // Erase always routes to the mask. Stencil can write to
                // either base color (default) or displacement (when the
                // user picks Height on the brush) — the projection
                // pipeline has a variant for each target format.
                // Paint / Fill honor mask_edit.
                let channel = match self.tool {
                    Tool::Erase => PaintChannel::Mask,
                    // Stencil projects into whatever channel the brush is
                    // pointed at — base color, displacement, roughness, or
                    // metallic. Mask falls back to base color (projection
                    // onto a per-layer mask isn't implemented yet).
                    Tool::Stencil => match self.brush.channel {
                        PaintChannel::Displacement => PaintChannel::Displacement,
                        PaintChannel::Roughness => PaintChannel::Roughness,
                        PaintChannel::Metallic => PaintChannel::Metallic,
                        _ => PaintChannel::BaseColor,
                    },
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
                    let stamps: &[(u32, Vec2, egui::Pos2)] = if self.tool == Tool::Fill {
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
                        // Only at stroke start. Displacement lives on
                        // PaintTarget in v0 — outside the per-Layer
                        // snapshot machinery — so we skip undo for it.
                        if stroke_starting && channel != PaintChannel::Displacement {
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

                        // Erase zeroes the channel; every other tool takes its
                        // write value from the brush color (directly for base
                        // color, luminance-derived for scalar channels).
                        let color_comp = if self.tool == Tool::Erase {
                            match channel {
                                PaintChannel::BaseColor => [0.0, 0.0, 0.0],
                                PaintChannel::Displacement => [0.0, 1.0, 0.0],
                                _ => [0.0, 0.0, 0.0],
                            }
                        } else {
                            brush_color_components(&self.brush, channel)
                        };
                        let uniform_fill = if self.tool == Tool::Fill { 1u32 } else { 0u32 };

                        // brush.radius lives in screen pixels; the shader
                        // expects UV. Probe the UV gradient at the first
                        // stamp's screen position and reuse it for the
                        // whole batch — drag interpolation keeps stamps
                        // in the same neighborhood so one Jacobian is fine.
                        let pixels_per_uv: Option<f32> =
                            stamp_positions.first().and_then(|p| {
                                let off = 2.0;
                                let ray_uv = |sp: egui::Pos2| -> Option<Vec2> {
                                    if !rect.contains(sp) {
                                        return None;
                                    }
                                    let (orig, dir) =
                                        pick::screen_to_ray(sp, rect, view_proj, eye);
                                    pick::pick(&self.cpu_mesh, orig, dir).map(|h| h.uv)
                                };
                                let uv_c = ray_uv(*p)?;
                                let uv_x = ray_uv(*p + egui::vec2(off, 0.0))?;
                                let uv_y = ray_uv(*p + egui::vec2(0.0, off))?;
                                let d_x = (uv_x - uv_c) / off;
                                let d_y = (uv_y - uv_c) / off;
                                let px_x = 1.0 / d_x.length().max(1e-6);
                                let px_y = 1.0 / d_y.length().max(1e-6);
                                Some(0.5 * (px_x + px_y))
                            });
                        // Convert pixels → UV. Fallback uses a sane default
                        // so a rare Jacobian miss doesn't skip the stamp.
                        //
                        // Reject implausibly tiny pixels_per_uv values (near-
                        // singular UV gradients on silhouettes or degenerate
                        // triangles). Without this guard a ratio of e.g. 0.5
                        // flips a 40-px brush into a 40-UV brush → whole tile
                        // gets painted every stamp, "fills the object" bug
                        // when the user drags across mesh edges. 16 px per UV
                        // is already an extremely zoomed-out view; anything
                        // below that almost certainly came from a bad pick.
                        let sane_ppu = pixels_per_uv
                            .filter(|&p| p >= 16.0)
                            .unwrap_or(rect.height() * 0.5)
                            .max(16.0);
                        // Hard cap at half a tile regardless — the shader and
                        // scissor would otherwise cover every texel, and no
                        // single stamp should ever exceed that footprint.
                        let uv_radius = (self.brush.radius / sane_ppu).min(0.5);

                        // Projection painting: when a stencil is selected
                        // and we're brushing into base color on a paint
                        // layer with a baked position map, route stamps
                        // through the projection pipeline instead. Falls
                        // back to the regular radial brush otherwise.
                        // Projection routes through a stencil for base
                        // color OR displacement (the shader branches on
                        // mode). Stencil tool + baked position map + a
                        // stencil asset are the shared prerequisites.
                        let projection_active = stencil_view.is_some()
                            && matches!(
                                channel,
                                PaintChannel::BaseColor
                                    | PaintChannel::Displacement
                                    | PaintChannel::Roughness
                                    | PaintChannel::Metallic,
                            )
                            && self.tool == Tool::Stencil
                            && self.mesh_maps.baked;

                        for (layer, local_uv, screen_pos) in stamps {
                            let layer_view = match channel {
                                PaintChannel::BaseColor => {
                                    &active.base_color_layer_views[*layer as usize]
                                }
                                PaintChannel::Roughness => {
                                    &active.roughness_layer_views[*layer as usize]
                                }
                                PaintChannel::Metallic => {
                                    &active.metallic_layer_views[*layer as usize]
                                }
                                PaintChannel::Mask => {
                                    &active.mask.as_ref().unwrap().layer_views[*layer as usize]
                                }
                                PaintChannel::Displacement => {
                                    // Displacement lives on PaintTarget
                                    // for v0, not per-Layer.
                                    &self.paint_target.displacement_layer_views
                                        [*layer as usize]
                                }
                            };

                            if projection_active {
                                // Screen pos → NDC. rect.left()/top() are the
                                // viewport origin in screen coords.
                                let ndc_x =
                                    2.0 * (screen_pos.x - rect.left()) / rect.width() - 1.0;
                                let ndc_y =
                                    1.0 - 2.0 * (screen_pos.y - rect.top()) / rect.height();
                                // Brush radius (screen px) → NDC. NDC covers
                                // [-1, 1] over the viewport height → 2 units.
                                let radius_ndc =
                                    self.brush.radius * 2.0 / rect.height();
                                let proj_uniforms = ProjBrushUniforms {
                                    view_proj: view_proj.to_cols_array_2d(),
                                    center_screen: [ndc_x, ndc_y],
                                    radius_screen: radius_ndc,
                                    opacity: self.brush.opacity,
                                    hardness: self.brush.hardness,
                                    aspect: rect.width() / rect.height(),
                                    stencil_offset: self.stencil_transform.offset,
                                    stencil_scale: self.stencil_transform.scale,
                                    stencil_cos_rot: self.stencil_transform.rotation.cos(),
                                    stencil_sin_rot: self.stencil_transform.rotation.sin(),
                                    stencil_aspect,
                                    // mode tells the shader how to pack its
                                    // output so it matches the target format:
                                    // 0 = Rgba8UnormSrgb (premultiplied RGB),
                                    // 1 = Rg16Float (h·a, a),
                                    // 2 = R8Unorm (luminance·a in R).
                                    mode: match channel {
                                        PaintChannel::Displacement => 1,
                                        PaintChannel::Roughness | PaintChannel::Metallic => 2,
                                        _ => 0,
                                    },
                                    _pad: [0.0; 3],
                                };
                                let position_view = self
                                    .mesh_maps
                                    .world_position
                                    .create_view(&wgpu::TextureViewDescriptor {
                                        label: Some("mesh_maps.world_position.tile_view"),
                                        dimension: Some(wgpu::TextureViewDimension::D2),
                                        base_array_layer: *layer,
                                        array_layer_count: Some(1),
                                        ..Default::default()
                                    });
                                self.projection_brush.stamp(
                                    &render_state.device,
                                    &render_state.queue,
                                    &mut encoder,
                                    layer_view,
                                    &position_view,
                                    stencil_view.unwrap(),
                                    &proj_uniforms,
                                    self.paint_target.resolution,
                                    local_uv.to_array(),
                                    uv_radius,
                                );
                            } else {
                                let uniforms = BrushUniforms {
                                    color: [
                                        color_comp[0],
                                        color_comp[1],
                                        color_comp[2],
                                        self.brush.opacity,
                                    ],
                                    center_uv: local_uv.to_array(),
                                    radius: uv_radius,
                                    hardness: self.brush.hardness,
                                    uniform_fill,
                                    _pad: [0; 3],
                                };
                                self.brush_pipeline.stamp(
                                    &render_state.queue,
                                    &mut encoder,
                                    layer_view,
                                    channel,
                                    &uniforms,
                                    self.paint_target.resolution,
                                );
                            }
                        }
                        // Flatten only the tiles that received stamps this
                        // frame — full recomposite runs on layer-property
                        // changes, not per paint frame. De-dupe before the
                        // composite call since drag interpolation usually
                        // lands repeated tile indices. Displacement is
                        // painted directly onto paint_target outside the
                        // layer stack, so no composite is needed for it.
                        if channel != PaintChannel::Displacement {
                            let mut dirty: Vec<usize> =
                                stamps.iter().map(|(l, _, _)| *l as usize).collect();
                            dirty.sort_unstable();
                            dirty.dedup();
                            self.compositor.run_sparse(
                                &render_state.device,
                                &render_state.queue,
                                &mut encoder,
                                &self.layer_stack,
                                &self.paint_target,
                                &dirty,
                            );
                        }
                    }
                }
            }

            // Mesh pass → HDR intermediate (Rgba16Float). The post pass
            // below reads this and writes the tonemapped result into the
            // egui-facing color texture.
            {
                let hdr_view = self.renderer.hdr_view();
                let depth_view = self.renderer.depth_view();

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("forge_paint_hdr_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
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

                // Gradient background first — fills the whole viewport
                // before anything else lands. Skybox (if visible) and
                // mesh overwrite it where pixels actually have content.
                // Skipped in channel-isolation views so the background
                // doesn't tint the displayed value.
                if matches!(self.view_mode, ViewMode::Material) {
                    pass.set_pipeline(&self.background.pipeline);
                    pass.draw(0..3, 0..1);
                }

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

                // Wireframe overlay sits on top of the shaded mesh in
                // the SAME pass so it gets depth-tested against it.
                if self.wireframe.visible && matches!(self.view_mode, ViewMode::Material) {
                    pass.set_pipeline(&self.wireframe.pipeline);
                    pass.set_vertex_buffer(0, self.mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        self.mesh.line_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(0..self.mesh.line_index_count, 0, 0..1);
                }
            }

            // Post pass: HDR → tonemapped LDR intermediate.
            {
                let hdr_view = self.renderer.hdr_view();
                let ldr_view = self.renderer.ldr_view();

                self.post.write_uniforms(
                    &render_state.queue,
                    &PostUniforms {
                        exposure: (2.0_f32).powf(self.exposure_stops),
                        view_mode: self.view_mode.as_u32(),
                        tonemap_mode: self.tonemap_mode.as_u32(),
                        contrast: self.grading_contrast,
                        saturation: self.grading_saturation,
                        clarity: self.grading_clarity,
                        texel_size: [1.0 / w as f32, 1.0 / h as f32],
                    },
                );
                let post_bg = self.post.make_bind_group(&render_state.device, hdr_view);

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("forge_paint_post_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ldr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.post.pipeline);
                pass.set_bind_group(0, &post_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            // FXAA pass: LDR → final egui viewport texture.
            {
                let ldr_view = self.renderer.ldr_view();
                let color_view = &self.color.as_ref().unwrap().1;
                self.fxaa.write_uniforms(&render_state.queue);
                let fxaa_bg = self.fxaa.make_bind_group(&render_state.device, ldr_view);

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("forge_paint_fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&self.fxaa.pipeline);
                pass.set_bind_group(0, &fxaa_bg, &[]);
                pass.draw(0..3, 0..1);
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

        // Stencil preview overlay — faint projection of the stencil at
        // its current transform, hidden while actively painting so the
        // stroke is clearly visible. Shown while idling and while
        // manipulating the transform via M/R/T.
        if let (Some(tex_id), Some(_)) = (stencil_egui_tex, self.active_stencil) {
            let is_painting_now = primary_active
                && !adjust_mode
                && !stencil_xf_mode
                && no_mods;
            if !is_painting_now {
                let xf = self.stencil_transform;
                let cr = xf.rotation.cos();
                let sr = xf.rotation.sin();
                let viewport_aspect = rect.width() / rect.height();
                // Stencil corners in stencil-local space. X spans
                // [-stencil_aspect, +stencil_aspect], Y spans [-1, 1]
                // — "isotropic" units where X and Y have the same
                // visual scale on screen.
                let local = [
                    ([-stencil_aspect,  1.0], [0.0, 0.0]),
                    ([ stencil_aspect,  1.0], [1.0, 0.0]),
                    ([ stencil_aspect, -1.0], [1.0, 1.0]),
                    ([-stencil_aspect, -1.0], [0.0, 1.0]),
                ];
                let ndc_to_screen = |nx: f32, ny: f32| -> egui::Pos2 {
                    egui::pos2(
                        rect.left() + (nx * 0.5 + 0.5) * rect.width(),
                        rect.top() + (0.5 - ny * 0.5) * rect.height(),
                    )
                };
                // Transform a local stencil corner into screen NDC.
                // Rotation happens in isotropic space; the final step
                // divides X by the viewport's aspect so NDC X ranges
                // match the screen's horizontal extent.
                let place = |lx: f32, ly: f32| -> egui::Pos2 {
                    let sx = lx * xf.scale;
                    let sy = ly * xf.scale;
                    let rx = sx * cr - sy * sr;
                    let ry = sx * sr + sy * cr;
                    let ndc_x = rx / viewport_aspect + xf.offset[0];
                    let ndc_y = ry + xf.offset[1];
                    ndc_to_screen(ndc_x, ndc_y)
                };
                let mut mesh = egui::epaint::Mesh::with_texture(tex_id);
                let alpha = if stencil_xf_mode { 180 } else { 110 };
                let tint = egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha);
                let mut pts = Vec::with_capacity(4);
                for ([lx, ly], [u, v]) in local {
                    let pos = place(lx, ly);
                    pts.push(pos);
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos,
                        uv: egui::pos2(u, v),
                        color: tint,
                    });
                }
                mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                ui.painter().add(egui::Shape::mesh(mesh));

                // Rectangle outline so the footprint is readable against
                // busy backgrounds.
                let outline = egui::Stroke::new(
                    1.5,
                    egui::Color32::from_rgb(255, 220, 100),
                );
                ui.painter().line_segment([pts[0], pts[1]], outline);
                ui.painter().line_segment([pts[1], pts[2]], outline);
                ui.painter().line_segment([pts[2], pts[3]], outline);
                ui.painter().line_segment([pts[3], pts[0]], outline);

                // Shortcut reminder at the bottom of the viewport while
                // a stencil is active.
                let hint = "Stencil: M+drag move · R+drag rotate · T+drag scale";
                ui.painter().text(
                    egui::pos2(rect.center().x, rect.bottom() - 10.0),
                    egui::Align2::CENTER_BOTTOM,
                    hint,
                    egui::FontId::monospace(12.0),
                    egui::Color32::from_rgb(255, 220, 100),
                );
            }
        }

        // Renderer label / picker now lives in a shared combo overlay
        // drawn by `App::draw_renderer_picker` after this function
        // returns. No badge here — the picker is the single
        // wgpu/Storm/3Delight selector.

        // Default brush-shortcut hint — shown at the bottom of the
        // viewport whenever the stencil-transform hint isn't taking the
        // slot. Mirrors the stencil hint's style so the bottom line is
        // always a predictable "where are the shortcuts?" cheat-sheet.
        let stencil_hint_visible = self.tool == Tool::Stencil && self.active_stencil.is_some();
        if !stencil_hint_visible {
            let hint = "S/D/F + drag: radius · hardness · opacity  ·  B/E/G/I/T: tools  ·  Home: reset view  ·  ⌘Z / ⌘⇧Z: undo/redo";
            ui.painter().text(
                egui::pos2(rect.center().x, rect.bottom() - 10.0),
                egui::Align2::CENTER_BOTTOM,
                hint,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgba_unmultiplied(200, 200, 210, 160),
            );
        }

        // Brush cursor — ring over the cursor in paint mode, or locked
        // to the adjust anchor while scrubbing brush parameters.
        if response.hovered() && (no_mods || adjust_mode) {
            let cursor_pos = if adjust_mode {
                self.adjust_anchor.or_else(|| response.hover_pos())
            } else {
                response.hover_pos()
            };
            if let Some(pos) = cursor_pos {
                // Brush radius is already in screen pixels — the cursor
                // ring is a trivial mapping now. No more UV-gradient
                // picker dance.
                let on_mesh = self.last_hit_uv.is_some();
                let screen_radius = self.brush.radius.clamp(2.0, rect.height() * 0.5);

                let (color, stroke_width) = if adjust_mode {
                    (egui::Color32::from_rgb(255, 220, 100), 2.0)
                } else if on_mesh {
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
                // Live value readout while adjusting.
                if adjust_mode {
                    let label = if adj_s {
                        format!("radius {:.0} px", self.brush.radius)
                    } else if adj_d {
                        format!("hardness {:.2}", self.brush.hardness)
                    } else {
                        format!("opacity {:.2}", self.brush.opacity)
                    };
                    painter.text(
                        pos + egui::vec2(screen_radius + 10.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        label,
                        egui::FontId::monospace(12.0),
                        color,
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

        // Keep the redraw going while painting, adjusting brush, or
        // manipulating the stencil transform so drags stay smooth.
        if paint_pos.is_some()
            || (primary_active && (adjust_mode || stencil_xf_mode))
        {
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

/// Shared brush-color → stamp-RGB mapping used by both the 3D viewport
/// paint path and `stamp_at_uv`. One color picker drives every channel:
/// BaseColor gets linear RGB directly, scalar channels use Rec.709
/// luminance, Displacement remaps luminance to signed height so a gray
/// brush is neutral and a white/black brush pushes/carves.
fn brush_color_components(brush: &BrushState, channel: PaintChannel) -> [f32; 3] {
    match channel {
        PaintChannel::BaseColor => {
            let lin = brush.color_linear();
            [lin[0], lin[1], lin[2]]
        }
        PaintChannel::Roughness | PaintChannel::Metallic => {
            let v = brush.luminance();
            [v, v, v]
        }
        PaintChannel::Mask => {
            let v = if brush.luminance() >= 0.5 { 1.0 } else { 0.0 };
            [v, v, v]
        }
        PaintChannel::Displacement => {
            // Luminance 0..1 → signed [-1, 1]. Gray (0.5) is a no-op so
            // tapering a stroke with a neutral swatch cleanly feathers
            // out without tearing a crater. Shader packs (h·a, 1·a, 0)
            // into the Rg16Float accumulator.
            let h = 2.0 * brush.luminance() - 1.0;
            [h, 1.0, 0.0]
        }
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

    /// Rec.709 luminance of the linear brush color. Maps the single
    /// color picker onto scalar paint channels: roughness/metallic use
    /// this directly (0..1), mask thresholds it, displacement remaps
    /// to a signed height (0.5 → 0, <0.5 carves, >0.5 pushes).
    pub fn luminance(&self) -> f32 {
        let l = self.color_linear();
        0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]
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
