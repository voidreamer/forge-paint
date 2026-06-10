use anyhow::Context;
use eframe::egui;
use std::path::{Path, PathBuf};

use crate::{
    assets::{self, AssetBrowser},
    mesh,
    viewport::{Tool, Viewport, ViewportSelection},
};

const HYDRA_STARTUP_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

struct HydraStartupProbe {
    stage_path: PathBuf,
    delegate: Option<String>,
    child: std::process::Child,
    started_at: std::time::Instant,
}

impl HydraStartupProbe {
    fn start(stage_path: &Path, delegate: Option<&str>) -> anyhow::Result<Self> {
        let exe = std::env::current_exe().context("locating forge-paint executable")?;
        let normalized_delegate = delegate
            .filter(|id| !id.is_empty())
            .map(std::string::ToString::to_string);
        let mut cmd = std::process::Command::new(exe);
        cmd.env("FORGE_PAINT_HYDRA_PROBE", "1")
            .env("FORGE_PAINT_HYDRA_PROBE_STAGE", stage_path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if let Some(delegate) = normalized_delegate.as_deref() {
            cmd.env("FORGE_PAINT_HYDRA_PROBE_DELEGATE", delegate);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let child = cmd.spawn().with_context(|| {
            format!(
                "starting Hydra startup probe for {}",
                normalized_delegate
                    .as_deref()
                    .unwrap_or("the default Hydra delegate")
            )
        })?;
        Ok(Self {
            stage_path: stage_path.to_path_buf(),
            delegate: normalized_delegate,
            child,
            started_at: std::time::Instant::now(),
        })
    }

    fn matches(&self, stage_path: &Path, delegate: Option<&str>) -> bool {
        self.stage_path == stage_path
            && self.delegate.as_deref() == delegate.filter(|id| !id.is_empty())
    }
}

impl Drop for HydraStartupProbe {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct HydraStartupFailure {
    stage_path: PathBuf,
    delegate: Option<String>,
    message: String,
}

enum HydraStartupState {
    Running(HydraStartupProbe),
    Failed(HydraStartupFailure),
}

impl HydraStartupState {
    fn matches(&self, stage_path: &Path, delegate: Option<&str>) -> bool {
        let normalized_delegate = delegate.filter(|id| !id.is_empty());
        match self {
            Self::Running(probe) => probe.matches(stage_path, normalized_delegate),
            Self::Failed(failure) => {
                failure.stage_path == stage_path
                    && failure.delegate.as_deref() == normalized_delegate
            }
        }
    }
}

#[derive(Default)]
pub struct App {
    viewport: Option<Viewport>,
    status: String,
    current_usd_path: Option<PathBuf>,
    /// USD path passed on the CLI — consumed once the viewport is ready.
    pending_open: Option<PathBuf>,

    // Open URI dialog state
    show_uri_dialog: bool,
    uri_buffer: String,
    pending_conversion: Option<PendingModelConversion>,

    browser: AssetBrowser,

    /// When true, the stencil picker modal is open. Set by the Stencil
    /// tool button; closed when the user picks a texture, imports a new
    /// one, or cancels.
    show_stencil_picker: bool,

    /// First-frame flag — we scan `assets/stencils/` and
    /// `assets/displacement/` once the wgpu renderer is ready and
    /// auto-import everything there so the user doesn't have to click
    /// "+ Import" for bundled assets.
    asset_scan_done: bool,

    /// When `Some`, the material slot picker modal is open. The value
    /// tracks which channel the picked texture should be assigned to.
    show_slot_picker: Option<MaterialSlot>,

    /// Docked 2D UV painting view. Splits the central area when on.
    show_uv_view: bool,
    /// Left-side dockable stage browser (read-only tree view of the
    /// loaded USD prim hierarchy). Off by default — switch on via
    /// View → Stage browser. Dock pattern mirrors the UV view.
    show_stage_browser: bool,
    stage_browser_undocked: bool,
    stage_browser: crate::stage_browser::StageBrowser,
    /// Last selection set pushed to the viewport's vertex-buffer
    /// highlight mask. Compared against the browser's current set
    /// every frame; the GPU write only fires on change so we don't
    /// re-upload the mask (12+ MB on SimReady-class assets) per
    /// repaint.
    last_pushed_selection: std::collections::HashSet<String>,
    /// Last expanded selection set pushed to Hydra/Storm. Tracked
    /// separately from `last_pushed_selection` because the wgpu path
    /// consumes the raw clicked set, while Storm needs explicit
    /// descendants and may be constructed after the user selected
    /// something in the stage browser.
    last_pushed_hydra_selection: std::collections::HashSet<String>,
    /// Which renderer owns the central viewport this frame. Solaris-
    /// style toggle (one viewport, swap between rasteriser and
    /// path-tracer) rather than side-by-side panels — keeps full
    /// resolution + a single source of input for orbit/zoom. The
    /// `▶ wgpu painter` / `▶ Hydra <delegate>` badge in the top-left
    /// is the swap button.
    renderer_mode: RendererMode,
    hydra: Option<crate::hydra_view::HydraView>,
    /// egui texture handle for the Hydra render result. Cached so the
    /// previous frame's pixels don't disappear during a re-render.
    hydra_egui_tex: Option<egui::TextureHandle>,
    /// User-selected Hydra render delegate plugin ID. `None` until
    /// the first frame populates it from `HydraView::current_delegate`,
    /// then sticks across stage opens so the user's pick (Storm /
    /// 3Delight / Arnold / ...) survives a close-reopen.
    hydra_delegate: Option<String>,
    /// Windows Hydra startup guard. HydraNSI can block forever while
    /// loading 3Delight or its shader compiler, and the first Hgi/GL
    /// bring-up can hang or die in native code when no usable GL
    /// context exists; probing each delegate in a child process keeps
    /// the main egui frame responsive and gives us a timeout path
    /// before constructing the real renderer.
    hydra_startup: Option<HydraStartupState>,
    /// Root cache dir for painted-material syncs. Each sync writes
    /// into a versioned subdir (`v<seq>`) of this so the material's
    /// asset paths change every time — without that, Hydra's texture
    /// cache keys on the resolved path and returns stale samples
    /// even though we've overwritten the PNG on disk. Versioned
    /// subdirs are the simplest invalidation hack.
    hydra_paint_cache_dir: Option<std::path::PathBuf>,
    /// Monotonic counter feeding the `vN` subdir name. Bumped on
    /// every successful sync. Resets per process (re-launch starts
    /// at 0); since the cache root is also per-pid, there's no risk
    /// of collision with an older run's textures.
    hydra_paint_sync_seq: u64,
    /// Status string for the sync button overlay — set after a sync
    /// run completes. Either "synced 14:32:05" on success or a
    /// short error message. Cleared when stage changes.
    hydra_paint_sync_status: Option<String>,
    /// `UsdGeomImageable::purpose` filter toggles for the Hydra view.
    /// Defaults: render = on (pipeline assets wrap detail geom in
    /// `Scope { purpose = "render" }`), proxy = on (playback proxies),
    /// guides = off (overlay-only annotations). Mirrored into the
    /// bridge every frame.
    hydra_show_render: bool,
    hydra_show_proxy: bool,
    hydra_show_guides: bool,
    /// Latched true whenever the user flips wgpu → Hydra. Consumed by
    /// `draw_hydra_central` to run one painted-material sync at mode
    /// entry, so any wgpu strokes get reflected in the Hydra panel
    /// without the user needing to click `↻ Sync paint`. Cleared
    /// after the sync (or after a failure that won't repeat).
    hydra_paint_sync_pending: bool,
    /// Concurrent material bindings authored against this stage.
    /// Each entry references a library material and binds it to
    /// either a set of target prims or the whole stage. Pushed to
    /// hydra-rs each frame via apply_material_binding (dirty-checked
    /// against `last_pushed_bindings`).
    material_bindings: Vec<MaterialBindingInstance>,
    /// Which entry of `material_bindings` is currently focused in
    /// the Material Editor (slider edits target this binding's
    /// `inputs`). `None` = no binding active; editor shows a
    /// placeholder prompt.
    active_binding_id: Option<u64>,
    /// Monotonic generator for new MaterialBindingInstance ids.
    /// Stable across the session so the hydra-rs side can key
    /// `apply_material_binding`/`remove_material_binding` calls
    /// without re-authoring the whole binding network when a
    /// single binding moves.
    next_binding_id: u64,
    /// Snapshot of each binding last pushed to hydra-rs, keyed by
    /// id. Used per-frame to figure out which bindings are new
    /// (apply_material_binding), which had their target_prims /
    /// source changed (re-apply), which had only input edits
    /// (set_binding_input_*), and which were removed
    /// (remove_material_binding).
    last_pushed_bindings: std::collections::HashMap<u64, MaterialBindingSnapshot>,
    /// Node-graph state for the Material Editor. Rebuilt when the
    /// active binding changes (so each binding's graph layout is
    /// independent); preserved across frames so node positions and
    /// texture wiring stick within a binding's lifetime.
    material_graph: crate::material_graph::MaterialGraph,
    /// Material editor pops out into a floating window when true,
    /// otherwise it sits as a bottom strip inside the central area
    /// (same dock pattern as the UV view).
    material_editor_undocked: bool,
    /// Pixels-per-UV-unit for the UV atlas. Modified by scroll.
    uv_zoom: f32,
    /// Screen-pixel offset applied to the atlas (drag to pan).
    uv_pan: egui::Vec2,
    /// Cached egui TextureIds for the composited paint_target channel
    /// tiles, rebuilt when the tile count or selected channel changes.
    uv_thumb_ids: Vec<Option<egui::TextureId>>,
    /// Which channel `uv_thumb_ids` currently reflects. When the user
    /// switches channel we clear the cache and re-register views.
    uv_thumb_channel: Option<crate::render::ViewMode>,
    /// Active layer index at the time `uv_thumb_ids` was populated —
    /// only relevant for the Mask channel (which reads views off the
    /// active Layer, so switching layers must invalidate the cache).
    uv_thumb_layer_idx: Option<usize>,
    /// Which channel the UV atlas is displaying. Tracks `view_mode`
    /// for paintable/viewable channels; limited to ones that map to a
    /// per-tile texture (BaseColor, Roughness, Metallic, Normal, Mask,
    /// Height).
    uv_channel: crate::render::ViewMode,
    /// Overlay the mesh's UV wireframe on top of the atlas.
    uv_show_wireframe: bool,
    /// When true, the UV view renders as a floating `egui::Window` that
    /// the user can drag around / resize freely. Otherwise it sits as a
    /// bottom-docked panel inside the central viewport area.
    uv_view_undocked: bool,
    /// Last atlas-UV position painted inside the UV view. Used to bridge
    /// fast drags with interpolated stamps, same pattern as the 3D
    /// viewport's `last_paint_pos`. None between strokes.
    uv_last_paint_atlas_uv: Option<egui::Vec2>,
    /// Tablet pressure paired with `uv_last_paint_atlas_uv`.
    uv_last_paint_pressure: Option<f32>,
    /// Which page of the right Properties panel is currently visible.
    /// Mirrors how the left tool column works — one focused view at a
    /// time, switched via the icon strip on the right edge.
    props_tab: PropertiesTab,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvertibleModelKind {
    Obj,
}

impl ConvertibleModelKind {
    fn from_path(path: &Path) -> Option<Self> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("obj") => Some(Self::Obj),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Obj => "OBJ",
        }
    }
}

#[derive(Debug, Clone)]
struct PendingModelConversion {
    source: PathBuf,
    kind: ConvertibleModelKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversionDialogAction {
    None,
    Convert,
    Cancel,
}

/// Which renderer owns the central viewport. Toggled by clicking the
/// renderer badge in the canvas top-left. Wgpu mode is the painting
/// surface (brush, stencil, paint ops); Hydra mode is a read-only
/// production-reference preview through `hydra-rs` / `UsdImagingGL`.
/// Camera state lives on `Viewport::camera` and is mirrored into Hydra
/// each frame, so orbit/zoom feels identical in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RendererMode {
    #[default]
    Wgpu,
    Hydra,
}

impl RendererMode {
    fn toggled(self) -> Self {
        match self {
            Self::Wgpu => Self::Hydra,
            Self::Hydra => Self::Wgpu,
        }
    }
}

/// Pages of the right Properties panel. Default = Layers (the active
/// painting context). Mirrored visually by a phosphor-icon strip on the
/// right edge of the window so left + right read as a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PropertiesTab {
    #[default]
    Layers,
    Lighting,
    Bake,
    Material,
    Project,
}

impl PropertiesTab {
    fn label(self) -> &'static str {
        match self {
            PropertiesTab::Layers => "Layers",
            PropertiesTab::Lighting => "Lighting",
            PropertiesTab::Bake => "Bake",
            PropertiesTab::Material => "Material",
            PropertiesTab::Project => "Project",
        }
    }

    fn glyph(self) -> &'static str {
        // Phosphor icons — same family the left tool column uses.
        match self {
            PropertiesTab::Layers => egui_phosphor::regular::STACK,
            PropertiesTab::Lighting => egui_phosphor::regular::SUN,
            PropertiesTab::Bake => egui_phosphor::regular::MAGIC_WAND,
            PropertiesTab::Material => egui_phosphor::regular::PALETTE,
            PropertiesTab::Project => egui_phosphor::regular::GEAR,
        }
    }

    const ALL: &'static [PropertiesTab] = &[
        PropertiesTab::Layers,
        PropertiesTab::Lighting,
        PropertiesTab::Bake,
        PropertiesTab::Material,
        PropertiesTab::Project,
    ];
}

/// Which channel a material-slot assignment should target on the
/// active layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum MaterialSlot {
    BaseColor,
    Roughness,
    Metallic,
    Normal,
}

impl MaterialSlot {
    fn label(self) -> &'static str {
        match self {
            MaterialSlot::BaseColor => "Base color",
            MaterialSlot::Roughness => "Roughness",
            MaterialSlot::Metallic => "Metallic",
            MaterialSlot::Normal => "Normal",
        }
    }
    const ALL: &'static [MaterialSlot] = &[
        MaterialSlot::BaseColor,
        MaterialSlot::Roughness,
        MaterialSlot::Metallic,
        MaterialSlot::Normal,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DetectedMaterialTextures {
    uv_primvar: String,
    base_color: Option<PathBuf>,
    roughness: Option<PathBuf>,
    metallic: Option<PathBuf>,
    normal: Option<PathBuf>,
    emission: Option<PathBuf>,
    occlusion: Option<PathBuf>,
}

impl DetectedMaterialTextures {
    fn empty(uv_primvar: impl Into<String>) -> Self {
        Self {
            uv_primvar: uv_primvar.into(),
            base_color: None,
            roughness: None,
            metallic: None,
            normal: None,
            emission: None,
            occlusion: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.base_color.is_none()
            && self.roughness.is_none()
            && self.metallic.is_none()
            && self.normal.is_none()
            && self.emission.is_none()
            && self.occlusion.is_none()
    }

    fn set(&mut self, pin: crate::material_graph::ShaderPin, path: PathBuf) {
        match pin {
            crate::material_graph::ShaderPin::DiffuseColor => self.base_color = Some(path),
            crate::material_graph::ShaderPin::Roughness => self.roughness = Some(path),
            crate::material_graph::ShaderPin::Metallic => self.metallic = Some(path),
            crate::material_graph::ShaderPin::Normal => self.normal = Some(path),
            crate::material_graph::ShaderPin::EmissionColor => self.emission = Some(path),
            crate::material_graph::ShaderPin::Occlusion => self.occlusion = Some(path),
            _ => {}
        }
    }

    fn texture_nodes(&self) -> Vec<(PathBuf, crate::material_graph::ShaderPin)> {
        let mut nodes = Vec::new();
        if let Some(path) = &self.base_color {
            nodes.push((path.clone(), crate::material_graph::ShaderPin::DiffuseColor));
        }
        if let Some(path) = &self.roughness {
            nodes.push((path.clone(), crate::material_graph::ShaderPin::Roughness));
        }
        if let Some(path) = &self.metallic {
            nodes.push((path.clone(), crate::material_graph::ShaderPin::Metallic));
        }
        if let Some(path) = &self.normal {
            nodes.push((path.clone(), crate::material_graph::ShaderPin::Normal));
        }
        if let Some(path) = &self.emission {
            nodes.push((
                path.clone(),
                crate::material_graph::ShaderPin::EmissionColor,
            ));
        }
        if let Some(path) = &self.occlusion {
            nodes.push((path.clone(), crate::material_graph::ShaderPin::Occlusion));
        }
        nodes
    }
}

#[derive(Debug, Clone)]
struct ResolvedStageTexture {
    path: PathBuf,
    slot: Option<MaterialSlot>,
    udim: Option<u32>,
}

/// One concurrent material binding. The library material at `source`
/// is referenced into the stage and bound either to every
/// UsdGeomMesh (target_prims empty) or to the listed SdfPaths
/// (Xform / Scope entries cascade to descendants hydra-rs-side).
#[derive(Debug, Clone)]
pub struct MaterialBindingInstance {
    pub id: u64,
    pub source: PathBuf,
    pub prim_path: String,
    pub kind: crate::assets::MaterialKind,
    pub inputs: crate::assets::MaterialInputs,
    /// SdfPaths the binding restricts itself to when `assigned`.
    /// Empty + `assigned == true` ⇒ stage-wide. Empty +
    /// `assigned == false` ⇒ shader node sits in the graph but
    /// hasn't been bound to anything yet (chip click default).
    pub target_prims: Vec<String>,
    /// False when the binding only exists in the node editor as a
    /// staged-but-unbound shader; true after the user picked
    /// "Assign to selection" / "Assign to stage" from the node's
    /// right-click menu. Per-frame Hydra push skips unassigned.
    pub assigned: bool,
}

/// What we last sent to hydra-rs for a given binding id, used to
/// figure out per-frame what changed. Stored separately from
/// `MaterialBindingInstance` so the snapshot stays Eq-comparable
/// independently of UI fields we may add later.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialBindingSnapshot {
    pub source: PathBuf,
    pub prim_path: String,
    pub inputs: crate::assets::MaterialInputs,
    pub target_prims: Vec<String>,
    pub assigned: bool,
}

impl MaterialBindingSnapshot {
    pub fn of(b: &MaterialBindingInstance) -> Self {
        Self {
            source: b.source.clone(),
            prim_path: b.prim_path.clone(),
            inputs: b.inputs,
            target_prims: b.target_prims.clone(),
            assigned: b.assigned,
        }
    }
}

fn bundled_asset_candidates(rel: impl AsRef<Path>) -> Vec<PathBuf> {
    let rel = rel.as_ref();
    let mut candidates = Vec::new();

    candidates.push(PathBuf::from(rel));
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(rel));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join(rel));
            if let Some(p2) = parent.parent() {
                candidates.push(p2.join(rel));
            }
            if let Some(p3) = parent.parent().and_then(|p| p.parent()) {
                candidates.push(p3.join(rel));
            }
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel));

    let mut seen = std::collections::HashSet::new();
    candidates.retain(|p| seen.insert(p.clone()));
    candidates
}

fn resolve_bundled_asset_dir(rel: impl AsRef<Path>) -> PathBuf {
    let rel = rel.as_ref();
    bundled_asset_candidates(rel)
        .into_iter()
        .find(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from(rel))
}

fn resolve_bundled_asset_file(rel: impl AsRef<Path>) -> Option<PathBuf> {
    bundled_asset_candidates(rel)
        .into_iter()
        .find(|p| p.is_file())
}

impl App {
    pub fn new(initial_usd: Option<PathBuf>) -> Self {
        // CLI path wins. Otherwise fall back to a user-provided default
        // mesh at assets/default_mesh/default.usda. Override the location via
        // FORGE_PAINT_DEFAULT_MESH if you run the binary from elsewhere.
        let pending_open = initial_usd.or_else(|| {
            if let Some(override_path) = std::env::var_os("FORGE_PAINT_DEFAULT_MESH") {
                let p = PathBuf::from(override_path);
                if p.exists() {
                    return Some(p);
                }
            }
            resolve_bundled_asset_file("assets/default_mesh/default.usda")
        });
        Self {
            pending_open,
            // Sane non-zero defaults for the UV view. bool/Vec2/Vec
            // defaults (false, (0,0), empty) are already what we want.
            uv_zoom: 400.0,
            uv_show_wireframe: true,
            uv_channel: crate::render::ViewMode::BaseColor,
            // Defaults — render + proxy on, guides off. Matches
            // usdview and Solaris's "show me the asset" defaults.
            hydra_show_render: true,
            hydra_show_proxy: true,
            hydra_show_guides: false,
            ..Default::default()
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.viewport.is_none() {
            if let Some(render_state) = frame.wgpu_render_state() {
                let cpu = mesh::cube();
                self.viewport = Some(Viewport::new(
                    &render_state.device,
                    &render_state.queue,
                    &cpu,
                ));
                log::info!(
                    "Viewport initialized with unit cube ({} verts)",
                    cpu.positions.len()
                );
            }
        }

        // If a path was passed on the CLI, open it now that the viewport exists.
        if self.viewport.is_some() {
            if let Some(path) = self.pending_open.take() {
                self.open_stage_or_offer_conversion(frame, path);
            }
        }

        let dropped_paths: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if let Some(path) = dropped_paths.into_iter().next() {
            self.open_stage_or_offer_conversion(frame, path);
        }

        // Stage-browser selection → viewport vertex-highlight mask.
        // Dirty-tracked against `last_pushed_selection` so we only
        // write the buffer when the set actually changes (else the
        // train asset would push 12 MB / frame for no reason).
        if let (Some(vp), Some(rs)) = (&self.viewport, frame.wgpu_render_state()) {
            let now = self.stage_browser.selected();
            if now != &self.last_pushed_selection {
                vp.set_selection(&rs.queue, now);
                self.last_pushed_selection = now.clone();
            }
        }
        if let Some(hydra) = self.hydra.as_mut() {
            // Push the expanded set (with descendants of selected
            // Xforms) to Storm too: Storm doesn't auto-cascade
            // selection like our wgpu prefix logic. This is tracked
            // independently so selection made before Hydra is lazily
            // constructed still appears on its first frame.
            let effective = self.stage_browser.effective_selection();
            if effective != self.last_pushed_hydra_selection {
                let paths: Vec<&str> = effective.iter().map(|s| s.as_str()).collect();
                hydra.set_selection(&paths);
                self.last_pushed_hydra_selection = effective;
            }
        }

        // First frame with a live viewport: import bundled stencils +
        // displacement maps into the asset browser.
        if !self.asset_scan_done && self.viewport.is_some() {
            if let Some(rs) = frame.wgpu_render_state() {
                self.asset_scan_done = true;
                self.scan_bundled_assets(&rs);
            }
        }

        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("forge-paint");
                ui.separator();
                ui.menu_button("Edit", |ui| {
                    let can_undo = self.viewport.as_ref().is_some_and(|vp| vp.can_undo());
                    let can_redo = self.viewport.as_ref().is_some_and(|vp| vp.can_redo());
                    if ui
                        .add_enabled(can_undo, egui::Button::new("Undo   ⌘Z / Ctrl+Z"))
                        .clicked()
                    {
                        self.do_undo(frame);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(can_redo, egui::Button::new("Redo   ⇧⌘Z / Ctrl+Shift+Z"))
                        .clicked()
                    {
                        self.do_redo(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        self.open_stage_dialog(frame);
                        ui.close_menu();
                    }
                    if ui.button("Open URI…").clicked() {
                        self.show_uri_dialog = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    let save_enabled = self.current_usd_path.is_some();
                    if ui
                        .add_enabled(save_enabled, egui::Button::new("Save   ⌘S / Ctrl+S"))
                        .clicked()
                    {
                        self.save_to_work_dir(frame);
                        ui.close_menu();
                    }
                    if ui.button("Save As…").clicked() {
                        self.export_textures_dialog(frame);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(save_enabled, egui::Button::new("Reload Sidecars"))
                        .clicked()
                    {
                        self.reload_sidecars(frame);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Export To Folder…").clicked() {
                        self.export_textures_dialog(frame);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.checkbox(&mut self.show_uv_view, "UV view").clicked() {
                        ui.close_menu();
                    }
                    if ui
                        .checkbox(&mut self.show_stage_browser, "Stage browser")
                        .clicked()
                    {
                        ui.close_menu();
                    }
                    // Hydra preview is no longer a separate panel —
                    // click the `▶ wgpu painter` badge in the canvas
                    // top-left to swap the central viewport over to
                    // Hydra. View menu carries the keyboard shortcut
                    // so it's still discoverable.
                    let label = match self.renderer_mode {
                        RendererMode::Wgpu => "Switch to Hydra preview",
                        RendererMode::Hydra => "Switch to wgpu painter",
                    };
                    if ui.button(label).clicked() {
                        self.renderer_mode = self.renderer_mode.toggled();
                        ui.close_menu();
                    }
                });
                ui.separator();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    match std::env::var("FORGE_PROJECT") {
                        Ok(proj) if !proj.is_empty() => {
                            ui.weak(format!("forge:{proj}"));
                        }
                        _ => {
                            ui.weak("standalone");
                        }
                    }
                });
            });
        });

        // Open URI modal — string entry for forge:// URIs or filesystem paths.
        if self.show_uri_dialog {
            let mut open = true;
            let mut load_requested: Option<String> = None;
            egui::Window::new("Open URI")
                .open(&mut open)
                .resizable(false)
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.label("USD URI or path (forge://… or a filesystem path):");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.uri_buffer)
                            .desired_width(f32::INFINITY),
                    );
                    ui.horizontal(|ui| {
                        let ok = ui.button("Load").clicked();
                        if ui.button("Cancel").clicked() {
                            self.show_uri_dialog = false;
                        }
                        if ok && !self.uri_buffer.trim().is_empty() {
                            load_requested = Some(self.uri_buffer.trim().to_string());
                        }
                    });
                });
            if !open {
                self.show_uri_dialog = false;
            }
            if let Some(uri) = load_requested {
                self.show_uri_dialog = false;
                self.open_stage_or_offer_conversion(frame, PathBuf::from(uri));
            }
        }

        if let Some(request) = self.pending_conversion.clone() {
            let mut open = true;
            let mut action = ConversionDialogAction::None;
            egui::Window::new(format!("Convert {} to USD?", request.kind.label()))
                .open(&mut open)
                .resizable(false)
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "{} files need to be converted to USD before forge-paint can open them.",
                        request.kind.label()
                    ));
                    ui.add_space(6.0);
                    ui.weak(request.source.display().to_string());
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button("Convert…").clicked() {
                            action = ConversionDialogAction::Convert;
                        }
                        if ui.button("Cancel").clicked() {
                            action = ConversionDialogAction::Cancel;
                        }
                    });
                });
            if !open {
                action = ConversionDialogAction::Cancel;
            }
            match action {
                ConversionDialogAction::None => {}
                ConversionDialogAction::Cancel => {
                    self.pending_conversion = None;
                    self.status = "Conversion cancelled.".to_string();
                }
                ConversionDialogAction::Convert => {
                    self.pending_conversion = None;
                    self.convert_model_dialog(frame, request);
                }
            }
        }

        // Stencil picker modal — grid of already-imported textures plus
        // an Import button for when the user wants a fresh one. Clicking
        // a thumbnail activates it as the stencil.
        if self.show_stencil_picker {
            let mut open = true;
            let mut picked: Option<usize> = None;
            let mut want_import = false;
            let mut want_cancel = false;
            egui::Window::new("Pick stencil")
                .open(&mut open)
                .resizable(true)
                .default_width(440.0)
                .default_height(360.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("+ Import new…").clicked() {
                            want_import = true;
                        }
                        if ui.button("Cancel").clicked() {
                            want_cancel = true;
                        }
                    });
                    ui.separator();
                    if self.browser.textures.is_empty() {
                        ui.weak("No textures imported yet. Click \"+ Import new…\".");
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("stencil_picker_scroll")
                            .show(ui, |ui| {
                                let columns = 4;
                                let thumb = egui::vec2(90.0, 90.0);
                                egui::Grid::new("stencil_picker_grid")
                                    .num_columns(columns)
                                    .spacing(egui::vec2(8.0, 8.0))
                                    .show(ui, |ui| {
                                        for (i, asset) in self.browser.textures.iter().enumerate() {
                                            ui.vertical(|ui| {
                                                let img = egui::Image::new((asset.thumb_id, thumb))
                                                    .fit_to_exact_size(thumb)
                                                    .sense(egui::Sense::click());
                                                if ui.add(img).on_hover_text(&asset.name).clicked()
                                                {
                                                    picked = Some(i);
                                                }
                                                ui.label(egui::RichText::new(&asset.name).small());
                                            });
                                            if (i + 1) % columns == 0 {
                                                ui.end_row();
                                            }
                                        }
                                    });
                            });
                    }
                });
            if !open || want_cancel {
                self.show_stencil_picker = false;
            }
            if let Some(idx) = picked {
                self.activate_stencil(idx, frame);
                self.show_stencil_picker = false;
            }
            if want_import {
                self.open_stencil_dialog(frame);
                self.show_stencil_picker = false;
            }
        }

        // Material slot picker — same grid as the stencil picker but
        // clicking a texture routes it to whichever channel the user
        // selected in the Material section.
        if let Some(slot) = self.show_slot_picker {
            let mut open = true;
            let mut picked: Option<usize> = None;
            let mut want_cancel = false;
            egui::Window::new(format!("Assign {} texture", slot.label()))
                .open(&mut open)
                .resizable(true)
                .default_width(440.0)
                .default_height(360.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            want_cancel = true;
                        }
                    });
                    ui.separator();
                    if self.browser.textures.is_empty() {
                        ui.weak(
                            "No textures imported yet. Use + Import in the \
                             bottom asset browser first.",
                        );
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("slot_picker_scroll")
                            .show(ui, |ui| {
                                let columns = 4;
                                let thumb = egui::vec2(90.0, 90.0);
                                egui::Grid::new("slot_picker_grid")
                                    .num_columns(columns)
                                    .spacing(egui::vec2(8.0, 8.0))
                                    .show(ui, |ui| {
                                        for (i, asset) in self.browser.textures.iter().enumerate() {
                                            ui.vertical(|ui| {
                                                let img = egui::Image::new((asset.thumb_id, thumb))
                                                    .fit_to_exact_size(thumb)
                                                    .sense(egui::Sense::click());
                                                if ui.add(img).on_hover_text(&asset.name).clicked()
                                                {
                                                    picked = Some(i);
                                                }
                                                ui.label(egui::RichText::new(&asset.name).small());
                                            });
                                            if (i + 1) % columns == 0 {
                                                ui.end_row();
                                            }
                                        }
                                    });
                            });
                    }
                });
            if !open || want_cancel {
                self.show_slot_picker = None;
            }
            if let Some(idx) = picked {
                self.apply_slot(slot, idx, frame);
                self.show_slot_picker = None;
            }
        }

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.status.is_empty() {
                    ui.weak("ready");
                } else {
                    ui.label(&self.status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("V select / Alt+LMB quick select · LMB paint · Ctrl+LMB orbit · Shift+LMB / MMB pan · wheel zoom · S/D/F+LMB brush size/hardness/opacity · M/R/T+LMB stencil move/rotate/scale");
                });
            });
        });

        egui::TopBottomPanel::bottom("assets")
            .resizable(true)
            .default_height(160.0)
            .min_height(80.0)
            .show(ctx, |ui| {
                self.asset_browser_panel(ui, frame);
            });

        let mut tool_clicked: Option<Tool> = None;
        // Thin icon-only tool column. Brush radius / hardness / opacity
        // are on S / D / F drag shortcuts; per-channel color & value are
        // picked from the viewport via the Eyedropper tool (E). Camera
        // reset lives on the Home key.
        egui::SidePanel::left("tools")
            .resizable(false)
            .exact_width(48.0)
            .show(ctx, |ui| {
                if let Some(vp) = &self.viewport {
                    tool_clicked = tool_strip(ui, vp);
                }
            });
        if let Some(t) = tool_clicked {
            self.switch_tool(t, frame);
        }

        // Stage browser — left of the tool strip (closer to the
        // viewport's edge). Mounted as its own SidePanel so it has
        // independent resize handles and doesn't fight the tool
        // strip's fixed-width layout. Only shown when toggled on
        // AND docked; the floating Window below handles the
        // undocked case.
        if self.show_stage_browser && !self.stage_browser_undocked {
            if let Some(path) = self.current_usd_path.clone() {
                self.stage_browser.ensure_loaded(&path);
            }
            egui::SidePanel::left("stage_browser_panel")
                .resizable(true)
                .default_width(260.0)
                .min_width(180.0)
                .show(ctx, |ui| {
                    self.stage_browser
                        .show(ui, &mut self.stage_browser_undocked);
                });
        }

        // Outer tab strip (right-most) — fixed width icon column, same
        // styling as the left tool column so the window reads as a pair
        // of icon strips bracketing the viewport.
        egui::SidePanel::right("props_tabs")
            .resizable(false)
            .exact_width(48.0)
            .show(ctx, |ui| {
                tab_strip(ui, &mut self.props_tab);
            });

        // Inner panel — the active tab's body. One header per tab so
        // each page has room to breathe; the `CollapsingHeader` stack
        // is gone in favour of subsection headings inside each tab.
        let mut slot_clicked: Option<MaterialSlot> = None;
        egui::SidePanel::right("props")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                if let Some(vp) = &mut self.viewport {
                    // Tab title block.
                    ui.heading(self.props_tab.label());
                    ui.separator();
                    // Uniform slider width — keeps every page's controls
                    // aligned regardless of label length.
                    ui.style_mut().spacing.slider_width = 140.0;

                    egui::ScrollArea::vertical()
                        .id_salt("right_panel_scroll")
                        .show(ui, |ui| match self.props_tab {
                            PropertiesTab::Layers => {
                                layers_tab(ui, vp, frame);
                            }
                            PropertiesTab::Lighting => {
                                lighting_tab(ui, vp, frame);
                            }
                            PropertiesTab::Bake => {
                                bake_tab(ui, vp, frame);
                            }
                            PropertiesTab::Material => {
                                // Two modes, mutually exclusive:
                                //  * Library material bound → show
                                //    the library editor only. The
                                //    wgpu paint-slot / factor knobs
                                //    aren't driving Hydra (the
                                //    library material overrides
                                //    them), so showing them would
                                //    read as "duplicate material UI".
                                //  * No library material → show the
                                //    paint-pipeline UI (slot
                                //    assignments + numeric factors).
                                //    Library cards in the bottom
                                //    pane are the path to the other
                                //    mode.
                                if self.material_bindings.is_empty() {
                                    slot_clicked = material_tab(ui, vp);
                                } else {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} binding(s) active",
                                            self.material_bindings.len()
                                        ))
                                        .strong()
                                        .color(egui::Color32::from_rgb(180, 220, 180)),
                                    );
                                    for b in &self.material_bindings {
                                        let scope = if b.target_prims.is_empty() {
                                            "stage-wide".to_string()
                                        } else {
                                            format!("{} prim(s)", b.target_prims.len())
                                        };
                                        let name = b
                                            .source
                                            .file_stem()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("(material)");
                                        ui.weak(format!("· {name} → {scope}"));
                                    }
                                    ui.add_space(8.0);
                                    ui.weak(
                                        "Live edits live in the Material Editor at the bottom of the viewport.",
                                    );
                                }
                            }
                            PropertiesTab::Project => {
                                project_tab(ui, vp, frame, &mut self.status);
                            }
                        });
                }
            });
        // If the user clicked a slot's "Assign…" button, queue the
        // picker modal to open next frame. (The modal lives near the
        // URI dialog block above; see show_slot_picker below.)
        if let Some(slot) = slot_clicked {
            self.show_slot_picker = Some(slot);
        }

        // Resolve the active stencil's GPU view + metadata up front,
        // outside the mutable borrow of viewport inside the CentralPanel
        // closure. We also hand the egui TextureId through for the
        // preview overlay.
        let stencil_idx = self.viewport.as_ref().and_then(|vp| vp.active_stencil);
        let stencil_asset = stencil_idx.and_then(|i| self.browser.textures.get(i));
        let stencil_view = stencil_asset.map(|a| &a.view);
        let stencil_aspect = stencil_asset
            .map(|a| a.width as f32 / a.height.max(1) as f32)
            .unwrap_or(1.0);
        let stencil_tex_id = stencil_asset.map(|a| a.thumb_id);

        let mut pending_viewport_selection: Option<ViewportSelection> = None;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(18, 18, 22)))
            .show(ctx, |ui| {
                if let Some(vp) = &mut self.viewport {
                    ui.horizontal(|ui| {
                        ui.label("View");
                        let prev = vp.view_mode;
                        egui::ComboBox::from_id_salt("viewport_view_mode")
                            .selected_text(prev.label())
                            .show_ui(ui, |ui| {
                                for &mode in crate::render::ViewMode::ALL {
                                    ui.selectable_value(&mut vp.view_mode, mode, mode.label());
                                }
                            });
                        // When the view is set to a single paintable channel,
                        // follow with the brush + UV view so strokes land on
                        // what's shown. The view mode, UV view channel, and
                        // brush channel are all tied together: whichever you
                        // change is the new source of truth.
                        if vp.view_mode != prev {
                            self.uv_channel = vp.view_mode;
                            if let Some(ch) = paint_channel_from_view_mode(vp.view_mode) {
                                if ch == crate::paint::PaintChannel::Mask {
                                    if vp.layer_stack.active_layer().mask.is_some() {
                                        vp.brush.mask_edit = true;
                                    }
                                } else {
                                    vp.brush.channel = ch;
                                    vp.brush.mask_edit = false;
                                }
                            }
                        }
                    });
                    // UV view — a bottom strip inside the central area
                    // when docked. When undocked, skip the panel here and
                    // the floating Window below handles it.
                    if self.show_uv_view && !self.uv_view_undocked {
                        let prev_uv_channel = self.uv_channel;
                        egui::TopBottomPanel::bottom("uv_view_panel")
                            .default_height(280.0)
                            .resizable(true)
                            .show_inside(ui, |ui| {
                                uv_view_body(
                                    ui,
                                    vp,
                                    frame,
                                    &mut self.uv_zoom,
                                    &mut self.uv_pan,
                                    &mut self.uv_thumb_ids,
                                    &mut self.uv_thumb_channel,
                                    &mut self.uv_thumb_layer_idx,
                                    &mut self.uv_channel,
                                    &mut self.uv_show_wireframe,
                                    &mut self.uv_view_undocked,
                                    &mut self.uv_last_paint_atlas_uv,
                                    &mut self.uv_last_paint_pressure,
                                );
                            });
                        if self.uv_channel != prev_uv_channel {
                            vp.view_mode = self.uv_channel;
                            apply_uv_channel_to_brush(self.uv_channel, vp);
                        }
                    }
                    // Material Editor — bottom strip when docked. Same
                    // pattern as the UV view above. Renders whenever
                    // there's at least one active binding OR the user
                    // has picked a chip and may want to assign it
                    // (so the "Assign…" buttons are reachable from a
                    // fresh stage).
                    let editor_visible = !self.material_bindings.is_empty();
                    if editor_visible && !self.material_editor_undocked {
                        let browser_sel = self.stage_browser.effective_selection();
                        egui::TopBottomPanel::bottom("material_editor_panel")
                            .default_height(320.0)
                            .resizable(true)
                            .show_inside(ui, |ui| {
                                material_editor_body(
                                    ui,
                                    &mut self.material_bindings,
                                    &mut self.material_graph,
                                    &browser_sel,
                                    &mut self.material_editor_undocked,
                                );
                            });
                    }
                    let mut swap_renderer = false;
                    let viewport_selection = match self.renderer_mode {
                        RendererMode::Wgpu => {
                            vp.show(ui, frame, stencil_view, stencil_aspect, stencil_tex_id)
                        }
                        RendererMode::Hydra => Self::draw_hydra_central(
                            ui,
                            frame,
                            vp,
                            &mut self.hydra,
                            &mut self.hydra_egui_tex,
                            &mut self.hydra_delegate,
                            &mut self.hydra_startup,
                            &mut self.hydra_paint_cache_dir,
                            &mut self.hydra_paint_sync_seq,
                            &mut self.hydra_paint_sync_status,
                            &mut self.hydra_paint_sync_pending,
                            &mut self.hydra_show_render,
                            &mut self.hydra_show_proxy,
                            &mut self.hydra_show_guides,
                            &self.material_bindings,
                            &mut self.last_pushed_bindings,
                            self.current_usd_path.as_deref(),
                            &mut swap_renderer,
                        ),
                    };
                    if let Some(selection) = viewport_selection {
                        pending_viewport_selection = Some(selection);
                    }
                    // Renderer / delegate picker — single combo
                    // overlay covering both modes. Placed top-right
                    // of the canvas, mirrored from the badges'
                    // previous home. Drives renderer_mode and
                    // hydra_delegate directly. Also stamps the Storm
                    // warning banner when Storm is the active Hydra
                    // delegate.
                    let prev_mode = self.renderer_mode;
                    let prev_delegate = self.hydra_delegate.clone();
                    let viewport_rect = ui.max_rect();
                    Self::draw_renderer_picker(
                        ui,
                        viewport_rect,
                        &mut self.renderer_mode,
                        &mut self.hydra_delegate,
                    );
                    let _ = swap_renderer;
                    if prev_mode == RendererMode::Wgpu && self.renderer_mode == RendererMode::Hydra
                    {
                        // Just entered Hydra → schedule the one-
                        // shot paint sync so any strokes painted in
                        // wgpu mode show up here without an extra
                        // click. Hydra → Wgpu needs nothing — the
                        // wgpu painter has been seeing the live
                        // paint targets the whole time.
                        self.hydra_paint_sync_pending = true;
                    }
                    // If the user picked a different Hydra delegate
                    // while already in Hydra mode, no sync needed —
                    // the delegate switch alone is fine.
                    let _ = prev_delegate;
                } else {
                    ui.centered_and_justified(|ui| ui.label("Initializing GPU…"));
                }
            });
        if let Some(selection) = pending_viewport_selection {
            self.apply_viewport_selection(selection, ctx);
        }

        // Floating Stage browser. Closing the window via [×] hides
        // the browser entirely (same as unchecking View → Stage
        // browser).
        if self.show_stage_browser && self.stage_browser_undocked {
            if let Some(path) = self.current_usd_path.clone() {
                self.stage_browser.ensure_loaded(&path);
            }
            let mut open = true;
            egui::Window::new("Stage browser")
                .open(&mut open)
                .default_size(egui::vec2(360.0, 560.0))
                .resizable(true)
                .show(ctx, |ui| {
                    self.stage_browser
                        .show(ui, &mut self.stage_browser_undocked);
                });
            if !open {
                self.show_stage_browser = false;
            }
        }

        // Floating UV view — only when the feature is enabled AND the
        // user has undocked it. Closing the window via its [×] hides the
        // UV view entirely (same as unchecking View → UV view).
        // Floating Material Editor — only when a library material is
        // bound AND the user has popped it out. Closing the window
        // re-docks (returns to the bottom strip); to actually drop
        // the material, use the "Clear material" button in the right-
        // side properties tab.
        let editor_visible = !self.material_bindings.is_empty();
        if editor_visible && self.material_editor_undocked {
            let browser_sel = self.stage_browser.effective_selection();
            let mut open = true;
            egui::Window::new("Material Editor")
                .open(&mut open)
                .default_size(egui::vec2(720.0, 480.0))
                .resizable(true)
                .show(ctx, |ui| {
                    material_editor_body(
                        ui,
                        &mut self.material_bindings,
                        &mut self.material_graph,
                        &browser_sel,
                        &mut self.material_editor_undocked,
                    );
                });
            if !open {
                self.material_editor_undocked = false;
            }
        }

        if self.show_uv_view && self.uv_view_undocked {
            if let Some(vp) = &mut self.viewport {
                let mut open = true;
                let prev_uv_channel = self.uv_channel;
                egui::Window::new("UV view")
                    .open(&mut open)
                    .default_size(egui::vec2(720.0, 480.0))
                    .resizable(true)
                    .show(ctx, |ui| {
                        uv_view_body(
                            ui,
                            vp,
                            frame,
                            &mut self.uv_zoom,
                            &mut self.uv_pan,
                            &mut self.uv_thumb_ids,
                            &mut self.uv_thumb_channel,
                            &mut self.uv_thumb_layer_idx,
                            &mut self.uv_channel,
                            &mut self.uv_show_wireframe,
                            &mut self.uv_view_undocked,
                            &mut self.uv_last_paint_atlas_uv,
                            &mut self.uv_last_paint_pressure,
                        );
                    });
                if self.uv_channel != prev_uv_channel {
                    vp.view_mode = self.uv_channel;
                    apply_uv_channel_to_brush(self.uv_channel, vp);
                }
                if !open {
                    self.show_uv_view = false;
                }
            }
        }

        // Cmd/Ctrl+S: save to default work dir.
        let save_shortcut = ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::S,
            ))
        });
        if save_shortcut {
            self.save_to_work_dir(frame);
        }

        // Cmd/Ctrl+Z: undo. Cmd/Ctrl+Shift+Z: redo. Check shift variant first so
        // the shift+Z form doesn't trigger the plain undo binding too.
        let redo_shortcut = ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                egui::Key::Z,
            ))
        });
        if redo_shortcut {
            self.do_redo(frame);
        }
        let undo_shortcut = ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Z,
            ))
        });
        if undo_shortcut {
            self.do_undo(frame);
        }

        // Home: reset the orbit camera to its default pose, preserving the
        // current target so the reset still frames the loaded mesh. No
        // left-panel "Reset camera" button any more — this is the only
        // way to reset the view.
        let home_pressed = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Home));
        if home_pressed {
            if let Some(vp) = &mut self.viewport {
                let t = vp.camera.target;
                vp.camera = crate::camera::OrbitCamera::default();
                vp.camera.target = t;
            }
        }

        // Tool hotkeys (single key, no modifiers). consume_key respects
        // focus — text fields intercept before these fire.
        let tool_change = ctx.input_mut(|i| {
            use egui::{Key, Modifiers};
            if i.consume_key(Modifiers::NONE, Key::V) {
                Some(Tool::Select)
            } else if i.consume_key(Modifiers::NONE, Key::B) {
                Some(Tool::Paint)
            } else if i.consume_key(Modifiers::NONE, Key::E) {
                Some(Tool::Erase)
            } else if i.consume_key(Modifiers::NONE, Key::G) {
                Some(Tool::Fill)
            } else if i.consume_key(Modifiers::NONE, Key::I) {
                Some(Tool::Eyedropper)
            } else {
                None
            }
        });
        if let Some(tool) = tool_change {
            self.switch_tool(tool, frame);
        }
    }
}

/// Renders the tool strip and reports any tool the user just clicked.
/// Caller handles the side effects (file dialog for stencil, clearing
/// the active stencil when switching away, etc.).
/// Map the current view mode to its paintable channel, if any. Material /
/// Normal / WorldNormalBaked aren't direct paint targets so they return
/// None (and the brush's current channel is preserved). Kept out-of-line
/// so view → brush syncing stays consistent across the 3D view dropdown
/// and the UV-view channel picker.
fn paint_channel_from_view_mode(vm: crate::render::ViewMode) -> Option<crate::paint::PaintChannel> {
    use crate::paint::PaintChannel;
    use crate::render::ViewMode;
    match vm {
        ViewMode::BaseColor => Some(PaintChannel::BaseColor),
        ViewMode::Roughness => Some(PaintChannel::Roughness),
        ViewMode::Metallic => Some(PaintChannel::Metallic),
        ViewMode::Mask => Some(PaintChannel::Mask),
        ViewMode::Height => Some(PaintChannel::Displacement),
        ViewMode::Material | ViewMode::Normal | ViewMode::WorldNormalBaked => None,
    }
}

/// Apply a UV-view channel change to the brush. Paintable channels
/// (BaseColor / Roughness / Metallic / Height) route to `brush.channel`
/// and clear `mask_edit`. Mask flips `mask_edit` on (if the active layer
/// has a mask). Non-paintable view channels (Normal, Material, World
/// Normal) leave the brush as-is — the user is only viewing.
fn apply_uv_channel_to_brush(vm: crate::render::ViewMode, vp: &mut Viewport) {
    use crate::paint::PaintChannel;
    match paint_channel_from_view_mode(vm) {
        Some(PaintChannel::Mask) => {
            if vp.layer_stack.active_layer().mask.is_some() {
                vp.brush.mask_edit = true;
            }
        }
        Some(ch) => {
            vp.brush.channel = ch;
            vp.brush.mask_edit = false;
        }
        None => {}
    }
}

/// Vertical icon column for the right Properties panel — picks which
/// tab's body the inner panel renders. Mirrors `tool_strip` so left and
/// right edges of the window read as a pair of icon strips.
fn tab_strip(ui: &mut egui::Ui, current: &mut PropertiesTab) {
    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        for &tab in PropertiesTab::ALL {
            let selected = *current == tab;
            let fill = if selected {
                egui::Color32::from_rgb(46, 92, 148)
            } else {
                ui.style().visuals.widgets.inactive.bg_fill
            };
            let btn = egui::Button::new(egui::RichText::new(tab.glyph()).size(22.0))
                .min_size(egui::vec2(36.0, 36.0))
                .fill(fill);
            if ui.add(btn).on_hover_text(tab.label()).clicked() {
                *current = tab;
            }
            ui.add_space(2.0);
        }
    });
}

// --- Tab bodies. Each composes existing section helpers; reparenting
//     into tabs keeps content per-page focused without duplicating the
//     existing logic. Subsection labels use `ui.heading` rather than
//     CollapsingHeader so the page reads top-to-bottom in one glance.

fn layers_tab(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    color_section(ui, vp);
    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new("Brush").strong());
    brush_section(ui, vp);
    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new("Layers").strong());
    layer_panel(ui, vp, frame);
}

fn lighting_tab(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    // Environment carries the HDRI + tonemap + grading + display
    // controls; light_section adds the analytic key/fill/rim rig. They
    // belong on the same page because changing any of them is a "how
    // does the scene look?" decision.
    env_panel(ui, vp, frame);
    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new("Light").strong());
    light_section(ui, vp);
}

fn bake_tab(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    mesh_maps_panel(ui, vp, frame);
}

/// Material Editor body — header row (dock toggle + material name)
/// followed by the egui-snarl node graph. Reused by both the docked
/// bottom-strip panel and the floating undocked window so the two
/// render identically.
fn material_editor_body(
    ui: &mut egui::Ui,
    bindings: &mut Vec<MaterialBindingInstance>,
    graph: &mut crate::material_graph::MaterialGraph,
    browser_selection: &std::collections::HashSet<String>,
    undocked: &mut bool,
) {
    ui.horizontal(|ui| {
        let n = bindings.len();
        ui.strong(format!("Material network ({n} shader(s))"));
        ui.weak("Right-click a shader node to assign / remove.");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (label, tip) = if *undocked {
                (
                    "⮌ Dock",
                    "Dock the Material Editor back into the main layout",
                )
            } else {
                (
                    "⮎ Undock",
                    "Pop out the Material Editor into a floating window",
                )
            };
            if ui.button(label).on_hover_text(tip).clicked() {
                *undocked = !*undocked;
            }
        });
    });
    ui.separator();

    let mut pending: Vec<crate::material_graph::GraphAction> = Vec::new();
    let mut viewer = crate::material_graph::GraphViewer {
        bindings,
        browser_selection,
        pending_actions: &mut pending,
    };
    let graph_rect = ui.available_rect_before_wrap();
    let graph_size = egui::vec2(graph_rect.width().max(1.0), graph_rect.height().max(160.0));
    let (graph_rect, _) = ui.allocate_exact_size(graph_size, egui::Sense::hover());
    let mut graph_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt("material_graph_canvas")
            .max_rect(graph_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    let style = material_graph_style(&graph_ui);
    graph
        .snarl
        .show(&mut viewer, &style, "material_graph", &mut graph_ui);
    material_graph_canvas_nav(ui, graph_rect, graph);

    // Apply right-click menu actions emitted by the viewer. Done
    // here (outside the snarl.show borrow) so we can mutate the
    // bindings Vec freely without aliasing the &mut Vec the
    // viewer is holding.
    for act in pending {
        match act {
            crate::material_graph::GraphAction::AssignToSelection(id) => {
                if let Some(b) = bindings.iter_mut().find(|b| b.id == id) {
                    b.target_prims = browser_selection.iter().cloned().collect();
                    b.assigned = true;
                }
            }
            crate::material_graph::GraphAction::AssignToStage(id) => {
                if let Some(b) = bindings.iter_mut().find(|b| b.id == id) {
                    b.target_prims.clear();
                    b.assigned = true;
                }
            }
            crate::material_graph::GraphAction::Unassign(id) => {
                if let Some(b) = bindings.iter_mut().find(|b| b.id == id) {
                    b.assigned = false;
                }
            }
            crate::material_graph::GraphAction::Remove(id) => {
                bindings.retain(|b| b.id != id);
                graph.remove_shader_node(id);
            }
        }
    }
}

fn material_graph_style(ui: &egui::Ui) -> egui_snarl::ui::SnarlStyle {
    let visuals = ui.visuals();
    egui_snarl::ui::SnarlStyle {
        min_scale: Some(0.12),
        max_scale: Some(3.5),
        animate_zoom: Some(0.06),
        bg_pattern: Some(egui_snarl::ui::BackgroundPattern::grid(
            egui::vec2(44.0, 44.0),
            0.0,
        )),
        bg_pattern_stroke: Some(egui::Stroke::new(
            1.0,
            visuals
                .widgets
                .noninteractive
                .bg_stroke
                .color
                .gamma_multiply(0.55),
        )),
        pin_size: Some(10.0),
        wire_width: Some(3.0),
        header_drag_space: Some(egui::vec2(18.0, 18.0)),
        ..Default::default()
    }
}

fn material_graph_canvas_nav(
    ui: &egui::Ui,
    rect: egui::Rect,
    graph: &mut crate::material_graph::MaterialGraph,
) {
    let (inside, middle_down, delta) = ui.ctx().input(|i| {
        let pos = i.pointer.hover_pos().or(i.pointer.interact_pos());
        (
            pos.map(|p| rect.contains(p)).unwrap_or(false),
            i.pointer.button_down(egui::PointerButton::Middle),
            i.pointer.delta(),
        )
    });
    if inside && middle_down && delta != egui::Vec2::ZERO {
        graph.pan_nodes_by(delta);
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        ui.ctx().request_repaint();
    }
}

fn material_tab(ui: &mut egui::Ui, vp: &mut Viewport) -> Option<MaterialSlot> {
    let clicked = material_slots_section(ui, vp);
    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new("Material factors").strong());
    material_factors_section(ui, vp);
    clicked
}

fn project_tab(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame, status: &mut String) {
    paint_target_section(ui, vp, frame, status);
}

fn tool_strip(ui: &mut egui::Ui, vp: &Viewport) -> Option<Tool> {
    ui.add_space(4.0);
    let mut clicked: Option<Tool> = None;
    ui.vertical_centered(|ui| {
        let entries = [
            (Tool::Select, egui_phosphor::regular::CURSOR_CLICK),
            (Tool::Paint, egui_phosphor::regular::PAINT_BRUSH),
            (Tool::Erase, egui_phosphor::regular::ERASER),
            (Tool::Fill, egui_phosphor::regular::PAINT_BUCKET),
            (Tool::Eyedropper, egui_phosphor::regular::EYEDROPPER),
            (Tool::Stencil, egui_phosphor::regular::PROJECTOR_SCREEN),
        ];
        for (tool, glyph) in entries {
            let selected = vp.tool == tool;
            let fill = if selected {
                egui::Color32::from_rgb(46, 92, 148)
            } else {
                ui.style().visuals.widgets.inactive.bg_fill
            };
            let btn = egui::Button::new(egui::RichText::new(glyph).size(22.0))
                .min_size(egui::vec2(36.0, 36.0))
                .fill(fill);
            let shortcut = tool.shortcut();
            let tooltip = if shortcut.is_empty() {
                tool.label().to_string()
            } else {
                format!("{} [{}]", tool.label(), shortcut)
            };
            if ui.add(btn).on_hover_text(tooltip).clicked() {
                clicked = Some(tool);
            }
            ui.add_space(2.0);
        }
    });
    clicked
}

/// Run the command template in `FORGE_PAINT_POST_EXPORT` (if set), substituting
/// `{dir}` with the export directory. Returns a status suffix to tack onto the
/// main status message; empty if the hook isn't configured.
fn run_post_export_hook(dir: &std::path::Path) -> String {
    let Some(tmpl) = std::env::var_os("FORGE_PAINT_POST_EXPORT") else {
        return String::new();
    };
    let tmpl_str = tmpl.to_string_lossy();
    if tmpl_str.trim().is_empty() {
        return String::new();
    }
    let cmd_str = tmpl_str.replace("{dir}", &dir.to_string_lossy());
    log::info!("post-export hook: sh -c {cmd_str:?}");
    match std::process::Command::new("sh")
        .arg("-c")
        .arg(&cmd_str)
        .status()
    {
        Ok(s) if s.success() => " · post-export hook ok".to_string(),
        Ok(s) => format!(" · post-export hook failed ({s})"),
        Err(e) => format!(" · post-export hook error: {e}"),
    }
}

fn color_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    use crate::paint::PaintChannel;

    // A single color drives every channel: base color directly, scalar
    // channels via luminance. Show the current channel's derived write
    // value below the picker so the user can see what the swatch means
    // for Roughness / Metallic / Displacement / Mask.
    let channel = if vp.brush.mask_edit {
        PaintChannel::Mask
    } else {
        vp.brush.channel
    };
    ui.horizontal(|ui| {
        ui.label("Paint color");
        ui.color_edit_button_rgb(&mut vp.brush.color_srgb);
    });

    // Channel-specific readout of what this swatch paints right now.
    let lum = vp.brush.luminance();
    match channel {
        PaintChannel::BaseColor => {
            ui.weak("base color — RGB used directly");
        }
        PaintChannel::Roughness => {
            ui.weak(format!("roughness value: {lum:.2}  (color luminance)"));
        }
        PaintChannel::Metallic => {
            ui.weak(format!("metallic value: {lum:.2}  (color luminance)"));
        }
        PaintChannel::Displacement => {
            let h = 2.0 * lum - 1.0;
            ui.weak(format!(
                "height: {h:+.2}  (gray = 0, black carves, white pushes)"
            ));
        }
        PaintChannel::Mask => {
            let tag = if lum >= 0.5 {
                "reveal (white)"
            } else {
                "hide (black)"
            };
            ui.weak(format!("mask: {tag}  (threshold at 0.5)"));
        }
    }

    // Quick swatches — common neutrals + pure channel extremes. Makes
    // "set roughness to 0" or "set metallic to 1" a single click.
    ui.add_space(4.0);
    ui.weak("swatches");
    let swatches: &[([f32; 3], &str)] = &[
        ([0.0, 0.0, 0.0], "black"),
        ([0.5, 0.5, 0.5], "gray 0.5"),
        ([1.0, 1.0, 1.0], "white"),
        ([0.95, 0.2, 0.2], "red"),
        ([0.2, 0.9, 0.35], "green"),
        ([0.2, 0.45, 0.95], "blue"),
    ];
    ui.horizontal_wrapped(|ui| {
        for (rgb, tip) in swatches {
            let fill = egui::Color32::from_rgb(
                (rgb[0] * 255.0) as u8,
                (rgb[1] * 255.0) as u8,
                (rgb[2] * 255.0) as u8,
            );
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
            ui.painter().rect_filled(rect, 3.0, fill);
            ui.painter().rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(40)),
                egui::StrokeKind::Outside,
            );
            let resp = resp.on_hover_text(*tip);
            if resp.clicked() {
                vp.brush.color_srgb = *rgb;
            }
        }
    });
}

fn brush_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.add(egui::Slider::new(&mut vp.brush.radius, 2.0..=500.0).text("radius px"));
    ui.add(egui::Slider::new(&mut vp.brush.opacity, 0.0..=1.0).text("opacity"));
    ui.add(egui::Slider::new(&mut vp.brush.hardness, 0.0..=1.0).text("hardness"));

    ui.add_space(6.0);
    ui.label(egui::RichText::new("Tablet pressure").strong());
    let live = vp
        .input_pressure()
        .map(|p| format!("{p:.2}"))
        .unwrap_or_else(|| "idle".to_string());
    ui.horizontal(|ui| {
        ui.label("input");
        ui.weak(live);
    });

    let pressure = &mut vp.brush.pressure;
    ui.horizontal(|ui| {
        ui.checkbox(&mut pressure.size_enabled, "size")
            .on_hover_text("Scale brush radius from tablet pressure");
        ui.add_enabled(
            pressure.size_enabled,
            egui::Slider::new(&mut pressure.min_size, 0.0..=1.0).text("min"),
        )
        .on_hover_text("Radius multiplier at light pressure");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut pressure.opacity_enabled, "opacity")
            .on_hover_text("Scale per-dab opacity from tablet pressure");
        ui.add_enabled(
            pressure.opacity_enabled,
            egui::Slider::new(&mut pressure.min_opacity, 0.0..=1.0).text("min"),
        )
        .on_hover_text("Opacity multiplier at light pressure");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut pressure.hardness_enabled, "hardness")
            .on_hover_text("Blend hardness from min to the brush hardness");
        ui.add_enabled(
            pressure.hardness_enabled,
            egui::Slider::new(&mut pressure.min_hardness, 0.0..=1.0).text("min"),
        )
        .on_hover_text("Hardness at light pressure");
    });
    ui.add(egui::Slider::new(&mut pressure.curve, 0.25..=4.0).text("curve"))
        .on_hover_text("1 is linear, lower is softer, higher is firmer");
}

fn layer_panel(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    // Global Paint target: Content vs Mask. Disabled if active layer has no mask.
    ui.horizontal(|ui| {
        ui.label("Paint:");
        let mut edit = vp.brush.mask_edit;
        ui.radio_value(&mut edit, false, "content");
        let has_mask = vp.layer_stack.active_layer().mask.is_some();
        ui.add_enabled_ui(has_mask, |ui| {
            ui.radio_value(&mut edit, true, "mask");
        });
        if !has_mask && edit {
            edit = false;
        }
        vp.brush.mask_edit = edit;
    });

    let mut needs_recomposite = false;
    let mut delete_idx: Option<usize> = None;
    let mut add_requested = false;
    let mut mask_add: Option<usize> = None;
    let mut mask_remove: Option<usize> = None;
    // Layers whose smart-mask params changed this frame and need a GPU
    // re-bake. Drained after the per-layer iteration so the borrow
    // checker doesn't complain about double-mutating `vp`.
    let mut smart_regen_request: Vec<usize> = Vec::new();

    // Top-down list — index 0 is bottom of stack, so iterate reversed so the
    // topmost (last to composite) is drawn first in the panel.
    let n = vp.layer_stack.layers.len();
    egui::ScrollArea::vertical()
        .id_salt("layers_scroll")
        .max_height(280.0)
        .show(ui, |ui| {
            for i in (0..n).rev() {
                let is_active = vp.layer_stack.active == i;
                let row_bg = if is_active {
                    egui::Color32::from_rgb(46, 62, 96)
                } else {
                    egui::Color32::TRANSPARENT
                };
                // Thumbnail registration happens before the row mutates the
                // layer so we don't fight the borrow checker inside the row.
                let thumb = frame.wgpu_render_state().and_then(|rs| {
                    let mut renderer = rs.renderer.write();
                    vp.ensure_layer_thumb(&rs.device, &mut renderer, i)
                });
                let mut activate = false;
                egui::Frame::NONE
                    .fill(row_bg)
                    .inner_margin(4.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let layer = &mut vp.layer_stack.layers[i];

                            if ui.checkbox(&mut layer.visible, "").changed() {
                                needs_recomposite = true;
                            }

                            if let Some(id) = thumb {
                                // Click the thumbnail → activate the layer.
                                let img = egui::Image::new((id, egui::vec2(32.0, 32.0)))
                                    .fit_to_exact_size(egui::vec2(32.0, 32.0))
                                    .sense(egui::Sense::click());
                                if ui.add(img).clicked() {
                                    activate = true;
                                }
                            }

                            // Clickable name → sets active
                            let label = egui::Label::new(egui::RichText::new(&layer.name).strong())
                                .sense(egui::Sense::click())
                                .truncate();
                            if ui.add(label).clicked() {
                                activate = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            let layer = &mut vp.layer_stack.layers[i];
                            use crate::paint::BlendMode;
                            let before = layer.blend_mode;
                            egui::ComboBox::from_id_salt(("blend_mode", i))
                                .selected_text(layer.blend_mode.label())
                                .show_ui(ui, |ui| {
                                    for &mode in BlendMode::ALL {
                                        ui.selectable_value(
                                            &mut layer.blend_mode,
                                            mode,
                                            mode.label(),
                                        );
                                    }
                                });
                            if layer.blend_mode != before {
                                needs_recomposite = true;
                            }
                            let resp = ui.add(
                                egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                                    .show_value(true)
                                    .text("opacity"),
                            );
                            if resp.changed() {
                                needs_recomposite = true;
                            }
                            if n > 1 && ui.small_button("delete").clicked() {
                                delete_idx = Some(i);
                            }
                        });

                        // Fill-layer controls replace the content/mask buttons —
                        // Fill can't be painted directly; the sliders drive it.
                        if let Some(mut params) = vp.layer_stack.layers[i].fill_params() {
                            let before = params;
                            ui.horizontal(|ui| {
                                ui.label("fill color");
                                ui.color_edit_button_rgb(&mut params.base_color_srgb);
                            });
                            ui.add(
                                egui::Slider::new(&mut params.roughness, 0.0..=1.0)
                                    .text("roughness"),
                            );
                            ui.add(
                                egui::Slider::new(&mut params.metallic, 0.0..=1.0).text("metallic"),
                            );
                            if params != before {
                                if let Some(rs) = frame.wgpu_render_state() {
                                    vp.layer_stack.layers[i].set_fill_params(&rs.queue, params);
                                    needs_recomposite = true;
                                }
                            }
                        }
                        ui.horizontal(|ui| {
                            let has_mask = vp.layer_stack.layers[i].mask.is_some();
                            if has_mask {
                                let editing_this = is_active && vp.brush.mask_edit;
                                let (content_label, mask_label) = if editing_this {
                                    ("edit content", "[editing mask]")
                                } else if is_active {
                                    ("[editing content]", "edit mask")
                                } else {
                                    ("content", "mask")
                                };
                                if ui.small_button(content_label).clicked() {
                                    vp.layer_stack.active = i;
                                    vp.brush.mask_edit = false;
                                }
                                if ui.small_button(mask_label).clicked() {
                                    vp.layer_stack.active = i;
                                    vp.brush.mask_edit = true;
                                }
                                if ui.small_button("×").clicked() {
                                    mask_remove = Some(i);
                                }
                            } else if ui.small_button("+ mask").clicked() {
                                mask_add = Some(i);
                            }
                        });
                        // Smart-mask sub-block. Shown for any layer that
                        // has a mask. The block renders inside an indented
                        // group so the layer row stays compact when the
                        // mask is purely manual.
                        if vp.layer_stack.layers[i].mask.is_some() {
                            smart_mask_subpanel(ui, vp, i, frame, &mut smart_regen_request);
                        }
                    });
                if activate {
                    vp.layer_stack.active = i;
                }
            }
        });

    ui.add_space(4.0);
    let mut add_fill_requested = false;
    let mut preset_request: Option<crate::paint::SmartMaterialPreset> = None;
    ui.horizontal(|ui| {
        if ui.button("+ Paint").clicked() {
            add_requested = true;
        }
        if ui.button("+ Fill").clicked() {
            add_fill_requested = true;
        }
        ui.weak(format!("{n} layer{}", if n == 1 { "" } else { "s" }));
    });

    // Smart-material presets — one click adds a fill layer + smart
    // mask wired to the right baked-map source. Each button is gated
    // on the source bake being available, with a tooltip saying
    // which one to bake when it isn't.
    ui.add_space(4.0);
    ui.weak("Smart materials");
    ui.horizontal_wrapped(|ui| {
        for &preset in crate::paint::SmartMaterialPreset::ALL {
            let required = preset.smart_mask().source;
            let available = source_is_baked(vp, required);
            let tooltip = if available {
                format!("Adds a {} layer with a smart mask.", preset.label())
            } else {
                format!(
                    "Bake {} on the Bake tab first (this preset uses {} as its source).",
                    required.required_map().label(),
                    required.label()
                )
            };
            let resp = ui
                .add_enabled(
                    available,
                    egui::Button::new(format!("+ {}", preset.label())),
                )
                .on_hover_text(tooltip);
            if resp.clicked() {
                preset_request = Some(preset);
            }
        }
    });

    // Apply GPU-affecting actions after UI traversal.
    if let Some(render_state) = frame.wgpu_render_state() {
        if add_requested {
            vp.add_layer(&render_state.device, &render_state.queue);
        }
        if add_fill_requested {
            vp.add_fill_layer(&render_state.device, &render_state.queue);
        }
        if let Some(idx) = delete_idx {
            vp.remove_layer(&render_state.device, &render_state.queue, idx);
        }
        if let Some(idx) = mask_add {
            vp.layer_stack
                .add_mask_to(idx, &render_state.device, &render_state.queue);
            // Activate that layer and drop the brush straight into mask-edit —
            // this is almost always what the user wants after clicking "+ mask".
            vp.layer_stack.active = idx;
            vp.brush.mask_edit = true;
            vp.recomposite(&render_state.device, &render_state.queue);
        }
        if let Some(idx) = mask_remove {
            vp.layer_stack.remove_mask_from(idx);
            // If we just yanked the mask off the active layer while mask-edit
            // was on, drop the toggle so paint doesn't try to hit a vanished
            // texture next frame.
            if idx == vp.layer_stack.active && vp.brush.mask_edit {
                vp.brush.mask_edit = false;
            }
            vp.recomposite(&render_state.device, &render_state.queue);
        }
        // Run any smart-mask regenerations queued during this frame's
        // panel traversal. Errors (missing source bake) are logged and
        // surfaced via the panel on the next frame.
        for idx in &smart_regen_request {
            if let Err(e) =
                vp.regenerate_smart_mask(&render_state.device, &render_state.queue, *idx)
            {
                log::warn!("smart-mask regen failed on layer {idx}: {e}");
            }
        }

        // Smart-material preset → new layer with mask + smart config.
        if let Some(preset) = preset_request {
            match vp.apply_smart_material_preset(&render_state.device, &render_state.queue, preset)
            {
                Ok(idx) => log::info!("applied {} preset as layer {idx}", preset.label()),
                Err(e) => log::warn!("preset {} regen failed: {e}", preset.label()),
            }
        }
        if needs_recomposite {
            vp.recomposite(&render_state.device, &render_state.queue);
        }
    }
}

fn env_panel(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    // Dropdown of anything in assets/hdri/ + "Procedural default".
    let bundled = crate::env::discover_bundled_hdris(
        &std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
    );
    let current_name = vp.env.name.clone();
    let mut load_path: Option<std::path::PathBuf> = None;
    let mut load_procedural = false;

    ui.horizontal(|ui| {
        ui.label("sky");
        egui::ComboBox::from_id_salt("env_dropdown")
            .selected_text(&current_name)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current_name == "procedural_studio", "procedural (default)")
                    .clicked()
                {
                    load_procedural = true;
                }
                for (name, path) in &bundled {
                    if ui
                        .selectable_label(current_name == *name, name.as_str())
                        .clicked()
                    {
                        load_path = Some(path.clone());
                    }
                }
            });
    });

    ui.horizontal(|ui| {
        if ui.button("Load HDRI…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("HDRI", &["hdr", "exr"])
                .set_title("Load environment HDRI")
                .pick_file()
            {
                load_path = Some(path);
            }
        }
    });

    ui.add(egui::Slider::new(&mut vp.env_intensity, 0.0..=4.0).text("intensity"));
    ui.add(
        egui::Slider::new(
            &mut vp.env_rotation_y,
            -std::f32::consts::PI..=std::f32::consts::PI,
        )
        .text("rotation"),
    );
    ui.checkbox(&mut vp.env_skybox_visible, "show sky");

    ui.add_space(6.0);
    ui.label("Tonemap");
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("tonemap_mode")
            .selected_text(vp.tonemap_mode.label())
            .show_ui(ui, |ui| {
                for &mode in crate::render::TonemapMode::ALL {
                    ui.selectable_value(&mut vp.tonemap_mode, mode, mode.label());
                }
            });
    });
    ui.add(
        egui::Slider::new(&mut vp.exposure_stops, -4.0..=4.0)
            .text("exposure (stops)")
            .show_value(true),
    );

    ui.add_space(6.0);
    ui.label("Grading (post-tonemap)");
    ui.add(
        egui::Slider::new(&mut vp.grading_contrast, 0.5..=2.0)
            .text("contrast")
            .show_value(true),
    );
    ui.add(
        egui::Slider::new(&mut vp.grading_saturation, 0.0..=2.0)
            .text("saturation")
            .show_value(true),
    );
    ui.add(
        egui::Slider::new(&mut vp.grading_clarity, 0.0..=1.0)
            .text("clarity")
            .show_value(true),
    );
    if ui.button("Reset grading").clicked() {
        vp.grading_contrast = 1.10;
        vp.grading_saturation = 1.10;
        vp.grading_clarity = 0.15;
    }

    ui.add_space(6.0);
    ui.label("Display");
    ui.horizontal(|ui| {
        ui.checkbox(&mut vp.fxaa.enabled, "FXAA");
        ui.checkbox(&mut vp.wireframe.visible, "wireframe");
    });
    ui.add(
        egui::Slider::new(&mut vp.fxaa.sharpen, 0.0..=1.0)
            .text("sharpen (CAS)")
            .show_value(true),
    );
    ui.add(
        egui::Slider::new(&mut vp.fxaa.dither, 0.0..=2.0)
            .text("dither")
            .show_value(true),
    );

    if let Some(render_state) = frame.wgpu_render_state() {
        if load_procedural {
            vp.env = crate::env::Environment::new_procedural(
                &render_state.device,
                &render_state.queue,
                &vp.brdf_lut,
                &vp.irradiance_baker,
                &vp.prefilter_baker,
            );
        } else if let Some(path) = load_path {
            match crate::env::Environment::load_hdr(
                &render_state.device,
                &render_state.queue,
                &vp.brdf_lut,
                &vp.irradiance_baker,
                &vp.prefilter_baker,
                &path,
            ) {
                Ok(env) => {
                    log::info!(
                        "loaded HDRI {} ({}×{}, {} mips)",
                        env.name,
                        env.width,
                        env.height,
                        env.mip_count
                    );
                    vp.env = env;
                }
                Err(e) => {
                    log::error!("failed to load {}: {e:#}", path.display());
                }
            }
        }
    }
}

fn mesh_maps_panel(ui: &mut egui::Ui, vp: &mut Viewport, frame: &eframe::Frame) {
    use crate::bake::integration::MapKind;

    // Built-in MRT bake (world normal + position) — kept separate
    // because the projection brush relies on it being in lockstep with
    // the renderer's own pipeline, not the texture-baker path.
    ui.horizontal(|ui| {
        let baked = vp.mesh_maps.baked;
        ui.weak(if baked {
            "world maps: baked"
        } else {
            "world maps: not baked"
        });
        if ui.button("Bake").clicked() {
            if let Some(rs) = frame.wgpu_render_state() {
                vp.bake_mesh_maps(&rs.device, &rs.queue);
            }
        }
    });
    ui.weak("World normal + position baked via MRT. Used by projection paint.");

    ui.add_space(8.0);
    ui.label("Texture-baker maps");

    // Last-bake status strip. The bake call is sync (UI freezes for
    // its duration) — this is the post-hoc receipt so the user knows
    // it actually finished and how long it took. Threaded mid-bake
    // progress is a follow-up.
    if let Some(s) = vp.last_bake_status.clone() {
        let secs = s.duration_ms as f32 / 1000.0;
        let txt = match s.error {
            None => format!(
                "✓ {} baked in {secs:.2}s · {}×{} · {} tiles",
                s.kind.label(),
                s.resolution,
                s.resolution,
                s.tile_count
            ),
            Some(ref err) => format!("✗ {} bake failed: {err}", s.kind.label()),
        };
        let color = if s.error.is_none() {
            egui::Color32::from_rgb(120, 220, 140)
        } else {
            egui::Color32::from_rgb(255, 140, 120)
        };
        ui.colored_label(color, txt);
    }

    // Source meshes — optional high-poly + cage that drive low→high
    // projection bakes (normal / AO with HP detail / curvature / etc).
    // Both default to None (self-bakes from the low-poly).
    let lp_vert_count = vp.cpu_mesh().positions.len();
    ui.horizontal(|ui| {
        ui.label("HP");
        let label = vp
            .bake_high_poly_label
            .clone()
            .unwrap_or_else(|| "(none)".into());
        ui.weak(label);
    });
    ui.horizontal(|ui| {
        if ui.button("Load HP…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Mesh", &["obj", "gltf", "glb"])
                .pick_file()
            {
                match crate::bake::integration::load_high_poly(&path) {
                    Ok(m) => {
                        let stem = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "hp".into());
                        let tri = m.indices.len();
                        log::info!("loaded high-poly {} ({tri} triangles)", path.display());
                        vp.bake_high_poly = Some(m);
                        vp.bake_high_poly_label = Some(format!("{stem} · {tri} tris"));
                        vp.bake_high_poly_path = Some(path);
                    }
                    Err(e) => log::error!("HP load failed: {e}"),
                }
            }
        }
        if ui
            .add_enabled(vp.bake_high_poly.is_some(), egui::Button::new("Clear HP"))
            .clicked()
        {
            vp.bake_high_poly = None;
            vp.bake_high_poly_label = None;
            vp.bake_high_poly_path = None;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Cage");
        let label = vp
            .bake_cage_label
            .clone()
            .unwrap_or_else(|| "(none)".into());
        ui.weak(label);
    });
    ui.horizontal(|ui| {
        if ui
            .button("Load cage…")
            .on_hover_text(format!(
                "Cage must share the low-poly's vertex count ({lp_vert_count})"
            ))
            .clicked()
        {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Mesh", &["obj", "gltf", "glb"])
                .pick_file()
            {
                match crate::bake::integration::load_cage(&path) {
                    Ok(m) => {
                        let stem = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "cage".into());
                        if m.positions.len() != lp_vert_count {
                            log::error!(
                                "cage vertex count mismatch: cage={} vs low-poly={}",
                                m.positions.len(),
                                lp_vert_count
                            );
                            vp.bake_cage_label = Some(format!(
                                "⚠ {} verts ≠ {} (low-poly)",
                                m.positions.len(),
                                lp_vert_count
                            ));
                        } else {
                            log::info!(
                                "loaded cage {} ({} verts)",
                                path.display(),
                                m.positions.len()
                            );
                            vp.bake_cage = Some(m);
                            vp.bake_cage_label = Some(format!("{stem} · {lp_vert_count} verts"));
                            vp.bake_cage_path = Some(path);
                        }
                    }
                    Err(e) => log::error!("cage load failed: {e}"),
                }
            }
        }
        if ui
            .add_enabled(vp.bake_cage.is_some(), egui::Button::new("Clear cage"))
            .clicked()
        {
            vp.bake_cage = None;
            vp.bake_cage_label = None;
            vp.bake_cage_path = None;
        }
    });

    ui.add_space(4.0);

    // Bake settings — apply to every per-kind bake below. Kept compact;
    // power users can dial individual ray counts, defaults are sensible.
    egui::CollapsingHeader::new("Bake settings")
        .default_open(false)
        .show(ui, |ui| {
            let s = &mut vp.bake_settings;
            ui.add(
                egui::Slider::new(&mut s.ao_rays, 8..=512)
                    .logarithmic(true)
                    .text("AO rays"),
            );
            ui.add(
                egui::Slider::new(&mut s.thickness_rays, 8..=512)
                    .logarithmic(true)
                    .text("Thickness rays"),
            );
            ui.add(
                egui::Slider::new(&mut s.bent_rays, 8..=512)
                    .logarithmic(true)
                    .text("Bent normal rays"),
            );
            ui.add(egui::Slider::new(&mut s.spread_angle_deg, 0.0..=180.0).text("Spread°"));
            ui.add(egui::Slider::new(&mut s.max_distance, 0.0..=10.0).text("Max distance"));
            ui.add(egui::Slider::new(&mut s.aa_factor, 1..=8).text("AA factor"));
            ui.checkbox(&mut s.use_gpu, "Use GPU acceleration");
        });

    // Per-kind row: status + bake/clear buttons. Stale if the slot is
    // populated but its baked-at-revision lags the live mesh revision.
    let kinds: &[MapKind] = &[
        MapKind::AmbientOcclusion,
        MapKind::Curvature,
        MapKind::Thickness,
        MapKind::Height,
        MapKind::Normal,
        MapKind::BentNormal,
        MapKind::Id,
    ];

    let live_rev = vp.mesh_revision;

    let mut bake_request: Option<MapKind> = None;
    let mut clear_request: Option<MapKind> = None;
    let mut bake_all_request = false;

    ui.horizontal(|ui| {
        if ui
            .button("Bake all")
            .on_hover_text("Bake every map at the current resolution")
            .clicked()
        {
            bake_all_request = true;
        }
    });

    egui::Grid::new("mesh_maps_grid")
        .num_columns(4)
        .spacing(egui::vec2(6.0, 4.0))
        .show(ui, |ui| {
            for &k in kinds {
                ui.label(k.label());
                let slot = vp.mesh_maps.slot(k);
                let (status, color) = match slot {
                    None => ("·", egui::Color32::from_gray(120)),
                    Some(b) if b.tile_count != vp.tiles().len() as u32 => {
                        ("⚠ tiles", egui::Color32::from_rgb(255, 180, 100))
                    }
                    Some(_) if vp.mesh_maps.baked_at_revision != live_rev => {
                        ("⚠ stale", egui::Color32::from_rgb(255, 180, 100))
                    }
                    Some(_) => ("✓", egui::Color32::from_rgb(120, 220, 140)),
                };
                ui.colored_label(color, status);
                if ui.button("Bake").clicked() {
                    bake_request = Some(k);
                }
                if ui
                    .add_enabled(slot.is_some(), egui::Button::new("Clear"))
                    .clicked()
                {
                    clear_request = Some(k);
                }
                ui.end_row();
            }
        });

    // Drive bakes after the grid — borrows `vp` mutably and the closure
    // above borrowed it through `slot()`/`baked_at_revision`. Splitting
    // the read and write phases keeps the borrow checker happy.
    if let Some(rs) = frame.wgpu_render_state() {
        if bake_all_request {
            for &k in kinds {
                run_bake(vp, &rs.device, &rs.queue, k);
            }
        } else if let Some(k) = bake_request {
            run_bake(vp, &rs.device, &rs.queue, k);
        }
    }
    if let Some(k) = clear_request {
        vp.mesh_maps.clear(k);
    }

    ui.add_space(6.0);
    ui.label("Tessellation (for displacement)");
    let mut level = vp.subdivision_level;
    // Level 5 = 1024× triangles per base — expensive but workable on
    // small to mid meshes. Beyond that the no-dedupe storage balloons
    // VRAM hard.
    if ui
        .add(egui::Slider::new(&mut level, 0..=5).text("subdivision"))
        .changed()
    {
        if let Some(rs) = frame.wgpu_render_state() {
            vp.set_subdivision(&rs.device, level);
        }
    }
}

/// Run a single per-kind bake using the viewport's current mesh, tile
/// layout, paint-target resolution, and bake settings. Stamps the
/// resulting `BakedMap` into the matching `MeshMaps` slot.
fn run_bake(
    vp: &mut Viewport,
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    kind: crate::bake::integration::MapKind,
) {
    let tiles: Vec<u32> = vp.paint_target().tiles.iter().copied().collect();
    let resolution = vp.tile_resolution();
    let cpu_mesh = vp.cpu_mesh().clone();
    let hp_ref = vp.bake_high_poly.as_ref();
    let cage_ref = vp.bake_cage.as_ref();
    let started = std::time::Instant::now();
    let result = crate::bake::integration::bake_map(
        device,
        queue,
        &cpu_mesh,
        hp_ref,
        cage_ref,
        &tiles,
        resolution,
        kind,
        &vp.bake_settings,
    );
    let duration_ms = started.elapsed().as_millis();
    match result {
        Ok(baked) => {
            log::info!(
                "baked {:?} at {}×{} for {} tiles in {} ms",
                kind,
                resolution,
                resolution,
                tiles.len(),
                duration_ms
            );
            vp.mesh_maps.set(kind, baked);
            vp.mesh_maps.baked_at_revision = vp.mesh_revision;
            vp.last_bake_status = Some(crate::viewport::BakeStatus {
                kind,
                duration_ms,
                tile_count: tiles.len() as u32,
                resolution,
                error: None,
            });
        }
        Err(e) => {
            log::error!("bake {:?} failed: {e}", kind);
            vp.last_bake_status = Some(crate::viewport::BakeStatus {
                kind,
                duration_ms,
                tile_count: tiles.len() as u32,
                resolution,
                error: Some(e),
            });
        }
    }
}

/// Per-layer smart-mask sub-block. Renders below the mask row of any
/// layer that has a mask. Toggling `Smart` writes / clears
/// `Mask::smart`; param changes queue a regenerate request that the
/// caller drains after the panel traversal completes.
fn smart_mask_subpanel(
    ui: &mut egui::Ui,
    vp: &mut Viewport,
    layer_idx: usize,
    _frame: &eframe::Frame,
    regen_request: &mut Vec<usize>,
) {
    use crate::paint::smart_mask::{SmartMaskParams, SmartMaskSource};

    // Snapshot which sources are currently bakeable before taking the
    // mutable borrow on `mask` — the borrow checker rejects accessing
    // `vp.mesh_maps` from inside the source-picker closure once we've
    // already aliased `vp.layer_stack.layers[…].mask`.
    let availability: Vec<(SmartMaskSource, bool)> = SmartMaskSource::ALL
        .iter()
        .map(|&src| (src, source_is_baked(vp, src)))
        .collect();

    let Some(mask) = vp.layer_stack.layers[layer_idx].mask.as_mut() else {
        return;
    };

    ui.indent(("smart_mask_indent", layer_idx), |ui| {
        // Smart-mask toggle — small icon button (sparkle = "auto-
        // generated"). Filled blue when on, neutral when off, mirrors
        // the Properties tab strip's selection style. Reads cleaner
        // than a plain checkbox in the dense layer row.
        let is_smart_before = mask.smart.is_some();
        let fill = if is_smart_before {
            egui::Color32::from_rgb(46, 92, 148)
        } else {
            ui.style().visuals.widgets.inactive.bg_fill
        };
        let toggle = ui.add(
            egui::Button::new(
                egui::RichText::new(format!("{}  Smart mask", egui_phosphor::regular::SPARKLE))
                    .size(14.0),
            )
            .fill(fill)
            .min_size(egui::vec2(140.0, 22.0)),
        );
        if toggle.clicked() {
            if is_smart_before {
                mask.smart = None;
                // Don't recompose — the existing texture stays as a
                // starting point the user can paint over.
            } else {
                mask.smart = Some(SmartMaskParams::default());
                regen_request.push(layer_idx);
            }
        }

        let Some(params) = mask.smart.as_mut() else {
            return;
        };

        let prev = *params;

        ui.horizontal(|ui| {
            ui.label("Source");
            let cur = params.source;
            egui::ComboBox::from_id_salt(("smart_mask_source", layer_idx))
                .selected_text(cur.label())
                .show_ui(ui, |ui| {
                    for &(src, available) in &availability {
                        let label = if available {
                            src.label().to_string()
                        } else {
                            format!("{} (no bake)", src.label())
                        };
                        ui.add_enabled_ui(available, |ui| {
                            ui.selectable_value(&mut params.source, src, label);
                        });
                    }
                });
        });

        ui.add(egui::Slider::new(&mut params.low, 0.0..=1.0).text("low"));
        ui.add(egui::Slider::new(&mut params.high, 0.0..=1.0).text("high"));
        ui.add(egui::Slider::new(&mut params.contrast, 0.1..=4.0).text("contrast"));
        ui.checkbox(&mut params.invert, "Invert");
        if ui.button("Refresh").clicked() {
            regen_request.push(layer_idx);
        }

        if *params != prev {
            // Keep low < high so the smoothstep curve never collapses
            // to a 1-LSB step that artists can't see.
            if params.low > params.high {
                std::mem::swap(&mut params.low, &mut params.high);
            }
            regen_request.push(layer_idx);
        }
    });
}

fn source_is_baked(vp: &Viewport, src: crate::paint::smart_mask::SmartMaskSource) -> bool {
    use crate::paint::smart_mask::SmartMaskSource as S;
    match src {
        S::AoCrevice => vp.mesh_maps.ao.is_some(),
        S::CurvatureConvex | S::CurvatureConcave => vp.mesh_maps.curvature.is_some(),
        S::Thickness => vp.mesh_maps.thickness.is_some(),
        // World-Y up reads world_normal which always lives on
        // MeshMaps (1×1 dummy until the user runs the world-maps
        // bake). Treat it as always available — the UX is "click
        // Bake on world maps" rather than gating the row.
        S::WorldYUp => true,
    }
}

fn paint_target_section(
    ui: &mut egui::Ui,
    vp: &mut Viewport,
    frame: &eframe::Frame,
    status: &mut String,
) {
    let tiles = vp.tiles().to_vec();
    let cur_res = vp.tile_resolution();
    let vram = vp.paint_target_vram_bytes();
    ui.horizontal(|ui| {
        ui.label("resolution");
        let mut new_res = cur_res;
        egui::ComboBox::from_id_salt("tile_res_combo")
            .selected_text(format!("{cur_res}×{cur_res}"))
            .show_ui(ui, |ui| {
                for &r in &[1024u32, 2048, 4096, 8192] {
                    ui.selectable_value(&mut new_res, r, format!("{r}×{r}"));
                }
            });
        if new_res != cur_res {
            if let Some(rs) = frame.wgpu_render_state() {
                vp.set_tile_resolution(&rs.device, &rs.queue, new_res);
                *status = format!(
                    "Rebuilt paint target @ {new_res}×{new_res} (painted content discarded)"
                );
                log::info!("{}", *status);
            }
        }
    });
    ui.label(format!("tile count   {}", tiles.len()));
    ui.label(format!("vram approx  {}", human_bytes(vram)));
    egui::ScrollArea::vertical()
        .id_salt("tile_list_scroll")
        .max_height(120.0)
        .show(ui, |ui| {
            egui::Grid::new("tile_grid").num_columns(4).show(ui, |ui| {
                for (i, tid) in tiles.iter().enumerate() {
                    ui.label(format!("{tid}"));
                    if (i + 1) % 4 == 0 {
                        ui.end_row();
                    }
                }
            });
        });
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Render the 2D UV atlas painting view. Shows `paint_target.base_color`
/// composited tiles at their UDIM positions, supports pan (drag) + zoom
/// (scroll), paints via `Viewport::stamp_at_uv` on left-click drag.
fn uv_view_body(
    ui: &mut egui::Ui,
    vp: &mut Viewport,
    frame: &eframe::Frame,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
    thumb_ids: &mut Vec<Option<egui::TextureId>>,
    thumb_channel: &mut Option<crate::render::ViewMode>,
    thumb_layer_idx: &mut Option<usize>,
    channel: &mut crate::render::ViewMode,
    show_wireframe: &mut bool,
    undocked: &mut bool,
    last_paint_atlas_uv: &mut Option<egui::Vec2>,
    last_paint_pressure: &mut Option<f32>,
) {
    use crate::render::ViewMode;
    // Allowed atlas channels — everything that has a per-tile, 2D image:
    // the four paintable channels plus Normal (view-only) and Mask (the
    // active layer's mask, also paintable via mask_edit). Material and
    // WorldNormalBaked aren't flat UV atlases, so they're excluded.
    let atlas_channels: &[ViewMode] = &[
        ViewMode::BaseColor,
        ViewMode::Roughness,
        ViewMode::Metallic,
        ViewMode::Normal,
        ViewMode::Mask,
        ViewMode::Height,
    ];

    // Header row: channel picker, toggles, hints, dock/undock button.
    let active_layer_idx = vp.layer_stack.active;
    let active_has_mask = vp.layer_stack.active_layer().mask.is_some();
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("uv_view_channel")
            .selected_text(channel.label())
            .show_ui(ui, |ui| {
                for &m in atlas_channels {
                    // Gray out Mask when the active layer has none — picking
                    // it would just display the missing-mask placeholder.
                    let enabled = m != ViewMode::Mask || active_has_mask;
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.selectable_value(channel, m, m.label());
                    });
                }
            });
        ui.checkbox(show_wireframe, "UV wireframe");
        ui.weak(" · RMB / MMB drag to pan · scroll to zoom");
        // Right-aligned dock/undock toggle.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (label, tip) = if *undocked {
                ("⮌ Dock", "Dock the UV view back into the main layout")
            } else {
                ("⮎ Undock", "Pop out the UV view into a floating window")
            };
            if ui.button(label).on_hover_text(tip).clicked() {
                *undocked = !*undocked;
            }
        });
    });
    ui.separator();
    // Ensure our cache matches the current tile count, selected channel,
    // and — when showing Mask — the currently active layer. Mask views
    // live on each Layer, so switching layers requires re-registering.
    let tiles = vp.paint_target().tiles.to_vec();
    let channel_changed = *thumb_channel != Some(*channel);
    let mask_layer_changed =
        *channel == ViewMode::Mask && *thumb_layer_idx != Some(active_layer_idx);
    if thumb_ids.len() != tiles.len() || channel_changed || mask_layer_changed {
        thumb_ids.clear();
        thumb_ids.resize(tiles.len(), None);
        *thumb_channel = Some(*channel);
        *thumb_layer_idx = Some(active_layer_idx);
    }

    // Resolve the source tile-array texture view for the selected channel.
    // Mask is per-layer (active-layer's mask); everything else lives on
    // the PaintTarget. None for Mask when the active layer has no mask —
    // caller renders a placeholder.
    let pick_view = |i: usize| -> Option<&eframe::wgpu::TextureView> {
        match *channel {
            ViewMode::BaseColor => Some(&vp.paint_target().base_color_layer_views[i]),
            ViewMode::Roughness => Some(&vp.paint_target().roughness_layer_views[i]),
            ViewMode::Metallic => Some(&vp.paint_target().metallic_layer_views[i]),
            ViewMode::Normal => Some(&vp.paint_target().normal_layer_views[i]),
            ViewMode::Height => Some(&vp.paint_target().displacement_layer_views[i]),
            ViewMode::Mask => vp
                .layer_stack
                .active_layer()
                .mask
                .as_ref()
                .map(|m| &m.layer_views[i]),
            // These aren't per-tile atlases — the combo box excludes
            // them, but handle defensively so we don't crash if the
            // picker is ever bypassed.
            ViewMode::Material | ViewMode::WorldNormalBaked => None,
        }
    };

    // Lazily register each tile's current-channel layer view as an egui
    // texture — re-uses the live GPU view, so paints update the atlas
    // immediately.
    if let Some(rs) = frame.wgpu_render_state() {
        let mut renderer = rs.renderer.write();
        for i in 0..thumb_ids.len() {
            if thumb_ids[i].is_none() {
                if let Some(view) = pick_view(i) {
                    thumb_ids[i] = Some(renderer.register_native_texture(
                        &rs.device,
                        view,
                        eframe::wgpu::FilterMode::Linear,
                    ));
                }
            }
        }
    }

    let (rect, response) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

    // Pan (right-mouse drag or middle-mouse drag — leave LMB for paint).
    if response.dragged_by(egui::PointerButton::Secondary)
        || response.dragged_by(egui::PointerButton::Middle)
    {
        *pan += response.drag_delta();
    }

    // Zoom (scroll anywhere in the panel, cursor anchor).
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    if response.hovered() && scroll.abs() > 0.0 {
        let factor = (1.0 + scroll * 0.0015).clamp(0.1, 10.0);
        let cursor = response.hover_pos().unwrap_or_else(|| rect.center());
        let before = cursor - rect.min - *pan;
        *zoom = (*zoom * factor).clamp(16.0, 8000.0);
        let after = before * factor;
        *pan += before - after;
    }

    // Transforms: atlas UV → screen pixel.
    let to_screen = |uv: egui::Vec2| rect.min + *pan + uv * *zoom;

    // Draw tiles.
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(30, 30, 35));
    for (i, &tile_id) in tiles.iter().enumerate() {
        let Some(id) = thumb_ids[i] else { continue };
        // Inline UDIM offset: tile 1001 = (0,0), 1002 = (1,0), 1011 = (0,1) …
        let n = tile_id.saturating_sub(1001);
        let tu = (n % 10) as f32;
        let tv = (n / 10) as f32;
        // Tile UV range is [(tu, tv), (tu+1, tv+1)]. UV.y=0 lives at
        // the top of the tile texture (matches persist/export), so map
        // (tu, tv+1) → bottom-left of screen.
        let tl = to_screen(egui::vec2(tu, tv));
        let br = to_screen(egui::vec2(tu + 1.0, tv + 1.0));
        let tile_rect = egui::Rect::from_two_pos(tl, br);
        if tile_rect.intersects(rect) {
            painter.image(
                id,
                tile_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            // UDIM label at top-left of the tile.
            painter.text(
                tile_rect.left_top() + egui::vec2(4.0, 2.0),
                egui::Align2::LEFT_TOP,
                format!("{tile_id}"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(255, 220, 100),
            );
        }
    }

    // UDIM grid — 1-unit squares covering the visible extent.
    let grid_stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 30),
    );
    let uv_min =
        (egui::vec2(rect.min.x, rect.min.y) - egui::vec2(rect.min.x, rect.min.y) - *pan) / *zoom;
    let _ = uv_min;
    // Approximate visible UV range from rect.
    let uv_tl = (rect.min - rect.min - *pan) / *zoom;
    let uv_br = (rect.max - rect.min - *pan) / *zoom;
    let ix_min = uv_tl.x.floor() as i32;
    let ix_max = uv_br.x.ceil() as i32;
    let iy_min = uv_tl.y.floor() as i32;
    let iy_max = uv_br.y.ceil() as i32;
    // Hard cap the visible UDIM-grid extent so a zoomed-out / panned
    // view doesn't generate tens of thousands of line segments and
    // blow the egui index buffer (the wgpu validation error caps at
    // 256 MB per buffer, hit by SimReady-class assets with a wide
    // UV range). ±50 tiles covers every UDIM scheme we care about
    // (1001..1100), and the user can pan within that.
    const GRID_HALF: i32 = 50;
    let ix_min = ix_min.max(-GRID_HALF);
    let ix_max = ix_max.min(GRID_HALF);
    let iy_min = iy_min.max(-GRID_HALF);
    let iy_max = iy_max.min(GRID_HALF);
    for x in ix_min..=ix_max {
        let a = to_screen(egui::vec2(x as f32, iy_min as f32));
        let b = to_screen(egui::vec2(x as f32, iy_max as f32));
        painter.line_segment([a, b], grid_stroke);
    }
    for y in iy_min..=iy_max {
        let a = to_screen(egui::vec2(ix_min as f32, y as f32));
        let b = to_screen(egui::vec2(ix_max as f32, y as f32));
        painter.line_segment([a, b], grid_stroke);
    }

    // UV wireframe — every triangle becomes three line segments. egui
    // expands each segment into ~4 vertices + 6 indices for AA, so a
    // 100K-triangle mesh would land at ~1.8 M indices just for the
    // wireframe — fine on its own, but the SimReady warehouse rack
    // pushed past wgpu's per-buffer cap when combined with all the
    // other UI in the same egui frame. Skip the overlay when the
    // mesh is denser than a "small enough to read at a glance"
    // threshold; show a hint so the user knows the toggle still
    // worked, the renderer just declined the draw.
    if *show_wireframe {
        let mesh = vp.cpu_mesh();
        const TRI_CAP: usize = 50_000;
        if mesh.indices.len() > TRI_CAP {
            ui.ctx().debug_painter().text(
                rect.center_top() + egui::vec2(0.0, 20.0),
                egui::Align2::CENTER_TOP,
                format!(
                    "UV wireframe hidden — {} triangles exceed cap of {}",
                    mesh.indices.len(),
                    TRI_CAP,
                ),
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(180),
            );
        } else {
            let stroke = egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(100, 220, 255, 120),
            );
            let visible = rect.expand(8.0);
            for &[i0, i1, i2] in &mesh.indices {
                let uv0 = mesh.uvs[i0 as usize];
                let uv1 = mesh.uvs[i1 as usize];
                let uv2 = mesh.uvs[i2 as usize];
                let a = to_screen(egui::vec2(uv0.x, uv0.y));
                let b = to_screen(egui::vec2(uv1.x, uv1.y));
                let c = to_screen(egui::vec2(uv2.x, uv2.y));
                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                    // Cheap AABB cull — skip segments entirely
                    // off-panel.
                    let seg = egui::Rect::from_two_pos(p0, p1);
                    if seg.intersects(visible) {
                        painter.line_segment([p0, p1], stroke);
                    }
                }
            }
        }
    }

    // Paint: map LMB drag → atlas UV sequence (interpolated between last
    // frame and this frame so fast drags don't leave gaps) → tile +
    // local_uv → stamp. Mirrors the 3D viewport's `stamp_positions`
    // pattern; step spacing tracks the brush so dense strokes stay dense
    // and small brushes don't explode the step count.
    // Only paintable view channels stamp; Normal / Material / WorldNormal
    // are view-only. Paint routing already flips `mask_edit` appropriately
    // via `apply_uv_channel_to_brush`, so we can just check whether the
    // current ViewMode maps to any paint channel at all.
    let paintable = paint_channel_from_view_mode(*channel).is_some();
    let primary_active = paintable
        && (response.dragged_by(egui::PointerButton::Primary)
            || response.clicked_by(egui::PointerButton::Primary));
    let pressure_now = vp.refresh_input_pressure(ui.ctx(), primary_active);
    if primary_active {
        if let Some(pos) = response.interact_pointer_pos() {
            let atlas_uv_now = (pos - rect.min - *pan) / *zoom;
            // Step distance in UV: ~half the brush radius in UV units (so
            // stamps overlap by ~50%), floored to a small minimum.
            let uv_radius = (vp.brush.effective_radius(pressure_now) / 400.0).max(1e-4);
            let step_uv = (uv_radius * 0.5).max(1e-3);
            const MAX_STEPS: u32 = 128;

            let sequence: Vec<(egui::Vec2, f32)> = match *last_paint_atlas_uv {
                Some(prev) => {
                    let delta = atlas_uv_now - prev;
                    let dist = delta.length();
                    let steps = ((dist / step_uv).ceil() as u32).clamp(1, MAX_STEPS);
                    let prev_pressure = last_paint_pressure.unwrap_or(pressure_now);
                    (1..=steps)
                        .map(|i| {
                            let t = i as f32 / steps as f32;
                            (prev + delta * t, lerp(prev_pressure, pressure_now, t))
                        })
                        .collect()
                }
                None => vec![(atlas_uv_now, pressure_now)],
            };

            if let Some(rs) = frame.wgpu_render_state() {
                for (uv, pressure) in &sequence {
                    let tile_u = uv.x.floor() as i32;
                    let tile_v = uv.y.floor() as i32;
                    if tile_u < 0 || tile_v < 0 || tile_u >= 10 {
                        continue;
                    }
                    let tile_id = 1001 + tile_u as u32 + 10 * tile_v as u32;
                    let Some(tile_idx) = vp.paint_target().layer_for_tile(tile_id) else {
                        continue;
                    };
                    let local_uv = [uv.x.fract(), uv.y.fract()];
                    vp.stamp_at_uv(&rs.device, &rs.queue, tile_idx, local_uv, *pressure);
                }
                ui.ctx().request_repaint();
            }
            *last_paint_atlas_uv = Some(atlas_uv_now);
            *last_paint_pressure = Some(pressure_now);
        }
    } else {
        // Stroke ended — clear the anchor so the next stroke starts fresh
        // and doesn't interpolate a long line from wherever the previous
        // one ended.
        *last_paint_atlas_uv = None;
        *last_paint_pressure = None;
    }

    // Brush cursor preview — an outline ring matching the stamped radius
    // at the current zoom. Only visible while the pointer hovers the
    // panel and LMB isn't down on another widget.
    if response.hovered() {
        if let Some(hover) = response.hover_pos() {
            let preview_pressure = if primary_active { pressure_now } else { 1.0 };
            let uv_radius = vp.brush.effective_radius(preview_pressure) / 400.0;
            let screen_radius = uv_radius * *zoom;
            // Outer outline — dark, then bright, so it reads over both
            // light and dark atlas content.
            painter.circle_stroke(
                hover,
                screen_radius,
                egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180)),
            );
            painter.circle_stroke(
                hover,
                screen_radius,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 255, 255)),
            );
            // Inner hardness ring — matches how the brush falls off in
            // the shader (radius * hardness stays at full opacity, beyond
            // that falls to zero).
            let inner = screen_radius
                * vp.brush
                    .effective_hardness(preview_pressure)
                    .clamp(0.0, 1.0);
            if inner > 1.5 {
                painter.circle_stroke(
                    hover,
                    inner,
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 90),
                    ),
                );
            }
        }
    }

    // Simple footer — channel name + zoom readout.
    painter.text(
        rect.left_bottom() + egui::vec2(6.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        format!(
            "UV atlas · {:.0} px/UV · RMB or MMB drag to pan · scroll to zoom",
            *zoom
        ),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(160, 170, 180),
    );
}

fn material_slots_section(ui: &mut egui::Ui, vp: &Viewport) -> Option<MaterialSlot> {
    let mut clicked: Option<MaterialSlot> = None;
    let active_name = vp.layer_stack.layers[vp.layer_stack.active].name.clone();
    ui.weak(format!("Active layer: {}", active_name));
    ui.add_space(4.0);
    egui::Grid::new("material_slots_grid")
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            for &slot in MaterialSlot::ALL {
                ui.label(slot.label());
                if ui.button("Assign…").clicked() {
                    clicked = Some(slot);
                }
                ui.end_row();
            }
        });
    ui.add_space(4.0);
    ui.weak(
        "Clicking Assign uploads the picked texture into the active layer's \
         channel. Layers + masks compose on top.",
    );
    clicked
}

fn material_factors_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.horizontal(|ui| {
        ui.label("base color ×");
        ui.color_edit_button_rgb(&mut vp.base_color_factor);
    });
    ui.add(egui::Slider::new(&mut vp.metallic_factor, 0.0..=2.0).text("metallic ×"));
    ui.add(egui::Slider::new(&mut vp.roughness_factor, 0.0..=2.0).text("roughness ×"));
    ui.add(egui::Slider::new(&mut vp.normal_scale, 0.0..=2.0).text("normal scale"));
    ui.add(
        egui::Slider::new(&mut vp.displacement_scale, 0.0..=1.0)
            .text("displacement scale")
            .show_value(true),
    );

    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new("Bake effects").strong());
    ui.weak("Multipliers / blends on the baked mesh maps from the Bake tab.");

    // Baked-normal blend — mixes the painted tangent-space normal
    // toward the baked map from `MeshMaps::normal`. 0 = painted only,
    // 1 = baked only. Disabled until the user actually bakes Normal,
    // because the dummy is flat (0,0,1) and would just kill detail.
    let has_baked_normal = vp.mesh_maps.normal.is_some();
    ui.add_enabled(
        has_baked_normal,
        egui::Slider::new(&mut vp.baked_normal_blend, 0.0..=1.0)
            .text("baked normal blend")
            .show_value(true),
    );
    if !has_baked_normal {
        ui.weak("(bake the Normal map on the Bake tab to enable)");
    }

    // AO intensity — Substance / Marmoset call this "Ambient occlusion
    // strength". 0 = disabled (pass-through), 1 = full baked AO,
    // > 1 = exaggerated. Always interactable: when no AO is baked the
    // shader's source dummy is 1.0, so `mix(1.0, 1.0, k) = 1.0` and
    // the slider is harmless until a real bake lands.
    let has_baked_ao = vp.mesh_maps.ao.is_some();
    ui.add(
        egui::Slider::new(&mut vp.ao_intensity, 0.0..=2.0)
            .text("AO intensity")
            .show_value(true),
    );
    if !has_baked_ao {
        ui.weak("(bake AO on the Bake tab to see it on the model)");
    }
}

fn light_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    use crate::lights::{Light, LightKind, MAX_LIGHTS};

    // Header — add buttons + count.
    ui.horizontal(|ui| {
        ui.label(format!("Lights ({}/{})", vp.lights.len(), MAX_LIGHTS));
        let at_cap = vp.lights.len() >= MAX_LIGHTS;
        ui.add_enabled_ui(!at_cap, |ui| {
            if ui
                .button("+ Directional")
                .on_hover_text("Add a directional light")
                .clicked()
            {
                vp.lights.push(Light::new_directional());
            }
            if ui
                .button("+ Spot")
                .on_hover_text("Add a spot light")
                .clicked()
            {
                vp.lights.push(Light::new_spot());
            }
        });
    });
    if vp.lights.is_empty() {
        ui.weak("No analytic lights. HDRI (Environment) drives all lighting.");
        return;
    }

    let mut remove_idx: Option<usize> = None;
    for (i, light) in vp.lights.iter_mut().enumerate() {
        let header = format!(
            "#{i} {}{}",
            light.kind.label(),
            if light.enabled { "" } else { " — disabled" },
        );
        egui::CollapsingHeader::new(header)
            .id_salt(("light_block", i))
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut light.enabled, "enabled");
                    if ui.small_button("✕").on_hover_text("Remove light").clicked() {
                        remove_idx = Some(i);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("color");
                    let mut color = light.color;
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        light.color = color;
                    }
                });
                ui.add(
                    egui::Slider::new(&mut light.intensity, 0.0..=1000.0)
                        .text("intensity")
                        .show_value(true)
                        // Logarithmic so 0–10 still has good slider
                        // resolution while 100–1000 stays reachable
                        // without the slider getting unusable. Matches
                        // UsdLux's radiometric range, where 1 is dim
                        // and 1000+ is sun-bright.
                        .logarithmic(true),
                );
                match light.kind {
                    LightKind::Directional => {
                        ui.horizontal(|ui| {
                            ui.label("dir");
                            ui.add(
                                egui::DragValue::new(&mut light.direction[0])
                                    .speed(0.02)
                                    .prefix("x:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.direction[1])
                                    .speed(0.02)
                                    .prefix("y:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.direction[2])
                                    .speed(0.02)
                                    .prefix("z:"),
                            );
                        });
                    }
                    LightKind::Spot => {
                        ui.horizontal(|ui| {
                            ui.label("pos");
                            ui.add(
                                egui::DragValue::new(&mut light.position[0])
                                    .speed(0.05)
                                    .prefix("x:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.position[1])
                                    .speed(0.05)
                                    .prefix("y:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.position[2])
                                    .speed(0.05)
                                    .prefix("z:"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("dir");
                            ui.add(
                                egui::DragValue::new(&mut light.direction[0])
                                    .speed(0.02)
                                    .prefix("x:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.direction[1])
                                    .speed(0.02)
                                    .prefix("y:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut light.direction[2])
                                    .speed(0.02)
                                    .prefix("z:"),
                            );
                        });
                        // Inner ≤ outer is the invariant the shader's
                        // smoothstep assumes. Clamp inner against
                        // whatever outer is and vice versa after the
                        // user edits either slider.
                        ui.add(
                            egui::Slider::new(&mut light.inner_cone_deg, 0.0..=170.0)
                                .text("inner cone (deg)")
                                .show_value(true),
                        );
                        if light.inner_cone_deg > light.outer_cone_deg {
                            light.outer_cone_deg = light.inner_cone_deg;
                        }
                        ui.add(
                            egui::Slider::new(&mut light.outer_cone_deg, 0.0..=179.0)
                                .text("outer cone (deg)")
                                .show_value(true),
                        );
                        if light.outer_cone_deg < light.inner_cone_deg {
                            light.inner_cone_deg = light.outer_cone_deg;
                        }
                    }
                }
            });
    }
    if let Some(i) = remove_idx {
        vp.lights.remove(i);
    }
}

fn human_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if n >= GB {
        format!("{:.2} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

fn is_usdz_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("usdz"))
        .unwrap_or(false)
}

fn is_usd_stage_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "usd" | "usda" | "usdc" | "usdz"
            )
        })
        .unwrap_or(false)
}

fn default_converted_usd_path(source: &Path) -> PathBuf {
    let mut out = source.to_path_buf();
    out.set_extension("usda");
    out
}

fn extracted_texture_lookup(
    extracted: &[crate::usdz::ExtractedTexture],
) -> std::collections::HashMap<String, PathBuf> {
    let mut lookup = std::collections::HashMap::new();
    for tex in extracted {
        let normalized = crate::usdz::normalize_package_path(&tex.package_path);
        lookup.insert(normalized.to_ascii_lowercase(), tex.path.clone());
        if let Some(name) = Path::new(&normalized).file_name().and_then(|s| s.to_str()) {
            lookup.insert(name.to_ascii_lowercase(), tex.path.clone());
        }
    }
    lookup
}

fn package_inner_path(asset_ref: &str) -> Option<&str> {
    let end = asset_ref.rfind(']')?;
    let start = asset_ref[..end].rfind('[')?;
    Some(&asset_ref[start + 1..end])
}

fn resolve_texture_reference(
    stage_path: &Path,
    texture_ref: &str,
    extracted_lookup: &std::collections::HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    let trimmed = texture_ref.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(inner) = package_inner_path(trimmed) {
        let key = crate::usdz::normalize_package_path(inner).to_ascii_lowercase();
        if let Some(path) = extracted_lookup.get(&key) {
            return Some(path.clone());
        }
    }

    let normalized = crate::usdz::normalize_package_path(trimmed);
    if let Some(path) = extracted_lookup.get(&normalized.to_ascii_lowercase()) {
        return Some(path.clone());
    }
    if let Some(name) = Path::new(&normalized).file_name().and_then(|s| s.to_str()) {
        if let Some(path) = extracted_lookup.get(&name.to_ascii_lowercase()) {
            return Some(path.clone());
        }
    }

    let raw = PathBuf::from(trimmed);
    if raw.is_absolute() && raw.exists() {
        return Some(raw);
    }
    if let Some(parent) = stage_path.parent() {
        let candidate = parent.join(trimmed);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn classify_texture_pin(path_or_ref: &str) -> Option<crate::material_graph::ShaderPin> {
    let stem = Path::new(path_or_ref)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path_or_ref)
        .to_ascii_lowercase();
    let name = path_or_ref.to_ascii_lowercase();

    if stem.contains("basecolor")
        || stem.contains("base_color")
        || stem.contains("albedo")
        || stem.contains("diffuse")
        || stem.ends_with("_base")
    {
        return Some(crate::material_graph::ShaderPin::DiffuseColor);
    }
    if stem.contains("normal") || stem.contains("_nrm") || stem.contains("-nrm") {
        return Some(crate::material_graph::ShaderPin::Normal);
    }
    if stem.contains("emissive") || stem.contains("emission") || stem.contains("_emit") {
        return Some(crate::material_graph::ShaderPin::EmissionColor);
    }
    if stem.contains("occlusion") || stem.contains("_occl") || stem.ends_with("_ao") {
        return Some(crate::material_graph::ShaderPin::Occlusion);
    }
    if stem.contains("_rough")
        || stem.ends_with("rough")
        || stem.contains("roughness_rough")
        || stem.contains("-rough")
    {
        return Some(crate::material_graph::ShaderPin::Roughness);
    }
    if stem.contains("_metal")
        || stem.ends_with("metal")
        || stem.contains("metallic")
        || stem.contains("metalness")
    {
        return Some(crate::material_graph::ShaderPin::Metallic);
    }
    if name.contains("roughness") {
        return Some(crate::material_graph::ShaderPin::Roughness);
    }
    if name.contains("metal") {
        return Some(crate::material_graph::ShaderPin::Metallic);
    }
    None
}

fn material_slot_for_pin(pin: crate::material_graph::ShaderPin) -> Option<MaterialSlot> {
    match pin {
        crate::material_graph::ShaderPin::DiffuseColor => Some(MaterialSlot::BaseColor),
        crate::material_graph::ShaderPin::Roughness => Some(MaterialSlot::Roughness),
        crate::material_graph::ShaderPin::Metallic => Some(MaterialSlot::Metallic),
        crate::material_graph::ShaderPin::Normal => Some(MaterialSlot::Normal),
        _ => None,
    }
}

fn texture_udim(path: &Path) -> Option<u32> {
    let text = path.file_name()?.to_string_lossy();
    let bytes = text.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    for i in 0..=bytes.len() - 4 {
        let slice = &bytes[i..i + 4];
        if slice.iter().all(u8::is_ascii_digit) {
            let value = std::str::from_utf8(slice).ok()?.parse::<u32>().ok()?;
            if (1001..=1999).contains(&value) {
                return Some(value);
            }
        }
    }
    None
}

fn collect_resolved_stage_textures(
    stage_path: &Path,
    materials: &[crate::usd::MeshMaterialInfo],
    extracted: &[crate::usdz::ExtractedTexture],
) -> Vec<ResolvedStageTexture> {
    let lookup = extracted_texture_lookup(extracted);
    let mut out = Vec::new();
    for mat in materials {
        for texture_ref in &mat.texture_paths {
            let Some(pin) = classify_texture_pin(texture_ref) else {
                continue;
            };
            let Some(path) = resolve_texture_reference(stage_path, texture_ref, &lookup) else {
                continue;
            };
            out.push(ResolvedStageTexture {
                udim: texture_udim(&path),
                slot: material_slot_for_pin(pin),
                path,
            });
        }
    }

    if out.is_empty() {
        for texture in extracted {
            let Some(pin) = classify_texture_pin(&texture.package_path) else {
                continue;
            };
            out.push(ResolvedStageTexture {
                udim: texture_udim(&texture.path),
                slot: material_slot_for_pin(pin),
                path: texture.path.clone(),
            });
        }
    }
    out
}

fn stage_material_texture_groups(
    stage_path: &Path,
    materials: &[crate::usd::MeshMaterialInfo],
    extracted: &[crate::usdz::ExtractedTexture],
) -> std::collections::BTreeMap<DetectedMaterialTextures, Vec<String>> {
    let lookup = extracted_texture_lookup(extracted);
    let mut groups = std::collections::BTreeMap::new();
    for mat in materials {
        let mut detected = DetectedMaterialTextures::empty(
            mat.uv_primvar_name
                .clone()
                .unwrap_or_else(|| "st".to_string()),
        );
        for texture_ref in &mat.texture_paths {
            let Some(pin) = classify_texture_pin(texture_ref) else {
                continue;
            };
            if let Some(path) = resolve_texture_reference(stage_path, texture_ref, &lookup) {
                detected.set(pin, path);
            }
        }
        if !detected.is_empty() {
            groups
                .entry(detected)
                .or_insert_with(Vec::new)
                .push(mat.prim_path.clone());
        }
    }
    groups
}

fn usda_asset_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .replace('@', "%40")
}

fn write_texture_shader(
    text: &mut String,
    shader_name: &str,
    path: &Path,
    source_color_space: &str,
    normal_map: bool,
) {
    use std::fmt::Write as _;
    let asset = usda_asset_path(path);
    let _ = writeln!(text, "\n    def Shader \"{shader_name}\"");
    let _ = writeln!(text, "    {{");
    let _ = writeln!(text, "        uniform token info:id = \"UsdUVTexture\"");
    if normal_map {
        let _ = writeln!(text, "        float4 inputs:bias = (-1, -1, -1, -1)");
        let _ = writeln!(text, "        float4 inputs:scale = (2, 2, 2, 2)");
    }
    let _ = writeln!(text, "        asset inputs:file = @{asset}@");
    let _ = writeln!(
        text,
        "        token inputs:sourceColorSpace = \"{source_color_space}\""
    );
    let _ = writeln!(
        text,
        "        float2 inputs:st.connect = </Material/stReader.outputs:result>"
    );
    let _ = writeln!(text, "        token inputs:wrapS = \"repeat\"");
    let _ = writeln!(text, "        token inputs:wrapT = \"repeat\"");
    let _ = writeln!(text, "        float outputs:r");
    let _ = writeln!(text, "        float3 outputs:rgb");
    let _ = writeln!(text, "    }}");
}

fn write_usd_preview_material(
    path: &Path,
    textures: &DetectedMaterialTextures,
) -> anyhow::Result<()> {
    use std::fmt::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut text = String::new();
    let _ = writeln!(text, "#usda 1.0");
    let _ = writeln!(text, "(");
    let _ = writeln!(text, "    defaultPrim = \"Material\"");
    let _ = writeln!(
        text,
        "    doc = \"Imported USDZ material network generated by forge-paint.\""
    );
    let _ = writeln!(text, ")");
    let _ = writeln!(text);
    let _ = writeln!(text, "def Material \"Material\"");
    let _ = writeln!(text, "{{");
    let _ = writeln!(
        text,
        "    token outputs:surface.connect = </Material/Surface.outputs:surface>"
    );
    let _ = writeln!(text);
    let _ = writeln!(text, "    def Shader \"Surface\"");
    let _ = writeln!(text, "    {{");
    let _ = writeln!(
        text,
        "        uniform token info:id = \"UsdPreviewSurface\""
    );
    if textures.base_color.is_some() {
        let _ = writeln!(
            text,
            "        color3f inputs:diffuseColor.connect = </Material/BaseColorTexture.outputs:rgb>"
        );
    } else {
        let _ = writeln!(
            text,
            "        color3f inputs:diffuseColor = (0.8, 0.8, 0.8)"
        );
    }
    if textures.metallic.is_some() {
        let _ = writeln!(
            text,
            "        float inputs:metallic.connect = </Material/MetallicTexture.outputs:r>"
        );
    } else {
        let _ = writeln!(text, "        float inputs:metallic = 0.0");
    }
    if textures.roughness.is_some() {
        let _ = writeln!(
            text,
            "        float inputs:roughness.connect = </Material/RoughnessTexture.outputs:r>"
        );
    } else {
        let _ = writeln!(text, "        float inputs:roughness = 0.5");
    }
    if textures.normal.is_some() {
        let _ = writeln!(
            text,
            "        normal3f inputs:normal.connect = </Material/NormalTexture.outputs:rgb>"
        );
    }
    if textures.emission.is_some() {
        let _ = writeln!(
            text,
            "        color3f inputs:emissiveColor.connect = </Material/EmissionTexture.outputs:rgb>"
        );
    }
    if textures.occlusion.is_some() {
        let _ = writeln!(
            text,
            "        float inputs:occlusion.connect = </Material/OcclusionTexture.outputs:r>"
        );
    }
    let _ = writeln!(text, "        token outputs:surface");
    let _ = writeln!(text, "    }}");

    let _ = writeln!(text);
    let _ = writeln!(text, "    def Shader \"stReader\"");
    let _ = writeln!(text, "    {{");
    let _ = writeln!(
        text,
        "        uniform token info:id = \"UsdPrimvarReader_float2\""
    );
    let _ = writeln!(
        text,
        "        token inputs:varname = \"{}\"",
        textures.uv_primvar
    );
    let _ = writeln!(text, "        float2 outputs:result");
    let _ = writeln!(text, "    }}");

    if let Some(path) = &textures.base_color {
        write_texture_shader(&mut text, "BaseColorTexture", path, "sRGB", false);
    }
    if let Some(path) = &textures.metallic {
        write_texture_shader(&mut text, "MetallicTexture", path, "raw", false);
    }
    if let Some(path) = &textures.roughness {
        write_texture_shader(&mut text, "RoughnessTexture", path, "raw", false);
    }
    if let Some(path) = &textures.normal {
        write_texture_shader(&mut text, "NormalTexture", path, "raw", true);
    }
    if let Some(path) = &textures.emission {
        write_texture_shader(&mut text, "EmissionTexture", path, "sRGB", false);
    }
    if let Some(path) = &textures.occlusion {
        write_texture_shader(&mut text, "OcclusionTexture", path, "raw", false);
    }
    let _ = writeln!(text, "}}");

    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

impl App {
    fn export_textures_dialog(&mut self, frame: &eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.status = "No GPU render state available.".to_string();
            return;
        };
        let Some(vp) = &self.viewport else {
            self.status = "Viewport not initialized yet.".to_string();
            return;
        };

        let Some(dir) = rfd::FileDialog::new()
            .set_title("Export textures to folder")
            .pick_folder()
        else {
            return;
        };

        match crate::export::export_tiles(
            &render_state.device,
            &render_state.queue,
            vp.paint_target(),
            &dir,
        ) {
            Ok(exports) => {
                let hook_msg = run_post_export_hook(&dir);
                self.status = format!(
                    "Exported {} files to {}{hook_msg}",
                    exports.len(),
                    dir.display()
                );
                log::info!("{}", self.status);
                for e in &exports {
                    log::info!("  {} {} -> {}", e.channel, e.udim, e.path.display());
                }
            }
            Err(e) => {
                self.status = format!("Export failed: {e:#}");
                log::error!("{}", self.status);
            }
        }
    }

    fn open_stage_dialog(&mut self, frame: &eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("USD / OBJ", &["usd", "usda", "usdc", "usdz", "obj"])
            .add_filter("USD", &["usd", "usda", "usdc", "usdz"])
            .add_filter("OBJ", &["obj"])
            .set_title("Open")
            .pick_file()
        else {
            return;
        };
        self.open_stage_or_offer_conversion(frame, path);
    }

    fn open_stage_or_offer_conversion(&mut self, frame: &eframe::Frame, path: PathBuf) {
        if is_usd_stage_path(&path) || path.to_string_lossy().contains("://") {
            self.load_usd(frame, path);
            return;
        }
        if let Some(kind) = ConvertibleModelKind::from_path(&path) {
            self.pending_conversion = Some(PendingModelConversion { source: path, kind });
            self.status = format!("{} needs USD conversion.", kind.label());
            return;
        }
        self.status = format!(
            "Unsupported file type: {}. Choose a USD file directly, or choose OBJ to convert.",
            path.display()
        );
    }

    fn convert_model_dialog(&mut self, frame: &eframe::Frame, request: PendingModelConversion) {
        let suggested = default_converted_usd_path(&request.source);
        let mut dialog = rfd::FileDialog::new()
            .add_filter("USDA", &["usda"])
            .add_filter("USD", &["usd", "usda", "usdc"])
            .set_title("Save converted USD");
        if let Some(parent) = suggested.parent() {
            dialog = dialog.set_directory(parent);
        }
        if let Some(name) = suggested.file_name().and_then(|n| n.to_str()) {
            dialog = dialog.set_file_name(name);
        }
        let Some(dest) = dialog.save_file() else {
            self.status = "Conversion cancelled.".to_string();
            return;
        };

        self.status = format!(
            "Converting {} to {}…",
            request.source.display(),
            dest.display()
        );
        match request.kind {
            ConvertibleModelKind::Obj => {
                match crate::obj_to_usd::convert_obj_to_usd(&request.source, &dest) {
                    Ok(summary) => {
                        self.status = format!(
                            "Converted {} to {} — {} verts, {} tris",
                            request.source.display(),
                            dest.display(),
                            summary.vertices,
                            summary.triangles
                        );
                        log::info!("{}", self.status);
                        self.load_usd(frame, dest);
                    }
                    Err(e) => {
                        self.status = format!("OBJ conversion failed: {e:#}");
                        log::error!("{}", self.status);
                    }
                }
            }
        }
    }

    fn reset_stage_material_bindings(&mut self) {
        self.material_bindings.clear();
        self.active_binding_id = None;
        self.material_graph = crate::material_graph::MaterialGraph::default();
        self.last_pushed_bindings.clear();
    }

    fn restore_material_bindings_from_sidecar(
        &mut self,
        side: &crate::project::ProjectSidecar,
        work_dir: &Path,
    ) -> usize {
        let to_restore: Vec<&crate::project::BoundMaterialBinding> =
            if !side.bound_materials.is_empty() {
                side.bound_materials.iter().collect()
            } else if let Some(b) = side.bound_material.as_ref() {
                vec![b]
            } else {
                Vec::new()
            };
        let mut restored = 0usize;
        let generated_dir = work_dir.join("imported_materials");
        for binding in to_restore {
            let target =
                std::fs::canonicalize(&binding.source).unwrap_or_else(|_| binding.source.clone());
            let asset = self.browser.materials.iter().find(|m| {
                std::fs::canonicalize(&m.source)
                    .map(|p| p == target)
                    .unwrap_or(false)
                    || m.source == binding.source
            });
            match asset {
                Some(mat) => {
                    let new_id = self.next_binding_id;
                    self.next_binding_id += 1;
                    self.material_bindings.push(MaterialBindingInstance {
                        id: new_id,
                        source: mat.source.clone(),
                        prim_path: mat.prim_path.clone(),
                        kind: mat.kind,
                        inputs: binding.inputs,
                        target_prims: binding.target_prims.clone(),
                        assigned: true,
                    });
                    self.material_graph.spawn_shader_node(new_id);
                    self.active_binding_id = Some(new_id);
                    restored += 1;
                }
                None if binding.source.exists() && !binding.source.starts_with(&generated_dir) => {
                    let new_id = self.next_binding_id;
                    self.next_binding_id += 1;
                    self.material_bindings.push(MaterialBindingInstance {
                        id: new_id,
                        source: binding.source.clone(),
                        prim_path: binding.prim_path.clone(),
                        kind: crate::assets::MaterialKind::UsdPreviewSurface,
                        inputs: binding.inputs,
                        target_prims: binding.target_prims.clone(),
                        assigned: true,
                    });
                    self.material_graph.spawn_shader_node(new_id);
                    self.active_binding_id = Some(new_id);
                    restored += 1;
                }
                None if binding.source.starts_with(&generated_dir) => {
                    log::debug!(
                        "sidecar generated material will be rebuilt from stage textures: {}",
                        binding.source.display()
                    );
                }
                None => {
                    log::warn!(
                        "sidecar bound material source not in library and missing on disk: {}",
                        binding.source.display()
                    );
                }
            }
        }
        if restored > 0 {
            self.last_pushed_bindings.clear();
        }
        restored
    }

    fn import_resolved_stage_textures(
        &mut self,
        textures: &[ResolvedStageTexture],
        render_state: &eframe::egui_wgpu::RenderState,
    ) -> usize {
        let mut imported = 0usize;
        let mut renderer = render_state.renderer.write();
        for texture in textures {
            match self.browser.import_texture_once(
                &texture.path,
                &render_state.device,
                &render_state.queue,
                &mut renderer,
            ) {
                Ok(_) => imported += 1,
                Err(e) => log::warn!("stage texture import {}: {e:#}", texture.path.display()),
            }
        }
        imported
    }

    fn apply_resolved_stage_textures_to_active_layer(
        &mut self,
        textures: &[ResolvedStageTexture],
        render_state: &eframe::egui_wgpu::RenderState,
    ) -> usize {
        let Some(vp) = &mut self.viewport else {
            return 0;
        };
        let active_idx = vp.layer_stack.active;
        let tile_count = vp.paint_target().tiles.len() as u32;
        let res = vp.tile_resolution();
        let mut applied = 0usize;
        let mut seen = std::collections::HashSet::new();

        for texture in textures {
            let Some(slot) = texture.slot else {
                continue;
            };
            let Some(asset_idx) = self.browser.texture_index_for_source(&texture.path) else {
                continue;
            };
            let Some(asset) = self.browser.textures.get(asset_idx) else {
                continue;
            };

            let layers: Vec<u32> = if let Some(udim) = texture.udim {
                vp.paint_target()
                    .layer_for_tile(udim)
                    .map(|layer| vec![layer])
                    .unwrap_or_default()
            } else {
                (0..tile_count).collect()
            };
            let layer = &vp.layer_stack.layers[active_idx];
            for tile_layer in layers {
                if !seen.insert((slot, tile_layer, texture.path.clone())) {
                    continue;
                }
                let result = match slot {
                    MaterialSlot::BaseColor => assets::apply_as_base_color_tile(
                        &render_state.queue,
                        asset,
                        layer,
                        tile_layer,
                        res,
                    ),
                    MaterialSlot::Roughness => assets::apply_as_roughness_tile(
                        &render_state.queue,
                        asset,
                        layer,
                        tile_layer,
                        res,
                    ),
                    MaterialSlot::Metallic => assets::apply_as_metallic_tile(
                        &render_state.queue,
                        asset,
                        layer,
                        tile_layer,
                        res,
                    ),
                    MaterialSlot::Normal => assets::apply_as_normal_tile(
                        &render_state.queue,
                        asset,
                        layer,
                        tile_layer,
                        res,
                    ),
                };
                match result {
                    Ok(()) => applied += 1,
                    Err(e) => log::warn!(
                        "stage texture apply {} to {:?}: {e:#}",
                        texture.path.display(),
                        slot
                    ),
                }
            }
        }

        if applied > 0 {
            vp.recomposite(&render_state.device, &render_state.queue);
        }
        applied
    }

    fn replicate_stage_materials_in_editor(
        &mut self,
        work_dir: &Path,
        groups: std::collections::BTreeMap<DetectedMaterialTextures, Vec<String>>,
    ) -> usize {
        let material_dir = work_dir.join("imported_materials");
        let mut created = 0usize;
        for (textures, target_prims) in groups {
            let source = material_dir.join(format!("stage_material_{created:02}.usda"));
            if let Err(e) = write_usd_preview_material(&source, &textures) {
                log::warn!("stage material replication failed: {e:#}");
                continue;
            }

            let new_id = self.next_binding_id;
            self.next_binding_id += 1;
            self.material_bindings.push(MaterialBindingInstance {
                id: new_id,
                source,
                prim_path: "/Material".to_string(),
                kind: crate::assets::MaterialKind::UsdPreviewSurface,
                inputs: crate::assets::MaterialInputs::default(),
                target_prims,
                assigned: true,
            });
            let shader_node = self.material_graph.spawn_shader_node(new_id);
            for (idx, (path, pin)) in textures.texture_nodes().into_iter().enumerate() {
                let texture_node = self.material_graph.spawn_texture_node_at(
                    path,
                    egui::pos2(40.0 + (created as f32 * 240.0), 330.0 + idx as f32 * 86.0),
                );
                self.material_graph
                    .connect_texture_to_shader(texture_node, shader_node, pin);
            }
            self.active_binding_id = Some(new_id);
            created += 1;
        }
        if created > 0 {
            self.last_pushed_bindings.clear();
        }
        created
    }

    fn load_usd(&mut self, frame: &eframe::Frame, path: PathBuf) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.status = "No GPU render state available.".to_string();
            return;
        };
        if self.viewport.is_none() {
            self.status = "Viewport not initialized yet.".to_string();
            return;
        }

        match crate::usd::load_stage_merged_with_materials(&path) {
            Ok(loaded_stage) => {
                let crate::usd::LoadedStage {
                    mesh: cpu,
                    materials: stage_materials,
                } = loaded_stage;
                let tris = cpu.indices.len();
                let verts = cpu.positions.len();
                self.reset_stage_material_bindings();
                if let Some(vp) = &mut self.viewport {
                    vp.set_mesh(&render_state.device, &render_state.queue, &cpu);
                }

                let work_dir = crate::persist::default_work_dir(&path);
                let extracted_textures = if is_usdz_path(&path) {
                    let out_dir = work_dir.join("embedded_textures");
                    match crate::usdz::extract_embedded_textures(&path, &out_dir) {
                        Ok(textures) => {
                            if !textures.is_empty() {
                                log::info!(
                                    "extracted {} embedded USDZ texture(s) to {}",
                                    textures.len(),
                                    out_dir.display()
                                );
                            }
                            textures
                        }
                        Err(e) => {
                            log::warn!("USDZ texture extraction failed: {e:#}");
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };
                let resolved_textures =
                    collect_resolved_stage_textures(&path, &stage_materials, &extracted_textures);
                let imported_texture_count =
                    self.import_resolved_stage_textures(&resolved_textures, render_state);

                let loaded_n = if let Some(vp) = &mut self.viewport {
                    let loaded_n = crate::persist::load_sidecars(
                        &render_state.queue,
                        vp.active_layer(),
                        vp.tiles(),
                        vp.tile_resolution(),
                        &work_dir,
                    );
                    if loaded_n > 0 {
                        vp.recomposite(&render_state.device, &render_state.queue);
                    }
                    loaded_n
                } else {
                    0
                };

                // Project sidecar (JSON) — apply bake settings,
                // material factors, smart-mask params. HP / cage are
                // re-loaded from disk if their paths still exist.
                let mut restored_binding_count = 0usize;
                match crate::project::load_sidecar(&work_dir) {
                    Ok(Some(side)) => {
                        // Re-load HP / cage by replaying the same
                        // routine the panel buttons run, so the in-
                        // memory caches and labels stay coherent.
                        if let Some(vp) = &mut self.viewport {
                            if let Some(ref hp_path) = side.bake.high_poly_path {
                                match crate::bake::integration::load_high_poly(hp_path) {
                                    Ok(m) => {
                                        let stem = hp_path
                                            .file_stem()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| "hp".into());
                                        let tri = m.indices.len();
                                        vp.bake_high_poly = Some(m);
                                        vp.bake_high_poly_label =
                                            Some(format!("{stem} · {tri} tris"));
                                        vp.bake_high_poly_path = Some(hp_path.clone());
                                    }
                                    Err(e) => {
                                        log::warn!("sidecar HP load failed: {e}");
                                    }
                                }
                            }
                            if let Some(ref cage_path) = side.bake.cage_path {
                                match crate::bake::integration::load_cage(cage_path) {
                                    Ok(m) => {
                                        let stem = cage_path
                                            .file_stem()
                                            .map(|s| s.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| "cage".into());
                                        let lp_vert_count = vp.cpu_mesh().positions.len();
                                        if m.positions.len() == lp_vert_count {
                                            vp.bake_cage = Some(m);
                                            vp.bake_cage_label =
                                                Some(format!("{stem} · {lp_vert_count} verts"));
                                            vp.bake_cage_path = Some(cage_path.clone());
                                        } else {
                                            log::warn!(
                                                "sidecar cage vertex mismatch: cage={} vs low-poly={}",
                                                m.positions.len(),
                                                lp_vert_count
                                            );
                                        }
                                    }
                                    Err(e) => log::warn!("sidecar cage load failed: {e}"),
                                }
                            }
                            vp.apply_sidecar(&render_state.device, &render_state.queue, &side);
                        }
                        restored_binding_count =
                            self.restore_material_bindings_from_sidecar(&side, &work_dir);
                    }
                    Ok(None) => {}
                    Err(e) => log::warn!("project sidecar parse failed: {e:#}"),
                }

                let applied_texture_count = if loaded_n == 0 {
                    self.apply_resolved_stage_textures_to_active_layer(
                        &resolved_textures,
                        render_state,
                    )
                } else {
                    0
                };

                let replicated_material_count = if restored_binding_count == 0 {
                    let groups =
                        stage_material_texture_groups(&path, &stage_materials, &extracted_textures);
                    self.replicate_stage_materials_in_editor(&work_dir, groups)
                } else {
                    0
                };

                self.current_usd_path = Some(path.clone());
                self.stage_browser.ensure_loaded(&path);
                let mut extras = Vec::new();
                if loaded_n > 0 {
                    extras.push(format!("loaded {loaded_n} sidecar(s)"));
                }
                if !extracted_textures.is_empty() {
                    extras.push(format!(
                        "extracted {} embedded texture(s)",
                        extracted_textures.len()
                    ));
                }
                if imported_texture_count > 0 {
                    extras.push(format!("imported {imported_texture_count} texture(s)"));
                }
                if applied_texture_count > 0 {
                    extras.push(format!("seeded {applied_texture_count} paint tile(s)"));
                }
                if replicated_material_count > 0 {
                    extras.push(format!(
                        "replicated {replicated_material_count} material(s)"
                    ));
                }
                let extra_msg = if extras.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", extras.join(", "))
                };
                let tile_count = self
                    .viewport
                    .as_ref()
                    .map(|vp| vp.tiles().len())
                    .unwrap_or(0);
                self.status = format!(
                    "Loaded {} — {verts} verts, {tris} tris, {tile_count} UDIM tiles{extra_msg}",
                    path.display(),
                );
                log::info!("{}", self.status);
            }
            Err(e) => {
                self.status = format!("Failed to load {}: {e:#}", path.display());
                log::error!("{}", self.status);
            }
        }
    }

    fn apply_viewport_selection(&mut self, selection: ViewportSelection, ctx: &egui::Context) {
        if self
            .stage_browser
            .select_path(&selection.prim_path, selection.multi)
        {
            self.status = format!("Selected {}", selection.prim_path);
            ctx.request_repaint();
        }
    }

    fn save_to_work_dir(&mut self, frame: &eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.status = "No GPU render state available.".to_string();
            return;
        };
        let Some(vp) = &self.viewport else {
            self.status = "Viewport not initialized yet.".to_string();
            return;
        };
        let Some(usd_path) = &self.current_usd_path else {
            self.status = "No USD loaded — use Save As… to pick a folder.".to_string();
            return;
        };

        let dir = crate::persist::default_work_dir(usd_path);
        match crate::persist::save_sidecars(
            &render_state.device,
            &render_state.queue,
            vp.paint_target(),
            &dir,
        ) {
            Ok(exports) => {
                // Project sidecar — JSON metadata that travels next to
                // the PNGs (HP / cage paths, bake settings, smart-mask
                // params, material factors).
                let mut sidecar = vp.build_sidecar();
                // Fold the currently-bound library material (if any)
                // into the sidecar so it round-trips across sessions.
                // The viewport doesn't own the binding state (it's
                // app-level), hence the post-build augment here.
                sidecar.bound_materials = self
                    .material_bindings
                    .iter()
                    // Only persist bindings that have actually been
                    // assigned via the right-click menu. Unassigned
                    // shader nodes are session-only scratch.
                    .filter(|b| b.assigned)
                    .map(|b| crate::project::BoundMaterialBinding {
                        source: b.source.clone(),
                        prim_path: b.prim_path.clone(),
                        inputs: b.inputs,
                        target_prims: b.target_prims.clone(),
                    })
                    .collect();
                // Clear the legacy single-binding field on save so we
                // don't double-restore on next reopen.
                sidecar.bound_material = None;
                if let Err(e) = crate::project::save_sidecar(&dir, &sidecar) {
                    log::warn!("project sidecar save failed: {e:#}");
                }
                let hook_msg = run_post_export_hook(&dir);
                self.status = format!(
                    "Saved {} files to {}{hook_msg}",
                    exports.len(),
                    dir.display()
                );
                log::info!("{}", self.status);
            }
            Err(e) => {
                self.status = format!("Save failed: {e:#}");
                log::error!("{}", self.status);
            }
        }
    }

    fn hydra_delegate_needs_startup_probe(delegate: Option<&str>) -> bool {
        // Every Hydra delegate goes through the out-of-process probe
        // on Windows, not just hdNSI: HydraNSI can block forever in
        // 3Delight startup, and the first Hgi/GL bring-up can hang or
        // terminate in native code on machines without a usable GL
        // context (remote desktop, missing driver). The probe turns
        // both into a visible overlay instead of a frozen or vanishing
        // app. Storm clears the probe in under a second and hdNSI in a
        // few, and the result is only re-checked per (stage, delegate)
        // change, so the cost stays negligible.
        #[cfg(windows)]
        {
            let _ = delegate;
            true
        }
        #[cfg(not(windows))]
        {
            let _ = delegate;
            false
        }
    }

    fn draw_hydra_startup_overlay(
        ui: &mut egui::Ui,
        rect: egui::Rect,
        title: &str,
        detail: &str,
        retry: bool,
    ) -> bool {
        let panel_w = rect.width().clamp(280.0, 460.0);
        let panel_h = if retry { 118.0 } else { 86.0 };
        let panel_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(panel_w, panel_h));
        let mut retry_clicked = false;
        egui::Area::new(egui::Id::new("hydra_startup_overlay"))
            .order(egui::Order::Foreground)
            .fixed_pos(panel_rect.min)
            .show(ui.ctx(), |ui| {
                ui.set_width(panel_w);
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(8, 8, 10, 235))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(180, 200, 255),
                    ))
                    .inner_margin(12.0)
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new(title)
                                    .strong()
                                    .color(egui::Color32::from_rgb(235, 240, 255)),
                            );
                            ui.add_space(4.0);
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(detail)
                                        .size(12.0)
                                        .color(egui::Color32::from_gray(205)),
                                )
                                .wrap(),
                            );
                            if retry {
                                ui.add_space(8.0);
                                retry_clicked = ui.button("Retry").clicked();
                            }
                        });
                    });
            });
        retry_clicked
    }

    fn ensure_hydra_startup_ready(
        ui: &mut egui::Ui,
        rect: egui::Rect,
        stage_path: &Path,
        delegate: Option<&str>,
        hydra_startup: &mut Option<HydraStartupState>,
    ) -> bool {
        let delegate = delegate.filter(|id| !id.is_empty());
        // Short UI label for overlay / error text: "Storm", "3Delight",
        // or the raw plugin ID for delegates without a mapping.
        let delegate_label = delegate
            .map(crate::hydra_view::delegate_label)
            .unwrap_or("Hydra");
        if !Self::hydra_delegate_needs_startup_probe(delegate) {
            if hydra_startup
                .as_ref()
                .is_some_and(|state| !state.matches(stage_path, delegate))
            {
                *hydra_startup = None;
            }
            return true;
        }

        if hydra_startup
            .as_ref()
            .is_some_and(|state| !state.matches(stage_path, delegate))
        {
            *hydra_startup = None;
        }

        if hydra_startup.is_none() {
            match HydraStartupProbe::start(stage_path, delegate) {
                Ok(probe) => {
                    log::info!(
                        "Hydra startup probe started for {} on {}",
                        delegate.unwrap_or("default delegate"),
                        stage_path.display()
                    );
                    *hydra_startup = Some(HydraStartupState::Running(probe));
                }
                Err(e) => {
                    let message =
                        format!("Could not start the {delegate_label} startup check: {e:#}");
                    log::warn!("Hydra startup probe launch failed: {message}");
                    *hydra_startup = Some(HydraStartupState::Failed(HydraStartupFailure {
                        stage_path: stage_path.to_path_buf(),
                        delegate: delegate.map(std::string::ToString::to_string),
                        message,
                    }));
                }
            }
        }

        let mut ready = false;
        let mut failure = None;
        if let Some(HydraStartupState::Running(probe)) = hydra_startup.as_mut() {
            match probe.child.try_wait() {
                Ok(Some(status)) if status.success() => {
                    log::info!(
                        "Hydra startup probe completed OK for {}",
                        delegate.unwrap_or("default delegate")
                    );
                    ready = true;
                }
                Ok(Some(status)) => {
                    let status_text = status.code().map_or_else(
                        || "terminated before reporting an exit code".to_string(),
                        |code| format!("exit code {code}"),
                    );
                    failure = Some(format!(
                        "{delegate_label} startup check failed ({status_text}). See forge-paint.log beside forge-paint.exe for the Hydra breadcrumb."
                    ));
                }
                Ok(None) => {
                    let elapsed = probe.started_at.elapsed();
                    if elapsed >= HYDRA_STARTUP_PROBE_TIMEOUT {
                        failure = Some(format!(
                            "{delegate_label} startup check timed out after {}s. The UI stayed alive; see forge-paint.log for the last Hydra breadcrumb.",
                            HYDRA_STARTUP_PROBE_TIMEOUT.as_secs()
                        ));
                    }
                }
                Err(e) => {
                    failure = Some(format!(
                        "{delegate_label} startup check failed to report status: {e}"
                    ));
                }
            }
        }

        if ready {
            *hydra_startup = None;
            return true;
        }
        if let Some(message) = failure {
            log::warn!(
                "Hydra startup probe failed for {} on {}: {}",
                delegate.unwrap_or("default delegate"),
                stage_path.display(),
                message
            );
            *hydra_startup = Some(HydraStartupState::Failed(HydraStartupFailure {
                stage_path: stage_path.to_path_buf(),
                delegate: delegate.map(std::string::ToString::to_string),
                message,
            }));
        }

        match hydra_startup.as_ref() {
            Some(HydraStartupState::Running(probe)) => {
                let elapsed = probe.started_at.elapsed().as_secs_f32();
                let detail =
                    format!("Checking {delegate_label} outside the UI process ({elapsed:.1}s).");
                let title = format!("Starting {delegate_label}");
                Self::draw_hydra_startup_overlay(ui, rect, &title, &detail, false);
            }
            Some(HydraStartupState::Failed(failure)) => {
                let title = format!("{delegate_label} did not start");
                let retry = Self::draw_hydra_startup_overlay(
                    ui,
                    rect,
                    &title,
                    &failure.message,
                    true,
                );
                if retry {
                    *hydra_startup = None;
                }
            }
            None => {}
        }
        false
    }

    /// Render the Hydra preview into the central viewport, full-size,
    /// in place of the wgpu painter. Solaris-style mode swap: orbit /
    /// zoom input still drives `vp.camera` (so flipping back to wgpu
    /// keeps the same framing), the renderer badge in the top-left is
    /// clickable and toggles `renderer_mode` back, the delegate combo
    /// overlay sits top-right.
    ///
    /// Associated function (no `&mut self`) so the caller can hand us
    /// disjoint field borrows of App — that's how we update
    /// `self.hydra`, `self.hydra_egui_tex`, etc. without colliding
    /// with the `&mut self.viewport` already in scope inside the
    /// central panel closure.
    fn draw_hydra_central(
        ui: &mut egui::Ui,
        frame: &eframe::Frame,
        vp: &mut Viewport,
        hydra_slot: &mut Option<crate::hydra_view::HydraView>,
        hydra_egui_tex: &mut Option<egui::TextureHandle>,
        hydra_delegate: &mut Option<String>,
        hydra_startup: &mut Option<HydraStartupState>,
        hydra_paint_cache_dir: &mut Option<std::path::PathBuf>,
        hydra_paint_sync_seq: &mut u64,
        hydra_paint_sync_status: &mut Option<String>,
        hydra_paint_sync_pending: &mut bool,
        show_render: &mut bool,
        show_proxy: &mut bool,
        show_guides: &mut bool,
        material_bindings: &[MaterialBindingInstance],
        last_pushed_bindings: &mut std::collections::HashMap<u64, MaterialBindingSnapshot>,
        stage_path: Option<&std::path::Path>,
        request_swap_renderer: &mut bool,
    ) -> Option<ViewportSelection> {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let w = (rect.width() as u32).max(64);
        let h = (rect.height() as u32).max(64);
        let aspect = rect.width() / rect.height();

        // Continuous repaint while the panel is up — keeps path-
        // tracer convergence going and orbit feeling real-time.
        ui.ctx().request_repaint();

        // Background fill so the not-yet-rendered area reads as part
        // of the panel rather than the dark canvas behind everything.
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgb(5, 5, 6));

        // Camera nav — same plumbing as the wgpu side, same scroll
        // wheel for zoom. Updates `vp.camera`, which is what we
        // snapshot below to drive Hydra's view matrix.
        let scroll_dy = if response.hovered() {
            ui.input(|i| i.smooth_scroll_delta.y)
        } else {
            0.0
        };
        vp.camera.handle_input(&response, scroll_dy);

        // Paint-in-Hydra is intentionally dropped — Hydra mode is
        // visualization-only. Brush input lives in wgpu mode; the
        // mode-entry sync below re-authors the painted material so
        // any new strokes show up here without extra ceremony.

        // Stop here if there's no stage to open — show a placeholder
        // overlay so the canvas isn't just black with no explanation.
        let Some(path) = stage_path else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open a USD stage to see the Hydra preview.",
                egui::FontId::proportional(14.0),
                egui::Color32::from_gray(200),
            );
            let _ = request_swap_renderer;
            let _ = frame;
            return None;
        };

        // Drop the previous `HydraView` if the user has loaded a
        // different stage since we constructed it — Hydra's C++
        // Renderer holds a `UsdStageRefPtr` baked in at construction
        // and has no swap-stage API, so the only safe way to switch
        // stages is to throw the renderer away and lazy-init a new
        // one below. Without this, opening a new USD leaves the
        // Hydra panel rendering the old one indefinitely.
        if let Some(existing) = hydra_slot.as_ref() {
            if existing.stage_path != path {
                log::info!(
                    "Hydra: stage changed from {} to {}, dropping renderer",
                    existing.stage_path.display(),
                    path.display(),
                );
                *hydra_slot = None;
                *hydra_egui_tex = None;
                *hydra_startup = None;
            }
        }

        // Lazy-init the renderer once we have both a stage path and a
        // viewport. Same shape as the previous side-panel
        // implementation — the only thing that changed is where the
        // rect comes from.
        if hydra_slot.is_none() {
            let explicit_delegate = hydra_delegate.as_deref().filter(|id| !id.is_empty());
            let inferred_delegate = explicit_delegate
                .is_none()
                .then(|| {
                    crate::hydra_view::HydraView::list_delegates()
                        .into_iter()
                        .next()
                })
                .flatten();
            let startup_delegate = explicit_delegate.or(inferred_delegate.as_deref());
            if !Self::ensure_hydra_startup_ready(ui, rect, path, startup_delegate, hydra_startup) {
                let _ = request_swap_renderer;
                let _ = frame;
                return None;
            }
            log::info!("Hydra: opening stage {}", path.display());
            match crate::hydra_view::HydraView::new_with_delegate(path, hydra_delegate.as_deref()) {
                Ok(mut v) => {
                    log::info!("Hydra: stage opened OK, size {}x{}", w, h);
                    // Match the wgpu side's warm-orange selection tint.
                    v.set_selection_color([1.0, 0.55, 0.15, 1.0]);
                    *hydra_startup = None;
                    *hydra_slot = Some(v);
                }
                Err(e) => {
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("Hydra init failed: {e:#}"),
                        egui::FontId::proportional(13.0),
                        egui::Color32::from_rgb(255, 140, 120),
                    );
                    let _ = request_swap_renderer;
                    let _ = frame;
                    return None;
                }
            }
        }
        let hydra = hydra_slot.as_mut().unwrap();

        // Same auto camera-clipping as the wgpu side does inside
        // `Viewport::show`. Hydra mode doesn't go through `show`, so
        // without re-deriving near/far here the projection would
        // stay at whatever the last wgpu frame set (or the original
        // 0.01 / 1000 defaults if Hydra was the first mode used).
        // Re-using the wgpu helper keeps the formula in lock-step.
        vp.refresh_clip_planes();
        let viewport_selection = vp.selection_from_response(&response, rect, true);
        let view = vp.camera.view().to_cols_array_2d();
        let view_row = crate::hydra_view::glam_to_hydra(&view);
        let proj_row = crate::hydra_view::perspective_for_hydra(
            vp.camera.fov_y_deg.to_radians(),
            aspect,
            vp.camera.z_near,
            vp.camera.z_far,
        );
        let env = vp
            .env
            .source_path
            .as_ref()
            .map(|p| crate::hydra_view::DomeEnv {
                path: p.clone(),
                intensity: vp.env_intensity,
                exposure_stops: vp.exposure_stops,
                rotation_y_radians: vp.env_rotation_y,
            });

        // Adopt the user's delegate pick, if any.
        let current_delegate = hydra.current_delegate();
        let desired_delegate = hydra_delegate
            .clone()
            .unwrap_or_else(|| current_delegate.clone());
        if !desired_delegate.is_empty() && desired_delegate != current_delegate {
            if !Self::ensure_hydra_startup_ready(
                ui,
                rect,
                path,
                Some(&desired_delegate),
                hydra_startup,
            ) {
                let _ = request_swap_renderer;
                let _ = frame;
                return None;
            }
            // Breadcrumb the switch — on Windows the delegate change
            // can trigger a native HgiGL crash on the next render, and
            // the console-less release build otherwise vanishes with no
            // trace. The matching "switch OK" line tells us whether the
            // crash is in SetRendererPlugin itself or the render after.
            log::info!(
                "Hydra: switching delegate '{current_delegate}' -> '{desired_delegate}'"
            );
            match hydra.set_delegate(&desired_delegate) {
                Ok(()) => log::info!("Hydra: delegate switch to '{desired_delegate}' returned OK"),
                Err(e) => log::warn!("Hydra: delegate switch failed: {e:#}"),
            }
        }
        if hydra_delegate.is_none() && !current_delegate.is_empty() {
            *hydra_delegate = Some(current_delegate.clone());
        }

        hydra.resize(w, h);
        // Mirror the wgpu side's analytic-light list into the Hydra
        // session layer as `UsdLuxDistantLight` / `UsdLuxSphereLight`
        // prims. Dirty-tracked inside `set_user_lights`, so when the
        // panel is idle this is a no-op. Combined with the dome
        // below, both renderers see the same lighting from the same
        // source of truth.
        if let Err(e) = hydra.set_user_lights(&vp.lights) {
            log::warn!("Hydra: set_user_lights failed: {e:#}");
        }
        // Concurrent material bindings — diff the current binding
        // list against what we last pushed, then call hydra-rs once
        // per change (apply_material_binding for new/scope-changed,
        // set_binding_input_* for slider edits, remove_material_
        // binding for vanished ids).
        let mut current_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for b in material_bindings {
            // Skip shader nodes the user has staged but not yet
            // assigned — those exist in the graph only and shouldn't
            // touch the Hydra session layer.
            if !b.assigned {
                continue;
            }
            current_ids.insert(b.id);
            let new_snap = MaterialBindingSnapshot::of(b);
            let prev = last_pushed_bindings.get(&b.id).cloned();
            let scope_or_source_changed = match prev.as_ref() {
                Some(p) => {
                    p.source != new_snap.source
                        || p.prim_path != new_snap.prim_path
                        || p.target_prims != new_snap.target_prims
                }
                None => true,
            };
            if scope_or_source_changed {
                if let Err(e) =
                    hydra.apply_material_binding(b.id, &b.source, &b.prim_path, &b.target_prims)
                {
                    log::warn!("Hydra: apply_material_binding failed: {e:#}");
                }
            }
            let inputs_changed = match prev.as_ref() {
                Some(p) => p.inputs != new_snap.inputs,
                None => true,
            };
            if inputs_changed || scope_or_source_changed {
                let names = b.kind.input_names();
                if let Some(n) = names.diffuse_color {
                    hydra.set_binding_input_color3(b.id, n, b.inputs.diffuse_color);
                }
                if let Some(n) = names.metallic {
                    hydra.set_binding_input_f(b.id, n, b.inputs.metallic);
                }
                if let Some(n) = names.roughness {
                    hydra.set_binding_input_f(b.id, n, b.inputs.roughness);
                }
                if let Some(n) = names.opacity {
                    hydra.set_binding_input_f(b.id, n, b.inputs.opacity);
                }
                if let Some(n) = names.clearcoat {
                    hydra.set_binding_input_f(b.id, n, b.inputs.clearcoat);
                }
                if let Some(n) = names.clearcoat_roughness {
                    hydra.set_binding_input_f(b.id, n, b.inputs.clearcoat_roughness);
                }
                if let Some(n) = names.emission_color {
                    hydra.set_binding_input_color3(b.id, n, b.inputs.emission_color);
                }
                if let Some(n) = names.emission_intensity {
                    hydra.set_binding_input_f(b.id, n, b.inputs.emission_intensity);
                }
            }
            last_pushed_bindings.insert(b.id, new_snap);
        }
        // Drop any bindings that disappeared from the App-side Vec.
        let dropped: Vec<u64> = last_pushed_bindings
            .keys()
            .copied()
            .filter(|id| !current_ids.contains(id))
            .collect();
        for id in dropped {
            hydra.remove_material_binding(id);
            last_pushed_bindings.remove(&id);
        }
        hydra.set_purposes(*show_render, *show_proxy, *show_guides);
        if let Err(e) = hydra.set_environment(env.as_ref()) {
            log::warn!("Hydra: set_environment failed: {e:#}");
        }

        // One-shot sync on Hydra mode entry. Triggered by the
        // `update()` dispatch when the badge click flips wgpu → Hydra.
        // Clearing the flag unconditionally (even on sync failure)
        // keeps us out of an infinite "try again next frame" loop;
        // the failure status is surfaced via `hydra_paint_sync_status`
        // for the overlay.
        //
        // Skip the paint sync entirely when a library material is
        // bound — the paint sync's `MaterialBindingAPI.Bind` would
        // overwrite the external-material binding (UsdShade strength
        // is "last-authored wins"), which the user sees as "the
        // material I picked got unassigned every time I bounce
        // through wgpu". With a library material active, leave the
        // session layer alone.
        if *hydra_paint_sync_pending {
            *hydra_paint_sync_pending = false;
            if material_bindings.is_empty() {
                let status = Self::sync_painted_material(
                    frame,
                    vp,
                    hydra,
                    hydra_paint_cache_dir,
                    hydra_paint_sync_seq,
                );
                log::info!("Hydra paint mode-entry sync: {status}");
                *hydra_paint_sync_status = Some(status);
            } else {
                log::info!("Hydra: skipping paint mode-entry sync — library material bound",);
            }
        }

        // Breadcrumb the first render on a freshly-switched delegate.
        // Throttled to one begin/end pair per delegate change (not per
        // frame) so the log stays readable. If the log shows "render
        // begin" on a delegate but never the matching "render end", the
        // crash is inside that delegate's first Render/AOV-readback —
        // the exact Windows halt we're chasing.
        thread_local! {
            static LAST_BREADCRUMBED: std::cell::RefCell<String> =
                const { std::cell::RefCell::new(String::new()) };
        }
        let active_delegate = hydra.current_delegate();
        let first_on_delegate = LAST_BREADCRUMBED.with(|c| {
            let mut last = c.borrow_mut();
            if *last != active_delegate {
                *last = active_delegate.clone();
                true
            } else {
                false
            }
        });
        if first_on_delegate {
            log::info!(
                "Hydra: first render begin on delegate '{active_delegate}' ({w}x{h})"
            );
        }
        match hydra.render(&view_row, &proj_row) {
            Ok(pixels) => {
                if first_on_delegate {
                    log::info!(
                        "Hydra: first render end on delegate '{active_delegate}' — {} bytes",
                        pixels.len()
                    );
                }
                let img =
                    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &pixels);
                let handle =
                    ui.ctx()
                        .load_texture("hydra_frame", img, egui::TextureOptions::default());
                ui.painter().image(
                    handle.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                *hydra_egui_tex = Some(handle);
            }
            Err(e) => {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("Hydra render failed: {e:#}"),
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_rgb(255, 140, 120),
                );
            }
        }

        // Renderer / delegate picker now lives in a single overlay
        // drawn by `App::draw_renderer_picker` (after this function
        // returns), shared between wgpu + Hydra modes. The picker's
        // selection drives `renderer_mode` and `hydra_delegate`
        // directly, so this function no longer needs the swap-request
        // bool that the old clickable badge consumed.
        let _ = request_swap_renderer;
        let combo_size = egui::vec2(150.0, 28.0);

        // "Sync paint" button — sits directly under the delegate
        // combo, same right-aligned x. Click reads the current paint
        // target back from the wgpu side, writes per-tile PNGs into
        // a cache dir, and authors a `UsdPreviewSurface` material
        // in the Hydra session layer bound to all meshes. Hydra
        // picks up the binding next frame.
        let sync_btn_rect = egui::Rect::from_min_size(
            egui::pos2(
                rect.right() - combo_size.x - 12.0,
                rect.top() + 12.0 + combo_size.y + 6.0,
            ),
            egui::vec2(combo_size.x, 26.0),
        );
        let sync_resp = ui.interact(
            sync_btn_rect,
            ui.id().with("hydra_sync_paint"),
            egui::Sense::click(),
        );
        if sync_resp.clicked() {
            // Reuse the live `hydra` borrow from above — re-grabbing
            // through `hydra_slot.as_mut()` here would shadow that
            // earlier borrow and trip the borrow checker.
            let status = Self::sync_painted_material(
                frame,
                vp,
                hydra,
                hydra_paint_cache_dir,
                hydra_paint_sync_seq,
            );
            log::info!("Hydra paint sync: {status}");
            *hydra_paint_sync_status = Some(status);
        }
        let (sync_fill_alpha, sync_stroke_w) = if sync_resp.hovered() {
            (240u8, 2.0)
        } else {
            (220u8, 1.5)
        };
        ui.painter().rect_filled(
            sync_btn_rect,
            6.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, sync_fill_alpha),
        );
        ui.painter().rect_stroke(
            sync_btn_rect,
            6.0,
            egui::Stroke::new(sync_stroke_w, egui::Color32::from_rgb(180, 200, 255)),
            egui::StrokeKind::Outside,
        );
        ui.painter().text(
            sync_btn_rect.center(),
            egui::Align2::CENTER_CENTER,
            "↻ Sync paint",
            egui::FontId::proportional(13.0),
            egui::Color32::from_rgb(180, 200, 255),
        );
        if sync_resp.hovered() {
            sync_resp.on_hover_text(
                "Read paint targets → write per-tile PNGs → author UsdPreviewSurface in Hydra",
            );
        }

        // Status line under the sync button — last sync's outcome.
        let status_y = if let Some(status) = hydra_paint_sync_status.as_deref() {
            let status_rect = egui::Rect::from_min_size(
                egui::pos2(
                    rect.right() - combo_size.x - 12.0,
                    sync_btn_rect.bottom() + 4.0,
                ),
                egui::vec2(combo_size.x, 18.0),
            );
            ui.painter().text(
                status_rect.right_center(),
                egui::Align2::RIGHT_CENTER,
                status,
                egui::FontId::proportional(11.0),
                egui::Color32::from_gray(180),
            );
            status_rect.bottom()
        } else {
            sync_btn_rect.bottom()
        };

        // Purpose toggles — render / proxy / guides. Three chip-style
        // buttons below the sync UI; filled when on, outlined when
        // off. Matches usdview / Solaris's purpose checkbox triplet.
        // Default-purpose prims always draw and aren't user-toggleable.
        let chip_w = (combo_size.x - 8.0) / 3.0; // 3 across, 4px gaps
        let chip_h = 22.0;
        let chips_y = status_y + 6.0;
        let chip_x0 = rect.right() - combo_size.x - 12.0;
        let mut draw_chip = |idx: usize, label: &str, on: &mut bool| {
            let chip_rect = egui::Rect::from_min_size(
                egui::pos2(chip_x0 + (chip_w + 4.0) * idx as f32, chips_y),
                egui::vec2(chip_w, chip_h),
            );
            let resp = ui.interact(
                chip_rect,
                ui.id().with(("hydra_purpose_chip", idx)),
                egui::Sense::click(),
            );
            if resp.clicked() {
                *on = !*on;
            }
            let active = *on;
            let stroke = egui::Color32::from_rgb(200, 200, 230);
            let fill = if active {
                egui::Color32::from_rgba_unmultiplied(70, 90, 130, 240)
            } else {
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200)
            };
            let stroke_w = if resp.hovered() { 1.8 } else { 1.0 };
            ui.painter().rect_filled(chip_rect, 4.0, fill);
            ui.painter().rect_stroke(
                chip_rect,
                4.0,
                egui::Stroke::new(stroke_w, stroke),
                egui::StrokeKind::Outside,
            );
            ui.painter().text(
                chip_rect.center(),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(11.0),
                stroke,
            );
            if resp.hovered() {
                resp.on_hover_text(format!(
                    "Toggle the `{label}` `UsdGeomImageable::purpose` filter"
                ));
            }
        };
        draw_chip(0, "render", show_render);
        draw_chip(1, "proxy", show_proxy);
        draw_chip(2, "guides", show_guides);

        let _ = frame;
        viewport_selection
    }

    /// Exports the current paint target to a cache dir as per-tile
    /// PNGs, then authors a `UsdPreviewSurface` material in the
    /// Hydra session layer pointing at those PNGs (via the `<UDIM>`
    /// token so a single material spans every tile). The Hydra
    /// preview re-renders next frame with the painted look applied.
    ///
    /// Cache dir is per-process under `std::env::temp_dir()`. It's
    /// re-used across sync calls so the material in the session
    /// layer stays valid — re-exporting just overwrites the same
    /// files in place. Returns a short status string for the
    /// overlay (success timestamp or error message).
    fn sync_painted_material(
        frame: &eframe::Frame,
        vp: &Viewport,
        hydra: &mut crate::hydra_view::HydraView,
        cache_root: &mut Option<std::path::PathBuf>,
        sync_seq: &mut u64,
    ) -> String {
        let Some(render_state) = frame.wgpu_render_state() else {
            return "Sync failed: no wgpu render state".to_string();
        };

        // Root cache dir is per-process so concurrent forge-paint
        // instances can't trample each other's PNGs and so /tmp's
        // own GC reclaims everything on process exit.
        let root: std::path::PathBuf = cache_root
            .get_or_insert_with(|| {
                let pid = std::process::id();
                std::env::temp_dir().join(format!("forge-paint-hydra-{pid}"))
            })
            .clone();
        if let Err(e) = std::fs::create_dir_all(&root) {
            return format!("Sync failed: create cache root: {e}");
        }

        // Versioned subdir per sync — Hydra's texture cache keys on
        // the resolved asset path, so re-writing the same PNG path
        // would hand back the old sampled data. Bumping the dir
        // forces a fresh fetch on the delegate side.
        *sync_seq = sync_seq.saturating_add(1);
        let seq = *sync_seq;
        let versioned_dir = root.join(format!("v{seq}"));
        if let Err(e) = std::fs::create_dir_all(&versioned_dir) {
            return format!("Sync failed: create versioned dir: {e}");
        }

        let exports = match crate::export::export_tiles(
            &render_state.device,
            &render_state.queue,
            &vp.paint_target,
            &versioned_dir,
        ) {
            Ok(e) => e,
            Err(e) => return format!("Sync failed: export: {e:#}"),
        };

        // Build one `<UDIM>` asset path per channel. `export_tiles`
        // writes `<channel>.<udim>.png` (e.g. basecolor.1001.png),
        // so the path with `<UDIM>` in place of the integer is
        // exactly what UsdUVTexture's resolver will substitute. The
        // resolver doesn't care if a particular tile's PNG is
        // missing — it just samples default for those UVs.
        let path_for = |channel: &str| -> String {
            if !exports.iter().any(|e| e.channel == channel) {
                return String::new();
            }
            versioned_dir
                .join(format!("{channel}.<UDIM>.png"))
                .to_string_lossy()
                .into_owned()
        };

        let base_color = path_for("basecolor");
        let roughness = path_for("roughness");
        let metallic = path_for("metallic");
        let normal = path_for("normal");

        let bc = std::path::PathBuf::from(&base_color);
        let ro = std::path::PathBuf::from(&roughness);
        let me = std::path::PathBuf::from(&metallic);
        let nm = std::path::PathBuf::from(&normal);
        if let Err(e) = hydra.set_painted_material(&bc, &ro, &me, &nm) {
            return format!("Sync failed: set_painted_material: {e:#}");
        }

        // Trim history: keep the current sync's dir plus one prior
        // (in case the delegate is still sampling from the previous
        // version mid-render — unlikely with our continuous-repaint
        // loop, but cheap insurance). Delete `v<seq-2>` and earlier.
        if seq >= 2 {
            let _ = std::fs::remove_dir_all(root.join(format!("v{}", seq - 2)));
        }

        format!("synced v{seq} ({} tiles)", exports.len() / 4)
    }

    /// One combo box that picks the renderer driving the central
    /// viewport: `wgpu` (paint mode), or any Hydra delegate the plug
    /// registry found (Storm, 3Delight, …). Drives `renderer_mode`
    /// and `hydra_delegate` together so the rest of the App just
    /// reads those two fields.
    ///
    /// Also stamps the Storm-mode warning banner when the active
    /// delegate is `HdStormRendererPlugin` — Storm renders the
    /// stage but doesn't paint, so the banner is a nudge to switch
    /// to wgpu when you're trying to author rather than preview.
    fn draw_renderer_picker(
        ui: &mut egui::Ui,
        rect: egui::Rect,
        renderer_mode: &mut RendererMode,
        hydra_delegate: &mut Option<String>,
    ) {
        const WGPU_ID: &str = "__wgpu";
        let delegates = crate::hydra_view::HydraView::list_delegates();
        if hydra_delegate
            .as_ref()
            .is_some_and(|id| !delegates.iter().any(|delegate| delegate == id))
        {
            *hydra_delegate = delegates.first().cloned();
        }
        if *renderer_mode == RendererMode::Hydra && delegates.is_empty() {
            *renderer_mode = RendererMode::Wgpu;
        }

        // Current selection in the combo:
        //  - Wgpu mode      → WGPU_ID
        //  - Hydra mode     → whichever delegate is active (falls back
        //                     to the first registered one if nothing's
        //                     been picked yet)
        let current_id = match *renderer_mode {
            RendererMode::Wgpu => WGPU_ID.to_string(),
            RendererMode::Hydra => hydra_delegate
                .clone()
                .or_else(|| delegates.first().cloned())
                .unwrap_or_default(),
        };
        let current_label = if current_id == WGPU_ID {
            "wgpu painter"
        } else {
            crate::hydra_view::delegate_label(&current_id)
        };

        // Combo overlay rect — top-right corner of the viewport,
        // mirrors where the old Hydra-only combo used to sit.
        let combo_size = egui::vec2(170.0, 28.0);
        let combo_rect = egui::Rect::from_min_size(
            egui::pos2(rect.right() - combo_size.x - 12.0, rect.top() + 12.0),
            combo_size,
        );
        let mut combo_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(combo_rect)
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
        );

        let mut chosen_id = current_id.clone();
        egui::ComboBox::from_id_salt("renderer_picker")
            .selected_text(format!("▶ {current_label}"))
            .show_ui(&mut combo_ui, |ui| {
                ui.selectable_value(&mut chosen_id, WGPU_ID.to_string(), "wgpu painter");
                for id in &delegates {
                    ui.selectable_value(
                        &mut chosen_id,
                        id.clone(),
                        crate::hydra_view::delegate_label(id),
                    );
                }
            });

        if chosen_id != current_id {
            if chosen_id == WGPU_ID {
                *renderer_mode = RendererMode::Wgpu;
            } else {
                *renderer_mode = RendererMode::Hydra;
                *hydra_delegate = Some(chosen_id.clone());
            }
        }

        // Storm warning banner. Storm-only — when 3Delight is the
        // active delegate the banner stays hidden. Centered at the
        // top of the canvas, just below the combo's y. Reads as a
        // nudge ("Storm is preview, paint in wgpu") rather than an
        // error.
        let storm_active =
            *renderer_mode == RendererMode::Hydra && chosen_id == "HdStormRendererPlugin";
        if storm_active {
            let banner_size = egui::vec2(260.0, 22.0);
            let banner_rect = egui::Rect::from_center_size(
                egui::pos2(rect.center().x, rect.top() + 12.0 + banner_size.y / 2.0),
                banner_size,
            );
            ui.painter().rect_filled(
                banner_rect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(60, 30, 0, 220),
            );
            ui.painter().rect_stroke(
                banner_rect,
                4.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 180, 80)),
                egui::StrokeKind::Outside,
            );
            ui.painter().text(
                banner_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Storm preview — switch to wgpu to paint",
                egui::FontId::proportional(12.0),
                egui::Color32::from_rgb(255, 220, 160),
            );
        }
    }

    /// Draw the green "▶ Hydra ..." badge in `rect`'s top-left as a
    /// clickable button. Mirrors `Viewport::show`'s yellow wgpu badge.
    /// Sets `*request_swap` to true on click so the App can flip
    /// `renderer_mode` back to Wgpu after the closure returns.
    fn draw_renderer_badge_hydra(
        ui: &mut egui::Ui,
        rect: egui::Rect,
        label: &str,
        request_swap: &mut bool,
    ) {
        let badge_size = egui::vec2(170.0, 28.0);
        let badge_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left() + 12.0, rect.top() + 12.0),
            badge_size,
        );
        let resp = ui.interact(
            badge_rect,
            ui.id().with("renderer_badge_hydra"),
            egui::Sense::click(),
        );
        if resp.clicked() {
            *request_swap = true;
        }
        let (fill_alpha, stroke_w) = if resp.hovered() {
            (240u8, 2.0)
        } else {
            (220u8, 1.5)
        };
        ui.painter().rect_filled(
            badge_rect,
            6.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, fill_alpha),
        );
        ui.painter().rect_stroke(
            badge_rect,
            6.0,
            egui::Stroke::new(stroke_w, egui::Color32::from_rgb(120, 220, 140)),
            egui::StrokeKind::Outside,
        );
        ui.painter().text(
            badge_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgb(120, 220, 140),
        );
        if resp.hovered() {
            resp.on_hover_text("Click to switch back to wgpu painter");
        }
    }

    fn do_undo(&mut self, frame: &eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            return;
        };
        let Some(vp) = &mut self.viewport else {
            return;
        };
        if vp.undo(&render_state.device, &render_state.queue) {
            self.status = "Undo".to_string();
        }
    }

    fn do_redo(&mut self, frame: &eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            return;
        };
        let Some(vp) = &mut self.viewport else {
            return;
        };
        if vp.redo(&render_state.device, &render_state.queue) {
            self.status = "Redo".to_string();
        }
    }

    fn reload_sidecars(&mut self, frame: &eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.status = "No GPU render state available.".to_string();
            return;
        };
        let Some(vp) = &self.viewport else {
            self.status = "Viewport not initialized yet.".to_string();
            return;
        };
        let Some(usd_path) = &self.current_usd_path else {
            self.status = "No USD loaded.".to_string();
            return;
        };
        let dir = crate::persist::default_work_dir(usd_path);
        let n = crate::persist::load_sidecars(
            &render_state.queue,
            vp.active_layer(),
            vp.tiles(),
            vp.tile_resolution(),
            &dir,
        );
        if n > 0 {
            vp.recomposite(&render_state.device, &render_state.queue);
        }
        self.status = if n > 0 {
            format!("Reloaded {n} sidecar(s) from {}", dir.display())
        } else {
            format!("No sidecars at {}", dir.display())
        };
        log::info!("{}", self.status);
    }

    /// Walk `assets/stencils/` and `assets/displacement/` and import
    /// every PNG / EXR we find into the asset browser so the user can
    /// right-click → Project with stencil without first clicking
    /// "+ Import" for bundled content.
    fn scan_bundled_assets(&mut self, rs: &eframe::egui_wgpu::RenderState) {
        let dirs = [
            resolve_bundled_asset_dir("assets/stencils"),
            resolve_bundled_asset_dir("assets/displacement"),
        ];
        let mut count = 0usize;
        for dir in &dirs {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<std::path::PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            for path in paths {
                let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                    continue;
                };
                let lower = ext.to_lowercase();
                if !matches!(lower.as_str(), "png" | "jpg" | "jpeg" | "exr") {
                    continue;
                }
                let mut renderer = rs.renderer.write();
                match self
                    .browser
                    .import_texture(&path, &rs.device, &rs.queue, &mut renderer)
                {
                    Ok(()) => count += 1,
                    Err(e) => log::warn!("bundled asset {}: {e:#}", path.display()),
                }
            }
        }

        // Also scan for USD meshes — populate the Meshes tab so the
        // default mesh is available without hunting through File > Open.
        for dir in &[
            resolve_bundled_asset_dir("assets/default_mesh"),
            resolve_bundled_asset_dir("assets/meshes"),
        ] {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<std::path::PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            for path in paths {
                let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                    continue;
                };
                let lower = ext.to_lowercase();
                if !matches!(lower.as_str(), "usd" | "usda" | "usdc" | "usdz") {
                    continue;
                }
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "mesh".into());
                self.browser.meshes.push(assets::MeshAsset {
                    name,
                    path: path.clone(),
                });
            }
        }

        // Materials library — scan once at startup. Same exe-relative
        // fallback as the HDRI discovery.
        let materials_root =
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        self.browser.materials = crate::assets::discover_materials(&materials_root);
        if !self.browser.materials.is_empty() {
            log::info!(
                "Materials library: {} entries",
                self.browser.materials.len(),
            );
        }

        if count > 0 || !self.browser.meshes.is_empty() {
            self.status = format!(
                "Imported {count} texture(s), {} mesh(es)",
                self.browser.meshes.len()
            );
            log::info!("{}", self.status);
        }
    }

    fn asset_browser_panel(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let mut want_import = false;
        let mut want_clear_stencil = false;
        // Snapshot the active-stencil label so we can render it inside
        // this closure without holding a mutable borrow of the viewport.
        let stencil_label = self
            .viewport
            .as_ref()
            .and_then(|vp| vp.active_stencil)
            .and_then(|i| self.browser.textures.get(i))
            .map(|a| a.name.clone());
        ui.horizontal(|ui| {
            for &tab in assets::Tab::ALL {
                ui.selectable_value(&mut self.browser.active_tab, tab, tab.label());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if matches!(self.browser.active_tab, assets::Tab::Textures)
                    && ui.button("+ Import").clicked()
                {
                    want_import = true;
                }
                if let Some(name) = &stencil_label {
                    if ui.button("Clear stencil").clicked() {
                        want_clear_stencil = true;
                    }
                    ui.label(
                        egui::RichText::new(format!("Stencil: {}", name))
                            .strong()
                            .color(egui::Color32::from_rgb(255, 220, 100)),
                    );
                }
            });
        });
        ui.separator();

        match self.browser.active_tab {
            assets::Tab::Textures => {
                self.texture_strip(ui, frame);
            }
            assets::Tab::Meshes => {
                self.mesh_strip(ui, frame);
            }
            assets::Tab::Materials => {
                self.material_strip(ui);
            }
            _ => {
                ui.weak("(this tab is not implemented yet)");
            }
        }

        if want_import {
            self.import_texture_dialog(frame);
        }
        if want_clear_stencil {
            if let Some(vp) = &mut self.viewport {
                vp.active_stencil = None;
            }
            self.status = "Stencil cleared".to_string();
        }
    }

    fn mesh_strip(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        if self.browser.meshes.is_empty() {
            ui.weak(
                "No meshes found. Drop .usd / .usda / .usdc / .usdz files \
                 into assets/default_mesh/ or assets/meshes/ and restart.",
            );
            return;
        }
        let mut load_requested: Option<std::path::PathBuf> = None;
        egui::ScrollArea::horizontal()
            .id_salt("asset_mesh_strip")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for mesh in &self.browser.meshes {
                        ui.vertical(|ui| {
                            // Placeholder: large Phosphor cube glyph in
                            // a 80×80 button. Real rendered-mesh
                            // thumbnails are a follow-up — would need
                            // a one-shot offscreen PBR render per mesh
                            // at import time.
                            let glyph =
                                egui::RichText::new(egui_phosphor::regular::CUBE).size(48.0);
                            let btn = egui::Button::new(glyph).min_size(egui::vec2(80.0, 80.0));
                            if ui
                                .add(btn)
                                .on_hover_text(format!(
                                    "Load '{}' ({})",
                                    mesh.name,
                                    mesh.path.display(),
                                ))
                                .clicked()
                            {
                                load_requested = Some(mesh.path.clone());
                            }
                            ui.label(
                                egui::RichText::new(&mesh.name)
                                    .small()
                                    .color(ui.style().visuals.weak_text_color()),
                            );
                        });
                    }
                });
            });
        if let Some(path) = load_requested {
            self.load_usd(frame, path);
        }
    }

    fn material_strip(&mut self, ui: &mut egui::Ui) {
        // Filter chip row + clear button at the top.
        ui.horizontal(|ui| {
            ui.label("Filter:");
            for (i, &kind) in crate::assets::MaterialKind::ALL.iter().enumerate() {
                let mut on = self.browser.material_kind_filter[i];
                if ui.toggle_value(&mut on, kind.label()).changed() {
                    self.browser.material_kind_filter[i] = on;
                }
            }
            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    let bound_n = self.material_bindings.len();
                    if ui
                        .add_enabled(
                            bound_n > 0,
                            egui::Button::new(format!("Clear bindings ({bound_n})")),
                        )
                        .on_hover_text(
                            "Remove every library-material binding from the stage. The stage's authored materials take over again.",
                        )
                        .clicked()
                    {
                        self.material_bindings.clear();
                        self.active_binding_id = None;
                    }
                    ui.weak("Click a chip to add its shader node to the editor.");
                },
            );
        });
        ui.separator();

        if self.browser.materials.is_empty() {
            ui.weak(
                "No materials found. Drop `*.usd` / `*.usda` / `*.usdc` files \
                 with a `def Material` prim into `assets/materials/` (or set \
                 `FORGE_PAINT_MATERIAL_DIR` to your library) and restart.",
            );
            return;
        }

        let mut clicked: Option<usize> = None;
        egui::ScrollArea::horizontal()
            .id_salt("asset_material_strip")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, mat) in self.browser.materials.iter().enumerate() {
                        let kind_idx = match mat.kind {
                            crate::assets::MaterialKind::UsdPreviewSurface => 0,
                            crate::assets::MaterialKind::MaterialX => 1,
                            crate::assets::MaterialKind::DlPrincipled => 2,
                            crate::assets::MaterialKind::Other => 3,
                        };
                        if !self.browser.material_kind_filter[kind_idx] {
                            continue;
                        }
                        ui.vertical(|ui| {
                            let (rect, resp) = ui
                                .allocate_exact_size(egui::vec2(84.0, 84.0), egui::Sense::click());
                            let visuals = ui.style().interact(&resp);
                            ui.painter().rect(
                                rect,
                                6.0,
                                visuals.bg_fill,
                                visuals.bg_stroke,
                                egui::StrokeKind::Inside,
                            );
                            crate::assets::paint_material_preview_ball(
                                ui,
                                rect.shrink(5.0),
                                mat.preview_inputs,
                            );
                            let resp = resp.on_hover_text(format!(
                                "{}\n{}\n{}",
                                mat.name,
                                mat.kind.label(),
                                mat.source.display(),
                            ));
                            if resp.clicked() {
                                clicked = Some(i);
                            }
                            // Tag chip under the card — kind label in
                            // small text so the type is visible at a
                            // glance without hovering.
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(mat.kind.label())
                                        .small()
                                        .color(egui::Color32::from_gray(160)),
                                )
                                .truncate(),
                            );
                            // Name under the chip, slightly larger,
                            // truncated to keep the card narrow.
                            ui.add(
                                egui::Label::new(egui::RichText::new(&mat.name).size(11.0))
                                    .truncate(),
                            );
                        });
                    }
                });
            });

        if let Some(i) = clicked {
            // Chip click → spawn a fresh Shader node in the Material
            // Editor for this library material, unassigned. The
            // user wires up the assignment via the node's right-
            // click menu ("Assign to selection" / "Assign to stage").
            if let Some(mat) = self.browser.materials.get(i) {
                let new_id = self.next_binding_id;
                self.next_binding_id += 1;
                self.material_bindings.push(MaterialBindingInstance {
                    id: new_id,
                    source: mat.source.clone(),
                    prim_path: mat.prim_path.clone(),
                    kind: mat.kind,
                    inputs: crate::assets::read_material_inputs(&mat.source),
                    target_prims: Vec::new(),
                    assigned: false,
                });
                self.material_graph.spawn_shader_node(new_id);
                self.active_binding_id = Some(new_id);
            }
        }
    }

    fn texture_strip(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        if self.browser.textures.is_empty() {
            ui.weak("Nothing imported. Click + Import to add a texture.");
            return;
        }
        let mut action: Option<(usize, AssetAction)> = None;
        egui::ScrollArea::horizontal()
            .id_salt("asset_texture_strip")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, asset) in self.browser.textures.iter().enumerate() {
                        ui.vertical(|ui| {
                            let img = egui::Image::new((asset.thumb_id, egui::vec2(80.0, 80.0)))
                                .fit_to_exact_size(egui::vec2(80.0, 80.0))
                                .sense(egui::Sense::click());
                            let response = ui.add(img);
                            response.context_menu(|ui| {
                                if ui.button("Project with stencil").clicked() {
                                    action = Some((i, AssetAction::SetStencil));
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui.button("New paint layer from texture").clicked() {
                                    action = Some((i, AssetAction::NewLayer));
                                    ui.close_menu();
                                }
                                if ui.button("Apply as base color to active layer").clicked() {
                                    action = Some((i, AssetAction::ApplyBaseColor));
                                    ui.close_menu();
                                }
                                if ui.button("Apply as mask to active layer").clicked() {
                                    action = Some((i, AssetAction::ApplyMask));
                                    ui.close_menu();
                                }
                            });
                            ui.label(
                                egui::RichText::new(&asset.name)
                                    .small()
                                    .color(ui.style().visuals.weak_text_color()),
                            );
                        });
                    }
                });
            });
        if let Some((idx, act)) = action {
            self.apply_asset_action(idx, act, frame);
        }
    }

    fn import_texture_dialog(&mut self, frame: &eframe::Frame) {
        // PNG + JPEG + EXR are enabled via the image crate's feature
        // flags in Cargo.toml. EXR loads as HDR floats; we tonemap /
        // clamp to LDR on upload for now.
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "exr"])
            .set_title("Import texture / stencil")
            .pick_file()
        else {
            return;
        };
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = "No GPU available.".to_string();
            return;
        };
        let mut renderer = rs.renderer.write();
        match self
            .browser
            .import_texture(&path, &rs.device, &rs.queue, &mut renderer)
        {
            Ok(_) => {
                self.status = format!("Imported {}", path.display());
                log::info!("{}", self.status);
            }
            Err(e) => {
                self.status = format!("Import failed: {e:#}");
                log::warn!("{}", self.status);
            }
        }
    }

    fn apply_asset_action(&mut self, idx: usize, action: AssetAction, frame: &eframe::Frame) {
        let Some(vp) = &mut self.viewport else {
            return;
        };
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = "No GPU available.".to_string();
            return;
        };
        let Some(asset) = self.browser.textures.get(idx) else {
            return;
        };

        match action {
            AssetAction::NewLayer => {
                vp.add_layer(&rs.device, &rs.queue);
                let last = vp.layer_stack.layers.len() - 1;
                vp.layer_stack.active = last;
                let tile_count = vp.paint_target().tiles.len() as u32;
                let res = vp.tile_resolution();
                let layer = &vp.layer_stack.layers[last];
                if let Err(e) =
                    assets::apply_as_base_color(&rs.queue, asset, layer, tile_count, res)
                {
                    self.status = format!("Apply failed: {e}");
                    return;
                }
                vp.recomposite(&rs.device, &rs.queue);
                self.status = format!("Created layer from '{}'", asset.name);
            }
            AssetAction::ApplyBaseColor => {
                let tile_count = vp.paint_target().tiles.len() as u32;
                let res = vp.tile_resolution();
                let active = vp.layer_stack.active;
                let layer = &vp.layer_stack.layers[active];
                if let Err(e) =
                    assets::apply_as_base_color(&rs.queue, asset, layer, tile_count, res)
                {
                    self.status = format!("Apply failed: {e}");
                    return;
                }
                vp.recomposite(&rs.device, &rs.queue);
                self.status = format!("Applied '{}' as base color", asset.name);
            }
            AssetAction::ApplyMask => {
                let active = vp.layer_stack.active;
                if vp.layer_stack.layers[active].mask.is_none() {
                    vp.layer_stack.add_mask_to(active, &rs.device, &rs.queue);
                }
                let tile_count = vp.paint_target().tiles.len() as u32;
                let res = vp.tile_resolution();
                let layer = &vp.layer_stack.layers[active];
                if let Err(e) = assets::apply_as_mask(&rs.queue, asset, layer, tile_count, res) {
                    self.status = format!("Apply failed: {e}");
                    return;
                }
                vp.recomposite(&rs.device, &rs.queue);
                self.status = format!("Applied '{}' as mask", asset.name);
            }
            AssetAction::SetStencil => {
                // Delegate to the shared helper so the context menu,
                // the tool-strip picker, and the file-dialog import
                // path all behave identically. Bail out early since
                // `rs` borrowed self and `activate_stencil` takes
                // `&mut self` again.
                let _ = rs;
                self.activate_stencil(idx, frame);
                return;
            }
        }
        log::info!("{}", self.status);
    }

    /// Central tool-switch. Clicking the Stencil button opens the
    /// picker modal — letting the user reuse already-imported textures
    /// instead of forcing a file dialog every time. Switching to any
    /// non-Stencil tool cancels the active stencil so the user is back
    /// in normal paint.
    fn switch_tool(&mut self, tool: Tool, _frame: &eframe::Frame) {
        if tool == Tool::Stencil {
            self.show_stencil_picker = true;
            return;
        }
        if let Some(vp) = &mut self.viewport {
            if vp.tool == Tool::Stencil {
                vp.active_stencil = None;
            }
            vp.tool = tool;
        }
        self.show_stencil_picker = false;
    }

    /// Activate the texture at `idx` in the asset browser as the
    /// projection stencil. Auto-bakes mesh maps if the position map
    /// isn't ready yet.
    /// Upload the texture at `idx` in the asset browser into the active
    /// layer's channel that corresponds to `slot`, then recomposite.
    fn apply_slot(&mut self, slot: MaterialSlot, idx: usize, frame: &eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = "No GPU available.".to_string();
            return;
        };
        let Some(asset) = self.browser.textures.get(idx) else {
            return;
        };
        let Some(vp) = &mut self.viewport else {
            return;
        };
        let tile_count = vp.paint_target().tiles.len() as u32;
        let res = vp.tile_resolution();
        let active_idx = vp.layer_stack.active;
        let layer = &vp.layer_stack.layers[active_idx];
        let result = match slot {
            MaterialSlot::BaseColor => {
                assets::apply_as_base_color(&rs.queue, asset, layer, tile_count, res)
            }
            MaterialSlot::Roughness => {
                assets::apply_as_roughness(&rs.queue, asset, layer, tile_count, res)
            }
            MaterialSlot::Metallic => {
                assets::apply_as_metallic(&rs.queue, asset, layer, tile_count, res)
            }
            MaterialSlot::Normal => {
                assets::apply_as_normal(&rs.queue, asset, layer, tile_count, res)
            }
        };
        match result {
            Ok(()) => {
                vp.recomposite(&rs.device, &rs.queue);
                self.status = format!(
                    "Assigned '{}' to {} on '{}'",
                    asset.name,
                    slot.label(),
                    vp.layer_stack.layers[active_idx].name,
                );
                log::info!("{}", self.status);
            }
            Err(e) => {
                self.status = format!("Slot assign failed: {e:#}");
                log::warn!("{}", self.status);
            }
        }
    }

    fn activate_stencil(&mut self, idx: usize, frame: &eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = "No GPU available.".to_string();
            return;
        };
        let name = match self.browser.textures.get(idx) {
            Some(a) => a.name.clone(),
            None => return,
        };
        if let Some(vp) = &mut self.viewport {
            if !vp.mesh_maps.baked {
                vp.bake_mesh_maps(&rs.device, &rs.queue);
                log::info!("Auto-baked mesh maps for stencil projection");
            }
            vp.tool = Tool::Stencil;
            vp.active_stencil = Some(idx);
            vp.stencil_transform = crate::viewport::StencilTransform::default();
        }
        self.status = format!("Stencil: '{}' · M/R/T + LMB to move/rotate/scale", name);
    }

    fn open_stencil_dialog(&mut self, frame: &eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "exr"])
            .set_title("Pick stencil")
            .pick_file()
        else {
            // Dialog cancelled — leave the current tool untouched.
            return;
        };
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = "No GPU available.".to_string();
            return;
        };
        // Auto-bake mesh maps on demand — projection needs the position
        // map, and requiring a manual Bake click is unfriendly.
        let baked = self
            .viewport
            .as_ref()
            .map(|vp| vp.mesh_maps.baked)
            .unwrap_or(true);
        if !baked {
            if let Some(vp) = &mut self.viewport {
                vp.bake_mesh_maps(&rs.device, &rs.queue);
                log::info!("Auto-baked mesh maps for stencil projection");
            }
        }
        let mut renderer = rs.renderer.write();
        let result = self
            .browser
            .import_texture(&path, &rs.device, &rs.queue, &mut renderer);
        drop(renderer);
        match result {
            Ok(()) => {
                let idx = self.browser.textures.len() - 1;
                self.activate_stencil(idx, frame);
            }
            Err(e) => {
                self.status = format!("Failed to load stencil: {e:#}");
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AssetAction {
    NewLayer,
    ApplyBaseColor,
    ApplyMask,
    SetStencil,
}
