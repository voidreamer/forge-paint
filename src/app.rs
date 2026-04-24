use eframe::egui;
use std::path::PathBuf;

use crate::{
    assets::{self, AssetBrowser},
    mesh,
    viewport::{Tool, Viewport},
};

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
}

/// Which channel a material-slot assignment should target on the
/// active layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl App {
    pub fn new(initial_usd: Option<PathBuf>) -> Self {
        // CLI path wins. Otherwise fall back to a user-provided default
        // mesh at assets/default_mesh/default.usda (relative to CWD).
        // Override the location via FORGE_PAINT_DEFAULT_MESH if you run
        // the binary from elsewhere.
        let pending_open = initial_usd.or_else(|| {
            if let Some(override_path) = std::env::var_os("FORGE_PAINT_DEFAULT_MESH") {
                let p = PathBuf::from(override_path);
                if p.exists() {
                    return Some(p);
                }
            }
            let default_path = PathBuf::from("assets/default_mesh/default.usda");
            if default_path.exists() {
                Some(default_path)
            } else {
                None
            }
        });
        Self {
            pending_open,
            // Sane non-zero defaults for the UV view. bool/Vec2/Vec
            // defaults (false, (0,0), empty) are already what we want.
            uv_zoom: 400.0,
            uv_show_wireframe: true,
            uv_channel: crate::render::ViewMode::BaseColor,
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
                log::info!("Viewport initialized with unit cube ({} verts)", cpu.positions.len());
            }
        }

        // If a path was passed on the CLI, open it now that the viewport exists.
        if self.viewport.is_some() {
            if let Some(path) = self.pending_open.take() {
                self.load_usd(frame, path);
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
                        .add_enabled(
                            can_redo,
                            egui::Button::new("Redo   ⇧⌘Z / Ctrl+Shift+Z"),
                        )
                        .clicked()
                    {
                        self.do_redo(frame);
                        ui.close_menu();
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("Open USD…").clicked() {
                        self.open_usd_dialog(frame);
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
                    if ui
                        .checkbox(&mut self.show_uv_view, "UV view")
                        .clicked()
                    {
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

        // Open URI modal — string entry for forge:// or any path usdcat accepts.
        if self.show_uri_dialog {
            let mut open = true;
            let mut load_requested: Option<String> = None;
            egui::Window::new("Open URI")
                .open(&mut open)
                .resizable(false)
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.label("USD URI or path (forge://…, file path, or anything usdcat can resolve):");
                    ui.add(egui::TextEdit::singleline(&mut self.uri_buffer).desired_width(f32::INFINITY));
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
                self.load_usd(frame, PathBuf::from(uri));
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
                                        for (i, asset) in
                                            self.browser.textures.iter().enumerate()
                                        {
                                            ui.vertical(|ui| {
                                                let img = egui::Image::new((
                                                    asset.thumb_id,
                                                    thumb,
                                                ))
                                                .fit_to_exact_size(thumb)
                                                .sense(egui::Sense::click());
                                                if ui.add(img).on_hover_text(&asset.name).clicked()
                                                {
                                                    picked = Some(i);
                                                }
                                                ui.label(
                                                    egui::RichText::new(&asset.name).small(),
                                                );
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
                                        for (i, asset) in
                                            self.browser.textures.iter().enumerate()
                                        {
                                            ui.vertical(|ui| {
                                                let img = egui::Image::new((
                                                    asset.thumb_id,
                                                    thumb,
                                                ))
                                                .fit_to_exact_size(thumb)
                                                .sense(egui::Sense::click());
                                                if ui
                                                    .add(img)
                                                    .on_hover_text(&asset.name)
                                                    .clicked()
                                                {
                                                    picked = Some(i);
                                                }
                                                ui.label(
                                                    egui::RichText::new(&asset.name).small(),
                                                );
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
                    ui.weak("LMB paint · Ctrl+LMB orbit · Shift+LMB / MMB pan · wheel zoom · S/D/F+LMB brush size/hardness/opacity · M/R/T+LMB stencil move/rotate/scale");
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

        let mut slot_clicked: Option<MaterialSlot> = None;
        egui::SidePanel::right("props")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("right_panel_scroll")
                    .show(ui, |ui| {
                        if let Some(vp) = &mut self.viewport {
                            egui::CollapsingHeader::new("Color")
                                .default_open(true)
                                .show(ui, |ui| color_section(ui, vp));
                            egui::CollapsingHeader::new("Layers")
                                .default_open(true)
                                .show(ui, |ui| layer_panel(ui, vp, frame));
                            egui::CollapsingHeader::new("Environment")
                                .default_open(true)
                                .show(ui, |ui| env_panel(ui, vp, frame));
                            egui::CollapsingHeader::new("Mesh maps")
                                .default_open(false)
                                .show(ui, |ui| mesh_maps_panel(ui, vp, frame));
                            egui::CollapsingHeader::new("Paint target")
                                .default_open(false)
                                .show(ui, |ui| {
                                    paint_target_section(ui, vp, frame, &mut self.status)
                                });
                            egui::CollapsingHeader::new("Material")
                                .default_open(false)
                                .show(ui, |ui| {
                                    slot_clicked = material_slots_section(ui, vp);
                                });
                            egui::CollapsingHeader::new("Material factors")
                                .default_open(false)
                                .show(ui, |ui| material_factors_section(ui, vp));
                            egui::CollapsingHeader::new("Light")
                                .default_open(false)
                                .show(ui, |ui| light_section(ui, vp));
                        }
                    });
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
                                );
                            });
                        if self.uv_channel != prev_uv_channel {
                            vp.view_mode = self.uv_channel;
                            apply_uv_channel_to_brush(self.uv_channel, vp);
                        }
                    }
                    vp.show(ui, frame, stencil_view, stencil_aspect, stencil_tex_id);
                } else {
                    ui.centered_and_justified(|ui| ui.label("Initializing GPU…"));
                }
            });

        // Floating UV view — only when the feature is enabled AND the
        // user has undocked it. Closing the window via its [×] hides the
        // UV view entirely (same as unchecking View → UV view).
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
            if i.consume_key(Modifiers::NONE, Key::B) {
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

fn tool_strip(ui: &mut egui::Ui, vp: &Viewport) -> Option<Tool> {
    ui.add_space(4.0);
    let mut clicked: Option<Tool> = None;
    ui.vertical_centered(|ui| {
        let entries = [
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
    match std::process::Command::new("sh").arg("-c").arg(&cmd_str).status() {
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
            let tag = if lum >= 0.5 { "reveal (white)" } else { "hide (black)" };
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
            let (rect, resp) = ui.allocate_exact_size(
                egui::vec2(22.0, 22.0),
                egui::Sense::click(),
            );
            ui.painter()
                .rect_filled(rect, 3.0, fill);
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
                egui::Frame::NONE.fill(row_bg).inner_margin(4.0).show(ui, |ui| {
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
                });
                if activate {
                    vp.layer_stack.active = i;
                }
            }
        });

    ui.add_space(4.0);
    let mut add_fill_requested = false;
    ui.horizontal(|ui| {
        if ui.button("+ Paint").clicked() {
            add_requested = true;
        }
        if ui.button("+ Fill").clicked() {
            add_fill_requested = true;
        }
        ui.weak(format!("{n} layer{}", if n == 1 { "" } else { "s" }));
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
        egui::Slider::new(&mut vp.env_rotation_y, -std::f32::consts::PI..=std::f32::consts::PI)
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
    ui.label("Display");
    ui.horizontal(|ui| {
        ui.checkbox(&mut vp.fxaa.enabled, "FXAA");
        ui.checkbox(&mut vp.wireframe.visible, "wireframe");
    });

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
    ui.horizontal(|ui| {
        let baked = vp.mesh_maps.baked;
        ui.weak(if baked {
            "status: baked"
        } else {
            "status: not baked"
        });
        if ui.button("Bake").clicked() {
            if let Some(rs) = frame.wgpu_render_state() {
                vp.bake_mesh_maps(&rs.device, &rs.queue);
            }
        }
    });
    ui.weak("World normal + position baked via MRT. Used by projection paint.");

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
    let mask_layer_changed = *channel == ViewMode::Mask
        && *thumb_layer_idx != Some(active_layer_idx);
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

    let (rect, response) = ui.allocate_exact_size(
        ui.available_size(),
        egui::Sense::click_and_drag(),
    );

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
        let cursor = response
            .hover_pos()
            .unwrap_or_else(|| rect.center());
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
    let uv_min = (egui::vec2(rect.min.x, rect.min.y) - egui::vec2(rect.min.x, rect.min.y)
        - *pan)
        / *zoom;
    let _ = uv_min;
    // Approximate visible UV range from rect.
    let uv_tl = (rect.min - rect.min - *pan) / *zoom;
    let uv_br = (rect.max - rect.min - *pan) / *zoom;
    let ix_min = uv_tl.x.floor() as i32;
    let ix_max = uv_br.x.ceil() as i32;
    let iy_min = uv_tl.y.floor() as i32;
    let iy_max = uv_br.y.ceil() as i32;
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

    // UV wireframe — draw every triangle's three edges at the mesh's
    // UV coordinates. Dedup-free for simplicity; egui batches shapes
    // internally. Clipped to the panel rect so off-screen edges cost
    // nothing. Drawn under the paint cursor but over the tile images.
    if *show_wireframe {
        let mesh = vp.cpu_mesh();
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
                // Cheap AABB cull — skip segments entirely off-panel.
                let seg = egui::Rect::from_two_pos(p0, p1);
                if seg.intersects(visible) {
                    painter.line_segment([p0, p1], stroke);
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
    if primary_active {
        if let Some(pos) = response.interact_pointer_pos() {
            let atlas_uv_now = (pos - rect.min - *pan) / *zoom;
            // Step distance in UV: ~half the brush radius in UV units (so
            // stamps overlap by ~50%), floored to a small minimum.
            let uv_radius = (vp.brush.radius / 400.0).max(1e-4);
            let step_uv = (uv_radius * 0.5).max(1e-3);
            const MAX_STEPS: u32 = 128;

            let sequence: Vec<egui::Vec2> = match *last_paint_atlas_uv {
                Some(prev) => {
                    let delta = atlas_uv_now - prev;
                    let dist = delta.length();
                    let steps = ((dist / step_uv).ceil() as u32).clamp(1, MAX_STEPS);
                    (1..=steps)
                        .map(|i| prev + delta * (i as f32 / steps as f32))
                        .collect()
                }
                None => vec![atlas_uv_now],
            };

            if let Some(rs) = frame.wgpu_render_state() {
                for uv in &sequence {
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
                    vp.stamp_at_uv(&rs.device, &rs.queue, tile_idx, local_uv);
                }
                ui.ctx().request_repaint();
            }
            *last_paint_atlas_uv = Some(atlas_uv_now);
        }
    } else {
        // Stroke ended — clear the anchor so the next stroke starts fresh
        // and doesn't interpolate a long line from wherever the previous
        // one ended.
        *last_paint_atlas_uv = None;
    }

    // Brush cursor preview — an outline ring matching the stamped radius
    // at the current zoom. Only visible while the pointer hovers the
    // panel and LMB isn't down on another widget.
    if response.hovered() {
        if let Some(hover) = response.hover_pos() {
            let uv_radius = vp.brush.radius / 400.0;
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
            let inner = screen_radius * vp.brush.hardness.clamp(0.0, 1.0);
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
        format!("UV atlas · {:.0} px/UV · RMB or MMB drag to pan · scroll to zoom", *zoom),
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
}

fn light_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.add(egui::Slider::new(&mut vp.light_intensity, 0.0..=10.0).text("intensity"));
    ui.horizontal(|ui| {
        ui.label("dir");
        ui.add(egui::DragValue::new(&mut vp.light_dir[0]).speed(0.02).prefix("x:"));
        ui.add(egui::DragValue::new(&mut vp.light_dir[1]).speed(0.02).prefix("y:"));
        ui.add(egui::DragValue::new(&mut vp.light_dir[2]).speed(0.02).prefix("z:"));
    });
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

    fn open_usd_dialog(&mut self, frame: &eframe::Frame) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("USD", &["usd", "usda", "usdc", "usdz"])
            .set_title("Open USD stage")
            .pick_file()
        else {
            return;
        };
        self.load_usd(frame, path);
    }

    fn load_usd(&mut self, frame: &eframe::Frame, path: PathBuf) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.status = "No GPU render state available.".to_string();
            return;
        };
        let Some(vp) = &mut self.viewport else {
            self.status = "Viewport not initialized yet.".to_string();
            return;
        };

        match crate::usd::load_stage_merged(&path) {
            Ok(cpu) => {
                let tris = cpu.indices.len();
                let verts = cpu.positions.len();
                vp.set_mesh(&render_state.device, &render_state.queue, &cpu);

                let work_dir = crate::persist::default_work_dir(&path);
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

                self.current_usd_path = Some(path.clone());
                let sidecar_msg = if loaded_n > 0 {
                    format!(" — loaded {loaded_n} sidecar(s) from {}", work_dir.display())
                } else {
                    String::new()
                };
                self.status = format!(
                    "Loaded {} — {verts} verts, {tris} tris, {} UDIM tiles{sidecar_msg}",
                    path.display(),
                    vp.tiles().len()
                );
                log::info!("{}", self.status);
            }
            Err(e) => {
                self.status = format!("Failed to load {}: {e:#}", path.display());
                log::error!("{}", self.status);
            }
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
        // Resolve `assets/…` first relative to cwd, then fall back to
        // the path baked at compile time so the scan still works when
        // the binary is launched from an unexpected working directory.
        let resolve = |rel: &str| -> std::path::PathBuf {
            let cwd = std::path::PathBuf::from(rel);
            if cwd.is_dir() {
                return cwd;
            }
            let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
            manifest
        };
        let dirs = [
            resolve("assets/stencils"),
            resolve("assets/displacement"),
        ];
        let mut count = 0usize;
        for dir in &dirs {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<std::path::PathBuf> =
                entries.flatten().map(|e| e.path()).collect();
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
                match self.browser.import_texture(&path, &rs.device, &rs.queue, &mut renderer) {
                    Ok(()) => count += 1,
                    Err(e) => log::warn!("bundled asset {}: {e:#}", path.display()),
                }
            }
        }

        // Also scan for USD meshes — populate the Meshes tab so the
        // default mesh is available without hunting through File > Open.
        let resolve = |rel: &str| -> std::path::PathBuf {
            let cwd = std::path::PathBuf::from(rel);
            if cwd.is_dir() {
                return cwd;
            }
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
        };
        for dir in &[resolve("assets/default_mesh"), resolve("assets/meshes")] {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<std::path::PathBuf> =
                entries.flatten().map(|e| e.path()).collect();
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
                    ui.label(egui::RichText::new(format!("Stencil: {}", name)).strong().color(
                        egui::Color32::from_rgb(255, 220, 100),
                    ));
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
                            let btn = egui::Button::new(glyph)
                                .min_size(egui::vec2(80.0, 80.0));
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
                                egui::RichText::new(&mesh.name).small().color(
                                    ui.style().visuals.weak_text_color(),
                                ),
                            );
                        });
                    }
                });
            });
        if let Some(path) = load_requested {
            self.load_usd(frame, path);
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
                            let img = egui::Image::new((
                                asset.thumb_id,
                                egui::vec2(80.0, 80.0),
                            ))
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
                                egui::RichText::new(&asset.name).small().color(
                                    ui.style().visuals.weak_text_color(),
                                ),
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

    fn apply_asset_action(
        &mut self,
        idx: usize,
        action: AssetAction,
        frame: &eframe::Frame,
    ) {
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
                    vp.layer_stack
                        .add_mask_to(active, &rs.device, &rs.queue);
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
        self.status = format!(
            "Stencil: '{}' · M/R/T + LMB to move/rotate/scale",
            name
        );
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
        let result = self.browser.import_texture(
            &path,
            &rs.device,
            &rs.queue,
            &mut renderer,
        );
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
