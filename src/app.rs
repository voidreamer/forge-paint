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
}

impl App {
    pub fn new(initial_usd: Option<PathBuf>) -> Self {
        Self {
            pending_open: initial_usd,
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

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.status.is_empty() {
                    ui.weak("ready");
                } else {
                    ui.label(&self.status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("LMB paint · Ctrl+LMB orbit · Shift+LMB or MMB pan · wheel zoom");
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

        egui::SidePanel::left("tools")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                if let Some(vp) = &mut self.viewport {
                    tool_strip(ui, vp);
                    ui.separator();
                }
                egui::ScrollArea::vertical()
                    .id_salt("left_panel_scroll")
                    .show(ui, |ui| {
                        if let Some(vp) = &mut self.viewport {
                            egui::CollapsingHeader::new("Brush")
                                .default_open(true)
                                .show(ui, |ui| brush_section(ui, vp));
                            egui::CollapsingHeader::new("Cursor")
                                .default_open(false)
                                .show(ui, |ui| cursor_section(ui, vp));
                            egui::CollapsingHeader::new("View")
                                .default_open(false)
                                .show(ui, |ui| view_section(ui, vp));
                        }
                    });
            });

        egui::SidePanel::right("props")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("right_panel_scroll")
                    .show(ui, |ui| {
                        if let Some(vp) = &mut self.viewport {
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
                            egui::CollapsingHeader::new("Material factors")
                                .default_open(false)
                                .show(ui, |ui| material_factors_section(ui, vp));
                            egui::CollapsingHeader::new("Light")
                                .default_open(false)
                                .show(ui, |ui| light_section(ui, vp));
                        }
                    });
            });

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
                        // follow with the brush so strokes land on what's shown.
                        if vp.view_mode != prev {
                            use crate::paint::PaintChannel;
                            use crate::render::ViewMode;
                            match vp.view_mode {
                                ViewMode::BaseColor => {
                                    vp.brush.channel = PaintChannel::BaseColor;
                                    vp.brush.mask_edit = false;
                                }
                                ViewMode::Roughness => {
                                    vp.brush.channel = PaintChannel::Roughness;
                                    vp.brush.mask_edit = false;
                                }
                                ViewMode::Metallic => {
                                    vp.brush.channel = PaintChannel::Metallic;
                                    vp.brush.mask_edit = false;
                                }
                                ViewMode::Mask => {
                                    // Switch to mask edit if the active layer has
                                    // one; otherwise leave the brush alone and the
                                    // viewer just previews the dummy mask.
                                    if vp.layer_stack.active_layer().mask.is_some() {
                                        vp.brush.mask_edit = true;
                                    }
                                }
                                ViewMode::Material
                                | ViewMode::Normal
                                | ViewMode::WorldNormalBaked => {}
                            }
                        }
                    });
                    vp.show(ui, frame);
                } else {
                    ui.centered_and_justified(|ui| ui.label("Initializing GPU…"));
                }
            });

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
            if let Some(vp) = &mut self.viewport {
                vp.tool = tool;
            }
        }
    }
}

fn tool_strip(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        let entries = [
            (Tool::Paint, egui_phosphor::regular::PAINT_BRUSH),
            (Tool::Erase, egui_phosphor::regular::ERASER),
            (Tool::Fill, egui_phosphor::regular::PAINT_BUCKET),
            (Tool::Eyedropper, egui_phosphor::regular::EYEDROPPER),
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
            let tooltip = format!("{} [{}]", tool.label(), tool.shortcut());
            if ui.add(btn).on_hover_text(tooltip).clicked() {
                vp.tool = tool;
            }
        }
    });
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
    ui.weak("Currently: world normal. Position / curvature / AO land in D.3+.");
}

fn brush_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    use crate::paint::PaintChannel;
    if vp.brush.mask_edit {
        ui.weak("painting mask — black/white only");
        ui.horizontal(|ui| {
            let mut white = vp.brush.value >= 0.5;
            if ui.radio_value(&mut white, true, "paint (white)").changed() && white {
                vp.brush.value = 1.0;
            }
            if ui.radio_value(&mut white, false, "erase (black)").changed() && !white {
                vp.brush.value = 0.0;
            }
        });
    } else {
        ui.horizontal(|ui| {
            ui.radio_value(&mut vp.brush.channel, PaintChannel::BaseColor, "Color");
            ui.radio_value(&mut vp.brush.channel, PaintChannel::Roughness, "Rough");
            ui.radio_value(&mut vp.brush.channel, PaintChannel::Metallic, "Metal");
        });
        match vp.brush.channel {
            PaintChannel::BaseColor => {
                ui.horizontal(|ui| {
                    ui.label("color");
                    ui.color_edit_button_rgb(&mut vp.brush.color_srgb);
                });
            }
            PaintChannel::Roughness | PaintChannel::Metallic | PaintChannel::Mask => {
                ui.add(egui::Slider::new(&mut vp.brush.value, 0.0..=1.0).text("value"));
            }
        }
    }
    ui.add(egui::Slider::new(&mut vp.brush.radius, 0.002..=0.3).text("radius"));
    ui.add(egui::Slider::new(&mut vp.brush.hardness, 0.0..=1.0).text("hardness"));
    ui.add(egui::Slider::new(&mut vp.brush.opacity, 0.0..=1.0).text("opacity"));
}

fn cursor_section(ui: &mut egui::Ui, vp: &Viewport) {
    match (vp.last_hit_uv, vp.last_hit_tile) {
        (Some(uv), Some(tile)) => {
            ui.label(format!("uv    {:.3}, {:.3}", uv[0], uv[1]));
            ui.label(format!("tile  {tile}"));
        }
        _ => {
            ui.weak("(point at mesh)");
        }
    }
}

fn view_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.label(format!("yaw   {:>6.2}", vp.camera.yaw));
    ui.label(format!("pitch {:>6.2}", vp.camera.pitch));
    ui.label(format!("dist  {:>6.2}", vp.camera.distance));
    if ui.button("Reset camera").clicked() {
        let t = vp.camera.target;
        vp.camera = crate::camera::OrbitCamera::default();
        vp.camera.target = t;
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

fn material_factors_section(ui: &mut egui::Ui, vp: &mut Viewport) {
    ui.horizontal(|ui| {
        ui.label("base color ×");
        ui.color_edit_button_rgb(&mut vp.base_color_factor);
    });
    ui.add(egui::Slider::new(&mut vp.metallic_factor, 0.0..=2.0).text("metallic ×"));
    ui.add(egui::Slider::new(&mut vp.roughness_factor, 0.0..=2.0).text("roughness ×"));
    ui.add(egui::Slider::new(&mut vp.normal_scale, 0.0..=2.0).text("normal scale"));
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

    fn asset_browser_panel(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let mut want_import = false;
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
            });
        });
        ui.separator();

        match self.browser.active_tab {
            assets::Tab::Textures => {
                self.texture_strip(ui, frame);
            }
            _ => {
                ui.weak("(this tab is not implemented yet)");
            }
        }

        if want_import {
            self.import_texture_dialog(frame);
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
        // Only PNG is enabled in the image crate's features today; adding
        // JPEG/TGA/BMP means enabling more features in Cargo.toml.
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png"])
            .set_title("Import texture")
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
                self.status = format!("Import failed: {e}");
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
        }
        log::info!("{}", self.status);
    }
}

#[derive(Debug, Clone, Copy)]
enum AssetAction {
    NewLayer,
    ApplyBaseColor,
    ApplyMask,
}
