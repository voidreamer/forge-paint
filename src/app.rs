use eframe::egui;
use std::path::PathBuf;

use crate::{mesh, viewport::Viewport};

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

        egui::SidePanel::left("tools")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Brush");
                if let Some(vp) = &mut self.viewport {
                    use crate::paint::PaintChannel;
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
                        PaintChannel::Roughness | PaintChannel::Metallic => {
                            ui.add(egui::Slider::new(&mut vp.brush.value, 0.0..=1.0).text("value"));
                        }
                    }
                    ui.add(egui::Slider::new(&mut vp.brush.radius, 0.002..=0.3).text("radius"));
                    ui.add(egui::Slider::new(&mut vp.brush.hardness, 0.0..=1.0).text("hardness"));
                    ui.add(egui::Slider::new(&mut vp.brush.opacity, 0.0..=1.0).text("opacity"));
                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("Cursor");
                    match (vp.last_hit_uv, vp.last_hit_tile) {
                        (Some(uv), Some(tile)) => {
                            ui.label(format!("uv    {:.3}, {:.3}", uv[0], uv[1]));
                            ui.label(format!("tile  {tile}"));
                        }
                        _ => {
                            ui.weak("(point at mesh)");
                        }
                    }
                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("View");
                    ui.label(format!("yaw   {:>6.2}", vp.camera.yaw));
                    ui.label(format!("pitch {:>6.2}", vp.camera.pitch));
                    ui.label(format!("dist  {:>6.2}", vp.camera.distance));
                    if ui.button("Reset camera").clicked() {
                        let t = vp.camera.target;
                        vp.camera = crate::camera::OrbitCamera::default();
                        vp.camera.target = t;
                    }
                }
            });

        egui::SidePanel::right("props")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Paint target");
                if let Some(vp) = &mut self.viewport {
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
                                self.status = format!(
                                    "Rebuilt paint target @ {new_res}×{new_res} (painted content discarded)"
                                );
                                log::info!("{}", self.status);
                            }
                        }
                    });
                    ui.label(format!("tile count   {}", tiles.len()));
                    ui.label(format!("vram approx  {}", human_bytes(vram)));
                    egui::ScrollArea::vertical()
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
                    ui.add_space(8.0);
                    ui.separator();
                    ui.heading("Material factors");
                    ui.horizontal(|ui| {
                        ui.label("base color ×");
                        ui.color_edit_button_rgb(&mut vp.base_color_factor);
                    });
                    ui.add(
                        egui::Slider::new(&mut vp.metallic_factor, 0.0..=2.0).text("metallic ×"),
                    );
                    ui.add(
                        egui::Slider::new(&mut vp.roughness_factor, 0.0..=2.0).text("roughness ×"),
                    );
                    ui.add(
                        egui::Slider::new(&mut vp.normal_scale, 0.0..=2.0).text("normal scale"),
                    );
                    ui.add_space(8.0);
                    ui.separator();
                    ui.heading("Light");
                    ui.add(
                        egui::Slider::new(&mut vp.light_intensity, 0.0..=10.0).text("intensity"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("dir");
                        ui.add(egui::DragValue::new(&mut vp.light_dir[0]).speed(0.02).prefix("x:"));
                        ui.add(egui::DragValue::new(&mut vp.light_dir[1]).speed(0.02).prefix("y:"));
                        ui.add(egui::DragValue::new(&mut vp.light_dir[2]).speed(0.02).prefix("z:"));
                    });
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(18, 18, 22)))
            .show(ctx, |ui| {
                if let Some(vp) = &mut self.viewport {
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
    }
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
                    vp.paint_target(),
                    &work_dir,
                );

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
        let n = crate::persist::load_sidecars(&render_state.queue, vp.paint_target(), &dir);
        self.status = if n > 0 {
            format!("Reloaded {n} sidecar(s) from {}", dir.display())
        } else {
            format!("No sidecars at {}", dir.display())
        };
        log::info!("{}", self.status);
    }
}
