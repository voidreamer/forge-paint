#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod accel;
mod app;
mod camera;
mod export;
mod mesh;
mod paint;
mod pick;
mod render;
mod tangents;
mod usd;
mod viewport;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("forge-paint"),
        ..Default::default()
    };

    eframe::run_native(
        "forge-paint",
        options,
        Box::new(|_cc| Ok(Box::new(app::App::default()))),
    )
}
