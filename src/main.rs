#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod accel;
mod app;
mod camera;
mod env;
mod export;
mod mesh;
mod paint;
mod persist;
mod pick;
mod render;
mod tangents;
mod undo;
mod usd;
mod viewport;

use clap::Parser;
use std::path::PathBuf;

/// forge-paint — USD-centric Rust painter (standalone / anvil-aware).
///
/// Env vars respected when present (all optional):
///   FORGE_PAINT_WORK_DIR   — base dir for sidecar save/load (default: next to USD)
///   FORGE_PAINT_RESOLUTION — default tile resolution (default: 2048)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// USD file to open on startup (optional). Plain paths or `forge://` URIs
    /// (resolved by usdcat when its env is active) are both accepted.
    path: Option<PathBuf>,
}

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();

    let args = Args::parse();

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
        Box::new(move |_cc| Ok(Box::new(app::App::new(args.path)))),
    )
}
