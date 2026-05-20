#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod accel;
mod app;
mod assets;
mod background;
mod bake;
mod bake_cli;
mod camera;
mod env;
mod export;
mod fxaa;
mod hydra_view;
mod lights;
mod mesh;
mod paint;
mod persist;
mod pick;
mod post;
mod project;
mod render;
mod tangents;
mod wireframe;
mod undo;
mod usd;
mod viewport;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// forge-paint — USD-centric Rust painter (standalone / anvil-aware).
///
/// Run with no subcommand (and an optional path) to launch the GUI;
/// run `forge-paint bake …` to bake mesh maps headlessly via the
/// vendored texture-baker crate.
///
/// Env vars respected when present (all optional):
///   FORGE_PAINT_WORK_DIR   — base dir for sidecar save/load (default: next to USD)
///   FORGE_PAINT_RESOLUTION — default tile resolution (default: 2048)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// USD / mesh file to open on startup (GUI mode only). Plain paths
    /// or `forge://` URIs are both accepted; URIs resolve through the
    /// C++ ForgeResolver loaded via PXR_PLUGINPATH_NAME (set by anvil).
    path: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Bake mesh maps (AO, normal, curvature, position, …) headlessly.
    /// Drop-in compatible with the standalone `texture-baker` CLI.
    Bake(bake_cli::BakeArgs),
}

fn main() -> eframe::Result<()> {
    setup_bundled_usd_env();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();

    let args = Args::parse();

    if let Some(Cmd::Bake(b)) = args.cmd {
        std::process::exit(bake_cli::run(b));
    }

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
        Box::new(move |cc| {
            let mut fonts = eframe::egui::FontDefinitions::default();
            egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(app::App::new(args.path)))
        }),
    )
}

/// Self-locating env setup for the Windows / macOS hand-off bundles.
///
/// USD's plugin discovery reads `PXR_PLUGINPATH_NAME` lazily, the
/// first time any USD library function runs. Without it, none of the
/// file-format readers (usda, usdc, usdz) register and `Stage::open`
/// returns null for every input. Anvil and locally-built dev runs set
/// the variable via the surrounding shell; the hand-off zips ship the
/// entire USD install as a sibling `usd/` directory next to the EXE
/// and rely on this function to point USD at it instead.
///
/// On Windows we additionally prepend `usd/lib` and `usd/bin` to PATH
/// so the dynamic loader resolves the USD + TBB DLLs. On macOS the
/// equivalent — making the loader find `libusd_*.dylib` — is handled
/// by `install_name_tool -add_rpath @executable_path/usd/lib` in the
/// CI workflow, so no env-var work is needed at runtime here.
fn setup_bundled_usd_env() {
    #[cfg(any(windows, target_os = "macos"))]
    {
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let Some(dir) = exe.parent() else {
            return;
        };
        let usd = dir.join("usd");
        if !usd.is_dir() {
            return;
        }
        #[cfg(windows)]
        let sep = ";";
        #[cfg(not(windows))]
        let sep = ":";
        let plugin_path = format!(
            "{}{sep}{}",
            usd.join("plugin").join("usd").display(),
            usd.join("lib").join("usd").display(),
        );
        // SAFETY: called from main before any threads spawn — eframe's
        // render thread only starts inside run_native(). Edition 2024
        // marks env::set_var unsafe because of cross-thread races; we
        // have none here.
        unsafe {
            std::env::set_var("PXR_PLUGINPATH_NAME", plugin_path);
        }
        #[cfg(windows)]
        {
            let new_path = match std::env::var("PATH") {
                Ok(p) => format!(
                    "{};{};{}",
                    usd.join("lib").display(),
                    usd.join("bin").display(),
                    p,
                ),
                Err(_) => format!(
                    "{};{}",
                    usd.join("lib").display(),
                    usd.join("bin").display(),
                ),
            };
            unsafe {
                std::env::set_var("PATH", new_path);
            }
        }
    }
}
