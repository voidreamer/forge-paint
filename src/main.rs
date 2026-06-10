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
mod gltf_to_usd;
mod hydra_view;
mod lights;
mod material_graph;
mod mesh;
mod obj_to_usd;
mod paint;
mod persist;
mod pick;
mod post;
mod project;
mod render;
mod stage_browser;
mod tangents;
mod undo;
mod usd;
mod usd_out;
mod usdz;
mod viewport;
mod wireframe;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

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

    /// Convert a model file to USD headlessly. OBJ and glTF/GLB go
    /// through the built-in static converters; Alembic (and USD
    /// itself) are re-encoded through the USD runtime, which needs
    /// the usdAbc plugin for .abc input. The output format follows
    /// the destination extension: .usd/.usdc = crate binary
    /// (recommended), .usda = text.
    Convert(ConvertArgs),
}

#[derive(clap::Args, Debug)]
struct ConvertArgs {
    /// Source model: .obj, .gltf, .glb, .abc, or any USD file.
    source: PathBuf,
    /// Destination USD file: .usdc / .usd (binary) or .usda (text).
    dest: PathBuf,
}

fn run_convert(args: &ConvertArgs) -> i32 {
    let ext = args
        .source
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let result = match ext.as_str() {
        "obj" => obj_to_usd::convert_obj_to_usd(&args.source, &args.dest).map(|summary| {
            format!("{} verts, {} tris", summary.vertices, summary.triangles)
        }),
        "gltf" | "glb" => {
            gltf_to_usd::convert_gltf_to_usd(&args.source, &args.dest).map(|summary| {
                format!(
                    "{} meshes, {} verts, {} tris",
                    summary.meshes, summary.vertices, summary.triangles
                )
            })
        }
        // Anything USD itself can read — .abc through the usdAbc
        // plugin, or a USD file being re-encoded text<->binary.
        "abc" | "usd" | "usda" | "usdc" => {
            if rust_usd::convert_usd_file(&args.source, &args.dest) {
                Ok("re-encoded through USD".to_string())
            } else {
                Err(anyhow::anyhow!(
                    "USD could not open {} as a layer (for .abc this needs the usdAbc plugin) or could not write {}",
                    args.source.display(),
                    args.dest.display()
                ))
            }
        }
        other => Err(anyhow::anyhow!(
            "unsupported source format `.{other}` (supported: .obj, .gltf, .glb, .abc, .usd*)"
        )),
    };
    match result {
        Ok(detail) => {
            println!(
                "converted {} -> {} ({detail})",
                args.source.display(),
                args.dest.display()
            );
            0
        }
        Err(e) => {
            eprintln!("conversion failed: {e:#}");
            1
        }
    }
}

fn main() -> eframe::Result<()> {
    // Hand-off bundles must not run the app from the bundle root —
    // OpenUSD would load twice and deadlock. Relaunch from usd\lib
    // before anything touches USD. See the function comment.
    #[cfg(windows)]
    if let Some(code) = relaunch_from_bundled_usd_lib() {
        std::process::exit(code);
    }

    setup_bundled_usd_env();

    // On Windows release builds `windows_subsystem = "windows"` detaches
    // the console, so env_logger's stderr output and panic messages go
    // nowhere — a crash looks like the app silently vanishing. Tee the
    // log to a file beside the exe (falling back to the temp dir) so we
    // always have a post-mortem trail. The path is logged to stderr too
    // for dev runs that DO have a console.
    let log_path = init_file_log();

    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
    );
    if let Some(ref path) = log_path {
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            builder.target(env_logger::Target::Pipe(Box::new(file)));
        }
    }
    builder.init();

    install_panic_logger(log_path.clone());
    #[cfg(windows)]
    install_native_crash_logger(log_path.clone());

    if let Some(ref path) = log_path {
        log::info!("forge-paint starting — log file: {}", path.display());
        // Hand the log path to the C++ Hydra bridge so it can append
        // its own breadcrumbs — the bridge's stderr is dead on the
        // console-less Windows build. The bridge reads this env var and
        // no-ops if it's unset.
        // SAFETY: single-threaded startup, before eframe spawns threads.
        unsafe {
            std::env::set_var("FORGE_PAINT_HYDRA_LOG", path);
        }
    }

    if let Some(code) = run_hydra_probe_from_env() {
        std::process::exit(code);
    }

    let args = Args::parse();

    match args.cmd {
        Some(Cmd::Bake(b)) => std::process::exit(bake_cli::run(b)),
        Some(Cmd::Convert(c)) => std::process::exit(run_convert(&c)),
        None => {}
    }

    // Bump the wgpu device's per-buffer cap so dense meshes
    // (SimReady-class assets, hi-res hero geometry) can land their
    // vertex / index buffers in a single allocation. The downlevel
    // default is 256 MB; modern desktop GPUs report several GBs, and
    // refusing to push past 256 MB means assets with ~10 M verts
    // fail to load with a `forge_paint_mesh_vb` validation error. We
    // start from the adapter's reported limits (the actual GPU
    // ceiling) and just raise `max_buffer_size` to 2 GB on top — the
    // adapter-default for non-buffer limits stays unchanged.
    let wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: std::sync::Arc::new(|adapter| {
                    let mut limits = adapter.limits();
                    // 2 GB — comfortably covers ~40 M verts at 48 B
                    // each. Bump again if a real consumer hits it.
                    const TWO_GB: u64 = 2 * 1024 * 1024 * 1024;
                    limits.max_buffer_size = limits.max_buffer_size.max(TWO_GB);
                    egui_wgpu::wgpu::DeviceDescriptor {
                        label: Some("forge-paint device"),
                        required_features: egui_wgpu::wgpu::Features::empty(),
                        required_limits: limits,
                        memory_hints: egui_wgpu::wgpu::MemoryHints::default(),
                    }
                }),
                ..Default::default()
            },
        ),
        ..Default::default()
    };

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("forge-paint"),
        wgpu_options,
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

fn run_hydra_probe_from_env() -> Option<i32> {
    std::env::var_os("FORGE_PAINT_HYDRA_PROBE")?;
    let Some(stage) = std::env::var_os("FORGE_PAINT_HYDRA_PROBE_STAGE").map(PathBuf::from) else {
        log::error!("Hydra startup probe requested without FORGE_PAINT_HYDRA_PROBE_STAGE");
        return Some(2);
    };
    let delegate = std::env::var("FORGE_PAINT_HYDRA_PROBE_DELEGATE")
        .ok()
        .filter(|id| !id.is_empty());
    let delegate_label = delegate.as_deref().unwrap_or("default delegate");
    log::info!(
        "Hydra startup probe opening {} via {}",
        stage.display(),
        delegate_label
    );
    let started = std::time::Instant::now();
    let result = (|| -> anyhow::Result<()> {
        let mut view = hydra_view::HydraView::new_with_delegate(&stage, delegate.as_deref())
            .with_context(|| format!("constructing Hydra renderer via {delegate_label}"))?;
        view.resize(64, 64);
        let view_matrix = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let proj = hydra_view::perspective_for_hydra(45.0_f32.to_radians(), 1.0, 0.01, 1000.0);
        let pixels = view
            .render(&view_matrix, &proj)
            .context("rendering one Hydra startup probe frame")?;
        anyhow::ensure!(
            pixels.len() == 64 * 64 * 4,
            "Hydra probe returned {} bytes, expected {}",
            pixels.len(),
            64 * 64 * 4
        );
        Ok(())
    })();
    match result {
        Ok(()) => {
            log::info!(
                "Hydra startup probe OK for {} in {:.2}s",
                delegate_label,
                started.elapsed().as_secs_f32()
            );
            Some(0)
        }
        Err(e) => {
            log::error!(
                "Hydra startup probe failed for {} after {:.2}s: {e:#}",
                delegate_label,
                started.elapsed().as_secs_f32()
            );
            Some(3)
        }
    }
}

/// Choose a writable log path: `forge-paint.log` next to the exe if
/// that directory is writable, else `<temp>/forge-paint.log`. Returns
/// None only if neither is usable (logging then stays stderr-only).
fn init_file_log() -> Option<std::path::PathBuf> {
    // FORGE_PAINT_LOG_FILE wins when set: the bundle relaunch parent
    // pins the log beside the user-facing EXE so the relaunched child
    // in usd\lib doesn't scatter logs into the USD runtime tree.
    let candidates = std::env::var_os("FORGE_PAINT_LOG_FILE")
        .map(std::path::PathBuf::from)
        .into_iter()
        .chain(
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|p| p.join("forge-paint.log"))),
        )
        .chain(std::iter::once(std::env::temp_dir().join("forge-paint.log")));
    for path in candidates {
        // Probe writability by opening in append/create mode.
        if std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .is_ok()
        {
            return Some(path);
        }
    }
    None
}

/// Route Rust panics to both the default handler and the log file, so
/// a panic on the console-less Windows build still leaves a trace.
fn install_panic_logger(log_path: Option<std::path::PathBuf>) {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("PANIC: {info}");
        log::error!("{msg}");
        if let Some(ref path) = log_path {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(f, "{msg}");
                let bt = std::backtrace::Backtrace::force_capture();
                let _ = writeln!(f, "{bt}");
            }
        }
        default(info);
    }));
}

/// Windows-only: install a vectored exception handler that appends the
/// exception code + faulting address to the log right before a native
/// (C++ / Hydra) access violation tears the process down. A Rust panic
/// hook can't see these — they bypass unwinding entirely — so without
/// this an HgiGL / delegate-switch crash just vanishes with no trace.
#[cfg(windows)]
fn install_native_crash_logger(log_path: Option<std::path::PathBuf>) {
    use std::sync::Mutex;
    // Stash the path in a static so the bare `extern "system"` handler
    // (which can't capture) can reach it.
    static LOG_PATH: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);
    if let Ok(mut g) = LOG_PATH.lock() {
        *g = log_path;
    }

    // Minimal manual FFI — avoids pulling the whole `windows` crate in
    // just for one handler. AddVectoredExceptionHandler fires for every
    // structured exception in-process; we log the fatal ones and let
    // the OS continue its normal (terminating) search so behaviour is
    // otherwise unchanged.
    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut std::ffi::c_void,
        address: *mut std::ffi::c_void,
        // remaining fields unused
    }
    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut std::ffi::c_void,
    }
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    // Codes we treat as fatal-worth-logging.
    const ACCESS_VIOLATION: u32 = 0xC000_0005;
    const ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
    const STACK_OVERFLOW: u32 = 0xC000_00FD;

    // One-shot guard: an access violation may have corrupted the heap,
    // and the file-open below allocates — if that re-faults we must not
    // recurse into ourselves forever. Log at most once.
    static LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    unsafe extern "system" fn handler(info: *mut ExceptionPointers) -> i32 {
        unsafe {
            if !info.is_null() {
                let rec = (*info).exception_record;
                if !rec.is_null() {
                    let code = (*rec).code;
                    if (code == ACCESS_VIOLATION
                        || code == ILLEGAL_INSTRUCTION
                        || code == STACK_OVERFLOW)
                        && !LOGGED.swap(true, std::sync::atomic::Ordering::SeqCst)
                    {
                        if let Ok(g) = LOG_PATH.lock() {
                            if let Some(ref path) = *g {
                                use std::io::Write;
                                if let Ok(mut f) = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(path)
                                {
                                    let _ = writeln!(
                                        f,
                                        "NATIVE EXCEPTION 0x{:08X} at {:p} — likely inside the Hydra/Hgi bridge. See the last log line above for the call that was in flight.",
                                        code,
                                        (*rec).address
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    unsafe extern "system" {
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
        ) -> *mut std::ffi::c_void;
    }
    unsafe {
        AddVectoredExceptionHandler(1, handler);
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn push_usd_plugin_path_dirs(plugin_paths: &mut Vec<PathBuf>, root: &Path) {
    if !root.is_dir() {
        return;
    }

    plugin_paths.push(root.to_path_buf());
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("plugInfo.json").is_file() {
            plugin_paths.push(dir.clone());
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn dedup_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::new();
    paths.retain(|p| seen.insert(p.clone()));
}

#[cfg(windows)]
fn delight_install_roots(bundle_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["FORGE_PAINT_3DELIGHT_DIR", "DELIGHT", "Delight"] {
        if let Some(value) = std::env::var_os(var) {
            roots.push(PathBuf::from(value));
        }
    }
    roots.push(bundle_dir.join("3Delight"));

    for var in ["ProgramFiles", "ProgramFiles(x86)"] {
        let Some(base) = std::env::var_os(var).map(PathBuf::from) else {
            continue;
        };
        roots.push(base.join("3Delight"));
        roots.push(base.join("3DelightNSI"));
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.contains("3delight") {
                    roots.push(path);
                }
            }
        }
    }

    roots.retain(|root| root.join("bin").join("renderdl.exe").is_file());
    dedup_paths(&mut roots);
    roots
}

#[cfg(windows)]
fn delight_runtime_dirs(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for root in roots.iter().filter(|root| root.is_dir()) {
        dirs.push(root.to_path_buf());
        if root.is_dir() {
            dirs.push(root.join("bin"));
            dirs.push(root.join("lib"));
        }
    }
    dirs.retain(|p| p.is_dir());
    dedup_paths(&mut dirs);
    dirs
}

#[cfg(windows)]
fn path_has_component_case_insensitive(path: &Path, needle: &str) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(needle)
    })
}

#[cfg(windows)]
fn push_hdnsi_plugin_dirs_from_delight(plugin_paths: &mut Vec<PathBuf>, root: &Path) {
    let known_roots = [
        root.join("hdNSI"),
        root.join("usd").join("hdNSI"),
        root.join("plugin").join("usd").join("hdNSI"),
        root.join("plugins").join("usd").join("hdNSI"),
        root.join("hydra").join("hdNSI"),
    ];
    for known_root in known_roots {
        push_usd_plugin_path_dirs(plugin_paths, &known_root);
    }

    // Some 3Delight installers put DCC integrations in product-specific
    // subtrees. Keep this bounded so startup does not crawl arbitrary
    // Program Files contents, but still find e.g. .../hdNSI/resources.
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if dir.join("plugInfo.json").is_file() && path_has_component_case_insensitive(&dir, "hdNSI")
        {
            plugin_paths.push(dir.clone());
        }
        if depth >= 7 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
}

#[cfg(windows)]
fn usd_plugin_dll_search_dirs(plugin_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for plugin_path in plugin_paths {
        dirs.push(plugin_path.clone());
        let mut current = plugin_path.as_path();
        for _ in 0..3 {
            let Some(parent) = current.parent() else {
                break;
            };
            dirs.push(parent.to_path_buf());
            current = parent;
        }
    }
    dirs.retain(|p| p.is_dir());
    dedup_paths(&mut dirs);
    dirs
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
        // The relaunched bundle child (Windows) already inherits the
        // fully-formed environment from its parent. Recomputing here
        // would anchor everything at usd\lib (where this copy lives)
        // and prepend nonsense paths.
        #[cfg(windows)]
        if std::env::var_os("FORGE_PAINT_BUNDLE_BOOTSTRAPPED").is_some() {
            return;
        }
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let Some(dir) = exe.parent() else {
            return;
        };
        for (name, value) in compute_bundled_usd_env(dir) {
            // SAFETY: called from main before any threads spawn — eframe's
            // render thread only starts inside run_native(). Edition 2024
            // marks env::set_var unsafe because of cross-thread races; we
            // have none here.
            //
            // Windows caveat: this is best-effort only. USD captures
            // PXR_PLUGINPATH_NAME inside an ARCH_CONSTRUCTOR while
            // usd_plug.dll is loading (before main), and C-runtime
            // getenv snapshots the environment at process start, so
            // these set_var calls are invisible to both. Bundled runs
            // therefore go through relaunch_from_bundled_usd_lib(),
            // which applies this same environment to a child process
            // where it IS present from the first instruction. This
            // in-process path remains for macOS (where the bundled
            // plugins are found via plug's library-relative search)
            // and as a fallback for old zips without the usd\lib EXE.
            unsafe {
                std::env::set_var(name, value);
            }
        }
    }
}

/// Environment the self-contained bundle needs, computed against the
/// bundle root `dir` (the directory holding the user-facing EXE, the
/// `usd/` runtime tree, and the optional `plugins/usd/` extras).
/// Returns an empty list when `dir` doesn't look like a bundle.
#[cfg(any(windows, target_os = "macos"))]
fn compute_bundled_usd_env(dir: &Path) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut vars = Vec::new();
    let usd = dir.join("usd");
    if !usd.is_dir() {
        return vars;
    }
    #[cfg(windows)]
    let sep = ";";
    #[cfg(not(windows))]
    let sep = ":";
    #[cfg(windows)]
    let delight_roots = delight_install_roots(dir);
    #[cfg(windows)]
    if std::env::var_os("DELIGHT").is_none() {
        if let Some(delight_root) = delight_roots.first() {
            vars.push(("DELIGHT", delight_root.clone().into_os_string()));
        }
    }
    let mut plugin_paths = Vec::new();
    push_usd_plugin_path_dirs(&mut plugin_paths, &usd.join("plugin").join("usd"));
    push_usd_plugin_path_dirs(&mut plugin_paths, &usd.join("lib").join("usd"));
    let optional_plugins = dir.join("plugins").join("usd");
    push_usd_plugin_path_dirs(&mut plugin_paths, &optional_plugins);
    #[cfg(windows)]
    for delight_root in &delight_roots {
        push_hdnsi_plugin_dirs_from_delight(&mut plugin_paths, delight_root);
    }
    dedup_paths(&mut plugin_paths);
    let mut plugin_path = plugin_paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(sep);
    if let Ok(existing) = std::env::var("PXR_PLUGINPATH_NAME") {
        if !existing.is_empty() {
            plugin_path.push_str(sep);
            plugin_path.push_str(&existing);
        }
    }
    vars.push(("PXR_PLUGINPATH_NAME", plugin_path.into()));
    #[cfg(windows)]
    {
        let mut path_dirs = vec![usd.join("lib"), usd.join("bin")];
        path_dirs.extend(usd_plugin_dll_search_dirs(&plugin_paths));
        path_dirs.extend(delight_runtime_dirs(&delight_roots));
        dedup_paths(&mut path_dirs);
        let prefix = path_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(";");
        let new_path = match std::env::var("PATH") {
            Ok(p) if !p.is_empty() => format!("{prefix};{p}"),
            _ => prefix,
        };
        vars.push(("PATH", new_path.into()));
    }
    vars
}

/// Relaunch hand-off bundles from `usd\lib` so OpenUSD loads exactly
/// once and the environment exists before USD initializes.
///
/// Two Windows-specific traps make running the bundle-root EXE
/// directly unworkable:
///
/// 1. Double-loaded OpenUSD. The zip keeps copies of the USD runtime
///    DLLs beside forge-paint.exe because the loader resolves the
///    import table before main() runs. But USD's plug registry later
///    LoadLibrary()s the same libraries by absolute path from
///    `usd\lib\usd_*.dll`. The loader dedupes modules by path, not by
///    name, so every USD module then exists twice in the process —
///    duplicate type registries and all — and the first Hydra renderer
///    bring-up deadlocks inside Hgi::CreatePlatformDefaultHgi.
///
/// 2. Too-late environment. USD captures PXR_PLUGINPATH_NAME in an
///    ARCH_CONSTRUCTOR while usd_plug.dll loads (before main), and
///    C-runtime getenv (the 3Delight runtime, the hydra bridge's log
///    gate) snapshots the environment at process start. The set_var
///    calls in setup_bundled_usd_env are invisible to both, which is
///    why double-clicked artifacts saw zero USD plugins ("Available:
///    []", Stage::open null) while forge-paint.bat launches worked.
///
/// Both disappear by relaunching the second EXE copy that CI places at
/// `usd\lib\forge-paint.exe` with the bundle environment computed and
/// applied to the child up front: the child's import table resolves
/// from usd\lib — the same files plug loads later — and every
/// constructor / getenv sees the right environment from the start.
///
/// Returns Some(child exit code) when the relaunch ran; None when this
/// process should continue normally (dev runs, the relaunched child
/// itself, or old zips without the usd\lib EXE copy).
#[cfg(windows)]
fn relaunch_from_bundled_usd_lib() -> Option<i32> {
    if std::env::var_os("FORGE_PAINT_BUNDLE_BOOTSTRAPPED").is_some() {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    if !dir.join("usd").is_dir() {
        return None;
    }
    let child_exe = dir.join("usd").join("lib").join(exe.file_name()?);
    if !child_exe.is_file() {
        // Old zip layout without the usd\lib copy: fall through to the
        // best-effort in-process setup (forge-paint.bat still works).
        return None;
    }

    let mut cmd = std::process::Command::new(&child_exe);
    cmd.args(std::env::args_os().skip(1));
    cmd.env("FORGE_PAINT_BUNDLE_BOOTSTRAPPED", "1");
    // Keep the log beside the user-facing EXE rather than in usd\lib.
    // Handing the bridge's log variable to the child here (instead of
    // via set_var inside the child) also makes it visible to the C++
    // side's getenv, so bridge breadcrumbs work in the app process and
    // not just in probe children.
    let log_path = dir.join("forge-paint.log");
    cmd.env("FORGE_PAINT_LOG_FILE", &log_path);
    if std::env::var_os("FORGE_PAINT_HYDRA_LOG").is_none() {
        cmd.env("FORGE_PAINT_HYDRA_LOG", &log_path);
    }
    for (name, value) in compute_bundled_usd_env(dir) {
        cmd.env(name, value);
    }
    match cmd.status() {
        Ok(status) => Some(status.code().unwrap_or(0)),
        Err(err) => {
            // The logger isn't up yet (it initializes after this), so
            // stderr is the best signal available; falling through to
            // the in-process setup keeps launches with a preconfigured
            // shell environment working.
            eprintln!(
                "forge-paint: bundle relaunch via {} failed: {err}",
                child_exe.display()
            );
            None
        }
    }
}
