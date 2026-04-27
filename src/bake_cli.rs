//! `forge-paint bake …` — headless mesh-map baker. Exposes texture-baker's
//! standalone CLI as a subcommand so the same workflows (CI, batch jobs)
//! work without launching the GUI. Args mirror the upstream texture-baker
//! binary 1:1 — drop-in compatible.

use clap::Args;
use std::path::Path;
use std::time::Instant;

use texture_baker::baker::{BakeConfig, BakeRequest};
use texture_baker::bakers::ao::{Distribution, RaySettings};
use texture_baker::bakers::curvature::CurvatureSettings;
use texture_baker::bakers::id::IdSource;
use texture_baker::bakers::normal::NormalMapFormat;
use texture_baker::mesh::Mesh;

#[derive(Args, Debug)]
pub struct BakeArgs {
    /// Path to the low-poly mesh (OBJ, glTF, GLB).
    #[arg(short, long)]
    lowpoly: String,

    /// Path to the high-poly mesh. If omitted, bakes from low-poly only.
    #[arg(short = 'H', long)]
    highpoly: Option<String>,

    /// Path to a cage mesh (OBJ, glTF, GLB). Must share topology with the
    /// low-poly. Used to drive the projection rays — generally improves
    /// quality on thin geometry.
    #[arg(short = 'C', long)]
    cage: Option<String>,

    /// Output texture width in pixels.
    #[arg(long, default_value_t = 2048)]
    width: u32,
    /// Output texture height in pixels.
    #[arg(long, default_value_t = 2048)]
    height: u32,

    /// Output directory.
    #[arg(short, long, default_value = ".")]
    output: String,
    /// Output filename prefix.
    #[arg(long, default_value = "bake")]
    prefix: String,

    /// Max frontal ray distance for low→high projection.
    #[arg(long, default_value_t = 0.5)]
    frontal_distance: f32,
    /// Max rear ray distance for low→high projection.
    #[arg(long, default_value_t = 0.5)]
    rear_distance: f32,
    /// Don't ignore back-face hits during projection (rare).
    #[arg(long)]
    no_ignore_backface: bool,
    /// Dilation/padding pixels (0 = infinite — fills all empty texels).
    #[arg(long, default_value_t = 0)]
    dilation: u32,

    // --- Map selection (skip-flags follow the upstream convention so the
    //     defaults match — normal/ao/curvature/position bake by default).
    /// Skip tangent-space normal map.
    #[arg(long)]
    no_normal: bool,
    /// Bake world-space normal map (off by default unless curvature is on).
    #[arg(long)]
    world_normal: bool,
    /// Skip ambient occlusion.
    #[arg(long)]
    no_ao: bool,
    /// Skip curvature.
    #[arg(long)]
    no_curvature: bool,
    /// Skip position map.
    #[arg(long)]
    no_position: bool,
    /// Bake thickness map.
    #[arg(long)]
    thickness: bool,
    /// Bake height / displacement map.
    #[arg(long)]
    height_map: bool,
    /// Bake bent-normals map.
    #[arg(long)]
    bent_normals: bool,
    /// Bake mesh-id / material-id map.
    #[arg(long)]
    id: bool,

    // --- Per-map ray settings.
    #[arg(long, default_value_t = 128)]
    ao_rays: u32,
    #[arg(long, default_value_t = 0.0)]
    ao_max_distance: f32,
    #[arg(long, default_value_t = 180.0)]
    ao_spread: f32,
    /// AO ray distribution: "cosine" or "uniform".
    #[arg(long, default_value = "cosine")]
    ao_distribution: String,

    #[arg(long)]
    thickness_rays: Option<u32>,
    #[arg(long)]
    thickness_max_distance: Option<f32>,
    #[arg(long, default_value_t = 180.0)]
    thickness_spread: f32,

    #[arg(long)]
    bent_rays: Option<u32>,
    #[arg(long)]
    bent_max_distance: Option<f32>,

    /// Normal map Y convention: "directx" (default) or "opengl".
    #[arg(long, default_value = "directx")]
    normal_format: String,

    #[arg(long, default_value_t = 1.0)]
    curvature_intensity: f32,
    #[arg(long, default_value_t = 1.0)]
    curvature_detail: f32,
    #[arg(long, default_value_t = 1.0)]
    curvature_radius: f32,

    /// Anti-aliasing supersampling factor (1 = none, 2 = 2×, 4 = 4×, 8 = 8×).
    #[arg(long, default_value_t = 1)]
    aa: u32,

    /// Disable GPU acceleration (force CPU paths).
    #[arg(long)]
    no_gpu: bool,

    // --- Multi-mesh matching (low→high by name suffix).
    #[arg(long)]
    match_by_name: bool,
    #[arg(long, default_value = "_low")]
    low_suffix: String,
    #[arg(long, default_value = "_high")]
    high_suffix: String,
}

/// Run a bake job and return the process exit code (0 = success).
pub fn run(args: BakeArgs) -> i32 {
    let start = Instant::now();

    log::info!("Loading low-poly mesh: {}", args.lowpoly);
    let low_poly = match Mesh::load(Path::new(&args.lowpoly)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading {}: {e}", args.lowpoly);
            return 1;
        }
    };
    log::info!(
        "  {} mesh(es), {} total triangles",
        low_poly.len(),
        low_poly.iter().map(|m| m.triangle_count()).sum::<usize>()
    );

    let cage = if let Some(ref cage_path) = args.cage {
        log::info!("Loading cage mesh: {cage_path}");
        match Mesh::load(Path::new(cage_path)) {
            Ok(c) => {
                log::info!("  {} mesh(es)", c.len());
                Some(c)
            }
            Err(e) => {
                eprintln!("Error loading {cage_path}: {e}");
                return 1;
            }
        }
    } else {
        None
    };

    let high_poly = if let Some(ref hp_path) = args.highpoly {
        log::info!("Loading high-poly mesh: {hp_path}");
        match Mesh::load(Path::new(hp_path)) {
            Ok(hp) => {
                log::info!(
                    "  {} mesh(es), {} total triangles",
                    hp.len(),
                    hp.iter().map(|m| m.triangle_count()).sum::<usize>()
                );
                hp
            }
            Err(e) => {
                eprintln!("Error loading {hp_path}: {e}");
                return 1;
            }
        }
    } else {
        vec![]
    };

    let normal_format = match args.normal_format.to_lowercase().as_str() {
        "opengl" | "gl" => NormalMapFormat::OpenGL,
        _ => NormalMapFormat::DirectX,
    };
    let ao_distribution = match args.ao_distribution.to_lowercase().as_str() {
        "uniform" => Distribution::Uniform,
        _ => Distribution::Cosine,
    };
    // 0 → unlimited; downstream uses f32::MAX as the sentinel.
    let ao_max_dist = if args.ao_max_distance <= 0.0 {
        f32::MAX
    } else {
        args.ao_max_distance
    };
    let thickness_ray_count = args.thickness_rays.unwrap_or(args.ao_rays);
    let thickness_max_dist = args
        .thickness_max_distance
        .map(|d| if d <= 0.0 { f32::MAX } else { d })
        .unwrap_or(ao_max_dist);
    let bent_ray_count = args.bent_rays.unwrap_or(args.ao_rays);
    let bent_max_dist = args
        .bent_max_distance
        .map(|d| if d <= 0.0 { f32::MAX } else { d })
        .unwrap_or(ao_max_dist);

    let config = BakeConfig {
        width: args.width,
        height: args.height,
        maps: BakeRequest {
            normal: !args.no_normal,
            // Curvature is computed from world normals, so opting in to
            // curvature implicitly enables world_normal even if the user
            // didn't pass --world-normal explicitly.
            world_normal: args.world_normal || !args.no_curvature,
            ao: !args.no_ao,
            curvature: !args.no_curvature,
            position: !args.no_position,
            thickness: args.thickness,
            height: args.height_map,
            bent_normals: args.bent_normals,
            id: args.id,
        },
        normal_format,
        ao_settings: RaySettings {
            ray_count: args.ao_rays,
            max_distance: ao_max_dist,
            spread_angle: args.ao_spread,
            distribution: ao_distribution,
            ..RaySettings::default()
        },
        thickness_settings: RaySettings {
            ray_count: thickness_ray_count,
            max_distance: thickness_max_dist,
            spread_angle: args.thickness_spread,
            ..RaySettings::default()
        },
        bent_normal_settings: RaySettings {
            ray_count: bent_ray_count,
            max_distance: bent_max_dist,
            ..RaySettings::default()
        },
        max_frontal_distance: args.frontal_distance,
        max_rear_distance: args.rear_distance,
        ignore_backface: !args.no_ignore_backface,
        dilation: args.dilation,
        curvature_settings: CurvatureSettings {
            intensity: args.curvature_intensity,
            detail: args.curvature_detail,
            radius_scale: args.curvature_radius,
        },
        curvature_intensity: args.curvature_intensity,
        id_source: IdSource::MeshId,
        output_dir: args.output,
        output_prefix: args.prefix,
        match_by_name: args.match_by_name,
        low_suffix: args.low_suffix,
        high_suffix: args.high_suffix,
        aa_factor: args.aa,
        use_gpu: !args.no_gpu,
    };

    if let Err(e) = texture_baker::baker::bake(&low_poly, &high_poly, cage.as_deref(), &config) {
        eprintln!("Bake failed: {e}");
        return 1;
    }

    let elapsed = start.elapsed();
    log::info!("Total time: {:.2}s", elapsed.as_secs_f64());
    0
}
