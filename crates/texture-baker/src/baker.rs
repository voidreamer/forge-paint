use rayon::prelude::*;
use std::path::Path;

use crate::accel::AccelStructure;
use crate::bakers::ao::{self, AoSettings, RaySettings};
use crate::bakers::curvature::{self, CurvatureSettings};
use crate::bakers::height;
use crate::bakers::id::{self, IdSource};
use crate::bakers::normal::{self, NormalMapFormat};
use crate::bakers::position::{self, PositionNormalization};
use crate::dilate;
use crate::gpu;
use crate::mesh::Mesh;
use crate::output;
use crate::raster;
use crate::supersample;
use crate::tangent;

/// Which maps to bake.
#[derive(Debug, Clone)]
pub struct BakeRequest {
    pub normal: bool,
    pub world_normal: bool,
    pub ao: bool,
    pub curvature: bool,
    pub position: bool,
    pub thickness: bool,
    pub height: bool,
    pub bent_normals: bool,
    pub id: bool,
}

impl BakeRequest {
    /// All maps disabled.
    pub fn none() -> Self {
        Self {
            normal: false,
            world_normal: false,
            ao: false,
            curvature: false,
            position: false,
            thickness: false,
            height: false,
            bent_normals: false,
            id: false,
        }
    }
}

impl Default for BakeRequest {
    fn default() -> Self {
        Self {
            normal: true,
            world_normal: true,
            ao: true,
            curvature: true,
            position: true,
            thickness: false,
            height: true,
            bent_normals: false,
            id: false,
        }
    }
}

/// Full bake configuration.
#[derive(Debug, Clone)]
pub struct BakeConfig {
    pub width: u32,
    pub height: u32,
    pub maps: BakeRequest,
    pub normal_format: NormalMapFormat,
    pub ao_settings: AoSettings,
    /// Per-map ray settings for thickness (falls back to ao_settings values when default).
    pub thickness_settings: RaySettings,
    /// Per-map ray settings for bent normals (falls back to ao_settings values when default).
    pub bent_normal_settings: RaySettings,
    pub max_frontal_distance: f32,
    pub max_rear_distance: f32,
    pub ignore_backface: bool,
    pub dilation: u32,
    /// Curvature settings (replaces the old `curvature_intensity` scalar).
    pub curvature_settings: CurvatureSettings,
    /// Backward-compatible accessor — reads/writes `curvature_settings.intensity`.
    pub curvature_intensity: f32,
    pub id_source: IdSource,
    pub output_dir: String,
    pub output_prefix: String,
    /// Match high-poly to low-poly by mesh name suffix.
    pub match_by_name: bool,
    pub low_suffix: String,
    pub high_suffix: String,
    /// Anti-aliasing supersampling factor (1 = none, 2 = 2x2, 4 = 4x4, 8 = 8x8).
    pub aa_factor: u32,
    /// Use GPU acceleration for ray-heavy bakers (AO, thickness).
    pub use_gpu: bool,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            maps: BakeRequest::default(),
            normal_format: NormalMapFormat::DirectX,
            ao_settings: AoSettings::default(),
            thickness_settings: RaySettings::default(),
            bent_normal_settings: RaySettings::default(),
            max_frontal_distance: 0.5,
            max_rear_distance: 0.5,
            ignore_backface: true,
            dilation: 0, // infinite
            curvature_settings: CurvatureSettings::default(),
            curvature_intensity: 1.0,
            id_source: IdSource::MeshId,
            output_dir: ".".to_string(),
            output_prefix: "bake".to_string(),
            match_by_name: false,
            low_suffix: "_low".to_string(),
            high_suffix: "_high".to_string(),
            aa_factor: 1,
            use_gpu: true,
        }
    }
}

/// Which single map to preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapType {
    Normal,
    WorldNormal,
    AO,
    Curvature,
    Position,
    Thickness,
    Height,
    BentNormals,
    Id,
}

/// Result of a preview bake — raw pixel buffer in memory, no disk I/O.
pub enum PreviewResult {
    Rgb(Vec<[f32; 3]>, u32, u32),
    Gray(Vec<f32>, u32, u32),
}

/// Bake a single map at preview quality, returning the raw pixel buffer.
/// Reuses pre-built accel structure and GPU context for speed.
///
/// `cage_meshes`, when supplied, must be parallel to `low_poly_meshes`
/// (one cage mesh per low-poly mesh, sharing topology). Cage drives the
/// projection rays so thin geometry doesn't punch through the high-poly.
/// `None` (or fewer cages than low-poly meshes) falls back to expanding
/// along the low-poly normal.
pub fn bake_single_map_preview(
    low_poly_meshes: &[Mesh],
    merged_hp: Option<&Mesh>,
    cage_meshes: Option<&[Mesh]>,
    map_type: MapType,
    config: &BakeConfig,
) -> Result<PreviewResult, String> {
    let w = config.width;
    let h = config.height;
    let total = (w * h) as usize;

    // Compute tangents for the first low-poly mesh
    let tangent_data: Vec<_> = low_poly_meshes
        .iter()
        .map(tangent::compute_tangents)
        .collect();

    // Rasterize
    let raster_inputs: Vec<raster::RasterInput> = low_poly_meshes
        .iter()
        .zip(tangent_data.iter())
        .enumerate()
        .map(|(i, (m, t))| raster::RasterInput {
            mesh_index: i,
            mesh: m,
            tangent_data: t,
            cage: cage_meshes.and_then(|c| c.get(i)),
        })
        .collect();
    let grid = raster::rasterize_uv_space(&raster_inputs, w, h);
    let mask: Vec<bool> = grid.data.iter().map(|t| t.is_some()).collect();

    // Build BVH from merged HP or low-poly
    let bvh_meshes: &[Mesh] = if let Some(merged) = merged_hp {
        std::slice::from_ref(merged)
    } else {
        low_poly_meshes
    };
    let accel = AccelStructure::build(bvh_meshes);

    // Auto-bias
    let mut bbox_min = glam::Vec3::splat(f32::MAX);
    let mut bbox_max = glam::Vec3::splat(f32::MIN);
    for mesh in bvh_meshes {
        for pos in &mesh.positions {
            bbox_min = bbox_min.min(*pos);
            bbox_max = bbox_max.max(*pos);
        }
    }
    let diagonal = (bbox_max - bbox_min).length();
    let auto_bias = (diagonal * 0.0005).max(0.0001);

    // GPU init
    let gpu_ctx = if config.use_gpu {
        gpu::context::GpuContext::new().map(|ctx| {
            let flat_bvh = gpu::flat_bvh::FlatBvh::from_accel(&accel);
            (ctx, flat_bvh)
        })
    } else {
        None
    };

    // Bake the requested map
    match map_type {
        MapType::AO => {
            let mut settings = config.ao_settings.clone();
            settings.bias = auto_bias;
            let buffer = if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
                gpu::ao_baker::bake_ao_gpu(ctx, &grid.data, flat_bvh, &settings, false)
            } else {
                (0..total)
                    .into_par_iter()
                    .map(|idx| match &grid.data[idx] {
                        Some(texel) => ao::bake_ao_texel(texel, &accel, &settings, idx as u32),
                        None => 1.0,
                    })
                    .collect()
            };
            let mut buf = buffer;
            dilate::dilate_gray(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Gray(buf, w, h))
        }
        MapType::Curvature => {
            // Curvature needs world-space normals first
            let mut wn_buffer = vec![[0.5f32, 0.5, 1.0]; total];
            wn_buffer
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, pixel)| {
                    if let Some(texel) = &grid.data[idx] {
                        *pixel = normal::bake_world_normal_texel(texel, None);
                    }
                });
            let buffer = curvature::compute_curvature_from_normals(
                &wn_buffer,
                &mask,
                w,
                h,
                &config.curvature_settings,
            );
            let mut buf = buffer;
            dilate::dilate_gray(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Gray(buf, w, h))
        }
        MapType::Thickness => {
            let mut settings = config.thickness_settings.clone();
            settings.bias = auto_bias;
            if settings.max_distance > diagonal {
                settings.max_distance = diagonal * 0.1;
            }
            let buffer = if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
                gpu::ao_baker::bake_ao_gpu(ctx, &grid.data, flat_bvh, &settings, true)
            } else {
                (0..total)
                    .into_par_iter()
                    .map(|idx| match &grid.data[idx] {
                        Some(texel) => {
                            ao::bake_thickness_texel(texel, &accel, &settings, idx as u32)
                        }
                        None => 1.0,
                    })
                    .collect()
            };
            let mut buf = buffer;
            dilate::dilate_gray(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Gray(buf, w, h))
        }
        MapType::Normal => {
            if merged_hp.is_none() {
                return Err("Normal map requires high-poly".into());
            }
            let hp = merged_hp.unwrap();
            // Projection rays
            let hits: Vec<Option<crate::accel::HitRecord>> = (0..total)
                .into_par_iter()
                .map(|idx| {
                    let texel = match &grid.data[idx] {
                        Some(t) => t,
                        None => return None,
                    };
                    let origin = texel.position + texel.normal * config.max_frontal_distance;
                    let direction = -texel.normal;
                    let max_t = config.max_frontal_distance + config.max_rear_distance;
                    accel.trace_closest(origin, direction, max_t, auto_bias, config.ignore_backface)
                })
                .collect();

            let mut buffer = vec![[0.5f32, 0.5, 1.0]; total];
            buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                if let (Some(texel), Some(hit)) = (&grid.data[idx], &hits[idx]) {
                    *pixel = normal::bake_normal_texel(
                        texel,
                        hit,
                        std::slice::from_ref(hp),
                        config.normal_format,
                    );
                }
            });
            let mut buf = buffer;
            dilate::dilate_rgb(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Rgb(buf, w, h))
        }
        MapType::WorldNormal => {
            let mut buffer = vec![[0.5f32, 0.5, 1.0]; total];
            buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                if let Some(texel) = &grid.data[idx] {
                    *pixel = normal::bake_world_normal_texel(texel, None);
                }
            });
            let mut buf = buffer;
            dilate::dilate_rgb(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Rgb(buf, w, h))
        }
        MapType::Position => {
            let (bmin, bmax) = position::compute_texel_bounds(&grid.data);
            let norm = PositionNormalization::BoundingBox {
                min: bmin,
                max: bmax,
            };
            let mut buffer = vec![[0.0f32; 3]; total];
            buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                if let Some(texel) = &grid.data[idx] {
                    *pixel = position::bake_position_texel(texel, &norm);
                }
            });
            let mut buf = buffer;
            dilate::dilate_rgb(&mut buf, &mask, w, h, 0);
            Ok(PreviewResult::Rgb(buf, w, h))
        }
        _ => Err(format!("{:?} preview not yet supported", map_type)),
    }
}

/// Run the full bake pipeline.
/// `progress` is an optional callback called before each map bake starts.
pub fn bake(
    low_poly_meshes: &[Mesh],
    high_poly_meshes: &[Mesh],
    cage_meshes: Option<&[Mesh]>,
    config: &BakeConfig,
) -> Result<(), String> {
    bake_with_progress(
        low_poly_meshes,
        high_poly_meshes,
        cage_meshes,
        config,
        |_| {},
    )
}

/// Run the full bake pipeline with a progress callback.
pub fn bake_with_progress(
    low_poly_meshes: &[Mesh],
    high_poly_meshes: &[Mesh],
    cage_meshes: Option<&[Mesh]>,
    config: &BakeConfig,
    progress: impl Fn(&str),
) -> Result<(), String> {
    let aa = config.aa_factor.max(1);
    let out_w = config.width;
    let out_h = config.height;
    // Internal bake resolution (upscaled for AA)
    let w = out_w * aa;
    let h = out_h * aa;
    let total = (w * h) as usize;

    if aa > 1 {
        log::info!("Supersampling {}x{} (internal {}x{})", aa, aa, w, h);
    }

    // 1. Compute tangent basis for all low-poly meshes
    log::info!(
        "Computing MikkTSpace tangents for {} low-poly mesh(es)...",
        low_poly_meshes.len()
    );
    let tangent_data: Vec<_> = low_poly_meshes
        .iter()
        .map(tangent::compute_tangents)
        .collect();

    // Validate cage meshes if provided
    if let Some(cages) = cage_meshes {
        if cages.len() != low_poly_meshes.len() {
            return Err(format!(
                "Cage mesh count ({}) must match low-poly mesh count ({})",
                cages.len(),
                low_poly_meshes.len()
            ));
        }
        for (i, (lp, cage)) in low_poly_meshes.iter().zip(cages.iter()).enumerate() {
            if lp.positions.len() != cage.positions.len() {
                return Err(format!(
                    "Cage mesh {} ('{}') has {} vertices but low-poly ('{}') has {} — must match",
                    i,
                    cage.name,
                    cage.positions.len(),
                    lp.name,
                    lp.positions.len()
                ));
            }
        }
        log::info!("Using custom cage mesh(es) for projection");
    }

    // 2. Rasterize low-poly into UV space
    log::info!("Rasterizing low-poly into {}x{} texture space...", w, h);
    let raster_inputs: Vec<raster::RasterInput> = low_poly_meshes
        .iter()
        .zip(tangent_data.iter())
        .enumerate()
        .map(|(i, (m, t))| raster::RasterInput {
            mesh_index: i,
            mesh: m,
            tangent_data: t,
            cage: cage_meshes.and_then(|c| c.get(i)),
        })
        .collect();
    let grid = raster::rasterize_uv_space(&raster_inputs, w, h);

    // Compute validity mask
    let mask: Vec<bool> = grid.data.iter().map(|t| t.is_some()).collect();
    let valid_count = mask.iter().filter(|&&v| v).count();
    log::info!(
        "{} texels covered ({:.1}% of texture)",
        valid_count,
        valid_count as f64 / total as f64 * 100.0
    );

    // 3. Merge and build BVH for high-poly
    // We merge all high-poly meshes into one so that GPU projection's global
    // tri_index maps directly to the merged mesh's triangle/vertex arrays.
    let has_high_poly = !high_poly_meshes.is_empty();
    let merged_hp = if has_high_poly {
        let hp = if config.match_by_name {
            filter_matched_meshes(
                high_poly_meshes,
                low_poly_meshes,
                &config.low_suffix,
                &config.high_suffix,
            )
        } else {
            high_poly_meshes.to_vec()
        };
        log::info!("Merging {} high-poly mesh(es)...", hp.len());
        Some(Mesh::merge(&hp))
    } else {
        None
    };

    let accel = if let Some(ref merged) = merged_hp {
        log::info!("Building BVH ({} triangles)...", merged.triangle_count());
        Some(AccelStructure::build(std::slice::from_ref(merged)))
    } else {
        // For self-baking (AO, curvature from low-poly)
        log::info!("No high-poly provided, using low-poly for self-baking...");
        Some(AccelStructure::build(low_poly_meshes))
    };

    let accel_ref = accel.as_ref().unwrap();

    // All ray-based bakers (AO, thickness, bent normals) use the high-poly BVH
    // when available. Thickness uses a short max_distance (auto-capped to 10% of
    // diagonal) so it only detects nearby opposing surfaces, not the far wall.
    let thickness_accel = accel_ref;

    // Compute bounding box diagonal (used for auto-bias on all ray settings)
    let all_meshes: &[Mesh] = if let Some(ref merged) = merged_hp {
        std::slice::from_ref(merged)
    } else {
        low_poly_meshes
    };
    let mut bbox_min = glam::Vec3::splat(f32::MAX);
    let mut bbox_max = glam::Vec3::splat(f32::MIN);
    for mesh in all_meshes {
        for pos in &mesh.positions {
            bbox_min = bbox_min.min(*pos);
            bbox_max = bbox_max.max(*pos);
        }
    }
    let diagonal = (bbox_max - bbox_min).length();

    // Helper: auto-bias a RaySettings independently
    let auto_bias_settings = |mut s: RaySettings, label: &str, cap_distance: bool| -> RaySettings {
        if s.bias <= 0.001 {
            s.bias = (diagonal * 0.0005).max(0.0001);
            if cap_distance && s.max_distance > diagonal {
                s.max_distance = diagonal * 0.1;
            }
            log::info!(
                "Auto {} bias: {:.6} (mesh diagonal: {:.3})",
                label,
                s.bias,
                diagonal
            );
        }
        s
    };

    // Resolve curvature_intensity into curvature_settings (backward compat)
    let curvature_settings = {
        let mut cs = config.curvature_settings.clone();
        // If the user set curvature_intensity explicitly (i.e. it differs from default 1.0
        // while curvature_settings.intensity is still 1.0), honour the legacy field.
        if (config.curvature_intensity - 1.0).abs() > f32::EPSILON
            && (cs.intensity - 1.0).abs() < f32::EPSILON
        {
            cs.intensity = config.curvature_intensity;
        }
        cs
    };

    // Auto-bias each ray-based baker independently
    let ao_settings = auto_bias_settings(config.ao_settings.clone(), "AO", false);
    let thickness_settings =
        auto_bias_settings(config.thickness_settings.clone(), "thickness", true);
    let bent_normal_settings =
        auto_bias_settings(config.bent_normal_settings.clone(), "bent-normal", false);

    // Initialize GPU if requested
    // We keep two flat BVHs: one for projection (high-poly) and one for
    // All ray-based operations use the same BVH (high-poly when available).
    let gpu_ctx = if config.use_gpu {
        log::info!("Initializing GPU...");
        match gpu::context::GpuContext::new() {
            Some(ctx) => {
                let flat_bvh = gpu::flat_bvh::FlatBvh::from_accel(accel_ref);
                log::info!(
                    "  GPU BVH: {} nodes, {} tris",
                    flat_bvh.nodes.len(),
                    flat_bvh.triangles.len()
                );
                Some((ctx, flat_bvh))
            }
            None => {
                log::warn!("No GPU available, falling back to CPU");
                None
            }
        }
    } else {
        None
    };

    let out_dir = Path::new(&config.output_dir);
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("Failed to create output directory: {e}"))?;

    // 4. Per-texel ray casting (for maps that need high-poly projection)
    let needs_projection =
        has_high_poly && (config.maps.normal || config.maps.world_normal || config.maps.height);

    let has_cage = cage_meshes.is_some();
    let hits: Vec<Option<crate::accel::HitRecord>> = if needs_projection {
        // Try GPU projection (only for non-cage, simple normal-based projection)
        if !has_cage {
            if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
                log::info!("Casting projection rays on GPU...");
                gpu::projection::project_rays_gpu(
                    ctx,
                    &grid.data,
                    flat_bvh,
                    config.max_frontal_distance,
                    config.max_rear_distance,
                    config.ignore_backface,
                    ao_settings.bias,
                )
            } else {
                log::info!("Casting projection rays (CPU)...");
                project_rays_cpu(&grid.data, total, accel_ref, config, ao_settings.bias)
            }
        } else {
            log::info!("Casting projection rays (cage-based, CPU)...");
            project_rays_cpu(&grid.data, total, accel_ref, config, ao_settings.bias)
        }
    } else {
        vec![None; total]
    };

    // Extract GPU context ref for use in closures
    let gpu_ctx_ref = gpu_ctx.as_ref().map(|(ctx, _)| ctx);

    // --- Helper closures for AA-aware output ---
    // Downsample + dilate + write for RGB maps
    type RgbWriteFn = fn(&[[f32; 3]], u32, u32, &std::path::Path) -> Result<(), String>;
    let write_rgb = |buffer: &[[f32; 3]],
                     name: &str,
                     ext: &str,
                     dil: u32,
                     write_fn: RgbWriteFn|
     -> Result<(), String> {
        let (mut final_buf, final_mask) = if aa > 1 {
            (
                supersample::downsample_rgb(buffer, w, h, aa),
                supersample::downsample_mask(&mask, w, h, aa),
            )
        } else {
            (buffer.to_vec(), mask.clone())
        };
        // Use GPU JFA dilation if available and dilation is infinite (0)
        if let (Some(ctx), 0) = (gpu_ctx_ref, dil) {
            gpu::jfa_dilate::dilate_rgb_gpu(ctx, &mut final_buf, &final_mask, out_w, out_h);
        } else {
            dilate::dilate_rgb(&mut final_buf, &final_mask, out_w, out_h, dil);
        }
        let path = out_dir.join(format!("{}_{}.{}", config.output_prefix, name, ext));
        write_fn(&final_buf, out_w, out_h, &path)?;
        log::info!("  -> {}", path.display());
        Ok(())
    };

    // Downsample + dilate + write for grayscale maps
    let write_gray = |buffer: &[f32], name: &str, dil: u32| -> Result<(), String> {
        let (mut final_buf, final_mask) = if aa > 1 {
            (
                supersample::downsample_gray(buffer, w, h, aa),
                supersample::downsample_mask(&mask, w, h, aa),
            )
        } else {
            (buffer.to_vec(), mask.clone())
        };
        if let (Some(ctx), 0) = (gpu_ctx_ref, dil) {
            gpu::jfa_dilate::dilate_gray_gpu(ctx, &mut final_buf, &final_mask, out_w, out_h);
        } else {
            dilate::dilate_gray(&mut final_buf, &final_mask, out_w, out_h, dil);
        }
        let path = out_dir.join(format!("{}_{}.png", config.output_prefix, name));
        output::write_gray_png(&final_buf, out_w, out_h, &path)?;
        log::info!("  -> {}", path.display());
        Ok(())
    };

    // 5. Bake each requested map
    // --- Normal Map ---
    if config.maps.normal && has_high_poly {
        log::info!("Baking normal map (tangent space)...");
        progress("Baking normal map...");
        let mut buffer = vec![[0.5f32, 0.5, 1.0]; total];

        buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            if let (Some(texel), Some(hit)) = (&grid.data[idx], &hits[idx]) {
                let hp = merged_hp.as_ref().unwrap();
                *pixel = normal::bake_normal_texel(
                    texel,
                    hit,
                    std::slice::from_ref(hp),
                    config.normal_format,
                );
            }
        });

        write_rgb(
            &buffer,
            "normal",
            "png",
            config.dilation,
            output::write_rgb_png,
        )?;
    }

    // --- World Space Normal ---
    let world_normal_buffer = if config.maps.world_normal || config.maps.curvature {
        log::info!("Baking world-space normals...");
        progress("Baking world-space normals...");
        let mut buffer = vec![[0.5f32, 0.5, 1.0]; total];

        buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            if let Some(texel) = &grid.data[idx] {
                let hit_ref = if has_high_poly {
                    hits[idx]
                        .as_ref()
                        .map(|h| (h, std::slice::from_ref(merged_hp.as_ref().unwrap())))
                } else {
                    None
                };
                *pixel = normal::bake_world_normal_texel(texel, hit_ref);
            }
        });

        if config.maps.world_normal {
            write_rgb(
                &buffer,
                "world_normal",
                "png",
                config.dilation,
                output::write_rgb_png,
            )?;
        }

        Some(buffer)
    } else {
        None
    };

    // --- Ambient Occlusion ---
    if config.maps.ao {
        let buffer = if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
            log::info!(
                "Baking ambient occlusion on GPU ({} rays per texel)...",
                ao_settings.ray_count
            );
            progress("Baking ambient occlusion...");
            gpu::ao_baker::bake_ao_gpu(ctx, &grid.data, flat_bvh, &ao_settings, false)
        } else {
            log::info!(
                "Baking ambient occlusion ({} rays per texel)...",
                ao_settings.ray_count
            );
            (0..total)
                .into_par_iter()
                .map(|idx| match &grid.data[idx] {
                    Some(texel) => ao::bake_ao_texel(texel, accel_ref, &ao_settings, idx as u32),
                    None => 1.0,
                })
                .collect()
        };

        write_gray(&buffer, "ao", config.dilation)?;
    }

    // --- Curvature ---
    if config.maps.curvature {
        log::info!("Baking curvature...");
        progress("Baking curvature...");
        let wn = world_normal_buffer
            .as_ref()
            .expect("World-space normals needed for curvature");
        let buffer =
            curvature::compute_curvature_from_normals(wn, &mask, w, h, &curvature_settings);

        write_gray(&buffer, "curvature", config.dilation)?;
    }

    // --- Position ---
    if config.maps.position {
        log::info!("Baking position map...");
        progress("Baking position map...");
        let (bbox_min, bbox_max) = position::compute_texel_bounds(&grid.data);
        let norm = PositionNormalization::BoundingBox {
            min: bbox_min,
            max: bbox_max,
        };

        let mut buffer = vec![[0.0f32; 3]; total];
        buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            if let Some(texel) = &grid.data[idx] {
                *pixel = position::bake_position_texel(texel, &norm);
            }
        });

        write_rgb(
            &buffer,
            "position",
            "exr",
            config.dilation,
            output::write_rgb_exr,
        )?;
    }

    // --- Thickness ---
    if config.maps.thickness {
        let buffer = if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
            log::info!(
                "Baking thickness on GPU ({} rays per texel)...",
                thickness_settings.ray_count
            );
            progress("Baking thickness...");
            gpu::ao_baker::bake_ao_gpu(ctx, &grid.data, flat_bvh, &thickness_settings, true)
        } else {
            log::info!(
                "Baking thickness ({} rays per texel)...",
                thickness_settings.ray_count
            );
            (0..total)
                .into_par_iter()
                .map(|idx| match &grid.data[idx] {
                    Some(texel) => ao::bake_thickness_texel(
                        texel,
                        thickness_accel,
                        &thickness_settings,
                        idx as u32,
                    ),
                    None => 1.0,
                })
                .collect()
        };

        write_gray(&buffer, "thickness", config.dilation)?;
    }

    // --- Height ---
    if config.maps.height && has_high_poly {
        log::info!("Baking height map...");
        progress("Baking height map...");
        let mut raw_heights: Vec<Option<f32>> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let texel = grid.data[idx].as_ref()?;
                let hit = hits[idx].as_ref()?;
                let origin = texel
                    .cage_position
                    .unwrap_or(texel.position + texel.normal * config.max_frontal_distance);
                Some(height::bake_height_texel(texel, hit, origin))
            })
            .collect();

        height::normalize_height_map(&mut raw_heights);

        let buffer: Vec<f32> = raw_heights.iter().map(|h| h.unwrap_or(0.5)).collect();
        write_gray(&buffer, "height", config.dilation)?;
    }

    // --- Bent Normals ---
    if config.maps.bent_normals {
        let buffer = if let Some((ref ctx, ref flat_bvh)) = gpu_ctx {
            log::info!(
                "Baking bent normals on GPU ({} rays per texel)...",
                bent_normal_settings.ray_count
            );
            progress("Baking bent normals...");
            gpu::bent_normals::bake_bent_normals_gpu(
                ctx,
                &grid.data,
                flat_bvh,
                &bent_normal_settings,
            )
        } else {
            log::info!(
                "Baking bent normals ({} rays per texel)...",
                bent_normal_settings.ray_count
            );
            (0..total)
                .into_par_iter()
                .map(|idx| match &grid.data[idx] {
                    Some(texel) => ao::bake_bent_normal_texel(
                        texel,
                        accel_ref,
                        &bent_normal_settings,
                        idx as u32,
                    ),
                    None => [0.5, 0.5, 1.0],
                })
                .collect()
        };

        write_rgb(
            &buffer,
            "bent_normals",
            "png",
            config.dilation,
            output::write_rgb_png,
        )?;
    }

    // --- ID Map ---
    if config.maps.id {
        log::info!("Baking ID map...");
        progress("Baking ID map...");
        let mut buffer = vec![[0.0f32; 3]; total];

        buffer.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
            if let Some(hit) = &hits[idx] {
                *pixel = id::bake_id_texel(hit, config.id_source);
            } else if let Some(texel) = &grid.data[idx] {
                *pixel =
                    id::bake_id_from_lowpoly(texel.mesh_index, texel.tri_index, config.id_source);
            }
        });

        // Minimal dilation for ID maps (crisp edges)
        dilate::dilate_rgb(&mut buffer, &mask, w, h, 4);
        let final_buf = if aa > 1 {
            supersample::downsample_rgb(&buffer, w, h, aa)
        } else {
            buffer
        };
        let path = out_dir.join(format!("{}_id.png", config.output_prefix));
        output::write_rgb_png(&final_buf, out_w, out_h, &path)?;
        log::info!("  -> {}", path.display());
    }

    log::info!("Baking complete.");
    Ok(())
}

/// CPU fallback for projection ray casting.
fn project_rays_cpu(
    data: &[Option<crate::raster::TexelData>],
    total: usize,
    accel: &AccelStructure,
    config: &BakeConfig,
    min_t: f32,
) -> Vec<Option<crate::accel::HitRecord>> {
    (0..total)
        .into_par_iter()
        .map(|idx| {
            let texel = match &data[idx] {
                Some(t) => t,
                None => return None,
            };

            let (origin, direction, max_t) = if let (Some(cage_pos), Some(cage_dir)) =
                (texel.cage_position, texel.cage_direction)
            {
                let dist = (cage_pos - texel.position).length();
                (cage_pos, cage_dir, dist + config.max_rear_distance)
            } else {
                let origin = texel.position + texel.normal * config.max_frontal_distance;
                let direction = -texel.normal;
                let max_t = config.max_frontal_distance + config.max_rear_distance;
                (origin, direction, max_t)
            };

            accel.trace_closest(origin, direction, max_t, min_t, config.ignore_backface)
        })
        .collect()
}

/// Filter high-poly meshes to those matching low-poly meshes by name suffix.
fn filter_matched_meshes(
    high_poly: &[Mesh],
    low_poly: &[Mesh],
    low_suffix: &str,
    high_suffix: &str,
) -> Vec<Mesh> {
    let low_base_names: Vec<String> = low_poly
        .iter()
        .map(|m| {
            m.name
                .strip_suffix(low_suffix)
                .unwrap_or(&m.name)
                .to_string()
        })
        .collect();

    high_poly
        .iter()
        .filter(|hp| {
            let hp_base = hp.name.strip_suffix(high_suffix).unwrap_or(&hp.name);
            low_base_names.iter().any(|lb| hp_base.starts_with(lb))
        })
        .cloned()
        .collect()
}
