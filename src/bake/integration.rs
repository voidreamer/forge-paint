//! UDIM-aware adapter for the vendored `texture_baker` crate.
//!
//! `texture_baker` is single-tile: UVs are assumed to live in `[0, 1]`
//! and rasterisation runs against one square. forge-paint is UDIM-native
//! — paint targets are `texture_2d_array`s with one layer per tile.
//!
//! The bridge: split the mesh per UDIM tile (triangles are bucketed by
//! their centroid's tile id, UVs remapped to `[0, 1]` inside that tile),
//! call `bake_single_map_preview` once per tile, stage each tile's pixel
//! buffer into the matching layer of a freshly created texture array.

use egui_wgpu::wgpu;
use glam::Vec3;

use texture_baker::baker::{BakeConfig, BakeRequest, MapType, PreviewResult};
use texture_baker::bakers::ao::{Distribution, RaySettings};
use texture_baker::bakers::curvature::CurvatureSettings;
use texture_baker::bakers::id::IdSource;
use texture_baker::bakers::normal::NormalMapFormat;
use texture_baker::mesh::Mesh as BakerMesh;

use crate::mesh::CpuMesh;
use crate::paint::udim;

/// Paint-target-shaped output for a baked map. The texture is a
/// `texture_2d_array` with one layer per tile in `tiles`, matching the
/// layout of `PaintTarget`'s atlas textures so the PBR shader can index
/// it with the same math.
pub struct BakedMap {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub kind: MapKind,
    pub resolution: u32,
    pub tile_count: u32,
}

/// Which map to bake. Maps onto `texture_baker::baker::MapType` plus a
/// hint for the GPU storage format we'll allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    AmbientOcclusion,
    Curvature,
    Thickness,
    Height,
    Normal,
    WorldNormal,
    BentNormal,
    Position,
    Id,
}

impl MapKind {
    pub fn label(self) -> &'static str {
        match self {
            MapKind::AmbientOcclusion => "Ambient occlusion",
            MapKind::Curvature => "Curvature",
            MapKind::Thickness => "Thickness",
            MapKind::Height => "Height",
            MapKind::Normal => "Normal (tangent)",
            MapKind::WorldNormal => "World normal",
            MapKind::BentNormal => "Bent normal",
            MapKind::Position => "World position",
            MapKind::Id => "Id",
        }
    }

    fn map_type(self) -> MapType {
        match self {
            MapKind::AmbientOcclusion => MapType::AO,
            MapKind::Curvature => MapType::Curvature,
            MapKind::Thickness => MapType::Thickness,
            MapKind::Height => MapType::Height,
            MapKind::Normal => MapType::Normal,
            MapKind::WorldNormal => MapType::WorldNormal,
            MapKind::BentNormal => MapType::BentNormals,
            MapKind::Position => MapType::Position,
            MapKind::Id => MapType::Id,
        }
    }

    /// GPU format for the layered output. Scalar maps (AO / curvature /
    /// thickness / height) collapse to R8Unorm — that matches how
    /// roughness / metallic atlases are stored on the paint side. Three-
    /// channel maps stay at Rgba8Unorm; position is HDR (Rgba16Float)
    /// because world coords routinely exceed [0, 1].
    fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            MapKind::AmbientOcclusion
            | MapKind::Curvature
            | MapKind::Thickness
            | MapKind::Height => wgpu::TextureFormat::R8Unorm,
            MapKind::Position => wgpu::TextureFormat::Rgba16Float,
            MapKind::Normal | MapKind::WorldNormal | MapKind::BentNormal | MapKind::Id => {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
    }
}

/// Knobs the orchestrator forwards to `texture_baker`. Defaults match
/// the standalone CLI's defaults so headless and in-app behaviour stays
/// the same. Per-map ray counts are exposed because AO / thickness /
/// bent normal are the slow paths most users actually want to tune.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BakeSettings {
    pub ao_rays: u32,
    pub thickness_rays: u32,
    pub bent_rays: u32,
    pub spread_angle_deg: f32,
    pub max_distance: f32,
    pub aa_factor: u32,
    pub use_gpu: bool,
    pub normal_format: NormalConvention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormalConvention {
    /// Y+ green (Substance / Marmoset / glTF default).
    DirectX,
    /// Y- green.
    OpenGL,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            ao_rays: 64,
            thickness_rays: 64,
            bent_rays: 64,
            spread_angle_deg: 180.0,
            max_distance: 0.0, // 0 → unlimited (texture-baker uses f32::MAX)
            aa_factor: 1,
            use_gpu: true,
            normal_format: NormalConvention::DirectX,
        }
    }
}

/// Load + merge a high-poly source from disk. The result is the input
/// to texture-baker's BVH; topology / UVs of the HP don't matter to
/// downstream baking (only its positions / triangles), so we collapse
/// every part of the file into one mesh.
pub fn load_high_poly(path: &std::path::Path) -> Result<BakerMesh, String> {
    let parts = BakerMesh::load(path).map_err(|e| format!("load {}: {e}", path.display()))?;
    if parts.is_empty() {
        return Err(format!("{}: no meshes found", path.display()));
    }
    Ok(BakerMesh::merge(&parts))
}

/// Load a cage from disk and convert into a forge-paint `CpuMesh`. The
/// caller is responsible for verifying that the cage shares the active
/// low-poly's vertex count (required for per-vertex offset baking).
pub fn load_cage(path: &std::path::Path) -> Result<CpuMesh, String> {
    let parts = BakerMesh::load(path).map_err(|e| format!("load {}: {e}", path.display()))?;
    if parts.is_empty() {
        return Err(format!("{}: no meshes found", path.display()));
    }
    let merged = BakerMesh::merge(&parts);
    Ok(CpuMesh {
        positions: merged.positions,
        normals: merged.normals,
        uvs: merged
            .uvs
            .into_iter()
            .map(|uv| glam::Vec2::new(uv[0], uv[1]))
            .collect(),
        indices: merged.indices,
        prim_ranges: Vec::new(),
    })
}

/// Convert a forge-paint mesh into a single texture-baker mesh. UVs
/// stay in their UDIM-tiled space; the per-tile splitter below handles
/// the [0, 1] remap.
#[allow(dead_code)]
fn to_baker_mesh(name: &str, mesh: &CpuMesh) -> BakerMesh {
    BakerMesh {
        name: name.to_string(),
        positions: mesh.positions.clone(),
        normals: mesh.normals.clone(),
        uvs: mesh.uvs.iter().map(|uv| [uv.x, uv.y]).collect(),
        indices: mesh.indices.clone(),
    }
}

/// Build a per-tile mesh + optional matching cage. Triangles are kept
/// when their UV centroid lands in `tile_id`; UVs get remapped into
/// `[0, 1]` so texture-baker rasterises them as a single tile.
///
/// The cage (if supplied) must share the low-poly's vertex count and
/// index layout — texture-baker treats it as a per-vertex displacement
/// of the low-poly. We reuse the same remap table when collapsing the
/// per-tile vertex list so the resulting cage mesh stays in lockstep
/// with the low-poly tile mesh.
///
/// Returns `None` if no triangles land in this tile.
fn extract_tile_mesh(
    name: &str,
    mesh: &CpuMesh,
    cage: Option<&CpuMesh>,
    tile_id: u32,
) -> Option<(BakerMesh, Option<BakerMesh>)> {
    // Tile origin in UV space. tile 1001 = (0,0), 1002 = (1,0), 1011 = (0,1).
    let n = tile_id.saturating_sub(1001);
    let tile_u = (n % 10) as f32;
    let tile_v = (n / 10) as f32;

    let mut new_indices: Vec<[u32; 3]> = Vec::new();
    // We rebuild positions/normals/uvs so the per-tile mesh stays
    // compact and texture-baker's BVH only sees what it needs. Map
    // original-vertex → new-vertex on first sighting.
    let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut cage_positions: Vec<Vec3> = Vec::new();
    let mut cage_normals: Vec<Vec3> = Vec::new();

    // Cage must share vertex count with the low-poly. If the user
    // supplied a mismatched cage we silently fall back to no cage —
    // safer than panicking mid-bake.
    let cage_ok = match cage {
        Some(c) => c.positions.len() == mesh.positions.len(),
        None => false,
    };

    for tri in &mesh.indices {
        let uv0 = mesh.uvs[tri[0] as usize];
        let uv1 = mesh.uvs[tri[1] as usize];
        let uv2 = mesh.uvs[tri[2] as usize];
        let cu = (uv0.x + uv1.x + uv2.x) / 3.0;
        let cv = (uv0.y + uv1.y + uv2.y) / 3.0;
        if udim::tile_id([cu, cv]) != tile_id {
            continue;
        }

        let mut new_tri = [0u32; 3];
        for (slot, &orig) in tri.iter().enumerate() {
            let new_idx = if let Some(&i) = remap.get(&orig) {
                i
            } else {
                let i = positions.len() as u32;
                positions.push(mesh.positions[orig as usize]);
                normals.push(mesh.normals[orig as usize]);
                let uv = mesh.uvs[orig as usize];
                uvs.push([uv.x - tile_u, uv.y - tile_v]);
                if cage_ok {
                    let c = cage.unwrap();
                    cage_positions.push(c.positions[orig as usize]);
                    cage_normals.push(c.normals[orig as usize]);
                }
                remap.insert(orig, i);
                i
            };
            new_tri[slot] = new_idx;
        }
        new_indices.push(new_tri);
    }

    if new_indices.is_empty() {
        return None;
    }

    let lp = BakerMesh {
        name: format!("{name}.tile{tile_id}"),
        positions,
        normals,
        uvs: uvs.clone(),
        indices: new_indices.clone(),
    };
    let cage_mesh = if cage_ok {
        Some(BakerMesh {
            name: format!("{name}.cage.tile{tile_id}"),
            positions: cage_positions,
            normals: cage_normals,
            uvs,
            indices: new_indices,
        })
    } else {
        None
    };
    Some((lp, cage_mesh))
}

fn make_config(resolution: u32, settings: &BakeSettings) -> BakeConfig {
    let max_dist = if settings.max_distance <= 0.0 {
        f32::MAX
    } else {
        settings.max_distance
    };
    BakeConfig {
        width: resolution,
        height: resolution,
        maps: BakeRequest::default(),
        normal_format: match settings.normal_format {
            NormalConvention::DirectX => NormalMapFormat::DirectX,
            NormalConvention::OpenGL => NormalMapFormat::OpenGL,
        },
        ao_settings: RaySettings {
            ray_count: settings.ao_rays,
            max_distance: max_dist,
            spread_angle: settings.spread_angle_deg,
            distribution: Distribution::Cosine,
            ..RaySettings::default()
        },
        thickness_settings: RaySettings {
            ray_count: settings.thickness_rays,
            max_distance: max_dist,
            spread_angle: settings.spread_angle_deg,
            ..RaySettings::default()
        },
        bent_normal_settings: RaySettings {
            ray_count: settings.bent_rays,
            max_distance: max_dist,
            ..RaySettings::default()
        },
        max_frontal_distance: 0.5,
        max_rear_distance: 0.5,
        ignore_backface: true,
        dilation: 0,
        curvature_settings: CurvatureSettings::default(),
        curvature_intensity: 1.0,
        id_source: IdSource::MeshId,
        output_dir: String::new(),
        output_prefix: String::new(),
        match_by_name: false,
        low_suffix: String::new(),
        high_suffix: String::new(),
        aa_factor: settings.aa_factor,
        use_gpu: settings.use_gpu,
    }
}

/// V-flip a row-major byte buffer in place. forge-paint's texture
/// convention is V=0 at the *bottom* of the array (matches glTF / our
/// PNG import flow in `assets.rs`); texture-baker's rasteriser writes
/// V=0 at the bottom *row of the visible image*, which puts it at
/// array[H-1] in memory. Without this flip the AO bake renders upside
/// down on the model.
fn flip_rows(bytes: &mut [u8], width: u32, height: u32, bytes_per_pixel: u32) {
    let row = (width * bytes_per_pixel) as usize;
    let h = height as usize;
    for y in 0..h / 2 {
        let top = y * row;
        let bot = (h - 1 - y) * row;
        for i in 0..row {
            bytes.swap(top + i, bot + i);
        }
    }
}

/// Convert a `PreviewResult` into the byte buffer that goes into the
/// matching wgpu texture format. Output layout matches `MapKind::texture_format`.
fn pixels_to_bytes(kind: MapKind, preview: PreviewResult) -> (Vec<u8>, u32, u32) {
    match (kind, preview) {
        // Scalar → R8Unorm.
        (
            MapKind::AmbientOcclusion | MapKind::Curvature | MapKind::Thickness | MapKind::Height,
            PreviewResult::Gray(buf, w, h),
        ) => {
            let bytes: Vec<u8> = buf
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect();
            (bytes, w, h)
        }
        // RGB → Rgba8Unorm (alpha = 1).
        (
            MapKind::Normal | MapKind::WorldNormal | MapKind::BentNormal | MapKind::Id,
            PreviewResult::Rgb(buf, w, h),
        ) => {
            let mut bytes = Vec::with_capacity(buf.len() * 4);
            for px in &buf {
                bytes.push((px[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                bytes.push((px[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                bytes.push((px[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                bytes.push(255);
            }
            (bytes, w, h)
        }
        // Position → Rgba16Float, HDR.
        (MapKind::Position, PreviewResult::Rgb(buf, w, h)) => {
            let mut bytes = Vec::with_capacity(buf.len() * 8);
            for px in &buf {
                for &component in px {
                    bytes.extend_from_slice(&half::f16::from_f32(component).to_le_bytes());
                }
                // Alpha = 1.0 in f16.
                bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            }
            (bytes, w, h)
        }
        // Mismatched preview/kind shouldn't happen — texture-baker
        // returns a fixed shape per map type. Treat as empty.
        _ => (Vec::new(), 0, 0),
    }
}

/// Top-level entry: bake `kind` for the given mesh across every UDIM
/// tile in `tiles`, return a `texture_2d_array` ready to bind. Tiles
/// with no triangles become zero-initialised layers (alpha 0 / black).
///
/// `high_poly` and `cage` are optional. The high-poly is forwarded to
/// texture-baker as a single merged mesh — only the BVH is used, so
/// it's loaded once at the call site and shared across every tile bake.
/// The cage must share the low-poly's vertex count and index layout.
pub fn bake_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cpu_mesh: &CpuMesh,
    high_poly: Option<&BakerMesh>,
    cage: Option<&CpuMesh>,
    tiles: &[u32],
    resolution: u32,
    kind: MapKind,
    settings: &BakeSettings,
) -> Result<BakedMap, String> {
    let tile_count = tiles.len() as u32;
    if tile_count == 0 {
        return Err("no tiles to bake".into());
    }

    let format = kind.texture_format();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("mesh_maps.{:?}", kind)),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: tile_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("mesh_maps.{:?}.array_view", kind)),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let bytes_per_row = match format {
        wgpu::TextureFormat::R8Unorm => resolution,
        wgpu::TextureFormat::Rgba8Unorm => resolution * 4,
        wgpu::TextureFormat::Rgba16Float => resolution * 8,
        _ => unreachable!("MapKind only emits R8 / Rgba8 / Rgba16Float"),
    };

    let config = make_config(resolution, settings);
    let map_type = kind.map_type();

    // Bake one tile at a time. HP is shared across all tiles (BVH-only
    // input), cage gets per-tile-extracted alongside the low-poly.
    for (layer, &tile_id) in tiles.iter().enumerate() {
        let Some((tile_mesh, tile_cage)) = extract_tile_mesh("forge", cpu_mesh, cage, tile_id)
        else {
            log::info!("bake: tile {tile_id} (layer {layer}) has no triangles, skipping");
            continue;
        };
        let cage_slice = tile_cage.as_ref().map(std::slice::from_ref);
        let preview = texture_baker::baker::bake_single_map_preview(
            std::slice::from_ref(&tile_mesh),
            high_poly,
            cage_slice,
            map_type,
            &config,
        )?;

        let (mut bytes, w, h) = pixels_to_bytes(kind, preview);
        if bytes.is_empty() {
            log::warn!(
                "bake: kind {:?} produced no bytes for tile {tile_id}, skipping",
                kind
            );
            continue;
        }
        if w != resolution || h != resolution {
            return Err(format!(
                "bake: tile {tile_id} returned {w}x{h}, expected {resolution}x{resolution}"
            ));
        }

        // V-flip — see `flip_rows` doc comment.
        let bpp = match format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Rgba8Unorm => 4,
            wgpu::TextureFormat::Rgba16Float => 8,
            _ => unreachable!(),
        };
        flip_rows(&mut bytes, w, h, bpp);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(resolution),
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(BakedMap {
        texture,
        view,
        kind,
        resolution,
        tile_count,
    })
}

// Use sites in the GUI come in phase 3 (MeshMaps slots) and phase 4
// (panel UI). Until then the function is dead; the unused-import
// allowance keeps cargo quiet across the partial-rollout window.
#[allow(dead_code)]
const _: () = ();
