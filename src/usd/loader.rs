//! Stage loader. Uses `rust_usd` to open a USD stage in-process; the C++
//! ForgeResolver registered via PXR_PLUGINPATH_NAME handles `forge://`
//! URIs through USD's URI-scheme dispatch — rust-usd's own
//! `ForgeAwareResolver` becomes the no-op primary, our resolver wins for
//! `forge://`.

use anyhow::{Result, anyhow, bail};
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::path::Path;

use crate::mesh::CpuMesh;

#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub path: String,
    pub texture_paths: Vec<String>,
    pub uv_primvar_name: Option<String>,
    pub mesh: CpuMesh,
}

#[derive(Debug, Clone)]
pub struct MeshMaterialInfo {
    pub prim_path: String,
    pub texture_paths: Vec<String>,
    pub uv_primvar_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadedStage {
    pub mesh: CpuMesh,
    pub materials: Vec<MeshMaterialInfo>,
}

/// Load a USD stage by path or `forge://` URI. Returns one `LoadedMesh`
/// per `UsdGeomMesh` prim, with xforms baked into positions and UVs
/// unwelded to per-face-vert.
pub fn load_stage(path: &Path) -> Result<Vec<LoadedMesh>> {
    let stage = rust_usd::Stage::open(path)
        .map_err(|e| anyhow!("Stage::open({}) failed: {}", path.display(), e.what()))?;

    // Pick a default variant for every `VariantSet` that doesn't have
    // an authored selection. Nvidia Omniverse SimReady assets (and
    // anything else authored by tools that lean on global variant
    // fallbacks) leave variantSets unset; the meshes live behind
    // those variants, so the stage looks empty without this pass.
    // usdview / Houdini Solaris use `PcpVariantFallbackMap`'s global
    // fallbacks to the same effect — we use per-prim first-variant
    // because rust_usd already exposes that and it doesn't require
    // pre-Stage-Open plumbing.
    //
    // Run until stable: each variant selection can pull in new
    // references whose prims have their own variantSets, so we keep
    // walking until a pass selects nothing new (capped to keep
    // pathological cycles from spinning).
    let mut total_selected = 0;
    for _ in 0..8 {
        let n = auto_select_first_variants(&stage.pseudo_root());
        total_selected += n;
        if n == 0 {
            break;
        }
    }
    if total_selected > 0 {
        log::info!(
            "load_stage({}): auto-selected {} variant(s) on previously-unset variantSets",
            path.display(),
            total_selected
        );
    }

    let pxr_meshes = stage.meshes();
    if pxr_meshes.is_empty() {
        bail!("stage at {} contains no UsdGeomMesh prims", path.display());
    }

    let mut loaded = Vec::with_capacity(pxr_meshes.len());
    for m in pxr_meshes {
        let texture_paths = m.bound_texture_paths();
        let intermediate = build_intermediate(&m)?;
        let mesh = triangulate(&intermediate)?;
        loaded.push(LoadedMesh {
            path: intermediate.path,
            texture_paths,
            uv_primvar_name: intermediate.uv_primvar_name,
            mesh,
        });
    }
    Ok(loaded)
}

/// Walk the stage from `prim` down, set the first variant of any
/// `VariantSet` whose selection hasn't been authored yet. Returns the
/// number of selections written so the caller can loop until stable.
fn auto_select_first_variants(prim: &rust_usd::Prim) -> usize {
    let mut selected = 0;
    for vs_name in prim.variant_set_names() {
        if let Some(vs) = prim.variant_set(&vs_name) {
            if !vs.has_authored_selection() {
                let variants = vs.variants();
                if let Some(first) = variants.first() {
                    if vs.set_selection(first) {
                        selected += 1;
                        log::debug!(
                            "auto-selected variant `{first}` on variantSet `{vs_name}` of `{}`",
                            prim.path()
                        );
                    }
                }
            }
        }
    }
    // Re-fetch children AFTER our selections — picking a variant can
    // pull in new references whose child prims weren't visible to
    // the first traversal.
    for child in prim.children() {
        selected += auto_select_first_variants(&child);
    }
    selected
}

/// Convenience: load a stage and merge all mesh prims into one CpuMesh.
/// Keeps per-face-vert output; preserves UVs; concatenates index buffers.
pub fn load_stage_merged(path: &Path) -> Result<CpuMesh> {
    let loaded = load_stage(path)?;
    let count = loaded.len();
    let mut iter = loaded.into_iter();
    let LoadedMesh {
        path: first_path,
        texture_paths: _,
        uv_primvar_name: _,
        mesh: first,
    } = iter.next().unwrap();
    let mut out = first;
    let first_count = out.positions.len() as u32;
    out.prim_ranges.push(crate::mesh::PrimRange {
        prim_path: first_path,
        vert_start: 0,
        vert_count: first_count,
    });
    if count == 1 {
        return Ok(out);
    }

    for LoadedMesh {
        path: ppath,
        texture_paths: _,
        uv_primvar_name: _,
        mesh,
    } in iter
    {
        let offset = out.positions.len() as u32;
        let added = mesh.positions.len() as u32;
        out.positions.extend(mesh.positions);
        out.normals.extend(mesh.normals);
        out.uvs.extend(mesh.uvs);
        for tri in mesh.indices {
            out.indices
                .push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
        out.prim_ranges.push(crate::mesh::PrimRange {
            prim_path: ppath,
            vert_start: offset,
            vert_count: added,
        });
    }
    log::info!("merged {count} mesh prims into a single CpuMesh");
    Ok(out)
}

/// Load and merge a stage while preserving the bound texture references
/// discovered on each mesh prim. The merged CpuMesh remains the WGPU paint
/// surface; `materials` lets the app reconstruct assigned material graph nodes
/// and resolve embedded USDZ texture assets.
pub fn load_stage_merged_with_materials(path: &Path) -> Result<LoadedStage> {
    let loaded = load_stage(path)?;
    let materials = loaded
        .iter()
        .filter(|m| !m.texture_paths.is_empty())
        .map(|m| MeshMaterialInfo {
            prim_path: m.path.clone(),
            texture_paths: m.texture_paths.clone(),
            uv_primvar_name: m.uv_primvar_name.clone(),
        })
        .collect();
    let count = loaded.len();
    let mut iter = loaded.into_iter();
    let LoadedMesh {
        path: first_path,
        texture_paths: _,
        uv_primvar_name: _,
        mesh: first,
    } = iter.next().unwrap();
    let mut out = first;
    let first_count = out.positions.len() as u32;
    out.prim_ranges.push(crate::mesh::PrimRange {
        prim_path: first_path,
        vert_start: 0,
        vert_count: first_count,
    });

    for LoadedMesh {
        path: ppath,
        texture_paths: _,
        uv_primvar_name: _,
        mesh,
    } in iter
    {
        let offset = out.positions.len() as u32;
        let added = mesh.positions.len() as u32;
        out.positions.extend(mesh.positions);
        out.normals.extend(mesh.normals);
        out.uvs.extend(mesh.uvs);
        for tri in mesh.indices {
            out.indices
                .push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
        out.prim_ranges.push(crate::mesh::PrimRange {
            prim_path: ppath,
            vert_start: offset,
            vert_count: added,
        });
    }
    log::info!("merged {count} mesh prims into a single CpuMesh");
    Ok(LoadedStage {
        mesh: out,
        materials,
    })
}

// ---------------------------------------------------------------------------
// Intermediate representation + rust-usd → IR conversion
//
// We keep a small struct here rather than handing rust_usd::Mesh directly
// to triangulate() because:
//   * rust-usd flattens points/normals/uvs to Vec<f32>; chunking into
//     [f32; N] arrays once at the boundary keeps the inner loops clean.
//   * Interpolation comes through as a String in rust-usd; we convert
//     it to a Rust enum so the match arms in triangulate stay typed.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interpolation {
    Vertex,
    FaceVarying,
    Uniform,
    Constant,
    Unknown,
}

impl Interpolation {
    fn parse(s: &str) -> Self {
        match s {
            "vertex" => Self::Vertex,
            "faceVarying" => Self::FaceVarying,
            "uniform" => Self::Uniform,
            "constant" => Self::Constant,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
struct StPrimvar {
    name: String,
    data: Vec<[f32; 2]>,
    indices: Option<Vec<u32>>,
    interpolation: Interpolation,
}

#[derive(Debug, Clone)]
struct NormalPrimvar {
    data: Vec<[f32; 3]>,
    indices: Option<Vec<u32>>,
    interpolation: Interpolation,
}

#[derive(Debug, Clone)]
struct UsdMesh {
    path: String,
    points: Vec<[f32; 3]>,
    face_vertex_counts: Vec<u32>,
    face_vertex_indices: Vec<u32>,
    st: Option<StPrimvar>,
    uv_primvar_name: Option<String>,
    normals: Option<NormalPrimvar>,
    world_xform: Mat4,
    // UsdGeomMesh::orientation == "leftHanded" ⇒ face vertices wind CW.
    // Tracked so the triangulator can flip the emit order (otherwise
    // wgpu's `front_face = Ccw` rasterizer culls the visible side) and
    // so `compute_face_normal` knows the cross-product points INWARD
    // and needs negating to recover the outward-pointing normal.
    left_handed: bool,
}

fn build_intermediate(m: &rust_usd::Mesh) -> Result<UsdMesh> {
    let path = m.prim_path();

    // USD uses row-vector convention with row-major storage:
    // `v_world_row = v_local_row * M_USD`, translation lives in
    // `M_USD[3]` (row 3). rust-usd hands it back as `[[f32; 4]; 4]`
    // indexed `lt_w[row][col]` — so `lt_w[3] = [Tx, Ty, Tz, 1]`.
    //
    // glam uses column-vector convention with column-major storage:
    // `v_world_col = M_glam * v_local_col`, translation in
    // `M_glam.col(3)`. The semantic conversion from USD's row-vector
    // form to glam's column-vector form IS a transpose:
    //   M_glam = M_USD^T,  i.e.  M_glam.col(c) = M_USD.row(c)
    //
    // Since `lt_w[c]` is already a `[f32; 4]` carrying USD's row c,
    // feeding it straight into `from_cols_array_2d` does the right
    // thing — that constructor reads `data[c]` as glam's column c.
    //
    // Previously this code did a per-element shuffle
    // (`data[c] = [lt_w[0][c], lt_w[1][c], lt_w[2][c], lt_w[3][c]]`)
    // that's actually a transpose-of-a-transpose, equivalent to
    // skipping the convention conversion entirely. The result: any
    // mesh with a non-trivial xform (non-identity rotation /
    // non-zero translation) landed in the wrong world position, and
    // its normal matrix's rotation was effectively flipped, so the
    // normals on those meshes faced the wrong way. Hydra reads the
    // same stage through its own scene index — which DOES the
    // convention swap correctly — which is why the same asset
    // looked right on the Hydra side and wrong in wgpu.
    let lt_w = m.local_to_world();
    let world_xform = Mat4::from_cols_array_2d(&lt_w);

    let normals_data = m.normals_xyz();
    let normals = if normals_data.is_empty() {
        None
    } else {
        Some(NormalPrimvar {
            data: normals_data,
            // rust-usd's Mesh API doesn't surface normals_indices yet;
            // we never relied on them anyway. The dispatch in the old
            // .usda parser silently dropped `normals:indices`.
            indices: None,
            interpolation: Interpolation::parse(&m.normals_interpolation()),
        })
    };

    let st = choose_uv_primvar(m);
    let uv_primvar_name = st.as_ref().map(|pv| pv.name.clone());

    let left_handed = m.orientation() == "leftHanded";

    Ok(UsdMesh {
        path,
        points: m.points_xyz(),
        face_vertex_counts: m.face_vertex_counts_u32(),
        face_vertex_indices: m.face_vertex_indices_u32(),
        st,
        uv_primvar_name,
        normals,
        world_xform,
        left_handed,
    })
}

fn choose_uv_primvar(m: &rust_usd::Mesh) -> Option<StPrimvar> {
    let names = m.primvar_names();
    let mut candidates = Vec::new();
    for preferred in ["st", "st0"] {
        if names.iter().any(|name| name == preferred) {
            candidates.push(preferred.to_string());
        }
    }
    for name in names.iter().filter(|name| name.starts_with("st")) {
        if !candidates.iter().any(|candidate| candidate == name) {
            candidates.push(name.clone());
        }
    }
    for name in names.iter().filter(|name| {
        let lower = name.to_ascii_lowercase();
        lower.contains("uv") || lower.contains("texcoord")
    }) {
        if !candidates.iter().any(|candidate| candidate == name) {
            candidates.push(name.clone());
        }
    }
    for name in names {
        if !candidates.iter().any(|candidate| candidate == &name) {
            candidates.push(name);
        }
    }

    for name in candidates {
        let Some(pv) = m.primvar(&name) else {
            continue;
        };
        let raw = pv.as_vec2f_array();
        if raw.is_empty() {
            continue;
        }
        let data: Vec<[f32; 2]> = raw.chunks_exact(2).map(|uv| [uv[0], uv[1]]).collect();
        if data.is_empty() {
            continue;
        }
        let indices: Vec<u32> = pv
            .indices()
            .into_iter()
            .filter_map(|idx| (idx >= 0).then_some(idx as u32))
            .collect();
        let indices = if indices.is_empty() {
            None
        } else {
            Some(indices)
        };
        let interpolation = Interpolation::parse(&pv.interpolation());
        log::debug!("mesh {} using UV primvar `{name}`", m.prim_path());
        return Some(StPrimvar {
            name,
            data,
            indices,
            interpolation,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Triangulation (unchanged from the previous .usda-text-driven path —
// this is mesh logic, not USD I/O).
// ---------------------------------------------------------------------------

fn triangulate(m: &UsdMesh) -> Result<CpuMesh> {
    let total_fvi: u32 = m.face_vertex_counts.iter().sum();
    if total_fvi as usize != m.face_vertex_indices.len() {
        bail!(
            "mesh {}: faceVertexIndices length {} != sum of faceVertexCounts {}",
            m.path,
            m.face_vertex_indices.len(),
            total_fvi
        );
    }

    let normal_mat = normal_matrix_from_xform(&m.world_xform);

    // wgpu's pipeline culls back-faces using screen-space winding
    // (front_face defaults to Ccw). The mesh's effective winding in
    // world / screen space flips for either a leftHanded orientation
    // OR a mirror xform — XOR because two flips cancel out. When the
    // effective winding ends up CW we need to swap the emitted index
    // order so the rasterizer sees CCW and keeps the visible side.
    let xform_negative_det = Mat3::from_mat4(m.world_xform).determinant() < 0.0;
    let flip_winding = m.left_handed ^ xform_negative_det;

    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    let mut fv_cursor: usize = 0; // running index into face-varying streams

    for (face_i, &count) in m.face_vertex_counts.iter().enumerate() {
        let count = count as usize;
        if count < 3 {
            fv_cursor += count;
            continue; // skip degenerate
        }
        let off = fv_cursor;

        // Fan triangulate: (0, k, k+1) for k in 1..count-1
        let face_normal = compute_face_normal(m, off, count);

        // Pre-compute the 3 output verts for k=0 (the pivot) once per face
        let pivot_out = emit_face_vert(
            off,
            m,
            face_i,
            face_normal,
            &m.world_xform,
            &normal_mat,
            &mut positions,
            &mut normals,
            &mut uvs,
        );

        for k in 1..(count - 1) {
            let a_out = emit_face_vert(
                off + k,
                m,
                face_i,
                face_normal,
                &m.world_xform,
                &normal_mat,
                &mut positions,
                &mut normals,
                &mut uvs,
            );
            let b_out = emit_face_vert(
                off + k + 1,
                m,
                face_i,
                face_normal,
                &m.world_xform,
                &normal_mat,
                &mut positions,
                &mut normals,
                &mut uvs,
            );
            if flip_winding {
                indices.push([pivot_out, b_out, a_out]);
            } else {
                indices.push([pivot_out, a_out, b_out]);
            }
        }

        fv_cursor += count;
    }

    Ok(CpuMesh {
        positions,
        normals,
        uvs,
        indices,
        prim_ranges: Vec::new(),
    })
}

fn normal_matrix_from_xform(m: &Mat4) -> Mat3 {
    let m3 = Mat3::from_mat4(*m);
    let it = m3.inverse().transpose();
    // Inverse-transpose preserves perpendicularity through a
    // negative-determinant (mirror / odd-scale) xform but the
    // resulting normal flips to point INTO the surface in world
    // space. Negate so it stays outward — Hydra/Storm apply the same
    // correction when reading the stage, which is why mirrored
    // sub-meshes look right in the Hydra view but were wrong on the
    // wgpu side.
    if m3.determinant() < 0.0 { -it } else { it }
}

fn compute_face_normal(m: &UsdMesh, off: usize, count: usize) -> Vec3 {
    if count < 3 {
        return Vec3::Y;
    }
    let i0 = m.face_vertex_indices[off] as usize;
    let i1 = m.face_vertex_indices[off + 1] as usize;
    let i2 = m.face_vertex_indices[off + 2] as usize;
    let p0 = Vec3::from_array(m.points[i0]);
    let p1 = Vec3::from_array(m.points[i1]);
    let p2 = Vec3::from_array(m.points[i2]);
    let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
    // leftHanded winds CW from the outside, so the cross product
    // yields the inward normal. Flip so the per-face fallback (used
    // when no authored normals primvar exists) still points outward.
    if m.left_handed { -n } else { n }
}

#[allow(clippy::too_many_arguments)]
fn emit_face_vert(
    fv_index: usize,
    m: &UsdMesh,
    face_i: usize,
    face_normal_local: Vec3,
    world_xform: &Mat4,
    normal_mat: &Mat3,
    positions: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    uvs: &mut Vec<Vec2>,
) -> u32 {
    let point_idx = m.face_vertex_indices[fv_index] as usize;
    let pos_local = Vec3::from_array(m.points[point_idx]);
    let pos_world = world_xform.transform_point3(pos_local);

    let normal_local = resolve_normal(m, fv_index, point_idx, face_i, face_normal_local);
    let normal_world = (*normal_mat * normal_local).normalize_or_zero();

    let uv = resolve_uv(m.st.as_ref(), fv_index, point_idx);

    let out_idx = positions.len() as u32;
    positions.push(pos_world);
    normals.push(normal_world);
    uvs.push(uv);
    out_idx
}

fn resolve_normal(
    m: &UsdMesh,
    fv_index: usize,
    point_idx: usize,
    face_i: usize,
    face_normal_local: Vec3,
) -> Vec3 {
    let Some(n) = m.normals.as_ref() else {
        return face_normal_local;
    };
    let idx_opt = match n.interpolation {
        Interpolation::FaceVarying => {
            if let Some(idxs) = &n.indices {
                idxs.get(fv_index).copied().map(|i| i as usize)
            } else {
                Some(fv_index)
            }
        }
        Interpolation::Vertex => Some(point_idx),
        Interpolation::Uniform => Some(face_i),
        Interpolation::Constant => Some(0),
        Interpolation::Unknown => Some(point_idx), // best-guess
    };
    idx_opt
        .and_then(|i| n.data.get(i).copied())
        .map(Vec3::from_array)
        .unwrap_or(face_normal_local)
}

fn resolve_uv(st: Option<&StPrimvar>, fv_index: usize, point_idx: usize) -> Vec2 {
    let Some(st) = st else { return Vec2::ZERO };
    let idx = match st.interpolation {
        Interpolation::FaceVarying => {
            if let Some(idxs) = &st.indices {
                idxs.get(fv_index).copied().map(|i| i as usize)
            } else {
                Some(fv_index)
            }
        }
        Interpolation::Vertex => Some(point_idx),
        Interpolation::Uniform | Interpolation::Constant => Some(0),
        Interpolation::Unknown => Some(point_idx),
    };
    idx.and_then(|i| st.data.get(i).copied())
        .map(Vec2::from_array)
        .unwrap_or(Vec2::ZERO)
}
