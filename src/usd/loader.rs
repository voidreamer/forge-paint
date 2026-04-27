//! Stage loader. Uses `rust_usd` to open a USD stage in-process; the C++
//! ForgeResolver registered via PXR_PLUGINPATH_NAME handles `forge://`
//! URIs through USD's URI-scheme dispatch — rust-usd's own
//! `ForgeAwareResolver` becomes the no-op primary, our resolver wins for
//! `forge://`.

use anyhow::{anyhow, bail, Result};
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::path::Path;

use crate::mesh::CpuMesh;

#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub path: String,
    pub mesh: CpuMesh,
}

/// Load a USD stage by path or `forge://` URI. Returns one `LoadedMesh`
/// per `UsdGeomMesh` prim, with xforms baked into positions and UVs
/// unwelded to per-face-vert.
pub fn load_stage(path: &Path) -> Result<Vec<LoadedMesh>> {
    let stage = rust_usd::Stage::open(path)
        .map_err(|e| anyhow!("Stage::open({}) failed: {}", path.display(), e.what()))?;

    let pxr_meshes = stage.meshes();
    if pxr_meshes.is_empty() {
        bail!("stage at {} contains no UsdGeomMesh prims", path.display());
    }

    let mut loaded = Vec::with_capacity(pxr_meshes.len());
    for m in pxr_meshes {
        let intermediate = build_intermediate(&m)?;
        let mesh = triangulate(&intermediate)?;
        loaded.push(LoadedMesh {
            path: intermediate.path,
            mesh,
        });
    }
    Ok(loaded)
}

/// Convenience: load a stage and merge all mesh prims into one CpuMesh.
/// Keeps per-face-vert output; preserves UVs; concatenates index buffers.
pub fn load_stage_merged(path: &Path) -> Result<CpuMesh> {
    let loaded = load_stage(path)?;
    let count = loaded.len();
    let mut iter = loaded.into_iter();
    let first = iter.next().unwrap().mesh;
    if count == 1 {
        return Ok(first);
    }

    let mut out = first;
    for LoadedMesh { mesh, .. } in iter {
        let offset = out.positions.len() as u32;
        out.positions.extend(mesh.positions);
        out.normals.extend(mesh.normals);
        out.uvs.extend(mesh.uvs);
        for tri in mesh.indices {
            out.indices
                .push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
    }
    log::info!("merged {count} mesh prims into a single CpuMesh");
    Ok(out)
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
    normals: Option<NormalPrimvar>,
    world_xform: Mat4,
}

fn build_intermediate(m: &rust_usd::Mesh) -> Result<UsdMesh> {
    let path = m.prim_path();

    // USD authors `local_to_world` as row-major; rust-usd hands it back
    // as [[f32;4];4] in the same convention. glam::Mat4 is column-major.
    let lt_w = m.local_to_world();
    let world_xform = Mat4::from_cols_array_2d(&[
        [lt_w[0][0], lt_w[1][0], lt_w[2][0], lt_w[3][0]],
        [lt_w[0][1], lt_w[1][1], lt_w[2][1], lt_w[3][1]],
        [lt_w[0][2], lt_w[1][2], lt_w[2][2], lt_w[3][2]],
        [lt_w[0][3], lt_w[1][3], lt_w[2][3], lt_w[3][3]],
    ]);

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

    let st_data = m.st_uv();
    let st = if st_data.is_empty() {
        None
    } else {
        let st_idx = m.st_indices_u32();
        let indices = if st_idx.is_empty() { None } else { Some(st_idx) };
        // Query the Primvar handle for the actual interpolation token.
        // FaceVarying is the typical authored value but rust-usd surfaces
        // whatever was set via UsdGeomPrimvarsAPI — vertex-interpolated
        // UVs (common on subdivision-friendly meshes) get routed through
        // the Vertex branch in `expand_to_corners` correctly when we
        // pull the live string instead of hardcoding it.
        let interp = m
            .primvar("st")
            .map(|pv| Interpolation::parse(&pv.interpolation()))
            .unwrap_or(Interpolation::FaceVarying);
        Some(StPrimvar {
            data: st_data,
            indices,
            interpolation: interp,
        })
    };

    Ok(UsdMesh {
        path,
        points: m.points_xyz(),
        face_vertex_counts: m.face_vertex_counts_u32(),
        face_vertex_indices: m.face_vertex_indices_u32(),
        st,
        normals,
        world_xform,
    })
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
            indices.push([pivot_out, a_out, b_out]);
        }

        fv_cursor += count;
    }

    Ok(CpuMesh {
        positions,
        normals,
        uvs,
        indices,
    })
}

fn normal_matrix_from_xform(m: &Mat4) -> Mat3 {
    let m3 = Mat3::from_mat4(*m);
    m3.inverse().transpose()
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
    (p1 - p0).cross(p2 - p0).normalize_or_zero()
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
