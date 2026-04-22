use anyhow::{anyhow, bail, Context, Result};
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::mesh::CpuMesh;
use crate::usd::parser::{parse_usda, Interpolation, StPrimvar, UsdMesh};

#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub path: String,
    pub mesh: CpuMesh,
}

/// Locate the `usdcat` binary. Prefer $HOME/USD/bin/usdcat (the anvil-managed
/// install), fall back to PATH.
fn locate_usdcat() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(&home).join("USD/bin/usdcat");
        if p.exists() {
            return Ok(p);
        }
    }
    // PATH lookup via `which`
    let out = Command::new("which")
        .arg("usdcat")
        .output()
        .context("failed to run `which usdcat`")?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Ok(PathBuf::from(s));
        }
    }
    Err(anyhow!(
        "usdcat not found. Install USD or source the anvil pipeline environment."
    ))
}

fn flatten_to_usda(input: &Path) -> Result<String> {
    let usdcat = locate_usdcat()?;
    let out_path = std::env::temp_dir().join(format!(
        "forge-paint-flat-{}.usda",
        std::process::id()
    ));
    let status = Command::new(&usdcat)
        .arg("--flatten")
        .arg(input)
        .arg("-o")
        .arg(&out_path)
        .status()
        .with_context(|| format!("failed to execute {}", usdcat.display()))?;
    if !status.success() {
        bail!("usdcat --flatten exited with {status}");
    }
    let text = std::fs::read_to_string(&out_path)
        .with_context(|| format!("failed to read flattened output at {}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);
    Ok(text)
}

/// Load a USD stage by flattening it and parsing the resulting .usda text.
/// Returns one `LoadedMesh` per `UsdGeomMesh` prim, with xforms baked into
/// positions and UVs unwelded to per-face-vert.
pub fn load_stage(path: &Path) -> Result<Vec<LoadedMesh>> {
    let text = flatten_to_usda(path)?;
    let usd_meshes = parse_usda(&text).with_context(|| "parsing flattened .usda")?;
    if usd_meshes.is_empty() {
        bail!("stage contains no UsdGeomMesh prims");
    }

    let mut loaded = Vec::with_capacity(usd_meshes.len());
    for m in usd_meshes {
        let mesh = triangulate(&m)?;
        loaded.push(LoadedMesh {
            path: m.path,
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

/// Fan-triangulate n-gons with per-face-vert output (UVs unwelded).
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
