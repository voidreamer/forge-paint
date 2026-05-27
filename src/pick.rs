use eframe::egui;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::mesh::CpuMesh;

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub tri: usize,
    pub uv: Vec2,
    pub world_pos: Vec3,
    pub dist: f32,
}

/// Convert a screen-space cursor position within `rect` to a world-space ray.
/// Ray origin is the camera eye; direction points through the cursor.
pub fn screen_to_ray(
    screen_pos: egui::Pos2,
    rect: egui::Rect,
    view_proj: Mat4,
    eye: Vec3,
) -> (Vec3, Vec3) {
    let local = screen_pos - rect.min;
    let x_ndc = (local.x / rect.width().max(1.0)) * 2.0 - 1.0;
    let y_ndc = 1.0 - (local.y / rect.height().max(1.0)) * 2.0;

    let inv_vp = view_proj.inverse();
    // Pick a point on the far plane, unproject.
    let far = inv_vp * Vec4::new(x_ndc, y_ndc, 1.0, 1.0);
    let far_world = Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);
    let dir = (far_world - eye).normalize_or_zero();
    (eye, dir)
}

/// Möller-Trumbore ray-triangle intersection.
/// Returns (t, u, v) where t is parametric distance along `dir`, and (u, v)
/// are barycentric coords for `v1`/`v2` (w = 1 - u - v corresponds to `v0`).
///
/// Double-sided — `det.abs() < EPS` just guards the parallel-ray
/// degenerate. The wgpu pipeline renders double-sided too (paint
/// workflows always want both sides paintable), so picking matches
/// what the user sees.
fn ray_tri(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-8;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let pvec = dir.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = orig - v0;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    if t < 0.0 {
        return None;
    }
    Some((t, u, v))
}

/// Nearest-hit triangle pick. Brute force; fine for prototype-scale meshes.
pub fn pick(mesh: &CpuMesh, orig: Vec3, dir: Vec3) -> Option<Hit> {
    let mut best: Option<(usize, f32, f32, f32)> = None;
    for (i, tri) in mesh.indices.iter().enumerate() {
        let v0 = mesh.positions[tri[0] as usize];
        let v1 = mesh.positions[tri[1] as usize];
        let v2 = mesh.positions[tri[2] as usize];
        if let Some((t, u, v)) = ray_tri(orig, dir, v0, v1, v2) {
            if best.map_or(true, |(_, pt, _, _)| t < pt) {
                best = Some((i, t, u, v));
            }
        }
    }
    best.map(|(i, t, u, v)| {
        let tri = mesh.indices[i];
        let w = 1.0 - u - v;
        let uv = w * mesh.uvs[tri[0] as usize]
            + u * mesh.uvs[tri[1] as usize]
            + v * mesh.uvs[tri[2] as usize];
        let p0 = mesh.positions[tri[0] as usize];
        let p1 = mesh.positions[tri[1] as usize];
        let p2 = mesh.positions[tri[2] as usize];
        let world_pos = w * p0 + u * p1 + v * p2;
        Hit {
            tri: i,
            uv,
            world_pos,
            dist: t,
        }
    })
}
