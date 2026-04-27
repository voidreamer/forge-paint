use glam::Vec3;

use crate::accel::HitRecord;
use crate::mesh::Mesh;
use crate::raster::TexelData;

/// Y-axis convention for tangent-space normal maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalMapFormat {
    /// Green channel = Y+ (DirectX convention, Unreal Engine)
    DirectX,
    /// Green channel = Y- (OpenGL convention)
    OpenGL,
}

/// Bake a tangent-space normal at a single texel.
pub fn bake_normal_texel(
    texel: &TexelData,
    hit: &HitRecord,
    high_poly_meshes: &[Mesh],
    format: NormalMapFormat,
) -> [f32; 3] {
    let hp_mesh = &high_poly_meshes[hit.mesh_index];
    let hp_tri = &hp_mesh.indices[hit.tri_index];
    let hp_normals = hp_mesh.tri_normals(hp_tri);

    // Interpolate high-poly normal at hit point using barycentric coords
    let w0 = 1.0 - hit.u - hit.v;
    let hp_normal = (hp_normals[0] * w0 + hp_normals[1] * hit.u + hp_normals[2] * hit.v).normalize();

    // Transform high-poly world-space normal into low-poly tangent space
    // TBN^-1 * N_highpoly = (dot(T, N_hp), dot(B, N_hp), dot(N, N_hp))
    let ts_x = texel.tangent.dot(hp_normal);
    let ts_y = texel.bitangent.dot(hp_normal);
    let ts_z = texel.normal.dot(hp_normal);

    let ts_normal = Vec3::new(ts_x, ts_y, ts_z).normalize();

    // Encode to [0, 1]
    let y = match format {
        NormalMapFormat::DirectX => ts_normal.y,
        NormalMapFormat::OpenGL => -ts_normal.y,
    };

    [
        ts_normal.x * 0.5 + 0.5,
        y * 0.5 + 0.5,
        ts_normal.z * 0.5 + 0.5,
    ]
}

/// Bake a world-space normal at a texel (from low-poly if no hit, from high-poly if hit).
pub fn bake_world_normal_texel(
    texel: &TexelData,
    hit: Option<(&HitRecord, &[Mesh])>,
) -> [f32; 3] {
    let normal = if let Some((h, meshes)) = hit {
        let hp_mesh = &meshes[h.mesh_index];
        let hp_tri = &hp_mesh.indices[h.tri_index];
        let hp_normals = hp_mesh.tri_normals(hp_tri);
        let w0 = 1.0 - h.u - h.v;
        (hp_normals[0] * w0 + hp_normals[1] * h.u + hp_normals[2] * h.v).normalize()
    } else {
        texel.normal
    };

    [
        normal.x * 0.5 + 0.5,
        normal.y * 0.5 + 0.5,
        normal.z * 0.5 + 0.5,
    ]
}
