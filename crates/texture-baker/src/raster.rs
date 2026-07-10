use glam::Vec3;

use crate::mesh::Mesh;
use crate::tangent::TangentData;

/// Data interpolated at each texel from the low-poly mesh.
#[derive(Debug, Clone, Copy)]
pub struct TexelData {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    /// Cage position at this texel (if cage mesh provided). Used as ray origin.
    pub cage_position: Option<Vec3>,
    /// Cage-to-lowpoly direction (if cage mesh provided). Used as ray direction.
    pub cage_direction: Option<Vec3>,
    /// Which low-poly triangle this texel came from.
    pub tri_index: usize,
    /// Which low-poly mesh this texel came from.
    pub mesh_index: usize,
}

/// A 2D buffer of texel data. `None` entries are outside all UV islands.
pub struct TexelGrid {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Option<TexelData>>,
}

impl TexelGrid {
    pub fn get(&self, x: u32, y: u32) -> Option<&TexelData> {
        self.data[(y * self.width + x) as usize].as_ref()
    }
}

/// Input data for a single mesh during rasterization.
pub struct RasterInput<'a> {
    pub mesh_index: usize,
    pub mesh: &'a Mesh,
    pub tangent_data: &'a TangentData,
    /// Optional cage mesh (must have identical vertex count and topology).
    pub cage: Option<&'a Mesh>,
}

/// Rasterize the low-poly mesh into UV/texture space.
///
/// For each texel covered by a UV-space triangle, we interpolate the world-space
/// position, normal, tangent, and bitangent using barycentric coordinates.
/// If a cage mesh is provided, cage positions and cage→lowpoly directions are also interpolated.
/// Uses conservative rasterization (half-pixel expansion) to avoid gaps.
pub fn rasterize_uv_space(inputs: &[RasterInput], width: u32, height: u32) -> TexelGrid {
    let total = (width * height) as usize;
    let mut data: Vec<Option<TexelData>> = vec![None; total];

    for input in inputs {
        let mesh = input.mesh;
        let tangent_data = input.tangent_data;
        let mesh_index = input.mesh_index;

        for (tri_idx, tri) in mesh.indices.iter().enumerate() {
            let uvs = mesh.tri_uvs(tri);
            let positions = mesh.tri_positions(tri);
            let normals = mesh.tri_normals(tri);

            let (tan0, bitan0) = tangent_data.tangent_bitangent(tri[0] as usize, normals[0]);
            let (tan1, bitan1) = tangent_data.tangent_bitangent(tri[1] as usize, normals[1]);
            let (tan2, bitan2) = tangent_data.tangent_bitangent(tri[2] as usize, normals[2]);

            // Optional cage positions for this triangle
            let cage_positions = input.cage.map(|c| c.tri_positions(tri));

            // Convert UV to pixel coordinates
            let px = [
                [uvs[0][0] * width as f32, (1.0 - uvs[0][1]) * height as f32],
                [uvs[1][0] * width as f32, (1.0 - uvs[1][1]) * height as f32],
                [uvs[2][0] * width as f32, (1.0 - uvs[2][1]) * height as f32],
            ];

            // Bounding box of the triangle in pixel space (conservative: expand by 0.5)
            let min_x = px[0][0].min(px[1][0]).min(px[2][0]) - 0.5;
            let max_x = px[0][0].max(px[1][0]).max(px[2][0]) + 0.5;
            let min_y = px[0][1].min(px[1][1]).min(px[2][1]) - 0.5;
            let max_y = px[0][1].max(px[1][1]).max(px[2][1]) + 0.5;

            let x0 = (min_x.floor() as i32).max(0) as u32;
            let x1 = (max_x.ceil() as i32).min(width as i32) as u32;
            let y0 = (min_y.floor() as i32).max(0) as u32;
            let y1 = (max_y.ceil() as i32).min(height as i32) as u32;

            for y in y0..y1 {
                for x in x0..x1 {
                    // Sample at texel center
                    let px_center = [x as f32 + 0.5, y as f32 + 0.5];

                    if let Some((w0, w1, w2)) = barycentric_2d(px_center, px) {
                        // Conservative rasterization: accept if close to the triangle
                        if w0 >= -0.01 && w1 >= -0.01 && w2 >= -0.01 {
                            let position =
                                positions[0] * w0 + positions[1] * w1 + positions[2] * w2;
                            let normal =
                                (normals[0] * w0 + normals[1] * w1 + normals[2] * w2).normalize();
                            let tangent = (tan0 * w0 + tan1 * w1 + tan2 * w2).normalize();
                            let bitangent = (bitan0 * w0 + bitan1 * w1 + bitan2 * w2).normalize();

                            // Interpolate cage data if available
                            let (cage_position, cage_direction) = if let Some(cp) = &cage_positions
                            {
                                let cage_pos = cp[0] * w0 + cp[1] * w1 + cp[2] * w2;
                                let dir = (position - cage_pos).normalize();
                                (Some(cage_pos), Some(dir))
                            } else {
                                (None, None)
                            };

                            let idx = (y * width + x) as usize;
                            data[idx] = Some(TexelData {
                                position,
                                normal,
                                tangent,
                                bitangent,
                                cage_position,
                                cage_direction,
                                tri_index: tri_idx,
                                mesh_index,
                            });
                        }
                    }
                }
            }
        }
    }

    TexelGrid {
        width,
        height,
        data,
    }
}

/// Compute barycentric coordinates for point P inside triangle defined by pixel coords.
fn barycentric_2d(p: [f32; 2], tri: [[f32; 2]; 3]) -> Option<(f32, f32, f32)> {
    let v0 = [tri[1][0] - tri[0][0], tri[1][1] - tri[0][1]];
    let v1 = [tri[2][0] - tri[0][0], tri[2][1] - tri[0][1]];
    let v2 = [p[0] - tri[0][0], p[1] - tri[0][1]];

    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-10 {
        return None; // degenerate triangle
    }

    let inv_denom = 1.0 / denom;
    let w1 = (d11 * d20 - d01 * d21) * inv_denom;
    let w2 = (d00 * d21 - d01 * d20) * inv_denom;
    let w0 = 1.0 - w1 - w2;

    Some((w0, w1, w2))
}
