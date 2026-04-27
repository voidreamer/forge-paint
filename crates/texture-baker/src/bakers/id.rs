use crate::accel::HitRecord;

/// Source for ID map colors.
#[derive(Debug, Clone, Copy)]
pub enum IdSource {
    /// Assign a unique color per mesh object.
    MeshId,
    /// Assign a unique color per triangle material/group index.
    MaterialId,
}

/// Bake an ID color at a single texel.
pub fn bake_id_texel(hit: &HitRecord, source: IdSource) -> [f32; 3] {
    let id = match source {
        IdSource::MeshId => hit.mesh_index,
        IdSource::MaterialId => {
            // Use mesh_index * large_prime + tri_index to differentiate
            // In a real implementation, we'd read material IDs from the mesh
            hit.mesh_index * 997 + hit.tri_index
        }
    };

    id_to_color(id)
}

/// Bake an ID color from the low-poly mesh (no high-poly needed).
pub fn bake_id_from_lowpoly(mesh_index: usize, tri_index: usize, source: IdSource) -> [f32; 3] {
    let id = match source {
        IdSource::MeshId => mesh_index,
        IdSource::MaterialId => mesh_index * 997 + tri_index,
    };

    id_to_color(id)
}

/// Convert an integer ID to a distinct color using a hash-based approach.
fn id_to_color(id: usize) -> [f32; 3] {
    // Use a simple hash to generate visually distinct colors
    let hash = hash_u32(id as u32);

    // Extract RGB from hash bits, ensuring high saturation for visual distinction
    let hue = (hash & 0xFF) as f32 / 255.0;
    let sat = 0.6 + ((hash >> 8) & 0xFF) as f32 / 255.0 * 0.4; // 0.6-1.0
    let val = 0.5 + ((hash >> 16) & 0xFF) as f32 / 255.0 * 0.5; // 0.5-1.0

    hsv_to_rgb(hue, sat, val)
}

fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e3779b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x45d9f3b);
    x ^= x >> 16;
    x
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}
