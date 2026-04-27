use crate::accel::HitRecord;
use crate::raster::TexelData;

/// Bake height (displacement) at a single texel.
/// Returns the signed distance from the low-poly surface to the high-poly hit along the normal.
pub fn bake_height_texel(texel: &TexelData, hit: &HitRecord, ray_origin: glam::Vec3) -> f32 {
    // The hit position along the ray
    // ray was cast from: origin = texel.position + normal * frontal_distance
    // in direction: -normal
    // So hit position = origin + dir * t
    // Distance from low-poly surface = frontal_distance - t (if ray goes from cage inward)
    // But more precisely, we want the signed distance from the low-poly surface to the hit.
    let hit_position = ray_origin + (-texel.normal) * hit.t;
    let offset = hit_position - texel.position;
    texel.normal.dot(offset)
}

/// Normalize height values to [0, 1] range.
pub fn normalize_height_map(heights: &mut [Option<f32>]) {
    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;

    for h in heights.iter().flatten() {
        min_h = min_h.min(*h);
        max_h = max_h.max(*h);
    }

    let range = max_h - min_h;
    if range.abs() < 1e-10 {
        for h in heights.iter_mut().flatten() {
            *h = 0.5;
        }
        return;
    }

    for h in heights.iter_mut().flatten() {
        *h = (*h - min_h) / range;
    }
}
