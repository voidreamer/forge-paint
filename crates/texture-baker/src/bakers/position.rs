use glam::Vec3;

use crate::raster::TexelData;

/// Normalization mode for position maps.
#[derive(Debug, Clone, Copy)]
pub enum PositionNormalization {
    /// Normalize to bounding box [0, 1].
    BoundingBox { min: Vec3, max: Vec3 },
    /// Raw world-space position (requires float output).
    None,
}

/// Bake world-space position at a single texel.
pub fn bake_position_texel(texel: &TexelData, normalization: &PositionNormalization) -> [f32; 3] {
    match normalization {
        PositionNormalization::BoundingBox { min, max } => {
            let range = *max - *min;
            let safe_range = Vec3::new(
                if range.x.abs() < 1e-8 { 1.0 } else { range.x },
                if range.y.abs() < 1e-8 { 1.0 } else { range.y },
                if range.z.abs() < 1e-8 { 1.0 } else { range.z },
            );
            let normalized = (texel.position - *min) / safe_range;
            [normalized.x, normalized.y, normalized.z]
        }
        PositionNormalization::None => [texel.position.x, texel.position.y, texel.position.z],
    }
}

/// Compute the bounding box of all texels in the grid.
pub fn compute_texel_bounds(
    data: &[Option<TexelData>],
) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for texel in data.iter().flatten() {
        min = min.min(texel.position);
        max = max.max(texel.position);
    }

    (min, max)
}
