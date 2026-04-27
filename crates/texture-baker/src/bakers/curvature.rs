/// Settings for curvature baking.
#[derive(Debug, Clone)]
pub struct CurvatureSettings {
    /// Overall intensity multiplier.
    pub intensity: f32,
    /// Detail weight — higher values emphasise fine-scale curvature over broad.
    pub detail: f32,
    /// Multiplier applied to each kernel radius (1.0 = default 1/2/4 px radii).
    pub radius_scale: f32,
}

impl Default for CurvatureSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            detail: 1.0,
            radius_scale: 1.0,
        }
    }
}

/// Compute curvature map from a world-space normal map using multi-scale Sobel filtering.
///
/// Uses 3 scales (1px, 2px, 4px radius) blended together for both fine detail
/// and broader edge detection, matching Substance Painter's curvature quality.
/// Convex edges (ridges) → bright (> 0.5), concave (valleys) → dark (< 0.5).
pub fn compute_curvature_from_normals(
    world_normals: &[[f32; 3]],
    mask: &[bool],
    width: u32,
    height: u32,
    settings: &CurvatureSettings,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut curvature = vec![0.5f32; w * h];

    // Apply radius_scale to the three kernel radii (clamped to >= 1)
    let r1 = (1.0 * settings.radius_scale).round().max(1.0) as usize;
    let r2 = (2.0 * settings.radius_scale).round().max(1.0) as usize;
    let r4 = (4.0 * settings.radius_scale).round().max(1.0) as usize;

    // Detail controls the blend: higher detail puts more weight on the fine scale.
    // Base weights: [0.5, 0.3, 0.2]. With detail > 1, fine gets boosted.
    let d = settings.detail.max(0.01);
    let w1 = 0.5 * d;
    let w2 = 0.3;
    let w3 = 0.2 / d;
    let wsum = w1 + w2 + w3;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask[idx] {
                continue;
            }

            // Multi-scale: blend curvature at 3 different radii
            let c1 = sobel_at_radius(world_normals, mask, x, y, w, h, r1);
            let c2 = sobel_at_radius(world_normals, mask, x, y, w, h, r2);
            let c4 = sobel_at_radius(world_normals, mask, x, y, w, h, r4);

            // Normalized weighted blend
            let curv = (c1 * w1 + c2 * w2 + c4 * w3) / wsum;

            curvature[idx] = (0.5 + curv * settings.intensity * 2.0).clamp(0.0, 1.0);
        }
    }

    curvature
}

/// Compute curvature at a pixel using Sobel-like sampling at a given radius.
fn sobel_at_radius(
    normals: &[[f32; 3]],
    mask: &[bool],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    radius: usize,
) -> f32 {
    let decode = |sx: usize, sy: usize| -> [f32; 3] {
        let sx = sx.min(w - 1);
        let sy = sy.min(h - 1);
        let idx = sy * w + sx;
        if mask[idx] {
            let n = normals[idx];
            [n[0] * 2.0 - 1.0, n[1] * 2.0 - 1.0, n[2] * 2.0 - 1.0]
        } else {
            let center = normals[y * w + x];
            [center[0] * 2.0 - 1.0, center[1] * 2.0 - 1.0, center[2] * 2.0 - 1.0]
        }
    };

    let r = radius;
    let xl = x.saturating_sub(r);
    let xr = (x + r).min(w - 1);
    let yu = y.saturating_sub(r);
    let yd = (y + r).min(h - 1);

    // 3x3 Sobel at the given radius
    let tl = decode(xl, yu);
    let tc = decode(x, yu);
    let tr = decode(xr, yu);
    let ml = decode(xl, y);
    let mr = decode(xr, y);
    let bl = decode(xl, yd);
    let bc = decode(x, yd);
    let br = decode(xr, yd);

    let gx = |c: usize| -> f32 {
        (-tl[c] + tr[c] - 2.0 * ml[c] + 2.0 * mr[c] - bl[c] + br[c]) / 8.0
    };
    let gy = |c: usize| -> f32 {
        (-tl[c] - 2.0 * tc[c] - tr[c] + bl[c] + 2.0 * bc[c] + br[c]) / 8.0
    };

    // Divergence (signed curvature)
    let div = gx(0) + gy(1);

    // Edge magnitude boost
    let edge_x = gx(0) * gx(0) + gx(1) * gx(1) + gx(2) * gx(2);
    let edge_y = gy(0) * gy(0) + gy(1) * gy(1) + gy(2) * gy(2);
    let edge = (edge_x + edge_y).sqrt();

    div + edge * div.signum() * 0.5
}
