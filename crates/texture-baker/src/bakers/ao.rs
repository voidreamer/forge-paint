use glam::Vec3;

use crate::accel::AccelStructure;
use crate::raster::TexelData;

/// Ray distribution mode for hemisphere sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Cosine-weighted hemisphere sampling (default, energy-conserving).
    Cosine,
    /// Uniform hemisphere sampling (equal probability in all directions).
    Uniform,
}

/// Settings for ray-based bakers (AO, thickness, bent normals).
#[derive(Debug, Clone)]
pub struct RaySettings {
    /// Number of rays per texel (higher = less noise).
    pub ray_count: u32,
    /// Maximum occlusion distance.
    pub max_distance: f32,
    /// Minimum occlusion distance (ignore very close hits).
    pub min_distance: f32,
    /// Whether to ignore backface hits.
    pub ignore_backface: bool,
    /// Bias to push ray origin away from surface to avoid self-intersection.
    pub bias: f32,
    /// Spread angle in degrees (180 = full hemisphere, < 180 = narrower cone).
    pub spread_angle: f32,
    /// Ray distribution mode.
    pub distribution: Distribution,
}

impl Default for RaySettings {
    fn default() -> Self {
        Self {
            ray_count: 128,
            max_distance: f32::MAX,
            min_distance: 0.001,
            ignore_backface: true,
            bias: 0.001,
            spread_angle: 180.0,
            distribution: Distribution::Cosine,
        }
    }
}

/// Backward-compatible alias.
pub type AoSettings = RaySettings;

/// Bake ambient occlusion at a single texel.
/// Returns a value in [0, 1] where 1.0 = fully unoccluded, 0.0 = fully occluded.
pub fn bake_ao_texel(
    texel: &TexelData,
    accel: &AccelStructure,
    settings: &RaySettings,
    rng_seed: u32,
) -> f32 {
    let origin = texel.position + texel.normal * settings.bias;
    let mut unoccluded = 0u32;

    for i in 0..settings.ray_count {
        let dir = sample_hemisphere(texel.normal, i, settings.ray_count, rng_seed, settings.spread_angle, settings.distribution);

        let occluded = accel.trace_any_hit(
            origin,
            dir,
            settings.max_distance,
            settings.bias,
            settings.ignore_backface,
        );

        if !occluded {
            unoccluded += 1;
        }
    }

    unoccluded as f32 / settings.ray_count as f32
}

/// Bake bent normal at a single texel.
/// Returns the average direction of unoccluded rays (world space, encoded to [0,1]).
pub fn bake_bent_normal_texel(
    texel: &TexelData,
    accel: &AccelStructure,
    settings: &RaySettings,
    rng_seed: u32,
) -> [f32; 3] {
    let origin = texel.position + texel.normal * settings.bias;
    let mut bent = Vec3::ZERO;

    for i in 0..settings.ray_count {
        let dir = sample_hemisphere(texel.normal, i, settings.ray_count, rng_seed, settings.spread_angle, settings.distribution);

        let occluded = accel.trace_any_hit(
            origin,
            dir,
            settings.max_distance,
            settings.bias,
            settings.ignore_backface,
        );

        if !occluded {
            bent += dir;
        }
    }

    let bent = if bent.length_squared() > 1e-8 {
        bent.normalize()
    } else {
        texel.normal
    };

    [
        bent.x * 0.5 + 0.5,
        bent.y * 0.5 + 0.5,
        bent.z * 0.5 + 0.5,
    ]
}

/// Bake thickness at a single texel.
/// Casts rays *inward* (opposite normal) to measure how enclosed the surface is.
/// Returns a value in [0, 1] where 1.0 = thick (solid), 0.0 = thin (edge).
/// Convention matches Substance Painter: white = thick, black = thin.
pub fn bake_thickness_texel(
    texel: &TexelData,
    accel: &AccelStructure,
    settings: &RaySettings,
    rng_seed: u32,
) -> f32 {
    // Push origin INWARD so we start inside the mesh. With backface skip ON,
    // the shell we're inside of is a backface → ignored. Only frontface hits
    // (opposite walls) count as occlusion, making the point "thin".
    let thickness_bias = settings.bias * 2.0;
    let origin = texel.position - texel.normal * thickness_bias;
    let inverted_normal = -texel.normal;
    let min_t = thickness_bias;
    let mut unoccluded = 0u32;

    for i in 0..settings.ray_count {
        let dir = sample_hemisphere(inverted_normal, i, settings.ray_count, rng_seed, settings.spread_angle, settings.distribution);

        let occluded = accel.trace_any_hit(
            origin,
            dir,
            settings.max_distance,
            min_t,
            true, // skip backfaces — our own shell is a backface from inside
        );

        if !occluded {
            unoccluded += 1;
        }
    }

    unoccluded as f32 / settings.ray_count as f32
}

/// Generate a direction on the hemisphere around `normal`, respecting
/// `spread_angle` (degrees, 180 = full hemisphere) and `distribution`.
///
/// Uses Hammersley sequence for stratified sampling (low discrepancy).
fn sample_hemisphere(
    normal: Vec3,
    sample_index: u32,
    total_samples: u32,
    seed: u32,
    spread_angle: f32,
    distribution: Distribution,
) -> Vec3 {
    // Hammersley 2D sequence
    let i = sample_index.wrapping_add(seed);
    let xi1 = i as f32 / total_samples as f32;
    let xi2 = radical_inverse_vdc(i);

    let phi = 2.0 * std::f32::consts::PI * xi1;

    // Clamp spread to [0, 180] and convert to the cosine of the half-angle.
    // cos_max = cos(spread/2): at 180 deg this is cos(90) = 0 (full hemisphere),
    // at smaller angles the cone narrows.
    let half_angle_rad = (spread_angle.clamp(0.0, 180.0) * 0.5).to_radians();
    let cos_max = half_angle_rad.cos();

    let (cos_theta, sin_theta) = match distribution {
        Distribution::Cosine => {
            // Cosine-weighted hemisphere: remap xi2 into [cos_max, 1] range
            // so that samples stay within the cone.
            let remapped = 1.0 - xi2 * (1.0 - cos_max * cos_max);
            let ct = remapped.sqrt();
            let st = (1.0 - remapped).sqrt();
            (ct, st)
        }
        Distribution::Uniform => {
            // Uniform hemisphere within cone: theta in [0, half_angle]
            let ct = 1.0 - xi2 * (1.0 - cos_max);
            let st = (1.0 - ct * ct).sqrt();
            (ct, st)
        }
    };

    let x = phi.cos() * sin_theta;
    let y = phi.sin() * sin_theta;
    let z = cos_theta;

    // Build an orthonormal basis around the normal
    let (tangent, bitangent) = build_orthonormal_basis(normal);

    (tangent * x + bitangent * y + normal * z).normalize()
}

/// Van der Corput radical inverse (base 2) for Hammersley sequence.
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 0x100000000 as f32
}

/// Build an orthonormal basis from a normal vector (Frisvad's method).
fn build_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    if n.z < -0.999_999_9 {
        return (Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
    }
    let a = 1.0 / (1.0 + n.z);
    let b = -n.x * n.y * a;
    let tangent = Vec3::new(1.0 - n.x * n.x * a, b, -n.x);
    let bitangent = Vec3::new(b, 1.0 - n.y * n.y * a, -n.y);
    (tangent, bitangent)
}
