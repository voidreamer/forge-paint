// Diffuse-irradiance convolution: for each output UV direction N, integrate
// the cosine-weighted hemisphere sampled from the source equirect. With
// cosine-importance sampling the PDF = cos(θ)/π, so the Monte Carlo
// estimator of ∫ L(ω) cos(θ) dω is simply (π/N) · Σ L(ωi). We drop the π
// factor here and re-apply it implicitly when the PBR shader divides by π in
// its Lambertian term — net: stored value = (1/N) · Σ L, consumed as
// diffuse_ibl = stored * albedo * (1 - metallic).

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

const PI: f32 = 3.14159265359;
const SAMPLES: u32 = 512u;

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    // Inverse of the equirect mapping used everywhere else (with rotation 0):
    //   phi   = atan2(dir.z, dir.x)  in [-π, π]
    //   theta = asin(dir.y)         in [-π/2, π/2]
    //   u = (phi + π) / (2π),  v = 0.5 - theta/π
    let phi = uv.x * 2.0 * PI - PI;
    let theta = (0.5 - uv.y) * PI;
    let cos_t = cos(theta);
    return vec3<f32>(cos_t * cos(phi), sin(theta), cos_t * sin(phi));
}

fn dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2<f32>((phi + PI) / (2.0 * PI), 0.5 - theta / PI);
}

fn cosine_hemisphere(xi: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt(1.0 - xi.y);
    let sin_theta = sqrt(max(0.0, xi.y));
    let h_ts = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h_ts.x + bitangent * h_ts.y + n * h_ts.z);
}

@fragment
fn fs_irradiance(in: VsOut) -> @location(0) vec4<f32> {
    let n = uv_to_dir(in.uv);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < SAMPLES; i = i + 1u) {
        let xi = hammersley(i, SAMPLES);
        let dir = cosine_hemisphere(xi, n);
        sum = sum + textureSampleLevel(src, src_sampler, dir_to_uv(dir), 0.0).rgb;
    }
    return vec4<f32>(sum / f32(SAMPLES), 1.0);
}
