// GGX specular prefilter — one mip of the destination equirect per
// roughness step (mip 0 = roughness 0 / mirror, top mip = roughness 1).
// The fragment importance-samples the GGX distribution at (params.roughness)
// and weights the accumulated environment sample by NdotL.

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

struct Params {
    roughness: f32,
    // 12 bytes of implicit trailing padding to 16 — matches Rust's
    // `_pad: [f32; 3]`. Adding `vec3<f32>` explicitly would bump the struct
    // to 32 and desynchronise the layout.
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

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

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let h_ts = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h_ts.x + bitangent * h_ts.y + n * h_ts.z);
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let n = uv_to_dir(in.uv);
    let v = n; // Karis split-sum simplification: assume view == normal
    let rough = params.roughness;

    // Mip 0 (roughness≈0) is a mirror — bypass MC and just sample the source
    // direction directly so we don't waste samples on a delta distribution.
    if rough < 0.01 {
        return vec4<f32>(
            textureSampleLevel(src, src_sampler, dir_to_uv(n), 0.0).rgb,
            1.0,
        );
    }

    var sum = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < SAMPLES; i = i + 1u) {
        let xi = hammersley(i, SAMPLES);
        let h = importance_sample_ggx(xi, n, rough);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(dot(n, l), 0.0);
        if n_dot_l > 0.0 {
            sum = sum + textureSampleLevel(src, src_sampler, dir_to_uv(l), 0.0).rgb * n_dot_l;
            total_w = total_w + n_dot_l;
        }
    }
    return vec4<f32>(sum / max(total_w, 0.0001), 1.0);
}
