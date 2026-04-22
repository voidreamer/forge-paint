// Precomputed BRDF integration LUT (Karis 2013 / Epic split-sum).
// Output: RG16Float where
//   R = GGX integral with (1 - F) weight  — "scale"
//   G = GGX integral with F weight        — "bias"
// Sampled at (NdotV, roughness) in the PBR shader as:
//   ibl_specular = prefilter(R, roughness) * (F0 * lut.r + lut.g);

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
    // uv.x ∈ [0, 1] maps to NdotV, uv.y ∈ [0, 1] maps to roughness.
    // No V-flip needed — texture is sampled directly.
    out.uv = vec2<f32>(x, y);
    return out;
}

const PI: f32 = 3.14159265359;

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

fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) * 0.5;   // IBL variant (direct uses (a+1)²/8)
    let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return ggx_v * ggx_l;
}

fn integrate_brdf(n_dot_v_in: f32, roughness_in: f32) -> vec2<f32> {
    let n_dot_v = max(n_dot_v_in, 1e-4);
    let roughness = max(roughness_in, 1e-4);

    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var scale = 0.0;
    var bias = 0.0;
    const SAMPLE_COUNT: u32 = 1024u;

    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        if l.z > 0.0 {
            let n_dot_l = clamp(l.z, 0.0, 1.0);
            let n_dot_h = clamp(h.z, 0.0, 1.0);
            let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
            let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = pow(1.0 - v_dot_h, 5.0);
            scale = scale + (1.0 - fc) * g_vis;
            bias = bias + fc * g_vis;
        }
    }
    return vec2<f32>(scale, bias) / f32(SAMPLE_COUNT);
}

@fragment
fn fs_brdf(in: VsOut) -> @location(0) vec4<f32> {
    let n_dot_v = in.uv.x;
    let roughness = in.uv.y;
    let result = integrate_brdf(n_dot_v, roughness);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
}
