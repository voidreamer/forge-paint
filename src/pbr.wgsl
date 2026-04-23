// Phase 1b PBR: samples UDIM-indexed texture_2d_array for base_color,
// rough_metal (glTF packing: R=AO, G=roughness, B=metallic), and normal.
// `uv_to_layer` maps a mesh UV into the array layer for that UDIM tile.

struct Frame {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    ambient_sky: vec4<f32>,
    ambient_ground: vec4<f32>,
    view_mode: u32,        // 0 Material, 1 BaseColor, 2 Rough, 3 Metal, 4 Normal, 5 Mask
    tonemap_mode: u32,     // 0 None, 1 Reinhard, 2 ACES, 3 Filmic (UC2)
    exposure: f32,         // pre-tonemap linear multiplier (= 2^stops)
    _pad: u32,
    inv_view_proj: mat4x4<f32>,
}

struct Material {
    base_color_factor: vec4<f32>,
    params: vec4<f32>,              // x=metallic, y=roughness, z=normal_scale, w=_
    tile_count: u32,
    // 12 bytes of implicit padding here; matches Rust's `_pad0: [u32; 3]`
    // and the 16-byte alignment requirement of the array member below.
    tile_ids: array<vec4<u32>, 8>,  // 32 tile ids packed
}

struct Env {
    intensity: f32,
    rotation_y: f32,
    skybox_visible: u32,
    mip_count: f32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_tex: texture_2d_array<f32>;
@group(1) @binding(2) var rough_metal_tex: texture_2d_array<f32>;
@group(1) @binding(3) var normal_tex: texture_2d_array<f32>;
@group(1) @binding(4) var active_mask_tex: texture_2d_array<f32>;
@group(1) @binding(5) var texset_sampler: sampler;
@group(1) @binding(6) var world_normal_map: texture_2d_array<f32>;
@group(2) @binding(0) var<uniform> env: Env;
@group(2) @binding(1) var env_tex: texture_2d<f32>;
@group(2) @binding(2) var irradiance_tex: texture_2d<f32>;
@group(2) @binding(3) var prefilter_tex: texture_2d<f32>;
@group(2) @binding(4) var env_sampler: sampler;
@group(2) @binding(5) var brdf_lut: texture_2d<f32>;
@group(2) @binding(6) var brdf_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.world_tangent = in.tangent;
    out.uv = in.uv;
    return out;
}

const PI: f32 = 3.14159265359;

fn tile_id_at(i: u32) -> u32 {
    let v = material.tile_ids[i >> 2u];
    let r = i & 3u;
    if r == 0u { return v.x; }
    if r == 1u { return v.y; }
    if r == 2u { return v.z; }
    return v.w;
}

/// Map a mesh UV to the texture-array layer that backs its UDIM tile.
/// Returns -1 if the tile isn't in the paint target.
fn uv_to_layer(uv: vec2<f32>) -> i32 {
    let tu = u32(max(floor(uv.x), 0.0));
    let tv = u32(max(floor(uv.y), 0.0));
    let want_id = 1001u + tu + 10u * tv;
    for (var i = 0u; i < material.tile_count; i = i + 1u) {
        if tile_id_at(i) == want_id { return i32(i); }
    }
    return -1;
}

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-5);
}

fn v_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness;
    let ggx_v = n_dot_l * (n_dot_v * (1.0 - a) + a);
    let ggx_l = n_dot_v * (n_dot_l * (1.0 - a) + a);
    return 0.5 / max(ggx_v + ggx_l, 1e-5);
}

fn f_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Karis' closed-form approximation of the split-sum BRDF LUT.
// Returns vec2(scale, bias) such that `F0 * scale + bias` equals the baked
// LUT's `(F0 * r + g)` term. Avoids the BRDF LUT texture path entirely —
// important while that path is still blocked on task #45.
// Reference: Unreal Engine 4 presentation, Karis 2013 ("Real Shading in UE4").
fn env_brdf_approx(roughness: f32, n_dot_v: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>( 1.0,  0.0425,  1.040, -0.040);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

/// ACES filmic tonemap — Narkowicz 2015 fit.
fn aces_narkowicz(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Filmic tonemap (Hable / Uncharted 2). Similar filmic feel to ACES but
/// with a different shoulder/toe shape — many artists prefer this one for
/// lookdev.
fn filmic_uc2(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    let W = 11.2;
    let curr = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    let white = vec3<f32>(W);
    let white_scale = ((white * (A * white + C * B) + D * E)
        / (white * (A * white + B) + D * F))
        - E / F;
    return clamp(curr / white_scale, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_tonemap(x: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: {
            return x / (x + vec3<f32>(1.0));
        }
        case 2u: {
            return aces_narkowicz(x);
        }
        case 3u: {
            return filmic_uc2(x);
        }
        default: {
            return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
}

// Equirectangular mapping: world-space direction → (u, v) in [0, 1].
// Applies env.rotation_y so the user can spin the sky around the mesh.
fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let c = cos(env.rotation_y);
    let s = sin(env.rotation_y);
    let rotated = vec3<f32>(
        dir.x * c + dir.z * s,
        dir.y,
        -dir.x * s + dir.z * c,
    );
    let phi = atan2(rotated.z, rotated.x);
    let theta = asin(clamp(rotated.y, -1.0, 1.0));
    let u = (phi + PI) / (2.0 * PI);
    let v = 0.5 - theta / PI;
    return vec2<f32>(u, v);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    let uv = dir_to_equirect_uv(dir);
    return textureSampleLevel(env_tex, env_sampler, uv, lod).rgb * env.intensity;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let layer = uv_to_layer(in.uv);

    var base_color: vec3<f32>;
    var metallic: f32;
    var roughness: f32;
    var n_tangent_space: vec3<f32>;

    if layer < 0 {
        // Off-atlas fallback: use material factors only, flat tangent-space normal.
        base_color = material.base_color_factor.rgb;
        metallic = material.params.x;
        roughness = material.params.y;
        n_tangent_space = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        let local_uv = fract(in.uv);
        let bc = textureSample(base_color_tex, texset_sampler, local_uv, layer).rgb;
        base_color = bc * material.base_color_factor.rgb;

        let rm = textureSample(rough_metal_tex, texset_sampler, local_uv, layer);
        // glTF packing: R=AO (unused Phase 1b), G=roughness, B=metallic
        roughness = rm.g * material.params.y;
        metallic = rm.b * material.params.x;

        let nn = textureSample(normal_tex, texset_sampler, local_uv, layer).rgb * 2.0 - 1.0;
        n_tangent_space = normalize(nn);
    }

    let n_geom = normalize(in.world_normal);
    let t_raw = in.world_tangent.xyz;
    let t = normalize(t_raw - n_geom * dot(n_geom, t_raw));
    let b = cross(n_geom, t) * in.world_tangent.w;
    let n = normalize(
        t * n_tangent_space.x * material.params.z
        + b * n_tangent_space.y * material.params.z
        + n_geom * n_tangent_space.z,
    );

    let v = normalize(frame.camera_pos.xyz - in.world_pos);
    let l = normalize(-frame.light_dir.xyz);
    let h = normalize(v + l);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.04, 1.0);

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let d = d_ggx(n_dot_h, roughness);
    let g = v_smith_ggx(n_dot_v, n_dot_l, roughness);
    let f = f_schlick(v_dot_h, f0);
    let specular = (d * g) * f;

    let ks = f;
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;

    let direct = (diffuse + specular) * frame.light_color.rgb * frame.light_color.w * n_dot_l;

    // Karis split-sum IBL with the closed-form LUT approximation (task #45
    // blocks the baked LUT path). `env_brdf_approx` returns (scale, bias)
    // equivalent to the baked (F0 * r + g) term. Plus Fdez-Agüera
    // multi-scattering compensation so metals brighten correctly at grazing
    // angles.
    //   diffuse  = irradiance(N) * albedo * (1 - metallic)
    //   specular = prefilter(R, roughness) * (F0 * scale + bias) * ms_comp
    // Irradiance stores (1/N) · Σ L cos-weighted samples (no π factors);
    // prefilter mip 0 is mirror, last mip is fully rough.
    let irr_uv = dir_to_equirect_uv(n);
    let ibl_diffuse_raw =
        textureSampleLevel(irradiance_tex, env_sampler, irr_uv, 0.0).rgb * env.intensity;

    let r = reflect(-v, n);
    let r_uv = dir_to_equirect_uv(r);
    let spec_lod = roughness * max(env.mip_count - 1.0, 0.0);
    let ibl_spec_raw =
        textureSampleLevel(prefilter_tex, env_sampler, r_uv, spec_lod).rgb * env.intensity;

    let env_ab = env_brdf_approx(roughness, n_dot_v);
    let f_ss = f0 * env_ab.x + vec3<f32>(env_ab.y);
    // Multi-scatter energy compensation (Fdez-Agüera 2019). `fms` is the
    // fraction of light that bounced multiple times in the microfacet
    // distribution and would otherwise be lost.
    let f_avg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let e_ss = env_ab.x + env_ab.y;
    let fms = f_avg * e_ss / (vec3<f32>(1.0) - f_avg * (1.0 - e_ss));
    let kd_ibl = (vec3<f32>(1.0) - f_ss - fms) * (1.0 - metallic);
    let ibl_diffuse = (kd_ibl * base_color) * ibl_diffuse_raw;
    let ibl_specular = ibl_spec_raw * (f_ss + fms);

    let lit = direct + ibl_diffuse + ibl_specular;
    let lit_exposed = lit * frame.exposure;
    let tonemapped = apply_tonemap(lit_exposed, frame.tonemap_mode);

    // View mode override — isolate a channel for debugging / inspection.
    // The render target is Bgra8UnormSrgb, so the hardware applies sRGB
    // encoding on store. We output linear values throughout.
    var out_rgb: vec3<f32>;
    switch frame.view_mode {
        case 1u: {
            // Base color — already linear (sampled from Rgba8UnormSrgb).
            out_rgb = base_color;
        }
        case 2u: {
            out_rgb = vec3<f32>(roughness);
        }
        case 3u: {
            out_rgb = vec3<f32>(metallic);
        }
        case 4u: {
            // Visualise the tangent-space normal as the typical (0.5,0.5,1.0)-
            // biased RGB.
            out_rgb = n_tangent_space * 0.5 + vec3<f32>(0.5);
        }
        case 5u: {
            // Mask preview — grayscale R of the active layer's mask.
            if layer < 0 {
                out_rgb = vec3<f32>(0.5);
            } else {
                let local_uv = fract(in.uv);
                let m = textureSample(active_mask_tex, texset_sampler, local_uv, layer).r;
                out_rgb = vec3<f32>(m);
            }
        }
        case 6u: {
            // World-normal bake preview — stored 0..1 biased, display as RGB.
            if layer < 0 {
                out_rgb = vec3<f32>(0.5, 0.5, 1.0);
            } else {
                let local_uv = fract(in.uv);
                out_rgb =
                    textureSample(world_normal_map, texset_sampler, local_uv, layer).rgb;
            }
        }
        default: {
            out_rgb = tonemapped;
        }
    }
    return vec4<f32>(out_rgb, 1.0);
}
