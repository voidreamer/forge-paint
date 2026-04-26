// Skybox background pass: fullscreen triangle at clip z=1, the vertex shader
// unprojects NDC via frame.inv_view_proj to get a world-space ray, the
// fragment samples the equirect env at that direction. Rendered before the
// PBR mesh pass with LessEqual depth test so the mesh draws over.

struct Frame {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    fill_dir: vec4<f32>,
    fill_color: vec4<f32>,
    rim_dir: vec4<f32>,
    rim_color: vec4<f32>,
    ambient_sky: vec4<f32>,
    ambient_ground: vec4<f32>,
    view_mode: u32,
    tonemap_mode: u32,
    exposure: f32,
    ibl_scale: f32,
    inv_view_proj: mat4x4<f32>,
};

struct Env {
    intensity: f32,
    rotation_y: f32,
    skybox_visible: u32,
    mip_count: f32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(2) @binding(0) var<uniform> env: Env;
@group(2) @binding(1) var env_tex: texture_2d<f32>;
// bindings 2 (irradiance) and 3 (prefilter) unused by the skybox
@group(2) @binding(4) var env_sampler: sampler;
// bindings 5 (brdf_lut) and 6 (brdf_sampler) unused too

const PI: f32 = 3.14159265359;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_dir: vec3<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    // Fullscreen triangle at the far plane (NDC z = 1 after perspective).
    let ndc = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 1.0, 1.0);
    var out: VsOut;
    out.pos = ndc;
    let world = frame.inv_view_proj * ndc;
    let world_pt = world.xyz / world.w;
    out.world_dir = world_pt - frame.camera_pos.xyz;
    return out;
}

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
    return vec2<f32>((phi + PI) / (2.0 * PI), 0.5 - theta / PI);
}

// Tonemap implementations kept in sync with pbr.wgsl so the skybox matches
// whatever the mesh rendered with.
fn aces_narkowicz(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

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

fn khronos_pbr_neutral(color: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    var c = color;
    let mn = min(c.r, min(c.g, c.b));
    var offset = 0.04;
    if mn < 0.08 { offset = mn - 6.25 * mn * mn; }
    c = c - vec3<f32>(offset);
    let peak = max(c.r, max(c.g, c.b));
    if peak < start_compression { return c; }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    c = c * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), vec3<f32>(g));
}

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - vec3<f32>(0.00232);
}

fn agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0784335999999992, 0.0792237451477643),
        vec3<f32>(0.0423282422610123, 0.878468636469772, 0.0791661274605434),
        vec3<f32>(0.0423756549057051, 0.0784336, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0980208811401368, -0.0990297440797205),
        vec3<f32>(-0.0528968517574562, 1.15190312990417, -0.0989611768448433),
        vec3<f32>(-0.0529716355144438, -0.0980434501171241, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    let v = inset * max(color, vec3<f32>(0.0));
    let v_log = clamp(log2(v + vec3<f32>(1e-10)), vec3<f32>(min_ev), vec3<f32>(max_ev));
    let v_norm = (v_log - vec3<f32>(min_ev)) / (max_ev - min_ev);
    let curve = agx_default_contrast_approx(v_norm);
    return clamp(outset * curve, vec3<f32>(0.0), vec3<f32>(1.0));
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
        case 4u: {
            return khronos_pbr_neutral(x);
        }
        case 5u: {
            return agx(x);
        }
        default: {
            return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
}

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_dir);
    let uv = dir_to_equirect_uv(dir);
    let rgb = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb * env.intensity;
    let tonemapped = apply_tonemap(rgb * frame.exposure, frame.tonemap_mode);
    return vec4<f32>(tonemapped, 1.0);
}
