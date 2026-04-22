// Skybox background pass: fullscreen triangle at clip z=1, the vertex shader
// unprojects NDC via frame.inv_view_proj to get a world-space ray, the
// fragment samples the equirect env at that direction. Rendered before the
// PBR mesh pass with LessEqual depth test so the mesh draws over.

struct Frame {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    ambient_sky: vec4<f32>,
    ambient_ground: vec4<f32>,
    view_mode: u32,
    tonemap_mode: u32,
    exposure: f32,
    _pad: u32,
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

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_dir);
    let uv = dir_to_equirect_uv(dir);
    let rgb = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb * env.intensity;
    let tonemapped = apply_tonemap(rgb * frame.exposure, frame.tonemap_mode);
    return vec4<f32>(tonemapped, 1.0);
}
