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
@group(2) @binding(2) var env_sampler: sampler;

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

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_dir);
    let uv = dir_to_equirect_uv(dir);
    let rgb = textureSampleLevel(env_tex, env_sampler, uv, 0.0).rgb * env.intensity;
    // Reinhard tonemap so the skybox matches the mesh's output encoding.
    let tonemapped = rgb / (rgb + vec3<f32>(1.0));
    return vec4<f32>(tonemapped, 1.0);
}
