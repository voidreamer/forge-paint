// Unlit wireframe overlay — draws the mesh's edge list as lines in
// world space, depth-tested against the mesh pass's depth buffer so
// hidden edges don't bleed through.

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

@group(0) @binding(0) var<uniform> frame: Frame;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_wire(in: Vertex) -> VsOut {
    var out: VsOut;
    out.pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_wire() -> @location(0) vec4<f32> {
    // Unlit linear HDR — tonemap will compress. Keep the wire color
    // bright enough to stay visible through the tonemap curve.
    return vec4<f32>(0.55, 0.65, 0.75, 1.0);
}
