// Mesh-map bake: world-space normal. Per-tile UDIM pass — vertex shader maps
// the mesh UV (minus the tile offset) into clip space and the fragment writes
// the world-space normal, 0.5-biased into unorm-friendly range.

struct BakeParams {
    tile_u: f32,
    tile_v: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: BakeParams;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_bake(in: VsIn) -> VsOut {
    // Local UV within the tile, remapped to [-1, 1] clip space.
    let local_u = in.uv.x - params.tile_u;
    let local_v = in.uv.y - params.tile_v;
    // V-flip matches the brush/composite convention so this map aligns with
    // painted content pixel-for-pixel on the same texture coordinates.
    let clip_x = local_u * 2.0 - 1.0;
    let clip_y = (1.0 - local_v) * 2.0 - 1.0;
    var out: VsOut;
    out.clip_pos = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.world_normal = in.normal;
    return out;
}

@fragment
fn fs_bake(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    // Bake as 0..1 unorm-style; PBR preview remaps back if needed.
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
}
