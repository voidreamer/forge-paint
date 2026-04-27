// Smart-mask regenerator. One fullscreen triangle per tile, samples a
// baked map (one of the MeshMaps slots — AO / curvature / thickness /
// world normal), runs the threshold / falloff curve, writes R8.
//
// `source_kind` selects the per-source formula:
//   0  AO crevice          1 - smoothstep(low, high, ao)
//   1  Curvature convex    smoothstep(low, high, max(curv - 0.5, 0) * 2)
//   2  Curvature concave   smoothstep(low, high, max(0.5 - curv, 0) * 2)
//   3  Thickness           smoothstep(low, high, thickness)
//   4  World Y up          smoothstep(low, high, world_normal.g * 2 - 1)
//
// The bound source texture switches per-kind on the CPU side; the
// shader assumes scalar maps return the value in .r and world normal
// in .rgb.

struct SmartUniforms {
    low: f32,
    high: f32,
    contrast: f32,
    invert: u32,
    source_kind: u32,
    /// Which UDIM layer of the source D2Array to sample. Updated
    /// before each per-tile draw so a single source-binding can serve
    /// every tile of the destination mask.
    tile_layer: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: SmartUniforms;
@group(0) @binding(1) var src_tex: texture_2d_array<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_smart(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn smooth_step(low: f32, high: f32, x: f32) -> f32 {
    let t = clamp((x - low) / max(high - low, 1e-5), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fs_smart(in: VsOut) -> @location(0) vec4<f32> {
    // Source D2Array layer matches the destination tile — driven from
    // the CPU side via `u.tile_layer` (updated per draw).
    let s = textureSampleLevel(src_tex, src_sampler, in.uv, i32(u.tile_layer), 0.0);
    var t: f32;
    switch u.source_kind {
        case 0u: { // AO crevice
            t = 1.0 - smooth_step(u.low, u.high, s.r);
        }
        case 1u: { // Curvature convex
            let convex = max(s.r - 0.5, 0.0) * 2.0;
            t = smooth_step(u.low, u.high, convex);
        }
        case 2u: { // Curvature concave
            let concave = max(0.5 - s.r, 0.0) * 2.0;
            t = smooth_step(u.low, u.high, concave);
        }
        case 3u: { // Thickness
            t = smooth_step(u.low, u.high, s.r);
        }
        case 4u: { // World Y up — world_normal is encoded (0.5 + 0.5 n).
            let ny = s.g * 2.0 - 1.0;
            t = smooth_step(u.low, u.high, max(ny, 0.0));
        }
        default: {
            t = s.r;
        }
    }

    // Contrast around mid-gray. 1.0 = identity.
    t = clamp((t - 0.5) * u.contrast + 0.5, 0.0, 1.0);
    if u.invert == 1u {
        t = 1.0 - t;
    }
    return vec4<f32>(t, 0.0, 0.0, 1.0);
}
