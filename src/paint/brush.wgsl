// Brush stamp: fullscreen triangle, radial falloff, premultiplied-alpha output.
// Render target is the single-layer view of the UDIM tile we're stamping into.

struct Brush {
    color: vec4<f32>,        // linear rgb + alpha (opacity)
    center_uv: vec2<f32>,    // local UV [0,1] within the tile
    radius: f32,             // in local UV units
    hardness: f32,           // 0 = soft, 1 = hard
    uniform_fill: u32,       // 1 = bypass radius/falloff (Fill tool)
}

@group(0) @binding(0) var<uniform> brush: Brush;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One fullscreen triangle that covers the viewport in clip space.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    // Texture origin is top-left; clip-space Y is up. Flip V.
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_stamp(in: VsOut) -> @location(0) vec4<f32> {
    if brush.uniform_fill == 1u {
        // Full-tile flood fill — ignore radius/hardness, use opacity directly.
        let a = brush.color.a;
        return vec4<f32>(brush.color.rgb * a, a);
    }
    let d = distance(in.uv, brush.center_uv);
    if d > brush.radius { discard; }
    let t = d / max(brush.radius, 1e-6);
    let inner = clamp(brush.hardness, 0.0, 0.95);
    // falloff: 1 at center, smoothly to 0 at outer edge
    let falloff = 1.0 - smoothstep(inner, 1.0, t);
    let a = falloff * brush.color.a;
    // Premultiplied alpha
    return vec4<f32>(brush.color.rgb * a, a);
}
