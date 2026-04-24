// Fullscreen gradient background for the HDR pass. Drawn before the
// mesh (and before the skybox when visible) so mesh pixels cleanly
// overwrite it. Output is linear HDR because the HDR target is
// Rgba16Float — values here get exposure + tonemap applied downstream.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_bg(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    // Far plane so LessEqual depth tests from skybox / mesh still pass.
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 1.0, 1.0);
    // uv.y = 0 at screen top, = 1 at bottom — matches post.wgsl.
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_bg(in: VsOut) -> @location(0) vec4<f32> {
    // Cool slate at the top, warmer graphite toward the bottom. Values
    // tuned for the Filmic tonemap default so the output reads as a
    // rich "studio" dark rather than flat gray.
    let top = vec3<f32>(0.030, 0.035, 0.055);
    let bot = vec3<f32>(0.075, 0.065, 0.060);
    let t = smoothstep(0.0, 1.0, in.uv.y);
    var color = mix(top, bot, t);

    // Subtle warm accent in the upper-right quadrant — adds character
    // without pulling focus from the subject.
    let accent = vec2<f32>(0.72, 0.28);
    let ad = distance(in.uv, accent);
    let glow = smoothstep(0.9, 0.0, ad);
    color = color + vec3<f32>(0.08, 0.04, 0.02) * glow * 0.35;

    // Corner vignette — 30% darker at the edges.
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let v = smoothstep(1.05, 0.35, d);
    color = color * mix(0.65, 1.0, v);

    return vec4<f32>(color, 1.0);
}
