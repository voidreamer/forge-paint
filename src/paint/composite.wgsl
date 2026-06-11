// Compositor — per-tile fullscreen pass that samples one source layer and
// OVER-blends its per-channel textures into the display PaintTarget's four
// MRT outputs (base_color sRGB, roughness R8, metallic R8, normal Rgba8).

struct LayerParams {
    opacity: f32,
    // ChannelMask bits: bit0 base_color, bit1 roughness, bit2 metallic,
    // bit3 normal. A cleared bit zeroes that channel's coverage so the
    // layers below show through.
    affects: u32,
    // 8 bytes of trailing padding implicit (struct aligned to 16)
}

@group(0) @binding(0) var<uniform> params: LayerParams;
@group(0) @binding(1) var base_color_tex: texture_2d<f32>;
@group(0) @binding(2) var roughness_tex: texture_2d<f32>;
@group(0) @binding(3) var metallic_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;
@group(0) @binding(5) var mask_tex: texture_2d<f32>;      // R8, 1.0 = visible
@group(0) @binding(6) var src_sampler: sampler;

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
    // Top-left-origin UV, matches the brush-stamp convention so paint lands
    // pixel-for-pixel into the display target.
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

struct FsOut {
    @location(0) base_color: vec4<f32>,
    @location(1) roughness: vec4<f32>,
    @location(2) metallic: vec4<f32>,
    @location(3) normal: vec4<f32>,
};

@fragment
fn fs_composite(in: VsOut) -> FsOut {
    let bc = textureSample(base_color_tex, src_sampler, in.uv);
    let r = textureSample(roughness_tex, src_sampler, in.uv).r;
    let m = textureSample(metallic_tex, src_sampler, in.uv).r;
    let nm = textureSample(normal_tex, src_sampler, in.uv);
    let mk = textureSample(mask_tex, src_sampler, in.uv).r;
    let a = clamp(params.opacity * mk, 0.0, 1.0);
    // Per-channel coverage from the layer's channel mask. A disabled
    // channel emits (0, 0) premultiplied, which every shipped blend mode
    // resolves to "dst unchanged".
    let a_bc = a * f32(params.affects & 1u);
    let a_r = a * f32((params.affects >> 1u) & 1u);
    let a_m = a * f32((params.affects >> 2u) & 1u);
    let a_n = a * f32((params.affects >> 3u) & 1u);
    var out: FsOut;
    // Premultiplied-alpha over-blend is set as the pipeline's per-target
    // blend state; we emit (rgb*a, a) for each output. Single-channel
    // outputs still use the R channel; G is the coverage term.
    out.base_color = vec4<f32>(bc.rgb * a_bc, a_bc);
    out.roughness = vec4<f32>(r * a_r, a_r, 0.0, a_r);
    out.metallic = vec4<f32>(m * a_m, a_m, 0.0, a_m);
    out.normal = vec4<f32>(nm.rgb * a_n, a_n);
    return out;
}
