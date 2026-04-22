// Compositor — per-tile fullscreen pass that samples one source layer and
// OVER-blends its per-channel textures into the display PaintTarget's three
// MRT outputs (base_color sRGB, rough_metal unorm, normal unorm).

struct LayerParams {
    opacity: f32,
    // 12 bytes of trailing padding implicit (struct aligned to 16)
}

@group(0) @binding(0) var<uniform> params: LayerParams;
@group(0) @binding(1) var base_color_tex: texture_2d<f32>;
@group(0) @binding(2) var rough_metal_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var mask_tex: texture_2d<f32>;      // R8, 1.0 = visible
@group(0) @binding(5) var src_sampler: sampler;

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
    @location(1) rough_metal: vec4<f32>,
    @location(2) normal: vec4<f32>,
};

@fragment
fn fs_composite(in: VsOut) -> FsOut {
    let bc = textureSample(base_color_tex, src_sampler, in.uv);
    let rm = textureSample(rough_metal_tex, src_sampler, in.uv);
    let nm = textureSample(normal_tex, src_sampler, in.uv);
    let mk = textureSample(mask_tex, src_sampler, in.uv).r;
    let a = clamp(params.opacity * mk, 0.0, 1.0);
    var out: FsOut;
    // Premultiplied-alpha over-blend is set as the pipeline's per-target
    // blend state; we emit (rgb*a, a) for each output.
    out.base_color = vec4<f32>(bc.rgb * a, a);
    out.rough_metal = vec4<f32>(rm.rgb * a, a);
    out.normal = vec4<f32>(nm.rgb * a, a);
    return out;
}
