// Compact FXAA-style fullscreen edge smoothing, adapted for wgpu.
// Based on FXAA 3.11 (Timothy Lottes, 2011). Operates on the tonemapped
// LDR image — samples luminance in a 3x3 neighborhood, detects edges,
// and blurs along the dominant edge direction.

struct FxaaUniforms {
    /// Enable/disable toggle — 0 passes the center texel through
    /// untouched so the shader is always safe to include in the chain.
    enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> fxaa: FxaaUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fxaa(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

const EDGE_THRESHOLD: f32 = 0.0312;
const EDGE_THRESHOLD_MIN: f32 = 0.0078;
const SUBPIXEL_QUALITY: f32 = 0.75;
const ITERATIONS: i32 = 12;
const QUALITY_STEPS: array<f32, 12> = array<f32, 12>(
    1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0,
);

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_tex));
    let px = 1.0 / tex_size;

    let center = textureSample(src_tex, src_sampler, in.uv);
    if fxaa.enabled == 0u {
        return center;
    }

    // Luminance at cross neighbors.
    let l_c = luma(center.rgb);
    let l_n = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0, -px.y)).rgb);
    let l_s = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0,  px.y)).rgb);
    let l_e = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>( px.x, 0.0)).rgb);
    let l_w = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-px.x, 0.0)).rgb);

    let l_min = min(l_c, min(min(l_n, l_s), min(l_e, l_w)));
    let l_max = max(l_c, max(max(l_n, l_s), max(l_e, l_w)));
    let l_range = l_max - l_min;

    // Early-out for low-contrast pixels — not an edge, pass through.
    if l_range < max(EDGE_THRESHOLD_MIN, l_max * EDGE_THRESHOLD) {
        return center;
    }

    // Diagonal neighbors for richer edge orientation detection.
    let l_nw = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-px.x, -px.y)).rgb);
    let l_ne = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>( px.x, -px.y)).rgb);
    let l_sw = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-px.x,  px.y)).rgb);
    let l_se = luma(textureSample(src_tex, src_sampler, in.uv + vec2<f32>( px.x,  px.y)).rgb);

    // Horizontal vs vertical edge detection — compare cross sums.
    let ns = l_n + l_s;
    let ew = l_e + l_w;
    let corners_top = l_nw + l_ne;
    let corners_bot = l_sw + l_se;
    let corners_l = l_nw + l_sw;
    let corners_r = l_ne + l_se;

    let edge_horz = abs(corners_top - 2.0 * l_n) + 2.0 * abs(ns - 2.0 * l_c) + abs(corners_bot - 2.0 * l_s);
    let edge_vert = abs(corners_l   - 2.0 * l_w) + 2.0 * abs(ew - 2.0 * l_c) + abs(corners_r   - 2.0 * l_e);
    let is_horz = edge_horz >= edge_vert;

    // Step size along the edge normal (perpendicular to the edge).
    let step_size = select(px.x, px.y, is_horz);

    // Which side of the edge has the larger luminance gradient?
    let l1 = select(l_w, l_n, is_horz);
    let l2 = select(l_e, l_s, is_horz);
    let grad1 = l1 - l_c;
    let grad2 = l2 - l_c;
    let is_1_steeper = abs(grad1) >= abs(grad2);
    let grad_scaled = 0.25 * max(abs(grad1), abs(grad2));

    let luma_local_avg = select(
        0.5 * (l_c + l2),
        0.5 * (l_c + l1),
        is_1_steeper,
    );

    var uv = in.uv;
    // Shift half a pixel toward the darker side so subsequent taps
    // straddle the edge.
    if is_horz {
        uv.y = uv.y + select(step_size * 0.5, -step_size * 0.5, is_1_steeper);
    } else {
        uv.x = uv.x + select(step_size * 0.5, -step_size * 0.5, is_1_steeper);
    }

    // Walk along the edge in both directions, accumulating luminance
    // deltas until we hit an endpoint.
    var dir = vec2<f32>(0.0);
    if is_horz {
        dir.x = px.x;
    } else {
        dir.y = px.y;
    }
    var uv1 = uv - dir;
    var uv2 = uv + dir;

    var done1 = false;
    var done2 = false;
    var delta1 = 0.0;
    var delta2 = 0.0;

    for (var i: i32 = 0; i < ITERATIONS; i = i + 1) {
        if !done1 {
            let s = textureSample(src_tex, src_sampler, uv1).rgb;
            delta1 = luma(s) - luma_local_avg;
        }
        if !done2 {
            let s = textureSample(src_tex, src_sampler, uv2).rgb;
            delta2 = luma(s) - luma_local_avg;
        }
        done1 = abs(delta1) >= grad_scaled;
        done2 = abs(delta2) >= grad_scaled;
        if done1 && done2 {
            break;
        }
        let q = QUALITY_STEPS[i];
        if !done1 {
            uv1 = uv1 - dir * q;
        }
        if !done2 {
            uv2 = uv2 + dir * q;
        }
    }

    // Edge distances in pixels along the search direction.
    var dist1: f32;
    var dist2: f32;
    if is_horz {
        dist1 = uv.x - uv1.x;
        dist2 = uv2.x - uv.x;
    } else {
        dist1 = uv.y - uv1.y;
        dist2 = uv2.y - uv.y;
    }
    let is_dir1_shorter = dist1 < dist2;
    let dist_final = min(dist1, dist2);
    let edge_length = dist1 + dist2;
    let pixel_offset = -dist_final / edge_length + 0.5;

    // If the center is on the darker side, only shift inward if we're
    // actually within the edge span — otherwise pass through.
    let is_center_below = l_c < luma_local_avg;
    let is_endpoint_below = select(delta2, delta1, is_dir1_shorter) < 0.0;
    let is_correct_side = is_center_below != is_endpoint_below;
    let final_offset = select(0.0, pixel_offset, is_correct_side);

    // Subpixel antialiasing using luminance of the 3x3 neighborhood.
    let avg = (1.0 / 12.0)
        * (2.0 * (ns + ew) + l_nw + l_ne + l_sw + l_se);
    let subpix_offset1 = clamp(abs(avg - l_c) / l_range, 0.0, 1.0);
    let subpix_offset2 = (-2.0 * subpix_offset1 + 3.0) * subpix_offset1 * subpix_offset1;
    let subpix_final = subpix_offset2 * subpix_offset2 * SUBPIXEL_QUALITY;

    let offset = max(final_offset, subpix_final);

    var final_uv = in.uv;
    if is_horz {
        final_uv.y = final_uv.y + offset * step_size * select(1.0, -1.0, is_1_steeper);
    } else {
        final_uv.x = final_uv.x + offset * step_size * select(1.0, -1.0, is_1_steeper);
    }
    return textureSample(src_tex, src_sampler, final_uv);
}
