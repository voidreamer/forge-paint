// Projection brush — samples the baked world-position map at each tile
// UV, projects that world point to screen via the current camera's
// view_proj, and deposits the stencil's color at that screen location.
// Brush falloff is applied in screen-space so the stamp feels like a
// circular "reveal" of the projected texture.

struct ProjBrush {
    view_proj: mat4x4<f32>,
    center_screen: vec2<f32>,
    radius_screen: f32,
    opacity: f32,
    hardness: f32,
    aspect: f32,
    stencil_offset: vec2<f32>,
    stencil_scale: f32,
    stencil_cos_rot: f32,
    stencil_sin_rot: f32,
    stencil_aspect: f32,
    /// 0 = paint base color. 1 = paint displacement (output packed as
    /// (height × coverage, coverage) into Rg16Float).
    mode: u32,
};

@group(0) @binding(0) var<uniform> brush: ProjBrush;
@group(0) @binding(1) var position_tex: texture_2d<f32>;
@group(0) @binding(2) var map_sampler: sampler;
@group(0) @binding(3) var stencil_tex: texture_2d<f32>;
@group(0) @binding(4) var stencil_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_project(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@fragment
fn fs_project(in: VsOut) -> @location(0) vec4<f32> {
    // Sample the world-position map at this tile UV.
    let wp4 = textureSampleLevel(position_tex, map_sampler, in.uv, 0.0);
    let world_pos = wp4.xyz;
    // Unbaked / outside-UV texels are (0,0,0) — skip them. Coincidental
    // zero hits on real meshes round-trip through projection fine; the
    // false-positive skip here is acceptable.
    if all(world_pos == vec3<f32>(0.0)) { discard; }

    // Project world → clip → NDC.
    let clip = brush.view_proj * vec4<f32>(world_pos, 1.0);
    // Behind the camera? clip.w ≤ 0 maps outside the hemisphere.
    if clip.w <= 0.0 { discard; }
    let ndc = clip.xyz / clip.w;
    // Off-screen in either axis — nothing to sample.
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z > 1.0 {
        discard;
    }

    // Is the screen point within the brush's circular footprint?
    // Aspect-correct distance so circles stay round on non-square viewports.
    let d_aspect = vec2<f32>((ndc.x - brush.center_screen.x) * brush.aspect,
                              ndc.y - brush.center_screen.y);
    let d = length(d_aspect);
    if d > brush.radius_screen { discard; }

    // NDC → stencil UV, with the inverse of the stencil transform applied
    // in aspect-corrected (isotropic) space so the rotation looks like a
    // clean 2D spin on screen instead of an axis-dependent shear.
    //   translate: s = ndc - offset
    //   isotropic: iso = (s.x * viewport_aspect, s.y)
    //   rotate by -θ:  rotated = R(-θ) · iso
    //   unscale (with stencil aspect): local_x = rotated.x / (scale * stencil_aspect)
    //                                   local_y = rotated.y / scale
    let s = ndc.xy - brush.stencil_offset;
    let iso = vec2<f32>(s.x * brush.aspect, s.y);
    let cr = brush.stencil_cos_rot;
    let sr = brush.stencil_sin_rot;
    let rotated = vec2<f32>(iso.x * cr + iso.y * sr, -iso.x * sr + iso.y * cr);
    let local_x = rotated.x / max(brush.stencil_scale * brush.stencil_aspect, 1e-4);
    let local_y = rotated.y / max(brush.stencil_scale, 1e-4);
    if abs(local_x) > 1.0 || abs(local_y) > 1.0 { discard; }
    let stencil_uv = vec2<f32>(local_x * 0.5 + 0.5, 0.5 - local_y * 0.5);
    let stencil = textureSample(stencil_tex, stencil_sampler, stencil_uv);

    // Radial falloff on the brush circle, same shape as the regular brush.
    let t = d / max(brush.radius_screen, 1e-6);
    let inner = clamp(brush.hardness, 0.0, 0.95);
    let falloff = 1.0 - smoothstep(inner, 1.0, t);
    let a = falloff * brush.opacity * stencil.a;
    if brush.mode == 1u {
        // Displacement path — convert stencil to a scalar height
        // (Rec.709 luminance) and pack premultiplied (h·a, a) into
        // RG so the Rg16Float accumulator stays consistent with the
        // regular displacement brush.
        let height = dot(stencil.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        return vec4<f32>(height * a, a, 0.0, a);
    }
    return vec4<f32>(stencil.rgb * a, a);
}
