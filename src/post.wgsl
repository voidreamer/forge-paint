// Post-process pass: reads the HDR linear color from the PBR pass, applies
// exposure + tonemap (only in Material view mode; inspection modes pass
// through), writes to the Bgra8UnormSrgb viewport texture that egui samples.

struct PostUniforms {
    /// `2^exposure_stops` — already precomputed on the CPU side.
    exposure: f32,
    /// See ViewMode::as_u32 in render.rs.
    view_mode: u32,
    /// See TonemapMode::as_u32 in render.rs.
    tonemap_mode: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> post: PostUniforms;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle.
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    // Texture origin is top-left; clip +Y is up. Flip V to match.
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// --- Tonemap curves (identical to pbr.wgsl's — kept in sync manually) ---

fn aces_narkowicz(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmic_uc2(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    let W = 11.2;
    let curr = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    let white = vec3<f32>(W);
    let white_scale = ((white * (A * white + C * B) + D * E)
        / (white * (A * white + B) + D * F))
        - E / F;
    return clamp(curr / white_scale, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_tonemap(x: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: {
            return x / (x + vec3<f32>(1.0));
        }
        case 2u: {
            return aces_narkowicz(x);
        }
        case 3u: {
            return filmic_uc2(x);
        }
        default: {
            return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
}

@fragment
fn fs_post(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    // Inspection views: pass the channel value through directly. Tonemap
    // and AO would both misrepresent e.g. a stored roughness of 0.5.
    if post.view_mode != 0u {
        return vec4<f32>(hdr, 1.0);
    }
    let exposed = hdr * post.exposure;
    let tonemapped = apply_tonemap(exposed, post.tonemap_mode);
    return vec4<f32>(tonemapped, 1.0);
}
