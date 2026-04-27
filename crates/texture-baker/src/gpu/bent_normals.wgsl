// Bent normals compute shader: accumulate unoccluded ray directions per texel.
// Output is RGB (world-space bent normal encoded to [0,1]).

struct Texel {
    position: vec3<f32>,
    normal_x: f32,
    normal: vec3<f32>,
    _pad: f32,
};

struct BvhNode {
    aabb_min: vec3<f32>,
    left_child: u32,
    aabb_max: vec3<f32>,
    right_child: u32,
    tri_idx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Triangle {
    v0: vec3<f32>,
    _pad0: f32,
    v1: vec3<f32>,
    _pad1: f32,
    v2: vec3<f32>,
    _pad2: f32,
};

struct Params {
    ray_count: u32,
    max_distance: f32,
    bias: f32,
    total_texels: u32,
    node_count: u32,
    _pad0: u32,
    workgroups_x: u32,
    _pad1: u32,
    spread_angle: f32,
    distribution: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> texels: array<Texel>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // 3 floats per texel (r,g,b)

const INVALID: u32 = 0xFFFFFFFFu;

fn radical_inverse(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
    if n.z < -0.9999999 {
        return mat3x3<f32>(
            vec3<f32>(0.0, -1.0, 0.0),
            vec3<f32>(-1.0, 0.0, 0.0),
            n
        );
    }
    let a = 1.0 / (1.0 + n.z);
    let b = -n.x * n.y * a;
    return mat3x3<f32>(
        vec3<f32>(1.0 - n.x * n.x * a, b, -n.x),
        vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y),
        n
    );
}

fn hemisphere_sample(normal: vec3<f32>, i: u32, total: u32, seed: u32) -> vec3<f32> {
    let idx = i + seed;
    let xi1 = f32(idx) / f32(total);
    let xi2 = radical_inverse(idx);
    let phi = 2.0 * 3.14159265 * xi1;

    // Clamp spread to [0, 180] and convert to cosine of half-angle
    let half_angle = clamp(params.spread_angle, 0.0, 180.0) * 0.5 * 3.14159265 / 180.0;
    let cos_max = cos(half_angle);

    var cos_theta: f32;
    var sin_theta: f32;
    if params.distribution == 0u {
        // Cosine-weighted, narrowed to cone
        let remapped = 1.0 - xi2 * (1.0 - cos_max * cos_max);
        cos_theta = sqrt(remapped);
        sin_theta = sqrt(1.0 - remapped);
    } else {
        // Uniform within cone
        cos_theta = 1.0 - xi2 * (1.0 - cos_max);
        sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    }

    let basis = build_basis(normal);
    return normalize(basis * vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
}

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>, max_t: f32) -> bool {
    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    return tmax >= max(tmin, 0.0) && tmin < max_t;
}

fn intersect_triangle(origin: vec3<f32>, dir: vec3<f32>, tri_idx: u32) -> vec2<f32> {
    let tri = triangles[tri_idx];
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if abs(a) < 1e-7 { return vec2<f32>(-1.0, 0.0); }
    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 { return vec2<f32>(-1.0, 0.0); }
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 { return vec2<f32>(-1.0, 0.0); }
    let t = f * dot(edge2, q);
    let is_backface = select(0.0, 1.0, a < 0.0);
    return vec2<f32>(t, is_backface);
}

fn trace_any_hit(origin: vec3<f32>, dir: vec3<f32>, max_t: f32, min_t: f32) -> bool {
    let inv_dir = 1.0 / dir;
    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    while sp > 0 {
        sp -= 1;
        let node_idx = stack[sp];
        if node_idx == INVALID || node_idx >= params.node_count { continue; }
        let node = bvh_nodes[node_idx];
        if !intersect_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, max_t) { continue; }
        if node.tri_idx != INVALID {
            let result = intersect_triangle(origin, dir, node.tri_idx);
            let t = result.x;
            let is_backface = result.y > 0.5;
            if t > min_t && t < max_t && !is_backface { return true; }
        } else {
            if node.left_child != INVALID && sp < 31 { stack[sp] = node.left_child; sp += 1; }
            if node.right_child != INVALID && sp < 31 { stack[sp] = node.right_child; sp += 1; }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * params.workgroups_x * 64u + global_id.x;
    if idx >= params.total_texels {
        return;
    }

    let texel = texels[idx];
    if length(texel.normal) < 0.5 {
        output[idx * 3u + 0u] = 0.5;
        output[idx * 3u + 1u] = 0.5;
        output[idx * 3u + 2u] = 1.0;
        return;
    }

    let origin = texel.position + texel.normal * params.bias;
    let min_t = params.bias;
    var bent = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < params.ray_count; i++) {
        let dir = hemisphere_sample(texel.normal, i, params.ray_count, idx);
        if !trace_any_hit(origin, dir, params.max_distance, min_t) {
            bent += dir;
        }
    }

    let bent_len = length(bent);
    var result: vec3<f32>;
    if bent_len > 1e-8 {
        result = normalize(bent);
    } else {
        result = texel.normal;
    }

    // Encode to [0, 1]
    output[idx * 3u + 0u] = result.x * 0.5 + 0.5;
    output[idx * 3u + 1u] = result.y * 0.5 + 0.5;
    output[idx * 3u + 2u] = result.z * 0.5 + 0.5;
}
