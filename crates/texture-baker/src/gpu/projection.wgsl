// Projection ray casting: for each texel, cast a ray from cage/inflated position toward
// the high-poly mesh and record the closest hit (t, u, v, tri_idx).

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
    frontal_distance: f32,
    rear_distance: f32,
    total_texels: u32,
    node_count: u32,
    workgroups_x: u32,
    ignore_backface: u32,
    min_t: f32,    // minimum hit distance to reject self-intersection
    _pad1: u32,
};

// Hit result: t, u, v, tri_idx (packed as 4 floats for simplicity)
// tri_idx stored as float via bitcast; -1.0 means no hit.
struct HitResult {
    t: f32,
    u: f32,
    v: f32,
    tri_idx_f: f32, // bitcast<f32>(tri_idx), or bitcast<f32>(0xFFFFFFFF) for miss
};

@group(0) @binding(0) var<storage, read> texels: array<Texel>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> hits: array<HitResult>;

const INVALID: u32 = 0xFFFFFFFFu;

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>, max_t: f32) -> bool {
    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    return tmax >= max(tmin, 0.0) && tmin < max_t;
}

fn intersect_triangle(origin: vec3<f32>, dir: vec3<f32>, tri_idx: u32) -> vec4<f32> {
    // Returns (t, u, v, backface_flag). t < 0 means miss.
    let tri = triangles[tri_idx];
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);

    let is_backface = select(0.0, 1.0, a < 0.0);

    if abs(a) < 1e-8 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let t = f * dot(edge2, q);
    return vec4<f32>(t, u, v, is_backface);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * params.workgroups_x * 64u + global_id.x;
    if idx >= params.total_texels {
        return;
    }

    let texel = texels[idx];
    if length(texel.normal) < 0.5 {
        hits[idx] = HitResult(-1.0, 0.0, 0.0, bitcast<f32>(INVALID));
        return;
    }

    let origin = texel.position + texel.normal * params.frontal_distance;
    let dir = -texel.normal;
    let max_t = params.frontal_distance + params.rear_distance;
    let inv_dir = 1.0 / dir;

    var best_t: f32 = max_t + 1.0;
    var best_u: f32 = 0.0;
    var best_v: f32 = 0.0;
    var best_tri: u32 = INVALID;

    var stack: array<u32, 32>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;

    while sp > 0 {
        sp -= 1;
        let node_idx = stack[sp];

        if node_idx == INVALID || node_idx >= params.node_count {
            continue;
        }

        let node = bvh_nodes[node_idx];

        if !intersect_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, best_t) {
            continue;
        }

        if node.tri_idx != INVALID {
            let result = intersect_triangle(origin, dir, node.tri_idx);
            let t = result.x;
            let is_backface = result.w > 0.5;

            if t > params.min_t && t < best_t {
                if !(params.ignore_backface == 1u && is_backface) {
                    best_t = t;
                    best_u = result.y;
                    best_v = result.z;
                    best_tri = node.tri_idx;
                }
            }
        } else {
            if node.left_child != INVALID && sp < 31 {
                stack[sp] = node.left_child;
                sp += 1;
            }
            if node.right_child != INVALID && sp < 31 {
                stack[sp] = node.right_child;
                sp += 1;
            }
        }
    }

    if best_tri != INVALID {
        hits[idx] = HitResult(best_t, best_u, best_v, bitcast<f32>(best_tri));
    } else {
        hits[idx] = HitResult(-1.0, 0.0, 0.0, bitcast<f32>(INVALID));
    }
}
