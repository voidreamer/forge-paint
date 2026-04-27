// Jump Flooding Algorithm for UV dilation.
// Each pass, every pixel checks neighbors at distance `step_size` and copies
// the nearest valid pixel's color. After log2(max_dim) passes, all pixels are filled.

struct Params {
    width: u32,
    height: u32,
    step_size: u32,
    num_channels: u32, // 1 = grayscale, 3 = RGB
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<storage, read> input_mask: array<u32>; // 1 = valid, 0 = empty
@group(0) @binding(4) var<storage, read_write> output_mask: array<u32>;

fn pixel_idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn sample_valid(x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || u32(x) >= params.width || u32(y) >= params.height {
        return false;
    }
    return input_mask[pixel_idx(u32(x), u32(y))] != 0u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let idx = pixel_idx(x, y);
    let nc = params.num_channels;
    let step = i32(params.step_size);

    // If already valid, just copy through
    if input_mask[idx] != 0u {
        for (var c: u32 = 0u; c < nc; c++) {
            output_data[idx * nc + c] = input_data[idx * nc + c];
        }
        output_mask[idx] = 1u;
        return;
    }

    // Check 8 neighbors at step_size distance + center
    let ix = i32(x);
    let iy = i32(y);

    var best_dist: f32 = 1e10;
    var best_idx: u32 = idx;
    var found: bool = false;

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nx = ix + dx * step;
            let ny = iy + dy * step;

            if sample_valid(nx, ny) {
                let dist = f32((ix - nx) * (ix - nx) + (iy - ny) * (iy - ny));
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = pixel_idx(u32(nx), u32(ny));
                    found = true;
                }
            }
        }
    }

    if found {
        for (var c: u32 = 0u; c < nc; c++) {
            output_data[idx * nc + c] = input_data[best_idx * nc + c];
        }
        output_mask[idx] = 1u;
    } else {
        for (var c: u32 = 0u; c < nc; c++) {
            output_data[idx * nc + c] = input_data[idx * nc + c];
        }
        output_mask[idx] = 0u;
    }
}
