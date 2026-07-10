/// Downsample an RGB buffer from (w*factor, h*factor) to (w, h) using box filter.
pub fn downsample_rgb(
    buffer: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    factor: u32,
) -> Vec<[f32; 3]> {
    let dst_width = src_width / factor;
    let dst_height = src_height / factor;
    let mut result = vec![[0.0f32; 3]; (dst_width * dst_height) as usize];
    let inv = 1.0 / (factor * factor) as f32;

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let mut sum = [0.0f32; 3];
            for sy in 0..factor {
                for sx in 0..factor {
                    let src_idx = ((dy * factor + sy) * src_width + dx * factor + sx) as usize;
                    let c = buffer[src_idx];
                    sum[0] += c[0];
                    sum[1] += c[1];
                    sum[2] += c[2];
                }
            }
            let dst_idx = (dy * dst_width + dx) as usize;
            result[dst_idx] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
        }
    }

    result
}

/// Downsample a grayscale buffer from (w*factor, h*factor) to (w, h) using box filter.
pub fn downsample_gray(buffer: &[f32], src_width: u32, src_height: u32, factor: u32) -> Vec<f32> {
    let dst_width = src_width / factor;
    let dst_height = src_height / factor;
    let mut result = vec![0.0f32; (dst_width * dst_height) as usize];
    let inv = 1.0 / (factor * factor) as f32;

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let mut sum = 0.0f32;
            for sy in 0..factor {
                for sx in 0..factor {
                    let src_idx = ((dy * factor + sy) * src_width + dx * factor + sx) as usize;
                    sum += buffer[src_idx];
                }
            }
            let dst_idx = (dy * dst_width + dx) as usize;
            result[dst_idx] = sum * inv;
        }
    }

    result
}

/// Downsample a boolean mask — a texel is valid if any source texel was valid.
pub fn downsample_mask(mask: &[bool], src_width: u32, src_height: u32, factor: u32) -> Vec<bool> {
    let dst_width = src_width / factor;
    let dst_height = src_height / factor;
    let mut result = vec![false; (dst_width * dst_height) as usize];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let mut any = false;
            for sy in 0..factor {
                for sx in 0..factor {
                    let src_idx = ((dy * factor + sy) * src_width + dx * factor + sx) as usize;
                    if mask[src_idx] {
                        any = true;
                        break;
                    }
                }
                if any {
                    break;
                }
            }
            let dst_idx = (dy * dst_width + dx) as usize;
            result[dst_idx] = any;
        }
    }

    result
}
