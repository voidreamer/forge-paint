/// Dilate (edge-pad) a texture buffer by expanding valid pixels into empty regions.
///
/// Uses iterative flood-fill: each pass, empty pixels adjacent to valid ones
/// copy the nearest valid pixel. Runs `iterations` passes (0 = infinite until filled).
pub fn dilate_rgb(
    buffer: &mut Vec<[f32; 3]>,
    mask: &[bool],
    width: u32,
    height: u32,
    iterations: u32,
) {
    let w = width as usize;
    let h = height as usize;
    let mut current_mask = mask.to_vec();
    let max_iters = if iterations == 0 { w.max(h) as u32 } else { iterations };

    for _ in 0..max_iters {
        let mut new_pixels: Vec<(usize, [f32; 3])> = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if current_mask[idx] {
                    continue; // already filled
                }

                // Check 4-connected neighbors
                let mut sum = [0.0f32; 3];
                let mut count = 0u32;

                if x > 0 && current_mask[idx - 1] {
                    let n = buffer[idx - 1];
                    sum[0] += n[0]; sum[1] += n[1]; sum[2] += n[2];
                    count += 1;
                }
                if x + 1 < w && current_mask[idx + 1] {
                    let n = buffer[idx + 1];
                    sum[0] += n[0]; sum[1] += n[1]; sum[2] += n[2];
                    count += 1;
                }
                if y > 0 && current_mask[idx - w] {
                    let n = buffer[idx - w];
                    sum[0] += n[0]; sum[1] += n[1]; sum[2] += n[2];
                    count += 1;
                }
                if y + 1 < h && current_mask[idx + w] {
                    let n = buffer[idx + w];
                    sum[0] += n[0]; sum[1] += n[1]; sum[2] += n[2];
                    count += 1;
                }

                if count > 0 {
                    let inv = 1.0 / count as f32;
                    new_pixels.push((idx, [sum[0] * inv, sum[1] * inv, sum[2] * inv]));
                }
            }
        }

        if new_pixels.is_empty() {
            break; // nothing left to fill
        }

        for (idx, color) in new_pixels {
            buffer[idx] = color;
            current_mask[idx] = true;
        }
    }
}

/// Dilate a single-channel (grayscale) buffer.
pub fn dilate_gray(
    buffer: &mut Vec<f32>,
    mask: &[bool],
    width: u32,
    height: u32,
    iterations: u32,
) {
    let w = width as usize;
    let h = height as usize;
    let mut current_mask = mask.to_vec();
    let max_iters = if iterations == 0 { w.max(h) as u32 } else { iterations };

    for _ in 0..max_iters {
        let mut new_pixels: Vec<(usize, f32)> = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if current_mask[idx] {
                    continue;
                }

                let mut sum = 0.0f32;
                let mut count = 0u32;

                if x > 0 && current_mask[idx - 1] {
                    sum += buffer[idx - 1];
                    count += 1;
                }
                if x + 1 < w && current_mask[idx + 1] {
                    sum += buffer[idx + 1];
                    count += 1;
                }
                if y > 0 && current_mask[idx - w] {
                    sum += buffer[idx - w];
                    count += 1;
                }
                if y + 1 < h && current_mask[idx + w] {
                    sum += buffer[idx + w];
                    count += 1;
                }

                if count > 0 {
                    new_pixels.push((idx, sum / count as f32));
                }
            }
        }

        if new_pixels.is_empty() {
            break;
        }

        for (idx, val) in new_pixels {
            buffer[idx] = val;
            current_mask[idx] = true;
        }
    }
}
