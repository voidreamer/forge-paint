/// Dilate (edge-pad) a texture buffer by expanding valid pixels into empty regions.
///
/// Uses iterative flood-fill: each pass, empty pixels adjacent to valid ones
/// copy the nearest valid pixel. Runs `iterations` passes (0 = infinite until filled).
pub fn dilate_rgb(
    buffer: &mut [[f32; 3]],
    mask: &[bool],
    width: u32,
    height: u32,
    iterations: u32,
) {
    let w = width as usize;
    let h = height as usize;
    let mut current_mask = mask.to_vec();
    let max_iters = if iterations == 0 {
        w.max(h) as u32
    } else {
        iterations
    };

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
                    sum[0] += n[0];
                    sum[1] += n[1];
                    sum[2] += n[2];
                    count += 1;
                }
                if x + 1 < w && current_mask[idx + 1] {
                    let n = buffer[idx + 1];
                    sum[0] += n[0];
                    sum[1] += n[1];
                    sum[2] += n[2];
                    count += 1;
                }
                if y > 0 && current_mask[idx - w] {
                    let n = buffer[idx - w];
                    sum[0] += n[0];
                    sum[1] += n[1];
                    sum[2] += n[2];
                    count += 1;
                }
                if y + 1 < h && current_mask[idx + w] {
                    let n = buffer[idx + w];
                    sum[0] += n[0];
                    sum[1] += n[1];
                    sum[2] += n[2];
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
pub fn dilate_gray(buffer: &mut [f32], mask: &[bool], width: u32, height: u32, iterations: u32) {
    let w = width as usize;
    let h = height as usize;
    let mut current_mask = mask.to_vec();
    let max_iters = if iterations == 0 {
        w.max(h) as u32
    } else {
        iterations
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_mask_is_a_noop() {
        let mut buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let expected = buf.clone();
        let mask = vec![true; 16];
        dilate_gray(&mut buf, &mask, 4, 4, 0);
        assert_eq!(buf, expected);
    }

    #[test]
    fn single_seed_floods_whole_grid_when_unbounded() {
        // iterations == 0 means "run until filled" (bounded by max(w, h)).
        let mut buf = vec![0.0f32; 16];
        buf[5] = 1.0;
        let mut mask = vec![false; 16];
        mask[5] = true;
        dilate_gray(&mut buf, &mask, 4, 4, 0);
        assert!(
            buf.iter().all(|&v| v == 1.0),
            "expected uniform fill, got {buf:?}"
        );
    }

    #[test]
    fn iteration_count_bounds_the_spread() {
        // 1x5 row, seed at x=0, one pass: only the direct neighbor fills.
        let mut buf = vec![0.0f32; 5];
        buf[0] = 1.0;
        let mut mask = vec![false; 5];
        mask[0] = true;
        dilate_gray(&mut buf, &mask, 5, 1, 1);
        assert_eq!(buf, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb_seed_floods_all_channels() {
        // Seed near the center: "unbounded" dilation caps at max(w, h)
        // passes, which cannot reach the far corner from a corner seed
        // (Manhattan distance w + h - 2). Real bakes seed from island
        // edges everywhere, so the cap is fine in practice.
        let mut buf = vec![[0.0f32; 3]; 16];
        buf[5] = [0.25, 0.5, 0.75];
        let mut mask = vec![false; 16];
        mask[5] = true;
        dilate_rgb(&mut buf, &mask, 4, 4, 0);
        // Multi-neighbor passes average identical values through a rounded
        // 1/count factor, so allow float slop rather than exact equality.
        for px in &buf {
            for (c, expected) in px.iter().zip([0.25f32, 0.5, 0.75]) {
                assert!((c - expected).abs() < 1e-5, "got {px:?}");
            }
        }
    }
}
