use image::{ImageBuffer, Luma, Rgb};
use std::path::Path;

/// Write an RGB float buffer as a PNG (8-bit per channel).
pub fn write_rgb_png(
    buffer: &[[f32; 3]],
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), String> {
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let idx = (y * width + x) as usize;
        let c = buffer[idx];
        Rgb([
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        ])
    });

    img.save(path)
        .map_err(|e| format!("Failed to write '{}': {e}", path.display()))
}

/// Write a grayscale float buffer as a PNG (8-bit).
pub fn write_gray_png(buffer: &[f32], width: u32, height: u32, path: &Path) -> Result<(), String> {
    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let idx = (y * width + x) as usize;
        Luma([(buffer[idx].clamp(0.0, 1.0) * 255.0) as u8])
    });

    img.save(path)
        .map_err(|e| format!("Failed to write '{}': {e}", path.display()))
}

/// Write an RGB float buffer as 32-bit EXR.
pub fn write_rgb_exr(
    buffer: &[[f32; 3]],
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), String> {
    use exr::prelude::*;

    let pixels: Vec<(f32, f32, f32)> = buffer.iter().map(|c| (c[0], c[1], c[2])).collect();

    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::named("default"),
        Encoding::SMALL_LOSSLESS,
        SpecificChannels::rgb(|pos: Vec2<usize>| pixels[pos.y() * width as usize + pos.x()]),
    );

    let image = Image::from_layer(layer);
    image
        .write()
        .to_file(path)
        .map_err(|e| format!("Failed to write EXR '{}': {e}", path.display()))
}
