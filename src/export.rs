//! Readback + PNG export of painted UDIM tiles.
//!
//! All copies are batched into a single encoder submission, all buffer maps
//! into one `device.poll(Wait)`, and PNG encoding is parallelised with rayon.

use anyhow::{anyhow, Context, Result};
use egui_wgpu::wgpu;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::paint::PaintTarget;

#[derive(Debug, Clone)]
pub struct Export {
    pub path: PathBuf,
    pub channel: &'static str,
    pub udim: u32,
}

struct Readback {
    buffer: wgpu::Buffer,
    path: PathBuf,
    channel: &'static str,
    udim: u32,
}

const CHANNELS: &[(&str, fn(&PaintTarget) -> &wgpu::Texture)] = &[
    ("basecolor", |p| &p.base_color),
    ("roughmetal", |p| &p.rough_metal),
    ("normal", |p| &p.normal),
];

/// Export every channel × every tile from `paint_target` into `dir` as PNGs.
pub fn export_tiles(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    paint_target: &PaintTarget,
    dir: &Path,
) -> Result<Vec<Export>> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create export dir {}", dir.display()))?;
    }
    if !dir.is_dir() {
        return Err(anyhow!("export path is not a directory: {}", dir.display()));
    }

    let resolution = paint_target.resolution;
    let bytes_per_row = resolution * 4;
    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    if bytes_per_row % ALIGN != 0 {
        return Err(anyhow!(
            "export resolution {resolution} yields stride {bytes_per_row}, not {ALIGN}-aligned"
        ));
    }
    let per_layer_bytes = (bytes_per_row * resolution) as u64;

    // Stage 1: allocate one readback buffer per (tile × channel) and queue all
    // copies into a single encoder submission.
    let mut readbacks: Vec<Readback> = Vec::with_capacity(paint_target.tiles.len() * CHANNELS.len());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("export_copy_enc"),
    });

    for (layer_idx, udim) in paint_target.tiles.iter().enumerate() {
        for (channel_name, tex_getter) in CHANNELS {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("export_readback"),
                size: per_layer_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex_getter(paint_target),
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_idx as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(resolution),
                    },
                },
                wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
            );
            let path = dir.join(format!("{channel_name}.{udim}.png"));
            readbacks.push(Readback {
                buffer,
                path,
                channel: channel_name,
                udim: *udim,
            });
        }
    }
    queue.submit(Some(encoder.finish()));

    // Stage 2: request maps on every readback buffer, wait on all with one poll.
    let (tx, rx) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    for rb in &readbacks {
        let tx = tx.clone();
        rb.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
    }
    drop(tx);
    let _ = device.poll(wgpu::Maintain::Wait);
    for _ in 0..readbacks.len() {
        rx.recv()
            .map_err(|e| anyhow!("map_async channel closed: {e}"))?
            .map_err(|e| anyhow!("map_async failed: {e:?}"))?;
    }

    // Stage 3: copy mapped bytes to owned Vec<u8> (single thread — BufferView
    // isn't Send), unmap, then encode PNGs in parallel.
    let blocks: Vec<Vec<u8>> = readbacks
        .iter()
        .map(|rb| {
            let range = rb.buffer.slice(..).get_mapped_range();
            let owned = range.to_vec();
            drop(range);
            rb.buffer.unmap();
            owned
        })
        .collect();

    let exports: Result<Vec<Export>> = readbacks
        .par_iter()
        .zip(blocks.par_iter())
        .map(|(rb, rgba)| {
            // V-flip: produce PNGs with the OpenGL-convention external tools
            // (Maya / Houdini / Substance / glTF) expect. Mirrors the flip
            // we apply on import, so round-trips through forge-paint are
            // stable.
            let flipped = crate::persist::flip_rows_rgba8(rgba, resolution, resolution);
            image::save_buffer_with_format(
                &rb.path,
                &flipped,
                resolution,
                resolution,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )
            .with_context(|| format!("writing {}", rb.path.display()))?;
            Ok(Export {
                path: rb.path.clone(),
                channel: rb.channel,
                udim: rb.udim,
            })
        })
        .collect();

    exports
}
