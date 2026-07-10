use bytemuck::{Pod, Zeroable};

use crate::raster::TexelData;

/// Texel data packed for GPU upload (shared between AO and projection shaders).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTexel {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// Pack optional texel data into GPU-friendly structs.
pub fn pack_texels(texel_data: &[Option<TexelData>]) -> Vec<GpuTexel> {
    texel_data
        .iter()
        .map(|t| match t {
            Some(td) => GpuTexel {
                position: [td.position.x, td.position.y, td.position.z],
                _pad0: 0.0,
                normal: [td.normal.x, td.normal.y, td.normal.z],
                _pad1: 0.0,
            },
            None => GpuTexel {
                position: [0.0; 3],
                _pad0: 0.0,
                normal: [0.0; 3],
                _pad1: 0.0,
            },
        })
        .collect()
}

/// Read back data from a GPU buffer. Blocks until the data is available.
pub fn read_back_buffer<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
) -> wgpu::BufferView<'a> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    // wgpu 24 still uses Maintain (renamed to PollType in 25).
    let _ = device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    slice.get_mapped_range()
}
