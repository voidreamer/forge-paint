use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::common;
use super::context::GpuContext;
use super::flat_bvh::FlatBvh;
use crate::bakers::ao::{AoSettings, Distribution};
use crate::raster::TexelData;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    ray_count: u32,
    max_distance: f32,
    bias: f32,
    total_texels: u32,
    node_count: u32,
    _pad0: u32,
    workgroups_x: u32,
    _pad1: u32,
    spread_angle: f32,
    distribution: u32, // 0 = cosine, 1 = uniform
    _pad2: u32,
    _pad3: u32,
}

/// Bake bent normals on the GPU. Returns Vec<[f32; 3]> with one RGB value per texel.
pub fn bake_bent_normals_gpu(
    ctx: &GpuContext,
    texel_data: &[Option<TexelData>],
    flat_bvh: &FlatBvh,
    settings: &AoSettings,
) -> Vec<[f32; 3]> {
    let total = texel_data.len();
    let gpu_texels = common::pack_texels(texel_data);

    let workgroup_size = 64u32;
    let total_workgroups = (total as u32).div_ceil(workgroup_size);
    let max_dim = 65535u32;
    let wg_x = total_workgroups.min(max_dim);
    let wg_y = total_workgroups.div_ceil(wg_x);

    let params = GpuParams {
        ray_count: settings.ray_count,
        max_distance: settings.max_distance,
        bias: settings.bias,
        total_texels: total as u32,
        node_count: flat_bvh.nodes.len() as u32,
        _pad0: 0,
        workgroups_x: wg_x,
        _pad1: 0,
        spread_angle: settings.spread_angle,
        distribution: match settings.distribution {
            Distribution::Cosine => 0,
            Distribution::Uniform => 1,
        },
        _pad2: 0,
        _pad3: 0,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    let texel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_texels"),
        contents: bytemuck::cast_slice(&gpu_texels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_nodes"),
        contents: bytemuck::cast_slice(&flat_bvh.nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_tris"),
        contents: bytemuck::cast_slice(&flat_bvh.triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bn_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Output: 3 floats per texel
    let output_size = (total * 3 * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bn_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bn_readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bn_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("bent_normals.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bn_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(4, false),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bn_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: texel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: node_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: tri_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bn_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bn_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bn_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let data = common::read_back_buffer(device, &readback_buffer);
    let flat: &[f32] = bytemuck::cast_slice(&data);

    let result: Vec<[f32; 3]> = (0..total)
        .map(|i| [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]])
        .collect();

    drop(data);
    readback_buffer.unmap();

    result
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
