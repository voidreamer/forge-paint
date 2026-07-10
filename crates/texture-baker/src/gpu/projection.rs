use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::common;
use super::context::GpuContext;
use super::flat_bvh::FlatBvh;
use crate::accel::HitRecord;
use crate::raster::TexelData;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    frontal_distance: f32,
    rear_distance: f32,
    total_texels: u32,
    node_count: u32,
    workgroups_x: u32,
    ignore_backface: u32,
    min_t: f32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuHitResult {
    t: f32,
    u: f32,
    v: f32,
    tri_idx_f: f32,
}

/// Run projection ray casting on the GPU.
/// Returns a Vec of Option<HitRecord> matching the texel grid.
pub fn project_rays_gpu(
    ctx: &GpuContext,
    texel_data: &[Option<TexelData>],
    flat_bvh: &FlatBvh,
    frontal_distance: f32,
    rear_distance: f32,
    ignore_backface: bool,
    min_t: f32,
) -> Vec<Option<HitRecord>> {
    let total = texel_data.len();

    let gpu_texels = common::pack_texels(texel_data);

    let workgroup_size = 64u32;
    let total_workgroups = (total as u32).div_ceil(workgroup_size);
    let max_dim = 65535u32;
    let wg_x = total_workgroups.min(max_dim);
    let wg_y = total_workgroups.div_ceil(wg_x);

    let params = GpuParams {
        frontal_distance,
        rear_distance,
        total_texels: total as u32,
        node_count: flat_bvh.nodes.len() as u32,
        workgroups_x: wg_x,
        ignore_backface: if ignore_backface { 1 } else { 0 },
        min_t,
        _pad1: 0,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    let texel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("proj_texels"),
        contents: bytemuck::cast_slice(&gpu_texels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("proj_bvh"),
        contents: bytemuck::cast_slice(&flat_bvh.nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("proj_tris"),
        contents: bytemuck::cast_slice(&flat_bvh.triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("proj_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_size = (total * std::mem::size_of::<GpuHitResult>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("proj_output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("proj_readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("proj_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("projection.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("proj_bgl"),
        entries: &[
            bgl_entry(0, true),
            bgl_entry(1, true),
            bgl_entry(2, true),
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
            bgl_entry(4, false),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("proj_bg"),
        layout: &bind_group_layout,
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
        label: Some("proj_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("proj_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("proj_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proj_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let data = common::read_back_buffer(device, &readback_buffer);
    let gpu_hits: &[GpuHitResult] = bytemuck::cast_slice(&data);

    let result: Vec<Option<HitRecord>> = gpu_hits
        .iter()
        .map(|h| {
            let tri_idx = bytemuck::cast::<f32, u32>(h.tri_idx_f);
            if tri_idx == 0xFFFFFFFF {
                None
            } else {
                Some(HitRecord {
                    t: h.t,
                    u: h.u,
                    v: h.v,
                    tri_index: tri_idx as usize,
                    mesh_index: 0, // GPU path assumes single merged mesh
                    is_backface: false,
                })
            }
        })
        .collect();

    drop(data);
    readback_buffer.unmap();

    result
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
