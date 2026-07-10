use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::common;
use super::context::GpuContext;
use super::flat_bvh::FlatBvh;
use crate::bakers::ao::{AoSettings, Distribution};
use crate::raster::TexelData;

/// Shader parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    ray_count: u32,
    max_distance: f32,
    bias: f32,
    total_texels: u32,
    node_count: u32,
    mode: u32, // 0 = AO, 1 = thickness
    workgroups_x: u32,
    y_offset: u32,
    spread_angle: f32,
    distribution: u32, // 0 = cosine, 1 = uniform
    _pad0: u32,
    _pad1: u32,
}

/// Bake AO (or thickness) on the GPU.
/// Returns a Vec<f32> with one value per texel.
pub fn bake_ao_gpu(
    ctx: &GpuContext,
    texel_data: &[Option<TexelData>],
    flat_bvh: &FlatBvh,
    settings: &AoSettings,
    thickness_mode: bool,
) -> Vec<f32> {
    let total = texel_data.len();

    // Pack texel data for GPU
    let gpu_texels = common::pack_texels(texel_data);

    let workgroup_size = 64u32;
    let total_workgroups = (total as u32).div_ceil(workgroup_size);
    let max_dim = 65535u32;
    let wg_x = total_workgroups.min(max_dim);

    let params = GpuParams {
        ray_count: settings.ray_count,
        max_distance: settings.max_distance,
        bias: settings.bias,
        total_texels: total as u32,
        node_count: flat_bvh.nodes.len() as u32,
        mode: if thickness_mode { 1 } else { 0 },
        workgroups_x: wg_x,
        y_offset: 0,
        spread_angle: settings.spread_angle,
        distribution: match settings.distribution {
            Distribution::Cosine => 0,
            Distribution::Uniform => 1,
        },
        _pad0: 0,
        _pad1: 0,
    };

    let device = &ctx.device;
    let queue = &ctx.queue;

    // Create buffers
    let texel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("texels"),
        contents: bytemuck::cast_slice(&gpu_texels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bvh_nodes"),
        contents: bytemuck::cast_slice(&flat_bvh.nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangles"),
        contents: bytemuck::cast_slice(&flat_bvh.triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (total * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ao_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("ao.wgsl").into()),
    });

    // Bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ao_bgl"),
        entries: &[
            // texels
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // bvh_nodes
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // triangles
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params
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
            // output
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // The actual bind groups are created per batch in the dispatch loop
    // below (each batch rewrites the y_offset param buffer).

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ao_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ao_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch (wg_x, wg_y already computed above)
    let wg_y = total_workgroups.div_ceil(wg_x);

    // For large workloads (4K+, 256 rays), a single dispatch can exceed
    // Metal's command buffer timeout (~5s). We add a y_offset uniform to
    // the params and dispatch in row-batches, each in its own command buffer.
    let max_rows_per_batch = 128u32;
    let num_batches = wg_y.div_ceil(max_rows_per_batch);

    if num_batches > 1 {
        log::info!(
            "  GPU AO: splitting into {} batches to avoid timeout",
            num_batches
        );
    }

    for batch in 0..num_batches {
        let y_start = batch * max_rows_per_batch;
        let y_count = max_rows_per_batch.min(wg_y - y_start);

        // Update params with y_offset for this batch
        let mut batch_params = params;
        // Encode y_offset in the y_offset field (shader will add it to global_id.y)
        batch_params.y_offset = y_start;

        let batch_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ao_params_batch"),
            contents: bytemuck::bytes_of(&batch_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let batch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao_bg_batch"),
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: batch_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ao_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ao_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &batch_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, y_count, 1);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
    }

    // Copy results to readback
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ao_copy"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // Read back results
    let data = common::read_back_buffer(device, &readback_buffer);
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buffer.unmap();

    result
}
