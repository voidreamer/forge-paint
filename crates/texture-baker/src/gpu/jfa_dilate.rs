use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct JfaParams {
    width: u32,
    height: u32,
    step_size: u32,
    num_channels: u32,
}

/// Dilate an RGB buffer on the GPU using Jump Flooding Algorithm.
/// Runs in O(log N) passes where N = max(width, height).
pub fn dilate_rgb_gpu(
    ctx: &GpuContext,
    buffer: &mut Vec<[f32; 3]>,
    mask: &[bool],
    width: u32,
    height: u32,
) {
    let total = (width * height) as usize;
    let nc = 3u32;

    // Flatten RGB to flat f32 array
    let flat_data: Vec<f32> = buffer.iter().flat_map(|c| c.iter().copied()).collect();
    let mask_u32: Vec<u32> = mask.iter().map(|&v| if v { 1u32 } else { 0u32 }).collect();

    let result = run_jfa(ctx, &flat_data, &mask_u32, width, height, nc);

    // Unflatten back to RGB
    for i in 0..total {
        buffer[i] = [
            result[i * 3],
            result[i * 3 + 1],
            result[i * 3 + 2],
        ];
    }
}

/// Dilate a grayscale buffer on the GPU using Jump Flooding Algorithm.
pub fn dilate_gray_gpu(
    ctx: &GpuContext,
    buffer: &mut Vec<f32>,
    mask: &[bool],
    width: u32,
    height: u32,
) {
    let mask_u32: Vec<u32> = mask.iter().map(|&v| if v { 1u32 } else { 0u32 }).collect();
    let result = run_jfa(ctx, buffer, &mask_u32, width, height, 1);
    buffer.copy_from_slice(&result);
}

fn run_jfa(
    ctx: &GpuContext,
    data: &[f32],
    mask: &[u32],
    width: u32,
    height: u32,
    num_channels: u32,
) -> Vec<f32> {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let total_floats = data.len();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jfa_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("jfa.wgsl").into()),
    });

    // Create two ping-pong buffer pairs (data + mask)
    let data_buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jfa_data_a"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let data_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jfa_data_b"),
        size: (total_floats * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mask_buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jfa_mask_a"),
        contents: bytemuck::cast_slice(mask),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mask_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jfa_mask_b"),
        size: (mask.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jfa_params"),
        size: std::mem::size_of::<JfaParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("jfa_bgl"),
        entries: &[
            // params
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // input_data (read)
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
            // output_data (write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // input_mask (read)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // output_mask (write)
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

    // Two bind groups for ping-pong
    let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jfa_bg_a2b"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: data_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: data_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mask_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: mask_buf_b.as_entire_binding() },
        ],
    });
    let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jfa_bg_b2a"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: data_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: data_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mask_buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: mask_buf_a.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("jfa_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jfa_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let wg_x = (width + 7) / 8;
    let wg_y = (height + 7) / 8;

    // JFA passes: step_size = max_dim/2, max_dim/4, ..., 2, 1
    let max_dim = width.max(height);
    let mut step = max_dim.next_power_of_two() / 2;
    let mut use_a_as_input = true;
    let mut pass_count = 0u32;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jfa_encoder"),
    });

    while step >= 1 {
        let params = JfaParams {
            width,
            height,
            step_size: step,
            num_channels,
        };
        queue.write_buffer(&param_buffer, 0, bytemuck::bytes_of(&params));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jfa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            if use_a_as_input {
                pass.set_bind_group(0, &bg_a_to_b, &[]);
            } else {
                pass.set_bind_group(0, &bg_b_to_a, &[]);
            }
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        use_a_as_input = !use_a_as_input;
        pass_count += 1;
        step /= 2;
    }

    // Read back from whichever buffer has the final result
    let final_data_buf = if use_a_as_input { &data_buf_a } else { &data_buf_b };
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jfa_readback"),
        size: (total_floats * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(final_data_buf, 0, &readback, 0, (total_floats * 4) as u64);

    queue.submit(Some(encoder.finish()));

    let mapped = super::common::read_back_buffer(device, &readback);
    let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();

    log::info!("  JFA dilation: {} passes", pass_count);
    result
}
