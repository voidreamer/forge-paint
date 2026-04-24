//! Wireframe overlay pipeline — draws mesh edges as lines into the HDR
//! pass's color + depth attachments, shares the frame bind group with
//! the PBR pipeline.

use egui_wgpu::wgpu;

use crate::mesh::Vertex;
use crate::render::HDR_FORMAT;

pub struct WireframePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub visible: bool,
}

impl WireframePipeline {
    pub fn new(device: &wgpu::Device, frame_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wireframe.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wireframe.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wireframe.pl"),
            bind_group_layouts: &[frame_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe.pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_wire"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_wire"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            // Depth-test against the mesh pass's depth so hidden edges
            // are occluded; slight bias pulls the lines toward camera to
            // beat z-fighting with the underlying shaded surface.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -100,
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            visible: false,
        }
    }
}
