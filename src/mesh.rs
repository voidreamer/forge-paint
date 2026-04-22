use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 24, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 40, shader_location: 3 },
        ],
    };
}

#[derive(Clone, Default, Debug)]
pub struct CpuMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<[u32; 3]>,
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub center: Vec3,
    pub radius: f32,
}

impl GpuMesh {
    pub fn from_cpu(device: &wgpu::Device, cpu: &CpuMesh) -> Self {
        let tangents = crate::tangents::compute(cpu);

        let vertices: Vec<Vertex> = (0..cpu.positions.len())
            .map(|i| Vertex {
                position: cpu.positions[i].to_array(),
                normal: cpu.normals[i].to_array(),
                tangent: tangents[i],
                uv: cpu.uvs[i].to_array(),
            })
            .collect();

        let flat_indices: Vec<u32> = cpu.indices.iter().flatten().copied().collect();

        let mut center = Vec3::ZERO;
        for p in &cpu.positions {
            center += *p;
        }
        center /= cpu.positions.len().max(1) as f32;
        let mut radius: f32 = 0.0;
        for p in &cpu.positions {
            radius = radius.max((*p - center).length());
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forge_paint_mesh_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forge_paint_mesh_ib"),
            contents: bytemuck::cast_slice(&flat_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: flat_indices.len() as u32,
            center,
            radius,
        }
    }
}

/// Unit cube centered at origin with hard-split faces so each face gets its
/// own normal + full 0..1 UV square. 24 verts, 12 tris.
pub fn cube() -> CpuMesh {
    // (four corners CCW when viewed from outside, face normal)
    let faces: [([Vec3; 4], Vec3); 6] = [
        ([Vec3::new( 0.5,-0.5,-0.5), Vec3::new( 0.5,-0.5, 0.5), Vec3::new( 0.5, 0.5, 0.5), Vec3::new( 0.5, 0.5,-0.5)], Vec3::X),
        ([Vec3::new(-0.5,-0.5, 0.5), Vec3::new(-0.5,-0.5,-0.5), Vec3::new(-0.5, 0.5,-0.5), Vec3::new(-0.5, 0.5, 0.5)], Vec3::NEG_X),
        ([Vec3::new(-0.5, 0.5,-0.5), Vec3::new( 0.5, 0.5,-0.5), Vec3::new( 0.5, 0.5, 0.5), Vec3::new(-0.5, 0.5, 0.5)], Vec3::Y),
        ([Vec3::new(-0.5,-0.5, 0.5), Vec3::new( 0.5,-0.5, 0.5), Vec3::new( 0.5,-0.5,-0.5), Vec3::new(-0.5,-0.5,-0.5)], Vec3::NEG_Y),
        ([Vec3::new(-0.5,-0.5, 0.5), Vec3::new(-0.5, 0.5, 0.5), Vec3::new( 0.5, 0.5, 0.5), Vec3::new( 0.5,-0.5, 0.5)], Vec3::Z),
        ([Vec3::new( 0.5,-0.5,-0.5), Vec3::new( 0.5, 0.5,-0.5), Vec3::new(-0.5, 0.5,-0.5), Vec3::new(-0.5,-0.5,-0.5)], Vec3::NEG_Z),
    ];

    let uv_corners = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];

    let mut positions = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut uvs = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(12);

    for (corners, normal) in faces {
        let base = positions.len() as u32;
        for i in 0..4 {
            positions.push(corners[i]);
            normals.push(normal);
            uvs.push(uv_corners[i]);
        }
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
    }

    CpuMesh { positions, normals, uvs, indices }
}
