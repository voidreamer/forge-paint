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
    /// Line-list index buffer — each triangle contributes 3 edges (6
    /// indices). Used by the wireframe overlay pipeline.
    pub line_index_buffer: wgpu::Buffer,
    pub line_index_count: u32,
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
        let line_indices: Vec<u32> = cpu
            .indices
            .iter()
            .flat_map(|&[a, b, c]| [a, b, b, c, c, a])
            .collect();

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
        let line_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forge_paint_mesh_wireframe_ib"),
            contents: bytemuck::cast_slice(&line_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: flat_indices.len() as u32,
            line_index_buffer,
            line_index_count: line_indices.len() as u32,
            center,
            radius,
        }
    }
}

/// Midpoint-subdivide `mesh` `levels` times. Each level splits every
/// triangle into 4 via edge midpoints, quadrupling the triangle count.
/// Positions / UVs interpolate linearly; normals average-and-normalize.
/// Vertices are emitted per-triangle (no dedupe) so the data model stays
/// simple — OK for viewport display; costs memory for high levels.
pub fn subdivide(mesh: &CpuMesh, levels: u32) -> CpuMesh {
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current);
    }
    current
}

fn subdivide_once(mesh: &CpuMesh) -> CpuMesh {
    let tri_count = mesh.indices.len();
    let mut out = CpuMesh {
        positions: Vec::with_capacity(tri_count * 6),
        normals: Vec::with_capacity(tri_count * 6),
        uvs: Vec::with_capacity(tri_count * 6),
        indices: Vec::with_capacity(tri_count * 4),
    };
    for &[i0, i1, i2] in &mesh.indices {
        let (p0, p1, p2) = (
            mesh.positions[i0 as usize],
            mesh.positions[i1 as usize],
            mesh.positions[i2 as usize],
        );
        let (n0, n1, n2) = (
            mesh.normals[i0 as usize],
            mesh.normals[i1 as usize],
            mesh.normals[i2 as usize],
        );
        let (u0, u1, u2) = (
            mesh.uvs[i0 as usize],
            mesh.uvs[i1 as usize],
            mesh.uvs[i2 as usize],
        );
        let p01 = (p0 + p1) * 0.5;
        let p12 = (p1 + p2) * 0.5;
        let p20 = (p2 + p0) * 0.5;
        let n01 = (n0 + n1).normalize_or(Vec3::Y);
        let n12 = (n1 + n2).normalize_or(Vec3::Y);
        let n20 = (n2 + n0).normalize_or(Vec3::Y);
        let u01 = (u0 + u1) * 0.5;
        let u12 = (u1 + u2) * 0.5;
        let u20 = (u2 + u0) * 0.5;

        let base = out.positions.len() as u32;
        out.positions
            .extend_from_slice(&[p0, p1, p2, p01, p12, p20]);
        out.normals
            .extend_from_slice(&[n0, n1, n2, n01, n12, n20]);
        out.uvs.extend_from_slice(&[u0, u1, u2, u01, u12, u20]);
        // 4 new triangles per original (corners use original verts, the
        // central triangle uses the three new midpoints).
        out.indices.push([base, base + 3, base + 5]);
        out.indices.push([base + 3, base + 1, base + 4]);
        out.indices.push([base + 5, base + 4, base + 2]);
        out.indices.push([base + 3, base + 4, base + 5]);
    }
    out
}

trait Vec3NormalizeOr {
    fn normalize_or(self, fallback: Vec3) -> Vec3;
}
impl Vec3NormalizeOr for Vec3 {
    fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let len = self.length();
        if len > 1e-6 {
            self / len
        } else {
            fallback
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
