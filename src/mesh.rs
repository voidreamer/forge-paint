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
    /// Origin tracking for per-source-prim ranges in the merged
    /// vertex / index buffers. Populated by `load_stage_merged`;
    /// empty after non-merging loaders. The stage browser uses
    /// this to mark verts as "selected" by their owning UsdGeomMesh
    /// path so the wgpu shader can highlight them.
    pub prim_ranges: Vec<PrimRange>,
}

#[derive(Clone, Debug)]
pub struct PrimRange {
    pub prim_path: String,
    /// First vertex index in `CpuMesh::positions` owned by this prim.
    pub vert_start: u32,
    pub vert_count: u32,
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
    /// One f32 per vertex — 1.0 if the vertex's source prim is
    /// currently selected in the stage browser, 0.0 otherwise. Bound
    /// as a second vertex buffer; the PBR shader brightens flagged
    /// verts via interpolation (constant across triangle since all 3
    /// corners share the source prim). Updated via `set_selection`.
    pub selection_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub prim_ranges: Vec<PrimRange>,
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

        // Default everyone unselected — a fresh load shows no
        // highlight. set_selection writes the diff when the user
        // picks a prim in the stage browser.
        let zero_selection = vec![0.0_f32; cpu.positions.len()];
        let selection_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forge_paint_mesh_selection_vb"),
            contents: bytemuck::cast_slice(&zero_selection),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: flat_indices.len() as u32,
            line_index_buffer,
            line_index_count: line_indices.len() as u32,
            center,
            radius,
            selection_buffer,
            vertex_count: cpu.positions.len() as u32,
            prim_ranges: cpu.prim_ranges.clone(),
        }
    }

    /// Rewrites the per-vertex selection mask: 1.0 for any vertex
    /// whose owning `prim_path` is in `selected` OR whose path lives
    /// under a selected ancestor (so picking an Xform cascades to
    /// all the Mesh leaves it contains, matching Solaris/Houdini
    /// outliner behaviour). No-op if the mesh has no `prim_ranges`
    /// populated (i.e. came from a non-merging load path).
    pub fn set_selection(
        &self,
        queue: &wgpu::Queue,
        selected: &std::collections::HashSet<String>,
    ) {
        if self.prim_ranges.is_empty() {
            return;
        }
        let mut mask = vec![0.0_f32; self.vertex_count as usize];
        for range in &self.prim_ranges {
            if !is_selected_or_descendant(&range.prim_path, selected) {
                continue;
            }
            let start = range.vert_start as usize;
            let end = start + range.vert_count as usize;
            for v in &mut mask[start..end] {
                *v = 1.0;
            }
        }
        queue.write_buffer(&self.selection_buffer, 0, bytemuck::cast_slice(&mask));
    }
}

/// `prim_path` is selected iff the set holds it directly OR holds
/// an ancestor (any prefix that ends at a `/` boundary, so
/// `/World/F` doesn't match `/World/Foo`).
fn is_selected_or_descendant(
    prim_path: &str,
    selected: &std::collections::HashSet<String>,
) -> bool {
    selected.iter().any(|sel| {
        if sel == prim_path {
            return true;
        }
        // Pseudo-root "/" is selected → everything matches.
        if sel == "/" {
            return true;
        }
        prim_path.starts_with(sel.as_str())
            && prim_path.as_bytes().get(sel.len()) == Some(&b'/')
    })
}

/// Vertex-buffer layout for the per-vertex selection flag (slot 1
/// in the PBR pipeline). One f32 per vertex; matches `@location(4)
/// selected_in: f32` in pbr.wgsl.
pub const SELECTION_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<f32>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 0,
        shader_location: 4,
    }],
};

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
        prim_ranges: Vec::new(),
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

    CpuMesh {
        positions,
        normals,
        uvs,
        indices,
        prim_ranges: Vec::new(),
    }
}
