use bevy_mikktspace::{Geometry, TangentSpace, generate_tangents};
use glam::{Vec3, Vec4};

use crate::mesh::Mesh;

/// Result of MikkTSpace tangent generation: per-vertex tangent (xyz) + sign (w).
#[derive(Debug, Clone)]
pub struct TangentData {
    pub tangents: Vec<Vec4>, // xyz = tangent direction, w = sign for bitangent
}

impl TangentData {
    /// Get tangent and bitangent at a vertex given its normal.
    pub fn tangent_bitangent(&self, vertex: usize, normal: Vec3) -> (Vec3, Vec3) {
        let t = self.tangents[vertex];
        let tangent = Vec3::new(t.x, t.y, t.z);
        let bitangent = normal.cross(tangent) * t.w;
        (tangent, bitangent)
    }
}

/// Wrapper that implements bevy_mikktspace::Geometry for our Mesh.
struct MikkTSpaceContext<'a> {
    mesh: &'a Mesh,
    tangents: Vec<Vec4>,
}

impl Geometry for MikkTSpaceContext<'_> {
    fn num_faces(&self) -> usize {
        self.mesh.indices.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3 // always triangles
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let idx = self.mesh.indices[face][vert] as usize;
        self.mesh.positions[idx].into()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let idx = self.mesh.indices[face][vert] as usize;
        self.mesh.normals[idx].into()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let idx = self.mesh.indices[face][vert] as usize;
        self.mesh.uvs[idx]
    }

    fn set_tangent(&mut self, tangent: Option<TangentSpace>, face: usize, vert: usize) {
        let idx = self.mesh.indices[face][vert] as usize;
        if let Some(ts) = tangent {
            let encoded = ts.tangent_encoded();
            self.tangents[idx] = Vec4::from(encoded);
        }
    }
}

/// Compute MikkTSpace tangents for a mesh.
pub fn compute_tangents(mesh: &Mesh) -> TangentData {
    let mut ctx = MikkTSpaceContext {
        mesh,
        tangents: vec![Vec4::new(1.0, 0.0, 0.0, 1.0); mesh.positions.len()],
    };

    if let Err(e) = generate_tangents(&mut ctx) {
        log::warn!(
            "MikkTSpace tangent generation failed for mesh '{}': {e:?}, using fallback tangents",
            mesh.name
        );
    }

    TangentData {
        tangents: ctx.tangents,
    }
}
