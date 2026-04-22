use crate::mesh::CpuMesh;

struct MikktGeom<'a> {
    mesh: &'a CpuMesh,
    tangents: Vec<[f32; 4]>,
}

impl<'a> bevy_mikktspace::Geometry for MikktGeom<'a> {
    fn num_faces(&self) -> usize {
        self.mesh.indices.len()
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let vi = self.mesh.indices[face][vert] as usize;
        self.mesh.positions[vi].to_array()
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let vi = self.mesh.indices[face][vert] as usize;
        self.mesh.normals[vi].to_array()
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let vi = self.mesh.indices[face][vert] as usize;
        self.mesh.uvs[vi].to_array()
    }
    fn set_tangent(
        &mut self,
        tangent_space: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        let vi = self.mesh.indices[face][vert] as usize;
        self.tangents[vi] = tangent_space
            .map(|ts| ts.tangent_encoded())
            .unwrap_or([1.0, 0.0, 0.0, 1.0]);
    }
}

pub fn compute(mesh: &CpuMesh) -> Vec<[f32; 4]> {
    let mut geom = MikktGeom {
        mesh,
        tangents: vec![[1.0, 0.0, 0.0, 1.0]; mesh.positions.len()],
    };
    if let Err(e) = bevy_mikktspace::generate_tangents(&mut geom) {
        log::warn!("mikktspace tangent generation failed: {e}; using defaults");
    }
    geom.tangents
}
