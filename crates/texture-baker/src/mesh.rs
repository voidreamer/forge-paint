use glam::Vec3;
use std::path::Path;

/// A loaded triangle mesh with positions, normals, UVs, and triangle indices.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: String,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
}

impl Mesh {
    /// Load all meshes from an OBJ file. Returns one `Mesh` per object/group.
    pub fn load_obj(path: &Path) -> Result<Vec<Mesh>, String> {
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .map_err(|e| format!("Failed to load OBJ '{}': {e}", path.display()))?;

        let mut meshes = Vec::with_capacity(models.len());

        for model in models {
            let m = &model.mesh;

            if m.positions.len() < 3 {
                continue;
            }

            let vertex_count = m.positions.len() / 3;

            let positions: Vec<Vec3> = (0..vertex_count)
                .map(|i| {
                    Vec3::new(
                        m.positions[i * 3],
                        m.positions[i * 3 + 1],
                        m.positions[i * 3 + 2],
                    )
                })
                .collect();

            let normals: Vec<Vec3> = if m.normals.len() == vertex_count * 3 {
                (0..vertex_count)
                    .map(|i| {
                        Vec3::new(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2])
                            .normalize()
                    })
                    .collect()
            } else {
                // Compute face normals and accumulate per-vertex
                compute_vertex_normals(&positions, &m.indices)
            };

            let uvs: Vec<[f32; 2]> = if m.texcoords.len() == vertex_count * 2 {
                (0..vertex_count)
                    .map(|i| [m.texcoords[i * 2], m.texcoords[i * 2 + 1]])
                    .collect()
            } else {
                vec![[0.0, 0.0]; vertex_count]
            };

            let tri_count = m.indices.len() / 3;
            let indices: Vec<[u32; 3]> = (0..tri_count)
                .map(|i| [m.indices[i * 3], m.indices[i * 3 + 1], m.indices[i * 3 + 2]])
                .collect();

            meshes.push(Mesh {
                name: model.name,
                positions,
                normals,
                uvs,
                indices,
            });
        }

        Ok(meshes)
    }

    /// Load all meshes from a glTF/GLB file.
    pub fn load_gltf(path: &Path) -> Result<Vec<Mesh>, String> {
        let (document, buffers, _images) = gltf::import(path)
            .map_err(|e| format!("Failed to load glTF '{}': {e}", path.display()))?;

        let mut meshes = Vec::new();

        for gltf_mesh in document.meshes() {
            for (prim_idx, primitive) in gltf_mesh.primitives().enumerate() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions: Vec<Vec3> = reader
                    .read_positions()
                    .map(|iter| iter.map(|p| Vec3::new(p[0], p[1], p[2])).collect())
                    .unwrap_or_default();

                if positions.is_empty() {
                    continue;
                }

                let indices: Vec<[u32; 3]> = if let Some(idx_reader) = reader.read_indices() {
                    let flat: Vec<u32> = idx_reader.into_u32().collect();
                    let tri_count = flat.len() / 3;
                    (0..tri_count)
                        .map(|i| [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]])
                        .collect()
                } else {
                    // Non-indexed: every 3 vertices form a triangle
                    let tri_count = positions.len() / 3;
                    (0..tri_count)
                        .map(|i| [i as u32 * 3, i as u32 * 3 + 1, i as u32 * 3 + 2])
                        .collect()
                };

                let normals: Vec<Vec3> = reader
                    .read_normals()
                    .map(|iter| {
                        iter.map(|n| Vec3::new(n[0], n[1], n[2]).normalize())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        let flat_indices: Vec<u32> =
                            indices.iter().flat_map(|t| t.iter().copied()).collect();
                        compute_vertex_normals(&positions, &flat_indices)
                    });

                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|iter| iter.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let name = format!(
                    "{}{}",
                    gltf_mesh.name().unwrap_or("mesh"),
                    if gltf_mesh.primitives().len() > 1 {
                        format!("_prim{}", prim_idx)
                    } else {
                        String::new()
                    }
                );

                meshes.push(Mesh {
                    name,
                    positions,
                    normals,
                    uvs,
                    indices,
                });
            }
        }

        Ok(meshes)
    }

    /// Load meshes from a file, auto-detecting format by extension.
    pub fn load(path: &Path) -> Result<Vec<Mesh>, String> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("obj") => Self::load_obj(path),
            Some("gltf") | Some("glb") => Self::load_gltf(path),
            Some(ext) => Err(format!(
                "Unsupported mesh format: .{ext} (supported: .obj, .gltf, .glb)"
            )),
            None => Err("No file extension, cannot determine mesh format".to_string()),
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Merge multiple meshes into a single mesh.
    /// Vertex indices are offset so the merged mesh is self-consistent.
    pub fn merge(meshes: &[Mesh]) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        for mesh in meshes {
            let offset = positions.len() as u32;
            positions.extend_from_slice(&mesh.positions);
            normals.extend_from_slice(&mesh.normals);
            uvs.extend_from_slice(&mesh.uvs);
            for tri in &mesh.indices {
                indices.push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
            }
        }

        Mesh {
            name: "merged".to_string(),
            positions,
            normals,
            uvs,
            indices,
        }
    }

    /// Get the three vertex positions for a triangle.
    pub fn tri_positions(&self, tri: &[u32; 3]) -> [Vec3; 3] {
        [
            self.positions[tri[0] as usize],
            self.positions[tri[1] as usize],
            self.positions[tri[2] as usize],
        ]
    }

    /// Get the three vertex normals for a triangle.
    pub fn tri_normals(&self, tri: &[u32; 3]) -> [Vec3; 3] {
        [
            self.normals[tri[0] as usize],
            self.normals[tri[1] as usize],
            self.normals[tri[2] as usize],
        ]
    }

    /// Get the three vertex UVs for a triangle.
    pub fn tri_uvs(&self, tri: &[u32; 3]) -> [[f32; 2]; 3] {
        [
            self.uvs[tri[0] as usize],
            self.uvs[tri[1] as usize],
            self.uvs[tri[2] as usize],
        ]
    }
}

/// Compute smooth vertex normals by averaging face normals (area-weighted).
fn compute_vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    let tri_count = indices.len() / 3;

    for i in 0..tri_count {
        let i0 = indices[i * 3] as usize;
        let i1 = indices[i * 3 + 1] as usize;
        let i2 = indices[i * 3 + 2] as usize;

        let v0 = positions[i0];
        let v1 = positions[i1];
        let v2 = positions[i2];

        let face_normal = (v1 - v0).cross(v2 - v0); // area-weighted (not normalized)

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    for n in &mut normals {
        let len = n.length();
        if len > 1e-10 {
            *n /= len;
        } else {
            *n = Vec3::Y; // fallback
        }
    }

    normals
}
