//! Static glTF / GLB -> USD converter.
//!
//! Same scope philosophy as the OBJ converter: geometry only.
//! Positions, normals, UVs, and triangle faces survive; materials,
//! skins, morph targets, and animations are ignored for now. The
//! default scene's node hierarchy is flattened — node transforms are
//! baked into the points — and each (node, triangle primitive) pair
//! becomes one `Mesh` prim under a root `Xform`.
//!
//! Unlike the OBJ path (which must flatten to faceVarying because OBJ
//! indexes positions/UVs/normals independently), glTF is single-indexed
//! per vertex, so the output keeps the index buffer and writes
//! vertex-interpolated normals/st — much smaller files.

use crate::usd_out::{fmt_f32, sanitize_identifier, write_usda_document};
use anyhow::{Context, Result, bail};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfConversionSummary {
    pub meshes: usize,
    pub vertices: usize,
    pub triangles: usize,
}

struct UsdMeshData {
    name: String,
    points: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    /// Flat triangle index buffer, `len == 3 * triangle_count`.
    indices: Vec<u32>,
}

pub fn convert_gltf_to_usd(source: &Path, dest: &Path) -> Result<GltfConversionSummary> {
    let (document, buffers, _images) =
        gltf::import(source).with_context(|| format!("read glTF {}", source.display()))?;

    let mut meshes = Vec::<UsdMeshData>::new();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    let Some(scene) = scene else {
        bail!("glTF has no scenes: {}", source.display());
    };
    for node in scene.nodes() {
        collect_node(&node, glam::Mat4::IDENTITY, &buffers, &mut meshes);
    }

    if meshes.is_empty() {
        bail!(
            "glTF contains no triangle geometry in its default scene: {}",
            source.display()
        );
    }
    dedup_names(&mut meshes);

    let text = emit_usda_text(&meshes)?;
    write_usda_document(&text, dest)?;

    Ok(GltfConversionSummary {
        meshes: meshes.len(),
        vertices: meshes.iter().map(|m| m.points.len()).sum(),
        triangles: meshes.iter().map(|m| m.indices.len() / 3).sum(),
    })
}

fn collect_node(
    node: &gltf::Node,
    parent: glam::Mat4,
    buffers: &[gltf::buffer::Data],
    out: &mut Vec<UsdMeshData>,
) {
    // gltf reports the local transform column-major, matching glam.
    let world = parent * glam::Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(mesh) = node.mesh() {
        let multi = mesh.primitives().len() > 1;
        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let raw_points: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            if raw_points.is_empty() {
                continue;
            }

            let mut indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                // Non-indexed: every 3 vertices form a triangle.
                None => (0..raw_points.len() as u32).collect(),
            };
            indices.truncate(indices.len() - indices.len() % 3);
            if indices.is_empty() {
                continue;
            }

            // Bake the node's world transform into the geometry.
            // Normals use the inverse-transpose so non-uniform scale
            // doesn't shear them.
            let normal_mat = glam::Mat3::from_mat4(world).inverse().transpose();
            let points: Vec<[f32; 3]> = raw_points
                .iter()
                .map(|p| {
                    world
                        .transform_point3(glam::Vec3::from_array(*p))
                        .to_array()
                })
                .collect();
            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(iter) => iter
                    .map(|n| {
                        (normal_mat * glam::Vec3::from_array(n))
                            .normalize_or(glam::Vec3::Y)
                            .to_array()
                    })
                    .collect(),
                None => compute_vertex_normals(&points, &indices),
            };
            // glTF puts the UV origin at the top-left (v grows down);
            // USD's st convention is bottom-left, so flip v or every
            // texture samples upside down.
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().map(|uv| [uv[0], 1.0 - uv[1]]).collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; points.len()]);

            // A mirroring transform (negative determinant) flips the
            // winding; swap each triangle back to keep faces front-side
            // out under USD's rightHanded orientation.
            if world.determinant() < 0.0 {
                for tri in indices.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }

            let base = node
                .name()
                .or_else(|| mesh.name())
                .map(sanitize_identifier)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "mesh".to_string());
            let name = if multi {
                format!("{base}_prim{prim_idx}")
            } else {
                base
            };

            out.push(UsdMeshData {
                name,
                points,
                normals,
                uvs,
                indices,
            });
        }
    }

    for child in node.children() {
        collect_node(&child, world, buffers, out);
    }
}

/// Area-weighted per-vertex normals for primitives that author none.
fn compute_vertex_normals(points: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![glam::Vec3::ZERO; points.len()];
    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p0 = glam::Vec3::from_array(points[i0]);
        let p1 = glam::Vec3::from_array(points[i1]);
        let p2 = glam::Vec3::from_array(points[i2]);
        // Cross product magnitude is twice the triangle area, so the
        // un-normalized sum is the area weighting.
        let n = (p1 - p0).cross(p2 - p0);
        acc[i0] += n;
        acc[i1] += n;
        acc[i2] += n;
    }
    acc.into_iter()
        .map(|n| n.normalize_or(glam::Vec3::Y).to_array())
        .collect()
}

/// USD sibling prims need unique names; glTF nodes don't.
fn dedup_names(meshes: &mut [UsdMeshData]) {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for mesh in meshes.iter_mut() {
        let count = seen.entry(mesh.name.clone()).or_insert(0);
        if *count > 0 {
            mesh.name = format!("{}_{}", mesh.name, count);
        }
        *count += 1;
    }
}

fn emit_usda_text(meshes: &[UsdMeshData]) -> Result<String> {
    let mut text = String::new();
    writeln!(text, "#usda 1.0")?;
    writeln!(text, "(")?;
    writeln!(text, "    defaultPrim = \"root\"")?;
    // glTF mandates meters and +Y up, same as the OBJ converter's
    // assumption, so no unit or axis remapping is needed.
    writeln!(text, "    metersPerUnit = 1")?;
    writeln!(text, "    upAxis = \"Y\"")?;
    writeln!(text, ")")?;
    writeln!(text)?;
    writeln!(text, "def Xform \"root\"")?;
    writeln!(text, "{{")?;
    for mesh in meshes {
        writeln!(text, "    def Mesh \"{}\"", mesh.name)?;
        writeln!(text, "    {{")?;

        write!(text, "        int[] faceVertexCounts = [")?;
        for i in 0..mesh.indices.len() / 3 {
            if i > 0 {
                write!(text, ", ")?;
            }
            write!(text, "3")?;
        }
        writeln!(text, "]")?;

        write!(text, "        int[] faceVertexIndices = [")?;
        for (i, index) in mesh.indices.iter().enumerate() {
            if i > 0 {
                write!(text, ", ")?;
            }
            write!(text, "{index}")?;
        }
        writeln!(text, "]")?;

        writeln!(text, "        point3f[] points = [")?;
        for (i, p) in mesh.points.iter().enumerate() {
            let comma = if i + 1 == mesh.points.len() { "" } else { "," };
            writeln!(
                text,
                "            ({}, {}, {}){comma}",
                fmt_f32(p[0]),
                fmt_f32(p[1]),
                fmt_f32(p[2])
            )?;
        }
        writeln!(text, "        ]")?;

        writeln!(text, "        normal3f[] normals = [")?;
        for (i, n) in mesh.normals.iter().enumerate() {
            let comma = if i + 1 == mesh.normals.len() { "" } else { "," };
            writeln!(
                text,
                "            ({}, {}, {}){comma}",
                fmt_f32(n[0]),
                fmt_f32(n[1]),
                fmt_f32(n[2])
            )?;
        }
        writeln!(text, "        ] (")?;
        writeln!(text, "            interpolation = \"vertex\"")?;
        writeln!(text, "        )")?;

        writeln!(text, "        texCoord2f[] primvars:st = [")?;
        for (i, uv) in mesh.uvs.iter().enumerate() {
            let comma = if i + 1 == mesh.uvs.len() { "" } else { "," };
            writeln!(
                text,
                "            ({}, {}){comma}",
                fmt_f32(uv[0]),
                fmt_f32(uv[1])
            )?;
        }
        writeln!(text, "        ] (")?;
        writeln!(text, "            interpolation = \"vertex\"")?;
        writeln!(text, "        )")?;

        writeln!(text, "        uniform token subdivisionScheme = \"none\"")?;
        writeln!(text, "    }}")?;
    }
    writeln!(text, "}}")?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "forge_paint_gltf_to_usd_{}_{}",
            std::process::id(),
            name
        ))
    }

    /// Minimal hand-built glTF: one quad (4 verts, 2 triangles) with
    /// UVs, no normals, on a node translated +1 in X. Buffer data lives
    /// in a sidecar .bin the importer resolves relative to the .gltf.
    #[test]
    fn converts_translated_quad_and_flips_v() -> Result<()> {
        let gltf_path = temp_file("quad.gltf");
        let bin_path = temp_file("quad.bin");
        let usda = temp_file("quad_out.usda");

        let positions: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // v=1 at the bottom row in glTF's top-left convention; the
        // converter must emit st v=0 there.
        let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let mut bin = Vec::<u8>::new();
        for p in &positions {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for uv in &uvs {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in &indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        std::fs::write(&bin_path, &bin)?;

        let bin_name = bin_path.file_name().unwrap().to_str().unwrap();
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "buffers": [{{"uri": "{bin_name}", "byteLength": {}}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 48, "byteLength": 32}},
    {{"buffer": 0, "byteOffset": 80, "byteLength": 12}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
      "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"}},
    {{"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}}
  ],
  "meshes": [{{"primitives": [{{
    "attributes": {{"POSITION": 0, "TEXCOORD_0": 1}}, "indices": 2}}]}}],
  "nodes": [{{"mesh": 0, "name": "quad", "translation": [1, 0, 0]}}],
  "scenes": [{{"nodes": [0]}}],
  "scene": 0
}}"#,
            bin.len()
        );
        std::fs::write(&gltf_path, json)?;

        let summary = convert_gltf_to_usd(&gltf_path, &usda)?;
        assert_eq!(summary.meshes, 1);
        assert_eq!(summary.vertices, 4);
        assert_eq!(summary.triangles, 2);

        let text = std::fs::read_to_string(&usda)?;
        assert!(text.contains("def Mesh \"quad\""));
        assert!(text.contains("int[] faceVertexCounts = [3, 3]"));
        assert!(text.contains("int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]"));
        // Node translation baked into points: origin vertex -> (1, 0, 0).
        assert!(text.contains("(1, 0, 0)"));
        // The quad faces +Z and authors no normals -> computed (0, 0, 1)
        // with vertex interpolation.
        assert!(text.contains("(0, 0, 1)"));
        assert!(text.contains("interpolation = \"vertex\""));
        // glTF v=1 rows flip to st v=0: the corner authored (0, 1) must
        // come out as (0, 0).
        assert!(text.contains("            (0, 0),") || text.contains("            (0, 0)\n"));

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_file(usda);
        Ok(())
    }
}
