use crate::usd_out::{fmt_f32, sanitize_identifier, write_usda_document};
use anyhow::{Context, Result, anyhow, bail};
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjConversionSummary {
    pub vertices: usize,
    pub triangles: usize,
}

#[derive(Debug, Clone, Copy)]
struct ObjRef {
    position: usize,
    texcoord: Option<usize>,
    normal: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct FaceVertex {
    position: [f32; 3],
    texcoord: [f32; 2],
    normal: [f32; 3],
}

pub fn convert_obj_to_usd(source: &Path, dest: &Path) -> Result<ObjConversionSummary> {
    let text = std::fs::read_to_string(source)
        .with_context(|| format!("read OBJ {}", source.display()))?;
    let mut positions = Vec::<[f32; 3]>::new();
    let mut texcoords = Vec::<[f32; 2]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut flattened = Vec::<FaceVertex>::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else {
            continue;
        };
        match tag {
            "v" => positions.push(parse_vec3(parts, line_no, "v")?),
            "vt" => texcoords.push(parse_vec2(parts, line_no, "vt")?),
            "vn" => normals.push(normalize(parse_vec3(parts, line_no, "vn")?)),
            "f" => {
                let refs = parts
                    .map(|part| {
                        parse_obj_ref(
                            part,
                            positions.len(),
                            texcoords.len(),
                            normals.len(),
                            line_no,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                if refs.len() < 3 {
                    bail!("OBJ line {line_no}: face has fewer than 3 vertices");
                }
                for i in 1..refs.len() - 1 {
                    append_triangle(
                        &mut flattened,
                        [refs[0], refs[i], refs[i + 1]],
                        &positions,
                        &texcoords,
                        &normals,
                    );
                }
            }
            _ => {}
        }
    }

    if flattened.is_empty() {
        bail!("OBJ contains no polygon faces: {}", source.display());
    }

    write_usda(source, dest, &flattened)?;
    Ok(ObjConversionSummary {
        vertices: flattened.len(),
        triangles: flattened.len() / 3,
    })
}

fn parse_vec3<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    line_no: usize,
    tag: &str,
) -> Result<[f32; 3]> {
    let x = parse_f32(parts.next(), line_no, tag)?;
    let y = parse_f32(parts.next(), line_no, tag)?;
    let z = parse_f32(parts.next(), line_no, tag)?;
    Ok([x, y, z])
}

fn parse_vec2<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    line_no: usize,
    tag: &str,
) -> Result<[f32; 2]> {
    let u = parse_f32(parts.next(), line_no, tag)?;
    let v = parse_f32(parts.next(), line_no, tag)?;
    Ok([u, v])
}

fn parse_f32(value: Option<&str>, line_no: usize, tag: &str) -> Result<f32> {
    let value =
        value.ok_or_else(|| anyhow!("OBJ line {line_no}: `{tag}` is missing a component"))?;
    let parsed = value
        .parse::<f32>()
        .with_context(|| format!("OBJ line {line_no}: parse float `{value}`"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        bail!("OBJ line {line_no}: non-finite float `{value}`")
    }
}

fn parse_obj_ref(
    text: &str,
    position_count: usize,
    texcoord_count: usize,
    normal_count: usize,
    line_no: usize,
) -> Result<ObjRef> {
    let fields: Vec<&str> = text.split('/').collect();
    if fields.is_empty() || fields[0].is_empty() || fields.len() > 3 {
        bail!("OBJ line {line_no}: invalid face vertex `{text}`");
    }
    let position = resolve_obj_index(fields[0], position_count, line_no, "position")?;
    let texcoord = fields
        .get(1)
        .filter(|field| !field.is_empty())
        .map(|field| resolve_obj_index(field, texcoord_count, line_no, "texcoord"))
        .transpose()?;
    let normal = fields
        .get(2)
        .filter(|field| !field.is_empty())
        .map(|field| resolve_obj_index(field, normal_count, line_no, "normal"))
        .transpose()?;

    Ok(ObjRef {
        position,
        texcoord,
        normal,
    })
}

fn resolve_obj_index(text: &str, len: usize, line_no: usize, label: &str) -> Result<usize> {
    let index = text
        .parse::<isize>()
        .with_context(|| format!("OBJ line {line_no}: parse {label} index `{text}`"))?;
    if index == 0 {
        bail!("OBJ line {line_no}: OBJ indices are 1-based, got 0");
    }
    let resolved = if index > 0 {
        index - 1
    } else {
        len as isize + index
    };
    if resolved < 0 || resolved as usize >= len {
        bail!("OBJ line {line_no}: {label} index `{text}` is out of range for {len} entries");
    }
    Ok(resolved as usize)
}

fn append_triangle(
    out: &mut Vec<FaceVertex>,
    refs: [ObjRef; 3],
    positions: &[[f32; 3]],
    texcoords: &[[f32; 2]],
    normals: &[[f32; 3]],
) {
    let p0 = positions[refs[0].position];
    let p1 = positions[refs[1].position];
    let p2 = positions[refs[2].position];
    let fallback_normal = face_normal(p0, p1, p2);
    for reference in refs {
        out.push(FaceVertex {
            position: positions[reference.position],
            texcoord: reference
                .texcoord
                .map(|idx| texcoords[idx])
                .unwrap_or([0.0, 0.0]),
            normal: reference
                .normal
                .map(|idx| normals[idx])
                .unwrap_or(fallback_normal),
        });
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize([
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn write_usda(source: &Path, dest: &Path, vertices: &[FaceVertex]) -> Result<()> {
    let mesh_name = source
        .file_stem()
        .and_then(|s| s.to_str())
        .map(sanitize_identifier)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "mesh".to_string());

    let mut text = String::new();
    writeln!(text, "#usda 1.0")?;
    writeln!(text, "(")?;
    writeln!(text, "    defaultPrim = \"root\"")?;
    writeln!(text, "    metersPerUnit = 1")?;
    writeln!(text, "    upAxis = \"Y\"")?;
    writeln!(text, ")")?;
    writeln!(text)?;
    writeln!(text, "def Xform \"root\"")?;
    writeln!(text, "{{")?;
    writeln!(text, "    def Mesh \"{mesh_name}\"")?;
    writeln!(text, "    {{")?;
    write_face_vertex_counts(&mut text, vertices.len() / 3)?;
    write_face_vertex_indices(&mut text, vertices.len())?;
    write_points(&mut text, vertices)?;
    write_normals(&mut text, vertices)?;
    write_uvs(&mut text, vertices)?;
    writeln!(text, "        uniform token subdivisionScheme = \"none\"")?;
    writeln!(text, "    }}")?;
    writeln!(text, "}}")?;

    write_usda_document(&text, dest)
}

fn write_face_vertex_counts(text: &mut String, triangles: usize) -> Result<()> {
    write!(text, "        int[] faceVertexCounts = [")?;
    for i in 0..triangles {
        if i > 0 {
            write!(text, ", ")?;
        }
        write!(text, "3")?;
    }
    writeln!(text, "]")?;
    Ok(())
}

fn write_face_vertex_indices(text: &mut String, vertex_count: usize) -> Result<()> {
    write!(text, "        int[] faceVertexIndices = [")?;
    for i in 0..vertex_count {
        if i > 0 {
            write!(text, ", ")?;
        }
        write!(text, "{i}")?;
    }
    writeln!(text, "]")?;
    Ok(())
}

fn write_points(text: &mut String, vertices: &[FaceVertex]) -> Result<()> {
    writeln!(text, "        point3f[] points = [")?;
    for (i, v) in vertices.iter().enumerate() {
        let comma = if i + 1 == vertices.len() { "" } else { "," };
        writeln!(
            text,
            "            ({}, {}, {}){comma}",
            fmt_f32(v.position[0]),
            fmt_f32(v.position[1]),
            fmt_f32(v.position[2])
        )?;
    }
    writeln!(text, "        ]")?;
    Ok(())
}

fn write_normals(text: &mut String, vertices: &[FaceVertex]) -> Result<()> {
    writeln!(text, "        normal3f[] normals = [")?;
    for (i, v) in vertices.iter().enumerate() {
        let comma = if i + 1 == vertices.len() { "" } else { "," };
        writeln!(
            text,
            "            ({}, {}, {}){comma}",
            fmt_f32(v.normal[0]),
            fmt_f32(v.normal[1]),
            fmt_f32(v.normal[2])
        )?;
    }
    writeln!(text, "        ] (")?;
    writeln!(text, "            interpolation = \"faceVarying\"")?;
    writeln!(text, "        )")?;
    Ok(())
}

fn write_uvs(text: &mut String, vertices: &[FaceVertex]) -> Result<()> {
    writeln!(text, "        texCoord2f[] primvars:st = [")?;
    for (i, v) in vertices.iter().enumerate() {
        let comma = if i + 1 == vertices.len() { "" } else { "," };
        writeln!(
            text,
            "            ({}, {}){comma}",
            fmt_f32(v.texcoord[0]),
            fmt_f32(v.texcoord[1])
        )?;
    }
    writeln!(text, "        ] (")?;
    writeln!(text, "            interpolation = \"faceVarying\"")?;
    writeln!(text, "        )")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "forge_paint_obj_to_usd_{}_{}",
            std::process::id(),
            name
        ))
    }

    #[test]
    fn converts_quad_with_uvs_and_normals() -> Result<()> {
        let obj = temp_file("quad.obj");
        let usda = temp_file("quad.usda");
        std::fs::write(
            &obj,
            "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
",
        )?;

        let summary = convert_obj_to_usd(&obj, &usda)?;
        assert_eq!(summary.vertices, 6);
        assert_eq!(summary.triangles, 2);

        let text = std::fs::read_to_string(&usda)?;
        assert!(text.contains("int[] faceVertexCounts = [3, 3]"));
        assert!(text.contains("texCoord2f[] primvars:st"));
        assert!(text.contains("(0, 0, 1)"));

        let _ = std::fs::remove_file(obj);
        let _ = std::fs::remove_file(usda);
        Ok(())
    }

    #[test]
    fn accepts_negative_obj_indices() -> Result<()> {
        let obj = temp_file("negative.obj");
        let usda = temp_file("negative.usda");
        std::fs::write(
            &obj,
            "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f -4 -3 -2 -1
",
        )?;

        let summary = convert_obj_to_usd(&obj, &usda)?;
        assert_eq!(summary.vertices, 6);
        assert_eq!(summary.triangles, 2);

        let text = std::fs::read_to_string(&usda)?;
        assert!(text.contains("normal3f[] normals"));
        assert!(text.contains("texCoord2f[] primvars:st"));

        let _ = std::fs::remove_file(obj);
        let _ = std::fs::remove_file(usda);
        Ok(())
    }
}
