// Minimal .usda text parser — narrow enough to read what `usdcat --flatten`
// emits for geometry-only stages. Not a full USD parser; just enough for
// UsdGeomMesh extraction with points / faceVertex* / primvars:st / normals,
// plus single-op xformOp:transform on ancestor Xform prims.

use anyhow::{anyhow, bail, Context, Result};
use glam::Mat4;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Vertex,
    FaceVarying,
    Uniform,
    Constant,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct StPrimvar {
    pub data: Vec<[f32; 2]>,
    pub indices: Option<Vec<u32>>,
    pub interpolation: Interpolation,
}

#[derive(Debug, Clone)]
pub struct NormalPrimvar {
    pub data: Vec<[f32; 3]>,
    pub indices: Option<Vec<u32>>,
    pub interpolation: Interpolation,
}

#[derive(Debug, Clone)]
pub struct UsdMesh {
    pub path: String,
    pub points: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<u32>,
    pub face_vertex_indices: Vec<u32>,
    pub st: Option<StPrimvar>,
    pub normals: Option<NormalPrimvar>,
    pub world_xform: Mat4,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Eq,
    /// Asset path `@...@` — consumed as one token including arbitrary
    /// punctuation (dashes, parentheses, spaces) that would otherwise
    /// trip the tokenizer. Triple-`@` paths (`@@@...@@@`) aren't
    /// handled yet but are rare in flattened output.
    Asset(String),
    /// USD path literal inside `<...>` — e.g. `</World/foo>`. Stored
    /// opaquely; the mesh parser ignores these (they appear inside
    /// `rel` / `material:binding` / `inherits` / etc., which the mesh
    /// extractor doesn't need).
    Path(String),
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();

    while i < bytes.len() {
        let c = bytes[i];

        // Whitespace
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Comment: `#...` to end of line (also skips `#usda 1.0` header)
        if c == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Triple-quoted string: `"""..."""`
        if c == b'"' && i + 2 < bytes.len() && bytes[i + 1] == b'"' && bytes[i + 2] == b'"' {
            i += 3;
            let start = i;
            while i + 2 < bytes.len()
                && !(bytes[i] == b'"' && bytes[i + 1] == b'"' && bytes[i + 2] == b'"')
            {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .with_context(|| "non-utf8 in triple-string")?
                .to_string();
            i = (i + 3).min(bytes.len());
            out.push(Tok::Str(s));
            continue;
        }

        // Single-quoted string: "..." (no escape handling beyond \\" and \n)
        if c == b'"' || c == b'\'' {
            let quote = c;
            i += 1;
            let mut s = String::new();
            while i < bytes.len() && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    s.push(bytes[i + 1] as char);
                    i += 2;
                } else {
                    s.push(bytes[i] as char);
                    i += 1;
                }
            }
            i += 1; // skip closing quote
            out.push(Tok::Str(s));
            continue;
        }

        // Numbers (including negatives when context allows — the '-' binds here)
        if c.is_ascii_digit()
            || (c == b'-' && i + 1 < bytes.len() && (bytes[i + 1].is_ascii_digit() || bytes[i + 1] == b'.'))
            || (c == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            if bytes[i] == b'-' {
                i += 1;
            }
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || bytes[i] == b'.'
                    || bytes[i] == b'e'
                    || bytes[i] == b'E'
                    || ((bytes[i] == b'+' || bytes[i] == b'-')
                        && i > 0
                        && (bytes[i - 1] == b'e' || bytes[i - 1] == b'E')))
            {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap();
            let n: f64 = s
                .parse()
                .with_context(|| format!("bad number literal: {s:?}"))?;
            out.push(Tok::Num(n));
            continue;
        }

        // Single-char punctuation
        let t = match c {
            b'{' => Some(Tok::LBrace),
            b'}' => Some(Tok::RBrace),
            b'[' => Some(Tok::LBracket),
            b']' => Some(Tok::RBracket),
            b'(' => Some(Tok::LParen),
            b')' => Some(Tok::RParen),
            b',' => Some(Tok::Comma),
            b'=' => Some(Tok::Eq),
            _ => None,
        };
        if let Some(t) = t {
            out.push(t);
            i += 1;
            continue;
        }

        // Asset path `@...@` — single token, arbitrary content inside
        // (paths frequently contain dashes, dots, spaces).
        if c == b'@' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'@' {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .with_context(|| "non-utf8 in asset path")?
                .to_string();
            if i < bytes.len() {
                i += 1; // skip closing '@'
            }
            out.push(Tok::Asset(s));
            continue;
        }

        // USD path literal: `<...>` — used by rel / material:binding /
        // inherits / etc. We don't use the value (the mesh extractor
        // ignores relationships) but we do need to consume it cleanly.
        if c == b'<' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i])
                .with_context(|| "non-utf8 in path literal")?
                .to_string();
            if i < bytes.len() {
                i += 1; // skip '>'
            }
            out.push(Tok::Path(s));
            continue;
        }

        // Identifier (allows `:` and `.` and `/` for paths)
        if c.is_ascii_alphabetic() || c == b'_' || c == b'/' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'_'
                    || bytes[i] == b':'
                    || bytes[i] == b'.'
                    || bytes[i] == b'/')
            {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap().to_string();
            out.push(Tok::Ident(s));
            continue;
        }

        // Show a window of context so the user has a chance to see
        // which USD construct we can't handle yet (the parser is hand
        // rolled and only covers UsdGeomMesh attributes — references,
        // variants, material graphs, etc. fall over here).
        let window_start = i.saturating_sub(40);
        let window_end = (i + 40).min(bytes.len());
        let context = String::from_utf8_lossy(&bytes[window_start..window_end]);
        let line = bytes[..i].iter().filter(|&&b| b == b'\n').count() + 1;
        bail!(
            "tokenizer: unexpected byte {:?} at offset {} (line {}):\n    ...{}...",
            c as char,
            i,
            line,
            context.trim_end(),
        );
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Value tree (generic so we can post-extract per attribute type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Value {
    Num(f64),
    Str(String),
    Ident(String),
    Tuple(Vec<Value>),
    Array(Vec<Value>),
    Asset(String),
    None,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(toks: &'a [Tok]) -> Self {
        Self { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn advance(&mut self) -> Option<&Tok> {
        let t = self.toks.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn expect(&mut self, expected: &Tok) -> Result<()> {
        match self.advance() {
            Some(t) if t == expected => Ok(()),
            Some(t) => Err(anyhow!("expected {expected:?}, got {t:?}")),
            None => Err(anyhow!("expected {expected:?}, got EOF")),
        }
    }

    fn skip_optional_metadata_block(&mut self) -> Result<()> {
        // Stage metadata or prim/attribute metadata: `( ... )` at a position where
        // it's a metadata block (not a tuple value).
        if matches!(self.peek(), Some(Tok::LParen)) {
            self.advance();
            let mut depth = 1;
            while depth > 0 {
                match self.advance() {
                    Some(Tok::LParen) => depth += 1,
                    Some(Tok::RParen) => depth -= 1,
                    Some(_) => {}
                    None => bail!("unterminated metadata block"),
                }
            }
        }
        Ok(())
    }

    /// Parse a value following `=`.
    fn parse_value(&mut self) -> Result<Value> {
        match self.peek() {
            Some(Tok::Num(_)) => {
                if let Some(Tok::Num(n)) = self.advance() {
                    Ok(Value::Num(*n))
                } else {
                    unreachable!()
                }
            }
            Some(Tok::Str(_)) => {
                if let Some(Tok::Str(s)) = self.advance() {
                    Ok(Value::Str(s.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Tok::Asset(_)) => {
                if let Some(Tok::Asset(s)) = self.advance() {
                    Ok(Value::Asset(s.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Tok::Ident(s)) if s == "None" => {
                self.advance();
                Ok(Value::None)
            }
            Some(Tok::Ident(_)) => {
                if let Some(Tok::Ident(s)) = self.advance() {
                    Ok(Value::Ident(s.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Tok::LParen) => {
                self.advance();
                let mut elems = Vec::new();
                loop {
                    if matches!(self.peek(), Some(Tok::RParen)) {
                        self.advance();
                        break;
                    }
                    elems.push(self.parse_value()?);
                    if matches!(self.peek(), Some(Tok::Comma)) {
                        self.advance();
                    }
                }
                Ok(Value::Tuple(elems))
            }
            Some(Tok::LBracket) => {
                self.advance();
                let mut elems = Vec::new();
                loop {
                    if matches!(self.peek(), Some(Tok::RBracket)) {
                        self.advance();
                        break;
                    }
                    elems.push(self.parse_value()?);
                    if matches!(self.peek(), Some(Tok::Comma)) {
                        self.advance();
                    }
                }
                Ok(Value::Array(elems))
            }
            Some(Tok::Path(_)) => {
                // Opaque path literal — we don't actually use these
                // (relationships / inherits / material:binding etc.
                // aren't part of the geometry extraction path).
                self.advance();
                Ok(Value::None)
            }
            other => Err(anyhow!("parse_value: unexpected token {other:?}")),
        }
    }

    /// Walk all prims from the current position.
    fn walk_prims(
        &mut self,
        out: &mut Vec<UsdMesh>,
        path_prefix: &str,
        parent_xform: Mat4,
    ) -> Result<()> {
        loop {
            match self.peek() {
                None | Some(Tok::RBrace) => return Ok(()),
                Some(Tok::Ident(w)) if (w == "def" || w == "over" || w == "class") => {
                    self.advance();
                    self.parse_prim(out, path_prefix, parent_xform)?;
                }
                _ => {
                    bail!(
                        "walk_prims: unexpected token {:?} at pos {}",
                        self.peek(),
                        self.pos
                    );
                }
            }
        }
    }

    fn parse_prim(
        &mut self,
        out: &mut Vec<UsdMesh>,
        path_prefix: &str,
        parent_xform: Mat4,
    ) -> Result<()> {
        // After `def` keyword: optional type, then "name" (quoted).
        let mut prim_type: Option<String> = None;
        if let Some(Tok::Ident(t)) = self.peek() {
            // The next token is the type unless it's immediately a string name
            // (schema-less "def"). USDA conventionally writes the type.
            prim_type = Some(t.clone());
            self.advance();
        }
        let name = match self.advance() {
            Some(Tok::Str(s)) => s.clone(),
            other => bail!("expected prim name string, got {other:?}"),
        };

        // Skip prim metadata (e.g. variant sets etc.)
        self.skip_optional_metadata_block()?;

        // Open body
        self.expect(&Tok::LBrace)?;

        let full_path = format!("{path_prefix}/{name}");
        let mut local_xform = Mat4::IDENTITY;
        let mut xform_op_order: Vec<String> = Vec::new();
        let mut xform_op_values: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();

        let mut mesh_points: Option<Vec<[f32; 3]>> = None;
        let mut mesh_fvc: Option<Vec<u32>> = None;
        let mut mesh_fvi: Option<Vec<u32>> = None;
        let mut mesh_st: Option<StPrimvar> = None;
        let mut mesh_st_indices: Option<Vec<u32>> = None;
        let mut mesh_st_interp: Interpolation = Interpolation::Unknown;
        let mut mesh_normals: Option<NormalPrimvar> = None;
        let mut mesh_normals_indices: Option<Vec<u32>> = None;
        let mut mesh_normals_interp: Interpolation = Interpolation::Unknown;

        // Walk body: either nested prims or attributes
        loop {
            match self.peek() {
                None => bail!("unterminated prim body for {full_path}"),
                Some(Tok::RBrace) => {
                    self.advance();
                    break;
                }
                Some(Tok::Ident(w)) if (w == "def" || w == "over" || w == "class") => {
                    // Nested prim — first finalize our xform so children inherit it
                    let local = compose_xform(&xform_op_order, &xform_op_values).unwrap_or(local_xform);
                    local_xform = local;
                    let world = parent_xform * local_xform;

                    self.advance();
                    self.parse_prim(out, &full_path, world)?;
                }
                Some(_) => {
                    // Attribute declaration: [variability] [custom] type name = value (metadata)
                    self.parse_attribute(
                        &mut local_xform,
                        &mut xform_op_order,
                        &mut xform_op_values,
                        &mut mesh_points,
                        &mut mesh_fvc,
                        &mut mesh_fvi,
                        &mut mesh_st,
                        &mut mesh_st_indices,
                        &mut mesh_st_interp,
                        &mut mesh_normals,
                        &mut mesh_normals_indices,
                        &mut mesh_normals_interp,
                    )?;
                }
            }
        }

        // Compose final local xform for this prim (if any xformOps were set)
        let local = compose_xform(&xform_op_order, &xform_op_values).unwrap_or(local_xform);
        let world = parent_xform * local;

        // Emit a mesh if this prim was a Mesh and had the required attrs
        if prim_type.as_deref() == Some("Mesh") {
            let points = mesh_points.ok_or_else(|| anyhow!("Mesh {full_path} missing 'points'"))?;
            let fvc = mesh_fvc
                .ok_or_else(|| anyhow!("Mesh {full_path} missing 'faceVertexCounts'"))?;
            let fvi = mesh_fvi
                .ok_or_else(|| anyhow!("Mesh {full_path} missing 'faceVertexIndices'"))?;

            let st = mesh_st.map(|mut st| {
                if st.indices.is_none() {
                    st.indices = mesh_st_indices;
                }
                if !matches!(mesh_st_interp, Interpolation::Unknown) {
                    st.interpolation = mesh_st_interp;
                }
                st
            });
            let normals = mesh_normals.map(|mut n| {
                if n.indices.is_none() {
                    n.indices = mesh_normals_indices;
                }
                if !matches!(mesh_normals_interp, Interpolation::Unknown) {
                    n.interpolation = mesh_normals_interp;
                }
                n
            });

            out.push(UsdMesh {
                path: full_path,
                points,
                face_vertex_counts: fvc,
                face_vertex_indices: fvi,
                st,
                normals,
                world_xform: world,
            });
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_attribute(
        &mut self,
        _local_xform: &mut Mat4,
        xform_op_order: &mut Vec<String>,
        xform_op_values: &mut std::collections::HashMap<String, Value>,
        mesh_points: &mut Option<Vec<[f32; 3]>>,
        mesh_fvc: &mut Option<Vec<u32>>,
        mesh_fvi: &mut Option<Vec<u32>>,
        mesh_st: &mut Option<StPrimvar>,
        mesh_st_indices: &mut Option<Vec<u32>>,
        mesh_st_interp: &mut Interpolation,
        mesh_normals: &mut Option<NormalPrimvar>,
        _mesh_normals_indices: &mut Option<Vec<u32>>,
        mesh_normals_interp: &mut Interpolation,
    ) -> Result<()> {
        // Consume leading tokens until we find `=`. The *last* Ident before `=` is
        // the attribute name; everything before it is type/variability modifiers
        // which we don't need.
        let mut collected: Vec<Tok> = Vec::new();
        while let Some(t) = self.peek() {
            if matches!(t, Tok::Eq) {
                break;
            }
            // Skip array-type brackets that appear as standalone `[` `]` tokens
            if matches!(t, Tok::LBrace | Tok::RBrace) {
                break;
            }
            match self.advance() {
                Some(tok) => collected.push(tok.clone()),
                None => break,
            }
        }
        // Attribute declaration without `=` (bare output / connectable
        // with no default) — skip it, there's nothing for the mesh
        // extractor to consume.
        if !matches!(self.peek(), Some(Tok::Eq)) {
            return Ok(());
        }
        self.advance();

        // Attribute name is the last Ident we collected
        let name = collected
            .iter()
            .rev()
            .find_map(|t| if let Tok::Ident(s) = t { Some(s.clone()) } else { None })
            .ok_or_else(|| anyhow!("parse_attribute: could not find name in {collected:?}"))?;

        // RHS value
        let value = self.parse_value()?;

        // Optional trailing metadata block for this attribute: e.g. `(interpolation = "faceVarying")`
        let mut interp_for_this_attr = Interpolation::Unknown;
        if matches!(self.peek(), Some(Tok::LParen)) {
            // Parse into a key/value map so we can pull interpolation out
            self.advance();
            while !matches!(self.peek(), Some(Tok::RParen) | None) {
                if let Some(Tok::Ident(k)) = self.peek().cloned() {
                    self.advance();
                    if matches!(self.peek(), Some(Tok::Eq)) {
                        self.advance();
                        let v = self.parse_value()?;
                        if k == "interpolation" {
                            if let Value::Str(s) = v {
                                interp_for_this_attr = match s.as_str() {
                                    "vertex" => Interpolation::Vertex,
                                    "faceVarying" => Interpolation::FaceVarying,
                                    "uniform" => Interpolation::Uniform,
                                    "constant" => Interpolation::Constant,
                                    _ => Interpolation::Unknown,
                                };
                            }
                        }
                    } else {
                        // e.g. `hidden` bare flag; ignore
                    }
                } else {
                    // Skip unknown tokens within metadata
                    self.advance();
                }
            }
            if matches!(self.peek(), Some(Tok::RParen)) {
                self.advance();
            }
        }

        // Dispatch by attribute name
        match name.as_str() {
            "points" => *mesh_points = Some(to_vec3_array(&value)?),
            "faceVertexCounts" => *mesh_fvc = Some(to_u32_array(&value)?),
            "faceVertexIndices" => *mesh_fvi = Some(to_u32_array(&value)?),
            "primvars:st" | "st" => {
                *mesh_st = Some(StPrimvar {
                    data: to_vec2_array(&value)?,
                    indices: None,
                    interpolation: interp_for_this_attr,
                });
                // If interp was found on this attr, stash for finalizer too
                if !matches!(interp_for_this_attr, Interpolation::Unknown) {
                    *mesh_st_interp = interp_for_this_attr;
                }
            }
            "primvars:st:indices" => *mesh_st_indices = Some(to_u32_array(&value)?),
            "normals" => {
                *mesh_normals = Some(NormalPrimvar {
                    data: to_vec3_array(&value)?,
                    indices: None,
                    interpolation: interp_for_this_attr,
                });
                if !matches!(interp_for_this_attr, Interpolation::Unknown) {
                    *mesh_normals_interp = interp_for_this_attr;
                }
            }
            "normals:indices" | "primvars:normals" => { /* ignore for prototype */ }
            "xformOpOrder" => {
                if let Value::Array(arr) = value {
                    for v in arr {
                        if let Value::Str(s) = v {
                            xform_op_order.push(s);
                        }
                    }
                }
            }
            n if n.starts_with("xformOp:") => {
                xform_op_values.insert(n.to_string(), value);
            }
            _ => { /* ignore unknown attrs for prototype */ }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Value extractors
// ---------------------------------------------------------------------------

fn to_f32(v: &Value) -> Result<f32> {
    match v {
        Value::Num(n) => Ok(*n as f32),
        _ => Err(anyhow!("expected number, got {v:?}")),
    }
}
fn to_u32(v: &Value) -> Result<u32> {
    match v {
        Value::Num(n) => {
            if *n < 0.0 {
                bail!("expected unsigned int, got {n}");
            }
            Ok(*n as u32)
        }
        _ => Err(anyhow!("expected int, got {v:?}")),
    }
}

fn to_u32_array(v: &Value) -> Result<Vec<u32>> {
    match v {
        Value::Array(arr) => arr.iter().map(to_u32).collect(),
        _ => Err(anyhow!("expected array, got {v:?}")),
    }
}

fn to_vec2_array(v: &Value) -> Result<Vec<[f32; 2]>> {
    let arr = match v {
        Value::Array(a) => a,
        _ => return Err(anyhow!("expected array of 2-tuples, got {v:?}")),
    };
    let mut out = Vec::with_capacity(arr.len());
    for elem in arr {
        match elem {
            Value::Tuple(t) if t.len() == 2 => out.push([to_f32(&t[0])?, to_f32(&t[1])?]),
            _ => bail!("vec2 array: unexpected element {elem:?}"),
        }
    }
    Ok(out)
}

fn to_vec3_array(v: &Value) -> Result<Vec<[f32; 3]>> {
    let arr = match v {
        Value::Array(a) => a,
        _ => return Err(anyhow!("expected array of 3-tuples, got {v:?}")),
    };
    let mut out = Vec::with_capacity(arr.len());
    for elem in arr {
        match elem {
            Value::Tuple(t) if t.len() == 3 => {
                out.push([to_f32(&t[0])?, to_f32(&t[1])?, to_f32(&t[2])?])
            }
            _ => bail!("vec3 array: unexpected element {elem:?}"),
        }
    }
    Ok(out)
}

fn to_vec3(v: &Value) -> Result<[f32; 3]> {
    match v {
        Value::Tuple(t) if t.len() == 3 => Ok([to_f32(&t[0])?, to_f32(&t[1])?, to_f32(&t[2])?]),
        _ => Err(anyhow!("expected vec3 tuple, got {v:?}")),
    }
}

fn to_mat4(v: &Value) -> Result<Mat4> {
    // matrix4d is a tuple of 4 tuples each of 4 numbers (row-major in USD).
    let rows = match v {
        Value::Tuple(t) if t.len() == 4 => t,
        _ => return Err(anyhow!("expected matrix4d, got {v:?}")),
    };
    let mut m = [[0.0f32; 4]; 4];
    for (i, row) in rows.iter().enumerate() {
        match row {
            Value::Tuple(r) if r.len() == 4 => {
                for j in 0..4 {
                    m[i][j] = to_f32(&r[j])?;
                }
            }
            _ => bail!("matrix4d row not 4-tuple: {row:?}"),
        }
    }
    // USD matrices are row-major with the translation in the last row.
    // glam::Mat4 is column-major, so transpose.
    Ok(Mat4::from_cols_array_2d(&[
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [m[3][0], m[3][1], m[3][2], m[3][3]],
    ]))
}

// ---------------------------------------------------------------------------
// Xform composition
// ---------------------------------------------------------------------------

fn compose_xform(
    order: &[String],
    values: &std::collections::HashMap<String, Value>,
) -> Option<Mat4> {
    if values.is_empty() {
        return None;
    }
    // If xformOpOrder wasn't specified, use the single op that was set (common case).
    let effective_order: Vec<String> = if order.is_empty() {
        values.keys().cloned().collect()
    } else {
        order.to_vec()
    };

    let mut m = Mat4::IDENTITY;
    for op in effective_order.iter() {
        let Some(v) = values.get(op.as_str()) else { continue };
        let op_m = match op.as_str() {
            "xformOp:transform" => match to_mat4(v) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("bad xformOp:transform value: {e}; ignoring");
                    Mat4::IDENTITY
                }
            },
            "xformOp:translate" => to_vec3(v)
                .map(|t| Mat4::from_translation(t.into()))
                .unwrap_or(Mat4::IDENTITY),
            "xformOp:scale" => to_vec3(v)
                .map(|s| Mat4::from_scale(s.into()))
                .unwrap_or(Mat4::IDENTITY),
            "xformOp:rotateX" => to_f32(v)
                .map(|a| Mat4::from_rotation_x(a.to_radians()))
                .unwrap_or(Mat4::IDENTITY),
            "xformOp:rotateY" => to_f32(v)
                .map(|a| Mat4::from_rotation_y(a.to_radians()))
                .unwrap_or(Mat4::IDENTITY),
            "xformOp:rotateZ" => to_f32(v)
                .map(|a| Mat4::from_rotation_z(a.to_radians()))
                .unwrap_or(Mat4::IDENTITY),
            "xformOp:rotateXYZ" => to_vec3(v)
                .map(|e| {
                    Mat4::from_rotation_x(e[0].to_radians())
                        * Mat4::from_rotation_y(e[1].to_radians())
                        * Mat4::from_rotation_z(e[2].to_radians())
                })
                .unwrap_or(Mat4::IDENTITY),
            other => {
                log::warn!("unsupported xformOp {other}; treating as identity");
                Mat4::IDENTITY
            }
        };
        // USD applies ops in the order they appear in xformOpOrder; each op
        // transforms the *result of the previous ops*. Using column vectors,
        // that means M_total = M_n * ... * M_1 * M_0.
        m = op_m * m;
    }
    Some(m)
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

pub fn parse_usda(text: &str) -> Result<Vec<UsdMesh>> {
    let tokens = tokenize(text)?;
    let mut p = Parser::new(&tokens);

    // Top-level stage metadata (after the #usda header which the tokenizer strips)
    p.skip_optional_metadata_block()?;

    let mut out = Vec::new();
    p.walk_prims(&mut out, "", Mat4::IDENTITY)?;
    Ok(out)
}
