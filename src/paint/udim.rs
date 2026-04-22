//! UDIM tile conventions.
//!
//! A UDIM is a tile of UV space, numbered `1001 + u + 10 * v` where
//! `(u, v)` are non-negative integer offsets from the origin. Tile 1001
//! is the unit square `[0, 1] × [0, 1]`, tile 1002 is `[1, 2] × [0, 1]`,
//! tile 1011 is `[0, 1] × [1, 2]`, etc.

use std::collections::BTreeSet;

use crate::mesh::CpuMesh;

/// UDIM tile id from a UV sample. Negative UVs clamp to tile 1001.
#[inline]
pub fn tile_id(uv: [f32; 2]) -> u32 {
    let u = uv[0].floor().max(0.0) as u32;
    let v = uv[1].floor().max(0.0) as u32;
    1001 + u + 10 * v
}

/// Return the sorted set of UDIM tile ids touched by a mesh's UVs.
/// Uses a per-triangle bounding box so a triangle that straddles tiles
/// contributes to every tile it overlaps.
pub fn tiles_for_mesh(mesh: &CpuMesh) -> Vec<u32> {
    let mut set = BTreeSet::new();
    for tri in &mesh.indices {
        let a = mesh.uvs[tri[0] as usize];
        let b = mesh.uvs[tri[1] as usize];
        let c = mesh.uvs[tri[2] as usize];
        let min_u = a.x.min(b.x).min(c.x);
        let max_u = a.x.max(b.x).max(c.x);
        let min_v = a.y.min(b.y).min(c.y);
        let max_v = a.y.max(b.y).max(c.y);

        let tu_min = min_u.floor().max(0.0) as u32;
        let tu_max = max_u.floor().max(0.0) as u32;
        let tv_min = min_v.floor().max(0.0) as u32;
        let tv_max = max_v.floor().max(0.0) as u32;

        for tv in tv_min..=tv_max {
            for tu in tu_min..=tu_max {
                set.insert(1001 + tu + 10 * tv);
            }
        }
    }
    set.into_iter().collect()
}
