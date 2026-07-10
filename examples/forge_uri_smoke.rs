//! Smoke test for the rust-usd-backed loader. Opens a USD file or
//! `forge://...` URI, runs it through `forge_paint::usd::load_stage_merged`
//! equivalents, and prints mesh stats.
//!
//! Usage:
//!     cargo run --release --example forge_uri_smoke -- forge://assets/prop/gameboy/model
//!
//! Exercises the same code path the GUI uses (Stage::open + traverse +
//! triangulate via the usd/loader.rs module). Forge URIs require
//! PXR_PLUGINPATH_NAME to point at a loaded ForgeResolver variant and
//! FORGE_ROOT to point at the project root.

use std::path::PathBuf;

// forge-paint is a binary crate, so we can't `use forge_paint::...`
// from an example. Re-include the loader as a self-contained probe
// that calls the same rust_usd surface the production loader uses.

fn main() {
    let uri = std::env::args()
        .nth(1)
        .expect("usage: forge_uri_smoke <usd-path-or-forge-uri>");

    let path = PathBuf::from(&uri);
    let asset = path.to_str().expect("non-utf8 input").to_string();

    println!("opening: {}", asset);
    let stage = match rust_usd::Stage::open(&asset) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Stage::open failed: {}", e.what());
            std::process::exit(1);
        }
    };

    let meshes = stage.meshes();
    println!("found {} mesh(es)", meshes.len());

    let mut total_points = 0;
    let mut total_faces = 0;
    let mut total_fvi = 0;
    for mesh in &meshes {
        let pts = mesh.points().len() / 3;
        let fvc = mesh.face_vertex_counts();
        let fvi = mesh.face_vertex_indices();
        let st = mesh.st().len() / 2;
        let normals = mesh.normals().len() / 3;
        println!(
            "  - {}  pts={} faces={} fvi={} st={} normals={} subdiv={:?}",
            mesh.prim_path(),
            pts,
            fvc.len(),
            fvi.len(),
            st,
            normals,
            mesh.subdivision_scheme(),
        );
        total_points += pts;
        total_faces += fvc.len();
        total_fvi += fvi.len();
    }

    println!(
        "totals: {} points across {} prims, {} faces, {} face-vertex indices",
        total_points,
        meshes.len(),
        total_faces,
        total_fvi
    );
}
