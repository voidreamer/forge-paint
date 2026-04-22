//! BVH acceleration structure for ray-mesh picking.
//!
//! NOTE (2026-04-22): currently unused. Wiring this in as the pick path caused
//! visibly wrong hits (cursor-to-UV miss, backface hits) while the brute-force
//! Möller-Trumbore in `pick.rs` is correct on the same meshes. Diagnosis
//! pending — suspicion is a precision/AABB issue in bvh 0.9's intersect, or a
//! subtle contract I'm mis-using. At typical asset sizes (≲50k tris) brute
//! force is <1 ms per pick, so this isn't blocking anything.
#![allow(dead_code)]

use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use glam::{Vec2, Vec3};
use nalgebra::{Point3, Vector3};

use crate::mesh::CpuMesh;
use crate::pick::Hit;

#[derive(Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub tri_index: usize,
    node_index: usize,
}

impl Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> Aabb<f32, 3> {
        let min = self.v0.min(self.v1).min(self.v2);
        let max = self.v0.max(self.v1).max(self.v2);
        Aabb::with_bounds(
            Point3::new(min.x, min.y, min.z),
            Point3::new(max.x, max.y, max.z),
        )
    }
}

impl BHShape<f32, 3> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub struct MeshAccel {
    pub bvh: Bvh<f32, 3>,
    pub triangles: Vec<Triangle>,
}

impl MeshAccel {
    pub fn build(mesh: &CpuMesh) -> Self {
        let mut triangles: Vec<Triangle> = mesh
            .indices
            .iter()
            .enumerate()
            .map(|(i, tri)| Triangle {
                v0: mesh.positions[tri[0] as usize],
                v1: mesh.positions[tri[1] as usize],
                v2: mesh.positions[tri[2] as usize],
                tri_index: i,
                node_index: 0,
            })
            .collect();
        let bvh = Bvh::build(&mut triangles);
        Self { bvh, triangles }
    }

    /// Cast a ray, return the nearest hit with UV + world position interpolated
    /// from the owning mesh.
    pub fn pick(&self, mesh: &CpuMesh, origin: Vec3, direction: Vec3) -> Option<Hit> {
        let ray = Ray::new(
            Point3::new(origin.x, origin.y, origin.z),
            Vector3::new(direction.x, direction.y, direction.z),
        );
        let candidates = self.bvh.traverse(&ray, &self.triangles);

        let mut best: Option<(usize, f32, f32, f32)> = None;
        for tri in candidates {
            if let Some((t, u, v)) = ray_triangle(origin, direction, tri) {
                if best.map_or(true, |(_, pt, _, _)| t < pt) {
                    best = Some((tri.tri_index, t, u, v));
                }
            }
        }

        best.map(|(tri_idx, t, u, v)| {
            let tri = mesh.indices[tri_idx];
            let w = 1.0 - u - v;
            let uv: Vec2 = w * mesh.uvs[tri[0] as usize]
                + u * mesh.uvs[tri[1] as usize]
                + v * mesh.uvs[tri[2] as usize];
            let p0 = mesh.positions[tri[0] as usize];
            let p1 = mesh.positions[tri[1] as usize];
            let p2 = mesh.positions[tri[2] as usize];
            let world_pos = w * p0 + u * p1 + v * p2;
            Hit {
                tri: tri_idx,
                uv,
                world_pos,
                dist: t,
            }
        })
    }
}

fn ray_triangle(origin: Vec3, dir: Vec3, tri: &Triangle) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-8;
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < EPS {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t < 0.0 {
        return None;
    }
    Some((t, u, v))
}
