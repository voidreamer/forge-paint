use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::ray::Ray;
use glam::Vec3;
use nalgebra::{Point3, Vector3};

use crate::mesh::Mesh;

/// A triangle stored in the BVH with precomputed data for fast intersection.
#[derive(Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub tri_index: usize,
    pub mesh_index: usize,
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

/// Result of a ray-triangle intersection.
#[derive(Debug, Clone, Copy)]
pub struct HitRecord {
    pub t: f32,
    pub u: f32, // barycentric u
    pub v: f32, // barycentric v
    pub tri_index: usize,
    pub mesh_index: usize,
    pub is_backface: bool,
}

/// Acceleration structure wrapping a BVH over all high-poly triangles.
pub struct AccelStructure {
    pub bvh: Bvh<f32, 3>,
    pub triangles: Vec<Triangle>,
}

impl AccelStructure {
    /// Build a BVH from one or more high-poly meshes.
    pub fn build(meshes: &[Mesh]) -> Self {
        let mut triangles = Vec::new();

        for (mesh_idx, mesh) in meshes.iter().enumerate() {
            for (tri_idx, tri) in mesh.indices.iter().enumerate() {
                let [v0, v1, v2] = mesh.tri_positions(tri);
                triangles.push(Triangle {
                    v0,
                    v1,
                    v2,
                    tri_index: tri_idx,
                    mesh_index: mesh_idx,
                    node_index: 0,
                });
            }
        }

        let bvh = Bvh::build(&mut triangles);

        AccelStructure { bvh, triangles }
    }

    /// Cast a ray and find the closest intersection.
    /// `ignore_backface`: if true, skip hits where the ray enters from behind the triangle.
    pub fn trace_closest(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_t: f32,
        min_t: f32,
        ignore_backface: bool,
    ) -> Option<HitRecord> {
        let ray = Ray::new(
            Point3::new(origin.x, origin.y, origin.z),
            Vector3::new(direction.x, direction.y, direction.z),
        );

        let hits = self.bvh.traverse(&ray, &self.triangles);

        let mut closest: Option<HitRecord> = None;

        for tri in hits {
            if let Some(hit) = ray_triangle_intersect(origin, direction, tri) {
                if hit.t < min_t || hit.t > max_t {
                    continue;
                }
                if ignore_backface && hit.is_backface {
                    continue;
                }
                match closest {
                    None => closest = Some(hit),
                    Some(ref c) if hit.t < c.t => closest = Some(hit),
                    _ => {}
                }
            }
        }

        closest
    }

    /// Test if any triangle occludes a ray within `max_t`. Fast early-out for AO.
    pub fn trace_any_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_t: f32,
        min_t: f32,
        ignore_backface: bool,
    ) -> bool {
        let ray = Ray::new(
            Point3::new(origin.x, origin.y, origin.z),
            Vector3::new(direction.x, direction.y, direction.z),
        );

        let hits = self.bvh.traverse(&ray, &self.triangles);

        for tri in hits {
            if let Some(hit) = ray_triangle_intersect(origin, direction, tri) {
                if hit.t > min_t && hit.t < max_t {
                    if ignore_backface && hit.is_backface {
                        continue;
                    }
                    return true;
                }
            }
        }

        false
    }
}

/// Moller-Trumbore ray-triangle intersection.
fn ray_triangle_intersect(origin: Vec3, dir: Vec3, tri: &Triangle) -> Option<HitRecord> {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);

    let is_backface = a < 0.0;

    if a.abs() < 1e-8 {
        return None; // parallel
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

    Some(HitRecord {
        t,
        u,
        v,
        tri_index: tri.tri_index,
        mesh_index: tri.mesh_index,
        is_backface,
    })
}
