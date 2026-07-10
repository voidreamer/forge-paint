use bytemuck::{Pod, Zeroable};

use crate::accel::AccelStructure;

/// A BVH node flattened for GPU traversal.
/// Internal nodes: left_child and right_child are node indices, tri_idx = 0xFFFFFFFF
/// Leaf nodes: tri_idx is the triangle index, left_child = right_child = 0xFFFFFFFF
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 3],
    pub left_child: u32,
    pub aabb_max: [f32; 3],
    pub right_child: u32,
    pub tri_idx: u32, // 0xFFFFFFFF for internal nodes
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// A triangle for GPU intersection testing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTriangle {
    pub v0: [f32; 3],
    pub _pad0: f32,
    pub v1: [f32; 3],
    pub _pad1: f32,
    pub v2: [f32; 3],
    pub _pad2: f32,
}

const INVALID: u32 = 0xFFFFFFFF;

/// Flattened BVH ready for GPU upload.
pub struct FlatBvh {
    pub nodes: Vec<GpuBvhNode>,
    pub triangles: Vec<GpuTriangle>,
}

impl FlatBvh {
    /// Flatten an AccelStructure's BVH into GPU-friendly arrays.
    pub fn from_accel(accel: &AccelStructure) -> Self {
        use bvh::bvh::BvhNode;

        let bvh_nodes = &accel.bvh.nodes;
        let mut nodes = Vec::with_capacity(bvh_nodes.len());
        let mut triangles = Vec::new();

        // Build the flat triangle array
        for tri in &accel.triangles {
            triangles.push(GpuTriangle {
                v0: [tri.v0.x, tri.v0.y, tri.v0.z],
                _pad0: 0.0,
                v1: [tri.v1.x, tri.v1.y, tri.v1.z],
                _pad1: 0.0,
                v2: [tri.v2.x, tri.v2.y, tri.v2.z],
                _pad2: 0.0,
            });
        }

        // Flatten the BVH nodes, preserving the crate's index layout
        for node in bvh_nodes {
            match node {
                BvhNode::Node {
                    child_l_aabb,
                    child_l_index,
                    child_r_aabb,
                    child_r_index,
                    ..
                } => {
                    // Internal node: merged AABB, store both child indices
                    let min_x = child_l_aabb.min.x.min(child_r_aabb.min.x);
                    let min_y = child_l_aabb.min.y.min(child_r_aabb.min.y);
                    let min_z = child_l_aabb.min.z.min(child_r_aabb.min.z);
                    let max_x = child_l_aabb.max.x.max(child_r_aabb.max.x);
                    let max_y = child_l_aabb.max.y.max(child_r_aabb.max.y);
                    let max_z = child_l_aabb.max.z.max(child_r_aabb.max.z);

                    nodes.push(GpuBvhNode {
                        aabb_min: [min_x, min_y, min_z],
                        left_child: *child_l_index as u32,
                        aabb_max: [max_x, max_y, max_z],
                        right_child: *child_r_index as u32,
                        tri_idx: INVALID,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    });
                }
                BvhNode::Leaf { shape_index, .. } => {
                    let tri = &accel.triangles[*shape_index];
                    let min = tri.v0.min(tri.v1).min(tri.v2);
                    let max = tri.v0.max(tri.v1).max(tri.v2);

                    nodes.push(GpuBvhNode {
                        aabb_min: [min.x, min.y, min.z],
                        left_child: INVALID,
                        aabb_max: [max.x, max.y, max.z],
                        right_child: INVALID,
                        tri_idx: *shape_index as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    });
                }
            }
        }

        FlatBvh { nodes, triangles }
    }
}
