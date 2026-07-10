# Texture Baking: Deep Technical Research on Algorithms and Implementation

## Table of Contents

1. [Ray Casting from Low-Poly to High-Poly](#1-ray-casting-from-low-poly-to-high-poly)
2. [BVH / Acceleration Structures](#2-bvh--acceleration-structures-for-ray-mesh-intersection)
3. [Normal Map Baking Algorithm](#3-normal-map-baking-algorithm)
4. [Ambient Occlusion Baking](#4-ambient-occlusion-baking)
5. [Curvature Computation Algorithms](#5-curvature-computation-algorithms)
6. [Anti-Aliasing Strategies](#6-anti-aliasing-strategies-for-baked-textures)
7. [Padding / Dilation of UV Islands](#7-paddingdilation-of-uv-islands)
8. [Multiple Samples Per Texel Averaging](#8-how-multiple-samples-per-texel-are-averaged)

---

## 1. Ray Casting from Low-Poly to High-Poly

### 1.1 Overall Pipeline

The texture baking pipeline operates as follows:

1. **UV-Space Rasterization**: The low-poly mesh's triangles are rasterized in UV space (texture space), not screen space. For each texel in the output texture, the system determines which low-poly triangle covers that texel.

2. **Attribute Interpolation**: Using barycentric coordinates within the covering triangle, the system interpolates the world-space position, normal, tangent, and bitangent at the texel's location on the low-poly surface.

3. **Ray Construction**: A ray is constructed at the interpolated position, directed outward along the interpolated normal (or cage direction).

4. **Ray Casting**: The ray is traced against the high-poly mesh to find the closest intersection.

5. **Signal Sampling**: At the intersection point, the desired signal (normal, color, AO, etc.) is sampled from the high-poly and written to the texel.

### 1.2 How Rays Are Generated Per Texel

**UV-Space Rasterization** is the foundation. Each low-poly triangle is rasterized in UV coordinates as if the UV (u,v) values were screen coordinates. The process mirrors classic software rasterization:

```
For each low-poly triangle with vertices (v0, v1, v2):
    UV coordinates: (uv0, uv1, uv2)
    World positions: (p0, p1, p2)
    World normals:   (n0, n1, n2)

    Rasterize the triangle in UV space (treating u,v as x,y pixel coords):
        For each texel (tx, ty) covered by the UV triangle:
            Compute barycentric coordinates (w0, w1, w2) of texel center within UV triangle
            Interpolated world position: P = w0*p0 + w1*p1 + w2*p2
            Interpolated world normal:   N = normalize(w0*n0 + w1*n1 + w2*n2)
            
            Construct ray:
                ray.origin    = P + N * extrusion_distance
                ray.direction = -N
                ray.max_t     = max_ray_distance
```

The rasterization must follow **top-left fill rules** (same as GPU hardware rasterization) to prevent double-filling texels at shared triangle edges. Many implementations also use **conservative rasterization** to ensure no texel that partially overlaps a triangle is missed.

### 1.3 Direction of Rays

There are two primary strategies for determining ray direction:

#### Along Vertex Normal (Default/Simple)

The ray direction is the interpolated vertex normal of the low-poly surface at the texel position. The ray origin is offset outward along this normal by an "extrusion" distance, then cast inward (negative normal direction). This is the simplest approach and what tools like Blender use by default when no cage is specified.

**Limitation**: At hard edges and smoothing group boundaries, the averaged normals can cause rays to diverge or converge in undesirable ways, producing "skewed" bakes on floating geometry details.

#### Along Cage Direction (Professional Workflow)

A **cage mesh** is a ballooned-out copy of the low-poly mesh. The ray direction at each texel is determined by the vector from the cage vertex to the corresponding low-poly vertex:

```
cage_direction[vertex_i] = cage_position[vertex_i] - lowpoly_position[vertex_i]
```

During UV-space rasterization, the cage direction is interpolated per-texel using barycentric coordinates, just like the position and normal:

```
For each texel:
    cage_dir = normalize(w0*cage_dir0 + w1*cage_dir1 + w2*cage_dir2)
    cage_pos = w0*cage_pos0 + w1*cage_pos1 + w2*cage_pos2
    
    ray.origin    = cage_pos
    ray.direction = -cage_dir  (inward toward low-poly)
```

**From NVIDIA GPU Gems 3 (Chapter 22)**:
```hlsl
Ray ray;
ray.dir = -Input.cageDir.xyz;
ray.origin = Input.position + Input.cageDir.xyz;
```

#### Average Normals (Substance Painter)

Substance Painter offers an "Average Normals" mode that averages vertex normals across smoothing group boundaries to determine ray direction. This functions like an implicit cage but without requiring explicit cage geometry. The averaged normal direction differs from the per-smoothing-group normal, providing smoother ray coverage at hard edges.

### 1.4 How Intersections Are Found

The core ray-triangle intersection algorithm universally used is the **Moller-Trumbore algorithm** (1997):

**Mathematical Foundation**:

Given ray `R(t) = O + t*D` and triangle with vertices `v0, v1, v2`:

```
e1 = v1 - v0
e2 = v2 - v0
h  = D x e2          (cross product)
a  = e1 . h           (dot product = determinant)

if |a| < epsilon: ray is parallel, no intersection

f = 1/a
s = O - v0
u = f * (s . h)
if u < 0 or u > 1: miss

q = s x e1
v = f * (D . q)
if v < 0 or u + v > 1: miss

t = f * (e2 . q)
if t > epsilon:
    intersection at R(t), barycentric coords (1-u-v, u, v)
```

This yields:
- **t**: distance along ray to intersection
- **u, v**: barycentric coordinates on the triangle (used for interpolating normals, UVs, etc.)

For high-poly meshes with millions of triangles, brute-force intersection testing is infeasible. Acceleration structures (Section 2) reduce the number of triangles tested per ray from O(N) to O(log N).

### 1.5 Handling Ray Misses and Backface Hits

**Ray Misses**: When a ray finds no intersection within `max_ray_distance`, the texel is marked as "empty" or filled with a default value (e.g., flat tangent-space normal [0.5, 0.5, 1.0] for normal maps, or white for AO). The texel is flagged for later dilation/padding.

**Backface Hits**: When a ray intersects a backface of the high-poly mesh (the dot product of the ray direction and the triangle normal is positive), there are several strategies:

1. **Ignore backfaces entirely** -- skip the intersection and continue searching for front-face hits
2. **Accept backfaces** -- useful when the high-poly has open geometry or inverted normals
3. **Configurable behavior** -- tools like Substance Painter offer self-occlusion modes:
   - "Always": accept all hits
   - "Only Same Mesh Name": only accept hits on geometry with matching naming convention
   - "Only Different Mesh Name": only accept hits on other parts

**Bias/Offset**: To prevent self-intersection artifacts (where the ray immediately hits the low-poly surface it originated from), a small bias is applied. The ray origin is offset along the normal by a small epsilon before casting. In Substance Painter, this is controlled by the bias parameter (typically ~0.1).

**Multiple Intersection Selection**: When a ray intersects multiple high-poly triangles, the system selects the closest intersection to the ray origin (smallest positive t value). This ensures that the surface detail nearest to the low-poly is captured.

---

## 2. BVH / Acceleration Structures for Ray-Mesh Intersection

### 2.1 Overview of Spatial Data Structures

Two primary acceleration structures are used for texture baking:

1. **Bounding Volume Hierarchy (BVH)** -- the dominant choice in modern implementations
2. **Uniform Grid** -- simpler, used in some GPU implementations

**BVH** is a tree structure where:
- Each internal node stores an axis-aligned bounding box (AABB) that encloses all geometry in its subtree
- Leaf nodes contain a small set of primitives (triangles)
- Ray traversal skips entire subtrees when the ray doesn't intersect a node's AABB

**Uniform Grid** divides space into equal-sized voxels. Each voxel contains a list of triangles overlapping it. The ray traverses voxels in order using 3D-DDA (Digital Differential Analyzer). This approach was used in the NVIDIA GPU Gems 3 baking implementation.

### 2.2 BVH Data Structure

The standard BVH node is designed for cache efficiency at 32 bytes:

```cpp
struct BVHNode {
    float3 aabbMin;     // 12 bytes - minimum corner of bounding box
    float3 aabbMax;     // 12 bytes - maximum corner of bounding box
    uint32 leftFirst;   // 4 bytes  - index of left child OR first triangle
    uint32 triCount;    // 4 bytes  - 0 for interior nodes, >0 for leaves
};
// Total: 32 bytes (one cache line on many architectures)
```

Encoding convention:
- If `triCount > 0`: this is a **leaf node**; `leftFirst` is the index of the first triangle in the primitive array
- If `triCount == 0`: this is an **interior node**; `leftFirst` is the index of the left child (right child is always at `leftFirst + 1`)

Triangles are referenced indirectly through an index array `triIdx[]`, allowing primitives to be reordered without moving vertex data.

### 2.3 BVH Construction: Top-Down Recursive Subdivision

The standard construction algorithm:

```
function BuildBVH():
    For each triangle: compute centroid = (v0 + v1 + v2) / 3
    Create root node containing all N triangles
    Update root's AABB to enclose all triangles
    Subdivide(root)

function Subdivide(node):
    if node.triCount <= MAX_PRIMS_PER_LEAF: return  (termination)
    
    Choose split axis and position (see SAH below)
    Partition triangles: those with centroid < splitPos go left, rest go right
    
    Create left child with left partition
    Create right child with right partition
    Update both children's AABBs
    
    Subdivide(left child)
    Subdivide(right child)
```

**Primitive Partitioning** uses an in-place QuickSort-like approach:
```cpp
int i = node.firstTriIdx;
int j = i + node.triCount - 1;
while (i <= j) {
    if (tri[triIdx[i]].centroid[axis] < splitPos)
        i++;
    else
        swap(triIdx[i], triIdx[j--]);
}
```

### 2.4 Surface Area Heuristic (SAH)

The SAH is the key quality metric for BVH construction. It estimates the expected cost of traversing a node:

**Cost of making a node a leaf**:
```
C_leaf = N * t_isect
```

**Cost of splitting a node into children A and B**:
```
C_split = t_trav + P(A) * N_A * t_isect + P(B) * N_B * t_isect
```

Where:
- `t_trav` = cost of traversing an interior node (typically ~1.0)
- `t_isect` = cost of a ray-triangle intersection (typically ~1.0, or sometimes ~8x t_trav)
- `P(A)` = probability a ray passing through the parent also passes through child A
- `N_A`, `N_B` = number of primitives in each child

**Geometric Probability**: `P(A|parent) = SurfaceArea(A) / SurfaceArea(parent)`

This is based on the theorem that the probability of a random ray intersecting a convex object inside another convex object is proportional to the ratio of their surface areas.

#### Binned SAH (Practical Implementation)

Full SAH evaluation considers every possible split plane, which is O(N) per axis. **Binned SAH** reduces this to O(K) where K is typically 8-16 bins:

```
For each axis (x, y, z):
    Divide centroid extent into K bins of equal width
    For each triangle: assign to bin based on centroid position
    For each possible split (K-1 candidates):
        Compute left AABB (union of bins 0..i) and count
        Compute right AABB (union of bins i+1..K-1) and count
        cost[i] = t_trav + (count_left * SA_left + count_right * SA_right) / SA_parent
    
    Choose split with minimum cost
    
Compare best split cost against leaf cost (N * t_isect)
If split is cheaper: subdivide; otherwise: create leaf
```

From PBRT's implementation:
```cpp
Float cost[nBuckets - 1];
for (int i = 0; i < nBuckets - 1; ++i) {
    Bounds3f b0, b1;
    int count0 = 0, count1 = 0;
    for (int j = 0; j <= i; ++j) {
        b0 = Union(b0, buckets[j].bounds);
        count0 += buckets[j].count;
    }
    for (int j = i+1; j < nBuckets; ++j) {
        b1 = Union(b1, buckets[j].bounds);
        count1 += buckets[j].count;
    }
    cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea())
                      / bounds.SurfaceArea();
}
```

### 2.5 BVH Ray Traversal

```cpp
void IntersectBVH(Ray& ray, const uint nodeIdx) {
    BVHNode& node = bvhNode[nodeIdx];
    
    // Test ray against node's AABB using slab method
    if (!IntersectAABB(ray, node.aabbMin, node.aabbMax)) return;
    
    if (node.triCount > 0) {
        // Leaf: test all triangles
        for (uint i = 0; i < node.triCount; i++)
            IntersectTri(ray, tri[triIdx[node.firstTriIdx + i]]);
    } else {
        // Interior: recurse into both children
        IntersectBVH(ray, node.leftFirst);
        IntersectBVH(ray, node.leftFirst + 1);
    }
}
```

**AABB Intersection (Slab Test)**:
```cpp
bool IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax) {
    float tx1 = (bmin.x - ray.O.x) / ray.D.x;
    float tx2 = (bmax.x - ray.O.x) / ray.D.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    
    float ty1 = (bmin.y - ray.O.y) / ray.D.y;
    float ty2 = (bmax.y - ray.O.y) / ray.D.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    
    float tz1 = (bmin.z - ray.O.z) / ray.D.z;
    float tz2 = (bmax.z - ray.O.z) / ray.D.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
    
    return tmax >= tmin && tmin < ray.t && tmax > 0;
}
```

### 2.6 Uniform Grid (GPU Alternative)

Used in NVIDIA GPU Gems 3's baking implementation:

1. **Construction**: The high-poly mesh's AABB is subdivided into a 3D grid. Each cell stores a linked list (or index list) of triangles whose AABBs overlap the cell.

2. **Traversal via 3D-DDA**:
```hlsl
float3 tmax = (cellBoundary - ray.origin) / ray.dir;
float3 step = sign(ray.dir);  // +1 or -1 per axis
float3 tdelta = abs(cellSize / ray.dir);

while (inGrid) {
    TestTrianglesInCell(currentCell);
    
    // Step to next cell along axis with smallest tmax
    if (tmax.x < tmax.y && tmax.x < tmax.z) {
        currentCell.x += step.x;
        tmax.x += tdelta.x;
    } else if (tmax.y < tmax.z) {
        currentCell.y += step.y;
        tmax.y += tdelta.y;
    } else {
        currentCell.z += step.z;
        tmax.z += tdelta.z;
    }
}
```

The grid is stored as a 3D texture on the GPU, making it well-suited for GPU-based baking.

### 2.7 Intel Embree for Production Baking

Many production bakers (including some used internally at Adobe) use **Intel Embree**, a highly optimized CPU ray tracing library:

```cpp
// Setup
RTCDevice device = rtcNewDevice(NULL);
RTCScene scene = rtcNewScene(device);
RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

// Set vertex and index buffers
rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, ...);
rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, ...);
rtcCommitGeometry(geom);
rtcAttachGeometry(scene, geom);
rtcCommitScene(scene);  // builds BVH internally

// Per-texel ray query
RTCRayHit rayhit;
rayhit.ray.org_x = P.x;  rayhit.ray.org_y = P.y;  rayhit.ray.org_z = P.z;
rayhit.ray.dir_x = D.x;  rayhit.ray.dir_y = D.y;  rayhit.ray.dir_z = D.z;
rayhit.ray.tnear = 0.001f;
rayhit.ray.tfar = max_distance;
rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

rtcIntersect1(scene, &rayhit);

if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    // Hit found: rayhit.hit.u, rayhit.hit.v are barycentric coords
    // rayhit.ray.tfar is hit distance
    // rayhit.hit.primID is triangle index
}
```

Embree internally builds an optimized BVH using SAH with SIMD-accelerated traversal (SSE/AVX), achieving state-of-the-art performance for CPU ray tracing.

### 2.8 Linear BVH and HLBVH (GPU Construction)

For GPU-based bakers, fast BVH construction is critical:

**Morton Code Encoding** maps 3D positions to 1D:
```cpp
uint32 EncodeMorton3(float3 v) {
    // Interleave bits of x, y, z (10 bits each = 30 bits total)
    return (LeftShift3(v.z) << 2) | (LeftShift3(v.y) << 1) | LeftShift3(v.x);
}
```

**HLBVH Algorithm**:
1. Compute Morton codes for all triangle centroids
2. Radix sort by Morton code (O(N), 5 passes of 6 bits each)
3. Identify clusters sharing the top 12 Morton bits (4096 grid cells)
4. Build treelets within each cluster using remaining 18 bits
5. Connect treelets at the top level using SAH

This enables parallel BVH construction on the GPU while maintaining near-SAH quality.

---

## 3. Normal Map Baking Algorithm

### 3.1 The Tangent Basis (TBN Matrix)

A tangent-space normal map encodes surface perturbations relative to a local coordinate frame called the **tangent basis** or **TBN matrix**, constructed at each point on the low-poly surface:

- **T** (Tangent): aligned with the U direction of the texture coordinate
- **B** (Bitangent/Binormal): aligned with the V direction
- **N** (Normal): perpendicular to the surface

The TBN matrix is a 3x3 rotation matrix:
```
TBN = [Tx Bx Nx]
      [Ty By Ny]
      [Tz Bz Nz]
```

**Computing T and B from UV coordinates** (per triangle):

Given triangle vertices P0, P1, P2 with UVs (u0,v0), (u1,v1), (u2,v2):
```
Edge1 = P1 - P0,  Edge2 = P2 - P0
dUV1 = (u1-u0, v1-v0),  dUV2 = (u2-u0, v2-v0)

f = 1.0 / (dUV1.x * dUV2.y - dUV2.x * dUV1.y)

T.x = f * (dUV2.y * Edge1.x - dUV1.y * Edge2.x)
T.y = f * (dUV2.y * Edge1.y - dUV1.y * Edge2.y)
T.z = f * (dUV2.y * Edge1.z - dUV1.y * Edge2.z)

B.x = f * (-dUV2.x * Edge1.x + dUV1.x * Edge2.x)
B.y = f * (-dUV2.x * Edge1.y + dUV1.x * Edge2.y)
B.z = f * (-dUV2.x * Edge1.z + dUV1.x * Edge2.z)
```

### 3.2 MikkTSpace: The Standard Tangent Space Algorithm

**MikkTSpace** (by Morten S. Mikkelsen) is the industry-standard algorithm for computing tangent bases, used by Substance Painter, Blender, Unity, Unreal Engine, xNormal, and others. Its key properties:

**Why MikkTSpace Exists**: Different tools historically computed tangent bases differently. If the baker and the renderer use different tangent bases, normal maps display incorrect lighting, especially at UV seam boundaries. MikkTSpace provides a single canonical algorithm so baking and rendering are guaranteed to match.

**Algorithm Steps** (based on the reference implementation and Mikkelsen's thesis):

1. **Per-Face Tangent Computation**: For each face (triangle or quad), compute the tangent and bitangent from the UV gradient (as shown above).

2. **Internal Welding**: Vertices that share the same position, normal, and UV are welded together (treated as the same vertex). This is done in an **order-independent** manner -- face and vertex ordering do not affect results.

3. **Smoothing Group Respecting**: Tangents are only averaged across faces that share the same smoothing group. At hard edges (smoothing breaks), tangent spaces are computed independently.

4. **Weighted Averaging**: Per-face tangents are averaged at each vertex, weighted by the face's contribution (typically by angle at the vertex). The resulting tangent vector is orthogonalized against the vertex normal using Gram-Schmidt:
   ```
   T' = normalize(T - N * dot(N, T))
   ```

5. **Bitangent Sign**: Rather than storing a full bitangent vector, MikkTSpace computes a sign (handedness):
   ```
   sign = dot(cross(N, T), B) < 0 ? -1 : +1
   ```
   The bitangent is reconstructed at runtime: `B = sign * cross(N, T)`

6. **Degenerate Handling**: Degenerate primitives (collapsed triangles) inherit tangent space from neighboring valid primitives, ensuring no NaN or zero vectors propagate.

**Interface** (from mikktspace.h):
```c
typedef struct {
    int (*m_getNumFaces)(const SMikkTSpaceContext*);
    int (*m_getNumVerticesOfFace)(const SMikkTSpaceContext*, int);
    void (*m_getPosition)(const SMikkTSpaceContext*, float[], int, int);
    void (*m_getNormal)(const SMikkTSpaceContext*, float[], int, int);
    void (*m_getTexCoord)(const SMikkTSpaceContext*, float[], int, int);
    
    // Output (choose one):
    void (*m_setTSpaceBasic)(const SMikkTSpaceContext*, const float[], float, int, int);
    void (*m_setTSpace)(const SMikkTSpaceContext*, const float[], const float[],
                        float, float, tbool, int, int);
} SMikkTSpaceInterface;
```

The `m_setTSpaceBasic` callback returns:
- `fvTangent[3]`: unit tangent vector
- `fSign`: sign of the bitangent (+1 or -1)

### 3.3 How High-Poly Normal Is Transformed to Tangent Space

The core operation of normal map baking:

1. **Find intersection** on high-poly mesh at the texel's projected position
2. **Interpolate the high-poly face normal** at the intersection point using barycentric coordinates:
   ```
   N_highpoly = normalize(w0*n0 + w1*n1 + w2*n2)
   ```
   Where n0, n1, n2 are the vertex normals of the intersected high-poly triangle.

3. **Transform to tangent space** of the low-poly at that texel:
   ```
   // TBN matrix columns are T, B, N of the low-poly at this texel
   // To transform from world to tangent space, use the TRANSPOSE (inverse of rotation):
   N_tangent.x = dot(N_highpoly, T)   // or dot(N_highpoly, B) depending on convention
   N_tangent.y = dot(N_highpoly, B)   // or dot(N_highpoly, T)
   N_tangent.z = dot(N_highpoly, N)
   N_tangent = normalize(N_tangent)
   ```

4. **Encode and store**:
   ```
   texel_color = N_tangent * 0.5 + 0.5   // Map from [-1,1] to [0,1]
   ```

**From NVIDIA GPU Gems 3**:
```hlsl
float3 normalTS;
normalTS.x = dot(normal, Input.binormal);
normalTS.y = dot(normal, Input.tangent);
normalTS.z = dot(normal, Input.normal);
normalTS = normalize(normalTS);
return float4(normalTS * 0.5 + 0.5, 1);
```

Note: The order of T, B, N in the dot products depends on the convention (OpenGL vs DirectX Y-axis convention).

### 3.4 Synced vs Unsynced Tangent Space

**Synced tangent space** means the baker and the runtime renderer use the **exact same** tangent basis algorithm. This is critical because:

- The normal map encodes a **perturbation** relative to the tangent basis
- If the renderer reconstructs a different tangent basis than the baker assumed, the decoded world-space normal will be wrong
- This manifests as lighting errors, especially visible along UV seams

**Common scenarios**:
- **Synced** (correct): Substance Painter (MikkTSpace) -> Unreal Engine (MikkTSpace)
- **Synced** (correct): xNormal (MikkTSpace) -> Unity (MikkTSpace)
- **Unsynced** (incorrect): 3ds Max native tangents baked in 3ds Max -> Unity (MikkTSpace)

**Why seams are the canary**: At UV seam boundaries, the tangent basis is discontinuous. If the baker and renderer agree on how this discontinuity is handled, the normal map compensates perfectly. If they disagree, the compensation fails and a visible lighting seam appears.

**Per-fragment vs per-vertex tangent space**: Some older renderers computed tangent space per-fragment (using screen-space derivatives), which cannot match any per-vertex baker. Modern engines universally use per-vertex tangent space (MikkTSpace) interpolated across the triangle.

---

## 4. Ambient Occlusion Baking

### 4.1 Mathematical Formulation

Ambient Occlusion at a surface point **p** with normal **n** is defined as:

```
AO(p, n) = (1/pi) * integral_over_hemisphere V(p, w) * max(cos(theta), 0) dw
```

Where:
- The integral is over the hemisphere oriented around **n**
- `V(p, w)` = visibility function: 0 if a ray from **p** in direction **w** hits geometry within some distance, 1 otherwise
- `theta` = angle between **w** and **n**
- The `cos(theta)` term weights directions near the normal more heavily (Lambert's cosine law)

### 4.2 Monte Carlo Approximation

The integral is approximated via Monte Carlo sampling:

```
AO(p, n) ~= (1/N) * sum_{i=1}^{N} V(p, w_i) * weight(w_i)
```

The weight depends on the sampling distribution:

- **Uniform hemisphere sampling**: `weight = 2 * cos(theta)` (compensating for non-cosine PDF)
- **Cosine-weighted sampling**: `weight = 1` (the cosine term is baked into the PDF)

### 4.3 Hemisphere Sampling Strategies

#### Uniform Hemisphere Sampling

Generates directions with equal probability across the hemisphere:

```cpp
Vector3 UniformSampleHemisphere(float u1, float u2) {
    float r = sqrt(1.0f - u1 * u1);
    float phi = 2.0f * PI * u2;
    return Vector3(cos(phi) * r, sin(phi) * r, u1);
}
// PDF = 1 / (2*PI)
```

#### Cosine-Weighted Hemisphere Sampling (Preferred)

Generates more samples near the normal, matching the cosine weighting in the AO integral:

```cpp
Vector3 CosineSampleHemisphere(float u1, float u2) {
    float r = sqrt(u1);
    float theta = 2.0f * PI * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    return Vector3(x, y, sqrt(1.0f - u1));
}
// PDF = cos(theta) / PI
```

**Why cosine-weighted is better**: The Monte Carlo estimator simplifies because the cosine term in the integrand cancels with the PDF:
- Uniform: `AO ~= (2/N) * sum(V_i * cos(theta_i))`
- Cosine-weighted: `AO ~= (1/N) * sum(V_i)`

The cosine-weighted version has ~30% lower variance (less noise at the same sample count).

#### Stratified Sampling

Divide the hemisphere into equal-area sectors and take one sample per sector, with jitter:

```
For each of N_phi * N_theta strata:
    u1 = (stratum_theta + random()) / N_theta
    u2 = (stratum_phi + random()) / N_phi
    Generate direction from (u1, u2) using cosine-weighted formula
```

This ensures better coverage than pure random sampling.

#### Hammersley / Low-Discrepancy Sequences

The **Hammersley point set** provides deterministic, well-distributed samples:

```glsl
float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley2d(uint i, uint N) {
    return vec2(float(i) / float(N), radicalInverse_VdC(i));
}
```

Then mapped to hemisphere:
```glsl
vec3 hemisphereSample_cos(float u, float v) {
    float phi = v * 2.0 * PI;
    float cosTheta = sqrt(1.0 - u);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
```

### 4.4 Complete AO Baking Algorithm

```
For each texel in the output texture:
    Determine world-space position P and normal N on low-poly surface
    
    Construct tangent frame (T, B, N) for hemisphere orientation
    
    occluded_count = 0
    For i = 0 to num_samples - 1:
        Generate local direction w_local = CosineSampleHemisphere(u1, u2)
        Transform to world space: w_world = T * w_local.x + B * w_local.y + N * w_local.z
        
        Construct ray: origin = P + N * bias, direction = w_world
        
        if ray_intersects_scene(ray, max_distance):
            occluded_count++
    
    AO = 1.0 - (occluded_count / num_samples)
    Write AO to texel
```

### 4.5 Number of Rays and Quality

Typical sample counts in production tools:
- **Preview quality**: 16-64 samples per texel
- **Medium quality**: 128-256 samples per texel
- **High quality**: 512-2048 samples per texel
- **NVIDIA GPU Gems reference**: 512 rays per triangle

More samples reduce noise but increase computation time linearly. The relationship between noise and sample count follows: `noise ~= 1/sqrt(N)`.

### 4.6 Self-Occlusion vs External Occlusion

**Self-Occlusion**: The mesh occluding itself. This is the most common form of AO and captures creases, folds, and cavities. To prevent the ray from immediately intersecting the surface it originated from, a **bias** (small offset along the normal) is applied.

**External Occlusion**: Other objects in the scene occluding the point. When baking "Selected to Active," only the high-poly mesh participates. Some tools (Substance Painter) allow configuring which meshes contribute:
- "Self Occlusion: Always" -- all geometry contributes
- "Self Occlusion: Only Same Mesh Name" -- only geometry with matching naming prefix
- "Self Occlusion: Only Different Mesh Name" -- only other objects

**Occlusion Distance (Falloff)**: The `max_distance` parameter limits how far AO rays search. Short distances emphasize fine detail (cavity-like), while long distances capture broader shadowing. Typical default: 20% of the model's bounding box diagonal.

### 4.7 Ground Plane Occlusion

Some bakers add a virtual infinite ground plane at the object's lowest point. Any AO ray that would pass below this plane is treated as occluded. This simulates the object sitting on a surface and produces more natural-looking bottom-hemisphere darkening.

Implementation:
```
if w_world.y < 0 and (P + w_world * t).y < ground_y:
    treat as occluded
```

### 4.8 Bent Normal (Related Output)

The **bent normal** is the average direction of unoccluded rays:

```
bent_normal = Vector3(0, 0, 0)
For each unoccluded ray:
    bent_normal += ray_direction
bent_normal = normalize(bent_normal)
```

This indicates the dominant direction from which ambient light reaches the surface, useful for environment lighting.

---

## 5. Curvature Computation Algorithms

Curvature maps encode the rate of change of the surface normal. Convex areas (edges, ridges) are white, concave areas (cavities, creases) are black, and flat areas are 50% gray.

### 5.1 Method 1: Normal Map Derivative (Sobel Filter / Per-Pixel)

This is the method used by Substance Painter's standard curvature baker:

1. **Render a world-space normal map** of the mesh (or use the baked tangent-space normal converted to world space)
2. **Apply a Sobel filter** or similar derivative operator to compute the rate of change of the normal across the texture:

```
For each texel (x, y):
    // Sample neighboring normals
    N_left  = normal_map(x-1, y)
    N_right = normal_map(x+1, y)
    N_up    = normal_map(x, y-1)
    N_down  = normal_map(x, y+1)
    
    // Compute derivatives (Sobel or central differences)
    dN_dx = (N_right - N_left) / 2
    dN_dy = (N_down - N_up) / 2
    
    // Divergence of the normal field approximates mean curvature
    curvature = dot(dN_dx, tangent_x) + dot(dN_dy, tangent_y)
    
    // Map to [0, 1]: 0.5 = flat, >0.5 = convex, <0.5 = concave
    output = curvature * scale * 0.5 + 0.5
```

**Advantages**: Fast, works purely in 2D image space, can be computed from any normal map.
**Disadvantages**: Resolution-dependent, can miss fine geometry not captured in the normal map.

### 5.2 Method 2: Ray-Based (Curvature from Mesh / Per-Vertex)

Used by Substance's "Curvature from Mesh" baker, this traces secondary rays:

1. At each surface point, cast N rays (default: 32) into a small hemisphere
2. Rays that hit nearby geometry quickly indicate concavity; rays that travel far indicate convexity
3. The distribution of hit distances encodes the local curvature

This is conceptually similar to AO but with a very short search radius (typically 0.1-0.5 scene units).

### 5.3 Method 3: Discrete Differential Geometry (Mesh-Based)

This method works directly on the mesh topology using the **cotangent Laplacian**:

#### Mean Curvature via Laplace-Beltrami Operator

```
Delta(p_i) = (1 / 2*A_i) * sum_{j in N(i)} (cot(alpha_ij) + cot(beta_ij)) * (p_j - p_i)
```

Where:
- `p_i` is the vertex position
- `N(i)` is the set of neighboring vertices (1-ring)
- `alpha_ij`, `beta_ij` are the two angles opposite edge (i,j) in the adjacent triangles
- `A_i` is the Voronoi area of vertex i

The magnitude gives absolute mean curvature: `|H| = ||Delta(p_i)|| / 2`

The sign is determined by: `sign(H) = sign(dot(vertex_normal, -Delta(p_i)))`

#### Gaussian Curvature via Angle Deficit

```
K_gaussian(v_i) = (2*PI - sum(theta_j)) / A_i
```

Where `theta_j` are the interior angles at vertex `v_i` in each adjacent face.

#### Principal Curvatures from Mean and Gaussian

```
k1 = H + sqrt(H^2 - K_gaussian)
k2 = H - sqrt(H^2 - K_gaussian)
```

- `k1 > 0` and `k2 > 0`: convex (bowl-like, elliptic point)
- `k1 < 0` and `k2 < 0`: concave
- `k1 * k2 < 0`: saddle point (hyperbolic)
- `k1` or `k2 = 0`: parabolic

#### Implementation (using libigl):

```cpp
Eigen::MatrixXd HN;          // Mean curvature normals
Eigen::SparseMatrix<double> L, M, Minv;

igl::cotmatrix(V, F, L);     // Cotangent Laplacian
igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
igl::invert_diag(M, Minv);
HN = -Minv * (L * V);        // Mean curvature normal vectors
H = HN.rowwise().norm();     // Absolute mean curvature per vertex
```

### 5.4 Method 4: Dihedral Angle (Edge-Based)

For each edge shared by two triangles:

```
dihedral_angle = acos(dot(face_normal_1, face_normal_2))
signed_angle = dihedral_angle * sign(dot(cross(face_normal_1, face_normal_2), edge_direction))

if signed_angle > PI: edge is concave
if signed_angle < PI: edge is convex
```

The curvature per vertex is the weighted average of dihedral angles of incident edges:
```
curvature(v) = sum(signed_dihedral_angle(e) * edge_length(e)) / (4 * vertex_area(v))
```

### 5.5 Practical Curvature Map Encoding

Substance Painter's curvature output:
- **Grayscale**: 0.5 = flat, values > 0.5 = convex (edges/ridges), values < 0.5 = concave (cavities)
- **RGB (Curvature from Mesh)**: Red channel = convexity, Green channel = concavity

---

## 6. Anti-Aliasing Strategies for Baked Textures

### 6.1 Supersampling (SSAA)

The most common anti-aliasing method for texture baking. The texture is computed at a higher resolution and then downsampled:

```
For a 2K output with 2x2 supersampling:
    Bake at 4K resolution (4x the texel count)
    Downsample each 2x2 block by averaging:
        output[x,y] = (sample[2x,2y] + sample[2x+1,2y] + 
                       sample[2x,2y+1] + sample[2x+1,2y+1]) / 4
```

Common supersampling levels:
- **1x1**: No anti-aliasing
- **2x2**: 4 samples per output texel (Substance Painter default option)
- **4x4**: 16 samples per output texel (high quality)
- **8x8**: 64 samples per output texel (very high quality, slow)

**Cost**: Baking time and memory increase by the square of the supersampling factor.

### 6.2 Per-Texel Multisampling

Instead of rendering at higher resolution, multiple ray samples are cast per texel with **jittered offsets**:

```
For each texel at position (tx, ty):
    accumulated_value = 0
    For each sample s = 0 to N-1:
        // Jitter the texel position within its area
        jitter_u = (tx + stratified_random_x(s)) / texture_width
        jitter_v = (ty + stratified_random_y(s)) / texture_height
        
        // Recompute barycentric coords with jittered UV position
        // Interpolate world position and normal at jittered location
        // Cast ray and accumulate result
        accumulated_value += bake_sample(jitter_u, jitter_v)
    
    output[tx, ty] = accumulated_value / N
```

**Jitter Patterns**:
- **Random**: Simple but noisy at low sample counts
- **Stratified (NxN grid with jitter)**: Subdivide texel area into NxN sub-cells, sample one random point per cell
- **Rotated Grid**: 2x2 samples rotated 26.6 degrees -- good balance of quality and cost
- **Halton/Hammersley**: Low-discrepancy sequences for best coverage

### 6.3 Conservative Rasterization for Edge Anti-Aliasing

Standard rasterization tests whether the texel center is inside the UV triangle. This misses texels that are partially covered, leaving gaps. **Conservative rasterization** rasterizes any texel that has *any* overlap with the triangle:

**GPU Approach (pseudo-conservative)**:
```
Render the UV-space triangles multiple times with sub-pixel offsets:
    Pass 1: offset (+2, +2) texels
    Pass 2: offset (+2, -2) texels
    Pass 3: offset (-2, +2) texels
    Pass 4: offset (-2, -2) texels
    ... (medium and small offsets)
    Final pass: offset (0, 0) -- centered, rendered last to overwrite
```

**Software Approach**:
```
For each triangle edge:
    Expand the edge outward by 0.5 texels
    Rasterize the expanded triangle
```

### 6.4 Randomized Texel Origin for Hemisphere Sampling

When baking signals that require hemisphere integration (AO, bent normals), the texel center may not represent the full texel area well. Instead:

```
For each sample:
    Randomize ray origin within the texel's world-space footprint
    (not just the center)
```

This accounts for partial texel coverage and reduces aliasing at geometry boundaries.

---

## 7. Padding/Dilation of UV Islands

### 7.1 Why Padding Is Necessary

When GPU hardware samples a texture using bilinear filtering, it reads a 2x2 neighborhood of texels. If a texel at the edge of a UV island has an empty (black) neighbor outside the island, the black value bleeds into the filtered result, creating visible seams on the model.

**Mipmap amplification**: At lower mip levels (downsampled versions), the empty regions between UV islands increasingly contaminate valid texels. Without adequate padding, seams become progressively worse at lower LODs.

### 7.2 How Much Padding Is Needed

Rule of thumb:
- For a texture of resolution R with M mip levels: need at least `2^(M-1)` texels of padding
- Practical minimums:
  - **256x256**: 2-4 pixels padding
  - **1024x1024**: 4-8 pixels
  - **2048x2048**: 8-16 pixels
  - **4096x4096**: 16-32 pixels

### 7.3 Nearest-Neighbor Dilation Algorithm

The simplest and most common padding algorithm:

```
For each iteration (repeat P times for P pixels of padding):
    For each empty texel (x, y) in the texture:
        Check 8 neighbors (or 4 for faster computation)
        If any neighbor has a valid (non-empty) value:
            Copy the closest valid neighbor's color to this texel
            Mark this texel as "padding" (valid but not original)
```

This produces a "skirt" around each UV island that extends the edge colors outward.

**Optimization**: Instead of iterating over all texels each pass, maintain a frontier queue of empty texels adjacent to filled texels (BFS flood-fill approach).

### 7.4 Jump Flooding Algorithm (JFA) -- GPU-Optimized

For GPU-based dilation, the **Jump Flooding Algorithm** is highly efficient:

```
Initialize: Each valid texel stores its own coordinate. Empty texels store "no data."

For step_size = texture_size/2; step_size >= 1; step_size /= 2:
    For each texel (x, y):
        Check 9 neighbors at offset (dx, dy) * step_size
            where dx, dy in {-1, 0, +1}
        Among those neighbors and self, find the one storing the 
            coordinate of the closest valid texel (by Euclidean distance)
        Store that coordinate
        
Final pass: Each texel reads the color from its stored nearest-valid-texel coordinate
```

JFA runs in O(log N) passes and is embarrassingly parallel, making it ideal for compute shaders. It produces a **Voronoi diagram** of the nearest UV island edge, ensuring seamless dilation.

### 7.5 Weighted Mipmap Downsampling

Even with padding, naive mipmap generation can introduce artifacts. The solution is **weighted downsampling**:

```
For each 2x2 block during mip generation:
    total_weight = 0
    total_color = (0, 0, 0)
    For each of the 4 texels:
        weight = (texel is valid or padding) ? 1.0 : 0.0
        total_color += texel_color * weight
        total_weight += weight
    
    if total_weight > 0:
        mip_texel = total_color / total_weight
    else:
        mip_texel = (0, 0, 0)  // will be filled by dilation at this mip level
```

This prevents empty regions from averaging down into valid texels during mipmap generation.

### 7.6 Substance Painter's Dilation

Substance Painter generates padding after the main bake pass. The documentation states: "Padding (sometimes also called dilation) is a process that happens after the generation of a texture, with the purpose of dilating the borders of the UV islands to fill empty areas with similar pixels."

---

## 8. How Multiple Samples Per Texel Are Averaged

### 8.1 Simple Box Filter (Equal Weighting)

The most common approach for supersampled bakes:

```
output[x, y] = (1/N) * sum_{i=0}^{N-1} sample[i]
```

Where N is the number of sub-samples per texel. This treats all samples equally and is equivalent to a box filter in texture space.

### 8.2 Weighted Averaging for Normal Maps

Normal maps require special treatment when averaging because normals are unit vectors:

```
// WRONG: averaging encoded values
avg = (sample0 + sample1 + sample2 + sample3) / 4

// CORRECT: average in vector space, then renormalize
N_avg = decode(sample0) + decode(sample1) + decode(sample2) + decode(sample3)
N_avg = normalize(N_avg)
output = encode(N_avg)
```

Where `decode(c) = c * 2 - 1` and `encode(n) = n * 0.5 + 0.5`.

For supersampled normal maps, the averaging should happen in decoded vector space, not in the [0,1] encoded color space, to maintain correct unit-length normals.

### 8.3 Handling Mixed Valid/Invalid Samples

When some sub-samples fall outside any UV triangle (ray misses), only valid samples should contribute:

```
total = (0, 0, 0)
valid_count = 0
For each sub-sample:
    if sample.valid:
        total += sample.value
        valid_count++

if valid_count > 0:
    output = total / valid_count
else:
    output = default_value  // filled later by dilation
```

### 8.4 AO Sample Accumulation

For AO baking where each texel already uses hundreds of hemisphere samples, the supersampling multiplier compounds:

```
Total rays per output texel = supersample_count * ao_sample_count

Example: 4x4 supersampling with 256 AO samples = 4096 rays per output texel
```

To reduce computational cost, some bakers use lower AO sample counts with higher supersampling, or vice versa, trading between spatial aliasing and hemispheric noise.

### 8.5 Progressive/Incremental Averaging

For interactive bakers that display results progressively:

```
running_average[texel] = running_average[texel] * (n / (n+1)) + new_sample * (1 / (n+1))
```

Where n is the current sample count. This maintains a running average without storing all individual samples, enabling real-time preview of bake quality.

---

## References and Key Sources

### Implementation Resources
- [NVIDIA GPU Gems 3, Chapter 22: Baking Normal Maps on the GPU](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-22-baking-normal-maps-gpu)
- [NVIDIA GPU Gems, Chapter 17: Ambient Occlusion](https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-17-ambient-occlusion)
- [PBRT Book: Bounding Volume Hierarchies](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)
- [MikkTSpace Reference Implementation](https://github.com/mmikk/MikkTSpace)
- [Fornos: Open-Source GPU Texture Baking Tool](https://github.com/caosdoar/Fornos)
- [GPU-Zen-2-Baker: OpenGL Baking Example](https://github.com/alaingalvan/GPU-Zen-2-Baker)
- [aobaker: AO Baking Tool using Embree](https://github.com/prideout/aobaker)
- [Intel Embree Ray Tracing Kernels](https://github.com/RenderKit/embree)
- [madmann91/bvh: Modern C++ BVH Library](https://github.com/madmann91/bvh)

### Technical Articles
- [How to Build a BVH (Jacco Bikker)](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
- [Better Sampling (Rory Driscoll)](https://www.rorydriscoll.com/2009/01/07/better-sampling/)
- [Hammersley Points on the Hemisphere (Holger Dammertz)](https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html)
- [Baking Signals into Textures (Molecular Musings)](https://blog.molecular-matters.com/2011/12/30/baking-signals-into-textures/)
- [Baking Artifact-Free Lightmaps on the GPU](https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/)
- [Curvature of a Triangle Mesh (Rodolphe Vaillant)](https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation)
- [Cesium: Baking Ambient Occlusion in the glTF Pipeline](https://cesium.com/blog/2016/08/08/ambient-occlusion/)

### Academic Papers
- Moller, Trumbore: "Fast, Minimum Storage Ray/Triangle Intersection" (1997) -- [Wikipedia](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)
- Mikkelsen: Master's thesis on tangent space computation -- [mikktspace.com](http://www.mikktspace.com/)
- Wald: "On Fast Construction of SAH-based Bounding Volume Hierarchies" -- [PDF](https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf)
- Rusinkiewicz: "Estimating Curvatures and Their Derivatives on Triangle Meshes" -- [PDF](https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf)
- Meyer, Desbrun, Schroder, Barr: "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" -- [PDF](https://www.multires.caltech.edu/pubs/diffGeoOps.pdf)
- Sloan: "Ambient Obscurance Baking on the GPU" -- [ResearchGate](https://www.researchgate.net/publication/262242315_Ambient_Obscurance_baking_on_the_GPU)

### Documentation
- [Substance 3D Bakers: Common Parameters](https://helpx.adobe.com/substance-3d-bake/bakers-settings/common-parameters.html)
- [Substance 3D Bakers: Tangent Space](https://helpx.adobe.com/substance-3d-bake/features/tangent-space.html)
- [Substance 3D Bakers: Curvature](https://helpx.adobe.com/substance-3d-bake/bakers-settings/curvature.html)
- [Substance 3D Bakers: Curvature from Mesh](https://helpx.adobe.com/substance-3d-bake/bakers-settings/curvature-from-mesh.html)
- [Polycount Wiki: Texture Baking](http://wiki.polycount.com/wiki/Texture_Baking)
- [Polycount Wiki: Normal Map Technical Details](http://wiki.polycount.com/wiki/Normal_Map_Technical_Details)
- [Polycount Wiki: Edge Padding](http://wiki.polycount.com/wiki/Edge_padding)
- [Blender Manual: Render Baking](https://docs.blender.org/manual/en/latest/render/cycles/baking.html)
