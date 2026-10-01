//! Topology of a single refinement level (port of `opensubdiv/vtr/level.h`).

use std::collections::HashMap;

use crate::sdc::{Crease, SHARPNESS_INFINITE};
use crate::{Index, INDEX_INVALID};

/// Errors reported when constructing or validating topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// A face has fewer than three vertices.
    DegenerateFace {
        /// Index of the offending face.
        face: usize,
        /// Number of vertices the face was given.
        size: usize,
    },
    /// A face refers to a vertex index out of range.
    VertexIndexOutOfRange {
        /// Index of the offending face.
        face: usize,
        /// The out-of-range vertex index.
        vertex: Index,
    },
    /// A face uses the same vertex more than once.
    RepeatedVertexInFace {
        /// Index of the offending face.
        face: usize,
        /// The vertex index that appears more than once.
        vertex: Index,
    },
    /// The flattened face-vertex array does not match the per-face counts.
    FaceVertexCountMismatch {
        /// Total face-vertex count implied by the per-face vertex counts.
        expected: usize,
        /// Length of the flattened face-vertex array actually given.
        actual: usize,
    },
    /// The Loop scheme requires a purely triangular mesh.
    NonTriangularFaceForLoop {
        /// Index of the offending face.
        face: usize,
        /// Number of vertices of that face.
        size: usize,
    },
    /// A crease/corner/hole descriptor entry refers to an invalid component.
    InvalidDescriptorIndex {
        /// Name of the descriptor field holding the invalid index.
        what: &'static str,
        /// The invalid component index.
        index: Index,
    },
    /// A crease was specified between two vertices not connected by an edge.
    CreaseEdgeNotFound {
        /// The vertex pair for which no edge exists.
        vertices: [Index; 2],
    },
    /// A face-varying channel's value array does not have one entry per
    /// face-vertex.
    FVarValueCountMismatch {
        /// Index of the offending face-varying channel.
        channel: usize,
        /// Number of face-vertices in the mesh.
        expected: usize,
        /// Length of the channel's index array.
        actual: usize,
    },
    /// A face-varying channel refers to a value index out of range.
    FVarValueIndexOutOfRange {
        /// Index of the offending face-varying channel.
        channel: usize,
        /// The out-of-range face-varying value index.
        index: Index,
    },
    /// Building patches for a mesh with non-quad faces requires at least one
    /// level of refinement.
    PatchesRequireRefinement,
    /// A limit stencil was requested at a location lying in a hole, where
    /// no patch covers the limit surface.
    LimitLocationInHole {
        /// The ptex face of the offending location.
        ptex_face: Index,
        /// Position of the location across all requested location arrays,
        /// i.e. the index its stencil would have had.
        location: usize,
    },
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologyError::DegenerateFace { face, size } => {
                write!(f, "face {face} is degenerate ({size} vertices)")
            }
            TopologyError::VertexIndexOutOfRange { face, vertex } => {
                write!(f, "face {face} refers to out-of-range vertex {vertex}")
            }
            TopologyError::RepeatedVertexInFace { face, vertex } => {
                write!(f, "face {face} uses vertex {vertex} more than once")
            }
            TopologyError::FaceVertexCountMismatch { expected, actual } => {
                write!(
                    f,
                    "face-vertex array length {actual} does not match per-face counts (expected {expected})"
                )
            }
            TopologyError::NonTriangularFaceForLoop { face, size } => {
                write!(
                    f,
                    "Loop scheme requires triangles, but face {face} has {size} vertices"
                )
            }
            TopologyError::InvalidDescriptorIndex { what, index } => {
                write!(f, "descriptor {what} index {index} is out of range")
            }
            TopologyError::CreaseEdgeNotFound { vertices } => {
                write!(
                    f,
                    "no edge between vertices {} and {} for crease",
                    vertices[0], vertices[1]
                )
            }
            TopologyError::FVarValueCountMismatch {
                channel,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "face-varying channel {channel} has {actual} value indices (expected {expected}, one per face-vertex)"
                )
            }
            TopologyError::FVarValueIndexOutOfRange { channel, index } => {
                write!(
                    f,
                    "face-varying channel {channel} refers to out-of-range value index {index}"
                )
            }
            TopologyError::PatchesRequireRefinement => {
                write!(
                    f,
                    "patches for meshes with non-quad faces require at least one refinement level"
                )
            }
            TopologyError::LimitLocationInHole {
                ptex_face,
                location,
            } => {
                write!(
                    f,
                    "limit location {location} lies in a hole of ptex face {ptex_face}"
                )
            }
        }
    }
}

impl std::error::Error for TopologyError {}

/// A complete specification of the topology of one level of refinement
/// (`Vtr::internal::Level`).
///
/// All neighborhood *relations* between the three component types (vertices,
/// edges, faces) are stored as flat arrays with companion offset arrays
/// ("CSR" layout), exactly as in OpenSubdiv:
///
/// * face-verts:  vertices incident each face, in winding order
/// * face-edges:  edges incident each face (edge `i` joins corners `i` and `i+1`)
/// * edge-verts:  the two vertices at the ends of each edge
/// * edge-faces:  faces incident each edge
/// * vert-faces:  faces incident each vertex
/// * vert-edges:  edges incident each vertex
///
/// Per-component *sharpness* is stored for edges and vertices; boundary
/// status is derived from the relations.
#[derive(Debug, Clone, Default)]
pub struct Level {
    num_vertices: usize,

    face_vert_offsets: Vec<u32>,
    face_verts: Vec<Index>,
    face_edges: Vec<Index>,

    edge_verts: Vec<[Index; 2]>,
    edge_face_offsets: Vec<u32>,
    edge_faces: Vec<Index>,

    vert_face_offsets: Vec<u32>,
    vert_faces: Vec<Index>,
    vert_edge_offsets: Vec<u32>,
    vert_edges: Vec<Index>,

    edge_sharpness: Vec<f32>,
    vert_sharpness: Vec<f32>,

    face_holes: Vec<bool>,
}

impl Level {
    /// Build a level from raw face-vertex topology (the core of
    /// `Vtr::internal::Level`'s construction from a descriptor): every other
    /// relation (edges and all adjacencies) is derived from it.
    pub fn from_face_vertices(
        num_vertices: usize,
        verts_per_face: &[usize],
        face_verts: &[Index],
    ) -> Result<Level, TopologyError> {
        let expected: usize = verts_per_face.iter().sum();
        if expected != face_verts.len() {
            return Err(TopologyError::FaceVertexCountMismatch {
                expected,
                actual: face_verts.len(),
            });
        }

        let num_faces = verts_per_face.len();
        let mut level = Level {
            num_vertices,
            face_vert_offsets: Vec::with_capacity(num_faces + 1),
            face_verts: face_verts.to_vec(),
            face_edges: vec![INDEX_INVALID; face_verts.len()],
            ..Default::default()
        };

        // Face-vert offsets, with validation.
        let mut offset = 0u32;
        level.face_vert_offsets.push(0);
        for (f, &size) in verts_per_face.iter().enumerate() {
            if size < 3 {
                return Err(TopologyError::DegenerateFace { face: f, size });
            }
            let corners = &face_verts[offset as usize..offset as usize + size];
            for (i, &v) in corners.iter().enumerate() {
                if v as usize >= num_vertices {
                    return Err(TopologyError::VertexIndexOutOfRange { face: f, vertex: v });
                }
                if corners[..i].contains(&v) {
                    return Err(TopologyError::RepeatedVertexInFace { face: f, vertex: v });
                }
            }
            offset += size as u32;
            level.face_vert_offsets.push(offset);
        }

        // Derive edges: one per unique (unordered) vertex pair traversed
        // along face boundaries, numbered in order of first encounter.
        let mut edge_map: HashMap<(Index, Index), Index> = HashMap::new();
        for f in 0..num_faces {
            let (start, end) = level.face_vert_range(f);
            let n = end - start;
            for i in 0..n {
                let a = level.face_verts[start + i];
                let b = level.face_verts[start + (i + 1) % n];
                let key = (a.min(b), a.max(b));
                let edge = *edge_map.entry(key).or_insert_with(|| {
                    let e = level.edge_verts.len() as Index;
                    level.edge_verts.push([a, b]);
                    e
                });
                level.face_edges[start + i] = edge;
            }
        }

        let num_edges = level.edge_verts.len();

        // Edge-face relation (counting sort into CSR).
        let mut edge_face_counts = vec![0u32; num_edges];
        for &e in &level.face_edges {
            edge_face_counts[e as usize] += 1;
        }
        level.edge_face_offsets = prefix_sum(&edge_face_counts);
        level.edge_faces = vec![INDEX_INVALID; level.face_edges.len()];
        let mut cursor: Vec<u32> = level.edge_face_offsets[..num_edges].to_vec();
        for f in 0..num_faces {
            let (start, end) = level.face_vert_range(f);
            for i in start..end {
                let e = level.face_edges[i] as usize;
                level.edge_faces[cursor[e] as usize] = f as Index;
                cursor[e] += 1;
            }
        }

        // Vert-face relation.
        let mut vert_face_counts = vec![0u32; num_vertices];
        for &v in &level.face_verts {
            vert_face_counts[v as usize] += 1;
        }
        level.vert_face_offsets = prefix_sum(&vert_face_counts);
        level.vert_faces = vec![INDEX_INVALID; level.face_verts.len()];
        let mut cursor: Vec<u32> = level.vert_face_offsets[..num_vertices].to_vec();
        for f in 0..num_faces {
            let (start, end) = level.face_vert_range(f);
            for i in start..end {
                let v = level.face_verts[i] as usize;
                level.vert_faces[cursor[v] as usize] = f as Index;
                cursor[v] += 1;
            }
        }

        // Vert-edge relation.
        let mut vert_edge_counts = vec![0u32; num_vertices];
        for ev in &level.edge_verts {
            vert_edge_counts[ev[0] as usize] += 1;
            vert_edge_counts[ev[1] as usize] += 1;
        }
        level.vert_edge_offsets = prefix_sum(&vert_edge_counts);
        level.vert_edges = vec![INDEX_INVALID; 2 * num_edges];
        let mut cursor: Vec<u32> = level.vert_edge_offsets[..num_vertices].to_vec();
        for (e, ev) in level.edge_verts.iter().enumerate() {
            for &v in ev {
                level.vert_edges[cursor[v as usize] as usize] = e as Index;
                cursor[v as usize] += 1;
            }
        }

        level.edge_sharpness = vec![0.0; num_edges];
        level.vert_sharpness = vec![0.0; num_vertices];
        level.face_holes = vec![false; num_faces];

        Ok(level)
    }

    // ------------------------------------------------------------------
    //  Sizes
    // ------------------------------------------------------------------

    /// Number of vertices in this level (`GetNumVertices`).
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Number of edges in this level (`GetNumEdges`).
    pub fn num_edges(&self) -> usize {
        self.edge_verts.len()
    }

    /// Number of faces in this level (`GetNumFaces`).
    pub fn num_faces(&self) -> usize {
        self.face_vert_offsets.len().saturating_sub(1)
    }

    /// Total number of face-vertex pairs (`GetNumFaceVerticesTotal`).
    pub fn num_face_vertices_total(&self) -> usize {
        self.face_verts.len()
    }

    // ------------------------------------------------------------------
    //  Relations
    // ------------------------------------------------------------------

    fn face_vert_range(&self, face: usize) -> (usize, usize) {
        (
            self.face_vert_offsets[face] as usize,
            self.face_vert_offsets[face + 1] as usize,
        )
    }

    /// The offset of the first corner of `face` in the level's
    /// face-vertex array (the corners of face `f` occupy
    /// `face_vertices_offset(f)..face_vertices_offset(f + 1)`).
    pub fn face_vertices_offset(&self, face: usize) -> usize {
        self.face_vert_offsets[face] as usize
    }

    /// The vertices of `face`, in winding order.
    pub fn face_vertices(&self, face: usize) -> &[Index] {
        let (s, e) = self.face_vert_range(face);
        &self.face_verts[s..e]
    }

    /// The edges of `face`: edge `i` connects corners `i` and `(i+1) % n`.
    pub fn face_edges(&self, face: usize) -> &[Index] {
        let (s, e) = self.face_vert_range(face);
        &self.face_edges[s..e]
    }

    /// The two end vertices of `edge`, in the winding order of the first
    /// face that introduced the edge.
    pub fn edge_vertices(&self, edge: usize) -> [Index; 2] {
        self.edge_verts[edge]
    }

    /// The faces incident `edge`.
    pub fn edge_faces(&self, edge: usize) -> &[Index] {
        let s = self.edge_face_offsets[edge] as usize;
        let e = self.edge_face_offsets[edge + 1] as usize;
        &self.edge_faces[s..e]
    }

    /// The faces incident `vertex`.
    pub fn vertex_faces(&self, vertex: usize) -> &[Index] {
        let s = self.vert_face_offsets[vertex] as usize;
        let e = self.vert_face_offsets[vertex + 1] as usize;
        &self.vert_faces[s..e]
    }

    /// The edges incident `vertex`.
    pub fn vertex_edges(&self, vertex: usize) -> &[Index] {
        let s = self.vert_edge_offsets[vertex] as usize;
        let e = self.vert_edge_offsets[vertex + 1] as usize;
        &self.vert_edges[s..e]
    }

    /// Find the edge connecting `v0` and `v1` (`Level::findEdge`).
    pub fn find_edge(&self, v0: Index, v1: Index) -> Option<Index> {
        self.vertex_edges(v0 as usize).iter().copied().find(|&e| {
            let ev = self.edge_verts[e as usize];
            (ev[0] == v0 && ev[1] == v1) || (ev[0] == v1 && ev[1] == v0)
        })
    }

    /// The vertex at the end of `edge` opposite `vertex`.
    pub fn edge_opposite_vertex(&self, edge: usize, vertex: Index) -> Index {
        let ev = self.edge_verts[edge];
        if ev[0] == vertex {
            ev[1]
        } else {
            debug_assert_eq!(ev[1], vertex);
            ev[0]
        }
    }

    // ------------------------------------------------------------------
    //  Boundary queries
    // ------------------------------------------------------------------

    /// Is `edge` on a boundary (incident exactly one face)?
    pub fn is_edge_boundary(&self, edge: usize) -> bool {
        self.edge_faces(edge).len() == 1
    }

    /// Is `edge` non-manifold (incident more than two faces)?
    pub fn is_edge_non_manifold(&self, edge: usize) -> bool {
        self.edge_faces(edge).len() > 2
    }

    /// Is `vertex` non-manifold: incident a non-manifold edge, or whose
    /// incident faces do not form a single fan — a closed ring (as many
    /// edges as faces, none on a boundary) or one open fan bounded by two
    /// boundary edges? Vertices shared by otherwise separate fans ("bow
    /// ties") and dangling edges are non-manifold.
    pub fn is_vertex_non_manifold(&self, vertex: usize) -> bool {
        let edges = self.vertex_edges(vertex);
        let faces = self.vertex_faces(vertex);
        let mut boundary_edges = 0;
        for &e in edges {
            match self.edge_faces(e as usize).len() {
                0 => return true,
                1 => boundary_edges += 1,
                2 => {}
                _ => return true,
            }
        }
        if faces.is_empty() {
            return false; // an isolated vertex
        }
        let counts_ok = (boundary_edges == 0 && edges.len() == faces.len())
            || (boundary_edges == 2 && edges.len() == faces.len() + 1);
        !counts_ok || self.vertex_fans(vertex).1 > 1
    }

    /// The edges of `face` incident `vertex` (two per occurrence of the
    /// vertex in the face).
    fn face_edges_at_vertex(&self, face: usize, vertex: Index) -> impl Iterator<Item = Index> + '_ {
        let fv = self.face_vertices(face);
        let fe = self.face_edges(face);
        let n = fv.len();
        (0..n)
            .filter(move |&i| fv[i] == vertex)
            .flat_map(move |i| [fe[i], fe[(i + n - 1) % n]])
    }

    /// Partition the faces incident `vertex` into *fans*: sets of faces
    /// connected across manifold (two-face) edges incident the vertex.
    /// Returns the fan of each entry of [`Self::vertex_faces`] and the
    /// number of fans. Linear in the size of the vertex's neighborhood.
    fn vertex_fans(&self, vertex: usize) -> (Vec<usize>, usize) {
        let faces = self.vertex_faces(vertex);
        let slot: HashMap<Index, usize> = faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let mut fan = vec![usize::MAX; faces.len()];
        let mut num_fans = 0;
        let mut stack = Vec::new();
        for seed in 0..faces.len() {
            if fan[seed] != usize::MAX {
                continue;
            }
            fan[seed] = num_fans;
            stack.push(seed);
            while let Some(i) = stack.pop() {
                for e in self.face_edges_at_vertex(faces[i] as usize, vertex as Index) {
                    let edge_faces = self.edge_faces(e as usize);
                    if edge_faces.len() != 2 {
                        continue;
                    }
                    for &other in edge_faces {
                        if let Some(&j) = slot.get(&other) {
                            if fan[j] == usize::MAX {
                                fan[j] = num_fans;
                                stack.push(j);
                            }
                        }
                    }
                }
            }
            num_fans += 1;
        }
        (fan, num_fans)
    }

    /// Does non-manifold `vertex` lie on a *non-manifold crease*: exactly
    /// two non-manifold edges, no boundary edges, and every fan of faces
    /// around it bounded by those two edges? Only then do the crease rules
    /// along the two edges describe the limit of every fan; a vertex also
    /// shared by a fan away from the crease must be pinned instead.
    fn is_vertex_non_manifold_crease(&self, vertex: usize) -> bool {
        let mut crease = Vec::with_capacity(2);
        for &e in self.vertex_edges(vertex) {
            match self.edge_faces(e as usize).len() {
                2 => {}
                n if n > 2 => crease.push(e),
                _ => return false,
            }
        }
        if crease.len() != 2 {
            return false;
        }
        let (fan, num_fans) = self.vertex_fans(vertex);
        let mut touches = vec![[false; 2]; num_fans];
        for (i, &f) in self.vertex_faces(vertex).iter().enumerate() {
            for e in self.face_edges_at_vertex(f as usize, vertex as Index) {
                if let Some(k) = crease.iter().position(|&c| c == e) {
                    touches[fan[i]][k] = true;
                }
            }
        }
        touches.iter().all(|&[a, b]| a && b)
    }

    /// Is `vertex` on a boundary (incident a boundary edge)?
    pub fn is_vertex_boundary(&self, vertex: usize) -> bool {
        self.vertex_edges(vertex)
            .iter()
            .any(|&e| self.is_edge_boundary(e as usize))
    }

    /// Is `vertex` a boundary corner (incident exactly one face)?
    pub fn is_vertex_corner(&self, vertex: usize) -> bool {
        self.vertex_faces(vertex).len() == 1 && self.is_vertex_boundary(vertex)
    }

    // ------------------------------------------------------------------
    //  Sharpness and holes
    // ------------------------------------------------------------------

    /// Sharpness assigned to `edge` ([`SHARPNESS_SMOOTH`](crate::sdc::SHARPNESS_SMOOTH)
    /// = smooth, [`SHARPNESS_INFINITE`](crate::sdc::SHARPNESS_INFINITE) = fully
    /// sharp).
    pub fn edge_sharpness(&self, edge: usize) -> f32 {
        self.edge_sharpness[edge]
    }

    /// Sharpness assigned to `vertex` (a corner weight).
    pub fn vertex_sharpness(&self, vertex: usize) -> f32 {
        self.vert_sharpness[vertex]
    }

    /// Assign `edge`'s sharpness, clamped to the valid range by
    /// [`Crease::clamp`].
    pub fn set_edge_sharpness(&mut self, edge: usize, sharpness: f32) {
        self.edge_sharpness[edge] = Crease::clamp(sharpness);
    }

    /// Assign `vertex`'s corner sharpness, clamped to the valid range by
    /// [`Crease::clamp`].
    pub fn set_vertex_sharpness(&mut self, vertex: usize, sharpness: f32) {
        self.vert_sharpness[vertex] = Crease::clamp(sharpness);
    }

    /// Is `face` tagged as a hole? Holes are subdivided but not rendered.
    pub fn is_face_hole(&self, face: usize) -> bool {
        self.face_holes[face]
    }

    /// Tag (or untag) `face` as a hole.
    pub fn set_face_hole(&mut self, face: usize, hole: bool) {
        self.face_holes[face] = hole;
    }

    /// Apply the boundary-interpolation rules by sharpening boundary edges
    /// (and, for `EdgeAndCorner`, pinning boundary corner vertices), and
    /// sharpen non-manifold features. Called once on the base level after
    /// construction, mirroring how Far applies
    /// `Sdc::Options::VtxBoundaryInterpolation`
    /// (`applyComponentTagsAndBoundarySharpness`).
    ///
    /// As in OpenSubdiv, non-manifold edges are made infinitely sharp —
    /// the surface is split along them like along a boundary — and every
    /// non-manifold vertex is made infinitely sharp too, except one lying
    /// on a non-manifold crease (exactly two non-manifold edges bounding
    /// every fan of faces around it), which follows the crease rule along
    /// them. Refinement propagates both to
    /// the children, so every level stays consistent.
    pub fn sharpen_boundaries(&mut self, crease: &Crease) {
        for e in 0..self.num_edges() {
            if self.is_edge_boundary(e) {
                self.edge_sharpness[e] = crease.sharpen_boundary_edge(self.edge_sharpness[e]);
            } else if self.is_edge_non_manifold(e) {
                self.edge_sharpness[e] = SHARPNESS_INFINITE;
            }
        }
        for v in 0..self.num_vertices() {
            if self.is_vertex_corner(v) {
                self.vert_sharpness[v] =
                    crease.sharpen_boundary_corner_vertex(self.vert_sharpness[v]);
            } else if self.is_vertex_non_manifold(v) && !self.is_vertex_non_manifold_crease(v) {
                self.vert_sharpness[v] = SHARPNESS_INFINITE;
            }
        }
    }

    /// The number of sharp edges incident `vertex`, and the sharpness values
    /// of all its incident edges gathered into `sharpness_out`.
    pub fn gather_vertex_edge_sharpness<'a>(
        &self,
        vertex: usize,
        sharpness_out: &'a mut Vec<f32>,
    ) -> &'a [f32] {
        sharpness_out.clear();
        for &e in self.vertex_edges(vertex) {
            sharpness_out.push(self.edge_sharpness[e as usize]);
        }
        sharpness_out
    }
}

fn prefix_sum(counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut sum = 0u32;
    offsets.push(0);
    for &c in counts {
        sum += c;
        offsets.push(sum);
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad and a single triangle sharing an edge:
    ///
    /// ```text
    ///   3 ---- 2
    ///   |      | \
    ///   |      |  4
    ///   |      | /
    ///   0 ---- 1
    /// ```
    fn quad_and_tri() -> Level {
        Level::from_face_vertices(5, &[4, 3], &[0, 1, 2, 3, 1, 4, 2]).unwrap()
    }

    #[test]
    fn non_manifold_features_are_sharpened() {
        // Three quads sharing edge 0-1 (a "T" fin), and a fourth quad
        // touching the fin only at vertex 3 (a bow tie).
        let mut level = Level::from_face_vertices(
            11,
            &[4, 4, 4, 4],
            &[0, 1, 2, 3, 1, 0, 4, 5, 0, 1, 6, 7, 3, 8, 9, 10],
        )
        .unwrap();
        let seam = level.find_edge(0, 1).unwrap() as usize;
        assert!(level.is_edge_non_manifold(seam));
        for v in [0, 1, 3] {
            assert!(level.is_vertex_non_manifold(v), "vertex {v}");
        }
        for v in [2, 4, 5, 6, 7, 8] {
            assert!(!level.is_vertex_non_manifold(v), "vertex {v}");
        }
        // The quad and triangle of the helper mesh are manifold throughout.
        let manifold = quad_and_tri();
        assert!((0..5).all(|v| !manifold.is_vertex_non_manifold(v)));

        level.sharpen_boundaries(&Crease::new(crate::sdc::Options::default()));
        assert!(Crease::is_infinite(level.edge_sharpness(seam)));
        for v in [0, 1, 3] {
            assert!(Crease::is_infinite(level.vertex_sharpness(v)), "vertex {v}");
        }
        assert_eq!(level.vertex_sharpness(2), 0.0);
    }

    #[test]
    fn non_manifold_crease_vertex_is_not_pinned() {
        // Three 2x1 fins sharing a seam 0-1-2: the middle seam vertex lies
        // on a crease of exactly two non-manifold edges and keeps its
        // sharpness; the seam ends are pinned.
        let mut face_verts = Vec::new();
        for k in 0..3 {
            let b = 3 + 3 * k;
            face_verts.extend_from_slice(&[0, 1, b + 1, b, 1, 2, b + 2, b + 1]);
        }
        let mut level = Level::from_face_vertices(12, &[4; 6], &face_verts).unwrap();
        assert!(level.is_vertex_non_manifold(1));
        level.sharpen_boundaries(&Crease::new(crate::sdc::Options::default()));
        assert_eq!(level.vertex_sharpness(1), 0.0);
        assert!(Crease::is_infinite(level.vertex_sharpness(0)));
        assert!(Crease::is_infinite(level.vertex_sharpness(2)));
    }

    #[test]
    fn non_manifold_crease_vertex_with_a_separate_fan_is_pinned() {
        // The three-fin seam, plus a closed tetrahedron touching only the
        // middle seam vertex: that vertex's two non-manifold edges do not
        // bound the tetrahedron's fan, so it is pinned.
        let mut face_verts = Vec::new();
        let mut verts_per_face = vec![4usize; 6];
        for k in 0..3 {
            let b = 3 + 3 * k;
            face_verts.extend_from_slice(&[0, 1, b + 1, b, 1, 2, b + 2, b + 1]);
        }
        face_verts.extend_from_slice(&[1, 12, 13, 1, 13, 14, 1, 14, 12, 12, 14, 13]);
        verts_per_face.extend_from_slice(&[3; 4]);
        let mut level = Level::from_face_vertices(15, &verts_per_face, &face_verts).unwrap();
        assert!(level.is_vertex_non_manifold(1));
        assert!(!level.is_vertex_non_manifold(12));
        level.sharpen_boundaries(&Crease::new(crate::sdc::Options::default()));
        assert!(Crease::is_infinite(level.vertex_sharpness(1)));
        assert_eq!(level.vertex_sharpness(12), 0.0);
    }

    #[test]
    fn high_valence_vertices_are_classified() {
        // A closed fan of 4000 triangles around vertex 0 (a flattened
        // double cone), and the same fan split into two halves sharing only
        // the pole: fan detection stays linear in the valence.
        let n = 4000u32;
        let mut face_verts = Vec::new();
        for i in 0..n {
            face_verts.extend_from_slice(&[0, 1 + i, 1 + (i + 1) % n]);
        }
        let level =
            Level::from_face_vertices(1 + n as usize, &vec![3; n as usize], &face_verts).unwrap();
        assert!(!level.is_vertex_non_manifold(0));

        // Two open fans sharing the pole: a bow tie.
        let mut face_verts = Vec::new();
        for half in 0..2 {
            let b = 1 + half * (n + 1);
            for i in 0..n {
                face_verts.extend_from_slice(&[0, b + i, b + i + 1]);
            }
        }
        let level = Level::from_face_vertices(
            1 + 2 * (n as usize + 1),
            &vec![3; 2 * n as usize],
            &face_verts,
        )
        .unwrap();
        assert!(level.is_vertex_non_manifold(0));
    }

    #[test]
    fn build_relations() {
        let level = quad_and_tri();
        assert_eq!(level.num_vertices(), 5);
        assert_eq!(level.num_faces(), 2);
        assert_eq!(level.num_edges(), 6);

        assert_eq!(level.face_vertices(0), &[0, 1, 2, 3]);
        assert_eq!(level.face_vertices(1), &[1, 4, 2]);

        let shared = level.find_edge(1, 2).unwrap();
        assert_eq!(level.edge_faces(shared as usize), &[0, 1]);
        assert!(!level.is_edge_boundary(shared as usize));

        let boundary = level.find_edge(0, 1).unwrap();
        assert!(level.is_edge_boundary(boundary as usize));

        assert_eq!(level.vertex_faces(1), &[0, 1]);
        assert_eq!(level.vertex_edges(1).len(), 3);
        assert!(level.is_vertex_boundary(0));
        assert!(level.is_vertex_corner(0));
        assert!(!level.is_vertex_corner(1));
    }

    #[test]
    fn rejects_bad_topology() {
        assert!(matches!(
            Level::from_face_vertices(3, &[2], &[0, 1]),
            Err(TopologyError::DegenerateFace { .. })
        ));
        assert!(matches!(
            Level::from_face_vertices(2, &[3], &[0, 1, 2]),
            Err(TopologyError::VertexIndexOutOfRange { .. })
        ));
        assert!(matches!(
            Level::from_face_vertices(3, &[3], &[0, 1, 1]),
            Err(TopologyError::RepeatedVertexInFace { .. })
        ));
        assert!(matches!(
            Level::from_face_vertices(3, &[3], &[0, 1]),
            Err(TopologyError::FaceVertexCountMismatch { .. })
        ));
    }
}
