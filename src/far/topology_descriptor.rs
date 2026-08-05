//! Borrowed mesh-topology description (port of
//! `opensubdiv/far/topologyDescriptor.h`).

use crate::Index;

/// A simple reference to raw topology data for use with
/// [`TopologyRefinerFactory`](super::TopologyRefinerFactory) — the port of
/// `Far::TopologyDescriptor`.
///
/// All slices are borrowed from the caller; nothing is copied until the
/// factory builds the refiner. Only `num_vertices`, `num_verts_per_face` and
/// `vert_indices_per_face` are required; creases, corners and holes are
/// optional.
#[derive(Debug, Clone, Copy, Default)]
pub struct TopologyDescriptor<'a> {
    /// Number of vertices in the mesh.
    pub num_vertices: usize,
    /// Number of vertices of each face.
    pub num_verts_per_face: &'a [usize],
    /// Flattened, per-face vertex indices, in winding order.
    pub vert_indices_per_face: &'a [Index],

    /// Pairs of vertex indices identifying creased edges.
    pub crease_vertex_index_pairs: &'a [[Index; 2]],
    /// Sharpness of each creased edge (parallel to
    /// `crease_vertex_index_pairs`).
    pub crease_weights: &'a [f32],

    /// Indices of sharpened (corner) vertices.
    pub corner_vertex_indices: &'a [Index],
    /// Sharpness of each corner vertex (parallel to
    /// `corner_vertex_indices`).
    pub corner_weights: &'a [f32],

    /// Indices of faces tagged as holes.
    pub hole_indices: &'a [Index],
}

impl<'a> TopologyDescriptor<'a> {
    /// Create a descriptor from the required topology: the number of
    /// vertices, the size of each face, and the flattened face-vertex list.
    pub fn new(
        num_vertices: usize,
        num_verts_per_face: &'a [usize],
        vert_indices_per_face: &'a [Index],
    ) -> Self {
        TopologyDescriptor {
            num_vertices,
            num_verts_per_face,
            vert_indices_per_face,
            ..Default::default()
        }
    }

    /// Builder-style setter for creased edges.
    pub fn with_creases(mut self, vertex_pairs: &'a [[Index; 2]], weights: &'a [f32]) -> Self {
        self.crease_vertex_index_pairs = vertex_pairs;
        self.crease_weights = weights;
        self
    }

    /// Builder-style setter for corner vertices.
    pub fn with_corners(mut self, vertices: &'a [Index], weights: &'a [f32]) -> Self {
        self.corner_vertex_indices = vertices;
        self.corner_weights = weights;
        self
    }

    /// Builder-style setter for hole faces.
    pub fn with_holes(mut self, hole_indices: &'a [Index]) -> Self {
        self.hole_indices = hole_indices;
        self
    }
}
