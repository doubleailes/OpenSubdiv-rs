//! Borrowed mesh-topology description (port of
//! `opensubdiv/far/topologyDescriptor.h`).

use crate::Index;

/// Description of one face-varying channel for a [`TopologyDescriptor`]
/// (`Far::TopologyDescriptor::FVarChannel`).
///
/// A face-varying channel assigns one *value index* to every face-vertex
/// (corner) of the mesh, allowing data such as UVs to be discontinuous
/// ("seamed") across edges: two faces sharing an edge may refer to different
/// values at the shared vertices.
#[derive(Debug, Clone, Copy, Default)]
pub struct FVarChannelDescriptor<'a> {
    /// Number of distinct values in the channel.
    pub num_values: usize,
    /// One value index per face-vertex, flattened in the same order as
    /// [`TopologyDescriptor::vert_indices_per_face`].
    pub value_indices: &'a [Index],
}

impl<'a> FVarChannelDescriptor<'a> {
    /// Describe a channel with `num_values` distinct values, indexed once per
    /// face-vertex by `value_indices`.
    pub fn new(num_values: usize, value_indices: &'a [Index]) -> Self {
        Self {
            num_values,
            value_indices,
        }
    }
}

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

    /// Face-varying channels (UVs, per-corner colors, …).
    pub fvar_channels: &'a [FVarChannelDescriptor<'a>],
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

    /// Builder-style setter for face-varying channels.
    pub fn with_fvar_channels(mut self, channels: &'a [FVarChannelDescriptor<'a>]) -> Self {
        self.fvar_channels = channels;
        self
    }
}
