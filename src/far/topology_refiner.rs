//! The topology refiner and its factory (port of
//! `opensubdiv/far/topologyRefiner.h` and
//! `opensubdiv/far/topologyRefinerFactory.h`).

use super::topology_descriptor::TopologyDescriptor;
use crate::sdc::{self, Scheme, SchemeType};
use crate::vtr::{Level, Refinement, TopologyError};
use crate::Index;

/// Options controlling uniform refinement
/// (`Far::TopologyRefiner::UniformOptions`).
#[derive(Debug, Clone, Copy)]
pub struct UniformOptions {
    /// The number of refinement iterations to apply.
    pub refinement_level: usize,
}

impl UniformOptions {
    pub fn new(refinement_level: usize) -> Self {
        Self { refinement_level }
    }
}

/// Public interface to the topology of one refinement level
/// (`Far::TopologyLevel`) — a thin view over a [`crate::vtr::Level`].
#[derive(Debug, Clone, Copy)]
pub struct TopologyLevel<'a> {
    level: &'a Level,
}

impl<'a> TopologyLevel<'a> {
    pub fn num_vertices(&self) -> usize {
        self.level.num_vertices()
    }

    pub fn num_edges(&self) -> usize {
        self.level.num_edges()
    }

    pub fn num_faces(&self) -> usize {
        self.level.num_faces()
    }

    /// Total number of face-vertex pairs (`GetNumFaceVerticesTotal`).
    pub fn num_face_vertices_total(&self) -> usize {
        self.level.num_face_vertices_total()
    }

    /// The vertices of `face`, in winding order (`GetFaceVertices`).
    pub fn face_vertices(&self, face: usize) -> &'a [Index] {
        self.level.face_vertices(face)
    }

    /// The edges of `face` (`GetFaceEdges`).
    pub fn face_edges(&self, face: usize) -> &'a [Index] {
        self.level.face_edges(face)
    }

    /// The two end vertices of `edge` (`GetEdgeVertices`).
    pub fn edge_vertices(&self, edge: usize) -> [Index; 2] {
        self.level.edge_vertices(edge)
    }

    /// The faces incident `edge` (`GetEdgeFaces`).
    pub fn edge_faces(&self, edge: usize) -> &'a [Index] {
        self.level.edge_faces(edge)
    }

    /// The faces incident `vertex` (`GetVertexFaces`).
    pub fn vertex_faces(&self, vertex: usize) -> &'a [Index] {
        self.level.vertex_faces(vertex)
    }

    /// The edges incident `vertex` (`GetVertexEdges`).
    pub fn vertex_edges(&self, vertex: usize) -> &'a [Index] {
        self.level.vertex_edges(vertex)
    }

    /// Find the edge connecting two vertices (`FindEdge`).
    pub fn find_edge(&self, v0: Index, v1: Index) -> Option<Index> {
        self.level.find_edge(v0, v1)
    }

    /// The effective sharpness of `edge` (`GetEdgeSharpness`). Boundary
    /// sharpening implied by the boundary-interpolation option is included.
    pub fn edge_sharpness(&self, edge: usize) -> f32 {
        self.level.edge_sharpness(edge)
    }

    /// The effective sharpness of `vertex` (`GetVertexSharpness`).
    pub fn vertex_sharpness(&self, vertex: usize) -> f32 {
        self.level.vertex_sharpness(vertex)
    }

    pub fn is_edge_boundary(&self, edge: usize) -> bool {
        self.level.is_edge_boundary(edge)
    }

    pub fn is_edge_non_manifold(&self, edge: usize) -> bool {
        self.level.is_edge_non_manifold(edge)
    }

    pub fn is_vertex_boundary(&self, vertex: usize) -> bool {
        self.level.is_vertex_boundary(vertex)
    }

    /// Is `face` tagged as a hole (`IsFaceHole`)?
    pub fn is_face_hole(&self, face: usize) -> bool {
        self.level.is_face_hole(face)
    }

    pub(super) fn inner(&self) -> &'a Level {
        self.level
    }
}

/// Stores the hierarchy of refined topology levels and the refinements
/// between them (`Far::TopologyRefiner`).
#[derive(Debug, Clone)]
pub struct TopologyRefiner {
    scheme: Scheme,
    options: sdc::Options,
    levels: Vec<Level>,
    refinements: Vec<Refinement>,
}

impl TopologyRefiner {
    /// The subdivision scheme of this refiner (`GetSchemeType`).
    pub fn scheme_type(&self) -> SchemeType {
        self.scheme.scheme_type()
    }

    /// The subdivision options of this refiner (`GetSchemeOptions`).
    pub fn scheme_options(&self) -> sdc::Options {
        self.options
    }

    pub(super) fn scheme(&self) -> &Scheme {
        &self.scheme
    }

    /// The number of refinement levels, including the base level
    /// (`GetNumLevels`).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// The highest refinement level (`GetMaxLevel`).
    pub fn max_level(&self) -> usize {
        self.levels.len() - 1
    }

    /// The topology of refinement level `level` (`GetLevel`).
    pub fn level(&self, level: usize) -> TopologyLevel<'_> {
        TopologyLevel {
            level: &self.levels[level],
        }
    }

    /// The refinement that produced level `level` from level `level - 1`.
    pub fn refinement(&self, level: usize) -> &Refinement {
        &self.refinements[level - 1]
    }

    /// Sum of the vertex counts of all levels (`GetNumVerticesTotal`).
    pub fn num_vertices_total(&self) -> usize {
        self.levels.iter().map(|l| l.num_vertices()).sum()
    }

    /// Sum of the face counts of all levels (`GetNumFacesTotal`).
    pub fn num_faces_total(&self) -> usize {
        self.levels.iter().map(|l| l.num_faces()).sum()
    }

    /// Refine the topology uniformly (`RefineUniform`): every face is
    /// subdivided at every iteration until `options.refinement_level` levels
    /// of refinement exist.
    ///
    /// Calling this more than once extends the existing hierarchy if the new
    /// target is deeper; it never discards levels.
    pub fn refine_uniform(&mut self, options: UniformOptions) {
        while self.max_level() < options.refinement_level {
            let (child, refinement) = Refinement::refine(self.levels.last().unwrap(), &self.scheme)
                .expect("refined topology is always internally consistent");
            self.levels.push(child);
            self.refinements.push(refinement);
        }
    }
}

/// Factory constructing a [`TopologyRefiner`] from a
/// [`TopologyDescriptor`]
/// (`Far::TopologyRefinerFactory<TopologyDescriptor>`).
pub struct TopologyRefinerFactory;

impl TopologyRefinerFactory {
    /// Validate the descriptor and instantiate the base level of a
    /// [`TopologyRefiner`] (`TopologyRefinerFactory::Create`).
    pub fn create(
        descriptor: TopologyDescriptor<'_>,
        scheme_type: SchemeType,
        options: sdc::Options,
    ) -> Result<TopologyRefiner, TopologyError> {
        // The Loop scheme is only defined for triangular meshes.
        if scheme_type == SchemeType::Loop {
            for (f, &size) in descriptor.num_verts_per_face.iter().enumerate() {
                if size != 3 {
                    return Err(TopologyError::NonTriangularFaceForLoop { face: f, size });
                }
            }
        }

        let mut level = Level::from_face_vertices(
            descriptor.num_vertices,
            descriptor.num_verts_per_face,
            descriptor.vert_indices_per_face,
        )?;

        // Creases.
        for (pair, &weight) in descriptor
            .crease_vertex_index_pairs
            .iter()
            .zip(descriptor.crease_weights)
        {
            for &v in pair {
                if v as usize >= level.num_vertices() {
                    return Err(TopologyError::InvalidDescriptorIndex {
                        what: "crease vertex",
                        index: v,
                    });
                }
            }
            let edge = level
                .find_edge(pair[0], pair[1])
                .ok_or(TopologyError::CreaseEdgeNotFound { vertices: *pair })?;
            level.set_edge_sharpness(edge as usize, weight);
        }

        // Corners.
        for (&v, &weight) in descriptor
            .corner_vertex_indices
            .iter()
            .zip(descriptor.corner_weights)
        {
            if v as usize >= level.num_vertices() {
                return Err(TopologyError::InvalidDescriptorIndex {
                    what: "corner vertex",
                    index: v,
                });
            }
            level.set_vertex_sharpness(v as usize, weight);
        }

        // Holes.
        for &f in descriptor.hole_indices {
            if f as usize >= level.num_faces() {
                return Err(TopologyError::InvalidDescriptorIndex {
                    what: "hole face",
                    index: f,
                });
            }
            level.set_face_hole(f as usize, true);
        }

        let scheme = Scheme::new(scheme_type, options);
        level.sharpen_boundaries(scheme.crease());

        Ok(TopologyRefiner {
            scheme,
            options,
            levels: vec![level],
            refinements: Vec::new(),
        })
    }
}
