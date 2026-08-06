//! The topology refiner and its factory (port of
//! `opensubdiv/far/topologyRefiner.h` and
//! `opensubdiv/far/topologyRefinerFactory.h`).

use super::fvar::FVarChannel;
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
    /// Refine uniformly `refinement_level` times.
    pub fn new(refinement_level: usize) -> Self {
        Self { refinement_level }
    }
}

/// Options controlling feature-adaptive refinement
/// (`Far::TopologyRefiner::AdaptiveOptions`).
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveOptions {
    /// The maximum level of refinement applied to isolate irregular
    /// features (`isolationLevel`).
    pub isolation_level: usize,
}

impl AdaptiveOptions {
    /// Isolate irregular features up to `isolation_level` levels deep.
    pub fn new(isolation_level: usize) -> Self {
        Self { isolation_level }
    }
}

/// Public interface to the topology of one refinement level
/// (`Far::TopologyLevel`) — a thin view over a [`crate::vtr::Level`].
#[derive(Debug, Clone, Copy)]
pub struct TopologyLevel<'a> {
    level: &'a Level,
    fvar_channels: &'a [FVarChannel],
    level_index: usize,
}

impl<'a> TopologyLevel<'a> {
    /// The number of vertices in this level (`GetNumVertices`).
    pub fn num_vertices(&self) -> usize {
        self.level.num_vertices()
    }

    /// The number of edges in this level (`GetNumEdges`).
    pub fn num_edges(&self) -> usize {
        self.level.num_edges()
    }

    /// The number of faces in this level (`GetNumFaces`).
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

    /// Does `edge` lie on a mesh boundary (`IsEdgeBoundary`)?
    pub fn is_edge_boundary(&self, edge: usize) -> bool {
        self.level.is_edge_boundary(edge)
    }

    /// Is `edge` non-manifold, i.e. shared by more than two faces
    /// (`IsEdgeNonManifold`)?
    pub fn is_edge_non_manifold(&self, edge: usize) -> bool {
        self.level.is_edge_non_manifold(edge)
    }

    /// Does `vertex` lie on a mesh boundary (`IsVertexBoundary`)?
    pub fn is_vertex_boundary(&self, vertex: usize) -> bool {
        self.level.is_vertex_boundary(vertex)
    }

    /// Is `face` tagged as a hole (`IsFaceHole`)?
    pub fn is_face_hole(&self, face: usize) -> bool {
        self.level.is_face_hole(face)
    }

    /// The number of face-varying channels (`GetNumFVarChannels`).
    pub fn num_fvar_channels(&self) -> usize {
        self.fvar_channels.len()
    }

    /// The number of face-varying values of `channel` at this level
    /// (`GetNumFVarValues`).
    pub fn num_fvar_values(&self, channel: usize) -> usize {
        self.fvar_channels[channel]
            .level(self.level_index)
            .num_vertices()
    }

    /// The face-varying values associated with the corners of `face`, in the
    /// same winding order as [`face_vertices`](Self::face_vertices)
    /// (`GetFaceFVarValues`).
    pub fn face_fvar_values(&self, face: usize, channel: usize) -> &'a [Index] {
        self.fvar_channels[channel]
            .level(self.level_index)
            .face_vertices(face)
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
    fvar_channels: Vec<FVarChannel>,
    /// True when the hierarchy was produced by [`refine_adaptive`]
    /// (`Self::refine_adaptive`); levels above 0 are then sparse.
    adaptive: bool,
    /// For each refinement step, the faces of the parent level that were
    /// selected for refinement (all `true` for uniform steps).
    selections: Vec<Vec<bool>>,
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
            fvar_channels: &self.fvar_channels,
            level_index: level,
        }
    }

    /// The number of face-varying channels (`GetNumFVarChannels`).
    pub fn num_fvar_channels(&self) -> usize {
        self.fvar_channels.len()
    }

    /// The face-varying channel `channel`.
    pub fn fvar_channel(&self, channel: usize) -> &FVarChannel {
        &self.fvar_channels[channel]
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
        assert!(
            !self.adaptive,
            "cannot mix uniform and adaptive refinement on one refiner"
        );
        while self.max_level() < options.refinement_level {
            let num_faces = self.levels.last().unwrap().num_faces();
            let (child, refinement) = Refinement::refine(self.levels.last().unwrap(), &self.scheme)
                .expect("refined topology is always internally consistent");
            self.levels.push(child);
            self.refinements.push(refinement);
            self.selections.push(vec![true; num_faces]);
            // Face-varying channels refine in lockstep with the geometry.
            for channel in &mut self.fvar_channels {
                channel.refine_once(None);
            }
        }
    }

    /// Feature-adaptively refine the topology (`RefineAdaptive`): starting
    /// from the base level, only faces whose neighborhood prevents the
    /// limit surface from being a (boundary-aware) bicubic B-spline patch —
    /// extraordinary vertices, non-quads, creases — are *selected* and
    /// subdivided, together with their one-ring support, until they resolve
    /// or `options.isolation_level` is reached. Levels above 0 are sparse:
    /// memory grows with the mesh's irregular features, not with `4^level`.
    ///
    /// Patches for an adaptively refined mesh live at mixed depths — build
    /// a [`super::PatchTable`] to evaluate the limit surface.
    /// [`super::PrimvarRefiner::interpolate`] works level by level as
    /// usual; [`super::PrimvarRefiner::limit`] and stencil tables require
    /// uniform refinement.
    ///
    /// Adaptive refinement applies to the quad-split schemes: for Bilinear
    /// only non-quad base faces need one round of isolation, and for Loop
    /// (whose patches are not yet supported) this is a no-op.
    pub fn refine_adaptive(&mut self, options: AdaptiveOptions) {
        assert!(
            self.max_level() == 0,
            "adaptive refinement must start from an unrefined refiner"
        );
        if self.scheme_type() == SchemeType::Loop {
            return;
        }
        self.adaptive = true;

        while self.max_level() < options.isolation_level {
            let level_index = self.max_level();
            let level = self.levels.last().unwrap();

            let mut selected = vec![false; level.num_faces()];
            let mut any = false;
            for (f, sel) in selected.iter_mut().enumerate() {
                if level.is_face_hole(f) || !self.face_is_candidate(level_index, f) {
                    continue;
                }
                let needs_isolation = match self.scheme_type() {
                    SchemeType::Catmark => {
                        super::patch_table::gather_regular_patch(level, f).is_none()
                    }
                    SchemeType::Bilinear => level.face_vertices(f).len() != 4,
                    SchemeType::Loop => false,
                };
                if needs_isolation {
                    *sel = true;
                    any = true;
                }
            }
            if !any {
                break;
            }

            let (child, refinement) = Refinement::refine_selected(
                self.levels.last().unwrap(),
                &self.scheme,
                Some(&selected),
            )
            .expect("refined topology is always internally consistent");
            self.levels.push(child);
            self.refinements.push(refinement);
            for channel in &mut self.fvar_channels {
                channel.refine_once(Some(&selected));
            }
            self.selections.push(selected);
        }
    }

    /// Was this refiner refined adaptively (`IsAdaptive`)?
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// Is `face` of `level` a *candidate* for patches or further isolation:
    /// the base level entirely, and above it the children of selected
    /// faces (whose neighborhoods are guaranteed complete).
    pub(super) fn face_is_candidate(&self, level: usize, face: usize) -> bool {
        if level == 0 {
            return true;
        }
        let parent = self.refinements[level - 1].child_face_parent_face(face) as usize;
        self.selections[level - 1][parent]
    }

    /// Was `face` of `level` selected for further refinement?
    pub(super) fn face_is_selected(&self, level: usize, face: usize) -> bool {
        level < self.selections.len() && self.selections[level][face]
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

        // Face-varying channels — built after boundary sharpening so the
        // geometry's effective sharpness transfers onto the value meshes.
        let mut fvar_channels = Vec::with_capacity(descriptor.fvar_channels.len());
        for (c, channel) in descriptor.fvar_channels.iter().enumerate() {
            fvar_channels.push(FVarChannel::create(
                &level,
                scheme_type,
                options,
                c,
                channel.num_values,
                channel.value_indices,
            )?);
        }

        Ok(TopologyRefiner {
            scheme,
            options,
            levels: vec![level],
            refinements: Vec::new(),
            fvar_channels,
            adaptive: false,
            selections: Vec::new(),
        })
    }
}
