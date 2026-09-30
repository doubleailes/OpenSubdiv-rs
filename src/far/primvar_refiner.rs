//! Primvar interpolation between refinement levels and onto the limit
//! surface (port of `opensubdiv/far/primvarRefiner.h`).

use super::topology_refiner::TopologyRefiner;
use crate::sdc::{
    Crease, EdgeNeighborhood, EdgeVertexMask, Scheme, Split, VertexNeighborhood, VertexVertexMask,
};
use crate::vtr::{Level, Refinement};
use crate::{Index, INDEX_INVALID};
use std::borrow::Cow;

/// Interface for primvar data interpolated by [`PrimvarRefiner`].
///
/// This is the Rust equivalent of the duck-typed `Clear()` /
/// `AddWithWeight()` protocol used throughout OpenSubdiv's Far layer.
pub trait Primvar: Clone {
    /// Reset the value to zero.
    fn clear(&mut self);
    /// Accumulate `src` scaled by `weight`.
    fn add_with_weight(&mut self, src: &Self, weight: f32);
}

impl Primvar for f32 {
    fn clear(&mut self) {
        *self = 0.0;
    }
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        *self += src * weight;
    }
}

impl Primvar for f64 {
    fn clear(&mut self) {
        *self = 0.0;
    }
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        *self += src * weight as f64;
    }
}

impl<const N: usize> Primvar for [f32; N] {
    fn clear(&mut self) {
        *self = [0.0; N];
    }
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        for (d, s) in self.iter_mut().zip(src) {
            *d += s * weight;
        }
    }
}

impl<const N: usize> Primvar for [f64; N] {
    fn clear(&mut self) {
        *self = [0.0; N];
    }
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        for (d, s) in self.iter_mut().zip(src) {
            *d += s * weight as f64;
        }
    }
}

/// Applies refinement operations to generic primvar data
/// (`Far::PrimvarRefiner`).
pub struct PrimvarRefiner<'a> {
    refiner: &'a TopologyRefiner,
}

impl<'a> PrimvarRefiner<'a> {
    /// Create a primvar refiner driven by the topology of `refiner`.
    pub fn new(refiner: &'a TopologyRefiner) -> Self {
        Self { refiner }
    }

    /// The refiner this primvar refiner was created with
    /// (`GetTopologyRefiner`).
    pub fn topology_refiner(&self) -> &TopologyRefiner {
        self.refiner
    }

    /// Apply vertex interpolation weights to a primvar buffer for a single
    /// level of refinement (`PrimvarRefiner::Interpolate`).
    ///
    /// `src` holds one value per vertex of level `level - 1`; `dst` receives
    /// one value per vertex of level `level`.
    ///
    /// # Panics
    /// Panics if `level` is zero or exceeds the refiner's
    /// [max level](TopologyRefiner::max_level), or if the buffers are too
    /// small.
    pub fn interpolate<T: Primvar>(&self, level: usize, src: &[T], dst: &mut [T]) {
        assert!(
            level >= 1 && level <= self.refiner.max_level(),
            "level {level} out of range 1..={}",
            self.refiner.max_level()
        );
        interpolate_level(
            self.refiner.scheme(),
            false,
            self.refiner.level(level - 1).inner(),
            self.refiner.level(level).inner(),
            self.refiner.refinement(level),
            src,
            dst,
        );
    }

    /// Apply face-varying interpolation weights to a primvar buffer for a
    /// single level of refinement
    /// (`PrimvarRefiner::InterpolateFaceVarying`).
    ///
    /// `src` holds one value per face-varying *value* of `channel` at level
    /// `level - 1`; `dst` receives one value per face-varying value at level
    /// `level` (see
    /// [`TopologyLevel::num_fvar_values`](super::TopologyLevel::num_fvar_values)).
    pub fn interpolate_face_varying<T: Primvar>(
        &self,
        level: usize,
        channel: usize,
        src: &[T],
        dst: &mut [T],
    ) {
        assert!(
            level >= 1 && level <= self.refiner.max_level(),
            "level {level} out of range 1..={}",
            self.refiner.max_level()
        );
        let fvar = self.refiner.fvar_channel(channel);
        // Base-level values are the caller's; the value mesh may split an
        // index reused at several vertices into several values.
        let src = if level == 1 {
            fvar.base_source_values(src)
        } else {
            Cow::Borrowed(src)
        };
        interpolate_level(
            fvar.scheme(),
            fvar.is_linear(),
            fvar.level(level - 1),
            fvar.level(level),
            fvar.refinement(level),
            &src,
            dst,
        );
    }

    /// Interpolate the whole hierarchy in one call: `base` holds one value
    /// per base-level (level 0) vertex, and one buffer per refined level is
    /// returned. A convenience not present in OpenSubdiv, where clients loop
    /// over `Interpolate(level, src, dst)` themselves.
    pub fn interpolate_all<T: Primvar + Default>(&self, base: &[T]) -> Vec<Vec<T>> {
        let mut levels = Vec::with_capacity(self.refiner.max_level());
        let mut src = base.to_vec();
        for level in 1..=self.refiner.max_level() {
            let mut dst = vec![T::default(); self.refiner.level(level).num_vertices()];
            self.interpolate(level, &src, &mut dst);
            src.clone_from(&dst);
            levels.push(dst);
        }
        levels
    }

    /// Apply *limit* weights to a primvar buffer (`PrimvarRefiner::Limit`):
    /// `src` holds one value per vertex of the refiner's last level, and
    /// `dst` receives the limit-surface position of each of those vertices.
    ///
    /// As in OpenSubdiv, evaluating the limit requires the neighborhood of
    /// every vertex to be "regularized" with respect to face size: for the
    /// Catmark scheme with non-quad base faces, refine at least once before
    /// calling this.
    pub fn limit<T: Primvar>(&self, src: &[T], dst: &mut [T]) {
        assert!(
            !self.refiner.is_adaptive(),
            "Limit requires uniform refinement; evaluate adaptive refiners through a PatchTable"
        );
        limit_level(
            self.refiner.scheme(),
            false,
            self.refiner.level(self.refiner.max_level()).inner(),
            src,
            dst,
        );
    }

    /// Apply face-varying *limit* weights to a primvar buffer
    /// (`PrimvarRefiner::LimitFaceVarying`): `src` holds one value per
    /// face-varying value of `channel` at the refiner's last level.
    pub fn limit_face_varying<T: Primvar>(&self, channel: usize, src: &[T], dst: &mut [T]) {
        assert!(
            !self.refiner.is_adaptive(),
            "Limit requires uniform refinement; evaluate adaptive refiners through a PatchTable"
        );
        let fvar = self.refiner.fvar_channel(channel);
        let max_level = self.refiner.max_level();
        let src = if max_level == 0 {
            fvar.base_source_values(src)
        } else {
            Cow::Borrowed(src)
        };
        limit_level(
            fvar.scheme(),
            fvar.is_linear(),
            fvar.level(max_level),
            &src,
            dst,
        );
    }
}

// ----------------------------------------------------------------------
//  Level-to-level interpolation (PrimvarRefiner::interpChildVertsFrom*)
//
//  These are parameterized over the scheme and the level pair so that the
//  same passes serve both vertex data (the geometry levels) and
//  face-varying data (a channel's value-mesh levels). `linear` selects the
//  purely linear masks used for FVarLinearInterpolation::All.
// ----------------------------------------------------------------------

fn interpolate_level<T: Primvar>(
    scheme: &Scheme,
    linear: bool,
    parent: &Level,
    child: &Level,
    refinement: &Refinement,
    src: &[T],
    dst: &mut [T],
) {
    assert!(src.len() >= parent.num_vertices(), "src buffer too small");
    assert!(dst.len() >= child.num_vertices(), "dst buffer too small");
    interpolate_child_verts_from_faces(scheme, refinement, parent, src, dst);
    interpolate_child_verts_from_edges(scheme, linear, refinement, parent, src, dst);
    interpolate_child_verts_from_verts(scheme, linear, refinement, parent, src, dst);
}

fn interpolate_child_verts_from_faces<T: Primvar>(
    scheme: &Scheme,
    refinement: &Refinement,
    parent: &Level,
    src: &[T],
    dst: &mut [T],
) {
    if refinement.split() == Split::ToTris {
        return; // Triangular splits generate no face child-vertices.
    }
    for f in 0..parent.num_faces() {
        let cv = refinement.face_child_vertex(f);
        if cv == INDEX_INVALID {
            continue; // not refined (sparse refinement)
        }
        let fv = parent.face_vertices(f);
        // The face mask is the centroid for linear and smooth schemes alike.
        let mask = scheme.compute_face_vertex_mask(fv.len());

        let mut acc = src[fv[0] as usize].clone();
        acc.clear();
        for &v in fv {
            acc.add_with_weight(&src[v as usize], mask.vertex_weight);
        }
        dst[cv as usize] = acc;
    }
}

fn interpolate_child_verts_from_edges<T: Primvar>(
    scheme: &Scheme,
    linear: bool,
    refinement: &Refinement,
    parent: &Level,
    src: &[T],
    dst: &mut [T],
) {
    let mut mask = EdgeVertexMask::default();

    for e in 0..parent.num_edges() {
        let ecv = refinement.edge_child_vertex(e);
        if ecv == INDEX_INVALID {
            continue; // not refined (sparse refinement)
        }
        let [v0, v1] = parent.edge_vertices(e);
        let mut acc = src[v0 as usize].clone();
        acc.clear();

        let edge_faces = parent.edge_faces(e);
        // A fringe edge of a sparse refinement can be missing the face
        // child-vertices its smooth mask references; such vertices only
        // support the region and are never used by patches, so a crease
        // (midpoint) fallback suffices.
        let incomplete = edge_faces
            .iter()
            .any(|&f| refinement.face_child_vertex(f as usize) == INDEX_INVALID)
            && refinement.split() == Split::ToQuads;

        if linear || incomplete {
            acc.add_with_weight(&src[v0 as usize], 0.5);
            acc.add_with_weight(&src[v1 as usize], 0.5);
            dst[ecv as usize] = acc;
            continue;
        }

        scheme.compute_edge_vertex_mask(
            &EdgeNeighborhood {
                sharpness: parent.edge_sharpness(e),
                num_faces: edge_faces.len(),
            },
            &mut mask,
        );

        acc.add_with_weight(&src[v0 as usize], mask.vertex_weights[0]);
        acc.add_with_weight(&src[v1 as usize], mask.vertex_weights[1]);

        for (&f, &w) in edge_faces.iter().zip(&mask.face_weights) {
            if w == 0.0 {
                continue;
            }
            if mask.face_weights_for_face_centers {
                // Catmark: weight the child vertex at the face center,
                // already computed in the face pass.
                let fcv = refinement.face_child_vertex(f as usize) as usize;
                let value = dst[fcv].clone();
                acc.add_with_weight(&value, w);
            } else {
                // Loop: weight the vertex of the face opposite the edge.
                let opposite = tri_vertex_opposite_edge(parent, f as usize, [v0, v1]);
                acc.add_with_weight(&src[opposite as usize], w);
            }
        }
        dst[ecv as usize] = acc;
    }
}

fn interpolate_child_verts_from_verts<T: Primvar>(
    scheme: &Scheme,
    linear: bool,
    refinement: &Refinement,
    parent: &Level,
    src: &[T],
    dst: &mut [T],
) {
    let crease = scheme.crease();
    let mut mask = VertexVertexMask::default();
    let mut edge_sharpness = Vec::new();
    let mut child_edge_sharpness = Vec::new();

    for v in 0..parent.num_vertices() {
        let cv = refinement.vertex_child_vertex(v);
        if cv == INDEX_INVALID {
            continue; // not refined (sparse refinement)
        }

        // A fringe vertex of a sparse quad refinement can be missing the
        // face child-vertices its smooth mask references; its child only
        // supports the region and is never used by patches, so carrying the
        // parent value suffices. Triangular masks reference parent vertices
        // only, so they are always complete.
        let incomplete = refinement.split() == Split::ToQuads
            && parent
                .vertex_faces(v)
                .iter()
                .any(|&f| refinement.face_child_vertex(f as usize) == INDEX_INVALID);

        if linear || incomplete {
            dst[cv as usize] = src[v].clone();
            continue;
        }

        // Gather the parent-level sharpness of the incident edges, and
        // derive the child-level sharpness directly through the creasing
        // rules — the same computation the refinement applied, valid even
        // when child topology is sparse.
        parent.gather_vertex_edge_sharpness(v, &mut edge_sharpness);
        child_edge_sharpness.clear();
        for &sharpness in edge_sharpness.iter() {
            child_edge_sharpness.push(if Crease::is_sharp(sharpness) {
                crease.subdivide_edge_sharpness_at_vertex(sharpness, &edge_sharpness)
            } else {
                0.0
            });
        }

        let neighborhood = VertexNeighborhood {
            sharpness: parent.vertex_sharpness(v),
            child_sharpness: crease.subdivide_vertex_sharpness(parent.vertex_sharpness(v)),
            edge_sharpness: &edge_sharpness,
            child_edge_sharpness: &child_edge_sharpness,
            num_faces: parent.vertex_faces(v).len(),
        };
        scheme.compute_vertex_vertex_mask(&neighborhood, &mut mask);

        let mut acc = src[v].clone();
        acc.clear();
        acc.add_with_weight(&src[v], mask.vertex_weight);
        for (&e, &w) in parent.vertex_edges(v).iter().zip(&mask.edge_weights) {
            if w != 0.0 {
                let opposite = parent.edge_opposite_vertex(e as usize, v as Index);
                acc.add_with_weight(&src[opposite as usize], w);
            }
        }
        if !mask.face_weights.is_empty() {
            debug_assert!(mask.face_weights_for_face_centers);
            for (&f, &w) in parent.vertex_faces(v).iter().zip(&mask.face_weights) {
                if w != 0.0 {
                    let fcv = refinement.face_child_vertex(f as usize) as usize;
                    let value = dst[fcv].clone();
                    acc.add_with_weight(&value, w);
                }
            }
        }
        dst[cv as usize] = acc;
    }
}

// ----------------------------------------------------------------------
//  Limit evaluation
// ----------------------------------------------------------------------

fn limit_level<T: Primvar>(scheme: &Scheme, linear: bool, level: &Level, src: &[T], dst: &mut [T]) {
    assert!(src.len() >= level.num_vertices(), "src buffer too small");
    assert!(dst.len() >= level.num_vertices(), "dst buffer too small");

    if linear {
        // The limit of linear interpolation is the data itself.
        dst[..level.num_vertices()].clone_from_slice(&src[..level.num_vertices()]);
        return;
    }

    let mut mask = VertexVertexMask::default();
    let mut edge_sharpness = Vec::new();

    for v in 0..level.num_vertices() {
        level.gather_vertex_edge_sharpness(v, &mut edge_sharpness);
        let neighborhood = VertexNeighborhood {
            sharpness: level.vertex_sharpness(v),
            child_sharpness: 0.0,
            edge_sharpness: &edge_sharpness,
            child_edge_sharpness: &[],
            num_faces: level.vertex_faces(v).len(),
        };
        scheme.compute_vertex_limit_mask(&neighborhood, &mut mask);

        let mut acc = src[v].clone();
        acc.clear();
        acc.add_with_weight(&src[v], mask.vertex_weight);
        for (&e, &w) in level.vertex_edges(v).iter().zip(&mask.edge_weights) {
            if w != 0.0 {
                let opposite = level.edge_opposite_vertex(e as usize, v as Index);
                acc.add_with_weight(&src[opposite as usize], w);
            }
        }
        if !mask.face_weights.is_empty() {
            debug_assert!(!mask.face_weights_for_face_centers);
            for (&f, &w) in level.vertex_faces(v).iter().zip(&mask.face_weights) {
                if w != 0.0 {
                    let diagonal = face_vertex_opposite(level, f as usize, v as Index);
                    acc.add_with_weight(&src[diagonal as usize], w);
                }
            }
        }
        dst[v] = acc;
    }
}

/// The vertex of (triangular) face `face` opposite the edge with end
/// vertices `edge_verts`.
fn tri_vertex_opposite_edge(level: &Level, face: usize, edge_verts: [Index; 2]) -> Index {
    let fv = level.face_vertices(face);
    debug_assert_eq!(fv.len(), 3);
    *fv.iter()
        .find(|&&v| v != edge_verts[0] && v != edge_verts[1])
        .expect("triangle must have a vertex opposite each edge")
}

/// The vertex of (quad) face `face` diagonally opposite `vertex`.
fn face_vertex_opposite(level: &Level, face: usize, vertex: Index) -> Index {
    let fv = level.face_vertices(face);
    debug_assert_eq!(
        fv.len(),
        4,
        "limit masks with face weights require quad neighborhoods; refine at least once"
    );
    let i = fv
        .iter()
        .position(|&v| v == vertex)
        .expect("vertex must belong to its incident face");
    fv[(i + 2) % fv.len()]
}
