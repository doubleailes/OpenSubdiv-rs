//! Face-varying channel refinement.
//!
//! Port of the face-varying support spread across OpenSubdiv's
//! `vtr/fvarLevel.{h,cpp}`, `vtr/fvarRefinement.{h,cpp}` and the
//! face-varying paths of `far/topologyRefiner` / `far/primvarRefiner`.
//!
//! A face-varying channel assigns a *value index* to every face-corner, so
//! data such as UVs can be discontinuous ("seamed") across edges. The key
//! observation — equivalent to how OpenSubdiv tags fvar topology — is that a
//! channel forms a mesh of its own: its "vertices" are the channel *values*
//! and its faces mirror the mesh faces, with corners remapped to value
//! indices. Where neighboring faces disagree on values, that value-mesh
//! simply has two distinct (boundary) edges instead of one shared interior
//! edge: **seams are boundaries of the value mesh**.
//!
//! A channel is therefore represented as a [`Level`] per refinement level and
//! refined with the same [`Refinement`] machinery as the geometry, after
//! encoding the channel's [linear-interpolation
//! rule](crate::sdc::FVarLinearInterpolation) as sharpness on that value
//! mesh. As in OpenSubdiv, a value is a property of a *geometric vertex*: a
//! value index reused at several vertices (a deduplicated UV, say) is one
//! independent value at each of them, so the base value mesh gets one vertex
//! per distinct (vertex, value index) pair. The caller's indices are kept for
//! the base level's API and for gathering the source values of the first
//! interpolation step; refined levels only ever see the split values.
//!
//! The linear-interpolation rules map onto the value mesh as follows:
//!
//! * `All` — every mask is linear; no sharpening needed.
//! * `None` — value-mesh boundaries (seams) subdivide as smooth crease
//!   curves: boundary edges are sharpened (like
//!   [`VtxBoundaryInterpolation::EdgeOnly`](crate::sdc::VtxBoundaryInterpolation)).
//! * `CornersOnly` — additionally pins values incident exactly one face
//!   (like `EdgeAndCorner`).
//! * `CornersPlus1` — additionally pins all values at vertices where three
//!   or more distinct values meet ("junctions").
//! * `CornersPlus2` — additionally pins values at fvar darts (an interior
//!   vertex where a seam terminates) and at concave corners: where exactly
//!   two values meet and one of them is a corner (spans a single face), the
//!   other spans the remaining faces around the vertex and is pinned too.
//! * `Boundaries` — every boundary/seam value is pinned, making all fvar
//!   boundaries piecewise linear.
//!
//! Geometric sharpness (creases/corners and sharpened mesh boundaries) is
//! transferred onto the value mesh so that fvar data respects the same sharp
//! features as vertex data.

use crate::sdc::{self, Crease, FVarLinearInterpolation, Scheme, SchemeType};
use crate::vtr::{Level, Refinement, TopologyError};
use crate::Index;
use std::borrow::Cow;

/// One face-varying channel of a
/// [`TopologyRefiner`](super::TopologyRefiner): the hierarchy of value-mesh
/// levels refined in lockstep with the geometry.
#[derive(Debug, Clone)]
pub struct FVarChannel {
    linear_interpolation: FVarLinearInterpolation,
    /// Scheme used to refine and interpolate the value mesh. Same scheme
    /// type as the geometry, but with the boundary-interpolation option
    /// derived from the channel's linear-interpolation rule.
    scheme: Scheme,
    /// True when every mask is linear (`FVarLinearInterpolation::All`).
    linear: bool,
    /// The number of values the channel was described with.
    num_base_values: usize,
    /// The caller's value index of each base-level face-corner.
    base_face_values: Vec<Index>,
    /// The caller's value index of each base value-mesh vertex, when some
    /// index is reused at several geometric vertices and the value mesh
    /// therefore has more vertices than the channel has values; `None`
    /// when the two coincide.
    base_value_sources: Option<Vec<Index>>,
    levels: Vec<Level>,
    refinements: Vec<Refinement>,
}

impl FVarChannel {
    /// Build the base level of a channel from its per-corner value indices,
    /// validating them against the geometry's base level.
    pub(super) fn create(
        geometry: &Level,
        scheme_type: SchemeType,
        options: sdc::Options,
        channel_index: usize,
        num_values: usize,
        value_indices: &[Index],
    ) -> Result<FVarChannel, TopologyError> {
        if value_indices.len() != geometry.num_face_vertices_total() {
            return Err(TopologyError::FVarValueCountMismatch {
                channel: channel_index,
                expected: geometry.num_face_vertices_total(),
                actual: value_indices.len(),
            });
        }
        for &v in value_indices {
            if v as usize >= num_values {
                return Err(TopologyError::FVarValueIndexOutOfRange {
                    channel: channel_index,
                    index: v,
                });
            }
        }

        let mode = options.fvar_linear_interpolation;
        let linear = mode == FVarLinearInterpolation::All;
        let scheme = Scheme::new(scheme_type, channel_options(options));

        // Build the value mesh: same faces as the geometry, corners remapped
        // to value indices, with an index reused at several geometric
        // vertices split into one value-mesh vertex per vertex.
        let verts_per_face: Vec<usize> = (0..geometry.num_faces())
            .map(|f| geometry.face_vertices(f).len())
            .collect();
        let (split_indices, base_value_sources) =
            split_reused_values(geometry, num_values, value_indices);
        let num_split_values = base_value_sources.as_ref().map_or(num_values, Vec::len);
        let mut value_mesh =
            Level::from_face_vertices(num_split_values, &verts_per_face, &split_indices)?;

        if !linear {
            // Transfer geometric sharpness onto the value mesh so creases
            // and sharpened mesh boundaries also affect fvar data. Corner
            // `i` of face `f` associates geometric edge/vertex with the
            // value-mesh edge/value at the same position.
            for f in 0..geometry.num_faces() {
                let geom_edges = geometry.face_edges(f);
                let geom_verts = geometry.face_vertices(f);
                let fvar_edges = value_mesh.face_edges(f).to_vec();
                let fvar_values = value_mesh.face_vertices(f).to_vec();
                for i in 0..geom_verts.len() {
                    let ge = geom_edges[i] as usize;
                    let fe = fvar_edges[i] as usize;
                    let es = geometry
                        .edge_sharpness(ge)
                        .max(value_mesh.edge_sharpness(fe));
                    value_mesh.set_edge_sharpness(fe, es);

                    let gv = geom_verts[i] as usize;
                    let fv = fvar_values[i] as usize;
                    let vs = geometry
                        .vertex_sharpness(gv)
                        .max(value_mesh.vertex_sharpness(fv));
                    value_mesh.set_vertex_sharpness(fv, vs);
                }
            }

            // Fvar boundaries (mesh boundaries and seams) subdivide as
            // creases; `EdgeAndCorner`-mapped modes also pin one-face values.
            value_mesh.sharpen_boundaries(scheme.crease());

            apply_mode_sharpening(mode, geometry, &mut value_mesh);
        }

        Ok(FVarChannel {
            linear_interpolation: mode,
            scheme,
            linear,
            num_base_values: num_values,
            base_face_values: value_indices.to_vec(),
            base_value_sources,
            levels: vec![value_mesh],
            refinements: Vec::new(),
        })
    }

    /// Refine the channel by one level, in lockstep with the geometry:
    /// `included` is the geometry's mask of faces to refine (its selection
    /// already expanded to the one-ring support), or `None` for uniform
    /// refinement. The value mesh's faces mirror the geometry's, so the
    /// same mask applies — it must not be re-expanded on the value mesh,
    /// whose seams split vertices and would include fewer support faces,
    /// leaving the two meshes with different child faces.
    pub(super) fn refine_once(&mut self, included: Option<&[bool]>) {
        let parent = self.levels.last().unwrap();
        let (mut child, refinement) = match included {
            Some(included) => Refinement::refine_included(parent, &self.scheme, included),
            None => Refinement::refine(parent, &self.scheme),
        }
        .expect("refined face-varying topology is always internally consistent");
        // `Boundaries` pins *every* boundary value, including the values
        // newly created on boundary/seam edges, so it must be re-applied at
        // each level (the other modes' pins persist through vertex-sharpness
        // subdivision or `sharpen_boundaries`).
        if self.linear_interpolation == FVarLinearInterpolation::Boundaries {
            pin_boundary_values(&mut child);
        }
        self.levels.push(child);
        self.refinements.push(refinement);
    }

    /// The channel's linear-interpolation rule.
    pub fn linear_interpolation(&self) -> FVarLinearInterpolation {
        self.linear_interpolation
    }

    pub(super) fn scheme(&self) -> &Scheme {
        &self.scheme
    }

    pub(super) fn is_linear(&self) -> bool {
        self.linear
    }

    pub(super) fn level(&self, level: usize) -> &Level {
        &self.levels[level]
    }

    /// The number of values at `level` as seen by the channel's user: the
    /// number of values it was described with at the base level, and the
    /// value-mesh vertex count above.
    pub(super) fn num_values(&self, level: usize) -> usize {
        if level == 0 {
            self.num_base_values
        } else {
            self.levels[level].num_vertices()
        }
    }

    /// The values at the corners of `face` at `level`: the caller's indices
    /// at the base level, value-mesh vertices above.
    pub(super) fn face_values(&self, level: usize, face: usize) -> &[Index] {
        if level == 0 {
            let level0 = &self.levels[0];
            let start = level0.face_vertices_offset(face);
            &self.base_face_values[start..start + level0.face_vertices(face).len()]
        } else {
            self.levels[level].face_vertices(face)
        }
    }

    /// Source values for interpolating from the base level, one per base
    /// value-mesh vertex: `src` itself when every value is used at a single
    /// geometric vertex, or `src` gathered through the split.
    pub(super) fn base_source_values<'a, T: Clone>(&self, src: &'a [T]) -> Cow<'a, [T]> {
        assert!(
            src.len() >= self.num_base_values,
            "src buffer too small: {} values, expected at least {}",
            src.len(),
            self.num_base_values
        );
        match &self.base_value_sources {
            None => Cow::Borrowed(src),
            Some(sources) => Cow::Owned(sources.iter().map(|&v| src[v as usize].clone()).collect()),
        }
    }

    pub(super) fn refinement(&self, level: usize) -> &Refinement {
        &self.refinements[level - 1]
    }
}

/// Split value indices reused at several geometric vertices into one value
/// per (vertex, index) pair. Returns the per-corner indices of the split
/// values and, when any index had to be split, the caller's index of each
/// split value. The first vertex using an index keeps that index, so that
/// the split is the identity whenever no index is reused.
fn split_reused_values(
    geometry: &Level,
    num_values: usize,
    value_indices: &[Index],
) -> (Vec<Index>, Option<Vec<Index>>) {
    // Geometric vertex at which each value was first seen.
    let mut first_vertex: Vec<Option<Index>> = vec![None; num_values];
    // Extra (value, split id) pairs of each geometric vertex.
    let mut extra_at_vertex: Vec<Vec<(Index, Index)>> = vec![Vec::new(); geometry.num_vertices()];
    let mut sources: Vec<Index> = (0..num_values as Index).collect();

    let mut split = Vec::with_capacity(value_indices.len());
    let mut corner = 0;
    for f in 0..geometry.num_faces() {
        for &v in geometry.face_vertices(f) {
            let val = value_indices[corner];
            corner += 1;
            let id = match first_vertex[val as usize] {
                None => {
                    first_vertex[val as usize] = Some(v);
                    val
                }
                Some(first) if first == v => val,
                Some(_) => {
                    let extra = &mut extra_at_vertex[v as usize];
                    match extra.iter().find(|(existing, _)| *existing == val) {
                        Some(&(_, id)) => id,
                        None => {
                            let id = sources.len() as Index;
                            sources.push(val);
                            extra.push((val, id));
                            id
                        }
                    }
                }
            };
            split.push(id);
        }
    }
    let sources = (sources.len() > num_values).then_some(sources);
    (split, sources)
}

/// The value-mesh boundary interpolation implied by each fvar
/// linear-interpolation rule.
fn channel_options(options: sdc::Options) -> sdc::Options {
    use sdc::VtxBoundaryInterpolation::{EdgeAndCorner, EdgeOnly};
    let vtx = match options.fvar_linear_interpolation {
        FVarLinearInterpolation::All | FVarLinearInterpolation::None => EdgeOnly,
        FVarLinearInterpolation::CornersOnly
        | FVarLinearInterpolation::CornersPlus1
        | FVarLinearInterpolation::CornersPlus2
        | FVarLinearInterpolation::Boundaries => EdgeAndCorner,
    };
    options.with_vtx_boundary_interpolation(vtx)
}

fn apply_mode_sharpening(mode: FVarLinearInterpolation, geometry: &Level, value_mesh: &mut Level) {
    match mode {
        FVarLinearInterpolation::All
        | FVarLinearInterpolation::None
        | FVarLinearInterpolation::CornersOnly => {}
        FVarLinearInterpolation::CornersPlus1 | FVarLinearInterpolation::CornersPlus2 => {
            // Distinct values meeting at each geometric vertex, with the
            // number of incident faces each value spans there.
            let mut values_at_vertex: Vec<Vec<(Index, usize)>> =
                vec![Vec::new(); geometry.num_vertices()];
            for f in 0..geometry.num_faces() {
                let gv = geometry.face_vertices(f);
                let fv = value_mesh.face_vertices(f);
                for (&v, &val) in gv.iter().zip(fv) {
                    let vals = &mut values_at_vertex[v as usize];
                    match vals.iter_mut().find(|(existing, _)| *existing == val) {
                        Some((_, span)) => *span += 1,
                        None => vals.push((val, 1)),
                    }
                }
            }
            for (v, vals) in values_at_vertex.iter().enumerate() {
                // Junctions: three or more values meeting at a vertex.
                let junction = vals.len() > 2;
                // Darts (`CornersPlus2` only): an interior vertex whose single
                // value nevertheless lies on an fvar boundary — a seam
                // terminates here.
                let dart = mode == FVarLinearInterpolation::CornersPlus2
                    && vals.len() == 1
                    && !geometry.is_vertex_boundary(v)
                    && value_mesh.is_vertex_boundary(vals[0].0 as usize);
                // Concave corners (`CornersPlus2` only): exactly two values
                // meet and one of them is a corner (spans a single face), so
                // the other spans all remaining faces around the vertex — a
                // reflex corner of its UV island. OpenSubdiv sharpens both
                // (`sharpenBothIfOneCorner` in `FVarLevel`); the corner is
                // already pinned, so this pins the concave value.
                let concave = mode == FVarLinearInterpolation::CornersPlus2
                    && vals.len() == 2
                    && (vals[0].1 == 1 || vals[1].1 == 1);
                if junction || dart || concave {
                    for &(val, _) in vals {
                        value_mesh.set_vertex_sharpness(val as usize, sdc::SHARPNESS_INFINITE);
                    }
                }
            }
        }
        FVarLinearInterpolation::Boundaries => pin_boundary_values(value_mesh),
    }
}

/// Pin (make infinitely sharp) every value lying on a boundary of the value
/// mesh, so fvar boundaries and seams interpolate linearly.
fn pin_boundary_values(value_mesh: &mut Level) {
    for v in 0..value_mesh.num_vertices() {
        if value_mesh.is_vertex_boundary(v) && !Crease::is_infinite(value_mesh.vertex_sharpness(v))
        {
            value_mesh.set_vertex_sharpness(v, sdc::SHARPNESS_INFINITE);
        }
    }
}
