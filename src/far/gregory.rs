//! Gregory-basis end-cap patches for irregular faces.
//!
//! Port of the role of `opensubdiv/far/catmarkPatchBuilder.cpp`
//! (`CatmarkLimits` and `GregoryConverter`): where the Catmull-Clark limit
//! surface over a face is not a bicubic B-spline — a face with an
//! extraordinary corner, a corner on an irregular boundary or infinitely
//! sharp crease, a sharp (pinned) corner, or a dart — the surface is
//! approximated by a *Gregory patch*: 20 control points forming a bicubic
//! Bézier patch whose four interior points are rational blends, giving
//! corner limit-point interpolation, exact C0 boundaries with neighboring
//! patches, and approximate G1 smoothness. This is OpenSubdiv's
//! `ENDCAP_GREGORY_BASIS` end cap, including its treatment of infinitely
//! sharp features (`useInfSharpPatch`).
//!
//! Each of the 20 points is stored as a sparse stencil on the refined
//! mesh's vertices, so evaluation stays weight-based like every other patch
//! type. The coefficients are OpenSubdiv's own (they predate and differ
//! slightly from Loop, Schaefer, Nießner & Castaño's published
//! construction): on a fully regular neighborhood — interior, boundary,
//! pinned corner or infinitely sharp crease alike — the construction
//! degenerates to the exact bicubic B-spline patch, a property the unit
//! tests verify for arbitrary control data, pinning down every coefficient.
//!
//! ## Corner classification
//!
//! Following the reference, each corner of the face is described by the
//! *span* of faces around its vertex that contains the patch face: the
//! fan bounded by *singular* edges (boundary edges, non-manifold edges and
//! — unless the vertex is a dart — infinitely sharp creases), or the full
//! periodic one-ring when no singular edge is met. A corner is then
//!
//! * **regular** (2 faces on a boundary/crease, 4 faces interior) — its
//!   points are those of the B-spline patch;
//! * **smooth interior** — limit point and scaled limit tangents of the
//!   full ring (`CatmarkLimits::ComputeInteriorPointWeights`);
//! * **smooth boundary/crease** — limit point on the boundary curve, one
//!   edge point along the boundary curve (two non-zero weights) and one
//!   along the interior tangent (`ComputeBoundaryPointWeights`);
//! * **sharp** (infinitely sharp vertex, or three or more infinitely sharp
//!   edges) — the corner point is the vertex itself, and the edge points
//!   lie a third of the way along the face's edges;
//! * a **smooth corner** of a single face — the crease limit rules of its
//!   two boundary edges.
//!
//! Semi-sharp creases still unresolved at the isolation level are ignored
//! (their vertices are treated as smooth, or as darts), as OpenSubdiv does.
//! Non-manifold neighborhoods need no special case: their edges and
//! vertices are made infinitely sharp when the base level is built (as in
//! OpenSubdiv), so each face is capped over its own manifold span. Only
//! unsharpened boundaries (`VtxBoundaryInterpolation::None`, whose smooth
//! boundary rules have no Gregory counterpart) are left to the bilinear
//! fallback of [`super::PatchTableFactory`].

use crate::sdc::{self, Crease, Rule};
use crate::vtr::Level;
use crate::Index;

/// A point expressed as a sparse weighted combination of mesh vertices.
#[derive(Debug, Clone, Default)]
pub(crate) struct SparsePoint(pub(crate) Vec<(Index, f32)>);

impl SparsePoint {
    pub(super) fn of(entries: &[(Index, f32)]) -> SparsePoint {
        let mut p = SparsePoint::default();
        for &(i, w) in entries {
            p.add(i, w);
        }
        p
    }

    pub(super) fn add(&mut self, index: Index, weight: f32) {
        match self.0.iter_mut().find(|(i, _)| *i == index) {
            Some((_, w)) => *w += weight,
            None => self.0.push((index, weight)),
        }
    }

    pub(super) fn add_scaled(&mut self, other: &SparsePoint, scale: f32) {
        for &(i, w) in &other.0 {
            self.add(i, w * scale);
        }
    }

    pub(super) fn scaled(&self, scale: f32) -> SparsePoint {
        SparsePoint(self.0.iter().map(|&(i, w)| (i, w * scale)).collect())
    }
}

/// The number of control points of a Gregory patch.
pub(crate) const NUM_POINTS: usize = 20;

/// The 20 Gregory control points, ordered per corner as
/// `[P, E+, E-, F+, F-]` (matching OpenSubdiv's Gregory-basis layout).
pub(crate) type GregoryPoints = [SparsePoint; NUM_POINTS];

// ----------------------------------------------------------------------
//  Corner spans
// ----------------------------------------------------------------------

/// Is `edge` a wall for the neighborhood of a patch: a boundary or
/// non-manifold edge, or — when `inf_sharp` is set — an infinitely sharp
/// crease (which the limit surface treats exactly like a boundary)?
pub(super) fn is_edge_singular(level: &Level, edge: Index, inf_sharp: bool) -> bool {
    let e = edge as usize;
    level.edge_faces(e).len() != 2 || (inf_sharp && Crease::is_infinite(level.edge_sharpness(e)))
}

/// The fan of faces around one corner of a face that contains the patch
/// face, bounded by singular edges (`Vtr::Level::VSpan` plus the ring
/// ordering of `Far::SourcePatch`). Every face of the span has the same
/// size as the patch face (quads for Catmark, triangles for Loop).
///
/// `edges` are the incident edges in counter-clockwise order and `faces[i]`
/// sits between `edges[i]` and `edges[i + 1]`. For a periodic span
/// (`boundary == false`) `edges[0]` is the edge toward the face's next
/// corner, `faces[0]` is the patch face and `edges.len() == faces.len()`.
/// For a bounded span `edges[0]` is the leading singular edge (the one
/// reached walking clockwise from the patch face), `edges` closes with the
/// trailing singular edge — the same edge again when only one singular
/// edge exists — and `edges.len() == faces.len() + 1`.
pub(super) struct CornerSpan {
    pub(super) vertex: Index,
    pub(super) edges: Vec<Index>,
    pub(super) faces: Vec<Index>,
    /// Index of the patch face within `faces`.
    pub(super) face_in_span: usize,
    /// Is the span bounded by singular edges (as opposed to periodic)?
    pub(super) boundary: bool,
}

impl CornerSpan {
    pub(super) fn num_faces(&self) -> usize {
        self.faces.len()
    }

    fn valence(&self) -> usize {
        self.edges.len()
    }

    /// The far end of `edges[i]`.
    pub(super) fn edge_end(&self, level: &Level, i: usize) -> Index {
        level.edge_opposite_vertex(self.edges[i] as usize, self.vertex)
    }

    /// The vertex of (quad) `faces[i]` diagonally opposite the corner
    /// vertex.
    fn diagonal(&self, level: &Level, i: usize) -> Index {
        let fv = level.face_vertices(self.faces[i] as usize);
        debug_assert_eq!(fv.len(), 4);
        let k = fv
            .iter()
            .position(|&v| v == self.vertex)
            .expect("span faces are incident the corner vertex");
        fv[(k + 2) % 4]
    }
}

fn other_face_of_edge(level: &Level, edge: Index, face: Index) -> Option<Index> {
    let faces = level.edge_faces(edge as usize);
    if faces.len() != 2 {
        return None;
    }
    Some(if faces[0] == face { faces[1] } else { faces[0] })
}

/// The other edge of `face` incident `vertex`.
fn other_edge_at_vertex(level: &Level, face: Index, vertex: Index, edge: Index) -> Option<Index> {
    let fv = level.face_vertices(face as usize);
    let fe = level.face_edges(face as usize);
    let n = fv.len();
    let i = fv.iter().position(|&x| x == vertex)?;
    let leading = fe[i];
    let trailing = fe[(i + n - 1) % n];
    if leading == edge {
        Some(trailing)
    } else if trailing == edge {
        Some(leading)
    } else {
        None
    }
}

/// Identify the span of faces around corner `corner` of `face`
/// (`identifyManifoldCornerSpan`), treating infinitely sharp edges as
/// singular when `inf_sharp_singular` is set. `None` when the span contains
/// faces of a different size than `face` (non-quads around a quad, or
/// non-triangles around a triangle).
///
/// Non-manifold vertices need no special case: the walk never crosses a
/// non-manifold edge, so the span is the manifold fan of faces containing
/// the patch face, as in the reference. (Base-level sharpening makes every
/// non-manifold vertex infinitely sharp or a crease along its non-manifold
/// edges, so its other fans do not affect the limit surface over this one.)
pub(super) fn corner_span(
    level: &Level,
    face: usize,
    corner: usize,
    inf_sharp_singular: bool,
) -> Option<CornerSpan> {
    let fv = level.face_vertices(face);
    let fe = level.face_edges(face);
    let vertex = fv[corner];
    let size = fv.len();
    let num_faces = level.vertex_faces(vertex as usize).len();

    let singular = |e: Index| is_edge_singular(level, e, inf_sharp_singular);
    let face = face as Index;
    let e_lead = fe[corner]; // toward the face's next corner

    // Walk clockwise (across the leading edge, away from the face) until a
    // singular edge is met or the ring closes.
    let mut start_face = face;
    let mut start_edge = e_lead;
    let mut periodic = false;
    let mut steps = 0;
    loop {
        if singular(start_edge) {
            break;
        }
        let next = other_face_of_edge(level, start_edge, start_face)?;
        if next == face {
            periodic = true;
            break;
        }
        if level.face_vertices(next as usize).len() != size {
            return None;
        }
        start_edge = other_edge_at_vertex(level, next, vertex, start_edge)?;
        start_face = next;
        steps += 1;
        if steps > num_faces {
            return None;
        }
    }
    if periodic {
        start_face = face;
        start_edge = e_lead;
    }

    // Gather the span counter-clockwise from its leading edge.
    let mut edges = vec![start_edge];
    let mut faces = vec![start_face];
    let mut current_face = start_face;
    let mut current_edge = other_edge_at_vertex(level, start_face, vertex, start_edge)?;
    loop {
        if periodic && current_edge == e_lead {
            break;
        }
        if !periodic && singular(current_edge) {
            edges.push(current_edge);
            break;
        }
        edges.push(current_edge);
        let next = other_face_of_edge(level, current_edge, current_face)?;
        if level.face_vertices(next as usize).len() != size {
            return None;
        }
        faces.push(next);
        current_edge = other_edge_at_vertex(level, next, vertex, current_edge)?;
        current_face = next;
        if faces.len() > num_faces {
            return None;
        }
    }
    let face_in_span = faces.iter().position(|&f| f == face)?;
    Some(CornerSpan {
        vertex,
        edges,
        faces,
        face_in_span,
        boundary: !periodic,
    })
}

/// The `Sdc::Crease::Rule` at a vertex with the given sharpness and number
/// of sharp incident edges (the rule does not depend on the options).
pub(super) fn vertex_rule(vertex_sharpness: f32, sharp_edges: usize) -> Rule {
    Crease::new(sdc::Options::default()).determine_vertex_vertex_rule(vertex_sharpness, sharp_edges)
}

// ----------------------------------------------------------------------
//  Corner, edge and face points (CatmarkLimits / GregoryConverter)
// ----------------------------------------------------------------------

/// The scale factor applied to limit tangents at an interior vertex of the
/// given valence, arising from the eigenvalues of the subdivision matrix
/// (`CatmarkLimits::computeCoefficient`); `1/2` at valence 4.
fn edge_factor(valence: usize) -> f32 {
    let inv = 1.0 / valence as f64;
    let cos_t = (2.0 * std::f64::consts::PI * inv).cos();
    let divisor = (cos_t + 5.0) + ((cos_t + 9.0) * (cos_t + 1.0)).sqrt();
    (16.0 * inv / divisor) as f32
}

/// Corner point and edge points of a smooth interior corner
/// (`CatmarkLimits::ComputeInteriorPointWeights`): the Catmull-Clark limit
/// point of the vertex, offset along its scaled limit tangents in the
/// directions of the face's two edges.
fn interior_points(level: &Level, span: &CornerSpan) -> (SparsePoint, SparsePoint, SparsePoint) {
    let n = span.valence();
    let nf = n as f32;
    let p_coeff = 1.0 / (nf * (nf + 5.0));

    let mut p = SparsePoint::of(&[(span.vertex, nf / (nf + 5.0))]);
    for i in 0..n {
        p.add(span.edge_end(level, i), 4.0 * p_coeff);
        p.add(span.diagonal(level, i), p_coeff);
    }

    // The limit tangent along edge `j`: each edge point contributes three
    // cosine terms (its own angle and its two neighbors'), each face point
    // two.
    let tan_coeff = edge_factor(n) * 0.5 / (nf + 5.0);
    let theta = std::f32::consts::TAU / nf;
    let tangent = |j: usize| -> SparsePoint {
        let mut t = SparsePoint::default();
        for i in 0..n {
            let a = (i as f32 - j as f32) * theta;
            t.add(
                span.edge_end(level, i),
                tan_coeff * (2.0 * (a + theta).cos() + 4.0 * a.cos() + 2.0 * (a - theta).cos()),
            );
            t.add(
                span.diagonal(level, i),
                tan_coeff * (a.cos() + (a + theta).cos()),
            );
        }
        t
    };

    let mut ep = p.clone();
    ep.add_scaled(&tangent(0), 1.0);
    let mut em = p.clone();
    em.add_scaled(&tangent(1), 1.0);
    (p, ep, em)
}

/// Corner point and edge points of a smooth corner on a boundary or
/// infinitely sharp crease with two or more faces in its span
/// (`CatmarkLimits::ComputeBoundaryPointWeights`). The limit point and the
/// tangent along the boundary come from the cubic B-spline boundary curve;
/// the tangent across it from the crease limit-tangent rule; an edge point
/// on an interior edge at angle `a` from the leading boundary edge is
/// offset along `cos(a)·t_boundary + sin(a)·t_interior`.
fn boundary_points(level: &Level, span: &CornerSpan) -> (SparsePoint, SparsePoint, SparsePoint) {
    let k = span.num_faces();
    debug_assert!(span.boundary && k > 1);
    let v = span.vertex;
    let e0 = span.edge_end(level, 0);
    let ek = span.edge_end(level, k);

    let p = SparsePoint::of(&[(v, 4.0 / 6.0), (e0, 1.0 / 6.0), (ek, 1.0 / 6.0)]);
    let t_boundary = SparsePoint::of(&[(e0, 1.0 / 6.0), (ek, -1.0 / 6.0)]);

    let kf = k as f32;
    let theta = std::f32::consts::PI / kf;
    let c = theta.cos();
    let s = theta.sin();
    let div3 = 1.0 / 3.0;
    let div3kc = 1.0 / (3.0 * kf + c);
    let gamma = -4.0 * s * div3kc;
    let alpha_0k = -((1.0 + 2.0 * c) * (1.0 + c).sqrt()) * div3kc / (1.0 - c).sqrt();
    let beta_0 = s * div3kc;

    let mut t_interior = SparsePoint::of(&[
        (v, gamma * div3),
        (e0, alpha_0k * div3),
        (ek, alpha_0k * div3),
        (span.diagonal(level, 0), beta_0 * div3),
    ]);
    for i in 1..k {
        let sin_i = (theta * i as f32).sin();
        let sin_i1 = (theta * (i + 1) as f32).sin();
        t_interior.add(span.edge_end(level, i), 4.0 * sin_i * div3kc * div3);
        t_interior.add(span.diagonal(level, i), (sin_i + sin_i1) * div3kc * div3);
    }

    let along = |angle: f32| -> SparsePoint {
        let mut e = p.clone();
        e.add_scaled(&t_boundary, angle.cos());
        e.add_scaled(&t_interior, angle.sin());
        e
    };
    let f = span.face_in_span;
    let ep = if f == 0 {
        SparsePoint::of(&[(v, 2.0 / 3.0), (e0, 1.0 / 3.0)])
    } else {
        along(theta * f as f32)
    };
    let em = if f + 1 == k {
        SparsePoint::of(&[(v, 2.0 / 3.0), (ek, 1.0 / 3.0)])
    } else {
        along(theta * (f + 1) as f32)
    };
    (p, ep, em)
}

/// Everything known about one corner of the patch face.
struct Corner {
    span: CornerSpan,
    /// A regular corner: two faces on a boundary/crease or four interior.
    regular: bool,
    /// `cos` of the angle between consecutive edges of the span, as used by
    /// the face-point blend (zero for regular corners).
    cos_angle: f32,
    /// Does the edge point `E+` (`E-`) lie on the span's leading (trailing)
    /// singular edge?
    ep_on_boundary: bool,
    em_on_boundary: bool,
    p: SparsePoint,
    ep: SparsePoint,
    em: SparsePoint,
}

/// The face point `F+` (`plus`) or `F-` of corner `near`, blending toward
/// the adjacent corner `far` across the face's edge between them
/// (`computeIrregularFacePoint`):
/// `f = [c_far P + (3 - 2 c_near - c_far) E_near + 2 c_near E_far] / 3 + R`,
/// where `R` is the transversal component taken from the two pairs of
/// ring points on either side of that edge in the near corner's span.
fn irregular_face_point(level: &Level, near: &Corner, far: &Corner, plus: bool) -> SparsePoint {
    let (e_near, e_far, edge, sign) = if plus {
        (&near.ep, &far.em, near.span.face_in_span, 1.0)
    } else {
        (&near.em, &far.ep, near.span.face_in_span + 1, -1.0)
    };
    let cos_near = near.cos_angle;
    let cos_far = far.cos_angle;
    let mut f = near.p.scaled(cos_far / 3.0);
    f.add_scaled(e_near, (3.0 - 2.0 * cos_near - cos_far) / 3.0);
    f.add_scaled(e_far, 2.0 * cos_near / 3.0);

    let span = &near.span;
    let valence = span.valence();
    let prev = (edge + valence - 1) % valence;
    let next = (edge + 1) % valence;
    debug_assert!(prev < span.num_faces() && edge < span.num_faces());
    f.add(span.edge_end(level, prev), -sign / 9.0);
    f.add(span.diagonal(level, prev), -sign / 18.0);
    f.add(span.diagonal(level, edge), sign / 18.0);
    f.add(span.edge_end(level, next), sign / 9.0);
    f
}

/// Build the 20 Gregory control-point stencils for `face`
/// (`GregoryConverter::Convert`), or `None` when a corner neighborhood
/// contains non-quad faces or lies on an unsharpened boundary.
pub(crate) fn build(level: &Level, face: usize) -> Option<GregoryPoints> {
    let fv = level.face_vertices(face);
    if fv.len() != 4 {
        return None;
    }

    // Corner points P and edge points E+/E- first: the face points depend
    // on the edge points of adjacent corners.
    let mut corners = Vec::with_capacity(4);
    for k in 0..4 {
        let v = fv[k];
        let next = fv[(k + 1) % 4];
        let prev = fv[(k + 3) % 4];

        // Only infinitely sharp features shape the cap: semi-sharp creases
        // still unresolved at the isolation level decay to smooth, so their
        // vertices are treated as smooth (or as darts) here rather than as
        // creases or corners that would pin the cap to a moving vertex.
        let vertex_sharpness = level.vertex_sharpness(v as usize);
        let inf_vertex = Crease::is_infinite(vertex_sharpness);
        let inf_edges = level
            .vertex_edges(v as usize)
            .iter()
            .filter(|&&e| Crease::is_infinite(level.edge_sharpness(e as usize)))
            .count();
        let inf_rule = vertex_rule(if inf_vertex { vertex_sharpness } else { 0.0 }, inf_edges);
        // Infinitely sharp edges partition the ring — except at a dart,
        // whose single crease leaves the (smooth) limit neighborhood whole.
        let split_at_inf_sharp = inf_edges > 0 && inf_rule != Rule::Dart;
        let span = corner_span(level, face, k, split_at_inf_sharp)?;

        // A smooth boundary (`VtxBoundaryInterpolation::None`) follows the
        // smooth vertex rules, not the crease rules of the boundary points
        // below; such faces keep the bilinear fallback.
        if span.boundary {
            let smooth_boundary = |&e: &Index| {
                level.is_edge_boundary(e as usize)
                    && !Crease::is_infinite(level.edge_sharpness(e as usize))
            };
            if [span.edges[0], span.edges[span.valence() - 1]]
                .iter()
                .any(smooth_boundary)
            {
                return None;
            }
        }

        // A sharp corner interpolates its vertex: an infinitely sharp vertex,
        // or a vertex whose infinitely sharp edges do not form a crease.
        let sharp = if split_at_inf_sharp {
            inf_rule != Rule::Crease
        } else {
            inf_vertex
        };

        let num_faces = span.num_faces();
        let boundary = span.boundary;
        let regular = !sharp && (num_faces << (boundary as usize)) == 4;
        let cos_angle = if regular {
            0.0
        } else {
            let full = if boundary {
                std::f32::consts::PI
            } else {
                std::f32::consts::TAU
            };
            (full / num_faces as f32).cos()
        };
        let f = span.face_in_span;
        let ep_on_boundary = boundary && f == 0;
        let em_on_boundary = boundary && f + 1 == num_faces;

        let (p, ep, em) = if sharp {
            (
                SparsePoint::of(&[(v, 1.0)]),
                SparsePoint::of(&[(v, 2.0 / 3.0), (next, 1.0 / 3.0)]),
                SparsePoint::of(&[(v, 2.0 / 3.0), (prev, 1.0 / 3.0)]),
            )
        } else if !boundary {
            interior_points(level, &span)
        } else if num_faces > 1 {
            boundary_points(level, &span)
        } else {
            // A smooth corner of a single face: the crease rules of its two
            // boundary edges.
            (
                SparsePoint::of(&[(v, 4.0 / 6.0), (next, 1.0 / 6.0), (prev, 1.0 / 6.0)]),
                SparsePoint::of(&[(v, 2.0 / 3.0), (next, 1.0 / 3.0)]),
                SparsePoint::of(&[(v, 2.0 / 3.0), (prev, 1.0 / 3.0)]),
            )
        };

        corners.push(Corner {
            span,
            regular,
            cos_angle,
            ep_on_boundary,
            em_on_boundary,
            p,
            ep,
            em,
        });
    }

    // Face points F+/F-. Between two regular corners they are the interior
    // Bézier points of the B-spline patch (a fixed blend of the face's
    // corners); a face point next to a boundary edge is shared with its
    // partner across the corner, as the surface is only one-sided there.
    let regular_face_point = |k: usize| -> SparsePoint {
        SparsePoint::of(&[
            (fv[k], 4.0 / 9.0),
            (fv[(k + 3) % 4], 2.0 / 9.0),
            (fv[(k + 1) % 4], 2.0 / 9.0),
            (fv[(k + 2) % 4], 1.0 / 9.0),
        ])
    };

    let mut points: GregoryPoints = Default::default();
    for k in 0..4 {
        let next = (k + 1) % 4;
        let prev = (k + 3) % 4;
        let c = &corners[k];

        let mut fp_regular = c.regular && corners[next].regular;
        let mut fm_regular = c.regular && corners[prev].regular;
        let mut fp_copied = false;
        let mut fm_copied = false;
        if c.span.boundary {
            if c.span.num_faces() > 1 {
                if c.ep_on_boundary {
                    fp_regular = fm_regular;
                    fp_copied = !fp_regular;
                }
                if c.em_on_boundary {
                    fm_regular = fp_regular;
                    fm_copied = !fm_regular;
                }
            } else {
                fp_regular = true;
                fm_regular = true;
            }
        }

        let fp = if fp_regular {
            regular_face_point(k)
        } else if fp_copied {
            SparsePoint::default()
        } else {
            irregular_face_point(level, c, &corners[next], true)
        };
        let fm = if fm_regular {
            regular_face_point(k)
        } else if fm_copied {
            SparsePoint::default()
        } else {
            irregular_face_point(level, c, &corners[prev], false)
        };
        let (fp, fm) = if fp_copied {
            (fm.clone(), fm)
        } else if fm_copied {
            (fp.clone(), fp)
        } else {
            (fp, fm)
        };

        points[5 * k] = c.p.clone();
        points[5 * k + 1] = c.ep.clone();
        points[5 * k + 2] = c.em.clone();
        points[5 * k + 3] = fp;
        points[5 * k + 4] = fm;
    }
    Some(points)
}

// ----------------------------------------------------------------------
//  Evaluation
// ----------------------------------------------------------------------

fn bernstein(t: f32) -> ([f32; 4], [f32; 4]) {
    let s = 1.0 - t;
    let basis = [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t];
    let deriv = [
        -3.0 * s * s,
        3.0 * s * s - 6.0 * t * s,
        6.0 * t * s - 3.0 * t * t,
        3.0 * t * t,
    ];
    (basis, deriv)
}

/// Bézier slots mapped directly to a single Gregory point:
/// `(i, j, point index)` with `i` along `s` and `j` along `t`.
const DIRECT_SLOTS: [(usize, usize, usize); 12] = [
    (0, 0, 0),  // P0
    (3, 0, 5),  // P1
    (3, 3, 10), // P2
    (0, 3, 15), // P3
    (1, 0, 1),  // E0+
    (0, 1, 2),  // E0-
    (3, 1, 6),  // E1+
    (2, 0, 7),  // E1-
    (2, 3, 11), // E2+
    (3, 2, 12), // E2-
    (0, 2, 16), // E3+
    (1, 3, 17), // E3-
];

/// Evaluate the Gregory basis at patch-local `(s, t)`: weights (and local
/// `d/ds`, `d/dt` derivative weights) on the 20 control points.
pub(crate) fn evaluate_basis(s: f32, t: f32) -> ([f32; 20], [f32; 20], [f32; 20]) {
    let (bs, dbs) = bernstein(s);
    let (bt, dbt) = bernstein(t);

    let mut w = [0.0f32; 20];
    let mut ws = [0.0f32; 20];
    let mut wt = [0.0f32; 20];

    for &(i, j, point) in &DIRECT_SLOTS {
        w[point] += bs[i] * bt[j];
        ws[point] += dbs[i] * bt[j];
        wt[point] += bs[i] * dbt[j];
    }

    // Interior Bézier points are rational blends of the two face points of
    // their corner: q = (x P + y Q) / (x + y), with x and y affine in
    // (s, t). Their parametric dependence contributes extra derivative
    // terms: d(wP)/du = (x_u - wP (x_u + y_u)) / (x + y).
    struct InteriorSlot {
        i: usize,
        j: usize,
        p: usize, // Gregory point blended by x
        q: usize, // Gregory point blended by y
        x: fn(f32, f32) -> f32,
        y: fn(f32, f32) -> f32,
        x_grad: [f32; 2],
        y_grad: [f32; 2],
    }
    let interior = [
        // b11: (s F0+ + t F0-) / (s + t)
        InteriorSlot {
            i: 1,
            j: 1,
            p: 3,
            q: 4,
            x: |s, _| s,
            y: |_, t| t,
            x_grad: [1.0, 0.0],
            y_grad: [0.0, 1.0],
        },
        // b21: ((1-s) F1- + t F1+) / (1 - s + t)
        InteriorSlot {
            i: 2,
            j: 1,
            p: 9,
            q: 8,
            x: |s, _| 1.0 - s,
            y: |_, t| t,
            x_grad: [-1.0, 0.0],
            y_grad: [0.0, 1.0],
        },
        // b22: ((1-s) F2+ + (1-t) F2-) / (2 - s - t)
        InteriorSlot {
            i: 2,
            j: 2,
            p: 13,
            q: 14,
            x: |s, _| 1.0 - s,
            y: |_, t| 1.0 - t,
            x_grad: [-1.0, 0.0],
            y_grad: [0.0, -1.0],
        },
        // b12: (s F3- + (1-t) F3+) / (1 + s - t)
        InteriorSlot {
            i: 1,
            j: 2,
            p: 19,
            q: 18,
            x: |s, _| s,
            y: |_, t| 1.0 - t,
            x_grad: [1.0, 0.0],
            y_grad: [0.0, -1.0],
        },
    ];

    for slot in &interior {
        let x = (slot.x)(s, t);
        let y = (slot.y)(s, t);
        let d = x + y;
        let b = bs[slot.i] * bt[slot.j];
        let b_ds = dbs[slot.i] * bt[slot.j];
        let b_dt = bs[slot.i] * dbt[slot.j];

        if d < 1e-5 {
            // Exactly at the corner: the Bernstein factor vanishes, so the
            // blend choice is irrelevant; use the average and no rational
            // derivative term.
            w[slot.p] += 0.5 * b;
            w[slot.q] += 0.5 * b;
            ws[slot.p] += 0.5 * b_ds;
            ws[slot.q] += 0.5 * b_ds;
            wt[slot.p] += 0.5 * b_dt;
            wt[slot.q] += 0.5 * b_dt;
            continue;
        }

        let wp = x / d;
        let wq = y / d;
        w[slot.p] += b * wp;
        w[slot.q] += b * wq;

        // s axis (0) then t axis (1).
        let apply = |arr: &mut [f32; 20], bd: f32, axis: usize| {
            let d_sum = slot.x_grad[axis] + slot.y_grad[axis];
            let dwp = (slot.x_grad[axis] - wp * d_sum) / d;
            let dwq = (slot.y_grad[axis] - wq * d_sum) / d;
            arr[slot.p] += bd * wp + b * dwp;
            arr[slot.q] += bd * wq + b * dwq;
        };
        apply(&mut ws, b_ds, 0);
        apply(&mut wt, b_dt, 1);
    }

    (w, ws, wt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::far::{
        PatchTable, PatchTableFactory, PatchType, TopologyDescriptor, TopologyRefiner,
        TopologyRefinerFactory,
    };
    use crate::sdc;

    /// Pseudo-random control positions (simple LCG) for a `w x h` vertex
    /// grid: the equivalences below must hold for any data, not just
    /// smooth samples.
    fn jittered_grid(w: usize, h: usize) -> Vec<[f32; 3]> {
        let mut seed = 0x12345678u32;
        let mut rand = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        (0..w * h)
            .map(|k| {
                [
                    (k % w) as f32 + rand() * 0.4,
                    (k / w) as f32 + rand() * 0.4,
                    rand(),
                ]
            })
            .collect()
    }

    /// Face-vertex lists of a quad grid with `w x h` vertices.
    fn grid_faces(w: u32, h: u32) -> (Vec<usize>, Vec<u32>) {
        let mut verts_per_face = Vec::new();
        let mut face_verts = Vec::new();
        for j in 0..h - 1 {
            for i in 0..w - 1 {
                verts_per_face.push(4usize);
                let v = j * w + i;
                face_verts.extend_from_slice(&[v, v + 1, v + w + 1, v + w]);
            }
        }
        (verts_per_face, face_verts)
    }

    /// THE oracle for the whole Gregory construction: wherever the patch
    /// table extracts a regular B-spline patch (interior, boundary, pinned
    /// corner or infinitely sharp crease), the Gregory patch built on the
    /// same face must reproduce it — positions *and* derivatives — for
    /// arbitrary control data. This pins down the limit-point weights, the
    /// edge factor, the tangent stencils and the face-point formulas at
    /// once. Returns the number of faces compared.
    fn assert_gregory_matches_regular_patches(
        refiner: &TopologyRefiner,
        table: &PatchTable,
        positions: &[[f32; 3]],
    ) -> usize {
        assert_eq!(refiner.max_level(), 0, "oracle expects base-level patches");
        let level = refiner.level(0).inner();
        let samples = [
            (0.0f32, 0.0f32),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.15, 0.85),
            (0.9, 0.2),
            (0.35, 0.05),
            (0.65, 0.45),
        ];
        let mut compared = 0;
        for patch in 0..table.num_patches() {
            if table.patch_type(patch) != PatchType::Regular {
                continue;
            }
            let face = table.patch_face(patch) as usize;
            let param = table.patch_param(patch);
            assert_eq!(param.depth, 0);
            assert_eq!(param.rotation, 0);
            let points = build(level, face).expect("every sharpened quad face builds");

            for &(s, t) in &samples {
                let (w, ws, wt) = evaluate_basis(s, t);
                let mut g = [[0.0f32; 3]; 3];
                for (pt, stencil) in points.iter().enumerate() {
                    for &(cv, sw) in &stencil.0 {
                        for c in 0..3 {
                            g[0][c] += w[pt] * sw * positions[cv as usize][c];
                            g[1][c] += ws[pt] * sw * positions[cv as usize][c];
                            g[2][c] += wt[pt] * sw * positions[cv as usize][c];
                        }
                    }
                }
                let (bp, bdu, bdv) = table.evaluate(patch, s, t, positions);
                for c in 0..3 {
                    assert!(
                        (g[0][c] - bp[c]).abs() < 2e-4,
                        "face {face}: position mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                        g[0],
                        bp
                    );
                    assert!(
                        (g[1][c] - bdu[c]).abs() < 2e-3,
                        "face {face}: du mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                        g[1],
                        bdu
                    );
                    assert!(
                        (g[2][c] - bdv[c]).abs() < 2e-3,
                        "face {face}: dv mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                        g[2],
                        bdv
                    );
                }
            }
            compared += 1;
        }
        compared
    }

    #[test]
    fn gregory_reproduces_bspline_on_interior_and_boundary_faces() {
        // A 6x6-vertex grid with unpinned corners: the 3x3 central faces
        // have fully interior neighborhoods and the border faces regular
        // boundary corners; only the four faces at the grid's smooth
        // (unpinned) corners are irregular and skipped by the oracle.
        let (verts_per_face, face_verts) = grid_faces(6, 6);
        let positions = jittered_grid(6, 6);
        let descriptor = TopologyDescriptor::new(36, &verts_per_face, &face_verts);
        let refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        let table = PatchTableFactory::create(&refiner).unwrap();
        assert_eq!(
            assert_gregory_matches_regular_patches(&refiner, &table, &positions),
            21
        );
    }

    #[test]
    fn gregory_reproduces_bspline_on_boundary_and_corner_faces() {
        // A 5x5-vertex grid with pinned corners: all 16 faces are regular —
        // interior, boundary (valence-3 corners, one edge point along the
        // boundary curve) and pinned corners (sharp corner points).
        let (verts_per_face, face_verts) = grid_faces(5, 5);
        let positions = jittered_grid(5, 5);
        let options = sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
        let descriptor = TopologyDescriptor::new(25, &verts_per_face, &face_verts);
        let refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
        let table = PatchTableFactory::create(&refiner).unwrap();
        assert_eq!(
            assert_gregory_matches_regular_patches(&refiner, &table, &positions),
            16
        );
    }

    #[test]
    fn gregory_reproduces_bspline_along_infinitely_sharp_creases() {
        // A 6x6-vertex grid split by an infinitely sharp crease along its
        // middle row of edges, plus a second crease along a column meeting
        // it: every face is regular (crease vertices are regular crease
        // corners, and the crossing is a regular set of inf-sharp corners),
        // and the crease-adjacent faces are boundary-type B-spline patches.
        let (verts_per_face, face_verts) = grid_faces(6, 6);
        let positions = jittered_grid(6, 6);
        let mut creases = Vec::new();
        for i in 0..5u32 {
            creases.push([18 + i, 19 + i]); // row 3
        }
        for j in 0..5u32 {
            creases.push([2 + 6 * j, 8 + 6 * j]); // column 2
        }
        let weights = vec![sdc::SHARPNESS_INFINITE; creases.len()];
        let options = sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
        let descriptor = TopologyDescriptor::new(36, &verts_per_face, &face_verts)
            .with_creases(&creases, &weights);
        let refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
        let table = PatchTableFactory::create(&refiner).unwrap();
        assert_eq!(
            assert_gregory_matches_regular_patches(&refiner, &table, &positions),
            25
        );
    }

    #[test]
    fn edge_factor_matches_reference_table() {
        // The first entries of OpenSubdiv's `efTable`.
        let expected = [
            (3, 8.128_157_290_637_231e-1),
            (4, 0.5),
            (5, 3.636_440_632_914_28e-1),
            (6, 2.875_137_970_607_708_5e-1),
            (7, 2.386_878_668_585_167_8e-1),
            (12, 1.312_756_841_588_301_7e-1),
            (29, 5.296_209_143_379_613_4e-2),
        ];
        for (valence, value) in expected {
            assert!(
                (edge_factor(valence) as f64 - value).abs() < 1e-7,
                "valence {valence}"
            );
        }
    }

    #[test]
    fn corner_spans_follow_boundaries_and_creases() {
        // A 3x3-face grid with an infinitely sharp crease along the middle
        // column: the central face's corners split the ring of their
        // valence-4 vertices into two-face spans.
        let (verts_per_face, face_verts) = grid_faces(4, 4);
        let creases = [[1u32, 5], [5, 9], [9, 13]];
        let weights = [sdc::SHARPNESS_INFINITE; 3];
        let descriptor = TopologyDescriptor::new(16, &verts_per_face, &face_verts)
            .with_creases(&creases, &weights);
        let refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        let level = refiner.level(0).inner();

        // Face 4 (vertices 5, 6, 10, 9): corner 0 is vertex 5, on the crease.
        let span = corner_span(level, 4, 0, true).unwrap();
        assert!(span.boundary);
        assert_eq!(span.num_faces(), 2);
        assert_eq!(span.edges.len(), 3);
        // Both singular edges are crease edges incident vertex 5.
        assert!(level.edge_vertices(span.edges[0] as usize).contains(&1));
        assert!(level.edge_vertices(span.edges[2] as usize).contains(&9));
        // The patch face is the trailing face of the span (its edge toward
        // the previous corner, 5-9, is the trailing crease edge).
        assert_eq!(span.face_in_span, 1);
        assert_eq!(span.faces[1], 4);

        // Ignoring sharpness, the same corner is a periodic valence-4 ring
        // starting from the edge toward the next corner.
        let ring = corner_span(level, 4, 0, false).unwrap();
        assert!(!ring.boundary);
        assert_eq!(ring.num_faces(), 4);
        assert_eq!(ring.faces[0], 4);
        assert_eq!(ring.face_in_span, 0);
        assert!(level.edge_vertices(ring.edges[0] as usize).contains(&6));
        assert!(level.edge_vertices(ring.edges[1] as usize).contains(&9));

        // Corner 1 of face 0 is vertex 1, a boundary vertex on the crease's
        // end: its span is bounded by a boundary edge and the crease.
        let span = corner_span(level, 0, 1, true).unwrap();
        assert!(span.boundary);
        assert_eq!(span.num_faces(), 1);
        assert_eq!(span.face_in_span, 0);
    }
}
