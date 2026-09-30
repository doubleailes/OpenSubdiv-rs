//! Patches for the Loop scheme: quartic box-spline triangles on regular
//! faces and Gregory triangle end caps on irregular ones.
//!
//! Port of the role of `opensubdiv/far/loopPatchBuilder.cpp` (`LoopLimits`
//! and `GregoryTriConverter`) and of the box-spline and Gregory-triangle
//! bases of `far/patchBasis.cpp`.
//!
//! ## Regular patches
//!
//! Over a triangle whose three corners are *regular* — interior smooth
//! vertices of valence 6, regular boundary/crease vertices (three faces in
//! the span on the patch's side) and pinned corners — the Loop limit
//! surface is exactly the quartic three-direction box spline on the 12
//! control vertices of the face's one-ring neighborhood. The basis
//! polynomials are derived by exact subdivision of the regular lattice
//! (Stam, "Evaluation of Loop Subdivision Surfaces"); the unit tests pin
//! them against uniform refinement and the limit masks.
//!
//! Boundaries and infinitely sharp creases (`useInfSharpPatch`) are handled
//! as for B-spline patches: the control vertices beyond the wall are
//! *phantom* slots whose weights are folded onto real ones at evaluation
//! time. The reflection is the parallelogram rule `a + b - c` across the
//! wall edge `(a, b)` from the opposite vertex `c` (and the point reflection
//! `2v - n` through a pinned corner `v`), which reproduces the crease
//! subdivision rules exactly: the boundary curve is the cubic B-spline of
//! the boundary control vertices, as with the crease limit masks.
//!
//! ## Gregory triangles
//!
//! Every other face at its isolation level is capped with a quartic
//! Gregory triangle (OpenSubdiv's `GREGORY_TRIANGLE` end cap): 18 points —
//! per corner a limit point `P`, two edge points `E±` and two face points
//! `F±`, plus one mid-edge point `M` per edge — that form a quartic Bézier
//! triangle whose three interior points are rational blends of the face
//! points. Corner points interpolate the Loop limit points; edge and
//! mid-edge points are shared with the neighbor across each edge, so
//! adjacent patches join with exact C0 continuity; the face points are
//! chosen from a G1 condition between neighboring patches, giving
//! approximate tangent-plane continuity. On a regular neighborhood the
//! construction degenerates to the exact box-spline patch, which the unit
//! tests verify for arbitrary control data.

use super::gregory::{corner_span, is_edge_singular, vertex_rule, CornerSpan, SparsePoint};
use crate::sdc::{Crease, Rule};
use crate::vtr::Level;
use crate::{Index, INDEX_INVALID};

// ----------------------------------------------------------------------
//  Box-spline basis
// ----------------------------------------------------------------------

/// The monomials `s^i t^j` (as `(i, j)`) of the quartic basis polynomials.
const MONOMIALS: [(u32, u32); 15] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (4, 0),
];

/// Coefficients (times 12) of the 12 box-spline basis polynomials on
/// [`MONOMIALS`], one row per control-vertex slot of [`SLOT_POSITIONS`].
const BOX_SPLINE_COEFFS: [[i8; 15]; 12] = [
    [1, -4, 6, -4, 1, -2, 6, -6, 2, 0, 0, 0, 2, -2, -1],
    [1, -2, 0, 2, -1, 2, -6, 6, -2, 0, 0, 0, -4, 4, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, -1],
    [1, -2, 0, 2, -1, -4, 6, 0, -2, 6, -6, 0, -4, 2, 1],
    [6, 0, -12, 8, -1, 0, -12, 12, -2, -12, 12, 0, 8, -2, -1],
    [1, 2, 0, -4, 2, 4, 6, -12, 4, 6, -6, 0, -4, -2, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
    [1, 2, 0, -4, 2, -2, -6, 0, 4, 0, 6, 0, 2, -2, -1],
    [1, 4, 6, -4, -1, 2, 6, -6, -2, 0, -12, 0, -4, 4, 2],
    [0, 0, 0, 2, -1, 0, 0, 6, -2, 0, 6, 0, 2, -2, -1],
    [0, 0, 0, 2, -1, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
];

/// Evaluate the quartic box-spline basis (and its `d/ds`, `d/dt`) at
/// patch-local `(s, t)` — `s, t >= 0`, `s + t <= 1` — on the 12 control
/// vertices laid out as in [`gather_regular_patch`].
pub(crate) fn box_spline_basis(s: f32, t: f32) -> ([f32; 12], [f32; 12], [f32; 12]) {
    let (s, t) = (f64::from(s), f64::from(t));
    let mut pow_s = [1.0f64; 5];
    let mut pow_t = [1.0f64; 5];
    for k in 1..5 {
        pow_s[k] = pow_s[k - 1] * s;
        pow_t[k] = pow_t[k - 1] * t;
    }
    let mut w = [0.0f32; 12];
    let mut ws = [0.0f32; 12];
    let mut wt = [0.0f32; 12];
    for (slot, coeffs) in BOX_SPLINE_COEFFS.iter().enumerate() {
        let mut value = 0.0f64;
        let mut ds = 0.0f64;
        let mut dt = 0.0f64;
        for (&c, &(i, j)) in coeffs.iter().zip(&MONOMIALS) {
            if c == 0 {
                continue;
            }
            let c = f64::from(c) / 12.0;
            let (i, j) = (i as usize, j as usize);
            value += c * pow_s[i] * pow_t[j];
            if i > 0 {
                ds += c * i as f64 * pow_s[i - 1] * pow_t[j];
            }
            if j > 0 {
                dt += c * j as f64 * pow_s[i] * pow_t[j - 1];
            }
        }
        w[slot] = value as f32;
        ws[slot] = ds as f32;
        wt[slot] = dt as f32;
    }
    (w, ws, wt)
}

// ----------------------------------------------------------------------
//  Regular-patch classification and control-vertex gathering
// ----------------------------------------------------------------------

/// The 12 control-vertex slots of a regular patch on the triangular
/// lattice spanned by the face's edges: the face is `(0,0), (1,0), (0,1)`
/// and the third lattice direction is `(-1, 1)`.
///
/// ```text
///          10  11
///        7   8   9            corners: 4 = c0, 5 = c1, 8 = c2
///      3   4   5   6
///        0   1   2
/// ```
#[allow(dead_code)] // documents the layout; checked by the unit tests
const SLOT_POSITIONS: [(i8, i8); 12] = [
    (0, -1),
    (1, -1),
    (2, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (2, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 2),
    (0, 2),
];

/// The slots of each corner's six-ring, counter-clockwise from the face's
/// next corner.
const RING_SLOTS: [[usize; 6]; 3] = [[5, 8, 7, 3, 0, 1], [8, 4, 1, 2, 6, 9], [4, 5, 9, 11, 10, 7]];

/// The phantom slots beyond each edge of the face when that edge is a
/// wall (boundary, non-manifold edge or infinitely sharp crease).
#[cfg(test)]
const EDGE_WALL_SLOTS: [[usize; 3]; 3] = [[0, 1, 2], [6, 9, 11], [3, 7, 10]];

/// The phantom slots across a wall passing *through* each corner along
/// the lattice line parallel to the opposite edge (the face touches the
/// boundary at that vertex only, as the middle face of its three-face
/// span).
#[cfg(test)]
const VERTEX_WALL_SLOTS: [[usize; 2]; 3] = [[0, 3], [2, 6], [10, 11]];

/// Is corner `corner` of (triangular) `face` regular for box-spline patch
/// extraction (`PatchBuilder::IsPatchRegular`, one corner)? Regularity is
/// decided by the span of faces around the vertex containing the face —
/// the fan bounded by boundary edges and infinitely sharp creases:
///
/// * a smooth vertex must be interior with valence 6;
/// * a vertex on a boundary or infinitely sharp crease must have exactly
///   three faces in its span;
/// * an infinitely sharp corner (a pinned vertex, or three or more
///   infinitely sharp edges) must have a single-face span;
/// * darts and semi-sharp features are never regular.
fn is_corner_regular(level: &Level, face: usize, corner: usize) -> bool {
    let v = level.face_vertices(face)[corner] as usize;
    let edges = level.vertex_edges(v);
    let num_faces = level.vertex_faces(v).len();
    let vertex_sharpness = level.vertex_sharpness(v);

    if Crease::is_semi_sharp(vertex_sharpness)
        || edges
            .iter()
            .any(|&e| Crease::is_semi_sharp(level.edge_sharpness(e as usize)))
    {
        return false;
    }
    let inf_edges = edges
        .iter()
        .filter(|&&e| Crease::is_infinite(level.edge_sharpness(e as usize)))
        .count();
    let boundary = level.is_vertex_boundary(v);
    let span_faces = || corner_span(level, face, corner, true).map(|span| span.num_faces());

    match vertex_rule(vertex_sharpness, inf_edges) {
        Rule::Smooth => !boundary && edges.len() == 6 && num_faces == 6,
        Rule::Dart => false,
        Rule::Crease => {
            if boundary {
                num_faces == 3 && edges.len() == 4
            } else {
                span_faces() == Some(3)
            }
        }
        Rule::Corner => inf_edges > 0 && span_faces() == Some(1),
        Rule::Unknown => false,
    }
}

/// Attempt to classify (triangular) `face` as a regular box-spline patch
/// and gather its 12 control vertices, laid out as in [`SLOT_POSITIONS`];
/// phantom slots beyond a boundary or infinitely sharp crease are
/// `INDEX_INVALID`. Also used by adaptive refinement to decide which faces
/// need isolation.
pub(crate) fn gather_regular_patch(level: &Level, face: usize) -> Option<[Index; 12]> {
    let fv = level.face_vertices(face);
    if fv.len() != 3 {
        return None;
    }
    for corner in 0..3 {
        if !is_corner_regular(level, face, corner) {
            return None;
        }
    }

    let mut cvs = [INDEX_INVALID; 12];
    let mut set = |slot: usize, v: Index| -> Option<()> {
        if cvs[slot] != INDEX_INVALID && cvs[slot] != v {
            return None; // the rings disagree: not a lattice neighborhood
        }
        cvs[slot] = v;
        Some(())
    };
    for (corner, ring) in RING_SLOTS.iter().enumerate() {
        let span = corner_span(level, face, corner, true)?;
        if !span.boundary {
            if span.edges.len() != 6 {
                return None;
            }
            for (m, &slot) in ring.iter().enumerate() {
                set(slot, span.edge_end(level, m))?;
            }
        } else {
            let num_edges = span.edges.len();
            if num_edges > 6 {
                return None;
            }
            for m in 0..num_edges {
                let r = (m + 6 - span.face_in_span) % 6;
                set(ring[r], span.edge_end(level, m))?;
            }
        }
    }

    // Every phantom slot must be explained by a wall — an edge of the face
    // or a lattice line through one of its corners — with the real slots
    // its reflection needs present.
    for slot in 0..12 {
        if cvs[slot] == INDEX_INVALID {
            let rule = phantom_rule(&cvs, slot)?;
            if rule.iter().any(|&(target, _)| cvs[target] == INDEX_INVALID) {
                return None;
            }
        }
    }
    Some(cvs)
}

/// The walls of a regular patch's neighborhood, read off its phantom
/// pattern: per edge of the face (phantom slots `0, 1, 2` beyond edge 0,
/// `6, 9, 11` beyond edge 1 and `3, 7, 10` beyond edge 2), and per corner
/// along the lattice line through it parallel to the opposite edge
/// (slots `0, 3` beyond corner 0, `2, 6` beyond corner 1 and `10, 11`
/// beyond corner 2).
fn walls(cvs: &[Index; 12]) -> ([bool; 3], [bool; 3]) {
    let phantom = |slot: usize| cvs[slot] == INDEX_INVALID;
    let edge = [phantom(1), phantom(9), phantom(7)];
    let vertex = [
        !edge[0] && !edge[2] && phantom(0),
        !edge[0] && !edge[1] && phantom(2),
        !edge[1] && !edge[2] && phantom(10),
    ];
    (edge, vertex)
}

/// The real slots (and factors) a phantom `slot`'s weight is folded onto:
/// beyond a wall edge `(a, b)` the parallelogram reflection `a + b - c` of
/// the vertex `c` opposite the edge; beyond a pinned corner `v` (two wall
/// edges meeting at it) the point reflection `2v - n` of the corner's
/// neighbor `n`. `None` when no wall explains the slot.
fn phantom_rule(cvs: &[Index; 12], slot: usize) -> Option<[(usize, f32); 3]> {
    let (e, v) = walls(cvs);
    let parallelogram = |a: usize, b: usize, c: usize| Some([(a, 1.0), (b, 1.0), (c, -1.0)]);
    let point = |v: usize, n: usize| Some([(v, 2.0), (n, -1.0), (n, 0.0)]);
    match slot {
        // Beyond edge 0 (c0, c1), or the line through c0 or c1.
        1 if e[0] => parallelogram(4, 5, 8),
        0 if e[0] && e[2] => point(4, 8),
        0 if e[0] => parallelogram(3, 4, 7),
        0 if v[0] => parallelogram(4, 1, 5),
        2 if e[0] && e[1] => point(5, 8),
        2 if e[0] => parallelogram(5, 6, 9),
        2 if v[1] => parallelogram(5, 1, 4),
        // Beyond edge 1 (c1, c2), or the line through c1 or c2.
        9 if e[1] => parallelogram(5, 8, 4),
        6 if e[1] && e[0] => point(5, 4),
        6 if e[1] => parallelogram(5, 2, 1),
        6 if v[1] => parallelogram(5, 9, 8),
        11 if e[1] && e[2] => point(8, 4),
        11 if e[1] => parallelogram(8, 10, 7),
        11 if v[2] => parallelogram(8, 9, 5),
        // Beyond edge 2 (c2, c0), or the line through c2 or c0.
        7 if e[2] => parallelogram(4, 8, 5),
        3 if e[2] && e[0] => point(4, 5),
        3 if e[2] => parallelogram(0, 4, 1),
        3 if v[0] => parallelogram(4, 7, 8),
        10 if e[2] && e[1] => point(8, 5),
        10 if e[2] => parallelogram(8, 11, 9),
        10 if v[2] => parallelogram(8, 7, 4),
        _ => None,
    }
}

/// Redistribute the weights of phantom control-vertex slots onto real
/// slots (see [`phantom_rule`]). The reflections reproduce the crease and
/// corner subdivision rules exactly.
pub(crate) fn fold_phantom_weights(cvs: &[Index; 12], weights: &mut [f32; 12]) {
    for slot in 0..12 {
        if cvs[slot] != INDEX_INVALID || weights[slot] == 0.0 {
            continue;
        }
        let w = weights[slot];
        weights[slot] = 0.0;
        let rule = phantom_rule(cvs, slot).expect("gathered patches have valid phantoms");
        for (target, factor) in rule {
            debug_assert!(cvs[target] != INDEX_INVALID);
            weights[target] += factor * w;
        }
    }
}

// ----------------------------------------------------------------------
//  Gregory triangles (LoopLimits / GregoryTriConverter)
// ----------------------------------------------------------------------

/// The 18 Gregory triangle control points: per corner `[P, E+, E-, F+, F-]`
/// (points `5k..5k+5`), then the mid-edge points `M0, M1, M2` of edges
/// `0, 1, 2` (points `15..18`), matching OpenSubdiv's `GREGORY_TRIANGLE`
/// layout.
pub(crate) type GregoryTriPoints = [SparsePoint; 18];

/// Loop subdivision's `beta(n)` (`Sdc::Scheme<SCHEME_LOOP>`).
fn loop_beta(valence: usize) -> f32 {
    let inv = 1.0 / valence as f32;
    let t = 0.375 + 0.25 * (std::f32::consts::TAU * inv).cos();
    (0.625 - t * t) * inv
}

/// The scale of the cosine-weighted ring sum giving the limit tangent
/// along an edge at an interior vertex: `1/3` is the exact box-spline
/// derivative at a regular vertex, and — unlike a scale derived from the
/// valence-dependent subdominant eigenvalue — it also comes closest to
/// the limit surface at extraordinary vertices of every valence (see the
/// module's tests, which measure the caps against deep refinement).
const INTERIOR_TANGENT_SCALE: f32 = 1.0 / 3.0;

/// The scale of the across-boundary tangent eigenvector at a crease
/// vertex: `2/3` is exact for a regular boundary vertex (three faces) and
/// likewise closest to the limit surface for other face counts.
const BOUNDARY_TANGENT_SCALE: f32 = 2.0 / 3.0;

/// The corner point `P`, edge points `E+` / `E-` and the vertex's child
/// position under one subdivision step, for one corner of the face.
struct CornerPoints {
    p: SparsePoint,
    ep: SparsePoint,
    em: SparsePoint,
    child: SparsePoint,
}

/// Corner point and edge points of a smooth interior corner
/// (`LoopLimits::ComputeInteriorPointWeights`): the Loop limit point of the
/// vertex, offset by a quarter of its limit tangent along each of the
/// face's two edges — the first interior Bézier points of the quartic edge
/// curves.
fn interior_points(level: &Level, span: &CornerSpan) -> CornerPoints {
    let n = span.edges.len();
    let nf = n as f32;
    let beta = loop_beta(n);
    let omega = 3.0 / (8.0 * beta);
    let e_weight = 1.0 / (omega + nf);
    let mut p = SparsePoint::of(&[(span.vertex, omega * e_weight)]);
    let mut child = SparsePoint::of(&[(span.vertex, 1.0 - nf * beta)]);
    for i in 0..n {
        p.add(span.edge_end(level, i), e_weight);
        child.add(span.edge_end(level, i), beta);
    }

    let theta = std::f32::consts::TAU / nf;
    let tangent = |j: usize| -> SparsePoint {
        let mut t = SparsePoint::default();
        for i in 0..n {
            t.add(
                span.edge_end(level, i),
                INTERIOR_TANGENT_SCALE * ((i as f32 - j as f32) * theta).cos(),
            );
        }
        t
    };
    let mut ep = p.clone();
    ep.add_scaled(&tangent(0), 0.25);
    let mut em = p.clone();
    em.add_scaled(&tangent(1), 0.25);
    CornerPoints { p, ep, em, child }
}

/// Corner point and edge points of a smooth corner on a boundary or
/// infinitely sharp crease (`LoopLimits::ComputeBoundaryPointWeights`) with
/// `k` faces in its span. The limit point and the tangent along the
/// boundary come from the cubic B-spline boundary curve; the tangent across
/// it is the left eigenvector of the crease-vertex subdivision matrix for
/// its subdominant eigenvalue; an edge point at angle `a` from the leading
/// boundary edge is offset along `cos(a)·t_boundary + sin(a)·t_interior`.
fn boundary_points(level: &Level, span: &CornerSpan) -> CornerPoints {
    let k = span.num_faces();
    debug_assert!(span.boundary && k >= 1);
    let v = span.vertex;
    let e0 = span.edge_end(level, 0);
    let ek = span.edge_end(level, k);

    let p = SparsePoint::of(&[(v, 2.0 / 3.0), (e0, 1.0 / 6.0), (ek, 1.0 / 6.0)]);
    let child = SparsePoint::of(&[(v, 0.75), (e0, 0.125), (ek, 0.125)]);
    let t_boundary = SparsePoint::of(&[(e0, 0.5), (ek, -0.5)]);

    let kf = k as f32;
    let theta = std::f32::consts::PI / kf;
    let c = theta.cos();
    let s = theta.sin();
    // Interior ring weights sin(i·theta) sum to cot(theta / 2).
    let ring_sum = (1.0 + c) / s;
    let w_v = (3.0 * (1.0 - 2.0 * c) * ring_sum - 8.0 * s) / (5.0 + 8.0 * c - 4.0 * c * c);
    let w_0 = -(3.0 - 2.0 * c) / 8.0 * w_v - 3.0 / 8.0 * ring_sum;
    let scale = BOUNDARY_TANGENT_SCALE;
    let mut t_interior = SparsePoint::of(&[(v, scale * w_v), (e0, scale * w_0), (ek, scale * w_0)]);
    for i in 1..k {
        t_interior.add(span.edge_end(level, i), scale * (theta * i as f32).sin());
    }

    let along = |angle: f32| -> SparsePoint {
        let mut e = p.clone();
        e.add_scaled(&t_boundary, 0.25 * angle.cos());
        e.add_scaled(&t_interior, 0.25 * angle.sin());
        e
    };
    let f = span.face_in_span;
    let ep = along(theta * f as f32);
    let em = along(theta * (f + 1) as f32);
    CornerPoints { p, ep, em, child }
}

/// Everything known about one corner of the patch face.
struct Corner {
    span: CornerSpan,
    /// `cos` of the angle between consecutive edges of the span.
    cos_angle: f32,
    points: CornerPoints,
}

/// The span edge index of ring position `r` (counter-clockwise from the
/// face's next corner) — only valid for positions inside the span.
fn wrap(span: &CornerSpan, r: isize) -> usize {
    if span.boundary {
        (r + span.face_in_span as isize) as usize
    } else {
        r.rem_euclid(span.edges.len() as isize) as usize
    }
}

impl Corner {
    /// Ring vertex `r` of the corner's span, counter-clockwise from the
    /// face's next corner (`r = 0`). Beyond the walls of a bounded span the
    /// ring continues with the parallelogram reflections of the real ring
    /// across the wall edges, as the regular-lattice phantoms do.
    fn ring(&self, level: &Level, r: isize) -> SparsePoint {
        let span = &self.span;
        let v = span.vertex;
        let e = |i: usize| span.edge_end(level, i);
        if !span.boundary {
            let n = span.edges.len() as isize;
            return SparsePoint::of(&[(e(r.rem_euclid(n) as usize), 1.0)]);
        }
        let k = span.num_faces() as isize;
        let r = r + span.face_in_span as isize;
        if (0..=k).contains(&r) {
            SparsePoint::of(&[(e(r as usize), 1.0)])
        } else if r == -1 || r == k + 2 {
            SparsePoint::of(&[(v, 1.0), (e(0), 1.0), (e(1), -1.0)])
        } else {
            debug_assert!(r == -2 || r == k + 1);
            SparsePoint::of(&[(v, 1.0), (e(k as usize), 1.0), (e(k as usize - 1), -1.0)])
        }
    }
}

/// The face point `F+` (`plus`) or `F-` of corner `near`, adjacent to the
/// face's edge toward corner `far` whose mid-edge point is `mid`. From the
/// G1 condition along the edge between this patch and its neighbor
/// (`D_left + D_right = 2 (c_near (1 - s) + c_far s) C'(s)` matched on the
/// Bézier coefficients next to the corner), split symmetrically between
/// the two patches plus the transversal component of the near corner's
/// ring, which the regular box-spline patch determines.
fn face_point(
    level: &Level,
    near: &Corner,
    far: &Corner,
    mid: &SparsePoint,
    plus: bool,
) -> SparsePoint {
    let (e_near, sign) = if plus {
        (&near.points.ep, 1.0)
    } else {
        (&near.points.em, -1.0)
    };
    let (c_near, c_far) = (near.cos_angle, far.cos_angle);

    // f = E + c_near (M - E) + (c_far - c_near) (E - P) / 3 + R
    let mut f = e_near.scaled(1.0 - c_near + (c_far - c_near) / 3.0);
    f.add_scaled(mid, c_near);
    f.add_scaled(&near.points.p, -(c_far - c_near) / 3.0);

    // Ring index of the edge: 0 for E+ (toward the next corner), 1 for E-.
    let m: isize = if plus { 0 } else { 1 };
    f.add_scaled(&near.ring(level, m + 1), sign * 5.0 / 48.0);
    f.add_scaled(&near.ring(level, m - 1), -sign * 5.0 / 48.0);
    f.add_scaled(&near.ring(level, m + 2), sign / 48.0);
    f.add_scaled(&near.ring(level, m - 2), -sign / 48.0);
    f
}

/// Build the 18 Gregory triangle control-point stencils for `face`
/// (`GregoryTriConverter::Convert`), or `None` when a corner neighborhood
/// contains non-triangular faces or lies on an unsharpened boundary.
pub(crate) fn build(level: &Level, face: usize) -> Option<Box<GregoryTriPoints>> {
    let fv = level.face_vertices(face);
    if fv.len() != 3 {
        return None;
    }
    let fe = level.face_edges(face);

    let mut corners = Vec::with_capacity(3);
    for k in 0..3 {
        let v = fv[k];
        let next = fv[(k + 1) % 3];
        let prev = fv[(k + 2) % 3];

        // Only infinitely sharp features shape the cap: semi-sharp creases
        // still unresolved at the isolation level decay to smooth, so their
        // vertices are treated as smooth (or as darts) here.
        let vertex_sharpness = level.vertex_sharpness(v as usize);
        let inf_vertex = Crease::is_infinite(vertex_sharpness);
        let inf_edges = level
            .vertex_edges(v as usize)
            .iter()
            .filter(|&&e| Crease::is_infinite(level.edge_sharpness(e as usize)))
            .count();
        let inf_rule = vertex_rule(if inf_vertex { vertex_sharpness } else { 0.0 }, inf_edges);
        let split_at_inf_sharp = inf_edges > 0 && inf_rule != Rule::Dart;
        let span = corner_span(level, face, k, split_at_inf_sharp)?;

        // A smooth boundary (`VtxBoundaryInterpolation::None`) follows the
        // smooth vertex rules, not the crease rules of the boundary points
        // below; such faces keep the linear fallback.
        if span.boundary {
            let smooth_boundary = |&e: &Index| {
                level.is_edge_boundary(e as usize)
                    && !Crease::is_infinite(level.edge_sharpness(e as usize))
            };
            if [span.edges[0], span.edges[span.edges.len() - 1]]
                .iter()
                .any(smooth_boundary)
            {
                return None;
            }
        }

        let sharp = if split_at_inf_sharp {
            inf_rule != Rule::Crease
        } else {
            inf_vertex
        };
        let full = if span.boundary {
            std::f32::consts::PI
        } else {
            std::f32::consts::TAU
        };
        let cos_angle = (full / span.num_faces() as f32).cos();

        let points = if sharp {
            CornerPoints {
                p: SparsePoint::of(&[(v, 1.0)]),
                ep: SparsePoint::of(&[(v, 0.75), (next, 0.25)]),
                em: SparsePoint::of(&[(v, 0.75), (prev, 0.25)]),
                child: SparsePoint::of(&[(v, 1.0)]),
            }
        } else if !span.boundary {
            interior_points(level, &span)
        } else {
            boundary_points(level, &span)
        };
        corners.push(Corner {
            span,
            cos_angle,
            points,
        });
    }

    // Mid-edge points: the middle Bézier point of each quartic edge curve.
    // Across a wall the curve is the (degree-elevated) cubic B-spline of
    // the wall's vertices, whose middle point is `(a + b) / 2`. Elsewhere
    // the curve is made to interpolate the limit surface at the edge's
    // midpoint — the limit of the edge's child vertex, a regular vertex
    // whose ring is the children of the edge's two ends and of the four
    // edges around it — which on a regular neighborhood is exactly the
    // box spline's `(a + b) / 3 + (c + d) / 6`.
    let mut mids: [SparsePoint; 3] = Default::default();
    for (edge, mid) in mids.iter_mut().enumerate() {
        let a = fv[edge];
        let b = fv[(edge + 1) % 3];
        if is_edge_singular(level, fe[edge], true) {
            *mid = SparsePoint::of(&[(a, 0.5), (b, 0.5)]);
            continue;
        }
        let ca = &corners[edge];
        let cb = &corners[(edge + 1) % 3];
        // In corner `a`'s ring (counter-clockwise from `b`) the face's
        // third vertex `c` is at index 1 and the vertex `d` opposite the
        // edge in the face across it at index -1; in `b`'s ring `c` is at
        // 0, `a` at 1 and `d` at 2.
        let mut mid_edge = SparsePoint::of(&[(a, 0.375), (b, 0.375)]);
        mid_edge.add_scaled(&ca.ring(level, 1), 0.125);
        mid_edge.add_scaled(&ca.ring(level, -1), 0.125);
        // The child of the edge from `corner` to its ring vertex `near`,
        // whose opposite vertices are the ring vertices on either side of
        // it (`far` and its mirror).
        let side = |corner: &Corner, near: isize, far: isize| -> SparsePoint {
            let e = corner.span.edges[wrap(&corner.span, near)];
            let mut child = SparsePoint::of(&[(corner.span.vertex, 0.5)]);
            if is_edge_singular(level, e, true) {
                child.add_scaled(&corner.ring(level, near), 0.5);
            } else {
                child.add(corner.span.vertex, -0.125);
                child.add_scaled(&corner.ring(level, near), 0.375);
                child.add_scaled(&corner.ring(level, far), 0.125);
                child.add_scaled(&corner.ring(level, 2 * near - far), 0.125);
            }
            child
        };
        let mut limit = mid_edge.scaled(0.5);
        for neighbor in [
            &ca.points.child,
            &cb.points.child,
            &side(ca, 1, 0),  // (a, c)
            &side(cb, 0, 1),  // (b, c)
            &side(cb, 2, 1),  // (b, d)
            &side(ca, -1, 0), // (a, d)
        ] {
            limit.add_scaled(neighbor, 1.0 / 12.0);
        }
        // The quartic Bézier curve through P_a, E_a+, M, E_b-, P_b takes
        // (P_a + 4 E_a+ + 6 M + 4 E_b- + P_b) / 16 at its midpoint.
        let mut m = limit.scaled(16.0 / 6.0);
        m.add_scaled(&ca.points.p, -1.0 / 6.0);
        m.add_scaled(&cb.points.p, -1.0 / 6.0);
        m.add_scaled(&ca.points.ep, -4.0 / 6.0);
        m.add_scaled(&cb.points.em, -4.0 / 6.0);
        *mid = m;
    }

    let mut points: GregoryTriPoints = Default::default();
    for k in 0..3 {
        let next = (k + 1) % 3;
        let prev = (k + 2) % 3;
        let c = &corners[k];
        let span = &c.span;

        // A face point next to a wall edge is shared with its partner
        // across the corner, as the surface is only one-sided there; at a
        // single-face corner both are the regular lattice's interior point
        // with the phantoms folded in.
        let (fp, fm) = if span.boundary && span.num_faces() == 1 {
            let f = SparsePoint::of(&[(fv[k], 0.5), (fv[next], 0.25), (fv[prev], 0.25)]);
            (f.clone(), f)
        } else {
            let ep_on_wall = span.boundary && span.face_in_span == 0;
            let em_on_wall = span.boundary && span.face_in_span + 1 == span.num_faces();
            let fp = (!ep_on_wall).then(|| face_point(level, c, &corners[next], &mids[k], true));
            let fm =
                (!em_on_wall).then(|| face_point(level, c, &corners[prev], &mids[prev], false));
            match (fp, fm) {
                (Some(fp), Some(fm)) => (fp, fm),
                (Some(fp), None) => (fp.clone(), fp),
                (None, Some(fm)) => (fm.clone(), fm),
                (None, None) => unreachable!("a span with two walls has a single face"),
            }
        };

        points[5 * k] = c.points.p.clone();
        points[5 * k + 1] = c.points.ep.clone();
        points[5 * k + 2] = c.points.em.clone();
        points[5 * k + 3] = fp;
        points[5 * k + 4] = fm;
    }
    points[15..18].clone_from_slice(&mids);
    Some(Box::new(points))
}

// ----------------------------------------------------------------------
//  Gregory triangle evaluation
// ----------------------------------------------------------------------

/// Quartic Bézier triangle slots `(i, j, k)` (`i + j + k = 4`, weights on
/// corners 0, 1, 2) mapped directly to a single Gregory point.
const DIRECT_SLOTS: [((usize, usize, usize), usize); 12] = [
    ((4, 0, 0), 0),  // P0
    ((3, 1, 0), 1),  // E0+
    ((3, 0, 1), 2),  // E0-
    ((0, 4, 0), 5),  // P1
    ((0, 3, 1), 6),  // E1+
    ((1, 3, 0), 7),  // E1-
    ((0, 0, 4), 10), // P2
    ((1, 0, 3), 11), // E2+
    ((0, 1, 3), 12), // E2-
    ((2, 2, 0), 15), // M0
    ((0, 2, 2), 16), // M1
    ((2, 0, 2), 17), // M2
];

/// Bernstein polynomial `B^4_ijk` at barycentrics `(u, s, t)` with its
/// derivatives with respect to `s` and `t` (`u = 1 - s - t`).
fn bernstein_tri(i: usize, j: usize, k: usize, u: f32, s: f32, t: f32) -> (f32, f32, f32) {
    const FACT: [f32; 5] = [1.0, 1.0, 2.0, 6.0, 24.0];
    let c = 24.0 / (FACT[i] * FACT[j] * FACT[k]);
    let p = |x: f32, n: usize| x.powi(n as i32);
    let dp = |x: f32, n: usize| if n == 0 { 0.0 } else { n as f32 * p(x, n - 1) };
    let value = c * p(u, i) * p(s, j) * p(t, k);
    let ds = c * (-dp(u, i) * p(s, j) * p(t, k) + p(u, i) * dp(s, j) * p(t, k));
    let dt = c * (-dp(u, i) * p(s, j) * p(t, k) + p(u, i) * p(s, j) * dp(t, k));
    (value, ds, dt)
}

/// Evaluate the Gregory triangle basis at patch-local `(s, t)`: weights
/// (and local `d/ds`, `d/dt` derivative weights) on the 18 control points.
pub(crate) fn evaluate_basis(s: f32, t: f32) -> ([f32; 18], [f32; 18], [f32; 18]) {
    let u = 1.0 - s - t;
    let mut w = [0.0f32; 18];
    let mut ws = [0.0f32; 18];
    let mut wt = [0.0f32; 18];

    for &((i, j, k), point) in &DIRECT_SLOTS {
        let (b, bs, bt) = bernstein_tri(i, j, k, u, s, t);
        w[point] += b;
        ws[point] += bs;
        wt[point] += bt;
    }

    // The interior Bézier point near corner `c` is the rational blend
    // `(x F+ + y F-) / (x + y)` of its face points, `x` the barycentric of
    // the next corner and `y` that of the previous one, so that each face
    // point takes over on its own edge. Their parametric dependence adds
    // the derivative terms `d(x / (x + y)) = (x' - (x / (x + y)) (x' + y'))
    // / (x + y)`.
    struct InteriorSlot {
        ijk: (usize, usize, usize),
        p: usize,
        q: usize,
        x: f32,
        y: f32,
        x_grad: [f32; 2],
        y_grad: [f32; 2],
    }
    let interior = [
        InteriorSlot {
            ijk: (2, 1, 1),
            p: 3,
            q: 4,
            x: s,
            y: t,
            x_grad: [1.0, 0.0],
            y_grad: [0.0, 1.0],
        },
        InteriorSlot {
            ijk: (1, 2, 1),
            p: 8,
            q: 9,
            x: t,
            y: u,
            x_grad: [0.0, 1.0],
            y_grad: [-1.0, -1.0],
        },
        InteriorSlot {
            ijk: (1, 1, 2),
            p: 13,
            q: 14,
            x: u,
            y: s,
            x_grad: [-1.0, -1.0],
            y_grad: [1.0, 0.0],
        },
    ];
    for slot in &interior {
        let (i, j, k) = slot.ijk;
        let (b, b_ds, b_dt) = bernstein_tri(i, j, k, u, s, t);
        let d = slot.x + slot.y;
        if d < 1e-5 {
            // Exactly at the corner the Bernstein factor vanishes; the
            // blend choice is irrelevant.
            w[slot.p] += 0.5 * b;
            w[slot.q] += 0.5 * b;
            ws[slot.p] += 0.5 * b_ds;
            ws[slot.q] += 0.5 * b_ds;
            wt[slot.p] += 0.5 * b_dt;
            wt[slot.q] += 0.5 * b_dt;
            continue;
        }
        let wp = slot.x / d;
        let wq = slot.y / d;
        w[slot.p] += b * wp;
        w[slot.q] += b * wq;
        let apply = |arr: &mut [f32; 18], bd: f32, axis: usize| {
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

    #[test]
    fn box_spline_basis_is_a_partition_of_unity_with_limit_masks() {
        for &(s, t) in &[(0.0f32, 0.0f32), (0.3, 0.2), (0.5, 0.5), (0.1, 0.85)] {
            let (w, ws, wt) = box_spline_basis(s, t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(ws.iter().sum::<f32>().abs() < 1e-6);
            assert!(wt.iter().sum::<f32>().abs() < 1e-6);
        }
        // At the corner the basis is the regular Loop limit mask: 1/2 on
        // the vertex and 1/12 on its six neighbors.
        let (w, ws, _) = box_spline_basis(0.0, 0.0);
        assert!((w[4] - 0.5).abs() < 1e-6);
        for &slot in &RING_SLOTS[0] {
            assert!((w[slot] - 1.0 / 12.0).abs() < 1e-6, "slot {slot}");
        }
        // And the derivative along the edge is the cosine tangent mask,
        // scaled by 1/3.
        for (m, &slot) in RING_SLOTS[0].iter().enumerate() {
            let expected = (m as f32 * std::f32::consts::FRAC_PI_3).cos() / 3.0;
            assert!((ws[slot] - expected).abs() < 1e-6, "slot {slot}");
        }
    }

    #[test]
    fn slot_layout_is_consistent() {
        // Every ring slot sits at unit lattice distance from its corner,
        // in counter-clockwise order starting from the face's next corner
        // (two lattice directions further around at each corner).
        let dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];
        for corner in 0..3 {
            let (cx, cy) = SLOT_POSITIONS[[4, 5, 8][corner]];
            for (m, &slot) in RING_SLOTS[corner].iter().enumerate() {
                let (x, y) = SLOT_POSITIONS[slot];
                let dir = dirs[(m + 2 * corner) % 6];
                assert_eq!((x - cx, y - cy), dir, "corner {corner} ring {m}");
            }
        }
        // Edge wall slots are exactly those beyond each edge's line, and
        // vertex wall slots those beyond the line through the corner
        // parallel to the opposite edge.
        for (slot, &(x, y)) in SLOT_POSITIONS.iter().enumerate() {
            let beyond_edge = [y < 0, x + y > 1, x < 0];
            for (edge, slots) in EDGE_WALL_SLOTS.iter().enumerate() {
                assert_eq!(
                    slots.contains(&slot),
                    beyond_edge[edge],
                    "edge {edge} slot {slot}"
                );
            }
            let beyond_corner = [x + y < 0, x > 1, y > 1];
            for (corner, slots) in VERTEX_WALL_SLOTS.iter().enumerate() {
                assert_eq!(
                    slots.contains(&slot),
                    beyond_corner[corner],
                    "corner {corner} slot {slot}"
                );
            }
        }
        // Every phantom pattern of a regular neighborhood folds onto real
        // slots: each wall alone, pairs of edge walls (pinned corners) and
        // all three edges (an isolated pinned triangle).
        let patterns: Vec<Vec<usize>> = vec![
            EDGE_WALL_SLOTS[0].to_vec(),
            EDGE_WALL_SLOTS[1].to_vec(),
            EDGE_WALL_SLOTS[2].to_vec(),
            VERTEX_WALL_SLOTS[0].to_vec(),
            VERTEX_WALL_SLOTS[1].to_vec(),
            VERTEX_WALL_SLOTS[2].to_vec(),
            [EDGE_WALL_SLOTS[0], EDGE_WALL_SLOTS[1]].concat(),
            [EDGE_WALL_SLOTS[1], EDGE_WALL_SLOTS[2]].concat(),
            [EDGE_WALL_SLOTS[2], EDGE_WALL_SLOTS[0]].concat(),
            EDGE_WALL_SLOTS.concat(),
            [&VERTEX_WALL_SLOTS[0][..], &EDGE_WALL_SLOTS[1][..]].concat(),
        ];
        for pattern in patterns {
            let mut cvs = [0 as Index; 12];
            for &slot in &pattern {
                cvs[slot] = INDEX_INVALID;
            }
            for &slot in &pattern {
                let rule = phantom_rule(&cvs, slot).unwrap_or_else(|| panic!("{pattern:?} {slot}"));
                assert!(
                    rule.iter().all(|&(t, _)| cvs[t] != INDEX_INVALID),
                    "{pattern:?} {slot}"
                );
                assert!((rule.iter().map(|&(_, f)| f).sum::<f32>() - 1.0).abs() < 1e-6);
            }
            let mut weights = [1.0f32; 12];
            fold_phantom_weights(&cvs, &mut weights);
            assert!((weights.iter().sum::<f32>() - 12.0).abs() < 1e-5);
        }
    }

    #[test]
    fn gregory_tri_basis_is_a_partition_of_unity() {
        for &(s, t) in &[
            (0.0f32, 0.0f32),
            (0.3, 0.2),
            (0.5, 0.5),
            (0.0, 1.0),
            (0.25, 0.7),
        ] {
            let (w, ws, wt) = evaluate_basis(s, t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5, "({s},{t})");
            assert!(ws.iter().sum::<f32>().abs() < 1e-4, "({s},{t})");
            assert!(wt.iter().sum::<f32>().abs() < 1e-4, "({s},{t})");
        }
        // Corners interpolate their corner point.
        assert!((evaluate_basis(0.0, 0.0).0[0] - 1.0).abs() < 1e-6);
        assert!((evaluate_basis(1.0, 0.0).0[5] - 1.0).abs() < 1e-6);
        assert!((evaluate_basis(0.0, 1.0).0[10] - 1.0).abs() < 1e-6);
    }

    use crate::far::{
        PatchMap, PatchTable, PatchTableFactory, PatchType, PrimvarRefiner, TopologyDescriptor,
        TopologyRefiner, TopologyRefinerFactory, UniformOptions,
    };
    use crate::sdc;

    /// Pseudo-random control positions (simple LCG) for a triangulated
    /// `w x h` vertex grid, split along the `(i, j) - (i + 1, j + 1)`
    /// diagonals: the equivalences below must hold for any data.
    fn jittered_tri_grid(w: u32, h: u32) -> (Vec<usize>, Vec<u32>, Vec<[f32; 3]>) {
        let mut seed = 0x12345678u32;
        let mut rand = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let positions = (0..w * h)
            .map(|k| {
                [
                    (k % w) as f32 + rand() * 0.4,
                    (k / w) as f32 + rand() * 0.4,
                    rand(),
                ]
            })
            .collect();
        let mut verts_per_face = Vec::new();
        let mut face_verts = Vec::new();
        for j in 0..h - 1 {
            for i in 0..w - 1 {
                let v = j * w + i;
                verts_per_face.push(3usize);
                face_verts.extend_from_slice(&[v, v + 1, v + w + 1]);
                verts_per_face.push(3usize);
                face_verts.extend_from_slice(&[v, v + w + 1, v + w]);
            }
        }
        (verts_per_face, face_verts, positions)
    }

    /// THE oracle for the Gregory triangle construction: wherever the
    /// patch table extracts a regular box-spline patch (interior,
    /// boundary — by edge or by vertex — and pinned corner), the Gregory
    /// triangle built on the same face must reproduce it, positions *and*
    /// derivatives, for arbitrary control data. This pins down the limit
    /// point, tangent, mid-edge and face-point formulas at once. Returns
    /// the number of faces compared.
    fn assert_gregory_matches_box_spline_patches(
        refiner: &TopologyRefiner,
        table: &PatchTable,
        positions: &[[f32; 3]],
    ) -> usize {
        assert_eq!(refiner.max_level(), 0, "oracle expects base-level patches");
        let level = refiner.level(0).inner();
        let samples = [
            (0.0f32, 0.0f32),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5),
            (0.15, 0.6),
            (0.7, 0.2),
            (0.35, 0.05),
            (0.3, 0.3),
        ];
        let mut compared = 0;
        for patch in 0..table.num_patches() {
            if table.patch_type(patch) != PatchType::Loop {
                continue;
            }
            let face = table.patch_face(patch) as usize;
            assert_eq!(table.patch_param(patch).depth, 0);
            let points = build(level, face).expect("every sharpened triangle builds");
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
                        "face {face}: position mismatch at ({s},{t}): gregory {:?} vs box spline {:?}",
                        g[0],
                        bp
                    );
                    assert!(
                        (g[1][c] - bdu[c]).abs() < 2e-3,
                        "face {face}: du mismatch at ({s},{t}): gregory {:?} vs box spline {:?}",
                        g[1],
                        bdu
                    );
                    assert!(
                        (g[2][c] - bdv[c]).abs() < 2e-3,
                        "face {face}: dv mismatch at ({s},{t}): gregory {:?} vs box spline {:?}",
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
    fn gregory_triangle_reproduces_box_spline_on_regular_faces() {
        // A 6x5-vertex grid with pinned corners: every face but the two
        // at the valence-3 corners is regular — interior, boundary by
        // edge, boundary by vertex and pinned corner alike.
        let (verts_per_face, face_verts, positions) = jittered_tri_grid(6, 5);
        let options = sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
        let descriptor = TopologyDescriptor::new(30, &verts_per_face, &face_verts);
        let refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, options).unwrap();
        let table = PatchTableFactory::create(&refiner).unwrap();
        assert_eq!(
            assert_gregory_matches_box_spline_patches(&refiner, &table, &positions),
            36
        );
    }

    #[test]
    fn gregory_triangle_reproduces_box_spline_along_infinitely_sharp_creases() {
        // The same grid split by an infinitely sharp crease along a row
        // and a column of edges: crease vertices are regular crease
        // corners (or, at the crossing and the ends, regular sharp
        // corners of single-face spans), so every face stays regular but
        // the four at the valence-3 grid corners.
        let (verts_per_face, face_verts, positions) = jittered_tri_grid(7, 7);
        let mut creases = Vec::new();
        for i in 0..6u32 {
            creases.push([21 + i, 22 + i]); // row 3
        }
        for j in 0..6u32 {
            creases.push([3 + 7 * j, 10 + 7 * j]); // column 3
        }
        let weights = vec![sdc::SHARPNESS_INFINITE; creases.len()];
        let options = sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
        let descriptor = TopologyDescriptor::new(49, &verts_per_face, &face_verts)
            .with_creases(&creases, &weights);
        let refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, options).unwrap();
        let table = PatchTableFactory::create(&refiner).unwrap();
        // The creases' four ends are boundary vertices with three sharp
        // edges whose two-face sides are irregular (8 faces); at the
        // crossing the four sharp edges cut the six faces into spans of
        // 1, 2, 1 and 2 faces, the two-face ones irregular (4 faces); and
        // the valence-3 grid corners are irregular (4 faces).
        assert_eq!(
            assert_gregory_matches_box_spline_patches(&refiner, &table, &positions),
            72 - 16
        );
    }

    /// A vertex of valence `n` at the pole of a spherical cap, its ring
    /// vertices regular (valence 6) and a second ring on the boundary.
    fn fan(n: u32) -> (Vec<usize>, Vec<u32>, Vec<[f32; 3]>) {
        let mut faces = Vec::new();
        let r = |i: u32| 1 + i % n;
        let o = |i: u32| 1 + n + i % (2 * n);
        for i in 0..n {
            faces.extend_from_slice(&[0, r(i), r(i + 1)]);
            faces.extend_from_slice(&[r(i), o(2 * i), o(2 * i + 1)]);
            faces.extend_from_slice(&[r(i), o(2 * i + 1), r(i + 1)]);
            faces.extend_from_slice(&[r(i + 1), o(2 * i + 1), o(2 * i + 2)]);
        }
        let mut pos = vec![[0.0, 0.0, 1.0f32]];
        let sph =
            |polar: f32, az: f32| [polar.sin() * az.cos(), polar.sin() * az.sin(), polar.cos()];
        for i in 0..n {
            pos.push(sph(0.7, std::f32::consts::TAU * i as f32 / n as f32));
        }
        for i in 0..2 * n {
            let az = std::f32::consts::TAU * (i as f32 + 0.5) / (2.0 * n as f32);
            pos.push(sph(1.4, az));
        }
        (vec![3; faces.len() / 3], faces, pos)
    }

    /// The same cap cut in half: a boundary vertex with `k` faces at the
    /// pole, regular ring vertices, and a straight boundary.
    fn half_fan(k: u32) -> (Vec<usize>, Vec<u32>, Vec<[f32; 3]>) {
        let mut faces = Vec::new();
        let r = |i: u32| 1 + i;
        let o = |i: u32| 2 + k + i;
        for i in 0..k {
            faces.extend_from_slice(&[0, r(i), r(i + 1)]);
            faces.extend_from_slice(&[r(i), o(2 * i), o(2 * i + 1)]);
            faces.extend_from_slice(&[r(i), o(2 * i + 1), r(i + 1)]);
            faces.extend_from_slice(&[r(i + 1), o(2 * i + 1), o(2 * i + 2)]);
        }
        let mut pos = vec![[0.0, 0.0, 1.0f32]];
        let sph =
            |polar: f32, az: f32| [polar.sin() * az.cos(), polar.sin() * az.sin(), polar.cos()];
        for i in 0..=k {
            pos.push(sph(0.7, std::f32::consts::PI * i as f32 / k as f32));
        }
        for i in 0..=2 * k {
            pos.push(sph(1.4, std::f32::consts::PI * i as f32 / (2.0 * k as f32)));
        }
        (vec![3; faces.len() / 3], faces, pos)
    }

    /// The largest distance between the level-1 caps over the `num_faces`
    /// first faces of the mesh and the surface of a five-times refined
    /// table, sampled on a regular grid of the parametric domain.
    fn cap_error(
        verts_per_face: &[usize],
        face_verts: &[u32],
        positions: &[[f32; 3]],
        num_faces: usize,
    ) -> f32 {
        let descriptor = TopologyDescriptor::new(positions.len(), verts_per_face, face_verts);
        let build = |levels: usize| {
            let mut refiner = TopologyRefinerFactory::create(
                descriptor,
                sdc::SchemeType::Loop,
                sdc::Options::default(),
            )
            .unwrap();
            refiner.refine_uniform(UniformOptions::new(levels));
            let mut controls = positions.to_vec();
            for level in PrimvarRefiner::new(&refiner).interpolate_all(positions) {
                controls.extend(level);
            }
            (PatchTableFactory::create(&refiner).unwrap(), controls)
        };
        let (coarse, coarse_controls) = build(1);
        let (fine, fine_controls) = build(5);
        let (coarse_map, fine_map) = (PatchMap::new(&coarse), PatchMap::new(&fine));
        let mut max = 0.0f32;
        let n = 24;
        for ptex in 0..num_faces {
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                    let a = coarse_map.find_patch(ptex, u, v).unwrap();
                    let b = fine_map.find_patch(ptex, u, v).unwrap();
                    let (pa, _, _) = coarse.evaluate(a, u, v, &coarse_controls);
                    let (pb, _, _) = fine.evaluate(b, u, v, &fine_controls);
                    let d: f32 = (0..3).map(|c| (pa[c] - pb[c]).powi(2)).sum::<f32>().sqrt();
                    max = max.max(d);
                }
            }
        }
        max
    }

    #[test]
    fn caps_stay_close_to_the_limit_surface_at_extraordinary_vertices() {
        // The caps around an extraordinary vertex, built one level down,
        // stay within about 1% of the edge length (0.7 here) of the deeply
        // refined surface, for every valence — the measure that fixed the
        // tangent scales and the mid-edge interpolation.
        for n in [3u32, 4, 5, 7, 8, 10, 12] {
            let (verts_per_face, face_verts, positions) = fan(n);
            let error = cap_error(&verts_per_face, &face_verts, &positions, n as usize);
            assert!(error < 0.011, "valence {n}: cap error {error}");
        }
        // Likewise for boundary vertices with 1 to 6 faces; the regular
        // case (3) and the single-face corner are exact.
        for k in [1u32, 2, 3, 4, 5, 6] {
            let (verts_per_face, face_verts, positions) = half_fan(k);
            let error = cap_error(&verts_per_face, &face_verts, &positions, k as usize);
            let bound = if k == 1 || k == 3 { 1e-4 } else { 0.018 };
            assert!(error < bound, "{k} faces: cap error {error}");
        }
    }
}
