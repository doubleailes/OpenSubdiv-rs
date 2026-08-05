//! Gregory-basis end-cap patches for faces around extraordinary vertices.
//!
//! Port of the role of `opensubdiv/far/gregoryBasis.{h,cpp}` /
//! `endCapGregoryBasisPatchFactory`: where the Catmull-Clark limit surface
//! is not polynomial (faces with an extraordinary corner), the surface is
//! approximated by a *Gregory patch* — 20 control points forming a bicubic
//! Bézier patch whose four interior points are rational blends, giving
//! corner limit-point interpolation, exact C0 boundaries with neighboring
//! patches, and approximate G1 smoothness.
//!
//! The construction follows Loop, Schaefer, Nießner & Castaño,
//! *"Approximating Subdivision Surfaces with Gregory Patches for Hardware
//! Tessellation"* (the basis OpenSubdiv's end caps build on). Each of the
//! 20 points is stored as a sparse stencil on the refined mesh's vertices,
//! so evaluation stays weight-based like every other patch type. On a
//! fully regular neighborhood the construction degenerates to the exact
//! bicubic B-spline patch — a property the unit tests verify, pinning down
//! every coefficient.
//!
//! Only *smooth interior* irregular neighborhoods are handled here;
//! boundary or creased irregular faces keep the bilinear fallback.

use crate::vtr::Level;
use crate::Index;

/// A point expressed as a sparse weighted combination of mesh vertices.
#[derive(Debug, Clone, Default)]
pub(crate) struct SparsePoint(pub(crate) Vec<(Index, f32)>);

impl SparsePoint {
    fn add(&mut self, index: Index, weight: f32) {
        match self.0.iter_mut().find(|(i, _)| *i == index) {
            Some((_, w)) => *w += weight,
            None => self.0.push((index, weight)),
        }
    }

    fn add_scaled(&mut self, other: &SparsePoint, scale: f32) {
        for &(i, w) in &other.0 {
            self.add(i, w * scale);
        }
    }

    fn scaled(&self, scale: f32) -> SparsePoint {
        SparsePoint(self.0.iter().map(|&(i, w)| (i, w * scale)).collect())
    }
}

/// The 20 Gregory control points, ordered per corner as
/// `[P, E+, E-, F+, F-]` (matching OpenSubdiv's Gregory-basis layout).
pub(crate) type GregoryPoints = [SparsePoint; 20];

/// The ordered one-ring around one corner of a quad: `edges[j]` are the
/// incident edges in counter-clockwise order starting from the edge toward
/// the face's next corner, and `faces[j]` sits between `edges[j]` and
/// `edges[j+1]` (`faces[0]` is the patch face itself).
struct CornerRing {
    vertex: Index,
    edges: Vec<Index>,
    faces: Vec<Index>,
}

impl CornerRing {
    fn valence(&self) -> usize {
        self.edges.len()
    }

    /// The same ring traversed clockwise, re-aligned to start at the edge
    /// toward the face's *previous* corner — used to build the "minus" side
    /// tangents and r-vectors by mirror symmetry.
    fn reversed(&self) -> CornerRing {
        let n = self.edges.len();
        let mut edges = Vec::with_capacity(n);
        let mut faces = Vec::with_capacity(n);
        edges.push(self.edges[1]);
        faces.push(self.faces[0]);
        for j in 0..n - 1 {
            edges.push(self.edges[(n - j) % n]);
        }
        for j in 1..n {
            faces.push(self.faces[n - j]);
        }
        CornerRing {
            vertex: self.vertex,
            edges,
            faces,
        }
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

/// Walk the ordered one-ring around corner `corner` of `face`, requiring a
/// smooth interior manifold neighborhood of quads.
fn corner_ring(level: &Level, face: usize, corner: usize) -> Option<CornerRing> {
    let fv = level.face_vertices(face);
    let fe = level.face_edges(face);
    let n = fv.len();
    let vertex = fv[corner];

    if level.is_vertex_boundary(vertex as usize) || level.vertex_sharpness(vertex as usize) > 0.0 {
        return None;
    }
    let valence = level.vertex_edges(vertex as usize).len();
    if level.vertex_faces(vertex as usize).len() != valence {
        return None; // non-manifold
    }

    let e0 = fe[corner]; // edge toward the next corner
    let e1 = fe[(corner + n - 1) % n]; // edge toward the previous corner

    let mut edges = vec![e0];
    let mut faces = vec![face as Index];
    let mut current_face = face as Index;
    let mut current_edge = e1;
    while edges.len() < valence {
        edges.push(current_edge);
        let next_face = other_face_of_edge(level, current_edge, current_face)?;
        if level.face_vertices(next_face as usize).len() != 4 {
            return None;
        }
        faces.push(next_face);
        current_edge = other_edge_at_vertex(level, next_face, vertex, current_edge)?;
        current_face = next_face;
    }
    if current_edge != e0 {
        return None; // failed to close the ring
    }
    for &e in &edges {
        if level.edge_sharpness(e as usize) > 0.0 || level.edge_faces(e as usize).len() != 2 {
            return None;
        }
    }
    Some(CornerRing {
        vertex,
        edges,
        faces,
    })
}

/// Stencil of the midpoint of `edge`.
fn edge_midpoint(level: &Level, edge: Index) -> SparsePoint {
    let [a, b] = level.edge_vertices(edge as usize);
    SparsePoint(vec![(a, 0.5), (b, 0.5)])
}

/// Stencil of the centroid of (quad) `face`.
fn face_centroid(level: &Level, face: Index) -> SparsePoint {
    let fv = level.face_vertices(face as usize);
    SparsePoint(fv.iter().map(|&v| (v, 1.0 / fv.len() as f32)).collect())
}

/// The Catmull-Clark limit point of the ring's vertex:
/// `p = (n-3)/(n+5) v + 4/(n(n+5)) Σ (m_j + c_j)`.
fn limit_point(level: &Level, ring: &CornerRing) -> SparsePoint {
    let n = ring.valence() as f32;
    let mut p = SparsePoint::default();
    p.add(ring.vertex, (n - 3.0) / (n + 5.0));
    let w = 4.0 / (n * (n + 5.0));
    for j in 0..ring.valence() {
        p.add_scaled(&edge_midpoint(level, ring.edges[j]), w);
        p.add_scaled(&face_centroid(level, ring.faces[j]), w);
    }
    p
}

/// The scaled limit tangent along the ring's first edge:
/// `t = (2/n) Σ_j [ (1 - σ cos(π/n)) cos(2πj/n) m_j + 2σ cos((2πj+π)/n) c_j ]`
/// with `σ = 1/√(4 + cos²(π/n))`.
fn limit_tangent(level: &Level, ring: &CornerRing) -> SparsePoint {
    let n = ring.valence() as f32;
    let theta = std::f32::consts::TAU / n;
    let cos_half = (std::f32::consts::PI / n).cos();
    let sigma = 1.0 / (4.0 + cos_half * cos_half).sqrt();
    let m_scale = (2.0 / n) * (1.0 - sigma * cos_half);
    let c_scale = (2.0 / n) * 2.0 * sigma;

    let mut t = SparsePoint::default();
    for j in 0..ring.valence() {
        let jf = j as f32;
        t.add_scaled(
            &edge_midpoint(level, ring.edges[j]),
            m_scale * (jf * theta).cos(),
        );
        t.add_scaled(
            &face_centroid(level, ring.faces[j]),
            c_scale * (jf * theta + 0.5 * theta).cos(),
        );
    }
    t
}

/// The subdominant eigenvalue of Catmull-Clark subdivision at valence `n`
/// (`λ = 1/2` for the regular case `n = 4`).
fn subdominant_eigenvalue(n: usize) -> f32 {
    let theta = std::f32::consts::TAU / n as f32;
    let cos_half = (std::f32::consts::PI / n as f32).cos();
    (5.0 + theta.cos() + cos_half * (18.0 + 2.0 * theta.cos()).sqrt()) / 16.0
}

/// The `r` vector of the ring's first edge:
/// `r = (m_1 - m_{n-1})/3 + 2(c_0 - c_{n-1})/3` — in the regular case this
/// equals `P_t + P_st/3`, the transversal component of the Bézier interior
/// point.
fn r_vector(level: &Level, ring: &CornerRing) -> SparsePoint {
    let n = ring.valence();
    let mut r = SparsePoint::default();
    r.add_scaled(&edge_midpoint(level, ring.edges[1 % n]), 1.0 / 3.0);
    r.add_scaled(&edge_midpoint(level, ring.edges[n - 1]), -1.0 / 3.0);
    r.add_scaled(&face_centroid(level, ring.faces[0]), 2.0 / 3.0);
    r.add_scaled(&face_centroid(level, ring.faces[n - 1]), -2.0 / 3.0);
    r
}

/// Build the 20 Gregory control-point stencils for `face`, or `None` when
/// any corner neighborhood is not smooth-interior-manifold.
pub(crate) fn build(level: &Level, face: usize) -> Option<Box<GregoryPoints>> {
    if level.face_vertices(face).len() != 4 {
        return None;
    }

    struct Corner {
        p: SparsePoint,
        ep: SparsePoint,
        em: SparsePoint,
        rp: SparsePoint,
        rm: SparsePoint,
        cos_theta: f32,
    }

    let mut corners = Vec::with_capacity(4);
    for k in 0..4 {
        let ring = corner_ring(level, face, k)?;
        let reversed = ring.reversed();
        let n = ring.valence();
        let lambda = subdominant_eigenvalue(n);

        let p = limit_point(level, &ring);
        let mut ep = p.clone();
        ep.add_scaled(&limit_tangent(level, &ring), 2.0 * lambda / 3.0);
        let mut em = p.clone();
        em.add_scaled(&limit_tangent(level, &reversed), 2.0 * lambda / 3.0);

        corners.push(Corner {
            p,
            ep,
            em,
            rp: r_vector(level, &ring),
            rm: r_vector(level, &reversed),
            cos_theta: (std::f32::consts::TAU / n as f32).cos(),
        });
    }

    // Interior (face) points, blending across each edge of the patch:
    // f_k^+ = [ c_{k+1} p_k + (3 - 2c_k - c_{k+1}) e_k^+ + 2c_k e_{k+1}^- + r_k^+ ] / 3
    // f_k^- = [ c_{k-1} p_k + (3 - 2c_k - c_{k-1}) e_k^- + 2c_k e_{k-1}^+ + r_k^- ] / 3
    let mut points: GregoryPoints = Default::default();
    for k in 0..4 {
        let next = (k + 1) % 4;
        let prev = (k + 3) % 4;
        let c0 = corners[k].cos_theta;
        let c1 = corners[next].cos_theta;
        let cm = corners[prev].cos_theta;

        let mut fp = corners[k].p.scaled(c1 / 3.0);
        fp.add_scaled(&corners[k].ep, (3.0 - 2.0 * c0 - c1) / 3.0);
        fp.add_scaled(&corners[next].em, 2.0 * c0 / 3.0);
        fp.add_scaled(&corners[k].rp, 1.0 / 3.0);

        let mut fm = corners[k].p.scaled(cm / 3.0);
        fm.add_scaled(&corners[k].em, (3.0 - 2.0 * c0 - cm) / 3.0);
        fm.add_scaled(&corners[prev].ep, 2.0 * c0 / 3.0);
        fm.add_scaled(&corners[k].rm, 1.0 / 3.0);

        points[5 * k] = corners[k].p.clone();
        points[5 * k + 1] = corners[k].ep.clone();
        points[5 * k + 2] = corners[k].em.clone();
        points[5 * k + 3] = fp;
        points[5 * k + 4] = fm;
    }
    Some(Box::new(points))
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
    use crate::far::{PatchTableFactory, PatchType, TopologyDescriptor, TopologyRefinerFactory};
    use crate::sdc;

    /// THE oracle for the whole Gregory construction: on a fully regular
    /// neighborhood the Gregory patch must degenerate to the exact bicubic
    /// B-spline patch — for *arbitrary* control data. This pins down the
    /// limit-point weights, the subdominant eigenvalue, the tangent
    /// stencils, the r-vectors and the face-point formulas all at once
    /// (positions *and* derivatives are compared).
    #[test]
    fn gregory_reproduces_bspline_on_regular_face() {
        // A 6x6-vertex grid; the central face has a fully interior
        // neighborhood.
        let mut verts_per_face = Vec::new();
        let mut face_verts: Vec<u32> = Vec::new();
        for j in 0..5u32 {
            for i in 0..5u32 {
                verts_per_face.push(4usize);
                let v = j * 6 + i;
                face_verts.extend_from_slice(&[v, v + 1, v + 7, v + 6]);
            }
        }
        // Pseudo-random control positions (simple LCG): the equivalence
        // must hold for any data, not just smooth samples.
        let mut seed = 0x12345678u32;
        let mut rand = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let positions: Vec<[f32; 3]> = (0..36)
            .map(|k| {
                [
                    (k % 6) as f32 + rand() * 0.4,
                    (k / 6) as f32 + rand() * 0.4,
                    rand(),
                ]
            })
            .collect();

        let descriptor = TopologyDescriptor::new(36, &verts_per_face, &face_verts);
        let refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();

        // The central face (grid coordinates 2,2) is face 12.
        let face = 12usize;
        let table = PatchTableFactory::create(&refiner).unwrap();
        let patch = (0..table.num_patches())
            .find(|&p| table.patch_face(p) as usize == face)
            .unwrap();
        assert_eq!(table.patch_type(patch), PatchType::Regular);
        // Base level, no refinement: patch-local (s,t) == ptex (u,v).
        let param = table.patch_param(patch);
        assert_eq!(param.depth, 0);
        assert_eq!(param.rotation, 0);

        let level = refiner.level(0).inner();
        let points = build(level, face).expect("interior regular face qualifies");

        let eval_gregory = |s: f32, t: f32| -> [[f32; 3]; 3] {
            let (w, ws, wt) = evaluate_basis(s, t);
            let mut out = [[0.0f32; 3]; 3];
            for (pt, stencil) in points.iter().enumerate() {
                for &(cv, sw) in &stencil.0 {
                    for c in 0..3 {
                        out[0][c] += w[pt] * sw * positions[cv as usize][c];
                        out[1][c] += ws[pt] * sw * positions[cv as usize][c];
                        out[2][c] += wt[pt] * sw * positions[cv as usize][c];
                    }
                }
            }
            out
        };

        let samples = [
            (0.0f32, 0.0f32),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.15, 0.85),
            (0.9, 0.2),
            (0.35, 0.05),
            (0.65, 0.45),
        ];
        for &(s, t) in &samples {
            let (bp, bdu, bdv) = table.evaluate(patch, s, t, &positions);
            let g = eval_gregory(s, t);
            for c in 0..3 {
                assert!(
                    (g[0][c] - bp[c]).abs() < 2e-4,
                    "position mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                    g[0],
                    bp
                );
                assert!(
                    (g[1][c] - bdu[c]).abs() < 2e-3,
                    "du mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                    g[1],
                    bdu
                );
                assert!(
                    (g[2][c] - bdv[c]).abs() < 2e-3,
                    "dv mismatch at ({s},{t}): gregory {:?} vs bspline {:?}",
                    g[2],
                    bdv
                );
            }
        }
    }
}
