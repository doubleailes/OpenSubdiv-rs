// Portions derived from OpenSubdiv (https://github.com/PixarAnimationStudios/OpenSubdiv):
//   osd/patchBasis.h — Copyright 2016-2018 Pixar.
// Licensed under the Tomorrow Open Source Technology License 1.0; see
// LICENSE-OPENSUBDIV at the root of this crate.

//! Patch basis evaluation from a packed [`PatchParam`] (port of
//! `OsdEvaluatePatchBasis` in `opensubdiv/osd/patchBasis.h`).
//!
//! This is the function a GPU patch kernel runs per evaluation point, kept
//! on the CPU as the reference a shader port is checked against: it reads
//! only the descriptor code, the packed parameter and the ptex-face
//! coordinates, and returns weights over the patch's control points in
//! index-buffer order.

use super::types::{patch_descriptor, PatchParam};
use crate::far::gregory;
use crate::far::loop_patch;
use crate::far::patch_table::{bspline_basis, crease_basis_oriented};

/// The basis weights of one patch at one location: the limit point is
/// `Σ point[k] · cv[k]` over the patch's
/// [`num_points`](Self::num_points) control points, and likewise its first
/// derivatives along the ptex face's `u` and `v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchWeights {
    /// The number of control points of the patch; the arrays are zero past
    /// it.
    pub num_points: usize,
    /// Weights of the limit position.
    pub point: [f32; 20],
    /// Weights of the derivative along the ptex face's `u`.
    pub du: [f32; 20],
    /// Weights of the derivative along the ptex face's `v`.
    pub dv: [f32; 20],
}

/// Evaluate the basis of a patch of descriptor code `desc` with packed
/// parameter `param` at ptex-face coordinates `(u, v)`
/// (`OsdEvaluatePatchBasis`), or `None` for a code no
/// [`GpuPatchTable`](super::GpuPatchTable) emits.
///
/// Boundary masks and single creases are read as described on
/// [`PatchParam`]; derivatives are scaled to the ptex face's parameter
/// space, and negated for inverted triangles.
pub fn evaluate_patch_basis(desc: u32, param: &PatchParam, u: f32, v: f32) -> Option<PatchWeights> {
    let num_points = patch_descriptor::num_control_vertices(desc);
    if num_points == 0 {
        return None;
    }
    let triangular = patch_descriptor::is_triangular(desc);
    let (s, t) = if triangular {
        param.normalize_triangle(u, v)
    } else {
        param.normalize(u, v)
    };
    let mut scale = 1.0 / param.param_fraction();
    if triangular && param.is_triangle_rotated() {
        scale = -scale;
    }

    let mut w = PatchWeights {
        num_points,
        point: [0.0; 20],
        du: [0.0; 20],
        dv: [0.0; 20],
    };
    let mut set = |k: usize, p: f32, ds: f32, dt: f32| {
        w.point[k] = p;
        w.du[k] = ds * scale;
        w.dv[k] = dt * scale;
    };

    match desc {
        patch_descriptor::REGULAR => {
            let (bu, dbu, bv, dbv) = if param.sharpness > 0.0 {
                single_crease_basis(param.boundary(), param.sharpness, s, t)
            } else {
                let (mut bu, mut dbu) = bspline_basis(s);
                let (mut bv, mut dbv) = bspline_basis(t);
                adjust_boundary_weights(param.boundary(), &mut bu, &mut bv);
                adjust_boundary_weights(param.boundary(), &mut dbu, &mut dbv);
                (bu, dbu, bv, dbv)
            };
            for j in 0..4 {
                for i in 0..4 {
                    set(4 * j + i, bu[i] * bv[j], dbu[i] * bv[j], bu[i] * dbv[j]);
                }
            }
        }
        patch_descriptor::GREGORY_BASIS => {
            let (p, ds, dt) = gregory::evaluate_basis(s, t);
            for k in 0..20 {
                set(k, p[k], ds[k], dt[k]);
            }
        }
        patch_descriptor::QUADS => {
            let p = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
            let ds = [-(1.0 - t), 1.0 - t, t, -t];
            let dt = [-(1.0 - s), -s, s, 1.0 - s];
            for k in 0..4 {
                set(k, p[k], ds[k], dt[k]);
            }
        }
        patch_descriptor::LOOP => {
            let (mut p, mut ds, mut dt) = loop_patch::box_spline_basis(s, t);
            let boundary = param.boundary();
            adjust_box_spline_tri_boundary_weights(boundary, &mut p);
            adjust_box_spline_tri_boundary_weights(boundary, &mut ds);
            adjust_box_spline_tri_boundary_weights(boundary, &mut dt);
            for k in 0..12 {
                set(k, p[k], ds[k], dt[k]);
            }
        }
        patch_descriptor::GREGORY_TRIANGLE => {
            let (p, ds, dt) = loop_patch::evaluate_basis(s, t);
            for k in 0..18 {
                set(k, p[k], ds[k], dt[k]);
            }
        }
        _ => {
            debug_assert_eq!(desc, patch_descriptor::TRIANGLES);
            let p = [1.0 - s - t, s, t];
            let ds = [-1.0, 1.0, 0.0];
            let dt = [-1.0, 0.0, 1.0];
            for k in 0..3 {
                set(k, p[k], ds[k], dt[k]);
            }
        }
    }
    Some(w)
}

type Basis4 = ([f32; 4], [f32; 4]);

/// The two univariate bases of a single-crease patch whose crease lies on
/// the face edge named by the one bit set in `boundary`: the exact limit
/// profile of a crease of sharpness `sharpness` across it, the B-spline
/// along it.
fn single_crease_basis(
    boundary: u32,
    sharpness: f32,
    s: f32,
    t: f32,
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    // Edges 3 and 1 are the columns at s = 0 and s = 1, edges 0 and 2 the
    // rows at t = 0 and t = 1.
    let edge = boundary.trailing_zeros();
    let (bu, dbu): Basis4 = match edge {
        3 => crease_basis_oriented(s, sharpness, false),
        1 => crease_basis_oriented(s, sharpness, true),
        _ => bspline_basis(s),
    };
    let (bv, dbv): Basis4 = match edge {
        0 => crease_basis_oriented(t, sharpness, false),
        2 => crease_basis_oriented(t, sharpness, true),
        _ => bspline_basis(t),
    };
    (bu, dbu, bv, dbv)
}

/// Fold the phantom rows and columns of a boundary B-spline patch onto the
/// next two by the reflection `2a - b`, in each univariate basis
/// (`OsdAdjustBoundaryWeights`).
fn adjust_boundary_weights(boundary: u32, s_weights: &mut [f32; 4], t_weights: &mut [f32; 4]) {
    if boundary & 1 != 0 {
        t_weights[2] -= t_weights[0];
        t_weights[1] += t_weights[0] * 2.0;
        t_weights[0] = 0.0;
    }
    if boundary & 2 != 0 {
        s_weights[1] -= s_weights[3];
        s_weights[2] += s_weights[3] * 2.0;
        s_weights[3] = 0.0;
    }
    if boundary & 4 != 0 {
        t_weights[1] -= t_weights[3];
        t_weights[2] += t_weights[3] * 2.0;
        t_weights[3] = 0.0;
    }
    if boundary & 8 != 0 {
        s_weights[2] -= s_weights[0];
        s_weights[1] += s_weights[0] * 2.0;
        s_weights[0] = 0.0;
    }
}

/// Fold the weights of the phantom points of a boundary box-spline
/// triangle onto real ones (`Osd_adjustBoxSplineTriBoundaryWeights`):
/// beyond a boundary edge by the parallelogram rule, through a corner where
/// two boundary edges meet by the point reflection, and beyond a boundary
/// vertex along the lattice line through it.
fn adjust_box_spline_tri_boundary_weights(boundary: u32, w: &mut [f32; 12]) {
    if boundary == 0 {
        return;
    }
    let upper = (boundary >> 3) & 0x3;
    let lower = boundary & 0x7;
    let (e_bits, v_bits) = match upper {
        // Boundary vertices only.
        1 => (0, lower),
        // Boundary edges, and the vertex opposite each: the edge bits
        // rotated one to the right.
        2 => (lower, ((lower & 1) << 2) | (lower >> 1)),
        _ => (lower, 0),
    };
    let edge = [e_bits & 1 != 0, e_bits & 2 != 0, e_bits & 4 != 0];

    // `fold(w, src, [(dst, factor)..])`: move src's weight onto dst.
    fn fold(w: &mut [f32; 12], src: usize, targets: &[(usize, f32)]) {
        let x = w[src];
        for &(dst, factor) in targets {
            w[dst] += factor * x;
        }
    }
    // Edge e's three phantoms (P0, P1, P2), and per corner phantom the rule
    // when the adjacent edge is also a boundary (point reflection through
    // the corner) or not (parallelogram).
    if edge[0] {
        if edge[2] {
            fold(w, 0, &[(4, 2.0), (8, -1.0)]);
        } else {
            fold(w, 0, &[(4, 1.0), (3, 1.0), (7, -1.0)]);
        }
        fold(w, 1, &[(4, 1.0), (5, 1.0), (8, -1.0)]);
        if edge[1] {
            fold(w, 2, &[(5, 2.0), (8, -1.0)]);
        } else {
            fold(w, 2, &[(5, 1.0), (6, 1.0), (9, -1.0)]);
        }
        w[0] = 0.0;
        w[1] = 0.0;
        w[2] = 0.0;
    }
    if edge[1] {
        if edge[0] {
            fold(w, 6, &[(5, 2.0), (4, -1.0)]);
        } else {
            fold(w, 6, &[(5, 1.0), (2, 1.0), (1, -1.0)]);
        }
        fold(w, 9, &[(5, 1.0), (8, 1.0), (4, -1.0)]);
        if edge[2] {
            fold(w, 11, &[(8, 2.0), (4, -1.0)]);
        } else {
            fold(w, 11, &[(8, 1.0), (10, 1.0), (7, -1.0)]);
        }
        w[6] = 0.0;
        w[9] = 0.0;
        w[11] = 0.0;
    }
    if edge[2] {
        if edge[1] {
            fold(w, 10, &[(8, 2.0), (5, -1.0)]);
        } else {
            fold(w, 10, &[(8, 1.0), (11, 1.0), (9, -1.0)]);
        }
        fold(w, 7, &[(8, 1.0), (4, 1.0), (5, -1.0)]);
        if edge[0] {
            fold(w, 3, &[(4, 2.0), (5, -1.0)]);
        } else {
            fold(w, 3, &[(4, 1.0), (0, 1.0), (1, -1.0)]);
        }
        w[10] = 0.0;
        w[7] = 0.0;
        w[3] = 0.0;
    }
    if v_bits & 1 != 0 {
        fold(w, 3, &[(4, 1.0), (7, 1.0), (8, -1.0)]);
        fold(w, 0, &[(4, 1.0), (1, 1.0), (5, -1.0)]);
        w[3] = 0.0;
        w[0] = 0.0;
    }
    if v_bits & 2 != 0 {
        fold(w, 2, &[(5, 1.0), (1, 1.0), (4, -1.0)]);
        fold(w, 6, &[(5, 1.0), (9, 1.0), (8, -1.0)]);
        w[2] = 0.0;
        w[6] = 0.0;
    }
    if v_bits & 4 != 0 {
        fold(w, 11, &[(8, 1.0), (9, 1.0), (5, -1.0)]);
        fold(w, 10, &[(8, 1.0), (7, 1.0), (4, -1.0)]);
        w[11] = 0.0;
        w[10] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::far::loop_patch::fold_phantom_weights;
    use crate::INDEX_INVALID;

    /// The OpenSubdiv triangle fold agrees with this crate's own phantom
    /// rules for every wall pattern a regular neighborhood can have.
    #[test]
    fn box_spline_folds_match_the_phantom_rules() {
        let edge_slots = [[0usize, 1, 2], [6, 9, 11], [3, 7, 10]];
        let vertex_slots = [[0usize, 3], [2, 6], [10, 11]];
        let mut cases = Vec::new();
        for edges in 0u32..8 {
            cases.push((edges, 0u32));
        }
        for vertices in 1u32..8 {
            cases.push((0, vertices));
        }
        for e in 0..3u32 {
            // An edge and the vertex opposite it.
            cases.push((1 << e, 1 << ((e + 2) % 3)));
        }
        for (edges, vertices) in cases {
            let mut cvs = [0u32; 12];
            for k in 0..3 {
                if edges & (1 << k) != 0 {
                    edge_slots[k].iter().for_each(|&s| cvs[s] = INDEX_INVALID);
                }
                if vertices & (1 << k) != 0 {
                    vertex_slots[k].iter().for_each(|&s| cvs[s] = INDEX_INVALID);
                }
            }
            let mask = match (edges, vertices) {
                (e, 0) => e,
                (0, v) => v | 1 << 3,
                (e, _) => e | 2 << 3,
            };
            for (s, t) in [(0.2f32, 0.3f32), (0.6, 0.1), (0.05, 0.9)] {
                let (mut expected, _, _) = loop_patch::box_spline_basis(s, t);
                fold_phantom_weights(&cvs, &mut expected);
                let (mut actual, _, _) = loop_patch::box_spline_basis(s, t);
                adjust_box_spline_tri_boundary_weights(mask, &mut actual);
                for k in 0..12 {
                    assert!(
                        (actual[k] - expected[k]).abs() < 1e-6,
                        "mask {mask:#b} slot {k}: {} vs {}",
                        actual[k],
                        expected[k]
                    );
                }
            }
        }
    }
}
