//! Patch tables for parametric limit-surface evaluation.
//!
//! Port of the essential functionality of `opensubdiv/far/patchTable.h`,
//! `far/patchTableFactory.h`, `far/patchParam.h` and `far/patchMap.h`: a
//! collection of parametric patches covering the refined mesh, addressable
//! by ptex face and `(u, v)` location, and evaluable — with first
//! derivatives — anywhere on their domain.
//!
//! ## Patch types
//!
//! Every non-hole face of the refiner's last level yields one patch:
//!
//! * [`PatchType::Regular`] — where the face's four corners are *regular*
//!   (interior valence-4 smooth vertices, or regular boundary/pinned-corner
//!   vertices with sharpened boundaries), the limit surface over the face is
//!   exactly a bicubic B-spline patch on 16 control vertices of the last
//!   level. Boundary and corner patches are represented by marking the
//!   missing control vertices and folding their *phantom-point* reflection
//!   (`2a - b`) into the basis weights at evaluation time — mathematically
//!   equivalent to OpenSubdiv's boundary/corner basis masks.
//! * [`PatchType::GregoryBasis`] — faces whose corners are smooth interior
//!   vertices but include extraordinary valences are capped with a Gregory
//!   patch (see [`super::gregory`]): 20 derived points giving corner
//!   limit-point interpolation, exact C0 boundaries and approximate G1
//!   smoothness, as OpenSubdiv's `ENDCAP_GREGORY_BASIS` does.
//! * [`PatchType::Quads`] — anywhere else (irregularity involving creases,
//!   boundaries such as unsharpened `VtxBoundaryInterpolation::None`, or
//!   non-manifold neighborhoods), the patch falls back to bilinear
//!   interpolation of the refined face. The approximation error is
//!   confined to those faces and shrinks with each refinement level.
//!
//! Control-vertex indices refer to the vertices of the refiner's **last
//! level**: evaluate patches against the primvar buffer produced by
//! [`PrimvarRefiner::interpolate`](super::PrimvarRefiner::interpolate) for
//! that level.

use super::gregory::{self, GregoryPoints};
use super::primvar_refiner::Primvar;
use super::ptex::PtexIndices;
use super::topology_refiner::TopologyRefiner;
use crate::sdc::{Crease, SchemeType, Split};
use crate::vtr::{Level, TopologyError};
use crate::{Index, INDEX_INVALID};

/// The basis of a patch (`Far::PatchDescriptor::Type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchType {
    /// Bicubic B-spline patch on a 4x4 control-vertex grid.
    Regular,
    /// Gregory end-cap patch: 20 derived points around an extraordinary
    /// vertex (`PatchDescriptor::GREGORY_BASIS`).
    GregoryBasis,
    /// Bilinear quad on the face's 4 vertices.
    Quads,
}

/// Location and orientation of a patch within the parametric space of its
/// ptex face (`Far::PatchParam`).
///
/// A patch's local coordinates `(s, t) ∈ [0,1]²` map to ptex-face
/// coordinates via `(u, v) = origin + R(rotation) · (s, t) / 2^depth`.
#[derive(Debug, Clone, Copy)]
pub struct PatchParam {
    /// The ptex face this patch belongs to.
    pub ptex_face: Index,
    /// Subdivision depth of the patch below its ptex root face.
    pub depth: u8,
    /// Orientation of the patch within the ptex face: number of 90° CCW
    /// rotations from patch frame to ptex frame.
    pub rotation: u8,
    /// Ptex-frame coordinates of the patch's `(0, 0)` corner.
    pub origin: [f32; 2],
}

impl PatchParam {
    /// Convert ptex-face coordinates to the patch's local `(s, t)`
    /// (`PatchParam::Normalize`).
    pub fn normalize(&self, u: f32, v: f32) -> (f32, f32) {
        let scale = (1u32 << self.depth) as f32;
        let x = (u - self.origin[0]) * scale;
        let y = (v - self.origin[1]) * scale;
        rotate_ccw((4 - self.rotation as u32) % 4, x, y)
    }

    /// Convert the patch's local `(s, t)` to ptex-face coordinates — the
    /// inverse of [`normalize`](Self::normalize).
    pub fn unnormalize(&self, s: f32, t: f32) -> (f32, f32) {
        let scale = 1.0 / (1u32 << self.depth) as f32;
        let (x, y) = rotate_ccw(self.rotation as u32, s, t);
        (self.origin[0] + x * scale, self.origin[1] + y * scale)
    }

    /// The Jacobian `d(s,t)/d(u,v)` of [`normalize`](Self::normalize), as
    /// `[ds/du, ds/dv, dt/du, dt/dv]`.
    fn derivative_matrix(&self) -> [f32; 4] {
        let s = (1u32 << self.depth) as f32;
        match self.rotation % 4 {
            0 => [s, 0.0, 0.0, s],
            1 => [0.0, s, -s, 0.0],
            2 => [-s, 0.0, 0.0, -s],
            _ => [0.0, -s, s, 0.0],
        }
    }
}

/// Basis weights of a patch at one parametric location: the limit point is
/// `Σ weights[i] · control[indices[i]]`, and similarly for the first
/// derivatives with respect to the ptex-face `(u, v)`.
#[derive(Debug, Clone, Default)]
pub struct PatchBasis {
    pub indices: Vec<Index>,
    pub weights: Vec<f32>,
    pub du_weights: Vec<f32>,
    pub dv_weights: Vec<f32>,
}

#[derive(Debug, Clone)]
enum PatchKind {
    /// A 4x4 control-vertex grid (row-major, rows along `t`), with
    /// `INDEX_INVALID` marking phantom boundary slots.
    Regular([Index; 16]),
    /// The 20 Gregory control points as stencils on the last level's
    /// vertices.
    Gregory(Box<GregoryPoints>),
    /// The face's 4 corner vertices.
    Quads([Index; 4]),
}

#[derive(Debug, Clone)]
struct Patch {
    param: PatchParam,
    /// The face of the last level this patch covers.
    face: Index,
    kind: PatchKind,
}

/// A table of patches describing the limit surface of a refined mesh
/// (`Far::PatchTable`).
#[derive(Debug, Clone)]
pub struct PatchTable {
    patches: Vec<Patch>,
    /// Patch index of each face of the last level (`INDEX_INVALID` for
    /// holes).
    face_to_patch: Vec<Index>,
    ptex: PtexIndices,
    max_level: usize,
    /// First level-1 child face of each base face.
    level1_offsets: Vec<u32>,
}

impl PatchTable {
    /// The number of patches in the table (`GetNumPatchesTotal`).
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// The basis type of patch `patch`.
    pub fn patch_type(&self, patch: usize) -> PatchType {
        match &self.patches[patch].kind {
            PatchKind::Regular(_) => PatchType::Regular,
            PatchKind::Gregory(_) => PatchType::GregoryBasis,
            PatchKind::Quads(_) => PatchType::Quads,
        }
    }

    /// The [`PatchParam`] of patch `patch` (`GetPatchParam`).
    pub fn patch_param(&self, patch: usize) -> PatchParam {
        self.patches[patch].param
    }

    /// The face of the refiner's last level covered by patch `patch`.
    pub fn patch_face(&self, patch: usize) -> Index {
        self.patches[patch].face
    }

    /// The control vertices of patch `patch` (`GetPatchVertices`), as
    /// indices into the last level's vertices. Phantom boundary slots of
    /// regular patches are omitted; for Gregory patches this is the union
    /// of the vertices supporting its 20 derived points.
    pub fn patch_vertices(&self, patch: usize) -> Vec<Index> {
        match &self.patches[patch].kind {
            PatchKind::Regular(cvs) => cvs
                .iter()
                .copied()
                .filter(|&cv| cv != INDEX_INVALID)
                .collect(),
            PatchKind::Quads(cvs) => cvs.to_vec(),
            PatchKind::Gregory(points) => {
                let mut cvs: Vec<Index> = Vec::new();
                for point in points.iter() {
                    for &(cv, _) in &point.0 {
                        if !cvs.contains(&cv) {
                            cvs.push(cv);
                        }
                    }
                }
                cvs
            }
        }
    }

    /// The ptex indexing of the base mesh.
    pub fn ptex_indices(&self) -> &PtexIndices {
        &self.ptex
    }

    /// Evaluate the patch basis at a location given in the parametric space
    /// of the patch's *ptex face* (`EvaluateBasis`): returns point and
    /// first-derivative weights on the patch's control vertices.
    pub fn evaluate_basis(&self, patch: usize, u: f32, v: f32) -> PatchBasis {
        let p = &self.patches[patch];
        let (s, t) = p.param.normalize(u, v);
        // Chain rule to ptex frame: dP/du = Ps·ds/du + Pt·dt/du with
        // m = [ds/du, ds/dv, dt/du, dt/dv].
        let m = p.param.derivative_matrix();

        let mut basis = PatchBasis::default();
        let mut push = |cv: Index, w: f32, ws: f32, wt: f32| {
            let du = ws * m[0] + wt * m[2];
            let dv = ws * m[1] + wt * m[3];
            match basis.indices.iter().position(|&i| i == cv) {
                Some(k) => {
                    basis.weights[k] += w;
                    basis.du_weights[k] += du;
                    basis.dv_weights[k] += dv;
                }
                None => {
                    basis.indices.push(cv);
                    basis.weights.push(w);
                    basis.du_weights.push(du);
                    basis.dv_weights.push(dv);
                }
            }
        };

        match &p.kind {
            PatchKind::Regular(cvs) => {
                let (bu, dbu) = bspline_basis(s);
                let (bv, dbv) = bspline_basis(t);
                let mut w = [0.0f32; 16];
                let mut ws = [0.0f32; 16];
                let mut wt = [0.0f32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        w[4 * j + i] = bu[i] * bv[j];
                        ws[4 * j + i] = dbu[i] * bv[j];
                        wt[4 * j + i] = bu[i] * dbv[j];
                    }
                }
                fold_phantom_weights(cvs, &mut w);
                fold_phantom_weights(cvs, &mut ws);
                fold_phantom_weights(cvs, &mut wt);
                for (slot, &cv) in cvs.iter().enumerate() {
                    if cv == INDEX_INVALID {
                        debug_assert!(w[slot] == 0.0 && ws[slot] == 0.0 && wt[slot] == 0.0);
                        continue;
                    }
                    push(cv, w[slot], ws[slot], wt[slot]);
                }
            }
            PatchKind::Quads(cvs) => {
                let w = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
                let ws = [-(1.0 - t), 1.0 - t, t, -t];
                let wt = [-(1.0 - s), -s, s, 1.0 - s];
                for (slot, &cv) in cvs.iter().enumerate() {
                    push(cv, w[slot], ws[slot], wt[slot]);
                }
            }
            PatchKind::Gregory(points) => {
                let (w20, ws20, wt20) = gregory::evaluate_basis(s, t);
                for (point, stencil) in points.iter().enumerate() {
                    for &(cv, sw) in &stencil.0 {
                        push(cv, w20[point] * sw, ws20[point] * sw, wt20[point] * sw);
                    }
                }
            }
        }
        basis
    }

    /// Evaluate primvar data on patch `patch` at ptex-face coordinates
    /// `(u, v)`: returns the limit point and its two first derivatives.
    /// `control_values` holds one value per vertex of the refiner's last
    /// level.
    pub fn evaluate<T: Primvar>(
        &self,
        patch: usize,
        u: f32,
        v: f32,
        control_values: &[T],
    ) -> (T, T, T) {
        let basis = self.evaluate_basis(patch, u, v);
        let mut point = control_values[0].clone();
        let mut du = control_values[0].clone();
        let mut dv = control_values[0].clone();
        point.clear();
        du.clear();
        dv.clear();
        for (k, &cv) in basis.indices.iter().enumerate() {
            let value = &control_values[cv as usize];
            point.add_with_weight(value, basis.weights[k]);
            du.add_with_weight(value, basis.du_weights[k]);
            dv.add_with_weight(value, basis.dv_weights[k]);
        }
        (point, du, dv)
    }
}

/// Accelerated location of the patch covering a `(ptex face, u, v)`
/// location (`Far::PatchMap`).
pub struct PatchMap<'a> {
    table: &'a PatchTable,
}

impl<'a> PatchMap<'a> {
    pub fn new(table: &'a PatchTable) -> Self {
        Self { table }
    }

    /// The patch covering coordinates `(u, v)` of `ptex_face`
    /// (`FindPatch`); `None` only when the location lies in a hole.
    pub fn find_patch(&self, ptex_face: usize, u: f32, v: f32) -> Option<usize> {
        let table = self.table;
        let base_face = table.ptex.base_face(ptex_face) as usize;
        // Quad base faces map to exactly one ptex face; N-gons to N.
        let base_is_quad = table.ptex.face_ptex_count(base_face) == 1;

        let (mut face, mut level, mut u, mut v) = if base_is_quad {
            (base_face, 0usize, u, v)
        } else {
            // Non-quad base faces root their ptex faces at level 1.
            let child =
                table.level1_offsets[base_face] as usize + table.ptex.base_face_corner(ptex_face);
            (child, 1usize, u, v)
        };

        while level < table.max_level {
            let first_child = if level == 0 {
                table.level1_offsets[face] as usize
            } else {
                4 * face
            };
            let (k, nu, nv) = descend_quadrant(u, v);
            face = first_child + k;
            u = nu;
            v = nv;
            level += 1;
        }

        match table.face_to_patch[face] {
            INDEX_INVALID => None,
            p => Some(p as usize),
        }
    }
}

/// Factory constructing a [`PatchTable`] from a refined [`TopologyRefiner`]
/// (`Far::PatchTableFactory`).
pub struct PatchTableFactory;

impl PatchTableFactory {
    pub fn create(refiner: &TopologyRefiner) -> Result<PatchTable, TopologyError> {
        if refiner.scheme_type().topological_split_type() != Split::ToQuads {
            return Err(TopologyError::LoopPatchesNotSupported);
        }
        let base = refiner.level(0);
        let all_quads = (0..base.num_faces()).all(|f| base.face_vertices(f).len() == 4);
        if !all_quads && refiner.max_level() == 0 {
            return Err(TopologyError::PatchesRequireRefinement);
        }

        // First level-1 child face of each base face (an N-gon yields N
        // children).
        let mut level1_offsets = Vec::with_capacity(base.num_faces());
        let mut offset = 0u32;
        for f in 0..base.num_faces() {
            level1_offsets.push(offset);
            offset += base.face_vertices(f).len() as u32;
        }

        let ptex = PtexIndices::new(refiner);
        let max_level = refiner.max_level();
        let last = refiner.level(max_level);
        let last_inner = last.inner();
        let smooth_scheme = refiner.scheme_type() == SchemeType::Catmark;

        let mut patches = Vec::new();
        let mut face_to_patch = vec![INDEX_INVALID; last.num_faces()];

        for (face, patch_slot) in face_to_patch.iter_mut().enumerate() {
            if last.is_face_hole(face) {
                continue;
            }
            let param = compute_patch_param(refiner, &ptex, &level1_offsets, face);
            let kind = if smooth_scheme {
                if let Some(cvs) = gather_regular_patch(last_inner, face) {
                    PatchKind::Regular(cvs)
                } else if let Some(points) = gregory::build(last_inner, face) {
                    // Smooth interior extraordinary neighborhood: Gregory
                    // end cap.
                    PatchKind::Gregory(points)
                } else {
                    // Creased or boundary irregularity: bilinear fallback.
                    PatchKind::Quads(quad_cvs(last_inner, face))
                }
            } else {
                // Bilinear: the mesh is its own limit surface.
                PatchKind::Quads(quad_cvs(last_inner, face))
            };
            *patch_slot = patches.len() as Index;
            patches.push(Patch {
                param,
                face: face as Index,
                kind,
            });
        }

        Ok(PatchTable {
            patches,
            face_to_patch,
            ptex,
            max_level,
            level1_offsets,
        })
    }
}

// ----------------------------------------------------------------------
//  Parametric transforms
// ----------------------------------------------------------------------

/// Positions of a quad's corners in its unit parametric square.
const CORNER_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

fn rotate_ccw(times: u32, x: f32, y: f32) -> (f32, f32) {
    match times % 4 {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        _ => (y, -x),
    }
}

/// Select the child quadrant containing `(u, v)` and transform into the
/// child's parametric frame. Child `k` of a quad occupies the quarter at
/// corner `k`, rotated by `k`·90°: its corners are
/// `[corner_k, edge_k, center, edge_{k-1}]`.
fn descend_quadrant(u: f32, v: f32) -> (usize, f32, f32) {
    let k = match (u < 0.5, v < 0.5) {
        (true, true) => 0,
        (false, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
    };
    let (x, y) = (u - CORNER_UV[k][0], v - CORNER_UV[k][1]);
    let (x, y) = rotate_ccw((4 - k as u32) % 4, x, y);
    (k, x * 2.0, y * 2.0)
}

/// Compute the [`PatchParam`] of a face of the last level by walking its
/// ancestry down to the ptex root, composing the quadrant transforms
/// `uv = corner_k + R_k · (child uv) / 2`.
fn compute_patch_param(
    refiner: &TopologyRefiner,
    ptex: &PtexIndices,
    level1_offsets: &[u32],
    face: usize,
) -> PatchParam {
    let max_level = refiner.max_level();
    let mut origin = [0.0f32; 2];
    let mut rotation = 0u8;
    let mut depth = 0u8;
    let mut face = face;

    let compose = |origin: &mut [f32; 2], rotation: &mut u8, depth: &mut u8, k: usize| {
        let (x, y) = rotate_ccw(k as u32, origin[0], origin[1]);
        origin[0] = CORNER_UV[k][0] + 0.5 * x;
        origin[1] = CORNER_UV[k][1] + 0.5 * y;
        *rotation = ((*rotation as usize + k) % 4) as u8;
        *depth += 1;
    };

    // Steps between levels >= 2 are always quadrants of a quad parent.
    for level in (2..=max_level).rev() {
        let parent = refiner.refinement(level).child_face_parent_face(face) as usize;
        let k = face - 4 * parent;
        compose(&mut origin, &mut rotation, &mut depth, k);
        face = parent;
    }

    if max_level == 0 {
        // All-quad base mesh with no refinement: the base face is the patch.
        return PatchParam {
            ptex_face: ptex.face_id(face),
            depth: 0,
            rotation: 0,
            origin: [0.0, 0.0],
        };
    }

    // The final step from level 1 to the base: for quad base faces it is a
    // regular quadrant step of the (single) ptex face; for N-gons the
    // level-1 child *is* the ptex root.
    let base = refiner.refinement(1).child_face_parent_face(face) as usize;
    let k = face - level1_offsets[base] as usize;
    if refiner.level(0).face_vertices(base).len() == 4 {
        compose(&mut origin, &mut rotation, &mut depth, k);
        PatchParam {
            ptex_face: ptex.face_id(base),
            depth,
            rotation,
            origin,
        }
    } else {
        PatchParam {
            ptex_face: ptex.face_id(base) + k as Index,
            depth,
            rotation,
            origin,
        }
    }
}

// ----------------------------------------------------------------------
//  Regular-patch classification and control-vertex gathering
// ----------------------------------------------------------------------

fn quad_cvs(level: &Level, face: usize) -> [Index; 4] {
    let fv = level.face_vertices(face);
    let mut cvs = [INDEX_INVALID; 4];
    cvs.copy_from_slice(fv);
    cvs
}

/// Is this vertex regular for the purpose of B-spline patch extraction?
fn is_corner_regular(level: &Level, vertex: Index) -> bool {
    let v = vertex as usize;
    let num_faces = level.vertex_faces(v).len();
    let num_edges = level.vertex_edges(v).len();
    let sharpness = level.vertex_sharpness(v);

    let edges_ok = |require_boundary_sharp: bool| {
        level.vertex_edges(v).iter().all(|&e| {
            let e = e as usize;
            if level.is_edge_boundary(e) {
                require_boundary_sharp && Crease::is_infinite(level.edge_sharpness(e))
            } else {
                level.edge_sharpness(e) == 0.0
            }
        })
    };

    if !level.is_vertex_boundary(v) {
        // Interior: valence 4, smooth vertex, smooth edges.
        num_edges == 4 && num_faces == 4 && sharpness == 0.0 && edges_ok(false)
    } else if num_faces == 2 && num_edges == 3 {
        // Regular boundary vertex on a sharpened boundary.
        sharpness == 0.0 && edges_ok(true)
    } else if num_faces == 1 && num_edges == 2 {
        // Boundary corner: regular only when pinned (EdgeAndCorner), which
        // matches the interpolating end-condition of the phantom-point
        // reflection.
        Crease::is_infinite(sharpness) && edges_ok(true)
    } else {
        false
    }
}

fn other_face_of_edge(level: &Level, edge: Index, face: Index) -> Option<Index> {
    let faces = level.edge_faces(edge as usize);
    if faces.len() != 2 {
        return None;
    }
    Some(if faces[0] == face { faces[1] } else { faces[0] })
}

/// The neighbor of `v` within (quad) `face` that is not `exclude`.
fn neighbor_in_face(level: &Level, face: Index, v: Index, exclude: Index) -> Option<Index> {
    let fv = level.face_vertices(face as usize);
    if fv.len() != 4 {
        return None;
    }
    let i = fv.iter().position(|&x| x == v)?;
    let n = fv.len();
    let prev = fv[(i + n - 1) % n];
    let next = fv[(i + 1) % n];
    if prev == exclude {
        Some(next)
    } else if next == exclude {
        Some(prev)
    } else {
        None
    }
}

/// The vertex of (quad) `face` diagonally opposite `v`.
fn diagonal_in_face(level: &Level, face: Index, v: Index) -> Option<Index> {
    let fv = level.face_vertices(face as usize);
    if fv.len() != 4 {
        return None;
    }
    let i = fv.iter().position(|&x| x == v)?;
    Some(fv[(i + 2) % 4])
}

/// Attempt to classify `face` as a regular B-spline patch and gather its
/// 4x4 control-vertex grid; phantom (boundary) slots are `INDEX_INVALID`.
///
/// The grid is row-major with rows along `t`:
///
/// ```text
///   12 13 14 15
///    8  9 10 11        with the face being the center quad
///    4  5  6  7        (5, 6, 10, 9) = (c0, c1, c2, c3).
///    0  1  2  3
/// ```
fn gather_regular_patch(level: &Level, face: usize) -> Option<[Index; 16]> {
    let fv = level.face_vertices(face);
    if fv.len() != 4 {
        return None;
    }
    let [c0, c1, c2, c3] = [fv[0], fv[1], fv[2], fv[3]];
    for &c in &[c0, c1, c2, c3] {
        if !is_corner_regular(level, c) {
            return None;
        }
    }

    let fe = level.face_edges(face);
    let face = face as Index;
    let mut cvs = [INDEX_INVALID; 16];
    cvs[5] = c0;
    cvs[6] = c1;
    cvs[10] = c2;
    cvs[9] = c3;

    // Row/column neighbors across the face's four edges.
    let a0 = other_face_of_edge(level, fe[0], face); // below  (row 0/1)
    let a1 = other_face_of_edge(level, fe[1], face); // right  (col 3)
    let a2 = other_face_of_edge(level, fe[2], face); // above  (row 3)
    let a3 = other_face_of_edge(level, fe[3], face); // left   (col 0)

    if let Some(a) = a0 {
        cvs[1] = neighbor_in_face(level, a, c0, c1)?;
        cvs[2] = neighbor_in_face(level, a, c1, c0)?;
    }
    if let Some(a) = a1 {
        cvs[7] = neighbor_in_face(level, a, c1, c2)?;
        cvs[11] = neighbor_in_face(level, a, c2, c1)?;
    }
    if let Some(a) = a2 {
        cvs[14] = neighbor_in_face(level, a, c2, c3)?;
        cvs[13] = neighbor_in_face(level, a, c3, c2)?;
    }
    if let Some(a) = a3 {
        cvs[4] = neighbor_in_face(level, a, c0, c3)?;
        cvs[8] = neighbor_in_face(level, a, c3, c0)?;
    }

    // Diagonal (corner) faces, reached across the row-neighbor faces.
    cvs[0] = gather_corner_cv(level, a0, c0, cvs[1])?; // via A0 across edge (c0, cv1)
    cvs[3] = gather_corner_cv(level, a0, c1, cvs[2])?;
    cvs[15] = gather_corner_cv(level, a1, c2, cvs[11])?;
    cvs[12] = gather_corner_cv(level, a3, c3, cvs[8])?;

    // The phantom pattern must consist of complete missing border rows and
    // columns — anything else indicates an irregular configuration.
    let row0 = a0.is_none();
    let col3 = a1.is_none();
    let row3 = a2.is_none();
    let col0 = a3.is_none();
    for j in 0..4 {
        for i in 0..4 {
            let phantom_expected =
                (j == 0 && row0) || (j == 3 && row3) || (i == 0 && col0) || (i == 3 && col3);
            let is_phantom = cvs[4 * j + i] == INDEX_INVALID;
            if (1..3).contains(&i) && (1..3).contains(&j) {
                debug_assert!(!is_phantom);
            }
            if phantom_expected != is_phantom {
                return None;
            }
        }
    }
    Some(cvs)
}

/// The diagonal control vertex at a grid corner, found by crossing from the
/// row-neighbor face `a` over the edge `(c, inner)` into the corner face.
/// Returns `INDEX_INVALID` (a phantom slot) when the neighborhood ends at a
/// boundary, and `None` when a non-quad face makes the patch irregular.
fn gather_corner_cv(level: &Level, a: Option<Index>, c: Index, inner: Index) -> Option<Index> {
    if let Some(a) = a {
        if inner != INDEX_INVALID {
            if let Some(edge) = level.find_edge(c, inner) {
                if let Some(d) = other_face_of_edge(level, edge, a) {
                    // A non-quad corner face is disqualifying, not phantom.
                    return diagonal_in_face(level, d, c);
                }
            }
        }
    }
    Some(INDEX_INVALID)
}

/// Redistribute the weights of phantom (missing boundary) control-vertex
/// slots onto real slots via the reflection `p = 2a - b`: the B-spline
/// natural end-condition reproducing sharpened-boundary (crease) behavior,
/// and — after double reflection at corners — interpolating pinned corners.
fn fold_phantom_weights(cvs: &[Index; 16], weights: &mut [f32]) {
    for _ in 0..2 {
        for j in 0..4usize {
            for i in 0..4usize {
                let slot = 4 * j + i;
                if cvs[slot] != INDEX_INVALID || weights[slot] == 0.0 {
                    continue;
                }
                let (a, b) = if j == 0 {
                    (4 + i, 8 + i) // reflect off rows 1, 2
                } else if j == 3 {
                    (8 + i, 4 + i)
                } else if i == 0 {
                    (4 * j + 1, 4 * j + 2) // reflect off cols 1, 2
                } else {
                    (4 * j + 2, 4 * j + 1)
                };
                let w = weights[slot];
                weights[a] += 2.0 * w;
                weights[b] -= w;
                weights[slot] = 0.0;
            }
        }
    }
}

/// Uniform cubic B-spline basis functions (and derivatives) over the
/// central knot interval, `t ∈ [0, 1]`.
fn bspline_basis(t: f32) -> ([f32; 4], [f32; 4]) {
    let t2 = t * t;
    let t3 = t2 * t;
    let one_minus = 1.0 - t;
    let basis = [
        one_minus * one_minus * one_minus / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ];
    let deriv = [
        -0.5 * one_minus * one_minus,
        1.5 * t2 - 2.0 * t,
        -1.5 * t2 + t + 0.5,
        0.5 * t2,
    ];
    (basis, deriv)
}
