//! Patch tables for parametric limit-surface evaluation.
//!
//! Port of the essential functionality of `opensubdiv/far/patchTable.h`,
//! `far/patchTableFactory.h`, `far/patchParam.h` and `far/patchMap.h`: a
//! collection of parametric patches covering the refined mesh, addressable
//! by ptex face and `(u, v)` location, and evaluable — with first
//! derivatives — anywhere on their domain.
//!
//! For a uniformly refined mesh, every non-hole face of the last level
//! yields one patch. For a feature-adaptively refined mesh
//! ([`TopologyRefiner::refine_adaptive`](super::TopologyRefiner::refine_adaptive)),
//! each face is patched at the level where it becomes regular, and only
//! the irregular remainder descends to the isolation cap — the patch count
//! grows with the mesh's features rather than with `4^level`. Adjacent
//! patches at different depths evaluate the same limit surface, so
//! parametric evaluation is seamless (crack-free *tessellation* support —
//! transition-edge tagging — is not needed for evaluation and is not
//! provided).
//!
//! ## Patch types
//!
//! * [`PatchType::Regular`] — where the face's four corners are *regular*
//!   (interior valence-4 smooth vertices; regular boundary vertices with
//!   sharpened boundaries; pinned corners; and vertices on an infinitely
//!   sharp crease whose span of faces on the patch's side is regular), the
//!   limit surface over the face is exactly a bicubic B-spline patch on 16
//!   control vertices of the last level. Boundary, crease and corner
//!   patches are represented by marking the missing control vertices and
//!   folding their *phantom-point* reflection (`2a - b`) into the basis
//!   weights at evaluation time — mathematically equivalent to OpenSubdiv's
//!   boundary/corner basis masks, with infinitely sharp creases treated as
//!   boundaries (OpenSubdiv's `useInfSharpPatch`).
//! * [`PatchType::GregoryBasis`] — every other manifold face at its
//!   isolation level is capped with a Gregory patch (see
//!   [`super::gregory`]): extraordinary vertices, irregular boundary and
//!   crease corners, sharp corners and darts. The 20 derived points give
//!   corner limit-point interpolation, exact C0 boundaries and approximate
//!   G1 smoothness, as OpenSubdiv's `ENDCAP_GREGORY_BASIS` does.
//! * [`PatchType::Quads`] — non-manifold neighborhoods (and every face of
//!   the Bilinear scheme, whose mesh is its own limit surface) fall back to
//!   bilinear interpolation of the refined face.
//!
//! Control-vertex indices refer to the **concatenation of every level's
//! vertices**, base level first: evaluate patches against the base values
//! followed by each level's
//! [`PrimvarRefiner::interpolate`](super::PrimvarRefiner::interpolate)
//! output.

use super::gregory::{self, corner_span, is_edge_singular, vertex_rule, GregoryPoints};
use super::primvar_refiner::Primvar;
use super::ptex::PtexIndices;
use super::topology_refiner::TopologyRefiner;
use crate::sdc::{Crease, Rule, SchemeType, Split};
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
    /// Indices of the contributing control vertices.
    pub indices: Vec<Index>,
    /// Weight of each control vertex for the limit position.
    pub weights: Vec<f32>,
    /// Weight of each control vertex for the derivative along `u`.
    pub du_weights: Vec<f32>,
    /// Weight of each control vertex for the derivative along `v`.
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
///
/// Patches may live at *mixed depths* (feature-adaptive refinement emits a
/// patch at the level where its face becomes regular). Control-vertex
/// indices refer to the **concatenation of every level's vertices**, base
/// level first — the buffer produced by evaluating
/// [`PrimvarRefiner::interpolate`](super::PrimvarRefiner::interpolate)
/// level by level and appending each result to the base values
/// ([`num_control_values`](Self::num_control_values) in total).
#[derive(Debug, Clone)]
pub struct PatchTable {
    patches: Vec<Patch>,
    /// Per level: patch index of each face (`INDEX_INVALID` where no patch
    /// was emitted — holes, refined faces, or unsupported support faces).
    face_to_patch: Vec<Vec<Index>>,
    ptex: PtexIndices,
    max_level: usize,
    /// Per refinement step: first child face of each parent face
    /// (`INDEX_INVALID` for faces without children under sparse
    /// refinement).
    first_child: Vec<Vec<Index>>,
    /// Total control values (sum of all levels' vertex counts).
    num_control_values: usize,
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

    /// The face covered by patch `patch`, within the level given by the
    /// patch's [`PatchParam::depth`] ancestry (for uniform refiners, the
    /// last level).
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

    /// The number of control values patch evaluation expects: one value per
    /// vertex of every refinement level, base level first
    /// (`GetNumControlVertices`).
    pub fn num_control_values(&self) -> usize {
        self.num_control_values
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
    /// `control_values` holds one value per vertex of *every* refinement
    /// level, base level first
    /// ([`num_control_values`](Self::num_control_values) in total).
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
    /// Build a map over the patches of `table`.
    pub fn new(table: &'a PatchTable) -> Self {
        Self { table }
    }

    /// The patch covering coordinates `(u, v)` of `ptex_face`
    /// (`FindPatch`): descends the quadrant hierarchy until a patch is
    /// found, supporting mixed-depth (adaptive) tables. `None` only when
    /// the location lies in a hole.
    pub fn find_patch(&self, ptex_face: usize, u: f32, v: f32) -> Option<usize> {
        let table = self.table;
        let base_face = table.ptex.base_face(ptex_face) as usize;
        // Quad base faces map to exactly one ptex face; N-gons to N.
        let base_is_quad = table.ptex.face_ptex_count(base_face) == 1;

        let (mut face, mut level, mut u, mut v) = if base_is_quad {
            (base_face, 0usize, u, v)
        } else {
            // Non-quad base faces root their ptex faces at level 1 (their
            // isolation guarantees at least one refinement).
            let child =
                table.first_child[0][base_face] as usize + table.ptex.base_face_corner(ptex_face);
            (child, 1usize, u, v)
        };

        loop {
            match table.face_to_patch[level][face] {
                INDEX_INVALID => {}
                p => return Some(p as usize),
            }
            if level >= table.max_level {
                return None;
            }
            let first_child = table.first_child[level][face];
            if first_child == INDEX_INVALID {
                return None; // unrefined support face without a patch
            }
            let (k, nu, nv) = descend_quadrant(u, v);
            face = first_child as usize + k;
            u = nu;
            v = nv;
            level += 1;
        }
    }
}

/// Factory constructing a [`PatchTable`] from a refined [`TopologyRefiner`]
/// (`Far::PatchTableFactory`).
pub struct PatchTableFactory;

impl PatchTableFactory {
    /// Build a [`PatchTable`] covering the limit surface of `refiner`
    /// (`PatchTableFactory::Create`).
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::LoopPatchesNotSupported`] for schemes that do
    /// not split faces into quads, and
    /// [`TopologyError::PatchesRequireRefinement`] when the base mesh contains
    /// non-quad faces and `refiner` has not been refined at least once.
    pub fn create(refiner: &TopologyRefiner) -> Result<PatchTable, TopologyError> {
        if refiner.scheme_type().topological_split_type() != Split::ToQuads {
            return Err(TopologyError::LoopPatchesNotSupported);
        }
        let base = refiner.level(0);
        let all_quads = (0..base.num_faces()).all(|f| base.face_vertices(f).len() == 4);
        if !all_quads && refiner.max_level() == 0 {
            return Err(TopologyError::PatchesRequireRefinement);
        }

        let max_level = refiner.max_level();

        // Per refinement step: the first child face of each parent face
        // (children of one parent are contiguous, in corner order).
        let mut first_child: Vec<Vec<Index>> = Vec::with_capacity(max_level);
        for l in 1..=max_level {
            let refinement = refiner.refinement(l);
            let mut fc = vec![INDEX_INVALID; refiner.level(l - 1).num_faces()];
            for cf in 0..refiner.level(l).num_faces() {
                let parent = refinement.child_face_parent_face(cf) as usize;
                if fc[parent] == INDEX_INVALID {
                    fc[parent] = cf as Index;
                }
            }
            first_child.push(fc);
        }

        // Control values are the concatenation of every level's vertices.
        let mut level_offsets = Vec::with_capacity(max_level + 1);
        let mut total = 0usize;
        for l in 0..=max_level {
            level_offsets.push(total as Index);
            total += refiner.level(l).num_vertices();
        }

        let ptex = PtexIndices::new(refiner);
        let smooth_scheme = refiner.scheme_type() == SchemeType::Catmark;

        let mut patches = Vec::new();
        let mut face_to_patch: Vec<Vec<Index>> = (0..=max_level)
            .map(|l| vec![INDEX_INVALID; refiner.level(l).num_faces()])
            .collect();

        for level in 0..=max_level {
            let level_view = refiner.level(level);
            let inner = level_view.inner();
            let offset = level_offsets[level];
            for (face, patch_slot) in face_to_patch[level].iter_mut().enumerate() {
                if level_view.is_face_hole(face) || !refiner.face_is_candidate(level, face) {
                    continue;
                }
                if level < max_level && refiner.face_is_selected(level, face) {
                    continue; // refined further; patches come from children
                }
                let param = compute_patch_param(refiner, &ptex, &first_child, face, level);
                let kind = if smooth_scheme {
                    if let Some(mut cvs) = gather_regular_patch(inner, face) {
                        for cv in cvs.iter_mut().filter(|cv| **cv != INDEX_INVALID) {
                            *cv += offset;
                        }
                        PatchKind::Regular(cvs)
                    } else if let Some(mut points) = gregory::build(inner, face) {
                        // Irregular manifold neighborhood (extraordinary,
                        // boundary, crease, sharp or dart corners): Gregory
                        // end cap.
                        for point in points.iter_mut() {
                            for (cv, _) in point.0.iter_mut() {
                                *cv += offset;
                            }
                        }
                        PatchKind::Gregory(points)
                    } else {
                        // Non-manifold neighborhood: bilinear fallback.
                        PatchKind::Quads(quad_cvs(inner, face).map(|cv| cv + offset))
                    }
                } else {
                    // Bilinear: the mesh is its own limit surface.
                    PatchKind::Quads(quad_cvs(inner, face).map(|cv| cv + offset))
                };
                *patch_slot = patches.len() as Index;
                patches.push(Patch {
                    param,
                    face: face as Index,
                    kind,
                });
            }
        }

        Ok(PatchTable {
            patches,
            face_to_patch,
            ptex,
            max_level,
            first_child,
            num_control_values: total,
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

/// Compute the [`PatchParam`] of a face of level `level` by walking its
/// ancestry down to the ptex root, composing the quadrant transforms
/// `uv = corner_k + R_k · (child uv) / 2`.
fn compute_patch_param(
    refiner: &TopologyRefiner,
    ptex: &PtexIndices,
    first_child: &[Vec<Index>],
    face: usize,
    level: usize,
) -> PatchParam {
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
    for l in (2..=level).rev() {
        let parent = refiner.refinement(l).child_face_parent_face(face) as usize;
        let k = face - first_child[l - 1][parent] as usize;
        compose(&mut origin, &mut rotation, &mut depth, k);
        face = parent;
    }

    if level == 0 {
        // A base-level (quad) face is its own patch domain.
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
    let k = face - first_child[0][base] as usize;
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

/// Is corner `corner` of `face` regular for the purpose of B-spline patch
/// extraction (`PatchBuilder::IsPatchRegular`, one corner)?
///
/// Regularity is decided by the *span* of faces around the vertex that
/// contains the face — the fan bounded by boundary edges and infinitely
/// sharp creases, which the limit surface treats alike:
///
/// * a smooth vertex must be interior with valence 4;
/// * a vertex on an infinitely sharp crease (exactly two infinitely sharp
///   edges) must be a regular boundary vertex, or an interior vertex whose
///   span on the patch's side holds exactly two faces;
/// * an infinitely sharp corner (a pinned vertex, or three or more
///   infinitely sharp edges) must have a single-face span;
/// * darts are never regular.
///
/// Semi-sharp features are never regular: adaptive refinement isolates
/// them until they decay (or the isolation cap is reached, where they are
/// capped with Gregory patches).
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
        Rule::Smooth => !boundary && edges.len() == 4 && num_faces == 4,
        Rule::Dart => false,
        Rule::Crease => {
            if boundary {
                num_faces == 2 && edges.len() == 3
            } else {
                span_faces() == Some(2)
            }
        }
        Rule::Corner => inf_edges > 0 && span_faces() == Some(1),
        Rule::Unknown => false,
    }
}

/// The face across `edge` from `face`, or `None` when the edge is a wall of
/// the patch's neighborhood: a boundary, a non-manifold edge or an
/// infinitely sharp crease.
fn cross_edge(level: &Level, edge: Index, face: Index) -> Option<Index> {
    if is_edge_singular(level, edge, true) {
        return None;
    }
    let faces = level.edge_faces(edge as usize);
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
/// 4x4 control-vertex grid; phantom slots (beyond a boundary or an
/// infinitely sharp crease) are `INDEX_INVALID`. Also used by adaptive
/// refinement to decide which faces need isolation.
///
/// The grid is row-major with rows along `t`:
///
/// ```text
///   12 13 14 15
///    8  9 10 11        with the face being the center quad
///    4  5  6  7        (5, 6, 10, 9) = (c0, c1, c2, c3).
///    0  1  2  3
/// ```
pub(super) fn gather_regular_patch(level: &Level, face: usize) -> Option<[Index; 16]> {
    let fv = level.face_vertices(face);
    if fv.len() != 4 {
        return None;
    }
    let [c0, c1, c2, c3] = [fv[0], fv[1], fv[2], fv[3]];
    for corner in 0..4 {
        if !is_corner_regular(level, face, corner) {
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

    // Row/column neighbors across the face's four edges (missing across
    // boundaries and infinitely sharp creases).
    let a0 = cross_edge(level, fe[0], face); // below  (row 0/1)
    let a1 = cross_edge(level, fe[1], face); // right  (col 3)
    let a2 = cross_edge(level, fe[2], face); // above  (row 3)
    let a3 = cross_edge(level, fe[3], face); // left   (col 0)

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
/// boundary or infinitely sharp crease, and `None` when a non-quad face
/// makes the patch irregular.
fn gather_corner_cv(level: &Level, a: Option<Index>, c: Index, inner: Index) -> Option<Index> {
    if let Some(a) = a {
        if inner != INDEX_INVALID {
            if let Some(edge) = level.find_edge(c, inner) {
                if let Some(d) = cross_edge(level, edge, a) {
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
    // Which borders are actually missing (interior border slots are phantom
    // if and only if their whole row/column is).
    let row0 = cvs[1] == INDEX_INVALID;
    let row3 = cvs[13] == INDEX_INVALID;
    let col0 = cvs[4] == INDEX_INVALID;
    let col3 = cvs[7] == INDEX_INVALID;

    for _ in 0..2 {
        for j in 0..4usize {
            for i in 0..4usize {
                let slot = 4 * j + i;
                if cvs[slot] != INDEX_INVALID || weights[slot] == 0.0 {
                    continue;
                }
                // Reflect across the border the slot actually hangs off;
                // corner slots on two missing borders fold in two passes
                // (the reflections commute).
                let (a, b) = if j == 0 && row0 {
                    (4 + i, 8 + i) // reflect off rows 1, 2
                } else if j == 3 && row3 {
                    (8 + i, 4 + i)
                } else if i == 0 && col0 {
                    (4 * j + 1, 4 * j + 2) // reflect off cols 1, 2
                } else {
                    debug_assert!(i == 3 && col3);
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
