//! Stencil tables (port of `opensubdiv/far/stencilTable.h`,
//! `opensubdiv/far/stencilTableFactory.h` and the limit-stencil half of
//! them, `Far::LimitStencilTable` / `Far::LimitStencilTableFactory`).
//!
//! A *stencil* expresses one refined (or limit) vertex as a weighted sum of
//! the *base level* control vertices, factoring the whole chain of
//! subdivision steps into a single flat table. Once built, a
//! [`StencilTable`] can re-compute all refined/limit values for new control
//! values in a single pass — the core of OpenSubdiv's efficient re-posing of
//! animated meshes — without touching the topology again.
//!
//! Stencils are built for uniform and feature-adaptive hierarchies alike:
//! [`StencilTableFactory::create`] drives identity stencils through the
//! same [`PrimvarRefiner`] passes that interpolate vertex data, sparse
//! levels included.
//!
//! A *limit stencil* ([`LimitStencilTable`]) goes one step further: it
//! evaluates the limit surface — position and first derivatives — at an
//! arbitrary `(ptex face, u, v)` location directly from the base cage.
//! [`LimitStencilTableFactory`] obtains it by factorizing the basis weights
//! of the [`PatchTable`] patch covering the location through the stencils
//! of the patch's control vertices, so each evaluation becomes one sparse
//! dot product over cage vertices with no per-level intermediate buffers.
//! The stencils depend on the topology only and are reused across frames
//! of a deforming cage.

use super::patch_table::{PatchMap, PatchTable, PatchTableFactory};
use super::primvar_refiner::{Primvar, PrimvarRefiner};
use super::topology_refiner::TopologyRefiner;
use crate::vtr::TopologyError;
use crate::Index;

/// A single stencil: parallel slices of control-vertex indices and weights
/// (`Far::Stencil`).
#[derive(Debug, Clone, Copy)]
pub struct Stencil<'a> {
    /// Indices of the contributing control vertices.
    pub indices: &'a [Index],
    /// Weight of each control vertex, in the same order as `indices`.
    pub weights: &'a [f32],
}

impl Stencil<'_> {
    /// The number of control vertices contributing to this stencil
    /// (`GetSize`).
    pub fn size(&self) -> usize {
        self.indices.len()
    }
}

/// Options controlling [`StencilTableFactory`]
/// (`Far::StencilTableFactory::Options`).
#[derive(Debug, Clone, Copy)]
pub struct StencilTableOptions {
    /// Generate stencils for the vertices of all intermediate levels, not
    /// just the last one (`generateIntermediateLevels`; OpenSubdiv default:
    /// true).
    pub generate_intermediate_levels: bool,
    /// Prepend identity stencils for the control (base level) vertices
    /// themselves (`generateControlVerts`; OpenSubdiv default: false).
    pub generate_control_verts: bool,
}

impl Default for StencilTableOptions {
    fn default() -> Self {
        Self {
            generate_intermediate_levels: true,
            generate_control_verts: false,
        }
    }
}

impl StencilTableOptions {
    /// Options producing one stencil per *patch control value*: identity
    /// stencils for the base vertices followed by the stencils of every
    /// refined level, so that stencil `i` evaluates control value `i` of the
    /// buffer a [`PatchTable`] is evaluated against
    /// ([`PatchTable::num_control_values`]). This is the table
    /// [`LimitStencilTableFactory`] factorizes patches through.
    pub fn for_patch_controls() -> Self {
        Self {
            generate_intermediate_levels: true,
            generate_control_verts: true,
        }
    }
}

/// Table of subdivision stencils (`Far::StencilTable`).
///
/// Stencils are stored in flat arrays; stencil `i` covers the half-open
/// range `offsets[i]..offsets[i+1]` of `indices`/`weights`.
#[derive(Debug, Clone)]
pub struct StencilTable {
    num_control_vertices: usize,
    offsets: Vec<u32>,
    indices: Vec<Index>,
    weights: Vec<f32>,
}

impl StencilTable {
    /// The number of stencils in the table (`GetNumStencils`).
    pub fn num_stencils(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The number of control vertices indexed by the table
    /// (`GetNumControlVertices`).
    pub fn num_control_vertices(&self) -> usize {
        self.num_control_vertices
    }

    /// The `i`-th stencil (`GetStencil`).
    pub fn stencil(&self, i: usize) -> Stencil<'_> {
        let (s, e) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        Stencil {
            indices: &self.indices[s..e],
            weights: &self.weights[s..e],
        }
    }

    /// Update every stencil value from the given control values
    /// (`UpdateValues`): `dst[i]` receives the weighted sum of
    /// `control_values` described by stencil `i`.
    pub fn update_values<T: Primvar>(&self, control_values: &[T], dst: &mut [T]) {
        assert!(
            control_values.len() >= self.num_control_vertices,
            "control buffer too small"
        );
        assert!(dst.len() >= self.num_stencils(), "dst buffer too small");
        apply_weights(
            &self.offsets,
            &self.indices,
            &self.weights,
            control_values,
            dst,
        );
    }
}

/// Accumulate `dst[i] = Σ weights · control_values` over the flat stencil
/// arrays `offsets` / `indices` / `weights`.
fn apply_weights<T: Primvar>(
    offsets: &[u32],
    indices: &[Index],
    weights: &[f32],
    control_values: &[T],
    dst: &mut [T],
) {
    for (i, out) in dst.iter_mut().enumerate().take(offsets.len() - 1) {
        let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
        let mut acc = out.clone();
        acc.clear();
        for (&idx, &w) in indices[s..e].iter().zip(&weights[s..e]) {
            acc.add_with_weight(&control_values[idx as usize], w);
        }
        *out = acc;
    }
}

/// A sparse weight vector over the base-level control vertices, driven
/// through [`PrimvarRefiner`] like any other primvar: interpolating identity
/// stencils level by level *factorizes* the subdivision into stencils —
/// exactly the approach of `Far::StencilTableFactory`.
#[derive(Debug, Clone, Default)]
struct StencilAccumulator {
    entries: Vec<(Index, f32)>,
}

impl Primvar for StencilAccumulator {
    fn clear(&mut self) {
        self.entries.clear();
    }
    fn add_with_weight(&mut self, src: &Self, weight: f32) {
        for &(index, w) in &src.entries {
            match self.entries.iter_mut().find(|(i, _)| *i == index) {
                Some((_, acc)) => *acc += w * weight,
                None => self.entries.push((index, w * weight)),
            }
        }
    }
}

/// Factory constructing [`StencilTable`]s from a [`TopologyRefiner`]
/// (`Far::StencilTableFactory`).
pub struct StencilTableFactory;

impl StencilTableFactory {
    /// Instantiate a [`StencilTable`] from a refined [`TopologyRefiner`]
    /// (`StencilTableFactory::Create`), uniformly or feature-adaptively
    /// refined.
    ///
    /// Stencils are ordered by level (level 1 first, then level 2, …); with
    /// [`generate_intermediate_levels`](StencilTableOptions::generate_intermediate_levels)
    /// disabled only the last level is emitted, and with
    /// [`generate_control_verts`](StencilTableOptions::generate_control_verts)
    /// enabled identity stencils for the base vertices are prepended. With
    /// both enabled ([`StencilTableOptions::for_patch_controls`]) the table
    /// has one stencil per control value of the refiner's [`PatchTable`],
    /// in the same order, so [`StencilTable::update_values`] fills the
    /// buffer that patches are evaluated against in one pass.
    ///
    /// For an adaptively refined hierarchy the levels are sparse: each
    /// level's stencils cover exactly the vertices that level holds, with
    /// the values [`PrimvarRefiner::interpolate`] computes for them
    /// (including its fringe fallbacks for support-only vertices, which no
    /// patch references).
    pub fn create(refiner: &TopologyRefiner, options: StencilTableOptions) -> StencilTable {
        let num_control = refiner.level(0).num_vertices();
        let primvar = PrimvarRefiner::new(refiner);

        let mut collected: Vec<StencilAccumulator> = Vec::new();
        if options.generate_control_verts {
            collected.extend((0..num_control).map(identity_stencil));
        }

        let mut src: Vec<StencilAccumulator> = (0..num_control).map(identity_stencil).collect();
        for level in 1..=refiner.max_level() {
            let mut dst = vec![StencilAccumulator::default(); refiner.level(level).num_vertices()];
            primvar.interpolate(level, &src, &mut dst);
            if options.generate_intermediate_levels || level == refiner.max_level() {
                collected.extend(dst.iter().cloned());
            }
            src = dst;
        }

        build_table(num_control, collected)
    }

    /// Instantiate a table of *limit* stencils: one stencil per vertex of
    /// the refiner's last level, evaluating that vertex's limit position
    /// directly from the base-level control vertices
    /// (`LimitStencilTableFactory`, for the vertices of the last level).
    ///
    /// The same constraints as [`PrimvarRefiner::limit`] apply: the refiner
    /// must be uniformly refined (the last level of an adaptive hierarchy
    /// is sparse, so its fringe vertices lack the neighborhoods the limit
    /// masks need — build a [`LimitStencilTableFactory`] table over the
    /// patches instead), and for the Catmark scheme with non-quad base
    /// faces it must hold at least one level of refinement.
    pub fn create_limit(refiner: &TopologyRefiner) -> StencilTable {
        assert!(
            !refiner.is_adaptive(),
            "vertex limit stencils require uniform refinement; use LimitStencilTableFactory for adaptive refiners"
        );
        let num_control = refiner.level(0).num_vertices();
        let primvar = PrimvarRefiner::new(refiner);

        let mut src: Vec<StencilAccumulator> = (0..num_control).map(identity_stencil).collect();
        for level in 1..=refiner.max_level() {
            let mut dst = vec![StencilAccumulator::default(); refiner.level(level).num_vertices()];
            primvar.interpolate(level, &src, &mut dst);
            src = dst;
        }

        let mut limits = vec![StencilAccumulator::default(); src.len()];
        primvar.limit(&src, &mut limits);

        build_table(num_control, limits)
    }
}

fn identity_stencil(v: usize) -> StencilAccumulator {
    StencilAccumulator {
        entries: vec![(v as Index, 1.0)],
    }
}

fn build_table(num_control_vertices: usize, stencils: Vec<StencilAccumulator>) -> StencilTable {
    let total: usize = stencils.iter().map(|s| s.entries.len()).sum();
    let mut table = StencilTable {
        num_control_vertices,
        offsets: Vec::with_capacity(stencils.len() + 1),
        indices: Vec::with_capacity(total),
        weights: Vec::with_capacity(total),
    };
    table.offsets.push(0);
    for mut stencil in stencils {
        // Sort entries by control index for deterministic output.
        stencil.entries.sort_by_key(|&(i, _)| i);
        for (index, weight) in stencil.entries {
            table.indices.push(index);
            table.weights.push(weight);
        }
        table.offsets.push(table.indices.len() as u32);
    }
    table
}

// ----------------------------------------------------------------------
//  Limit stencils at arbitrary parametric locations
// ----------------------------------------------------------------------

/// A set of parametric locations on one ptex face at which limit stencils
/// are requested (`Far::LimitStencilTableFactory::LocationArray`).
///
/// `u` and `v` are parallel slices of coordinates in the parametric space
/// of `ptex_face` — the same space [`PatchMap::find_patch`] and
/// [`PatchTable::evaluate`] use.
#[derive(Debug, Clone, Copy)]
pub struct LocationArray<'a> {
    /// The ptex face the locations lie on (see
    /// [`PtexIndices`](super::PtexIndices)).
    pub ptex_face: Index,
    /// The `u` coordinate of each location.
    pub u: &'a [f32],
    /// The `v` coordinate of each location, parallel to `u`.
    pub v: &'a [f32],
}

impl<'a> LocationArray<'a> {
    /// Locations `(u[i], v[i])` on `ptex_face`.
    ///
    /// # Panics
    /// Panics if `u` and `v` differ in length.
    pub fn new(ptex_face: Index, u: &'a [f32], v: &'a [f32]) -> Self {
        assert_eq!(
            u.len(),
            v.len(),
            "u and v coordinate arrays must be parallel"
        );
        Self { ptex_face, u, v }
    }

    /// The number of locations in the array (`numLocations`).
    pub fn len(&self) -> usize {
        self.u.len()
    }

    /// Does the array hold no locations?
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }
}

/// A single limit stencil (`Far::LimitStencil`): weights on the base-level
/// control vertices for the limit position and its first derivatives with
/// respect to the ptex-face `(u, v)` coordinates.
#[derive(Debug, Clone, Copy)]
pub struct LimitStencil<'a> {
    /// Indices of the contributing control vertices.
    pub indices: &'a [Index],
    /// Weight of each control vertex for the limit position.
    pub weights: &'a [f32],
    /// Weight of each control vertex for the derivative along `u`
    /// (`GetDuWeights`).
    pub du_weights: &'a [f32],
    /// Weight of each control vertex for the derivative along `v`
    /// (`GetDvWeights`).
    pub dv_weights: &'a [f32],
}

impl LimitStencil<'_> {
    /// The number of control vertices contributing to this stencil
    /// (`GetSize`).
    pub fn size(&self) -> usize {
        self.indices.len()
    }
}

/// Table of limit stencils (`Far::LimitStencilTable`): each stencil
/// evaluates the limit surface at one parametric location — position and
/// first derivatives — directly from the base-level control vertices.
///
/// The position weights form an ordinary [`StencilTable`]
/// ([`point_stencils`](Self::point_stencils)); the derivative weights are
/// stored alongside, over the same control-vertex indices.
#[derive(Debug, Clone)]
pub struct LimitStencilTable {
    points: StencilTable,
    du_weights: Vec<f32>,
    dv_weights: Vec<f32>,
}

impl LimitStencilTable {
    /// The number of stencils in the table (`GetNumStencils`): one per
    /// requested location, in request order.
    pub fn num_stencils(&self) -> usize {
        self.points.num_stencils()
    }

    /// The number of control vertices indexed by the table
    /// (`GetNumControlVertices`): the base level's vertex count.
    pub fn num_control_vertices(&self) -> usize {
        self.points.num_control_vertices()
    }

    /// The `i`-th limit stencil (`GetLimitStencil`).
    pub fn stencil(&self, i: usize) -> LimitStencil<'_> {
        let (s, e) = (
            self.points.offsets[i] as usize,
            self.points.offsets[i + 1] as usize,
        );
        LimitStencil {
            indices: &self.points.indices[s..e],
            weights: &self.points.weights[s..e],
            du_weights: &self.du_weights[s..e],
            dv_weights: &self.dv_weights[s..e],
        }
    }

    /// The position stencils alone, as a plain [`StencilTable`].
    pub fn point_stencils(&self) -> &StencilTable {
        &self.points
    }

    /// Evaluate the limit position of every stencil from the given control
    /// values (`UpdateValues`): `dst[i]` receives the limit position at
    /// location `i`.
    pub fn update_values<T: Primvar>(&self, control_values: &[T], dst: &mut [T]) {
        self.points.update_values(control_values, dst);
    }

    /// Evaluate the first derivatives of every stencil from the given
    /// control values (`UpdateDerivs`): `du[i]` and `dv[i]` receive the
    /// derivatives of the limit surface along the ptex-face `u` and `v`
    /// directions at location `i`.
    pub fn update_derivs<T: Primvar>(&self, control_values: &[T], du: &mut [T], dv: &mut [T]) {
        assert!(
            control_values.len() >= self.num_control_vertices(),
            "control buffer too small"
        );
        assert!(
            du.len() >= self.num_stencils() && dv.len() >= self.num_stencils(),
            "derivative buffers too small"
        );
        apply_weights(
            &self.points.offsets,
            &self.points.indices,
            &self.du_weights,
            control_values,
            du,
        );
        apply_weights(
            &self.points.offsets,
            &self.points.indices,
            &self.dv_weights,
            control_values,
            dv,
        );
    }
}

/// Factory constructing [`LimitStencilTable`]s
/// (`Far::LimitStencilTableFactory`).
pub struct LimitStencilTableFactory;

impl LimitStencilTableFactory {
    /// Build limit stencils at the given parametric locations
    /// (`LimitStencilTableFactory::Create`), for a uniformly or adaptively
    /// refined `refiner`.
    ///
    /// Every location is covered by a patch of the refiner's [`PatchTable`]
    /// (bicubic B-spline, single-crease, Gregory, box-spline, Gregory
    /// triangle or linear alike); the patch's basis weights at the
    /// location — point and first derivatives — are factorized through the
    /// stencils of its control vertices, which puts the limit evaluation
    /// directly on the base cage. The resulting stencils are listed in the
    /// order of `locations` (arrays in order, locations within each array
    /// in order) and reproduce [`PatchTable::evaluate`] over the
    /// concatenated level buffer to floating-point round-off. Gregory and
    /// Gregory-triangle end caps need no separate treatment: their derived
    /// points are themselves stencils on the refined vertices and fold
    /// through the same factorization (the role of OpenSubdiv's
    /// `AppendLocalPointStencilTable`).
    ///
    /// `cv_stencils` and `patch_table` may be supplied to reuse tables
    /// already built for `refiner`; either is built on demand otherwise.
    /// `cv_stencils` must hold one stencil per control value of the patch
    /// table, in order — the table [`StencilTableFactory::create`] builds
    /// with [`StencilTableOptions::for_patch_controls`].
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::PatchesRequireRefinement`] when a patch
    /// table has to be built for an unrefined mesh with non-quad faces, and
    /// [`TopologyError::LimitLocationInHole`] when a location lies in a
    /// hole, where the limit surface is undefined.
    ///
    /// # Panics
    ///
    /// Panics when the supplied `cv_stencils` has fewer stencils than the
    /// patch table has control values.
    pub fn create(
        refiner: &TopologyRefiner,
        locations: &[LocationArray<'_>],
        cv_stencils: Option<&StencilTable>,
        patch_table: Option<&PatchTable>,
    ) -> Result<LimitStencilTable, TopologyError> {
        let owned_patches;
        let patch_table = match patch_table {
            Some(table) => table,
            None => {
                owned_patches = PatchTableFactory::create(refiner)?;
                &owned_patches
            }
        };
        let owned_stencils;
        let cv_stencils = match cv_stencils {
            Some(table) => table,
            None => {
                owned_stencils =
                    StencilTableFactory::create(refiner, StencilTableOptions::for_patch_controls());
                &owned_stencils
            }
        };
        assert!(
            cv_stencils.num_stencils() >= patch_table.num_control_values(),
            "control-vertex stencil table has {} stencils but the patch table has {} control values; \
             build it with StencilTableOptions::for_patch_controls()",
            cv_stencils.num_stencils(),
            patch_table.num_control_values()
        );

        let map = PatchMap::new(patch_table);
        let num_locations: usize = locations.iter().map(LocationArray::len).sum();
        let mut offsets = Vec::with_capacity(num_locations + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        let mut du_weights = Vec::new();
        let mut dv_weights = Vec::new();
        offsets.push(0u32);

        // Per location: weights on the base cage for (point, du, dv).
        let mut acc: Vec<(Index, [f32; 3])> = Vec::new();
        for array in locations {
            assert_eq!(
                array.u.len(),
                array.v.len(),
                "u and v coordinate arrays must be parallel"
            );
            for (&u, &v) in array.u.iter().zip(array.v) {
                let patch = map.find_patch(array.ptex_face as usize, u, v).ok_or(
                    TopologyError::LimitLocationInHole {
                        ptex_face: array.ptex_face,
                        location: offsets.len() - 1,
                    },
                )?;
                let basis = patch_table.evaluate_basis(patch, u, v);

                acc.clear();
                for (k, &cv) in basis.indices.iter().enumerate() {
                    let stencil = cv_stencils.stencil(cv as usize);
                    let factors = [basis.weights[k], basis.du_weights[k], basis.dv_weights[k]];
                    for (&index, &w) in stencil.indices.iter().zip(stencil.weights) {
                        let contribution = factors.map(|f| f * w);
                        match acc.iter_mut().find(|(i, _)| *i == index) {
                            Some((_, sum)) => {
                                for (s, c) in sum.iter_mut().zip(contribution) {
                                    *s += c;
                                }
                            }
                            None => acc.push((index, contribution)),
                        }
                    }
                }
                acc.sort_by_key(|&(i, _)| i);
                for &(index, [w, du, dv]) in &acc {
                    indices.push(index);
                    weights.push(w);
                    du_weights.push(du);
                    dv_weights.push(dv);
                }
                offsets.push(indices.len() as u32);
            }
        }

        Ok(LimitStencilTable {
            points: StencilTable {
                num_control_vertices: cv_stencils.num_control_vertices(),
                offsets,
                indices,
                weights,
            },
            du_weights,
            dv_weights,
        })
    }
}
