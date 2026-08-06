//! Stencil tables (port of `opensubdiv/far/stencilTable.h` and
//! `opensubdiv/far/stencilTableFactory.h`).
//!
//! A *stencil* expresses one refined (or limit) vertex as a weighted sum of
//! the *base level* control vertices, factoring the whole chain of
//! subdivision steps into a single flat table. Once built, a
//! [`StencilTable`] can re-compute all refined/limit values for new control
//! values in a single pass — the core of OpenSubdiv's efficient re-posing of
//! animated meshes — without touching the topology again.

use super::primvar_refiner::{Primvar, PrimvarRefiner};
use super::topology_refiner::TopologyRefiner;
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
        for (i, out) in dst.iter_mut().enumerate().take(self.num_stencils()) {
            let stencil = self.stencil(i);
            let mut acc = out.clone();
            acc.clear();
            for (&idx, &w) in stencil.indices.iter().zip(stencil.weights) {
                acc.add_with_weight(&control_values[idx as usize], w);
            }
            *out = acc;
        }
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
    /// (`StencilTableFactory::Create`).
    ///
    /// Stencils are ordered by level (level 1 first, then level 2, …); with
    /// [`generate_intermediate_levels`](StencilTableOptions::generate_intermediate_levels)
    /// disabled only the last level is emitted, and with
    /// [`generate_control_verts`](StencilTableOptions::generate_control_verts)
    /// enabled identity stencils for the base vertices are prepended.
    pub fn create(refiner: &TopologyRefiner, options: StencilTableOptions) -> StencilTable {
        assert!(
            !refiner.is_adaptive(),
            "stencil tables require uniform refinement"
        );
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
    /// The same constraint as [`PrimvarRefiner::limit`] applies: for the
    /// Catmark scheme with non-quad base faces, the refiner must hold at
    /// least one level of refinement.
    pub fn create_limit(refiner: &TopologyRefiner) -> StencilTable {
        assert!(
            !refiner.is_adaptive(),
            "limit stencil tables require uniform refinement"
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
