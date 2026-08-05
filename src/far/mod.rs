//! **Far** — Feature Adaptive Representation.
//!
//! Port of OpenSubdiv's `opensubdiv/far` layer: the public, client-facing
//! API of the library.
//!
//! * [`TopologyDescriptor`] — a lightweight, borrowed description of a
//!   mesh's topology (face-vertex lists, creases, corners, holes).
//! * [`TopologyRefinerFactory`] — validates a descriptor and constructs a
//!   [`TopologyRefiner`].
//! * [`TopologyRefiner`] — stores the hierarchy of refinement
//!   [levels](TopologyLevel) and performs [uniform
//!   refinement](TopologyRefiner::refine_uniform).
//! * [`PrimvarRefiner`] — interpolates primvar data (positions, colors, …)
//!   from one level to the next, and onto the limit surface — including
//!   face-varying channels (UVs) declared on the descriptor.
//! * [`StencilTable`] / [`StencilTableFactory`] — factorize the whole
//!   refinement (or limit evaluation) into flat per-vertex stencils on the
//!   base-level control vertices.

mod fvar;
mod primvar_refiner;
mod stencil_table;
mod topology_descriptor;
mod topology_refiner;

pub use crate::vtr::TopologyError as Error;
pub use fvar::FVarChannel;
pub use primvar_refiner::{Primvar, PrimvarRefiner};
pub use stencil_table::{Stencil, StencilTable, StencilTableFactory, StencilTableOptions};
pub use topology_descriptor::{FVarChannelDescriptor, TopologyDescriptor};
pub use topology_refiner::{
    TopologyLevel, TopologyRefiner, TopologyRefinerFactory, UniformOptions,
};
