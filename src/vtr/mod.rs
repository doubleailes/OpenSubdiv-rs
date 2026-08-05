//! **Vtr** — Vectorized Topology Representation.
//!
//! Port of OpenSubdiv's `opensubdiv/vtr` layer: an efficient intermediate
//! topological representation intended for internal use by the [`crate::far`]
//! layer.
//!
//! * [`Level`] — the complete topology of a mesh at one refinement level,
//!   stored as flat "CSR"-style index arrays (the *relations* between
//!   vertices, edges and faces), plus per-component sharpness and tags.
//! * [`Refinement`] — the mapping from a parent [`Level`] to the child
//!   [`Level`] produced by one step of uniform refinement.

mod level;
mod refinement;

pub use level::{Level, TopologyError};
pub use refinement::Refinement;
