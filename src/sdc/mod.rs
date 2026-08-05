//! **Sdc** — Subdivision Core.
//!
//! Port of OpenSubdiv's `opensubdiv/sdc` layer: the definitions of the
//! supported subdivision schemes, the standard subdivision options, the
//! (semi-sharp) creasing rules, and the scheme-specific subdivision and limit
//! masks.
//!
//! This layer is intentionally free of any topology *representation* — masks
//! are computed from small neighborhood descriptions supplied by higher
//! layers (see [`crate::vtr`] and [`crate::far`]).

mod crease;
mod options;
mod scheme;
mod types;

pub use crease::{Crease, Rule, SHARPNESS_INFINITE, SHARPNESS_SMOOTH};
pub use options::{
    CreasingMethod, FVarLinearInterpolation, Options, TriangleSubdivision, VtxBoundaryInterpolation,
};
pub use scheme::{
    EdgeNeighborhood, EdgeVertexMask, FaceVertexMask, Scheme, VertexNeighborhood, VertexVertexMask,
};
pub use types::{SchemeType, Split};
