//! # opensubdiv-rs
//!
//! A faithful Rust port of [Pixar's OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv)
//! subdivision-surface library.
//!
//! The port mirrors OpenSubdiv's layered architecture:
//!
//! * [`sdc`] — *Subdivision Core*: the lowest layer. Defines the subdivision
//!   schemes ([Bilinear](sdc::SchemeType::Bilinear), [Catmark](sdc::SchemeType::Catmark),
//!   [Loop](sdc::SchemeType::Loop)), the standard set of subdivision
//!   [options](sdc::Options), the semi-sharp creasing rules ([`sdc::Crease`])
//!   and the scheme-specific subdivision/limit *masks* (stencil weights).
//! * [`vtr`] — *Vectorized Topology Representation*: intermediate topology
//!   representation. A [`vtr::Level`] stores the complete topology of a mesh at
//!   a refinement level as flat index arrays, and a [`vtr::Refinement`] maps a
//!   parent level to its uniformly refined child level.
//! * [`far`] — *Feature Adaptive Representation*: the public client-facing
//!   layer. A [`far::TopologyRefiner`] is built from a
//!   [`far::TopologyDescriptor`] via [`far::TopologyRefinerFactory`], refined
//!   with [`far::TopologyRefiner::refine_uniform`], and primvar data is
//!   interpolated between levels (and onto the limit surface) with a
//!   [`far::PrimvarRefiner`].
//!
//! ## Example
//!
//! Uniformly subdivide a cube (the equivalent of OpenSubdiv's
//! `far_tutorial_1_1`):
//!
//! ```
//! use opensubdiv_rs::far::{
//!     PrimvarRefiner, TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
//! };
//! use opensubdiv_rs::sdc;
//!
//! let verts_per_face = [4usize; 6];
//! let face_verts: [u32; 24] = [
//!     0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6,
//!     6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
//! ];
//! let positions: [[f32; 3]; 8] = [
//!     [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
//!     [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
//!     [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
//!     [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
//! ];
//!
//! let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
//! let mut refiner = TopologyRefinerFactory::create(
//!     descriptor,
//!     sdc::SchemeType::Catmark,
//!     sdc::Options::default(),
//! )
//! .unwrap();
//! refiner.refine_uniform(UniformOptions::new(2));
//!
//! // Interpolate the positions up to the last refinement level.
//! let primvar_refiner = PrimvarRefiner::new(&refiner);
//! let mut src = positions.to_vec();
//! for level in 1..=refiner.max_level() {
//!     let mut dst = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
//!     primvar_refiner.interpolate(level, &src, &mut dst);
//!     src = dst;
//! }
//! assert_eq!(src.len(), 98);
//! ```
//!
//! ## Fidelity notes
//!
//! The subdivision rules (smooth / crease / corner masks, semi-sharp
//! fractional blending, `Uniform` and `Chaikin` crease subdivision, boundary
//! interpolation modes and limit masks) follow the reference implementation in
//! OpenSubdiv's `opensubdiv/sdc/*Scheme.h` and `opensubdiv/sdc/crease.{h,cpp}`.
//! Refined topology is identical in structure to OpenSubdiv's, though the
//! numbering of edges (and hence of edge child-vertices) may differ, as it
//! depends on traversal order internals.
//!
//! Face-varying channels (UVs with seams, all `FVarLinearInterpolation`
//! rules), stencil tables ([`far::StencilTable`]), patch tables
//! ([`far::PatchTable`], for parametric limit evaluation with derivatives,
//! including Gregory end caps at extraordinary vertices and on irregular
//! boundaries and infinitely sharp creases) and
//! feature-adaptive refinement
//! ([`far::TopologyRefiner::refine_adaptive`], with sparse levels and
//! mixed-depth patches) are supported. Not yet ported: the `Osd` GPU
//! layer. See the project README for the roadmap.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod far;
pub mod sdc;
pub mod vtr;

/// The type used for all topological indices (vertices, edges, faces).
///
/// Mirrors `OpenSubdiv::Far::Index` / `Vtr::Index` (a 32-bit integer).
pub type Index = u32;

/// Sentinel for an invalid/absent index (`Vtr::INDEX_INVALID`).
pub const INDEX_INVALID: Index = u32::MAX;

/// The type used for small per-component local indices
/// (`Vtr::LocalIndex`).
pub type LocalIndex = u16;
