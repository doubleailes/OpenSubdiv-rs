// Contains the cube mesh of OpenSubdiv's tutorials/far/tutorial_1_1
// (https://github.com/PixarAnimationStudios/OpenSubdiv) — Copyright 2013 Pixar.
// Licensed under the Tomorrow Open Source Technology License 1.0; see
// LICENSE-OPENSUBDIV at the root of this crate.

//! **Osd** — the data a GPU evaluator needs.
//!
//! OpenSubdiv's `opensubdiv/osd` layer adds no subdivision math: it uploads
//! tables built by `Far` to a device and runs small kernels on them, in a
//! version per API (CPU, OpenMP, TBB, CUDA, OpenCL, GLSL, HLSL, Metal). This
//! module ports the backend-independent part — the flat tables and the
//! patch basis the kernels evaluate — so any GPU API, such as `wgpu`, can
//! run them; it has no GPU dependency itself.
//!
//! Two kernels cover what a renderer needs:
//!
//! * **Stencil evaluation** (`EvalStencils`): one invocation per output
//!   vertex computes `dst[i] = Σ weights[j] · src[indices[j]]` over a
//!   [`StencilTable`](crate::far::StencilTable)'s flat
//!   [`offsets`](crate::far::StencilTable::offsets),
//!   [`indices`](crate::far::StencilTable::indices) and
//!   [`weights`](crate::far::StencilTable::weights). This refines a
//!   deforming cage each frame — uniformly, for drawing a refined level, or
//!   into the control buffer of a patch table.
//! * **Patch evaluation** (`EvalPatches`): one invocation per
//!   [`PatchCoord`] evaluates [`evaluate_patch_basis`] on the patch's
//!   control points, read from a [`GpuPatchTable`]'s buffers — the limit
//!   surface at arbitrary locations, with derivatives, for example on a
//!   tessellation grid per patch.
//!
//! ```
//! use opensubdiv_rs::far::{
//!     AdaptiveOptions, PatchMap, PatchTableFactory, StencilTableFactory, StencilTableOptions,
//!     TopologyDescriptor, TopologyRefinerFactory,
//! };
//! use opensubdiv_rs::osd::GpuPatchTable;
//! use opensubdiv_rs::sdc;
//!
//! // A cube, adaptively refined: B-spline patches and Gregory end caps.
//! let verts_per_face = [4; 6];
//! let face_verts = [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4];
//! let positions: [[f32; 3]; 8] = [
//!     [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
//!     [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
//! ];
//! let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
//! let mut refiner = TopologyRefinerFactory::create(
//!     descriptor, sdc::SchemeType::Catmark, sdc::Options::default(),
//! ).unwrap();
//! refiner.refine_adaptive(AdaptiveOptions::new(3));
//! let patches = PatchTableFactory::create(&refiner).unwrap();
//! let gpu = GpuPatchTable::new(&patches).unwrap();
//!
//! // Once per topology: one stencil table from the cage to every point the
//! // patches index, and the buffers to upload.
//! let stencils = StencilTableFactory::append_local_point_stencils(
//!     &StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls()),
//!     gpu.local_point_stencils(),
//! );
//! let (_arrays, _indices, _params) =
//!     (gpu.patch_array_words(), gpu.patch_index_buffer(), gpu.patch_param_words());
//!
//! // Per frame — here on the CPU, as the GPU kernels would: refine the cage
//! // into the patches' control points, then evaluate a location.
//! let mut points = vec![[0.0f32; 3]; stencils.num_stencils()];
//! stencils.update_values(&positions, &mut points);
//! let patch = PatchMap::new(&patches).find_patch(0, 0.25, 0.5).unwrap();
//! let (p, _du, _dv) = gpu.evaluate(&gpu.patch_coord(patch, 0.25, 0.5), &points);
//! assert!(p[2] > 0.0); // on the cube's front face, inside its +z cage face
//! ```

mod patch_basis;
mod patch_table;
mod types;

pub use patch_basis::{evaluate_patch_basis, PatchWeights};
pub use patch_table::GpuPatchTable;
pub use types::{patch_descriptor, PatchArray, PatchCoord, PatchParam};
