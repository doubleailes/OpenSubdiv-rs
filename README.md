# OpenSubdiv-rs

[![crates.io](https://img.shields.io/crates/v/opensubdiv-rs.svg)](https://crates.io/crates/opensubdiv-rs)
[![docs.rs](https://img.shields.io/docsrs/opensubdiv-rs)](https://docs.rs/opensubdiv-rs)
[![CI](https://github.com/doubleailes/OpenSubdiv-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/doubleailes/OpenSubdiv-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/opensubdiv-rs.svg)](LICENSE)

A faithful Rust port of [Pixar's OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv)
subdivision-surface library.

OpenSubdiv-rs implements the same subdivision rules as OpenSubdiv — the RenderMan
/ industry-standard semantics for Catmull-Clark and Loop subdivision with
semi-sharp creasing and boundary interpolation — behind an API that mirrors
OpenSubdiv's layered architecture, translated to idiomatic, safe Rust
(no `unsafe`, no dependencies).

## Architecture

The port follows OpenSubdiv's layer structure one-to-one:

| Module | OpenSubdiv layer | Contents |
|--------|------------------|----------|
| `sdc`  | `opensubdiv/sdc` | Scheme types (`Bilinear`, `Catmark`, `Loop`), subdivision `Options`, semi-sharp `Crease` rules (`Uniform` and `Chaikin`), and the scheme-specific subdivision & limit **masks** |
| `vtr`  | `opensubdiv/vtr` | `Level` — flat-array topology of one refinement level (face-verts, face-edges, edge-verts, edge-faces, vert-faces, vert-edges, sharpness, tags); `Refinement` — one step of uniform quad/tri refinement |
| `far`  | `opensubdiv/far` | `TopologyDescriptor`, `TopologyRefinerFactory`, `TopologyRefiner` / `TopologyLevel`, `PrimvarRefiner` (`interpolate`, `interpolate_face_varying`, `limit`, `limit_face_varying`), `StencilTable` / `StencilTableFactory`, and `PatchTable` / `PatchMap` / `PatchParam` / `PtexIndices` |

## Installation

```sh
cargo add opensubdiv-rs
```

or, in `Cargo.toml`:

```toml
[dependencies]
opensubdiv-rs = "0.1"
```

The crate has no dependencies and contains no `unsafe` code
(`#![forbid(unsafe_code)]`).

## Usage

The equivalent of OpenSubdiv's `far_tutorial_1_1` — uniformly subdivide a cube:

```rust
use opensubdiv_rs::far::{
    PrimvarRefiner, TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

let verts_per_face = [4usize; 6];
let face_verts: [u32; 24] = [
    0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6,
    6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
];
let positions: [[f32; 3]; 8] = [
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
    [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
    [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
];

// Describe the topology and build a refiner.
let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
let mut refiner = TopologyRefinerFactory::create(
    descriptor,
    sdc::SchemeType::Catmark,
    sdc::Options::default(),
)
.expect("valid topology");

// Refine uniformly.
refiner.refine_uniform(UniformOptions::new(2));

// Interpolate vertex data level by level.
let primvar_refiner = PrimvarRefiner::new(&refiner);
let mut verts = positions.to_vec();
for level in 1..=refiner.max_level() {
    let mut refined = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
    primvar_refiner.interpolate(level, &verts, &mut refined);
    verts = refined;
}

// Evaluate limit-surface positions of the last level.
let mut limits = verts.clone();
primvar_refiner.limit(&verts, &mut limits);
```

Run the ported tutorial, which writes a refined cube as OBJ:

```sh
cargo run --example far_tutorial_1_1 > cube.obj
```

Primvar data is anything implementing the `far::Primvar` trait (the Rust
equivalent of OpenSubdiv's `Clear()` / `AddWithWeight()` protocol); it is
provided out of the box for `f32`, `f64`, `[f32; N]` and `[f64; N]`.

## Features

- **Schemes**: Bilinear, Catmull-Clark (arbitrary n-gons), Loop (triangle meshes)
- **Uniform refinement** to any depth, with full topology (all component
  relations) available at every level
- **Feature-adaptive refinement** (`refine_adaptive`): only irregular
  features — extraordinary vertices, non-quads, semi-sharp creases and
  irregular infinitely sharp features — are isolated, together with their
  one-ring support, until they resolve or reach the isolation level. Levels above the base are *sparse*: memory grows with
  the mesh's features, not with `4^level`, while the patch table evaluates
  the identical limit surface with far fewer patches
- **Semi-sharp creasing**: edge creases and vertex corners with fractional
  sharpness, `Uniform` and `Chaikin` crease subdivision, and the transitional
  blending of smooth/crease/corner masks across levels
- **Boundary interpolation**: `None`, `EdgeOnly`, `EdgeAndCorner`
- **Face-varying channels** (UVs, per-corner colors) with seams, refined in
  lockstep with the geometry, supporting all `FVarLinearInterpolation` rules
  (`All`, `None`, `CornersOnly`, `CornersPlus1`, `CornersPlus2`,
  `Boundaries`) and face-varying limit evaluation
- **Stencil tables** (`StencilTable` / `StencilTableFactory`): the whole
  refinement — or the limit evaluation — factorized into flat per-vertex
  stencils on the base control vertices, for fast re-posing of animated
  meshes via `update_values`
- **Limit-surface evaluation** of vertex positions (`PrimvarRefiner::limit`),
  including crease and corner limit rules
- **Patch tables** (`PatchTable` / `PatchMap` / `PatchParam` / `PtexIndices`):
  parametric evaluation of the limit surface — with first derivatives — at
  arbitrary `(ptex face, u, v)` locations, supporting **mixed-depth**
  (adaptive) hierarchies: each face is patched at the level where it becomes
  regular. Regular neighborhoods (including sharpened boundaries, pinned
  corners and infinitely sharp creases, which are treated as boundaries as
  with OpenSubdiv's `useInfSharpPatch`) become exact bicubic B-spline
  patches; every other manifold face at its isolation level — extraordinary
  vertices, irregular boundary and crease corners, sharp corners, darts —
  is capped with a **Gregory patch** (as OpenSubdiv's
  `ENDCAP_GREGORY_BASIS`), interpolating the corner limit points with C0
  boundaries and approximate G1 smoothness; only non-manifold neighborhoods
  and unsharpened (`VtxBoundaryInterpolation::None`) boundaries fall back to
  bilinear quads
- **Hole tags**, propagated through refinement
- Topology validation with typed errors (degenerate faces, out-of-range
  indices, non-triangular meshes for Loop, …)

## Fidelity notes

The numerical rules are ported directly from OpenSubdiv's reference
implementation (`sdc/catmarkScheme.h`, `sdc/loopScheme.h`, `sdc/crease.cpp`):
smooth/crease/corner masks, the Warren-Loop vertex weights, the
`(n² V + 4 Σ E + Σ D) / (n (n + 5))` Catmark limit stencil, etc. Refined
topology is structurally identical to OpenSubdiv's (child vertices ordered by
parent faces, then edges, then vertices); the *numbering* of edges — and hence
of edge child-vertices — can differ from OpenSubdiv's, as it depends on
internal traversal order.

Face-varying channels are represented as OpenSubdiv represents them
conceptually: a channel's values form a mesh of their own in which UV seams
are boundaries, refined with the same machinery as the geometry after
encoding the channel's linear-interpolation rule as sharpness. One
approximation is documented in `far::fvar`: `CornersPlus2` implements
junction and dart sharpening but not OpenSubdiv's additional concave-corner
analysis.

Patch tables extract exact bicubic B-spline patches wherever the limit
surface is polynomial; boundary, crease and corner patches are realized by
folding the phantom-point reflection `2a − b` into the basis weights, which
is mathematically equivalent to OpenSubdiv's boundary basis masks. Every
other manifold face at its isolation level is capped with a Gregory patch
built as OpenSubdiv's `GregoryConverter` builds it
(`far/catmarkPatchBuilder.cpp`): corners are classified by the *span* of
faces around them bounded by boundaries and infinitely sharp creases, and
the corner, edge and face points use OpenSubdiv's own coefficients for
smooth interior, smooth boundary/crease, sharp and dart corners. The test
suite verifies that the construction degenerates to the exact B-spline
patch on regular interior, boundary, pinned-corner and crease neighborhoods
(for arbitrary control data), pinning every coefficient. Only non-manifold
neighborhoods and unsharpened (`VtxBoundaryInterpolation::None`) boundaries
fall back to bilinear quads of the refined level; semi-sharp features that
are still unresolved at the isolation cap are capped as if smooth, as
OpenSubdiv does.

Feature-adaptive refinement follows OpenSubdiv's approach: faces needing
isolation are selected level by level with their one-ring support included
(the role of `Vtr::SparseSelector`), producing sparse levels, and the test
suite verifies that adaptive and uniform evaluation agree everywhere on the
same meshes. Transition-edge tagging for crack-free hardware tessellation
is not provided — parametric evaluation needs none, as adjacent patches at
different depths evaluate the same limit surface.

Not yet ported (roadmap):

- Single-crease patches for semi-sharp creases (`useSingleCreasePatch`)
  and non-manifold patches
- Loop-scheme (box-spline) patches and adaptive refinement for Loop
- Stencil tables for adaptively refined hierarchies
- The `Osd` GPU/compute back-ends
- `TRI_SUB_SMOOTH` triangle-subdivision option for Catmark

## Minimum supported Rust version

The MSRV is **1.85**, checked in CI. Raising it is treated as a
minor-version change.

## Changelog

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

This crate is distributed under the [MIT License](LICENSE).

It is a from-scratch Rust port based on the algorithms and public reference
implementation of [OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv),
Copyright © Pixar, released under the Modified Apache 2.0 License.
