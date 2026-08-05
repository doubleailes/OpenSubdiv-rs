# OpenSubdiv-rs

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
| `far`  | `opensubdiv/far` | `TopologyDescriptor`, `TopologyRefinerFactory`, `TopologyRefiner` / `TopologyLevel`, `PrimvarRefiner` (`interpolate`, `interpolate_face_varying`, `limit`, `limit_face_varying`), and `StencilTable` / `StencilTableFactory` |

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

Not yet ported (roadmap):

- Adaptive (feature-adaptive) refinement
- `Far::PatchTable` and patch-based limit evaluation (`EvalLimit`)
- The `Osd` GPU/compute back-ends
- `TRI_SUB_SMOOTH` triangle-subdivision option for Catmark

## License

This crate is distributed under the [MIT License](LICENSE).

It is a from-scratch Rust port based on the algorithms and public reference
implementation of [OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv),
Copyright © Pixar, released under the Modified Apache 2.0 License.
