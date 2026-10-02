# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`opensubdiv-rs` is a faithful, from-scratch Rust port of Pixar's OpenSubdiv (Sdc / Vtr / Far layers; the `Osd` GPU layer is not ported). It has **no dependencies** and **no `unsafe`** (`#![forbid(unsafe_code)]`); keep it that way. `#![warn(missing_docs)]` and `#![deny(rustdoc::broken_intra_doc_links)]` are on, so every public item needs a doc comment and intra-doc links must resolve.

## Commands

CI (`.github/workflows/ci.yml`) runs these on stable, and all must pass:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

It also builds on the MSRV, **Rust 1.85** (`cargo +1.85 build --all-targets`). Don't use newer language or std features. Raising the MSRV counts as a minor-version change.

Run part of the suite:

```sh
cargo test --test patch                 # one integration-test file (tests/patch.rs)
cargo test --test fvar seam             # tests in one file whose names contain "seam"
cargo test --lib gregory                # unit tests inside src/ matching "gregory"
cargo test --doc                        # doctests (lib.rs example)
cargo run --example far_tutorial_1_1 > cube.obj
```

`tests/single_crease.rs` is the slowest file (~7 s in debug).

## Architecture

The modules mirror OpenSubdiv's layers one-to-one, and the doc comments name the OpenSubdiv source file each Rust file ports (for example `far/catmarkPatchBuilder.cpp` → `src/far/gregory.rs`). When you change the numerical behaviour, check it against that reference file.

- **`sdc`**: pure math with no topology storage. Holds `SchemeType`, `Options`, the `Crease` rules (Uniform/Chaikin, the semi-sharp `Rule` blending) and `Scheme`, which computes subdivision and limit masks from small neighborhood traits (`VertexNeighborhood`, `EdgeNeighborhood`) that higher layers implement.
- **`vtr`**: `Level` is one refinement level stored as flat CSR-style relation arrays (face-verts, face-edges, edge-verts, edge-faces, vert-faces, vert-edges) plus sharpness and tags. `Refinement` maps a parent level to its child. In the child, vertices are ordered as face-children first, then edge-children, then vertex-children. Edge numbering may differ from OpenSubdiv's. `TopologyError` lives here and is re-exported as `far::Error`.
- **`far`**: the public API. The data flows like this:
  1. `TopologyDescriptor`: borrowed input (faces, creases, corners, holes, face-varying channels).
  2. `TopologyRefinerFactory::create`: validates the descriptor and builds level 0.
  3. `TopologyRefiner::refine_uniform` or `refine_adaptive`.
  4. One of three consumers:
     - `PrimvarRefiner`: level-by-level interpolation and limit evaluation of anything implementing the `Primvar` trait.
     - `StencilTableFactory`: factorizes the refinement into stencils on the base cage.
     - `PatchTableFactory`: builds parametric patches, located with `PatchMap` and `PtexIndices`. `LimitStencilTableFactory` combines patch basis weights with control-vertex stencils.

### Cross-cutting design points

- **Adaptive refinement uses sparse levels.** At each step `refine_adaptive` selects the faces that need isolation. A Catmark face needs isolation when `patch_table::gather_regular_patch` fails (and, if enabled, so does `single_crease_patch`). A Loop face needs it when `loop_patch::gather_regular_patch` fails. The selection is expanded to its one-ring support with `Refinement::expand_selection`, and only the included faces are refined (`refine_included`). The refiner records each step's selection; `face_is_candidate` and `face_is_selected` tell the patch table at which level each face gets patched (mixed depth). `refine_adaptive_selected` only restricts which base faces are inspected at level 0; candidacy does the rest below it. The "is this face regular?" predicates in the patch modules therefore also drive refinement. If you change one, you change both what gets isolated and what gets patched.
- **Face-varying channels are value meshes** (`far/fvar.rs`). Each channel is its own `Level` whose vertices are (vertex, value index) pairs, so seams become boundaries. The `FVarLinearInterpolation` rule is encoded as sharpness on that mesh, and the mesh is refined with the same `Refinement` code. In adaptive mode, the one inclusion mask computed on the geometry is passed to every channel so their faces stay in lockstep.
- **Stencil tables reuse `PrimvarRefiner`.** `StencilTableFactory` pushes identity stencils through the same interpolation passes (sparse levels included). The stencil order must match the order in which `PrimvarRefiner::interpolate` produces vertices. `StencilTableOptions::for_patch_controls` produces the patch table's concatenated control buffer.
- **Patch types.**
  - Catmark: regular bicubic B-spline (boundaries, creases and pinned corners fold the phantom-point reflection `2a − b` into the basis weights), single-crease, Gregory basis end caps (`gregory.rs`, 20 points stored as sparse stencils), and `Quads` for unsharpened boundaries and Bilinear.
  - Loop (`loop_patch.rs`): quartic box-spline on 12 control vertices (parallelogram phantom reflection `a + b − c`), Gregory triangles (18 points), and linear `Triangles`.
  - Non-manifold edges and vertices are made infinitely sharp, so they patch like sharp features.
- **Documented deviations from OpenSubdiv** are listed in the README's "Fidelity notes": Loop Gregory-triangle coefficients, `CornersPlus2` concave-corner analysis, single-crease sharpness capping, and no transition-edge tagging. Keep that section, the crate-level docs in `src/lib.rs` and the module docs in sync when behaviour changes.

### Tests

The integration tests in `tests/` are grouped by feature (catmark, loop, fvar, fvar_patch, adaptive, patch, endcaps, single_crease, loop_patch, stencil, nonmanifold, selected_faces). Each file defines its own fixtures (such as the far-tutorial cube) and its own `assert_close` helper; there is no shared `common` module. The main checks are:

- The Gregory and Loop caps reduce to the exact B-spline or box-spline patch on regular neighborhoods, for arbitrary control data.
- Adaptive and uniform evaluation agree on the same mesh.
- `LimitStencilTable` reproduces `PatchTable::evaluate`.

New features are expected to come with tests of the same kind.

## Releases and changelog

- `CHANGELOG.md` follows Keep a Changelog. Add user-visible changes under `## [Unreleased]`, linking the GitHub issue as the existing entries do.
- Version bumps are separate commits ("Bump version to X.Y.Z").
- Pushing a tag triggers `.github/workflows/release.yml`. It checks that the tag (with any leading `v` removed) equals the `Cargo.toml` version, then runs `cargo publish`.
