# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The Catmark "smooth triangle" rule, `TriangleSubdivision::Smooth`
  (OpenSubdiv's `TRI_SUB_SMOOTH`, USD's `triangleSubdivisionRule =
  "smooth"`), which was accepted but silently refined as `Catmark`
  ([#14](https://github.com/doubleailes/OpenSubdiv-rs/issues/14)). As in
  OpenSubdiv, an interior edge with a triangle on either side now gives
  each incident triangle's face point 0.470 instead of 1/4, averages the
  two face weights and leaves the remainder to the end vertices; face
  points, vertex points, boundary edges and the `Catmark` rule are
  unchanged. Face-varying channels and stencil tables follow the rule
  through the shared refinement masks. `Scheme::triangle_subdivision`
  reports the rule in effect.

- Stencil tables for adaptively refined hierarchies and limit stencils
  over the patch table
  ([#13](https://github.com/doubleailes/OpenSubdiv-rs/issues/13)).
  `StencilTableFactory::create` now accepts feature-adaptive refiners,
  producing stencils for the vertices of every sparse level in the order
  `PrimvarRefiner::interpolate` computes them;
  `StencilTableOptions::for_patch_controls` names the options whose table
  fills a `PatchTable`'s concatenated control buffer in one
  `update_values` pass. The new `LimitStencilTableFactory::create` builds
  a `LimitStencilTable` of limit stencils — position and first derivatives
  — at arbitrary `(ptex face, u, v)` locations given as `LocationArray`s,
  by factorizing the covering patch's basis weights (B-spline,
  single-crease, Gregory, box-spline, Gregory triangle or linear) through
  the control-vertex stencils down to the base cage; it reproduces
  `PatchTable::evaluate` to floating-point round-off, and accepts
  prebuilt stencil and patch tables for reuse. `TopologyError` gained the
  variants `LimitLocationInHole` for locations no patch covers and
  `PtexFaceOutOfRange` for location arrays naming a ptex face the patch
  table does not have.

- Patches and feature-adaptive refinement for the Loop scheme
  ([#12](https://github.com/doubleailes/OpenSubdiv-rs/issues/12)).
  `PatchTableFactory::create` now builds patch tables for Loop refiners:
  regular triangles — interior valence-6 corners, regular boundary and
  infinitely sharp crease vertices, and pinned corners — become exact
  quartic box-spline patches on 12 control vertices (`PatchType::Loop`),
  every other face at its isolation level a Gregory triangle end cap on 18
  derived points (`PatchType::GregoryTriangle`, OpenSubdiv's
  `GREGORY_TRIANGLE`), and faces on unsharpened boundaries linear
  triangles (`PatchType::Triangles`). `TopologyRefiner::refine_adaptive`
  isolates Loop's irregular features exactly as it does for Catmark, with
  sparse levels; `PatchParam` carries a triangle's parametric sub-domain
  (`PatchParam::is_triangle_rotated` for the inverted central children),
  `PatchMap::find_patch` locates triangles by ptex face and `(u, v)`, and
  `PtexIndices` maps every triangular base face to one ptex face.
- `vtr::Refinement::refine_selected` supports sparse triangular splits;
  `Refinement::expand_selection` and `Refinement::refine_included` expose
  the one-ring expansion and the refinement of an explicit face mask.
- `AdaptiveOptions::MAX_ISOLATION_LEVEL` (10, as in OpenSubdiv):
  `refine_adaptive` clamps deeper requests to it, keeping patch depths
  within the range their parametric scale `2^depth` is computed for.
- Single-crease patches for semi-sharp creases (OpenSubdiv's
  `useSingleCreasePatch`), enabled with
  `AdaptiveOptions::with_single_crease_patch(true)`
  ([#11](https://github.com/doubleailes/OpenSubdiv-rs/issues/11)). Regular
  faces bounded on one side by a straight semi-sharp crease of uniform
  sharpness are no longer isolated by `refine_adaptive`: the patch table
  covers each with one exact `PatchType::Regular` patch whose basis across
  the crease is the crease's limit profile, and
  `PatchTable::single_crease_sharpness` reports its sharpness. Along such a
  crease the patch count drops from `4^level` per face to one.
- Patches around non-manifold features
  ([#10](https://github.com/doubleailes/OpenSubdiv-rs/issues/10)): faces
  incident non-manifold edges or vertices are now patched over their own
  manifold span — regular B-spline patches where the span is regular,
  Gregory caps elsewhere — instead of falling back to bilinear quads.
  `PatchType::Quads` now only appears for irregular faces on unsharpened
  (`VtxBoundaryInterpolation::None`) boundaries and the Bilinear scheme.
- `vtr::Level::is_vertex_non_manifold`.

### Changed

- `sdc::EdgeNeighborhood` gained the public field `face_vertex_counts`
  (the number of vertices of each of the first two incident faces), which
  the smooth-triangle rule inspects; code building it with a struct
  literal must set the field.
- `StencilTableFactory::create` no longer panics on adaptively refined
  refiners. `StencilTableFactory::create_limit` (the limit of the last
  level's vertices) still requires uniform refinement, as its sparse last
  level lacks complete neighborhoods; its panic message now points to
  `LimitStencilTableFactory`.
- `PatchParam` gained the public field `triangular` (true for the Loop
  scheme's patches); `PatchParam::is_triangle_rotated` is false for every
  quad patch, including those in the third quadrant of their ptex face.
  Code building `PatchParam` with a struct literal must set the field.
- The four children of a triangle under Loop refinement are now ordered
  and oriented as OpenSubdiv's `TriRefinement` orders them — the corner
  children `(v0, e0, e2)`, `(e0, v1, e1)`, `(e2, e1, v2)` keeping the
  parent's orientation and the central child `(e1, e2, e0)` inverted —
  instead of `(v_i, e_i, e_{i-1})` corner children; the face-vertex order
  of refined Loop levels changes accordingly (vertex numbering does not).
- `AdaptiveOptions` gained the public field `use_single_crease_patch`
  (default `false`); code building it with a struct literal must set it or
  use `AdaptiveOptions::new`.
- Non-manifold edges are made infinitely sharp when the base level is
  built, and non-manifold vertices infinitely sharp unless they lie on a
  crease of exactly two non-manifold edges bounding every fan of faces
  around them, as OpenSubdiv does
  (`applyComponentTagsAndBoundarySharpness`). This changes refinement and
  limit positions around non-manifold features, which previously followed
  the smooth rules.

### Fixed

- Face-varying channels of an adaptively refined mesh are refined with the
  geometry's expanded face mask instead of expanding the selection on
  their own value meshes. Across seams the value mesh shares fewer
  vertices, so it included fewer support faces than the geometry, and the
  next isolation step panicked on the mismatched face counts.

### Removed

- `TopologyError::LoopPatchesNotSupported`: Loop patch tables are now
  supported, so the variant no longer had a use.

## [0.2.0] - 2026-09-30

### Added

- Gregory end caps for every irregular manifold face, not only smooth
  interior extraordinary vertices: irregular boundary corners, corners on
  infinitely sharp creases, sharp (pinned or multiply creased) corners,
  darts and smooth boundary corners are now capped with Gregory patches
  built as OpenSubdiv's `GregoryConverter` builds them, instead of falling
  back to bilinear quads
  ([#8](https://github.com/doubleailes/OpenSubdiv-rs/issues/8)).
  `PatchType::Quads` now only appears for non-manifold neighborhoods,
  irregular faces on unsharpened (`VtxBoundaryInterpolation::None`)
  boundaries, and the Bilinear scheme.
- Infinitely sharp creases are treated as boundaries by regular patches
  (OpenSubdiv's `useInfSharpPatch`): faces whose corners are regular crease
  vertices, or fully creased corners, become exact B-spline patches with the
  phantom-point reflection across the crease, and feature-adaptive
  refinement no longer isolates such regular creases to the cap.

### Changed

- Smooth interior extraordinary corners of Gregory caps now use
  OpenSubdiv's own edge-point coefficients (`CatmarkLimits`, with its
  valence-dependent edge factor) rather than Loop, Schaefer, Nießner &
  Castaño's published construction, so caps match the reference's surface
  at the same isolation level. The two agree exactly on regular
  neighborhoods and differ slightly at extraordinary vertices.

## [0.1.4] - 2026-09-29

### Fixed

- `FVarLinearInterpolation::CornersPlus2` now sharpens concave corners as
  OpenSubdiv does: where exactly two face-varying values meet at a vertex
  and one of them is a corner (spans a single face), the other value is
  pinned too, so reflex corners of a UV island keep their authored position
  under refinement and at the limit
  ([#5](https://github.com/doubleailes/OpenSubdiv-rs/issues/5)).
- A face-varying value index reused at several geometric vertices (as in a
  deduplicated UV buffer) is now one independent value at each of them, as
  in OpenSubdiv, instead of one shared value-mesh vertex whose sharpness and
  neighborhood mixed all of its uses. Refined levels report one value per
  use; the base level still exposes the channel exactly as described.
- Removed the stale "face-varying refinement is not yet implemented" note
  from the `FVarLinearInterpolation` docs.

## [0.1.0] - 2026-08-06

### Added

- Initial release: a faithful Rust port of
  [Pixar's OpenSubdiv](https://github.com/PixarAnimationStudios/OpenSubdiv),
  covering the `Sdc`, `Vtr` and `Far` layers with no dependencies and no
  `unsafe` code.
- `sdc` — subdivision schemes (`Bilinear`, `Catmark`, `Loop`), subdivision
  `Options` (boundary interpolation, face-varying linear interpolation,
  creasing method), semi-sharp `Crease` rules (`Uniform` and `Chaikin`), and
  the scheme-specific subdivision and limit masks.
- `vtr` — `Level`, the flat-array topology of one refinement level, and
  `Refinement`, one step of uniform quad/tri refinement.
- `far` — `TopologyDescriptor`, `TopologyRefinerFactory`, `TopologyRefiner` /
  `TopologyLevel`, `PrimvarRefiner` (`interpolate`, `interpolate_face_varying`,
  `limit`, `limit_face_varying`), `StencilTable` / `StencilTableFactory`, and
  `PatchTable` / `PatchMap` / `PatchParam` / `PtexIndices`.
- Feature-adaptive refinement (`TopologyRefiner::refine_adaptive`) with sparse
  levels and mixed-depth patch tables.
- Gregory end caps at smooth extraordinary vertices, as OpenSubdiv's
  `ENDCAP_GREGORY_BASIS`.
- Face-varying channels with seams, supporting every
  `FVarLinearInterpolation` rule.
- The `far_tutorial_1_1` example, ported from OpenSubdiv, and a test suite
  covering Catmark, Loop, face-varying, stencil, patch and adaptive behaviour.
- Full crates.io metadata (authors, documentation, readme, keywords,
  categories) and an explicit MSRV (`rust-version = "1.85"`).
- Documentation for every public item, enforced with `#![warn(missing_docs)]`
  and `#![deny(rustdoc::broken_intra_doc_links)]`; `#![forbid(unsafe_code)]`.
- GitHub Actions CI (rustfmt, clippy, tests, rustdoc, MSRV check) and a
  publish-on-tag release workflow.
- This changelog.

[Unreleased]: https://github.com/doubleailes/OpenSubdiv-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/doubleailes/OpenSubdiv-rs/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/doubleailes/OpenSubdiv-rs/compare/v0.1.3...v0.1.4
[0.1.0]: https://github.com/doubleailes/OpenSubdiv-rs/releases/tag/v0.1.0
