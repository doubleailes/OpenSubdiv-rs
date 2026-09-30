# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Patches around non-manifold features
  ([#10](https://github.com/doubleailes/OpenSubdiv-rs/issues/10)): faces
  incident non-manifold edges or vertices are now patched over their own
  manifold span — regular B-spline patches where the span is regular,
  Gregory caps elsewhere — instead of falling back to bilinear quads.
  `PatchType::Quads` now only appears for irregular faces on unsharpened
  (`VtxBoundaryInterpolation::None`) boundaries and the Bilinear scheme.
- `vtr::Level::is_vertex_non_manifold`.

### Changed

- Non-manifold edges are made infinitely sharp when the base level is
  built, and non-manifold vertices infinitely sharp unless they lie on a
  crease of exactly two non-manifold edges, as OpenSubdiv does
  (`applyComponentTagsAndBoundarySharpness`). This changes refinement and
  limit positions around non-manifold features, which previously followed
  the smooth rules.

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
