# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/doubleailes/OpenSubdiv-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/doubleailes/OpenSubdiv-rs/releases/tag/v0.1.0
