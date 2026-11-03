# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Evaluator`, holding the WGSL ports of OpenSubdiv's compute kernels:
  - `eval_stencils` (`EvalStencils`) over `StencilTableBuffers` and
    `LimitStencilTableBuffers`;
  - `eval_patches` (`EvalPatches`) over `PatchTableBuffers` and
    `PatchCoordBuffer`, for every patch type of opensubdiv-rs.
- Primvar buffers are described by `PrimvarBuffer` (offset, length and
  stride in floats), so they can be interleaved.
- The `tessellate` example, and GPU tests against the CPU evaluation.
