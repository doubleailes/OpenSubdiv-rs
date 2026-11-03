# opensubdiv-wgpu

[wgpu](https://wgpu.rs) compute kernels that evaluate
[opensubdiv-rs](https://crates.io/crates/opensubdiv-rs) tables on the GPU.
They play the role of OpenSubdiv's `Osd` evaluators (`GLComputeEvaluator`,
`CudaEvaluator`, …) on every backend wgpu supports: Vulkan, Metal, DX12,
OpenGL and WebGPU.

| Kernel | OpenSubdiv | Use |
|--------|------------|-----|
| `Evaluator::eval_stencils` | `EvalStencils` | Apply a `StencilTable` to a primvar buffer: refine a deforming cage each frame, into a refined level to draw or into a patch table's control points. `LimitStencilTableBuffers` adds limit-stencil derivatives. |
| `Evaluator::eval_patches` | `EvalPatches` | Evaluate the limit surface (point, `du`, `dv`) at a list of `PatchCoord`s from a `GpuPatchTable`, for example a tessellation grid per patch. |

The kernels are WGSL ports of OpenSubdiv's compute kernels. The patch kernel
covers every patch type of opensubdiv-rs:

- B-spline patches, with boundary masks
- single-crease patches
- Gregory end caps
- bilinear quads
- Loop box-spline patches, with boundary masks
- Gregory triangles
- linear triangles

Their WGSL sources are public (`STENCIL_KERNEL_WGSL`, `PATCH_KERNEL_WGSL`), so
a renderer can reuse `evaluate_patch_basis` in its own shaders.

## Per-frame pipeline

```text
cage positions ──eval_stencils──▶ control points + Gregory local points ──eval_patches──▶ limit points, du, dv
               (one table: for_patch_controls stencils with the
                local-point stencils appended)
```

See `examples/tessellate.rs`. It tessellates an adaptively refined cube on the
GPU and writes it as OBJ:

```sh
cargo run -p opensubdiv-wgpu --example tessellate > cube.obj
```

## Testing

`tests/gpu.rs` runs both kernels on a real adapter and compares them with the
CPU evaluation of the same tables (`StencilTable::update_values`,
`osd::GpuPatchTable::evaluate`, `PatchTable::evaluate`) on meshes that, between
them, exercise every patch type and encoding. A software adapter is enough: on
Linux, Mesa's lavapipe (`apt install mesa-vulkan-drivers`).

When no adapter is found, the GPU tests are skipped with a message. Set
`OPENSUBDIV_WGPU_REQUIRE_GPU=1` to make them fail instead, as CI does. The WGSL
validation test needs no adapter.

## Layout

This crate is a member of the opensubdiv-rs repository's Cargo workspace;
only it depends on wgpu, and `opensubdiv-rs` keeps no dependencies. Run its
checks from the repository root with `-p opensubdiv-wgpu` (or `--workspace`).

The MSRV is **1.90**, shared with `opensubdiv-rs` through
`[workspace.package]`. It is the highest `rust-version` among the current
releases of this crate's dependencies: wgpu 30 needs 1.87, and
`ordered-float`, one of wgpu's dependencies, needs 1.90. The workspace uses
Cargo's MSRV-aware resolver (`resolver = "3"`), so a future dependency release
that needs a newer Rust is not picked up silently.

The crate contains no `unsafe` code.
