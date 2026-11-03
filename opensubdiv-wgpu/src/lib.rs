//! # opensubdiv-wgpu
//!
//! [`wgpu`] compute kernels that evaluate [`opensubdiv_rs`] tables on the
//! GPU — the role of OpenSubdiv's `Osd` evaluators (`GLComputeEvaluator`,
//! `CudaEvaluator`, …), for any backend wgpu supports (Vulkan, Metal,
//! DX12, OpenGL, WebGPU).
//!
//! Two kernels cover a renderer's needs:
//!
//! * [`Evaluator::eval_stencils`] (`EvalStencils`) applies a
//!   [`StencilTable`] to a primvar buffer:
//!   one invocation per stencil computes `Σ weights · src[indices]`. Run it
//!   each frame to refine a deforming cage — uniformly, into the vertices of
//!   a refined level to draw, or into the control points of a patch table.
//!   [`LimitStencilTableBuffers`] adds the derivative weights of limit
//!   stencils.
//! * [`Evaluator::eval_patches`] (`EvalPatches`) evaluates the limit
//!   surface, with first derivatives, at a list of
//!   [`PatchCoord`]s from the buffers of a
//!   [`GpuPatchTable`] — for example a
//!   tessellation grid per patch, whose output a render pass then draws.
//!
//! The kernels are WGSL ports of OpenSubdiv's compute kernels and of
//! [`opensubdiv_rs::osd::evaluate_patch_basis`]; the tests check them
//! against the CPU evaluation of the same tables.
//!
//! Primvar buffers are arrays of `f32` described by a [`PrimvarBuffer`]:
//! element `i` occupies `length` floats at `offset + i * stride`, as with
//! OpenSubdiv's `BufferDescriptor`, so positions, normals and other data
//! may be interleaved in one vertex buffer. Buffers the kernels read or
//! write need [`wgpu::BufferUsages::STORAGE`]; a kernel cannot read and
//! write the same buffer.
//!
//! ```no_run
//! use opensubdiv_rs::far::{
//!     AdaptiveOptions, PatchTableFactory, StencilTableFactory, StencilTableOptions,
//!     TopologyDescriptor, TopologyRefinerFactory,
//! };
//! use opensubdiv_rs::osd::GpuPatchTable;
//! use opensubdiv_rs::sdc;
//! use opensubdiv_wgpu::{
//!     Evaluator, PatchCoordBuffer, PatchOutputs, PatchTableBuffers, PrimvarBuffer,
//!     StencilTableBuffers,
//! };
//!
//! # fn run(device: &wgpu::Device, queue: &wgpu::Queue,
//! #        cage: &wgpu::Buffer, controls: &wgpu::Buffer, out: &wgpu::Buffer) {
//! # let verts_per_face = [4; 6];
//! # let face_verts = [0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4];
//! // Once per topology: refine, build the tables, upload them.
//! let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
//! let mut refiner = TopologyRefinerFactory::create(
//!     descriptor, sdc::SchemeType::Catmark, sdc::Options::default(),
//! ).unwrap();
//! refiner.refine_adaptive(AdaptiveOptions::new(3));
//! let patches = PatchTableFactory::create(&refiner).unwrap();
//! let gpu_patches = GpuPatchTable::new(&patches).unwrap();
//! let stencils = StencilTableFactory::append_local_point_stencils(
//!     &StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls()),
//!     gpu_patches.local_point_stencils(),
//! );
//! let evaluator = Evaluator::new(device);
//! let stencil_buffers = StencilTableBuffers::new(device, &stencils);
//! let patch_buffers = PatchTableBuffers::new(device, &gpu_patches);
//! let coords: Vec<_> = (0..gpu_patches.num_patches())
//!     .map(|p| {
//!         let (u, v) = patches.patch_param(p).unnormalize(0.5, 0.5);
//!         gpu_patches.patch_coord(p, u, v)
//!     })
//!     .collect();
//! let coord_buffer = PatchCoordBuffer::new(device, &coords);
//!
//! // Per frame, after writing the cage positions (xyz) to `cage`:
//! let mut encoder = device.create_command_encoder(&Default::default());
//! evaluator.eval_stencils(
//!     device, &mut encoder, &stencil_buffers,
//!     PrimvarBuffer::new(cage, 0, 3, 3),
//!     PrimvarBuffer::new(controls, 0, 3, 3),
//! );
//! evaluator.eval_patches(
//!     device, &mut encoder, &patch_buffers, &coord_buffer,
//!     PrimvarBuffer::new(controls, 0, 3, 3),
//!     PatchOutputs::points(PrimvarBuffer::new(out, 0, 3, 3)),
//! );
//! queue.submit([encoder.finish()]);
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

use opensubdiv_rs::far::{LimitStencilTable, StencilTable};
use opensubdiv_rs::osd::{GpuPatchTable, PatchCoord};
use wgpu::util::DeviceExt;

pub use opensubdiv_rs;
pub use wgpu;

/// The WGSL source of the stencil kernel (entry point `eval_stencils`).
pub const STENCIL_KERNEL_WGSL: &str = include_str!("stencil.wgsl");

/// The WGSL source of the patch kernel (entry point `eval_patches`). Its
/// `evaluate_patch_basis` function can be reused by other shaders, for
/// example to evaluate patches in a vertex or mesh shader.
pub const PATCH_KERNEL_WGSL: &str = include_str!("patch.wgsl");

/// Invocations per workgroup of both kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The most workgroups a dispatch has along one dimension (WebGPU's
/// guaranteed `maxComputeWorkgroupsPerDimension`).
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// A primvar array in a `f32` buffer (`Osd::BufferDescriptor` with its
/// buffer): element `i` occupies `length` floats at `offset + i * stride`,
/// all counted in floats.
#[derive(Debug, Clone, Copy)]
pub struct PrimvarBuffer<'a> {
    /// The buffer, with [`wgpu::BufferUsages::STORAGE`].
    pub buffer: &'a wgpu::Buffer,
    /// Floats before the first element.
    pub offset: u32,
    /// Floats per element (3 for positions).
    pub length: u32,
    /// Floats from one element to the next (at least `length`).
    pub stride: u32,
}

impl<'a> PrimvarBuffer<'a> {
    /// Describe `length` floats per element, `stride` apart, from `offset`.
    pub fn new(buffer: &'a wgpu::Buffer, offset: u32, length: u32, stride: u32) -> Self {
        PrimvarBuffer {
            buffer,
            offset,
            length,
            stride,
        }
    }

    /// Panic unless the buffer holds `count` elements.
    fn check(&self, count: u32, what: &str) {
        assert!(
            self.stride >= self.length,
            "{what}: stride {} is less than length {}",
            self.stride,
            self.length
        );
        if count == 0 {
            return;
        }
        let floats = u64::from(self.offset)
            + u64::from(count - 1) * u64::from(self.stride)
            + u64::from(self.length);
        assert!(
            floats * 4 <= self.buffer.size(),
            "{what}: {count} elements need {} bytes, the buffer has {}",
            floats * 4,
            self.buffer.size()
        );
    }
}

/// A [`StencilTable`] uploaded to the GPU, for
/// [`Evaluator::eval_stencils`].
#[derive(Debug, Clone)]
pub struct StencilTableBuffers {
    offsets: wgpu::Buffer,
    indices: wgpu::Buffer,
    weights: wgpu::Buffer,
    num_stencils: u32,
    num_control_vertices: u32,
}

impl StencilTableBuffers {
    /// Upload `table`'s [`offsets`](StencilTable::offsets),
    /// [`indices`](StencilTable::indices) and
    /// [`weights`](StencilTable::weights).
    pub fn new(device: &wgpu::Device, table: &StencilTable) -> Self {
        StencilTableBuffers {
            offsets: storage_buffer(device, "stencil offsets", table.offsets(), 1),
            indices: storage_buffer(device, "stencil indices", table.indices(), 1),
            weights: storage_buffer(device, "stencil weights", &float_words(table.weights()), 1),
            num_stencils: to_u32(table.num_stencils()),
            num_control_vertices: to_u32(table.num_control_vertices()),
        }
    }

    /// The number of stencils, and of elements the kernel writes.
    pub fn num_stencils(&self) -> u32 {
        self.num_stencils
    }

    /// The number of control vertices, the elements the kernel reads.
    pub fn num_control_vertices(&self) -> u32 {
        self.num_control_vertices
    }
}

/// A [`LimitStencilTable`] uploaded to the GPU: three stencil tables over
/// the same offsets and indices, for the limit position and its
/// derivatives along `u` and `v`.
#[derive(Debug, Clone)]
pub struct LimitStencilTableBuffers {
    points: StencilTableBuffers,
    du: StencilTableBuffers,
    dv: StencilTableBuffers,
}

impl LimitStencilTableBuffers {
    /// Upload `table`.
    pub fn new(device: &wgpu::Device, table: &LimitStencilTable) -> Self {
        let points = StencilTableBuffers::new(device, table.point_stencils());
        let with_weights = |label, weights: &[f32]| StencilTableBuffers {
            weights: storage_buffer(device, label, &float_words(weights), 1),
            ..points.clone()
        };
        LimitStencilTableBuffers {
            du: with_weights("limit stencil du weights", table.du_weights()),
            dv: with_weights("limit stencil dv weights", table.dv_weights()),
            points,
        }
    }

    /// The position stencils.
    pub fn points(&self) -> &StencilTableBuffers {
        &self.points
    }

    /// The stencils of the derivative along `u`.
    pub fn du(&self) -> &StencilTableBuffers {
        &self.du
    }

    /// The stencils of the derivative along `v`.
    pub fn dv(&self) -> &StencilTableBuffers {
        &self.dv
    }
}

/// A [`GpuPatchTable`] uploaded to the GPU, for
/// [`Evaluator::eval_patches`] (`Osd::GLPatchTable` and its siblings).
#[derive(Debug, Clone)]
pub struct PatchTableBuffers {
    arrays: wgpu::Buffer,
    indices: wgpu::Buffer,
    params: wgpu::Buffer,
    num_patches: u32,
    num_values: u32,
}

impl PatchTableBuffers {
    /// Upload `table`'s patch arrays, index buffer and patch parameters.
    pub fn new(device: &wgpu::Device, table: &GpuPatchTable) -> Self {
        PatchTableBuffers {
            arrays: storage_buffer(device, "patch arrays", &table.patch_array_words(), 6),
            indices: storage_buffer(device, "patch indices", table.patch_index_buffer(), 1),
            params: storage_buffer(device, "patch params", &table.patch_param_words(), 3),
            num_patches: to_u32(table.num_patches()),
            num_values: to_u32(table.num_control_values() + table.num_local_points()),
        }
    }

    /// The number of patches.
    pub fn num_patches(&self) -> u32 {
        self.num_patches
    }

    /// The number of elements the patches index: the control values
    /// followed by the local points.
    pub fn num_values(&self) -> u32 {
        self.num_values
    }
}

/// The [`PatchCoord`]s to evaluate, uploaded to the GPU.
#[derive(Debug, Clone)]
pub struct PatchCoordBuffer {
    buffer: wgpu::Buffer,
    len: u32,
}

impl PatchCoordBuffer {
    /// Upload `coords`.
    pub fn new(device: &wgpu::Device, coords: &[PatchCoord]) -> Self {
        let words: Vec<u32> = coords.iter().flat_map(PatchCoord::to_words).collect();
        PatchCoordBuffer {
            buffer: storage_buffer(device, "patch coords", &words, 5),
            len: to_u32(coords.len()),
        }
    }

    /// Wrap a buffer of `len` `OsdPatchCoord` records (20 bytes each) the
    /// application filled itself, for example with a compute pass that
    /// generates tessellation grids.
    ///
    /// # Panics
    ///
    /// Panics when the buffer is smaller than `len` records.
    pub fn from_buffer(buffer: wgpu::Buffer, len: u32) -> Self {
        assert!(
            buffer.size() >= u64::from(len) * 20,
            "{len} patch coords need {} bytes, the buffer has {}",
            u64::from(len) * 20,
            buffer.size()
        );
        PatchCoordBuffer { buffer, len }
    }

    /// The number of coordinates, and of elements the patch kernel writes.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Is the buffer empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Where [`Evaluator::eval_patches`] writes: the limit points, and
/// optionally their derivatives along the ptex face's `u` and `v`. The
/// three buffers must be distinct, and have the same element `length`.
#[derive(Debug, Clone, Copy)]
pub struct PatchOutputs<'a> {
    /// The limit points.
    pub points: PrimvarBuffer<'a>,
    /// The derivatives along `u`, if wanted.
    pub du: Option<PrimvarBuffer<'a>>,
    /// The derivatives along `v`, if wanted.
    pub dv: Option<PrimvarBuffer<'a>>,
}

impl<'a> PatchOutputs<'a> {
    /// Write the limit points only.
    pub fn points(points: PrimvarBuffer<'a>) -> Self {
        PatchOutputs {
            points,
            du: None,
            dv: None,
        }
    }

    /// Write the limit points and both derivatives.
    pub fn with_derivatives(
        points: PrimvarBuffer<'a>,
        du: PrimvarBuffer<'a>,
        dv: PrimvarBuffer<'a>,
    ) -> Self {
        PatchOutputs {
            points,
            du: Some(du),
            dv: Some(dv),
        }
    }
}

/// The compiled stencil and patch kernels (`Osd::GLComputeEvaluator`'s
/// role). Create one per device and reuse it.
#[derive(Debug)]
pub struct Evaluator {
    stencil_pipeline: wgpu::ComputePipeline,
    stencil_layout: wgpu::BindGroupLayout,
    patch_pipeline: wgpu::ComputePipeline,
    patch_layout: wgpu::BindGroupLayout,
    /// Bound in place of derivative outputs that are not wanted.
    placeholder: wgpu::Buffer,
}

impl Evaluator {
    /// Compile both kernels for `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let pipeline = |label, source: &str, entry_point| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipeline.get_bind_group_layout(0);
            (pipeline, layout)
        };
        let (stencil_pipeline, stencil_layout) = pipeline(
            "opensubdiv stencil kernel",
            STENCIL_KERNEL_WGSL,
            "eval_stencils",
        );
        let (patch_pipeline, patch_layout) =
            pipeline("opensubdiv patch kernel", PATCH_KERNEL_WGSL, "eval_patches");
        Evaluator {
            stencil_pipeline,
            stencil_layout,
            patch_pipeline,
            patch_layout,
            placeholder: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("opensubdiv placeholder output"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        }
    }

    /// Record a compute pass applying `table` to `src` (one element per
    /// control vertex) and writing `dst` (one element per stencil)
    /// (`EvalStencils`).
    ///
    /// # Panics
    ///
    /// Panics when the element lengths differ, or a buffer is too small
    /// for its elements.
    pub fn eval_stencils(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        table: &StencilTableBuffers,
        src: PrimvarBuffer<'_>,
        dst: PrimvarBuffer<'_>,
    ) {
        assert_eq!(src.length, dst.length, "src and dst element lengths differ");
        src.check(table.num_control_vertices, "stencil source");
        dst.check(table.num_stencils, "stencil destination");
        if table.num_stencils == 0 {
            return;
        }
        let (groups, row_width) = dispatch_size(table.num_stencils);
        let params = [
            table.num_stencils,
            row_width,
            src.length,
            src.offset,
            src.stride,
            dst.offset,
            dst.stride,
            0,
        ];
        let params = uniform_buffer(device, "stencil params", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opensubdiv stencil bindings"),
            layout: &self.stencil_layout,
            entries: &bind_entries(&[
                &params,
                src.buffer,
                dst.buffer,
                &table.offsets,
                &table.indices,
                &table.weights,
            ]),
        });
        dispatch(
            encoder,
            "opensubdiv eval_stencils",
            &self.stencil_pipeline,
            &bind_group,
            groups,
        );
    }

    /// Record a compute pass evaluating the patches of `table` at every
    /// coordinate of `coords` (`EvalPatches`): `src` holds the patches'
    /// control values followed by their local points
    /// ([`PatchTableBuffers::num_values`] elements), and `outputs` receive
    /// one element per coordinate.
    ///
    /// # Panics
    ///
    /// Panics when the element lengths differ, or a buffer is too small
    /// for its elements.
    pub fn eval_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        table: &PatchTableBuffers,
        coords: &PatchCoordBuffer,
        src: PrimvarBuffer<'_>,
        outputs: PatchOutputs<'_>,
    ) {
        let n = coords.len;
        src.check(table.num_values, "patch source");
        outputs.points.check(n, "patch points");
        assert_eq!(
            src.length, outputs.points.length,
            "src and dst element lengths differ"
        );
        for (output, what) in [(outputs.du, "patch du"), (outputs.dv, "patch dv")] {
            if let Some(output) = output {
                output.check(n, what);
                assert_eq!(
                    src.length, output.length,
                    "src and {what} element lengths differ"
                );
            }
        }
        if n == 0 {
            return;
        }
        let (groups, row_width) = dispatch_size(n);
        let du = outputs
            .du
            .unwrap_or(PrimvarBuffer::new(&self.placeholder, 0, 0, 0));
        let dv = outputs
            .dv
            .unwrap_or(PrimvarBuffer::new(&self.placeholder, 0, 0, 0));
        let flags = outputs.du.is_some() as u32 | (outputs.dv.is_some() as u32) << 1;
        let params = [
            n,
            row_width,
            src.length,
            flags,
            src.offset,
            src.stride,
            outputs.points.offset,
            outputs.points.stride,
            du.offset,
            du.stride,
            dv.offset,
            dv.stride,
        ];
        let params = uniform_buffer(device, "patch params", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("opensubdiv patch bindings"),
            layout: &self.patch_layout,
            entries: &bind_entries(&[
                &params,
                src.buffer,
                outputs.points.buffer,
                du.buffer,
                dv.buffer,
                &table.arrays,
                &coords.buffer,
                &table.indices,
                &table.params,
            ]),
        });
        dispatch(
            encoder,
            "opensubdiv eval_patches",
            &self.patch_pipeline,
            &bind_group,
            groups,
        );
    }
}

/// The workgroup grid covering `n` invocations, and its row width in
/// invocations.
fn dispatch_size(n: u32) -> ([u32; 2], u32) {
    let groups = n.div_ceil(WORKGROUP_SIZE);
    let x = groups.min(MAX_WORKGROUPS_PER_DIMENSION);
    let y = groups.div_ceil(x);
    ([x, y], x * WORKGROUP_SIZE)
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups[0], groups[1], 1);
}

fn bind_entries<'a>(buffers: &[&'a wgpu::Buffer]) -> Vec<wgpu::BindGroupEntry<'a>> {
    buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect()
}

/// A storage buffer of `words`, padded to at least `min_words` (a binding
/// must hold one element of its array, even for an empty table).
fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    words: &[u32],
    min_words: usize,
) -> wgpu::Buffer {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    bytes.resize(bytes.len().max(4 * min_words), 0);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn uniform_buffer(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn float_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).expect("table sizes fit in 32 bits on the GPU")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_grids_cover_their_invocations() {
        for n in [1u32, 63, 64, 65, 64 * 65535, 64 * 65535 + 1, 10_000_000] {
            let ([x, y], row) = dispatch_size(n);
            assert!(x <= MAX_WORKGROUPS_PER_DIMENSION && y <= MAX_WORKGROUPS_PER_DIMENSION);
            assert_eq!(row, x * WORKGROUP_SIZE);
            assert!(u64::from(x) * u64::from(y) * u64::from(WORKGROUP_SIZE) >= u64::from(n));
        }
    }
}
