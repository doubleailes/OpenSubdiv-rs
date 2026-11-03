//! GPU kernel tests: the stencil and patch kernels, run on a real device,
//! must reproduce the CPU evaluation of the same tables.
//!
//! The tests need a GPU adapter — a software one such as Mesa's lavapipe
//! will do. Without one they are skipped with a message, unless
//! `OPENSUBDIV_WGPU_REQUIRE_GPU` is set (as in CI), when they fail. The
//! shader validation test needs no adapter.

use opensubdiv_rs::far::{
    AdaptiveOptions, LimitStencilTableFactory, LocationArray, PatchTable, PatchTableFactory,
    PrimvarRefiner, StencilTable, StencilTableFactory, StencilTableOptions, TopologyDescriptor,
    TopologyRefiner, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::osd::{GpuPatchTable, PatchCoord};
use opensubdiv_rs::sdc;
use opensubdiv_wgpu::{
    Evaluator, LimitStencilTableBuffers, PatchCoordBuffer, PatchOutputs, PatchTableBuffers,
    PrimvarBuffer, StencilTableBuffers, PATCH_KERNEL_WGSL, STENCIL_KERNEL_WGSL,
};
use wgpu::util::DeviceExt;

type P3 = [f32; 3];

// ----------------------------------------------------------------------
//  Device and buffer helpers
// ----------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    evaluator: Evaluator,
}

/// The test device, or `None` (after a message) when no adapter exists and
/// none is required.
fn gpu() -> Option<Gpu> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = match pollster::block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            if std::env::var_os("OPENSUBDIV_WGPU_REQUIRE_GPU").is_some() {
                panic!("no GPU adapter: {error}");
            }
            eprintln!("skipping: no GPU adapter ({error})");
            return None;
        }
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let evaluator = Evaluator::new(&device);
    Some(Gpu {
        device,
        queue,
        evaluator,
    })
}

impl Gpu {
    fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let mut bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
        bytes.resize(bytes.len().max(4), 0);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test data"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// A storage buffer of `floats` floats, filled with NaN so that any
    /// element the kernel fails to write shows.
    fn output(&self, floats: usize) -> wgpu::Buffer {
        self.upload(&vec![f32::NAN; floats.max(1)])
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let view = staging.get_mapped_range(..).expect("mapped range");
        view.chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn submit(&self, record: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        record(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }
}

/// Spread 3-vectors into a strided buffer: `offset` floats of padding,
/// then each vector followed by `stride - 3` padding floats.
fn interleave(values: &[P3], offset: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![-7.0f32; offset + values.len() * stride];
    for (i, v) in values.iter().enumerate() {
        out[offset + i * stride..][..3].copy_from_slice(v);
    }
    out
}

fn extract(buffer: &[f32], count: usize, offset: usize, stride: usize) -> Vec<P3> {
    (0..count)
        .map(|i| {
            let b = &buffer[offset + i * stride..];
            [b[0], b[1], b[2]]
        })
        .collect()
}

fn assert_close(actual: &[P3], expected: &[P3], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        for k in 0..3 {
            assert!(
                (a[k] - e[k]).abs() <= tol * (1.0 + e[k].abs()),
                "{what} {i}: expected {e:?}, got {a:?}"
            );
        }
    }
}

// ----------------------------------------------------------------------
//  Fixtures
// ----------------------------------------------------------------------

const CUBE_FACE_VERTS: [u32; 24] = [
    0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
];
const CUBE_POSITIONS: [P3; 8] = [
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
];

fn edge_and_corner() -> sdc::Options {
    sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
}

/// A refiner over `faces` (vertex counts and indices) of `scheme`.
fn refiner(
    scheme: sdc::SchemeType,
    options: sdc::Options,
    num_vertices: usize,
    verts_per_face: &[usize],
    face_verts: &[u32],
    creases: Option<(&[[u32; 2]], &[f32])>,
) -> TopologyRefiner {
    let mut descriptor = TopologyDescriptor::new(num_vertices, verts_per_face, face_verts);
    if let Some((pairs, weights)) = creases {
        descriptor = descriptor.with_creases(pairs, weights);
    }
    TopologyRefinerFactory::create(descriptor, scheme, options).unwrap()
}

/// A bumpy `w x h`-vertex grid of quads, or of triangles (one diagonal
/// flipped, for valence-5 and -7 vertices).
fn grid(w: u32, h: u32, triangles: bool) -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let v = j * w + i;
            if !triangles {
                verts_per_face.push(4);
                face_verts.extend_from_slice(&[v, v + 1, v + w + 1, v + w]);
            } else if (i, j) == (2, 2) {
                verts_per_face.extend([3, 3]);
                face_verts.extend_from_slice(&[v, v + 1, v + w, v + 1, v + w + 1, v + w]);
            } else {
                verts_per_face.extend([3, 3]);
                face_verts.extend_from_slice(&[v, v + 1, v + w + 1, v, v + w + 1, v + w]);
            }
        }
    }
    let positions = (0..h)
        .flat_map(|j| {
            (0..w).map(move |i| {
                let (x, y) = (i as f32, j as f32);
                [
                    x + 0.1 * y.sin(),
                    y,
                    0.3 * (0.9 * x).sin() * (1.3 * y).cos(),
                ]
            })
        })
        .collect();
    (verts_per_face, face_verts, positions)
}

/// The meshes the patch kernel is checked on, with their cage positions:
/// between them, every patch type, rotation, boundary mask, single crease,
/// non-quad root and inverted triangle.
fn patch_fixtures() -> Vec<(&'static str, TopologyRefiner, Vec<P3>)> {
    let mut fixtures = Vec::new();

    // Regular and Gregory patches with semi-sharp single-crease patches.
    let creases = [[0u32, 1], [1, 3], [3, 2], [2, 0]];
    let weights = [2.5f32; 4];
    let mut cube = refiner(
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
        8,
        &[4; 6],
        &CUBE_FACE_VERTS,
        Some((&creases, &weights)),
    );
    cube.refine_adaptive(AdaptiveOptions::new(3).with_single_crease_patch(true));
    fixtures.push(("creased cube", cube, CUBE_POSITIONS.to_vec()));

    // Boundaries, corners, an infinitely sharp crease, a triangle pair
    // (non-quad roots) and extraordinary vertices.
    let (w, h) = (6u32, 5u32);
    let (mut verts_per_face, mut face_verts, positions) = grid(w, h, false);
    let q = w as usize; // the quad at (1, 1)
    let quad: Vec<u32> = face_verts[4 * q..4 * q + 4].to_vec();
    face_verts.splice(
        4 * q..4 * q + 4,
        [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]],
    );
    verts_per_face.splice(q..q + 1, [3, 3]);
    let sharp: Vec<[u32; 2]> = (0..w - 1).map(|i| [3 * w + i, 3 * w + i + 1]).collect();
    let sharpness = vec![sdc::SHARPNESS_INFINITE; sharp.len()];
    let mut quads = refiner(
        sdc::SchemeType::Catmark,
        edge_and_corner(),
        (w * h) as usize,
        &verts_per_face,
        &face_verts,
        Some((&sharp, &sharpness)),
    );
    quads.refine_adaptive(AdaptiveOptions::new(3));
    fixtures.push(("boundary grid", quads, positions));

    // Box-spline patches with boundary edges and vertices, Gregory
    // triangles and inverted triangles.
    let (w, h) = (7u32, 6u32);
    let (verts_per_face, face_verts, positions) = grid(w, h, true);
    let mut tris = refiner(
        sdc::SchemeType::Loop,
        edge_and_corner(),
        (w * h) as usize,
        &verts_per_face,
        &face_verts,
        None,
    );
    tris.refine_adaptive(AdaptiveOptions::new(3));
    fixtures.push(("loop grid", tris, positions.clone()));

    // Linear triangles on unsharpened Loop boundaries.
    let mut linear = refiner(
        sdc::SchemeType::Loop,
        sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::None),
        (w * h) as usize,
        &verts_per_face,
        &face_verts,
        None,
    );
    linear.refine_uniform(UniformOptions::new(1));
    fixtures.push(("linear triangles", linear, positions));

    // Bilinear quads.
    let mut bilinear = refiner(
        sdc::SchemeType::Bilinear,
        sdc::Options::default(),
        8,
        &[4; 6],
        &CUBE_FACE_VERTS,
        None,
    );
    bilinear.refine_uniform(UniformOptions::new(1));
    fixtures.push(("bilinear cube", bilinear, CUBE_POSITIONS.to_vec()));

    fixtures
}

/// Patch control values: every level's vertex values, base level first.
fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

/// Coordinates on every patch: its corners, center and a few interior
/// points, in ptex-face `(u, v)`.
fn patch_coords(table: &PatchTable, gpu: &GpuPatchTable) -> Vec<PatchCoord> {
    let quad = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.7),
        (0.9, 0.35),
    ];
    let tri = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 1.0 / 3.0),
        (0.6, 0.1),
        (0.15, 0.7),
    ];
    let mut coords = Vec::new();
    for patch in 0..table.num_patches() {
        let param = table.patch_param(patch);
        let samples: &[(f32, f32)] = if param.triangular { &tri } else { &quad };
        for &(s, t) in samples {
            let (u, v) = param.unnormalize(s, t);
            coords.push(gpu.patch_coord(patch, u, v));
        }
    }
    coords
}

// ----------------------------------------------------------------------
//  Tests
// ----------------------------------------------------------------------

#[test]
fn kernels_are_valid_wgsl() {
    for (name, source) in [
        ("stencil", STENCIL_KERNEL_WGSL),
        ("patch", PATCH_KERNEL_WGSL),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} kernel: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} kernel: {}", e.emit_to_string(source)));
    }
}

/// Run `table` on the GPU over `src` (laid out at `src_layout`), writing
/// `dst_layout`; return the 3-vectors written.
fn gpu_stencils(
    gpu: &Gpu,
    table: &StencilTable,
    src: &[P3],
    src_layout: (usize, usize),
    dst_layout: (usize, usize),
) -> Vec<P3> {
    let buffers = StencilTableBuffers::new(&gpu.device, table);
    let src_buffer = gpu.upload(&interleave(src, src_layout.0, src_layout.1));
    let n = table.num_stencils();
    let dst_buffer = gpu.output(dst_layout.0 + n * dst_layout.1);
    gpu.submit(|encoder| {
        gpu.evaluator.eval_stencils(
            &gpu.device,
            encoder,
            &buffers,
            PrimvarBuffer::new(&src_buffer, src_layout.0 as u32, 3, src_layout.1 as u32),
            PrimvarBuffer::new(&dst_buffer, dst_layout.0 as u32, 3, dst_layout.1 as u32),
        );
    });
    extract(&gpu.read(&dst_buffer), n, dst_layout.0, dst_layout.1)
}

#[test]
fn stencil_kernel_matches_update_values() {
    let Some(gpu) = gpu() else { return };

    let mut uniform = refiner(
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
        8,
        &[4; 6],
        &CUBE_FACE_VERTS,
        None,
    );
    uniform.refine_uniform(UniformOptions::new(3));
    let fixtures = patch_fixtures();
    let mut tables = vec![
        StencilTableFactory::create(&uniform, StencilTableOptions::default()),
        StencilTableFactory::create_limit(&uniform),
    ];
    let mut cages = vec![CUBE_POSITIONS.to_vec(), CUBE_POSITIONS.to_vec()];
    for (_, refiner, positions) in &fixtures {
        tables.push(StencilTableFactory::create(
            refiner,
            StencilTableOptions::for_patch_controls(),
        ));
        cages.push(positions.clone());
    }

    for (table, cage) in tables.iter().zip(&cages) {
        let mut expected = vec![[0.0f32; 3]; table.num_stencils()];
        table.update_values(cage, &mut expected);
        // Packed, and interleaved in wider vertices with leading padding.
        for (src_layout, dst_layout) in [((0, 3), (0, 3)), ((2, 4), (5, 7))] {
            let actual = gpu_stencils(&gpu, table, cage, src_layout, dst_layout);
            assert_close(&actual, &expected, 1e-5, "stencil");
        }
    }
}

#[test]
fn limit_stencil_kernels_match_update_derivs() {
    let Some(gpu) = gpu() else { return };
    let fixtures = patch_fixtures();
    let (_, refiner, cage) = &fixtures[1];
    let u: Vec<f32> = (0..16).map(|k| (k as f32 + 0.5) / 16.0).collect();
    let v: Vec<f32> = (0..16).map(|k| ((k * 7) % 16) as f32 / 15.0).collect();
    let arrays: Vec<LocationArray> = (0..6).map(|f| LocationArray::new(f, &u, &v)).collect();
    let table = LimitStencilTableFactory::create(refiner, &arrays, None, None).unwrap();
    let n = table.num_stencils();
    let mut expected = vec![[0.0f32; 3]; n];
    let (mut du, mut dv) = (expected.clone(), expected.clone());
    table.update_values(cage, &mut expected);
    table.update_derivs(cage, &mut du, &mut dv);

    let buffers = LimitStencilTableBuffers::new(&gpu.device, &table);
    let src = gpu.upload(&interleave(cage, 0, 3));
    let outputs = [gpu.output(3 * n), gpu.output(3 * n), gpu.output(3 * n)];
    gpu.submit(|encoder| {
        for (stencils, dst) in [buffers.points(), buffers.du(), buffers.dv()]
            .into_iter()
            .zip(&outputs)
        {
            gpu.evaluator.eval_stencils(
                &gpu.device,
                encoder,
                stencils,
                PrimvarBuffer::new(&src, 0, 3, 3),
                PrimvarBuffer::new(dst, 0, 3, 3),
            );
        }
    });
    for (output, expected, what) in [
        (&outputs[0], &expected, "point"),
        (&outputs[1], &du, "du"),
        (&outputs[2], &dv, "dv"),
    ] {
        assert_close(&extract(&gpu.read(output), n, 0, 3), expected, 1e-4, what);
    }
}

#[test]
fn patch_kernel_matches_cpu_evaluation() {
    let Some(gpu) = gpu() else { return };
    for (name, refiner, cage) in patch_fixtures() {
        let table = PatchTableFactory::create(&refiner).unwrap();
        let gpu_table = GpuPatchTable::new(&table).unwrap();
        let coords = patch_coords(&table, &gpu_table);
        let n = coords.len();

        // The control values and local points, by the CPU.
        let controls = patch_controls(&refiner, &cage);
        let mut local = vec![[0.0f32; 3]; gpu_table.num_local_points()];
        gpu_table
            .local_point_stencils()
            .update_values(&controls, &mut local);
        let values: Vec<P3> = controls.iter().chain(&local).copied().collect();

        let patch_buffers = PatchTableBuffers::new(&gpu.device, &gpu_table);
        let coord_buffer = PatchCoordBuffer::new(&gpu.device, &coords);
        let src = gpu.upload(&interleave(&values, 1, 4));
        let (points, du, dv) = (gpu.output(6 * n), gpu.output(3 * n), gpu.output(5 + 3 * n));
        gpu.submit(|encoder| {
            gpu.evaluator.eval_patches(
                &gpu.device,
                encoder,
                &patch_buffers,
                &coord_buffer,
                PrimvarBuffer::new(&src, 1, 3, 4),
                PatchOutputs::with_derivatives(
                    PrimvarBuffer::new(&points, 0, 3, 6),
                    PrimvarBuffer::new(&du, 0, 3, 3),
                    PrimvarBuffer::new(&dv, 5, 3, 3),
                ),
            );
        });
        let actual = (
            extract(&gpu.read(&points), n, 0, 6),
            extract(&gpu.read(&du), n, 0, 3),
            extract(&gpu.read(&dv), n, 5, 3),
        );

        let mut expected = (Vec::new(), Vec::new(), Vec::new());
        for coord in &coords {
            let (p, pu, pv) = gpu_table.evaluate(coord, &values);
            // The CPU reference itself agrees with the far patch table.
            let (fp, _, _) =
                table.evaluate(coord.patch_index as usize, coord.s, coord.t, &controls);
            assert_close(&[p], &[fp], 1e-5, "reference");
            expected.0.push(p);
            expected.1.push(pu);
            expected.2.push(pv);
        }
        // The kernel evaluates in single precision where the CPU basis
        // uses double precision in places.
        assert_close(&actual.0, &expected.0, 1e-4, &format!("{name} point"));
        assert_close(&actual.1, &expected.1, 1e-3, &format!("{name} du"));
        assert_close(&actual.2, &expected.2, 1e-3, &format!("{name} dv"));
    }
}

#[test]
fn stencils_then_patches_in_one_submission() {
    // The per-frame pipeline of a renderer: the appended stencil table
    // refines the cage into every point the patches index, then the patch
    // kernel evaluates them, without a CPU round trip.
    let Some(gpu) = gpu() else { return };
    for (name, refiner, cage) in patch_fixtures() {
        let table = PatchTableFactory::create(&refiner).unwrap();
        let gpu_table = GpuPatchTable::new(&table).unwrap();
        let stencils = StencilTableFactory::append_local_point_stencils(
            &StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls()),
            gpu_table.local_point_stencils(),
        );
        let coords = patch_coords(&table, &gpu_table);
        let n = coords.len();

        let stencil_buffers = StencilTableBuffers::new(&gpu.device, &stencils);
        let patch_buffers = PatchTableBuffers::new(&gpu.device, &gpu_table);
        let coord_buffer = PatchCoordBuffer::new(&gpu.device, &coords);
        let cage_buffer = gpu.upload(&interleave(&cage, 0, 3));
        let values = gpu.output(3 * stencils.num_stencils());
        let points = gpu.output(3 * n);
        gpu.submit(|encoder| {
            gpu.evaluator.eval_stencils(
                &gpu.device,
                encoder,
                &stencil_buffers,
                PrimvarBuffer::new(&cage_buffer, 0, 3, 3),
                PrimvarBuffer::new(&values, 0, 3, 3),
            );
            gpu.evaluator.eval_patches(
                &gpu.device,
                encoder,
                &patch_buffers,
                &coord_buffer,
                PrimvarBuffer::new(&values, 0, 3, 3),
                PatchOutputs::points(PrimvarBuffer::new(&points, 0, 3, 3)),
            );
        });

        let controls = patch_controls(&refiner, &cage);
        let expected: Vec<P3> = coords
            .iter()
            .map(|c| {
                table
                    .evaluate(c.patch_index as usize, c.s, c.t, &controls)
                    .0
            })
            .collect();
        assert_close(&extract(&gpu.read(&points), n, 0, 3), &expected, 1e-4, name);
    }
}
