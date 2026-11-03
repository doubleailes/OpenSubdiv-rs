//! Time the CPU evaluation (`opensubdiv-rs`, single-threaded) against the
//! GPU kernels on the same tables:
//!
//! ```sh
//! cargo run --release -p opensubdiv-wgpu --example bench
//! ```
//!
//! Two workloads on a bumpy `R x R` quad grid:
//!
//! - **uniform**: `StencilTable::update_values` against one stencil dispatch,
//!   producing the last level of a uniform refinement.
//! - **limit**: the tessellate example's frame (stencils to patch controls,
//!   then an `N x N` grid of limit points and derivatives on every patch)
//!   against `GpuPatchTable::evaluate` per coordinate.
//!
//! The tables are built once on the CPU and are not timed. A GPU frame
//! uploads the cage, records and submits the dispatches; its time is the
//! average over many queued frames (throughput), and the latency of one
//! frame waited on alone is printed beside it. Read-backs are timed
//! separately. Each GPU result is checked against the CPU one.

use std::time::Instant;

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchTableFactory, StencilTable, StencilTableFactory, StencilTableOptions,
    TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::osd::GpuPatchTable;
use opensubdiv_rs::sdc;
use opensubdiv_wgpu::{
    Evaluator, PatchCoordBuffer, PatchOutputs, PatchTableBuffers, PrimvarBuffer,
    StencilTableBuffers,
};

type P3 = [f32; 3];

/// Frames queued per GPU measurement.
const FRAMES: u32 = 50;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    evaluator: Evaluator,
}

impl Gpu {
    fn storage(&self, floats: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 * floats as u64,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        })
    }

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    }

    /// Milliseconds per frame over `FRAMES` queued frames, and the latency
    /// of one frame waited on alone.
    fn time_frames(&self, mut frame: impl FnMut(&mut wgpu::CommandEncoder)) -> (f64, f64) {
        let submit = |frame: &mut dyn FnMut(&mut wgpu::CommandEncoder)| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            frame(&mut encoder);
            self.queue.submit([encoder.finish()]);
        };
        // Warm up (pipeline creation, first-use allocations).
        for _ in 0..3 {
            submit(&mut frame);
        }
        self.wait();

        let start = Instant::now();
        for _ in 0..FRAMES {
            submit(&mut frame);
        }
        self.wait();
        let throughput = ms(start) / FRAMES as f64;

        let mut latency = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            submit(&mut frame);
            self.wait();
            latency = latency.min(ms(start));
        }
        (throughput, latency)
    }

    /// Read `buffer` back, returning the points and the milliseconds taken.
    fn read(&self, buffer: &wgpu::Buffer) -> (Vec<P3>, f64) {
        let start = Instant::now();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map"));
        self.wait();
        let bytes = staging.get_mapped_range(..).expect("mapped range");
        let points = bytes
            .as_chunks::<4>()
            .0
            .as_chunks::<3>()
            .0
            .iter()
            .map(|xyz| xyz.map(f32::from_ne_bytes))
            .collect();
        (points, ms(start))
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

/// Best of several runs, in milliseconds, repeating for at least ~0.3 s.
fn time_cpu(mut run: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    let begin = Instant::now();
    let mut runs = 0;
    while runs < 3 || (runs < 50 && begin.elapsed().as_secs_f64() < 0.3) {
        let start = Instant::now();
        run();
        best = best.min(ms(start));
        runs += 1;
    }
    best
}

fn max_diff(a: &[P3], b: &[P3]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
        .fold(0.0, f32::max)
}

fn flat(points: &[P3]) -> Vec<u8> {
    points
        .iter()
        .flatten()
        .flat_map(|x| x.to_ne_bytes())
        .collect()
}

/// A bumpy `r x r` quad grid.
fn grid(r: usize) -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let row = r + 1;
    let mut face_verts = Vec::with_capacity(4 * r * r);
    for j in 0..r {
        for i in 0..r {
            let v = (j * row + i) as u32;
            face_verts.extend([v, v + 1, v + 1 + row as u32, v + row as u32]);
        }
    }
    let cage = (0..row * row)
        .map(|k| {
            let (x, y) = ((k % row) as f32 / r as f32, (k / row) as f32 / r as f32);
            [x, y, 0.05 * ((17.0 * x).sin() + (13.0 * y).cos())]
        })
        .collect();
    (vec![4; r * r], face_verts, cage)
}

fn refiner(r: usize) -> (opensubdiv_rs::far::TopologyRefiner, Vec<P3>) {
    let (verts_per_face, face_verts, cage) = grid(r);
    let descriptor = TopologyDescriptor::new(cage.len(), &verts_per_face, &face_verts);
    let refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .expect("valid topology");
    (refiner, cage)
}

fn report(name: &str, outputs: usize, cpu: f64, gpu: (f64, f64), read: f64, diff: f32) {
    println!(
        "{name:<26} {outputs:>10} {cpu:>10.3} {:>10.3} {:>10.3} {read:>10.3} {:>8.1}x {diff:>9.1e}",
        gpu.0,
        gpu.1,
        cpu / gpu.0
    );
}

fn bench_uniform(gpu: &Gpu, r: usize, level: usize) {
    let (mut refiner, cage) = refiner(r);
    refiner.refine_uniform(UniformOptions::new(level));
    let stencils: StencilTable = StencilTableFactory::create(
        &refiner,
        StencilTableOptions {
            generate_intermediate_levels: false,
            generate_control_verts: false,
        },
    );
    let n = stencils.num_stencils();
    eprintln!(
        "  {n} stencils, {} weights ({:.1} per stencil)",
        stencils.weights().len(),
        stencils.weights().len() as f64 / n as f64
    );

    let mut cpu_out = vec![[0.0; 3]; n];
    let cpu = time_cpu(|| stencils.update_values(&cage, &mut cpu_out));

    let buffers = StencilTableBuffers::new(&gpu.device, &stencils);
    let cage_buffer = gpu.storage(3 * cage.len(), wgpu::BufferUsages::COPY_DST);
    let out = gpu.storage(3 * n, wgpu::BufferUsages::COPY_SRC);
    let cage_bytes = flat(&cage);
    let timing = gpu.time_frames(|encoder| {
        gpu.queue.write_buffer(&cage_buffer, 0, &cage_bytes);
        gpu.evaluator.eval_stencils(
            &gpu.device,
            encoder,
            &buffers,
            PrimvarBuffer::new(&cage_buffer, 0, 3, 3),
            PrimvarBuffer::new(&out, 0, 3, 3),
        );
    });
    let (gpu_out, read) = gpu.read(&out);
    report(
        &format!("uniform {r}x{r} level {level}"),
        n,
        cpu,
        timing,
        read,
        max_diff(&cpu_out, &gpu_out),
    );
}

fn bench_limit(gpu: &Gpu, r: usize, isolation: usize, segments: usize) {
    let (mut refiner, cage) = refiner(r);
    refiner.refine_adaptive(AdaptiveOptions::new(isolation));
    let patches = PatchTableFactory::create(&refiner).expect("patch table");
    let gpu_patches = GpuPatchTable::new(&patches).expect("GPU patch table");
    let stencils = StencilTableFactory::append_local_point_stencils(
        &StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls()),
        gpu_patches.local_point_stencils(),
    );
    let mut coords = Vec::new();
    for patch in 0..patches.num_patches() {
        let param = patches.patch_param(patch);
        for j in 0..=segments {
            for i in 0..=segments {
                let (u, v) =
                    param.unnormalize(i as f32 / segments as f32, j as f32 / segments as f32);
                coords.push(gpu_patches.patch_coord(patch, u, v));
            }
        }
    }
    let n = coords.len();

    let mut controls = vec![[0.0; 3]; stencils.num_stencils()];
    let mut cpu_points = vec![[0.0; 3]; n];
    let cpu = time_cpu(|| {
        stencils.update_values(&cage, &mut controls);
        for (coord, point) in coords.iter().zip(&mut cpu_points) {
            // The derivatives are computed alongside; keep them alive.
            let (p, du, dv) = gpu_patches.evaluate(coord, &controls);
            *point = std::hint::black_box((p, du, dv)).0;
        }
    });

    let stencil_buffers = StencilTableBuffers::new(&gpu.device, &stencils);
    let patch_buffers = PatchTableBuffers::new(&gpu.device, &gpu_patches);
    let coord_buffer = PatchCoordBuffer::new(&gpu.device, &coords);
    let cage_buffer = gpu.storage(3 * cage.len(), wgpu::BufferUsages::COPY_DST);
    let control_buffer = gpu.storage(3 * stencils.num_stencils(), wgpu::BufferUsages::empty());
    let [points, du, dv] = [(); 3].map(|_| gpu.storage(3 * n, wgpu::BufferUsages::COPY_SRC));
    let cage_bytes = flat(&cage);
    let timing = gpu.time_frames(|encoder| {
        gpu.queue.write_buffer(&cage_buffer, 0, &cage_bytes);
        gpu.evaluator.eval_stencils(
            &gpu.device,
            encoder,
            &stencil_buffers,
            PrimvarBuffer::new(&cage_buffer, 0, 3, 3),
            PrimvarBuffer::new(&control_buffer, 0, 3, 3),
        );
        gpu.evaluator.eval_patches(
            &gpu.device,
            encoder,
            &patch_buffers,
            &coord_buffer,
            PrimvarBuffer::new(&control_buffer, 0, 3, 3),
            PatchOutputs::with_derivatives(
                PrimvarBuffer::new(&points, 0, 3, 3),
                PrimvarBuffer::new(&du, 0, 3, 3),
                PrimvarBuffer::new(&dv, 0, 3, 3),
            ),
        );
    });
    let (gpu_points, read) = gpu.read(&points);
    report(
        &format!("limit {r}x{r} iso {isolation} {segments}x{segments}"),
        n,
        cpu,
        timing,
        read,
        max_diff(&cpu_points, &gpu_points),
    );
}

fn main() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("a GPU adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("a device");
    let evaluator = Evaluator::new(&device);
    let gpu = Gpu {
        device,
        queue,
        evaluator,
    };
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "workload", "outputs", "cpu ms", "gpu ms", "gpu lat", "readback", "speedup", "max diff"
    );
    for (r, level) in [(32, 2), (32, 4), (128, 2), (128, 3), (128, 4), (256, 4)] {
        bench_uniform(&gpu, r, level);
    }
    for (r, isolation, segments) in [(32, 2, 8), (128, 2, 8), (256, 2, 8), (256, 2, 16)] {
        bench_limit(&gpu, r, isolation, segments);
    }
}
