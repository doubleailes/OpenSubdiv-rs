//! Tessellate the limit surface of an adaptively refined cube on the GPU
//! and write it as OBJ, with normals:
//!
//! ```sh
//! cargo run -p opensubdiv-wgpu --example tessellate > cube.obj
//! ```
//!
//! The per-frame work of a renderer, headless: one stencil dispatch refines
//! the cage into every control point the patches index, then one patch
//! dispatch evaluates a uniform `N x N` grid on each patch, with
//! derivatives for the normals. A renderer would draw the output buffers
//! directly (with `VERTEX` usage added) instead of reading them back.
//!
//! Each patch is tessellated on its own, so where patches of different
//! depths meet, their grids do not line up: the surface is continuous but
//! the mesh has T-junctions there.

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchTableFactory, StencilTableFactory, StencilTableOptions,
    TopologyDescriptor, TopologyRefinerFactory,
};
use opensubdiv_rs::osd::GpuPatchTable;
use opensubdiv_rs::sdc;
use opensubdiv_wgpu::{
    Evaluator, PatchCoordBuffer, PatchOutputs, PatchTableBuffers, PrimvarBuffer,
    StencilTableBuffers,
};
use wgpu::util::DeviceExt;

/// Grid segments per patch edge.
const N: usize = 6;

fn main() {
    // The far tutorial's cube.
    let verts_per_face = [4usize; 6];
    let face_verts: [u32; 24] = [
        0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
    ];
    let cage: [f32; 24] = [
        -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
        -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5,
    ];

    // Once per topology, on the CPU.
    let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .expect("valid topology");
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    let patches = PatchTableFactory::create(&refiner).expect("patch table");
    let gpu_patches = GpuPatchTable::new(&patches).expect("GPU patch table");
    let stencils = StencilTableFactory::append_local_point_stencils(
        &StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls()),
        gpu_patches.local_point_stencils(),
    );

    // An N x N grid on every patch, in ptex-face coordinates.
    let mut coords = Vec::new();
    for patch in 0..patches.num_patches() {
        let param = patches.patch_param(patch);
        for j in 0..=N {
            for i in 0..=N {
                let (u, v) = param.unnormalize(i as f32 / N as f32, j as f32 / N as f32);
                coords.push(gpu_patches.patch_coord(patch, u, v));
            }
        }
    }

    // The device.
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("a GPU adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("a device");
    eprintln!("adapter: {}", adapter.get_info().name);

    // Upload the tables and allocate the buffers.
    let evaluator = Evaluator::new(&device);
    let stencil_buffers = StencilTableBuffers::new(&device, &stencils);
    let patch_buffers = PatchTableBuffers::new(&device, &gpu_patches);
    let coord_buffer = PatchCoordBuffer::new(&device, &coords);
    let storage = |floats: usize, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 * floats as u64,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        })
    };
    let cage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cage"),
        contents: &cage
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let controls = storage(3 * stencils.num_stencils(), wgpu::BufferUsages::empty());
    let n = coords.len();
    let [points, du, dv] = [(); 3].map(|_| storage(3 * n, wgpu::BufferUsages::COPY_SRC));

    // The per-frame work: two dispatches.
    let mut encoder = device.create_command_encoder(&Default::default());
    evaluator.eval_stencils(
        &device,
        &mut encoder,
        &stencil_buffers,
        PrimvarBuffer::new(&cage_buffer, 0, 3, 3),
        PrimvarBuffer::new(&controls, 0, 3, 3),
    );
    evaluator.eval_patches(
        &device,
        &mut encoder,
        &patch_buffers,
        &coord_buffer,
        PrimvarBuffer::new(&controls, 0, 3, 3),
        PatchOutputs::with_derivatives(
            PrimvarBuffer::new(&points, 0, 3, 3),
            PrimvarBuffer::new(&du, 0, 3, 3),
            PrimvarBuffer::new(&dv, 0, 3, 3),
        ),
    );
    queue.submit([encoder.finish()]);

    // Read back and write OBJ.
    let read = |buffer: &wgpu::Buffer| -> Vec<[f32; 3]> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let bytes = staging.get_mapped_range(..).expect("mapped range");
        bytes
            .chunks_exact(12)
            .map(|c| [0, 4, 8].map(|k| f32::from_ne_bytes([c[k], c[k + 1], c[k + 2], c[k + 3]])))
            .collect()
    };
    let (points, du, dv) = (read(&points), read(&du), read(&dv));

    for p in &points {
        println!("v {} {} {}", p[0], p[1], p[2]);
    }
    for (a, b) in du.iter().zip(&dv) {
        let n = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        println!("vn {} {} {}", n[0] / len, n[1] / len, n[2] / len);
    }
    let row = N + 1;
    for patch in 0..patches.num_patches() {
        let base = patch * row * row + 1; // OBJ indices start at 1
        for j in 0..N {
            for i in 0..N {
                let corners = [
                    base + j * row + i,
                    base + j * row + i + 1,
                    base + (j + 1) * row + i + 1,
                    base + (j + 1) * row + i,
                ];
                let [a, b, c, d] = corners.map(|k| format!("{k}//{k}"));
                println!("f {a} {b} {c} {d}");
            }
        }
    }
    eprintln!(
        "{} patches, {} control points, {} vertices",
        patches.num_patches(),
        stencils.num_stencils(),
        n
    );
}
