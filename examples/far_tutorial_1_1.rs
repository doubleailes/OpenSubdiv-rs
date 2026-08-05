//! Port of OpenSubdiv's `far_tutorial_1_1`: instantiate a
//! `Far::TopologyRefiner` from a descriptor for a cube, uniformly refine it,
//! interpolate the vertex positions and print the refined mesh as an OBJ
//! file on stdout.
//!
//! Run with:
//! ```sh
//! cargo run --example far_tutorial_1_1 > cube.obj
//! ```

use opensubdiv_rs::far::{
    PrimvarRefiner, TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

const MAX_LEVEL: usize = 2;

fn main() {
    // Cube geometry from catmark_cube.h.
    let verts_per_face = [4usize; 6];
    let face_verts: [u32; 24] = [
        0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
    ];
    let positions: [[f32; 3]; 8] = [
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
    ];

    // Populate a topology descriptor with our raw data and instantiate a
    // TopologyRefiner with it.
    let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default()
            .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeOnly),
    )
    .expect("cube topology is valid");

    // Uniformly refine the topology up to MAX_LEVEL.
    refiner.refine_uniform(UniformOptions::new(MAX_LEVEL));

    // Interpolate vertex primvar data: refine the positions level by level.
    let primvar_refiner = PrimvarRefiner::new(&refiner);
    let mut verts = positions.to_vec();
    for level in 1..=MAX_LEVEL {
        let mut refined = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
        primvar_refiner.interpolate(level, &verts, &mut refined);
        verts = refined;
    }

    // Output an OBJ of the highest level refined.
    let last_level = refiner.level(MAX_LEVEL);

    for p in &verts {
        println!("v {} {} {}", p[0], p[1], p[2]);
    }
    for face in 0..last_level.num_faces() {
        print!("f");
        for &v in last_level.face_vertices(face) {
            print!(" {}", v + 1); // OBJ indices are 1-based
        }
        println!();
    }
}
