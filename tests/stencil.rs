//! Stencil-table tests: factorized stencils must reproduce the results of
//! running the PrimvarRefiner level by level.

use opensubdiv_rs::far::{
    PrimvarRefiner, StencilTableFactory, StencilTableOptions, TopologyDescriptor,
    TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

const CUBE_VERTS_PER_FACE: [usize; 6] = [4; 6];
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

fn cube_refiner(levels: usize) -> opensubdiv_rs::far::TopologyRefiner {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(levels));
    refiner
}

fn refine_positions(refiner: &opensubdiv_rs::far::TopologyRefiner) -> Vec<Vec<P3>> {
    PrimvarRefiner::new(refiner).interpolate_all(&CUBE_POSITIONS)
}

fn assert_all_close(actual: &[P3], expected: &[P3]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        for (x, y) in a.iter().zip(e) {
            assert!((x - y).abs() < 1e-6, "expected {e:?}, got {a:?}");
        }
    }
}

#[test]
fn stencils_match_primvar_refinement() {
    let refiner = cube_refiner(2);
    let by_level = refine_positions(&refiner);

    // Default options: stencils for all refined levels, no control verts.
    let table = StencilTableFactory::create(&refiner, StencilTableOptions::default());
    assert_eq!(table.num_control_vertices(), 8);
    assert_eq!(table.num_stencils(), 26 + 98);

    let mut values = vec![[0.0f32; 3]; table.num_stencils()];
    table.update_values(&CUBE_POSITIONS, &mut values);
    assert_all_close(&values[..26], &by_level[0]);
    assert_all_close(&values[26..], &by_level[1]);

    // Every stencil is a convex-combination: its weights sum to one.
    for i in 0..table.num_stencils() {
        let sum: f32 = table.stencil(i).weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "stencil {i} weights sum to {sum}");
    }
}

#[test]
fn stencil_options_control_shape() {
    let refiner = cube_refiner(2);
    let by_level = refine_positions(&refiner);

    // Last level only.
    let table = StencilTableFactory::create(
        &refiner,
        StencilTableOptions {
            generate_intermediate_levels: false,
            generate_control_verts: false,
        },
    );
    assert_eq!(table.num_stencils(), 98);
    let mut values = vec![[0.0f32; 3]; 98];
    table.update_values(&CUBE_POSITIONS, &mut values);
    assert_all_close(&values, &by_level[1]);

    // With prepended identity stencils for the control vertices.
    let table = StencilTableFactory::create(
        &refiner,
        StencilTableOptions {
            generate_intermediate_levels: false,
            generate_control_verts: true,
        },
    );
    assert_eq!(table.num_stencils(), 8 + 98);
    let mut values = vec![[0.0f32; 3]; 8 + 98];
    table.update_values(&CUBE_POSITIONS, &mut values);
    assert_all_close(&values[..8], &CUBE_POSITIONS);
    assert_all_close(&values[8..], &by_level[1]);
    for v in 0..8 {
        let stencil = table.stencil(v);
        assert_eq!(stencil.indices, &[v as u32]);
        assert_eq!(stencil.weights, &[1.0]);
    }
}

#[test]
fn limit_stencils_match_primvar_limit() {
    let refiner = cube_refiner(2);
    let by_level = refine_positions(&refiner);

    let primvar = PrimvarRefiner::new(&refiner);
    let mut expected = by_level[1].clone();
    primvar.limit(&by_level[1], &mut expected);

    let table = StencilTableFactory::create_limit(&refiner);
    assert_eq!(table.num_stencils(), 98);
    let mut values = vec![[0.0f32; 3]; 98];
    table.update_values(&CUBE_POSITIONS, &mut values);
    assert_all_close(&values, &expected);

    for i in 0..table.num_stencils() {
        let sum: f32 = table.stencil(i).weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "limit stencil {i} sums to {sum}");
    }
}

#[test]
fn stencils_work_for_loop_scheme() {
    let verts_per_face = [3usize; 4];
    let face_verts = [0u32, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];
    let positions: [P3; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let primvar = PrimvarRefiner::new(&refiner);
    let by_level = primvar.interpolate_all(&positions);

    let table = StencilTableFactory::create(
        &refiner,
        StencilTableOptions {
            generate_intermediate_levels: false,
            generate_control_verts: false,
        },
    );
    assert_eq!(table.num_stencils(), by_level[1].len());
    let mut values = vec![[0.0f32; 3]; table.num_stencils()];
    table.update_values(&positions, &mut values);
    assert_all_close(&values, &by_level[1]);
}
