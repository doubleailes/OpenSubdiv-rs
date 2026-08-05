//! Feature-adaptive refinement tests.
//!
//! The central invariant: an adaptively refined mesh describes the *same*
//! limit surface as a uniformly refined one — with far fewer patches and
//! sparse levels — so parametric evaluation of both must agree everywhere.

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchMap, PatchTable, PatchTableFactory, PatchType, PrimvarRefiner,
    TopologyDescriptor, TopologyRefiner, TopologyRefinerFactory, UniformOptions,
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

fn assert_close(actual: P3, expected: P3, tol: f32) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - e).abs() < tol, "expected {expected:?}, got {actual:?}");
    }
}

fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

/// Deterministic pseudo-random parametric samples.
fn samples(count: usize) -> Vec<(f32, f32)> {
    let mut seed = 0x2468aceu32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..count).map(|_| (rand(), rand())).collect()
}

/// Evaluate two patch tables at identical parametric locations and require
/// identical limit positions and derivatives.
fn assert_tables_agree(
    a: &PatchTable,
    a_controls: &[P3],
    b: &PatchTable,
    b_controls: &[P3],
    num_ptex: usize,
) {
    let map_a = PatchMap::new(a);
    let map_b = PatchMap::new(b);
    for ptex in 0..num_ptex {
        for &(u, v) in &samples(12) {
            let pa = map_a.find_patch(ptex, u, v).unwrap();
            let pb = map_b.find_patch(ptex, u, v).unwrap();
            let (qa, dua, dva) = a.evaluate(pa, u, v, a_controls);
            let (qb, dub, dvb) = b.evaluate(pb, u, v, b_controls);
            assert_close(qa, qb, 1e-4);
            assert_close(dua, dub, 2e-3);
            assert_close(dva, dvb, 2e-3);
        }
    }
}

#[test]
fn adaptive_cube_matches_uniform_with_fewer_patches() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut uniform = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    uniform.refine_uniform(UniformOptions::new(4));
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    let uniform_controls = patch_controls(&uniform, &CUBE_POSITIONS);

    let mut adaptive = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    adaptive.refine_adaptive(AdaptiveOptions::new(4));
    assert!(adaptive.is_adaptive());
    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();
    let adaptive_controls = patch_controls(&adaptive, &CUBE_POSITIONS);

    // Every cube face touches a corner EV, so early levels refine fully; at
    // each level from 2 on, 72 regular faces are emitted and only the 24 EV
    // faces descend, capped with Gregory patches at the isolation level:
    //   72 (L2) + 72 (L3) + 72 + 24 (L4) = 240 patches vs 1536 uniform.
    assert_eq!(uniform_table.num_patches(), 1536);
    assert_eq!(adaptive_table.num_patches(), 240);
    let count = |t: PatchType| {
        (0..adaptive_table.num_patches())
            .filter(|&p| adaptive_table.patch_type(p) == t)
            .count()
    };
    assert_eq!(count(PatchType::Regular), 216);
    assert_eq!(count(PatchType::GregoryBasis), 24);
    assert_eq!(count(PatchType::Quads), 0);

    // Level 4 is sparse: only the EV neighborhoods were refined.
    assert!(adaptive.level(4).num_faces() < uniform.level(4).num_faces());
    assert!(adaptive.num_vertices_total() < uniform.num_vertices_total());

    // Same limit surface everywhere.
    assert_tables_agree(
        &adaptive_table,
        &adaptive_controls,
        &uniform_table,
        &uniform_controls,
        6,
    );
}

#[test]
fn adaptive_does_not_refine_regular_meshes() {
    // A regular grid has nothing to isolate: adaptive refinement is a
    // no-op and patches live directly on the base level.
    let linear = |x: f32, y: f32| [x, y, 0.25 * x - 0.5 * y];
    let mut positions = Vec::new();
    for j in 0..5 {
        for i in 0..5 {
            positions.push(linear(i as f32, j as f32));
        }
    }
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            verts_per_face.push(4);
            let v = j * 5 + i;
            face_verts.extend_from_slice(&[v, v + 1, v + 6, v + 5]);
        }
    }
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor = TopologyDescriptor::new(25, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(5));

    assert_eq!(refiner.max_level(), 0);
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_patches(), 16);
    for p in 0..table.num_patches() {
        assert_eq!(table.patch_type(p), PatchType::Regular);
        assert_eq!(table.patch_param(p).depth, 0);
    }

    // Evaluation still has linear precision everywhere.
    let map = PatchMap::new(&table);
    for ptex in 0..16usize {
        let (i, j) = ((ptex % 4) as f32, (ptex / 4) as f32);
        for &(u, v) in &samples(6) {
            let patch = map.find_patch(ptex, u, v).unwrap();
            let (point, _, _) = table.evaluate(patch, u, v, &positions);
            assert_close(point, linear(i + u, j + v), 1e-4);
        }
    }
}

#[test]
fn semi_sharp_creases_stop_isolating_once_resolved() {
    // A 4x4 grid with a semi-sharp interior crease of sharpness 1.5: after
    // two subdivisions the sharpness decays to zero, so isolation stops at
    // level 2 even though 5 levels were requested.
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            verts_per_face.push(4usize);
            let v = j * 5 + i;
            face_verts.extend_from_slice(&[v, v + 1, v + 6, v + 5]);
        }
    }
    let creases = [[12u32, 17u32]]; // an interior vertical edge
    let weights = [1.5f32];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor =
        TopologyDescriptor::new(25, &verts_per_face, &face_verts).with_creases(&creases, &weights);

    let mut adaptive =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    adaptive.refine_adaptive(AdaptiveOptions::new(5));
    assert_eq!(adaptive.max_level(), 2);

    // The adaptive surface agrees with the uniformly refined one.
    let mut positions = Vec::new();
    for j in 0..5 {
        for i in 0..5 {
            positions.push([i as f32, j as f32, ((i * 7 + j * 3) % 5) as f32 * 0.1]);
        }
    }
    let mut uniform =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    uniform.refine_uniform(UniformOptions::new(2));

    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert!(adaptive_table.num_patches() < uniform_table.num_patches());
    assert_tables_agree(
        &adaptive_table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        16,
    );
}

#[test]
fn adaptive_pentagon_is_sparse_and_matches_uniform() {
    let verts_per_face = [5usize];
    let face_verts = [0u32, 1, 2, 3, 4];
    let positions: Vec<P3> = (0..5)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / 5.0;
            [a.cos(), a.sin(), 0.1 * k as f32]
        })
        .collect();
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor = TopologyDescriptor::new(5, &verts_per_face, &face_verts);

    let mut adaptive =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    adaptive.refine_adaptive(AdaptiveOptions::new(4));
    let mut uniform =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    uniform.refine_uniform(UniformOptions::new(4));

    // The interior EV keeps isolating to the cap; on a mesh this small its
    // one-ring covers the shallow levels entirely, but the deepest level is
    // sparse: only the EV neighborhood was refined.
    assert_eq!(adaptive.max_level(), 4);
    assert!(
        adaptive.level(4).num_faces() < uniform.level(4).num_faces(),
        "adaptive deepest level is not sparse"
    );
    assert!(adaptive.num_vertices_total() < uniform.num_vertices_total());

    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert!(adaptive_table.num_patches() < uniform_table.num_patches());
    assert_tables_agree(
        &adaptive_table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        5,
    );
}

#[test]
fn adaptive_bilinear_isolates_non_quads_once() {
    let verts_per_face = [5usize];
    let face_verts = [0u32, 1, 2, 3, 4];
    let positions: Vec<P3> = (0..5)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / 5.0;
            [a.cos(), a.sin(), 0.0]
        })
        .collect();
    let descriptor = TopologyDescriptor::new(5, &verts_per_face, &face_verts);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Bilinear,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(3));

    // Bilinear needs a single round to quad the pentagon; the children are
    // then regular (bilinear quads).
    assert_eq!(refiner.max_level(), 1);
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_patches(), 5);

    let controls = patch_controls(&refiner, &positions);
    let map = PatchMap::new(&table);
    let patch = map.find_patch(2, 0.25, 0.75).unwrap();
    let (point, _, _) = table.evaluate(patch, 0.25, 0.75, &controls);
    assert!(point[2].abs() < 1e-6);
}

#[test]
#[should_panic(expected = "stencil tables require uniform refinement")]
fn stencils_reject_adaptive_refiners() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(2));
    let _ = opensubdiv_rs::far::StencilTableFactory::create(
        &refiner,
        opensubdiv_rs::far::StencilTableOptions::default(),
    );
}
