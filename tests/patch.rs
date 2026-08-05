//! Patch-table tests: B-spline patch extraction, boundary/corner phantom
//! folding, parametric transforms (PatchParam/PatchMap) and derivative
//! evaluation, validated against limit stencils and linear precision.

use opensubdiv_rs::far::{
    Error, PatchMap, PatchTableFactory, PatchType, PrimvarRefiner, TopologyDescriptor,
    TopologyRefiner, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

fn assert_close(actual: P3, expected: P3, tol: f32) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - e).abs() < tol, "expected {expected:?}, got {actual:?}");
    }
}

fn refined_positions(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    if refiner.max_level() == 0 {
        return base.to_vec();
    }
    PrimvarRefiner::new(refiner)
        .interpolate_all(base)
        .pop()
        .unwrap()
}

/// A flat 5x5-vertex grid carrying a linear function: the Catmull-Clark
/// limit surface reproduces linear data exactly — across interior,
/// boundary and (pinned) corner patches alike — so patch evaluation must
/// return the exact plane, with exact constant derivatives, at any
/// parametric location. This exercises basis functions, phantom folding,
/// PatchParam rotations and the PatchMap in one sweep.
#[test]
fn grid_patches_have_linear_precision_everywhere() {
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
    refiner.refine_uniform(UniformOptions::new(1));

    let controls = refined_positions(&refiner, &positions);
    let table = PatchTableFactory::create(&refiner).unwrap();

    // Every vertex of the grid is regular (interior valence-4, boundary
    // valence-3, or pinned corner): every patch is a B-spline patch.
    assert_eq!(table.num_patches(), 64);
    for p in 0..table.num_patches() {
        assert_eq!(table.patch_type(p), PatchType::Regular);
    }

    let map = PatchMap::new(&table);
    let samples = [
        (0.1f32, 0.7f32),
        (0.5, 0.5),
        (0.95, 0.05),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.33, 0.99),
    ];
    for base_face in 0..16usize {
        let (i, j) = ((base_face % 4) as f32, (base_face / 4) as f32);
        let ptex = table.ptex_indices().face_id(base_face) as usize;
        for &(u, v) in &samples {
            let patch = map.find_patch(ptex, u, v).expect("no holes in the grid");
            let (point, du, dv) = table.evaluate(patch, u, v, &controls);
            assert_close(point, linear(i + u, j + v), 1e-4);
            assert_close(du, [1.0, 0.0, 0.25], 1e-4);
            assert_close(dv, [0.0, 1.0, -0.5], 1e-4);
        }
    }
}

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

#[test]
fn cube_patch_corners_match_limit_stencils() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let controls = refined_positions(&refiner, &CUBE_POSITIONS);
    let mut limits = controls.clone();
    PrimvarRefiner::new(&refiner).limit(&controls, &mut limits);

    let table = PatchTableFactory::create(&refiner).unwrap();

    // At level 2 the only extraordinary vertices are the 8 descendants of
    // the cube corners (valence 3), each incident 3 faces: 24 bilinear
    // fallback patches, 72 regular B-spline patches.
    let regular = (0..table.num_patches())
        .filter(|&p| table.patch_type(p) == PatchType::Regular)
        .count();
    assert_eq!(table.num_patches(), 96);
    assert_eq!(regular, 72);

    // For every regular patch, evaluating at its (s,t) = (0,0) corner must
    // reproduce the limit position of the corresponding vertex — validating
    // the B-spline basis against the independent limit masks.
    let level = refiner.level(2);
    for p in 0..table.num_patches() {
        if table.patch_type(p) != PatchType::Regular {
            continue;
        }
        let param = table.patch_param(p);
        let (u, v) = param.unnormalize(0.0, 0.0);
        let (point, _, _) = table.evaluate(p, u, v, &controls);
        let c0 = level.face_vertices(table.patch_face(p) as usize)[0] as usize;
        assert_close(point, limits[c0], 1e-5);
    }
}

#[test]
fn boundary_and_corner_patches_match_limit_stencils() {
    // A single quad with pinned corners: at level 2 every patch is regular
    // (interior, boundary, or corner), so every patch corner must evaluate
    // to the limit of its vertex — validating phantom-point folding against
    // the crease/corner limit masks.
    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];
    let positions: [P3; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.3],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, -0.2],
    ];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let controls = refined_positions(&refiner, &positions);
    let mut limits = controls.clone();
    PrimvarRefiner::new(&refiner).limit(&controls, &mut limits);

    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_patches(), 16);
    for p in 0..table.num_patches() {
        assert_eq!(table.patch_type(p), PatchType::Regular);
    }

    let level = refiner.level(2);
    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        let (u, v) = param.unnormalize(0.0, 0.0);
        let (point, _, _) = table.evaluate(p, u, v, &controls);
        let c0 = level.face_vertices(table.patch_face(p) as usize)[0] as usize;
        assert_close(point, limits[c0], 1e-5);
    }

    // The pinned mesh corners are interpolated exactly by their patches.
    let map = PatchMap::new(&table);
    let corners = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    for (corner, &(u, v)) in corners.iter().enumerate() {
        let patch = map.find_patch(0, u, v).unwrap();
        let (point, _, _) = table.evaluate(patch, u, v, &controls);
        assert_close(point, positions[corner], 1e-5);
    }
}

#[test]
fn patch_param_and_patch_map_are_consistent() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(3));

    let table = PatchTableFactory::create(&refiner).unwrap();
    let map = PatchMap::new(&table);

    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        // A point strictly inside the patch maps back to the same patch.
        let (u, v) = param.unnormalize(0.3, 0.6);
        assert_eq!(map.find_patch(param.ptex_face as usize, u, v), Some(p));
        // normalize is the inverse of unnormalize.
        let (s, t) = param.normalize(u, v);
        assert!((s - 0.3).abs() < 1e-5 && (t - 0.6).abs() < 1e-5);

        // Basis invariants: point weights sum to 1, derivatives to 0.
        let basis = table.evaluate_basis(p, u, v);
        let sum: f32 = basis.weights.iter().sum();
        let dsum: f32 = basis.du_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(dsum.abs() < 1e-2);
    }
}

#[test]
fn non_quad_base_faces_use_ptex_subfaces() {
    // A pentagon: 5 ptex faces rooted at its level-1 children.
    let verts_per_face = [5usize];
    let face_verts = [0u32, 1, 2, 3, 4];
    let positions: Vec<P3> = (0..5)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / 5.0;
            [a.cos(), a.sin(), 0.0]
        })
        .collect();
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor = TopologyDescriptor::new(5, &verts_per_face, &face_verts);

    // Without refinement, non-quad faces cannot be patched.
    let refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    assert!(matches!(
        PatchTableFactory::create(&refiner),
        Err(Error::PatchesRequireRefinement)
    ));

    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();

    assert_eq!(table.ptex_indices().num_faces(), 5);
    assert_eq!(table.num_patches(), 5 * 4); // level 2: each ptex face holds 4 patches

    let map = PatchMap::new(&table);
    let controls = refined_positions(&refiner, &positions);
    for ptex in 0..5 {
        let patch = map.find_patch(ptex, 0.3, 0.4).unwrap();
        assert_eq!(table.patch_param(patch).ptex_face as usize, ptex);
        let (point, _, _) = table.evaluate(patch, 0.3, 0.4, &controls);
        // The pentagon is flat: the limit surface stays in the plane and
        // within the convex hull.
        assert!(point[2].abs() < 1e-5);
        assert!(point[0].abs() <= 1.0 + 1e-5 && point[1].abs() <= 1.0 + 1e-5);
    }
}

#[test]
fn loop_patches_unsupported() {
    let verts_per_face = [3usize; 4];
    let face_verts = [0u32, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));
    assert!(matches!(
        PatchTableFactory::create(&refiner),
        Err(Error::LoopPatchesNotSupported)
    ));
}

#[test]
fn irregular_patches_converge_to_limit() {
    // Bilinear fallback patches near extraordinary vertices shrink with
    // refinement: evaluating near a cube corner must converge toward the
    // true limit position as the refinement level grows.
    let evaluate_near_ev = |levels: usize| -> P3 {
        let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
        let mut refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        refiner.refine_uniform(UniformOptions::new(levels));
        let controls = refined_positions(&refiner, &CUBE_POSITIONS);
        let table = PatchTableFactory::create(&refiner).unwrap();
        let map = PatchMap::new(&table);
        // Ptex face 0 is base face 0 (verts 0,1,3,2): its (0,0) corner is
        // the extraordinary vertex 0.
        let patch = map.find_patch(0, 0.0, 0.0).unwrap();
        let (point, _, _) = table.evaluate(patch, 0.0, 0.0, &controls);
        point
    };

    // True limit of vertex 0, from the limit stencils.
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));
    let level1 = refined_positions(&refiner, &CUBE_POSITIONS);
    let mut limits = level1.clone();
    PrimvarRefiner::new(&refiner).limit(&level1, &mut limits);
    // Child of vertex 0 at level 1 is vertex 6 + 12 + 0 = 18.
    let truth = limits[18];

    let distance = |p: P3| -> f32 {
        p.iter()
            .zip(&truth)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    };
    let e2 = distance(evaluate_near_ev(2));
    let e4 = distance(evaluate_near_ev(4));
    assert!(
        e4 < e2 * 0.5,
        "EV evaluation did not converge: level2 err {e2}, level4 err {e4}"
    );
    assert!(e4 < 0.01, "level-4 EV error too large: {e4}");
}
