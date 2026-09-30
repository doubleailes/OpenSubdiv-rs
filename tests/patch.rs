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

/// Patch control values: every level's vertex values concatenated, base
/// level first.
fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

/// Vertex values of the refiner's last level only.
fn last_level_positions(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
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

    let controls = patch_controls(&refiner, &positions);
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

    let controls = patch_controls(&refiner, &CUBE_POSITIONS);
    let last = last_level_positions(&refiner, &CUBE_POSITIONS);
    let mut limits = last.clone();
    PrimvarRefiner::new(&refiner).limit(&last, &mut limits);

    let table = PatchTableFactory::create(&refiner).unwrap();

    // At level 2 the only extraordinary vertices are the 8 descendants of
    // the cube corners (valence 3), each incident 3 faces: 24 Gregory
    // end-cap patches, 72 regular B-spline patches, no bilinear fallback.
    let count = |t: PatchType| {
        (0..table.num_patches())
            .filter(|&p| table.patch_type(p) == t)
            .count()
    };
    assert_eq!(table.num_patches(), 96);
    assert_eq!(count(PatchType::Regular), 72);
    assert_eq!(count(PatchType::GregoryBasis), 24);
    assert_eq!(count(PatchType::Quads), 0);

    // For every patch — B-spline and Gregory alike — evaluating at its
    // (s,t) = (0,0) corner must reproduce the limit position of the
    // corresponding vertex, validating both bases against the independent
    // limit masks.
    let level = refiner.level(2);
    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        let (u, v) = param.unnormalize(0.0, 0.0);
        let (point, _, _) = table.evaluate(p, u, v, &controls);
        let c0 = level.face_vertices(table.patch_face(p) as usize)[0] as usize;
        assert_close(point, limits[c0], 1e-5);
    }
}

#[test]
fn patches_are_c0_across_boundaries() {
    // Evaluate on both sides of internal patch boundaries — including
    // Gregory/B-spline junctions around the extraordinary vertices — at the
    // exact same ptex location: the surface must be continuous.
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));
    let controls = patch_controls(&refiner, &CUBE_POSITIONS);
    let table = PatchTableFactory::create(&refiner).unwrap();
    let map = PatchMap::new(&table);

    let eps = 1e-3f32;
    let lines = [0.25f32, 0.5, 0.75];
    let along = [0.05f32, 0.2, 0.4, 0.65, 0.9];
    for ptex in 0..6usize {
        for &line in &lines {
            for &x in &along {
                // Vertical boundary u = line.
                let a = map.find_patch(ptex, line - eps, x).unwrap();
                let b = map.find_patch(ptex, line + eps, x).unwrap();
                assert_ne!(a, b);
                let (pa, _, _) = table.evaluate(a, line, x, &controls);
                let (pb, _, _) = table.evaluate(b, line, x, &controls);
                assert_close(pa, pb, 1e-4);
                // Horizontal boundary v = line.
                let a = map.find_patch(ptex, x, line - eps).unwrap();
                let b = map.find_patch(ptex, x, line + eps).unwrap();
                let (pa, _, _) = table.evaluate(a, x, line, &controls);
                let (pb, _, _) = table.evaluate(b, x, line, &controls);
                assert_close(pa, pb, 1e-4);
            }
        }
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

    let controls = patch_controls(&refiner, &positions);
    let last = last_level_positions(&refiner, &positions);
    let mut limits = last.clone();
    PrimvarRefiner::new(&refiner).limit(&last, &mut limits);

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
    let controls = patch_controls(&refiner, &positions);
    for ptex in 0..5 {
        let patch = map.find_patch(ptex, 0.3, 0.4).unwrap();
        assert_eq!(table.patch_param(patch).ptex_face as usize, ptex);
        let (point, _, _) = table.evaluate(patch, 0.3, 0.4, &controls);
        // The pentagon is flat: the limit surface stays in the plane (the
        // Gregory stencils are affine combinations) and near the hull.
        assert!(point[2].abs() < 1e-4);
        assert!(point[0].abs() <= 1.02 && point[1].abs() <= 1.02);
    }
}

#[test]
fn gregory_end_caps_are_exact_at_evs_and_consistent_across_levels() {
    let evaluate_at = |levels: usize, u: f32, v: f32| -> P3 {
        let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
        let mut refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        refiner.refine_uniform(UniformOptions::new(levels));
        let controls = patch_controls(&refiner, &CUBE_POSITIONS);
        let table = PatchTableFactory::create(&refiner).unwrap();
        let map = PatchMap::new(&table);
        // Ptex face 0 is base face 0 (verts 0,1,3,2): its (0,0) corner is
        // the extraordinary vertex 0.
        let patch = map.find_patch(0, u, v).unwrap();
        let (point, _, _) = table.evaluate(patch, u, v, &controls);
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
    let level1 = last_level_positions(&refiner, &CUBE_POSITIONS);
    let mut limits = level1.clone();
    PrimvarRefiner::new(&refiner).limit(&level1, &mut limits);
    // Child of vertex 0 at level 1 is vertex 6 + 12 + 0 = 18.
    let truth = limits[18];

    // Gregory end caps interpolate the extraordinary vertex's limit
    // position exactly, at any refinement level.
    assert_close(evaluate_at(2, 0.0, 0.0), truth, 1e-5);
    assert_close(evaluate_at(4, 0.0, 0.0), truth, 1e-5);

    // Near (but not at) the EV, the end caps built at different refinement
    // levels approximate the same limit surface.
    let p2 = evaluate_at(2, 0.03, 0.05);
    let p4 = evaluate_at(4, 0.03, 0.05);
    assert_close(p2, p4, 1e-2);
}
