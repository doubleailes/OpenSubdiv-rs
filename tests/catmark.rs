//! Catmull-Clark refinement tests, validated against the classic
//! Catmull-Clark subdivision rules and OpenSubdiv's behavior.

use opensubdiv_rs::far::{
    PrimvarRefiner, TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

/// The cube used by OpenSubdiv's far tutorials.
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

fn cube_refiner(options: sdc::Options, levels: usize) -> opensubdiv_rs::far::TopologyRefiner {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(levels));
    refiner
}

fn assert_close(actual: P3, expected: P3) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!(
            (a - e).abs() < 1e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn cube_refinement_counts() {
    let refiner = cube_refiner(sdc::Options::default(), 2);

    // Level 1: 6 face + 12 edge + 8 vertex children; 24 quads; 48 edges.
    let level1 = refiner.level(1);
    assert_eq!(level1.num_vertices(), 26);
    assert_eq!(level1.num_faces(), 24);
    assert_eq!(level1.num_edges(), 48);

    // Level 2: 24 + 48 + 26 = 98 vertices; 96 quads; 192 edges.
    let level2 = refiner.level(2);
    assert_eq!(level2.num_vertices(), 98);
    assert_eq!(level2.num_faces(), 96);
    assert_eq!(level2.num_edges(), 192);

    // Euler characteristic of a sphere-like mesh: V - E + F = 2.
    for l in 0..=2 {
        let level = refiner.level(l);
        assert_eq!(
            level.num_vertices() as i64 - level.num_edges() as i64 + level.num_faces() as i64,
            2
        );
    }
}

#[test]
fn cube_level1_positions_match_catmull_clark() {
    let refiner = cube_refiner(sdc::Options::default(), 1);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.interpolate(1, &CUBE_POSITIONS, &mut dst);

    // Child vertices are ordered faces (0..6), edges (6..18), verts (18..26).

    // Face point of face 0 (verts 0,1,3,2): the +z face center.
    assert_close(dst[0], [0.0, 0.0, 0.5]);

    // Edge point of edge (0,1) — the first edge encountered, index 0:
    // e' = (v0 + v1)/4 + (f_a + f_b)/4 with adjacent face centers
    // (0,0,0.5) and (0,-0.5,0):
    let e = refiner.level(0).find_edge(0, 1).unwrap() as usize;
    assert_close(dst[6 + e], [0.0, -0.375, 0.375]);

    // Vertex point of corner 0 (valence 3):
    // v' = 1/3 v + 1/9 sum(edge-adjacent) + 1/9 sum(face centers)
    //    = 5/9 * (-0.5, -0.5, 0.5).
    assert_close(dst[18], [-5.0 / 18.0, -5.0 / 18.0, 5.0 / 18.0]);

    // All smooth masks are convex: the refined cube must stay within the
    // original bounding box.
    for p in &dst {
        for c in p {
            assert!(c.abs() <= 0.5 + 1e-6);
        }
    }
}

#[test]
fn infinitely_sharp_cube_stays_a_cube() {
    // Crease all 12 edges: subdivision must reproduce the cube exactly.
    let mut crease_pairs = Vec::new();
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    {
        let base = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        for e in 0..base.level(0).num_edges() {
            crease_pairs.push(base.level(0).edge_vertices(e));
        }
    }
    let weights = vec![sdc::SHARPNESS_INFINITE; crease_pairs.len()];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&crease_pairs, &weights);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut src = CUBE_POSITIONS.to_vec();
    for level in 1..=2 {
        let mut dst = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
        primvar.interpolate(level, &src, &mut dst);
        src = dst;
    }
    // Every refined vertex must lie on the surface of the cube:
    // at least one coordinate at +-0.5, all within the box.
    for p in &src {
        let mx = p.iter().fold(0.0f32, |m, c| m.max(c.abs()));
        assert!((mx - 0.5).abs() < 1e-6, "point {p:?} left the cube surface");
    }
    // Original corners persist (corner rule from three sharp edges).
    for corner in &CUBE_POSITIONS {
        assert!(
            src.iter()
                .any(|p| p.iter().zip(corner).all(|(a, b)| (a - b).abs() < 1e-6)),
            "corner {corner:?} not preserved"
        );
    }
}

#[test]
fn semi_sharp_edge_blends_masks() {
    // A cube edge with sharpness 0.5: the edge child vertex must be the
    // 50/50 blend of the crease (midpoint) and smooth masks.
    let crease_pairs = [[0u32, 1u32]];
    let weights = [0.5f32];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&crease_pairs, &weights);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.interpolate(1, &CUBE_POSITIONS, &mut dst);

    let e = refiner.level(0).find_edge(0, 1).unwrap() as usize;
    // Smooth: (0, -0.375, 0.375); crease midpoint: (0, -0.5, 0.5).
    assert_close(dst[6 + e], [0.0, -0.4375, 0.4375]);
}

#[test]
fn boundary_edge_and_corner_pins_quad_corners() {
    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];
    let positions: [P3; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut src = positions.to_vec();
    for level in 1..=2 {
        let mut dst = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
        primvar.interpolate(level, &src, &mut dst);
        src = dst;
    }
    // The four corners must be interpolated exactly, and all boundary
    // edge-children must stay on the unit square's boundary.
    for corner in &positions {
        assert!(
            src.iter()
                .any(|p| p.iter().zip(corner).all(|(a, b)| (a - b).abs() < 1e-6)),
            "corner {corner:?} not interpolated"
        );
    }
    for p in &src {
        assert!(p[0] >= -1e-6 && p[0] <= 1.0 + 1e-6);
        assert!(p[1] >= -1e-6 && p[1] <= 1.0 + 1e-6);
        assert!(p[2].abs() < 1e-6);
    }
}

#[test]
fn boundary_edge_only_rounds_corners() {
    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];
    let positions: [P3; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeOnly);
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(1));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.interpolate(1, &positions, &mut dst);

    // With edge-only boundaries a corner vertex is governed by the crease
    // rule along its two boundary edges: v' = 3/4 v + 1/8 (a + b).
    // For corner (0,0): 1/8 * ((1,0) + (0,1)) = (0.125, 0.125).
    assert_close(dst[5], [0.125, 0.125, 0.0]);
}

#[test]
fn limit_has_linear_precision_on_regular_grid() {
    // A 4x4 grid of vertices (3x3 quads). The central vertices are regular
    // (valence 4): Catmull-Clark limit evaluation reproduces linear data
    // exactly, so the interior limit points must equal their positions.
    let mut positions = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            positions.push([i as f32, j as f32, 0.25 * i as f32 - 0.5 * j as f32]);
        }
    }
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    for j in 0..3u32 {
        for i in 0..3u32 {
            verts_per_face.push(4);
            let v = j * 4 + i;
            face_verts.extend_from_slice(&[v, v + 1, v + 5, v + 4]);
        }
    }
    let descriptor = TopologyDescriptor::new(16, &verts_per_face, &face_verts);
    let refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();

    let primvar = PrimvarRefiner::new(&refiner);
    let mut limits = vec![[0.0f32; 3]; 16];
    primvar.limit(&positions, &mut limits);

    // Interior vertices: 5, 6, 9, 10.
    for v in [5usize, 6, 9, 10] {
        assert_close(limits[v], positions[v]);
    }
}

#[test]
fn holes_propagate_to_children() {
    let holes = [0u32];
    let descriptor =
        TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS).with_holes(&holes);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));

    let level1 = refiner.level(1);
    let holes: Vec<usize> = (0..level1.num_faces())
        .filter(|&f| level1.is_face_hole(f))
        .collect();
    // The hole quad subdivides into 4 hole children (the first 4 child faces).
    assert_eq!(holes, vec![0, 1, 2, 3]);
}

#[test]
fn interpolate_all_convenience() {
    let refiner = cube_refiner(sdc::Options::default(), 3);
    let primvar = PrimvarRefiner::new(&refiner);
    let levels = primvar.interpolate_all(&CUBE_POSITIONS);
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].len(), 26);
    assert_eq!(levels[1].len(), 98);
    assert_eq!(levels[2].len(), refiner.level(3).num_vertices());
}
