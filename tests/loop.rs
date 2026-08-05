//! Loop-scheme refinement tests, validated against the classic Loop
//! subdivision rules.

use opensubdiv_rs::far::{
    Error, PrimvarRefiner, TopologyDescriptor, TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

/// A regular tetrahedron: 4 vertices of valence 3, 4 triangles, 6 edges.
const TET_VERTS_PER_FACE: [usize; 4] = [3; 4];
const TET_FACE_VERTS: [u32; 12] = [0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];
const TET_POSITIONS: [P3; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
];

fn assert_close(actual: P3, expected: P3) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!(
            (a - e).abs() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn loop_requires_triangles() {
    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let result =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default());
    assert!(matches!(
        result,
        Err(Error::NonTriangularFaceForLoop { face: 0, size: 4 })
    ));
}

#[test]
fn tetrahedron_refinement_counts() {
    let descriptor = TopologyDescriptor::new(4, &TET_VERTS_PER_FACE, &TET_FACE_VERTS);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    // Level 1: 6 edge children + 4 vertex children; 16 tris; 24 edges.
    let level1 = refiner.level(1);
    assert_eq!(level1.num_vertices(), 10);
    assert_eq!(level1.num_faces(), 16);
    assert_eq!(level1.num_edges(), 24);

    // Level 2: 24 + 10 = 34 vertices; 64 tris; 96 edges.
    let level2 = refiner.level(2);
    assert_eq!(level2.num_vertices(), 34);
    assert_eq!(level2.num_faces(), 64);
    assert_eq!(level2.num_edges(), 96);

    for l in 0..=2 {
        let level = refiner.level(l);
        assert_eq!(
            level.num_vertices() as i64 - level.num_edges() as i64 + level.num_faces() as i64,
            2
        );
    }
}

#[test]
fn tetrahedron_level1_positions_match_loop_rules() {
    let descriptor = TopologyDescriptor::new(4, &TET_VERTS_PER_FACE, &TET_FACE_VERTS);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.interpolate(1, &TET_POSITIONS, &mut dst);

    // Edge children are ordered first (0..6), then vertex children (6..10).

    // Edge (0,1): e' = 3/8 (v0 + v1) + 1/8 (opposite vertices).
    // Its two adjacent triangles are (0,1,2) and (0,3,1): opposites 2 and 3.
    let e = refiner.level(0).find_edge(0, 1).unwrap() as usize;
    let expected = {
        let mut p = [0.0f32; 3];
        for k in 0..3 {
            p[k] = 0.375 * (TET_POSITIONS[0][k] + TET_POSITIONS[1][k])
                + 0.125 * (TET_POSITIONS[2][k] + TET_POSITIONS[3][k]);
        }
        p
    };
    assert_close(dst[e], expected);

    // Vertex child of v0 (valence 3): beta = 3/16, so
    // v' = (1 - 3*beta) v + beta * sum(neighbors)
    //    = 0.4375 (1,1,1) + 0.1875 (-1,-1,-1) = (0.25, 0.25, 0.25).
    assert_close(dst[6], [0.25, 0.25, 0.25]);
}

#[test]
fn loop_limit_has_linear_precision_on_regular_vertex() {
    // A hexagonal fan around a central regular (valence 6) vertex, with
    // linear ("planar ramp") data: the limit of the center must reproduce
    // its position exactly.
    let mut positions: Vec<P3> = vec![[0.0, 0.0, 0.0]];
    for i in 0..6 {
        let a = std::f32::consts::TAU * i as f32 / 6.0;
        positions.push([a.cos(), a.sin(), 0.0]);
    }
    for p in positions.iter_mut() {
        p[2] = 0.3 * p[0] - 0.7 * p[1]; // linear function of (x, y)
    }
    let verts_per_face = [3usize; 6];
    let mut face_verts: Vec<u32> = Vec::new();
    for i in 0..6u32 {
        face_verts.extend_from_slice(&[0, 1 + i, 1 + (i + 1) % 6]);
    }
    let descriptor = TopologyDescriptor::new(7, &verts_per_face, &face_verts);
    let refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();

    let primvar = PrimvarRefiner::new(&refiner);
    let mut limits = vec![[0.0f32; 3]; 7];
    primvar.limit(&positions, &mut limits);
    assert_close(limits[0], positions[0]);
}

#[test]
fn bilinear_refinement_is_linear() {
    // Bilinear: face points are centroids, edge points are midpoints, and
    // vertex points are interpolated (identity).
    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];
    let positions: [P3; 4] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let descriptor = TopologyDescriptor::new(4, &verts_per_face, &face_verts);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Bilinear,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(1));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.interpolate(1, &positions, &mut dst);

    assert_close(dst[0], [1.0, 1.0, 0.0]); // face centroid
    let e = refiner.level(0).find_edge(0, 1).unwrap() as usize;
    assert_close(dst[1 + e], [1.0, 0.0, 0.0]); // edge midpoint
    for (i, &p) in positions.iter().enumerate() {
        assert_close(dst[5 + i], p); // vertices unchanged
    }

    // Bilinear limit is the mesh itself.
    let mut limits = vec![[0.0f32; 3]; refiner.level(1).num_vertices()];
    primvar.limit(&dst, &mut limits);
    for (l, d) in limits.iter().zip(&dst) {
        assert_close(*l, *d);
    }
}
