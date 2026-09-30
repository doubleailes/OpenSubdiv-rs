//! Patches around non-manifold features (GitHub issue #10): as in
//! OpenSubdiv, non-manifold edges are infinitely sharp and non-manifold
//! vertices sharp (or creased along their non-manifold edges), and the faces
//! around them are patched over their own manifold span — regular B-spline
//! patches where the span is regular under that rule, Gregory caps
//! elsewhere — rather than falling back to bilinear quads.

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchMap, PatchTableFactory, PatchType, PrimvarRefiner, TopologyDescriptor,
    TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

fn distance(a: P3, b: P3) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Limit positions of every base face's centre, from uniform refinement and
/// `PrimvarRefiner::limit` at the face-centre vertex.
fn face_centre_limits(
    descriptor: TopologyDescriptor<'_>,
    positions: &[P3],
    levels: usize,
) -> Vec<P3> {
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(levels));
    let primvar = PrimvarRefiner::new(&refiner);
    let last = primvar.interpolate_all(positions).pop().unwrap();
    let mut limits = last.clone();
    primvar.limit(&last, &mut limits);
    (0..refiner.level(0).num_faces())
        .map(|f| {
            let mut v = refiner.refinement(1).face_child_vertex(f);
            for l in 2..=levels {
                v = refiner.refinement(l).vertex_child_vertex(v as usize);
            }
            limits[v as usize]
        })
        .collect()
}

/// Adaptively patch the mesh, check no face falls back to bilinear quads,
/// and compare every base face's centre with the uniform limit.
fn check_against_uniform(
    descriptor: TopologyDescriptor<'_>,
    positions: &[P3],
    isolation: usize,
) -> opensubdiv_rs::far::PatchTable {
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(isolation));
    let table = PatchTableFactory::create(&refiner).unwrap();
    for p in 0..table.num_patches() {
        assert_ne!(table.patch_type(p), PatchType::Quads, "patch {p}");
    }

    let mut controls = positions.to_vec();
    for level in PrimvarRefiner::new(&refiner).interpolate_all(positions) {
        controls.extend(level);
    }
    let truth = face_centre_limits(descriptor, positions, 3);
    let map = PatchMap::new(&table);
    for (face, &expected) in truth.iter().enumerate() {
        let patch = map.find_patch(face, 0.5, 0.5).unwrap();
        let actual = table.evaluate(patch, 0.5, 0.5, &controls).0;
        let error = distance(actual, expected);
        assert!(
            error < 1e-3,
            "face {face} ({:?}): {actual:?} vs limit {expected:?}",
            table.patch_type(patch)
        );
    }
    table
}

#[test]
fn t_fin_faces_are_patched() {
    // Three quads sharing edge 0-1: the edge is non-manifold, so it is
    // infinitely sharp and its end vertices are sharp corners.
    let verts_per_face = [4usize; 3];
    let face_verts: [u32; 12] = [0, 1, 2, 3, 1, 0, 4, 5, 0, 1, 7, 6];
    let positions: [P3; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [1.1, 1.0, 0.2],
        [-0.1, 0.9, -0.1],
        [0.1, 0.0, 1.0],
        [0.9, -0.2, 1.1],
        [0.0, -0.8, -0.6],
        [1.2, -0.6, -0.8],
    ];
    let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);

    let refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    let base = refiner.level(0);
    let seam = (0..base.num_edges())
        .find(|&e| base.is_edge_non_manifold(e))
        .unwrap();
    assert!(base.edge_sharpness(seam) >= sdc::SHARPNESS_INFINITE);
    assert!(base.vertex_sharpness(0) >= sdc::SHARPNESS_INFINITE);
    assert!(base.vertex_sharpness(1) >= sdc::SHARPNESS_INFINITE);

    let table = check_against_uniform(descriptor, &positions, 3);
    assert!(
        (0..table.num_patches()).any(|p| table.patch_type(p) == PatchType::GregoryBasis),
        "the fins' smooth outer corners are irregular"
    );
}

#[test]
fn faces_along_a_non_manifold_seam_are_regular() {
    // Three 2x2-face fins sharing a seam of two edges (vertices 0-1-2).
    // The middle seam vertex has exactly two non-manifold edges: it is a
    // regular crease vertex of each fin, and the seam ends are sharp
    // corners with single-face spans, so every face touching the seam is
    // an exact B-spline patch at the base level.
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    let mut positions: Vec<P3> = vec![[0.0, 0.0, 0.0], [1.0, 0.05, -0.05], [2.0, 0.0, 0.1]];
    for k in 0..3u32 {
        let angle = std::f32::consts::TAU * k as f32 / 3.0 + 0.3 * k as f32;
        let (s, c) = angle.sin_cos();
        let base = 3 + 6 * k;
        for j in 1..3 {
            for i in 0..3 {
                let r = j as f32 + 0.1 * ((i + 2 * j + k as usize) % 3) as f32;
                positions.push([i as f32 + 0.07 * j as f32, c * r, s * r]);
            }
        }
        let row = |j: u32, i: u32| if j == 0 { i } else { base + 3 * (j - 1) + i };
        for j in 0..2 {
            for i in 0..2 {
                verts_per_face.push(4usize);
                face_verts.extend_from_slice(&[
                    row(j, i),
                    row(j, i + 1),
                    row(j + 1, i + 1),
                    row(j + 1, i),
                ]);
            }
        }
    }
    let descriptor = TopologyDescriptor::new(positions.len(), &verts_per_face, &face_verts);

    let table = check_against_uniform(descriptor, &positions, 3);
    let seam_faces: Vec<usize> = (0..verts_per_face.len())
        .filter(|&f| face_verts[4 * f..4 * f + 4].iter().any(|&v| v < 3))
        .collect();
    assert_eq!(seam_faces.len(), 6);
    for p in 0..table.num_patches() {
        let face = table.patch_face(p) as usize;
        if table.patch_param(p).depth == 0 && seam_faces.contains(&face) {
            assert_eq!(table.patch_type(p), PatchType::Regular, "face {face}");
        }
    }
    let base_regular = (0..table.num_patches())
        .filter(|&p| {
            table.patch_param(p).depth == 0 && seam_faces.contains(&(table.patch_face(p) as usize))
        })
        .count();
    assert_eq!(base_regular, 6);
}
