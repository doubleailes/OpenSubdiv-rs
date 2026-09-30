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

/// Limit positions of every base face's centre and corners (in face-vertex
/// order), from uniform refinement and `PrimvarRefiner::limit` at the
/// face-centre vertex and at the base vertices' descendants.
fn face_limits(
    descriptor: TopologyDescriptor<'_>,
    positions: &[P3],
    levels: usize,
) -> Vec<(P3, [P3; 4])> {
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
    let descend = |mut v: u32, from: usize| {
        for l in from..=levels {
            v = refiner.refinement(l).vertex_child_vertex(v as usize);
        }
        limits[v as usize]
    };
    (0..refiner.level(0).num_faces())
        .map(|f| {
            let centre = descend(refiner.refinement(1).face_child_vertex(f), 2);
            let fv = refiner.level(0).face_vertices(f);
            (centre, [0, 1, 2, 3].map(|k| descend(fv[k], 1)))
        })
        .collect()
}

/// Adaptively patch the mesh, check no face falls back to bilinear quads,
/// and compare every base face's centre and corners with the uniform limit.
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
    let map = PatchMap::new(&table);
    let check = |face: usize, u: f32, v: f32, expected: P3, tol: f32| {
        let patch = map.find_patch(face, u, v).unwrap();
        let actual = table.evaluate(patch, u, v, &controls).0;
        let error = distance(actual, expected);
        assert!(
            error < tol,
            "face {face} at ({u}, {v}) ({:?}): {actual:?} vs limit {expected:?}",
            table.patch_type(patch)
        );
    };
    for (face, (centre, corners)) in face_limits(descriptor, positions, 3)
        .into_iter()
        .enumerate()
    {
        check(face, 0.5, 0.5, centre, 1e-3);
        // Patches interpolate the limit points of their corners exactly.
        for (&(u, v), &corner) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .zip(&corners)
        {
            check(face, u, v, corner, 1e-4);
        }
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

#[test]
fn closed_fan_touching_a_seam_vertex_is_pinned() {
    // The three-fin seam of the previous test, plus a closed fan — the
    // corner of a cube — touching only the seam's middle vertex. That
    // vertex has exactly two non-manifold edges, but they do not bound the
    // cube's fan, so the crease rule along the seam would not describe the
    // cube's limit there: the vertex must be pinned, and the seam patches
    // and the cube's caps must agree with the refined limit.
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
    // A cube whose vertex 0 is the seam's middle vertex (1), extending
    // along x only (away from the fins, which lie in the y-z plane).
    let cube_offset = [1.3f32, 0.4, 0.3];
    let first = positions.len() as u32;
    let mut cube_index = [1u32; 8];
    for (k, index) in cube_index.iter_mut().enumerate().skip(1) {
        *index = first + k as u32 - 1;
        let bits = [(k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32];
        positions.push([
            positions[1][0] + 0.2 + bits[0] * cube_offset[0],
            positions[1][1] + 0.2 + bits[1] * cube_offset[1],
            positions[1][2] + 0.2 + bits[2] * cube_offset[2],
        ]);
    }
    let cube_faces: [[usize; 4]; 6] = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    for f in cube_faces {
        verts_per_face.push(4);
        face_verts.extend(f.map(|k| cube_index[k]));
    }
    let descriptor = TopologyDescriptor::new(positions.len(), &verts_per_face, &face_verts);

    let refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    assert!(refiner.level(0).vertex_sharpness(1) >= sdc::SHARPNESS_INFINITE);
    check_against_uniform(descriptor, &positions, 3);
}
