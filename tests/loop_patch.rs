//! Loop-scheme patch tests (GitHub issue #12): quartic box-spline patches
//! on regular triangles, Gregory triangle end caps on irregular ones,
//! triangular `PatchParam` / `PatchMap` handling, and feature-adaptive
//! refinement for Loop. The central oracle is the crate's own uniform Loop
//! refinement and limit masks: patches evaluated at refined vertices must
//! reproduce their limit positions, and adaptive tables must agree with
//! uniform ones everywhere.

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchMap, PatchTable, PatchTableFactory, PatchType, PrimvarRefiner,
    TopologyDescriptor, TopologyRefiner, TopologyRefinerFactory, UniformOptions,
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

fn count(table: &PatchTable, t: PatchType) -> usize {
    (0..table.num_patches())
        .filter(|&p| table.patch_type(p) == t)
        .count()
}

/// Deterministic pseudo-random samples in the unit triangle
/// (`u, v >= 0`, `u + v <= 1`).
fn tri_samples(count: usize) -> Vec<(f32, f32)> {
    let mut seed = 0x2468aceu32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..count)
        .map(|_| {
            let (a, b) = (rand(), rand());
            if a + b > 1.0 {
                (1.0 - a, 1.0 - b)
            } else {
                (a, b)
            }
        })
        .collect()
}

/// A triangulated `w x h`-vertex grid: each quad is split along its
/// `(i, j) - (i + 1, j + 1)` diagonal, so interior vertices have valence 6
/// and boundary vertices valence 4 (three faces), except the corners:
/// two of valence 3 (two faces) and two of valence 2 (one face).
fn tri_grid(w: u32, h: u32) -> (Vec<usize>, Vec<u32>) {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let v = j * w + i;
            verts_per_face.push(3usize);
            face_verts.extend_from_slice(&[v, v + 1, v + w + 1]);
            verts_per_face.push(3usize);
            face_verts.extend_from_slice(&[v, v + w + 1, v + w]);
        }
    }
    (verts_per_face, face_verts)
}

fn grid_positions(w: u32, h: u32, f: impl Fn(f32, f32) -> P3) -> Vec<P3> {
    let mut positions = Vec::new();
    for j in 0..h {
        for i in 0..w {
            positions.push(f(i as f32, j as f32));
        }
    }
    positions
}

/// A jittered height field over the grid.
fn bumpy(x: f32, y: f32) -> P3 {
    [
        x + 0.13 * (3.0 * y).sin(),
        y + 0.11 * (2.0 * x).cos(),
        0.4 * (x * 0.9).sin() * (y * 1.3).cos() + 0.05 * x * y,
    ]
}

const ICOSA_VERTS_PER_FACE: [usize; 20] = [3; 20];
const ICOSA_FACE_VERTS: [u32; 60] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1,
    8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
];

fn icosahedron_positions() -> Vec<P3> {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let raw: [P3; 12] = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    raw.iter()
        .map(|p| {
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / n, p[1] / n, p[2] / n]
        })
        .collect()
}

fn loop_refiner(descriptor: TopologyDescriptor<'_>, options: sdc::Options) -> TopologyRefiner {
    TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, options).unwrap()
}

fn edge_and_corner() -> sdc::Options {
    sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
}

/// Every patch of `table` — box-spline and Gregory triangle alike —
/// evaluated at its three corners must reproduce the limit positions of the
/// corresponding vertices of the refiner's last level (uniform refiners
/// only).
fn assert_patch_corners_match_limits(refiner: &TopologyRefiner, table: &PatchTable, base: &[P3]) {
    let controls = patch_controls(refiner, base);
    let last = last_level_positions(refiner, base);
    let mut limits = last.clone();
    PrimvarRefiner::new(refiner).limit(&last, &mut limits);
    let level = refiner.level(refiner.max_level());
    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        let fv = level.face_vertices(table.patch_face(p) as usize);
        for (corner, &(s, t)) in [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 1.0)]
            .iter()
            .enumerate()
        {
            let (u, v) = param.unnormalize(s, t);
            let (point, _, _) = table.evaluate(p, u, v, &controls);
            assert_close(point, limits[fv[corner] as usize], 1e-5);
        }
    }
}

#[test]
fn ptex_indices_map_one_face_per_triangle() {
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let mut refiner = loop_refiner(descriptor, sdc::Options::default());
    refiner.refine_uniform(UniformOptions::new(1));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let ptex = table.ptex_indices();
    assert_eq!(ptex.num_faces(), 20);
    for f in 0..20 {
        assert_eq!(ptex.face_id(f), f as u32);
        assert_eq!(ptex.face_ptex_count(f), 1);
        assert_eq!(ptex.base_face(f), f as u32);
        assert_eq!(ptex.base_face_corner(f), 0);
    }
    // No refinement is required for patches on a triangular base mesh.
    let refiner = loop_refiner(descriptor, sdc::Options::default());
    assert!(PatchTableFactory::create(&refiner).is_ok());
}

#[test]
fn regular_grid_patches_have_linear_precision() {
    // Linear data over a triangulated grid: the Loop limit surface is the
    // plane itself, so every box-spline patch — interior, boundary and
    // pinned corner — must evaluate to the exact plane with exact
    // constant derivatives; the two valence-3 corners are Gregory caps.
    let linear = |x: f32, y: f32| [x, y, 0.25 * x - 0.5 * y];
    let (w, h) = (6u32, 5u32);
    let (verts_per_face, face_verts) = tri_grid(w, h);
    let positions = grid_positions(w, h, linear);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let refiner = loop_refiner(descriptor, edge_and_corner());
    let table = PatchTableFactory::create(&refiner).unwrap();

    let num_faces = 2 * ((w - 1) * (h - 1)) as usize;
    assert_eq!(table.num_patches(), num_faces);
    // Two faces touch each valence-3 corner: (0, 0) has faces (0,1,w+1)
    // and (0,w+1,w); the opposite corner likewise.
    assert_eq!(count(&table, PatchType::GregoryTriangle), 4);
    assert_eq!(count(&table, PatchType::Loop), num_faces - 4);
    assert_eq!(count(&table, PatchType::Triangles), 0);

    let map = PatchMap::new(&table);
    let level = refiner.level(0);
    for ptex in 0..num_faces {
        if table.patch_type(ptex) != PatchType::Loop {
            continue;
        }
        let fv = level.face_vertices(ptex);
        let corner = |k: usize| positions[fv[k] as usize];
        for &(u, v) in &tri_samples(10) {
            let patch = map.find_patch(ptex, u, v).unwrap();
            assert_eq!(patch, ptex);
            let (point, du, dv) = table.evaluate(patch, u, v, &positions);
            let mut expected = [0.0f32; 3];
            let mut edu = [0.0f32; 3];
            let mut edv = [0.0f32; 3];
            for c in 0..3 {
                expected[c] = (1.0 - u - v) * corner(0)[c] + u * corner(1)[c] + v * corner(2)[c];
                edu[c] = corner(1)[c] - corner(0)[c];
                edv[c] = corner(2)[c] - corner(0)[c];
            }
            assert_close(point, expected, 1e-4);
            assert_close(du, edu, 1e-3);
            assert_close(dv, edv, 1e-3);
        }
    }
}

#[test]
fn box_spline_patches_reproduce_the_limit_surface() {
    // Refine a bumpy grid once: the level-1 patches, evaluated at the
    // midpoints of their edges, must reproduce the limit positions of the
    // level-2 vertices that sit there — across interior, boundary and
    // pinned-corner patches — and at their corners those of the level-1
    // vertices. This pins the box-spline basis and the phantom folding
    // against the independent subdivision and limit masks.
    let (w, h) = (6u32, 5u32);
    let (verts_per_face, face_verts) = tri_grid(w, h);
    let positions = grid_positions(w, h, bumpy);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);

    let mut refiner = loop_refiner(descriptor, edge_and_corner());
    refiner.refine_uniform(UniformOptions::new(1));
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_patch_corners_match_limits(&refiner, &table, &positions);

    let mut deeper = loop_refiner(descriptor, edge_and_corner());
    deeper.refine_uniform(UniformOptions::new(2));
    let level2 = last_level_positions(&deeper, &positions);
    let mut limits2 = level2.clone();
    PrimvarRefiner::new(&deeper).limit(&level2, &mut limits2);

    let controls = patch_controls(&refiner, &positions);
    let level1 = refiner.level(1);
    let refinement = deeper.refinement(2);
    let mut checked = 0;
    for p in 0..table.num_patches() {
        if table.patch_type(p) != PatchType::Loop {
            continue;
        }
        let face = table.patch_face(p) as usize;
        let param = table.patch_param(p);
        let fe = level1.face_edges(face);
        for (edge, &(s, t)) in [(0.5f32, 0.0f32), (0.5, 0.5), (0.0, 0.5)]
            .iter()
            .enumerate()
        {
            let child = refinement.edge_child_vertex(fe[edge] as usize) as usize;
            let (u, v) = param.unnormalize(s, t);
            let (point, _, _) = table.evaluate(p, u, v, &controls);
            assert_close(point, limits2[child], 1e-5);
            checked += 1;
        }
    }
    assert!(checked > 100);
}

#[test]
fn icosahedron_is_capped_with_gregory_triangles() {
    // Every vertex of an icosahedron has valence 5, so every face is
    // irregular at every level: the caps must interpolate the limit
    // points exactly and stay near the limit surface — which, for so
    // coarse a cage, lies well inside the unit sphere: the limit point of
    // a valence-5 vertex is `(omega v + sum e) / (omega + 5)` with
    // `omega = 4.46`, at radius 0.708.
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let positions = icosahedron_positions();
    let mut refiner = loop_refiner(descriptor, sdc::Options::default());
    refiner.refine_uniform(UniformOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();

    assert_eq!(table.num_patches(), 20 * 16);
    // At level 2 each base face has 16 children, 3 of which touch an
    // original (valence-5) vertex.
    assert_eq!(count(&table, PatchType::GregoryTriangle), 60);
    assert_eq!(count(&table, PatchType::Loop), 260);
    assert_patch_corners_match_limits(&refiner, &table, &positions);

    let controls = patch_controls(&refiner, &positions);
    let map = PatchMap::new(&table);
    for ptex in 0..20 {
        for &(u, v) in &tri_samples(8) {
            let patch = map.find_patch(ptex, u, v).unwrap();
            assert_eq!(table.patch_param(patch).ptex_face as usize, ptex);
            let (point, _, _) = table.evaluate(patch, u, v, &controls);
            let r = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            assert!(r > 0.69 && r < 0.72, "radius {r}");
        }
    }
}

#[test]
fn patches_are_c0_across_triangle_boundaries() {
    // Evaluate on both sides of the internal patch boundaries of each ptex
    // face — the three lines splitting a triangle into its four children,
    // including Gregory/box-spline junctions around the corner vertices —
    // at the exact same location: the surface must be continuous.
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let positions = icosahedron_positions();
    let mut refiner = loop_refiner(descriptor, sdc::Options::default());
    refiner.refine_uniform(UniformOptions::new(1));
    let controls = patch_controls(&refiner, &positions);
    let table = PatchTableFactory::create(&refiner).unwrap();
    let map = PatchMap::new(&table);

    let eps = 1e-3f32;
    let along = [0.05f32, 0.2, 0.35, 0.45];
    for ptex in 0..20usize {
        for &x in &along {
            // u = 0.5 (between children 0/3 and 1), v = 0.5, u + v = 0.5.
            let pairs = [
                ((0.5 - eps, x), (0.5 + eps, x), (0.5, x)),
                ((x, 0.5 - eps), (x, 0.5 + eps), (x, 0.5)),
                (
                    (x - eps, 0.5 - x - eps),
                    (x + eps, 0.5 - x + eps),
                    (x, 0.5 - x),
                ),
            ];
            for &((ua, va), (ub, vb), (u, v)) in &pairs {
                let a = map.find_patch(ptex, ua, va).unwrap();
                let b = map.find_patch(ptex, ub, vb).unwrap();
                assert_ne!(a, b);
                let (pa, _, _) = table.evaluate(a, u, v, &controls);
                let (pb, _, _) = table.evaluate(b, u, v, &controls);
                assert_close(pa, pb, 1e-4);
            }
        }
    }
}

#[test]
fn patch_param_and_patch_map_are_consistent_for_triangles() {
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let mut refiner = loop_refiner(descriptor, sdc::Options::default());
    refiner.refine_uniform(UniformOptions::new(3));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let map = PatchMap::new(&table);

    let mut rotated = 0;
    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        assert_eq!(param.depth, 3);
        assert!(param.rotation == 0 || param.rotation == 2);
        rotated += param.is_triangle_rotated() as usize;
        // A point strictly inside the patch maps back to the same patch.
        let (u, v) = param.unnormalize(0.3, 0.2);
        assert!(u >= 0.0 && v >= 0.0 && u + v <= 1.0);
        assert_eq!(map.find_patch(param.ptex_face as usize, u, v), Some(p));
        let (s, t) = param.normalize(u, v);
        assert!((s - 0.3).abs() < 1e-5 && (t - 0.2).abs() < 1e-5);

        // Basis invariants: point weights sum to 1, derivatives to 0.
        let basis = table.evaluate_basis(p, u, v);
        let sum: f32 = basis.weights.iter().sum();
        let dsum: f32 = basis.du_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(dsum.abs() < 1e-2);
    }
    // Of the 64 patches of a ptex face at depth 3, 1 + 3 + 9 = 13 have an
    // odd number of central ancestors and are inverted... in fact 28:
    // each level inverts a quarter of the upright and three quarters of
    // the inverted sub-triangles.
    assert_eq!(rotated, 20 * 28);
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
        for &(u, v) in &tri_samples(12) {
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

/// The triangulated grid with the diagonal of one interior quad flipped:
/// two valence-5 and two valence-7 vertices in an otherwise regular grid.
fn grid_with_flipped_edge(w: u32, h: u32) -> (Vec<usize>, Vec<u32>) {
    let (verts_per_face, mut face_verts) = tri_grid(w, h);
    let (i, j) = (2u32, 2u32);
    let quad = ((j * (w - 1) + i) * 2) as usize;
    let v = j * w + i;
    // (v, v+1, v+w+1), (v, v+w+1, v+w)  ->  (v, v+1, v+w), (v+1, v+w+1, v+w)
    face_verts[3 * quad..3 * quad + 6].copy_from_slice(&[v, v + 1, v + w, v + 1, v + w + 1, v + w]);
    (verts_per_face, face_verts)
}

#[test]
fn adaptive_grid_with_extraordinary_vertices_matches_uniform() {
    let (w, h) = (8u32, 7u32);
    let (verts_per_face, face_verts) = grid_with_flipped_edge(w, h);
    let positions = grid_positions(w, h, bumpy);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let level0 = loop_refiner(descriptor, edge_and_corner());
    let valences: Vec<usize> = (0..(w * h) as usize)
        .map(|v| level0.level(0).vertex_edges(v).len())
        .collect();
    assert_eq!(valences.iter().filter(|&&n| n == 7).count(), 2);
    assert_eq!(valences.iter().filter(|&&n| n == 5).count(), 2);

    let mut adaptive = loop_refiner(descriptor, edge_and_corner());
    adaptive.refine_adaptive(AdaptiveOptions::new(3));
    assert!(adaptive.is_adaptive());
    assert_eq!(adaptive.max_level(), 3);
    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();

    let mut uniform = loop_refiner(descriptor, edge_and_corner());
    uniform.refine_uniform(UniformOptions::new(3));
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();

    // Regular faces are box-spline patches at depth 0; only the faces
    // around the four extraordinary vertices and the two valence-3 grid
    // corners descend, capped with Gregory triangles at the isolation
    // level.
    let num_faces = 2 * ((w - 1) * (h - 1)) as usize;
    assert_eq!(uniform_table.num_patches(), num_faces * 64);
    assert!(adaptive_table.num_patches() < uniform_table.num_patches() / 4);
    assert!(adaptive.level(3).num_faces() < uniform.level(3).num_faces());
    assert!(adaptive.num_vertices_total() < uniform.num_vertices_total());
    let mut depth0_loop = 0;
    for p in 0..adaptive_table.num_patches() {
        let param = adaptive_table.patch_param(p);
        match adaptive_table.patch_type(p) {
            PatchType::Loop => depth0_loop += (param.depth == 0) as usize,
            PatchType::GregoryTriangle => assert_eq!(param.depth, 3),
            other => panic!("unexpected {other:?} patch"),
        }
    }
    assert!(depth0_loop > num_faces / 2);
    assert_eq!(
        count(&adaptive_table, PatchType::GregoryTriangle),
        6 * 4 + 2 * 2
    );

    // Same limit surface everywhere.
    assert_tables_agree(
        &adaptive_table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        num_faces,
    );
}

#[test]
fn adaptive_icosahedron_is_sparse_and_matches_uniform() {
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let positions = icosahedron_positions();

    let mut adaptive = loop_refiner(descriptor, sdc::Options::default());
    adaptive.refine_adaptive(AdaptiveOptions::new(3));
    let mut uniform = loop_refiner(descriptor, sdc::Options::default());
    uniform.refine_uniform(UniformOptions::new(3));

    // Every face touches a valence-5 vertex, so level 1 refines fully and
    // every level-1 face still touches one; at level 2 the 200 children
    // away from the vertices are regular and only the 60 corner children
    // descend, yielding 180 regular and 60 capped faces at the isolation
    // level: 440 patches against 1280 uniform ones.
    assert_eq!(adaptive.max_level(), 3);
    assert!(adaptive.level(3).num_faces() < uniform.level(3).num_faces());
    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert_eq!(uniform_table.num_patches(), 20 * 64);
    assert_eq!(adaptive_table.num_patches(), 440);
    assert_eq!(count(&adaptive_table, PatchType::GregoryTriangle), 60);
    assert_eq!(count(&adaptive_table, PatchType::Loop), 380);

    assert_tables_agree(
        &adaptive_table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        20,
    );
}

#[test]
fn gregory_triangles_are_exact_at_evs_and_consistent_across_levels() {
    let descriptor = TopologyDescriptor::new(12, &ICOSA_VERTS_PER_FACE, &ICOSA_FACE_VERTS);
    let positions = icosahedron_positions();
    let evaluate_at = |levels: usize, u: f32, v: f32| -> P3 {
        let mut refiner = loop_refiner(descriptor, sdc::Options::default());
        refiner.refine_uniform(UniformOptions::new(levels));
        let controls = patch_controls(&refiner, &positions);
        let table = PatchTableFactory::create(&refiner).unwrap();
        let map = PatchMap::new(&table);
        // Ptex face 0 is base face 0 (verts 0, 11, 5): its (0, 0) corner
        // is vertex 0.
        let patch = map.find_patch(0, u, v).unwrap();
        let (point, _, _) = table.evaluate(patch, u, v, &controls);
        point
    };

    // True limit of vertex 0, from the limit stencils.
    let refiner = loop_refiner(descriptor, sdc::Options::default());
    let mut limits = positions.clone();
    PrimvarRefiner::new(&refiner).limit(&positions, &mut limits);
    let truth = limits[0];

    assert_close(evaluate_at(1, 0.0, 0.0), truth, 1e-5);
    assert_close(evaluate_at(3, 0.0, 0.0), truth, 1e-5);

    // Near (but not at) the extraordinary vertex, the caps built at
    // different levels approximate the same limit surface.
    let p1 = evaluate_at(1, 0.03, 0.05);
    let p3 = evaluate_at(3, 0.03, 0.05);
    assert_close(p1, p3, 1e-2);
}

#[test]
fn smooth_boundaries_fall_back_to_linear_triangles() {
    // With `VtxBoundaryInterpolation::None` the boundary follows the
    // smooth rules, which no patch reproduces: boundary faces are linear
    // triangles, interior faces box-spline patches.
    let (w, h) = (5u32, 5u32);
    let (verts_per_face, face_verts) = tri_grid(w, h);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::None);
    let mut refiner = loop_refiner(descriptor, options);
    refiner.refine_uniform(UniformOptions::new(1));
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert!(count(&table, PatchType::Triangles) > 0);
    assert!(count(&table, PatchType::Loop) > 0);
    assert_eq!(count(&table, PatchType::GregoryTriangle), 0);
}

#[test]
fn infinitely_sharp_creases_split_loop_patches() {
    // An infinitely sharp crease along a grid row is treated as a
    // boundary: the faces on either side are exact box-spline patches with
    // the crease's limit curve (the cubic B-spline of the crease vertices)
    // as their edge, and adaptive refinement does not isolate it. Only the
    // crease's ends — boundary vertices with three sharp edges, whose
    // two-face side is irregular — and the valence-3 grid corners isolate.
    let (w, h) = (7u32, 7u32);
    let (verts_per_face, face_verts) = tri_grid(w, h);
    let positions = grid_positions(w, h, bumpy);
    let creases: Vec<[u32; 2]> = (0..w - 1).map(|i| [3 * w + i, 3 * w + i + 1]).collect();
    let weights = vec![sdc::SHARPNESS_INFINITE; creases.len()];
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts)
        .with_creases(&creases, &weights);

    let mut adaptive = loop_refiner(descriptor, edge_and_corner());
    adaptive.refine_adaptive(AdaptiveOptions::new(4));
    let table = PatchTableFactory::create(&adaptive).unwrap();
    let level0 = adaptive.level(0);
    let crease_interior: Vec<u32> = (1..w - 1).map(|i| 3 * w + i).collect();
    let mut on_crease = 0;
    for p in 0..table.num_patches() {
        let param = table.patch_param(p);
        if param.depth == 0 {
            let fv = level0.face_vertices(table.patch_face(p) as usize);
            if fv.iter().any(|v| crease_interior.contains(v)) {
                on_crease += 1;
                assert_eq!(table.patch_type(p), PatchType::Loop);
            }
        }
    }
    // Every face touching the interior of the crease is a depth-0 patch:
    // 6 per interior crease vertex, minus the shared ones.
    assert_eq!(
        on_crease,
        2 * (w as usize - 2) + 2 * (w as usize - 1) - 2 - 2 + 2
    );
    assert_eq!(count(&table, PatchType::GregoryTriangle), 2 * 2 + 2 * 2);

    // The crease-adjacent patches reproduce the limit of the crease
    // vertices and of the refined vertices around them, and the adaptive
    // surface agrees with the uniform one at the same isolation depth.
    let mut uniform = loop_refiner(descriptor, edge_and_corner());
    uniform.refine_uniform(UniformOptions::new(1));
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert_patch_corners_match_limits(&uniform, &uniform_table, &positions);
    let mut uniform = loop_refiner(descriptor, edge_and_corner());
    uniform.refine_uniform(UniformOptions::new(4));
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert_tables_agree(
        &table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        2 * ((w - 1) * (h - 1)) as usize,
    );
}

#[test]
fn semi_sharp_creases_are_isolated_until_they_decay() {
    let (w, h) = (7u32, 7u32);
    let (verts_per_face, face_verts) = tri_grid(w, h);
    let positions = grid_positions(w, h, bumpy);
    let creases = [[3 * w + 2, 3 * w + 3], [3 * w + 3, 3 * w + 4]];
    let weights = [1.5f32, 1.5];
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts)
        .with_creases(&creases, &weights);

    let mut adaptive = loop_refiner(descriptor, edge_and_corner());
    adaptive.refine_adaptive(AdaptiveOptions::new(5));
    // Sharpness 1.5 decays to 0 after two subdivisions; the grid corners
    // of valence 3 keep isolating to the cap, though.
    assert_eq!(adaptive.max_level(), 5);
    let adaptive_table = PatchTableFactory::create(&adaptive).unwrap();

    let mut uniform = loop_refiner(descriptor, edge_and_corner());
    uniform.refine_uniform(UniformOptions::new(5));
    let uniform_table = PatchTableFactory::create(&uniform).unwrap();
    assert!(adaptive_table.num_patches() < uniform_table.num_patches() / 10);
    assert_tables_agree(
        &adaptive_table,
        &patch_controls(&adaptive, &positions),
        &uniform_table,
        &patch_controls(&uniform, &positions),
        2 * ((w - 1) * (h - 1)) as usize,
    );
}
