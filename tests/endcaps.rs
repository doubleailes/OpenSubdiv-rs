//! Gregory end caps for irregular faces on boundaries and infinitely sharp
//! creases (GitHub issue #8): the patch table must never fall back to
//! bilinear quads on manifold meshes, the caps must reproduce the boundary
//! limit curves and the limit points of crease and boundary vertices
//! exactly, join regular neighbors with exact C0 continuity, and approximate
//! the limit surface closely at the isolation level.

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

fn distance(a: P3, b: P3) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

fn count(table: &PatchTable, t: PatchType) -> usize {
    (0..table.num_patches())
        .filter(|&p| table.patch_type(p) == t)
        .count()
}

/// A refiner, its patch table and its control values.
struct Surface {
    refiner: TopologyRefiner,
    table: PatchTable,
    controls: Vec<P3>,
}

impl Surface {
    fn adaptive(
        descriptor: TopologyDescriptor<'_>,
        options: sdc::Options,
        isolation: usize,
        positions: &[P3],
    ) -> Surface {
        let mut refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
        refiner.refine_adaptive(AdaptiveOptions::new(isolation));
        let table = PatchTableFactory::create(&refiner).unwrap();
        let controls = patch_controls(&refiner, positions);
        Surface {
            refiner,
            table,
            controls,
        }
    }

    /// The same mesh uniformly refined `levels` deep: its regular patches
    /// are the exact limit surface, so evaluating it wherever a regular
    /// patch covers the location gives ground truth.
    fn uniform(
        descriptor: TopologyDescriptor<'_>,
        options: sdc::Options,
        levels: usize,
        positions: &[P3],
    ) -> Surface {
        let mut refiner =
            TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
        refiner.refine_uniform(UniformOptions::new(levels));
        let table = PatchTableFactory::create(&refiner).unwrap();
        let controls = patch_controls(&refiner, positions);
        Surface {
            refiner,
            table,
            controls,
        }
    }

    fn evaluate(&self, patch: usize, u: f32, v: f32) -> P3 {
        self.table.evaluate(patch, u, v, &self.controls).0
    }

    /// The exact limit position at `(ptex, u, v)`, asserting that a regular
    /// (exact) patch covers it.
    fn exact_limit(&self, ptex: usize, u: f32, v: f32) -> P3 {
        let patch = PatchMap::new(&self.table).find_patch(ptex, u, v).unwrap();
        assert_eq!(
            self.table.patch_type(patch),
            PatchType::Regular,
            "ground truth must come from an exact patch"
        );
        self.evaluate(patch, u, v)
    }
}

/// Largest distances between the Gregory caps of `surface` and the exact
/// limit surface `truth`, sampled at each cap's face centre, at the
/// midpoints of its edges lying on the mesh boundary, and at the midpoints
/// of its edges shared with regular patches.
#[derive(Debug, Default)]
struct CapErrors {
    centre: f32,
    boundary_edge: f32,
    regular_edge: f32,
    boundary_edges: usize,
    regular_edges: usize,
}

fn cap_errors(surface: &Surface, truth: &Surface) -> CapErrors {
    let table = &surface.table;
    let map = PatchMap::new(table);
    let mut errors = CapErrors::default();
    for p in 0..table.num_patches() {
        if table.patch_type(p) != PatchType::GregoryBasis {
            continue;
        }
        let param = table.patch_param(p);
        let ptex = param.ptex_face as usize;
        let level = surface.refiner.level(param.depth as usize);
        let face = table.patch_face(p) as usize;

        let (u, v) = param.unnormalize(0.5, 0.5);
        let error = distance(surface.evaluate(p, u, v), truth.exact_limit(ptex, u, v));
        errors.centre = errors.centre.max(error);

        // Edge k runs from corner k to corner k+1 in the patch frame.
        let midpoints = [(0.5f32, 0.0f32), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)];
        let nudges = [(0.0f32, -1.0f32), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
        for k in 0..4 {
            let (s, t) = midpoints[k];
            let (u, v) = param.unnormalize(s, t);
            let edge = level.face_edges(face)[k] as usize;
            let error = distance(surface.evaluate(p, u, v), truth.exact_limit(ptex, u, v));
            if level.is_edge_boundary(edge) {
                errors.boundary_edge = errors.boundary_edge.max(error);
                errors.boundary_edges += 1;
                continue;
            }
            // Step across the edge to identify the neighboring patch.
            let eps = 1e-3;
            let (nu, nv) = param.unnormalize(s + eps * nudges[k].0, t + eps * nudges[k].1);
            if !(0.0..=1.0).contains(&nu) || !(0.0..=1.0).contains(&nv) {
                continue; // the neighbor lies in another ptex face
            }
            let neighbor = map.find_patch(ptex, nu, nv).unwrap();
            if neighbor != p && table.patch_type(neighbor) == PatchType::Regular {
                errors.regular_edge = errors.regular_edge.max(error);
                errors.regular_edges += 1;
            }
        }
    }
    errors
}

/// Basis invariants of every Gregory cap: point weights sum to one and
/// derivative weights to zero (affine invariance).
fn assert_caps_are_affine(surface: &Surface) {
    let table = &surface.table;
    for p in 0..table.num_patches() {
        if table.patch_type(p) != PatchType::GregoryBasis {
            continue;
        }
        let param = table.patch_param(p);
        for (s, t) in [(0.0f32, 0.0f32), (0.5, 0.5), (0.2, 0.9), (1.0, 0.3)] {
            let (u, v) = param.unnormalize(s, t);
            let basis = table.evaluate_basis(p, u, v);
            let sum: f32 = basis.weights.iter().sum();
            let du: f32 = basis.du_weights.iter().sum();
            let dv: f32 = basis.dv_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "patch {p}: weights sum to {sum}");
            assert!(du.abs() < 1e-2 && dv.abs() < 1e-2, "patch {p}: {du} {dv}");
        }
    }
}

/// An open fan of five quads around one vertex: the centre is an interior
/// extraordinary vertex (valence 5); the rim spokes are regular boundary
/// vertices (valence 3) and the rim corners boundary corners (valence 2).
///
/// ```text
///        b_i --- b_{i+1}
///        |          |
///       a_i ------ a_{i+1}
///          \      /
///           centre
/// ```
fn fan() -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let n = 5u32;
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for i in 0..n {
        verts_per_face.push(4usize);
        // (centre, a_i, b_i, a_{i+1}) — b_i lies between the two spokes.
        face_verts.extend_from_slice(&[0, 1 + i, 1 + n + i, 1 + (i + 1) % n]);
    }
    let mut positions = vec![[0.0, 0.0, 0.3]];
    let angle = |k: f32| std::f32::consts::TAU * k / n as f32;
    for i in 0..n {
        let a = angle(i as f32);
        positions.push([a.cos(), a.sin(), 0.0]);
    }
    for i in 0..n {
        let a = angle(i as f32 + 0.5);
        positions.push([1.7 * a.cos(), 1.7 * a.sin(), -0.2]);
    }
    (verts_per_face, face_verts, positions)
}

/// Three quads around a *boundary* vertex: the centre is an extraordinary
/// boundary vertex (three faces, four edges) whose caps need the irregular
/// boundary corner rules (`ComputeBoundaryPointWeights` with `k = 3`).
///
/// ```text
///     b0 ----- b1 ----- b2
///     |        |        |
///     a0 - a1 -c- a2 - a3       (a0-c and c-a3 are boundary edges)
/// ```
fn half_fan() -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let verts_per_face = vec![4usize; 3];
    // c = 0, a0..a3 = 1..4, b0..b2 = 5..7.
    let face_verts = vec![0u32, 1, 5, 2, 0, 2, 6, 3, 0, 3, 7, 4];
    let positions = vec![
        [0.0, 0.0, 0.25],
        [-1.0, 0.0, 0.0],
        [-0.5, 0.85, 0.05],
        [0.5, 0.85, -0.05],
        [1.0, 0.0, 0.0],
        [-1.6, 1.3, -0.1],
        [0.0, 1.8, 0.1],
        [1.6, 1.3, -0.1],
    ];
    (verts_per_face, face_verts, positions)
}

#[test]
fn boundary_extraordinary_fan_is_capped_with_gregory_patches() {
    // Issue #8, case 1. With the default (edge-only) boundary rule the rim
    // corners are smooth, unpinned corners — irregular boundary corners —
    // and the centre is an extraordinary vertex, so at the isolation cap
    // every irregular face must be a Gregory patch, never a bilinear quad.
    let (verts_per_face, face_verts, positions) = fan();
    let descriptor = TopologyDescriptor::new(11, &verts_per_face, &face_verts);
    let options = sdc::Options::default();

    let surface = Surface::adaptive(descriptor, options, 3, &positions);
    assert_eq!(surface.refiner.max_level(), 3);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    // Per level, the 5 faces at the centre and the 5 at the rim corners
    // stay irregular while their other children become regular:
    //   10 (L1) + 30 (L2) + 30 + 10 Gregory (L3) = 80 patches.
    assert_eq!(surface.table.num_patches(), 80);
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 10);
    assert_eq!(count(&surface.table, PatchType::Regular), 70);
    assert_caps_are_affine(&surface);

    // Value check against deep uniform refinement: the caps at the rim
    // corners interpolate the boundary limit curve exactly, every cap is
    // C0 with its regular neighbors, and the face centres are close.
    let truth = Surface::uniform(descriptor, options, 6, &positions);
    let errors = cap_errors(&surface, &truth);
    assert_eq!(errors.boundary_edges, 10);
    assert!(errors.regular_edges >= 20);
    assert!(errors.boundary_edge < 1e-5, "{errors:?}");
    assert!(errors.regular_edge < 1e-5, "{errors:?}");
    assert!(errors.centre < 1e-3, "{errors:?}");

    // The adaptive and uniform surfaces agree everywhere (the uniform table
    // has its own caps only in tiny level-6 faces).
    let map_a = PatchMap::new(&surface.table);
    let map_u = PatchMap::new(&truth.table);
    for ptex in 0..5 {
        for &(u, v) in &[(0.1f32, 0.7f32), (0.5, 0.5), (0.93, 0.02), (0.02, 0.05)] {
            let pa = map_a.find_patch(ptex, u, v).unwrap();
            let pu = map_u.find_patch(ptex, u, v).unwrap();
            assert_close(surface.evaluate(pa, u, v), truth.evaluate(pu, u, v), 5e-3);
        }
    }
}

#[test]
fn boundary_extraordinary_fan_with_pinned_corners() {
    // Pinned rim corners are regular corner patches: only the centre's
    // five faces remain irregular at the cap.
    let (verts_per_face, face_verts, positions) = fan();
    let descriptor = TopologyDescriptor::new(11, &verts_per_face, &face_verts);
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);

    let surface = Surface::adaptive(descriptor, options, 3, &positions);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 5);

    let truth = Surface::uniform(descriptor, options, 6, &positions);
    let errors = cap_errors(&surface, &truth);
    assert!(errors.regular_edges >= 10);
    assert!(errors.regular_edge < 1e-5, "{errors:?}");
    assert!(errors.centre < 1e-3, "{errors:?}");
}

#[test]
fn boundary_extraordinary_vertex_is_capped_with_gregory_patches() {
    // An extraordinary vertex *on* the boundary: its three caps are built
    // with the irregular boundary corner rules and interpolate the boundary
    // curve through the vertex exactly.
    let (verts_per_face, face_verts, positions) = half_fan();
    let descriptor = TopologyDescriptor::new(8, &verts_per_face, &face_verts);
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);

    let surface = Surface::adaptive(descriptor, options, 3, &positions);
    assert_eq!(surface.refiner.max_level(), 3);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 3);
    assert_caps_are_affine(&surface);

    let truth = Surface::uniform(descriptor, options, 6, &positions);
    let errors = cap_errors(&surface, &truth);
    assert_eq!(errors.boundary_edges, 2);
    assert!(errors.regular_edges >= 3);
    assert!(errors.boundary_edge < 1e-5, "{errors:?}");
    assert!(errors.regular_edge < 1e-5, "{errors:?}");
    assert!(errors.centre < 1e-3, "{errors:?}");

    // The cap corner at the boundary EV is the crease limit point of the
    // vertex: 2/3 c + 1/6 a0 + 1/6 a3 at the base level.
    let map = PatchMap::new(&surface.table);
    let patch = map.find_patch(0, 0.0, 0.0).unwrap();
    assert_eq!(surface.table.patch_type(patch), PatchType::GregoryBasis);
    let (c, a0, a3) = (positions[0], positions[1], positions[4]);
    let expected = [0, 1, 2].map(|i| 2.0 / 3.0 * c[i] + (a0[i] + a3[i]) / 6.0);
    assert_close(surface.evaluate(patch, 0.0, 0.0), expected, 1e-5);
}

#[test]
fn creased_cube_caps_are_gregory_along_the_crease() {
    // Issue #8, case 2: a cube with one infinitely sharp edge. Every cube
    // corner has valence 3, so all six faces are irregular; along the
    // crease the valence-4 edge vertices are regular crease corners, so the
    // faces beside the crease away from its ends are exact B-spline
    // patches, and exactly four faces incident to the crease — two at each
    // end, one per side — are Gregory caps built with the sharp-crease
    // rules (the cube corners at the crease's ends are darts).
    let creases = [[0u32, 1u32]];
    let weights = [sdc::SHARPNESS_INFINITE];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&creases, &weights);
    let options = sdc::Options::default();

    let surface = Surface::adaptive(descriptor, options, 3, &CUBE_POSITIONS);
    assert_eq!(surface.refiner.max_level(), 3);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    // Same patch structure as the smooth cube: 72 regular faces at levels 2
    // and 3 each, plus the 24 corner caps at level 3.
    assert_eq!(surface.table.num_patches(), 168);
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 24);
    assert_caps_are_affine(&surface);

    let level = surface.refiner.level(3);
    let crease_edge = |patch: usize| {
        level
            .face_edges(surface.table.patch_face(patch) as usize)
            .iter()
            .position(|&e| sdc::Crease::is_infinite(level.edge_sharpness(e as usize)))
    };
    let crease_caps: Vec<usize> = (0..surface.table.num_patches())
        .filter(|&p| {
            surface.table.patch_type(p) == PatchType::GregoryBasis && crease_edge(p).is_some()
        })
        .collect();
    assert_eq!(crease_caps.len(), 4);

    // Value check (issue #8, case 3): the caps are C0 with the regular
    // patches around them and match the limit surface closely at their
    // face centres.
    let truth = Surface::uniform(descriptor, options, 6, &CUBE_POSITIONS);
    let errors = cap_errors(&surface, &truth);
    assert!(errors.regular_edges >= 48);
    assert!(errors.regular_edge < 1e-5, "{errors:?}");
    assert!(errors.centre < 1e-3, "{errors:?}");

    // At the crease-end caps, the crease edge joins the cube corner — a
    // dart — to a crease vertex. The cap interpolates the crease vertex's
    // limit point (on the crease curve) exactly. At the dart, the cap's
    // corner is the smooth limit formula of its refined neighborhood (as
    // in OpenSubdiv, whose limit masks treat darts as smooth), which only
    // converges to the true limit with refinement, so compare loosely.
    let truth_map = PatchMap::new(&truth.table);
    for &p in &crease_caps {
        let param = surface.table.patch_param(p);
        let face = surface.table.patch_face(p) as usize;
        let k = crease_edge(p).unwrap();
        for corner in [k, (k + 1) % 4] {
            let (s, t) = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)][corner];
            let (u, v) = param.unnormalize(s, t);
            let point = surface.evaluate(p, u, v);
            let tp = truth_map
                .find_patch(param.ptex_face as usize, u, v)
                .unwrap();
            let vertex = level.face_vertices(face)[corner] as usize;
            let is_dart = level.vertex_edges(vertex).len() == 3;
            if is_dart {
                assert_eq!(truth.table.patch_type(tp), PatchType::GregoryBasis);
                assert_close(point, truth.evaluate(tp, u, v), 5e-3);
            } else {
                assert_eq!(truth.table.patch_type(tp), PatchType::Regular);
                assert_close(point, truth.evaluate(tp, u, v), 1e-5);
            }
        }
    }
}

#[test]
fn infinitely_sharp_crease_yields_regular_patches() {
    // A 4x4-face grid split by an infinitely sharp crease along an interior
    // column: every corner is regular (crease vertices have two-face
    // spans), so adaptive refinement has nothing to isolate and the base
    // level is patched with exact B-spline patches, the crease acting as a
    // boundary. Their corners evaluate to the crease limit points, and the
    // surface agrees with uniform refinement.
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            verts_per_face.push(4usize);
            let v = j * 5 + i;
            face_verts.extend_from_slice(&[v, v + 1, v + 6, v + 5]);
        }
    }
    let creases: Vec<[u32; 2]> = (0..4).map(|j| [2 + 5 * j, 7 + 5 * j]).collect();
    let weights = [sdc::SHARPNESS_INFINITE; 4];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor =
        TopologyDescriptor::new(25, &verts_per_face, &face_verts).with_creases(&creases, &weights);
    let mut positions = Vec::new();
    for j in 0..5 {
        for i in 0..5 {
            // A fold along the crease column.
            let x = i as f32;
            positions.push([
                x,
                j as f32,
                (x - 2.0).abs() + 0.1 * ((i * 3 + j * 5) % 4) as f32,
            ]);
        }
    }

    let surface = Surface::adaptive(descriptor, options, 3, &positions);
    assert_eq!(
        surface.refiner.max_level(),
        0,
        "regular creases need no isolation"
    );
    assert_eq!(surface.table.num_patches(), 16);
    assert_eq!(count(&surface.table, PatchType::Regular), 16);

    // Corners of the uniformly refined crease patches evaluate to the
    // crease-vertex limits.
    let uniform = Surface::uniform(descriptor, options, 2, &positions);
    assert_eq!(count(&uniform.table, PatchType::Regular), 256);
    let last = PrimvarRefiner::new(&uniform.refiner)
        .interpolate_all(&positions)
        .pop()
        .unwrap();
    let mut limits = last.clone();
    PrimvarRefiner::new(&uniform.refiner).limit(&last, &mut limits);
    let level2 = uniform.refiner.level(2);
    for p in 0..uniform.table.num_patches() {
        let param = uniform.table.patch_param(p);
        let (u, v) = param.unnormalize(0.0, 0.0);
        let c0 = level2.face_vertices(uniform.table.patch_face(p) as usize)[0] as usize;
        assert_close(uniform.evaluate(p, u, v), limits[c0], 1e-5);
    }

    // The base-level patches describe the same surface.
    let map_a = PatchMap::new(&surface.table);
    let map_u = PatchMap::new(&uniform.table);
    for ptex in 0..16 {
        for &(u, v) in &[
            (0.1f32, 0.7f32),
            (0.5, 0.5),
            (0.93, 0.02),
            (0.0, 0.5),
            (1.0, 1.0),
        ] {
            let pa = map_a.find_patch(ptex, u, v).unwrap();
            let pu = map_u.find_patch(ptex, u, v).unwrap();
            let (qa, dua, dva) = surface.table.evaluate(pa, u, v, &surface.controls);
            let (qu, duu, dvu) = uniform.table.evaluate(pu, u, v, &uniform.controls);
            assert_close(qa, qu, 1e-4);
            assert_close(dua, duu, 2e-3);
            assert_close(dva, dvu, 2e-3);
        }
    }
}

#[test]
fn sharp_corner_caps_interpolate_the_vertex() {
    // An infinitely sharp corner weight pins a cube vertex: its three caps
    // are Gregory patches whose corner point is the vertex itself.
    let corners = [0u32];
    let weights = [sdc::SHARPNESS_INFINITE];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_corners(&corners, &weights);
    let surface = Surface::adaptive(descriptor, sdc::Options::default(), 2, &CUBE_POSITIONS);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    assert_caps_are_affine(&surface);

    // Ptex face 0 is base face 0 (verts 0,1,3,2): its (0,0) corner is vertex 0.
    let map = PatchMap::new(&surface.table);
    let patch = map.find_patch(0, 0.0, 0.0).unwrap();
    assert_eq!(surface.table.patch_type(patch), PatchType::GregoryBasis);
    assert_close(surface.evaluate(patch, 0.0, 0.0), CUBE_POSITIONS[0], 1e-6);
}

#[test]
fn fully_creased_cube_corner_is_a_regular_corner_patch() {
    // When all three edges at a cube corner are infinitely sharp, the
    // corner is a regular set of inf-sharp corners: its faces become exact
    // B-spline corner patches (pinned at the vertex) instead of caps, and
    // the whole cube needs no isolation beyond level 1.
    let creases = [[0u32, 1u32], [0, 2], [0, 6]];
    let weights = [sdc::SHARPNESS_INFINITE; 3];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&creases, &weights);
    let surface = Surface::adaptive(descriptor, sdc::Options::default(), 4, &CUBE_POSITIONS);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    // 7 smooth corners keep 3 caps each; vertex 0's three faces are regular.
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 21);

    let map = PatchMap::new(&surface.table);
    let patch = map.find_patch(0, 0.0, 0.0).unwrap();
    assert_eq!(surface.table.patch_type(patch), PatchType::Regular);
    assert_eq!(surface.table.patch_param(patch).depth, 1);
    assert_close(surface.evaluate(patch, 0.0, 0.0), CUBE_POSITIONS[0], 1e-6);
    // Being a regular patch, it is the exact limit surface along the crease.
    let truth = Surface::uniform(descriptor, sdc::Options::default(), 6, &CUBE_POSITIONS);
    assert_close(
        surface.evaluate(patch, 0.3, 0.0),
        truth.exact_limit(0, 0.3, 0.0),
        1e-5,
    );
}

#[test]
fn unresolved_semi_sharp_crease_does_not_pin_the_cap() {
    // A grid vertex with one infinitely sharp edge and one semi-sharp edge
    // that has not decayed at a shallow isolation level: only infinitely
    // sharp features shape the cap, so the vertex is a dart — its cap
    // corner is the smooth limit of its neighborhood, not the vertex
    // itself, which the limit surface never reaches.
    let mut verts_per_face = Vec::new();
    let mut face_verts: Vec<u32> = Vec::new();
    for j in 0..4u32 {
        for i in 0..4u32 {
            verts_per_face.push(4usize);
            let v = j * 5 + i;
            face_verts.extend_from_slice(&[v, v + 1, v + 6, v + 5]);
        }
    }
    let creases = [[12u32, 13u32], [12, 7]];
    let weights = [sdc::SHARPNESS_INFINITE, 2.5];
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let descriptor =
        TopologyDescriptor::new(25, &verts_per_face, &face_verts).with_creases(&creases, &weights);
    let mut positions = Vec::new();
    for j in 0..5 {
        for i in 0..5 {
            positions.push([i as f32, j as f32, 0.0]);
        }
    }
    positions[12][2] = 1.0; // the creased vertex stands out of the plane

    let surface = Surface::adaptive(descriptor, options, 1, &positions);
    assert_eq!(surface.refiner.max_level(), 1);
    // Vertex 12 is corner 2 of face 5 (ptex 5, uv (1,1)).
    let map = PatchMap::new(&surface.table);
    let patch = map.find_patch(5, 1.0, 1.0).unwrap();
    assert_eq!(surface.table.patch_type(patch), PatchType::GregoryBasis);
    let corner = surface.evaluate(patch, 1.0, 1.0);

    // Deep uniform refinement approaches the true limit of the vertex. The
    // cap cannot be exact there — the crease still shapes two more levels
    // of refinement — but it must be much closer to the limit than the
    // vertex it would otherwise be pinned to.
    let truth = Surface::uniform(descriptor, options, 6, &positions);
    let tp = PatchMap::new(&truth.table).find_patch(5, 1.0, 1.0).unwrap();
    let limit = truth.evaluate(tp, 1.0, 1.0);
    let pinned_error = distance(positions[12], limit);
    let cap_error = distance(corner, limit);
    assert!(pinned_error > 0.3, "{limit:?}");
    assert!(
        cap_error < 0.35 * pinned_error,
        "cap {corner:?} vs limit {limit:?} (pinned error {pinned_error})"
    );
}

#[test]
fn smooth_boundaries_keep_the_bilinear_fallback() {
    // With `VtxBoundaryInterpolation::None` boundary edges stay smooth and
    // follow rules the Gregory construction does not model: faces on such
    // boundaries stay bilinear at the cap, while the interior extraordinary
    // vertex is still capped with Gregory patches.
    let (verts_per_face, face_verts, positions) = fan();
    let descriptor = TopologyDescriptor::new(11, &verts_per_face, &face_verts);
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::None);

    let surface = Surface::adaptive(descriptor, options, 2, &positions);
    assert_eq!(count(&surface.table, PatchType::GregoryBasis), 5);
    assert!(count(&surface.table, PatchType::Quads) > 0);
    let level = surface.refiner.level(2);
    for p in 0..surface.table.num_patches() {
        let face = surface.table.patch_face(p) as usize;
        let depth = surface.table.patch_param(p).depth as usize;
        let on_boundary = surface
            .refiner
            .level(depth)
            .face_vertices(face)
            .iter()
            .any(|&v| surface.refiner.level(depth).is_vertex_boundary(v as usize));
        match surface.table.patch_type(p) {
            PatchType::Quads => assert!(on_boundary && depth == 2),
            PatchType::GregoryBasis => {
                assert!(!on_boundary);
                assert!(level
                    .face_vertices(face)
                    .iter()
                    .any(|&v| level.vertex_edges(v as usize).len() == 5));
            }
            PatchType::Regular => assert!(!on_boundary),
            other => panic!("Catmark never builds {other:?} patches"),
        }
    }
}

#[test]
fn bow_tie_vertex_is_capped_consistently() {
    // A closed fan of four quads and an open fan of two quads sharing only
    // one vertex: the vertex is non-manifold, so — as in OpenSubdiv — it is
    // made infinitely sharp, and every face at it gets a Gregory cap over
    // its own fan whose corner interpolates the vertex, rather than mixing
    // caps on one fan with bilinear quads on the other.
    let verts_per_face = [4usize; 6];
    let face_verts: [u32; 24] = [
        0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7, // 2x2 grid around vertex 4
        4, 9, 10, 11, 4, 11, 12, 13, // open fan hanging off vertex 4
    ];
    let mut positions: Vec<P3> = (0..9)
        .map(|k| [(k % 3) as f32, (k / 3) as f32, 0.0])
        .collect();
    positions.extend_from_slice(&[
        [1.5, 1.0, 1.0],
        [1.5, 1.5, 2.0],
        [1.0, 1.5, 1.0],
        [0.5, 1.5, 2.0],
        [0.5, 1.0, 1.0],
    ]);
    let descriptor = TopologyDescriptor::new(14, &verts_per_face, &face_verts);
    let surface = Surface::adaptive(descriptor, sdc::Options::default(), 2, &positions);
    assert!(surface.refiner.level(0).vertex_sharpness(4) >= sdc::SHARPNESS_INFINITE);
    assert_eq!(count(&surface.table, PatchType::Quads), 0);
    assert_caps_are_affine(&surface);

    let level = surface.refiner.level(2);
    let child = |v: u32| -> u32 {
        let c1 = surface
            .refiner
            .refinement(1)
            .vertex_child_vertex(v as usize);
        surface
            .refiner
            .refinement(2)
            .vertex_child_vertex(c1 as usize)
    };
    let shared = child(4);
    assert_eq!(level.vertex_faces(shared as usize).len(), 6);
    let mut at_shared = 0;
    for p in 0..surface.table.num_patches() {
        let param = surface.table.patch_param(p);
        if param.depth != 2 {
            continue;
        }
        let face = surface.table.patch_face(p) as usize;
        if level.face_vertices(face).contains(&shared) {
            assert_eq!(surface.table.patch_type(p), PatchType::GregoryBasis);
            // One corner of the cap is pinned to the sharp vertex.
            let pinned = [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .iter()
                .map(|&(s, t)| {
                    let (u, v) = param.unnormalize(s, t);
                    distance(surface.evaluate(p, u, v), positions[4])
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                pinned < 1e-5,
                "patch {p}: corner misses the vertex by {pinned}"
            );
            at_shared += 1;
        }
    }
    assert_eq!(at_shared, 6);
}
