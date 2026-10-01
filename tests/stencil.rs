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

// ----------------------------------------------------------------------
//  Adaptive hierarchies and limit stencils over the patch table
// ----------------------------------------------------------------------

use opensubdiv_rs::far::{
    AdaptiveOptions, Error, LimitStencilTableFactory, LocationArray, PatchMap, PatchTable,
    PatchTableFactory, PatchType, StencilTable, TopologyRefiner,
};

/// The four edges of the cube's first face carry a semi-sharp crease, so
/// adaptive refinement isolates both the corner EVs and the crease.
const CUBE_CREASES: [[u32; 2]; 4] = [[0, 1], [1, 3], [3, 2], [2, 0]];
const CUBE_CREASE_WEIGHTS: [f32; 4] = [2.0; 4];

fn creased_cube_adaptive(levels: usize) -> TopologyRefiner {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&CUBE_CREASES, &CUBE_CREASE_WEIGHTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(levels));
    refiner
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

fn assert_close(actual: P3, expected: P3, tol: f32) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - e).abs() < tol, "expected {expected:?}, got {actual:?}");
    }
}

/// Deterministic pseudo-random parametric samples.
fn samples(count: usize) -> Vec<(f32, f32)> {
    let mut seed = 0x13579bdu32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..count).map(|_| (rand(), rand())).collect()
}

/// Evaluate every stencil of `table` from `base` and require it to match
/// `PatchTable::evaluate` at the corresponding `(ptex, u, v)` location over
/// the concatenated level buffer: point and first derivatives.
fn assert_limit_stencils_match_patches(
    table: &opensubdiv_rs::far::LimitStencilTable,
    patches: &PatchTable,
    refiner: &TopologyRefiner,
    base: &[P3],
    locations: &[(usize, f32, f32)],
) {
    assert_eq!(table.num_stencils(), locations.len());
    assert_eq!(table.num_control_vertices(), base.len());
    let controls = patch_controls(refiner, base);
    let map = PatchMap::new(patches);

    let mut points = vec![[0.0f32; 3]; table.num_stencils()];
    let mut du = points.clone();
    let mut dv = points.clone();
    table.update_values(base, &mut points);
    table.update_derivs(base, &mut du, &mut dv);

    for (i, &(ptex, u, v)) in locations.iter().enumerate() {
        let patch = map.find_patch(ptex, u, v).unwrap();
        let (p, pu, pv) = patches.evaluate(patch, u, v, &controls);
        assert_close(points[i], p, 1e-5);
        assert_close(du[i], pu, 1e-4);
        assert_close(dv[i], pv, 1e-4);

        // Position weights are a partition of unity; derivative weights sum
        // to zero, and the stencil never reaches outside the base cage.
        let stencil = table.stencil(i);
        let sum: f32 = stencil.weights.iter().sum();
        let dsum: f32 = stencil.du_weights.iter().sum();
        let tsum: f32 = stencil.dv_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "stencil {i} sums to {sum}");
        assert!(dsum.abs() < 1e-4 && tsum.abs() < 1e-4);
        assert!(stencil.indices.iter().all(|&cv| (cv as usize) < base.len()));
    }
}

#[test]
fn adaptive_stencils_match_primvar_refinement() {
    let refiner = creased_cube_adaptive(3);
    assert!(refiner.is_adaptive());
    assert_eq!(refiner.max_level(), 3);
    let by_level = refine_positions(&refiner);

    // Default options: every refined level, base level first.
    let table = StencilTableFactory::create(&refiner, StencilTableOptions::default());
    let total: usize = by_level.iter().map(Vec::len).sum();
    assert_eq!(table.num_stencils(), total);
    let mut values = vec![[0.0f32; 3]; total];
    table.update_values(&CUBE_POSITIONS, &mut values);
    let mut offset = 0;
    for level in &by_level {
        assert_all_close(&values[offset..offset + level.len()], level);
        offset += level.len();
    }

    // Last level only.
    let table = StencilTableFactory::create(
        &refiner,
        StencilTableOptions {
            generate_intermediate_levels: false,
            generate_control_verts: false,
        },
    );
    assert_eq!(table.num_stencils(), by_level.last().unwrap().len());

    // With control vertices: the patch table's control buffer, in order.
    let table = StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls());
    let patches = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_stencils(), patches.num_control_values());
    let mut values = vec![[0.0f32; 3]; table.num_stencils()];
    table.update_values(&CUBE_POSITIONS, &mut values);
    assert_all_close(&values, &patch_controls(&refiner, &CUBE_POSITIONS));
}

/// The issue's proposed test: for an adaptively refined creased cube,
/// limit stencils at the centre of every patch agree with `PatchTable::
/// evaluate` over the concatenated level buffer, since one factorizes the
/// other.
#[test]
fn limit_stencils_factorize_patch_evaluation() {
    let refiner = creased_cube_adaptive(4);
    let patches = PatchTableFactory::create(&refiner).unwrap();
    let count = |t: PatchType| {
        (0..patches.num_patches())
            .filter(|&p| patches.patch_type(p) == t)
            .count()
    };
    // Mixed depths and both bases are exercised.
    assert!(count(PatchType::Regular) > 0 && count(PatchType::GregoryBasis) > 0);
    assert!((0..patches.num_patches()).any(|p| patches.patch_param(p).depth == 2));
    assert!((0..patches.num_patches()).any(|p| patches.patch_param(p).depth == 4));

    // One location array per ptex face, holding the centre of each of its
    // patches.
    let mut per_face: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); 6];
    for p in 0..patches.num_patches() {
        let param = patches.patch_param(p);
        let (u, v) = param.unnormalize(0.5, 0.5);
        per_face[param.ptex_face as usize].0.push(u);
        per_face[param.ptex_face as usize].1.push(v);
    }
    let arrays: Vec<LocationArray> = per_face
        .iter()
        .enumerate()
        .map(|(f, (u, v))| LocationArray::new(f as u32, u, v))
        .collect();
    let locations: Vec<(usize, f32, f32)> = per_face
        .iter()
        .enumerate()
        .flat_map(|(f, (u, v))| u.iter().zip(v).map(move |(&u, &v)| (f, u, v)))
        .collect();

    let table = LimitStencilTableFactory::create(&refiner, &arrays, None, None).unwrap();
    assert_eq!(table.num_stencils(), patches.num_patches());
    assert_limit_stencils_match_patches(&table, &patches, &refiner, &CUBE_POSITIONS, &locations);

    // Each stencil is one sparse dot product over the cage: at most the 8
    // cube vertices.
    for i in 0..table.num_stencils() {
        assert!(table.stencil(i).size() <= 8);
        assert_eq!(
            table.stencil(i).size(),
            table.point_stencils().stencil(i).size()
        );
    }
}

#[test]
fn limit_stencils_are_reused_across_deforming_cages() {
    // The stencils depend on topology only: re-posing the cage re-evaluates
    // the limit surface with the same table.
    let refiner = creased_cube_adaptive(3);
    let patches = PatchTableFactory::create(&refiner).unwrap();
    let locations: Vec<(usize, f32, f32)> = (0..6usize)
        .flat_map(|f| samples(10).into_iter().map(move |(u, v)| (f, u, v)))
        .collect();
    let (u, v): (Vec<f32>, Vec<f32>) = locations.iter().map(|&(_, u, v)| (u, v)).unzip();
    let arrays: Vec<LocationArray> = (0..6usize)
        .map(|f| LocationArray::new(f as u32, &u[10 * f..10 * f + 10], &v[10 * f..10 * f + 10]))
        .collect();
    let table = LimitStencilTableFactory::create(&refiner, &arrays, None, None).unwrap();

    let twisted: Vec<P3> = CUBE_POSITIONS
        .iter()
        .map(|&[x, y, z]| {
            let a = 0.7 * (y + 0.5);
            [
                x * a.cos() - z * a.sin() + 0.1,
                1.5 * y,
                x * a.sin() + z * a.cos(),
            ]
        })
        .collect();
    assert_limit_stencils_match_patches(&table, &patches, &refiner, &CUBE_POSITIONS, &locations);
    assert_limit_stencils_match_patches(&table, &patches, &refiner, &twisted, &locations);
}

#[test]
fn limit_stencils_accept_prebuilt_tables_and_uniform_refiners() {
    // A uniformly refined cube: regular patches and Gregory caps at the
    // corner EVs, evaluated through tables built up front.
    let refiner = cube_refiner(2);
    let patches = PatchTableFactory::create(&refiner).unwrap();
    let cv_stencils: StencilTable =
        StencilTableFactory::create(&refiner, StencilTableOptions::for_patch_controls());

    let locations: Vec<(usize, f32, f32)> = (0..6usize)
        .flat_map(|f| samples(8).into_iter().map(move |(u, v)| (f, u, v)))
        .chain([(0, 0.0, 0.0), (3, 1.0, 1.0), (5, 0.5, 0.0)])
        .collect();
    let (u, v): (Vec<f32>, Vec<f32>) = locations.iter().map(|&(_, u, v)| (u, v)).unzip();
    let mut arrays: Vec<LocationArray> = (0..6usize)
        .map(|f| LocationArray::new(f as u32, &u[8 * f..8 * f + 8], &v[8 * f..8 * f + 8]))
        .collect();
    arrays.push(LocationArray::new(0, &u[48..49], &v[48..49]));
    arrays.push(LocationArray::new(3, &u[49..50], &v[49..50]));
    arrays.push(LocationArray::new(5, &u[50..51], &v[50..51]));

    let table =
        LimitStencilTableFactory::create(&refiner, &arrays, Some(&cv_stencils), Some(&patches))
            .unwrap();
    assert_limit_stencils_match_patches(&table, &patches, &refiner, &CUBE_POSITIONS, &locations);

    // Corner locations are the limit points of the cube corners: the
    // vertex limit stencils agree.
    let limits = StencilTableFactory::create_limit(&refiner);
    let mut limit_points = vec![[0.0f32; 3]; limits.num_stencils()];
    limits.update_values(&CUBE_POSITIONS, &mut limit_points);
    let mut points = vec![[0.0f32; 3]; table.num_stencils()];
    table.update_values(&CUBE_POSITIONS, &mut points);
    // Ptex face 0 is base face 0, whose (0, 0) corner is base vertex 0;
    // follow its descendants down to the last level.
    let mut corner = 0u32;
    for level in 1..=refiner.max_level() {
        corner = refiner
            .refinement(level)
            .vertex_child_vertex(corner as usize);
    }
    assert_close(points[48], limit_points[corner as usize], 1e-5);
}

#[test]
fn limit_stencils_cover_loop_patches() {
    // An adaptively refined tetrahedron under Loop: every vertex has
    // valence 3, so the Gregory triangles cap each isolation region.
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
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    assert!(refiner.is_adaptive());
    let patches = PatchTableFactory::create(&refiner).unwrap();
    assert!((0..patches.num_patches()).any(|p| patches.patch_type(p) == PatchType::Loop));
    assert!((0..patches.num_patches()).any(|p| patches.patch_type(p) == PatchType::GregoryTriangle));

    // Samples inside the triangular domain.
    let locations: Vec<(usize, f32, f32)> = (0..4usize)
        .flat_map(|f| {
            samples(10).into_iter().map(move |(u, v)| {
                if u + v > 1.0 {
                    (f, 1.0 - u, 1.0 - v)
                } else {
                    (f, u, v)
                }
            })
        })
        .collect();
    let (u, v): (Vec<f32>, Vec<f32>) = locations.iter().map(|&(_, u, v)| (u, v)).unzip();
    let arrays: Vec<LocationArray> = (0..4usize)
        .map(|f| LocationArray::new(f as u32, &u[10 * f..10 * f + 10], &v[10 * f..10 * f + 10]))
        .collect();
    let table = LimitStencilTableFactory::create(&refiner, &arrays, None, None).unwrap();
    assert_limit_stencils_match_patches(&table, &patches, &refiner, &positions, &locations);
}

#[test]
fn limit_stencils_report_holes() {
    let holes = [2u32];
    let descriptor =
        TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS).with_holes(&holes);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(2));

    let arrays = [
        LocationArray::new(0, &[0.5], &[0.5]),
        LocationArray::new(2, &[0.25, 0.75], &[0.25, 0.75]),
    ];
    let err = LimitStencilTableFactory::create(&refiner, &arrays, None, None).unwrap_err();
    assert_eq!(
        err,
        Error::LimitLocationInHole {
            ptex_face: 2,
            location: 1
        }
    );
    assert_eq!(
        err.to_string(),
        "limit location 1 lies in a hole of ptex face 2"
    );

    // Empty arrays are fine and yield no stencils.
    let empty = LocationArray::new(1, &[], &[]);
    assert!(empty.is_empty());
    let table = LimitStencilTableFactory::create(&refiner, &[empty], None, None).unwrap();
    assert_eq!(table.num_stencils(), 0);
}
