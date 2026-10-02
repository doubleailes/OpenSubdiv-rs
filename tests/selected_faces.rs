//! Selected-face refinement and patch tables:
//! `TopologyRefiner::refine_adaptive_selected` and
//! `PatchTableFactory::create_with_options_selected`.
//!
//! The central invariant: on the selected faces, a selected refinement and
//! a selected table give exactly the patches a full refinement and a full
//! table give — same types, parameterizations and sharpness, and
//! bit-identical limit positions and derivatives — while every other face
//! is left without a patch.

use opensubdiv_rs::far::{
    AdaptiveOptions, FVarChannelDescriptor, LimitStencilTableFactory, LocationArray, PatchMap,
    PatchTable, PatchTableFactory, PatchTableOptions, PatchType, PrimvarRefiner,
    TopologyDescriptor, TopologyRefiner, TopologyRefinerFactory,
};
use opensubdiv_rs::sdc::{self, FVarLinearInterpolation, SchemeType};
use opensubdiv_rs::vtr::TopologyError;

type P3 = [f32; 3];
type UV = [f32; 2];

/// A mesh with every kind of feature adaptive refinement isolates, and one
/// face-varying channel with a seam.
struct Mesh {
    verts_per_face: Vec<usize>,
    face_verts: Vec<u32>,
    positions: Vec<P3>,
    crease_pairs: Vec<[u32; 2]>,
    crease_weights: Vec<f32>,
    corners: Vec<u32>,
    corner_weights: Vec<f32>,
    holes: Vec<u32>,
    uv_indices: Vec<u32>,
    uvs: Vec<UV>,
}

const N: u32 = 8;

fn vid(i: u32, j: u32) -> u32 {
    j * (N + 1) + i
}

fn grid_positions() -> Vec<P3> {
    let mut positions = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            let (x, y) = (i as f32, j as f32);
            positions.push([x, y, 0.3 * (0.7 * x).sin() * (0.5 * y).cos() + 0.05 * x * y]);
        }
    }
    positions
}

/// The face-varying values of `face_verts`: the vertex's own value, except
/// that faces right of column 4 take separate values on that column, a
/// seam.
fn seamed_uvs(verts_per_face: &[usize], face_verts: &[u32]) -> (Vec<u32>, Vec<UV>) {
    let mut uvs: Vec<UV> = (0..=N)
        .flat_map(|j| (0..=N).map(move |i| [i as f32 / N as f32, j as f32 / N as f32]))
        .collect();
    let seam_base = uvs.len() as u32;
    for j in 0..=N {
        uvs.push([0.55, j as f32 / N as f32]);
    }
    let mut indices = Vec::with_capacity(face_verts.len());
    let mut start = 0;
    for &size in verts_per_face {
        let face = &face_verts[start..start + size];
        let right = face.iter().any(|&v| v % (N + 1) > 4);
        for &v in face {
            let (i, j) = (v % (N + 1), v / (N + 1));
            indices.push(if right && i == 4 { seam_base + j } else { v });
        }
        start += size;
    }
    (indices, uvs)
}

/// An 8×8 quad grid with two triangles, a hexagon (and so extraordinary
/// vertices), a semi-sharp crease of varying sharpness, a sharp corner and
/// a hole.
fn catmark_mesh() -> Mesh {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..N {
        for i in 0..N {
            let quad = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)];
            match (i, j) {
                (2, 2) => {
                    verts_per_face.extend([3, 3]);
                    face_verts.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
                }
                (5, 1) => {
                    verts_per_face.push(6);
                    face_verts.extend([
                        vid(5, 1),
                        vid(6, 1),
                        vid(7, 1),
                        vid(7, 2),
                        vid(6, 2),
                        vid(5, 2),
                    ]);
                }
                (6, 1) => {} // merged into the hexagon
                _ => {
                    verts_per_face.push(4);
                    face_verts.extend(quad);
                }
            }
        }
    }
    let crease_pairs: Vec<[u32; 2]> = (1..5).map(|i| [vid(i, 6), vid(i + 1, 6)]).collect();
    let (uv_indices, uvs) = seamed_uvs(&verts_per_face, &face_verts);
    Mesh {
        verts_per_face,
        face_verts,
        positions: grid_positions(),
        crease_pairs,
        crease_weights: vec![1.5, 2.0, 2.0, 0.5],
        corners: vec![0],
        corner_weights: vec![2.0],
        holes: vec![62],
        uv_indices,
        uvs,
    }
}

/// An 8×8 grid of triangle pairs, two of them with flipped diagonals
/// (extraordinary vertices), a semi-sharp crease and a hole.
fn loop_mesh() -> Mesh {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..N {
        for i in 0..N {
            let [a, b, c, d] = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)];
            verts_per_face.extend([3, 3]);
            if (i, j) == (2, 3) || (i, j) == (5, 5) {
                face_verts.extend([a, b, d, b, c, d]);
            } else {
                face_verts.extend([a, b, c, a, c, d]);
            }
        }
    }
    let crease_pairs: Vec<[u32; 2]> = (1..5).map(|i| [vid(i, 6), vid(i + 1, 6)]).collect();
    let (uv_indices, uvs) = seamed_uvs(&verts_per_face, &face_verts);
    Mesh {
        verts_per_face,
        face_verts,
        positions: grid_positions(),
        crease_pairs,
        crease_weights: vec![1.0, 1.5, 1.5, 3.0],
        corners: vec![],
        corner_weights: vec![],
        holes: vec![100],
        uv_indices,
        uvs,
    }
}

impl Mesh {
    fn num_faces(&self) -> usize {
        self.verts_per_face.len()
    }

    fn refiner(&self, scheme: SchemeType) -> TopologyRefiner {
        let channels = [FVarChannelDescriptor::new(self.uvs.len(), &self.uv_indices)];
        let descriptor =
            TopologyDescriptor::new(self.positions.len(), &self.verts_per_face, &self.face_verts)
                .with_creases(&self.crease_pairs, &self.crease_weights)
                .with_corners(&self.corners, &self.corner_weights)
                .with_holes(&self.holes)
                .with_fvar_channels(&channels);
        let options = sdc::Options {
            fvar_linear_interpolation: FVarLinearInterpolation::None,
            ..sdc::Options::default()
        };
        TopologyRefinerFactory::create(descriptor, scheme, options).unwrap()
    }
}

fn controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

fn fvar_values(refiner: &TopologyRefiner, base: &[UV]) -> Vec<UV> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_face_varying_all(0, base) {
        all.extend(level);
    }
    all
}

/// Deterministic pseudo-random parametric samples, including the corners.
fn samples(triangular: bool) -> Vec<(f32, f32)> {
    let mut seed = 0x1357_9bdfu32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut samples = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
    if !triangular {
        samples.push((1.0, 1.0));
    }
    for _ in 0..10 {
        let (u, v) = (rand(), rand());
        samples.push(if triangular && u + v > 1.0 {
            (1.0 - u, 1.0 - v)
        } else {
            (u, v)
        });
    }
    samples
}

/// A refined mesh and its patch table, control values and face-varying
/// values.
struct Patched {
    refiner: TopologyRefiner,
    table: PatchTable,
    controls: Vec<P3>,
    uvs: Vec<UV>,
}

fn table_options() -> PatchTableOptions {
    PatchTableOptions::new()
        .with_fvar_tables(true)
        .with_fvar_legacy_linear_patches(false)
}

fn patch_full(mesh: &Mesh, scheme: SchemeType, options: AdaptiveOptions) -> Patched {
    let mut refiner = mesh.refiner(scheme);
    refiner.refine_adaptive(options);
    let table = PatchTableFactory::create_with_options(&refiner, &table_options()).unwrap();
    let controls = controls(&refiner, &mesh.positions);
    let uvs = fvar_values(&refiner, &mesh.uvs);
    Patched {
        refiner,
        table,
        controls,
        uvs,
    }
}

fn patch_selected(
    mesh: &Mesh,
    scheme: SchemeType,
    options: AdaptiveOptions,
    selected: &[u32],
) -> Patched {
    let mut refiner = mesh.refiner(scheme);
    refiner.refine_adaptive_selected(options, selected);
    let table =
        PatchTableFactory::create_with_options_selected(&refiner, &table_options(), selected)
            .unwrap();
    let controls = controls(&refiner, &mesh.positions);
    let uvs = fvar_values(&refiner, &mesh.uvs);
    Patched {
        refiner,
        table,
        controls,
        uvs,
    }
}

/// On the faces in `selected`, `sel` must hold exactly `full`'s patches
/// and evaluate them bit-identically; elsewhere it must hold none.
fn assert_selected_matches_full(full: &Patched, sel: &Patched, selected: &[u32], holes: &[u32]) {
    let ptex = full.table.ptex_indices();
    let triangular = full.refiner.scheme_type() == SchemeType::Loop;
    let full_map = PatchMap::new(&full.table);
    let sel_map = PatchMap::new(&sel.table);
    let num_faces = full.refiner.level(0).num_faces();
    for face in 0..num_faces {
        let is_selected = selected.contains(&(face as u32));
        let is_hole = holes.contains(&(face as u32));
        let first = ptex.face_id(face) as usize;
        for ptex_face in first..first + ptex.face_ptex_count(face) {
            for &(u, v) in &samples(triangular) {
                let found = sel_map.find_patch(ptex_face, u, v);
                if !is_selected || is_hole {
                    assert_eq!(found, None, "face {face} has a patch but is not selected");
                    continue;
                }
                let p = full_map.find_patch(ptex_face, u, v).unwrap();
                let q = found.unwrap_or_else(|| panic!("selected face {face} has no patch"));
                assert_eq!(full.table.patch_type(p), sel.table.patch_type(q));
                let (a, b) = (full.table.patch_param(p), sel.table.patch_param(q));
                assert_eq!(
                    (a.ptex_face, a.depth, a.rotation, a.origin, a.triangular),
                    (b.ptex_face, b.depth, b.rotation, b.origin, b.triangular),
                );
                assert_eq!(
                    full.table.single_crease_sharpness(p),
                    sel.table.single_crease_sharpness(q)
                );
                assert_eq!(
                    full.table.evaluate(p, u, v, &full.controls),
                    sel.table.evaluate(q, u, v, &sel.controls),
                    "face {face}, ptex face {ptex_face} at ({u}, {v})"
                );
                assert_eq!(
                    full.table.fvar_patch_type(p, 0),
                    sel.table.fvar_patch_type(q, 0)
                );
                assert_eq!(
                    full.table.evaluate_face_varying(p, u, v, &full.uvs, 0),
                    sel.table.evaluate_face_varying(q, u, v, &sel.uvs, 0),
                    "face-varying: face {face}, ptex face {ptex_face} at ({u}, {v})"
                );
            }
        }
    }

    // Nothing more than the selected faces' patches.
    let expected = (0..full.table.num_patches())
        .filter(|&p| {
            let base = ptex.base_face(full.table.patch_param(p).ptex_face as usize);
            selected.contains(&base)
        })
        .count();
    assert_eq!(sel.table.num_patches(), expected);
}

/// Scattered selections touching every feature of the meshes.
fn selections(mesh: &Mesh) -> Vec<Vec<u32>> {
    let num_faces = mesh.num_faces() as u32;
    vec![
        vec![0],
        vec![18, 19, 13, 50, 51, 52],
        vec![27, 13, 13, mesh.holes[0], 40, num_faces - 1],
        (0..num_faces).step_by(5).collect(),
        (0..num_faces).collect(),
    ]
}

fn check_scheme(mesh: &Mesh, scheme: SchemeType, options: AdaptiveOptions) {
    let full = patch_full(mesh, scheme, options);
    // The fixtures isolate down to the cap and need end caps there.
    assert_eq!(full.refiner.max_level(), options.isolation_level);
    let end_cap = match scheme {
        SchemeType::Loop => PatchType::GregoryTriangle,
        _ => PatchType::GregoryBasis,
    };
    assert!((0..full.table.num_patches()).any(|p| full.table.patch_type(p) == end_cap));
    for selected in selections(mesh) {
        let sel = patch_selected(mesh, scheme, options, &selected);
        assert_selected_matches_full(&full, &sel, &selected, &mesh.holes);
    }
    // Each face on its own, too.
    for face in 0..mesh.num_faces() as u32 {
        let sel = patch_selected(mesh, scheme, options, &[face]);
        assert_selected_matches_full(&full, &sel, &[face], &mesh.holes);
    }
}

#[test]
fn catmark_selected_patches_match_full_table() {
    check_scheme(
        &catmark_mesh(),
        SchemeType::Catmark,
        AdaptiveOptions::new(3),
    );
}

#[test]
fn catmark_selected_patches_match_full_table_with_options() {
    let options = AdaptiveOptions::new(4)
        .with_single_crease_patch(true)
        .with_consider_fvar_channels(true);
    check_scheme(&catmark_mesh(), SchemeType::Catmark, options);
}

#[test]
fn loop_selected_patches_match_full_table() {
    check_scheme(&loop_mesh(), SchemeType::Loop, AdaptiveOptions::new(3));
    let options = AdaptiveOptions::new(3).with_consider_fvar_channels(true);
    check_scheme(&loop_mesh(), SchemeType::Loop, options);
}

#[test]
fn selected_refinement_refines_only_around_the_selection() {
    let mesh = catmark_mesh();
    let full = patch_full(&mesh, SchemeType::Catmark, AdaptiveOptions::new(4));
    // The hexagon alone.
    let sel = patch_selected(&mesh, SchemeType::Catmark, AdaptiveOptions::new(4), &[13]);
    assert_eq!(sel.refiner.max_level(), full.refiner.max_level());
    assert!(sel.refiner.num_faces_total() * 2 < full.refiner.num_faces_total());
    assert!(sel.table.num_patches() * 4 < full.table.num_patches());

    // A regular face needs no refinement at all.
    let regular = patch_selected(&mesh, SchemeType::Catmark, AdaptiveOptions::new(4), &[38]);
    assert_eq!(regular.refiner.max_level(), 0);
    assert_eq!(regular.table.num_patches(), 1);
}

#[test]
fn full_refinement_with_selected_table_shares_patches() {
    // With the full refiner, the selected table's patches are the full
    // table's, control vertices included.
    let mesh = catmark_mesh();
    let full = patch_full(&mesh, SchemeType::Catmark, AdaptiveOptions::new(3));
    let selected = [13u32, 27, 50];
    let table =
        PatchTableFactory::create_with_options_selected(&full.refiner, &table_options(), &selected)
            .unwrap();
    assert_eq!(table.num_control_values(), full.table.num_control_values());
    assert_eq!(table.num_fvar_values(0), full.table.num_fvar_values(0));
    let ptex = full.table.ptex_indices();
    let mut matched = 0;
    for q in 0..table.num_patches() {
        let param = table.patch_param(q);
        let base = ptex.base_face(param.ptex_face as usize);
        assert!(selected.contains(&base));
        let p = (0..full.table.num_patches())
            .find(|&p| {
                full.table.patch_face(p) == table.patch_face(q)
                    && full.table.patch_param(p).depth == param.depth
                    && full.table.patch_param(p).ptex_face == param.ptex_face
            })
            .unwrap();
        assert_eq!(full.table.patch_vertices(p), table.patch_vertices(q));
        assert_eq!(
            full.table.fvar_patch_values(p, 0),
            table.fvar_patch_values(q, 0)
        );
        matched += 1;
    }
    assert!(matched > 0);
}

#[test]
fn limit_stencils_on_a_selected_table() {
    let mesh = catmark_mesh();
    let selected = [13u32, 18, 19];
    let sel = patch_selected(
        &mesh,
        SchemeType::Catmark,
        AdaptiveOptions::new(3),
        &selected,
    );
    let ptex = sel.table.ptex_indices();
    let u = [0.0f32, 0.25, 0.5, 0.9];
    let v = [0.0f32, 0.75, 0.5, 0.1];
    let mut arrays = Vec::new();
    for &face in &selected {
        let first = ptex.face_id(face as usize);
        for k in 0..ptex.face_ptex_count(face as usize) {
            arrays.push(LocationArray::new(first + k as u32, &u, &v));
        }
    }
    let stencils =
        LimitStencilTableFactory::create(&sel.refiner, &arrays, None, Some(&sel.table)).unwrap();
    let mut points = vec![[0.0f32; 3]; stencils.num_stencils()];
    stencils.update_values(&mesh.positions, &mut points);
    let map = PatchMap::new(&sel.table);
    let mut i = 0;
    for array in &arrays {
        for (&u, &v) in array.u.iter().zip(array.v) {
            let patch = map.find_patch(array.ptex_face as usize, u, v).unwrap();
            let (expected, _, _) = sel.table.evaluate(patch, u, v, &sel.controls);
            for (a, e) in points[i].iter().zip(&expected) {
                assert!((a - e).abs() < 1e-4, "{:?} vs {expected:?}", points[i]);
            }
            i += 1;
        }
    }

    // A location on an unselected face has no patch.
    let outside = [LocationArray::new(ptex.face_id(40), &u, &v)];
    let err = LimitStencilTableFactory::create(&sel.refiner, &outside, None, Some(&sel.table));
    assert!(matches!(
        err,
        Err(TopologyError::LimitLocationInHole { location: 0, .. })
    ));
}

#[test]
fn empty_selection_refines_and_patches_nothing() {
    let mesh = catmark_mesh();
    let sel = patch_selected(&mesh, SchemeType::Catmark, AdaptiveOptions::new(3), &[]);
    assert_eq!(sel.refiner.max_level(), 0);
    assert_eq!(sel.table.num_patches(), 0);
    let map = PatchMap::new(&sel.table);
    for ptex_face in 0..sel.table.ptex_indices().num_faces() {
        assert_eq!(map.find_patch(ptex_face, 0.5, 0.5), None);
    }
}

#[test]
fn selected_face_out_of_range_is_an_error() {
    let mesh = catmark_mesh();
    let mut refiner = mesh.refiner(SchemeType::Catmark);
    refiner.refine_adaptive(AdaptiveOptions::new(2));
    let num_faces = mesh.num_faces();
    let result = PatchTableFactory::create_with_options_selected(
        &refiner,
        &PatchTableOptions::new(),
        &[3, num_faces as u32],
    );
    assert_eq!(
        result.err(),
        Some(TopologyError::SelectedFaceOutOfRange {
            face: num_faces as u32,
            num_faces,
        })
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn refining_an_out_of_range_selected_face_panics() {
    let mesh = catmark_mesh();
    let mut refiner = mesh.refiner(SchemeType::Catmark);
    refiner.refine_adaptive_selected(AdaptiveOptions::new(2), &[mesh.num_faces() as u32]);
}

#[test]
fn unrefined_non_quads_cannot_be_patched() {
    let mesh = catmark_mesh();
    let mut refiner = mesh.refiner(SchemeType::Catmark);
    refiner.refine_adaptive_selected(AdaptiveOptions::new(2), &[0, 1]);
    let options = PatchTableOptions::new();
    // The hexagon (face 13) was not refined: neither a selected table
    // including it nor a full table can patch it.
    assert_eq!(
        PatchTableFactory::create_with_options_selected(&refiner, &options, &[0, 13]).err(),
        Some(TopologyError::PatchesRequireRefinement)
    );
    assert_eq!(
        PatchTableFactory::create(&refiner).err(),
        Some(TopologyError::PatchesRequireRefinement)
    );
    // The faces that were selected can.
    assert!(PatchTableFactory::create_with_options_selected(&refiner, &options, &[0, 1]).is_ok());
}

#[test]
fn full_table_over_a_selected_refinement_caps_the_rest() {
    // On a quad mesh, a full table built from a selected refinement covers
    // every face: the selected ones as a full refinement would, the others
    // at the base level.
    let face_verts = [
        0u32, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
    ];
    let positions: Vec<P3> = vec![
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
    ];
    let create = || {
        TopologyRefinerFactory::create(
            TopologyDescriptor::new(8, &[4; 6], &face_verts),
            SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap()
    };
    let mut full = create();
    full.refine_adaptive(AdaptiveOptions::new(3));
    let full_table = PatchTableFactory::create(&full).unwrap();
    let full_controls = controls(&full, &positions);

    let mut sel = create();
    sel.refine_adaptive_selected(AdaptiveOptions::new(3), &[2]);
    let table = PatchTableFactory::create(&sel).unwrap();
    let sel_controls = controls(&sel, &positions);

    let (full_map, map) = (PatchMap::new(&full_table), PatchMap::new(&table));
    for ptex_face in 0..6 {
        for &(u, v) in &samples(false) {
            let p = full_map.find_patch(ptex_face, u, v).unwrap();
            let q = map.find_patch(ptex_face, u, v).unwrap();
            if ptex_face == 2 {
                assert_eq!(
                    full_table.evaluate(p, u, v, &full_controls),
                    table.evaluate(q, u, v, &sel_controls)
                );
            } else {
                assert_eq!(table.patch_param(q).depth, 0);
            }
        }
    }
}

#[test]
fn unrefined_non_quad_holes_need_no_patch() {
    // Non-quad faces are only required to be refined when they get a
    // patch: as holes they don't.
    let mut mesh = catmark_mesh();
    mesh.holes = vec![13, 17, 18];
    let refiner = mesh.refiner(SchemeType::Catmark);
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_patches(), mesh.num_faces() - 3);
    let ptex = table.ptex_indices();
    let map = PatchMap::new(&table);
    for face in [13usize, 17, 18] {
        let first = ptex.face_id(face) as usize;
        for ptex_face in first..first + ptex.face_ptex_count(face) {
            assert_eq!(map.find_patch(ptex_face, 0.5, 0.25), None);
        }
    }
}
