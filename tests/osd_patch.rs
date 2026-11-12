// Contains the cube mesh of OpenSubdiv's tutorials/far/tutorial_1_1
// (https://github.com/PixarAnimationStudios/OpenSubdiv) — Copyright 2013 Pixar.
// Licensed under the Tomorrow Open Source Technology License 1.0; see
// LICENSE-OPENSUBDIV at the root of this crate.

//! GPU patch-table tests: the flat buffers of an `osd::GpuPatchTable`,
//! evaluated the way a GPU kernel evaluates them, must reproduce
//! `PatchTable::evaluate` — for every patch type, rotation, boundary mask,
//! single crease, non-quad root and inverted triangle.

use opensubdiv_rs::far::{
    AdaptiveOptions, PatchMap, PatchTable, PatchTableFactory, PrimvarRefiner, StencilTableFactory,
    StencilTableOptions, TopologyDescriptor, TopologyRefiner, TopologyRefinerFactory,
    UniformOptions,
};
use opensubdiv_rs::osd::{patch_descriptor, GpuPatchTable, PatchParam};
use opensubdiv_rs::sdc;

type P3 = [f32; 3];

/// Patch control values: every level's vertex values concatenated, base
/// level first.
fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
}

fn assert_close(actual: P3, expected: P3, what: &str) {
    for (a, e) in actual.iter().zip(&expected) {
        let tol = 1e-4 * (1.0 + e.abs());
        assert!(
            (a - e).abs() < tol,
            "{what}: expected {expected:?}, got {actual:?}"
        );
    }
}

/// Deterministic pseudo-random samples of a patch domain, its corners
/// included.
fn samples(count: usize, triangular: bool) -> Vec<(f32, f32)> {
    let mut seed = 0x5eed_1234u32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut out: Vec<(f32, f32)> = if triangular {
        vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    } else {
        vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    };
    while out.len() < count {
        let (a, b) = (rand(), rand());
        out.push(if triangular && a + b > 1.0 {
            (1.0 - a, 1.0 - b)
        } else {
            (a, b)
        });
    }
    out
}

/// What a table exercised, for the fixtures to assert their coverage.
#[derive(Debug, Default)]
struct Coverage {
    per_desc: [usize; 11],
    rotated_quads: usize,
    boundary: usize,
    box_spline_vertex_walls: usize,
    single_crease: usize,
    non_quad_root: usize,
    rotated_triangles: usize,
    local_points: usize,
}

/// Build the GPU table of `table`, check its buffers, and require its CPU
/// reference evaluation to match `PatchTable::evaluate` everywhere.
fn check_gpu_table(refiner: &TopologyRefiner, table: &PatchTable, base: &[P3]) -> Coverage {
    let gpu = GpuPatchTable::new(table).unwrap();
    assert_eq!(gpu.num_patches(), table.num_patches());
    assert_eq!(gpu.num_control_values(), table.num_control_values());
    let num_values = gpu.num_control_values() + gpu.num_local_points();

    // Arrays tile the patches and the index buffer in order.
    let arrays = gpu.patch_arrays();
    let mut next_patch = 0;
    let mut next_index = 0;
    for array in arrays {
        assert_eq!(array.primitive_id_base, next_patch);
        assert_eq!(array.index_base, next_index);
        assert_eq!(array.reg_desc, array.desc);
        assert_eq!(
            array.stride as usize,
            patch_descriptor::num_control_vertices(array.desc)
        );
        next_patch += array.num_patches;
        next_index += array.num_patches * array.stride;
    }
    assert_eq!(next_patch as usize, gpu.num_patches());
    assert_eq!(next_index as usize, gpu.patch_index_buffer().len());
    assert!(gpu
        .patch_index_buffer()
        .iter()
        .all(|&i| (i as usize) < num_values));
    assert_eq!(gpu.patch_array_words().len(), 6 * arrays.len());
    assert_eq!(gpu.patch_param_words().len(), 3 * gpu.num_patches());

    // The buffer the patches index: control values, then local points —
    // computed in two passes, and in one through the appended stencils.
    let controls = patch_controls(refiner, base);
    let mut local = vec![[0.0f32; 3]; gpu.num_local_points()];
    gpu.local_point_stencils()
        .update_values(&controls, &mut local);
    let values: Vec<P3> = controls.iter().chain(&local).copied().collect();

    let appended = StencilTableFactory::append_local_point_stencils(
        &StencilTableFactory::create(refiner, StencilTableOptions::for_patch_controls()),
        gpu.local_point_stencils(),
    );
    assert_eq!(appended.num_stencils(), num_values);
    assert_eq!(appended.num_control_vertices(), base.len());
    let mut one_pass = vec![[0.0f32; 3]; num_values];
    appended.update_values(base, &mut one_pass);
    for (a, e) in one_pass.iter().zip(&values) {
        assert_close(*a, *e, "appended local-point stencils");
    }

    let mut coverage = Coverage {
        local_points: gpu.num_local_points(),
        ..Coverage::default()
    };
    for patch in 0..table.num_patches() {
        let far_param = table.patch_param(patch);
        let coord0 = gpu.patch_coord(patch, 0.0, 0.0);
        let array = &arrays[coord0.array_index as usize];
        let param: PatchParam = gpu.patch_param_buffer()[patch];
        assert_eq!(
            patch_descriptor::to_patch_type(array.desc),
            Some(table.patch_type(patch))
        );
        assert_eq!(param.face_id(), far_param.ptex_face);
        assert_eq!(param.transition(), 0);

        coverage.per_desc[array.desc as usize] += 1;
        coverage.rotated_quads += (!far_param.triangular && far_param.rotation != 0) as usize;
        coverage.boundary += (param.boundary() != 0 && param.sharpness == 0.0) as usize;
        coverage.box_spline_vertex_walls +=
            (array.desc == patch_descriptor::LOOP && param.boundary() >> 3 != 0) as usize;
        coverage.single_crease += (param.sharpness > 0.0) as usize;
        coverage.non_quad_root += param.non_quad_root() as usize;
        coverage.rotated_triangles +=
            (far_param.triangular && param.is_triangle_rotated()) as usize;

        for (s, t) in samples(12, far_param.triangular) {
            let (u, v) = far_param.unnormalize(s, t);
            // The packed parameter maps the location back into the patch.
            let (gs, gt) = if far_param.triangular {
                param.normalize_triangle(u, v)
            } else {
                param.normalize(u, v)
            };
            assert!(
                (-1e-4..=1.0 + 1e-4).contains(&gs) && (-1e-4..=1.0 + 1e-4).contains(&gt),
                "patch {patch}: ({u}, {v}) normalizes to ({gs}, {gt})"
            );

            let coord = gpu.patch_coord(patch, u, v);
            let (p, du, dv) = gpu.evaluate(&coord, &values);
            let (ep, edu, edv) = table.evaluate(patch, u, v, &controls);
            let what = |q: &str| {
                format!(
                    "patch {patch} ({:?}) {q} at ({s}, {t})",
                    table.patch_type(patch)
                )
            };
            assert_close(p, ep, &what("point"));
            assert_close(du, edu, &what("du"));
            assert_close(dv, edv, &what("dv"));
        }
    }

    // Locations found through the patch map evaluate alike.
    let map = PatchMap::new(table);
    for ptex in 0..table.ptex_indices().num_faces() {
        for (u, v) in samples(6, false) {
            let Some(patch) = map.find_patch(ptex, u, v) else {
                continue;
            };
            if table.patch_param(patch).triangular && u + v > 1.0 {
                continue;
            }
            let (p, _, _) = gpu.evaluate(&gpu.patch_coord(patch, u, v), &values);
            let (ep, _, _) = table.evaluate(patch, u, v, &controls);
            assert_close(p, ep, "patch-map location");
        }
    }
    coverage
}

// ----------------------------------------------------------------------
//  Catmark
// ----------------------------------------------------------------------

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

fn edge_and_corner() -> sdc::Options {
    sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
}

/// A `w x h`-vertex quad grid with a bumpy height field.
fn quad_grid(w: u32, h: u32) -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let v = j * w + i;
            verts_per_face.push(4);
            face_verts.extend_from_slice(&[v, v + 1, v + w + 1, v + w]);
        }
    }
    let positions = (0..h)
        .flat_map(|j| {
            (0..w).map(move |i| {
                let (x, y) = (i as f32, j as f32);
                [
                    x + 0.1 * y.sin(),
                    y,
                    0.3 * (0.9 * x).sin() * (1.3 * y).cos(),
                ]
            })
        })
        .collect();
    (verts_per_face, face_verts, positions)
}

#[test]
fn adaptive_cube_buffers_match_patch_evaluation() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &CUBE_POSITIONS);
    assert!(c.per_desc[patch_descriptor::REGULAR as usize] > 0);
    assert!(c.per_desc[patch_descriptor::GREGORY_BASIS as usize] > 0);
    assert!(c.rotated_quads > 0, "{c:?}");
    assert_eq!(
        c.local_points,
        20 * c.per_desc[patch_descriptor::GREGORY_BASIS as usize]
    );
}

#[test]
fn single_crease_buffers_match_patch_evaluation() {
    let creases = [[0u32, 1], [1, 3], [3, 2], [2, 0]];
    let weights = [2.5f32; 4];
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS)
        .with_creases(&creases, &weights);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(4).with_single_crease_patch(true));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &CUBE_POSITIONS);
    assert!(c.single_crease > 0, "{c:?}");
}

#[test]
fn boundary_and_sharp_grid_buffers_match_patch_evaluation() {
    // Boundaries and corners, an infinitely sharp crease across the middle
    // (boundary masks on interior patches), and an extraordinary vertex.
    let (w, h) = (7u32, 6u32);
    let (mut verts_per_face, mut face_verts, mut positions) = quad_grid(w, h);
    // Split the quad at (1, 1) into two triangles: valence-5 vertices.
    let q = (w - 1 + 1) as usize;
    let quad: Vec<u32> = face_verts[4 * q..4 * q + 4].to_vec();
    face_verts.splice(
        4 * q..4 * q + 4,
        [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]],
    );
    verts_per_face.splice(q..q + 1, [3, 3]);
    positions[0][2] += 0.2;
    let creases: Vec<[u32; 2]> = (0..w - 1).map(|i| [3 * w + i, 3 * w + i + 1]).collect();
    let weights = vec![sdc::SHARPNESS_INFINITE; creases.len()];
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts)
        .with_creases(&creases, &weights);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, edge_and_corner())
            .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert!(c.boundary > 0 && c.non_quad_root > 0, "{c:?}");
    assert!(c.per_desc[patch_descriptor::GREGORY_BASIS as usize] > 0);
}

#[test]
fn pentagon_buffers_match_patch_evaluation() {
    let positions: Vec<P3> = (0..5)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / 5.0;
            [a.cos(), a.sin(), 0.1 * k as f32]
        })
        .collect();
    let descriptor = TopologyDescriptor::new(5, &[5], &[0, 1, 2, 3, 4]);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, edge_and_corner())
            .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert_eq!(c.non_quad_root, table.num_patches());
}

#[test]
fn bilinear_quads_buffers_match_patch_evaluation() {
    let descriptor = TopologyDescriptor::new(8, &CUBE_VERTS_PER_FACE, &CUBE_FACE_VERTS);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Bilinear,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &CUBE_POSITIONS);
    assert_eq!(
        c.per_desc[patch_descriptor::QUADS as usize],
        table.num_patches()
    );
}

// ----------------------------------------------------------------------
//  Loop
// ----------------------------------------------------------------------

/// A triangulated `w x h`-vertex grid, one diagonal flipped to create
/// valence-5 and valence-7 vertices.
fn tri_grid(w: u32, h: u32) -> (Vec<usize>, Vec<u32>, Vec<P3>) {
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let v = j * w + i;
            verts_per_face.extend([3usize, 3]);
            if (i, j) == (2, 2) {
                face_verts.extend_from_slice(&[v, v + 1, v + w, v + 1, v + w + 1, v + w]);
            } else {
                face_verts.extend_from_slice(&[v, v + 1, v + w + 1, v, v + w + 1, v + w]);
            }
        }
    }
    let (_, _, positions) = quad_grid(w, h);
    (verts_per_face, face_verts, positions)
}

#[test]
fn loop_grid_buffers_match_patch_evaluation() {
    let (w, h) = (8u32, 7u32);
    let (verts_per_face, face_verts, positions) = tri_grid(w, h);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, edge_and_corner())
            .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert!(c.per_desc[patch_descriptor::LOOP as usize] > 0);
    assert!(c.per_desc[patch_descriptor::GREGORY_TRIANGLE as usize] > 0);
    assert!(c.boundary > 0 && c.box_spline_vertex_walls > 0, "{c:?}");
    assert!(c.rotated_triangles > 0, "{c:?}");
}

#[test]
fn loop_icosahedron_buffers_match_patch_evaluation() {
    let face_verts: [u32; 60] = [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let positions: Vec<P3> = [
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
    ]
    .to_vec();
    let descriptor = TopologyDescriptor::new(12, &[3; 20], &face_verts);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, sdc::Options::default())
            .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert_eq!(
        c.local_points,
        18 * c.per_desc[patch_descriptor::GREGORY_TRIANGLE as usize]
    );
    assert!(c.rotated_triangles > 0, "{c:?}");
}

#[test]
fn bilinear_triangles_are_quads_below_non_quad_roots() {
    // The Bilinear scheme splits triangles into quads, as OpenSubdiv does:
    // every patch is a bilinear quad below a non-quad root, most rotated.
    let (w, h) = (4u32, 4u32);
    let (verts_per_face, face_verts, positions) = tri_grid(w, h);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Bilinear,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_uniform(UniformOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert_eq!(
        c.per_desc[patch_descriptor::QUADS as usize],
        table.num_patches()
    );
    assert_eq!(c.non_quad_root, table.num_patches());
    assert!(c.rotated_quads > 0, "{c:?}");
}

#[test]
fn linear_triangle_buffers_match_patch_evaluation() {
    // Unsharpened Loop boundaries fall back to linear triangles.
    let (w, h) = (5u32, 5u32);
    let (verts_per_face, face_verts, positions) = tri_grid(w, h);
    let descriptor = TopologyDescriptor::new((w * h) as usize, &verts_per_face, &face_verts);
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::None);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Loop, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let c = check_gpu_table(&refiner, &table, &positions);
    assert!(
        c.per_desc[patch_descriptor::TRIANGLES as usize] > 0,
        "{c:?}"
    );
    assert!(c.rotated_triangles > 0, "{c:?}");
}
