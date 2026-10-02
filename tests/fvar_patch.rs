//! Face-varying patches: `PatchTableOptions::generate_fvar_tables`,
//! `PatchTable::evaluate_basis_face_varying` and
//! `AdaptiveOptions::consider_fvar_channels`.

use opensubdiv_rs::far::{
    AdaptiveOptions, FVarChannelDescriptor, PatchMap, PatchTable, PatchTableFactory,
    PatchTableOptions, PatchType, PrimvarRefiner, TopologyDescriptor, TopologyRefiner,
    TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc::{self, FVarLinearInterpolation, SchemeType};
use opensubdiv_rs::vtr::TopologyError;

type UV = [f32; 2];

const SMOOTH_MODES: [FVarLinearInterpolation; 5] = [
    FVarLinearInterpolation::None,
    FVarLinearInterpolation::CornersOnly,
    FVarLinearInterpolation::CornersPlus1,
    FVarLinearInterpolation::CornersPlus2,
    FVarLinearInterpolation::Boundaries,
];

/// A mesh with one face-varying channel.
struct Mesh {
    verts_per_face: Vec<usize>,
    face_verts: Vec<u32>,
    num_vertices: usize,
    uv_indices: Vec<u32>,
    uvs: Vec<UV>,
}

impl Mesh {
    fn refiner(
        &self,
        scheme: SchemeType,
        options: sdc::Options,
        refine: impl FnOnce(&mut TopologyRefiner),
    ) -> TopologyRefiner {
        let channels = [FVarChannelDescriptor::new(self.uvs.len(), &self.uv_indices)];
        let descriptor =
            TopologyDescriptor::new(self.num_vertices, &self.verts_per_face, &self.face_verts)
                .with_fvar_channels(&channels);
        let mut refiner = TopologyRefinerFactory::create(descriptor, scheme, options).unwrap();
        refine(&mut refiner);
        refiner
    }
}

/// The unit cube of `far_tutorial_1_1`, with a UV seam along the path of
/// edges 1-7-6-0: it passes through vertices 7 and 6, whose values split in
/// two, and ends at vertices 1 and 0 (face-varying darts).
fn seamed_cube() -> Mesh {
    let face_verts = vec![
        0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
    ];
    // Face 3 (6 7 1 0) takes new values 8 and 9 at vertices 6 and 7.
    let mut uv_indices = face_verts.clone();
    uv_indices[12] = 8;
    uv_indices[13] = 9;
    let uvs = (0..10)
        .map(|i| {
            let x = i as f32;
            [0.3 * x + 0.05 * x * x, 1.0 - 0.2 * x + 0.01 * x * x * x]
        })
        .collect();
    Mesh {
        verts_per_face: vec![4; 6],
        face_verts,
        num_vertices: 8,
        uv_indices,
        uvs,
    }
}

/// A 4x4 quad grid split into three UV islands — the left half, and the
/// bottom and top quarters of the right half — whose seams meet at a
/// junction of three values in the middle of the grid. The vertex topology
/// is regular everywhere (with pinned corners).
fn seamed_grid() -> Mesh {
    let n = 4u32;
    let mut face_verts = Vec::new();
    let mut islands = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let v = j * (n + 1) + i;
            face_verts.extend([v, v + 1, v + n + 2, v + n + 1]);
            islands.push(if i < 2 {
                0
            } else if j < 2 {
                1
            } else {
                2
            });
        }
    }
    // One value per (vertex, island) pair.
    let mut keys: Vec<(u32, u32)> = Vec::new();
    let mut uv_indices = Vec::new();
    for (f, corners) in face_verts.chunks(4).enumerate() {
        for &v in corners {
            let key = (v, islands[f]);
            let index = keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                keys.push(key);
                keys.len() - 1
            });
            uv_indices.push(index as u32);
        }
    }
    let uvs = keys
        .iter()
        .map(|&(v, island)| {
            let (x, y) = ((v % (n + 1)) as f32, (v / (n + 1)) as f32);
            let island = island as f32;
            [
                0.2 * x + 0.1 * island + 0.01 * x * y,
                0.3 * y - 0.05 * island * island,
            ]
        })
        .collect();
    Mesh {
        verts_per_face: vec![4; (n * n) as usize],
        face_verts,
        num_vertices: ((n + 1) * (n + 1)) as usize,
        uv_indices,
        uvs,
    }
}

/// A triangulated 4x4 grid with a UV seam down its middle column of
/// edges.
fn seamed_tri_grid() -> Mesh {
    let n = 4u32;
    let mut face_verts = Vec::new();
    let mut islands = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let v = j * (n + 1) + i;
            face_verts.extend([v, v + 1, v + n + 2, v, v + n + 2, v + n + 1]);
            islands.extend([(i >= 2) as u32; 2]);
        }
    }
    let mut keys: Vec<(u32, u32)> = Vec::new();
    let mut uv_indices = Vec::new();
    for (f, corners) in face_verts.chunks(3).enumerate() {
        for &v in corners {
            let key = (v, islands[f]);
            let index = keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                keys.push(key);
                keys.len() - 1
            });
            uv_indices.push(index as u32);
        }
    }
    let uvs = keys
        .iter()
        .map(|&(v, island)| {
            let (x, y) = ((v % (n + 1)) as f32, (v / (n + 1)) as f32);
            [
                0.25 * x + 0.2 * island as f32 + 0.02 * y * y,
                0.25 * y - 0.01 * x * y,
            ]
        })
        .collect();
    Mesh {
        verts_per_face: vec![3; (2 * n * n) as usize],
        face_verts,
        num_vertices: ((n + 1) * (n + 1)) as usize,
        uv_indices,
        uvs,
    }
}

/// The channel's values at every level, base level first: what
/// `PatchTable::evaluate_face_varying` expects.
fn fvar_values<T: opensubdiv_rs::far::Primvar + Default>(
    refiner: &TopologyRefiner,
    base: &[T],
) -> Vec<T> {
    let mut values = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_face_varying_all(0, base) {
        values.extend(level);
    }
    values
}

fn smooth_fvar() -> PatchTableOptions {
    PatchTableOptions::new()
        .with_fvar_tables(true)
        .with_fvar_legacy_linear_patches(false)
}

fn assert_close(actual: UV, expected: UV, tol: f32, what: &str) {
    for k in 0..2 {
        assert!(
            (actual[k] - expected[k]).abs() <= tol,
            "{what}: expected {expected:?}, got {actual:?}"
        );
    }
}

/// The local corners of a patch: quads, then triangles.
fn patch_corners(triangular: bool) -> &'static [(f32, f32)] {
    if triangular {
        &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    } else {
        &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }
}

/// A channel whose value indices are the vertex indices has the vertex
/// topology, so — without linear interpolation of its own — its patches
/// are the vertex patches, weight for weight.
#[test]
fn seamless_channel_patches_are_the_vertex_patches() {
    let cube = seamed_cube();
    let mut grid = seamed_grid();
    let mut tris = seamed_tri_grid();
    let mut cube = cube;
    for mesh in [&mut cube, &mut grid, &mut tris] {
        mesh.uv_indices = mesh.face_verts.clone();
        mesh.uvs = vec![[0.0; 2]; mesh.num_vertices];
    }
    let corners = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let cases = [
        (&cube, SchemeType::Catmark, sdc::Options::default()),
        (&grid, SchemeType::Catmark, corners),
        (&tris, SchemeType::Loop, corners),
    ];
    for (mesh, scheme, options) in cases {
        for mode in [
            FVarLinearInterpolation::None,
            FVarLinearInterpolation::CornersOnly,
        ] {
            let options = options.with_fvar_linear_interpolation(mode);
            let uniform = mesh.refiner(scheme, options, |r| {
                r.refine_uniform(UniformOptions::new(2));
            });
            let adaptive = mesh.refiner(scheme, options, |r| {
                r.refine_adaptive(AdaptiveOptions::new(3));
            });
            for refiner in [uniform, adaptive] {
                let table =
                    PatchTableFactory::create_with_options(&refiner, &smooth_fvar()).unwrap();
                assert_eq!(table.num_fvar_channels(), 1);
                assert_eq!(table.num_fvar_values(0), table.num_control_values());
                for patch in 0..table.num_patches() {
                    assert_eq!(table.fvar_patch_type(patch, 0), table.patch_type(patch));
                    assert_eq!(
                        table.fvar_patch_values(patch, 0),
                        table.patch_vertices(patch)
                    );
                    let param = table.patch_param(patch);
                    let (u, v) = param.unnormalize(0.3, 0.2);
                    let vertex = table.evaluate_basis(patch, u, v);
                    let fvar = table.evaluate_basis_face_varying(patch, u, v, 0);
                    assert_eq!(fvar.indices, vertex.indices);
                    assert_eq!(fvar.weights, vertex.weights);
                    assert_eq!(fvar.du_weights, vertex.du_weights);
                    assert_eq!(fvar.dv_weights, vertex.dv_weights);
                }
            }
        }
    }
}

/// At the corners of its face, a face-varying patch — regular or Gregory —
/// takes the channel's limit value there, across seams, darts, junctions
/// and every linear-interpolation rule.
#[test]
fn fvar_patches_interpolate_the_limit_at_face_corners() {
    let corners = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let cases = [
        (seamed_cube(), SchemeType::Catmark, sdc::Options::default()),
        (seamed_grid(), SchemeType::Catmark, corners),
        (seamed_grid(), SchemeType::Catmark, sdc::Options::default()),
        (seamed_tri_grid(), SchemeType::Loop, corners),
    ];
    let mut seen = Vec::new();
    for (mesh, scheme, options) in &cases {
        for mode in SMOOTH_MODES
            .into_iter()
            .chain([FVarLinearInterpolation::All])
        {
            let level = 2;
            let options = options.with_fvar_linear_interpolation(mode);
            let refiner = mesh.refiner(*scheme, options, |r| {
                r.refine_uniform(UniformOptions::new(level));
            });
            let table = PatchTableFactory::create_with_options(&refiner, &smooth_fvar()).unwrap();
            let values = fvar_values(&refiner, &mesh.uvs);
            assert_eq!(values.len(), table.num_fvar_values(0));

            // The limit of the last level's values.
            let last = refiner.level(level);
            let first_last = values.len() - last.num_fvar_values(0);
            let mut limit = vec![[0.0f32; 2]; last.num_fvar_values(0)];
            PrimvarRefiner::new(&refiner).limit_face_varying(0, &values[first_last..], &mut limit);

            for patch in 0..table.num_patches() {
                let kind = table.fvar_patch_type(patch, 0);
                if mode == FVarLinearInterpolation::All {
                    let linear = if *scheme == SchemeType::Loop {
                        PatchType::Triangles
                    } else {
                        PatchType::Quads
                    };
                    assert_eq!(kind, linear);
                }
                seen.push(kind);
                let face = table.patch_face(patch) as usize;
                let face_values = last.face_fvar_values(face, 0);
                let param = table.fvar_patch_param(patch, 0);
                for (corner, &(s, t)) in patch_corners(param.triangular).iter().enumerate() {
                    let (u, v) = param.unnormalize(s, t);
                    let (value, _, _) = table.evaluate_face_varying(patch, u, v, &values, 0);
                    let expected = limit[face_values[corner] as usize];
                    let what = format!("{mode:?}, {kind:?} patch {patch}, corner {corner}");
                    assert_close(value, expected, 2e-5, &what);
                }
            }
        }
    }
    for kind in [
        PatchType::Regular,
        PatchType::GregoryBasis,
        PatchType::Quads,
        PatchType::Loop,
        PatchType::GregoryTriangle,
        PatchType::Triangles,
    ] {
        assert!(seen.contains(&kind), "no {kind:?} face-varying patch");
    }
}

/// Adaptive refinement that considers the face-varying channel isolates its
/// features as uniform refinement does, so its face-varying patches
/// evaluate the same surface as those of a uniformly refined mesh — only
/// with far fewer patches.
#[test]
fn adaptive_fvar_patches_match_uniform_with_consider_fvar_channels() {
    let corners = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner);
    let cases = [
        (seamed_cube(), SchemeType::Catmark, sdc::Options::default()),
        (seamed_grid(), SchemeType::Catmark, corners),
        (seamed_tri_grid(), SchemeType::Loop, corners),
    ];
    let level = 3;
    for (mesh, scheme, options) in &cases {
        for mode in SMOOTH_MODES {
            let options = options.with_fvar_linear_interpolation(mode);
            let uniform = mesh.refiner(*scheme, options, |r| {
                r.refine_uniform(UniformOptions::new(level));
            });
            let adaptive = mesh.refiner(*scheme, options, |r| {
                r.refine_adaptive(AdaptiveOptions::new(level).with_consider_fvar_channels(true));
            });
            let uniform_table =
                PatchTableFactory::create_with_options(&uniform, &smooth_fvar()).unwrap();
            let adaptive_table =
                PatchTableFactory::create_with_options(&adaptive, &smooth_fvar()).unwrap();
            assert!(adaptive_table.num_patches() <= uniform_table.num_patches());
            let uniform_values = fvar_values(&uniform, &mesh.uvs);
            let adaptive_values = fvar_values(&adaptive, &mesh.uvs);

            let eval = |table: &PatchTable, values: &[UV], ptex: usize, u: f32, v: f32| {
                let patch = PatchMap::new(table).find_patch(ptex, u, v).unwrap();
                table.evaluate_face_varying(patch, u, v, values, 0)
            };
            let num_ptex = uniform_table.ptex_indices().num_faces();
            for ptex in 0..num_ptex {
                for (u, v) in [
                    (0.1, 0.13),
                    (0.5, 0.5),
                    (0.77, 0.21),
                    (0.31, 0.64),
                    (0.02, 0.9),
                ] {
                    if *scheme == SchemeType::Loop && u + v > 1.0 {
                        continue;
                    }
                    let (p0, du0, dv0) = eval(&uniform_table, &uniform_values, ptex, u, v);
                    let (p1, du1, dv1) = eval(&adaptive_table, &adaptive_values, ptex, u, v);
                    let what = format!("{mode:?}, ptex {ptex} at ({u}, {v})");
                    assert_close(p1, p0, 1e-5, &what);
                    assert_close(du1, du0, 1e-3, &what);
                    assert_close(dv1, dv0, 1e-3, &what);
                }
            }
        }
    }
}

/// Without `consider_fvar_channels`, a face whose vertex topology is
/// regular is patched at once, and its face-varying patch caps the
/// channel's irregularity at that level; with it, the face is isolated
/// like any vertex feature.
#[test]
fn consider_fvar_channels_isolates_face_varying_features() {
    let mesh = seamed_grid();
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
        .with_fvar_linear_interpolation(FVarLinearInterpolation::CornersPlus1);
    let level = 3;
    let coarse_caps = |consider: bool| {
        let refiner = mesh.refiner(SchemeType::Catmark, options, |r| {
            r.refine_adaptive(AdaptiveOptions::new(level).with_consider_fvar_channels(consider));
        });
        let table = PatchTableFactory::create_with_options(&refiner, &smooth_fvar()).unwrap();
        (0..table.num_patches())
            .filter(|&p| {
                table.fvar_patch_type(p, 0) == PatchType::GregoryBasis
                    && (table.patch_param(p).depth as usize) < level
            })
            .count()
    };
    // The vertex topology is regular: nothing is refined, and the
    // junction's faces are capped on the base level.
    assert!(coarse_caps(false) > 0);
    assert_eq!(coarse_caps(true), 0);
}

/// Linear face-varying patches: for `FVarLinearInterpolation::All`, and for
/// every channel under the (default) legacy linear patches.
#[test]
fn linear_fvar_patches_interpolate_their_refined_face() {
    let mesh = seamed_grid();
    for (mode, options) in [
        (FVarLinearInterpolation::All, smooth_fvar()),
        (
            FVarLinearInterpolation::CornersOnly,
            PatchTableOptions::new().with_fvar_tables(true),
        ),
    ] {
        let sdc_options = sdc::Options::default().with_fvar_linear_interpolation(mode);
        let refiner = mesh.refiner(SchemeType::Catmark, sdc_options, |r| {
            r.refine_adaptive(AdaptiveOptions::new(2));
        });
        let table = PatchTableFactory::create_with_options(&refiner, &options).unwrap();
        let values = fvar_values(&refiner, &mesh.uvs);
        let mut level_starts = vec![0];
        for level in 0..refiner.num_levels() {
            level_starts.push(level_starts[level] + refiner.level(level).num_fvar_values(0));
        }
        for patch in 0..table.num_patches() {
            assert_eq!(table.fvar_patch_type(patch, 0), PatchType::Quads);
            let param = table.patch_param(patch);
            let level = param.depth as usize;
            let face = table.patch_face(patch) as usize;
            let corner_values: Vec<UV> = refiner
                .level(level)
                .face_fvar_values(face, 0)
                .iter()
                .map(|&value| values[level_starts[level] + value as usize])
                .collect();
            let (s, t) = (0.3f32, 0.6f32);
            let w = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
            let mut expected = [0.0f32; 2];
            for (value, w) in corner_values.iter().zip(w) {
                expected[0] += w * value[0];
                expected[1] += w * value[1];
            }
            let (u, v) = param.unnormalize(s, t);
            let (value, _, _) = table.evaluate_face_varying(patch, u, v, &values, 0);
            assert_close(value, expected, 1e-6, &format!("{mode:?} patch {patch}"));
        }
    }
}

/// A value index reused at several vertices is one independent value at
/// each of them; base-level face-varying patches still refer to the
/// caller's indices.
#[test]
fn base_level_patches_refer_to_the_callers_values() {
    // A 3x3 grid whose UV index is the vertex's column: every index is
    // reused down its column.
    let n = 3u32;
    let mut face_verts = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let v = j * (n + 1) + i;
            face_verts.extend([v, v + 1, v + n + 2, v + n + 1]);
        }
    }
    let reused = Mesh {
        verts_per_face: vec![4; (n * n) as usize],
        uv_indices: face_verts.iter().map(|&v| v % (n + 1)).collect(),
        uvs: (0..=n)
            .map(|i| [i as f32 * 0.5, (i * i) as f32 * 0.1])
            .collect(),
        face_verts: face_verts.clone(),
        num_vertices: ((n + 1) * (n + 1)) as usize,
    };
    // The same values, one index per vertex.
    let distinct = Mesh {
        uv_indices: face_verts.clone(),
        uvs: (0..reused.num_vertices as u32)
            .map(|v| reused.uvs[(v % (n + 1)) as usize])
            .collect(),
        verts_per_face: reused.verts_per_face.clone(),
        face_verts,
        num_vertices: reused.num_vertices,
    };
    let options = sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
        .with_fvar_linear_interpolation(FVarLinearInterpolation::None);
    let tables: Vec<(PatchTable, Vec<UV>)> = [&reused, &distinct]
        .into_iter()
        .map(|mesh| {
            let refiner = mesh.refiner(SchemeType::Catmark, options, |_| {});
            let table = PatchTableFactory::create_with_options(&refiner, &smooth_fvar()).unwrap();
            (table, mesh.uvs.clone())
        })
        .collect();
    let (reused_table, reused_uvs) = &tables[0];
    let (distinct_table, distinct_uvs) = &tables[1];
    assert_eq!(reused_table.num_fvar_values(0), reused_uvs.len());
    for patch in 0..reused_table.num_patches() {
        assert_eq!(reused_table.fvar_patch_type(patch, 0), PatchType::Regular);
        for value in reused_table.fvar_patch_values(patch, 0) {
            assert!((value as usize) < reused_uvs.len());
        }
        for (u, v) in [(0.0, 0.0), (0.4, 0.7), (1.0, 0.5)] {
            let (a, _, _) = reused_table.evaluate_face_varying(patch, u, v, reused_uvs, 0);
            let (b, _, _) = distinct_table.evaluate_face_varying(patch, u, v, distinct_uvs, 0);
            assert_close(a, b, 1e-6, &format!("patch {patch} at ({u}, {v})"));
        }
    }
}

#[test]
fn fvar_tables_are_built_only_for_the_requested_channels() {
    let mesh = seamed_cube();
    let refiner = mesh.refiner(SchemeType::Catmark, sdc::Options::default(), |r| {
        r.refine_uniform(UniformOptions::new(1));
    });

    let table = PatchTableFactory::create(&refiner).unwrap();
    assert_eq!(table.num_fvar_channels(), 0);
    let ignored = PatchTableOptions::new().with_fvar_channels(&[0]);
    let table = PatchTableFactory::create_with_options(&refiner, &ignored).unwrap();
    assert_eq!(table.num_fvar_channels(), 0);

    let options = smooth_fvar().with_fvar_channels(&[0, 0]);
    let table = PatchTableFactory::create_with_options(&refiner, &options).unwrap();
    assert_eq!(table.num_fvar_channels(), 2);
    assert_eq!(table.fvar_refiner_channel(1), 0);
    assert_eq!(
        table.fvar_channel_linear_interpolation(0),
        FVarLinearInterpolation::CornersOnly
    );
    for patch in 0..table.num_patches() {
        let (a, b) = (table.patch_param(patch), table.fvar_patch_param(patch, 1));
        assert_eq!(
            (a.ptex_face, a.depth, a.rotation),
            (b.ptex_face, b.depth, b.rotation)
        );
        assert_eq!(a.origin, b.origin);
    }

    let out_of_range = smooth_fvar().with_fvar_channels(&[0, 1]);
    assert_eq!(
        PatchTableFactory::create_with_options(&refiner, &out_of_range).unwrap_err(),
        TopologyError::FVarChannelOutOfRange {
            channel: 1,
            num_channels: 1
        }
    );
}
