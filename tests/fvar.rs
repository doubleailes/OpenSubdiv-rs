//! Face-varying channel tests: seams, linear-interpolation modes, and
//! equivalence between face-varying refinement and vertex refinement of the
//! channel's value mesh.

use opensubdiv_rs::far::{
    FVarChannelDescriptor, PrimvarRefiner, TopologyDescriptor, TopologyRefiner,
    TopologyRefinerFactory, UniformOptions,
};
use opensubdiv_rs::sdc;

type UV = [f32; 2];

fn assert_close2(actual: UV, expected: UV) {
    for (a, e) in actual.iter().zip(&expected) {
        assert!(
            (a - e).abs() < 1e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

/// Two quads side by side, with a UV seam splitting the shared edge:
///
/// ```text
///   3 --- 4 --- 5        3---2 | 7---6
///   |  A  |  B  |   UVs: | A | | | B |
///   0 --- 1 --- 2        0---1 | 4---5
/// ```
fn seamed_two_quads(
    mode: sdc::FVarLinearInterpolation,
    levels: usize,
) -> (TopologyRefiner, Vec<UV>) {
    let verts_per_face = [4usize, 4];
    let face_verts = [0u32, 1, 4, 3, 1, 2, 5, 4];
    let uv_indices = [0u32, 1, 2, 3, 4, 5, 6, 7];
    let uvs = vec![
        [0.0, 0.0],
        [0.45, 0.0],
        [0.45, 1.0],
        [0.0, 1.0],
        [0.55, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.55, 1.0],
    ];
    let channels = [FVarChannelDescriptor::new(8, &uv_indices)];
    let descriptor =
        TopologyDescriptor::new(6, &verts_per_face, &face_verts).with_fvar_channels(&channels);
    let options = sdc::Options::default().with_fvar_linear_interpolation(mode);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(levels));
    (refiner, uvs)
}

#[test]
fn seam_splits_values() {
    let (refiner, uvs) = seamed_two_quads(sdc::FVarLinearInterpolation::All, 1);

    let level0 = refiner.level(0);
    assert_eq!(level0.num_fvar_channels(), 1);
    assert_eq!(level0.num_fvar_values(0), 8);
    assert_eq!(level0.face_fvar_values(0, 0), &[0, 1, 2, 3]);
    assert_eq!(level0.face_fvar_values(1, 0), &[4, 5, 6, 7]);

    // The value mesh of the channel is two disconnected quads: 8 fvar edges.
    // Child values: 2 (faces) + 8 (edges) + 8 (values) = 18, while the
    // geometry has 15 child vertices — the seam persists.
    let level1 = refiner.level(1);
    assert_eq!(level1.num_fvar_values(0), 18);
    assert_eq!(level1.num_vertices(), 15);

    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; 18];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);

    // Linear-all: face value = centroid of the face's UVs.
    assert_close2(dst[0], [0.225, 0.5]);
    // The two sides of the seam refine to *different* midpoints:
    // A-side seam edge (values 1,2) is fvar edge 1; B-side (7,4) is edge 7.
    assert_close2(dst[2 + 1], [0.45, 0.5]);
    assert_close2(dst[2 + 7], [0.55, 0.5]);
    // Linear-all leaves the original values in place.
    for (k, uv) in uvs.iter().enumerate() {
        assert_close2(dst[10 + k], *uv);
    }
}

#[test]
fn fvar_none_matches_vertex_refinement_of_value_mesh() {
    // FVarLinearInterpolation::None makes a channel behave exactly like
    // vertex data on its value mesh with EdgeOnly boundaries. Refine the
    // seamed mesh's channel and, in parallel, a *geometry* refiner built
    // from the value mesh itself, and compare all values at every level and
    // at the limit.
    let (refiner, uvs) = seamed_two_quads(sdc::FVarLinearInterpolation::None, 2);
    let primvar = PrimvarRefiner::new(&refiner);

    let mesh_verts_per_face = [4usize, 4];
    let mesh_face_verts = [0u32, 1, 2, 3, 4, 5, 6, 7];
    let descriptor = TopologyDescriptor::new(8, &mesh_verts_per_face, &mesh_face_verts);
    let mut value_mesh_refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(), // EdgeOnly boundaries
    )
    .unwrap();
    value_mesh_refiner.refine_uniform(UniformOptions::new(2));
    let value_mesh_primvar = PrimvarRefiner::new(&value_mesh_refiner);

    let mut fvar_src = uvs.clone();
    let mut mesh_src = uvs.clone();
    for level in 1..=2 {
        let n = refiner.level(level).num_fvar_values(0);
        assert_eq!(n, value_mesh_refiner.level(level).num_vertices());

        let mut fvar_dst = vec![[0.0f32; 2]; n];
        primvar.interpolate_face_varying(level, 0, &fvar_src, &mut fvar_dst);
        let mut mesh_dst = vec![[0.0f32; 2]; n];
        value_mesh_primvar.interpolate(level, &mesh_src, &mut mesh_dst);

        for (a, b) in fvar_dst.iter().zip(&mesh_dst) {
            assert_close2(*a, *b);
        }
        fvar_src = fvar_dst;
        mesh_src = mesh_dst;
    }

    // Limit surfaces agree too.
    let mut fvar_limit = fvar_src.clone();
    primvar.limit_face_varying(0, &fvar_src, &mut fvar_limit);
    let mut mesh_limit = mesh_src.clone();
    value_mesh_primvar.limit(&mesh_src, &mut mesh_limit);
    for (a, b) in fvar_limit.iter().zip(&mesh_limit) {
        assert_close2(*a, *b);
    }
}

#[test]
fn fvar_none_on_seamless_channel_matches_vertex_interpolation() {
    // A channel whose values coincide with the vertices (no seams) on a
    // closed mesh must interpolate identically to vertex data under
    // FVarLinearInterpolation::None.
    let verts_per_face = [4usize; 6];
    let face_verts: [u32; 24] = [
        0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4,
    ];
    let positions: [[f32; 3]; 8] = [
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
    ];
    let channels = [FVarChannelDescriptor::new(8, &face_verts)];
    let descriptor =
        TopologyDescriptor::new(8, &verts_per_face, &face_verts).with_fvar_channels(&channels);
    let options =
        sdc::Options::default().with_fvar_linear_interpolation(sdc::FVarLinearInterpolation::None);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    let primvar = PrimvarRefiner::new(&refiner);
    let mut vsrc = positions.to_vec();
    let mut fsrc = positions.to_vec();
    for level in 1..=2 {
        assert_eq!(
            refiner.level(level).num_fvar_values(0),
            refiner.level(level).num_vertices()
        );
        let mut vdst = vec![[0.0f32; 3]; refiner.level(level).num_vertices()];
        primvar.interpolate(level, &vsrc, &mut vdst);
        let mut fdst = vec![[0.0f32; 3]; refiner.level(level).num_fvar_values(0)];
        primvar.interpolate_face_varying(level, 0, &fsrc, &mut fdst);
        for (a, b) in vdst.iter().zip(&fdst) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-6);
            }
        }
        vsrc = vdst;
        fsrc = fdst;
    }
}

#[test]
fn boundaries_mode_pins_all_boundary_values() {
    let (refiner, uvs) = seamed_two_quads(sdc::FVarLinearInterpolation::Boundaries, 2);
    let primvar = PrimvarRefiner::new(&refiner);

    let mut src = uvs.clone();
    for level in 1..=2 {
        let mut dst = vec![[0.0f32; 2]; refiner.level(level).num_fvar_values(0)];
        primvar.interpolate_face_varying(level, 0, &src, &mut dst);
        src = dst;
    }
    // Every value of the seamed channel lies on an fvar boundary, so the
    // whole channel refines linearly: all original corner values persist
    // exactly, and every refined value stays inside the UV hull.
    for uv in &uvs {
        assert!(
            src.iter()
                .any(|p| (p[0] - uv[0]).abs() < 1e-6 && (p[1] - uv[1]).abs() < 1e-6),
            "value {uv:?} not preserved by linear boundaries"
        );
    }
    for p in &src {
        assert!((-1e-6..=1.0 + 1e-6).contains(&p[0]));
        assert!((-1e-6..=1.0 + 1e-6).contains(&p[1]));
    }
}

/// A 3x3-vertex grid of four quads used by the junction and dart tests:
///
/// ```text
///   6 --- 7 --- 8
///   |  F2 |  F3 |
///   3 --- 4 --- 5
///   |  F0 |  F1 |
///   0 --- 1 --- 2
/// ```
const GRID_VERTS_PER_FACE: [usize; 4] = [4; 4];
const GRID_FACE_VERTS: [u32; 16] = [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7];

fn grid_refiner(
    uv_indices: &[u32; 16],
    num_values: usize,
    mode: sdc::FVarLinearInterpolation,
) -> TopologyRefiner {
    let channels = [FVarChannelDescriptor::new(num_values, uv_indices)];
    let descriptor = TopologyDescriptor::new(9, &GRID_VERTS_PER_FACE, &GRID_FACE_VERTS)
        .with_fvar_channels(&channels);
    let options = sdc::Options::default().with_fvar_linear_interpolation(mode);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(1));
    refiner
}

#[test]
fn corners_plus1_pins_junctions() {
    // Three UV islands: F0 alone (values 0-3), F2 alone (values 4-7), and
    // F1+F3 continuous (values 8-13). Three distinct values meet at the
    // center vertex 4 — a junction.
    let uv_indices: [u32; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 11, 10, 12, 13];
    // Value 11 sits at the center for the right island; its seam neighbors
    // along the island boundary are values 8 (below) and 13 (above), spaced
    // non-uniformly so the crease mask visibly moves the value.
    let mut uvs = vec![[0.0f32; 2]; 14];
    for (k, uv) in uvs.iter_mut().enumerate() {
        *uv = [2.0 + k as f32, 5.0]; // arbitrary distinct filler
    }
    uvs[8] = [0.0, 0.0];
    uvs[11] = [0.0, 0.3];
    uvs[13] = [0.0, 1.0];

    // The value mesh has 15 edges, so value children start at 4 + 15 = 19.
    let child_of_11 = 19 + 11;

    // CornersOnly: value 11 spans two faces, so it is *not* a corner; it
    // subdivides as a crease along the island boundary.
    let refiner = grid_refiner(&uv_indices, 14, sdc::FVarLinearInterpolation::CornersOnly);
    assert_eq!(refiner.level(1).num_fvar_values(0), 4 + 15 + 14);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(dst[child_of_11], [0.0, 0.75 * 0.3 + 0.125 * (0.0 + 1.0)]);

    // CornersPlus1: three values meet at vertex 4 — the junction pins them.
    let refiner = grid_refiner(&uv_indices, 14, sdc::FVarLinearInterpolation::CornersPlus1);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(dst[child_of_11], uvs[11]);
}

#[test]
fn corners_plus2_pins_darts() {
    // A seam along edge (1,4) only, terminating at the interior vertex 4:
    // both sides share value 4 at the center (a dart) but use different
    // values (1 vs 9) at boundary vertex 1.
    let uv_indices: [u32; 16] = [0, 1, 4, 3, 9, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7];
    let mut uvs = vec![[0.0f32; 2]; 10];
    for (k, uv) in uvs.iter_mut().enumerate() {
        *uv = [3.0 + k as f32, 7.0]; // arbitrary distinct filler
    }
    uvs[1] = [0.4, 0.0];
    uvs[9] = [0.6, 0.0];
    uvs[4] = [0.5, 0.5];

    // The value mesh has 13 edges, so value children start at 4 + 13 = 17.
    let child_of_4 = 17 + 4;

    // CornersPlus1: the dart value creases along the two sides of the seam.
    let refiner = grid_refiner(&uv_indices, 10, sdc::FVarLinearInterpolation::CornersPlus1);
    assert_eq!(refiner.level(1).num_fvar_values(0), 4 + 13 + 10);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(
        dst[child_of_4],
        [
            0.75 * 0.5 + 0.125 * (0.4 + 0.6),
            0.75 * 0.5 + 0.125 * (0.0 + 0.0),
        ],
    );

    // CornersPlus2: the dart is pinned.
    let refiner = grid_refiner(&uv_indices, 10, sdc::FVarLinearInterpolation::CornersPlus2);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(dst[child_of_4], uvs[4]);
}

#[test]
fn fvar_channel_validation() {
    use opensubdiv_rs::far::Error;

    let verts_per_face = [4usize];
    let face_verts = [0u32, 1, 2, 3];

    // Wrong number of value indices.
    let bad_len = [0u32, 1, 2];
    let channels = [FVarChannelDescriptor::new(4, &bad_len)];
    let descriptor =
        TopologyDescriptor::new(4, &verts_per_face, &face_verts).with_fvar_channels(&channels);
    assert!(matches!(
        TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default()
        ),
        Err(Error::FVarValueCountMismatch { channel: 0, .. })
    ));

    // Out-of-range value index.
    let bad_index = [0u32, 1, 2, 7];
    let channels = [FVarChannelDescriptor::new(4, &bad_index)];
    let descriptor =
        TopologyDescriptor::new(4, &verts_per_face, &face_verts).with_fvar_channels(&channels);
    assert!(matches!(
        TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default()
        ),
        Err(Error::FVarValueIndexOutOfRange {
            channel: 0,
            index: 7
        })
    ));
}

/// UV layout of the L-shaped island used by the concave-corner tests: faces
/// F0, F1 and F2 share one island (values 0-7, matching vertex ids), while F3
/// is an island of its own (values 8-11). Value 4 at the center vertex thus
/// spans three faces — a reflex (concave) corner of the L — while value 8
/// spans F3 alone.
const L_ISLAND_UV_INDICES: [u32; 16] = [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 8, 9, 10, 11];

fn l_island_uvs() -> Vec<UV> {
    vec![
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1.0, 0.4],
        [0.0, 1.0],
        [0.6, 1.0],
        [0.5, 0.5],
        [1.0, 0.5],
        [1.0, 1.0],
        [0.5, 1.0],
    ]
}

#[test]
fn corners_plus2_pins_concave_corners() {
    let uvs = l_island_uvs();

    // The value mesh has 14 edges, so value children start at 4 + 14 = 18.
    let child_of_4 = 18 + 4;

    // CornersPlus1: value 4 is a smooth boundary value of the island; it
    // creases along the island boundary, between values 5 and 7.
    let refiner = grid_refiner(
        &L_ISLAND_UV_INDICES,
        12,
        sdc::FVarLinearInterpolation::CornersPlus1,
    );
    assert_eq!(refiner.level(1).num_fvar_values(0), 4 + 14 + 12);
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(
        dst[child_of_4],
        [
            0.75 * 0.5 + 0.125 * (1.0 + 0.6),
            0.75 * 0.5 + 0.125 * (0.4 + 1.0),
        ],
    );

    // CornersPlus2: value 8 is a corner of its island, so the concave value
    // 4 on the other side of the seam is pinned at its authored position.
    let refiner = grid_refiner(
        &L_ISLAND_UV_INDICES,
        12,
        sdc::FVarLinearInterpolation::CornersPlus2,
    );
    let primvar = PrimvarRefiner::new(&refiner);
    let mut dst = vec![[0.0f32; 2]; refiner.level(1).num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut dst);
    assert_close2(dst[child_of_4], uvs[4]);
}

/// Per-face corner UVs of the L-shaped island refined once under
/// `CornersPlus2`, as computed by OpenSubdiv 3.7
/// (`Far::PrimvarRefiner::InterpolateFaceVarying`).
const L_ISLAND_PLUS2_REFINED: [[UV; 4]; 16] = [
    [[0.0, 0.0], [0.25, 0.0], [0.25, 0.25], [0.0, 0.25]],
    [[0.25, 0.0], [0.5, 0.0], [0.5, 0.24375], [0.25, 0.25]],
    [[0.25, 0.25], [0.5, 0.24375], [0.5, 0.5], [0.25625, 0.5]],
    [[0.0, 0.25], [0.25, 0.25], [0.25625, 0.5], [0.0, 0.5]],
    [[0.5, 0.0], [0.75, 0.0], [0.75, 0.225], [0.5, 0.24375]],
    [[0.75, 0.0], [1.0, 0.0], [1.0, 0.2], [0.75, 0.225]],
    [[0.75, 0.225], [1.0, 0.2], [1.0, 0.4], [0.75, 0.45]],
    [[0.5, 0.24375], [0.75, 0.225], [0.75, 0.45], [0.5, 0.5]],
    [[0.0, 0.5], [0.25625, 0.5], [0.275, 0.75], [0.0, 0.75]],
    [[0.25625, 0.5], [0.5, 0.5], [0.55, 0.75], [0.275, 0.75]],
    [[0.275, 0.75], [0.55, 0.75], [0.6, 1.0], [0.3, 1.0]],
    [[0.0, 0.75], [0.275, 0.75], [0.3, 1.0], [0.0, 1.0]],
    [[0.5, 0.5], [0.75, 0.5], [0.75, 0.75], [0.5, 0.75]],
    [[0.75, 0.5], [1.0, 0.5], [1.0, 0.75], [0.75, 0.75]],
    [[0.75, 0.75], [1.0, 0.75], [1.0, 1.0], [0.75, 1.0]],
    [[0.5, 0.75], [0.75, 0.75], [0.75, 1.0], [0.5, 1.0]],
];

/// Limit values of the same level-1 corners, as computed by OpenSubdiv 3.7
/// (`Far::PrimvarRefiner::LimitFaceVarying`, printed to six decimals).
const L_ISLAND_PLUS2_LIMIT: [[UV; 4]; 16] = [
    [[0.0, 0.0], [0.25, 0.0], [0.250694, 0.249306], [0.0, 0.25]],
    [
        [0.25, 0.0],
        [0.5, 0.0],
        [0.500174, 0.243056],
        [0.250694, 0.249306],
    ],
    [
        [0.250694, 0.249306],
        [0.500174, 0.243056],
        [0.5, 0.5],
        [0.256944, 0.499826],
    ],
    [
        [0.0, 0.25],
        [0.250694, 0.249306],
        [0.256944, 0.499826],
        [0.0, 0.5],
    ],
    [
        [0.5, 0.0],
        [0.75, 0.0],
        [0.75, 0.224306],
        [0.500174, 0.243056],
    ],
    [[0.75, 0.0], [1.0, 0.0], [1.0, 0.2], [0.75, 0.224306]],
    [[0.75, 0.224306], [1.0, 0.2], [1.0, 0.4], [0.75, 0.45]],
    [
        [0.500174, 0.243056],
        [0.75, 0.224306],
        [0.75, 0.45],
        [0.5, 0.5],
    ],
    [
        [0.0, 0.5],
        [0.256944, 0.499826],
        [0.275694, 0.75],
        [0.0, 0.75],
    ],
    [
        [0.256944, 0.499826],
        [0.5, 0.5],
        [0.55, 0.75],
        [0.275694, 0.75],
    ],
    [[0.275694, 0.75], [0.55, 0.75], [0.6, 1.0], [0.3, 1.0]],
    [[0.0, 0.75], [0.275694, 0.75], [0.3, 1.0], [0.0, 1.0]],
    [[0.5, 0.5], [0.75, 0.5], [0.75, 0.75], [0.5, 0.75]],
    [[0.75, 0.5], [1.0, 0.5], [1.0, 0.75], [0.75, 0.75]],
    [[0.75, 0.75], [1.0, 0.75], [1.0, 1.0], [0.75, 1.0]],
    [[0.5, 0.75], [0.75, 0.75], [0.75, 1.0], [0.5, 1.0]],
];

#[test]
fn corners_plus2_matches_opensubdiv_on_concave_island() {
    let uvs = l_island_uvs();
    let refiner = grid_refiner(
        &L_ISLAND_UV_INDICES,
        12,
        sdc::FVarLinearInterpolation::CornersPlus2,
    );
    let level1 = refiner.level(1);
    assert_eq!(level1.num_faces(), 16);
    assert_eq!(level1.num_fvar_values(0), 30);

    let primvar = PrimvarRefiner::new(&refiner);
    let mut refined = vec![[0.0f32; 2]; level1.num_fvar_values(0)];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut refined);
    let mut limit = vec![[0.0f32; 2]; level1.num_fvar_values(0)];
    primvar.limit_face_varying(0, &refined, &mut limit);

    let assert_golden = |actual: UV, expected: UV, what: &str, f: usize, c: usize| {
        for (a, e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() < 1e-5,
                "{what}: face {f} corner {c}: expected {expected:?}, got {actual:?}"
            );
        }
    };
    // Child face `f` is child `f % 4` of base face `f / 4`, at that corner.
    // This port lists a child quad's corners as [corner, leading edge,
    // center, trailing edge], whereas OpenSubdiv keeps the parent corner at
    // the same local index `f % 4` (`QuadRefinement::
    // populateFaceVerticesFromParentFaces`): corner `c` here is corner
    // `(c + f % 4) % 4` of the same face in OpenSubdiv's tables.
    for f in 0..16 {
        let values = level1.face_fvar_values(f, 0);
        assert_eq!(values.len(), 4);
        for (c, &val) in values.iter().enumerate() {
            let osd_c = (c + f % 4) % 4;
            assert_golden(
                refined[val as usize],
                L_ISLAND_PLUS2_REFINED[f][osd_c],
                "refined",
                f,
                c,
            );
            assert_golden(
                limit[val as usize],
                L_ISLAND_PLUS2_LIMIT[f][osd_c],
                "limit",
                f,
                c,
            );
        }
    }
}

/// A 4x4-vertex grid of nine quads whose bottom-left eight form one UV
/// island and whose top-right quad (vertices 10, 11, 15, 14) is an island of
/// its own. The island's values are numbered in vertex order skipping the
/// vertices 5 and 15, except that the interior vertex 5 *reuses* the value
/// of vertex 10 — the concave corner of the island — as a deduplicated UV
/// buffer would. The value index is used at two vertices: as a pinned
/// concave corner at vertex 10 and as a smooth interior value at vertex 5.
///
/// ```text
///   12 -- 13 -- 14 -- 15
///   |  6  |  7  |  8  |
///    8 --- 9 -- 10 -- 11
///   |  3  |  4  |  5  |
///    4 --- 5 --- 6 --- 7
///   |  0  |  1  |  2  |
///    0 --- 1 --- 2 --- 3
/// ```
fn reused_value_grid() -> (Vec<u32>, Vec<u32>, Vec<UV>) {
    let mut id_of = [u32::MAX; 16];
    let mut next = 0;
    for (v, id) in id_of.iter_mut().enumerate() {
        if v != 5 && v != 15 {
            *id = next;
            next += 1;
        }
    }
    id_of[5] = id_of[10];

    let mut face_verts = Vec::with_capacity(36);
    let mut uv_indices = Vec::with_capacity(36);
    for y in 0..3u32 {
        for x in 0..3u32 {
            let q = y * 3 + x;
            let v0 = y * 4 + x;
            let corners = [v0, v0 + 1, v0 + 5, v0 + 4];
            face_verts.extend_from_slice(&corners);
            for (i, &c) in corners.iter().enumerate() {
                uv_indices.push(if q == 8 {
                    14 + i as u32
                } else {
                    id_of[c as usize]
                });
            }
        }
    }

    let mut uvs = vec![[0.0f32; 2]; 18];
    for v in 0..16 {
        if v != 5 && v != 15 {
            uvs[id_of[v] as usize] = [(v % 4) as f32 / 3.0, (v / 4) as f32 / 3.0];
        }
    }
    uvs[14] = [0.7, 0.7];
    uvs[15] = [1.0, 0.7];
    uvs[16] = [1.0, 1.0];
    uvs[17] = [0.7, 1.0];
    (face_verts, uv_indices, uvs)
}

/// Per-face corner UVs of the reused-value grid refined once under
/// `CornersPlus2`, as computed by OpenSubdiv 3.7
/// (`Far::PrimvarRefiner::InterpolateFaceVarying`).
const REUSED_VALUE_PLUS2_REFINED: [[UV; 4]; 36] = [
    [[0.0, 0.0], [0.166667, 0.0], [0.25, 0.25], [0.0, 0.166667]],
    [
        [0.166667, 0.0],
        [0.333333, 0.0],
        [0.458333, 0.291667],
        [0.25, 0.25],
    ],
    [
        [0.25, 0.25],
        [0.458333, 0.291667],
        [0.520833, 0.520833],
        [0.291667, 0.458333],
    ],
    [
        [0.0, 0.166667],
        [0.25, 0.25],
        [0.291667, 0.458333],
        [0.0, 0.333333],
    ],
    [
        [0.333333, 0.0],
        [0.5, 0.0],
        [0.583333, 0.25],
        [0.458333, 0.291667],
    ],
    [
        [0.5, 0.0],
        [0.666667, 0.0],
        [0.6875, 0.1875],
        [0.583333, 0.25],
    ],
    [
        [0.583333, 0.25],
        [0.6875, 0.1875],
        [0.697917, 0.364583],
        [0.625, 0.458333],
    ],
    [
        [0.458333, 0.291667],
        [0.583333, 0.25],
        [0.625, 0.458333],
        [0.520833, 0.520833],
    ],
    [
        [0.666667, 0.0],
        [0.833333, 0.0],
        [0.833333, 0.166667],
        [0.6875, 0.1875],
    ],
    [
        [0.833333, 0.0],
        [1.0, 0.0],
        [1.0, 0.166667],
        [0.833333, 0.166667],
    ],
    [
        [0.833333, 0.166667],
        [1.0, 0.166667],
        [1.0, 0.333333],
        [0.833333, 0.333333],
    ],
    [
        [0.6875, 0.1875],
        [0.833333, 0.166667],
        [0.833333, 0.333333],
        [0.697917, 0.364583],
    ],
    [
        [0.0, 0.333333],
        [0.291667, 0.458333],
        [0.25, 0.583333],
        [0.0, 0.5],
    ],
    [
        [0.291667, 0.458333],
        [0.520833, 0.520833],
        [0.458333, 0.625],
        [0.25, 0.583333],
    ],
    [
        [0.25, 0.583333],
        [0.458333, 0.625],
        [0.364583, 0.697917],
        [0.1875, 0.6875],
    ],
    [
        [0.0, 0.5],
        [0.25, 0.583333],
        [0.1875, 0.6875],
        [0.0, 0.666667],
    ],
    [
        [0.520833, 0.520833],
        [0.625, 0.458333],
        [0.583333, 0.583333],
        [0.458333, 0.625],
    ],
    [
        [0.625, 0.458333],
        [0.697917, 0.364583],
        [0.6875, 0.520833],
        [0.583333, 0.583333],
    ],
    [
        [0.583333, 0.583333],
        [0.6875, 0.520833],
        [0.666667, 0.666667],
        [0.520833, 0.6875],
    ],
    [
        [0.458333, 0.625],
        [0.583333, 0.583333],
        [0.520833, 0.6875],
        [0.364583, 0.697917],
    ],
    [
        [0.697917, 0.364583],
        [0.833333, 0.333333],
        [0.833333, 0.5],
        [0.6875, 0.520833],
    ],
    [
        [0.833333, 0.333333],
        [1.0, 0.333333],
        [1.0, 0.5],
        [0.833333, 0.5],
    ],
    [
        [0.833333, 0.5],
        [1.0, 0.5],
        [1.0, 0.666667],
        [0.833333, 0.666667],
    ],
    [
        [0.6875, 0.520833],
        [0.833333, 0.5],
        [0.833333, 0.666667],
        [0.666667, 0.666667],
    ],
    [
        [0.0, 0.666667],
        [0.1875, 0.6875],
        [0.166667, 0.833333],
        [0.0, 0.833333],
    ],
    [
        [0.1875, 0.6875],
        [0.364583, 0.697917],
        [0.333333, 0.833333],
        [0.166667, 0.833333],
    ],
    [
        [0.166667, 0.833333],
        [0.333333, 0.833333],
        [0.333333, 1.0],
        [0.166667, 1.0],
    ],
    [
        [0.0, 0.833333],
        [0.166667, 0.833333],
        [0.166667, 1.0],
        [0.0, 1.0],
    ],
    [
        [0.364583, 0.697917],
        [0.520833, 0.6875],
        [0.5, 0.833333],
        [0.333333, 0.833333],
    ],
    [
        [0.520833, 0.6875],
        [0.666667, 0.666667],
        [0.666667, 0.833333],
        [0.5, 0.833333],
    ],
    [
        [0.5, 0.833333],
        [0.666667, 0.833333],
        [0.666667, 1.0],
        [0.5, 1.0],
    ],
    [
        [0.333333, 0.833333],
        [0.5, 0.833333],
        [0.5, 1.0],
        [0.333333, 1.0],
    ],
    [[0.7, 0.7], [0.85, 0.7], [0.85, 0.85], [0.7, 0.85]],
    [[0.85, 0.7], [1.0, 0.7], [1.0, 0.85], [0.85, 0.85]],
    [[0.85, 0.85], [1.0, 0.85], [1.0, 1.0], [0.85, 1.0]],
    [[0.7, 0.85], [0.85, 0.85], [0.85, 1.0], [0.7, 1.0]],
];

#[test]
fn reused_value_index_is_independent_at_each_vertex() {
    let (face_verts, uv_indices, uvs) = reused_value_grid();
    let verts_per_face = [4usize; 9];
    let channels = [FVarChannelDescriptor::new(18, &uv_indices)];
    let descriptor =
        TopologyDescriptor::new(16, &verts_per_face, &face_verts).with_fvar_channels(&channels);
    let options = sdc::Options::default()
        .with_fvar_linear_interpolation(sdc::FVarLinearInterpolation::CornersPlus2);
    let mut refiner =
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap();
    refiner.refine_uniform(UniformOptions::new(2));

    // The base level exposes the channel exactly as described.
    let level0 = refiner.level(0);
    assert_eq!(level0.num_fvar_values(0), 18);
    for f in 0..9 {
        assert_eq!(level0.face_fvar_values(f, 0), &uv_indices[f * 4..f * 4 + 4]);
    }

    // Refined levels count the reused index once per vertex it is used at,
    // as OpenSubdiv does (54 and 178 values).
    let level1 = refiner.level(1);
    assert_eq!(level1.num_fvar_values(0), 54);
    assert_eq!(refiner.level(2).num_fvar_values(0), 178);

    let primvar = PrimvarRefiner::new(&refiner);
    let mut refined = vec![[0.0f32; 2]; 54];
    primvar.interpolate_face_varying(1, 0, &uvs, &mut refined);

    // Same child-corner rotation as `corners_plus2_matches_opensubdiv_on_concave_island`.
    for f in 0..36 {
        let values = level1.face_fvar_values(f, 0);
        for (c, &val) in values.iter().enumerate() {
            let expected = REUSED_VALUE_PLUS2_REFINED[f][(c + f % 4) % 4];
            let actual = refined[val as usize];
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() < 1e-5,
                    "face {f} corner {c}: expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    // The two occurrences of the shared index refine differently: pinned at
    // the concave corner (vertex 10 is corner 2 of quad 4, so its child
    // value is corner 0 of child face 4 * 4 + 2), smoothed at the interior
    // vertex 5 (corner 2 of quad 0: corner 0 of child face 2).
    let pinned = refined[level1.face_fvar_values(18, 0)[0] as usize];
    let smoothed = refined[level1.face_fvar_values(2, 0)[0] as usize];
    assert_close2(pinned, uvs[9]);
    assert!((smoothed[0] - uvs[9][0]).abs() > 0.1);

    // Level 2 interpolates from the split level-1 values without any gather.
    let mut refined2 = vec![[0.0f32; 2]; 178];
    primvar.interpolate_face_varying(2, 0, &refined, &mut refined2);
    let level2 = refiner.level(2);
    let pinned2 = refined2[level2.face_fvar_values(18 * 4, 0)[0] as usize];
    assert_close2(pinned2, uvs[9]);
}
