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
