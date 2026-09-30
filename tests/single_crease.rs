//! Single-crease patches (`useSingleCreasePatch`): regular faces bounded by
//! a semi-sharp crease are patched without isolating the crease, and the
//! patches evaluate the exact limit surface.

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

fn patch_controls(refiner: &TopologyRefiner, base: &[P3]) -> Vec<P3> {
    let mut all = base.to_vec();
    for level in PrimvarRefiner::new(refiner).interpolate_all(base) {
        all.extend(level);
    }
    all
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

/// An `n`x`n` quad grid: face `j * n + i` spans vertices `(i, j)` to
/// `(i + 1, j + 1)`, vertex `(i, j)` being `j * (n + 1) + i`.
struct Grid {
    n: u32,
    verts_per_face: Vec<usize>,
    face_verts: Vec<u32>,
    positions: Vec<P3>,
    creases: Vec<[u32; 2]>,
    weights: Vec<f32>,
}

impl Grid {
    fn new(n: u32) -> Self {
        let mut verts_per_face = Vec::new();
        let mut face_verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v = j * (n + 1) + i;
                verts_per_face.push(4);
                face_verts.extend_from_slice(&[v, v + 1, v + n + 2, v + n + 1]);
            }
        }
        let mut positions = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let z = ((i * i * 3 + j * 5 + i * j) % 7) as f32 * 0.15;
                positions.push([i as f32, j as f32, z]);
            }
        }
        Self {
            n,
            verts_per_face,
            face_verts,
            positions,
            creases: Vec::new(),
            weights: Vec::new(),
        }
    }

    fn vertex(&self, i: u32, j: u32) -> u32 {
        j * (self.n + 1) + i
    }

    /// Crease the vertical grid line `x = i` from `y = j0` to `y = j1`.
    fn crease_vertical(mut self, i: u32, j0: u32, j1: u32, sharpness: &[f32]) -> Self {
        for (k, j) in (j0..j1).enumerate() {
            self.creases
                .push([self.vertex(i, j), self.vertex(i, j + 1)]);
            self.weights.push(sharpness[k.min(sharpness.len() - 1)]);
        }
        self
    }

    /// Crease the horizontal grid line `y = j` from `x = i0` to `x = i1`.
    fn crease_horizontal(mut self, j: u32, i0: u32, i1: u32, sharpness: &[f32]) -> Self {
        for (k, i) in (i0..i1).enumerate() {
            self.creases
                .push([self.vertex(i, j), self.vertex(i + 1, j)]);
            self.weights.push(sharpness[k.min(sharpness.len() - 1)]);
        }
        self
    }

    fn refiner(&self, options: sdc::Options) -> TopologyRefiner {
        let descriptor =
            TopologyDescriptor::new(self.positions.len(), &self.verts_per_face, &self.face_verts)
                .with_creases(&self.creases, &self.weights);
        TopologyRefinerFactory::create(descriptor, sdc::SchemeType::Catmark, options).unwrap()
    }

    fn num_faces(&self) -> usize {
        (self.n * self.n) as usize
    }
}

fn options(creasing: sdc::CreasingMethod) -> sdc::Options {
    sdc::Options::default()
        .with_vtx_boundary_interpolation(sdc::VtxBoundaryInterpolation::EdgeAndCorner)
        .with_creasing_method(creasing)
}

/// A uniformly refined reference deep enough for every crease to have
/// decayed: its patch table consists of exact B-spline patches only.
fn exact_reference(grid: &Grid, options: sdc::Options, levels: usize) -> (PatchTable, Vec<P3>) {
    let mut refiner = grid.refiner(options);
    refiner.refine_uniform(UniformOptions::new(levels));
    let table = PatchTableFactory::create(&refiner).unwrap();
    for p in 0..table.num_patches() {
        assert_eq!(table.patch_type(p), PatchType::Regular);
        assert_eq!(table.single_crease_sharpness(p), 0.0);
    }
    let controls = patch_controls(&refiner, &grid.positions);
    (table, controls)
}

/// Require `table` to evaluate the same limit surface — positions and
/// derivatives — as `reference` everywhere.
fn assert_matches_reference(
    table: &PatchTable,
    controls: &[P3],
    reference: &(PatchTable, Vec<P3>),
    num_ptex: usize,
) {
    let map = PatchMap::new(table);
    let reference_map = PatchMap::new(&reference.0);
    let mut locations = samples(24);
    locations.extend_from_slice(&[(0.5, 0.5), (0.0, 0.0), (1.0, 0.5), (0.5, 1.0), (0.01, 0.99)]);
    for ptex in 0..num_ptex {
        for &(u, v) in &locations {
            let p = map.find_patch(ptex, u, v).unwrap();
            let r = reference_map.find_patch(ptex, u, v).unwrap();
            let (q, du, dv) = table.evaluate(p, u, v, controls);
            let (rq, rdu, rdv) = reference.0.evaluate(r, u, v, &reference.1);
            assert_close(q, rq, 2e-5);
            assert_close(du, rdu, 2e-4);
            assert_close(dv, rdv, 2e-4);
        }
    }
}

fn single_crease_patches(table: &PatchTable) -> Vec<usize> {
    (0..table.num_patches())
        .filter(|&p| table.single_crease_sharpness(p) > 0.0)
        .collect()
}

#[test]
fn semi_sharp_crease_faces_become_single_crease_patches() {
    // The issue's scenario: a 4x4 grid with one interior edge run at
    // sharpness 1.5, from boundary to boundary.
    let grid = Grid::new(4).crease_vertical(2, 0, 4, &[1.5]);
    let options = options(sdc::CreasingMethod::Uniform);

    let mut refiner = grid.refiner(options);
    refiner.refine_adaptive(AdaptiveOptions::new(4).with_single_crease_patch(true));
    let table = PatchTableFactory::create(&refiner).unwrap();
    let controls = patch_controls(&refiner, &grid.positions);
    let map = PatchMap::new(&table);

    // The interior faces beside the crease are single-crease patches of the
    // base level, carrying the authored sharpness.
    for face in [5usize, 6, 9, 10] {
        let patch = map.find_patch(face, 0.5, 0.5).unwrap();
        assert_eq!(table.patch_type(patch), PatchType::Regular);
        assert_eq!(table.patch_param(patch).depth, 0);
        assert_eq!(table.single_crease_sharpness(patch), 1.5);
    }
    // Along the rest of the crease — the faces touching its boundary ends —
    // the decayed crease is patched at depth 1 wherever it is regular.
    let shallow: Vec<_> = single_crease_patches(&table)
        .into_iter()
        .filter(|&p| table.patch_param(p).depth == 1)
        .collect();
    assert!(!shallow.is_empty());
    for p in shallow {
        assert_eq!(table.single_crease_sharpness(p), 0.5);
    }

    // At each face centre the patch matches the limit of uniform refinement
    // to a deep level.
    let levels = 5;
    let mut uniform = grid.refiner(options);
    uniform.refine_uniform(UniformOptions::new(levels));
    let mut refined = grid.positions.clone();
    for level in 1..=levels {
        let mut next = vec![[0.0f32; 3]; uniform.level(level).num_vertices()];
        PrimvarRefiner::new(&uniform).interpolate(level, &refined, &mut next);
        refined = next;
    }
    let mut limits = refined.clone();
    PrimvarRefiner::new(&uniform).limit(&refined, &mut limits);
    for face in 0..grid.num_faces() {
        let mut centre = uniform.refinement(1).face_child_vertex(face) as usize;
        for level in 2..=levels {
            centre = uniform.refinement(level).vertex_child_vertex(centre) as usize;
        }
        let patch = map.find_patch(face, 0.5, 0.5).unwrap();
        let (point, _, _) = table.evaluate(patch, 0.5, 0.5, &controls);
        assert_close(point, limits[centre], 1e-5);
    }

    // And everywhere else, derivatives included.
    let reference = exact_reference(&grid, options, 3);
    assert_matches_reference(&table, &controls, &reference, grid.num_faces());

    // Without single-crease patches the crease is isolated much deeper.
    let mut isolated = grid.refiner(options);
    isolated.refine_adaptive(AdaptiveOptions::new(4));
    let isolated_table = PatchTableFactory::create(&isolated).unwrap();
    assert!(single_crease_patches(&isolated_table).is_empty());
    assert!(table.num_patches() < isolated_table.num_patches());
    assert!(refiner.num_faces_total() < isolated.num_faces_total());
}

#[test]
fn single_crease_patches_are_exact_for_any_sharpness() {
    for &creasing in &[sdc::CreasingMethod::Uniform, sdc::CreasingMethod::Chaikin] {
        for &sharpness in &[0.4f32, 1.0, 1.5, 2.7, 3.0, 4.25] {
            let levels = sharpness.ceil() as usize + 1;
            let options = options(creasing);
            // The crease ends at darts inside the mesh: under Chaikin a
            // crease meeting the (infinitely sharp) boundary never decays
            // there, and no uniform reference would be free of it.
            for grid in [
                Grid::new(6).crease_vertical(3, 1, 5, &[sharpness]),
                Grid::new(6).crease_horizontal(2, 1, 5, &[sharpness]),
            ] {
                let mut refiner = grid.refiner(options);
                refiner.refine_adaptive(AdaptiveOptions::new(6).with_single_crease_patch(true));
                let table = PatchTableFactory::create(&refiner).unwrap();

                // The four faces flanking the crease away from its ends are
                // single-crease patches of the base level.
                let base: Vec<_> = single_crease_patches(&table)
                    .into_iter()
                    .filter(|&p| table.patch_param(p).depth == 0)
                    .collect();
                assert_eq!(base.len(), 4, "sharpness {sharpness}");
                for &p in &base {
                    assert_eq!(table.single_crease_sharpness(p), sharpness);
                }

                let controls = patch_controls(&refiner, &grid.positions);
                let reference = exact_reference(&grid, options, levels);
                assert_matches_reference(&table, &controls, &reference, grid.num_faces());
            }
        }
    }
}

#[test]
fn crease_ends_and_varying_sharpness_are_isolated() {
    // A crease ending inside the mesh (darts at both ends) whose sharpness
    // varies along it: only faces flanking a uniform stretch of the crease
    // are single-crease patches, the rest is isolated — and under both
    // creasing methods the surface still matches uniform refinement.
    let grid = Grid::new(8).crease_vertical(3, 1, 7, &[1.0, 2.0, 2.0, 2.0, 2.0, 1.25]);
    for &creasing in &[sdc::CreasingMethod::Uniform, sdc::CreasingMethod::Chaikin] {
        let options = options(creasing);
        let mut refiner = grid.refiner(options);
        refiner.refine_adaptive(AdaptiveOptions::new(5).with_single_crease_patch(true));
        let table = PatchTableFactory::create(&refiner).unwrap();

        let base: Vec<_> = single_crease_patches(&table)
            .into_iter()
            .filter(|&p| table.patch_param(p).depth == 0)
            .collect();
        // Only rows 3 and 4 have both crease corners inside the stretch of
        // sharpness 2 (the edges of rows 2..=5), on both sides of the crease.
        assert_eq!(base.len(), 4);
        for &p in &base {
            assert_eq!(table.single_crease_sharpness(p), 2.0);
            let row = table.patch_face(p) as u32 / 8;
            assert!(row == 3 || row == 4);
        }

        let controls = patch_controls(&refiner, &grid.positions);
        let reference = exact_reference(&grid, options, 4);
        assert_matches_reference(&table, &controls, &reference, grid.num_faces());
    }
}

#[test]
fn single_crease_patches_are_off_by_default() {
    let grid = Grid::new(4).crease_vertical(2, 0, 4, &[1.5]);
    let options = options(sdc::CreasingMethod::Uniform);
    let mut refiner = grid.refiner(options);
    refiner.refine_adaptive(AdaptiveOptions::new(3));
    assert!(!AdaptiveOptions::new(3).use_single_crease_patch);
    let table = PatchTableFactory::create(&refiner).unwrap();
    assert!(single_crease_patches(&table).is_empty());

    // Uniformly refined tables never carry single-crease patches either.
    let mut uniform = grid.refiner(options);
    uniform.refine_uniform(UniformOptions::new(1));
    let table = PatchTableFactory::create(&uniform).unwrap();
    assert!(single_crease_patches(&table).is_empty());
}
