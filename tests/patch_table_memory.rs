//! Patch-table memory: the live heap bytes a `PatchTable` holds per patch,
//! measured by a counting global allocator (this test binary only).

use opensubdiv_rs::far::{
    AdaptiveOptions, FVarChannelDescriptor, PatchTableFactory, PatchTableOptions, PatchType,
    TopologyDescriptor, TopologyRefinerFactory,
};
use opensubdiv_rs::sdc;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        LIVE.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Held by each test while it measures, so that tests running in parallel
/// do not count each other's allocations.
static MEASURING: Mutex<()> = Mutex::new(());

/// A regular quad cage keeps its patch table near OpenSubdiv's layout: 16
/// control-vertex indices, an 8-byte parameterization and the face per
/// regular patch, plus the per-level face maps (179 bytes per patch in
/// 0.3.0).
#[test]
fn regular_patches_cost_under_100_bytes() {
    let _measuring = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let n = 200u32;
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let v = j * (n + 1) + i;
            verts_per_face.push(4);
            face_verts.extend_from_slice(&[v, v + 1, v + n + 2, v + n + 1]);
        }
    }
    let num_verts = ((n + 1) * (n + 1)) as usize;
    for isolation in 1..=3 {
        let descriptor = TopologyDescriptor::new(num_verts, &verts_per_face, &face_verts);
        let mut refiner = TopologyRefinerFactory::create(
            descriptor,
            sdc::SchemeType::Catmark,
            sdc::Options::default(),
        )
        .unwrap();
        refiner.refine_adaptive(AdaptiveOptions::new(isolation));

        let before = LIVE.load(Ordering::Relaxed);
        let table = PatchTableFactory::create(&refiner).unwrap();
        let bytes = LIVE.load(Ordering::Relaxed) - before;

        let patches = table.num_patches();
        let regular = (0..patches)
            .filter(|&p| table.patch_type(p) == PatchType::Regular)
            .count();
        assert!(
            regular * 100 >= patches * 99,
            "{regular} of {patches} regular"
        );
        let per_patch = bytes as f64 / patches as f64;
        println!("isolation {isolation}: {patches} patches, {bytes} bytes, {per_patch:.1} B/patch");
        assert!(per_patch < 100.0, "{per_patch:.1} bytes per patch");
    }
}

/// A face-varying channel costs its patches' control values and nothing per
/// patch besides when a single patch type covers it: 16 indices per regular
/// patch, 4 per linear one.
#[test]
fn fvar_patches_cost_their_control_values() {
    let _measuring = MEASURING.lock().unwrap_or_else(|e| e.into_inner());
    let n = 200u32;
    let mut verts_per_face = Vec::new();
    let mut face_verts = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let v = j * (n + 1) + i;
            verts_per_face.push(4);
            face_verts.extend_from_slice(&[v, v + 1, v + n + 2, v + n + 1]);
        }
    }
    let num_verts = ((n + 1) * (n + 1)) as usize;
    let channels = [FVarChannelDescriptor::new(num_verts, &face_verts)];
    let descriptor = TopologyDescriptor::new(num_verts, &verts_per_face, &face_verts)
        .with_fvar_channels(&channels);
    let mut refiner = TopologyRefinerFactory::create(
        descriptor,
        sdc::SchemeType::Catmark,
        sdc::Options::default(),
    )
    .unwrap();
    refiner.refine_adaptive(AdaptiveOptions::new(2));

    let measure = |options: &PatchTableOptions| {
        let before = LIVE.load(Ordering::Relaxed);
        let table = PatchTableFactory::create_with_options(&refiner, options).unwrap();
        (LIVE.load(Ordering::Relaxed) - before, table)
    };
    let (vertex_only, table) = measure(&PatchTableOptions::new());
    let patches = table.num_patches();
    for (legacy_linear, budget) in [(false, 66.0), (true, 18.0)] {
        let options = PatchTableOptions::new()
            .with_fvar_tables(true)
            .with_fvar_legacy_linear_patches(legacy_linear);
        let (bytes, table) = measure(&options);
        let per_patch = (bytes - vertex_only) as f64 / patches as f64;
        let regular = (0..patches)
            .filter(|&p| table.fvar_patch_type(p, 0) == PatchType::Regular)
            .count();
        println!(
            "legacy linear {legacy_linear}: {regular} of {patches} regular, {per_patch:.1} B/patch"
        );
        assert!(
            per_patch < budget,
            "{per_patch:.1} face-varying bytes per patch"
        );
    }
}
