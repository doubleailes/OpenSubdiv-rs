//! A patch table flattened into the buffers a GPU evaluator reads (port of
//! `opensubdiv/osd/cpuPatchTable.cpp`, the common source of OpenSubdiv's
//! per-API patch tables, together with the end-cap local points of
//! `Far::PatchTable`).

use super::patch_basis::evaluate_patch_basis;
use super::types::{patch_descriptor, PatchArray, PatchCoord, PatchParam};
use crate::far::patch_table::PatchRef;
use crate::far::{PatchTable, Primvar, StencilTable};
use crate::vtr::TopologyError;
use crate::{Index, INDEX_INVALID};

/// A [`PatchTable`] as flat, GPU-ready buffers (`Osd::CpuPatchTable` and
/// its `GL`/`D3D11`/`Metal` siblings, which upload these same arrays).
///
/// It holds three buffers and one stencil table:
///
/// * the [patch arrays](Self::patch_arrays): runs of patches of one type;
/// * the [index buffer](Self::patch_index_buffer): every patch's control
///   points, [`PatchArray::stride`] per patch;
/// * the [patch-param buffer](Self::patch_param_buffer): one packed
///   [`PatchParam`] per patch;
/// * the [local-point stencils](Self::local_point_stencils): the derived
///   control points of the Gregory end caps, which the index buffer refers
///   to like any other control point.
///
/// Indices refer to one buffer of [`num_control_values`](Self::num_control_values)
/// control values — every refinement level's vertices, base level first,
/// as [`PatchTable::evaluate`] expects — followed by the
/// [`num_local_points`](Self::num_local_points) local points. Fill it on
/// the GPU with two stencil passes (the
/// [`for_patch_controls`](crate::far::StencilTableOptions::for_patch_controls)
/// table from the base cage, then the local-point stencils), or with one,
/// through [`StencilTableFactory::append_local_point_stencils`](crate::far::StencilTableFactory::append_local_point_stencils).
///
/// Patches keep the indices of the [`PatchTable`] they were built from:
/// the patch [`PatchMap::find_patch`](crate::far::PatchMap::find_patch)
/// returns is the same in both, and [`patch_coord`](Self::patch_coord)
/// turns it into the [`PatchCoord`] a kernel evaluates.
///
/// The layout is OpenSubdiv's, so ports of its shaders read it directly:
///
/// * patches carry no rotation — each patch's control points are ordered
///   so that its `(s, t)` axes are its ptex face's `(u, v)`;
/// * the phantom points of boundary B-spline and box-spline patches are
///   described by the [boundary mask](PatchParam::boundary) and their
///   index-buffer slots hold a valid but arbitrary index;
/// * single-crease patches are regular patches with a positive
///   [`sharpness`](PatchParam::sharpness), the crease edge in the
///   boundary mask;
/// * patches below a non-quad base face set
///   [`non_quad_root`](PatchParam::non_quad_root) and count the step to
///   their ptex face in their [`depth`](PatchParam::depth).
///
/// [`evaluate`](Self::evaluate) evaluates the buffers on the CPU, the way
/// a GPU kernel does, and is the reference for shader ports.
///
/// Face-varying patches are not exported yet.
#[derive(Debug, Clone)]
pub struct GpuPatchTable {
    arrays: Vec<PatchArray>,
    indices: Vec<Index>,
    params: Vec<PatchParam>,
    num_control_values: usize,
    local_point_stencils: StencilTable,
}

impl GpuPatchTable {
    /// Flatten `table`.
    ///
    /// Returns [`TopologyError::PatchDepthTooDeep`] when a patch lies deeper
    /// than [`PatchParam::MAX_DEPTH`] below its ptex face — possible only
    /// for uniform refinement past level 10, as adaptive isolation stops
    /// there.
    ///
    /// # Panics
    ///
    /// Panics when the mesh has more than `2^28` ptex faces, the most the
    /// packed [`PatchParam::face_id`] can address.
    pub fn new(table: &PatchTable) -> Result<GpuPatchTable, TopologyError> {
        let num_control_values = table.num_control_values();
        let ptex = table.ptex_indices();
        assert!(
            ptex.num_faces() <= 1 << 28,
            "{} ptex faces exceed the 2^28 a packed patch parameter can address",
            ptex.num_faces()
        );

        let mut arrays: Vec<PatchArray> = Vec::new();
        let mut indices: Vec<Index> = Vec::with_capacity(table.num_patches() * 16);
        let mut params = Vec::with_capacity(table.num_patches());
        let mut local_points: Vec<Vec<(Index, f32)>> = Vec::new();

        for patch in 0..table.num_patches() {
            let desc = patch_descriptor::from_patch_type(table.patch_type(patch));
            if arrays.last().is_none_or(|array| array.desc != desc) {
                arrays.push(PatchArray {
                    reg_desc: desc,
                    desc,
                    num_patches: 0,
                    index_base: indices.len() as u32,
                    stride: patch_descriptor::num_control_vertices(desc) as u32,
                    primitive_id_base: patch as u32,
                });
            }
            arrays.last_mut().unwrap().num_patches += 1;

            let far_param = table.patch_param(patch);
            if u32::from(far_param.depth) > PatchParam::MAX_DEPTH {
                return Err(TopologyError::PatchDepthTooDeep {
                    depth: far_param.depth as usize,
                    max: PatchParam::MAX_DEPTH as usize,
                });
            }
            let depth = u32::from(far_param.depth);
            let n = 1i64 << depth;
            // The patch's cell in the 2^depth grid of its ptex face, and its
            // rotation: quads are reordered to rotation 0, triangles keep
            // theirs (0, or 2 for an inverted triangle, encoded in u, v).
            let rotation = u32::from(far_param.rotation % 4);
            let corner = CORNER[rotation as usize];
            let cell = [0, 1].map(|k| (far_param.origin[k] * n as f32).round() as i64 - corner[k]);
            debug_assert!(cell.iter().all(|&c| (0..n).contains(&c)));
            let (u, v) = if far_param.triangular && rotation == 2 {
                (n - 1 - cell[0], n - 1 - cell[1])
            } else {
                (cell[0], cell[1])
            };
            let quad_rotation = if far_param.triangular { 0 } else { rotation };
            let base_face = ptex.base_face(far_param.ptex_face as usize) as usize;
            let non_quad_root = !far_param.triangular && ptex.face_ptex_count(base_face) > 1;

            let mut boundary = 0;
            let mut sharpness = 0.0;
            let mut regular = false;
            let mut end_cap = |range, num_points: usize, permute: &dyn Fn(usize) -> usize| {
                let first = local_points.len();
                local_points.resize(first + num_points, Vec::new());
                table.for_each_end_cap_entry(range, num_points, |point, cv, w| {
                    local_points[first + permute(point)].push((cv, w));
                });
                for k in 0..num_points {
                    indices.push((num_control_values + first + k) as Index);
                }
            };
            match table.patch_ref(patch) {
                PatchRef::Regular(cvs) => {
                    let grid = rotate_grid(cvs, quad_rotation);
                    let phantom = |slot: usize| grid[slot] == INDEX_INVALID;
                    // Rows t = -1 and t = 2, columns s = 2 and s = -1, as the
                    // edges 0, 2, 1 and 3 they lie beyond.
                    boundary = phantom(1) as u32
                        | (phantom(7) as u32) << 1
                        | (phantom(13) as u32) << 2
                        | (phantom(4) as u32) << 3;
                    for (slot, &cv) in grid.iter().enumerate() {
                        indices.push(if cv == INDEX_INVALID {
                            // Any real index: the slot's weight is folded
                            // away. The four central slots are the face's.
                            let (i, j) = (slot % 4, slot / 4);
                            grid[4 * j.clamp(1, 2) + i.clamp(1, 2)]
                        } else {
                            cv
                        });
                    }
                    regular = true;
                }
                PatchRef::SingleCrease {
                    cvs,
                    edge,
                    sharpness: crease,
                } => {
                    indices.extend_from_slice(&rotate_grid(cvs, quad_rotation));
                    boundary = 1 << ((u32::from(edge) + quad_rotation) % 4);
                    sharpness = crease;
                    regular = true;
                }
                PatchRef::Gregory(range) => {
                    // Points 5k..5k+5 belong to corner k, which moves to
                    // corner k + rotation.
                    let permute = |p: usize| 5 * ((p / 5 + quad_rotation as usize) % 4) + p % 5;
                    end_cap(range, 20, &permute);
                }
                PatchRef::Quads(cvs) => {
                    let mut out = [0; 4];
                    for (k, &cv) in cvs.iter().enumerate() {
                        out[(k + quad_rotation as usize) % 4] = cv;
                    }
                    indices.extend_from_slice(&out);
                }
                PatchRef::Loop(cvs) => {
                    boundary = box_spline_boundary_mask(cvs);
                    // Slot 4 is the face's first corner, always real.
                    indices.extend(
                        cvs.iter()
                            .map(|&cv| if cv == INDEX_INVALID { cvs[4] } else { cv }),
                    );
                    regular = true;
                }
                PatchRef::GregoryTriangle(range) => end_cap(range, 18, &|p| p),
                PatchRef::Triangles(cvs) => indices.extend_from_slice(cvs),
            }

            params.push(PatchParam::new(
                far_param.ptex_face,
                u as u32,
                v as u32,
                depth + non_quad_root as u32,
                non_quad_root,
                boundary,
                0,
                regular,
                sharpness,
            ));
        }

        Ok(GpuPatchTable {
            arrays,
            indices,
            params,
            num_control_values,
            local_point_stencils: StencilTable::from_stencils(num_control_values, local_points),
        })
    }

    /// The patch arrays, in patch order (`GetPatchArrayBuffer`).
    pub fn patch_arrays(&self) -> &[PatchArray] {
        &self.arrays
    }

    /// Every patch's control points, [`PatchArray::stride`] per patch
    /// (`GetPatchIndexBuffer`): indices into the control values followed by
    /// the local points.
    pub fn patch_index_buffer(&self) -> &[Index] {
        &self.indices
    }

    /// One packed parameter per patch (`GetPatchParamBuffer`).
    pub fn patch_param_buffer(&self) -> &[PatchParam] {
        &self.params
    }

    /// The patch arrays as the `u32` words of an `OsdPatchArray` buffer.
    pub fn patch_array_words(&self) -> Vec<u32> {
        self.arrays.iter().flat_map(PatchArray::to_words).collect()
    }

    /// The patch parameters as the `u32` words of an `OsdPatchParam`
    /// buffer.
    pub fn patch_param_words(&self) -> Vec<u32> {
        self.params.iter().flat_map(PatchParam::to_words).collect()
    }

    /// The number of patches.
    pub fn num_patches(&self) -> usize {
        self.params.len()
    }

    /// The number of control values the patches index first
    /// ([`PatchTable::num_control_values`]): every refinement level's
    /// vertices, base level first.
    pub fn num_control_values(&self) -> usize {
        self.num_control_values
    }

    /// The number of local points, which follow the control values in the
    /// buffer the patches index (`GetNumLocalPoints`).
    pub fn num_local_points(&self) -> usize {
        self.local_point_stencils.num_stencils()
    }

    /// The stencils of the local points over the control values
    /// (`GetLocalPointStencilTable`): applied to the
    /// [`num_control_values`](Self::num_control_values) control values,
    /// they give the local points that follow them.
    pub fn local_point_stencils(&self) -> &StencilTable {
        &self.local_point_stencils
    }

    /// The [`PatchCoord`] of ptex-face location `(u, v)` on patch `patch`,
    /// as found by [`PatchMap::find_patch`](crate::far::PatchMap::find_patch).
    ///
    /// # Panics
    ///
    /// Panics when `patch` is out of range.
    pub fn patch_coord(&self, patch: usize, u: f32, v: f32) -> PatchCoord {
        assert!(
            patch < self.num_patches(),
            "patch {patch} out of range ({} patches)",
            self.num_patches()
        );
        let array_index = self
            .arrays
            .partition_point(|array| array.primitive_id_base as usize <= patch)
            - 1;
        let array = &self.arrays[array_index];
        PatchCoord {
            array_index: array_index as u32,
            patch_index: patch as u32,
            vert_index: array.index_base + (patch as u32 - array.primitive_id_base) * array.stride,
            s: u,
            t: v,
        }
    }

    /// Evaluate the limit point and its first derivatives along the ptex
    /// face's `u` and `v` at `coord`, from `values`: the control values
    /// followed by the local points
    /// ([`num_control_values`](Self::num_control_values)` + `[`num_local_points`](Self::num_local_points)
    /// values).
    ///
    /// This is what OpenSubdiv's `EvalPatches` kernels do per coordinate,
    /// reading only the flat buffers, and it agrees with
    /// [`PatchTable::evaluate`] on the table it was built from.
    ///
    /// # Panics
    ///
    /// Panics when `coord` does not address a patch of the table, or when
    /// `values` is too short.
    pub fn evaluate<T: Primvar>(&self, coord: &PatchCoord, values: &[T]) -> (T, T, T) {
        assert!(
            values.len() >= self.num_control_values + self.num_local_points(),
            "{} values given, {} needed",
            values.len(),
            self.num_control_values + self.num_local_points()
        );
        let array = &self.arrays[coord.array_index as usize];
        let param = &self.params[coord.patch_index as usize];
        let weights = evaluate_patch_basis(array.desc, param, coord.s, coord.t)
            .expect("tables hold known patch types");
        let cvs = &self.indices[coord.vert_index as usize..][..weights.num_points];

        let mut point = values[0].clone();
        point.clear();
        let (mut du, mut dv) = (point.clone(), point.clone());
        for (k, &cv) in cvs.iter().enumerate() {
            let value = &values[cv as usize];
            point.add_with_weight(value, weights.point[k]);
            du.add_with_weight(value, weights.du[k]);
            dv.add_with_weight(value, weights.dv[k]);
        }
        (point, du, dv)
    }
}

/// The corner of a patch's cell that is its `(s, t) = (0, 0)` corner, per
/// rotation (counter-clockwise from the cell's lower-left corner).
const CORNER: [[i64; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];

/// Reorder a 4x4 B-spline grid (row-major, rows along `t`) of a patch at
/// `rotation` to rotation 0: the grid point at patch coordinates `(x, y)`
/// lies at `corner + R(rotation) · (x, y)` in its cell, `R` the
/// counter-clockwise quarter turn.
fn rotate_grid(cvs: &[Index; 16], rotation: u32) -> [Index; 16] {
    let corner = CORNER[rotation as usize];
    let mut out = [INDEX_INVALID; 16];
    for (slot, &cv) in cvs.iter().enumerate() {
        let (x, y) = ((slot % 4) as i64 - 1, (slot / 4) as i64 - 1);
        let (rx, ry) = match rotation {
            0 => (x, y),
            1 => (-y, x),
            2 => (-x, -y),
            _ => (y, -x),
        };
        let (i, j) = (corner[0] + rx + 1, corner[1] + ry + 1);
        out[(4 * j + i) as usize] = cv;
    }
    out
}

/// The OpenSubdiv boundary mask of a box-spline triangle from its phantom
/// slots (see [`PatchParam`]): its wall edges (phantom slots 1, 9 and 7
/// beyond edges 0, 1 and 2), or else its wall vertices (phantom slots 0, 2
/// and 10 beyond corners 0, 1 and 2, along the lattice line parallel to
/// the opposite edge). A vertex wall beside an edge wall can only be the
/// vertex opposite it.
fn box_spline_boundary_mask(cvs: &[Index; 12]) -> u32 {
    let phantom = |slot: usize| cvs[slot] == INDEX_INVALID;
    let edges = phantom(1) as u32 | (phantom(9) as u32) << 1 | (phantom(7) as u32) << 2;
    let e = |k: u32| edges & (1 << k) != 0;
    let vertices = (!e(0) && !e(2) && phantom(0)) as u32
        | ((!e(0) && !e(1) && phantom(2)) as u32) << 1
        | ((!e(1) && !e(2) && phantom(10)) as u32) << 2;
    match (edges, vertices) {
        (0, 0) => 0,
        (0, v) => v | 1 << 3,
        (e, 0) => e,
        (e, v) => {
            debug_assert_eq!(
                v,
                ((e & 1) << 2) | (e >> 1),
                "a vertex wall opposite the edge"
            );
            e | 2 << 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_grids_keep_the_face_in_the_center() {
        let cvs: [Index; 16] = std::array::from_fn(|k| k as Index);
        // The face's corners are slots 5, 6, 10, 9 (counter-clockwise from
        // (0, 0)); at rotation r, patch corner k is cell corner k + r.
        let corners = [5usize, 6, 10, 9];
        for r in 0..4u32 {
            let out = rotate_grid(&cvs, r);
            for k in 0..4 {
                assert_eq!(out[corners[(k + r as usize) % 4]], corners[k] as Index);
            }
            let mut sorted = out;
            sorted.sort();
            assert_eq!(sorted, cvs);
        }
    }
}
