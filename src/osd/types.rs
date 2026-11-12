// Portions derived from OpenSubdiv (https://github.com/PixarAnimationStudios/OpenSubdiv):
//   osd/types.h — Copyright 2015 Pixar.
//   osd/patchBasisTypes.h — Copyright 2018 Pixar.
//   far/patchParam.h — Copyright 2013 Pixar.
// Licensed under the Tomorrow Open Source Technology License 1.0; see
// LICENSE-OPENSUBDIV at the root of this crate.

//! The plain-data records a GPU evaluator reads: patch arrays, packed patch
//! parameters and patch coordinates (port of `opensubdiv/osd/types.h` and
//! of the `OsdPatchArray` / `OsdPatchParam` / `OsdPatchCoord` structs of
//! `osd/patchBasisTypes.h`, with `Far::PatchParam`'s bitfield layout).
//!
//! Every record is `#[repr(C)]` and made of 32-bit fields only, so it has
//! no padding and the same layout as the OpenSubdiv shader structs; its
//! `to_words` method gives that layout as `u32` words, ready to be written
//! to a storage buffer.

use crate::far::PatchType;

/// `Far::PatchDescriptor::Type` codes, as stored in [`PatchArray::desc`]
/// and read by OpenSubdiv's patch shaders.
pub mod patch_descriptor {
    use crate::far::PatchType;

    /// Bilinear quad on 4 control vertices ([`PatchType::Quads`]).
    pub const QUADS: u32 = 3;
    /// Linear triangle on 3 control vertices ([`PatchType::Triangles`]).
    pub const TRIANGLES: u32 = 4;
    /// Quartic box-spline triangle on 12 control vertices
    /// ([`PatchType::Loop`]).
    pub const LOOP: u32 = 5;
    /// Bicubic B-spline patch on 16 control vertices
    /// ([`PatchType::Regular`], single-crease patches included).
    pub const REGULAR: u32 = 6;
    /// Gregory patch on 20 control points ([`PatchType::GregoryBasis`]).
    pub const GREGORY_BASIS: u32 = 9;
    /// Gregory triangle on 18 control points
    /// ([`PatchType::GregoryTriangle`]).
    pub const GREGORY_TRIANGLE: u32 = 10;

    /// The descriptor code of `patch_type`.
    pub const fn from_patch_type(patch_type: PatchType) -> u32 {
        match patch_type {
            PatchType::Regular => REGULAR,
            PatchType::GregoryBasis => GREGORY_BASIS,
            PatchType::Quads => QUADS,
            PatchType::Loop => LOOP,
            PatchType::GregoryTriangle => GREGORY_TRIANGLE,
            PatchType::Triangles => TRIANGLES,
        }
    }

    /// The patch type of descriptor code `code`, or `None` for a code no
    /// [`GpuPatchTable`](super::super::GpuPatchTable) emits.
    pub const fn to_patch_type(code: u32) -> Option<PatchType> {
        match code {
            REGULAR => Some(PatchType::Regular),
            GREGORY_BASIS => Some(PatchType::GregoryBasis),
            QUADS => Some(PatchType::Quads),
            LOOP => Some(PatchType::Loop),
            GREGORY_TRIANGLE => Some(PatchType::GregoryTriangle),
            TRIANGLES => Some(PatchType::Triangles),
            _ => None,
        }
    }

    /// The number of control points of a patch of descriptor code `code`
    /// (`PatchDescriptor::GetNumControlVertices`), `0` for an unknown code.
    pub const fn num_control_vertices(code: u32) -> usize {
        match code {
            REGULAR => 16,
            GREGORY_BASIS => 20,
            QUADS => 4,
            LOOP => 12,
            GREGORY_TRIANGLE => 18,
            TRIANGLES => 3,
            _ => 0,
        }
    }

    /// Is a patch of descriptor code `code` triangular (its domain
    /// `s, t >= 0`, `s + t <= 1`)?
    pub const fn is_triangular(code: u32) -> bool {
        matches!(code, LOOP | GREGORY_TRIANGLE | TRIANGLES)
    }
}

/// A run of patches of one type in a [`GpuPatchTable`](super::GpuPatchTable)
/// (`Osd::PatchArray`, `OsdPatchArray` in the shaders).
///
/// The patches of the array are patches `primitive_id_base..` of the
/// table (indices into its patch-param buffer), and patch
/// `primitive_id_base + k` has its `stride` control points at
/// `index_base + k * stride` in the index buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchArray {
    /// Descriptor code of the array's regular patches
    /// ([`patch_descriptor`]); equal to [`desc`](Self::desc) for vertex
    /// patches, as in OpenSubdiv.
    pub reg_desc: u32,
    /// Descriptor code of the array's patches ([`patch_descriptor`]).
    pub desc: u32,
    /// The number of patches in the array.
    pub num_patches: u32,
    /// Where the array's control points start in the index buffer.
    pub index_base: u32,
    /// The number of index-buffer entries per patch.
    pub stride: u32,
    /// The index of the array's first patch in the table (and in its
    /// patch-param buffer).
    pub primitive_id_base: u32,
}

impl PatchArray {
    /// The patch type of the array, or `None` for a descriptor code no
    /// [`GpuPatchTable`](super::GpuPatchTable) emits.
    pub fn patch_type(&self) -> Option<PatchType> {
        patch_descriptor::to_patch_type(self.desc)
    }

    /// The record as the six 32-bit words of `OsdPatchArray`.
    pub fn to_words(&self) -> [u32; 6] {
        [
            self.reg_desc,
            self.desc,
            self.num_patches,
            self.index_base,
            self.stride,
            self.primitive_id_base,
        ]
    }
}

/// A patch's parameterization packed into two words, plus its crease
/// sharpness (`Osd::PatchParam`: `Far::PatchParam` with a `sharpness`,
/// `OsdPatchParam` in the shaders).
///
/// The bit layout is `Far::PatchParam`'s:
///
/// | word     | bits    | field                                    |
/// |----------|---------|------------------------------------------|
/// | `field0` | 0..28   | ptex face ([`face_id`](Self::face_id))   |
/// | `field0` | 28..32  | transition mask (always `0`, see below)  |
/// | `field1` | 0..4    | [`depth`](Self::depth)                   |
/// | `field1` | 4       | [`non_quad_root`](Self::non_quad_root)   |
/// | `field1` | 5       | [`is_regular`](Self::is_regular)         |
/// | `field1` | 7..12   | [`boundary`](Self::boundary) mask        |
/// | `field1` | 12..22  | [`v`](Self::v)                           |
/// | `field1` | 22..32  | [`u`](Self::u)                           |
///
/// Patches carry no rotation: a [`GpuPatchTable`](super::GpuPatchTable)
/// orders every patch's control points so that its `(s, t)` axes are the
/// ptex face's `(u, v)` axes, and an inverted triangle is told by its
/// position (`u + v >= 2^depth`, [`is_triangle_rotated`](Self::is_triangle_rotated)),
/// as in OpenSubdiv.
///
/// The boundary mask has OpenSubdiv's meaning for each patch type:
///
/// * B-spline ([`patch_descriptor::REGULAR`]) with zero
///   [`sharpness`](Self::sharpness): bit `k` set when face edge `k` (edge 0
///   at `t = 0`, then counter-clockwise: `s = 1`, `t = 1`, `s = 0`) is a
///   boundary or infinitely sharp crease, whose row or column of control
///   points beyond is then *phantom*: its slots hold an arbitrary valid
///   index, and its weights are folded onto the next two rows by the
///   reflection `2a - b` (`OsdAdjustBoundaryWeights`).
/// * B-spline with positive sharpness: a *single-crease* patch, the one
///   set bit naming the face edge carrying the semi-sharp crease, all 16
///   control points real. (OpenSubdiv's tessellation shaders read it the
///   same way; its compute evaluator ignores the sharpness.)
/// * Box-spline triangle ([`patch_descriptor::LOOP`]): the low three bits
///   are edges or vertices, as the upper two bits say
///   (`Osd_adjustBoxSplineTriBoundaryWeights`): `0` boundary edges only,
///   `1` boundary vertices only, `2` boundary edges plus the vertex
///   opposite each.
/// * Other patch types: unused, `0`.
///
/// The transition mask (edges where the neighboring patch is one level
/// deeper, for crack-free hardware tessellation) is not computed and is
/// always `0`; parametric evaluation does not need it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchParam {
    /// Ptex face and transition mask.
    pub field0: u32,
    /// Depth, flags, boundary mask and the patch's `(u, v)` cell.
    pub field1: u32,
    /// Sharpness of a single-crease patch's crease, `0.0` otherwise.
    pub sharpness: f32,
}

impl PatchParam {
    /// The deepest [`depth`](Self::depth), not counting
    /// [`non_quad_root`](Self::non_quad_root), the packing can hold: the
    /// `(u, v)` cell takes 10 bits per axis, as in OpenSubdiv.
    pub const MAX_DEPTH: u32 = 10;

    /// Pack the fields of a patch parameter (`Far::PatchParam::Set`). Each
    /// field is truncated to its bit width.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        face_id: u32,
        u: u32,
        v: u32,
        depth: u32,
        non_quad_root: bool,
        boundary: u32,
        transition: u32,
        regular: bool,
        sharpness: f32,
    ) -> Self {
        let pack = |value: u32, width: u32, offset: u32| (value & ((1 << width) - 1)) << offset;
        PatchParam {
            field0: pack(face_id, 28, 0) | pack(transition, 4, 28),
            field1: pack(u, 10, 22)
                | pack(v, 10, 12)
                | pack(boundary, 5, 7)
                | pack(regular as u32, 1, 5)
                | pack(non_quad_root as u32, 1, 4)
                | pack(depth, 4, 0),
            sharpness,
        }
    }

    /// The ptex face the patch belongs to (`GetFaceId`).
    pub fn face_id(&self) -> u32 {
        self.field0 & 0x0fff_ffff
    }

    /// The transition mask (`GetTransition`): always `0` in a
    /// [`GpuPatchTable`](super::GpuPatchTable).
    pub fn transition(&self) -> u32 {
        self.field0 >> 28
    }

    /// The patch's column in the `2^(depth - non_quad_root)`-square grid
    /// of its ptex face (`GetU`); see
    /// [`is_triangle_rotated`](Self::is_triangle_rotated) for inverted
    /// triangles.
    pub fn u(&self) -> u32 {
        (self.field1 >> 22) & 0x3ff
    }

    /// The patch's row in its ptex face's grid (`GetV`).
    pub fn v(&self) -> u32 {
        (self.field1 >> 12) & 0x3ff
    }

    /// The boundary mask (`GetBoundary`); see the [type docs](Self).
    pub fn boundary(&self) -> u32 {
        (self.field1 >> 7) & 0x1f
    }

    /// Is the patch a regular B-spline or box-spline patch (`IsRegular`)?
    pub fn is_regular(&self) -> bool {
        (self.field1 >> 5) & 1 != 0
    }

    /// Does the patch descend from a non-quad base face, whose level-1
    /// children are the ptex faces (`NonQuadRoot`)? Its
    /// [`depth`](Self::depth) then counts that first step.
    pub fn non_quad_root(&self) -> bool {
        (self.field1 >> 4) & 1 != 0
    }

    /// The refinement level of the patch (`GetDepth`).
    pub fn depth(&self) -> u32 {
        self.field1 & 0xf
    }

    /// The patch's size in its ptex face's parametric space
    /// (`GetParamFraction`): `2^-(depth - non_quad_root)`.
    pub fn param_fraction(&self) -> f32 {
        1.0 / (1u32 << (self.depth() - self.non_quad_root() as u32)) as f32
    }

    /// Is this triangular patch inverted with respect to its ptex face
    /// (`IsTriangleRotated`): the central child of a subdivided triangle,
    /// encoded by `u + v >= 2^depth`? Its `(s, t)` axes then point along
    /// the ptex face's `-u` and `-v`, from the corner at
    /// `(2^depth - u, 2^depth - v) · 2^-depth`.
    pub fn is_triangle_rotated(&self) -> bool {
        self.u() + self.v() >= 1 << self.depth()
    }

    /// Convert ptex-face coordinates to the quad patch's local `(s, t)`
    /// (`Normalize`).
    pub fn normalize(&self, u: f32, v: f32) -> (f32, f32) {
        let inv = 1.0 / self.param_fraction();
        (u * inv - self.u() as f32, v * inv - self.v() as f32)
    }

    /// Convert the quad patch's local `(s, t)` to ptex-face coordinates
    /// (`Unnormalize`).
    pub fn unnormalize(&self, s: f32, t: f32) -> (f32, f32) {
        let frac = self.param_fraction();
        ((s + self.u() as f32) * frac, (t + self.v() as f32) * frac)
    }

    /// Convert ptex-face coordinates to the triangular patch's local
    /// `(s, t)` (`NormalizeTriangle`).
    pub fn normalize_triangle(&self, u: f32, v: f32) -> (f32, f32) {
        if self.is_triangle_rotated() {
            let inv = 1.0 / self.param_fraction();
            let n = (1u32 << self.depth()) as f32;
            (n - self.u() as f32 - u * inv, n - self.v() as f32 - v * inv)
        } else {
            self.normalize(u, v)
        }
    }

    /// Convert the triangular patch's local `(s, t)` to ptex-face
    /// coordinates (`UnnormalizeTriangle`).
    pub fn unnormalize_triangle(&self, s: f32, t: f32) -> (f32, f32) {
        if self.is_triangle_rotated() {
            let frac = self.param_fraction();
            let n = (1u32 << self.depth()) as f32;
            (
                (n - self.u() as f32 - s) * frac,
                (n - self.v() as f32 - t) * frac,
            )
        } else {
            self.unnormalize(s, t)
        }
    }

    /// The record as the three 32-bit words of `OsdPatchParam`, the
    /// sharpness as its IEEE-754 bits.
    pub fn to_words(&self) -> [u32; 3] {
        [self.field0, self.field1, self.sharpness.to_bits()]
    }
}

/// A location to evaluate: a patch and a ptex-face `(u, v)` inside it
/// (`Osd::PatchCoord`, `OsdPatchCoord` in the shaders).
///
/// Build one with [`GpuPatchTable::patch_coord`](super::GpuPatchTable::patch_coord)
/// from the patch [`PatchMap::find_patch`](crate::far::PatchMap::find_patch)
/// returns.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchCoord {
    /// The [`PatchArray`] holding the patch.
    pub array_index: u32,
    /// The patch, an index into the patch-param buffer.
    pub patch_index: u32,
    /// Where the patch's control points start in the index buffer.
    pub vert_index: u32,
    /// The ptex-face `u` coordinate.
    pub s: f32,
    /// The ptex-face `v` coordinate.
    pub t: f32,
}

impl PatchCoord {
    /// The record as the five 32-bit words of `OsdPatchCoord`, the
    /// coordinates as their IEEE-754 bits.
    pub fn to_words(&self) -> [u32; 5] {
        [
            self.array_index,
            self.patch_index,
            self.vert_index,
            self.s.to_bits(),
            self.t.to_bits(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_the_shader_layouts() {
        assert_eq!(std::mem::size_of::<PatchArray>(), 24);
        assert_eq!(std::mem::size_of::<PatchParam>(), 12);
        assert_eq!(std::mem::size_of::<PatchCoord>(), 20);
    }

    #[test]
    fn patch_param_round_trips_its_fields() {
        let p = PatchParam::new(0x0abc_def1, 1023, 517, 11, true, 0x15, 0xa, true, 2.5);
        assert_eq!(p.face_id(), 0x0abc_def1);
        assert_eq!(p.u(), 1023);
        assert_eq!(p.v(), 517);
        assert_eq!(p.depth(), 11);
        assert!(p.non_quad_root());
        assert_eq!(p.boundary(), 0x15);
        assert_eq!(p.transition(), 0xa);
        assert!(p.is_regular());
        assert_eq!(p.to_words()[2], 2.5f32.to_bits());

        // Far::PatchParam::Set's bit positions.
        let q = PatchParam::new(7, 3, 5, 2, false, 0b1001, 0, false, 0.0);
        assert_eq!(q.field0, 7);
        assert_eq!(q.field1, 3 << 22 | 5 << 12 | 0b1001 << 7 | 2);
    }

    #[test]
    fn triangle_normalization_inverts() {
        // The central child of a depth-1 triangle.
        let p = PatchParam::new(0, 1, 1, 1, false, 0, 0, true, 0.0);
        assert!(p.is_triangle_rotated());
        assert_eq!(p.unnormalize_triangle(0.0, 0.0), (0.5, 0.5));
        assert_eq!(p.unnormalize_triangle(1.0, 0.0), (0.0, 0.5));
        let (s, t) = p.normalize_triangle(0.3, 0.4);
        assert!((s - 0.4).abs() < 1e-6 && (t - 0.2).abs() < 1e-6);
    }

    #[test]
    fn descriptor_codes_round_trip() {
        for t in [
            PatchType::Regular,
            PatchType::GregoryBasis,
            PatchType::Quads,
            PatchType::Loop,
            PatchType::GregoryTriangle,
            PatchType::Triangles,
        ] {
            let code = patch_descriptor::from_patch_type(t);
            assert_eq!(patch_descriptor::to_patch_type(code), Some(t));
            assert!(patch_descriptor::num_control_vertices(code) > 0);
        }
        assert_eq!(patch_descriptor::to_patch_type(0), None);
    }
}
