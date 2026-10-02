//! Ptex face indexing (port of `opensubdiv/far/ptexIndices.h`).

use super::topology_refiner::TopologyRefiner;
use crate::Index;

/// Mapping between base-level faces and *ptex* face indices
/// (`Far::PtexIndices`).
///
/// The ptex convention parameterizes the subdivision surface by the
/// scheme's regular face domain: a *regular* base face (a quadrilateral
/// for the quad-split schemes, a triangle for Loop) maps to a single ptex
/// face, while an N-sided base face of a quad-split scheme maps to N ptex
/// faces — one per corner, corresponding to its N quadrilateral child
/// faces at refinement level 1.
#[derive(Debug, Clone)]
pub struct PtexIndices {
    /// Ptex index of the first ptex face of each base face (+ total).
    offsets: Vec<u32>,
    /// Reverse map: base face of each ptex face.
    base_faces: Vec<Index>,
    /// Reverse map: corner of the base face each ptex face covers
    /// (always 0 for quad base faces).
    corners: Vec<u16>,
}

impl PtexIndices {
    /// Compute the ptex-face indexing for the base level of `refiner`
    /// (`PtexIndices::PtexIndices`).
    pub fn new(refiner: &TopologyRefiner) -> Self {
        let base = refiner.level(0);
        let regular_size = refiner.scheme_type().regular_face_size();
        let ptex_count = |f: usize| {
            let size = base.face_vertices(f).len();
            if size == regular_size {
                1
            } else {
                size
            }
        };
        let num_ptex = (0..base.num_faces()).map(ptex_count).sum();
        let mut offsets = Vec::with_capacity(base.num_faces() + 1);
        let mut base_faces = Vec::with_capacity(num_ptex);
        let mut corners = Vec::with_capacity(num_ptex);
        offsets.push(0);
        for f in 0..base.num_faces() {
            for k in 0..ptex_count(f) {
                base_faces.push(f as Index);
                corners.push(k as u16);
            }
            offsets.push(base_faces.len() as u32);
        }
        Self {
            offsets,
            base_faces,
            corners,
        }
    }

    /// The total number of ptex faces (`GetNumFaces`).
    pub fn num_faces(&self) -> usize {
        self.base_faces.len()
    }

    /// The ptex index of the first ptex face of base face `face`
    /// (`GetFaceId`).
    pub fn face_id(&self, face: usize) -> Index {
        self.offsets[face]
    }

    /// The number of ptex faces of base face `face`: 1 for regular faces
    /// (quads, or triangles under Loop), N for N-gons.
    pub fn face_ptex_count(&self, face: usize) -> usize {
        (self.offsets[face + 1] - self.offsets[face]) as usize
    }

    /// The base face covered by ptex face `ptex_face`.
    pub fn base_face(&self, ptex_face: usize) -> Index {
        self.base_faces[ptex_face]
    }

    /// The corner of the base face covered by ptex face `ptex_face` (always
    /// 0 for regular base faces, which map to a single ptex face).
    pub fn base_face_corner(&self, ptex_face: usize) -> usize {
        self.corners[ptex_face] as usize
    }
}
