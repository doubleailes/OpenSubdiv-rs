//! One step of refinement between two levels (port of
//! `opensubdiv/vtr/refinement.h` — the uniform `QuadRefinement` /
//! `TriRefinement` paths plus the *sparse* (selected) refinement used by
//! feature-adaptive isolation, the role of `Vtr::internal::SparseSelector`,
//! for both splits).

use super::level::{Level, TopologyError};
use crate::sdc::{Crease, Scheme, Split};
use crate::{Index, INDEX_INVALID};

/// Mapping between a parent [`Level`] and the child [`Level`] produced by one
/// step of refinement (`Vtr::internal::Refinement`).
///
/// Child vertices are ordered by their parent component type: first the
/// child vertices of *faces* (absent for triangular splits), then those of
/// *edges*, then those of *vertices*. Under sparse refinement, components
/// without children map to `INDEX_INVALID`.
#[derive(Debug, Clone)]
pub struct Refinement {
    split: Split,
    /// Child vertex originating from each parent face (`INDEX_INVALID` for
    /// triangular splits, which generate no face vertices, and for faces
    /// excluded by sparse refinement).
    face_child_vert: Vec<Index>,
    /// Child vertex at the midpoint of each parent edge.
    edge_child_vert: Vec<Index>,
    /// Child vertex corresponding to each parent vertex.
    vert_child_vert: Vec<Index>,
    /// Parent face of each child face.
    child_face_parent: Vec<Index>,
}

impl Refinement {
    /// The kind of topological split performed.
    pub fn split(&self) -> Split {
        self.split
    }

    /// The child vertex generated from parent face `face`
    /// (`Refinement::getFaceChildVertex`).
    pub fn face_child_vertex(&self, face: usize) -> Index {
        self.face_child_vert[face]
    }

    /// The child vertex generated from parent edge `edge`
    /// (`Refinement::getEdgeChildVertex`).
    pub fn edge_child_vertex(&self, edge: usize) -> Index {
        self.edge_child_vert[edge]
    }

    /// The child vertex generated from parent vertex `vertex`
    /// (`Refinement::getVertexChildVertex`).
    pub fn vertex_child_vertex(&self, vertex: usize) -> Index {
        self.vert_child_vert[vertex]
    }

    /// The parent face of child face `child_face`.
    pub fn child_face_parent_face(&self, child_face: usize) -> Index {
        self.child_face_parent[child_face]
    }

    /// Uniformly refine `parent`, returning the child level and the
    /// refinement mapping (`Refinement::refine`).
    ///
    /// The scheme determines the split (quads for Bilinear/Catmark, tris for
    /// Loop); creasing options govern how sharpness is subdivided. Boundary
    /// sharpening applied to the cage propagates through sharpness
    /// subdivision, so no per-level re-sharpening is required.
    pub fn refine(parent: &Level, scheme: &Scheme) -> Result<(Level, Refinement), TopologyError> {
        Self::refine_selected(parent, scheme, None)
    }

    /// Refine `parent` sparsely: children are generated only for the faces
    /// flagged in `selection` *plus their one-ring neighborhoods* (the
    /// supporting faces required so that selected faces' children have
    /// complete neighborhoods at the child level, as guaranteed by
    /// OpenSubdiv's `SparseSelector`). Pass `None` to refine everything.
    pub fn refine_selected(
        parent: &Level,
        scheme: &Scheme,
        selection: Option<&[bool]>,
    ) -> Result<(Level, Refinement), TopologyError> {
        match selection {
            None => Self::refine_included(parent, scheme, &vec![true; parent.num_faces()]),
            Some(selected) => {
                Self::refine_included(parent, scheme, &Self::expand_selection(parent, selected))
            }
        }
    }

    /// Expand a face selection to its one-ring support: every face sharing
    /// a vertex with a selected face is included (`SparseSelector`'s
    /// neighborhood expansion). The result is the face mask
    /// [`refine_included`](Self::refine_included) takes.
    pub fn expand_selection(parent: &Level, selected: &[bool]) -> Vec<bool> {
        assert_eq!(selected.len(), parent.num_faces());
        let mut vertex_marked = vec![false; parent.num_vertices()];
        for (f, &sel) in selected.iter().enumerate() {
            if sel {
                for &v in parent.face_vertices(f) {
                    vertex_marked[v as usize] = true;
                }
            }
        }
        (0..parent.num_faces())
            .map(|f| {
                parent
                    .face_vertices(f)
                    .iter()
                    .any(|&v| vertex_marked[v as usize])
            })
            .collect()
    }

    /// Refine exactly the faces flagged in `included` (one flag per face
    /// of `parent`), with no further expansion. The caller is responsible
    /// for including the support faces the refined faces' children need
    /// (see [`expand_selection`](Self::expand_selection)); passing one
    /// mask to several levels that mirror each other's faces — the
    /// geometry and its face-varying value meshes — keeps their child
    /// faces in lockstep even where the meshes' vertex connectivity
    /// differs (as it does across face-varying seams).
    pub fn refine_included(
        parent: &Level,
        scheme: &Scheme,
        included: &[bool],
    ) -> Result<(Level, Refinement), TopologyError> {
        assert_eq!(included.len(), parent.num_faces());
        match scheme.scheme_type().topological_split_type() {
            Split::ToQuads => Self::refine_quads(parent, scheme, included),
            Split::ToTris => Self::refine_tris(parent, scheme, included),
            Split::Hybrid => unimplemented!("hybrid splits are not used by any scheme"),
        }
    }

    fn refine_quads(
        parent: &Level,
        scheme: &Scheme,
        included: &[bool],
    ) -> Result<(Level, Refinement), TopologyError> {
        let (num_faces, num_edges, num_verts) = (
            parent.num_faces(),
            parent.num_edges(),
            parent.num_vertices(),
        );

        // Child vertices exist only for included faces and the edges and
        // vertices incident to them. Ordering: faces, then edges, then
        // vertices.
        let mut face_child_vert = vec![INDEX_INVALID; num_faces];
        let mut edge_child_vert = vec![INDEX_INVALID; num_edges];
        let mut vert_child_vert = vec![INDEX_INVALID; num_verts];

        let mut edge_has_child = vec![false; num_edges];
        let mut vert_has_child = vec![false; num_verts];
        for (f, &inc) in included.iter().enumerate() {
            if inc {
                for &e in parent.face_edges(f) {
                    edge_has_child[e as usize] = true;
                }
                for &v in parent.face_vertices(f) {
                    vert_has_child[v as usize] = true;
                }
            }
        }

        let mut next = 0 as Index;
        for (f, fcv) in face_child_vert.iter_mut().enumerate() {
            if included[f] {
                *fcv = next;
                next += 1;
            }
        }
        for (e, ecv) in edge_child_vert.iter_mut().enumerate() {
            if edge_has_child[e] {
                *ecv = next;
                next += 1;
            }
        }
        for (v, vcv) in vert_child_vert.iter_mut().enumerate() {
            if vert_has_child[v] {
                *vcv = next;
                next += 1;
            }
        }
        let num_child_verts = next as usize;

        // An N-sided included face yields N child quads. Child face `i` is
        // the quad at corner `i`: [corner, leading edge, center, trailing edge].
        let mut verts_per_face = Vec::new();
        let mut child_face_verts = Vec::new();
        let mut child_face_parent = Vec::new();
        for (f, &inc) in included.iter().enumerate() {
            if !inc {
                continue;
            }
            let center = face_child_vert[f];
            let fv = parent.face_vertices(f);
            let fe = parent.face_edges(f);
            let n = fv.len();
            for i in 0..n {
                let prev = (i + n - 1) % n;
                verts_per_face.push(4);
                child_face_verts.extend_from_slice(&[
                    vert_child_vert[fv[i] as usize],
                    edge_child_vert[fe[i] as usize],
                    center,
                    edge_child_vert[fe[prev] as usize],
                ]);
                child_face_parent.push(f as Index);
            }
        }

        let refinement = Refinement {
            split: Split::ToQuads,
            face_child_vert,
            edge_child_vert,
            vert_child_vert,
            child_face_parent,
        };
        let child = refinement.finalize_child_level(
            parent,
            scheme,
            num_child_verts,
            &verts_per_face,
            &child_face_verts,
        )?;
        Ok((child, refinement))
    }

    fn refine_tris(
        parent: &Level,
        scheme: &Scheme,
        included: &[bool],
    ) -> Result<(Level, Refinement), TopologyError> {
        let (num_faces, num_edges, num_verts) = (
            parent.num_faces(),
            parent.num_edges(),
            parent.num_vertices(),
        );

        // No face child-vertices: ordering is edges, then vertices. Under
        // sparse refinement, child vertices exist only for the edges and
        // vertices of included faces.
        let face_child_vert: Vec<Index> = vec![INDEX_INVALID; num_faces];
        let mut edge_child_vert = vec![INDEX_INVALID; num_edges];
        let mut vert_child_vert = vec![INDEX_INVALID; num_verts];

        let mut edge_has_child = vec![false; num_edges];
        let mut vert_has_child = vec![false; num_verts];
        for (f, &inc) in included.iter().enumerate() {
            if inc {
                for &e in parent.face_edges(f) {
                    edge_has_child[e as usize] = true;
                }
                for &v in parent.face_vertices(f) {
                    vert_has_child[v as usize] = true;
                }
            }
        }

        let mut next = 0 as Index;
        for (e, ecv) in edge_child_vert.iter_mut().enumerate() {
            if edge_has_child[e] {
                *ecv = next;
                next += 1;
            }
        }
        for (v, vcv) in vert_child_vert.iter_mut().enumerate() {
            if vert_has_child[v] {
                *vcv = next;
                next += 1;
            }
        }
        let num_child_verts = next as usize;

        // Each triangle yields 4 children, ordered and oriented as in
        // OpenSubdiv's `TriRefinement`: the three corner triangles keep the
        // parent's orientation, and the central triangle is inverted:
        //
        //   child 0: (v0, e0, e2)     child 1: (e0, v1, e1)
        //   child 2: (e2, e1, v2)     child 3: (e1, e2, e0)
        //
        // so that child `i < 3` has corner `i` at the parent's corner `i`
        // and child 3 has each corner at the middle of the opposite edge.
        let mut verts_per_face = Vec::new();
        let mut child_face_verts = Vec::new();
        let mut child_face_parent = Vec::new();
        for (f, &inc) in included.iter().enumerate() {
            if !inc {
                continue;
            }
            let fv = parent.face_vertices(f);
            let fe = parent.face_edges(f);
            debug_assert_eq!(fv.len(), 3, "tri split requires a triangulated level");
            let v = |i: usize| vert_child_vert[fv[i] as usize];
            let e = |i: usize| edge_child_vert[fe[i] as usize];
            let children = [
                [v(0), e(0), e(2)],
                [e(0), v(1), e(1)],
                [e(2), e(1), v(2)],
                [e(1), e(2), e(0)],
            ];
            for child in &children {
                verts_per_face.push(3);
                child_face_verts.extend_from_slice(child);
                child_face_parent.push(f as Index);
            }
        }

        let refinement = Refinement {
            split: Split::ToTris,
            face_child_vert,
            edge_child_vert,
            vert_child_vert,
            child_face_parent,
        };
        let child = refinement.finalize_child_level(
            parent,
            scheme,
            num_child_verts,
            &verts_per_face,
            &child_face_verts,
        )?;
        Ok((child, refinement))
    }

    /// Build the child level's topology from the generated face list, then
    /// propagate subdivided sharpness and hole tags.
    ///
    /// Boundary sharpening is *not* re-applied here: the cage's boundary
    /// sharpness (applied once at level 0) propagates exactly through
    /// sharpness subdivision — infinitely sharp edges and vertices beget
    /// infinitely sharp children, and refinement never creates new boundary
    /// corners. This also keeps sparse child levels correct, whose fringe
    /// edges must not be mistaken for actual mesh boundaries.
    fn finalize_child_level(
        &self,
        parent: &Level,
        scheme: &Scheme,
        num_child_verts: usize,
        verts_per_face: &[usize],
        child_face_verts: &[Index],
    ) -> Result<Level, TopologyError> {
        let mut child =
            Level::from_face_vertices(num_child_verts, verts_per_face, child_face_verts)?;
        let crease = scheme.crease();

        // Vertex sharpness: child vertices of parent vertices inherit the
        // subdivided vertex sharpness; all other child vertices are smooth.
        for v in 0..parent.num_vertices() {
            let cv = self.vert_child_vert[v];
            if cv == INDEX_INVALID {
                continue;
            }
            let s = crease.subdivide_vertex_sharpness(parent.vertex_sharpness(v));
            if Crease::is_sharp(s) {
                child.set_vertex_sharpness(cv as usize, s);
            }
        }

        // Edge sharpness: the two child (half) edges of each sharp parent
        // edge get the sharpness subdivided at their respective end vertex
        // (supporting the Chaikin rule); edges interior to a parent face are
        // smooth.
        let mut incident = Vec::new();
        for e in 0..parent.num_edges() {
            let mid = self.edge_child_vert[e];
            if mid == INDEX_INVALID {
                continue;
            }
            let sharpness = parent.edge_sharpness(e);
            if !Crease::is_sharp(sharpness) {
                continue;
            }
            for &end in &parent.edge_vertices(e) {
                parent.gather_vertex_edge_sharpness(end as usize, &mut incident);
                let child_sharpness =
                    crease.subdivide_edge_sharpness_at_vertex(sharpness, &incident);
                if Crease::is_sharp(child_sharpness) {
                    let child_edge = child
                        .find_edge(self.vert_child_vert[end as usize], mid)
                        .expect("child of a parent edge must exist in the child level");
                    child.set_edge_sharpness(child_edge as usize, child_sharpness);
                }
            }
        }

        // Hole tags propagate from parent face to all its child faces.
        for (cf, &pf) in self.child_face_parent.iter().enumerate() {
            if parent.is_face_hole(pf as usize) {
                child.set_face_hole(cf, true);
            }
        }

        Ok(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdc::{Options, SchemeType};

    #[test]
    fn quad_refinement_counts() {
        // A single quad: 4 verts, 4 edges, 1 face.
        let level = Level::from_face_vertices(4, &[4], &[0, 1, 2, 3]).unwrap();
        let scheme = Scheme::new(SchemeType::Catmark, Options::default());
        let (child, refinement) = Refinement::refine(&level, &scheme).unwrap();

        assert_eq!(child.num_vertices(), 1 + 4 + 4);
        assert_eq!(child.num_faces(), 4);
        assert_eq!(child.num_edges(), 12);
        assert_eq!(refinement.face_child_vertex(0), 0);
        assert_eq!(refinement.edge_child_vertex(0), 1);
        assert_eq!(refinement.vertex_child_vertex(0), 5);
    }

    #[test]
    fn tri_refinement_counts() {
        // A single triangle.
        let level = Level::from_face_vertices(3, &[3], &[0, 1, 2]).unwrap();
        let scheme = Scheme::new(SchemeType::Loop, Options::default());
        let (child, _) = Refinement::refine(&level, &scheme).unwrap();

        assert_eq!(child.num_vertices(), 3 + 3);
        assert_eq!(child.num_faces(), 4);
        assert_eq!(child.num_edges(), 9);
    }

    #[test]
    fn sharpness_propagates() {
        let level = {
            let mut l = Level::from_face_vertices(4, &[4], &[0, 1, 2, 3]).unwrap();
            let e = l.find_edge(0, 1).unwrap();
            l.set_edge_sharpness(e as usize, 2.0);
            l.set_vertex_sharpness(0, 3.0);
            l
        };
        // Boundary interpolation "None" so that the quad's (boundary) edges
        // keep their explicit sharpness instead of being pinned infinite.
        let options = Options::default()
            .with_vtx_boundary_interpolation(crate::sdc::VtxBoundaryInterpolation::None);
        let scheme = Scheme::new(SchemeType::Catmark, options);
        let (child, refinement) = Refinement::refine(&level, &scheme).unwrap();

        // Child vertex of vertex 0 has sharpness 2.
        assert_eq!(
            child.vertex_sharpness(refinement.vertex_child_vertex(0) as usize),
            2.0
        );
        // Both halves of the sharp edge have sharpness 1 (uniform creasing);
        // find them between the edge midpoint and the corner children.
        let e01 = level.find_edge(0, 1).unwrap() as usize;
        let mid = refinement.edge_child_vertex(e01);
        for corner in [0usize, 1] {
            let half = child
                .find_edge(refinement.vertex_child_vertex(corner), mid)
                .unwrap();
            assert_eq!(child.edge_sharpness(half as usize), 1.0);
        }
    }

    #[test]
    fn sparse_refinement_includes_one_ring() {
        // A 5x5 grid of quad faces (6x6 vertices); selecting only the
        // center face must include exactly its 3x3 one-ring neighborhood.
        let mut verts_per_face = Vec::new();
        let mut face_verts: Vec<Index> = Vec::new();
        for j in 0..5u32 {
            for i in 0..5u32 {
                verts_per_face.push(4);
                let v = j * 6 + i;
                face_verts.extend_from_slice(&[v, v + 1, v + 7, v + 6]);
            }
        }
        let level = Level::from_face_vertices(36, &verts_per_face, &face_verts).unwrap();
        let scheme = Scheme::new(SchemeType::Catmark, Options::default());

        let mut selected = vec![false; 25];
        selected[12] = true; // center face of the 5x5 face grid
        let (child, refinement) =
            Refinement::refine_selected(&level, &scheme, Some(&selected)).unwrap();

        // Included: the 3x3 block of faces around face 12 -> 9 faces, each
        // yielding 4 children.
        assert_eq!(child.num_faces(), 36);
        // 9 face children + 24 edge children + 16 vertex children.
        assert_eq!(child.num_vertices(), 9 + 24 + 16);
        // Faces outside the block have no children.
        assert_eq!(refinement.face_child_vertex(0), crate::INDEX_INVALID);
        assert_ne!(refinement.face_child_vertex(12), crate::INDEX_INVALID);
    }
}
