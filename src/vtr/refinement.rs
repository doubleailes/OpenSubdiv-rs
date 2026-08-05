//! One step of uniform refinement between two levels (port of
//! `opensubdiv/vtr/refinement.h` — specifically the uniform
//! `QuadRefinement` / `TriRefinement` paths).

use super::level::{Level, TopologyError};
use crate::sdc::{Crease, Scheme, Split};
use crate::{Index, INDEX_INVALID};

/// Mapping between a parent [`Level`] and the child [`Level`] produced by one
/// step of uniform refinement (`Vtr::internal::Refinement`).
///
/// Child vertices are ordered by their parent component type: first the
/// child vertices of *faces* (absent for triangular splits), then those of
/// *edges*, then those of *vertices*.
#[derive(Debug, Clone)]
pub struct Refinement {
    split: Split,
    /// Child vertex originating from each parent face (`INDEX_INVALID` for
    /// triangular splits, which generate no face vertices).
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
    /// Loop); creasing options govern how sharpness is subdivided; boundary
    /// rules are re-applied to the child level.
    pub fn refine(parent: &Level, scheme: &Scheme) -> Result<(Level, Refinement), TopologyError> {
        match scheme.scheme_type().topological_split_type() {
            Split::ToQuads => Self::refine_quads(parent, scheme),
            Split::ToTris => Self::refine_tris(parent, scheme),
            Split::Hybrid => unimplemented!("hybrid splits are not used by any scheme"),
        }
    }

    fn refine_quads(parent: &Level, scheme: &Scheme) -> Result<(Level, Refinement), TopologyError> {
        let (num_faces, num_edges, num_verts) = (
            parent.num_faces(),
            parent.num_edges(),
            parent.num_vertices(),
        );

        // Child vertex ordering: faces, then edges, then vertices.
        let face_child_vert: Vec<Index> = (0..num_faces as Index).collect();
        let edge_child_vert: Vec<Index> = (0..num_edges as Index)
            .map(|e| num_faces as Index + e)
            .collect();
        let vert_child_vert: Vec<Index> = (0..num_verts as Index)
            .map(|v| (num_faces + num_edges) as Index + v)
            .collect();
        let num_child_verts = num_faces + num_edges + num_verts;

        // An N-sided parent face yields N child quads. Child face `i` is the
        // quad at corner `i`:  [ corner, leading edge, center, trailing edge ].
        let num_child_faces = parent.num_face_vertices_total();
        let mut verts_per_face = Vec::with_capacity(num_child_faces);
        let mut child_face_verts = Vec::with_capacity(num_child_faces * 4);
        let mut child_face_parent = Vec::with_capacity(num_child_faces);
        for (f, &center) in face_child_vert.iter().enumerate() {
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

    fn refine_tris(parent: &Level, scheme: &Scheme) -> Result<(Level, Refinement), TopologyError> {
        let (num_faces, num_edges, num_verts) = (
            parent.num_faces(),
            parent.num_edges(),
            parent.num_vertices(),
        );

        // No face child-vertices: ordering is edges, then vertices.
        let face_child_vert: Vec<Index> = vec![INDEX_INVALID; num_faces];
        let edge_child_vert: Vec<Index> = (0..num_edges as Index).collect();
        let vert_child_vert: Vec<Index> = (0..num_verts as Index)
            .map(|v| num_edges as Index + v)
            .collect();
        let num_child_verts = num_edges + num_verts;

        // Each triangle yields 4 children: three corner triangles followed by
        // the central triangle.
        let num_child_faces = num_faces * 4;
        let mut verts_per_face = Vec::with_capacity(num_child_faces);
        let mut child_face_verts = Vec::with_capacity(num_child_faces * 3);
        let mut child_face_parent = Vec::with_capacity(num_child_faces);
        for f in 0..num_faces {
            let fv = parent.face_vertices(f);
            let fe = parent.face_edges(f);
            debug_assert_eq!(fv.len(), 3, "tri split requires a triangulated level");
            for i in 0..3 {
                let prev = (i + 2) % 3;
                verts_per_face.push(3);
                child_face_verts.extend_from_slice(&[
                    vert_child_vert[fv[i] as usize],
                    edge_child_vert[fe[i] as usize],
                    edge_child_vert[fe[prev] as usize],
                ]);
                child_face_parent.push(f as Index);
            }
            verts_per_face.push(3);
            child_face_verts.extend_from_slice(&[
                edge_child_vert[fe[0] as usize],
                edge_child_vert[fe[1] as usize],
                edge_child_vert[fe[2] as usize],
            ]);
            child_face_parent.push(f as Index);
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
    /// propagate subdivided sharpness, hole tags and boundary sharpening.
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
            let s = crease.subdivide_vertex_sharpness(parent.vertex_sharpness(v));
            if Crease::is_sharp(s) {
                child.set_vertex_sharpness(self.vert_child_vert[v] as usize, s);
            }
        }

        // Edge sharpness: the two child (half) edges of each sharp parent
        // edge get the sharpness subdivided at their respective end vertex
        // (supporting the Chaikin rule); edges interior to a parent face are
        // smooth.
        let mut incident = Vec::new();
        for e in 0..parent.num_edges() {
            let sharpness = parent.edge_sharpness(e);
            if !Crease::is_sharp(sharpness) {
                continue;
            }
            let mid = self.edge_child_vert[e];
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

        // Re-apply boundary rules at the child level (idempotent for
        // boundaries inherited from the parent).
        child.sharpen_boundaries(crease);

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
}
