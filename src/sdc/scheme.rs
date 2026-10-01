//! Scheme-specific subdivision and limit masks.
//!
//! Port of `opensubdiv/sdc/scheme.h` and the scheme specializations in
//! `bilinearScheme.h`, `catmarkScheme.h` and `loopScheme.h`.
//!
//! A *mask* is the set of stencil weights used to compute a refined (child)
//! vertex — or a limit point — from vertices in its parent neighborhood:
//!
//! * **face-vertex** masks weight the vertices of a parent face,
//! * **edge-vertex** masks weight the two end vertices of a parent edge plus
//!   one weight per incident face,
//! * **vertex-vertex** masks weight the parent vertex itself, the vertex at
//!   the opposite end of each incident edge, and one weight per incident
//!   face.
//!
//! As in OpenSubdiv, per-face weights are flagged as being either for the
//! *face centers* (the child vertex at the face's center — Catmark) or for
//! the vertex of the face *opposite* the subject component (Loop edge masks,
//! Catmark limit masks). See [`EdgeVertexMask::face_weights_for_face_centers`].

use super::crease::{Crease, Rule};
use super::options::{Options, TriangleSubdivision};
use super::types::SchemeType;

/// The per-face weight the Catmark "smooth triangle" rule gives an incident
/// triangle in an edge-vertex mask (`CATMARK_SMOOTH_TRI_EDGE_WEIGHT`), in
/// place of the standard 1/4.
const CATMARK_SMOOTH_TRI_EDGE_WEIGHT: f32 = 0.470;

/// Neighborhood description of an edge, as needed to compute its child
/// vertex mask.
#[derive(Debug, Clone, Copy)]
pub struct EdgeNeighborhood {
    /// Effective sharpness of the edge (boundary sharpening already applied).
    pub sharpness: f32,
    /// Number of faces incident the edge (1 for a boundary edge, 2 for a
    /// manifold interior edge).
    pub num_faces: usize,
    /// Number of vertices of each of the first two incident faces
    /// (`GetNumVerticesPerFace`), in the order of the mask's face weights.
    /// Only consulted by the Catmark scheme under
    /// [`TriangleSubdivision::Smooth`], and only for edges with exactly two
    /// incident faces; entries without a face are ignored and may be zero.
    pub face_vertex_counts: [usize; 2],
}

/// Neighborhood description of a vertex, as needed to compute its child
/// vertex mask (and its limit mask).
#[derive(Debug, Clone, Copy)]
pub struct VertexNeighborhood<'a> {
    /// Effective sharpness of the vertex at the parent level.
    pub sharpness: f32,
    /// Sharpness the vertex will have at the child level.
    pub child_sharpness: f32,
    /// Effective sharpness of each incident edge at the parent level.
    pub edge_sharpness: &'a [f32],
    /// Sharpness each incident edge will have at the child level (same
    /// order). May be empty when computing limit masks.
    pub child_edge_sharpness: &'a [f32],
    /// Number of incident faces.
    pub num_faces: usize,
}

/// Mask for a child vertex of a face: a uniform average of the face's
/// vertices for all schemes that generate face child-vertices.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaceVertexMask {
    /// The single weight applied to every vertex of the parent face.
    pub vertex_weight: f32,
}

/// Mask for the child vertex of an edge.
#[derive(Debug, Clone, Default)]
pub struct EdgeVertexMask {
    /// Weights for the two end vertices of the edge.
    pub vertex_weights: [f32; 2],
    /// One weight per incident face (empty when the mask is a pure crease).
    pub face_weights: Vec<f32>,
    /// When true (Catmark/Bilinear), `face_weights[i]` applies to the *child
    /// vertex at the center* of incident face `i`. When false (Loop),
    /// `face_weights[i]` applies to the vertex of incident face `i` opposite
    /// the edge.
    pub face_weights_for_face_centers: bool,
}

/// Mask for the child vertex of a vertex, or for its limit point.
#[derive(Debug, Clone, Default)]
pub struct VertexVertexMask {
    /// Weight applied to the parent vertex itself.
    pub vertex_weight: f32,
    /// One weight per incident edge, applied to the vertex at the *opposite
    /// end* of that edge.
    pub edge_weights: Vec<f32>,
    /// One weight per incident face. For refinement masks these apply to the
    /// *child vertex at the center* of the face
    /// (`face_weights_for_face_centers == true`); for Catmark limit masks
    /// they apply to the vertex of the (quad) face *opposite* the subject
    /// vertex.
    pub face_weights: Vec<f32>,
    /// See [`EdgeVertexMask::face_weights_for_face_centers`].
    pub face_weights_for_face_centers: bool,
}

impl VertexVertexMask {
    fn clear_for(&mut self, num_edges: usize, num_faces: usize) {
        self.vertex_weight = 0.0;
        self.edge_weights.clear();
        self.edge_weights.resize(num_edges, 0.0);
        self.face_weights.clear();
        self.face_weights.resize(num_faces, 0.0);
        self.face_weights_for_face_centers = true;
    }
}

/// Scheme-specific application of the subdivision rules
/// (`Sdc::Scheme<SCHEME_TYPE>`).
#[derive(Debug, Clone, Copy)]
pub struct Scheme {
    scheme_type: SchemeType,
    crease: Crease,
    triangle_subdivision: TriangleSubdivision,
}

impl Scheme {
    /// Create the rules for `scheme_type` under the given subdivision
    /// `options`.
    pub fn new(scheme_type: SchemeType, options: Options) -> Self {
        Self {
            scheme_type,
            crease: Crease::new(options),
            triangle_subdivision: options.triangle_subdivision,
        }
    }

    /// The subdivision scheme these rules apply.
    pub fn scheme_type(&self) -> SchemeType {
        self.scheme_type
    }

    /// The triangle-subdivision rule these rules apply (only meaningful for
    /// the Catmark scheme).
    pub fn triangle_subdivision(&self) -> TriangleSubdivision {
        self.triangle_subdivision
    }

    /// The creasing queries derived from this scheme's options.
    pub fn crease(&self) -> &Crease {
        &self.crease
    }

    // ------------------------------------------------------------------
    //  Face-vertex masks
    // ------------------------------------------------------------------

    /// Mask for the child vertex of a face with `num_vertices` vertices
    /// (`Scheme::ComputeFaceVertexMask`). Identical for all schemes that
    /// split faces to quads: the centroid of the face.
    pub fn compute_face_vertex_mask(&self, num_vertices: usize) -> FaceVertexMask {
        FaceVertexMask {
            vertex_weight: 1.0 / num_vertices as f32,
        }
    }

    // ------------------------------------------------------------------
    //  Edge-vertex masks
    // ------------------------------------------------------------------

    /// Mask for the child vertex of an edge (`Scheme::ComputeEdgeVertexMask`).
    pub fn compute_edge_vertex_mask(&self, edge: &EdgeNeighborhood, mask: &mut EdgeVertexMask) {
        mask.face_weights.clear();
        mask.face_weights_for_face_centers = !matches!(self.scheme_type, SchemeType::Loop);

        if self.scheme_type == SchemeType::Bilinear {
            mask.vertex_weights = [0.5, 0.5];
            return;
        }

        // The transitional crease weight blends the sharp (crease) mask with
        // the smooth mask for semi-sharp edges:
        let crease_weight = edge.sharpness.clamp(0.0, 1.0);

        let smooth_supported = match self.scheme_type {
            SchemeType::Catmark => edge.num_faces > 0,
            SchemeType::Loop => edge.num_faces == 2,
            SchemeType::Bilinear => unreachable!(),
        };

        if crease_weight >= 1.0 || !smooth_supported {
            mask.vertex_weights = [0.5, 0.5];
            return;
        }

        let smooth_weight = 1.0 - crease_weight;
        match self.scheme_type {
            SchemeType::Catmark => {
                // Smooth mask: 1/4 to each end vertex, the remaining 1/2
                // distributed over the incident faces' child vertices —
                // unless the smooth-triangle rule adjusts the split.
                let (v_weight, f_weight) = self.catmark_smooth_edge_weights(edge);
                let vw = crease_weight * 0.5 + smooth_weight * v_weight;
                mask.vertex_weights = [vw, vw];
                mask.face_weights
                    .resize(edge.num_faces, smooth_weight * f_weight);
            }
            SchemeType::Loop => {
                // Smooth mask: 3/8 to each end vertex, 1/8 to the vertex
                // opposite the edge in each of the two incident triangles.
                let vw = crease_weight * 0.5 + smooth_weight * 0.375;
                mask.vertex_weights = [vw, vw];
                mask.face_weights.resize(2, smooth_weight * 0.125);
            }
            SchemeType::Bilinear => unreachable!(),
        }
    }

    /// The (end-vertex, per-face) weights of the smooth Catmark edge-vertex
    /// mask (`Scheme<SCHEME_CATMARK>::assignSmoothMaskForEdge`).
    ///
    /// Under [`TriangleSubdivision::Smooth`], an interior edge with a
    /// triangle on either side uses the "smooth triangle" adjustment of
    /// Catmull and Clark's original paper: each incident triangle
    /// contributes [`CATMARK_SMOOTH_TRI_EDGE_WEIGHT`] instead of 1/4, the two
    /// face weights are averaged, and the end vertices share the remainder.
    /// This removes the pinching plain Catmark produces at triangles inside
    /// a quad mesh. The order of operations mirrors OpenSubdiv (and Hbr) so
    /// the weights match bit for bit.
    fn catmark_smooth_edge_weights(&self, edge: &EdgeNeighborhood) -> (f32, f32) {
        if self.triangle_subdivision == TriangleSubdivision::Smooth && edge.num_faces == 2 {
            let [face0_is_tri, face1_is_tri] = edge.face_vertex_counts.map(|n| n == 3);
            if face0_is_tri || face1_is_tri {
                let tri_weight = |is_tri: bool| {
                    if is_tri {
                        CATMARK_SMOOTH_TRI_EDGE_WEIGHT
                    } else {
                        0.25
                    }
                };
                let f_weight = 0.5 * (tri_weight(face0_is_tri) + tri_weight(face1_is_tri));
                let v_weight = 0.5 * (1.0 - 2.0 * f_weight);
                return (v_weight, f_weight);
            }
        }
        (0.25, 0.5 / edge.num_faces as f32)
    }

    // ------------------------------------------------------------------
    //  Vertex-vertex masks
    // ------------------------------------------------------------------

    /// Mask for the child vertex of a vertex
    /// (`Scheme::ComputeVertexVertexMask`).
    ///
    /// Handles the smooth/dart, crease and corner rules, including the
    /// fractional blending of masks across a refinement step for semi-sharp
    /// features.
    pub fn compute_vertex_vertex_mask(
        &self,
        vertex: &VertexNeighborhood<'_>,
        mask: &mut VertexVertexMask,
    ) {
        let num_edges = vertex.edge_sharpness.len();
        mask.clear_for(num_edges, vertex.num_faces);

        if self.scheme_type == SchemeType::Bilinear {
            mask.vertex_weight = 1.0;
            return;
        }

        let sharp_edges_parent = count_sharp(vertex.edge_sharpness);
        let parent_rule = self
            .crease
            .determine_vertex_vertex_rule(vertex.sharpness, sharp_edges_parent);

        // Fast path: no sharp features at all.
        if parent_rule == Rule::Smooth || parent_rule == Rule::Dart {
            self.assign_smooth_mask_for_vertex(vertex, 1.0, mask);
            return;
        }

        let sharp_edges_child = count_sharp(vertex.child_edge_sharpness);
        let child_rule = self
            .crease
            .determine_vertex_vertex_rule(vertex.child_sharpness, sharp_edges_child);

        if parent_rule == child_rule {
            self.assign_rule_mask(parent_rule, vertex, ByLevel::Parent, 1.0, mask);
            return;
        }

        // Transitional (semi-sharp) case: blend the parent-rule mask with the
        // child-rule mask by the fractional weight.
        let frac = self.crease.compute_fractional_weight_at_vertex(
            vertex.sharpness,
            vertex.child_sharpness,
            vertex.edge_sharpness,
            vertex.child_edge_sharpness,
        );
        self.assign_rule_mask(parent_rule, vertex, ByLevel::Parent, frac, mask);
        self.assign_rule_mask(child_rule, vertex, ByLevel::Child, 1.0 - frac, mask);
    }

    fn assign_rule_mask(
        &self,
        rule: Rule,
        vertex: &VertexNeighborhood<'_>,
        level: ByLevel,
        scale: f32,
        mask: &mut VertexVertexMask,
    ) {
        match rule {
            Rule::Smooth | Rule::Dart => self.assign_smooth_mask_for_vertex(vertex, scale, mask),
            Rule::Crease => {
                let sharpness = match level {
                    ByLevel::Parent => vertex.edge_sharpness,
                    ByLevel::Child => vertex.child_edge_sharpness,
                };
                assign_crease_mask_for_vertex(sharpness, scale, mask);
            }
            Rule::Corner => mask.vertex_weight += scale,
            Rule::Unknown => unreachable!(),
        }
    }

    fn assign_smooth_mask_for_vertex(
        &self,
        vertex: &VertexNeighborhood<'_>,
        scale: f32,
        mask: &mut VertexVertexMask,
    ) {
        let valence = vertex.edge_sharpness.len();
        if valence == 0 {
            mask.vertex_weight += scale;
            return;
        }
        match self.scheme_type {
            SchemeType::Catmark => {
                // V' = (n-2)/n V + 1/n^2 Sum(E_i) + 1/n^2 Sum(F_i)
                // with E_i the opposite end vertices of the incident edges
                // and F_i the child vertices of the incident faces.
                let n = valence as f32;
                let e_weight = 1.0 / (n * n);
                let f_weight = e_weight;
                let v_weight =
                    1.0 - (valence as f32) * e_weight - (vertex.num_faces as f32) * f_weight;
                mask.vertex_weight += scale * v_weight;
                for w in mask.edge_weights.iter_mut() {
                    *w += scale * e_weight;
                }
                for w in mask.face_weights.iter_mut() {
                    *w += scale * f_weight;
                }
            }
            SchemeType::Loop => {
                // V' = (1 - n*beta) V + beta Sum(E_i), with Warren/Loop's
                // beta = (5/8 - (3/8 + 1/4 cos(2 pi / n))^2) / n.
                let beta = loop_beta(valence);
                mask.vertex_weight += scale * (1.0 - valence as f32 * beta);
                for w in mask.edge_weights.iter_mut() {
                    *w += scale * beta;
                }
            }
            SchemeType::Bilinear => unreachable!(),
        }
    }

    // ------------------------------------------------------------------
    //  Limit masks
    // ------------------------------------------------------------------

    /// Mask evaluating the *limit position* of a vertex on the subdivision
    /// surface (`Scheme::ComputeVertexLimitMask`).
    ///
    /// For Catmark, the per-face weights apply to the vertex of each incident
    /// (quad) face diagonally *opposite* the subject vertex
    /// (`face_weights_for_face_centers == false`); the caller must therefore
    /// only invoke this on levels where all incident faces are quads (i.e.
    /// after at least one refinement for meshes with non-quad faces).
    pub fn compute_vertex_limit_mask(
        &self,
        vertex: &VertexNeighborhood<'_>,
        mask: &mut VertexVertexMask,
    ) {
        let num_edges = vertex.edge_sharpness.len();
        mask.clear_for(num_edges, vertex.num_faces);
        mask.face_weights_for_face_centers = false;

        if self.scheme_type == SchemeType::Bilinear || num_edges == 0 {
            mask.vertex_weight = 1.0;
            mask.face_weights.clear();
            mask.face_weights.resize(vertex.num_faces, 0.0);
            return;
        }

        let sharp_edges = count_sharp(vertex.edge_sharpness);
        let rule = self
            .crease
            .determine_vertex_vertex_rule(vertex.sharpness, sharp_edges);

        match rule {
            Rule::Smooth | Rule::Dart => match self.scheme_type {
                SchemeType::Catmark => {
                    // L = (n^2 V + 4 Sum(E_i) + Sum(D_i)) / (n (n + 5)),
                    // D_i the diagonally opposite vertex of incident quad i.
                    let n = num_edges as f32;
                    let f_weight = 1.0 / (n * (n + 5.0));
                    let e_weight = 4.0 * f_weight;
                    mask.vertex_weight = 1.0 - n * e_weight - vertex.num_faces as f32 * f_weight;
                    mask.edge_weights.fill(e_weight);
                    mask.face_weights.fill(f_weight);
                }
                SchemeType::Loop => {
                    // L = (omega V + Sum(E_i)) / (omega + n), omega = 3/(8 beta).
                    let beta = loop_beta(num_edges);
                    let omega = 3.0 / (8.0 * beta);
                    let e_weight = 1.0 / (omega + num_edges as f32);
                    mask.vertex_weight = omega * e_weight;
                    mask.edge_weights.fill(e_weight);
                }
                SchemeType::Bilinear => unreachable!(),
            },
            Rule::Crease => {
                // The boundary/crease limit curve is a cubic B-spline:
                // L = 2/3 V + 1/6 E_a + 1/6 E_b along the two sharp edges.
                mask.vertex_weight = 2.0 / 3.0;
                let mut assigned = 0;
                for (i, &s) in vertex.edge_sharpness.iter().enumerate() {
                    if Crease::is_sharp(s) && assigned < 2 {
                        mask.edge_weights[i] = 1.0 / 6.0;
                        assigned += 1;
                    }
                }
            }
            Rule::Corner => {
                mask.vertex_weight = 1.0;
            }
            Rule::Unknown => unreachable!(),
        }
    }
}

#[derive(Clone, Copy)]
enum ByLevel {
    Parent,
    Child,
}

fn count_sharp(sharpness: &[f32]) -> usize {
    sharpness.iter().filter(|&&s| Crease::is_sharp(s)).count()
}

fn assign_crease_mask_for_vertex(edge_sharpness: &[f32], scale: f32, mask: &mut VertexVertexMask) {
    // V' = 3/4 V + 1/8 (E_a + E_b) along the two sharpest edges.
    mask.vertex_weight += scale * 0.75;
    let mut assigned = 0;
    for (i, &s) in edge_sharpness.iter().enumerate() {
        if Crease::is_sharp(s) && assigned < 2 {
            mask.edge_weights[i] += scale * 0.125;
            assigned += 1;
        }
    }
}

/// Loop subdivision's `beta(n)` — the per-edge weight of the smooth
/// vertex-vertex mask (`Sdc::Scheme<SCHEME_LOOP>`).
fn loop_beta(valence: usize) -> f32 {
    let inv_valence = 1.0 / valence as f32;
    let t = 0.375 + 0.25 * (std::f32::consts::TAU * inv_valence).cos();
    (0.625 - t * t) * inv_valence
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme(t: SchemeType) -> Scheme {
        Scheme::new(t, Options::default())
    }

    #[test]
    fn catmark_smooth_edge_mask() {
        let s = scheme(SchemeType::Catmark);
        let mut mask = EdgeVertexMask::default();
        s.compute_edge_vertex_mask(
            &EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [4, 4],
            },
            &mut mask,
        );
        assert_eq!(mask.vertex_weights, [0.25, 0.25]);
        assert_eq!(mask.face_weights, vec![0.25, 0.25]);
        assert!(mask.face_weights_for_face_centers);
    }

    #[test]
    fn loop_smooth_edge_mask() {
        let s = scheme(SchemeType::Loop);
        let mut mask = EdgeVertexMask::default();
        s.compute_edge_vertex_mask(
            &EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [4, 4],
            },
            &mut mask,
        );
        assert_eq!(mask.vertex_weights, [0.375, 0.375]);
        assert_eq!(mask.face_weights, vec![0.125, 0.125]);
        assert!(!mask.face_weights_for_face_centers);
    }

    #[test]
    fn semi_sharp_edge_mask_blends() {
        let s = scheme(SchemeType::Catmark);
        let mut mask = EdgeVertexMask::default();
        s.compute_edge_vertex_mask(
            &EdgeNeighborhood {
                sharpness: 0.5,
                num_faces: 2,
                face_vertex_counts: [4, 4],
            },
            &mut mask,
        );
        // 0.5 * (0.5, 0.5) + 0.5 * (0.25, 0.25, f: 0.25, 0.25)
        assert_eq!(mask.vertex_weights, [0.375, 0.375]);
        assert_eq!(mask.face_weights, vec![0.125, 0.125]);
    }

    fn catmark_edge_mask(options: Options, edge: EdgeNeighborhood) -> EdgeVertexMask {
        let mut mask = EdgeVertexMask::default();
        Scheme::new(SchemeType::Catmark, options).compute_edge_vertex_mask(&edge, &mut mask);
        mask
    }

    #[test]
    fn smooth_triangle_rule_adjusts_edge_between_quad_and_triangle() {
        let smooth = Options::default().with_triangle_subdivision(TriangleSubdivision::Smooth);
        let mask = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [4, 3],
            },
        );
        // f = (1/4 + 0.470) / 2 = 0.36; v = (1 - 2 f) / 2 = 0.14.
        let f = 0.5 * (0.25 + 0.470);
        let v = 0.5 * (1.0 - 2.0 * f);
        assert_eq!(mask.vertex_weights, [v, v]);
        assert_eq!(mask.face_weights, vec![f, f]);
        assert!(mask.face_weights_for_face_centers);

        // The triangle may sit on either side of the edge.
        let flipped = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [3, 4],
            },
        );
        assert_eq!(flipped.vertex_weights, mask.vertex_weights);
        assert_eq!(flipped.face_weights, mask.face_weights);

        // Two triangles: each face takes the full 0.470.
        let both = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [3, 3],
            },
        );
        let v = 0.5 * (1.0 - 2.0 * 0.470);
        assert_eq!(both.vertex_weights, [v, v]);
        assert_eq!(both.face_weights, vec![0.470, 0.470]);
    }

    #[test]
    fn smooth_triangle_rule_leaves_other_edges_alone() {
        let smooth = Options::default().with_triangle_subdivision(TriangleSubdivision::Smooth);

        // No triangle: the standard weights.
        let quads = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [4, 4],
            },
        );
        assert_eq!(quads.vertex_weights, [0.25, 0.25]);
        assert_eq!(quads.face_weights, vec![0.25, 0.25]);

        // A smooth boundary edge of a triangle (one incident face): the
        // rule only applies to edges with two faces.
        let boundary = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 1,
                face_vertex_counts: [3, 0],
            },
        );
        assert_eq!(boundary.vertex_weights, [0.25, 0.25]);
        assert_eq!(boundary.face_weights, vec![0.5]);

        // Under the default rule, triangles get the standard weights.
        let catmark = catmark_edge_mask(
            Options::default(),
            EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [4, 3],
            },
        );
        assert_eq!(catmark.vertex_weights, [0.25, 0.25]);
        assert_eq!(catmark.face_weights, vec![0.25, 0.25]);

        // Loop ignores the option entirely.
        let mut mask = EdgeVertexMask::default();
        Scheme::new(SchemeType::Loop, smooth).compute_edge_vertex_mask(
            &EdgeNeighborhood {
                sharpness: 0.0,
                num_faces: 2,
                face_vertex_counts: [3, 3],
            },
            &mut mask,
        );
        assert_eq!(mask.vertex_weights, [0.375, 0.375]);
        assert_eq!(mask.face_weights, vec![0.125, 0.125]);
    }

    #[test]
    fn smooth_triangle_rule_blends_with_semi_sharp_crease() {
        let smooth = Options::default().with_triangle_subdivision(TriangleSubdivision::Smooth);
        let mask = catmark_edge_mask(
            smooth,
            EdgeNeighborhood {
                sharpness: 0.5,
                num_faces: 2,
                face_vertex_counts: [4, 3],
            },
        );
        // 0.5 * (0.5, 0.5) + 0.5 * (0.14, 0.14, f: 0.36, 0.36)
        let f = 0.5 * (0.25 + 0.470);
        let v = 0.5 * (1.0 - 2.0 * f);
        assert_eq!(mask.vertex_weights, [0.5 * 0.5 + 0.5 * v; 2]);
        assert_eq!(mask.face_weights, vec![0.5 * f; 2]);
    }

    #[test]
    fn catmark_smooth_vertex_mask_valence_4() {
        let s = scheme(SchemeType::Catmark);
        let mut mask = VertexVertexMask::default();
        let es = [0.0f32; 4];
        s.compute_vertex_vertex_mask(
            &VertexNeighborhood {
                sharpness: 0.0,
                child_sharpness: 0.0,
                edge_sharpness: &es,
                child_edge_sharpness: &es,
                num_faces: 4,
            },
            &mut mask,
        );
        assert!((mask.vertex_weight - 0.5).abs() < 1e-6);
        assert_eq!(mask.edge_weights, vec![0.0625; 4]);
        assert_eq!(mask.face_weights, vec![0.0625; 4]);
    }

    #[test]
    fn loop_regular_limit_mask() {
        let s = scheme(SchemeType::Loop);
        let mut mask = VertexVertexMask::default();
        let es = [0.0f32; 6];
        s.compute_vertex_limit_mask(
            &VertexNeighborhood {
                sharpness: 0.0,
                child_sharpness: 0.0,
                edge_sharpness: &es,
                child_edge_sharpness: &es,
                num_faces: 6,
            },
            &mut mask,
        );
        // Regular Loop limit stencil: 1/2 center, 1/12 per neighbor.
        assert!((mask.vertex_weight - 0.5).abs() < 1e-5);
        for w in &mask.edge_weights {
            assert!((w - 1.0 / 12.0).abs() < 1e-5);
        }
    }

    #[test]
    fn catmark_regular_limit_mask_is_bspline() {
        let s = scheme(SchemeType::Catmark);
        let mut mask = VertexVertexMask::default();
        let es = [0.0f32; 4];
        s.compute_vertex_limit_mask(
            &VertexNeighborhood {
                sharpness: 0.0,
                child_sharpness: 0.0,
                edge_sharpness: &es,
                child_edge_sharpness: &es,
                num_faces: 4,
            },
            &mut mask,
        );
        // Bicubic B-spline evaluation at a regular vertex:
        // 16/36 center, 4/36 edge neighbors, 1/36 diagonal neighbors.
        assert!((mask.vertex_weight - 16.0 / 36.0).abs() < 1e-6);
        assert_eq!(mask.edge_weights, vec![4.0 / 36.0; 4]);
        assert_eq!(mask.face_weights, vec![1.0 / 36.0; 4]);
    }
}
