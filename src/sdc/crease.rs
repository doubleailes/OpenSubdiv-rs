//! Semi-sharp creasing support (port of `opensubdiv/sdc/crease.{h,cpp}`).

use super::options::{CreasingMethod, Options};

/// Sharpness of a completely smooth edge or vertex (`Sdc::Crease::SHARPNESS_SMOOTH`).
pub const SHARPNESS_SMOOTH: f32 = 0.0;

/// Sharpness of an infinitely sharp edge or vertex (`Sdc::Crease::SHARPNESS_INFINITE`).
///
/// As in OpenSubdiv, `10.0` is the canonical "infinite" sharpness value: an
/// infinitely sharp feature never decays under subdivision.
pub const SHARPNESS_INFINITE: f32 = 10.0;

/// The subdivision *rule* in effect at a vertex, derived from the sharpness
/// of the vertex and of its incident edges (`Sdc::Crease::Rule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rule {
    /// The rule has not been determined yet.
    Unknown,
    /// No sharp features: the full smooth mask applies.
    Smooth,
    /// Exactly one sharp edge and a smooth vertex: interpolated like Smooth,
    /// but the limit neighborhood is only C1.
    Dart,
    /// Exactly two sharp edges (or a boundary): the crease mask applies.
    Crease,
    /// A sharp vertex, or three or more sharp edges: the vertex is pinned.
    Corner,
}

/// Types, constants and queries related to semi-sharp creasing
/// (`Sdc::Crease`).
///
/// All sharpness values are clamped to the range `[SHARPNESS_SMOOTH,
/// SHARPNESS_INFINITE]`, i.e. `[0.0, 10.0]`.
#[derive(Debug, Clone, Copy)]
pub struct Crease {
    options: Options,
}

impl Crease {
    /// Create the creasing queries for the given subdivision `options`.
    pub fn new(options: Options) -> Self {
        Self { options }
    }

    /// Is the given sharpness considered sharp (greater than smooth)?
    pub fn is_sharp(sharpness: f32) -> bool {
        sharpness > SHARPNESS_SMOOTH
    }

    /// Is the given sharpness smooth?
    pub fn is_smooth(sharpness: f32) -> bool {
        sharpness <= SHARPNESS_SMOOTH
    }

    /// Is the given sharpness semi-sharp, i.e. sharp but not infinitely so?
    pub fn is_semi_sharp(sharpness: f32) -> bool {
        (SHARPNESS_SMOOTH < sharpness) && (sharpness < SHARPNESS_INFINITE)
    }

    /// Is the given sharpness infinitely sharp?
    pub fn is_infinite(sharpness: f32) -> bool {
        sharpness >= SHARPNESS_INFINITE
    }

    /// Clamp an arbitrary user-provided sharpness into the valid range.
    pub fn clamp(sharpness: f32) -> f32 {
        sharpness.clamp(SHARPNESS_SMOOTH, SHARPNESS_INFINITE)
    }

    fn is_uniform(&self) -> bool {
        self.options.creasing_method == CreasingMethod::Uniform
    }

    fn decrement_sharpness(sharpness: f32) -> f32 {
        if Self::is_infinite(sharpness) {
            SHARPNESS_INFINITE
        } else if sharpness > 1.0 {
            sharpness - 1.0
        } else {
            SHARPNESS_SMOOTH
        }
    }

    /// Sharpness of the child of a vertex with the given sharpness
    /// (`Sdc::Crease::SubdivideVertexSharpness`). Both the `Uniform` and
    /// `Chaikin` methods simply decrement vertex sharpness.
    pub fn subdivide_vertex_sharpness(&self, sharpness: f32) -> f32 {
        Self::decrement_sharpness(sharpness)
    }

    /// Uniform subdivision of an edge sharpness value
    /// (`Sdc::Crease::SubdivideUniformSharpness`).
    pub fn subdivide_uniform_sharpness(&self, sharpness: f32) -> f32 {
        Self::decrement_sharpness(sharpness)
    }

    /// Sharpness of the child (half) of an edge nearest the given end vertex
    /// (`Sdc::Crease::SubdivideEdgeSharpnessAtVertex`).
    ///
    /// * `edge_sharpness` — sharpness of the parent edge,
    /// * `incident_edge_sharpness` — sharpness of *all* edges incident the
    ///   end vertex (including the parent edge itself, whose exact value must
    ///   appear in the slice for the Chaikin average to exclude it).
    pub fn subdivide_edge_sharpness_at_vertex(
        &self,
        edge_sharpness: f32,
        incident_edge_sharpness: &[f32],
    ) -> f32 {
        if self.is_uniform() || incident_edge_sharpness.len() < 2 {
            return Self::decrement_sharpness(edge_sharpness);
        }
        if Self::is_infinite(edge_sharpness) {
            return SHARPNESS_INFINITE;
        }
        if Self::is_smooth(edge_sharpness) {
            return SHARPNESS_SMOOTH;
        }

        // Chaikin creasing: blend the edge's sharpness with the average
        // sharpness of the *other* semi-sharp edges around the shared vertex
        // (3/4 self + 1/4 neighbor average), then decrement. Infinitely sharp
        // neighbors (boundary edges, infinite creases, non-manifold edges) do
        // not take part in the average; without semi-sharp neighbors the
        // sharpness is simply decremented.
        let mut sharp_sum = 0.0f32;
        let mut sharp_count = 0u32;
        for &s in incident_edge_sharpness {
            if Self::is_semi_sharp(s) {
                sharp_sum += s;
                sharp_count += 1;
            }
        }

        let blended = if sharp_count > 1 {
            // Exclude the subject edge itself from the neighbor average:
            let avg = (sharp_sum - edge_sharpness) / (sharp_count - 1) as f32;
            0.75 * edge_sharpness + 0.25 * avg
        } else {
            edge_sharpness
        };
        (blended - 1.0).max(SHARPNESS_SMOOTH)
    }

    /// Determine the [`Rule`] at a vertex from its own sharpness and the
    /// number of sharp edges incident to it
    /// (`Sdc::Crease::DetermineVertexVertexRule`).
    pub fn determine_vertex_vertex_rule(
        &self,
        vertex_sharpness: f32,
        sharp_edge_count: usize,
    ) -> Rule {
        if Self::is_sharp(vertex_sharpness) {
            return Rule::Corner;
        }
        match sharp_edge_count {
            0 => Rule::Smooth,
            1 => Rule::Dart,
            2 => Rule::Crease,
            _ => Rule::Corner,
        }
    }

    /// Transitional weight for blending the mask of the parent rule with the
    /// mask of the child rule when semi-sharp features decay across a
    /// refinement step (`Sdc::Crease::ComputeFractionalWeightAtVertex`).
    ///
    /// The result is in `[0, 1]`: `1` means the parent rule fully applies,
    /// `0` means the child rule fully applies. Matching OpenSubdiv, the
    /// weight is the average fractional sharpness of the *transitional*
    /// features — features that are sharp in the parent but smooth in the
    /// child.
    pub fn compute_fractional_weight_at_vertex(
        &self,
        vertex_sharpness: f32,
        child_vertex_sharpness: f32,
        incident_edge_sharpness: &[f32],
        child_edge_sharpness: &[f32],
    ) -> f32 {
        debug_assert_eq!(incident_edge_sharpness.len(), child_edge_sharpness.len());

        let mut weight_sum = 0.0f32;
        let mut count = 0u32;

        if Self::is_sharp(vertex_sharpness) && Self::is_smooth(child_vertex_sharpness) {
            weight_sum += vertex_sharpness.min(1.0);
            count += 1;
        }
        for (&s, &cs) in incident_edge_sharpness.iter().zip(child_edge_sharpness) {
            if Self::is_sharp(s) && Self::is_smooth(cs) {
                weight_sum += s.min(1.0);
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            weight_sum / count as f32
        }
    }

    /// Sharpen a boundary edge or boundary vertex according to the
    /// vertex-boundary-interpolation option. Returns the effective sharpness.
    pub(crate) fn sharpen_boundary_edge(&self, sharpness: f32) -> f32 {
        use super::options::VtxBoundaryInterpolation::*;
        match self.options.vtx_boundary_interpolation {
            None => sharpness,
            EdgeOnly | EdgeAndCorner => SHARPNESS_INFINITE,
        }
    }

    pub(crate) fn sharpen_boundary_corner_vertex(&self, sharpness: f32) -> f32 {
        use super::options::VtxBoundaryInterpolation::*;
        match self.options.vtx_boundary_interpolation {
            None | EdgeOnly => sharpness,
            EdgeAndCorner => SHARPNESS_INFINITE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_subdivision_decrements() {
        let c = Crease::new(Options::default());
        assert_eq!(c.subdivide_vertex_sharpness(2.5), 1.5);
        assert_eq!(c.subdivide_vertex_sharpness(0.5), 0.0);
        assert_eq!(
            c.subdivide_vertex_sharpness(SHARPNESS_INFINITE),
            SHARPNESS_INFINITE
        );
    }

    #[test]
    fn chaikin_averages_neighbors() {
        let opts = Options::default().with_creasing_method(CreasingMethod::Chaikin);
        let c = Crease::new(opts);
        // Edge of sharpness 2 meeting one other sharp edge (sharpness 4) and
        // two smooth edges: 0.75*2 + 0.25*4 - 1 = 1.5
        let s = c.subdivide_edge_sharpness_at_vertex(2.0, &[2.0, 4.0, 0.0, 0.0]);
        assert!((s - 1.5).abs() < 1e-6);
    }

    #[test]
    fn chaikin_ignores_infinitely_sharp_neighbors() {
        let opts = Options::default().with_creasing_method(CreasingMethod::Chaikin);
        let c = Crease::new(opts);
        // Edge of sharpness 2 at a boundary vertex whose two other edges are
        // infinitely sharp boundary edges: no semi-sharp neighbor, so the
        // sharpness is simply decremented (2 - 1 = 1), as in OpenSubdiv.
        let s = c.subdivide_edge_sharpness_at_vertex(
            2.0,
            &[2.0, SHARPNESS_INFINITE, SHARPNESS_INFINITE],
        );
        assert!((s - 1.0).abs() < 1e-6);
        // Infinite neighbors are skipped, semi-sharp ones still average:
        // 0.75*2 + 0.25*4 - 1 = 1.5
        let s = c.subdivide_edge_sharpness_at_vertex(
            2.0,
            &[2.0, SHARPNESS_INFINITE, 4.0, 0.0, SHARPNESS_INFINITE],
        );
        assert!((s - 1.5).abs() < 1e-6);
    }

    #[test]
    fn rules() {
        let c = Crease::new(Options::default());
        assert_eq!(c.determine_vertex_vertex_rule(0.0, 0), Rule::Smooth);
        assert_eq!(c.determine_vertex_vertex_rule(0.0, 1), Rule::Dart);
        assert_eq!(c.determine_vertex_vertex_rule(0.0, 2), Rule::Crease);
        assert_eq!(c.determine_vertex_vertex_rule(0.0, 3), Rule::Corner);
        assert_eq!(c.determine_vertex_vertex_rule(1.0, 0), Rule::Corner);
    }

    #[test]
    fn fractional_weight() {
        let c = Crease::new(Options::default());
        // A single semi-sharp crease (two edges of sharpness 0.5) decaying to
        // smooth: fractional weight is 0.5.
        let w = c.compute_fractional_weight_at_vertex(0.0, 0.0, &[0.5, 0.5, 0.0], &[0.0, 0.0, 0.0]);
        assert!((w - 0.5).abs() < 1e-6);
    }
}
