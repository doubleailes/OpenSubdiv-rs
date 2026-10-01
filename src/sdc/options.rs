//! Standard subdivision options (port of `opensubdiv/sdc/options.h`).

/// Boundary interpolation rules for vertex (position) data
/// (`Sdc::Options::VtxBoundaryInterpolation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VtxBoundaryInterpolation {
    /// Smooth boundaries: no boundary edge or vertex is sharpened, so
    /// boundaries of the mesh "shrink" toward the interior. Note that
    /// (matching OpenSubdiv) child vertices *are* still generated for
    /// boundary components.
    None,
    /// Sharpen boundary edges only ("edge only"): boundaries behave as
    /// creases; boundary corners are rounded.
    #[default]
    EdgeOnly,
    /// Sharpen boundary edges *and* pin boundary corners (vertices incident
    /// to exactly one face).
    EdgeAndCorner,
}

/// Face-varying linear-interpolation rules
/// (`Sdc::Options::FVarLinearInterpolation`).
///
/// These rules govern how face-varying channels (see
/// [`FVarChannelDescriptor`](crate::far::FVarChannelDescriptor)) are refined
/// and limited by
/// [`PrimvarRefiner::interpolate_face_varying`](crate::far::PrimvarRefiner::interpolate_face_varying)
/// and
/// [`PrimvarRefiner::limit_face_varying`](crate::far::PrimvarRefiner::limit_face_varying).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FVarLinearInterpolation {
    /// Smooth everywhere the mesh is smooth.
    None,
    /// Linearly interpolate (sharpen) corners only.
    #[default]
    CornersOnly,
    /// `CornersOnly` + sharpening of junctions of three or more edges.
    CornersPlus1,
    /// `CornersPlus1` + sharpening of darts and concave corners.
    CornersPlus2,
    /// Linear interpolation along all boundary edges and corners.
    Boundaries,
    /// Linear interpolation everywhere ("bilinear" face-varying).
    All,
}

/// Method used to subdivide (semi-sharp) crease sharpness values
/// (`Sdc::Options::CreasingMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CreasingMethod {
    /// Classic Catmull-Clark semi-sharp creasing: a child edge's sharpness is
    /// its parent's sharpness reduced by one.
    #[default]
    Uniform,
    /// Chaikin creasing: sharpness of child edges is additionally averaged
    /// with the other sharp edges around the shared vertex, giving smoother
    /// decay along a crease of varying sharpness.
    Chaikin,
}

/// How triangular faces are subdivided by the Catmark scheme
/// (`Sdc::Options::TriangleSubdivision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TriangleSubdivision {
    /// The standard Catmull-Clark weights.
    #[default]
    Catmark,
    /// The "smooth triangle" weight adjustment of the original Catmull-Clark
    /// paper (`TRI_SUB_SMOOTH`): the child vertex of an interior edge with a
    /// triangle on either side takes more of its position from the incident
    /// face points (0.470 per triangle instead of 1/4) and less from the
    /// edge's end vertices, removing the pinching plain Catmark produces at
    /// triangles inside a quad mesh. Face points and vertex points are not
    /// affected, and the rule only acts at the base level: after one
    /// refinement every face is a quad.
    ///
    /// This is USD's `triangleSubdivisionRule = "smooth"`.
    Smooth,
}

/// All supported options applying to subdivision scheme rules
/// (`Sdc::Options`).
///
/// Instances are lightweight and copyable; the default value of every field
/// matches OpenSubdiv's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Options {
    /// How vertices and edges on a mesh boundary are interpolated.
    pub vtx_boundary_interpolation: VtxBoundaryInterpolation,
    /// How face-varying data is interpolated around seams and boundaries.
    pub fvar_linear_interpolation: FVarLinearInterpolation,
    /// How semi-sharp creases decay under subdivision.
    pub creasing_method: CreasingMethod,
    /// Which weights are used when subdividing triangles.
    pub triangle_subdivision: TriangleSubdivision,
}

impl Options {
    /// Options with every field set to its OpenSubdiv default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for [`VtxBoundaryInterpolation`].
    pub fn with_vtx_boundary_interpolation(mut self, v: VtxBoundaryInterpolation) -> Self {
        self.vtx_boundary_interpolation = v;
        self
    }

    /// Builder-style setter for [`FVarLinearInterpolation`].
    pub fn with_fvar_linear_interpolation(mut self, v: FVarLinearInterpolation) -> Self {
        self.fvar_linear_interpolation = v;
        self
    }

    /// Builder-style setter for [`CreasingMethod`].
    pub fn with_creasing_method(mut self, v: CreasingMethod) -> Self {
        self.creasing_method = v;
        self
    }

    /// Builder-style setter for [`TriangleSubdivision`].
    pub fn with_triangle_subdivision(mut self, v: TriangleSubdivision) -> Self {
        self.triangle_subdivision = v;
        self
    }
}
