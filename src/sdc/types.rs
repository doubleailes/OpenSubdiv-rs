//! Basic scheme types and traits (port of `opensubdiv/sdc/types.h`).

/// Enumerated type for all subdivision schemes supported by OpenSubdiv
/// (`Sdc::SchemeType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SchemeType {
    /// Bilinear subdivision: faces are split into quads and primvar data is
    /// interpolated linearly. The limit surface is the control mesh itself.
    Bilinear,
    /// Catmull-Clark subdivision — the industry standard for quad-dominant
    /// meshes. Faces of any size are supported; all child faces are quads.
    #[default]
    Catmark,
    /// Loop subdivision — for purely triangular meshes; all child faces are
    /// triangles.
    Loop,
}

/// Enumerated type for the topological split applied by a scheme
/// (`Sdc::Split`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// Used by Catmark and Bilinear: an N-sided face is split into N quads.
    ToQuads,
    /// Used by Loop: a triangle is split into 4 triangles.
    ToTris,
    /// Not currently used (reserved, as in OpenSubdiv).
    Hybrid,
}

impl SchemeType {
    /// The topological split for this scheme
    /// (`Sdc::SchemeTypeTraits::GetTopologicalSplitType()`).
    pub fn topological_split_type(self) -> Split {
        match self {
            SchemeType::Bilinear | SchemeType::Catmark => Split::ToQuads,
            SchemeType::Loop => Split::ToTris,
        }
    }

    /// Number of vertices of a "regular" face for the scheme
    /// (`Sdc::SchemeTypeTraits::GetRegularFaceSize()`).
    pub fn regular_face_size(self) -> usize {
        match self {
            SchemeType::Bilinear | SchemeType::Catmark => 4,
            SchemeType::Loop => 3,
        }
    }

    /// Valence of a "regular" interior vertex for the scheme
    /// (`Sdc::SchemeTypeTraits::GetRegularVertexValence()`).
    pub fn regular_vertex_valence(self) -> usize {
        match self {
            SchemeType::Bilinear | SchemeType::Catmark => 4,
            SchemeType::Loop => 6,
        }
    }

    /// Whether local neighborhoods of the scheme's limit surface are purely
    /// linear (true only for Bilinear).
    pub fn is_linear(self) -> bool {
        matches!(self, SchemeType::Bilinear)
    }

    /// The name of the scheme, as in `Sdc::SchemeTypeTraits::GetName()`.
    pub fn name(self) -> &'static str {
        match self {
            SchemeType::Bilinear => "Bilinear",
            SchemeType::Catmark => "Catmark",
            SchemeType::Loop => "Loop",
        }
    }
}
