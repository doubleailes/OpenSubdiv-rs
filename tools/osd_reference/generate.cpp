//
// Generates the OpenSubdiv reference data read by tests/upstream_regression.rs.
//
// For every shape of OpenSubdiv's regression catalogue (the bfr_evaluate
// list), it writes two files into the output directory:
//
//   <shape>.mesh  the base mesh exactly as Far::TopologyRefinerFactory<Shape>
//                 sees it: scheme, Sdc options, positions, face-vertices,
//                 creases, corners, holes and the UV channel.
//   <shape>.ref   reference results computed by OpenSubdiv:
//                 - uniform refinement: per-level component counts, and a
//                   sample of face corners at the last level (refined
//                   position, limit position, UV and UV limit);
//                 - adaptive patch tables (Gregory-basis end caps,
//                   useInfSharpPatch on): patch counts per type and a
//                   sample of evaluations (position, du, dv, and UV with
//                   its derivatives when face-varying tables are built).
//
// Samples are chosen deterministically. Corners at irregular vertices and
// evaluations on end-cap patches are kept preferentially, as that is where
// implementations diverge; the rest are a regular stride over the mesh.
//
// See README.md next to this file for how to build and run it.
//

#include <regression/bfr_evaluate/init_shapes_all.h>
#include <regression/common/far_utils.h>

#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/version.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace OpenSubdiv;
using Far::TopologyRefiner;

namespace {

// Sample budgets per shape.
const int kIrregularCorners = 96;
const int kStrideCorners = 96;
const int kIrregularSamples = 48;
const int kStrideSamples = 48;
// Uniform refinement goes as deep as this many faces allow (levels 1 to 3).
const int kMaxUniformFaces = 20000;

struct V3 {
    float p[3];
    void Clear() { p[0] = p[1] = p[2] = 0.0f; }
    void AddWithWeight(V3 const &s, float w) {
        for (int i = 0; i < 3; ++i) p[i] += w * s.p[i];
    }
};

struct V2 {
    float p[2];
    void Clear() { p[0] = p[1] = 0.0f; }
    void AddWithWeight(V2 const &s, float w) {
        for (int i = 0; i < 2; ++i) p[i] += w * s.p[i];
    }
};

TopologyRefiner *makeRefiner(Shape const &shape) {
    typedef Far::TopologyRefinerFactory<Shape> Factory;
    return Factory::Create(shape, Factory::Options(GetSdcType(shape), GetSdcOptions(shape)));
}

std::vector<V3> basePositions(Shape const &shape, int n) {
    std::vector<V3> pos(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) pos[i].p[k] = shape.verts[3 * i + k];
    return pos;
}

std::vector<V2> baseUVs(Shape const &shape, int n) {
    std::vector<V2> uv(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 2; ++k) uv[i].p[k] = shape.uvs[2 * i + k];
    return uv;
}

// Keep up to `irregular` of the flagged indices and `stride` of the others,
// both evenly spaced.
std::vector<int> pick(std::vector<bool> const &flagged, int irregular, int stride) {
    std::vector<int> a, b, out;
    for (int i = 0; i < (int)flagged.size(); ++i) (flagged[i] ? a : b).push_back(i);
    for (std::vector<int> *v : {&a, &b}) {
        int budget = (v == &a) ? irregular : stride;
        int n = (int)v->size();
        int take = std::min(n, budget);
        for (int k = 0; k < take; ++k) out.push_back((*v)[(long)k * n / take]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

const char *patchTypeName(Far::PatchDescriptor::Type t) {
    switch (t) {
    case Far::PatchDescriptor::QUADS: return "QUADS";
    case Far::PatchDescriptor::TRIANGLES: return "TRIANGLES";
    case Far::PatchDescriptor::LOOP: return "LOOP";
    case Far::PatchDescriptor::REGULAR: return "REGULAR";
    case Far::PatchDescriptor::GREGORY_BASIS: return "GREGORY_BASIS";
    case Far::PatchDescriptor::GREGORY_TRIANGLE: return "GREGORY_TRIANGLE";
    default: return "OTHER";
    }
}

void writeMesh(FILE *f, Shape const &shape, TopologyRefiner const &r) {
    Sdc::Options o = r.GetSchemeOptions();
    Far::TopologyLevel const &L = r.GetLevel(0);
    fprintf(f, "scheme %d\n", (int)r.GetSchemeType());
    fprintf(f, "options %d %d %d %d\n", (int)o.GetVtxBoundaryInterpolation(),
            (int)o.GetFVarLinearInterpolation(), (int)o.GetCreasingMethod(),
            (int)o.GetTriangleSubdivision());
    fprintf(f, "verts %d\n", L.GetNumVertices());
    for (int i = 0; i < L.GetNumVertices(); ++i)
        fprintf(f, "%.9g %.9g %.9g\n", shape.verts[3 * i], shape.verts[3 * i + 1],
                shape.verts[3 * i + 2]);
    fprintf(f, "faces %d\n", L.GetNumFaces());
    for (int i = 0; i < L.GetNumFaces(); ++i) {
        Far::ConstIndexArray fv = L.GetFaceVertices(i);
        fprintf(f, "%d", fv.size());
        for (int j = 0; j < fv.size(); ++j) fprintf(f, " %d", fv[j]);
        fprintf(f, "\n");
    }
    // Creases, corners and holes as TopologyRefinerFactory<Shape> assigns
    // them (regression/common/far_utils.h), including its indexing of the
    // per-edge crease weights.
    std::vector<std::string> creases, corners, holes;
    char buf[128];
    for (size_t i = 0; i < shape.tags.size(); ++i) {
        Shape::tag const *t = shape.tags[i];
        int nfloat = (int)t->floatargs.size();
        if (t->name == "crease") {
            for (int j = 0; j < (int)t->intargs.size() - 1; j += 2) {
                float s = std::max(0.0f, (nfloat > 1) ? t->floatargs[j] : t->floatargs[0]);
                snprintf(buf, sizeof(buf), "%d %d %.9g", t->intargs[j], t->intargs[j + 1], s);
                creases.push_back(buf);
            }
        } else if (t->name == "corner") {
            for (int j = 0; j < (int)t->intargs.size(); ++j) {
                float s = std::max(0.0f, (nfloat > 1) ? t->floatargs[j] : t->floatargs[0]);
                snprintf(buf, sizeof(buf), "%d %.9g", t->intargs[j], s);
                corners.push_back(buf);
            }
        } else if (t->name == "hole") {
            for (int j = 0; j < (int)t->intargs.size(); ++j) {
                snprintf(buf, sizeof(buf), "%d", t->intargs[j]);
                holes.push_back(buf);
            }
        }
    }
    fprintf(f, "creases %d\n", (int)creases.size());
    for (size_t i = 0; i < creases.size(); ++i) fprintf(f, "%s\n", creases[i].c_str());
    fprintf(f, "corners %d\n", (int)corners.size());
    for (size_t i = 0; i < corners.size(); ++i) fprintf(f, "%s\n", corners[i].c_str());
    fprintf(f, "holes %d\n", (int)holes.size());
    for (size_t i = 0; i < holes.size(); ++i) fprintf(f, "%s\n", holes[i].c_str());
    if (r.GetNumFVarChannels() > 0) {
        fprintf(f, "fvar %d\n", L.GetNumFVarValues(0));
        for (int i = 0; i < L.GetNumFVarValues(0); ++i)
            fprintf(f, "%.9g %.9g\n", shape.uvs[2 * i], shape.uvs[2 * i + 1]);
        for (int i = 0; i < L.GetNumFaces(); ++i) {
            Far::ConstIndexArray fv = L.GetFaceFVarValues(i, 0);
            for (int j = 0; j < fv.size(); ++j) fprintf(f, "%s%d", j ? " " : "", fv[j]);
            fprintf(f, "\n");
        }
    } else {
        fprintf(f, "fvar 0\n");
    }
}

void writeUniform(FILE *f, Shape const &shape) {
    TopologyRefiner *r = makeRefiner(shape);
    int level = 1;
    // Every refinement splits a face into about four.
    int faces = r->GetLevel(0).GetNumFaces() * 4;
    while (level < 3 && faces * 4 <= kMaxUniformFaces) {
        faces *= 4;
        ++level;
    }
    TopologyRefiner::UniformOptions uo(level);
    uo.fullTopologyInLastLevel = true;
    r->RefineUniform(uo);

    bool hasFVar = r->GetNumFVarChannels() > 0;
    fprintf(f, "uniform %d %d\n", level, (int)hasFVar);
    for (int l = 0; l <= level; ++l)
        fprintf(f, "level %d %d %d %d\n", l, r->GetLevel(l).GetNumVertices(),
                r->GetLevel(l).GetNumEdges(), r->GetLevel(l).GetNumFaces());

    Far::PrimvarRefiner pr(*r);
    std::vector<V3> src = basePositions(shape, r->GetLevel(0).GetNumVertices());
    std::vector<V2> fsrc;
    if (hasFVar) fsrc = baseUVs(shape, r->GetLevel(0).GetNumFVarValues(0));
    for (int l = 1; l <= level; ++l) {
        std::vector<V3> dst(r->GetLevel(l).GetNumVertices());
        V3 *s = &src[0], *d = &dst[0];
        pr.Interpolate(l, s, d);
        src.swap(dst);
        if (hasFVar) {
            std::vector<V2> fdst(r->GetLevel(l).GetNumFVarValues(0));
            V2 *fs = &fsrc[0], *fd = &fdst[0];
            pr.InterpolateFaceVarying(l, fs, fd, 0);
            fsrc.swap(fdst);
        }
    }
    // Position-only limit: the tangent overload overflows a buffer on some
    // non-manifold shapes (catmark_nonman_edge100) in OpenSubdiv 3.7.0.
    std::vector<V3> lim(src.size());
    {
        V3 *s = &src[0], *d = &lim[0];
        pr.Limit(s, d);
    }
    std::vector<V2> flim(fsrc.size());
    if (hasFVar) {
        V2 *s = &fsrc[0], *d = &flim[0];
        pr.LimitFaceVarying(s, d, 0);
    }

    // Flag the corners at irregular vertices: extraordinary valence,
    // boundary, or any sharpness.
    Far::TopologyLevel const &L = r->GetLevel(level);
    int regularValence = r->GetSchemeType() == Sdc::SCHEME_LOOP ? 6 : 4;
    std::vector<int> cornerFace, cornerIndex;
    std::vector<bool> flagged;
    for (int i = 0; i < L.GetNumFaces(); ++i) {
        Far::ConstIndexArray fv = L.GetFaceVertices(i);
        for (int j = 0; j < fv.size(); ++j) {
            Far::Index v = fv[j];
            bool irregular = L.GetVertexEdges(v).size() != regularValence ||
                             L.IsVertexBoundary(v) || L.GetVertexSharpness(v) > 0.0f;
            Far::ConstIndexArray ve = L.GetVertexEdges(v);
            for (int k = 0; k < ve.size() && !irregular; ++k)
                irregular = L.GetEdgeSharpness(ve[k]) > 0.0f;
            cornerFace.push_back(i);
            cornerIndex.push_back(j);
            flagged.push_back(irregular);
        }
    }
    std::vector<int> picked = pick(flagged, kIrregularCorners, kStrideCorners);
    fprintf(f, "corners %d\n", (int)picked.size());
    for (size_t n = 0; n < picked.size(); ++n) {
        int face = cornerFace[picked[n]], j = cornerIndex[picked[n]];
        Far::Index v = L.GetFaceVertices(face)[j];
        fprintf(f, "%.7g %.7g %.7g %.7g %.7g %.7g", src[v].p[0], src[v].p[1], src[v].p[2],
                lim[v].p[0], lim[v].p[1], lim[v].p[2]);
        if (hasFVar) {
            Far::Index k = L.GetFaceFVarValues(face, 0)[j];
            fprintf(f, " %.7g %.7g %.7g %.7g", fsrc[k].p[0], fsrc[k].p[1], flim[k].p[0],
                    flim[k].p[1]);
        }
        fprintf(f, "\n");
    }
    delete r;
}

void writeAdaptive(FILE *f, Shape const &shape, int isolation, bool singleCrease, bool fvar) {
    TopologyRefiner *r = makeRefiner(shape);
    bool hasFVar = fvar && r->GetNumFVarChannels() > 0;

    Far::PatchTableFactory::Options po(isolation);
    po.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    po.useInfSharpPatch = true;
    po.useSingleCreasePatch = singleCrease;
    po.generateFVarTables = hasFVar;
    po.generateFVarLegacyLinearPatches = false;
    TopologyRefiner::AdaptiveOptions ao = po.GetRefineAdaptiveOptions();
    ao.considerFVarChannels = false;
    r->RefineAdaptive(ao);
    Far::PatchTable *pt = Far::PatchTableFactory::Create(*r, po);

    // Vertex data of every level, followed by the end-cap local points.
    std::vector<V3> pos = basePositions(shape, r->GetLevel(0).GetNumVertices());
    pos.resize(r->GetNumVerticesTotal() + pt->GetNumLocalPoints());
    Far::PrimvarRefiner pr(*r);
    V3 *src = &pos[0];
    for (int l = 1; l <= r->GetMaxLevel(); ++l) {
        V3 *dst = src + r->GetLevel(l - 1).GetNumVertices();
        pr.Interpolate(l, src, dst);
        src = dst;
    }
    if (pt->GetNumLocalPoints())
        pt->ComputeLocalPointValues(&pos[0], &pos[r->GetNumVerticesTotal()]);

    std::vector<V2> uvs;
    if (hasFVar) {
        int total = 0;
        for (int l = 0; l <= r->GetMaxLevel(); ++l) total += r->GetLevel(l).GetNumFVarValues(0);
        uvs = baseUVs(shape, r->GetLevel(0).GetNumFVarValues(0));
        uvs.resize(total + pt->GetNumLocalPointsFaceVarying(0));
        V2 *s = &uvs[0];
        for (int l = 1; l <= r->GetMaxLevel(); ++l) {
            V2 *d = s + r->GetLevel(l - 1).GetNumFVarValues(0);
            pr.InterpolateFaceVarying(l, s, d, 0);
            s = d;
        }
        if (pt->GetNumLocalPointsFaceVarying(0))
            pt->ComputeLocalPointValuesFaceVarying(&uvs[0], &uvs[total], 0);
    }

    fprintf(f, "adaptive %d %d %d\n", isolation, (int)singleCrease, (int)hasFVar);
    std::map<std::string, int> counts;
    for (int a = 0; a < pt->GetNumPatchArrays(); ++a)
        counts[patchTypeName(pt->GetPatchArrayDescriptor(a).GetType())] += pt->GetNumPatches(a);
    fprintf(f, "patchcounts %d", (int)counts.size());
    for (std::map<std::string, int>::const_iterator c = counts.begin(); c != counts.end(); ++c)
        fprintf(f, " %s %d", c->first.c_str(), c->second);
    fprintf(f, "\n");

    // Candidate locations: a fixed (u, v) pattern on every ptex face.
    bool tri = r->GetSchemeType() == Sdc::SCHEME_LOOP;
    std::vector<std::pair<float, float> > pattern;
    if (tri) {
        float s[][2] = {{0.1f, 0.1f},   {0.6f, 0.2f},  {0.2f, 0.6f},  {0.33f, 0.33f},
                        {0.05f, 0.9f},  {0.9f, 0.05f}, {0.45f, 0.45f}, {0.25f, 0.05f},
                        {0.0f, 0.0f},   {0.5f, 0.0f}};
        for (size_t i = 0; i < sizeof(s) / sizeof(s[0]); ++i)
            pattern.push_back(std::make_pair(s[i][0], s[i][1]));
    } else {
        float g[] = {0.0f, 0.13f, 0.5f, 0.77f, 0.97f};
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) pattern.push_back(std::make_pair(g[i], g[j]));
    }
    Far::PatchMap pm(*pt);
    Far::PtexIndices ptex(*r);
    std::vector<int> sFace;
    std::vector<float> sU, sV;
    std::vector<bool> flagged;
    for (int face = 0; face < ptex.GetNumFaces(); ++face) {
        for (size_t k = 0; k < pattern.size(); ++k) {
            Far::PatchTable::PatchHandle const *h =
                pm.FindPatch(face, pattern[k].first, pattern[k].second);
            Far::PatchDescriptor::Type t =
                h ? pt->GetPatchDescriptor(*h).GetType() : Far::PatchDescriptor::NON_PATCH;
            sFace.push_back(face);
            sU.push_back(pattern[k].first);
            sV.push_back(pattern[k].second);
            flagged.push_back(t != Far::PatchDescriptor::REGULAR && t != Far::PatchDescriptor::LOOP);
        }
    }
    std::vector<int> picked = pick(flagged, kIrregularSamples, kStrideSamples);
    fprintf(f, "samples %d\n", (int)picked.size());
    float wP[20], wDu[20], wDv[20];
    for (size_t n = 0; n < picked.size(); ++n) {
        int face = sFace[picked[n]];
        float u = sU[picked[n]], v = sV[picked[n]];
        Far::PatchTable::PatchHandle const *h = pm.FindPatch(face, u, v);
        if (!h) {
            fprintf(f, "%d %g %g none\n", face, u, v);
            continue;
        }
        pt->EvaluateBasis(*h, u, v, wP, wDu, wDv);
        Far::ConstIndexArray cvs = pt->GetPatchVertices(*h);
        V3 P, Du, Dv;
        P.Clear();
        Du.Clear();
        Dv.Clear();
        for (int k = 0; k < cvs.size(); ++k) {
            P.AddWithWeight(pos[cvs[k]], wP[k]);
            Du.AddWithWeight(pos[cvs[k]], wDu[k]);
            Dv.AddWithWeight(pos[cvs[k]], wDv[k]);
        }
        fprintf(f, "%d %g %g %s %d %.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g", face, u, v,
                patchTypeName(pt->GetPatchDescriptor(*h).GetType()),
                (int)pt->GetPatchParam(*h).NonQuadRoot(), P.p[0], P.p[1], P.p[2], Du.p[0],
                Du.p[1], Du.p[2], Dv.p[0], Dv.p[1], Dv.p[2]);
        if (hasFVar) {
            pt->EvaluateBasisFaceVarying(*h, u, v, wP, wDu, wDv, 0, 0, 0, 0);
            Far::ConstIndexArray fvs = pt->GetPatchFVarValues(*h, 0);
            V2 T, Tu, Tv;
            T.Clear();
            Tu.Clear();
            Tv.Clear();
            for (int k = 0; k < fvs.size(); ++k) {
                T.AddWithWeight(uvs[fvs[k]], wP[k]);
                Tu.AddWithWeight(uvs[fvs[k]], wDu[k]);
                Tv.AddWithWeight(uvs[fvs[k]], wDv[k]);
            }
            fprintf(f, " %.7g %.7g %.7g %.7g %.7g %.7g", T.p[0], T.p[1], Tu.p[0], Tu.p[1],
                    Tv.p[0], Tv.p[1]);
        }
        fprintf(f, "\n");
    }
    delete pt;
    delete r;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output directory>\n", argv[0]);
        return 1;
    }
    std::string out = argv[1];
    std::vector<ShapeDesc> shapes;
    initShapesAll(shapes);

    FILE *list = fopen((out + "/shapes.txt").c_str(), "w");
    if (!list) {
        fprintf(stderr, "cannot write to %s\n", out.c_str());
        return 1;
    }
    fprintf(list, "# Generated by tools/osd_reference from OpenSubdiv %d.%d.%d.\n",
            OPENSUBDIV_VERSION_MAJOR, OPENSUBDIV_VERSION_MINOR, OPENSUBDIV_VERSION_PATCH);
    for (size_t i = 0; i < shapes.size(); ++i) {
        Shape *shape = Shape::parseObj(shapes[i]);
        TopologyRefiner *r = makeRefiner(*shape);
        if (!r) {
            fprintf(stderr, "skipping %s: OpenSubdiv rejects it\n", shapes[i].name.c_str());
            delete shape;
            continue;
        }
        fprintf(list, "%s\n", shapes[i].name.c_str());

        FILE *f = fopen((out + "/" + shapes[i].name + ".mesh").c_str(), "w");
        writeMesh(f, *shape, *r);
        fclose(f);
        delete r;

        f = fopen((out + "/" + shapes[i].name + ".ref").c_str(), "w");
        writeUniform(f, *shape);
        if (shape->scheme != kBilinear) {
            writeAdaptive(f, *shape, 4, false, false);
            writeAdaptive(f, *shape, 4, true, false);
            writeAdaptive(f, *shape, 3, false, true);
        }
        fclose(f);
        delete shape;
    }
    fclose(list);
    return 0;
}
