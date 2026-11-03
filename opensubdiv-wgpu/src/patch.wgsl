// Patch evaluation: one invocation per patch coordinate (port of the patch
// kernel of OpenSubdiv's Osd evaluators, osd/glslComputeKernel.glsl, with
// the basis of opensubdiv-rs's `osd::evaluate_patch_basis`).
//
// Each invocation reads its PatchCoord, the patch's PatchArray and packed
// PatchParam, evaluates the patch basis at the coordinate's ptex-face
// (u, v), and combines the patch's control points from `src` into the
// limit point and, when requested, its derivatives along u and v.

struct PatchArray {
    reg_desc: u32,
    desc: u32,
    num_patches: u32,
    index_base: u32,
    stride: u32,
    primitive_id_base: u32,
}

struct PatchCoord {
    array_index: u32,
    patch_index: u32,
    vert_index: u32,
    s: f32,
    t: f32,
}

struct PatchParam {
    field0: u32,
    field1: u32,
    sharpness: f32,
}

struct Params {
    num_coords: u32,
    row_width: u32,
    length: u32,
    // Bit 0: write du; bit 1: write dv.
    flags: u32,
    src_offset: u32,
    src_stride: u32,
    dst_offset: u32,
    dst_stride: u32,
    du_offset: u32,
    du_stride: u32,
    dv_offset: u32,
    dv_stride: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> du_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> dv_out: array<f32>;
@group(0) @binding(5) var<storage, read> patch_arrays: array<PatchArray>;
@group(0) @binding(6) var<storage, read> patch_coords: array<PatchCoord>;
@group(0) @binding(7) var<storage, read> patch_indices: array<u32>;
@group(0) @binding(8) var<storage, read> patch_params: array<PatchParam>;

// Far::PatchDescriptor::Type codes.
const QUADS: u32 = 3u;
const TRIANGLES: u32 = 4u;
const LOOP: u32 = 5u;
const REGULAR: u32 = 6u;
const GREGORY_BASIS: u32 = 9u;
const GREGORY_TRIANGLE: u32 = 10u;

// The basis weights of the current patch: position, d/du, d/dv.
var<private> W: array<f32, 20>;
var<private> WS: array<f32, 20>;
var<private> WT: array<f32, 20>;

// ---------------------------------------------------------------------
//  Packed patch parameters (Far::PatchParam's bit layout)
// ---------------------------------------------------------------------

fn param_u(p: PatchParam) -> u32 { return (p.field1 >> 22u) & 0x3ffu; }
fn param_v(p: PatchParam) -> u32 { return (p.field1 >> 12u) & 0x3ffu; }
fn param_boundary(p: PatchParam) -> u32 { return (p.field1 >> 7u) & 0x1fu; }
fn param_is_regular(p: PatchParam) -> bool { return ((p.field1 >> 5u) & 1u) != 0u; }
fn param_non_quad_root(p: PatchParam) -> u32 { return (p.field1 >> 4u) & 1u; }
fn param_depth(p: PatchParam) -> u32 { return p.field1 & 0xfu; }

fn param_fraction(p: PatchParam) -> f32 {
    return 1.0 / f32(1u << (param_depth(p) - param_non_quad_root(p)));
}

fn is_triangle_rotated(p: PatchParam) -> bool {
    return param_u(p) + param_v(p) >= (1u << param_depth(p));
}

fn normalize_quad(p: PatchParam, uv: vec2<f32>) -> vec2<f32> {
    let inv = 1.0 / param_fraction(p);
    return uv * inv - vec2<f32>(f32(param_u(p)), f32(param_v(p)));
}

fn normalize_triangle(p: PatchParam, uv: vec2<f32>) -> vec2<f32> {
    if (is_triangle_rotated(p)) {
        let inv = 1.0 / param_fraction(p);
        let n = f32(1u << param_depth(p));
        return vec2<f32>(n - f32(param_u(p)), n - f32(param_v(p))) - uv * inv;
    }
    return normalize_quad(p, uv);
}

// ---------------------------------------------------------------------
//  Univariate bases
// ---------------------------------------------------------------------

struct Basis4 {
    w: vec4<f32>,
    d: vec4<f32>,
}

// Uniform cubic B-spline basis over the central knot interval.
fn bspline_basis(t: f32) -> Basis4 {
    let t2 = t * t;
    let t3 = t2 * t;
    let m = 1.0 - t;
    return Basis4(
        vec4<f32>(m * m * m, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0,
        vec4<f32>(-0.5 * m * m, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2),
    );
}

// Weights on (p-1, p0, p1, p2) with the phantom p-1 = 2 p0 - p1 folded in.
fn fold_end(x: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(0.0, x.y + 2.0 * x.x, x.z - x.x, x.w);
}

// The limit curve over [p0, p1] of (p-1, p0, p1, p2) whose p0 carries a
// crease of integer sharpness n: n steps of the crease rule, then smooth
// subdivision (see the crate's `integer_crease_basis`).
fn integer_crease_basis(t: f32, n: u32) -> Basis4 {
    let b = bspline_basis(t);
    if (n == 0u) {
        return b;
    }
    let h = exp2(-f32(n));
    if (t >= h) {
        return Basis4(fold_end(b.w), fold_end(b.d));
    }
    var window = array<vec4<f32>, 4>(
        vec4<f32>(0.0, 2.0, -1.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    for (var step = 0u; step < n; step++) {
        let q = window;
        window[0] = 0.5 * q[0] + 0.5 * q[1];
        window[1] = 0.125 * q[0] + 0.75 * q[1] + 0.125 * q[2];
        window[2] = 0.5 * q[1] + 0.5 * q[2];
        window[3] = 0.125 * q[1] + 0.75 * q[2] + 0.125 * q[3];
    }
    window[0] = vec4<f32>(h, 1.0 - h, 0.0, 0.0);
    let r = bspline_basis(t / h);
    var w = vec4<f32>(0.0);
    var d = vec4<f32>(0.0);
    for (var p = 0u; p < 4u; p++) {
        w += r.w[p] * window[p];
        d += (r.d[p] / h) * window[p];
    }
    return Basis4(w, d);
}

// Fractional sharpness blends the curves of the neighboring integers.
fn crease_basis(t: f32, sharpness: f32) -> Basis4 {
    let n = floor(sharpness);
    let f = sharpness - n;
    var r = integer_crease_basis(t, u32(n));
    if (f > 0.0) {
        let r1 = integer_crease_basis(t, u32(n) + 1u);
        r = Basis4(mix(r.w, r1.w, f), mix(r.d, r1.d, f));
    }
    return r;
}

// The crease on grid line 1 (x = 0) or, reversed, on grid line 2 (x = 1).
fn crease_basis_oriented(x: f32, sharpness: f32, reversed: bool) -> Basis4 {
    if (!reversed) {
        return crease_basis(x, sharpness);
    }
    let r = crease_basis(1.0 - x, sharpness);
    return Basis4(r.w.wzyx, -r.d.wzyx);
}

// OsdAdjustBoundaryWeights: fold phantom rows/columns by 2a - b.
fn adjust_boundary_weights(boundary: u32, sw: ptr<function, vec4<f32>>, tw: ptr<function, vec4<f32>>) {
    if ((boundary & 1u) != 0u) {
        (*tw).z -= (*tw).x;
        (*tw).y += (*tw).x * 2.0;
        (*tw).x = 0.0;
    }
    if ((boundary & 2u) != 0u) {
        (*sw).y -= (*sw).w;
        (*sw).z += (*sw).w * 2.0;
        (*sw).w = 0.0;
    }
    if ((boundary & 4u) != 0u) {
        (*tw).y -= (*tw).w;
        (*tw).z += (*tw).w * 2.0;
        (*tw).w = 0.0;
    }
    if ((boundary & 8u) != 0u) {
        (*sw).z -= (*sw).x;
        (*sw).y += (*sw).x * 2.0;
        (*sw).x = 0.0;
    }
}

// ---------------------------------------------------------------------
//  Quad patches
// ---------------------------------------------------------------------

fn eval_regular(param: PatchParam, st: vec2<f32>) {
    var bu: Basis4;
    var bv: Basis4;
    let boundary = param_boundary(param);
    if (param.sharpness > 0.0) {
        // Single crease: the one boundary bit names the crease edge.
        // Edges 3 and 1 are the columns at s = 0 and 1, 0 and 2 the rows
        // at t = 0 and 1.
        let edge = countTrailingZeros(boundary);
        if (edge == 3u) {
            bu = crease_basis_oriented(st.x, param.sharpness, false);
        } else if (edge == 1u) {
            bu = crease_basis_oriented(st.x, param.sharpness, true);
        } else {
            bu = bspline_basis(st.x);
        }
        if (edge == 0u) {
            bv = crease_basis_oriented(st.y, param.sharpness, false);
        } else if (edge == 2u) {
            bv = crease_basis_oriented(st.y, param.sharpness, true);
        } else {
            bv = bspline_basis(st.y);
        }
    } else {
        bu = bspline_basis(st.x);
        bv = bspline_basis(st.y);
        var sw = bu.w;
        var tw = bv.w;
        var sd = bu.d;
        var td = bv.d;
        adjust_boundary_weights(boundary, &sw, &tw);
        adjust_boundary_weights(boundary, &sd, &td);
        bu = Basis4(sw, sd);
        bv = Basis4(tw, td);
    }
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let k = 4u * j + i;
            W[k] = bu.w[i] * bv.w[j];
            WS[k] = bu.d[i] * bv.w[j];
            WT[k] = bu.w[i] * bv.d[j];
        }
    }
}

fn bernstein(t: f32) -> Basis4 {
    let s = 1.0 - t;
    return Basis4(
        vec4<f32>(s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t),
        vec4<f32>(-3.0 * s * s, 3.0 * s * s - 6.0 * t * s, 6.0 * t * s - 3.0 * t * t, 3.0 * t * t),
    );
}

// An interior Bezier point of a Gregory patch: the rational blend
// (x P + y Q) / (x + y) of two face points, x and y affine in (s, t) with
// gradients xg and yg. `b`, `bs`, `bt` are its Bernstein weight and
// derivatives.
fn rational_slot(p: u32, q: u32, x: f32, y: f32, xg: vec2<f32>, yg: vec2<f32>, b: f32, bs: f32, bt: f32) {
    let d = x + y;
    if (d < 1e-5) {
        // At the corner the Bernstein factor vanishes.
        W[p] += 0.5 * b;
        W[q] += 0.5 * b;
        WS[p] += 0.5 * bs;
        WS[q] += 0.5 * bs;
        WT[p] += 0.5 * bt;
        WT[q] += 0.5 * bt;
        return;
    }
    let wp = x / d;
    let wq = y / d;
    W[p] += b * wp;
    W[q] += b * wq;
    let dsum = xg + yg;
    let dwp = (xg - wp * dsum) / d;
    let dwq = (yg - wq * dsum) / d;
    WS[p] += bs * wp + b * dwp.x;
    WS[q] += bs * wq + b * dwq.x;
    WT[p] += bt * wp + b * dwp.y;
    WT[q] += bt * wq + b * dwq.y;
}

fn eval_gregory(st: vec2<f32>) {
    let bs = bernstein(st.x);
    let bt = bernstein(st.y);
    // Bezier slots (i along s, j along t) taken directly by one point.
    let direct = array<vec3<u32>, 12>(
        vec3<u32>(0u, 0u, 0u), vec3<u32>(3u, 0u, 5u), vec3<u32>(3u, 3u, 10u), vec3<u32>(0u, 3u, 15u),
        vec3<u32>(1u, 0u, 1u), vec3<u32>(0u, 1u, 2u), vec3<u32>(3u, 1u, 6u), vec3<u32>(2u, 0u, 7u),
        vec3<u32>(2u, 3u, 11u), vec3<u32>(3u, 2u, 12u), vec3<u32>(0u, 2u, 16u), vec3<u32>(1u, 3u, 17u),
    );
    for (var k = 0u; k < 12u; k++) {
        let e = direct[k];
        W[e.z] += bs.w[e.x] * bt.w[e.y];
        WS[e.z] += bs.d[e.x] * bt.w[e.y];
        WT[e.z] += bs.w[e.x] * bt.d[e.y];
    }
    let s = st.x;
    let t = st.y;
    // b11: (s F0+ + t F0-) / (s + t)
    rational_slot(3u, 4u, s, t, vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        bs.w[1] * bt.w[1], bs.d[1] * bt.w[1], bs.w[1] * bt.d[1]);
    // b21: ((1-s) F1- + t F1+) / (1 - s + t)
    rational_slot(9u, 8u, 1.0 - s, t, vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 1.0),
        bs.w[2] * bt.w[1], bs.d[2] * bt.w[1], bs.w[2] * bt.d[1]);
    // b22: ((1-s) F2+ + (1-t) F2-) / (2 - s - t)
    rational_slot(13u, 14u, 1.0 - s, 1.0 - t, vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, -1.0),
        bs.w[2] * bt.w[2], bs.d[2] * bt.w[2], bs.w[2] * bt.d[2]);
    // b12: (s F3- + (1-t) F3+) / (1 + s - t)
    rational_slot(19u, 18u, s, 1.0 - t, vec2<f32>(1.0, 0.0), vec2<f32>(0.0, -1.0),
        bs.w[1] * bt.w[2], bs.d[1] * bt.w[2], bs.w[1] * bt.d[2]);
}

fn eval_quads(st: vec2<f32>) {
    let s = st.x;
    let t = st.y;
    W[0] = (1.0 - s) * (1.0 - t);
    W[1] = s * (1.0 - t);
    W[2] = s * t;
    W[3] = (1.0 - s) * t;
    WS[0] = -(1.0 - t);
    WS[1] = 1.0 - t;
    WS[2] = t;
    WS[3] = -t;
    WT[0] = -(1.0 - s);
    WT[1] = -s;
    WT[2] = s;
    WT[3] = 1.0 - s;
}

// ---------------------------------------------------------------------
//  Triangular patches
// ---------------------------------------------------------------------

// Coefficients (times 12) of the 12 quartic box-spline basis polynomials
// on the monomials s^i t^j, (i, j) in MONOMIAL_I / MONOMIAL_J order.
const BOX_SPLINE_COEFFS = array<f32, 180>(
    1, -4, 6, -4, 1, -2, 6, -6, 2, 0, 0, 0, 2, -2, -1,
    1, -2, 0, 2, -1, 2, -6, 6, -2, 0, 0, 0, -4, 4, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, -1,
    1, -2, 0, 2, -1, -4, 6, 0, -2, 6, -6, 0, -4, 2, 1,
    6, 0, -12, 8, -1, 0, -12, 12, -2, -12, 12, 0, 8, -2, -1,
    1, 2, 0, -4, 2, 4, 6, -12, 4, 6, -6, 0, -4, -2, -1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1,
    1, 2, 0, -4, 2, -2, -6, 0, 4, 0, 6, 0, 2, -2, -1,
    1, 4, 6, -4, -1, 2, 6, -6, -2, 0, -12, 0, -4, 4, 2,
    0, 0, 0, 2, -1, 0, 0, 6, -2, 0, 6, 0, 2, -2, -1,
    0, 0, 0, 2, -1, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
);
const MONOMIAL_I = array<u32, 15>(0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4);
const MONOMIAL_J = array<u32, 15>(0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0);

fn eval_box_spline(st: vec2<f32>) {
    var ps = array<f32, 5>(1.0, 1.0, 1.0, 1.0, 1.0);
    var pt = array<f32, 5>(1.0, 1.0, 1.0, 1.0, 1.0);
    for (var k = 1u; k < 5u; k++) {
        ps[k] = ps[k - 1u] * st.x;
        pt[k] = pt[k - 1u] * st.y;
    }
    var coeffs = BOX_SPLINE_COEFFS;
    var mi = MONOMIAL_I;
    var mj = MONOMIAL_J;
    for (var slot = 0u; slot < 12u; slot++) {
        var value = 0.0;
        var ds = 0.0;
        var dt = 0.0;
        for (var m = 0u; m < 15u; m++) {
            let c = coeffs[15u * slot + m] / 12.0;
            let i = mi[m];
            let j = mj[m];
            value += c * ps[i] * pt[j];
            if (i > 0u) {
                ds += c * f32(i) * ps[i - 1u] * pt[j];
            }
            if (j > 0u) {
                dt += c * f32(j) * ps[i] * pt[j - 1u];
            }
        }
        W[slot] = value;
        WS[slot] = ds;
        WT[slot] = dt;
    }
}

// Move the weight of slot `src` onto `a`, `b`, `c` with factors fa, fb, fc
// (in all three weight arrays) and clear it.
fn fold_tri(src: u32, a: u32, fa: f32, b: u32, fb: f32, c: u32, fc: f32) {
    let w = W[src];
    let ws = WS[src];
    let wt = WT[src];
    W[a] += fa * w;
    W[b] += fb * w;
    W[c] += fc * w;
    WS[a] += fa * ws;
    WS[b] += fb * ws;
    WS[c] += fc * ws;
    WT[a] += fa * wt;
    WT[b] += fb * wt;
    WT[c] += fc * wt;
    W[src] = 0.0;
    WS[src] = 0.0;
    WT[src] = 0.0;
}

// Osd_adjustBoxSplineTriBoundaryWeights.
fn adjust_box_spline_boundary(boundary: u32) {
    if (boundary == 0u) {
        return;
    }
    let upper = (boundary >> 3u) & 3u;
    let lower = boundary & 7u;
    var e_bits = lower;
    var v_bits = 0u;
    if (upper == 1u) {
        v_bits = lower;
        e_bits = 0u;
    } else if (upper == 2u) {
        v_bits = ((lower & 1u) << 2u) | (lower >> 1u);
    }
    let e0 = (e_bits & 1u) != 0u;
    let e1 = (e_bits & 2u) != 0u;
    let e2 = (e_bits & 4u) != 0u;
    // Each fold clears its own source slot; no fold of a block targets
    // another source of the same block, so this matches OpenSubdiv's
    // clearing at the end of each block.
    if (e0) {
        if (e2) { fold_tri(0u, 4u, 2.0, 8u, -1.0, 8u, 0.0); } else { fold_tri(0u, 4u, 1.0, 3u, 1.0, 7u, -1.0); }
        fold_tri(1u, 4u, 1.0, 5u, 1.0, 8u, -1.0);
        if (e1) { fold_tri(2u, 5u, 2.0, 8u, -1.0, 8u, 0.0); } else { fold_tri(2u, 5u, 1.0, 6u, 1.0, 9u, -1.0); }
    }
    if (e1) {
        if (e0) { fold_tri(6u, 5u, 2.0, 4u, -1.0, 4u, 0.0); } else { fold_tri(6u, 5u, 1.0, 2u, 1.0, 1u, -1.0); }
        fold_tri(9u, 5u, 1.0, 8u, 1.0, 4u, -1.0);
        if (e2) { fold_tri(11u, 8u, 2.0, 4u, -1.0, 4u, 0.0); } else { fold_tri(11u, 8u, 1.0, 10u, 1.0, 7u, -1.0); }
    }
    if (e2) {
        if (e1) { fold_tri(10u, 8u, 2.0, 5u, -1.0, 5u, 0.0); } else { fold_tri(10u, 8u, 1.0, 11u, 1.0, 9u, -1.0); }
        fold_tri(7u, 8u, 1.0, 4u, 1.0, 5u, -1.0);
        if (e0) { fold_tri(3u, 4u, 2.0, 5u, -1.0, 5u, 0.0); } else { fold_tri(3u, 4u, 1.0, 0u, 1.0, 1u, -1.0); }
    }
    if ((v_bits & 1u) != 0u) {
        fold_tri(3u, 4u, 1.0, 7u, 1.0, 8u, -1.0);
        fold_tri(0u, 4u, 1.0, 1u, 1.0, 5u, -1.0);
    }
    if ((v_bits & 2u) != 0u) {
        fold_tri(2u, 5u, 1.0, 1u, 1.0, 4u, -1.0);
        fold_tri(6u, 5u, 1.0, 9u, 1.0, 8u, -1.0);
    }
    if ((v_bits & 4u) != 0u) {
        fold_tri(11u, 8u, 1.0, 9u, 1.0, 5u, -1.0);
        fold_tri(10u, 8u, 1.0, 7u, 1.0, 4u, -1.0);
    }
}

fn int_pow(x: f32, n: u32) -> f32 {
    var r = 1.0;
    for (var k = 0u; k < n; k++) {
        r *= x;
    }
    return r;
}

fn factorial(n: u32) -> f32 {
    var r = 1.0;
    for (var k = 2u; k <= n; k++) {
        r *= f32(k);
    }
    return r;
}

// Quartic Bernstein triangle polynomial B_ijk at barycentrics (u, s, t),
// u = 1 - s - t, with its derivatives along s and t.
fn bernstein_tri(i: u32, j: u32, k: u32, u: f32, s: f32, t: f32) -> vec3<f32> {
    let c = 24.0 / (factorial(i) * factorial(j) * factorial(k));
    let pu = int_pow(u, i);
    let ps = int_pow(s, j);
    let pt = int_pow(t, k);
    var du = 0.0;
    var ds = 0.0;
    var dt = 0.0;
    if (i > 0u) { du = f32(i) * int_pow(u, i - 1u); }
    if (j > 0u) { ds = f32(j) * int_pow(s, j - 1u); }
    if (k > 0u) { dt = f32(k) * int_pow(t, k - 1u); }
    return c * vec3<f32>(pu * ps * pt, -du * ps * pt + pu * ds * pt, -du * ps * pt + pu * ps * dt);
}

fn eval_gregory_triangle(st: vec2<f32>) {
    let s = st.x;
    let t = st.y;
    let u = 1.0 - s - t;
    // Bezier slots (i, j, k) taken directly by one point (the w component).
    let direct = array<vec4<u32>, 12>(
        vec4<u32>(4u, 0u, 0u, 0u), vec4<u32>(3u, 1u, 0u, 1u), vec4<u32>(3u, 0u, 1u, 2u),
        vec4<u32>(0u, 4u, 0u, 5u), vec4<u32>(0u, 3u, 1u, 6u), vec4<u32>(1u, 3u, 0u, 7u),
        vec4<u32>(0u, 0u, 4u, 10u), vec4<u32>(1u, 0u, 3u, 11u), vec4<u32>(0u, 1u, 3u, 12u),
        vec4<u32>(2u, 2u, 0u, 15u), vec4<u32>(0u, 2u, 2u, 16u), vec4<u32>(2u, 0u, 2u, 17u),
    );
    for (var n = 0u; n < 12u; n++) {
        let e = direct[n];
        let b = bernstein_tri(e.x, e.y, e.z, u, s, t);
        W[e.w] += b.x;
        WS[e.w] += b.y;
        WT[e.w] += b.z;
    }
    // Interior points: (x F+ + y F-) / (x + y), x the barycentric of the
    // corner's next corner and y of its previous one.
    let b0 = bernstein_tri(2u, 1u, 1u, u, s, t);
    rational_slot(3u, 4u, s, t, vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), b0.x, b0.y, b0.z);
    let b1 = bernstein_tri(1u, 2u, 1u, u, s, t);
    rational_slot(8u, 9u, t, u, vec2<f32>(0.0, 1.0), vec2<f32>(-1.0, -1.0), b1.x, b1.y, b1.z);
    let b2 = bernstein_tri(1u, 1u, 2u, u, s, t);
    rational_slot(13u, 14u, u, s, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 0.0), b2.x, b2.y, b2.z);
}

fn eval_triangles(st: vec2<f32>) {
    W[0] = 1.0 - st.x - st.y;
    W[1] = st.x;
    W[2] = st.y;
    WS[0] = -1.0;
    WS[1] = 1.0;
    WS[2] = 0.0;
    WT[0] = -1.0;
    WT[1] = 0.0;
    WT[2] = 1.0;
}

// ---------------------------------------------------------------------
//  OsdEvaluatePatchBasis
// ---------------------------------------------------------------------

// Fill W, WS, WT with the basis of a patch of type `desc` at ptex-face
// (u, v), derivatives along the ptex face's u and v; returns the number of
// control points (0 for an unknown type).
fn evaluate_patch_basis(desc: u32, param: PatchParam, uv: vec2<f32>) -> u32 {
    for (var k = 0u; k < 20u; k++) {
        W[k] = 0.0;
        WS[k] = 0.0;
        WT[k] = 0.0;
    }
    let triangular = desc == LOOP || desc == GREGORY_TRIANGLE || desc == TRIANGLES;
    var st: vec2<f32>;
    var scale = 1.0 / param_fraction(param);
    if (triangular) {
        st = normalize_triangle(param, uv);
        if (is_triangle_rotated(param)) {
            scale = -scale;
        }
    } else {
        st = normalize_quad(param, uv);
    }

    var n = 0u;
    switch desc {
        case REGULAR: {
            eval_regular(param, st);
            n = 16u;
        }
        case GREGORY_BASIS: {
            eval_gregory(st);
            n = 20u;
        }
        case QUADS: {
            eval_quads(st);
            n = 4u;
        }
        case LOOP: {
            eval_box_spline(st);
            adjust_box_spline_boundary(param_boundary(param));
            n = 12u;
        }
        case GREGORY_TRIANGLE: {
            eval_gregory_triangle(st);
            n = 18u;
        }
        case TRIANGLES: {
            eval_triangles(st);
            n = 3u;
        }
        default: {
            return 0u;
        }
    }
    for (var k = 0u; k < n; k++) {
        WS[k] *= scale;
        WT[k] *= scale;
    }
    return n;
}

@compute @workgroup_size(64)
fn eval_patches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row_width;
    if (i >= params.num_coords) {
        return;
    }
    let coord = patch_coords[i];
    let array = patch_arrays[coord.array_index];
    let param = patch_params[coord.patch_index];
    var desc = array.desc;
    if (param_is_regular(param)) {
        desc = array.reg_desc;
    }
    let n = evaluate_patch_basis(desc, param, vec2<f32>(coord.s, coord.t));

    let write_du = (params.flags & 1u) != 0u;
    let write_dv = (params.flags & 2u) != 0u;
    for (var k = 0u; k < params.length; k++) {
        var p = 0.0;
        var du = 0.0;
        var dv = 0.0;
        for (var j = 0u; j < n; j++) {
            let cv = patch_indices[coord.vert_index + j];
            let x = src[params.src_offset + cv * params.src_stride + k];
            p += W[j] * x;
            du += WS[j] * x;
            dv += WT[j] * x;
        }
        dst[params.dst_offset + i * params.dst_stride + k] = p;
        if (write_du) {
            du_out[params.du_offset + i * params.du_stride + k] = du;
        }
        if (write_dv) {
            dv_out[params.dv_offset + i * params.dv_stride + k] = dv;
        }
    }
}
