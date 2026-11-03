// Stencil evaluation: one invocation per stencil (port of the stencil
// kernel of OpenSubdiv's Osd evaluators, osd/glslComputeKernel.glsl).
//
//   dst[i] = sum over j in offsets[i]..offsets[i+1] of
//            weights[j] * src[indices[j]]
//
// for each of `length` consecutive floats of a primvar, the source and
// destination primvars laid out as `offset + index * stride` floats.

struct Params {
    num_stencils: u32,
    // Invocations per row of the dispatch grid (rows extend past the
    // 65535-workgroup limit of one dimension).
    row_width: u32,
    length: u32,
    src_offset: u32,
    src_stride: u32,
    dst_offset: u32,
    dst_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;

@compute @workgroup_size(64)
fn eval_stencils(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row_width;
    if (i >= params.num_stencils) {
        return;
    }
    let begin = offsets[i];
    let end = offsets[i + 1u];
    let d = params.dst_offset + i * params.dst_stride;
    if (params.length >= 1u && params.length <= 4u) {
        // One pass over the stencil for up to four floats per element
        // (named components: FXC cannot unroll a dynamically indexed
        // accumulator).
        let n = params.length;
        var acc = vec4<f32>(0.0);
        for (var j = begin; j < end; j++) {
            let w = weights[j];
            let s = params.src_offset + indices[j] * params.src_stride;
            acc.x += w * src[s];
            if (n > 1u) {
                acc.y += w * src[s + 1u];
            }
            if (n > 2u) {
                acc.z += w * src[s + 2u];
            }
            if (n > 3u) {
                acc.w += w * src[s + 3u];
            }
        }
        dst[d] = acc.x;
        if (n > 1u) {
            dst[d + 1u] = acc.y;
        }
        if (n > 2u) {
            dst[d + 2u] = acc.z;
        }
        if (n > 3u) {
            dst[d + 3u] = acc.w;
        }
        return;
    }
    for (var k = 0u; k < params.length; k++) {
        var acc = 0.0;
        for (var j = begin; j < end; j++) {
            acc += weights[j] * src[params.src_offset + indices[j] * params.src_stride + k];
        }
        dst[d + k] = acc;
    }
}
