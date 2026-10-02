#!/usr/bin/env bash
#
# Rebuild the OpenSubdiv reference data in tests/data/upstream/.
#
# Clones OpenSubdiv at a pinned tag, builds its CPU library, compiles
# generate.cpp against it and runs it. Needs git, cmake, a C++14 compiler
# and network access. Not run by CI: the generated files are committed.
#
# Usage: tools/osd_reference/generate.sh [work directory]

set -euo pipefail

OSD_TAG=v3_7_0
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
work="${1:-$repo/target/osd_reference}"
out="$repo/tests/data/upstream"

mkdir -p "$work"
if [ ! -d "$work/OpenSubdiv" ]; then
    git clone --depth 1 --branch "$OSD_TAG" \
        https://github.com/PixarAnimationStudios/OpenSubdiv.git "$work/OpenSubdiv"
fi
osd="$work/OpenSubdiv"

cmake -S "$osd" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
    -DNO_EXAMPLES=1 -DNO_TUTORIALS=1 -DNO_REGRESSION=1 -DNO_DOC=1 -DNO_TESTS=1 \
    -DNO_GLTESTS=1 -DNO_OMP=1 -DNO_TBB=1 -DNO_CUDA=1 -DNO_OPENCL=1 -DNO_OPENGL=1 \
    -DNO_DX=1 -DNO_GLFW=1 -DNO_PTEX=1 -DNO_METAL=1 > "$work/cmake.log"
cmake --build "$work/build" --target osd_static_cpu -j "$(nproc 2>/dev/null || echo 4)"

c++ -O2 -std=c++17 -w \
    -I"$osd" -I"$osd/opensubdiv" -I"$work/build" \
    "$here/generate.cpp" \
    "$osd/regression/common/shape_utils.cpp" \
    "$osd/regression/common/far_utils.cpp" \
    -L"$work/build/lib" -losdCPU -o "$work/generate"

rm -f "$out"/*.mesh "$out"/*.ref "$out"/shapes.txt
mkdir -p "$out"
"$work/generate" "$out"
cp "$osd/LICENSE.txt" "$out/LICENSE-OpenSubdiv.txt"
echo "Wrote $(ls "$out"/*.ref | wc -l) shapes to $out"
