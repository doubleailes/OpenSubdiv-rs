# OpenSubdiv reference data

`generate.cpp` runs OpenSubdiv itself on the shapes of its regression
catalogue and writes the reference data that `tests/upstream_regression.rs`
compares this crate against. The output is committed in
`tests/data/upstream/`, so neither CI nor `cargo test` needs OpenSubdiv.

## Regenerating

```sh
tools/osd_reference/generate.sh            # work directory: target/osd_reference
tools/osd_reference/generate.sh /tmp/osd   # or any other
```

The script clones OpenSubdiv at the pinned tag (`v3_7_0`), builds its CPU
library with CMake, compiles `generate.cpp` against it and rewrites
`tests/data/upstream/`. It needs git, CMake, a C++17 compiler and network
access, and takes a few minutes the first time.

Regenerate when moving to a new OpenSubdiv release (bump `OSD_TAG`) or when
changing what the generator records. Then run
`cargo test --test upstream_regression`: it lists every difference that is
not in `KNOWN_DEVIATIONS`, ready to paste, and every listed one that no
longer occurs.

## What is recorded

For each shape of `regression/bfr_evaluate/init_shapes_all.h` (115 shapes):

- `<shape>.mesh`: the base mesh as `Far::TopologyRefinerFactory<Shape>`
  builds it (scheme, `Sdc::Options`, positions, faces, creases, corners,
  holes and the UV channel). Crease and corner tags are expanded per edge
  and per vertex, with the same weight indexing as the factory in
  `regression/common/far_utils.h`.
- `<shape>.ref`:
  - uniform refinement as deep as 20 000 faces allow (levels 1 to 3): the
    vertex, edge and face counts of every level, and up to 192 face corners
    of the last level with their refined position, limit position
    (`PrimvarRefiner::Limit`), UV and UV limit;
  - three adaptive patch tables, all with `ENDCAP_GREGORY_BASIS`,
    `useInfSharpPatch` on and `generateFVarLegacyLinearPatches` off:
    isolation 4; isolation 4 with `useSingleCreasePatch`; isolation 3 with
    `generateFVarTables`. For each table: the patch counts per type, and up
    to 96 evaluations at `(ptex face, u, v)` (patch type, `NonQuadRoot`,
    position, du, dv, and UV with its derivatives).

Corners at irregular vertices (extraordinary valence, boundary or sharp) and
evaluations on end-cap patches make up half of each sample. The rest are
evenly spaced over the mesh. Values are single precision, written with 7
significant digits.

The tangent overload of `PrimvarRefiner::Limit` is not used: in OpenSubdiv
3.7.0 it overflows a heap buffer on `catmark_nonman_edge100`.

## License

The shapes and the values computed from them come from OpenSubdiv, Copyright
Pixar, under the license in `tests/data/upstream/LICENSE-OpenSubdiv.txt`
(copied from the OpenSubdiv release by `generate.sh`). `generate.cpp`
includes OpenSubdiv's regression headers at build time; it does not vendor
them.
