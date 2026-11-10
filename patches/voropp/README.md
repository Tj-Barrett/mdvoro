# voro++ patch for mdvoro

`0001-full-precision-output.patch` makes voro++ print coordinates, volumes and
the other floating-point output with `%.17g` (full double precision) instead
of `%g` (6 significant digits).

With `%g`, the copies of a Voronoi vertex shared by several cells differ by up
to ~1e-5 at coordinates of 10-100, and mdvoro has to merge them by rounding
to 4 decimals. That rounding can fold near-degenerate facets through each
other on a box face, and TetGen then rejects the PLC as self-intersecting
(e.g. 40 sigma boxes c40_s2 and c40_s4 of the semicrystalline RVE study).
With full-precision output mdvoro writes PLCs with 6 decimals (`tetgen.DECIMALS`),
and those boxes mesh.

mdvoro detects the output precision: with stock voro++ it falls back to 4
decimals and prints a warning.

Base: voro++ 0.4.6 (`voro++-0.4.6.tar.gz`, math.lbl.gov/voro++).

```
tar xzf voro++-0.4.6.tar.gz && cd voro++-0.4.6
patch -p1 < /path/to/mdvoro/patches/voropp/0001-full-precision-output.patch
make
```

Then point `voro++:` in `mdvoro/config.yaml` (or `config.local.yaml`) at
`src/voro++`.

License: voro++ is under a modified BSD license (see its LICENSE); this patch
is distributed under the same terms.
