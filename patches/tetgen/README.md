# TetGen patch for mdvoro

`0001-mdvoro-robustness.patch` fixes crashes and memory corruption that large
mdvoro PLCs (Voronoi cells of MD beads, 50-60 sigma RVEs) trigger in TetGen
1.6.1. The patch message lists the fixes.

Base: TetGen 1.6.1, commit `5203ac3` (2026-10-08) of
[TetGen/TetGen](https://github.com/TetGen/TetGen)
(primary repository: [codeberg.org/TetGen/TetGen](https://codeberg.org/TetGen/TetGen)).

```
git clone https://github.com/TetGen/TetGen.git && cd TetGen
git checkout 5203ac3540d5092163c1bce61737f75a21703e1b
git apply /path/to/mdvoro/patches/tetgen/0001-mdvoro-robustness.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Then point `tetgen:` in `mdvoro/config.yaml` at `build/tetgen`.

Without it, `-pA` segfaults on large PLCs. The semicrystalline RVE scripts try
`-pA`, then `-pAD5`, then `-pYA`, and rely on this build.

License: TetGen is AGPL-3.0; this patch is distributed under the same license.
