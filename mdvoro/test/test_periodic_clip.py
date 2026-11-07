"""Tests for mdvoro.mesh.periodic_clip: box-clipped periodic Voronoi tessellations."""
import collections
import os
import shutil

import numpy as np
import pytest

from mdvoro.mesh.periodic_clip import _clip, _volume, clip_periodic_cells, WALLS
from mdvoro.utils.read_config import read_config

LIMS = [[0.0, 6.0], [0.0, 7.0], [0.0, 8.0]]
L = np.array([6.0, 7.0, 8.0])


def cube(c, h):
    """axis-aligned cube centred at c with half-width h, outward loops"""
    P = np.array([[c[0] + sx * h, c[1] + sy * h, c[2] + sz * h]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    loops = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return P, [(lp, 0) for lp in loops]


def test_clip_cube_halves_volume_and_caps():
    P, F = cube([0.0, 0.0, 0.0], 1.0)
    assert _volume(P, F) == pytest.approx(8.0)
    Q, G = _clip(P, F, 0, 0.0, keep_below=True, tag=-2, eps=1e-12)
    assert _volume(Q, G) == pytest.approx(4.0)
    caps = [lp for lp, adj in G if adj == -2]
    assert len(caps) == 1 and np.allclose(Q[caps[0], 0], 0.0)


def test_clip_outside_and_inside():
    P, F = cube([5.0, 0.0, 0.0], 1.0)
    assert _clip(P, F, 0, 0.0, keep_below=True, tag=-2, eps=1e-12) is None
    Q, G = _clip(P, F, 0, 10.0, keep_below=True, tag=-2, eps=1e-12)
    assert Q is P and G is F


def test_clip_face_on_plane_becomes_wall():
    P, F = cube([1.0, 0.0, 0.0], 1.0)                 # face x = 0 lies on the plane
    Q, G = _clip(P, F, 0, 0.0, keep_below=False, tag=-1, eps=1e-12)
    assert _volume(Q, G) == pytest.approx(8.0)
    assert sum(adj == -1 for _, adj in G) == 1


@pytest.fixture(scope='module')
def pieces(tmp_path_factory):
    voropp = read_config().alias_voropp
    if not voropp or not (os.path.isfile(voropp) or shutil.which(voropp)):
        pytest.skip('voro++ not configured')
    from mdvoro.mesh.compute_voro import compute_voro
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('voro'))
    try:
        rng = np.random.default_rng(3)
        pts = rng.uniform([0, 0, 0], L, size=(400, 3))
        data = [[*p, int(rng.integers(0, 2))] for p in pts]
        voro = compute_voro(data, LIMS, periodic=[True] * 3, method='voro++')
    finally:
        os.chdir(cwd)
    return clip_periodic_cells(voro, LIMS)


def test_pieces_tile_the_box(pieces):
    assert sum(p['volume'] for p in pieces) == pytest.approx(L.prod(), rel=1e-6)
    assert min(p['volume'] for p in pieces) > 0


def test_adjacency_is_symmetric(pieces):
    for k, p in enumerate(pieces):
        for f in p['faces']:
            a = f['adjacent_cell']
            if a >= 0:
                assert k in [g['adjacent_cell'] for g in pieces[a]['faces']]


def test_opposite_box_faces_carry_identical_polygons(pieces):
    def key(Q):
        return tuple(sorted(map(tuple, np.round(Q, 8))))
    for d in range(3):
        lo, hi = collections.Counter(), collections.Counter()
        area = 0.0
        for p in pieces:
            P = np.array(p['vertices'])
            for f in p['faces']:
                Q = P[f['vertices']]
                if f['adjacent_cell'] == WALLS[(d, 0)]:
                    lo[key(Q)] += 1
                    area += 0.5 * np.linalg.norm(np.cross(Q[1:-1] - Q[0], Q[2:] - Q[0]), axis=1).sum()
                elif f['adjacent_cell'] == WALLS[(d, 1)]:
                    hi[key(Q - np.eye(3)[d] * L)] += 1
        assert lo == hi
        assert area == pytest.approx(L.prod() / L[d], rel=1e-6)


def test_region_points_inside_box(pieces):
    for p in pieces:
        x = np.array(p['original'][:3])
        assert ((x > 0) & (x < L)).all()


def test_clipped_mesh_has_matching_boundary(tmp_path):
    """periodic tessellation -> clip -> PLC -> TetGen -pYA: opposite faces carry
    the same nodes and the same boundary triangles"""
    import subprocess
    cfg = read_config()
    voropp, tetgen_bin = cfg.alias_voropp, cfg.alias_tetgen
    if not voropp or not tetgen_bin or not os.path.isfile(tetgen_bin):
        pytest.skip('voro++ or tetgen not configured')
    from mdvoro.mesh.compute_voro import compute_voro
    from mdvoro.mesh.tetgen import tetgen
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rng = np.random.default_rng(11)
        Lb = np.array([5.0, 5.5, 6.0])
        lims = [[0.0, 5.0], [0.0, 5.5], [0.0, 6.0]]
        pts = rng.uniform(0, Lb, size=(600, 3))
        d = pts - [1.0, 1.0, 1.0]
        d -= Lb * np.round(d / Lb)
        ph = np.where(np.linalg.norm(d, axis=1) < 1.6, 2, -1)    # one crystal across the corner
        voro = compute_voro([[*p, int(t)] for p, t in zip(pts, ph)], lims, periodic=[True] * 3,
                            method='voro++')
        tetgen().mesh(voro, lims, method='tetgen', nomesh=True, min_edge=0.05, clip_periodic=True)
        r = subprocess.run([tetgen_bin, '-pYA', 'temp.poly'], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1000:]
        nl = [l.split() for l in open('temp.1.node') if l.split() and not l.startswith('#')]
        X = np.array([[float(v) for v in p[1:4]] for p in nl[1:]])
        fl = [l.split() for l in open('temp.1.face') if l.split() and not l.startswith('#')]
        F = np.array([[int(v) for v in p[1:4]] for p in fl[1:]]) - 1
    finally:
        os.chdir(cwd)
    for ax in range(3):
        def face(side):
            on = np.abs(X[:, ax] - side) < 1e-4
            pts2 = set(map(tuple, np.round(np.delete(X[on], ax, 1), 4)))
            tris = {tuple(sorted(map(tuple, np.round(np.delete(X[t], ax, 1), 4)))) for t in F[on[F].all(1)]}
            return pts2, tris
        (p_lo, t_lo), (p_hi, t_hi) = face(0.0), face(Lb[ax])
        assert p_lo == p_hi and len(p_lo) > 0
        assert t_lo == t_hi and len(t_lo) > 0
