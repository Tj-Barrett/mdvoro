import numpy as np

from mdvoro.mesh.sliver_flip import fold_flips, tet_quality, read_poly, write_poly
from mdvoro.mesh.tetgen import _full_precision

# a fold: PLC triangles abc and abd meet along ab almost flat (d 1e-4 above z = 0);
# the sliver abcd (region 1) sits in the fold, tets abce and abde (region 2) below
A, B, C, D, E = range(5)
X = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1e-4], [0.5, 0.5, -1.0]])
T = np.array([[A, B, C, D], [A, B, C, E], [A, B, D, E]])
# neighbour k is across the face opposite vertex k: the sliver's faces abd (opposite c) and abc (opposite d)
NBR = np.array([[-2, -2, 2, 1], [-2, -2, -2, 0], [-2, -2, -2, 0]])
TRIS = np.array([[A, B, C], [A, B, D]]) + 1


def test_fold_is_flipped_onto_the_other_region():
    keep, new, st = fold_flips(X, T, np.array([1, 2, 2]), NBR, TRIS)
    assert st['flipped'] == 1
    assert not keep.any()
    assert sorted(tuple(sorted(f)) for f in new) == [(A + 1, C + 1, D + 1), (B + 1, C + 1, D + 1)]


def test_fold_between_different_regions_is_kept():
    keep, new, st = fold_flips(X, T, np.array([1, 2, 3]), NBR, TRIS)
    assert keep.all() and not new and st['skip_regions'] == 1


def test_box_face_triangles_are_never_flipped():
    lims = [[-5, 5], [-5, 5], [0, 5]]                 # abc lies on the box face z = 0
    keep, new, st = fold_flips(X, T, np.array([1, 2, 2]), NBR, TRIS, lims=lims)
    assert keep.all() and not new and st['skip_box'] == 1


def test_a_flip_is_not_undone():
    removed = {(C + 1, D + 1)}                         # cd was flipped away before
    keep, new, st = fold_flips(X, T, np.array([1, 2, 2]), NBR, TRIS, removed=removed)
    assert keep.all() and not new and st['skip_undo'] == 1


def test_tet_quality():
    reg = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    q = tet_quality(np.vstack([reg, X]), np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert abs(q[0] - 1) < 1e-12 and q[1] < 1e-3


def test_poly_round_trip(tmp_path):
    head = ['# Node count, 3 dim, attribute, boundary marker', '4 3 1 1', '# Node index, node coordinates']
    head += [f'{i + 1} {x} {y} {z}' for i, (x, y, z) in enumerate(X[:4])]
    tail = ['# Part 3 - hole list', '0', '# Part 4 - region list', '0']
    write_poly(tmp_path / 'p.poly', head, TRIS, tail)
    h, tris, t = read_poly(tmp_path / 'p.poly')
    assert h == head and t == tail and np.array_equal(tris, TRIS)


def test_full_precision_detects_voropp_output_digits():
    six = [{'vertices': [[1.23457, 2.5, 31.4159], [12.3457, 0.271828, 6.62607]]}]
    full = [{'vertices': [[1.2345678901234567, 2.718281828459045, 31.41592653589793],
                          [12.345678901234567, 0.2718281828459045, 6.626070150000001]]}]
    assert not _full_precision(six)
    assert _full_precision(full)


def test_two_folds_never_make_the_same_triangle():
    # a second sliver abcd' whose flip would also create acd: only one of the two may flip
    X2 = np.vstack([X, [[0, 1, -1e-4]]])
    F = 5
    t = np.array([[A, B, C, D], [A, B, C, E], [A, B, D, E], [A, B, F, D], [A, B, F, E]])
    nbr = np.array([[-2, -2, 2, 1], [-2, -2, -2, 0], [-2, -2, -2, 0], [-2, -2, 2, 4], [-2, -2, -2, 3]])
    tris = np.array([[A, B, C], [A, B, D], [A, B, F], [C, D, F]]) + 1   # acd' would duplicate a flip of the first
    keep, new, st = fold_flips(X2, t, np.array([1, 2, 2, 1, 2]), nbr, tris)
    keys = [tuple(sorted(f)) for f in new]
    assert len(keys) == len(set(keys))


def test_delete_slivers_keeps_nodes_and_box_edges():
    from mdvoro.mesh.sliver_flip import delete_slivers
    # the sliver abcd is the only tet holding nothing of its own: deleted
    keep, st = delete_slivers(X, T)
    assert keep.tolist() == [False, True, True] and st['deleted'] == 1
    # a sliver holding a node of its own is kept
    keep, st = delete_slivers(X, T[:1])
    assert keep.all() and st['restored'] == 1
    # an edge on a box face (z = 0: a, b, c) that only the sliver has is kept
    t = np.array([[A, B, C, D], [A, B, D, E], [A, C, D, E]])           # bc only in the sliver
    keep, _ = delete_slivers(X, t, lims=[[-5, 5], [-5, 5], [0, 5]])
    assert keep[0]
    keep, _ = delete_slivers(X, t)
    assert not keep[0]


def test_absorb_removes_a_flat_sliver_without_losing_volume():
    from mdvoro.mesh.sliver_flip import absorb_slivers
    # flat quad a c b d (d 1e-4 above z = 0) between two pyramids: apex e above (split along cd),
    # f below (split along ab); the sliver abcd fills the gap between the two diagonals
    x = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 1e-4], [0.5, 0.5, 1], [0.5, 0.5, -1]], float)
    a, b, c, d, e, f = range(6)
    t = np.array([[a, b, c, d], [a, c, d, e], [b, c, d, e], [a, b, c, f], [a, b, d, f]])
    V = lambda tt: np.abs(np.einsum('ij,ij->i', *(lambda p: (p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0],
                                                              p[:, 3] - p[:, 0])))(x[tt]))).sum()
    for lab, relab in (([1, 1, 1, 1, 1], 0), ([2, 1, 1, 1, 1], 1)):     # 2: the sliver joins the region across
        t2, a2, st = absorb_slivers(x, t, np.array(lab), 1e-3)
        assert st['removed_edges'] == 1 and st['below_after'] == 0 and st['relabelled'] == relab
        assert len(t2) == 4 and abs(V(t2) - V(t)) < 1e-12 and (a2 == 1).all()
        assert not any(set(r) == {a, b, c, d} for r in t2.tolist())


def _grid_mesh(node, offset, plane_x1=False):
    """Delaunay mesh of a jittered 5^3 grid on [0, 2]^3 plus a node at grid node (i, j, k) + offset"""
    from scipy.spatial import Delaunay
    g = np.linspace(0, 2, 5)
    x = np.array([[i, j, k] for i in g for j in g for k in g], float)
    rng = np.random.default_rng(0)                  # jitter off the box faces: no co-planar grid tets
    inner = (x > 0) & (x < 2)
    if plane_x1:                                    # keep the plane x = 1 flat (an interface)
        inner[:, 0] &= x[:, 0] != 1
    x[inner] += rng.uniform(-0.08, 0.08, inner.sum())
    x = np.vstack([x, x[25 * node[0] + 5 * node[1] + node[2]] + offset])
    t = Delaunay(x).simplices
    return x, t[tet_quality(x, t) > 1e-12]


def _vol(x, t):
    p = x[t]
    return np.abs(np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))).sum() / 6


def test_collapse_merges_a_near_duplicate_node_without_voids():
    from mdvoro.mesh.sliver_flip import collapse_slivers
    x, t = _grid_mesh((2, 2, 2), [0.002, 0, 0])           # a needle cluster at the centre node
    lims = [(0, 2)] * 3
    q = tet_quality(x, t)
    x2, t2, a2, st = collapse_slivers(x, t, np.ones(len(t), int), 0.05, lims=lims)
    assert (q < 0.05).any() and st['collapsed'] >= 1 and st['nodes_removed'] == st['collapsed']
    assert len(x2) == len(x) - st['nodes_removed'] and abs(_vol(x2, t2) - 8) < 1e-12
    assert tet_quality(x2, t2).min() > q.min() and st['moved_volume'] < 1e-12
    on_box = lambda p: (np.isclose(p, 0) | np.isclose(p, 2)).any(1)
    assert on_box(x2).sum() == on_box(x).sum()      # box-face nodes all kept
    assert np.bincount(t2.ravel(), minlength=len(x2)).min() > 0


def _kuhn_mesh(m=4, L=2.0):
    """periodic-conforming tet mesh of [0, L]^3: m^3 cubes, 6 Kuhn tets each (translation invariant)"""
    g = np.linspace(0, L, m + 1)
    x = np.array([[i, j, k] for i in g for j in g for k in g], float)
    nid = lambda i, j, k: (i * (m + 1) + j) * (m + 1) + k
    paths = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    t = []
    for i, j, k in np.ndindex(m, m, m):
        for p in paths:
            c, tet = [i, j, k], [nid(i, j, k)]
            for d in p:
                c[d] += 1
                tet.append(nid(*c))
            t.append(tet)
    return x, np.array(t), nid


def _box_faces(x, t, d, side, L):
    from mdvoro.mesh.sliver_flip import FACES
    out = set()
    for r in t:
        for f in FACES:
            p = x[r[list(f)]]
            if np.allclose(p[:, d], side):
                q = p.copy(); q[:, d] = 0
                out.add(frozenset(map(tuple, np.round(q, 9))))
    return out


def test_collapse_merges_box_face_nodes_with_their_periodic_images():
    from mdvoro.mesh.sliver_flip import collapse_slivers
    x, t, nid = _kuhn_mesh()
    for i in (0, 4):                                  # move a node of the face x = 0 and its image at x = 2
        x[nid(i, 2, 2), 1] += 0.45 * 0.5            # toward its neighbour in +y: needle tets on both sides
    lims = [(0, 2)] * 3
    q = tet_quality(x, t)
    x2, t2, a2, st = collapse_slivers(x, t, np.ones(len(t), int), 0.2, lims=lims)
    assert st['collapsed_box'] >= 1 and st['nodes_removed'] >= 2 and abs(_vol(x2, t2) - 8) < 1e-12
    assert tet_quality(x2, t2).min() > q.min()
    assert np.bincount(t2.ravel(), minlength=len(x2)).min() > 0
    for d in range(3):                               # the opposite face triangulations are still copies
        assert _box_faces(x2, t2, d, 0, 2) == _box_faces(x2, t2, d, 2, 2)
        a = {tuple(np.delete(p, d)) for p in x2[np.isclose(x2[:, d], 0)]}
        b = {tuple(np.delete(p, d)) for p in x2[np.isclose(x2[:, d], 2)]}
        assert a == b


def test_collapse_needs_the_periodic_image():
    from mdvoro.mesh.sliver_flip import collapse_slivers
    x, t, nid = _kuhn_mesh()
    x[nid(0, 2, 2), 1] += 0.45 * 0.5                 # moved without its image: no partner at x = 2
    x2, t2, _, st = collapse_slivers(x, t, np.ones(len(t), int), 0.2, lims=[(0, 2)] * 3)
    assert st['collapsed_box'] == 0 and st['reject']['box_faces'] > 0
    assert abs(_vol(x2, t2) - 8) < 1e-12


def test_collapse_keeps_interface_nodes_on_their_interface():
    from mdvoro.mesh.sliver_flip import collapse_slivers
    x, t = _grid_mesh((2, 2, 2), [0, 0, 0.002], plane_x1=True)
    c = x[t].mean(1)
    attr = np.where(c[:, 0] < 1, 1, 2)              # interface x = 1 through both close nodes
    x2, t2, a2, st = collapse_slivers(x, t, attr, 0.05, lims=[(0, 2)] * 3)
    assert st['collapsed'] >= 1 and st['moved_volume'] < 1e-12
    for lab in (1, 2):                               # each region keeps its volume (4)
        assert abs(_vol(x2, t2[a2 == lab]) - 4) < 1e-12


def _jagged_kuhn(m=4, L=2.0):
    """the Kuhn mesh with cubes (0, 1..2, 1..2) moved by +L in x: a periodic domain with a jagged boundary"""
    x, t, nid = _kuhn_mesh(m, L)
    h = L / m
    c = x[t].mean(1)
    mv = (c[:, 0] < h) & (np.abs(c[:, 1] - 2 * h) < h) & (np.abs(c[:, 2] - 2 * h) < h)
    x = list(map(tuple, x))
    idx = {p: i for i, p in enumerate(x)}
    t = t.copy()
    for r in np.nonzero(mv)[0]:
        for a in range(4):
            p = tuple(np.array(x[t[r, a]]) + [L, 0, 0])
            if p not in idx:
                idx[p] = len(x); x.append(p)
            t[r, a] = idx[p]
    x = np.array(x)
    used = np.unique(t)
    remap = -np.ones(len(x), int); remap[used] = np.arange(len(used))
    return x[used], remap[t], lambda p: int(remap[idx[tuple(map(float, p))]])


def test_collapse_periodic_whole_cells_moves_translates_together():
    from mdvoro.mesh.sliver_flip import collapse_slivers, FACES
    from mdvoro.mesh.poly_tools import translation_classes
    x, t, at = _jagged_kuhn()
    u, us = at((0.5, 1.0, 1.0)), at((2.5, 1.0, 1.0))   # inside the step face x = 0.5 and its translate
    for i in (u, us):
        x[i, 1] += 0.9 * 0.5                          # in the face, toward its +y neighbour: needles on both sides
    q = tet_quality(x, t)
    x2, t2, a2, st = collapse_slivers(x, t, np.ones(len(t), int), 0.1, period=[2, 2, 2])
    assert st['collapsed_box'] >= 1 and st['nodes_removed'] >= 2 and abs(_vol(x2, t2) - 8) < 1e-12
    assert tet_quality(x2, t2).min() > q.min()
    assert np.bincount(t2.ravel(), minlength=len(x2)).min() > 0
    f = np.sort(t2[:, FACES], axis=2).reshape(-1, 3)
    uf, cnt = np.unique(f, axis=0, return_counts=True)
    bn = np.unique(uf[cnt == 1])
    cls, _ = translation_classes(x2[bn], np.array([2.0, 2, 2]), 1e-9)
    assert np.bincount(cls)[cls].min() >= 2          # every boundary node still has a translate
    # the boundary triangles pair up by translation
    key = lambda tri: tuple(sorted(cls[np.searchsorted(bn, tri)].tolist()))
    from collections import Counter
    assert all(c % 2 == 0 for c in Counter(key(tri) for tri in uf[cnt == 1]).values())


def test_collapse_periodic_needs_the_translate():
    from mdvoro.mesh.sliver_flip import collapse_slivers
    x, t, at = _jagged_kuhn()
    x[at((0.5, 1.0, 1.0)), 1] += 0.9 * 0.5           # moved without its translate
    try:
        collapse_slivers(x, t, np.ones(len(t), int), 0.1, period=[2, 2, 2])
    except ValueError as e:
        assert 'translate' in str(e)
    else:
        raise AssertionError('expected a not-periodic error')


def test_periodic_recut_makes_a_seam_tet_interior():
    from mdvoro.mesh.sliver_flip import periodic_recut, FACES
    from mdvoro.mesh.poly_tools import translation_classes
    x, t, at = _jagged_kuhn()
    f = np.sort(t[:, FACES], axis=2).reshape(-1, 3)
    uf, inv, cnt = np.unique(f, axis=0, return_inverse=True, return_counts=True)
    bfc = (cnt[inv] == 1).reshape(-1, 4).sum(1)
    k = int(np.argmax(bfc))                             # a tet with a face on the boundary
    assert bfc[k] > 0
    x2, t2, moved = periodic_recut(x, t, [k], [2, 2, 2], 0.9)
    assert moved > 0 and len(t2) == len(t) and abs(_vol(x2, t2) - 8) < 1e-12
    f2 = np.sort(t2[:, FACES], axis=2).reshape(-1, 3)
    uf2, inv2, cnt2 = np.unique(f2, axis=0, return_inverse=True, return_counts=True)
    assert cnt2.max() == 2 and (cnt2[inv2] == 1).reshape(-1, 4)[k].sum() == 0   # conforming; tet k interior
    bn = np.unique(uf2[cnt2 == 1])
    cls, _ = translation_classes(x2[bn], np.array([2.0, 2, 2]), 1e-9)
    assert np.bincount(cls)[cls].min() >= 2            # still periodic by translation
    assert len(np.unique(np.round(x2, 9), axis=0)) == len(x2)
