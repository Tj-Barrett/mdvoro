import numpy as np

from mdvoro.mesh.poly_tools import snap_to_box


def test_snap_to_box_rounds_a_face_to_one_plane():
    # a face coordinate next to a 4-decimal rounding boundary: voro++ round-off
    # puts its vertices on both sides, which used to split the face in two
    lims = [[0.0, 15.0], [0.0, 15.0], [6.294652, 21.294652]]
    xyz = np.array([[1.0, 1.0, 6.2946499],     # rounds to 6.2946
                    [2.0, 1.0, 6.2946521],     # rounds to 6.2947
                    [3.0, 2.0, 21.2946531],
                    [4.0, 4.0, 10.0]])         # interior: untouched
    before = {round(z, 4) for z in xyz[:2, 2]}
    out = snap_to_box(xyz, lims)
    assert len(before) == 2
    assert len({round(z, 4) for z in out[:2, 2]}) == 1
    assert out[0, 2] == out[1, 2] == 6.294652
    assert out[2, 2] == 21.294652
    assert np.array_equal(out[3], xyz[3])
    assert out[0, 0] == 1.0                    # 1.0 is 1 from the x faces


def test_snap_to_box_leaves_vertices_off_the_faces():
    lims = [[0.0, 10.0]] * 3
    xyz = np.array([[0.001, 5.0, 5.0], [9.9995, 5.0, 5.0], [0.00005, 5.0, 5.0]])
    out = snap_to_box(xyz, lims)
    assert out[0, 0] == 0.001                  # beyond the 1e-4 tolerance
    assert out[1, 0] == 9.9995
    assert out[2, 0] == 0.0


# ---- periodic collapse along box edges -------------------------------------------------
from mdvoro.mesh.poly_tools import collapse_short_edges_periodic  # noqa: E402

L = 10.0
LIMS = [[0.0, L]] * 3


def _box_edge_plc():
    """Box-face polygons of a periodic box [0, 10]^3 along its x-y edge (z axis), in all four
    periodic copies of that edge: two edge vertices 1e-4 apart at z = 5, a face vertex 0.01 off
    the edge at z = 6, and the box corner (0, 0, 0)."""
    edge_z = [0.0, 0.01, 5.0, 5.0001, 6.0]
    polys = []
    for x0 in (0.0, L):                      # on the x faces, touching the edges y = 0 and y = L
        for y0, yin in ((0.0, 3.0), (L, 7.0)):
            ring = [(x0, y0, z) for z in edge_z] + [(x0, yin, 6.0), (x0, yin, 0.0)]
            polys.append(ring)
            polys.append([(x0, y0 + (0.01 if y0 == 0 else -0.01), 6.0), (x0, yin, 6.0), (x0, yin, 8.0)])
    for y0 in (0.0, L):                      # on the y faces, touching the edges x = 0 and x = L
        for x0, xin in ((0.0, 3.0), (L, 7.0)):
            polys.append([(x0, y0, z) for z in edge_z] + [(xin, y0, 6.0), (xin, y0, 0.0)])
    pts, faces = {}, []
    for ring in polys:
        faces.append([pts.setdefault(p, len(pts)) for p in ring])
    return faces, np.array(list(pts))


def _edge_copies(xyz):
    """{(x, y) of an x-y edge copy: sorted z of its vertices}"""
    out = {}
    for x, y, z in xyz:
        if x in (0.0, L) and y in (0.0, L):
            out.setdefault((x, y), []).append(round(z, 6))
    return {k: sorted(v) for k, v in out.items()}


def test_box_edge_vertices_stay_put_by_default():
    faces, xyz = _box_edge_plc()
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS)
    for zs in _edge_copies(out).values():
        assert 5.0 in zs and 5.0001 in zs


def test_merge_box_edges_merges_alike_in_all_four_copies():
    faces, xyz = _box_edge_plc()
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS, merge_box_edges=True)
    copies = _edge_copies(out)
    assert len(copies) == 4
    near5 = [[z for z in zs if 4.9 < z < 5.1] for zs in copies.values()]
    assert all(len(z) == 1 for z in near5)                   # the pair became one vertex ...
    assert len({z[0] for z in near5}) == 1                   # ... at the same z in every copy
    assert all(len(zs) == len(next(iter(copies.values()))) for zs in copies.values())


def test_merge_box_edges_keeps_corners_and_face_vertices():
    faces, xyz = _box_edge_plc()
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS, merge_box_edges=True)
    rows = {tuple(np.round(p, 6)) for p in out}
    assert (0.0, 0.0, 0.0) in rows                           # corner unmoved
    assert (0.0, 0.0, 0.01) in rows                          # the edge vertex next to it not merged into it
    assert (0.0, 0.01, 6.0) in rows and (L, 0.01, 6.0) in rows   # face vertex 0.01 off the edge kept ...
    assert (0.0, 0.0, 6.0) in rows                           # ... and the edge vertex too


def _interface_near_face():
    """an interior interface polygon whose vertex b lies 0.1 off the box-face vertex a (x = 0)"""
    xyz = np.array([[0.0, 5.0, 5.0], [0.0, 6.0, 5.0], [0.0, 5.5, 6.0],     # box-face triangle a, c, d
                    [0.1, 5.0, 5.0], [2.0, 5.0, 4.0], [2.0, 6.0, 6.0],     # interface b, e, f
                    [L, 5.0, 5.0], [L, 6.0, 5.0], [L, 5.5, 6.0]])          # the image of the face triangle
    faces = [[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 3, 5, 1]]
    return faces, xyz


def test_interior_vertex_stays_off_the_box_by_default():
    faces, xyz = _interface_near_face()
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS)
    assert len(out) == len(xyz)
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.2, LIMS)   # min_edge alone: still frozen
    assert len(out) == len(xyz)


def test_min_edge_box_merges_an_interior_vertex_into_the_box_vertex():
    faces, xyz = _interface_near_face()
    f2, out, keep, src, rep = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS, min_edge_box=0.2)
    rows = {tuple(np.round(p, 9)) for p in out}
    assert len(out) == len(xyz) - 1 and (0.1, 5.0, 5.0) not in rows
    for p in xyz[[0, 1, 2, 6, 7, 8]]:                         # box vertices unmoved, both copies
        assert tuple(p) in rows
    assert rep[3] == rep[0]
    tri = [sorted(tuple(out[v]) for v in f) for f in f2]
    assert sorted(map(tuple, xyz[[0, 4, 5]])) in tri          # the interface now ends on the face vertex


def test_min_edge_box_respects_its_distance_and_frozen():
    faces, xyz = _interface_near_face()
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS, min_edge_box=0.08)
    assert len(out) == len(xyz)                               # 0.1 apart: kept
    _, out, *_ = collapse_short_edges_periodic(faces, xyz, 0.05, LIMS, frozen={0}, min_edge_box=0.2)
    assert len(out) == len(xyz)                               # a frozen box vertex takes nothing in


# ------------------------------------------------------------- whole periodic cells
from mdvoro.mesh.poly_tools import (translation_classes, collapse_short_edges_translational,
                                    unique_triangulated_facets_translational)
from mdvoro.mesh.periodic_clip import snap_to_planes


def _pair_of_faces(short=0.01):
    '''a pentagon near x = 0 with a short edge, its translate by +L in x, and an interior
    triangle sharing a vertex with the first'''
    L = np.array([10.0, 10.0, 10.0])
    P = np.array([[0.1, 1, 1], [0.0, 3, 1], [0.0, 3, 3], [0.1, 2.0, 4], [0.1, 2.0 + short, 4.0]])
    xyz = np.vstack([P, P + [L[0], 0, 0], [[3.0, 2, 2]]])
    faces = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 10]]
    return xyz, faces, L


def test_translation_classes_pair_translates():
    xyz, faces, L = _pair_of_faces()
    cls, off = translation_classes(xyz, L)
    for v in range(5):
        assert cls[v] == cls[v + 5]
        assert np.allclose(xyz[v + 5], xyz[cls[v]] + off[v + 5] * L)
    assert len(set(cls[:5])) == 5 and cls[10] == 10


def test_translational_collapse_moves_every_translate_alike():
    xyz, faces, L = _pair_of_faces()
    f, p, keep, src, rep = collapse_short_edges_translational(faces, xyz, 0.05, L)
    assert len(p) == len(xyz) - 2                       # the short edge merged on both copies
    assert len(f[0]) == len(f[1]) == 4
    a, b = sorted(p[f[0]].tolist()), sorted((p[f[1]] - [L[0], 0, 0]).tolist())
    assert np.allclose(a, b)                            # still translates
    assert rep[3] == rep[4] and rep[8] == rep[9]


def test_translational_triangulation_matches_on_both_copies():
    xyz, faces, L = _pair_of_faces(short=0.5)
    tris, marks, n = unique_triangulated_facets_translational(faces, [0, 0, 0], xyz, L)
    cls, _ = translation_classes(xyz, L)
    first = {tuple(sorted(cls[list(t)])) for t in tris if max(t) < 5}
    second = {tuple(sorted(cls[list(t)])) for t in tris if 5 <= min(t) and max(t) < 10}
    assert first == second and len(first) == 3


def test_snap_to_planes_moves_near_plane_coordinates_only():
    V = np.array([[0.03, 5.0, 9.97], [10.04, 5.0, 5.0], [-0.02, 0.5, 5.0]])
    S = snap_to_planes(V, np.zeros(3), np.full(3, 10.0), 0.05)
    assert np.allclose(S, [[0.0, 5.0, 10.0], [10.0, 5.0, 5.0], [0.0, 0.5, 5.0]])
