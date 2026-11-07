'''
Facet clean-up for TetGen piecewise linear complexes.

The faces collected from the Voronoi cells contain every interface twice (once
per adjacent cell), and after vertices are merged and rounded many polygons are
slightly non-planar or no longer convex. TetGen needs each facet once and planar;
duplicated or self-overlapping facets crash its surface mesher. These helpers
remove duplicates and split each polygon into triangles (planar by construction)
by ear clipping in the polygon's best-fit plane, which also handles non-convex
polygons.

Voronoi diagrams of MD beads also have many near-degenerate vertices joined by
very short edges (10 % of the edges shorter than a fifth of the median for a
melt). TetGen keeps every PLC vertex, so these produce sliver tetrahedra and
ill-conditioned stiffness matrices; collapse_short_edges merges them first.
'''
import numpy as np


def ear_clip(vs, xyz, uv=None):
    '''triangles (vertex-id triples) of the simple polygon vs, xyz[v] its coordinates.
    uv: 2-D coordinates of the vertices (default: the best-fit plane)'''
    if len(vs) == 3:
        return [tuple(vs)]
    if uv is None:
        P = np.array([xyz[v] for v in vs], dtype=float)
        c = P.mean(axis=0)
        _, _, Vt = np.linalg.svd(P - c)
        uv = (P - c) @ Vt[:2].T                   # 2-D coordinates in the best-fit plane
    uv = np.asarray(uv, dtype=float)
    area = 0.5 * np.sum(uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1])
    idx = list(range(len(vs))) if area > 0 else list(range(len(vs)))[::-1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, d = uv[i0], uv[i1], uv[i2]
            if cross(a, b, d) <= 0:               # reflex vertex: not an ear
                continue
            if any(cross(a, b, uv[j]) >= 0 and cross(b, d, uv[j]) >= 0 and cross(d, a, uv[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((vs[i0], vs[i1], vs[i2]))
            idx.pop(k)
            break
        else:                                     # numerically degenerate remainder: fan
            break
    rest = [vs[i] for i in idx]
    tris += [(rest[0], rest[k], rest[k + 1]) for k in range(1, len(rest) - 1)]
    return tris


def unique_triangulated_facets(faces, boundaries, xyz, area_tol=1e-12, lims=None):
    '''Each distinct polygon once, split into triangles.

    faces: list of vertex-index lists, boundaries: marker per face, xyz: vertex
    coordinates indexed like the faces. Triangles with an area below area_tol
    (collinear vertices) are dropped. Returns (triangles, markers, n_unique).

    lims: periodic box. Polygons on a box face are then triangulated from their
    in-plane coordinates only, starting at the lexicographically smallest vertex,
    so that the same polygon on opposite faces gets the same triangles.
    '''
    xyz = np.asarray(xyz, dtype=float)
    seen, seen_tri, tris, marks = set(), set(), [], []
    for f, b in zip(faces, boundaries):
        key = tuple(sorted(f))
        if key in seen:
            continue
        seen.add(key)
        f, uv = list(f), None
        d = box_plane(f, xyz, lims) if lims is not None else None
        if d is not None:
            uv = xyz[f][:, [k for k in range(3) if k != d]]
            area = np.sum(uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1])
            order = list(range(len(f))) if area > 0 else list(range(len(f)))[::-1]
            k0 = min(order, key=lambda k: (round(uv[k, 0], 6), round(uv[k, 1], 6)))
            order = order[order.index(k0):] + order[:order.index(k0)]
            f, uv = [f[k] for k in order], uv[order]
        for t in ear_clip(f, xyz, uv):
            p = xyz[list(t)]
            tkey = tuple(sorted(t))
            if tkey not in seen_tri and 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) > area_tol:
                seen_tri.add(tkey)
                tris.append(t)
                marks.append(b)
    return tris, marks, len(seen)


def box_tolerance(lims):
    """distance within which a vertex counts as on a face of the box lims
    (voro++ writes vertices with ~6 significant digits)"""
    lo = np.array([l[0] for l in lims], float)
    hi = np.array([l[1] for l in lims], float)
    return lo, hi, 1e-4 + 1e-9 * (hi - lo)


def box_plane(f, xyz, lims):
    '''axis d if every vertex of polygon f lies on the same face of the box lims, else None'''
    lo, hi, tol = box_tolerance(lims)
    P = xyz[list(f)]
    for d in range(3):
        if (np.abs(P[:, d] - lo[d]) <= tol[d]).all() or (np.abs(P[:, d] - hi[d]) <= tol[d]).all():
            return d
    return None


def snap_to_box(xyz, lims):
    """Move vertices within box_tolerance of a face of lims exactly onto it.

    Vertices on a box face carry voro++ round-off; written with 4 decimals, a
    face whose coordinate lies near a rounding boundary (e.g. 6.294652) splits
    into two planes 1e-4 apart (6.2946 and 6.2947), and the mesh boundary is no
    longer the box. Returns a new (n, 3) array.
    """
    xyz = np.array(xyz, dtype=float)
    lo, hi, tol = box_tolerance(lims)
    for d in range(3):
        xyz[np.abs(xyz[:, d] - lo[d]) <= tol[d], d] = lo[d]
        xyz[np.abs(xyz[:, d] - hi[d]) <= tol[d], d] = hi[d]
    return xyz


def collapse_short_edges(faces, xyz, min_edge, lims=None, frozen=()):
    '''Merge polygon vertices joined by edges shorter than min_edge.

    Edges are taken shortest first; two vertex clusters merge if every member
    stays within min_edge of the merged centroid. A cluster with a member on a
    face of the box lims stays on that face. Polygons are rewritten with the
    cluster representatives: repeated consecutive vertices are dropped, a
    polygon pinched at a vertex is split into its loops, and loops of fewer than
    three vertices disappear. Unused vertices are removed. Vertices in frozen
    are never merged.

    Returns (faces, xyz, keep, src, rep): faces index the new xyz, keep maps
    new vertex -> old vertex (the representative's original index), src maps
    each new face to the face it came from and rep maps every old vertex to
    its representative.
    '''
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    edges = set()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            if a != b:
                edges.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edges)) if edges else np.zeros((0, 2), int)
    length = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1) if len(edges) else np.zeros(0)
    if len(frozen):
        fz = np.zeros(n, dtype=bool)
        fz[list(frozen)] = True
        length = np.where(fz[edges[:, 0]] | fz[edges[:, 1]], np.inf, length)
    short = np.nonzero(length < min_edge)[0]
    short = short[np.argsort(length[short])]

    parent = np.arange(n)
    members = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in short:
        ra, rb = find(edges[k, 0]), find(edges[k, 1])
        if ra == rb:
            continue
        m = members.get(ra, [ra]) + members.get(rb, [rb])
        P = xyz[m]
        if np.linalg.norm(P - P.mean(axis=0), axis=1).max() > min_edge:
            continue
        r, o = min(ra, rb), max(ra, rb)
        parent[o] = r
        members[r] = m
        members.pop(o, None)

    pos = xyz.copy()
    if lims is not None:
        lo, hi, tol = box_tolerance(lims)
    for r, m in members.items():
        c = xyz[m].mean(axis=0)
        if lims is not None:
            for d in range(3):
                on_lo = np.abs(xyz[m, d] - lo[d]) <= tol[d]
                on_hi = np.abs(xyz[m, d] - hi[d]) <= tol[d]
                if on_lo.any():
                    c[d] = lo[d]
                elif on_hi.any():
                    c[d] = hi[d]
        pos[r] = c

    rep = np.array([find(i) for i in range(n)])
    out, src = [], []
    for i, f in enumerate(faces):
        g = [int(rep[v]) for v in f]
        g = [v for k, v in enumerate(g) if v != g[k - 1]] if len(g) > 1 else g
        loops = _split_pinched(g)
        out += loops
        src += [i] * len(loops)
    used = sorted({v for f in out for v in f})
    new = {v: k for k, v in enumerate(used)}
    return [[new[v] for v in f] for f in out], pos[used], np.array(used, dtype=int), src, rep


def _split_pinched(f):
    '''loops of a closed vertex sequence that visits a vertex more than once'''
    if len(f) < 3:
        return []
    seen = {}
    for k, v in enumerate(f):
        if v in seen:
            i = seen[v]
            return _split_pinched(f[i:k]) + _split_pinched(f[k:] + f[:i])
        seen[v] = k
    return [f]


def collapse_short_edges_periodic(faces, xyz, min_edge, lims, frozen=()):
    '''collapse_short_edges for a periodic box: opposite box faces stay identical.

    Box-face vertices are first collapsed once per periodic class (a vertex on a
    high face is identified with its image on the low face) using the box-face
    polygons only, and every image takes the same merged position. The rest is
    then collapsed with all box-face vertices frozen. Same return values as
    collapse_short_edges.
    '''
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    lo, hi, tol = box_tolerance(lims)
    on_lo, on_hi = np.abs(xyz - lo) <= tol, np.abs(xyz - hi) <= tol
    on_box = (on_lo | on_hi).any(axis=1)
    canon = np.where(on_hi, lo, np.where(on_lo, lo, xyz))
    cls, cpos = {}, []
    for v in np.nonzero(on_box)[0]:
        key = tuple(np.round(canon[v], 6))
        if key not in cls:
            cls[key] = len(cpos)
            cpos.append(canon[v])
    vcls = {int(v): cls[tuple(np.round(canon[v], 6))] for v in np.nonzero(on_box)[0]}
    box_faces = [[vcls[v] for v in f] for f in faces
                 if all(on_box[v] for v in f) and box_plane(f, xyz, lims) is not None]
    frozen = set(int(v) for v in frozen)
    fcls = {vcls[v] for v in frozen if v in vcls}
    # vertices on a box edge or corner stay put: merging a face vertex into one
    # would need images on faces where that vertex has none
    fcls |= {c for v, c in vcls.items() if (on_lo[v] | on_hi[v]).sum() >= 2}

    # pass 1: periodic classes on the box surface
    target, pos = np.arange(n), xyz.copy()
    if box_faces:
        _, p1, keep1, _, rep1 = collapse_short_edges(box_faces, np.array(cpos), min_edge, lims, fcls)
        cls_pos = {int(k): p1[i] for i, k in enumerate(keep1)}
        groups = {}
        for v, c in vcls.items():
            r = int(rep1[c])
            groups.setdefault((r, tuple(on_hi[v])), []).append(v)
        for (r, side), vs in groups.items():
            q = np.array(cls_pos.get(r, cpos[r]), dtype=float)
            q[list(side)] = hi[list(side)]
            m = min(vs)
            target[vs] = m
            pos[m] = q
    out, src = [], []
    for i, f in enumerate(faces):
        g = [int(target[v]) for v in f]
        g = [v for k, v in enumerate(g) if v != g[k - 1]] if len(g) > 1 else g
        loops = _split_pinched(g)
        out += loops
        src += [i] * len(loops)
    used = sorted({v for f in out for v in f})
    new = {v: k for k, v in enumerate(used)}
    faces_a = [[new[v] for v in f] for f in out]
    keep_a = np.array(used, dtype=int)

    # pass 2: everything else, the box surface frozen
    frozen_a = {new[int(target[v])] for v in np.nonzero(on_box)[0] if int(target[v]) in new}
    frozen_a |= {new[int(target[v])] for v in frozen if int(target[v]) in new}
    f2, p2, keep2, src2, rep2 = collapse_short_edges(faces_a, pos[used], min_edge, lims, frozen_a)
    rep = np.array([keep_a[rep2[new[int(target[v])]]] if int(target[v]) in new else int(target[v])
                    for v in range(n)])
    return f2, p2, keep_a[keep2], [src[s] for s in src2], rep
