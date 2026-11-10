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


def collapse_short_edges(faces, xyz, min_edge, lims=None, frozen=(), group=None, anchors=(), min_edge_anchor=None):
    '''Merge polygon vertices joined by edges shorter than min_edge.

    Edges are taken shortest first; two vertex clusters merge if every member
    stays within min_edge of the merged centroid. A cluster with a member on a
    face of the box lims stays on that face. Polygons are rewritten with the
    cluster representatives: repeated consecutive vertices are dropped, a
    polygon pinched at a vertex is split into its loops, and loops of fewer than
    three vertices disappear. Unused vertices are removed. Vertices in frozen
    are never merged. group: optional (n,) labels; vertices merge only with
    vertices of the same label. anchors: vertices that never move but take in a
    cluster (at most one anchor per cluster) whose members all lie within
    min_edge_anchor (default min_edge) of the anchor; the merged vertex stays at
    the anchor. Edges at an anchor count as short below min_edge_anchor.

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
    fz = np.zeros(n, dtype=bool)
    fz[list(frozen)] = True
    an = np.zeros(n, dtype=bool)
    an[list(anchors)] = True
    an &= ~fz
    d_an = min_edge if min_edge_anchor is None else min_edge_anchor
    if len(edges):
        at_an = an[edges[:, 0]] | an[edges[:, 1]]
        length = np.where(fz[edges[:, 0]] | fz[edges[:, 1]] | (an[edges[:, 0]] & an[edges[:, 1]]), np.inf, length)
        thr = np.where(at_an, d_an, min_edge)
    else:
        thr = np.zeros(0)
    short = np.nonzero(length < thr)[0]
    short = short[np.argsort(length[short])]
    anchor = {int(v): int(v) for v in np.nonzero(an)[0]}   # cluster root -> its anchor

    parent = np.arange(n)
    members = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in short:
        ra, rb = find(edges[k, 0]), find(edges[k, 1])
        if ra == rb or (group is not None and group[ra] != group[rb]):
            continue
        aa, ab = anchor.get(ra), anchor.get(rb)
        if aa is not None and ab is not None:
            continue
        m = members.get(ra, [ra]) + members.get(rb, [rb])
        P = xyz[m]
        a = aa if aa is not None else ab
        if a is not None:
            if np.linalg.norm(P - xyz[a], axis=1).max() > d_an:
                continue
        elif np.linalg.norm(P - P.mean(axis=0), axis=1).max() > min_edge:
            continue
        r, o = min(ra, rb), max(ra, rb)
        parent[o] = r
        members[r] = m
        members.pop(o, None)
        anchor.pop(ra, None); anchor.pop(rb, None)
        if a is not None:
            anchor[r] = a

    pos = xyz.copy()
    if lims is not None:
        lo, hi, tol = box_tolerance(lims)
    for r, m in members.items():
        if r in anchor:
            pos[r] = xyz[anchor[r]]
            continue
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


def collapse_short_edges_periodic(faces, xyz, min_edge, lims, frozen=(), merge_box_edges=False, min_edge_box=None):
    '''collapse_short_edges for a periodic box: opposite box faces stay identical.

    Box-face vertices are first collapsed once per periodic class (a vertex on a
    high face is identified with its image on the low face) using the box-face
    polygons only, and every image takes the same merged position. The rest is
    then collapsed with all box-face vertices frozen. Same return values as
    collapse_short_edges.

    Box-edge vertices stay put by default. merge_box_edges: they may merge with
    vertices of the same periodic box edge (the merged vertex stays on the edge,
    and its four images move alike), never with face vertices; corners stay put.

    min_edge_box: in the second pass, an interior vertex (cluster) within this of a
    box-surface vertex merges into it; the box vertex keeps its place, so the box
    faces stay as they are (collapse_short_edges anchors). None: box vertices take
    nothing in (an interface vertex just off a box face stays: a sliver at best).
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
    # would need images on faces where that vertex has none. With merge_box_edges
    # only corners do, and a vertex merges only within its own kind (the bit mask
    # of the box planes it lies on: one face, or one edge)
    n_planes = {c: int((on_lo[v] | on_hi[v]).sum()) for v, c in vcls.items()}
    fcls |= {c for c, k in n_planes.items() if k >= (3 if merge_box_edges else 2)}
    group = None
    if merge_box_edges:
        group = np.zeros(len(cpos), dtype=int)
        for v, c in vcls.items():
            group[c] = int(np.dot(on_lo[v] | on_hi[v], [1, 2, 4]))

    # pass 1: periodic classes on the box surface
    target, pos = np.arange(n), xyz.copy()
    if box_faces:
        _, p1, keep1, _, rep1 = collapse_short_edges(box_faces, np.array(cpos), min_edge, lims, fcls, group)
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
    anchors_a = ()
    if min_edge_box:
        user = {new[int(target[v])] for v in frozen if int(target[v]) in new}
        anchors_a = {new[int(target[v])] for v in np.nonzero(on_box)[0] if int(target[v]) in new} - user
        frozen_a = user
    f2, p2, keep2, src2, rep2 = collapse_short_edges(faces_a, pos[used], min_edge, lims, frozen_a,
                                                     anchors=anchors_a, min_edge_anchor=min_edge_box)
    rep = np.array([keep_a[rep2[new[int(target[v])]]] if int(target[v]) in new else int(target[v])
                    for v in range(n)])
    return f2, p2, keep_a[keep2], [src[s] for s in src2], rep


# ---------------------------------------------------------------- whole periodic cells
# A fully periodic tessellation meshed as it is (no box cut): the domain is the union of the
# cells whose seeds lie in the box, a fundamental domain with a jagged boundary. Its boundary
# faces come in pairs that are translates of each other by a lattice vector n L. Vertices
# that are translates of each other are the same point of the periodic structure (a class);
# the PLC is cleaned on the classes, so every translate stays a translate.

def translation_classes(xyz, L, tol=1e-5):
    '''(cls, off): vertices that are translates of each other by n L (n in {-1, 0, 1}^3)
    share a class; xyz[v] = xyz[root of cls[v]] + off[v] L. Interior vertices are classes
    of their own. Raises ValueError if a vertex is a translate of itself (box too small).'''
    from scipy.spatial import cKDTree
    import itertools
    xyz = np.asarray(xyz, float)
    L = np.asarray(L, float)
    n = len(xyz)
    tree = cKDTree(xyz)
    adj = [[] for _ in range(n)]
    for s in itertools.product((-1, 0, 1), repeat=3):
        if s <= (0, 0, 0):                  # each pair once: s and -s
            continue
        s = np.array(s)
        hits = tree.query_ball_point(xyz + s * L, tol)
        for a, hs in enumerate(hits):
            for b in hs:                    # x_b = x_a + s L
                adj[a].append((b, s))
                adj[b].append((a, -s))
    cls = np.full(n, -1)
    off = np.zeros((n, 3), int)
    for r in range(n):
        if cls[r] >= 0:
            continue
        cls[r] = r
        stack = [r]
        while stack:
            a = stack.pop()
            for b, s in adj[a]:
                if cls[b] < 0:
                    cls[b], off[b] = r, off[a] + s
                    stack.append(b)
                elif (off[b] != off[a] + s).any():
                    raise ValueError('a vertex is a periodic image of itself: box too small for this tolerance')
    return cls, off


def collapse_short_edges_translational(faces, xyz, min_edge, L, frozen=(), tol=1e-5):
    '''collapse_short_edges for whole periodic cells (translation_classes): clusters are
    formed on the vertex classes, and every translate of a merged vertex moves alike, so
    paired boundary faces stay translates of each other. Nothing is pinned to a plane.
    Same return values as collapse_short_edges.'''
    xyz = np.asarray(xyz, float)
    L = np.asarray(L, float)
    n = len(xyz)
    cls, off = translation_classes(xyz, L, tol)
    roots = np.unique(cls)
    P = {int(c): xyz[c] for c in roots}                 # class position (its root vertex)
    fz = {int(cls[v]) for v in frozen}
    # quotient edges: (class a, class b, shift of b relative to a in units of L)
    E = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            if a == b:
                continue
            ca, cb = int(cls[a]), int(cls[b])
            sab = off[b] - off[a]
            if (ca, cb) > (cb, ca) or (ca == cb):
                ca, cb, sab = cb, ca, -sab
            key = (ca, cb, tuple(sab))
            if key not in E:
                E[key] = np.linalg.norm(xyz[b] - xyz[a])
    keys = sorted((k for k, l in E.items() if l < min_edge and k[0] not in fz and k[1] not in fz),
                  key=lambda k: E[k])
    parent = {int(c): int(c) for c in roots}
    k_of = {int(c): np.zeros(3, int) for c in roots}    # class offset in its cluster's frame
    members = {}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for ca, cb, s in keys:
        ra, rb = find(ca), find(cb)
        s = np.array(s)
        if ra == rb:
            continue
        ma, mb = members.get(ra, [ra]), members.get(rb, [rb])
        sh = k_of[ca] + s - k_of[cb]                    # B frame -> A frame
        pos = [P[c] + k_of[c] * L for c in ma] + [P[c] + (k_of[c] + sh) * L for c in mb]
        Q = np.array(pos)
        if np.linalg.norm(Q - Q.mean(axis=0), axis=1).max() > min_edge:
            continue
        # a class may not meet a translate of itself (pinches a boundary pair together)
        if set(ma) & set(mb):
            continue
        for c in mb:
            k_of[c] = k_of[c] + sh
            parent[c] = ra
        members[ra] = ma + mb
        members.pop(rb, None)
    center = {}
    for r, m in members.items():
        center[r] = np.mean([P[c] + k_of[c] * L for c in m], axis=0)
    # every vertex -> (cluster, offset); vertices with the same key are one new vertex
    newkey, newpos, rep_of_key = {}, [], {}
    vid = np.empty(n, int)
    for v in range(n):
        c = int(cls[v])
        r = find(c)
        o = off[v] - k_of[c]
        key = (r, tuple(o))
        if key not in newkey:
            newkey[key] = len(newpos)
            base = center[r] if r in center else P[r]
            newpos.append(base + o * L)
            rep_of_key[key] = v
        vid[v] = newkey[key]
    out, src = [], []
    for i, f in enumerate(faces):
        g = [int(vid[v]) for v in f]
        g = [v for k, v in enumerate(g) if v != g[k - 1]] if len(g) > 1 else g
        loops = _split_pinched(g)
        out += loops
        src += [i] * len(loops)
    used = sorted({v for f in out for v in f})
    new = {v: k for k, v in enumerate(used)}
    rep_vertex = np.array([rep_of_key[k] for k in newkey])  # new id -> an old vertex
    keep = rep_vertex[used]
    rep = rep_vertex[vid]
    return [[new[v] for v in f] for f in out], np.array(newpos)[used], keep, src, rep


def unique_triangulated_facets_translational(faces, boundaries, xyz, L, area_tol=1e-12, tol=1e-5):
    '''unique_triangulated_facets for whole periodic cells: a face and its translate (same
    vertex classes) get the same triangles, so the boundary mesh is periodic under -Y.'''
    xyz = np.asarray(xyz, float)
    cls, _ = translation_classes(xyz, L, tol)
    seen, seen_tri, tris, marks = set(), set(), [], []
    by_class = {}                                          # class key -> triangles as class triples
    for f, b in zip(faces, boundaries):
        key = tuple(sorted(f))
        if key in seen:
            continue
        seen.add(key)
        f = list(f)
        ck = tuple(sorted(int(cls[v]) for v in f))
        v_of = {int(cls[v]): v for v in f}
        if len(v_of) != len(f):                            # a face holding two translates of one point
            ctris = None
        elif ck in by_class:
            ctris = by_class[ck]
        else:
            k0 = min(range(len(f)), key=lambda k: cls[f[k]])
            g = f[k0:] + f[:k0]                            # start at the smallest class: same ears on both copies
            ctris = [tuple(int(cls[v]) for v in t) for t in ear_clip(g, xyz)]
            by_class[ck] = ctris
        new = ear_clip(f, xyz) if ctris is None else [tuple(v_of[c] for c in t) for t in ctris]
        for t in new:
            p = xyz[list(t)]
            if 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) > area_tol:
                tris.append(t)
                marks.append(b)
    # a triangle made by two faces (collapsed faces sharing a run of vertices make the same ear):
    # one copy, unless it then dangles (an edge no other triangle has: a fin TetGen removes as
    # exterior, jettisoning and renumbering its vertices); repeated, as a fin can hide another
    from collections import Counter
    n_of = Counter(tuple(sorted(t)) for t in tris)
    seen_tri.clear()
    out_t, out_m = [], []
    for t, m in zip(tris, marks):
        k = tuple(sorted(t))
        if k not in seen_tri:
            seen_tri.add(k)
            out_t.append(t)
            out_m.append(m)
    edges = lambda t: [(min(a, b), max(a, b)) for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))]
    while True:
        E = Counter(e for t in out_t for e in edges(t))
        drop = {i for i, t in enumerate(out_t) if n_of[tuple(sorted(t))] > 1 and any(E[e] == 1 for e in edges(t))}
        if not drop:
            break
        out_t = [t for i, t in enumerate(out_t) if i not in drop]
        out_m = [m for i, m in enumerate(out_m) if i not in drop]
    return out_t, out_m, len(seen)
