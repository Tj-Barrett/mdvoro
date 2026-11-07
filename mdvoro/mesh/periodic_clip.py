'''
Box-clipped periodic Voronoi tessellation.

A fully periodic voro++ tessellation (-p) gives every cell whole, around its
seed, so cells near the boundary stick out of the box. This module clips every
cell and its periodic images (shifts of -1, 0, +1 box lengths per axis) to the
box. The pieces tile the box exactly, and the cut faces on opposite box faces
are the same polygons: the piece of cell i shifted by s, cut at x = hi, and the
piece shifted by s - e_x, cut at x = lo, are the same cross-section of cell i.
A conforming mesh of the pieces therefore has matching nodes on opposite faces
(given a periodic-aware edge collapse and box-face triangulation, see
poly_tools), which periodic boundary conditions need.

The output has the format of a non-periodic voro++ tessellation of the box:
faces on the box carry the voro++ wall ids (-1 x lo, -2 x hi, -3 y lo, -4 y hi,
-5 z lo, -6 z hi) and the other faces the index of the adjacent piece, so the
rest of mdvoro (decimate, _voro_reader, _phase_regions) uses it unchanged.

Every cell is convex, so clipping by a plane is: keep the vertices inside, cut
each face polygon at the plane (Sutherland-Hodgman), and close the cut with a
cap polygon whose vertices are the points on the plane in angular order.
'''
import itertools

import numpy as np

WALLS = {(0, 0): -1, (0, 1): -2, (1, 0): -3, (1, 1): -4, (2, 0): -5, (2, 1): -6}


def _clip(P, faces, d, value, keep_below, tag, eps):
    '''clip a convex polyhedron by the plane x_d = value.

    P: (n, 3) vertices, faces: [(loop, adjacent)], keep_below: keep x_d <= value.
    The cut face gets adjacent = tag. Returns (P, faces) or None if nothing is left.
    '''
    dist = (P[:, d] - value) * (1 if keep_below else -1)     # > 0 outside
    on = np.abs(dist) <= eps
    dist[on] = 0.0
    if on.any():                               # within eps of the plane: exactly on it, so that
        P = P.copy()                           # slivers thinner than eps vanish instead of
        P[on, d] = value                       # rounding into self-intersecting facets
    if (dist <= 0).all():                      # inside; a face lying on the plane becomes the wall
        if any(all(dist[v] == 0 for v in loop) for loop, _ in faces):
            faces = [(loop, tag if all(dist[v] == 0 for v in loop) else adj) for loop, adj in faces]
        return P, faces
    if (dist >= 0).all():
        return None
    pts = [p for p in P]
    cut = {}                                   # edge -> new vertex on the plane

    def cut_vertex(a, b):
        key = (min(a, b), max(a, b))
        if key not in cut:
            t = dist[a] / (dist[a] - dist[b])
            q = P[a] + t * (P[b] - P[a])
            q[d] = value
            cut[key] = len(pts)
            pts.append(q)
        return cut[key]

    out, on_plane, has_cap = [], set(), False
    for loop, adj in faces:
        if all(dist[v] == 0 for v in loop):    # face already on the plane: it is the cap
            out.append((list(loop), tag))
            has_cap = True
            continue
        new = []
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if dist[a] <= 0:
                new.append(a)
                if dist[a] == 0:
                    on_plane.add(a)
            if (dist[a] < 0 < dist[b]) or (dist[b] < 0 < dist[a]):
                v = cut_vertex(a, b)
                new.append(v)
                on_plane.add(v)
        if len(new) >= 3:
            out.append((new, adj))
    if not has_cap and len(on_plane) >= 3:
        on_plane = sorted(on_plane)
        Q = np.array([pts[v] for v in on_plane])
        c = Q.mean(axis=0)
        u, w = [k for k in range(3) if k != d]
        ang = np.arctan2(Q[:, w] - c[w], Q[:, u] - c[u])
        cap = [on_plane[k] for k in np.argsort(ang, kind='stable')]
        if np.dot(_normal(np.array([pts[v] for v in cap])), np.eye(3)[d]) * (1 if keep_below else -1) < 0:
            cap = cap[::-1]                      # outward: +e_d when keeping below the plane
        out.append((cap, tag))
    used = sorted({v for loop, _ in out for v in loop})
    new_id = {v: k for k, v in enumerate(used)}
    return np.array([pts[v] for v in used]), [([new_id[v] for v in loop], adj) for loop, adj in out]


def _normal(Q):
    '''Newell normal (times twice the area) of the polygon Q'''
    R = np.roll(Q, -1, axis=0)
    return np.array([np.sum((Q[:, 1] - R[:, 1]) * (Q[:, 2] + R[:, 2])),
                     np.sum((Q[:, 2] - R[:, 2]) * (Q[:, 0] + R[:, 0])),
                     np.sum((Q[:, 0] - R[:, 0]) * (Q[:, 1] + R[:, 1]))])


def _volume(P, faces):
    vol = 0.0
    for loop, _ in faces:
        a = P[loop[0]]
        for b, c in zip(loop[1:-1], loop[2:]):
            vol += np.dot(a, np.cross(P[b], P[c]))
    return vol / 6.0


def shared_vertices(voro, lims, tol):
    '''cell vertices with every copy of a Voronoi vertex at one position.

    voro++ prints vertices with ~6 significant digits, so the copies of a vertex
    shared by several cells differ slightly. Cutting an edge that is nearly
    parallel to a box plane amplifies the difference, and the cut points of
    neighbouring pieces no longer coincide. Copies within tol (periodically) are
    moved to their mean. Returns a list of (n_i, 3) arrays.
    '''
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    lo = np.array([l[0] for l in lims], float)
    L = np.array([l[1] - l[0] for l in lims], float)
    V = [np.array(c['vertices'], float) for c in voro]
    X = np.concatenate(V)
    W = np.mod(X - lo, L)
    W[W >= L] = 0.0
    pairs = cKDTree(W, boxsize=L).query_pairs(tol, output_type='ndarray')
    n = len(X)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                  directed=False)
    anchor = np.full(lab.max() + 1, -1)
    anchor[lab[::-1]] = np.arange(n)[::-1]       # first member of each cluster
    off = W - W[anchor[lab]]
    off -= L * np.round(off / L)                 # minimum image relative to the anchor
    mean = np.zeros((len(anchor), 3))
    np.add.at(mean, lab, off)
    mean = W[anchor] + mean / np.bincount(lab)[:, None]
    # each copy: the cluster position, in the copy's own periodic image
    Y = mean[lab]
    Y += L * np.round((W - Y) / L)
    Y += X - W                                   # back to the unwrapped coordinates of the cell
    out, k = [], 0
    for v in V:
        out.append(Y[k:k + len(v)])
        k += len(v)
    return out


def clip_periodic_cells(voro, lims, eps=None, min_volume=1e-12):
    '''Clip a fully periodic voro++ tessellation of the box lims to the box.

    voro: cells as returned by compute_voro(..., periodic=[True]*3, method='voro++')
    (whole cells around seeds inside the box, faces with adjacent cell ids).
    Returns pieces in the same format; each piece also has 'source' = (cell, shift),
    shift an integer 3-vector in box lengths. 'original' is the seed moved by the
    shift, or the piece's vertex centroid when that seed is outside the box (it
    is used as a TetGen region point and must lie inside the piece).
    '''
    lo = np.array([l[0] for l in lims], float)
    hi = np.array([l[1] for l in lims], float)
    L = hi - lo
    if eps is None:                            # the box tolerance of poly_tools (4-decimal output)
        eps = 1e-4 + 1e-9 * L.max()
    seeds = np.array([c['original'][:3] for c in voro], float)
    verts = shared_vertices(voro, lims, eps)
    pieces, index = [], {}
    shifts = [np.array(s) for s in itertools.product((-1, 0, 1), repeat=3)]
    for i, cell in enumerate(voro):
        P0 = verts[i]
        faces0 = [(list(f['vertices']), f['adjacent_cell']) for f in cell['faces']]
        if _volume(P0, faces0) < 0:            # voro++ loops have inward normals: make them outward
            faces0 = [(loop[::-1], adj) for loop, adj in faces0]
        for s in shifts:
            P = P0 + s * L
            if (P.max(axis=0) <= lo + eps).any() or (P.min(axis=0) >= hi - eps).any():
                continue
            poly = (P, faces0)
            for (d, side), tag in WALLS.items():
                poly = _clip(*poly, d, hi[d] if side else lo[d], keep_below=bool(side), tag=tag, eps=eps)
                if poly is None:
                    break
            if poly is None:
                continue
            P, faces = poly
            vol = _volume(P, faces)
            if vol <= min_volume:
                continue
            index[(i, tuple(s))] = len(pieces)
            seed = seeds[i] + s * L
            inside = ((seed > lo + eps) & (seed < hi - eps)).all()
            pieces.append({'original': [*(seed if inside else P.mean(axis=0)), cell['original'][3]],
                           'volume': vol, 'vertices': P.tolist(), 'faces': faces, 'source': (i, tuple(s))})

    # adjacent cell ids -> adjacent piece ids: the neighbour j of cell i is the
    # image of seed j nearest to seed i; seen from piece (i, s) it is piece (j, s + t)
    for piece in pieces:
        i, s = piece['source']
        out = []
        for loop, adj in piece['faces']:
            if adj < 0:
                out.append({'vertices': loop, 'adjacent_cell': adj})
                continue
            t = np.round((seeds[i] - seeds[adj]) / L).astype(int)
            key = (adj, tuple(int(x) for x in np.array(s) + t))
            if key not in index:                # neighbour was a sliver thinner than eps
                Q = np.array(piece['vertices'])[loop]
                wall = [tag for (d, side), tag in WALLS.items()
                        if np.allclose(Q[:, d], hi[d] if side else lo[d], atol=eps)]
                if wall:
                    out.append({'vertices': loop, 'adjacent_cell': wall[0]})
                    continue
                if np.linalg.norm(_normal(Q)) <= eps * eps:
                    continue                    # degenerate face
                raise RuntimeError(f'piece {key} adjacent to {piece["source"]} was clipped away')
            out.append({'vertices': loop, 'adjacent_cell': index[key]})
        piece['faces'] = out
        piece['adjacency'] = [f['adjacent_cell'] for f in out]
    return pieces
