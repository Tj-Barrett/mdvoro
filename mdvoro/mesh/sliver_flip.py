'''Fold-sliver removal by PLC edge flips.

A flat tetrahedron abcd whose faces abc and abd are both PLC triangles fills a
fold of an interface: the two triangles meet along ab at nearly 180 degrees, and
TetGen must put a tet into the fold because it may not move PLC vertices. If the
tets across abc and across abd lie in the same region R2 (the sliver in R1 != R2),
replacing the PLC triangles abc + abd by acd + bcd, the sliver's other two faces,
moves the sliver (of near-zero volume) into R2 and removes the fold. The new
triangles are faces of a tet of a valid mesh, so they cannot intersect the PLC.

flip_fold_slivers iterates mesh -> flip -> remesh on a .poly file. Box-face
triangles are never changed (periodic copies stay identical under TetGen -Y). A
flip that would undo an earlier one (a flat quad: both diagonals leave a sliver)
is skipped.
'''
import os
import subprocess
import numpy as np

FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))      # face k is opposite vertex k


def tet_quality(x, t):
    '''q = 6 sqrt(2) V / l_max^3 per tet (1: regular, 0: flat); x (n, 3), t (m, 4) 0-based'''
    p = x[t]
    e = p[:, 1:] - p[:, :1]
    V = np.abs(np.einsum('ij,ij->i', e[:, 0], np.cross(e[:, 1], e[:, 2]))) / 6
    edges = np.stack([e[:, 0], e[:, 1], e[:, 2], p[:, 2] - p[:, 1], p[:, 3] - p[:, 1], p[:, 3] - p[:, 2]], 1)
    return 6 * np.sqrt(2) * V / np.linalg.norm(edges, axis=2).max(1) ** 3


def read_poly(path):
    '''(lines before the facet list, triangles (k, 3) 1-based, lines from the hole list on).
    The facet list must be triangles, one per facet, as mdvoro writes it.'''
    L = open(path).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith('# Facets'))
    j = next(k for k, l in enumerate(L) if l.startswith('# Part 3'))
    tris = []
    for l in L[i + 1:j]:
        p = l.split()
        if p and not l.startswith('#') and p[0] == '3' and len(p) == 4:
            tris.append([int(v) for v in p[1:]])
        elif p and not l.startswith('#') and len(p) == int(p[0]) + 1 and int(p[0]) > 3:
            raise ValueError(f'{path}: polygon facets are not supported, only triangles')
    return L[:i], np.array(tris, dtype=np.int64), L[j:]


def write_poly(path, head, tris, tail):
    out = head + ['# Facets, boundary marker', f'{len(tris)} 1', '# triangles (fold flips applied)']
    for a, b, c in tris:
        out += ['1 0 0', f'3 {a} {b} {c}']
    with open(path, 'w') as f:
        f.write('\n'.join(out + tail) + '\n')


def _read_mesh(prefix):
    def rows(ext):
        r = [l.split() for l in open(prefix + ext) if l.split() and not l.startswith('#')]
        return r[1:]
    x = np.array([[float(v) for v in p[1:4]] for p in rows('.node')])
    el = rows('.ele')
    t = np.array([[int(v) for v in p[1:5]] for p in el]) - 1
    attr = np.array([int(float(p[5])) for p in el])
    nbr = np.array([[int(v) for v in p[1:5]] for p in rows('.neigh')]) - 1     # < 0: no neighbour
    return x, t, attr, nbr


def _on_box(P, lims, tol=1e-6):
    return lims is not None and any((np.abs(P[:, d] - lims[d][s]) < tol).all() for d in range(3) for s in range(2))


def fold_flips(x, t, attr, nbr, tris, lims=None, qthr=1e-3, removed=None, boundary=None):
    '''PLC triangle replacements for the fold slivers of one mesh.

    x, t (0-based), attr, nbr: a TetGen mesh of the PLC whose triangles are tris
    (1-based, the PLC vertices first in x, as TetGen keeps them). Returns
    (keep mask over tris, new triangles 1-based, stats). removed: set of PLC edges
    (sorted 1-based pairs) taken out by earlier flips, updated in place.
    boundary: (n,) translation class of every node (whole periodic cells, poly_tools
    translation_classes; a class of one node is not on the boundary). A fold between two
    boundary triangles is flipped on every translate of the pair alike, so the boundary
    stays periodic (the sliver leaves the domain on one copy and joins it on the other);
    a fold between a boundary and an interior triangle stays.
    '''
    if removed is None:
        removed = set()
    q = tet_quality(x, t)
    idx = {tuple(sorted(r)): k for k, r in enumerate(tris.tolist())}
    keep, new, used, made = np.ones(len(tris), bool), [], set(), set()
    stats = dict.fromkeys(('candidates', 'folds', 'flipped', 'skip_box', 'skip_regions', 'skip_exists',
                           'skip_undo', 'skip_used'), 0)
    stats.update(boundary_flipped=0, skip_boundary=0)
    cls = boundary
    if cls is not None:
        on_b = np.bincount(cls, minlength=len(x))[cls] > 1
        by_cls = {}
        for i, r in enumerate(tris.tolist()):
            by_cls.setdefault(tuple(sorted(cls[np.array(r) - 1])), []).append(i)
    for k in np.nonzero(q < qthr)[0]:
        stats['candidates'] += 1
        v = t[k] + 1
        on = [idx.get(tuple(sorted(v[list(f)]))) for f in FACES]
        fk = [i for i in range(4) if on[i] is not None]
        if len(fk) != 2:
            continue
        stats['folds'] += 1
        c, d = v[fk[1]], v[fk[0]]                 # the PLC faces are opposite c and d: abd and abc
        a, b = [w for w in v if w not in (c, d)]
        old = [on[i] for i in fk]
        if any(_on_box(x[tris[o] - 1], lims) for o in old):
            stats['skip_box'] += 1
            continue
        n1, n2 = nbr[k, fk[0]], nbr[k, fk[1]]
        if cls is not None and any(on_b[tris[o] - 1].all() for o in old):
            # boundary fold: both faces on the domain boundary, flipped on every translate
            ok = n1 < 0 and n2 < 0
            copies = []
            if ok:
                keys = [tuple(sorted(cls[tris[o] - 1])) for o in old]
                for p1 in by_cls.get(keys[0], []):
                    if p1 == old[0]:
                        continue
                    for p2 in by_cls.get(keys[1], []):
                        if len(set(tris[p1]) & set(tris[p2])) == 2:
                            copies.append((p1, p2))
                ok = len(copies) >= 1
            if not ok:
                stats['skip_boundary'] += 1
                continue
            groups = [(old, {int(cls[w - 1]): int(w) for w in (a, b, c, d)})]
            for p1, p2 in copies:
                groups.append(([p1, p2], {int(cls[w - 1]): int(w) for w in set(tris[p1]) | set(tris[p2])}))
            ca, cb, cc, cd = (int(cls[w - 1]) for w in (a, b, c, d))
            plan = []
            for olds, v_of in groups:
                if not all(z in v_of for z in (ca, cb, cc, cd)):
                    plan = None
                    break
                A, B, C, D = (v_of[z] for z in (ca, cb, cc, cd))
                nt = [(A, C, D), (B, C, D)]
                if any(tuple(sorted(f)) in idx or tuple(sorted(f)) in made for f in nt) \
                        or (min(C, D), max(C, D)) in removed or used & set(olds):
                    plan = None
                    break
                plan.append((olds, nt, (min(A, B), max(A, B))))
            if not plan:
                stats['skip_boundary'] += 1
                continue
            for olds, nt, ab in plan:
                used |= set(olds)
                made |= {tuple(sorted(f)) for f in nt}
                removed.add(ab)
                keep[olds] = False
                new += nt
            stats['flipped'] += 1
            stats['boundary_flipped'] += 1
            continue
        if n1 < 0 or n2 < 0 or attr[n1] != attr[n2] or attr[n1] == attr[k]:
            stats['skip_regions'] += 1
            continue
        nt = [(a, c, d), (b, c, d)]
        if any(tuple(sorted(f)) in idx or tuple(sorted(f)) in made for f in nt):   # no duplicate triangles
            stats['skip_exists'] += 1
            continue
        if (min(c, d), max(c, d)) in removed:
            stats['skip_undo'] += 1
            continue
        if used & set(old):
            stats['skip_used'] += 1
            continue
        used |= set(old)
        made |= {tuple(sorted(f)) for f in nt}
        removed.add((min(a, b), max(a, b)))
        keep[old] = False
        new += nt
        stats['flipped'] += 1
    return keep, new, stats


def flip_fold_slivers(poly, tetgen, cmd='pYA', lims=None, qthr=1e-3, iters=6, timeout=None, log=print,
                      period=None, tol=3e-6):
    '''Remove fold slivers of the PLC in poly (rewritten in place) by edge flips.

    tetgen: TetGen binary; cmd: its switches for the trial meshes, or several tried in
    order (n is added for the neighbours; Y keeps the PLC triangles whole, which the
    fold search needs);
    lims: box whose face triangles stay as they are. period: box lengths of whole periodic
    cells; triangles on their boundary (vertices with a translate, within tol) stay.
    Returns one stats dict per
    iteration (with q_min and the counts of tets below 1e-5 and qthr).
    '''
    poly = os.path.abspath(poly)
    wd, name = os.path.split(poly)
    stem = name[:-len('.poly')]
    head, tris, tail = read_poly(poly)
    removed, history, prev = set(), [], None
    cmds = (cmd,) if isinstance(cmd, str) else tuple(cmd)
    for it in range(iters):
        for c in cmds:                    # TetGen's mesh improvement fails on some PLCs that it meshes with
            r = subprocess.run([tetgen, '-' + c + 'n', name], cwd=wd, capture_output=True, text=True,
                               timeout=timeout)   # other switches (e.g. -pYA fails where -pYAq1.4 passes)
            if r.returncode == 0:
                break
        if r.returncode:
            if prev is None:
                raise RuntimeError(f'tetgen -{c}n failed (rc={r.returncode}) during fold flips:\n'
                                   f'{r.stdout[-1500:]}')
            log(f'fold flips {it}: tetgen failed (rc={r.returncode}) on the flipped PLC with '
                f'{", ".join("-" + c for c in cmds)}: taking back the flips of round {it - 1}')
            write_poly(poly, head, prev, tail)
            history.append({'iter': it, 'failed': True, 'flipped': 0})
            break
        x, t, attr, nbr = _read_mesh(os.path.join(wd, stem + '.1'))
        boundary = None
        if period is not None:
            from mdvoro.mesh.poly_tools import translation_classes
            boundary, _ = translation_classes(x, period, tol)
        keep, new, st = fold_flips(x, t, attr, nbr, tris, lims, qthr, removed, boundary)
        q = tet_quality(x, t)
        st.update(iter=it, cmd=c, tets=int(len(t)), q_min=float(q.min()), n_q_1e5=int((q < 1e-5).sum()),
                  n_q_thr=int((q < qthr).sum()))
        history.append(st)
        log(f"fold flips {it}: {len(t)} tets, q_min {st['q_min']:.2e}, {st['n_q_thr']} below {qthr:g}, "
            f"{st['flipped']} of {st['folds']} folds flipped")
        if not new:
            break
        prev, tris = tris, np.vstack([tris[keep], np.array(new, dtype=np.int64)])
        write_poly(poly, head, tris, tail)
    for ext in ('node', 'ele', 'face', 'edge', 'neigh'):
        f = os.path.join(wd, f'{stem}.1.{ext}')
        if os.path.exists(f):
            os.remove(f)
    return history


def delete_slivers(x, t, qthr=1e-3, lims=None, tol=1e-6):
    '''Mask of the tets to keep when the tets of quality below qthr are deleted.

    A flat sliver shares its corners (and edges) with the tets around it, so deleting
    it leaves a void of its own, near-zero volume, not a crack, and takes out the
    near-singular stiffness that breaks the linear solvers. A deleted tet is kept after
    all if it holds a node no other tet has, or, on a box face (lims), an edge no other
    tet has: the periodic copy of that node or of the edge's C3D10 midside node would
    lose its partner. x (n, 3); t (m, 4), 0-based. Returns (keep, stats).
    '''
    q = tet_quality(x, t)
    keep = q >= qthr
    face = np.zeros((len(x), 6), bool)                     # node on box face (d, s)
    if lims is not None:
        for d in range(3):
            for s in range(2):
                face[:, 2 * d + s] = np.abs(x[:, d] - lims[d][s]) < tol
    E = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    n = len(x)
    node_use = np.bincount(t[keep].ravel(), minlength=n)
    key = lambda e: e[..., 0].astype(np.int64) * n + e[..., 1]
    edges = key(np.sort(t[:, E], axis=2))                  # (m, 6)
    cand = np.nonzero(~keep)[0]
    cand = cand[np.argsort(-q[cand])]                      # restore the best of the deleted tets first
    ce = edges[cand].ravel()
    cu = np.unique(ce)
    i, j = cu // n, cu % n
    on_face = (face[i] & face[j]).any(1)
    watch = cu[on_face]                                    # box-face edges of the deleted tets
    ke = edges[keep].ravel()
    edge_use = dict(zip(watch.tolist(), np.bincount(np.searchsorted(watch, ke[np.isin(ke, watch)]),
                                                    minlength=len(watch)).tolist()))
    restored, changed = 0, True
    while changed:                                         # restoring one tet may cover others' nodes/edges
        changed = False
        for k in cand:
            if keep[k]:
                continue
            if (node_use[t[k]] == 0).any() or any(edge_use.get(e, 1) == 0 for e in edges[k].tolist()):
                keep[k] = True
                node_use[t[k]] += 1
                for e in edges[k].tolist():
                    if e in edge_use:
                        edge_use[e] += 1
                restored += 1
                changed = True
    stats = {'qthr': qthr, 'below': int((q < qthr).sum()), 'deleted': int((~keep).sum()), 'restored': restored,
             'volume_fraction': 0.0, 'q_min_kept': float(q[keep].min())}
    p = x[t]
    V = np.abs(np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6
    stats['volume_fraction'] = float(V[~keep].sum() / V.sum())
    return keep, stats


def _orient(a, b, c, d):
    return float(np.dot(b - a, np.cross(c - a, d - a)))


def _ring(tets, t, c, d):
    '''the other vertices of the tets around edge cd as a closed cycle, or None (open: a boundary edge)'''
    nb = {}
    for k in tets:
        p, r = [v for v in t[k] if v != c and v != d]
        nb.setdefault(p, []).append(r)
        nb.setdefault(r, []).append(p)
    if any(len(v) != 2 for v in nb.values()):
        return None
    start = next(iter(nb))
    cyc, prev = [start], None
    while True:
        a, b = nb[cyc[-1]]
        nxt = b if a == prev else a
        if nxt == start:
            break
        prev = cyc[-1]
        cyc.append(nxt)
        if len(cyc) > len(nb):
            return None
    return cyc if len(cyc) == len(nb) else None


def _edge_removal(x, c, d, ring):
    '''best triangulation of the ring polygon for the tets (c, T), (d, T): (worst quality, triangles) or None'''
    n = len(ring)
    P = x[ring]
    best = None
    for S in (1, -1):
        Q = {}
        arg = {}
        def tq(i, m, j):
            a, b, e = P[i], P[m], P[j]
            if S * _orient(a, b, e, x[c]) <= 0 or S * _orient(a, b, e, x[d]) >= 0:
                return -1.0
            q = tet_quality(np.array([x[c], a, b, e, x[d]]), np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))
            return float(q.min())
        for span in range(2, n):
            for i in range(n - span):
                j = i + span
                bq, bm = -1.0, None
                for m in range(i + 1, j):
                    v = min(Q.get((i, m), np.inf), Q.get((m, j), np.inf), tq(i, m, j))
                    if v > bq:
                        bq, bm = v, m
                Q[(i, j)], arg[(i, j)] = bq, bm
        qv = Q[(0, n - 1)]
        if qv > 0 and (best is None or qv > best[0]):
            tris, stack = [], [(0, n - 1)]
            while stack:
                i, j = stack.pop()
                if j - i < 2:
                    continue
                m = arg[(i, j)]
                tris.append((ring[i], ring[m], ring[j]))
                stack += [(i, m), (m, j)]
            best = (qv, tris)
    return best


def absorb_slivers(x, t, attr, qthr=1e-3, passes=3, max_ring=12):
    '''Remove the tets of quality below qthr by edge removal, without deleting volume.

    For a bad tet, each edge cd with a closed ring of tets of one region (an edge on
    no interface and not on the boundary; the bad tet itself may lie in another region,
    and then joins that one: a chip on an interface is absorbed by the region across
    it, which moves the interface by the chip's thickness) can be removed: the ring polygon is
    triangulated (best worst-quality, Klincsek dynamic programme) and each triangle
    T gives the tets (c, T) and (d, T), which fill the same space. The sliver's space
    goes to the new tets of its region; no node moves, interfaces and box faces stay.
    Applied when the worst new tet beats the worst old one. x (n, 3); t (m, 4) 0-based.
    Returns (t, attr, stats); relabelled_volume: the volume that changed region.
    '''
    t, attr = t.copy(), attr.copy()
    p = x[t[:200]]
    s0 = np.sign(np.median(np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))))
    E = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    q0 = tet_quality(x, t)
    stats = {'qthr': qthr, 'below_before': int((q0 < qthr).sum()), 'q_min_before': float(q0.min()), 'removed_edges': 0,
             'relabelled': 0, 'relabelled_volume': 0.0}
    for _ in range(passes):
        q = tet_quality(x, t)
        bad = np.nonzero(q < qthr)[0]
        if not len(bad):
            break
        N = np.unique(t[bad])
        inN = np.zeros(len(x), bool)
        inN[N] = True
        pool = np.nonzero(inN[t].any(1))[0]
        tl = [list(r) for r in t.tolist()]
        al = attr.tolist()
        ql = dict(zip(pool.tolist(), q[pool].tolist()))
        alive = np.ones(len(t), bool)
        emap = {}
        def add(k):
            v = tl[k]
            for a, b in E:
                emap.setdefault((min(v[a], v[b]), max(v[a], v[b])), set()).add(k)
        def drop(k):
            v = tl[k]
            for a, b in E:
                emap[(min(v[a], v[b]), max(v[a], v[b]))].discard(k)
        for k in pool.tolist():
            add(k)
        done = 0
        for k in bad[np.argsort(q[bad])].tolist():
            if k >= len(alive) or not alive[k]:
                continue
            best = None
            for a, b in E:
                c, d = tl[k][a], tl[k][b]
                if not (inN[c] or inN[d]):
                    continue
                R = list(emap.get((min(c, d), max(c, d)), ()))
                regs = {al[r] for r in R if r != k}       # the bad tet itself may change region
                if len(R) > max_ring or len(regs) != 1:
                    continue
                ring = _ring(R, tl, c, d)
                if ring is None:
                    continue
                res = _edge_removal(x, c, d, ring)
                old = min(ql[r] for r in R)
                if res and res[0] > old and (best is None or res[0] > best[0]):
                    best = (res[0], c, d, R, res[1], regs.pop())
            if best is None:
                continue
            qn, c, d, R, tris, reg = best
            vol_old = sum(abs(_orient(*x[tl[r]])) for r in R)
            new = []
            for T in tris:
                for apex in (c, d):
                    v = [apex, *T]
                    if np.sign(_orient(*x[v])) != s0:
                        v[2], v[3] = v[3], v[2]
                    new.append(v)
            vol_new = sum(abs(_orient(*x[v])) for v in new)
            if abs(vol_new - vol_old) > 1e-9 * vol_old + 1e-15:
                continue
            if al[k] != reg:                               # absorbed by the region on the other side
                stats['relabelled'] += 1
                stats['relabelled_volume'] += abs(_orient(*x[tl[k]])) / 6
            for r in R:
                drop(r)
                alive[r] = False
            qq = tet_quality(x, np.array(new))
            for v, qv in zip(new, qq.tolist()):
                tl.append(v); al.append(reg)
                kk = len(tl) - 1
                ql[kk] = qv
                add(kk)
            alive = np.concatenate([alive, np.ones(len(new), bool)])
            done += 1
        stats['removed_edges'] += done
        t = np.array(tl, dtype=t.dtype)[alive]
        attr = np.array(al, dtype=attr.dtype)[alive]
        if not done:
            break
    q = tet_quality(x, t)
    stats.update(below_after=int((q < qthr).sum()), q_min_after=float(q.min()), tets_change=int(len(t) - len(q0)))
    return t, attr, stats


def collapse_slivers(x, t, attr, qthr=1e-3, lims=None, passes=5, tol=1e-6, interface_move=None, period=None,
                     ptol=3e-6):
    '''Remove the tets of quality below qthr by half-edge collapses, without leaving voids.

    For an edge uv of a bad tet, node u is merged into v (v does not move): the tets
    around uv vanish and the other tets of u's star take v in place of u, which fills
    the star again when every one keeps its orientation. A node u on box faces (lims)
    is merged only into a node v on the same faces (the edge uv lies in them), and
    together with its periodic images, each into the image of v by the same shift, so
    the opposite face triangulations stay copies of each other. A collapse (with all its
    images) is applied only if: the link condition holds (Lk u ∩ Lk v = Lk uv, with a
    virtual vertex over the box-face triangles: the mesh stays a manifold with the same
    boundary); every star keeps its volume and orientation; an interior node makes no
    new edge in a box face; an interface node u moves only along an interface edge into
    a node v of the same regions (attr), which moves the interface by a sliver-sized
    amount (moved_volume) -- or, with interface_move, any interface node u when the
    volume that changes region stays below interface_move times the mean tet volume (a
    short-edge cap between two interfaces); the image stars are disjoint; and the worst
    tet of the stars gets better. The best collapse of the six edges, both ways, is taken.
    period (box lengths, in place of lims): a periodic mesh of whole cells, whose boundary
    is not the box but the faces that appear once, paired by translation (ptol); a
    boundary node u merges only into a boundary node v with a translate v + s for every
    translate u + s of u, and the star volume check keeps the (jagged) boundary in place.
    x (n, 3); t (m, 4) 0-based. Returns (x, t, attr, stats) with the merged nodes
    removed and the rest renumbered.
    '''
    from itertools import combinations, product
    x = np.asarray(x, float)
    n = len(x)
    face = np.zeros((n, 6), bool)                          # on box face (d, side)
    shift = np.zeros(3)
    if lims is not None:
        lo, hi = np.array(lims, float).T
        for d in range(3):
            face[:, 2 * d] = np.abs(x[:, d] - lo[d]) < tol
            face[:, 2 * d + 1] = np.abs(x[:, d] - hi[d]) < tol
        shift = hi - lo
    on_box = face.any(1)
    if period is not None:                                 # boundary: the faces of one tet, paired by translation
        from .poly_tools import translation_classes
        tf = np.sort(np.asarray(t)[:, FACES], axis=2).reshape(-1, 3)
        uf, cnt = np.unique(tf, axis=0, return_counts=True)
        on_box[np.unique(uf[cnt == 1])] = True
    fidx = np.nonzero(on_box)[0]
    tree = None
    if len(fidx):
        from scipy.spatial import cKDTree
        tree = cKDTree(x[fidx])
    if period is not None:
        period = np.asarray(period, float)
        cls, _ = translation_classes(x[fidx], period, ptol)
        members = {}
        for i, c in zip(fidx.tolist(), cls.tolist()):
            members.setdefault(c, []).append(i)
        trans = {i: members[c] for i, c in zip(fidx.tolist(), cls.tolist())}
        if any(len(m) < 2 for m in members.values()):
            raise ValueError('collapse_slivers: a boundary node has no translate (mesh not periodic)')
    tl = [list(r) for r in np.asarray(t).tolist()]
    al = list(np.asarray(attr).tolist())
    q = tet_quality(x, np.asarray(t)).tolist()
    alive = [True] * len(tl)
    star = [set() for _ in range(n)]
    for k, v in enumerate(tl):
        for w in v:
            star[w].add(k)
    gone = np.zeros(n, bool)
    p_ = x[np.asarray(t)]
    vbar = float(np.abs(np.einsum('ij,ij->i', p_[:, 1] - p_[:, 0], np.cross(p_[:, 2] - p_[:, 0], p_[:, 3] - p_[:, 0]))).mean() / 6)
    W = -1                                                 # the virtual vertex over the box faces
    vol = lambda v: _orient(x[v[0]], x[v[1]], x[v[2]], x[v[3]]) / 6
    E = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    stats = {'qthr': qthr, 'below_before': int(sum(qq < qthr for qq in q)), 'q_min_before': float(min(q)),
             'collapsed': 0, 'collapsed_box': 0, 'moved_volume': 0.0, 'reject': dict.fromkeys(
                 ('box_faces', 'images', 'regions', 'link', 'orient', 'volume', 'box_edge', 'quality'), 0)}

    def bface(f):                                          # a triangle in a box face
        return (face[f[0]] & face[f[1]] & face[f[2]]).any()

    def bfaces(w):                                         # the box-face triangles at w
        if period is not None:                             # the faces at w that one tet of its star has
            from collections import Counter
            cnt = Counter(frozenset(tl[r][i] for i in f) for r in star[w] for f in FACES if w in [tl[r][i] for i in f])
            return {f for f, c in cnt.items() if c == 1}
        out = set()
        for r in star[w]:
            for f in FACES:
                tri = [tl[r][i] for i in f]
                if w in tri and bface(tri):
                    out.add(frozenset(tri))
        return out

    def link(w):
        tris = {frozenset(tl[r]) - {w} for r in star[w]}
        tris |= {f - {w} | {W} for f in bfaces(w)}
        edges = {frozenset(e) for f in tris for e in combinations(sorted(f), 2)}
        return tris, edges, set().union(*tris)

    def image(w, s):
        d, i = tree.query(x[w] + s)
        return int(fidx[i]) if d < 1e3 * tol else None

    def images(u, v):
        '''[(u_s, v_s)] for the periodic images of u (u itself first), or None'''
        if not on_box[u]:
            return [(u, v)]
        if period is not None:
            if not on_box[v]:
                return None
            out = [(u, v)]
            for us in trans[u]:
                if us != u:
                    d, i = tree.query(x[v] + x[us] - x[u])
                    if d > ptol:
                        return None
                    out.append((us, int(fidx[i])))
            return out
        if (face[u] & ~face[v]).any():                     # v must lie on every face of u
            return None
        axes = [(d, shift[d] if face[u, 2 * d] else -shift[d]) for d in range(3) if face[u, 2 * d:2 * d + 2].any()]
        out = []
        for take in product((0, 1), repeat=len(axes)):
            s = np.zeros(3)
            for (d, sd), k in zip(axes, take):
                s[d] = sd * k
            pair = (u, v) if not any(take) else (image(u, s), image(v, s))
            if None in pair:
                return None
            out.append(pair)
        return out

    def check(u, v):
        '''new tets for u -> v: (old_min_q, new [(r, tet)], new q, Suv, region volume change) or a reject key'''
        Suv = [r for r in star[u] if v in tl[r]]
        Tu, Eu, Vu = link(u)
        Tv, Ev, Vv = link(v)
        Le = {frozenset(tl[r]) - {u, v} for r in Suv}
        Le |= {f - {u, v} | {W} for f in bfaces(u) if v in f}
        Lv = set().union(*Le) if Le else set()
        if Tu & Tv or (Eu & Ev) != Le or (Vu & Vv) != Lv:
            return 'link'
        new = [(r, [v if w == u else w for w in tl[r]]) for r in star[u] if r not in Suv]
        V_old = {r: vol(tl[r]) for r in star[u]}
        V_new = [vol(vv) for _, vv in new]
        if not new or any(np.sign(vn) != np.sign(V_old[r]) or vn == 0 for (r, _), vn in zip(new, V_new)):
            return 'orient'
        a_old, a_new = sum(abs(w) for w in V_old.values()), sum(abs(w) for w in V_new)
        if abs(a_new - a_old) > 1e-9 * a_old:
            return 'volume'
        if not on_box[u]:
            nbr_v = set().union(*(set(tl[r]) for r in star[v]))
            for _, vv in new:
                for w in vv:
                    if w != v and w not in nbr_v and ((face[v] & face[w]).any() or
                                                      (period is not None and on_box[v] and on_box[w])):
                        return 'box_edge'
        by = {}
        for r, w in V_old.items():
            by[al[r]] = by.get(al[r], 0.0) - abs(w)
        for (r, _), w in zip(new, V_new):
            by[al[r]] = by.get(al[r], 0.0) + abs(w)
        qn = tet_quality(x, np.array([vv for _, vv in new])).tolist()
        return min(q[r] for r in star[u]), new, qn, Suv, 0.5 * sum(abs(w) for w in by.values())

    def trial(u, v):
        rej = stats['reject']
        pairs = images(u, v)
        if pairs is None:
            rej['box_faces'] += 1; return 'box_faces'
        tets = [star[a] | star[b] for a, b in pairs]
        if sum(map(len, tets)) != len(set().union(*tets)):
            rej['images'] += 1; return 'images'
        reg = lambda S: {al[r] for r in S}
        reg_u = set().union(*(reg(star[a]) for a, _ in pairs))
        if len(reg_u) > 1:
            reg_v = set().union(*(reg(star[b]) for _, b in pairs))
            reg_uv = set().union(*(reg(star[a] & star[b]) for a, b in pairs))
            if not (reg_u <= reg_v and reg_u <= reg_uv) and interface_move is None:
                rej['regions'] += 1; return 'regions'
        res = []
        for a, b in pairs:
            r = check(a, b)
            if isinstance(r, str):
                rej[r] += 1; return r
            res.append(r)
        if interface_move is not None and sum(r[4] for r in res) > interface_move * vbar:
            rej['regions'] += 1; return 'regions'
        q_old, q_new = min(r[0] for r in res), min(min(r[2]) for r in res)
        if q_new <= q_old:
            rej['quality'] += 1; return 'quality'
        return q_new, pairs, res

    why = {}
    for _ in range(passes):
        why = {}
        bad = sorted((k for k in range(len(tl)) if alive[k] and q[k] < qthr), key=lambda k: q[k])
        done = 0
        for k in bad:
            if not alive[k] or q[k] >= qthr:
                continue
            best, reasons = None, set()
            for a, b in E:
                for u, v in ((tl[k][a], tl[k][b]), (tl[k][b], tl[k][a])):
                    res = trial(u, v)
                    if isinstance(res, str):
                        reasons.add(res)
                    elif best is None or res[0] > best[0]:
                        best = res
            if best is None:
                why[k] = reasons
                continue
            _, pairs, res = best
            for (u, v), (_, new, qn, Suv, moved) in zip(pairs, res):
                for r in Suv:
                    alive[r] = False
                    for w in tl[r]:
                        star[w].discard(r)
                for (r, vv), qq in zip(new, qn):
                    star[u].discard(r)
                    star[v].add(r)
                    tl[r], q[r] = vv, qq
                gone[u] = True
                stats['moved_volume'] += moved
            done += 1
            stats['collapsed_box'] += int(on_box[pairs[0][0]])
        stats['collapsed'] += done
        if not done:
            break
    keep = np.array(alive)
    t = np.array(tl, dtype=np.int64)[keep]
    attr = np.array(al)[keep]
    remap = np.cumsum(~gone) - 1
    x, t = x[~gone], remap[t]
    qa = tet_quality(x, t)
    from collections import Counter
    stats['unresolved'] = dict(Counter('+'.join(sorted(r)) for k, r in why.items() if alive[k] and q[k] < qthr))
    stats.update(nodes_removed=int(gone.sum()), below_after=int((qa < qthr).sum()), q_min_after=float(qa.min()),
                 tets_change=int(len(t) - len(tl)))
    return x, t, attr, stats


def periodic_recut(x, t, bad, period, radius, tol=3e-6):
    '''Re-cut a periodic mesh of whole cells around the tets bad, so they lie inside it.

    The mesh is a fundamental domain of the torus R^3 / (period Z^3): boundary nodes that
    are translates (tol) are one torus node. In turn for each bad tet, every tet whose
    centroid lies within radius of the bad tet's (minimum image) is translated to that
    image, so the ball around it is one piece and the rings and stars of its edges and
    nodes, which the cut split, are whole (radius < period / 2). The result is again a
    fundamental domain, periodic by translation; nodes are made from the (torus node,
    lattice offset) pairs the tets use. x (n, 3); t (m, 4) 0-based. Returns (x, t, moved),
    moved: the number of tets translated.
    '''
    from .poly_tools import translation_classes
    x, t = np.asarray(x, float), np.asarray(t)
    L = np.asarray(period, float)
    if radius >= L.min() / 2:
        raise ValueError('periodic_recut: radius must be below half the period')
    cls, off = translation_classes(x, L, tol)
    c = x[t].mean(1)
    shift = np.zeros((len(t), 3), int)
    for k in bad:
        p = c[k] + shift[k] * L
        d = c + shift * L - p
        m = -np.round(d / L).astype(int)
        sel = np.linalg.norm(d + m * L, axis=1) < radius
        shift[sel] += m[sel]
    lat = off[t] + shift[:, None, :]                       # (m, 4, 3) lattice offset of each corner
    key = np.concatenate([cls[t][..., None], lat], 2).reshape(-1, 4)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    xn = x[uk[:, 0]] + uk[:, 1:] * L
    return xn, inv.reshape(-1, 4), int((shift != 0).any(1).sum())
