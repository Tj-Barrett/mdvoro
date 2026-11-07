'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

--------------------------------------------------------------------------------
TetGen
--------------------------------------------------------------------------------


================================================================================
TetGen
================================================================================

INPUTS:

OUTPUTS:


'''
import os, subprocess
from mdvoro.utils.read_config import read_config


def tetgen_timeout(filename):
    '''Seconds to allow TetGen on a .poly before treating it as hung: 30 s per MB
    of input, at least 10 min (a 28 MB PLC meshes in under a minute). The
    environment variable MDVORO_TETGEN_TIMEOUT overrides it; 0 disables it.'''
    env = os.environ.get('MDVORO_TETGEN_TIMEOUT')
    if env is not None:
        return float(env) or None
    return max(600.0, 30.0 * os.path.getsize(filename) / 2**20)


def _run(argv, filename, **kw):
    '''subprocess.run with the TetGen timeout; a hang raises RuntimeError'''
    timeout = tetgen_timeout(filename)
    try:
        return subprocess.run(argv, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'{" ".join(argv)} did not finish in {timeout:.0f} s') from None


def run_tetgen(filename, tetcmds):
    args = read_config()
    cmd = args.alias_tetgen
    p = _run([cmd,tetcmds,filename], filename)


def plc_self_intersects(filename):
    '''Coordinates (n, 3) of the vertices of the facets TetGen finds
    self-intersecting in a .poly (tetgen -d, no meshing); empty if it is clean'''
    import glob
    import numpy as np
    cmd = read_config().alias_tetgen or 'tetgen'
    r = _run([cmd, '-dQ', filename], filename, capture_output=True, text=True)
    stem = filename.rsplit('.', 1)[0]
    bad = np.zeros((0, 3))
    if r.returncode != 0:
        if os.path.exists(stem + '_skipped.node') and os.path.exists(stem + '_skipped.face'):
            read = lambda f: [l.split() for l in open(f) if l.split() and not l.lstrip().startswith('#')]
            nodes = {int(p[0]): [float(v) for v in p[1:4]] for p in read(stem + '_skipped.node')[1:]}
            used = sorted({int(v) for p in read(stem + '_skipped.face')[1:] for v in p[1:4]})
            bad = np.array([nodes[i] for i in used if i in nodes]).reshape(-1, 3)
        if len(bad) == 0:
            bad = np.full((1, 3), np.nan)        # failed without telling where
    for f in glob.glob(stem + '.1.*') + glob.glob(stem + '_skipped.*'):
        os.remove(f)
    return bad
