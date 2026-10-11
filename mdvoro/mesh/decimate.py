'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

================================================================================
Decimate
================================================================================

Nomenclature:
    c*          - current cell
    a*          - adjacent cell

    *_face_data - dictionary of face data with entries "vertices" for (x,y,z)
                       data of the vertices and "faces" which contains the vertices
                       of the closed hull (?) face
    *_face_num  - face number of cell

    *_cell_pts - array of [x,y,z] location data for the cell vertices

    *_cell_vts  - array of [x,y,z] location data for the face vertices



'''
import  numpy as np
from rich.progress import track
from mdvoro.utils.mdutils import mdutils
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def decimate(voro, lims, grain_boundary=False, surfpts=10, surf_cutoff=1.0, cell_cutoff=10.0, query=None, n_workers=None):
    """
    Decimate nonperiodic, internal cells and surfaces.

    Parameters
    ----------
    surf_cutoff : float, optional
        Distance threshold for surface decimation.
    n_workers : int, optional
        Number of worker processes for parallelization.
        If None, auto-detects and uses cpu_count() // 2.
        Set to 1 to disable parallelization.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() // 2)

    if query is not None:
        print('----------------\nPre Decimation\n----------------')
        print(voro[query])
    voro, vols = _decimate_faces(voro, grain_boundary, surf_cutoff, cell_cutoff, n_workers=n_workers)
    '''
    Decimate edges based on surface normals
    '''
    # voro = _decimate_edges(voro, lims)

    '''
    Decimate excess points
    '''
    voro = _decimate_points(voro, n_workers=n_workers)

    if query is not None:
        print('----------------\nPost Decimation\n----------------')
        print(voro[query])
    '''
    Merge surviving cells
    '''

    voro, phases, qn  = _merge_cells(voro, vols, query, n_workers=n_workers)

    # voro = _generate_convex_hull(voro, lims, surfpts)

    return voro, phases, qn

'''
--------------------------------------------------------------------------------
_decimate
--------------------------------------------------------------------------------
    Decimates internal cells

INPUTS:
    voro            -
    lim             -

OUTPUTS:
    voro            -


[ # list in same order as original points.
  {
    'volume' : 1.0,
    'vertices' : [[1.0, 2.0, 3.0], ...], # positions of vertices
    'adjacency' : [[1,3,4, ...], ...], # cell-vertices adjacent to i by index
    'faces' : [
      {
        'vertices' : [7,4,13, ...], # vertex ids in loop order
        'adjacent_cell' : 34 # *cell* id, negative if a wall
      }, ...]
    'original' : point[index] # the original instance from args
  },
  ...
]
'''
# decimate internal cells of phases
def _decimate_faces(voro, grain_boundary, surf_cutoff, cell_cutoff, n_workers=None):
    """
    Decimate faces with optional parallelization.

    Parameters
    ----------
    surf_cutoff : float
        Distance threshold for surface decimation.
    n_workers : int, optional
        Number of worker processes. Reserved for future parallelization.
    """
    seeds = {}
    # vol dict -- phase:volume
    vols = {}
    phases = []

    for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Decimating Faces ... ')):
        # seed data of this cell
        seed_data = c_cell_data['original']
        vol = c_cell_data['volume']
        rn = c_cell_num+1
        phase = seed_data[3]

        seeds[c_cell_num] = {'seed':rn,
                            'phase':phase,
                            'vol':vol,
                            'neighs':[]}

        if phase not in phases:
            phases.append(phase)
            vols[phase]=vol
        elif phase in phases:
            vols[phase]+= vol

        # face polyhedrals
        for c_face_data in c_cell_data['faces']:
            if len(c_face_data) > 0:
                '''
                a_cell_num : the adjacent cell number

                # cell subject
                c_local_vt : the local vertice indexes, []
                c_global_locs : array of local vertice locations, [[]]
                '''
                a_cell_num = c_face_data['adjacent_cell']

                seeds[c_cell_num]['neighs'].append(a_cell_num)

    # decimate internal cells
    decimate = set()
    for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Decimating Internal Cells ... ')):
        neighlist = seeds[c_cell_num]['neighs']
        allneigh = len(neighlist)
        n = 0
        for adj in neighlist:
            if adj >= 0 and adj < len(voro):
                close = mdutils.is_not_far( np.array(voro[c_cell_num]['original'][0:3]),
                                         np.array(voro[adj]['original'][0:3]),
                                         cell_cutoff)
                if seeds[c_cell_num]['phase']==seeds[adj]['phase'] and close:
                    n +=1

            elif not grain_boundary: # boundary condition
                n +=1

        rn = c_cell_num + 1
        if allneigh == n:
            actual = c_cell_num+1 # number \= index
            decimate.add(c_cell_num)

    survivors = set()
    for c_cell_num, c_cell_data in enumerate(voro):
        if c_cell_num not in decimate:
            survivors.add(c_cell_num)

    _voro = []

    for i, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Organizing Survivors ... ')):
        if i in survivors:
            new_faces = []
            new_verts = []
            for j, face in enumerate(voro[i]['faces']):
                if len(face)>0:
                    adj = face['adjacent_cell']
                    vert = face['vertices']

                    adjbool = False

                    if adj < 0:
                        if not grain_boundary: # boundary condition, comment out for grain boundary
                            new_faces.append({'adjacent_cell':adj,
                                              'vertices':vert})
                            adjbool = True
                    elif adj < len(voro):
                        close = mdutils.is_not_far( np.array(voro[i]['original'][0:3]),
                                                 np.array(voro[adj]['original'][0:3]),
                                                 cell_cutoff)

                        if adj in survivors and voro[i]['original'][3] != voro[adj]['original'][3]:
                            new_faces.append({'adjacent_cell':adj,
                                              'vertices':vert})
                            adjbool = True

                        elif adj in survivors and voro[i]['original'][3] == voro[adj]['original'][3] and not close:
                            new_faces.append({'adjacent_cell':adj,
                                              'vertices':vert})
                            adjbool = True

            voro[i]['faces']=new_faces

            _voro.append(voro[i])
        else:
            new_faces = []
            new_verts = []
            for j, face in enumerate(voro[i]['faces']):
                if len(face)>0:
                    adj = face['adjacent_cell']
                    vert = face['vertices']

                    if adj >= 0:

                        close = mdutils.is_not_far( np.array(voro[i]['original'][0:3]),
                                                 np.array(voro[adj]['original'][0:3]),
                                                 cell_cutoff)

                        if not close:
                            new_faces.append({'adjacent_cell':adj,
                                              'vertices':vert})
                    else:
                        # Surface points
                        new_faces.append({'adjacent_cell':adj,
                                          'vertices':vert})

            if len(new_faces) > 0:
                voro[i]['faces']=new_faces
                _voro.append(voro[i])
            else:
                _voro.append({'volume':-1,
                              'vertices':[],
                              'adjacency':[],
                              'faces':[],
                              'original':[]})

    return _voro, vols

# def _decimate_edges(voro):
#     _voro = []
#     for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Decimating Edges ... ')):
#         # face polyhedrals
#         merged_faces = []
#         if c_cell_data['volume'] != -1:
#             _normals = []
#             for c_face_data in c_cell_data['faces']:
#                 _pts = []
#                 for vt in c_face_data['vertices']:
#                     _pts.append(c_cell_data['vertices'][vt])
#
#                 _normals.append(_face_normal(_pts))
#
#             # assume it is cartesian
#             _retain = []
#             for i, _n in _normals:
#                 dx = 1.*_n[0] + 0.*_n[1] + 0.*_n[2]
#                 dy = 0.*_n[0] + 1.*_n[1] + 0.*_n[2]
#                 dz = 0.*_n[0] + 0.*_n[1] + 1.*_n[2]
#                 if 1E-4 > dx or 1E-4 > dy or 1E-4 > dz:
#                     continue
#                 elif:
#                     _retain.append(i)
#
#     return _voro

def _decimate_points(voro, n_workers=None):
    """
    Decimate excess points in Voronoi cells.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes. Reserved for future parallelization.
    """
    for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Decimating Points ... ')):
        # face polyhedrals
        if c_cell_data['volume'] != -1:
            used_vts = []
            for c_face_data in c_cell_data['faces']:
                for vt in c_face_data['vertices']:
                    used_vts.append(vt)

            # make unique
            used_vts = set(used_vts)

            old_new = {}
            old_keys = []
            redo_vts = []
            n = 0
            for vt in used_vts:
                old_keys.append(vt)
                old_new[vt] = n
                n+=1

                redo_vts.append(c_cell_data['vertices'][vt])

            voro[c_cell_num]['vertices'] = redo_vts

            for fc, c_face_data in enumerate(c_cell_data['faces']):
                redo_vertices = []
                for j in c_face_data['vertices']:
                    if j in old_keys:
                        redo_vertices.append(old_new[j])

                voro[c_cell_num]['faces'][fc]['vertices']=redo_vertices


    return voro

'''
--------------------------------------------------------------------------------
_merge_cells
--------------------------------------------------------------------------------
    Merge cells

INPUTS:
    voro            -

OUTPUTS:
    voro            -

'''


def _merge_cells(voro, vols, query=None, n_workers=None):
    """
    Merge cells by phase.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes. Reserved for future parallelization.
    """
    _voro = []
    qn = []

    # determine types and create cells based on type
    _types = set()
    for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Copying Cells ... ')):
        if c_cell_data['volume'] != -1 and c_cell_data['original'][3] not in _types:
            _types.add(c_cell_data['original'][3])

            _voro.append({'volume':vols[c_cell_data['original'][3]],
                          'vertices':[],
                          'adjacency':[],
                          'faces':[],
                          'original':[]})

    # sort existing cells into the new ones
    for ii, tt in track(enumerate(_types),  description= "{:<25}".format('Merging Cells ... ')):
        # face polyhedrals
        all_pts = []
        len_used_vts = 0
        for c_cell_num, c_cell_data in enumerate(voro):
            if c_cell_data['volume'] > 0 and c_cell_data['original'][3] == tt:
                # indexes
                for pt in c_cell_data['vertices']:
                    all_pts.append(pt)

                for c_face_data in c_cell_data['faces']:
                    current_vts = []
                    for vt in c_face_data['vertices']:
                        current_vts.append(vt+len_used_vts)

                    if query is not None:
                        if c_cell_num == query:
                            print('----------------\nAdj-Face\n----------------')
                            print(c_face_data['adjacent_cell'])
                            print('Ovito voro nodes')
                            print(np.array(current_vts)+1)
                            qn.append(len(_voro[ii]['faces'])+1)

                    _voro[ii]['faces'].append({'adjacent_cell':c_face_data['adjacent_cell'],
                                               'vertices':current_vts}) # doesnt really matter

                len_used_vts = len(all_pts)

        _x = 0
        _y = 0
        _z = 0
        n = 0
        for (x,y,z) in all_pts:
            n+=1
            _x += x
            _y += y
            _z += z
        x = _x/n
        y = _y/n
        z = _z/n

        _voro[ii]['vertices'] = all_pts
        _voro[ii]['original'] = [x,y,z,tt]

    return _voro, _types, qn

'''
--------------------------------------------------------------------------------
_face_normal
--------------------------------------------------------------------------------
    Determine face normal

INPUTS:
    voro            -

OUTPUTS:
    voro            -

'''
def _face_normal(face_pts):
    A = np.array(face_pts[0])
    B = np.array(face_pts[1])
    C = np.array(face_pts[-1])

    local_0 = A
    # local x axis
    local_x = B - A
    dec_x   = (local_x[0]**2 + local_x[1]**2 + local_x[2]**2 )**0.5

    # new vector to find normal
    AC = C - A
    _normal = np.cross(local_x, AC)
    dec_normal_x = (_normal[0]**2 + _normal[1]**2 + _normal[2]**2 )**0.5
    _normal = _normal / dec_normal_x

    return _normal
