'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

--------------------------------------------------------------------------------
MicroStructPy
--------------------------------------------------------------------------------
@article{HartKA2020,
    author       = "Hart, K.~A. and Rimoli, J.~J.",
    title        = "MicroStructPy: A statistical microstructure mesh generator in Python",
    journal    = "Computer Methods in Applied Mechanics and Engineering",
    volume       ="12",
    number       ="100595",
    year         = "2020",
    doi          = "https://doi.org/10.1016/j.softx.2020.100595"
}

--------------------------------------------------------------------------------
Periodic Face Method
--------------------------------------------------------------------------------
@article{FritzenF2009,
    author       = "Fritzen, F. and B{\"o}hlke, T. and Schnack, E.",
    title        = "Periodic three-dimensional mesh generation for crystalline aggregates based on Voronoi tessellations",
    journal    = "Computational Mechanics",
    volume       ="43",
    pages       ="701-713",
    year         = "2009",
    doi          = "http://dx.doi.org/10.1007/s00466-008-0339-2"
}

--------------------------------------------------------------------------------
TRIANGLE
--------------------------------------------------------------------------------
@ARTICLE{ShewchukJR1996,
    author       = "Shewchuk, J.~R.",
    title        = "Triangle: Engineering a 2\MakeUppercase{D} Quality Mesh Generator and Delaunay Triangulator",
    journal      = "In: Lin M. C. and Manocha, D. (eds) Applied Computational Geometry: Towards Geometric Engineering",
    volume       = "1148",
    pages        = "203-222",
    year         = "1996"
    }

================================================================================
TRIANGLE Face
================================================================================
    The code uses the method outlined by Fritzen (2009), which uses TRIANGLE
    (Shewchuk, 1996) to place additional nodes on the faces of Voronoi tessellations.
    This code does so for the periodic, periodic clipped, and non-periodic volumes.

    In periodic and periodic clipped, the periodic faces are identified as a (A,B) pair
    and the nodes from A are moved across the periodic face to B to mantain periodicity

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
from pymesh import triangle, resolve_self_intersection, collapse_short_edges
from rich.progress import track

from mdvoro.utils.mdutils import mdutils

def triangle_face(voro, lims, periodic=[False]*3, min_edge_dist = 1E-3, min_edge_cutoff = 1E-6, pt_pt_dist_tol = 1E-4, split = False, clip = False, max_area = 0.5):

    # if clip:
    #     voro = _clip_faces(voro, lims)

    if sum(periodic) > 0:
        '''
        Find periodic pairs

        '''
        periodic_faces, periodic_dist, pts_dict, new_periodic_faces, new_pf = _periodic_faces(voro, lims, periodic, split, max_area, pt_pt_dist_tol, min_edge_dist, min_edge_cutoff)

        '''
        If considered a periodic face, apply TRIANGLE face and replace already seen vertices

        '''
        seen_periodic_faces = set()
        for _pf, perface in enumerate(new_periodic_faces):
            if perface not in seen_periodic_faces:

                c_cell_num = perface[0]
                c_face_num = perface[1]

                if len(voro[c_cell_num]['faces'][c_face_num]) > 0:
                    a_cell_num = voro[c_cell_num]['faces'][c_face_num]['adjacent_cell']

                    # get adjacent cell on face
                    a_face_num = -1
                    for _a_face_num, a_face_data in enumerate(voro[a_cell_num]['faces']):
                        if len(a_face_data) > 0:
                            # get adjacent cell adjacent cell
                            _c_cell_num = a_face_data['adjacent_cell']
                            # when equal to main cell, get data and exit loop
                            if  _c_cell_num == c_cell_num:
                                a_face_num = _a_face_num

                    # if -1 still, chances are the Face is non periodic
                    if a_face_num > -1:
                        '''
                        Handle Current Cell

                        '''
                        c_new_face = new_periodic_faces[perface]['faces']
                        c_new_verts = new_periodic_faces[perface]['vertices']
                        c_existing_verts = voro[c_cell_num]['vertices']

                        c_final_face = _triangle_face_handle(voro, c_cell_num, c_new_face, c_new_verts, c_existing_verts, pt_pt_dist_tol)

                        voro[c_cell_num]['faces'][c_face_num] = []

                        '''
                        Handle Adjacent Cell

                        '''
                        a_new_face = c_new_face

                        a_new_verts = [0]*len(c_new_verts)
                        for _i, cnv in enumerate(c_new_verts):
                            cnv_x = cnv[0]+periodic_dist[perface][0]
                            cnv_y = cnv[1]+periodic_dist[perface][1]
                            cnv_z = cnv[2]+periodic_dist[perface][2]
                            a_new_verts[_i] = [cnv_x, cnv_y, cnv_z]

                        # need to prevent addition of redundent points
                        a_existing_verts = voro[a_cell_num]['vertices']

                        a_final_face = _triangle_face_handle(voro, a_cell_num, a_new_face, a_new_verts, a_existing_verts, pt_pt_dist_tol)

                        voro[a_cell_num]['faces'][a_face_num] = []

                        '''
                        Append to faces
                        '''

                        for _nf in c_final_face:
                            voro[c_cell_num]['faces'].append({'adjacent_cell':a_cell_num, 'vertices':_nf})

                        for _nf in a_final_face:
                            voro[a_cell_num]['faces'].append({'adjacent_cell':c_cell_num, 'vertices':_nf})

                        seen_periodic_faces.add((c_cell_num, c_face_num))
                        seen_periodic_faces.add((a_cell_num, a_face_num))
    else:
        '''
        Do the same for non periodic boundaries

        '''
        # new_faces = {}
        # new_fs = []
        seen_faces = set()
        for c_cell_num, c_cell_data in track(enumerate(voro),  description= "{:<25}".format('Triangling Faces ... ')):
            # face polyhedrals
            # generate bonds between each vertex for ovito
            for c_face_num, c_face_data in enumerate(c_cell_data['faces']):
                if (c_cell_num,c_face_num) not in seen_faces:
                    if len(c_face_data) > 0:
                        new_face, new_f, bb = _triangle_face(voro, c_cell_num, c_face_num, split, max_area, min_edge_dist, min_edge_cutoff)

                        if bb and len(voro[c_cell_num]['faces'][c_face_num]) > 0:
                            a_cell_num = voro[c_cell_num]['faces'][c_face_num]['adjacent_cell']

                            '''
                            Handle Current Cell

                            '''
                            c_new_face = new_face['faces']
                            c_new_verts = new_face['vertices']
                            c_existing_verts = voro[c_cell_num]['vertices']

                            c_final_face = _triangle_face_handle(voro, c_cell_num, c_new_face, c_new_verts, c_existing_verts, pt_pt_dist_tol)

                            if a_cell_num > -1 and a_cell_num < len(voro):
                                voro[c_cell_num]['faces'][c_face_num] = []

                                # get adjacent cell on face
                                a_face_num = -1
                                for _a_face_num, a_face_data in enumerate(voro[a_cell_num]['faces']):
                                    if len(a_face_data) > 0:
                                        # get adjacent cell adjacent cell
                                        _c_cell_num = a_face_data['adjacent_cell']
                                        # when equal to main cell, get data and exit loop
                                        if  _c_cell_num == c_cell_num:
                                            a_face_num = _a_face_num

                                '''
                                Handle Adjacent Cell

                                '''
                                a_new_face = c_new_face

                                a_new_verts = [0]*len(c_new_verts)
                                for _i, cnv in enumerate(c_new_verts):
                                    cnv_x = cnv[0]
                                    cnv_y = cnv[1]
                                    cnv_z = cnv[2]
                                    a_new_verts[_i] = [cnv_x, cnv_y, cnv_z]

                                # need to prevent addition of redundent points
                                a_existing_verts = voro[a_cell_num]['vertices']

                                a_final_face = _triangle_face_handle(voro, a_cell_num, a_new_face, a_new_verts, a_existing_verts, pt_pt_dist_tol)

                                voro[a_cell_num]['faces'][a_face_num] = []

                                '''
                                Append to faces
                                '''

                                for _nf in c_final_face:
                                    voro[c_cell_num]['faces'].append({'adjacent_cell':a_cell_num, 'vertices':_nf})

                                for _nf in a_final_face:
                                    voro[a_cell_num]['faces'].append({'adjacent_cell':c_cell_num, 'vertices':_nf})

                                seen_faces.add((c_cell_num, c_face_num))
                                seen_faces.add((a_cell_num, a_face_num))

                            else:
                                voro[c_cell_num]['faces'][c_face_num] = []

                                '''
                                Append to faces
                                '''

                                for _nf in c_final_face:
                                    voro[c_cell_num]['faces'].append({'adjacent_cell':a_cell_num, 'vertices':_nf})

                    # new_faces[(c_cell_num, c_face_num)] = new_face
                    # new_fs.append(new_f)

        # seen_faces = set()
        # for _pf, perface in track(enumerate(new_faces),  description= "{:<25}".format('Handling Triangles ... ')):
        #     if perface not in seen_faces:
        #
        #         c_cell_num = perface[0]
        #         c_face_num = perface[1]
        #
        #         if len(voro[c_cell_num]['faces'][c_face_num]) > 0:
        #             a_cell_num = voro[c_cell_num]['faces'][c_face_num]['adjacent_cell']
        #
        #             '''
        #             Handle Current Cell
        #
        #             '''
        #             c_new_face = new_faces[perface]['faces']
        #             c_new_verts = new_faces[perface]['vertices']
        #             c_existing_verts = voro[c_cell_num]['vertices']
        #
        #             c_final_face = _triangle_face_handle(voro, c_cell_num, c_new_face, c_new_verts, c_existing_verts, pt_pt_dist_tol)
        #
        #             voro[c_cell_num]['faces'][c_face_num] = []
        #
        #             '''
        #             Append to faces
        #             '''
        #
        #             for _nf in c_final_face:
        #                 voro[c_cell_num]['faces'].append({'adjacent_cell':a_cell_num, 'vertices':_nf})
        #
        #             seen_faces.add((c_cell_num, c_face_num))


    return voro


'''
--------------------------------------------------------------------------------
_clip_faces
--------------------------------------------------------------------------------
    Clips periodic cell face pairings

INPUTS:
    voro            -
    lim             -

OUTPUTS:
    voro            -

'''
def _clip_faces(voro, lims):
    # limits
    xlo = lims[0][0]
    xhi = lims[0][1]
    ylo = lims[1][0]
    yhi = lims[1][1]
    zlo = lims[2][0]
    zhi = lims[2][1]

    pts_global = []
    pts_seen = []
    pts_dict = {}

    periodic_pairs = []
    periodic_faces = []
    periodic_dist  = {}

    for c_cell_num, c_cell_data in enumerate(voro):
        # face polyhedrals
        # generate bonds between each vertex for ovito
        for c_face_num, c_face_data in enumerate(voro[c_cell_num]['faces']):
            if len(c_face_data) > 0:
                '''
                a_cell_num : the adjacent cell number

                c_cell_vts : the local vertice indexes, []
                c_global_locs : array of local vertice locations, [[]]
                '''

                new_pts = {}
                remove_pts = {}
                remove_face = {}
                new_vts = []
                new_cell_pts = {}

                for _i, vt in enumerate(voro[c_cell_num]['faces'][c_face_num]['vertices']):
                    # two points under investigation
                    pt2 = voro[c_cell_num]['vertices'][vt]
                    _vt = voro[c_cell_num]['faces'][c_face_num]['vertices'][_i-1]
                    pt1 = voro[c_cell_num]['vertices'][_vt]

                    # above lims, so use offset, clip, and move
                    offset, _f, _remove, new_pt, old_pt = mdutils.detect_plane_intersection(pt2, pt1, lims)

                    if _f:
                        new_pts[vt] = new_pt
                        new_vts.append(vt)
                        new_cell_pts[vt] = old_pt+offset
                        voro[c_cell_num]['vertices'][vt] = new_pt
                    if _remove:
                        remove_pts[c_cell_num] = vt
                        remove_face[c_cell_num] = c_face_num

                for _i, c_cell_num in enumerate(remove_face):
                    vt = remove_pts[c_cell_num]
                    # del voro[c_cell_num]['vertices'][vt]
                    del voro[c_cell_num]['faces'][c_face_num]['vertices']

                voro[c_cell_num]['faces'][c_face_num]['vertices'] = new_vts


    return voro

'''
--------------------------------------------------------------------------------
_periodic_faces
--------------------------------------------------------------------------------
    Finds periodic cell face pairings

INPUTS:
    voro            -

OUTPUTS:
    periodic_faces      -
    periodic_dist       -

'''
def _periodic_faces(voro, lims, periodic, split, max_area, pt_pt_dist_tol, min_edge_dist, min_edge_cutoff):
    new_periodic_faces = {}
    new_pfs = []

    pts_global = []
    pts_seen = []
    pts_dict = {}

    periodic_pairs = []
    periodic_pairs_set = set()
    periodic_faces = []
    periodic_dist  = {}

    for c_cell_num, c_cell_data in enumerate(voro):
        # face polyhedrals
        # generate bonds between each vertex for ovito
        for c_face_num, c_face_data in enumerate(c_cell_data['faces']):
            if len(c_face_data) > 0:
                '''
                a_cell_num : the adjacent cell number

                c_cell_vts : the local vertice indexes, []
                c_global_locs : array of local vertice locations, [[]]
                '''
                a_cell_num = c_face_data['adjacent_cell']
                c_cell_vts = c_face_data['vertices']
                c_cell_pts = c_cell_data['vertices']

                for c_vt in c_cell_vts:
                    '''
                    Check if point is alread connected
                    if already vertice is duplicate, get the original number
                    else add as new point to the lists

                    key is (current cell number, vertice)

                    '''
                    key = (c_cell_num, c_vt)

                    if key in pts_dict:
                        global_vt_num = pts_dict[key]

                    else:
                        # use length as counter for new pts
                        global_vt_num = len(pts_global)

                        pts_global.append(c_cell_pts[c_vt])
                        pts_seen.append({c_cell_num:key})
                        pts_dict[key]=global_vt_num

                    seen_list = set(pts_seen[global_vt_num])

                    if a_cell_num not in seen_list and len(voro)>1:
                        '''
                        a_cell_data : the adjacent cell data
                        a_cell_pts : the local point locations, [(x,y,z),...]
                        '''
                        a_cell_data = voro[a_cell_num]
                        a_cell_pts = np.array(a_cell_data['vertices'])

                        '''
                        Adjust relative location, determine distances, and find minimum

                        Minimum will be the overlap location between the same vertice
                        of both the cell and its neighbor
                        '''
                        rel_pos = a_cell_pts - pts_global[global_vt_num]
                        sq_dist = np.sum(rel_pos * rel_pos, axis=-1)
                        a_min_vt = np.argmin(sq_dist)

                        '''
                        Get minimum distance and determine if it is across the periodic
                            boundary. If it is not, then ok to be adjacent
                        '''
                        a_min_pt = a_cell_pts[a_min_vt]

                        adj_key = (a_cell_num, a_min_vt)
                        if mdutils.is_not_far(a_min_pt, pts_global[global_vt_num], pt_pt_dist_tol):
                            pts_dict[adj_key] = global_vt_num
                            pts_seen[global_vt_num][a_cell_num] = a_min_vt

                        else:
                            dist, offset, _peri =  mdutils.detect_zone_dist(a_min_pt, pts_global[global_vt_num], lims, periodic)

                            _mismatch = False
                            for _p, _bool in enumerate(_peri):
                                if periodic[_p] != _bool:
                                    _mismatch = True
                                    break

                            if not _mismatch :
                                cnp = []
                                if c_cell_num < a_cell_num:
                                    cnp = (c_cell_num, a_cell_num)
                                else:
                                    cnp = (a_cell_num, c_cell_num)

                                if cnp not in periodic_pairs_set:
                                    periodic_pairs_set.add(cnp)

                                    periodic_pairs.append([c_cell_num, a_cell_num])

                                    periodic_faces.append((c_cell_num, c_face_num))
                                    periodic_dist[(c_cell_num, c_face_num)] = offset

                                    new_periodic_face, new_pf, bb = _triangle_face(voro, c_cell_num, c_face_num, split, max_area, min_edge_dist, min_edge_cutoff)
                                    '''
                                    To do, only take lower face count
                                    '''
                                    # a_new_periodic_face, a_new_pf = _triangle_face(voro, a_cell_num, a_face_num, split)
                                    #
                                    # if len(c_new_periodic_face) > len(a_new_periodic_face):
                                    #     new_periodic_face = a_new_periodic_face
                                    #     new_pf = a_new_pf
                                    # else:
                                    #     new_periodic_face = c_new_periodic_face
                                    #     new_pf = c_new_pf
                                    if bb:
                                        new_periodic_faces[(c_cell_num, c_face_num)] = new_periodic_face
                                        new_pfs.append(new_pf)


    return periodic_faces, periodic_dist, pts_dict, new_periodic_faces, new_pfs

'''
--------------------------------------------------------------------------------
_split_edges
--------------------------------------------------------------------------------
    Split edges

INPUTS:
    voro            -

OUTPUTS:
    voro            -

'''
def _split_edges(proj_tri_pts, min_edge_dist, min_edge_cutoff):

    len_ptp = len(proj_tri_pts)

    for _i in range(len_ptp):
        pt2 = np.array(proj_tri_pts[_i])
        pt1 = np.array(proj_tri_pts[_i-1])

        if not mdutils.is_not_far(pt1, pt2, min_edge_dist):

            mid_point_pt = (pt2 + pt1)/2.0
            proj_tri_pts.append(mid_point_pt)

    return proj_tri_pts

'''
--------------------------------------------------------------------------------
_triangle_face
--------------------------------------------------------------------------------
    TRIANGLEs (Shewchuk, 1996) face using pymesh

INPUTS:
    voro            -

OUTPUTS:
    final_face      -

'''

def _triangle_face(voro, cell_num, face_num, split, max_area, min_edge_dist, min_edge_cutoff):
    # https://stackoverflow.com/questions/26369618/getting-local-2d-coordinates-of-vertices-of-a-planar-polygon-in-3d-space
    #begin triangling
    cell_data = voro[cell_num]
    face_data = cell_data['faces'][face_num]

    tri_pts = []
    for _i in face_data['vertices']:
        tri_pts.append(cell_data['vertices'][_i])

    A = np.array(tri_pts[0])
    B = np.array(tri_pts[1])
    C = np.array(tri_pts[-1])

    local_0 = A
    # local x axis
    local_x = B - A
    dec_x   = (local_x[0]**2 + local_x[1]**2 + local_x[2]**2 )**0.5


    # new vector to find normal
    AC = C - A
    _normal = np.cross(local_x, AC)
    dec_normal_x = (_normal[0]**2 + _normal[1]**2 + _normal[2]**2 )**0.5


    # new y axis
    local_y = np.cross( _normal, local_x)
    dec_local_y = (local_y[0]**2 + local_y[1]**2 + local_y[2]**2 )**0.5

    if dec_x > 0 and dec_normal_x > 0 and dec_local_y > 0:
        local_x = local_x / dec_x
        _normal = _normal / dec_normal_x
        local_y = local_y / dec_local_y

        # Project points to plane
        proj_tri_pts = []
        # _local_vts = []
        for _i, pt in enumerate(tri_pts):
            pt = np.array(pt)
            proj = [np.dot(pt - local_0, local_x),
                    np.dot(pt - local_0, local_y)]
            proj_tri_pts.append(proj)

        # if split:
        #     proj_tri_pts = _split_edges(proj_tri_pts, min_edge_dist, min_edge_cutoff)

        # triangulating faces
        tri = triangle()
        tri.points = proj_tri_pts
        tri.verbosity = 0
        tri.max_area = max_area
        # tri.algorithm = 'SWEEPLINE'
        # main issue
        tri.split_boundary = split # True causes self intersection it seems
        tri.keep_convex_hull = True
        tri.conforming_delaunay = True
        tri.run()
        mesh = tri.mesh

        new_pts = []
        for pt in mesh.vertices:
            pt = np.array(pt)
            p = local_0 + pt[0]*local_x + pt[1]*local_y
            new_pts.append(p)

        new_periodic_face = {'vertices': new_pts,
                             'faces': mesh.faces}

        new_pf = (cell_num, face_num)
        return new_periodic_face, new_pf, True
    else:
        return [], [], False

'''
--------------------------------------------------------------------------------
_triangle_face_handle
--------------------------------------------------------------------------------
    Handles TRIANGLEd face using pymesh's triangle, a wrapping of
    (Shewchuk, 1996).

INPUTS:
    voro            -
    new_face        -
    new_verts       -
    existing_verts  -

OUTPUTS:
    final_face      -

'''
def _triangle_face_handle(voro, cell_num, new_face, new_verts, existing_verts, pt_pt_dist_tol):
    temp_face = []
    for i in new_face:
        temp_face.append([-1, -1, -1])

    new_face = list(map(tuple, new_face))
    new_verts = list(map(tuple, new_verts))
    existing_verts = list(map(tuple, existing_verts))

    for _nv, pt in enumerate(new_verts):
        min_index, min_location, dist = mdutils.find_closest(pt, list(existing_verts))

        if pt_pt_dist_tol > dist:
            # if matching location, find and replace index in new faces
            # for all triangles in the new_face
            for _i, _verts in enumerate(new_face):
                # for all vertices in those triangles
                for _j, v in enumerate(_verts):
                    # if the index that returns the min is the current,
                    # replace with the global number
                    if _nv == v:
                        temp_face[_i][_j] = min_index
        else:
            ln = len(voro[cell_num]['vertices'])

            voro[cell_num]['vertices'].append(pt)

            for _i, _verts in enumerate(new_face):
                # for all vertices in those triangles
                for _j, v in enumerate(_verts):
                    # if the index that returns the min is the current,
                    # replace with the new index number
                    if _nv == v:
                        temp_face[_i][_j] = ln

    final_face = []
    for _nf in temp_face:
        _i, _j, _k = _nf
        if _i == _j or _i == _k or _j == _k:
            continue
        else:
            final_face.append(_nf)

    return final_face
