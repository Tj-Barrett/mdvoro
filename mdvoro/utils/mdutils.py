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
    title        = "Triangle: Engineering a 2{D} Quality Mesh Generator and Delaunay Triangulator",
    journal      = "In: Lin M. C. and Manocha, D. (eds) Applied Computational Geometry: Towards Geometric Engineering",
    volume       = "1148",
    pages        = "203-222",
    year         = "1996"
    }

================================================================================
Utils
================================================================================
'''
import numpy as np

class mdutils:
    '''
    --------------------------------------------------------------------------------
    detect_plane_intersection
    --------------------------------------------------------------------------------
        Determines the periodic offest of two cell faces when trying to map across
        periodic bounds. Distance is minimized between the current cell and adjacent
        cell face points, and the offset that determines that is used

    INPUTS:
        lims    - [[xlo, xhi] [ylo, yhi] [zlo, zhi]]
                  limits of the periodic cube
        a       - face vertice (x,y,z) locations

    OUTPUTS:
        offset  - offset used to find minimum distance
    '''
    def detect_plane_intersection(a, b, lims):
        # limits
        xlo = lims[0][0]
        xhi = lims[0][1]
        ylo = lims[1][0]
        yhi = lims[1][1]
        zlo = lims[2][0]
        zhi = lims[2][1]

        xlen = xhi - xlo
        ylen = yhi - ylo
        zlen = zhi - zlo

        pt2 = np.array(a)
        pt1 = np.array(b)

        offset = [0,0,0]
        found = False
        new_pt = pt2
        old_pt = pt2
        dir_type = []

        _clipping = True
        _remove   = False

        if pt2[0] > xhi and pt1[0] > xhi:
            _clipping = False
            _remove   = True
        if pt2[0] < xlo and pt1[0] < xlo:
            _clipping = False
            _remove   = True
        if pt2[1] > yhi and pt1[1] > yhi:
            _clipping = False
            _remove   = True
        if pt2[1] < ylo and pt1[1] < ylo:
            _clipping = False
            _remove   = True
        if pt2[2] > zhi and pt1[2] > zhi:
            _clipping = False
            _remove   = True
        if pt2[2] < zlo and pt1[2] < zlo:
            _clipping = False
            _remove   = True

        if _clipping:
            found = True
            x_lim_seen = y_lim_seen = z_lim_seen = False

            for _ in range(3):
                # +x +y +z / +x +y -z / +x -y +z / +x -y -z / -x +y +z / -x +y -z / -x -y +z / -x -y -z
                # if pt2[0] > xhi and pt2[1] > yhi and pt2[2] > zhi:
                #     offset = [-xlen,-ylen,-zlen]
                #     dir_type = ['x', 'y', 'z']
                #
                # elif pt2[0] > xhi and pt2[1] > yhi and pt2[2] < zlo:
                #     offset = [-xlen,-ylen,zlen]
                #     dir_type = ['x', 'y', '-z']
                #
                # elif pt2[0] > xhi and pt2[1] < ylo and pt2[2] > zhi:
                #     offset = [-xlen,ylen,-zlen]
                #     dir_type = ['x', '-y', 'z']
                #
                # elif pt2[0] > xhi and pt2[1] < ylo and pt2[2] < zlo:
                #     offset = [-xlen,ylen,zlen]
                #     dir_type = ['x', '-y', '-z']
                #
                # elif pt2[0] < xlo and pt2[1] > yhi and pt2[2] > zhi:
                #     offset = [xlen,-ylen,-zlen]
                #     dir_type = ['-x', 'y', 'z']
                #
                # elif pt2[0] < xlo and pt2[1] > yhi and pt2[2] < zlo:
                #     offset = [xlen,-ylen,zlen]
                #     dir_type = ['-x', 'y', '-z']
                #
                # elif pt2[0] < xlo and pt2[1] < ylo and pt2[2] > zhi:
                #     offset = [xlen,ylen,-zlen]
                #     dir_type = ['-x', '-y', 'z']
                #
                # elif pt2[0] < xlo and pt2[1] < ylo and pt2[2] < zlo:
                #     offset = [xlen,ylen,zlen]
                #     dir_type = ['-x', '-y', '-z']
                #
                #     # +x +z / +x -z / -x +z / -x -z
                # elif pt2[0] > xhi and pt2[2] > zhi:
                #     offset = [-xlen,0,-zlen]
                #     dir_type = ['x', 'z']
                #
                # elif pt2[0] > xhi and pt2[2] < zlo:
                #     offset = [-xlen,0,zlen]
                #     dir_type = ['x', '-z']
                #
                # elif pt2[0] < xlo and pt2[2] > zhi:
                #     offset = [xlen,0,-zlen]
                #     dir_type = ['-x', 'z']
                #
                # elif pt2[0] < xlo and pt2[2] < zhi:
                #     offset = [xlen,0,zlen]
                #     dir_type = ['-x', '-z']
                #
                #     # +x +y / +x -y / -x +y / -x -y
                # elif pt2[0] > xhi and pt2[1] > yhi:
                #     offset = [-xlen,-ylen,0]
                #     dir_type = ['x', 'y']
                #
                # elif pt2[0] > xhi and pt2[1] < ylo:
                #     offset = [-xlen,ylen,0]
                #     dir_type = ['x', '-y']
                #
                # elif pt2[0] < xlo and pt2[1] > yhi:
                #     offset = [xlen,-ylen,0]
                #     dir_type = ['-x', 'y']
                #
                # elif pt2[0] < xlo and pt2[1] < ylo:
                #     offset = [xlen,ylen,0]
                #     dir_type = ['-x', '-y']
                #
                #     # +y +z / +y -z / -y +z / -y -z
                # elif pt2[1] > yhi and pt2[2] > zhi:
                #     offset = [0,-ylen,-zlen]
                #     dir_type = ['y', 'z']
                #
                # elif pt2[1] > yhi and pt2[2] < zlo:
                #     offset = [0,-ylen,zlen]
                #     dir_type = ['y', '-z']
                #
                # elif pt2[1] < ylo and pt2[2] > zhi:
                #     offset = [0,ylen,-zlen]
                #     dir_type = ['-y', 'z']
                #
                # elif pt2[1] < ylo and pt2[2] < zlo:
                #     offset = [0,ylen,zlen]
                #     dir_type = ['-y', '-z']

                    # +x / -x
                if pt2[0] > xhi:
                    offset = [-xlen,0,0]
                    dir_type = ['x']

                elif pt2[0] < xlo:
                    offset = [xlen,0,0]
                    dir_type = ['-x']

                    # +y / -y
                elif pt2[1] > yhi:
                    offset = [0,-ylen,0]
                    dir_type = ['y']

                elif pt2[1] < ylo:
                    offset = [0,ylen,0]
                    dir_type = ['-y']

                    # +z / -z
                elif pt2[2] > zhi:
                    offset = [0,0,-zlen]
                    dir_type = ['z']


                elif pt2[2] < zlo:
                    offset = [0,0,zlen]
                    dir_type = ['-z']


                _v = pt2 - pt1

                if found:
                    for _i in dir_type:
                        if _i == 'x' and not x_lim_seen:
                            new_pt[0] = xhi
                            if not y_lim_seen:
                                new_pt[1] = _v[1]/_v[0]*(xhi-pt1[0])+pt1[1]
                            if not z_lim_seen:
                                new_pt[2] = _v[2]/_v[0]*(xhi-pt1[0])+pt1[2]
                            pt2 = new_pt

                            x_lim_seen = True

                        elif _i == '-x'and not x_lim_seen:
                            new_pt[0] = xlo
                            if not y_lim_seen:
                                new_pt[1] = _v[1]/_v[0]*(xlo-pt1[0])+pt1[1]
                            if not z_lim_seen:
                                new_pt[2] = _v[2]/_v[0]*(xlo-pt1[0])+pt1[2]
                            pt2 = new_pt

                            x_lim_seen = True

                        elif _i == 'y' and not y_lim_seen:
                            if not x_lim_seen:
                                new_pt[0] = _v[0]/_v[1]*(yhi-pt1[1])+pt1[0]
                            new_pt[1] = yhi
                            if not z_lim_seen:
                                new_pt[2] = _v[2]/_v[1]*(yhi-pt1[1])+pt1[2]
                            pt2 = new_pt

                            y_lim_seen = True

                        elif _i == '-y' and not y_lim_seen:
                            if not x_lim_seen:
                                new_pt[0] = _v[0]/_v[1]*(ylo-pt1[1])+pt1[0]
                            new_pt[1] = ylo
                            if not z_lim_seen:
                                new_pt[2] = _v[2]/_v[1]*(ylo-pt1[1])+pt1[2]
                            pt2 = new_pt

                            y_lim_seen = True

                        elif _i == 'z' and not z_lim_seen:
                            if not x_lim_seen:
                                new_pt[0] = _v[0]/_v[2]*(zhi-pt1[2])+pt1[0]
                            if not y_lim_seen:
                                new_pt[1] = _v[1]/_v[2]*(zhi-pt1[2])+pt1[1]
                            new_pt[2] = zhi
                            pt2 = new_pt

                            z_lim_seen = True

                        elif _i == '-z' and not z_lim_seen:
                            if not x_lim_seen:
                                new_pt[0] = _v[0]/_v[2]*(zlo-pt1[2])+pt1[0]
                            if not y_lim_seen:
                                new_pt[1] = _v[1]/_v[2]*(zlo-pt1[2])+pt1[1]
                            new_pt[2] = zlo
                            pt2 = new_pt

                            z_lim_seen = True

        return offset, found, _remove, new_pt, old_pt
    '''
    --------------------------------------------------------------------------------
    find_closest
    --------------------------------------------------------------------------------
        Finds closest point in a point array

    INPUTS:
        pt          -
        pt_array    -

    OUTPUTS:
        min_index       -
        min_location    -
        dist            -
    '''
    def find_closest(pt, pt_array):
        '''
        Adjust relative location, determine distances, and find minimum

        Minimum will be the overlap location between the same vertice
        of both the cell and its neighbor
        '''
        pt = np.array(pt)
        pt_array = np.array(pt_array, dtype='object')

        rel_pos = pt_array - pt
        sq_dist = (np.sum(rel_pos * rel_pos, axis=-1))**0.5
        min_index = np.argmin(sq_dist)
        dist = sq_dist[min_index]

        '''
        Get minimum distance and determine if it is across the periodic
            boundary. If it is not, then ok to be adjacent
        '''
        min_location = pt_array[min_index]

        return min_index, min_location, dist

    def find_mult_closest(pt, pt_array, cut):
        '''
        Adjust relative location, determine distances, and find minimum

        Minimum will be the overlap location between the same vertice
        of both the cell and its neighbor
        '''
        pt = np.array(pt)
        pt_array = np.array(pt_array, dtype='object')

        rel_pos = pt_array - pt
        sq_dist = (np.sum(rel_pos * rel_pos, axis=-1))**0.5
        _index = sq_dist < cut

        min_index = []
        for i, _ in enumerate(_index):
            if _:
                min_index.append(i)

        return min_index

    '''
    --------------------------------------------------------------------------------
    determine_offset
    --------------------------------------------------------------------------------
        Determines the periodic offest of two cell faces when trying to map across
        periodic bounds. Distance is minimized between the current cell and adjacent
        cell face points, and the offset that determines that is used

    INPUTS:
        lims    - [[xlo, xhi] [ylo, yhi] [zlo, zhi]]
                  limits of the periodic cube
        a , b   - face vertice (x,y,z) locations

    OUTPUTS:
        min_dist    - minimum distance used to find correct offset
        min_offset  - offset used to find minimum distance
    '''
    def detect_zone_dist(a, b, lims, periodic):
        # limits
        xlo = lims[0][0]
        xhi = lims[0][1]
        ylo = lims[1][0]
        yhi = lims[1][1]
        zlo = lims[2][0]
        zhi = lims[2][1]

        xlen = xhi - xlo
        ylen = yhi - ylo
        zlen = zhi - zlo

        min_dist = 10.
        min_offset = []
        offset = [0]*3
        mistmatch = [False]*3

        for i in range(26):
            # +x +y +z / +x +y -z / +x -y +z / +x -y -z / -x +y +z / -x +y -z / -x -y +z / -x -y -z
            if i == 0 and periodic[0] and periodic[1] and periodic[2]:
                offset = [-xlen,-ylen,-zlen]
                mistmatch = [True, True, True]
            elif i == 1 and periodic[0] and periodic[1] and periodic[2]:
                offset = [-xlen,-ylen,zlen]
                mistmatch = [True, True, True]
            elif i == 2 and periodic[0] and periodic[1] and periodic[2]:
                offset = [-xlen,ylen,-zlen]
                mistmatch = [True, True, True]
            elif i == 3 and periodic[0] and periodic[1] and periodic[2]:
                offset = [-xlen,ylen,zlen]
                mistmatch = [True, True, True]
            elif i == 4 and periodic[0] and periodic[1] and periodic[2]:
                offset = [xlen,-ylen,-zlen]
                mistmatch = [True, True, True]
            elif i == 5 and periodic[0] and periodic[1] and periodic[2]:
                offset = [xlen,-ylen,zlen]
                mistmatch = [True, True, True]
            elif i == 6 and periodic[0] and periodic[1] and periodic[2]:
                offset = [xlen,ylen,-zlen]
                mistmatch = [True, True, True]
            elif i == 7 and periodic[0] and periodic[1] and periodic[2]:
                offset = [xlen,ylen,zlen]
                mistmatch = [True, True, True]

                # +x +z / +x -z / -x +z / -x -z
            elif i == 8 and periodic[0] and periodic[2]:
                offset = [-xlen,0,-zlen]
                mistmatch = [True, False, True]
            elif i == 9 and periodic[0] and periodic[2]:
                offset = [-xlen,0,zlen]
                mistmatch = [True, False, True]
            elif i == 10 and periodic[0] and periodic[2]:
                offset = [xlen,0,-zlen]
                mistmatch = [True, False, True]
            elif i == 11 and periodic[0] and periodic[2]:
                offset = [xlen,0,zlen]
                mistmatch = [True, False, True]

                # +x +y / +x -y / -x +y / -x -y
            elif i == 12 and periodic[0] and periodic[1]:
                offset = [-xlen,-ylen,0]
                mistmatch = [True, True, False]
            elif i == 13 and periodic[0] and periodic[1]:
                offset = [-xlen,ylen,0]
                mistmatch = [True, True, False]
            elif i == 14 and periodic[0] and periodic[1]:
                offset = [xlen,-ylen,0]
                mistmatch = [True, True, False]
            elif i == 15 and periodic[0] and periodic[1]:
                offset = [xlen,ylen,0]
                mistmatch = [True, True, False]

                # +y +z / +y -z / -y +z / -y -z
            elif i == 16 and periodic[1] and periodic[2]:
                offset = [0,-ylen,-zlen]
                mistmatch = [False, True, True]
            elif i == 17 and periodic[1] and periodic[2]:
                offset = [0,-ylen,zlen]
                mistmatch = [False, True, True]
            elif i == 18 and periodic[1] and periodic[2]:
                offset = [0,ylen,-zlen]
                mistmatch = [False, True, True]
            elif i == 19 and periodic[1] and periodic[2]:
                offset = [0,ylen,zlen]
                mistmatch = [False, True, True]

                # +x / -x
            elif i == 20 and periodic[0]:
                offset = [-xlen,0,0]
                mistmatch = [True, False, False]
            elif i == 21 and periodic[0]:
                offset = [xlen,0,0]
                mistmatch = [True, False, False]

                # +y / -y
            elif i == 22 and periodic[1]:
                offset = [0,-ylen,0]
                mistmatch = [False, True, False]
            elif i == 23 and periodic[1]:
                offset = [0,ylen,0]
                mistmatch = [False, True, False]

                # +z / -z
            elif i == 24 and periodic[2]:
                offset = [0,0,-zlen]
                mistmatch = [False, False, True]
            elif i == 25 and periodic[2]:
                offset = [0,0,zlen]
                mistmatch = [False, False, True]

            rel_pos = a - b - offset
            dist = (np.sum(rel_pos * rel_pos, axis=-1) )**0.5
            if dist < min_dist:
                min_dist = dist
                min_offset = offset

        return min_dist, min_offset, mistmatch

    '''
    --------------------------------------------------------------------------------
    is_colinear
    --------------------------------------------------------------------------------
        Determines if 3 points are colinear by checking to see if the area between
        those points is below the tolerance.
        A perfectly colinear line will have an area of zero.

    INPUTS:
        pt1, pt2, pt3 - all (x,y,z) location data
        tol           - area tolerance to determine colinearity

    OUTPUTS:
        boolean TRUE / FALSE
    '''
    def is_colinear(pt1, pt2, pt3, tol=1E-6):
        # if 3 points are colinear the triangle area is zero
        AB = np.array(pt2)-np.array(pt1)
        AC = np.array(pt3)-np.array(pt1)

        S = 0.5 * ( (AB[1]*AC[2] - AB[2]*AC[1])**2 + \
                    (AB[2]*AC[0] - AB[0]*AC[2])**2 + \
                    (AB[0]*AC[1] - AB[1]*AC[0])**2 )**0.5

        if S < tol:
            return True
        else:
            return False

    '''
    --------------------------------------------------------------------------------
    is_not_far
    --------------------------------------------------------------------------------


    INPUTS:

    OUTPUTS:

    '''
    def is_not_far(a, b, cut = 1E-6):
        '''
        Check if matching point is far away, ie across periodic bound
        '''
        rel_pos = a - b
        dist = (np.sum(rel_pos * rel_pos, axis=-1) )**0.5

        # print(dist)

        if dist > cut:
            return False
        else:
            return True

    '''
    --------------------------------------------------------------------------------
    is_on_boundary
    --------------------------------------------------------------------------------


    INPUTS:

    OUTPUTS:

    '''
    def is_on_boundary(a, lim_array, cut = 1E-3):
        '''
        Check if point is on a boundary
        '''
        a = np.array(a)

        for pln in lim_array:
            dist = np.abs(a - np.array(pln))

            if dist[0] < cut or dist[1] < cut or dist[2] < cut:
                return True

        return False

    '''
    --------------------------------------------------------------------------------
    line_seg_interstect
    --------------------------------------------------------------------------------


    INPUTS:

    OUTPUTS:

    '''

    def line_seg_interstect(A,B,C,D):

        # for some reason not seen outside of function?
        def lshelper(A,B,C):
            Ax = A[0]
            Ay = A[1]
            Bx = B[0]
            By = B[1]
            Cx = C[0]
            Cy = C[1]
            return (Cy-Ay) * (Bx-Ax) > (By-Ay) * (Cx-Ax)

        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        D = np.array(D)

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
            proj_pts = []

            for _i, pt in enumerate([A, B, C, D]):
                pt = np.array(pt)
                proj = [np.dot(pt - local_0, local_x),
                        np.dot(pt - local_0, local_y)]
                proj_pts.append(proj)

            # check intersection https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
            Ax = proj_pts[0][0]
            Ay = proj_pts[0][1]
            Bx = proj_pts[1][0]
            By = proj_pts[1][1]
            Cx = proj_pts[2][0]
            Cy = proj_pts[2][1]
            Dx = proj_pts[3][0]
            Dy = proj_pts[3][1]

            ACD = lshelper([Ax, Ay], [Cx, Cy], [Dx, Dy])
            BCD = lshelper([Bx, By], [Cx, Cy], [Dx, Dy])
            ABC = lshelper([Ax, Ay], [Bx, By], [Cx, Cy])
            ABD = lshelper([Ax, Ay], [Bx, By], [Dx, Dy])

            if ACD != BCD and ABC != ABD:
                return True
            else:
                return False

        else:
            return False
