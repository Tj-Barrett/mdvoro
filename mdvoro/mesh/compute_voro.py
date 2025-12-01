'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

--------------------------------------------------------------------------------
MDvoro
--------------------------------------------------------------------------------


================================================================================
MDvoro
================================================================================

INPUTS:

OUTPUTS:


'''
# import pyvoro
import os
import math
import numpy as np

def compute_voro(data, lims, dispersion = 1.0, radii=[], periodic=[False]*3,
    shimmy = None, shimmy_atom = None, shift_bounds=False, drop= False,
    seg_lim=None, method = 'pyvoro'):

    if seg_lim is not None:
        xlo = lims[0][0] - seg_lim
        xhi = lims[0][1] + seg_lim
        ylo = lims[1][0] - seg_lim
        yhi = lims[1][1] + seg_lim
        zlo = lims[2][0] - seg_lim
        zhi = lims[2][1] + seg_lim
    else:
        xlo = lims[0][0]
        xhi = lims[0][1]
        ylo = lims[1][0]
        yhi = lims[1][1]
        zlo = lims[2][0]
        zhi = lims[2][1]

    lims = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

    if shimmy is not None and shimmy > 0:
        dispx = np.random.uniform(low = -shimmy, high= shimmy, size = len(data))
        dispy = np.random.uniform(low = -shimmy, high= shimmy, size = len(data))
        dispz = np.random.uniform(low = -shimmy, high= shimmy, size = len(data))

    if shimmy is not None:
        xyz = []
        if len(data[0]) == 4:
            for _i, _xyz in enumerate(data):
                _x = _xyz[0]+dispx[_i]
                _y = _xyz[1]+dispy[_i]
                _z = _xyz[2]+dispz[_i]
                xyz.append([_x,_y,_z,_xyz[3]])
        else:
            for _i, _xyz in enumerate(data):
                _x = _xyz[0]+dispx[_i]
                _y = _xyz[1]+dispy[_i]
                _z = _xyz[2]+dispz[_i]
                xyz.append([_x,_y,_z])
        data = xyz

    if shimmy_atom is not None and isinstance(shimmy_atom[0], int) and isinstance(shimmy_atom[1], float) :
        ind = shimmy_atom[0]
        dispx = np.random.uniform(low = -shimmy_atom[1], high= shimmy_atom[1], size = 1)
        dispy = np.random.uniform(low = -shimmy_atom[1], high= shimmy_atom[1], size = 1)
        dispz = np.random.uniform(low = -shimmy_atom[1], high= shimmy_atom[1], size = 1)
        xyz = []
        if len(data[0]) == 4:
            for _i, _xyz in enumerate(data):
                if _i == int(ind):
                    _x = _xyz[0]+dispx[0]
                    _y = _xyz[1]+dispy[0]
                    _z = _xyz[2]+dispz[0]
                    xyz.append([_x,_y,_z,_xyz[3]])
                else:
                    _x = _xyz[0]
                    _y = _xyz[1]
                    _z = _xyz[2]
                    xyz.append([_x,_y,_z,_xyz[3]])
        else:
            for _i, _xyz in enumerate(data):
                if _i == int(ind):
                    _x = _xyz[0]+dispx[0]
                    _y = _xyz[1]+dispy[0]
                    _z = _xyz[2]+dispz[0]
                    xyz.append([_x,_y,_z])
                else:
                    _x = _xyz[0]
                    _y = _xyz[1]
                    _z = _xyz[2]
                    xyz.append([_x,_y,_z])
        data = xyz

    elif shimmy_atom is not None :
        raise ValueError('shimmy_atom variable should be [int, float]')

    if shift_bounds is True or shimmy_atom is not None or shimmy is not None:
        _x = []
        _y = []
        _z = []
        for _i, _xyz in enumerate(data):
            _x.append(_xyz[0])
            _y.append(_xyz[1])
            _z.append(_xyz[2])

        if shimmy_atom is not None:
            xlo = np.min(_x) - shimmy_atom[1]
            ylo = np.min(_y) - shimmy_atom[1]
            zlo = np.min(_z) - shimmy_atom[1]

            xhi = np.max(_x) + shimmy_atom[1]
            yhi = np.max(_y) + shimmy_atom[1]
            zhi = np.max(_z) + shimmy_atom[1]
        elif shimmy is not None:
            xlo = np.min(_x) - np.max(dispx)
            ylo = np.min(_y) - np.max(dispy)
            zlo = np.min(_z) - np.max(dispz)

            xhi = np.max(_x) + np.max(dispx)
            yhi = np.max(_y) + np.max(dispy)
            zhi = np.max(_z) + np.max(dispz)

        lims = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

    if method == 'pyvoro':
        voro = pyvoro.compute_voronoi(data,
                                      lims,
                                      dispersion = dispersion,
                                      radii = radii,
                                      periodic=periodic)
        return voro

    elif method == 'voro++':
        # Dispersion to determine block sizes
        xlen = max([1, int(math.floor((xhi-xlo)/dispersion))])
        ylen = max([1, int(math.floor((yhi-ylo)/dispersion))])
        zlen = max([1, int(math.floor((zhi-zlo)/dispersion))])

        # shimy to modify location slightly

        # Make temporary voro file to read
        if len(radii) != len(data):
            rad_bool = False
            radical = ''
        else:
            rad_bool = True
            radical = '-r'

        seq = open('temp.voro', 'w+')
        if rad_bool:
            for _i, xyz in enumerate(data):
                    seq.write(f'{_i} {xyz[0]} {xyz[1]} {xyz[2]} {radii[_i]}\n')
        else:
            for _i, xyz in enumerate(data):
                    seq.write(f'{_i} {xyz[0]} {xyz[1]} {xyz[2]}\n')
        seq.close()

        # make correct periodic boundary assignment
        if periodic[0] and periodic[1] and periodic[0]:
            per = '-p'
        elif periodic[0] and periodic[1] :
            per = '-px -py'
        elif periodic[0] and periodic[2] :
            per = '-px -pz'
        elif periodic[1] and periodic[2] :
            per = '-py -pz'
        elif periodic[0]:
            per = '-px'
        elif periodic[0]:
            per = '-py'
        elif periodic[0]:
            per = '-py'
        elif sum(periodic) == 0:
            per = ''
        else:
            raise ValueError('Periodic List is empty.')

        '''
        -r radical tessellation

        -c custom output to build pyvoro output

        %i index
        %q x,y,z location of original
        %v volume

        %w number of vertices in cell
        %P vertices global (x,y,z) data

        %s number of faces
        %t bracketed list of face vertices
        %n adjacent cell number of face
        '''

        os.system(f'voro++ {per} -o {radical} -n {xlen} {ylen} {zlen} -c "%i %q %v %w %P %s %n %t %f" {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} temp.voro')

        f = open('temp.voro.vol', 'r')
        nodes = f.readlines()
        f.close()

        '''
        PyVoro caps out at ~250,000 points. Voro++ handles above 1,000,000 easily, so this side steps pyvoro

        Directly from pyvoro:
        Output format is a list of cells as follows:
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
        if len(nodes) != len(data):
            print('Mismatch in data lengths :')
            print('Voro nodes : '+str(len(nodes)))
            print('Data nodes : '+str(len(data)))
            cells = list(map(_map,nodes))
            _voro_data = set()
            for _i in range(len(cells)):
                _voro_data.add(cells[_i]['original'][3])

            _pop = set()
            for _i in range(len(data)):
                if _i in _voro_data:
                    continue
                else:
                    _pop.add(_i)

            if drop:
                print('Drop flag active, removing lost node ...')
                _data = data
                for _i in _pop:
                    _data.pop(_i)
                    print(f'Lost node index : {_i}')
                data = _data
            else:
                print('Lost node indexes :')
                print(_pop)
                print('\n')
                raise ValueError('voro++ did not return the same number of cells it was given')
        else:
            cells = list(map(_map,nodes))


        # checks and makes sure each face has a match on the other cell
        for c_cell_num, c_cell_data in enumerate(cells):
            # assign type correctly (only for voro, redundent for pyvoro)
            x = c_cell_data['original'][0]
            y = c_cell_data['original'][1]
            z = c_cell_data['original'][2]
            if len(data[0]) == 4:
                t = data[int(c_cell_data['original'][3])][3]
            else:
                t = int(c_cell_data['original'][3])
            cells[c_cell_num]['original'] = [x,y,z,t]
            # face polyhedrals
            _mismatch = True
            for c_face_num, c_face_data in enumerate(c_cell_data['faces']):
                '''
                a_cell_num : the adjacent cell number
                a_cell_data : the adjacent cell data
                '''
                a_cell_num = c_face_data['adjacent_cell']
                if a_cell_num > 0:
                    a_cell_data = cells[a_cell_num]
                    for a_face_num, a_face_data in enumerate(a_cell_data['faces']):
                        if _mismatch and a_face_data['adjacent_cell'] == c_cell_num:
                            _mismatch = False


                    if _mismatch:
                        del cells[c_cell_num]['faces'][c_face_num]

        return cells


'''
 Voro++ methodology only

'''

def _map(entry):
    min_area = 1E-3

    sp = entry.split()
    # seed location
    xyzt = [float(sp[1]),float(sp[2]),float(sp[3]),float(sp[0])]
    # volume
    vol  = float(sp[4])
    # number of vertices
    w = int(sp[5])
    # vertices (x,y,z) location data
    vertices = [list(eval(sp[i])) for i in range(6,w+6)]
    # number of faces
    s = int(sp[w+6])
    # adjacent cells
    adjacency = [int(sp[i]) for i in range(w+7, w+7+s)]
    # faces and face adjacency
    # face_pops = []
    # faces = []
    # for i in range(w+7, w+7+s):
    #     if float(sp[i+2*s]) > min_area:
    #         faces = [{'vertices':list(eval(sp[i+s])) ,'adjacent_cell':int(sp[i])}]
    #     else:

    faces = [{'adjacent_cell':int(sp[i]), 'vertices':list(eval(sp[i+s]))} for i in range(w+7, w+7+s)]# if float(sp[i+2*s]) > min_area]

    cell  =   {'original':xyzt,
               'volume':vol,
               'vertices':vertices,
               'adjacency':adjacency,
               'faces':faces}
    return cell
