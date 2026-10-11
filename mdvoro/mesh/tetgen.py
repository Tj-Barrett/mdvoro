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
import copy
import os
import json
import subprocess
import numpy as np
import glob
import mdvoro
import inspect
from rich.progress import track

from mdvoro.mesh.run_tetgen import run_tetgen
from mdvoro.mesh.decimate import decimate
# from mdvoro.mesh.triangle_face import triangle_face
from mdvoro.utils.mdutils import mdutils
from mdvoro.utils.ovito_writer import ovito_writer

class tetgen(object):
    def __init__(self):
        self.mesh_args = []
        self.seeds = []
        self.lims = []
        self.cleanup_temp_files()

    @staticmethod
    def cleanup_temp_files(meshname='temp'):
        """
        Clean up temporary mesh files from previous runs.

        Parameters
        ----------
        meshname : str, default='temp'
            Base name of temporary mesh files to remove.
        """
        temp_extensions = ['.1.node', '.1.edge', '.1.face', '.1.ele', '.poly']
        for ext in temp_extensions:
            filepath = meshname + ext
            if os.path.exists(filepath):
                os.remove(filepath)

    def mesh(self, voro, lims, method='tetgen',
             periodic = [False]*3, pbctype = None, tri_face=False,
             grain_boundary=False, nomesh = False, noovito=True,
             reduced_ovito=True, decimate_cutoff = None, query=None, tetcmds = '',
             n_workers=None):

        if pbctype == 'np' or pbctype == 'pbc':
            pbctype = pbctype

        elif pbctype is None:
            if sum(periodic) > 0:
                pbctype = 'pbc'
        else:
            print(f'Error : Unsupported periodic boundary type {pbctype}. Only \'np\', \'pbc\' supported.')
            exit()

        self.pbctype = pbctype

        self.lims = lims
        # limits
        xlo = lims[0][0]
        xhi = lims[0][1]
        ylo = lims[1][0]
        yhi = lims[1][1]
        zlo = lims[2][0]
        zhi = lims[2][1]

        if query is None:
            query = query
        elif isinstance(query, int) and len(voro) > query > 0:
            query = query
        else:
            print(f'Error : Query index is invalid or not an int. Remember query value is the node number indexed from zero (ie subtract 1 from the value in OVITO.)')
            exit()

        reference_cutoff = ( ((xhi-xlo) + (yhi-ylo) + (zhi-zlo))/3. )/3.
        if decimate_cutoff is None:
            decimate_cutoff = reference_cutoff
        elif isinstance(decimate_cutoff, float) and decimate_cutoff > 0:
            decimate_cutoff = decimate_cutoff
        elif isinstance(decimate_cutoff, int) and float(decimate_cutoff) > 0:
            decimate_cutoff = float(decimate_cutoff)
        else:
            print(f'Error : Decimate cutoff is not a float or int.')
            exit()

        if isinstance(tetcmds, str):
            if tetcmds[0]=='-':
                tetcmds=tetcmds[1:]
        else:
            print(f'Error : Tetcmds is not a string. Check and make sure it matches tetgen inputs.')
            exit()

        out = f'''
Important Meshing inputs:
---------------------------------------
If there is a hole in the mesh, lower decimate_cutoff.
A good reference is twice the bead diameter. The specified value is:

decimate_cutoff : {decimate_cutoff}
recommended cutoff based on cell size: {reference_cutoff}

If cell and data match error, either shimmy a single node or use shimmy.

For systems above 200,000, use voro++

      '''

        print(out)

        # if tri_face:
        #     voro = triangle_face(voro,
        #                   lims,
        #                   periodic=periodic,
        #                   min_edge_dist = 1E-3,
        #                   min_edge_cutoff = 1E-6,
        #                   pt_pt_dist_tol = 1E-4,
        #                   split = True,
        #                   clip = False,
        #                   max_area = 0.5)

        voro, phases, qn = decimate(voro,
                                lims,
                                grain_boundary=grain_boundary,
                                cell_cutoff = decimate_cutoff,
                                query = query,
                                n_workers = n_workers)

        pts_global, atts, faces, regions, boundaries, vols, self.seeds, facets, _voro = _voro_reader(self, voro, qn)

        meshname = 'temp'

        _voro = _add_faces_back(_voro)

        if not noovito:
            if not reduced_ovito:
                ovito_writer(meshname+'.dump', voro, lims)
            else:
                ovito_writer(meshname+'.dump', _voro, lims)

        _poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets)

        # fcts = _sep_poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets)
        if not nomesh:
            print(f'Running tetgen with -{tetcmds}')
            run_tetgen(meshname+'.poly', '-'+tetcmds)
            self.mesh_args = _tetgen_reader(meshname)

        return voro

    def write(self, filename, method='abaqus', supplemental_file=None, material_name='MATERIAL'):

        if method == 'abaqus':
            _abaqus_writer(filename, self.mesh_args, self.seeds, self.lims, supplemental_file)
        elif method == 'calculix':
            _calculix_writer(filename, self.mesh_args, self.seeds, self.lims, supplemental_file, material_name)

'''
--------------------------------------------------------------------------------
_abaqus_writer
--------------------------------------------------------------------------------
    Writes to *.inp file for ABAQUS

INPUTS:
    filename    -
    mesh_args   -
    seeds       -

OUTPUTS:
    none, file

'''
def _abaqus_writer(filename, mesh_args, seeds, lims, supplemental_file):

    # limits
    xlo = lims[0][0]
    xhi = lims[0][1]
    ylo = lims[1][0]
    yhi = lims[1][1]
    zlo = lims[2][0]
    zhi = lims[2][1]

    xmid = xlo + (xhi-xlo)/2.0
    ymid = ylo + (yhi-ylo)/2.0
    zmid = zlo + (zhi-zlo)/2.0

    points = mesh_args[0]
    elements = mesh_args[1]
    element_attributes = mesh_args[2]

    '''
        Modified to not carry all points in memory.
        - abaqus - seems to take too long for our application

        Writes only in C3D4, if C3D8 allow Abaqus to handle that conversion
    '''

    seq = open(filename, 'w+')
    seq.write('** Job name: MDvoro \n')
    seq.write('** Model name: tess \n')
    seq.write('*\n')
    seq.write('**\n** PARTS\n**\n')
    seq.write('*Part, name=tess \n \n*Node \n')


    lenprint = len(points)
    boundary_points = [False]*lenprint
    for i in track(range(lenprint),  description= "{:<25}".format('Writing Abaqus file : xyz data')):
        _point = points[i]
        # fix E-17 instances with rounding
        _x = round(_point[0], 4)
        _y = round(_point[1], 4)
        _z = round(_point[2], 4)
        _b = int(_point[3])
        seq.write(str(int(i+1))+', '+str(_x)+', ' \
                                    +str(_y)+', ' \
                                    +str(_z)+'\n')
        if _b == 1:
            boundary_points[i] = True

    del _x, _y, _z, _point

    seq.write('**\n')
    seq.write('*Element, type=C3D4 \n')
    seq.write('**\n')
    lenprint = len(elements)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing Abaqus file : element data')):
        els = elements[i]
        _string = str(int(i+1))+', '
        for _ind in els:
            if _ind == els[-1]:
                _string+= str(_ind)+' '
            else:
                _string+= str(_ind)+', '
        _string+='\n'
        seq.write(_string)

    del els, _string

    max_phase = 0
    lenprint = len(seeds)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing Abaqus file : handling seeds')):
        seed = seeds[i]
        if seed['phase'] > max_phase:
            max_phase = seed['phase']

    elem_atts = np.array(element_attributes)

    # Print Seeds Elset
    elset_seed = {i: [] for i in range(0,int(len(seeds)+1)) }

    # Phase Sets
    if max_phase < 1:
        elset_phase = {1: []}
    else:
        elset_phase = {i: [] for i in range(1,int(max_phase+1)) }

    for elem_ind, elem_att in enumerate(elem_atts):
        seed_num = int(elem_att)
        # seed
        if seed_num not in elset_phase:
            # if not there, assume it is amorphous
            elset_seed[1].append(elem_ind+1)
        else:
            elset_seed[seed_num].append(elem_ind+1)
            # phase
            phase_num = seeds[seed_num-1]['phase']
            if phase_num < 1:
                elset_phase[1].append(elem_ind+1)
            else:
                elset_phase[phase_num].append(elem_ind+1)

    n_elsets = len(elset_seed)

    lenprint = len(elset_phase)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing Abaqus file : phase data')):
        elms = elset_phase[i+1]
        _len = len(elms)
        if _len > 0:
            seq.write('\n*Elset, elset=POLY'+str(i+1)+' \n\n')
            for j in range(_len):
                if j == _len-1:
                    seq.write(str(elms[j])+' \n\n')
                else:
                    seq.write(str(elms[j])+', \n')
    del i, elms, _len
    seq.write('\n')

    lenprint = len(points)
    #
    #
    # to do: limit to just surface nodes -- boundary of 1 from voro
    #
    #
    for i in track(range(lenprint),  description= "{:<25}".format('Writing Abaqus file : Nodes')):
        _point  = points[i]
        if boundary_points[i]:
            if i < lenprint-1:
                seq.write('*Nset, nset=Node-'+str(i+1)+' \n'+str(i+1)+', \n')
            else:
                seq.write('*Nset, nset=Node-'+str(i+1)+' \n'+str(i+1)+' \n')


    '''
    Abaqus doesn't support *equation using node sets  _      _
        despite exporting it in their own inp files    \(ツ)/
    '''

    seq.write('\n')
    seq.write('*END PART\n')

    seq.write('** MDvoro ASSEMBLY Reference Points\n')
    seq.write('*ASSEMBLY, name=Assembly\n')
    seq.write('*INSTANCE, name=TESS, part=TESS\n')
    seq.write('*END INSTANCE\n')
    # x
    seq.write('*Node\n1, '+str(round(xhi,4))+', '+str(round(ymid,4))+', '+str(round(zmid,4))+'\n')
    # y
    seq.write('*Node\n2, '+str(round(xmid,4))+', '+str(round(yhi,4))+', '+str(round(zmid,4))+'\n')
    # z
    seq.write('*Node\n3, '+str(round(xmid,4))+', '+str(round(ymid,4))+', '+str(round(zhi,4))+'\n')
    # 0
    seq.write('*Node\n4,  0.0, 0.0, 0.0 \n')

    seq.write('*Nset, nset=RP-X \n1, \n')
    seq.write('*Nset, nset=RP-Y \n2, \n')
    seq.write('*Nset, nset=RP-Z \n3, \n')
    seq.write('*Nset, nset=RP-0 \n4, \n')

    seq.write('*END ASSEMBLY\n')



    if supplemental_file is not None:
        sf = open(supplemental_file, 'r')
        sf_lines = sf.readlines()
        sf.close()

        for _sf_line in sf_lines:
            seq.write(_sf_line)

    seq.close()

'''
--------------------------------------------------------------------------------
_calculix_writer
--------------------------------------------------------------------------------
    Writes a CalculiX-compatible flat INP file.

    No *Part/*Instance/*Assembly hierarchy — all nodes and elements at the
    global level as required by CalculiX.  The output is directly compatible
    with periodic_inp_writer.py, which adds *EQUATION constraints and load
    steps on top of this file.

INPUTS:
    filename        -
    mesh_args       -
    seeds           -
    lims            -
    supplemental_file -
    material_name   - name used in *Material and *Solid Section blocks

OUTPUTS:
    none, file

'''
def _calculix_writer(filename, mesh_args, seeds, lims, supplemental_file, material_name='MATERIAL'):

    xlo = lims[0][0];  xhi = lims[0][1]
    ylo = lims[1][0];  yhi = lims[1][1]
    zlo = lims[2][0];  zhi = lims[2][1]

    points            = mesh_args[0]
    elements          = mesh_args[1]
    element_attributes = mesh_args[2]

    seq = open(filename, 'w+')
    seq.write('** Job name: MDvoro\n')
    seq.write('** CalculiX flat format — compatible with periodic_inp_writer.py\n')
    seq.write('** Run periodic_inp_writer.write_periodic_inp() on this file to add\n')
    seq.write('** *EQUATION constraints and load steps before running ccx.\n')
    seq.write('**\n')

    # --- Nodes ---
    seq.write('*Node\n')
    lenprint = len(points)
    for i in track(range(lenprint), description="{:<25}".format('Writing CalculiX : nodes')):
        _point = points[i]
        _x = round(_point[0], 4)
        _y = round(_point[1], 4)
        _z = round(_point[2], 4)
        seq.write(str(int(i+1)) + ', ' + str(_x) + ', ' + str(_y) + ', ' + str(_z) + '\n')

    # --- Elements ---
    seq.write('**\n')
    seq.write('*Element, type=C3D4\n')
    lenprint = len(elements)
    for i in track(range(lenprint), description="{:<25}".format('Writing CalculiX : elements')):
        els = elements[i]
        _string = str(int(i+1)) + ', '
        for _ind in els:
            if _ind == els[-1]:
                _string += str(_ind) + '\n'
            else:
                _string += str(_ind) + ', '
        seq.write(_string)

    # --- Phase elsets (same logic as _abaqus_writer) ---
    max_phase = 0
    lenprint = len(seeds)
    for i in track(range(lenprint), description="{:<25}".format('Writing CalculiX : phases')):
        seed = seeds[i]
        if seed['phase'] > max_phase:
            max_phase = seed['phase']

    elem_atts = np.array(element_attributes)

    elset_seed  = {i: [] for i in range(0, int(len(seeds)+1))}
    if max_phase < 1:
        elset_phase = {1: []}
    else:
        elset_phase = {i: [] for i in range(1, int(max_phase+1))}

    for elem_ind, elem_att in enumerate(elem_atts):
        seed_num = int(elem_att)
        if seed_num not in elset_phase:
            elset_seed[1].append(elem_ind+1)
        else:
            elset_seed[seed_num].append(elem_ind+1)
            phase_num = seeds[seed_num-1]['phase']
            if phase_num < 1:
                elset_phase[1].append(elem_ind+1)
            else:
                elset_phase[phase_num].append(elem_ind+1)

    all_elems = []
    lenprint = len(elset_phase)
    for i in track(range(lenprint), description="{:<25}".format('Writing CalculiX : elsets')):
        elms = elset_phase[i+1]
        _len = len(elms)
        if _len > 0:
            seq.write('\n*Elset, elset=POLY' + str(i+1) + '\n\n')
            for j in range(_len):
                if j == _len-1:
                    seq.write(str(elms[j]) + ' \n\n')
                else:
                    seq.write(str(elms[j]) + ', \n')
            all_elems.extend(elms)

    # EALL elset — required by periodic_inp_writer.py step output blocks
    _len = len(all_elems)
    seq.write('\n*Elset, elset=EALL\n\n')
    for j in range(_len):
        if j == _len-1:
            seq.write(str(all_elems[j]) + ' \n\n')
        else:
            seq.write(str(all_elems[j]) + ', \n')

    # --- Material and section ---
    seq.write('\n*Material, name=' + material_name + '\n')
    seq.write('** TODO: Replace with actual material properties\n')
    seq.write('*Elastic\n')
    seq.write('** E, nu\n')
    seq.write('1000.0, 0.3\n')
    seq.write('\n')
    lenprint = len(elset_phase)
    for i in range(lenprint):
        elms = elset_phase[i+1]
        if elms:
            seq.write('*Solid Section, elset=POLY' + str(i+1) + ', material=' + material_name + '\n')

    if supplemental_file is not None:
        sf = open(supplemental_file, 'r')
        sf_lines = sf.readlines()
        sf.close()
        for _sf_line in sf_lines:
            seq.write(_sf_line)

    seq.close()

'''
--------------------------------------------------------------------------------
_add_faces_back
--------------------------------------------------------------------------------
    Adds faces back to complete face volumes

INPUTS:
    filename    -
    mesh_args   -
    seeds       -

OUTPUTS:
    none, file

'''
def _add_faces_back(voro):
    _voro = voro

    cellf = []
    all_faces = []

    for _i, v in track(enumerate(voro),description= "{:<25}".format('Reducing Voro ...')):
        for face in v['faces']:
            cellf.append(_i)
            all_faces.append(face)

    for _i in range(len(voro)):
        for j, _c in enumerate(cellf):
            if all_faces[j]['adjacent_cell'] == _i+1:
                _voro[_i]['faces'].append({'adjacent_cell': _c+1,
                                           'vertices': all_faces[j]['vertices']} )

    return _voro
#

'''
--------------------------------------------------------------------------------
_poly_writer
--------------------------------------------------------------------------------
    Writes to *.poly file for TetGen

    Appending to one big string then calling write is much faster

INPUTS:
    pts_global      -
    faces           -
    Regions         -

OUTPUTS:
    none, file

'''
def _poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets):

    from numpy import abs

    seq = open(meshname+'.poly', 'w+')

    poly_string = ''
    total_pts = len(pts_global)
    poly_string+= '# Node count, 3 dim, attribute, boundary marker \n'
    poly_string+= str(total_pts)+' 3 1 1 \n'
    poly_string+= '# Node index, node coordinates \n'

    lenprint = len(pts_global)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing .poly file : xyz data')):
        xyz = pts_global[i]
        att = int(atts[i])

        _string = str(int(i+1))+' '
        for _i in xyz:
            _loc = round(_i,4)
            _string+='{:.4f}'.format(_loc)+' '
        poly_string+=_string+' \n'#+f'{att} \n'

    seq.write(poly_string)
    poly_string = ''
    del xyz

    # One line: <# of facets> <boundary markers (0 or 1)>
    # Following lines list # of facets:
    #    <facet #>

    # lenprint = len(facets)
    # _faces = 0
    # for i in range(lenprint):
    #     _faces+=facets[i]['count']

    # poly_string+= '\n# Facets, boundary marker \n'
    # poly_string+= str(_faces)+' 1 \n'
    #
    # tot_ff = 0
    # for i in track(range(lenprint),  description= "{:<25}".format('Writing .poly file : facet data')):
    #
    #     fcts = facets[i]['count']
    #     _facets = '# Cell : '+str(i+1)+'\n'
    #     for _f, ff in enumerate(facets[i]['fcts']):
    #         active = int(_f+tot_ff)
    #         # _facets += f'{n} {i+1} # {active} \n'
    #         _facets += f'1 # {active} \n'
    #         _facets += str(len(ff))+' '
    #         for fc in ff:
    #             _facets+= str(fc)+' '
    #         _facets +=' \n'
    #     poly_string+= _facets
    #     tot_ff+=_f
    #
    # seq.write(poly_string)

    ff = []
    tt = []

    # begins at 2
    # for _i, _f in enumerate(faces):
    #     if boundaries[_i] == 0 :
    #         ff.append(_f)
    #         tt.append(boundaries[_i])
    #
    # faces = ff
    # boundaries= tt

    poly_string+= '\n# Facets, boundary marker \n'
    poly_string+= str(len(faces))+' 1 \n'
    poly_string+= '# Facets \n'

    lenprint = len(faces)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing .poly file : facet data')):
        fct = faces[i]
        bound = boundaries[i]
        _facets = f'1 # Facet : '+str(i+1)+'\n'+str(len(fct))+' '
        for fc in fct:
            _facets+= str(fc+1)+' '
        _facets +=f' {bound} \n'
        poly_string+= _facets

    seq.write(poly_string)

    poly_string = ''
    poly_string+= '\n# Part 3 - hole list   \n0 # no hole \n'
    poly_string+= '\n# Part 4 - region list \n'+str(len(regions))+'\n# <region #> <x> <y> <z> <region number> <region attribute> \n'

    lenprint = len(regions)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing .poly file : region data')):
        reg = regions[i]
        _reg = ''
        for r in reg:
            _reg+= str(r)+' '
        _reg +=' \n'
        poly_string+= _reg

    seq.write(poly_string)
    seq.close()

'''
--------------------------------------------------------------------------------
_pytetgen
--------------------------------------------------------------------------------
    Python implementation of tetgen from Pyvist

    Appending to one big string then calling write is much faster

INPUTS:
    pts_global      -
    faces           -
    Regions         -

OUTPUTS:
    none, file

'''
def _pytetgen(meshname, tetcmds, pts_global, atts, faces, regions):
    import pyvista as pv
    import tetgen

    pv.set_plot_theme('document')

    _atts = max(atts)

    # for i in range(_atts):
    #     pts_local = []
    #     for j, pt in enumerate(pts_global)
    #         if i == atts[j]:
    #             pts_local.append(pt)
    #
    #     facelist = faces
    #     faces = []
    #     for fct in facelist:
    #         _f = [len(fct)]
    #         for f in fct:
    #             _f.append(f)
    #         faces.append(_f)
    #     faces = np.concatenate(faces)
    #     # faces = np.hstack(np.array(facelist))
    #     mesh = pv.PolyData( np.array(pts_global), faces=faces)
    #
    #     # surf.plot(eye_dome_lighting=True,show_edges=True, line_width=5)
    #
    #     tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
    #     tet = tetgen.TetGen(np.array(pts_global), faces)
    #     tet.tetrahedralize(regionattrib=atts, switches=tetcmds)
    #     tet.write(f"{meshname}-{i}.vtk", binary=False)

    facelist = faces
    _faces = []
    for fct in facelist:
        _f = [len(fct)]
        for f in fct:
            _f.append(f+1)
        _faces.append(_f)
    _faces = np.concatenate(_faces)
    # faces = np.hstack(np.array(facelist))
    mesh = pv.PolyData( np.array(pts_global), faces=_faces)

    mesh.plot(eye_dome_lighting=True,show_edges=True, line_width=5)

    # tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
    # tet = tetgen.TetGen( np.array(pts_global), _faces )
    # print(f'Running pytetgen with -{tetcmds}')
    # tet.tetrahedralize()#regionattrib=atts, switches=tetcmds)
    # tet.write(f"{meshname}.vtk", binary=False)

def _netgen(meshname, pts_global, atts, faces, regions, boundaries, facets):

    # initialize
    mesh = Mesh()
    mesh.dim = 3

    # add pts
    lenprint = len(pts_global)
    pts = []
    for i in track(range(lenprint),  description= "{:<25}".format('Creating netgen points')):
        xyz = pts_global[i]
        att = int(atts[i])
        pt = mesh.Add(MeshPoint(Pnt(xyz[0], xyz[1], xyz[2])))
        pts.append(pt)

    mesh.SetMaterial(1, "mat")
    # add faces
    lenprint = len(faces)
    lines = []
    gfaces = []
    mesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
    for i in track(range(lenprint),  description= "{:<25}".format('Creating netgen lines')):
        fct = faces[i]

        xx = []
        yy = []
        zz = []
        for j, fc in enumerate(fct):
            xx.append(pts_global[fc][0])
            yy.append(pts_global[fc][1])
            zz.append(pts_global[fc][2])

        xx = np.mean(xx)
        yy = np.mean(yy)
        zz = np.mean(zz)

        pts.append(mesh.Add(MeshPoint(Pnt(xx, yy, zz))))

        bound = boundaries[i]
        gf = []
        for j, fc in enumerate(fct):
            mesh.Add(Element2D(1, [ pts[fc], pts[fct[j-1]], pts[-1] ]))

    mesh.GenerateVolumeMesh()
    # ngmesh = ngsolve.Mesh(mesh)
    # Draw(mesh);
    mesh.Export(f'{meshname}.mesh',"Neutral Format")
    mesh.Save (f"{meshname}.vol")

    # for visualization we need a NGSolve mesh
    # Draw(mesh)


'''
--------------------------------------------------------------------------------
_tetgen_reader
--------------------------------------------------------------------------------
    Reads outputs of TetGen

INPUTS:
    meshname

OUTPUTS:


'''
def _tetgen_reader(meshname):

    if os.path.exists(meshname+'.1.node'):
        pass
    else:
        print(f'Error : Tetgen Unsuccessful. {meshname}.1.node file missing.')
        exit()

    if os.path.exists(meshname+'.1.ele'):
        pass
    else:
        print(f'Error : Tetgen Unsuccessful. {meshname}.1.ele file missing.')
        exit()

    if os.path.exists(meshname+'.1.edge'):
        pass
    else:
        print(f'Error : Tetgen Unsuccessful. {meshname}.1.edge file missing.')
        exit()

    if os.path.exists(meshname+'.1.face'):
        pass
    else:
        print(f'Error : Tetgen Unsuccessful. {meshname}.1.face file missing.')
        exit()


    f = open(meshname+'.1.node', 'r')
    nodes = f.readlines()[1:-1]
    f.close()

    data = []
    att = []
    for i, entry in enumerate(nodes):
        x, n1, n2, n3, t, b = entry.split()
        n1 = float(n1)
        n2 = float(n2)
        n3 = float(n3)
        t  = float(t)
        b  = float(b)

        if t < 0:
            t = -t

        data.append([n1,n2,n3,b])
        att.append(t)

    del nodes

    f = open(meshname+'.1.ele', 'r')
    tetelm = f.readlines()[1:-1]
    f.close()
    cells = []
    elem_att = []
    for i , entry in enumerate(tetelm):
        n, e1, e2, e3, e4, t = entry.split()
        e1 = int(e1)
        e2 = int(e2)
        e3 = int(e3)
        e4 = int(e4)
        t  = float(t)

        cells.append([e1,e2,e3,e4])
        elem_att.append(t)

    del tetelm

    f = open(meshname+'.1.face', 'r')
    tetface = f.readlines()[1:-1]
    f.close()
    faces = []
    nodes = []
    boundary = []
    for i, entry in enumerate(tetface):
        f, n1, n2, n3, b = entry.split()
        f  = int(f)
        n1 = int(n1)
        n2 = int(n2)
        n3 = int(n3)
        b  = int(b)

        faces.append(f)
        nodes.append([n1,n2,n3])
        boundary.append(b)


    pts = np.array(data)
    elems = np.array(cells)
    e_atts = np.array(elem_att)

    # clean up
    if os.path.exists(meshname+'.1.node'):
        os.remove(meshname+'.1.node')
    if os.path.exists(meshname+'.1.edge'):
        os.remove(meshname+'.1.edge')
    if os.path.exists(meshname+'.1.face'):
        os.remove(meshname+'.1.face')
    if os.path.exists(meshname+'.1.ele'):
        os.remove(meshname+'.1.ele')
    # if os.path.exists(meshname+'.poly'):
    #     os.remove(meshname+'.poly')

    mesh_args = (pts, elems, e_atts)

    return mesh_args

'''
--------------------------------------------------------------------------------
_voro_reader
--------------------------------------------------------------------------------

INPUTS:
    meshname

OUTPUTS:

'''
def _voro_reader(self, voro, qn):
    from statistics import mode

    pts_global = []
    atts = []
    pts_seen = []
    pts_dict = {}

    n_cells = len(voro)

    pt_region = []

    seeds = {}

    lenvoro = len(voro)
    single_phase = lenvoro < 2
    lenact = lenvoro-1

    # for c_cell_num in track(range(lenvoro),  description= "{:<25}".format('Reading voronoi tesselation')):
    for c_cell_num in range(lenvoro):
        c_cell_data = voro[c_cell_num]

        if c_cell_data['volume'] < 0:
            seeds[c_cell_num] = {'seed':c_cell_num+1,
                                'seed_loc':[],
                                'phase':[],
                                'face_verts':[],
                                'face_elems':[],
                                'neighs':[],
                                'volume':-1}
        else:
            # seed data of this cell
            seed_data = c_cell_data['original']
            rn = c_cell_num+1
            rx = round(seed_data[0],4)
            ry = round(seed_data[1],4)
            rz = round(seed_data[2],4)
            phase = seed_data[3]

            seeds[c_cell_num] = {'seed':rn,
                                'seed_loc':[rx,ry,rz],
                                'phase':phase,
                                'face_verts':[],
                                'face_elems':[],
                                'neighs':[],
                                'volume':c_cell_data['volume']}

            # face polyhedrals
            for c_face_data in track(c_cell_data['faces'],description= "{:<25}".format(f'Processing Cell : {c_cell_num} of {lenact}')):
                if len(c_face_data) > 0:
                    '''
                    a_cell_num : the adjacent cell number

                    # cell subject
                    c_local_vt : the local vertice indexes, []
                    c_global_locs : array of local vertice locations, [[]]
                    '''
                    a_cell_num = c_face_data['adjacent_cell']
                    c_local_vt = set(c_face_data['vertices'])
                    c_local_pts = c_cell_data['vertices']

                    # Single phase will have no neighbors
                    if not single_phase:
                        if voro[c_cell_num]['volume'] == -1:
                            continue
                        else:
                            seeds[c_cell_num]['neighs'].append(a_cell_num)

                            for c_vt in c_local_vt:
                                # key is (cell_num, local face vert num)
                                key = (c_cell_num, c_vt)

                                '''
                                Check if point is alread connected
                                if already vertice is duplicate, get the original number
                                else add as new point to the lists

                                '''
                                if key in pts_dict:
                                    global_vt_num = pts_dict[key]

                                else:
                                    global_vt_num = len(pts_global) # use length as counter

                                    pt = c_local_pts[c_vt]

                                    # tuples hash
                                    # if the rounding is too coarse, some points may merge introducing holes to the mesh
                                    pts_global.append(tuple([round(pt[0],4),
                                                             round(pt[1],4),
                                                             round(pt[2],4)]))
                                    atts.append(seeds[c_cell_num]['phase'])

                                    pts_seen.append({c_cell_num:key})
                                    pts_dict[key]=global_vt_num

                                    pt_region.append(rn)

                                seen_list = set()
                                if  0 <= a_cell_num < n_cells:
                                    seen_list = set(pts_seen[global_vt_num])

                                    if a_cell_num not in seen_list:
                                        a_cell_data = voro[a_cell_num]
                                        a_local_pts = np.array(a_cell_data['vertices'])

                                        '''
                                        Adjust relative location, determine distances, and find minimum

                                        Minimum will be the overlap location between the same vertice
                                        of both the cell and its neighbor
                                        '''
                                        rel_pos = a_local_pts - pts_global[global_vt_num]
                                        sq_dist = np.sum(rel_pos * rel_pos, axis=-1)
                                        a_min_vt = np.argmin(sq_dist)

                                        '''
                                        Get minimum distance and determine if it is across the periodic
                                            boundary. If it is not, then ok to be adjacent
                                        '''
                                        a_min_pt = a_local_pts[a_min_vt]

                                        adj_key = (a_cell_num, a_min_vt)

                                        if mdutils.is_not_far(a_min_pt, pts_global[global_vt_num], 1E-3):
                                            # close, same nodal position
                                            pts_dict[adj_key] = global_vt_num
                                            pts_seen[global_vt_num][a_cell_num] = a_min_vt

                                            pt_region.append(c_cell_num+1)

                    else:
                        seeds[c_cell_num]['neighs'].append(a_cell_num)

                        for c_vt in c_local_vt:
                            # key is (cell_num, local face vert num)
                            key = (c_cell_num, c_vt)

                            '''
                            Check if point is alread connected
                            if already vertice is duplicate, get the original number
                            else add as new point to the lists

                            '''
                            if key in pts_dict:
                                global_vt_num = pts_dict[key]

                            else:
                                global_vt_num = len(pts_global) # use length as counter
                                pt = c_local_pts[c_vt]

                                # tuples hash
                                # if the rounding is too coarse, some points may merge introducing holes to the mesh
                                pts_global.append(tuple([round(pt[0],4),
                                                         round(pt[1],4),
                                                         round(pt[2],4)]))
                                atts.append(seeds[c_cell_num]['phase'])

                                pts_seen.append({c_cell_num:key})
                                pts_dict[key]=global_vt_num

                                pt_region.append(rn)

    '''
    cell_index = linear atom number
    attribute  = phase number


    --- tidy up duplicate points ---
    pts_global is a tuple so its hashable
    1) use x,y,z as hashable index to check if exists
    1a) if exists, add to indicies
    1b) if does not, make new dictionary entry

    2) remake pts_reduced using indexes and reduced x,y,z location data
    '''
    pg = {}
    _pts_global = []
    _pts_att = []
    for i, pt in track(enumerate(pts_global),description= "{:<25}".format(f'Finding verts to merge')):
        if pt in pg:
            pg[pt]['inds'].append(i)
        else:
            pg[pt] = {'original' : len(_pts_global),
                      'pt': pt,
                      'atts' : atts[i],
                      'inds' : [i]}

            _pts_global.append(pt)
            _pts_att.append(atts[i])

    _new_pts = [-1 for i in pts_global]

    for i, pt in track(enumerate(pg),description= "{:<25}".format(f'Merging verts ...')):
        o = pg[pt]['original']
        d = pg[pt]['inds']
        for j in d:
            _new_pts[j] = o

    pts_global = _pts_global
    atts = _pts_att

    '''
    Basic regions and clean

    '''

    faces = []
    adjs = []
    faces_seen = []
    boundaries = []
    cell_index = []

    # adjmerge = [ [] for i in range(voro)]
    regions = []
    total_regions = 0

    for c_cell_num, cell_data in enumerate(voro):
        _x = []
        _y = []
        _z = []

        for c_face_num, c_face_data in enumerate(cell_data['faces']):
            if len(c_face_data) > 0:
                # adjacent cell on face
                a_cell_num = c_face_data['adjacent_cell']
                # local
                c_local_vts = c_face_data['vertices']
                # global
                if not single_phase:
                    if voro[c_cell_num]['volume'] == -1:
                        continue
                    else:
                        cgv = []
                        for i, vt in enumerate(c_local_vts):
                            if (c_cell_num, vt) in pts_dict:
                                if i == 0:
                                    boundaries.append(seeds[c_cell_num]['neighs'][i])
                                ind = _new_pts[pts_dict[(c_cell_num, vt)]]
                                cgv.append(ind)
                                _x.append(pts_global[ind][0])
                                _y.append(pts_global[ind][1])
                                _z.append(pts_global[ind][2])

                        faces.append(cgv)
                        cell_index.append(c_cell_num)
                        seeds[c_cell_num]['face_verts'].append(cgv)
                        adjs.append(c_cell_num)

                else:
                    cgv = []
                    for i, vt in enumerate(c_local_vts):
                        if (c_cell_num, vt) in pts_dict:
                            if i == 0:
                                boundaries.append(seeds[c_cell_num]['neighs'][i])
                            ind = _new_pts[pts_dict[(c_cell_num, vt)]]
                            cgv.append(ind)
                            _x.append(pts_global[ind][0])
                            _y.append(pts_global[ind][1])
                            _z.append(pts_global[ind][2])

                    faces.append(cgv)
                    cell_index.append(c_cell_num)
                    seeds[c_cell_num]['face_verts'].append(cgv)
                    adjs.append(c_cell_num)

        total_regions += 1
        regions.append([total_regions, round(np.mean(_x),4) , round(np.mean(_y),4), round(np.mean(_z),4), c_cell_num, -1.])

    clean_index = []
    clean_faces = []
    clean_adjs  = []
    for _i, face in enumerate(faces):
        if len(set(face))>2:
            # dropping redundent points after merge. Using set breaks it for some reason that I dont feel like looking into right now
            f = []
            for _f in face:
                if _f not in f:
                    f.append(_f)

            clean_faces.append(f)
            clean_index.append(cell_index[_i])
            clean_adjs.append(adjs[_i])

    faces = clean_faces
    cell_index = clean_index
    adjs = clean_adjs

    _boundaries = []

    facets = {}
    for _i, face in track(enumerate(faces),description= "{:<25}".format('Renumbering Face vts ...')):
        new_face = []
        mm_atts = []
        if cell_index[_i] in facets:
            facets[cell_index[_i]]['count']+= 1
        else:
            facets[cell_index[_i]] = {'count':0,
                                      'fcts':[],
                                      'adj':[]}
        for f in face:
            # new_face.append(f)
            mm_atts.append(atts[f])
        phase = mode(mm_atts)

        _boundaries.append(cell_index[_i])

        facets[cell_index[_i]]['fcts'].append(face)
        facets[cell_index[_i]]['adj'].append(cell_index[_i])

    # Create volume list
    vols = [cell['volume'] for cell in voro]
    boundaries = _boundaries

    '''
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
    _voro = []
    for _i, f in track(enumerate(facets),description= "{:<25}".format('Reducing Voro ...')):
        _v = {'volume':vols[f],
              'vertices':[],
              'adjacency':[],
              'faces':[],
              'original':voro[f]['original']}

        _tot_pts = []
        _vvs = set()
        _vv = []
        for _j, face in enumerate(facets[f]['fcts']):
            _vts = []
            for vt in face:
                if vt not in _vvs:
                    _vts.append(len(_tot_pts))
                    _tot_pts.append(pts_global[vt])
                    _vvs.add(vt)
                    _vv.append(vt)
                else:
                    _vts.append(_vv.index(vt))

            _v['faces'].append({'vertices':_vts,
                                'adjacent_cell':_j})

        _v['vertices']= _tot_pts

        _voro.append(_v)

    '''
    Figure out if there are any disconnected areas. Need to recursively build out faces and check if there are any left at the end.

    Still a bug here. Tetgen still creates tetrahedrals that do not belong. If we keep the region count to zero it correctly identifies all.

    We can work around by ranking by decending volume, but thats not a good solution.

    Actual workaround seems to be just labeling crystals. Something happens with centroids that fall outside the volume

    '''
    sub_regions = []
    # new_atts = atts
    new_atts = np.zeros((len(pts_global)))
    total_regions = 0

    from random import randint

    for c_cell_num, cell_data in enumerate(voro):

        all_vts = []
        for c_face_num, c_face_data in enumerate(cell_data['faces']):

            c_local_vts = c_face_data['vertices']
            for i, vt in enumerate(c_local_vts):
                ind = _new_pts[pts_dict[(c_cell_num, vt)]]
                all_vts.append(ind)

        # build connectivity table
        bonds = []
        for c_face_num, c_face_data in enumerate(cell_data['faces']):
            for _i in range(0,len(c_face_data['vertices'])-1):
                if _i == 0:
                    a = c_face_data['vertices'][0]
                    inda = _new_pts[pts_dict[(c_cell_num, a)]]
                    b = c_face_data['vertices'][-1]
                    indb = _new_pts[pts_dict[(c_cell_num, b)]]
                    bonds.append([inda,indb])
                    bonds.append([indb,inda])

                a = c_face_data['vertices'][_i]
                inda = _new_pts[pts_dict[(c_cell_num, a)]]
                b = c_face_data['vertices'][_i+1]
                indb = _new_pts[pts_dict[(c_cell_num, b)]]
                bonds.append([inda,indb])
                bonds.append([indb,inda])

        # doubled so only do one
        all_vts = list(set(all_vts))
        lav = len(all_vts)
        popped = []

        while len(all_vts) > 0:
            r = randint(0,len(all_vts)-1)
            vt = all_vts[r]

            total_regions += 1
            pairs = []
            done = []

            # pairs.append(vt)
            for b in bonds:
                if vt == b[0] and b not in done :
                    pairs.append(b[1])
                    done.append(b)
            # recursively build pair list
            for p in pairs:
                # print(pairs)
                for b in bonds:
                    if p == b[0] and b not in done :
                        pairs.append(b[1])
                        done.append(b)

            # done, pop
            for p in set(pairs):
                if p not in popped:
                    all_vts.pop(all_vts.index(p))
                    popped.append(p)
                    new_atts[p]=total_regions

            _x = []
            _y = []
            _z = []
            for p in set(pairs):
                # new_atts[p]=total_regions
                _x.append(pts_global[p][0])
                _y.append(pts_global[p][1])
                _z.append(pts_global[p][2])

            if len(popped) == lav:
                sub_regions.append([total_regions, round(np.mean(_x),4) , round(np.mean(_y),4), round(np.mean(_z),4), total_regions, -1.])

    # print(sub_regions)
    return pts_global, new_atts, faces, sub_regions, boundaries, vols, seeds, facets, _voro
