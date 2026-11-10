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

# decimals of the PLC and mesh coordinates: also the key that merges the copies of a
# Voronoi vertex shared by several cells. 6 needs voro++ with full-precision output
# (patches/voropp); stock voro++ prints ~6 significant digits, and mesh() then falls
# back to 4 (coarser rounding folds facets through each other in some boxes, finer
# rounding leaves near-zero-volume slivers the solvers cannot factor)
DECIMALS = 6
_DEC = 4          # in effect for the current mesh() call


def _full_precision(voro, n=200):
    '''True if the cell vertices carry more than 6 significant digits'''
    v = np.concatenate([np.ravel(c['vertices']) for c in voro[:n] if len(c['vertices'])])
    v = v[np.abs(v) > 1e-3]
    return len(v) > 0 and np.mean([abs(x - float(f'{x:.6g}')) > 1e-12 * abs(x) for x in v[:5000]]) > 0.5

from mdvoro.mesh.run_tetgen import run_tetgen, plc_self_intersects
from mdvoro.mesh.poly_tools import (unique_triangulated_facets, snap_to_box, collapse_short_edges,
                                    collapse_short_edges_periodic, collapse_short_edges_translational,
                                    unique_triangulated_facets_translational)
from mdvoro.mesh.periodic_clip import clip_periodic_cells, whole_periodic_cells
from mdvoro.mesh.decimate import decimate
# from mdvoro.mesh.triangle_face import triangle_face
from mdvoro.utils.mdutils import mdutils
from mdvoro.utils.ovito_writer import ovito_writer

class tetgen(object):
    def __init__(self):
        self.mesh_args = []
        self.seeds = []
        self.region_phase = None
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
             n_workers=None, min_edge=None, clip_periodic=False, min_edge_box=None, clip_snap=None,
             whole_periodic=False):
        '''min_edge: merge PLC vertices joined by shorter edges before meshing
        (near-degenerate Voronoi vertices give sliver tetrahedra); None keeps all.
        min_edge_box (periodic): an interior vertex this close to a box-surface vertex
        merges into it (the box vertex stays), e.g. an interface vertex just off a face.
        clip_periodic: voro is a fully periodic tessellation of lims; clip it to the
        box (periodic_clip) and keep opposite box faces identical, so the mesh has
        matching boundary nodes for periodic BCs. Mesh it with TetGen -Y (no
        Steiner points on facets), or the box faces lose their match.
        clip_snap (clip_periodic): Voronoi vertices this close to a box plane move onto it
        before the cut (periodic_clip snap), so the cut makes no short edges.
        whole_periodic: voro is a fully periodic tessellation of lims, meshed as it is: the
        domain is the union of the whole cells (a jagged fundamental domain), its boundary
        the cell faces shared across the periodic boundary, in pairs of translates kept
        identical (poly_tools translation classes). Mesh it with TetGen -Y'''
        if clip_periodic and whole_periodic:
            raise ValueError('clip_periodic and whole_periodic exclude each other')
        global _DEC
        _DEC = DECIMALS if DECIMALS <= 4 or _full_precision(voro) else 4
        if _DEC != DECIMALS:
            print(f'WARNING: voro++ output has ~6 significant digits: writing {_DEC} decimals instead of '
                  f'{DECIMALS} (patch voro++ with mdvoro/patches/voropp for full precision)')
        self.decimals = _DEC
        self.clip_periodic = clip_periodic
        if clip_periodic:
            voro = clip_periodic_cells(voro, lims, snap=clip_snap)
            periodic = [False] * 3               # the pieces are a non-periodic tessellation of the box
        self.clip_snap = clip_snap if clip_periodic else None
        self.whole_periodic = whole_periodic
        translational = None
        if whole_periodic:
            voro = whole_periodic_cells(voro, lims)
            periodic = [False] * 3
            translational = [l[1] - l[0] for l in lims]

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
            if tetcmds.startswith('-'):
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

        # TetGen regions: one per connected same-phase set of cells, computed on
        # the full tessellation (decimate drops interior cells and edits faces)
        regions, self.region_phase = _phase_regions(voro, decimate_cutoff)

        voro, phases, qn = decimate(voro,
                                lims,
                                grain_boundary=grain_boundary,
                                cell_cutoff = decimate_cutoff,
                                query = query,
                                n_workers = n_workers)

        pts_global, atts, faces, _, boundaries, vols, self.seeds, facets, _voro = _voro_reader(self, voro, qn)

        meshname = 'temp'

        _voro = _add_faces_back(_voro)

        if not noovito:
            if not reduced_ovito:
                ovito_writer(meshname+'.dump', voro, lims)
            else:
                ovito_writer(meshname+'.dump', _voro, lims)

        # collapsing edges can fold a facet through another: TetGen reports the
        # intersecting facets; keep their vertices from merging and try again.
        # Periodic boxes merge box-edge vertices along their edges too: voro++
        # round-off leaves clusters 1e-4 apart there, slivers at best and
        # mismatched periodic copies at worst. Collapsing nothing at all is the
        # last resort.
        frozen = set()
        merge_box_edges = bool(clip_periodic)
        self.plc_intersections = np.zeros((0, 3))
        for attempt in range(20):
            collapse = _poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets,
                                    min_edge=min_edge, lims=lims, frozen=frozen, periodic=clip_periodic,
                                    merge_box_edges=merge_box_edges, min_edge_box=min_edge_box if min_edge else None,
                                    translational=translational)
            if not min_edge:
                break
            bad = plc_self_intersects(meshname+'.poly')
            if len(bad) == 0:
                break
            new_xyz, (rep, keep) = collapse
            if np.isnan(bad).any():
                min_edge = None                  # no location reported: do not collapse
                continue
            hit = [int(np.argmin(np.linalg.norm(new_xyz - b, axis=1))) for b in bad]
            reps = [int(keep[k]) for k in hit]   # new vertex -> its cluster representative
            added = set(np.nonzero(np.isin(rep, reps))[0].tolist()) - frozen
            print(f'PLC self-intersects at {len(bad)} vertices: keeping {len(added)} vertices unmerged')
            if not added:
                min_edge = None
                continue
            frozen |= added
        else:
            min_edge = None
            _poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets,
                         lims=lims if clip_periodic else None, periodic=clip_periodic, translational=translational)
        if not min_edge:                         # collapse given up: is the plain PLC clean?
            bad = plc_self_intersects(meshname+'.poly')
            if len(bad):
                self.plc_intersections = bad
                print(f'WARNING: the PLC still self-intersects at {len(bad)} vertices without collapsing '
                      f'(e.g. {np.round(bad[0], _DEC).tolist()}); TetGen will reject it')
        self.min_edge = min_edge
        self.min_edge_box = min_edge_box if min_edge else None
        self.merge_box_edges = merge_box_edges
        # TetGen's region attributes number the connected phase regions, which
        # only exist here: keep their phases so the mesh can be used on its own
        with open(meshname+'.regions', 'w') as f:
            f.write('# region attribute, phase\n')
            for att in sorted(self.region_phase):
                f.write(f"{att} {self.region_phase[att]}\n")

        # fcts = _sep_poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets)
        if not nomesh:
            print(f'Running tetgen with -{tetcmds}')
            run_tetgen(meshname+'.poly', '-'+tetcmds)
            self.mesh_args = _tetgen_reader(meshname)

        return voro

    def write(self, filename, method='abaqus', supplemental_file=None, material_name='MATERIAL'):

        if method == 'abaqus':
            _abaqus_writer(filename, self.mesh_args, self.seeds, self.lims, supplemental_file,
                           region_phase=self.region_phase)
        elif method == 'calculix':
            _calculix_writer(filename, self.mesh_args, self.seeds, self.lims, supplemental_file, material_name,
                             region_phase=self.region_phase)


def _phase_regions(voro, cell_cutoff):
    """TetGen region points: an attribute per connected set of same-phase cells.

    Cells are connected through shared faces of cells with the same phase whose
    seeds are within cell_cutoff (farther ones are neighbours across a periodic
    boundary, kept apart as in decimate). Every cell's seed, which lies strictly
    inside the cell, is a region point with its set's attribute: the PLC can
    split a set where it is joined only by a tiny face that collapse_short_edges
    closes, and each piece still needs a point. TetGen floods a region from its
    first point and only locates the others, so each set's first point is the
    seed of its largest cell with no neighbour of another phase.

    Returns (regions, region_phase): rows [n, x, y, z, attribute, -1.] for the
    .poly region list, and {attribute: phase}.
    """
    n = len(voro)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seed = np.array([c['original'][0:3] for c in voro], dtype=float)
    phase = [c['original'][3] for c in voro]
    mixed = np.zeros(n, dtype=bool)          # has a neighbour of another phase
    for i, cell in enumerate(voro):
        for face in cell['faces']:
            if len(face) == 0:
                continue
            j = face['adjacent_cell']
            if j < 0 or j >= n:
                continue
            if phase[i] != phase[j]:
                mixed[i] = True
            elif j > i and np.linalg.norm(seed[i] - seed[j]) <= cell_cutoff:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    best = {}
    for i in range(n):
        r = find(i)
        key = (not mixed[i], voro[i]['volume'])
        if r not in best or key > best[r][0]:
            best[r] = (key, i)

    regions, region_phase, att_of = [], {}, {}
    for att, r in enumerate(sorted(best), start=1):
        i = best[r][1]
        x, y, z = seed[i]
        regions.append([att, round(x, 6), round(y, 6), round(z, 6), att, -1.])
        region_phase[att] = phase[i]
        att_of[r] = att
    first = {i for _, i in best.values()}
    for i in range(n):
        if i not in first:
            x, y, z = seed[i]
            regions.append([len(regions) + 1, round(x, 6), round(y, 6), round(z, 6), att_of[find(i)], -1.])
    return regions, region_phase


def _phase_elsets(element_attributes, seeds, region_phase=None):
    """{phase elset number: [element ids]} from TetGen region attributes.

    region_phase maps attribute -> phase (tetgen.mesh); without it, attribute
    a is cell a-1 of seeds. Amorphous (phase < 1) and unknown attributes go
    to elset 1.
    """
    if region_phase is None:
        region_phase = {i + 1: seeds[i]['phase'] for i in range(len(seeds))}
    max_phase = max([1] + [int(p) for p in region_phase.values()])
    elset_phase = {i: [] for i in range(1, max_phase + 1)}
    for elem_ind, elem_att in enumerate(np.asarray(element_attributes)):
        p = region_phase.get(int(elem_att), 0)
        elset_phase[int(p) if p >= 1 else 1].append(elem_ind + 1)
    return elset_phase

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
def _abaqus_writer(filename, mesh_args, seeds, lims, supplemental_file, region_phase=None):

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
        _x = round(_point[0], _DEC)
        _y = round(_point[1], _DEC)
        _z = round(_point[2], _DEC)
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

    elset_phase = _phase_elsets(element_attributes, seeds, region_phase)

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
        despite exporting it in their own inp files    \\(ツ)/
    '''

    seq.write('\n')
    seq.write('*END PART\n')

    seq.write('** MDvoro ASSEMBLY Reference Points\n')
    seq.write('*ASSEMBLY, name=Assembly\n')
    seq.write('*INSTANCE, name=TESS, part=TESS\n')
    seq.write('*END INSTANCE\n')
    # x
    seq.write('*Node\n1, '+str(round(xhi, _DEC))+', '+str(round(ymid, _DEC))+', '+str(round(zmid, _DEC))+'\n')
    # y
    seq.write('*Node\n2, '+str(round(xmid, _DEC))+', '+str(round(yhi, _DEC))+', '+str(round(zmid, _DEC))+'\n')
    # z
    seq.write('*Node\n3, '+str(round(xmid, _DEC))+', '+str(round(ymid, _DEC))+', '+str(round(zhi, _DEC))+'\n')
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
def _calculix_writer(filename, mesh_args, seeds, lims, supplemental_file, material_name='MATERIAL',
                     region_phase=None):

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
        _x = round(_point[0], _DEC)
        _y = round(_point[1], _DEC)
        _z = round(_point[2], _DEC)
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
    elset_phase = _phase_elsets(element_attributes, seeds, region_phase)

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
def _poly_writer(meshname, pts_global, atts, faces, regions, boundaries, facets, min_edge=None, lims=None,
                 frozen=(), periodic=False, merge_box_edges=False, min_edge_box=None, translational=None):
    '''periodic: lims is a periodic box whose opposite faces carry the same polygons
    (periodic_clip); edge collapse and triangulation then keep them identical.
    translational: box lengths of whole periodic cells (tetgen.mesh whole_periodic); faces
    that are translates of each other stay so (poly_tools translation classes)'''
    ttol = 3 * 10.0 ** -_DEC                     # translates agree to the written decimals
    if translational is not None:
        lims = None

    from numpy import abs

    if lims is not None:
        # box-face vertices exactly on the faces, so that they round alike
        pts_global = snap_to_box(pts_global, lims)
    collapse = None
    if min_edge:
        # near-degenerate Voronoi vertices: merge them, or TetGen makes slivers
        n_before = len(pts_global)
        if translational is not None:
            faces, pts_global, keep, src, rep = collapse_short_edges_translational(
                [list(f) for f in faces], np.round(np.asarray(pts_global, float), _DEC), min_edge, translational,
                frozen, tol=ttol)
        elif periodic:
            faces, pts_global, keep, src, rep = collapse_short_edges_periodic(
                [list(f) for f in faces], pts_global, min_edge, lims, frozen, merge_box_edges=merge_box_edges,
                min_edge_box=min_edge_box)
        else:
            faces, pts_global, keep, src, rep = collapse_short_edges([list(f) for f in faces], pts_global,
                                                                     min_edge, lims, frozen)
        boundaries = [boundaries[i] for i in src]
        atts = [atts[i] for i in keep]
        collapse = (np.round(pts_global, _DEC), (rep, keep))
        print(f'Collapsed edges shorter than {min_edge}: {n_before} -> {len(pts_global)} vertices')

    if translational is not None:
        tris, marks, n_unique = unique_triangulated_facets_translational(
            faces, boundaries, np.round(np.asarray(pts_global, float), _DEC), translational, tol=ttol)
    else:
        tris, marks, n_unique = unique_triangulated_facets(faces, boundaries, pts_global,
                                                           lims=lims if periodic else None)
    # vertices no triangle uses (their faces degenerated): TetGen jettisons them and renumbers
    # the rest, which breaks every PLC index -> mesh node map downstream (fold flips)
    used = np.zeros(len(pts_global), bool)
    used[np.array(tris, dtype=int).ravel()] = True
    if not used.all():
        new_id = np.cumsum(used) - 1
        pts_global = [p for p, u in zip(pts_global, used) if u]
        atts = [a for a, u in zip(atts, used) if u]
        tris = [tuple(int(new_id[v]) for v in t) for t in tris]
        print(f'Dropped {int((~used).sum())} PLC vertices no facet uses')

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
            _loc = round(_i, _DEC)
            _string+=f'{_loc:.{_DEC}f} '
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

    # every interface is collected once per adjacent cell, and merged/rounded
    # vertices leave polygons slightly non-planar or non-convex: write each facet
    # once, as triangles (duplicates and self-overlapping polygons crash TetGen)

    poly_string+= '\n# Facets, boundary marker \n'
    poly_string+= str(len(tris))+' 1 \n'
    poly_string+= f'# {len(faces)} polygons, {n_unique} distinct, written as triangles \n'

    lenprint = len(tris)
    for i in track(range(lenprint),  description= "{:<25}".format('Writing .poly file : facet data')):
        a, b, c = tris[i]
        # <#polygons> <#holes> <boundary marker>, then <#corners> <corners>
        poly_string+= f'1 0 {marks[i]}\n3 {a+1} {b+1} {c+1}\n'

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
    return collapse

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
            rx = round(seed_data[0], _DEC)
            ry = round(seed_data[1], _DEC)
            rz = round(seed_data[2], _DEC)
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
                                    pts_global.append(tuple([round(pt[0], _DEC),
                                                             round(pt[1], _DEC),
                                                             round(pt[2], _DEC)]))
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
                                pts_global.append(tuple([round(pt[0], _DEC),
                                                         round(pt[1], _DEC),
                                                         round(pt[2], _DEC)]))
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
        regions.append([total_regions, round(np.mean(_x), _DEC) , round(np.mean(_y), _DEC), round(np.mean(_z), _DEC), c_cell_num, -1.])

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
        _vv = {}                            # global vertex -> local index
        for _j, face in enumerate(facets[f]['fcts']):
            _vts = []
            for vt in face:
                if vt not in _vv:
                    _vv[vt] = len(_tot_pts)
                    _tot_pts.append(pts_global[vt])
                _vts.append(_vv[vt])

            _v['faces'].append({'vertices':_vts,
                                'adjacent_cell':_j})

        _v['vertices']= _tot_pts

        _voro.append(_v)

    # region points come from _phase_regions; node attributes are not used
    sub_regions = []
    new_atts = np.zeros((len(pts_global)))

    return pts_global, new_atts, faces, sub_regions, boundaries, vols, seeds, facets, _voro
