'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

--------------------------------------------------------------------------------
Ovito
--------------------------------------------------------------------------------


================================================================================
Ovito Writer
================================================================================
    Writes OVITO compatible input to visualize relation between input seeds and
    the resulting tessellations

INPUTS:
    filename    -
    voro        -
    limits      -

OUTPUTS:
    prints filename

'''
import array

def ovito_writer(filename, voro, lims):
    # limits
    xlo = lims[0][0]
    xhi = lims[0][1]
    ylo = lims[1][0]
    yhi = lims[1][1]
    zlo = lims[2][0]
    zhi = lims[2][1]

    ''' cell global dictionary
        [cell_num-vertices_index] = {'atom_number':int,
                                     'seed':cell_num,
                                     'seed_loc':[x,y,z],
                                     'vert_loc':pt,
                                     'phase':phase,
                                     'bonds':[]}
    '''
    pts_global = []
    pts_reduced = {}
    cell_global = {}

    n_cells = len(voro)
    n_atoms = 0
    for cell_num, cell_data in enumerate(voro):
        if cell_data['volume']== -1:
            continue
        else:
            # voro++ verticies for this cell
            pts_local = cell_data['vertices']
            phase = cell_data['original'][3]
            for i, pt in enumerate(pts_local):
                c_v = str(cell_num)+'-'+str(i)
                cell_global[c_v] = {'atom_number':n_atoms,
                                    'phase':phase,
                                    'seed':cell_num,
                                    'vert_loc':pt,
                                    'bonds':[]}
                n_atoms+=1

            for face_data in cell_data['faces']:
                # check if not empty from triangle
                if len(face_data) > 0:
                    face_verts = face_data['vertices']
                    for i, bond_i in enumerate(face_verts):
                        c_v = str(cell_num)+'-'+str(bond_i)
                        cell_global[c_v]['bonds'].append(str(cell_num)+'-'+str(face_verts[i-1]) )

    n_bonds = 0
    for _cell in cell_global:
        for pair in cell_global[_cell]['bonds']:
            if pair in cell_global:
                n_bonds += 1

    del voro

    seq = open(filename, 'w+')
    # build string to write
    seq.write('Generated for VDBSCAN \n\n')
    seq.write(f'{len(cell_global)} atoms \n')
    seq.write(f'{n_bonds} bonds \n\n')
    seq.write(f'{n_cells} atom types \n')
    seq.write(f'{n_cells} bond types \n\n')

    seq.write(f'{xlo} {xhi} xlo xhi \n')
    seq.write(f'{ylo} {yhi} ylo yhi \n')
    seq.write(f'{zlo} {zhi} zlo zhi \n')

    seq.write(f'Atoms #molecular \n\n')

    atom_numbers = {}

    # Reduced will just use the first
    for i, _cell in enumerate(cell_global):
        cell_data = cell_global[_cell]
        xyz = cell_data['vert_loc']
        seed = cell_data['seed']
        phase = cell_data['phase']

        # base 0 to base 1 indexing
        atom_numbers[_cell] = cell_data['atom_number']+1

        _string = str(i+1)+' '+str(seed+1)+' '
        for j in xyz:
            _string = _string + str(j) + ' '
        _string = _string+' '+str(int(phase))+' 0 0\n'
        seq.write(f'{_string}')

    seq.write(f'\nBonds \n\n')

    n_bonds = 0
    for i, _cell in enumerate(cell_global):
        cell_data = cell_global[_cell]
        bond_type = cell_data['seed'] + 1

        _bond_i = atom_numbers[_cell]
        for pair in cell_data['bonds']:
            if pair in atom_numbers:
                n_bonds += 1
                _bond_j = atom_numbers[pair]
                seq.write(f'{n_bonds} {bond_type} {_bond_i} {_bond_j}\n')
    seq.write(f'\n')
    seq.close()
