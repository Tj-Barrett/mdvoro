# Periodic Boundary Condition writer for CalculiX / PrePoMax
#
# Reads a NEPER-generated INP mesh file, identifies surface nodes,
# matches periodic node pairs across opposite faces, and writes
# *EQUATION constraints directly into a new INP file ready for
# CalculiX (also readable by PrePoMax).
#
# No Abaqus dependency — pure Python 3, stdlib + numpy only.
#
# -------------------------------------------------------------------
# BUGS FOUND IN THE ORIGINAL abaqus_utils.periodic_constraints()
# -------------------------------------------------------------------
# This script fixes three systematic errors present in the Abaqus
# version (which only affects shear load cases; normal-only loads
# were unaffected):
#
# 1. WRONG EDGE PAIRING
#    The Abaqus version paired edges that share the same hi/lo face
#    value (e.g. zone7(+x,+y) with zone11(-x,+y), both on y=yhi).
#    The correct periodic partner of zone7(+x,+y) is zone12(-x,-y)
#    because the lattice translation vector for that edge is (Lx,Ly,0).
#
# 2. MISSING SIGN ON EDGE/CORNER RP COEFFICIENTS
#    When the periodic translation vector has a -1 component (e.g.
#    zone8(+x,-y) pairs with zone11(-x,+y) correctly, diff=(Lx,-Ly,0)),
#    the RP-Y term needs a +1 coefficient (not -1) in the equation:
#      u_8 - u_11 - u_RP_X + u_RP_Y = 0
#    The Abaqus version always used -1 for every RP term, which is
#    wrong for the "minus" direction contributions.
#
# 3. WRONG CORNER PAIRING
#    The Abaqus version paired, e.g., zone19(+x+y+z) with zone25(-x-y+z)
#    (differ in x,y only). The correct partner is zone26(-x-y-z)
#    (differ in x,y,z) so that all three RP terms are consistent.
# -------------------------------------------------------------------

import numpy as np
import re
import os

# ===================================================================
# Configuration
# ===================================================================

INPUT_INP    = 'mesh.inp'        # NEPER mesh file to read
OUTPUT_INP   = 'mesh_pbc.inp'    # output file with PBC equations added
TOL          = 1e-5              # node coordinate matching tolerance

DISPLACEMENT_STRAIN = 0.01       # applied macroscopic strain for each load case
MATERIAL_NAME       = 'PVA'      # must already exist in the INP material block
LOAD_CASES          = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ']

# ===================================================================
# INP Parser
# ===================================================================

def parse_inp_nodes(inp_path):
    """
    Return {node_id (int): np.array([x,y,z])} from *Node block.
    """
    nodes = {}
    in_node = False
    with open(inp_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith('*NODE'):
                in_node = True
                continue
            if stripped.startswith('*') and in_node:
                in_node = False
            if in_node and stripped and not stripped.startswith('**'):
                parts = stripped.split(',')
                if len(parts) >= 4:
                    nid = int(parts[0].strip())
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = np.array([x, y, z])
    return nodes


def parse_inp_raw(inp_path):
    """
    Return the full file as a list of lines for later rewriting.
    Also return the max node ID so we know where to insert new nodes.
    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    return lines


def get_box_bounds(nodes):
    coords = np.array(list(nodes.values()))
    return (coords[:,0].min(), coords[:,0].max(),
            coords[:,1].min(), coords[:,1].max(),
            coords[:,2].min(), coords[:,2].max())


# ===================================================================
# 26-Zone Surface Node Sorter
# ===================================================================

def sort_surface_nodes(nodes, xlo, xhi, ylo, yhi, zlo, zhi, tol):
    """
    Sort nodes into the 26 periodic zones (6 faces + 12 edges + 8 corners).
    Only nodes on the surface of the box (within tol of a boundary plane)
    are classified.

    Zone naming (same convention as abaqus_utils.py):
        Faces  : 1(+x) 2(-x) 3(+y) 4(-y) 5(+z) 6(-z)
        Edges  : 7(+x+y) 8(+x-y) 9(+x+z) 10(+x-z)
                 11(-x+y) 12(-x-y) 13(-x+z) 14(-x-z)
                 15(+y+z) 16(+y-z) 17(-y+z) 18(-y-z)
        Corners: 19(+x+y+z) 20(+x+y-z) 21(+x-y+z) 22(+x-y-z)
                 23(-x+y+z) 24(-x+y-z) 25(-x-y+z) 26(-x-y-z)

    Returns dict zone_id (1-26) → list of node IDs.
    """
    zones = {i: [] for i in range(1, 27)}

    for nid, c in nodes.items():
        on_xhi = c[0] >= xhi - tol
        on_xlo = c[0] <= xlo + tol
        on_yhi = c[1] >= yhi - tol
        on_ylo = c[1] <= ylo + tol
        on_zhi = c[2] >= zhi - tol
        on_zlo = c[2] <= zlo + tol

        n_faces = sum([on_xhi, on_xlo, on_yhi, on_ylo, on_zhi, on_zlo])
        if n_faces == 0:
            continue   # interior node

        # Corners (on 3 faces)
        if n_faces >= 3:
            if   on_xhi and on_yhi and on_zhi: zones[19].append(nid)
            elif on_xhi and on_yhi and on_zlo: zones[20].append(nid)
            elif on_xhi and on_ylo and on_zhi: zones[21].append(nid)
            elif on_xhi and on_ylo and on_zlo: zones[22].append(nid)
            elif on_xlo and on_yhi and on_zhi: zones[23].append(nid)
            elif on_xlo and on_yhi and on_zlo: zones[24].append(nid)
            elif on_xlo and on_ylo and on_zhi: zones[25].append(nid)
            elif on_xlo and on_ylo and on_zlo: zones[26].append(nid)
        # Edges (on exactly 2 faces)
        elif n_faces == 2:
            if   on_xhi and on_yhi: zones[7].append(nid)
            elif on_xhi and on_ylo: zones[8].append(nid)
            elif on_xhi and on_zhi: zones[9].append(nid)
            elif on_xhi and on_zlo: zones[10].append(nid)
            elif on_xlo and on_yhi: zones[11].append(nid)
            elif on_xlo and on_ylo: zones[12].append(nid)
            elif on_xlo and on_zhi: zones[13].append(nid)
            elif on_xlo and on_zlo: zones[14].append(nid)
            elif on_yhi and on_zhi: zones[15].append(nid)
            elif on_yhi and on_zlo: zones[16].append(nid)
            elif on_ylo and on_zhi: zones[17].append(nid)
            elif on_ylo and on_zlo: zones[18].append(nid)
        # Faces (on exactly 1 face)
        else:
            if   on_xhi: zones[1].append(nid)
            elif on_xlo: zones[2].append(nid)
            elif on_yhi: zones[3].append(nid)
            elif on_ylo: zones[4].append(nid)
            elif on_zhi: zones[5].append(nid)
            elif on_zlo: zones[6].append(nid)

    return zones


# ===================================================================
# Node Pair Matcher
# ===================================================================

def _match_pairs(nodes, zone_a, zone_b, free_axes):
    """
    Match nodes in zone_a with their periodic partners in zone_b.
    Matching is done on the coordinates of 'free_axes' (tuple of 0,1,2).
    Returns list of (nid_a, nid_b) pairs.
    """
    unmatched_b = list(zone_b)
    pairs = []
    for a in zone_a:
        ca = nodes[a]
        for b in unmatched_b:
            cb = nodes[b]
            if all(abs(ca[ax] - cb[ax]) <= TOL for ax in free_axes):
                pairs.append((a, b))
                unmatched_b.remove(b)
                break
    if unmatched_b:
        print('WARNING: {} unmatched nodes remain in zone pair'.format(len(unmatched_b)))
    return pairs


def build_periodic_pairs(nodes, zones):
    """
    Build all (master, slave, {rp: coeff}) triples for the 26-zone scheme.

    Convention: master node is on the '+' face/edge/corner.
                slave  node is its periodic image on the '-' side.

    Each entry: (master_nid, slave_nid, rp_coeffs)
    where rp_coeffs = {'X': ±1, 'Y': ±1, 'Z': ±1} (only included axes).

    The constraint written is:
        u_master - u_slave - Σ(coeff_k * u_RP_k) = 0   for each DOF

    Coefficient signs follow from the periodic translation vector
    (positive component → -1 on the corresponding RP term,
     negative component → +1 on the corresponding RP term).

    This fixes the sign bug present in abaqus_utils.periodic_constraints().
    """
    pairs = []

    # --- Faces (translation vectors along one axis only) ---
    # zone1(+x) ↔ zone2(-x), diff vector = (+Lx,0,0) → RP_X coeff -1
    for a, b in _match_pairs(nodes, zones[1], zones[2], free_axes=(1, 2)):
        pairs.append((a, b, {'X': -1}))

    # zone3(+y) ↔ zone4(-y), diff = (0,+Ly,0) → RP_Y coeff -1
    for a, b in _match_pairs(nodes, zones[3], zones[4], free_axes=(0, 2)):
        pairs.append((a, b, {'Y': -1}))

    # zone5(+z) ↔ zone6(-z), diff = (0,0,+Lz) → RP_Z coeff -1
    for a, b in _match_pairs(nodes, zones[5], zones[6], free_axes=(0, 1)):
        pairs.append((a, b, {'Z': -1}))

    # --- Edges (translation vectors along two axes) ---
    # Pair each '+' edge with the diagonally opposite '-' edge so that the
    # two axes in the translation vector both contribute.
    #
    # zone7(+x+y) ↔ zone12(-x-y), diff=(+Lx,+Ly,0) → X:-1 Y:-1
    for a, b in _match_pairs(nodes, zones[7], zones[12], free_axes=(2,)):
        pairs.append((a, b, {'X': -1, 'Y': -1}))

    # zone8(+x-y) ↔ zone11(-x+y), diff=(+Lx,-Ly,0) → X:-1 Y:+1
    for a, b in _match_pairs(nodes, zones[8], zones[11], free_axes=(2,)):
        pairs.append((a, b, {'X': -1, 'Y': +1}))

    # zone9(+x+z) ↔ zone14(-x-z), diff=(+Lx,0,+Lz) → X:-1 Z:-1
    for a, b in _match_pairs(nodes, zones[9], zones[14], free_axes=(1,)):
        pairs.append((a, b, {'X': -1, 'Z': -1}))

    # zone10(+x-z) ↔ zone13(-x+z), diff=(+Lx,0,-Lz) → X:-1 Z:+1
    for a, b in _match_pairs(nodes, zones[10], zones[13], free_axes=(1,)):
        pairs.append((a, b, {'X': -1, 'Z': +1}))

    # zone15(+y+z) ↔ zone18(-y-z), diff=(0,+Ly,+Lz) → Y:-1 Z:-1
    for a, b in _match_pairs(nodes, zones[15], zones[18], free_axes=(0,)):
        pairs.append((a, b, {'Y': -1, 'Z': -1}))

    # zone16(+y-z) ↔ zone17(-y+z), diff=(0,+Ly,-Lz) → Y:-1 Z:+1
    for a, b in _match_pairs(nodes, zones[16], zones[17], free_axes=(0,)):
        pairs.append((a, b, {'Y': -1, 'Z': +1}))

    # --- Corners (translation vectors along all three axes) ---
    # Each corner on the '+x' side is paired with its true periodic image.
    #
    # zone19(+x+y+z) ↔ zone26(-x-y-z), diff=(+Lx,+Ly,+Lz) → X:-1 Y:-1 Z:-1
    for a, b in _match_pairs(nodes, zones[19], zones[26], free_axes=()):
        pairs.append((a, b, {'X': -1, 'Y': -1, 'Z': -1}))

    # zone20(+x+y-z) ↔ zone25(-x-y+z), diff=(+Lx,+Ly,-Lz) → X:-1 Y:-1 Z:+1
    for a, b in _match_pairs(nodes, zones[20], zones[25], free_axes=()):
        pairs.append((a, b, {'X': -1, 'Y': -1, 'Z': +1}))

    # zone21(+x-y+z) ↔ zone24(-x+y-z), diff=(+Lx,-Ly,+Lz) → X:-1 Y:+1 Z:-1
    for a, b in _match_pairs(nodes, zones[21], zones[24], free_axes=()):
        pairs.append((a, b, {'X': -1, 'Y': +1, 'Z': -1}))

    # zone22(+x-y-z) ↔ zone23(-x+y+z), diff=(+Lx,-Ly,-Lz) → X:-1 Y:+1 Z:+1
    for a, b in _match_pairs(nodes, zones[22], zones[23], free_axes=()):
        pairs.append((a, b, {'X': -1, 'Y': +1, 'Z': +1}))

    return pairs


# ===================================================================
# INP Writer
# ===================================================================

def write_periodic_inp(input_inp, output_inp, tol=1e-5,
                        displacement_strain=0.01,
                        load_cases=None,
                        material_name='PVA'):
    """
    Read a NEPER INP file, append periodic equation constraints and
    six homogenisation load steps, write to output_inp.

    Reference point nodes are added immediately after the *Node block
    with IDs max_node+1, max_node+2, max_node+3 for RP-X, RP-Y, RP-Z.
    Their coordinates are placed slightly outside the box so they are
    never accidentally picked up as mesh nodes.

    Parameters
    ----------
    input_inp  : path to the NEPER mesh INP
    output_inp : path for the output CalculiX INP
    tol        : node coordinate matching tolerance
    displacement_strain : macroscopic strain applied per load case
    load_cases : list subset of ['XX','YY','ZZ','XY','YZ','XZ']
    material_name : material already defined in the INP
    """
    if load_cases is None:
        load_cases = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ']

    print('Parsing nodes from {}'.format(input_inp))
    nodes = parse_inp_nodes(input_inp)
    raw_lines = parse_inp_raw(input_inp)

    xlo, xhi, ylo, yhi, zlo, zhi = get_box_bounds(nodes)
    xlen = xhi - xlo
    ylen = yhi - ylo
    zlen = zhi - zlo
    print('Box: X[{:.4f},{:.4f}]  Y[{:.4f},{:.4f}]  Z[{:.4f},{:.4f}]'.format(
        xlo, xhi, ylo, yhi, zlo, zhi))

    # --- Assign RP node IDs ---
    max_nid = max(nodes.keys())
    rp_X = max_nid + 1
    rp_Y = max_nid + 2
    rp_Z = max_nid + 3
    # Place RPs outside the box (CalculiX ignores their position for equation constraints)
    rp_coords = {
        rp_X: np.array([xhi + xlen, (ylo+yhi)/2, (zlo+zhi)/2]),
        rp_Y: np.array([(xlo+xhi)/2, yhi + ylen, (zlo+zhi)/2]),
        rp_Z: np.array([(xlo+xhi)/2, (ylo+yhi)/2, zhi + zlen]),
    }
    rp_map = {'X': rp_X, 'Y': rp_Y, 'Z': rp_Z}

    # --- Zone sorting ---
    print('Sorting surface nodes into 26 zones...')
    zones = sort_surface_nodes(nodes, xlo, xhi, ylo, yhi, zlo, zhi, tol)
    for i in range(1, 27):
        if zones[i]:
            print('  Zone {:2d}: {:4d} nodes'.format(i, len(zones[i])))

    # --- Build periodic pairs ---
    print('Matching periodic node pairs...')
    all_nodes_for_matching = {**nodes, **rp_coords}
    pairs = build_periodic_pairs(nodes, zones)
    print('Total constraint pairs: {}'.format(len(pairs)))

    # --- Find insertion point for RP nodes (end of *Node block) ---
    node_end_line = 0
    in_node_block = False
    for idx, line in enumerate(raw_lines):
        stripped = line.strip().upper()
        if stripped.startswith('*NODE'):
            in_node_block = True
            continue
        if in_node_block and stripped.startswith('*'):
            node_end_line = idx
            in_node_block = False
            break

    if node_end_line == 0:
        # *Node was the last block — append at end of file
        node_end_line = len(raw_lines)

    # --- Build RP node lines ---
    rp_node_lines = ['** Reference point nodes for macroscopic loading\n']
    for rp_nid, rp_c in [(rp_X, rp_coords[rp_X]),
                          (rp_Y, rp_coords[rp_Y]),
                          (rp_Z, rp_coords[rp_Z])]:
        rp_node_lines.append('{:d}, {:.6f}, {:.6f}, {:.6f}\n'.format(
            rp_nid, rp_c[0], rp_c[1], rp_c[2]))

    # --- Build *EQUATION block ---
    eq_lines = ['\n** Periodic boundary condition equations\n']
    eq_count = 0
    for master, slave, rp_coeffs in pairs:
        for dof in [1, 2, 3]:
            terms = [(master, dof, 1.0), (slave, dof, -1.0)]
            for axis, coeff in rp_coeffs.items():
                terms.append((rp_map[axis], dof, float(coeff)))
            eq_lines.append('*EQUATION\n')
            eq_lines.append('{:d}\n'.format(len(terms)))
            eq_lines.append(', '.join(
                '{:d},{:d},{:.1f}'.format(nid, d, c)
                for nid, d, c in terms) + '\n')
            eq_count += 1

    print('Equations written: {}'.format(eq_count))

    # --- Build *NSET for RP nodes ---
    nset_lines = [
        '\n*NSET, NSET=RP-X\n{:d}\n'.format(rp_X),
        '*NSET, NSET=RP-Y\n{:d}\n'.format(rp_Y),
        '*NSET, NSET=RP-Z\n{:d}\n'.format(rp_Z),
    ]

    # --- Build load step blocks ---
    step_lines = []
    for lc in load_cases:
        step_lines.append('\n** Load case: {}\n'.format(lc))
        step_lines.append('*STEP\n')
        step_lines.append('*STATIC\n')

        if lc == 'XX':
            d = displacement_strain * xlen
            step_lines += _bc_lines('RP-X', u1=d,  u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Y', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Z', u1=0.0, u2=0.0, u3=0.0)

        elif lc == 'YY':
            d = displacement_strain * ylen
            step_lines += _bc_lines('RP-X', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Y', u1=0.0, u2=d,   u3=0.0)
            step_lines += _bc_lines('RP-Z', u1=0.0, u2=0.0, u3=0.0)

        elif lc == 'ZZ':
            d = displacement_strain * zlen
            step_lines += _bc_lines('RP-X', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Y', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Z', u1=0.0, u2=0.0, u3=d)

        elif lc == 'XY':
            # symmetric shear: u_RP_X = 0.5*gamma*Lx in y, u_RP_Y = 0.5*gamma*Ly in x
            d = 0.5 * displacement_strain * xlen
            e = 0.5 * displacement_strain * ylen
            step_lines += _bc_lines('RP-X', u1=0.0, u2=d,   u3=0.0)
            step_lines += _bc_lines('RP-Y', u1=e,   u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Z', u1=0.0, u2=0.0, u3=0.0)

        elif lc == 'YZ':
            d = 0.5 * displacement_strain * ylen
            e = 0.5 * displacement_strain * zlen
            step_lines += _bc_lines('RP-X', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Y', u1=0.0, u2=0.0, u3=d)
            step_lines += _bc_lines('RP-Z', u1=0.0, u2=e,   u3=0.0)

        elif lc == 'XZ':
            d = 0.5 * displacement_strain * xlen
            e = 0.5 * displacement_strain * zlen
            step_lines += _bc_lines('RP-X', u1=0.0, u2=0.0, u3=d)
            step_lines += _bc_lines('RP-Y', u1=0.0, u2=0.0, u3=0.0)
            step_lines += _bc_lines('RP-Z', u1=e,   u2=0.0, u3=0.0)

        step_lines.append('*NODE PRINT, NSET=RP-X\nU\n')
        step_lines.append('*NODE PRINT, NSET=RP-Y\nU\n')
        step_lines.append('*NODE PRINT, NSET=RP-Z\nU\n')
        step_lines.append('*EL PRINT, ELSET=EALL\nS,E\n')
        step_lines.append('*END STEP\n')

    # --- Assemble and write output ---
    out_lines = (raw_lines[:node_end_line] +
                 rp_node_lines +
                 raw_lines[node_end_line:] +
                 nset_lines +
                 eq_lines +
                 step_lines)

    with open(output_inp, 'w') as f:
        f.writelines(out_lines)

    print('Written: {}'.format(output_inp))
    print('RP-X node ID: {}  RP-Y: {}  RP-Z: {}'.format(rp_X, rp_Y, rp_Z))


def _bc_lines(nset, u1, u2, u3):
    """Format *BOUNDARY lines for a reference point nset (all 3 DOFs)."""
    return [
        '*BOUNDARY\n',
        '{},1,1,{:.6e}\n'.format(nset, u1),
        '{},2,2,{:.6e}\n'.format(nset, u2),
        '{},3,3,{:.6e}\n'.format(nset, u3),
    ]


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    write_periodic_inp(
        input_inp=INPUT_INP,
        output_inp=OUTPUT_INP,
        tol=TOL,
        displacement_strain=DISPLACEMENT_STRAIN,
        load_cases=LOAD_CASES,
        material_name=MATERIAL_NAME)
