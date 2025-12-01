from mdvoro.mesh.compute_voro import compute_voro
from pothos.align import align

pa = align('small_test.dump')

pa.ncpus=1
pa.series=False

pa.align_type = 'P2'
pa.length_coeff = 2

pa.legendre_cutoff = 0.90
pa.eps = 1.1
pa.lam = 0.95

pa.method = 'BallTree'

pa.leaf_size = 13
pa.min_pts = 2

pa.atom_types = 1
pa.just_atoms = True

pa.min_length = 2
pa.second_filt = True
pa.window = 'parzen'

pa.crystalmin = 25
pa.coloring = 'herman'

pa.find()

# Define seed points (x, y, z, [phase])
# Define simulation box limits
seeds, lims = pa.voro_dump()

# Compute Voronoi tessellation
voro = compute_voro(seeds, lims,
                    dispersion=1.0,
                    periodic=[False, False, False],
                    method='voro++')

from mdvoro.mesh.tetgen import tetgen

# Initialize tetgen mesher
mesh = tetgen()

# Generate mesh from Voronoi tessellation
voro_out = mesh.mesh(voro, lims,
                     method='tetgen',
                     periodic=[False, False, False],
                     tetcmds='pkYAV')

# Write Abaqus input file
mesh.write('output.inp', method='abaqus')

from mdvoro.utils.ovito_writer import ovito_writer

# Write OVITO-compatible dump file
ovito_writer('output.dump', voro, lims)