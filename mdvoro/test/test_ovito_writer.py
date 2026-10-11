"""
Tests for mdvoro.utils.ovito_writer

Covers:
  - Header bond count matches the number of bonds actually written
  - Atom count in header matches written atoms
  - Box bounds written correctly
"""
import pytest
import tempfile
import os
from mdvoro.utils.ovito_writer import ovito_writer


def _make_voro(n_cells=2):
    """Build a minimal voro list with shared face bonds."""
    voro = []
    for i in range(n_cells):
        voro.append({
            'volume': 1.0,
            'vertices': [
                [float(i),       0.0, 0.0],
                [float(i) + 1.0, 0.0, 0.0],
                [float(i),       1.0, 0.0],
                [float(i),       0.0, 1.0],
            ],
            'adjacency': [],
            'faces': [
                {'adjacent_cell': -1, 'vertices': [0, 1, 2]},
                {'adjacent_cell': -1, 'vertices': [0, 1, 3]},
            ],
            'original': [float(i) + 0.5, 0.5, 0.5, i + 1],
        })
    return voro


def _parse_lammps_dump(content):
    """Extract header counts and actual section lengths from a LAMMPS dump."""
    lines = content.splitlines()
    n_atoms_header = None
    n_bonds_header = None
    in_atoms = False
    in_bonds = False
    atom_lines = 0
    bond_lines = 0
    for line in lines:
        s = line.strip()
        if 'atoms' in s and 'types' not in s and n_atoms_header is None:
            try:
                n_atoms_header = int(s.split()[0])
            except ValueError:
                pass
        if 'bonds' in s and 'types' not in s and n_bonds_header is None:
            try:
                n_bonds_header = int(s.split()[0])
            except ValueError:
                pass
        if s == 'Atoms #molecular' or s == 'Atoms':
            in_atoms = True
            in_bonds = False
            continue
        if s.startswith('Bonds'):
            in_bonds = True
            in_atoms = False
            continue
        if in_atoms and s and not s.startswith('#'):
            atom_lines += 1
        if in_bonds and s and not s.startswith('#'):
            bond_lines += 1
    return n_atoms_header, n_bonds_header, atom_lines, bond_lines


class TestOvitoWriter:
    def test_bond_header_matches_written_bonds(self):
        """Bug fix: header used double-counted n_bonds; actual written may differ."""
        voro = _make_voro(n_cells=3)
        lims = [[0.0, 3.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False, mode='w') as f:
            fname = f.name
        try:
            ovito_writer(fname, voro, lims)
            with open(fname) as f:
                content = f.read()
            _, n_bonds_header, _, bond_lines = _parse_lammps_dump(content)
            assert n_bonds_header == bond_lines, (
                f"Header declares {n_bonds_header} bonds but {bond_lines} were written")
        finally:
            os.unlink(fname)

    def test_atom_header_matches_written_atoms(self):
        voro = _make_voro(n_cells=2)
        lims = [[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False, mode='w') as f:
            fname = f.name
        try:
            ovito_writer(fname, voro, lims)
            with open(fname) as f:
                content = f.read()
            n_atoms_header, _, atom_lines, _ = _parse_lammps_dump(content)
            assert n_atoms_header == atom_lines, (
                f"Header declares {n_atoms_header} atoms but {atom_lines} were written")
        finally:
            os.unlink(fname)

    def test_box_bounds_written(self):
        voro = _make_voro(n_cells=1)
        lims = [[-5.0, 5.0], [-3.0, 3.0], [-1.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False, mode='w') as f:
            fname = f.name
        try:
            ovito_writer(fname, voro, lims)
            with open(fname) as f:
                content = f.read()
            assert '-5.0 5.0 xlo xhi' in content
            assert '-3.0 3.0 ylo yhi' in content
            assert '-1.0 1.0 zlo zhi' in content
        finally:
            os.unlink(fname)

    def test_single_cell_no_crash(self):
        voro = _make_voro(n_cells=1)
        lims = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False, mode='w') as f:
            fname = f.name
        try:
            ovito_writer(fname, voro, lims)
            assert os.path.getsize(fname) > 0
        finally:
            os.unlink(fname)
