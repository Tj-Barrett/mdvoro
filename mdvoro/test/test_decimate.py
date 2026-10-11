"""
Tests for mdvoro.mesh.decimate

Covers:
  - _decimate_faces: adj >= 0 fix (cell index 0 is valid)
  - _decimate_points: vertex remapping preserves faces
  - _merge_cells: correct phase grouping
"""
import pytest
import numpy as np
from mdvoro.mesh.decimate import _decimate_faces, _decimate_points, _merge_cells


def _make_cell(x, y, z, phase, adj_list):
    """Build a minimal voro cell dict."""
    verts = [
        [x,       y,       z      ],
        [x + 1.0, y,       z      ],
        [x,       y + 1.0, z      ],
        [x,       y,       z + 1.0],
    ]
    faces = [{'adjacent_cell': a, 'vertices': [0, 1, 2]} for a in adj_list]
    return {
        'volume': 1.0,
        'vertices': verts,
        'adjacency': [],
        'faces': faces,
        'original': [x, y, z, phase],
    }


class TestDecimateFaces:
    def test_cell_zero_neighbor_included(self):
        """Bug fix: adj > 0 excluded valid cell index 0.
        Cell 1 borders cell 0 (different phases) — that face must survive."""
        voro = [
            _make_cell(0.0, 0.0, 0.0, phase=1, adj_list=[ 1, -1]),  # cell 0
            _make_cell(5.0, 0.0, 0.0, phase=2, adj_list=[ 0, -1]),  # cell 1
        ]
        # large cutoff so nothing is decimated by distance
        result, vols = _decimate_faces(voro, grain_boundary=False, cell_cutoff=100.0)
        # Both cells should have faces to each other
        cell1_adjs = [f['adjacent_cell'] for f in result[1]['faces']]
        assert 0 in cell1_adjs, "Face to cell 0 was incorrectly dropped"

    def test_same_phase_close_cells_decimated(self):
        """Cells of the same phase that are close should be decimated (interior)."""
        voro = [
            _make_cell(0.0, 0.0, 0.0, phase=1, adj_list=[1, 2, -1]),
            _make_cell(0.5, 0.0, 0.0, phase=1, adj_list=[0, 2, -1]),
            _make_cell(1.0, 0.0, 0.0, phase=1, adj_list=[0, 1, -1]),
        ]
        result, vols = _decimate_faces(voro, grain_boundary=False, cell_cutoff=100.0)
        # All three are same phase and close — all become decimated (volume=-1)
        # or their cross-phase faces are empty
        surviving_faces = [f for cell in result for f in cell['faces']
                           if cell['volume'] != -1]
        # Only boundary faces (adj < 0) should survive
        for f in surviving_faces:
            assert f['adjacent_cell'] < 0

    def test_different_phase_face_preserved(self):
        """Faces between different phases must always be preserved."""
        voro = [
            _make_cell(0.0, 0.0, 0.0, phase=1, adj_list=[1, -1]),
            _make_cell(5.0, 0.0, 0.0, phase=2, adj_list=[0, -1]),
        ]
        result, _ = _decimate_faces(voro, grain_boundary=False, cell_cutoff=100.0)
        all_adjs = [f['adjacent_cell'] for cell in result
                    for f in cell['faces'] if cell['volume'] != -1]
        # There should be faces connecting the two phases
        assert 0 in all_adjs or 1 in all_adjs


class TestDecimatePoints:
    def test_unused_vertices_removed(self):
        """Vertices not referenced by any face should be dropped."""
        voro = [{
            'volume': 1.0,
            'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]],  # index 3 unused
            'adjacency': [],
            'faces': [{'adjacent_cell': -1, 'vertices': [0, 1, 2]}],
            'original': [0, 0, 0, 1],
        }]
        result = _decimate_points(voro)
        assert len(result[0]['vertices']) == 3

    def test_face_indices_remapped(self):
        """After removing unused vertices, face indices must be updated."""
        voro = [{
            'volume': 1.0,
            'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]],
            'adjacency': [],
            'faces': [{'adjacent_cell': -1, 'vertices': [0, 1, 2]}],
            'original': [0, 0, 0, 1],
        }]
        result = _decimate_points(voro)
        for vt in result[0]['faces'][0]['vertices']:
            assert vt < len(result[0]['vertices'])


class TestMergeCells:
    def test_phases_merged_correctly(self):
        """Cells of the same phase type should merge into one entry."""
        voro = [
            {
                'volume': 2.0,
                'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                'adjacency': [],
                'faces': [{'adjacent_cell': 1, 'vertices': [0, 1, 2]}],
                'original': [0.5, 0.5, 0.5, 1],
            },
            {
                'volume': 2.0,
                'vertices': [[2, 0, 0], [3, 0, 0], [2, 1, 0]],
                'adjacency': [],
                'faces': [{'adjacent_cell': 0, 'vertices': [0, 1, 2]}],
                'original': [2.5, 0.5, 0.5, 1],
            },
        ]
        vols = {1: 4.0}
        result, types, _ = _merge_cells(voro, vols)
        assert len(result) == 1
        assert 1 in types

    def test_two_phases_produce_two_merged_cells(self):
        voro = [
            {
                'volume': 1.0,
                'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                'adjacency': [],
                'faces': [{'adjacent_cell': -1, 'vertices': [0, 1, 2]}],
                'original': [0.5, 0.5, 0.5, 1],
            },
            {
                'volume': 1.0,
                'vertices': [[2, 0, 0], [3, 0, 0], [2, 1, 0]],
                'adjacency': [],
                'faces': [{'adjacent_cell': -1, 'vertices': [0, 1, 2]}],
                'original': [2.5, 0.5, 0.5, 2],
            },
        ]
        vols = {1: 1.0, 2: 1.0}
        result, types, _ = _merge_cells(voro, vols)
        assert len(result) == 2
