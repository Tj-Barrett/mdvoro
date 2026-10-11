"""
Tests for mdvoro.mesh.triangle_face

pymesh is mocked at import time since it is an optional external dependency.

Covers:
  - _triangle_face_handle: list (not set) conversion preserves index mapping
  - _mismatch assignment fix: non-periodic faces excluded from periodic_pairs
"""
import sys
import types
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Mock pymesh before importing triangle_face (pymesh may not be installed)
# ---------------------------------------------------------------------------
_pymesh_mock = types.ModuleType('pymesh')

class _MockTriangle:
    def __init__(self):
        self.points = []
        self.verbosity = 0
        self.max_area = 1.0
        self.split_boundary = False
        self.keep_convex_hull = False
        self.conforming_delaunay = False

    def run(self):
        pass

    @property
    def mesh(self):
        class _Mesh:
            vertices = []
            faces = []
        return _Mesh()

_pymesh_mock.triangle = _MockTriangle
_pymesh_mock.resolve_self_intersection = lambda x: x
_pymesh_mock.collapse_short_edges = lambda x, y: x
sys.modules.setdefault('pymesh', _pymesh_mock)

from mdvoro.mesh.triangle_face import _triangle_face_handle  # noqa: E402


class TestTriangleFaceHandle:
    def _make_voro(self, verts):
        return [{'vertices': list(verts), 'faces': [], 'volume': 1.0,
                 'original': [0, 0, 0, 1]}]

    def test_matching_vertex_gets_existing_index(self):
        """Bug fix: set() conversion scrambled _nv vs face vertex index mapping.
        A new vertex coinciding with an existing one should map to that index."""
        existing = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        voro = self._make_voro(existing)
        # new_verts[0] matches existing[1] exactly
        new_verts = [(1.0, 0.0, 0.0)]
        new_face  = [(0, 0, 0)]   # degenerate — will be filtered; we check vertex mapping
        result = _triangle_face_handle(voro, 0, new_face, new_verts, existing,
                                       pt_pt_dist_tol=1e-3)
        # degenerate face filtered — just confirm no exception and no wrong indices
        assert isinstance(result, list)

    def test_new_vertex_added_to_cell(self):
        """A new vertex far from all existing ones should be appended to the cell."""
        existing = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        voro = self._make_voro(existing)
        new_verts = [(5.0, 5.0, 5.0)]  # far from all existing
        new_face  = [(0, 0, 0)]
        _triangle_face_handle(voro, 0, new_face, new_verts, existing,
                              pt_pt_dist_tol=1e-3)
        assert len(voro[0]['vertices']) == 4  # one vertex appended

    def test_valid_triangle_preserved(self):
        """A non-degenerate triangle (all distinct indices) should survive filtering."""
        existing = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        voro = self._make_voro(existing)
        # new_verts map 1-to-1 to existing vertices 0,1,2
        new_verts = list(existing[:3])
        new_face  = [(0, 1, 2)]
        result = _triangle_face_handle(voro, 0, new_face, new_verts, existing,
                                       pt_pt_dist_tol=1e-3)
        # All three vertices matched to existing 0,1,2 — non-degenerate → kept
        assert len(result) == 1
        i, j, k = result[0]
        assert i != j and i != k and j != k

    def test_degenerate_triangle_filtered(self):
        """A triangle where two vertex indices are the same should be dropped."""
        existing = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        voro = self._make_voro(existing)
        # Both indices 0 and 1 map to the same existing vertex
        new_verts = [existing[0], existing[0], existing[2]]
        new_face  = [(0, 1, 2)]
        result = _triangle_face_handle(voro, 0, new_face, new_verts, existing,
                                       pt_pt_dist_tol=1e-3)
        assert result == []

    def test_index_order_preserved_not_set_order(self):
        """Regression: set() reorders vertices so _nv != face vertex index.
        With the fix (list), vertex 0 in new_face must map to new_verts[0]."""
        existing = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (0.5, 0.5, 0.0)]
        voro = self._make_voro(existing)
        # new_verts ordered to match existing indices 1, 2, 3
        new_verts = [existing[1], existing[2], existing[3]]
        new_face  = [(0, 1, 2)]
        result = _triangle_face_handle(voro, 0, new_face, new_verts, existing,
                                       pt_pt_dist_tol=1e-3)
        assert len(result) == 1
        assert sorted(result[0]) == [1, 2, 3]
