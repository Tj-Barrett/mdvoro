"""
Tests for _abaqus_writer and _poly_writer in mdvoro.mesh.tetgen

Covers:
  - ** Model name comment prefix in .inp header
  - Reference point midpoint calculation for off-origin boxes
  - No abs() applied to negative coordinates in .poly output
"""
import os
import tempfile
import pytest
import numpy as np
from mdvoro.mesh.tetgen import _abaqus_writer, _poly_writer


def _minimal_mesh_args():
    """Minimal valid mesh_args for _abaqus_writer."""
    # 4 nodes: x, y, z, boundary_flag
    points = np.array([
        [0.0,  0.0,  0.0,  0.0],
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
    ])
    # 1 C3D4 element
    elements = np.array([[1, 2, 3, 4]])
    element_attributes = np.array([1.0])
    return (points, elements, element_attributes)


def _minimal_seeds():
    return {0: {'seed': 1, 'seed_loc': [0.5, 0.5, 0.5], 'phase': 1,
                'face_verts': [], 'face_elems': [], 'neighs': [], 'volume': 1.0}}


class TestAbaqusWriterHeader:
    def test_model_name_has_comment_prefix(self):
        """Bug fix: 'Model name: tess' was missing '**' prefix."""
        mesh_args = _minimal_mesh_args()
        seeds = _minimal_seeds()
        lims = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _abaqus_writer(fname, mesh_args, seeds, lims, None)
            with open(fname) as f:
                content = f.read()
            assert 'Model name: tess' in content
            # Must be a comment line — every occurrence must start with **
            for line in content.splitlines():
                if 'Model name: tess' in line:
                    assert line.strip().startswith('**'), (
                        f"'Model name' line is not a comment: {line!r}")
        finally:
            os.unlink(fname)

    def test_part_name_present(self):
        mesh_args = _minimal_mesh_args()
        seeds = _minimal_seeds()
        lims = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _abaqus_writer(fname, mesh_args, seeds, lims, None)
            with open(fname) as f:
                content = f.read()
            assert '*Part, name=tess' in content
            assert '*END PART' in content
        finally:
            os.unlink(fname)


class TestAbaqusWriterReferencePoints:
    def test_midpoint_on_origin_box(self):
        """For [0,1]^3, midpoints should be 0.5."""
        mesh_args = _minimal_mesh_args()
        seeds = _minimal_seeds()
        lims = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _abaqus_writer(fname, mesh_args, seeds, lims, None)
            with open(fname) as f:
                content = f.read()
            # RP-X node: x=xhi=1.0, y=ymid=0.5, z=zmid=0.5
            assert '1, 1.0, 0.5, 0.5' in content
        finally:
            os.unlink(fname)

    def test_midpoint_on_off_origin_box(self):
        """Bug fix: was (xhi-xlo)/2 instead of xlo+(xhi-xlo)/2."""
        mesh_args = _minimal_mesh_args()
        seeds = _minimal_seeds()
        # box from -5 to 5 — midpoint is 0.0, not 5.0
        lims = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _abaqus_writer(fname, mesh_args, seeds, lims, None)
            with open(fname) as f:
                content = f.read()
            # RP-X node: x=xhi=5.0, y=ymid=0.0, z=zmid=0.0
            assert '1, 5.0, 0.0, 0.0' in content
            # Regression: old bug would give '1, 5.0, 5.0, 5.0'
            assert '1, 5.0, 5.0, 5.0' not in content
        finally:
            os.unlink(fname)

    def test_midpoint_asymmetric_box(self):
        """Midpoint of [2, 8] is 5.0, not 3.0."""
        mesh_args = _minimal_mesh_args()
        seeds = _minimal_seeds()
        lims = [[2.0, 8.0], [2.0, 8.0], [2.0, 8.0]]
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _abaqus_writer(fname, mesh_args, seeds, lims, None)
            with open(fname) as f:
                content = f.read()
            # RP-X: x=xhi=8.0, y=ymid=5.0, z=zmid=5.0
            assert '1, 8.0, 5.0, 5.0' in content
        finally:
            os.unlink(fname)


class TestPolyWriter:
    def _make_poly_args(self, pts):
        faces = [[0, 1, 2]]
        boundaries = [1]
        regions = [[1, 0.3, 0.3, 0.3, 1, -1.0]]
        atts = [0] * len(pts)
        facets = {0: {'count': 1, 'fcts': [faces[0]], 'adj': [0]}}
        return pts, atts, faces, regions, boundaries, facets

    def test_negative_coords_preserved(self):
        """Bug fix: abs() was applied to all coords, collapsing negatives."""
        pts = [(-1.0, -2.0, -3.0), (1.0, 2.0, 3.0), (0.0, 0.0, 1.0)]
        args = self._make_poly_args(pts)
        with tempfile.TemporaryDirectory() as tmpdir:
            meshname = os.path.join(tmpdir, 'temp')
            _poly_writer(meshname, *args)
            with open(meshname + '.poly') as f:
                content = f.read()
        assert '-1.0000' in content
        assert '-2.0000' in content
        assert '-3.0000' in content

    def test_positive_coords_unchanged(self):
        pts = [(1.5, 2.5, 3.5), (4.0, 5.0, 6.0), (0.1, 0.2, 0.3)]
        args = self._make_poly_args(pts)
        with tempfile.TemporaryDirectory() as tmpdir:
            meshname = os.path.join(tmpdir, 'temp')
            _poly_writer(meshname, *args)
            with open(meshname + '.poly') as f:
                content = f.read()
        assert '1.5000' in content
        assert '4.0000' in content

    def test_node_count_in_header(self):
        pts = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        args = self._make_poly_args(pts)
        with tempfile.TemporaryDirectory() as tmpdir:
            meshname = os.path.join(tmpdir, 'temp')
            _poly_writer(meshname, *args)
            with open(meshname + '.poly') as f:
                lines = f.readlines()
        # second non-comment line should be "3 3 1 1"
        data_lines = [l for l in lines if not l.startswith('#')]
        assert data_lines[0].strip().startswith('3')
