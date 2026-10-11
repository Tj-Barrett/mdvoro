"""
Tests for _abaqus_writer, _calculix_writer, and _poly_writer in mdvoro.mesh.tetgen

Covers:
  - ** Model name comment prefix in .inp header
  - Reference point midpoint calculation for off-origin boxes
  - No abs() applied to negative coordinates in .poly output
  - _calculix_writer: flat format, no Part/Instance, EALL elset, material block
"""
import os
import tempfile
import pytest
import numpy as np
from mdvoro.mesh.tetgen import _abaqus_writer, _calculix_writer, _poly_writer


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


class TestCalculixWriter:
    def setup_method(self):
        self.mesh_args = _minimal_mesh_args()
        self.seeds = _minimal_seeds()
        self.lims = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    def _write(self, lims=None, material_name='MATERIAL', supplemental_file=None):
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        _calculix_writer(fname, self.mesh_args, self.seeds,
                         lims or self.lims, supplemental_file, material_name)
        with open(fname) as f:
            content = f.read()
        os.unlink(fname)
        return content

    def test_no_part_or_assembly_keywords(self):
        """CalculiX does not support *Part/*Instance/*Assembly hierarchy."""
        content = self._write()
        assert '*Part' not in content
        assert '*END PART' not in content
        assert '*ASSEMBLY' not in content
        assert '*INSTANCE' not in content

    def test_node_block_present(self):
        content = self._write()
        assert '*Node\n' in content

    def test_element_block_c3d4(self):
        content = self._write()
        assert '*Element, type=C3D4' in content

    def test_nodes_written_correctly(self):
        """All 4 nodes from minimal mesh_args should appear."""
        content = self._write()
        assert '1, 0.0, 0.0, 0.0' in content
        assert '2, 1.0, 0.0, 0.0' in content
        assert '3, 0.0, 1.0, 0.0' in content
        assert '4, 0.0, 0.0, 1.0' in content

    def test_element_written(self):
        content = self._write()
        assert '1, 1, 2, 3, 4' in content

    def test_phase_elset_present(self):
        content = self._write()
        assert '*Elset, elset=POLY1' in content

    def test_eall_elset_present(self):
        """EALL elset required by periodic_inp_writer.py step output blocks."""
        content = self._write()
        assert '*Elset, elset=EALL' in content

    def test_material_block_present(self):
        content = self._write()
        assert '*Material, name=MATERIAL' in content
        assert '*Elastic' in content

    def test_custom_material_name(self):
        content = self._write(material_name='PVA')
        assert '*Material, name=PVA' in content
        assert '*Solid Section, elset=POLY1, material=PVA' in content

    def test_solid_section_links_elset_and_material(self):
        content = self._write()
        assert '*Solid Section, elset=POLY1, material=MATERIAL' in content

    def test_compatible_with_periodic_inp_writer(self):
        """periodic_inp_writer.py must be able to parse the *Node block."""
        import tempfile, os
        from mdvoro.utils.periodic_inp_writer import parse_inp_nodes, get_box_bounds
        # Write to a real temp file (parse_inp_nodes needs a path)
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _calculix_writer(fname, self.mesh_args, self.seeds, self.lims, None)
            nodes = parse_inp_nodes(fname)
            assert len(nodes) == 4
            xlo, xhi, _, _, _, _ = get_box_bounds(nodes)
            assert abs(xlo - 0.0) < 1e-4
            assert abs(xhi - 1.0) < 1e-4
        finally:
            os.unlink(fname)

    def test_negative_coords_preserved(self):
        """Coordinates must not have abs() applied."""
        points = np.array([
            [-1.0, -2.0, -3.0, 0.0],
            [ 1.0,  2.0,  3.0, 0.0],
            [-0.5,  0.5,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
        ])
        mesh_args = (points, self.mesh_args[1], self.mesh_args[2])
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w') as f:
            fname = f.name
        try:
            _calculix_writer(fname, mesh_args, self.seeds, self.lims, None)
            content = open(fname).read()
        finally:
            os.unlink(fname)
        assert '-1.0' in content
        assert '-2.0' in content
        assert '-3.0' in content
