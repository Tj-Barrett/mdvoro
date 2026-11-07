"""Region points and phase element sets for TetGen (tetgen._phase_regions, _phase_elsets)."""
import numpy as np

from mdvoro.mesh.tetgen import _phase_regions, _phase_elsets


def _row_of_cells(phases, spacing=1.0):
    """1D row of unit cells along x; cell i touches i-1 and i+1, walls at the ends"""
    n = len(phases)
    voro = []
    for i, p in enumerate(phases):
        faces = [{'adjacent_cell': i - 1 if i > 0 else -1, 'vertices': [0, 1, 2]},
                 {'adjacent_cell': i + 1 if i < n - 1 else -2, 'vertices': [0, 1, 2]}]
        voro.append({'volume': 1.0 + 0.1 * i, 'faces': faces, 'vertices': [],
                     'original': [(i + 0.5) * spacing, 0.5, 0.5, p]})
    return voro


class TestPhaseRegions:
    def test_one_attribute_per_connected_component(self):
        # amorphous (-1) appears in two disconnected pieces
        regions, rp = _phase_regions(_row_of_cells([-1, -1, 2, 2, 2, -1]), cell_cutoff=10.0)
        assert len(rp) == 3
        assert sorted(rp.values()) == [-1, -1, 2]

    def test_every_cell_seed_is_a_region_point(self):
        # the PLC may split a component, every piece needs a point
        phases = [-1, -1, 2, 2, 2, -1]
        regions, rp = _phase_regions(_row_of_cells(phases), cell_cutoff=10.0)
        assert len(regions) == len(phases)
        assert [r[0] for r in regions] == list(range(1, len(phases) + 1))
        by_x = {r[1]: rp[r[4]] for r in regions}
        assert [by_x[i + 0.5] for i in range(len(phases))] == phases

    def test_first_point_is_an_interior_seed(self):
        regions, rp = _phase_regions(_row_of_cells([-1, 2, 2, 2, -1]), cell_cutoff=10.0)
        crystal = [r for r in regions if rp[r[4]] == 2][0]
        assert crystal[1] == 2.5                # middle cell: no neighbour of another phase
        # one point per component first, then the other cells
        assert [r[4] for r in regions[:len(rp)]] == list(range(1, len(rp) + 1))

    def test_attribute_numbering(self):
        regions, rp = _phase_regions(_row_of_cells([3, 3, 4]), cell_cutoff=10.0)
        assert [r[4] for r in regions[:2]] == [1, 2]
        assert set(rp) == {1, 2}
        assert {r[4] for r in regions} == {1, 2}

    def test_far_neighbours_are_not_joined(self):
        # same phase, but seeds farther apart than the cutoff (periodic image)
        regions, rp = _phase_regions(_row_of_cells([2, 2], spacing=5.0), cell_cutoff=1.0)
        assert len(rp) == 2


class TestPhaseElsets:
    def test_region_phase_map(self):
        sets = _phase_elsets(np.array([1, 2, 2, 3]), seeds={}, region_phase={1: -1, 2: 3, 3: -1})
        assert sets == {1: [1, 4], 2: [], 3: [2, 3]}

    def test_seed_fallback(self):
        seeds = {0: {'phase': 1}, 1: {'phase': 2}}
        assert _phase_elsets([1, 2, 2], seeds) == {1: [1], 2: [2, 3]}

    def test_unknown_attribute_is_amorphous(self):
        assert _phase_elsets([0, 7], seeds={}, region_phase={1: 2}) == {1: [1, 2], 2: []}


class TestCollapseShortEdges:
    def _two_squares(self, eps):
        # two unit squares sharing an edge; vertex 6 sits eps from vertex 1
        xyz = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0], [1 + eps, 0, 0]]
        faces = [[0, 1, 2, 3], [1, 6, 4, 5, 2]]
        return faces, np.array(xyz, float)

    def test_merges_short_edge_and_drops_unused(self):
        from mdvoro.mesh.poly_tools import collapse_short_edges
        faces, xyz = self._two_squares(1e-3)
        f2, x2, keep, src, _ = collapse_short_edges(faces, xyz, 0.05)
        assert len(x2) == 6 and src == [0, 1]
        assert [len(f) for f in f2] == [4, 4]
        assert np.allclose(x2[f2[0][1]], [1.0005, 0, 0])

    def test_long_edges_untouched(self):
        from mdvoro.mesh.poly_tools import collapse_short_edges
        faces, xyz = self._two_squares(0.2)
        f2, x2, keep, src, _ = collapse_short_edges(faces, xyz, 0.05)
        assert np.allclose(x2, xyz) and f2 == faces

    def test_box_vertices_stay_on_the_box(self):
        from mdvoro.mesh.poly_tools import collapse_short_edges
        xyz = np.array([[0, 0.5, 0.5], [0.01, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], float)
        f2, x2, _, _, _ = collapse_short_edges([[0, 1, 2], [1, 3, 2]], xyz, 0.05,
                                            lims=[[0, 1], [0, 1], [0, 1]])
        assert len(x2) == 3 and np.isclose(x2[0][0], 0.0)

    def test_cluster_size_is_bounded(self):
        from mdvoro.mesh.poly_tools import collapse_short_edges
        # a chain of short edges must not collapse into one far-reaching cluster
        xyz = np.array([[0.04 * k, 0, 0] for k in range(6)] + [[0.1, 1, 0]], float)
        f2, x2, _, _, _ = collapse_short_edges([list(range(7))], xyz, 0.05)
        assert len(x2) > 3

    def test_pinched_polygon_is_split(self):
        from mdvoro.mesh.poly_tools import _split_pinched
        assert _split_pinched([0, 1, 2, 3, 1, 4]) == [[1, 2, 3], [1, 4, 0]]
        assert _split_pinched([0, 1]) == []

    def test_frozen_vertices_are_not_merged(self):
        from mdvoro.mesh.poly_tools import collapse_short_edges
        faces, xyz = self._two_squares(1e-3)
        f2, x2, _, _, rep = collapse_short_edges(faces, xyz, 0.05, frozen={6})
        assert len(x2) == 7 and rep[6] == 6
