"""
Tests for mdvoro.utils.periodic_inp_writer

Covers:
  - sort_surface_nodes: face/edge/corner/interior classification
  - build_periodic_pairs: edge pairing uses diagonal opposite zones (bug fix 1),
      RP coefficient signs correct for shear (bug fix 2),
      corners paired with true diagonal (bug fix 3)
  - write_periodic_inp: integration — RP nodes inserted, equations written
"""
import pytest
import tempfile
import os
import numpy as np
from mdvoro.utils.periodic_inp_writer import (
    sort_surface_nodes, build_periodic_pairs, _match_pairs,
    write_periodic_inp, get_box_bounds, parse_inp_nodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_cube_nodes():
    """
    27 nodes of a 3x3x3 lattice on [0,1]^3.
    Each of the 26 surface positions is covered, plus one interior node.
    """
    nodes = {}
    nid = 1
    for x in [0.0, 0.5, 1.0]:
        for y in [0.0, 0.5, 1.0]:
            for z in [0.0, 0.5, 1.0]:
                nodes[nid] = np.array([x, y, z])
                nid += 1
    return nodes   # 27 nodes total; (0.5,0.5,0.5) is the sole interior node


def _zone_of(nodes, nid, tol=1e-5):
    """Return the zone dict for a single node (helper for inspection)."""
    zones = sort_surface_nodes(nodes, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, tol)
    for z, members in zones.items():
        if nid in members:
            return z
    return None   # interior


# ---------------------------------------------------------------------------
# sort_surface_nodes
# ---------------------------------------------------------------------------

class TestSortSurfaceNodes:
    def setup_method(self):
        self.nodes = _unit_cube_nodes()
        self.zones = sort_surface_nodes(self.nodes, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, tol=1e-5)

    def _nid(self, x, y, z):
        for nid, c in self.nodes.items():
            if np.allclose(c, [x, y, z]):
                return nid
        raise KeyError((x, y, z))

    def test_interior_node_not_in_any_zone(self):
        nid = self._nid(0.5, 0.5, 0.5)
        in_any = any(nid in v for v in self.zones.values())
        assert not in_any

    def test_face_xhi_in_zone1(self):
        # (1,0.5,0.5) — on +x face only
        nid = self._nid(1.0, 0.5, 0.5)
        assert nid in self.zones[1]

    def test_face_xlo_in_zone2(self):
        nid = self._nid(0.0, 0.5, 0.5)
        assert nid in self.zones[2]

    def test_face_yhi_in_zone3(self):
        nid = self._nid(0.5, 1.0, 0.5)
        assert nid in self.zones[3]

    def test_face_ylo_in_zone4(self):
        nid = self._nid(0.5, 0.0, 0.5)
        assert nid in self.zones[4]

    def test_face_zhi_in_zone5(self):
        nid = self._nid(0.5, 0.5, 1.0)
        assert nid in self.zones[5]

    def test_face_zlo_in_zone6(self):
        nid = self._nid(0.5, 0.5, 0.0)
        assert nid in self.zones[6]

    def test_edge_xhi_yhi_in_zone7(self):
        nid = self._nid(1.0, 1.0, 0.5)
        assert nid in self.zones[7]

    def test_edge_xlo_ylo_in_zone12(self):
        nid = self._nid(0.0, 0.0, 0.5)
        assert nid in self.zones[12]

    def test_corner_xhi_yhi_zhi_in_zone19(self):
        nid = self._nid(1.0, 1.0, 1.0)
        assert nid in self.zones[19]

    def test_corner_xlo_ylo_zlo_in_zone26(self):
        nid = self._nid(0.0, 0.0, 0.0)
        assert nid in self.zones[26]

    def test_total_surface_nodes(self):
        total = sum(len(v) for v in self.zones.values())
        # 27 nodes, 1 interior → 26 surface
        assert total == 26


# ---------------------------------------------------------------------------
# build_periodic_pairs — bug fix regressions
# ---------------------------------------------------------------------------

class TestBuildPeriodicPairs:
    """
    Use a minimal 2-node-per-zone setup so pair matching is deterministic.
    Each zone gets nodes placed at the correct boundary coordinates.
    """
    def setup_method(self):
        self.nodes = _unit_cube_nodes()
        self.zones = sort_surface_nodes(self.nodes, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, tol=1e-5)
        self.pairs = build_periodic_pairs(self.nodes, self.zones)

    def _pairs_for_zones(self, za, zb):
        """Return all (master,slave,coeffs) where master in zone za, slave in zone zb."""
        members_a = set(self.zones[za])
        members_b = set(self.zones[zb])
        return [(m, s, c) for m, s, c in self.pairs
                if m in members_a and s in members_b]

    # --- Bug fix 1: edge pairing must use diagonal opposite zones ---

    def test_zone7_paired_with_zone12_not_zone11(self):
        """zone7(+x+y) must pair with zone12(-x-y), not zone11(-x+y)."""
        # If bug were present, zone7 would pair with zone11 (same y=hi)
        correct = self._pairs_for_zones(7, 12)
        wrong   = self._pairs_for_zones(7, 11)
        assert len(correct) > 0, "zone7 must have pairs with zone12"
        assert len(wrong)   == 0, "zone7 must NOT pair with zone11"

    def test_zone8_paired_with_zone11_not_zone12(self):
        """zone8(+x-y) must pair with zone11(-x+y), not zone12(-x-y)."""
        correct = self._pairs_for_zones(8, 11)
        wrong   = self._pairs_for_zones(8, 12)
        assert len(correct) > 0
        assert len(wrong)   == 0

    def test_zone9_paired_with_zone14(self):
        """zone9(+x+z) must pair with zone14(-x-z)."""
        correct = self._pairs_for_zones(9, 14)
        assert len(correct) > 0

    def test_zone15_paired_with_zone18(self):
        """zone15(+y+z) must pair with zone18(-y-z)."""
        correct = self._pairs_for_zones(15, 18)
        assert len(correct) > 0

    # --- Bug fix 2: RP coefficient sign must match translation vector direction ---

    def test_face_x_rp_coefficient_is_minus1(self):
        """zone1(+x) ↔ zone2(-x): diff=(+Lx,0,0) → RP_X must be -1."""
        face_pairs = self._pairs_for_zones(1, 2)
        assert len(face_pairs) > 0
        for _, _, coeffs in face_pairs:
            assert coeffs == {'X': -1}, f"Expected {{'X': -1}}, got {coeffs}"

    def test_edge_zone8_zone11_y_coefficient_is_plus1(self):
        """zone8(+x-y) ↔ zone11(-x+y): diff=(+Lx,-Ly,0) → X:-1 Y:+1.
        Bug: Abaqus version always used Y:-1, which is wrong here."""
        ep = self._pairs_for_zones(8, 11)
        assert len(ep) > 0
        for _, _, coeffs in ep:
            assert coeffs.get('Y') == +1, f"Expected Y:+1, got Y:{coeffs.get('Y')}"
            assert coeffs.get('X') == -1, f"Expected X:-1, got X:{coeffs.get('X')}"

    def test_edge_zone7_zone12_both_coefficients_minus1(self):
        """zone7(+x+y) ↔ zone12(-x-y): diff=(+Lx,+Ly,0) → X:-1 Y:-1."""
        ep = self._pairs_for_zones(7, 12)
        assert len(ep) > 0
        for _, _, coeffs in ep:
            assert coeffs == {'X': -1, 'Y': -1}

    def test_edge_zone10_zone13_z_coefficient_is_plus1(self):
        """zone10(+x-z) ↔ zone13(-x+z): diff=(+Lx,0,-Lz) → X:-1 Z:+1."""
        ep = self._pairs_for_zones(10, 13)
        assert len(ep) > 0
        for _, _, coeffs in ep:
            assert coeffs.get('Z') == +1

    # --- Bug fix 3: corners must pair with true full diagonal ---

    def test_corner_zone19_paired_with_zone26_not_zone25(self):
        """zone19(+x+y+z) must pair with zone26(-x-y-z), not zone25(-x-y+z).
        Bug: Abaqus version paired zone19 with zone25 (wrong — same z)."""
        correct = self._pairs_for_zones(19, 26)
        wrong   = self._pairs_for_zones(19, 25)
        assert len(correct) > 0, "zone19 must have pairs with zone26"
        assert len(wrong)   == 0, "zone19 must NOT pair with zone25"

    def test_corner_zone20_paired_with_zone25(self):
        """zone20(+x+y-z) must pair with zone25(-x-y+z)."""
        correct = self._pairs_for_zones(20, 25)
        assert len(correct) > 0

    def test_corner_zone19_coefficients_all_minus1(self):
        """zone19(+x+y+z) ↔ zone26(-x-y-z): diff=(+Lx,+Ly,+Lz) → all -1."""
        cp = self._pairs_for_zones(19, 26)
        assert len(cp) > 0
        for _, _, coeffs in cp:
            assert coeffs == {'X': -1, 'Y': -1, 'Z': -1}

    def test_corner_zone21_coefficients(self):
        """zone21(+x-y+z) ↔ zone24(-x+y-z): diff=(+Lx,-Ly,+Lz) → X:-1 Y:+1 Z:-1."""
        cp = self._pairs_for_zones(21, 24)
        assert len(cp) > 0
        for _, _, coeffs in cp:
            assert coeffs == {'X': -1, 'Y': +1, 'Z': -1}


# ---------------------------------------------------------------------------
# write_periodic_inp — integration test
# ---------------------------------------------------------------------------

def _make_minimal_inp(path, nodes):
    """Write a minimal NEPER-style INP with the given node dict."""
    with open(path, 'w') as f:
        f.write('*Node\n')
        for nid, c in sorted(nodes.items()):
            f.write('{:d}, {:.6f}, {:.6f}, {:.6f}\n'.format(nid, c[0], c[1], c[2]))
        f.write('*Element, type=C3D4\n')
        # Minimal single element referencing first 4 nodes
        keys = sorted(nodes.keys())[:4]
        f.write('1, {:d}, {:d}, {:d}, {:d}\n'.format(*keys))
        f.write('*Elset, elset=EALL\n1\n')
        f.write('*Material, name=PVA\n*Elastic\n1000.0, 0.3\n')
        f.write('*Solid Section, elset=EALL, material=PVA\n')


class TestWritePeriodicInp:
    def setup_method(self):
        self.nodes = _unit_cube_nodes()
        self.tmpdir = tempfile.mkdtemp()
        self.inp_in  = os.path.join(self.tmpdir, 'mesh.inp')
        self.inp_out = os.path.join(self.tmpdir, 'mesh_pbc.inp')
        _make_minimal_inp(self.inp_in, self.nodes)

    def teardown_method(self):
        for f in [self.inp_in, self.inp_out]:
            if os.path.exists(f):
                os.unlink(f)
        os.rmdir(self.tmpdir)

    def test_output_file_created(self):
        write_periodic_inp(self.inp_in, self.inp_out)
        assert os.path.exists(self.inp_out)
        assert os.path.getsize(self.inp_out) > 0

    def test_rp_nodes_inserted(self):
        write_periodic_inp(self.inp_in, self.inp_out)
        content = open(self.inp_out).read()
        max_nid = max(self.nodes.keys())
        assert str(max_nid + 1) in content  # RP-X
        assert str(max_nid + 2) in content  # RP-Y
        assert str(max_nid + 3) in content  # RP-Z

    def test_equation_block_present(self):
        write_periodic_inp(self.inp_in, self.inp_out)
        content = open(self.inp_out).read()
        assert '*EQUATION' in content

    def test_equation_count_is_three_per_pair(self):
        """Each node pair generates 3 equations (one per DOF)."""
        write_periodic_inp(self.inp_in, self.inp_out)
        content = open(self.inp_out).read()
        n_eq = content.count('*EQUATION')
        # Number of pairs × 3 DOFs
        zones = sort_surface_nodes(self.nodes, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, tol=1e-5)
        pairs = build_periodic_pairs(self.nodes, zones)
        assert n_eq == len(pairs) * 3

    def test_nset_rp_blocks_present(self):
        write_periodic_inp(self.inp_in, self.inp_out)
        content = open(self.inp_out).read()
        assert 'NSET=RP-X' in content
        assert 'NSET=RP-Y' in content
        assert 'NSET=RP-Z' in content

    def test_load_cases_written(self):
        write_periodic_inp(self.inp_in, self.inp_out, load_cases=['XX', 'XY'])
        content = open(self.inp_out).read()
        assert 'Load case: XX' in content
        assert 'Load case: XY' in content
        assert 'Load case: YY' not in content
