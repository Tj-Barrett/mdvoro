"""
Tests for mdvoro.utils.mdutils

Covers:
  - is_not_far: correct distance comparison, no dead branches
  - find_closest: returns actual Euclidean distance (bug: was dist^4)
  - find_mult_closest: cutoff comparison against actual distance (bug: was dist^4)
  - is_on_boundary: absolute-value distance, explicit False return
  - is_colinear: basic geometry
  - detect_zone_dist: periodic offset selection
"""
import pytest
import numpy as np
from mdvoro.utils.mdutils import mdutils


class TestIsNotFar:
    def test_within_cutoff_returns_true(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1e-7])
        assert mdutils.is_not_far(a, b, cut=1e-6) is True

    def test_outside_cutoff_returns_false(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert mdutils.is_not_far(a, b, cut=1e-6) is False

    def test_exactly_at_cutoff_returns_true(self):
        # dist == cut: condition is dist > cut, so equal is not "far" → True
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1e-6, 0.0, 0.0])
        assert mdutils.is_not_far(a, b, cut=1e-6) is True

    def test_default_cutoff(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        assert mdutils.is_not_far(a, b) is True


class TestFindClosest:
    def test_returns_correct_min_index(self):
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
        ]
        min_index, _, _ = mdutils.find_closest(pt, pt_array)
        assert min_index == 1

    def test_returns_euclidean_distance_not_dist4(self):
        """Bug fix: exponent was **2, making returned dist = actual_dist^4."""
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [np.array([0.1, 0.0, 0.0])]
        _, _, dist = mdutils.find_closest(pt, pt_array)
        assert abs(dist - 0.1) < 1e-10, f"Expected ~0.1, got {dist}"

    def test_distance_not_inflated(self):
        """Regression: with **2 bug, dist of 0.5 would return 0.0625 (0.5^4)."""
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [np.array([0.5, 0.0, 0.0])]
        _, _, dist = mdutils.find_closest(pt, pt_array)
        assert abs(dist - 0.5) < 1e-10

    def test_returns_correct_location(self):
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [
            np.array([2.0, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
        ]
        _, min_location, _ = mdutils.find_closest(pt, pt_array)
        assert np.allclose(min_location, [0.3, 0.0, 0.0])


class TestFindMultClosest:
    def test_returns_indices_within_cutoff(self):
        """Bug fix: was comparing dist^4 < cut, so distant points incorrectly matched."""
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [
            np.array([0.05, 0.0, 0.0]),  # dist=0.05, within cut=0.1
            np.array([0.50, 0.0, 0.0]),  # dist=0.5,  outside cut=0.1
            np.array([0.08, 0.0, 0.0]),  # dist=0.08, within cut=0.1
        ]
        indices = mdutils.find_mult_closest(pt, pt_array, cut=0.1)
        assert 0 in indices
        assert 2 in indices

    def test_excludes_points_outside_cutoff(self):
        """Regression: with **2 bug, a point at dist=0.5 gives dist^4=0.0625
        which incorrectly passes a cut=0.1 threshold."""
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [np.array([0.5, 0.0, 0.0])]
        indices = mdutils.find_mult_closest(pt, pt_array, cut=0.1)
        assert 0 not in indices

    def test_empty_when_none_in_range(self):
        pt = np.array([0.0, 0.0, 0.0])
        pt_array = [np.array([10.0, 0.0, 0.0])]
        indices = mdutils.find_mult_closest(pt, pt_array, cut=0.1)
        assert indices == []


class TestIsOnBoundary:
    def test_point_on_boundary(self):
        a = np.array([5.0, 0.0, 0.0])
        lim_array = [np.array([5.0, 0.0, 0.0])]
        assert mdutils.is_on_boundary(a, lim_array, cut=1e-3) is True

    def test_point_off_boundary_returns_false_not_none(self):
        """Bug fix: used to return None (no explicit return False)."""
        a = np.array([100.0, 100.0, 100.0])
        lim_array = [np.array([0.0, 0.0, 0.0])]
        result = mdutils.is_on_boundary(a, lim_array, cut=1e-3)
        assert result is False
        assert result is not None

    def test_negative_coordinate_not_false_positive(self):
        """Bug fix: signed diff (no abs) made negative coords always satisfy < cut.
        Point must differ in all axes from the plane point to avoid per-axis match."""
        a = np.array([-5.0, 2.0, 3.0])
        lim_array = [np.array([5.0, 0.0, 0.0])]
        # abs distances: [10, 2, 3] — all >> cut=1e-3
        assert mdutils.is_on_boundary(a, lim_array, cut=1e-3) is False

    def test_large_negative_not_on_boundary(self):
        """Regression: without abs(), -1000 < 1e-3 was True."""
        a = np.array([-1000.0, -1000.0, -1000.0])
        lim_array = [np.array([0.0, 0.0, 0.0])]
        assert mdutils.is_on_boundary(a, lim_array, cut=1e-3) is False


class TestIsColinear:
    def test_colinear_points(self):
        assert mdutils.is_colinear([0, 0, 0], [1, 0, 0], [2, 0, 0]) is True

    def test_non_colinear_points(self):
        assert mdutils.is_colinear([0, 0, 0], [1, 0, 0], [0, 1, 0]) is False

    def test_colinear_diagonal(self):
        assert mdutils.is_colinear([0, 0, 0], [1, 1, 1], [2, 2, 2]) is True


class TestDetectZoneDist:
    def test_x_periodic_offset_applied(self):
        """Points near opposite x walls: periodic image distance should be ~0.2."""
        lims = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        periodic = [True, False, False]
        a = np.array([4.9, 0.0, 0.0])
        b = np.array([-4.9, 0.0, 0.0])
        dist, offset, _ = mdutils.detect_zone_dist(a, b, lims, periodic)
        assert dist < 1.0

    def test_y_periodic_offset_applied(self):
        """Points near opposite y walls with y-periodic only."""
        lims = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        periodic = [False, True, False]
        a = np.array([0.0, 4.9, 0.0])
        b = np.array([0.0, -4.9, 0.0])
        dist, offset, _ = mdutils.detect_zone_dist(a, b, lims, periodic)
        assert dist < 1.0

    def test_z_periodic_offset_applied(self):
        """Points near opposite z walls with z-periodic only."""
        lims = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        periodic = [False, False, True]
        a = np.array([0.0, 0.0, 4.9])
        b = np.array([0.0, 0.0, -4.9])
        dist, offset, _ = mdutils.detect_zone_dist(a, b, lims, periodic)
        assert dist < 1.0
