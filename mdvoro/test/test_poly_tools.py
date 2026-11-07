import numpy as np

from mdvoro.mesh.poly_tools import snap_to_box


def test_snap_to_box_rounds_a_face_to_one_plane():
    # a face coordinate next to a 4-decimal rounding boundary: voro++ round-off
    # puts its vertices on both sides, which used to split the face in two
    lims = [[0.0, 15.0], [0.0, 15.0], [6.294652, 21.294652]]
    xyz = np.array([[1.0, 1.0, 6.2946499],     # rounds to 6.2946
                    [2.0, 1.0, 6.2946521],     # rounds to 6.2947
                    [3.0, 2.0, 21.2946531],
                    [4.0, 4.0, 10.0]])         # interior: untouched
    before = {round(z, 4) for z in xyz[:2, 2]}
    out = snap_to_box(xyz, lims)
    assert len(before) == 2
    assert len({round(z, 4) for z in out[:2, 2]}) == 1
    assert out[0, 2] == out[1, 2] == 6.294652
    assert out[2, 2] == 21.294652
    assert np.array_equal(out[3], xyz[3])
    assert out[0, 0] == 1.0                    # 1.0 is 1 from the x faces


def test_snap_to_box_leaves_vertices_off_the_faces():
    lims = [[0.0, 10.0]] * 3
    xyz = np.array([[0.001, 5.0, 5.0], [9.9995, 5.0, 5.0], [0.00005, 5.0, 5.0]])
    out = snap_to_box(xyz, lims)
    assert out[0, 0] == 0.001                  # beyond the 1e-4 tolerance
    assert out[1, 0] == 9.9995
    assert out[2, 0] == 0.0
