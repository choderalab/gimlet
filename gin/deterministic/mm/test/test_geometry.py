import pytest
import gin
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import math

def test_angles():
    angle_idxs = [[0, 1, 2]]
    coordinates = [[
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ]]

    angles = gin.deterministic.mm.geometry.get_angles(angle_idxs, coordinates)

    npt.assert_almost_equal(
        tf.squeeze(angles),
        0.5 * math.pi)


def test_torsions():
    torsion_idxs = [[0, 1, 2, 3]]

    coordinates = [[
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 1.0]
    ]]

    torsions = gin.deterministic.mm.geometry.get_torsions(torsion_idxs, coordinates)

    npt.assert_almost_equal(
        tf.squeeze(torsions),
        0.0)
