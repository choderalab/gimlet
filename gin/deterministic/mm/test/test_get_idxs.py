import pytest
import gin
import numpy as np
import numpy.testing as npt
import tensorflow as tf


def test_get_idxs():
    adjacency_map = tf.constant(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ],
        dtype=tf.float32)

    bond_idxs, angle_idxs, torsion_idxs, one_four_idxs, nonbonded_idxs = gin.deterministic.mm.get_idxs.get_idxs(
        adjacency_map)

    npt.assert_almost_equal(
        bond_idxs,
        [
            [0, 1],
            [0, 3],
            [1, 2],
            [2, 3]
        ])

    npt.assert_almost_equal(
        angle_idxs,
        [
            [1, 0, 3],
            [0, 1, 2],
            [1, 2, 3],
            [0, 3, 2]
        ])

    npt.assert_almost_equal(
        torsion_idxs,
        [
            [2, 1, 0, 3],
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [0, 3, 2, 1]
        ])
