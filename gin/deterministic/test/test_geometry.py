import gin
import pytest
import numpy.testing as npt
import tensorflow as tf
import math

def test_get_angles():
    coordinates = tf.constant(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=tf.float32)

    angle_idxs = tf.constant(
        [
            [1, 0, 2],
            [1, 0, 3],
            [1, 3, 2],
            [0, 1, 3]
        ],
        dtype=tf.int64)

    npt.assert_almost_equal(
        gin.deterministic.md.get_angles(
            coordinates,
            angle_idxs).numpy(),
        [math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.25 * math.pi],
        decimal=2)

    coordinates = tf.constant(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, math.sqrt(3), 0]
        ],
        dtype=tf.float32)

    angle_idxs = tf.constant(
        [
            [0, 1, 2],
            [0, 2, 1],
            [2, 0, 1]
        ],
        dtype=tf.int64)

    npt.assert_almost_equal(
        gin.deterministic.md.get_angles(
            coordinates,
            angle_idxs).numpy(),
        [math.pi/3, math.pi/3, math.pi/3])
