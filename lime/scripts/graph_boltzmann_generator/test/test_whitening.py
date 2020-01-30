import flow
import tensorflow as tf
import gin
import numpy as np
import numpy.testing as npt

def test_whitening():
    seq_xyz = tf.random.normal(
        shape=(64, 16, 3),
        dtype=tf.float32)

    seq_xyz_whitened = flow.GraphFlow.whitening(seq_xyz)

    npt.assert_almost_equal(
        gin.deterministic.mm.geometry.get_distance_matrix(seq_xyz_whitened).numpy(),
        gin.deterministic.mm.geometry.get_distance_matrix(seq_xyz).numpy())

test_whitening()
