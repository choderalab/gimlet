import gin
import pytest
import numpy.testing as npt
import tensorflow as tf

def test_embed():
    ds = gin.i_o.from_sdf.read_sdf('data/caffeine.sdf')
    coordinates = list(ds)[0][2]

    distance_matrix = gin.deterministic.md.get_distance_matrix(coordinates)

    d_o_2 = tf.reduce_sum(
        tf.pow(
            coordinates - tf.reduce_mean(coordinates, 0),
            2),
        axis=1)

    npt.assert_almost_equal(
        distance_matrix.numpy(),
        gin.deterministic.md.get_distance_matrix(
            gin.deterministic.conformer.embed(distance_matrix)).numpy(),
        decimal=2)
