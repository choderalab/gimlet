import flow
import tensorflow as tf
import gin
import numpy as np
import numpy.testing as npt


def test_flow_zx_det():
    w = tf.random.normal(shape=(16, 3))
    b = tf.random.normal(shape=(16, 3))
    z_i = tf.random.normal(shape=(16, ))

    with tf.GradientTape() as tape:
        tape.watch(z_i)
        x, d_log_det = flow.GraphFlow.flow_zx(z_i, w, b)

    npt.assert_almost_equal(
        tf.math.log(tf.linalg.diag_part(tape.jacobian(x, z_i))).numpy(),
        d_log_det.numpy())

    w = tf.random.normal(shape=(16, 4, 3, 3))
    b = tf.random.normal(shape=(16, 4, 3))
    z_i = tf.random.normal(shape=(16, 3))

    with tf.GradientTape() as tape:
        tape.watch(z_i)
        x, d_log_det = flow.GraphFlow.flow_zx(z_i, w, b)

    dz_dx = tape.jacobian(x, z_i).numpy()
    for idx in range(16):

        npt.assert_almost_equal(
            d_log_det[idx].numpy(),
            tf.math.log(tf.linalg.det(dz_dx[idx, :, idx, :]).numpy()),
            decimal=1)


    w = tf.random.normal(shape=(16, 4, 2, 2))
    b = tf.random.normal(shape=(16, 4, 2))
    z_i = tf.random.normal(shape=(16, 2))

    with tf.GradientTape() as tape:
        tape.watch(z_i)
        x, d_log_det = flow.GraphFlow.flow_zx(z_i, w, b)

    dz_dx = tape.jacobian(x, z_i).numpy()
    for idx in range(16):

        npt.assert_almost_equal(
            d_log_det[idx].numpy(),
            tf.math.log(tf.linalg.det(dz_dx[idx, :, idx, :]).numpy()),
            decimal=1)


def test_flow_zx_det():
    w = tf.random.normal(shape=(16, 3))
    b = tf.random.normal(shape=(16, 3))
    x = tf.random.normal(shape=(16, ))

    with tf.GradientTape() as tape:
        tape.watch(x)
        z, d_log_det = flow.GraphFlow.flow_xz(x, w, b)

    npt.assert_almost_equal(
        tf.math.log(tf.linalg.diag_part(tape.jacobian(z, x))).numpy(),
        -d_log_det.numpy(),
        decimal=2)

    w = tf.random.normal(shape=(16, 4, 10, 3, 3))
    b = tf.random.normal(shape=(16, 4, 10, 3))
    x = tf.random.normal(shape=(16, 10, 3))

    with tf.GradientTape() as tape:
        tape.watch(x)
        z, log_det = flow.GraphFlow.flow_xz(x, w, b)

    dz_dx = tape.jacobian(z, x)

    for batch_idx in range(16):
        for walk_idx in range(10):

            npt.assert_almost_equal(
                tf.math.log(tf.linalg.det(dz_dx[batch_idx, walk_idx, :, batch_idx, walk_idx, :])).numpy(),
                -log_det[batch_idx, walk_idx].numpy(),
                decimal=1)


    w = tf.random.normal(shape=(16, 4, 2, 2))
    b = tf.random.normal(shape=(16, 4, 2))
    x = tf.random.normal(shape=(16, 2))

    with tf.GradientTape() as tape:
        tape.watch(x)
        z, log_det = flow.GraphFlow.flow_xz(x, w, b)

    dz_dx = tape.jacobian(z, x)

    for idx in range(16):
        npt.assert_almost_equal(
            tf.math.log(tf.linalg.det(dz_dx[idx, :, idx, :])).numpy(),
            -log_det[idx].numpy(),
            decimal=1)


def test_invertible():

    w = tf.random.normal(shape=(16, 3))
    b = tf.random.normal(shape=(16, 3))
    z = tf.random.normal(shape=(16, ))

    x, log_det_zx = flow.GraphFlow.flow_zx(z, w, b)
    z_, log_det_xz = flow.GraphFlow.flow_xz(x, w, b)

    npt.assert_almost_equal(z.numpy(), z_.numpy(), decimal=2)
    npt.assert_almost_equal(log_det_zx.numpy(), log_det_xz.numpy(), decimal=2)

    w = tf.random.normal(shape=(16, 3))
    b = tf.random.normal(shape=(16, 3))
    x = tf.random.normal(shape=(16, ))

    z, log_det_xz = flow.GraphFlow.flow_xz(x, w, b)
    x_, log_det_zx = flow.GraphFlow.flow_zx(z, w, b)

    npt.assert_almost_equal(x.numpy(), x_.numpy(), decimal=2)
    npt.assert_almost_equal(log_det_xz.numpy(), log_det_zx.numpy(), decimal=2)

    w = tf.random.normal(shape=(16, 4, 2, 2))
    b = tf.random.normal(shape=(16, 4, 2))
    z = tf.random.normal(shape=(16, 2))

    x, log_det_zx = flow.GraphFlow.flow_zx(z, w, b)
    z_, log_det_xz = flow.GraphFlow.flow_xz(x, w, b)

    npt.assert_almost_equal(z.numpy(), z_.numpy(), decimal=2)
    npt.assert_almost_equal(log_det_zx.numpy(), log_det_xz.numpy(), decimal=2)

    w = tf.random.normal(shape=(16, 4, 2, 2))
    b = tf.random.normal(shape=(16, 4, 2))
    x = tf.random.normal(shape=(16, 2))

    z, log_det_xz = flow.GraphFlow.flow_xz(x, w, b)
    x_, log_det_zx = flow.GraphFlow.flow_zx(z, w, b)

    npt.assert_almost_equal(x.numpy(), x_.numpy(), decimal=2)
    npt.assert_almost_equal(log_det_xz.numpy(), log_det_zx.numpy(), decimal=2)

    w = tf.random.normal(shape=(16, 4, 3, 3))
    b = tf.random.normal(shape=(16, 4, 3))
    z = tf.random.normal(shape=(16, 3))

    x, log_det_zx = flow.GraphFlow.flow_zx(z, w, b)
    z_, log_det_xz = flow.GraphFlow.flow_xz(
        tf.expand_dims(x, 1),
        tf.expand_dims(w, 2),
        tf.expand_dims(b, 2))

    npt.assert_almost_equal(z.numpy(), tf.squeeze(z_).numpy(), decimal=2)
    npt.assert_almost_equal(log_det_zx.numpy(), tf.squeeze(log_det_xz).numpy(), decimal=2)

    w = tf.random.normal(shape=(16, 4, 1, 3, 3))
    b = tf.random.normal(shape=(16, 4, 1, 3))
    x = tf.random.normal(shape=(16, 1, 3))

    z, log_det_xz = flow.GraphFlow.flow_xz(x, w, b)
    log_det_xz = tf.squeeze(log_det_xz)

    x_, log_det_zx = flow.GraphFlow.flow_zx(
        tf.squeeze(z),
        tf.squeeze(w),
        tf.squeeze(b))
    npt.assert_almost_equal(x_.numpy(), tf.squeeze(x).numpy(), decimal=2)
    npt.assert_almost_equal(log_det_xz.numpy(), log_det_zx.numpy(), decimal=2)


#
# def test_flow_zx():
#     chinese_postman_routes = tf.constant(
#         [
#             [0, 1, 4, 1, 5, 1, 2, 3, 2, 7, 2, 6],
#             [0, 1, 4, 1, 5, 1, 2, 3, 2, 6, 2, 7],
#             [0, 1, 4, 1, 5, 1, 2, 6, 2, 3, 2, 7],
#             [0, 1, 4, 1, 5, 1, 2, 6, 2, 7, 2, 3],
#             [0, 1, 4, 1, 5, 1, 2, 7, 2, 6, 2, 3],
#             [0, 1, 4, 1, 5, 1, 2, 7, 2, 3, 2, 6],
#
#             [0, 1, 5, 1, 4, 1, 2, 3, 2, 7, 2, 6],
#             [0, 1, 5, 1, 4, 1, 2, 3, 2, 6, 2, 7],
#             [0, 1, 5, 1, 4, 1, 2, 6, 2, 3, 2, 7],
#             [0, 1, 5, 1, 4, 1, 2, 6, 2, 7, 2, 3],
#             [0, 1, 5, 1, 4, 1, 2, 7, 2, 6, 2, 3],
#             [0, 1, 5, 1, 4, 1, 2, 7, 2, 3, 2, 6],
#
#             [5, 1, 0, 1, 4, 1, 2, 3, 2, 7, 2, 6],
#             [5, 1, 0, 1, 4, 1, 2, 3, 2, 6, 2, 7],
#             [5, 1, 0, 1, 4, 1, 2, 6, 2, 3, 2, 7],
#             [5, 1, 0, 1, 4, 1, 2, 6, 2, 7, 2, 3],
#             [5, 1, 0, 1, 4, 1, 2, 7, 2, 6, 2, 3],
#             [5, 1, 0, 1, 4, 1, 2, 7, 2, 3, 2, 6],
#
#             [5, 1, 4, 1, 0, 1, 2, 3, 2, 7, 2, 6],
#             [5, 1, 4, 1, 0, 1, 2, 3, 2, 6, 2, 7],
#             [5, 1, 4, 1, 0, 1, 2, 6, 2, 3, 2, 7],
#             [5, 1, 4, 1, 0, 1, 2, 6, 2, 7, 2, 3],
#             [5, 1, 4, 1, 0, 1, 2, 7, 2, 6, 2, 3],
#             [5, 1, 4, 1, 0, 1, 2, 7, 2, 3, 2, 6],
#
#             [4, 1, 5, 1, 0, 1, 2, 3, 2, 7, 2, 6],
#             [4, 1, 5, 1, 0, 1, 2, 3, 2, 6, 2, 7],
#             [4, 1, 5, 1, 0, 1, 2, 6, 2, 3, 2, 7],
#             [4, 1, 5, 1, 0, 1, 2, 6, 2, 7, 2, 3],
#             [4, 1, 5, 1, 0, 1, 2, 7, 2, 6, 2, 3],
#             [4, 1, 5, 1, 0, 1, 2, 7, 2, 3, 2, 6],
#
#             [4, 1, 0, 1, 5, 1, 2, 3, 2, 7, 2, 6],
#             [4, 1, 0, 1, 5, 1, 2, 3, 2, 6, 2, 7],
#             [4, 1, 0, 1, 5, 1, 2, 6, 2, 3, 2, 7],
#             [4, 1, 0, 1, 5, 1, 2, 6, 2, 7, 2, 3],
#             [4, 1, 0, 1, 5, 1, 2, 7, 2, 6, 2, 3],
#             [4, 1, 0, 1, 5, 1, 2, 7, 2, 3, 2, 6],
#
#         ],
#         dtype=tf.int64)
#
#     graph_flow = flow.GraphFlow()
#
#     z_i = tf.random.normal(shape=(16, 3))
#     seq_xyz = tf.random.normal(shape=(16, 8, 3))
#     h_graph = tf.random.normal(shape=(16, 64))
#
#
#     x = graph_flow.flow_zx(z_i, seq_xyz, h_graph)
