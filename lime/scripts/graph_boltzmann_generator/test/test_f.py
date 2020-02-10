import flow
import tensorflow as tf
import gin
import numpy as np
import numpy.testing as npt


def test_f():
    mol = gin.i_o.from_smiles.to_mol('CC')
    mol = gin.deterministic.hydrogen.add_hydrogen(mol)
    atoms, adjacency_map = mol

    chinese_postman_routes = tf.constant(
        [
            [0, 1, 4, 1, 5, 1, 2, 3, 2, 7, 2, 6],
            [0, 1, 4, 1, 5, 1, 2, 3, 2, 6, 2, 7],
            [0, 1, 4, 1, 5, 1, 2, 6, 2, 3, 2, 7],
            [0, 1, 4, 1, 5, 1, 2, 6, 2, 7, 2, 3],
            [0, 1, 4, 1, 5, 1, 2, 7, 2, 6, 2, 3],
            [0, 1, 4, 1, 5, 1, 2, 7, 2, 3, 2, 6],

            [0, 1, 5, 1, 4, 1, 2, 3, 2, 7, 2, 6],
            [0, 1, 5, 1, 4, 1, 2, 3, 2, 6, 2, 7],
            [0, 1, 5, 1, 4, 1, 2, 6, 2, 3, 2, 7],
            [0, 1, 5, 1, 4, 1, 2, 6, 2, 7, 2, 3],
            [0, 1, 5, 1, 4, 1, 2, 7, 2, 6, 2, 3],
            [0, 1, 5, 1, 4, 1, 2, 7, 2, 3, 2, 6],

            [5, 1, 0, 1, 4, 1, 2, 3, 2, 7, 2, 6],
            [5, 1, 0, 1, 4, 1, 2, 3, 2, 6, 2, 7],
            [5, 1, 0, 1, 4, 1, 2, 6, 2, 3, 2, 7],
            [5, 1, 0, 1, 4, 1, 2, 6, 2, 7, 2, 3],
            [5, 1, 0, 1, 4, 1, 2, 7, 2, 6, 2, 3],
            [5, 1, 0, 1, 4, 1, 2, 7, 2, 3, 2, 6],

            [5, 1, 4, 1, 0, 1, 2, 3, 2, 7, 2, 6],
            [5, 1, 4, 1, 0, 1, 2, 3, 2, 6, 2, 7],
            [5, 1, 4, 1, 0, 1, 2, 6, 2, 3, 2, 7],
            [5, 1, 4, 1, 0, 1, 2, 6, 2, 7, 2, 3],
            [5, 1, 4, 1, 0, 1, 2, 7, 2, 6, 2, 3],
            [5, 1, 4, 1, 0, 1, 2, 7, 2, 3, 2, 6],

            [4, 1, 5, 1, 0, 1, 2, 3, 2, 7, 2, 6],
            [4, 1, 5, 1, 0, 1, 2, 3, 2, 6, 2, 7],
            [4, 1, 5, 1, 0, 1, 2, 6, 2, 3, 2, 7],
            [4, 1, 5, 1, 0, 1, 2, 6, 2, 7, 2, 3],
            [4, 1, 5, 1, 0, 1, 2, 7, 2, 6, 2, 3],
            [4, 1, 5, 1, 0, 1, 2, 7, 2, 3, 2, 6],

            [4, 1, 0, 1, 5, 1, 2, 3, 2, 7, 2, 6],
            [4, 1, 0, 1, 5, 1, 2, 3, 2, 6, 2, 7],
            [4, 1, 0, 1, 5, 1, 2, 6, 2, 3, 2, 7],
            [4, 1, 0, 1, 5, 1, 2, 6, 2, 7, 2, 3],
            [4, 1, 0, 1, 5, 1, 2, 7, 2, 6, 2, 3],
            [4, 1, 0, 1, 5, 1, 2, 7, 2, 3, 2, 6],

        ],
        dtype=tf.int64)

    graph_flow = flow.GraphFlow(whiten=False)
    z = tf.random.normal(
        shape = (36, 6, 3))

    x, log_det_zx = graph_flow.f_zx(z, atoms, adjacency_map, chinese_postman_routes)
    z_, log_det_xz = graph_flow.f_xz(x, atoms, adjacency_map, chinese_postman_routes)

    npt.assert_almost_equal(to_return_0.numpy(), to_return_1.numpy())

    # x_, log_det_zx_ = graph_flow.f_zx(z_, atoms, adjacency_map, chinese_postman_routes)
    # npt.assert_almost_equal(z_.numpy(), z.numpy())
    # npt.assert_almost_equal(z.numpy(), z_.numpy())
    # npt.assert_almost_equal(log_det_zx.numpy(), log_det_xz.numpy())

    # npt.assert_almost_equal(x.numpy(), x_.numpy())
