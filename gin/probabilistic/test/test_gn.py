import pytest
import gin
import numpy as np
import numpy.testing as npt
import tensorflow as tf

def test_build_graph_net():
    gn = gin.probabilistic.gn.GraphNet()

@pytest.fixture
def empty_gn():
    return gin.probabilistic.gn.GraphNet()

def test_forward_update_bonds_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn.GraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane).numpy(),
        2 ** 5 * 10 * np.ones((4, 8)))

def test_forward_update_atoms_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4
    gn = gin.probabilistic.gn.GraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        f_r=lambda *x: x[1],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane).numpy(),
        4 ** 5 * np.ones((4, 16)))

def test_forward_update_global_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn.GraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: 5 * tf.ones((1, 32)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        f_r=lambda *x: x[2],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane).numpy(),
        3 ** 5 * 5 * np.ones((1, 32)))

def test_forward_rho_e_v_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn.GraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 16), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 16)),
        phi_v=lambda *x: x[1],
        f_r=lambda *x: x[1],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane).numpy(),
        20 * np.ones((4, 16)))
