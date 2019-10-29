import pytest
import gin
import numpy as np
import numpy.testing as npt
import tensorflow as tf

def test_build_graph_net():
    gn = gin.probabilistic.gn_hyper.HyperGraphNet()

@pytest.fixture
def empty_gn():
    return gin.probabilistic.gn_hyper.HyperGraphNet()

def test_forward_update_bonds_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32)),
        f_a=lambda *x: tf.zeros((6, 128)),
        f_t=lambda *x: tf.zeros((6, 256)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        f_r=lambda *x: x[1],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        2 ** 5 * 10 * np.ones((4, 8)))

def test_forward_update_atoms_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4
    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        f_r=lambda *x: x[0],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 ** 5 * np.ones((4, 16)))

def test_forward_update_angle_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32), dtype=tf.float32),
        f_a=lambda *x: tf.ones((4, 64), dtype=tf.float32),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        phi_a=lambda *x: 5 * x[0],
        f_r=lambda *x: x[2],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        5 ** 5 * np.ones((4, 64)))

def test_forward_update_torsion_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 32), dtype=tf.float32),
        f_a=lambda *x: tf.ones((4, 64), dtype=tf.float32),
        f_t=lambda *x: tf.ones((4, 128), dtype=tf.float32),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        phi_a=lambda *x: 5 * x[0],
        phi_t=lambda *x: 6 * x[0],
        f_r=lambda *x: x[3],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        6 ** 5 * np.ones((4, 128)))


def test_forward_update_global_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: 5 * tf.ones((1, 32)),
        phi_e=lambda *x: 2 * x[0],
        phi_u=lambda *x: 3 * x[0],
        phi_v=lambda *x: 4 * x[0],
        f_r=lambda *x: x[4],
        repeat=5)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        3 ** 5 * 5 * np.ones((1, 32)))

def test_forward_rho_e_v_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: 10 * tf.ones((4, 16), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.zeros((1, 16)),
        phi_v=lambda *x: x[2],
        f_r=lambda *x: x[0],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        20 * np.ones((4, 16)))

def test_forward_rho_e_u_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        phi_u=lambda *x: x[2],
        f_r=lambda *x: x[4],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 * np.ones((1, 8)))

def test_forward_rho_v_u_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        phi_u=lambda *x: x[3],
        f_r=lambda *x: x[4],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 * np.ones((1, 8)))

def test_forward_rho_v_u_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        f_a=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        phi_u=lambda *x: x[4],
        f_r=lambda *x: x[4],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 * np.ones((1, 8)))

def test_forward_rho_t_u_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 16), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        f_t=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        phi_u=lambda *x: x[5],
        f_r=lambda *x: x[4],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 * np.ones((1, 8)))

def test_forward_phi_u_cyclobutane_1():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: 10 * tf.ones((4, 8), dtype=tf.float32),
        f_a=lambda *x: 100 * tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        f_t=lambda *x: 1000 * tf.ones((4, 8), dtype=tf.float32),
        phi_u=lambda *x: 5 * x[2] + 4 * x[3] + 3 * x[4] + 2 * x[5],
        f_r=lambda *x: x[4],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        4 * 2345* np.ones((1, 8)))

def test_forward_phi_a_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_a=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        f_t=lambda *x: tf.ones((4, 8), dtype=tf.float32),

        phi_a=lambda *x: 2 * x[2] + 3 * x[3],

        f_r=lambda *x: x[2],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        8 * np.ones((4, 8)))

def test_forward_phi_t_cyclobutane():
    cyclobutane = gin.i_o.from_smiles.smiles_to_organic_topological_molecule(
        'C1CCC1') # n_bonds = 4, n_atoms = 4

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_v=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_a=lambda *x: tf.ones((4, 8), dtype=tf.float32),
        f_u=lambda *x: tf.ones((1, 8), dtype=tf.float32),
        f_t=lambda *x: tf.ones((4, 8), dtype=tf.float32),

        phi_t=lambda *x: 2 * x[2] + 3 * x[3],

        f_r=lambda *x: x[3],
        repeat=1)

    npt.assert_almost_equal(
        gn(cyclobutane[0], cyclobutane[1]).numpy(),
        10 * np.ones((4, 8)))
