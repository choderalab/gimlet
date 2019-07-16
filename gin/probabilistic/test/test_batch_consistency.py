import pytest
import gin
import tonic
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import pandas as pd



class f_v(tf.keras.Model):
    def __init__(self, units):
        super(f_v, self).__init__()
        self.d = tf.keras.layers.Dense(units)

    # @tf.function
    def call(self, x):
        return self.d(tf.one_hot(x, 8))

class phi_u(tf.keras.Model):
    def __init__(self):
        super(phi_u, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(
            (32, 'elu', 32, 'elu'))

    def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
        return self.d(h_u, h_u_0, h_e_bar, h_v_bar)



def test_consistency():

    class f_r(tf.keras.Model):
        def __init__(self, config):
            super(f_r, self).__init__()
            # self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

        # @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):
            # y = tf.reshape(self.d(h_u), [-1])
            return h_e_history, h_v_history, h_u_history

    gn = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(16),

        f_v=f_v(8),

        f_u=(lambda atoms, adjacency_map, batched_attr_mask: \
            tf.boolean_mask(
                tf.zeros((64, 32), dtype=tf.float32),
                batched_attr_mask)),

        phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'elu')),

        phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((8, 'elu', 8, 'elu')),

        phi_u=phi_u(),

        f_r=f_r((128, 'tanh', 128, 1)),

        repeat=5)

    cyclobutane = gin.i_o.from_smiles.to_mol('C1CCC1')
    caffeine = gin.i_o.from_smiles.to_mol('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')

    cyclobutane_atoms = cyclobutane[0]
    caffeine_atoms = caffeine[0]

    cyclobutane_adjacency_map = cyclobutane[1]
    caffeine_adjacency_map = caffeine[1]

    cyclobutane_n_atoms = tf.shape(cyclobutane_atoms, tf.int64)[0]
    caffeine_n_atoms = tf.shape(caffeine_atoms, tf.int64)[0]

    cyclobutane_n_bonds = tf.math.count_nonzero(cyclobutane_adjacency_map)
    caffeine_n_bonds = tf.math.count_nonzero(caffeine_adjacency_map)

    batched_n_atoms = tf.add(
        cyclobutane_n_atoms,
        caffeine_n_atoms)

    batched_atoms = tf.concat(
        [
            cyclobutane[0],
            caffeine[0],
            -1 * tf.ones(
                (64-caffeine_n_atoms-cyclobutane_n_atoms,),
                dtype=tf.int64)
        ],
        axis=0)

    batched_adjacency_map = tf.zeros(
        (64, 64),
        dtype=tf.float32)

    is_cyclobutane_1d = tf.less(
        tf.range(64, dtype=tf.int64),
        cyclobutane_n_atoms)

    is_caffeine_1d = tf.logical_and(
        tf.greater_equal(
            tf.range(64, dtype=tf.int64),
            cyclobutane_n_atoms),
        tf.less(
            tf.range(64, dtype=tf.int64),
            caffeine_n_atoms+cyclobutane_n_atoms))

    is_cyclobutane_2d = tf.logical_and(
        tf.tile(
            tf.expand_dims(
                is_cyclobutane_1d,
                0),
            [64, 1]),
        tf.tile(
            tf.expand_dims(
                is_cyclobutane_1d,
                1),
            [1, 64]))

    is_caffeine_2d = tf.logical_and(
        tf.tile(
            tf.expand_dims(
                is_caffeine_1d,
                0),
            [64, 1]),
        tf.tile(
            tf.expand_dims(
                is_caffeine_1d,
                1),
            [1, 64]))

    batched_adjacency_map = tf.where(
        is_cyclobutane_2d,
        tf.pad(
            cyclobutane_adjacency_map,
            [
                [0, 64-cyclobutane_n_atoms],
                [0, 64-cyclobutane_n_atoms]
            ]),
        batched_adjacency_map)

    batched_adjacency_map = tf.where(
        is_caffeine_2d,
        tf.pad(
            caffeine_adjacency_map,
            [
                [cyclobutane_n_atoms, 64-cyclobutane_n_atoms-caffeine_n_atoms],
                [cyclobutane_n_atoms, 64-cyclobutane_n_atoms-caffeine_n_atoms],
            ]),
        batched_adjacency_map)

    atom_in_mol = tf.concat(
        [
            tf.expand_dims(
                is_cyclobutane_1d,
                1),
            tf.expand_dims(
                is_caffeine_1d,
                1)
        ],
        axis=1)

    bond_in_mol = tf.concat(
        [
            tf.expand_dims(
                tf.less(
                    tf.range(128, dtype=tf.int64),
                    cyclobutane_n_bonds),
                1),
            tf.expand_dims(
                tf.logical_and(
                    tf.greater_equal(
                        tf.range(128, dtype=tf.int64),
                        cyclobutane_n_bonds),
                    tf.less(
                        tf.range(128, dtype=tf.int64),
                        cyclobutane_n_bonds + caffeine_n_bonds)),
                1)
        ],
        axis=1)

    y_mask = tf.concat(
        [
            tf.constant([True, True]),
            tf.tile(
                tf.constant([False]),
                [62])
        ],
        axis=0)


    (
        h_e_history_cyclobutane,
        h_v_history_cyclobutane,
        h_u_history_cyclobutane
    ) = gn(cyclobutane_atoms, cyclobutane_adjacency_map,
        batched_attr_mask=tf.concat(
            [
                tf.constant([True]),
                tf.tile(
                    tf.constant([False]),
                    [63])
            ],
            axis=0))

    (
        h_e_history_caffeine,
        h_v_history_caffeine,
        h_u_history_caffeine,
    ) = gn(caffeine_atoms, caffeine_adjacency_map,
        batched_attr_mask=tf.concat(
            [
                tf.constant([True]),
                tf.tile(
                    tf.constant([False]),
                    [63])
            ],
            axis=0))


    (
        h_e_history_batched,
        h_v_history_batched,
        h_u_history_batched
    ) = gn(
        batched_atoms,
        batched_adjacency_map,
        atom_in_mol,
        bond_in_mol,
        y_mask)

    # test edge function
    npt.assert_almost_equal(
        h_e_history_batched[:19, :, :].numpy(),
        tf.concat(
            [
                h_e_history_cyclobutane,
                h_e_history_caffeine
            ],
            axis=0).numpy(),
        decimal=3)

    npt.assert_almost_equal(
        h_v_history_batched[:18, :, :].numpy(),
        tf.concat(
            [
                h_v_history_cyclobutane,
                h_v_history_caffeine
            ],
            axis=0).numpy(),
        decimal=3)

    npt.assert_almost_equal(
        h_u_history_batched[:2, :, :].numpy(),
        tf.concat(
            [
                h_u_history_cyclobutane,
                h_u_history_caffeine
            ],
            axis=0).numpy(),
        decimal=2)



def test_consistency_ds():
    # read data
    df = pd.read_csv('data/delaney-processed.csv')
    df = df[:1024]
    df = df.loc[df['smiles'].str.len() > 1]
    x_array = df[['smiles']].values.flatten()
    y_array = \
        df[['measured log solubility in mols per litre']].values.flatten()
    y_array = (y_array - np.mean(y_array) / np.std(y_array))

    ds = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)

    ds_batched = gin.probabilistic.gn.GraphNet.batch(ds, 256)

    class f_r(tf.keras.Model):
        def __init__(self, config):
            super(f_r, self).__init__()
            self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

        # @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):
            y = tf.reshape(self.d(h_u), [-1])
            return y

    class f_v(tf.keras.Model):
        def __init__(self, units):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        # @tf.function
        def call(self, x):
            return self.d(tf.one_hot(x, 8))

    class phi_u(tf.keras.Model):
        def __init__(self):
            super(phi_u, self).__init__()
            self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(
                (128, 'elu', 128, 'elu'))

        def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
            return self.d(h_u, h_u_0, h_e_bar, h_v_bar)



    gn = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(128),

        f_v=f_v(128),

        f_u=(lambda atoms, adjacency_map, batched_attr_mask: \
            tf.boolean_mask(
                tf.zeros((64, 128), dtype=tf.float32),
                batched_attr_mask)),

        phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

        phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

        phi_u=phi_u(),

        f_r=f_r((128, 'tanh', 128, 1)),

        repeat=3)

    optimizer = tf.keras.optimizers.Adam(1e-5)
    n_epoch = 50


    idx = 0
    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask in ds_batched:

        start_idx = idx
        y_hat_batched = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_mask=y_mask)
        batch_size = int(tf.math.count_nonzero(y_mask).numpy())
        idx += batch_size

        ds_debatched = ds.skip(start_idx).take(batch_size)
        y_hat_debatched = tf.constant([-1], dtype=tf.float32)

        for atoms, adjacency_map, attr in ds_debatched:

            y_hat_debatched = tf.concat(
                [
                    y_hat_debatched,
                    gn(
                        atoms,
                        adjacency_map,
                        batched_attr_mask=tf.concat(
                            [
                                [True],
                                tf.tile(
                                    [False],
                                    [63])
                            ],
                            axis=0)),
                ],
                axis=0)

        y_hat_debatched = y_hat_debatched[1:]

        npt.assert_almost_equal(
            y_hat_batched.numpy(),
            y_hat_debatched.numpy(),
            decimal=2)
