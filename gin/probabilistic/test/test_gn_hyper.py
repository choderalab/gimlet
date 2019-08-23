# =============================================================================
# imports
# =============================================================================
from sklearn import metrics
import os
import sys
import tensorflow as tf
import gin
import lime
import pandas as pd
import numpy as np


mols_ds = gin.i_o.from_sdf.to_ds('/Users/yuanqingwang/Downloads/qm9-2/gdb9.sdf', has_charge=False)

attr_ds = tf.data.Dataset.from_tensor_slices(
    pd.read_csv('/Users/yuanqingwang/Downloads/qm9-2/gdb9.sdf.csv').values[:, 1:].astype(np.float32))

ds = tf.data.Dataset.zip((mols_ds, attr_ds))

ds = ds.map(
    lambda mol, attr: (mol[0], mol[1], mol[2], attr))

ds = ds.take(2048)

ds = gin.probabilistic.gn_hyper.HyperGraphNet.batch(
    ds,
    128,
    attr_dimension=19).cache(
        str(os.getcwd()) + 'temp')


class f_v(tf.keras.Model):
    """ Featurization of nodes.
    Here we simply featurize atoms using one-hot encoding.

    """
    def __init__(self, units=32):
        super(f_v, self).__init__()
        self.d = tf.keras.layers.Dense(units)

    @tf.function
    def call(self, x):
        x = tf.one_hot(x, 10)
        # set shape because Dense doesn't like variation
        x.set_shape([None, 10])
        return self.d(x)


class f_r(tf.keras.Model):
    """ Readout function
    """
    def __init__(self, units=32):
        super(f_r, self).__init__()
        self.d_k = tf.keras.layers.Dense(units)
        self.d_q = tf.keras.layers.Dense(units)

        self.d_v_0 = tf.keras.layers.Dense(units, activation='elu')
        self.d_v_1 = tf.keras.layers.Dense(units)

        self.d_pair_0 = tf.keras.layers.Dense(units, activation='sigmoid')
        self.d_pair_1 = tf.keras.layers.Dense(19)

        self.d = lime.nets.for_gn.ConcatenateThenFullyConnect((64, 'sigmoid', 64, 19))

    @tf.function
    def call(self, h_v, h_e, h_a, h_t, h_u,
        h_v_history, h_e_history, h_a_history,
        h_t_history, h_u_history,
        atom_in_mol, bond_in_mol, angle_in_mol, torsion_in_mol,
        adjacency_map, coordinates):

        per_mol_mask = tf.matmul(
            tf.where(
                atom_in_mol,
                tf.ones_like(atom_in_mol, dtype=tf.float32),
                tf.zeros_like(atom_in_mol, dtype=tf.float32)),
            tf.transpose(
                tf.where(
                    atom_in_mol,
                    tf.ones_like(atom_in_mol, dtype=tf.float32),
                    tf.zeros_like(atom_in_mol, dtype=tf.float32))))

        # get distance matrix
        distance = gin.deterministic.md.get_distance_matrix(coordinates)

        distance = tf.expand_dims(
            distance,
            2)

        n_atoms = tf.shape(distance, tf.int64)[0]

        # (n_atoms, n_atoms, units)
        v = tf.multiply(
            tf.tile(
                tf.expand_dims(
                    per_mol_mask,
                    2),
                [1, 1, 32]),
            self.d_v_1(self.d_v_0(distance)))

        # (n_atoms, n_atoms, units)
        k = tf.multiply(
            tf.tile(
                tf.expand_dims(
                    per_mol_mask,
                    2),
                [1, 1, 32]),
            tf.tile(
                tf.expand_dims(
                    self.d_k(h_v),
                    1),
                [1, n_atoms, 1]))

        # (n_atoms, n_atoms, units)
        q = tf.multiply(
            tf.tile(
                tf.expand_dims(
                    per_mol_mask,
                    2),
                [1, 1, 32]),
            tf.tile(
                tf.expand_dims(
                    self.d_q(h_v),
                    0),
                [n_atoms, 1, 1]))

        h_pair = tf.concat(
            [
                k,
                q,
                v
            ],
            axis=2)

        h_pair = tf.reduce_sum(
            self.d_pair_1(self.d_pair_0(h_pair)),
            axis=0)

        h_pair = tf.matmul(
            tf.transpose(
                tf.where(
                    atom_in_mol,
                    tf.ones_like(atom_in_mol, dtype=tf.float32),
                    tf.zeros_like(atom_in_mol, dtype=tf.float32))),
            h_pair)

        h_e_history.set_shape([None, 6, 32])
        h_u_history.set_shape([None, 6, 32])
        h_v_history.set_shape([None, 6, 32])
        h_t_history.set_shape([None, 6, 32])
        h_a_history.set_shape([None, 6, 32])

        h_e_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                        tf.multiply(
                            tf.tile(
                                tf.expand_dims(
                                    tf.expand_dims(
                                        tf.where( # (n_bonds, n_mols)
                                            tf.boolean_mask(
                                                bond_in_mol,
                                                tf.reduce_any(
                                                    bond_in_mol,
                                                    axis=1),
                                                axis=0),
                                            tf.ones_like(
                                                tf.boolean_mask(
                                                    bond_in_mol,
                                                    tf.reduce_any(
                                                        bond_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32),
                                            tf.zeros_like(
                                                tf.boolean_mask(
                                                    bond_in_mol,
                                                    tf.reduce_any(
                                                        bond_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32)),
                                        2),
                                    3),
                                [
                                    1,
                                    1,
                                    tf.shape(h_e_history)[1],
                                    tf.shape(h_e)[1]
                                ]),
                            tf.tile( # (n_bonds, n_mols, t, d_e)
                                tf.expand_dims(
                                    h_e_history, # (n_bonds, t, d_e)
                                    1),
                                [1, tf.shape(bond_in_mol)[1], 1, 1])),
                        axis=0)

        h_a_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                        tf.multiply(
                            tf.tile(
                                tf.expand_dims(
                                    tf.expand_dims(
                                        tf.where( # (n_bonds, n_mols)
                                            tf.boolean_mask(
                                                angle_in_mol,
                                                tf.reduce_any(
                                                    angle_in_mol,
                                                    axis=1),
                                                axis=0),
                                            tf.ones_like(
                                                tf.boolean_mask(
                                                    angle_in_mol,
                                                    tf.reduce_any(
                                                        angle_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32),
                                            tf.zeros_like(
                                                tf.boolean_mask(
                                                    angle_in_mol,
                                                    tf.reduce_any(
                                                        angle_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32)),
                                        2),
                                    3),
                                [
                                    1,
                                    1,
                                    tf.shape(h_a_history)[1],
                                    tf.shape(h_a)[1]
                                ]),
                            tf.tile( # (n_bonds, n_mols, t, d_e)
                                tf.expand_dims(
                                    h_a_history, # (n_bonds, t, d_e)
                                    1),
                                [1, tf.shape(angle_in_mol)[1], 1, 1])),
                        axis=0)

        h_t_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                        tf.multiply(
                            tf.tile(
                                tf.expand_dims(
                                    tf.expand_dims(
                                        tf.where( # (n_bonds, n_mols)
                                            tf.boolean_mask(
                                                torsion_in_mol,
                                                tf.reduce_any(
                                                    torsion_in_mol,
                                                    axis=1),
                                                axis=0),
                                            tf.ones_like(
                                                tf.boolean_mask(
                                                    torsion_in_mol,
                                                    tf.reduce_any(
                                                        torsion_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32),
                                            tf.zeros_like(
                                                tf.boolean_mask(
                                                    torsion_in_mol,
                                                    tf.reduce_any(
                                                        torsion_in_mol,
                                                        axis=1),
                                                    axis=0),
                                                dtype=tf.float32)),
                                        2),
                                    3),
                                [
                                    1,
                                    1,
                                    tf.shape(h_t_history)[1],
                                    tf.shape(h_t)[1]
                                ]),
                            tf.tile( # (n_bonds, n_mols, t, d_e)
                                tf.expand_dims(
                                    h_t_history, # (n_bonds, t, d_e)
                                    1),
                                [1, tf.shape(torsion_in_mol)[1], 1, 1])),
                        axis=0)

        h_v_bar_history = tf.reduce_sum( # (n_mols, t, d_e)
                tf.multiply(
                    tf.tile(
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.where( # (n_atoms, n_mols)
                                    atom_in_mol,
                                    tf.ones_like(
                                        atom_in_mol,
                                        dtype=tf.float32),
                                    tf.zeros_like(
                                        atom_in_mol,
                                        dtype=tf.float32)),
                                2),
                            3),
                        [1, 1, tf.shape(h_v_history)[1], tf.shape(h_v)[1]]),
                    tf.tile( # (n_atoms, n_mols, t, d_e)
                        tf.expand_dims(
                            h_v_history, # (n_atoms, t, d_e)
                            1),
                        [1, tf.shape(atom_in_mol)[1], 1, 1])),
                axis=0)

        y = self.d(
            tf.reshape(
                h_v_bar_history,
                [-1, 6 * 32]),
            tf.reshape(
                h_e_bar_history,
                [-1, 6 * 32]),
            tf.reshape(
                h_u_history,
                [-1, 6 * 32]),
            tf.reshape(
                h_a_bar_history,
                [-1, 6 * 32]),
            tf.reshape(
                h_a_bar_history,
                [-1, 6 * 32]))

        y = y + h_pair

        return y

gn = gin.probabilistic.gn_hyper.HyperGraphNet(
    f_e=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    f_a=tf.keras.layers.Dense(32),
    f_t=tf.keras.layers.Dense(32),
    f_v=f_v(),
    f_u=(lambda atoms, adjacency_map, batched_attr_in_mol: \
        tf.tile(
            tf.zeros((1, 32)),
            [
                 tf.math.count_nonzero(batched_attr_in_mol),
                 1
            ]
        )),
    phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    phi_a=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    phi_t=lime.nets.for_gn.ConcatenateThenFullyConnect((32, 'sigmoid', 32)),
    f_r=f_r(),
    repeat=5)


optimizer = tf.keras.optimizers.Adam(1e-5)

for dummy_idx in range(10):
    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds:
        with tf.GradientTape() as tape:
            y_hat = gn(atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

            y = tf.boolean_mask(
                attr,
                attr_in_mol)

            loss = tf.losses.mean_squared_error(
                y,
                y_hat)

        print(loss)
        variables = gn.variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            zip(grad, variables))
