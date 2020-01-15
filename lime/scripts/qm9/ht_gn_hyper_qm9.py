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

mols_ds = gin.i_o.from_sdf.to_ds('gdb9.sdf', has_charge=False)

attr_ds = pd.read_csv('gdb9.sdf.csv').values[:, 1:].astype(np.float32)

attr_ds = attr_ds / np.linalg.norm(attr_ds, axis=0) \
    - np.std(attr_ds, axis=0)

attr_ds = tf.data.Dataset.from_tensor_slices(attr_ds)

ds = tf.data.Dataset.zip((mols_ds, attr_ds))

ds = ds.map(
    lambda mol, attr: (mol[0], mol[1], mol[2], attr))

ds = gin.probabilistic.gn_hyper.HyperGraphNet.batch(
    ds,
    128,
    attr_dimension=19).cache(
        str(os.getcwd()) + 'temp')

n_batches = int(gin.probabilistic.gn.GraphNet.get_number_batches(ds))
n_te = n_batches // 10

ds_te = ds.take(n_te)
ds_vl = ds.skip(n_te).take(n_te)
ds_tr = ds.skip(2 * n_te)

config_space = {
    'D_V': [16, 32, 64, 128, 256],
    'D_E': [16, 32, 64, 128, 256],
    'D_A': [16, 32, 64, 128, 256],
    'D_T': [16, 32, 64, 128, 256],
    'D_U': [16, 32, 64, 128, 256],


    'phi_e_0': [32, 64, 128],
    'phi_e_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_a_0': [32, 64, 128],
    'phi_a_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_a_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_t_0': [32, 64, 128],
    'phi_t_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_t_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'f_e_0': [32, 64, 128],
    'f_e_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'f_e_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'f_r': [32, 64, 128],
    'f_r_a': ['elu', 'relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}

def init(point):
    global gn
    global optimizer

    class f_v(tf.keras.Model):
        """ Featurization of nodes.
        Here we simply featurize atoms using one-hot encoding.

        """
        def __init__(self, units=point['D_V']):
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
        def __init__(self, units=point['f_r'], f_r_a=point['f_r_a']):
            super(f_r, self).__init__()
            self.d_k = tf.keras.layers.Dense(units)
            self.d_q = tf.keras.layers.Dense(units)

            self.d_v_0 = tf.keras.layers.Dense(units, activation=f_r_a)
            self.d_v_1 = tf.keras.layers.Dense(units)

            self.d_pair_0 = tf.keras.layers.Dense(units, activation=f_r_a)
            self.d_pair_1 = tf.keras.layers.Dense(19)

            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect((units, f_r_a, units, 19))

            self.units = units
            self.d_v = point['D_V']
            self.d_e = point['D_E']
            self.d_a = point['D_A']
            self.d_t = point['D_T']
            self.d_u = point['D_U']

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
                    [1, 1, self.units]),
                self.d_v_1(self.d_v_0(distance)))

            # (n_atoms, n_atoms, units)
            k = tf.multiply(
                tf.tile(
                    tf.expand_dims(
                        per_mol_mask,
                        2),
                    [1, 1, self.units]),
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
                    [1, 1, self.units]),
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

            h_e_history.set_shape([None, 6, self.d_e])
            h_u_history.set_shape([None, 6, self.d_u])
            h_v_history.set_shape([None, 6, self.d_v])
            h_t_history.set_shape([None, 6, self.d_t])
            h_a_history.set_shape([None, 6, self.d_a])

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
                    [-1, 6 * self.d_v]),
                tf.reshape(
                    h_e_bar_history,
                    [-1, 6 * self.d_e]),
                tf.reshape(
                    h_u_history,
                    [-1, 6 * self.d_u]),
                tf.reshape(
                    h_a_bar_history,
                    [-1, 6 * self.d_a]),
                tf.reshape(
                    h_a_bar_history,
                    [-1, 6 * self.d_a]))

            y = y + h_pair

            return y

    gn = gin.probabilistic.gn_hyper.HyperGraphNet(
        f_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['f_e_0'], point['f_e_a_0'], point['D_E'], point['f_e_a_1'])),
        f_a=tf.keras.layers.Dense(point['D_A']),
        f_t=tf.keras.layers.Dense(point['D_T']),
        f_v=f_v(),
        f_u=(lambda atoms, adjacency_map, batched_attr_in_mol: \
            tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(batched_attr_in_mol),
                     1
                ]
            )),
        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'], point['phi_e_a_0'], point['D_E'], point['phi_e_a_1'])),
        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'], point['phi_u_a_0'], point['D_U'], point['phi_u_a_1'])),
        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'], point['phi_v_a_0'], point['D_V'], point['phi_v_a_1'])),
        phi_a=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_a_0'], point['phi_a_a_0'], point['D_A'], point['phi_a_a_1'])),
        phi_t=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_t_0'], point['phi_t_a_0'], point['D_T'], point['phi_t_a_1'])),
        f_r=f_r(),
        repeat=5)

    optimizer = tf.keras.optimizers.Adam(1e-5)

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(10):
        print(dummy_idx)
        for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_tr:
            with tf.GradientTape() as tape:
                y_hat = gn(atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

                y = tf.boolean_mask(
                    attr,
                    attr_in_mol)

                loss = tf.reduce_sum(tf.losses.mean_squared_error(
                    y,
                    y_hat))

            variables = gn.variables
            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                zip(grad, variables))

    y_true_tr = -1. * tf.ones([1, 19], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, 19], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, 19], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, 19], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, 19], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, 19], dtype=tf.float32)

    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_tr:

        y_hat = gn(atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        y = tf.boolean_mask(
            attr,
            attr_in_mol)

        y_true_tr = tf.concat([y_true_tr, y], axis=0)
        y_pred_tr = tf.concat([y_pred_tr, y_hat], axis=0)

    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_te:

        y_hat = gn(atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        y = tf.boolean_mask(
            attr,
            attr_in_mol)

        y_true_te = tf.concat([y_true_te, y], axis=0)
        y_pred_te = tf.concat([y_pred_te, y_hat], axis=0)

    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_vl:

        y_hat = gn(atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        y = tf.boolean_mask(
            attr,
            attr_in_mol)

        y_true_vl = tf.concat([y_true_vl, y], axis=0)
        y_pred_vl = tf.concat([y_pred_vl, y_hat], axis=0)

    r2_tr = metrics.r2_score(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())
    rmse_tr = metrics.mean_squared_error(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())

    r2_vl = metrics.r2_score(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())
    rmse_vl = metrics.mean_squared_error(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())

    r2_te = metrics.r2_score(y_true_te[1:].numpy(), y_pred_te[1:].numpy())
    rmse_te = metrics.mean_squared_error(y_true_te[1:].numpy(), y_pred_te[1:].numpy())

    print(point, flush=True)
    print(r2_tr, flush=True)
    print(rmse_tr, flush=True)
    print(r2_vl, flush=True)
    print(rmse_vl, flush=True)
    print(r2_te, flush=True)
    print(rmse_te, flush=True)

    return rmse_vl

lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
