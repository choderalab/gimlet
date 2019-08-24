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

attr_ds = tf.data.Dataset.from_tensor_slices(
    pd.read_csv('gdb9.sdf.csv').values[:, 1:].astype(np.float32))

mols_ds = mols_ds.map(
    lambda atoms, adjacency_map, coordinates, charges:\
        (tf.cast(atoms, tf.float32), adjacency_map, coordinates))

ds = tf.data.Dataset.zip((mols_ds, attr_ds))



ds = ds.map(
    lambda mol, attr:\
        (
            tf.concat(
                [
                    tf.expand_dims(mol[0], 1),
                    mol[2]
                ],
                axis=1),
            mol[1],
            attr
        )).shuffle(
            buffer_size=10000,
            seed=2666)

ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 128, attr_dimension=19, feature_dimension=4, atom_dtype=tf.float32).cache(
    str(os.getcwd()) + '/temp')


n_batches = int(gin.probabilistic.gn.GraphNet.get_number_batches(ds))
n_te = n_batches // 10

ds_te = ds.take(n_te)
ds_vl = ds.skip(n_te).take(n_te)
ds_tr = ds.skip(2 * n_te)

config_space = {
    'D_V': [16, 32, 64, 128, 256],
    'D_E': [16, 32, 64, 128, 256],
    'D_U': [16, 32, 64, 128, 256],


    'phi_e_0': [32, 64, 128],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128],
    'f_r_1': [32, 64, 128],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}


def init(point):
    global gn
    global optimizer

    class f_v(tf.keras.Model):
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            atom_type = x[:, 0]
            coordinates = x[:, 1:]

            x = tf.one_hot(
                tf.cast(
                    atom_type,
                    tf.int64),
                8)

            x.set_shape([None, 8])
            return self.d(x)

    f_e = tf.keras.layers.Dense(point['D_E'])

    f_u = lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
            tf.zeros((1, point['D_U'])),
            [
                 tf.math.count_nonzero(
                     batched_attr_in_mol),
                1
            ]
        )

    phi_v = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_v_0'], 
         point['phi_v_a_0'], 
         point['D_V'], 
         point['phi_v_a_1']))

    phi_e = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_e_0'], 
         point['phi_e_a_0'], 
         point['D_E'], 
         point['phi_e_a_1']))

    phi_u = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_u_0'], 
         point['phi_u_a_0'], 
         point['D_U'], 
         point['phi_u_a_1']))

    class f_r(tf.keras.Model):
        def __init__(self, config=[
          point['f_r_0'], 
          point['f_r_a_0'], 
          point['f_r_1'], 
          point['f_r_a_1'], 19],
          
          d_e=point['D_E'],
          d_u=point['D_U'],
          d_v=point['D_V']):
            super(f_r, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)
            self.f_r_1 = config[2]
            self.d_e = d_e
            self.d_u = d_u
            self.d_v = d_v

        # @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):

            h_e_history.set_shape([None, 6, self.d_e])
            h_u_history.set_shape([None, 6, self.d_u])
            h_v_history.set_shape([None, 6, self.d_v])

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
                    [-1, 6 * self.d_u]))

            # y = tf.reshape(y, [-1])

            return y


    gn = gin.probabilistic.gn.GraphNet(
        f_e=f_e,

        f_v=f_v(),

        f_u=f_u,

        phi_e=phi_e,

        phi_v=phi_v,

        phi_u=phi_u,

        f_r=f_r(),

        repeat=5)

    optimizer = tf.keras.optimizers.Adam(1e-3)


def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(30):
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
                in ds_tr:

            with tf.GradientTape() as tape:
                y_hat = gn(
                    atoms,
                    adjacency_map,
                    atom_in_mol=atom_in_mol,
                    bond_in_mol=bond_in_mol,
                    batched_attr_in_mol=y_mask)


                y = tf.boolean_mask(
                    y,
                    y_mask)

                loss = tf.losses.mean_squared_error(y, y_hat)

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

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_tr:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)


        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_tr = tf.concat([y_true_tr, y], axis=0)
        y_pred_tr = tf.concat([y_pred_tr, y_hat], axis=0)

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_vl:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_vl = tf.concat([y_true_vl, y], axis=0)
        y_pred_vl = tf.concat([y_pred_vl, y_hat], axis=0)

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_te:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_in_mol=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_te = tf.concat([y_true_te, y], axis=0)
        y_pred_te = tf.concat([y_pred_te, y_hat], axis=0)

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
