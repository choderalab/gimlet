"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, Nicea Research, and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import tensorflow as tf
import gin
import lime
import time
import pandas as pd
import numpy as np
from sklearn import metrics

N_EPOCH = 80

df = pd.read_csv('data/Lipophilicity.csv')
df = df[~df['smiles'].str.contains('B')]
df = df[~df['smiles'].str.contains('\%')]
df = df[~df['smiles'].str.contains('\.')]
df = df[~df['smiles'].str.contains('Se')]
df = df[~df['smiles'].str.contains('Si')]
df = df[~df['smiles'].str.contains('S@@')]
df = df[~df['smiles'].str.contains('6')]
df = df[~df['smiles'].str.contains('7')]
df = df[~df['smiles'].str.contains('8')]
df = df[~df['smiles'].str.contains('9')]
df = df[~df['smiles'].str.contains('\+')]
df = df[~df['smiles'].str.contains('\-')]

x_array = df[['smiles']].values.flatten()
y_array = df[['exp']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))
n_samples = y_array.shape[0]

ds_all = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds_all = ds_all.shuffle(n_samples)

ds_all, n_batched_samples_total \
    = gin.probabilistic.gn.GraphNet.batch(ds_all, 256)

n_batched_samples_total = int(n_batched_samples_total)
n_global_te = int(0.2 * n_batched_samples_total)
ds_global_tr = ds_all.skip(n_global_te)
ds_global_te = ds_all.take(n_global_te)

config_space = {
    'f_e_0': [32, 64, 128],
    'f_v_0': [32, 64, 128],
    'f_u_0': [32, 64, 128],

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
    'f_r_a': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_1': [32, 64, 128],

    'gru_unit': [64, 128, 256, 512],

    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2]
}

def init(point):
    global gn
    global optimizer

    class f_r(tf.keras.Model):
        def __init__(self, unit, config):
            super(f_r, self).__init__()
            self.gru_e = tf.keras.layers.GRU(
                unit,
                return_state=True,
                unroll=False)
            self.gru_v = tf.keras.layers.GRU(
                unit,
                return_state=True,
                unroll=False)
            self.gru_u = tf.keras.layers.GRU(
                unit,
                return_state=True,
                unroll=False)

            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

        @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):

            h_e_history.set_shape([None, None, point['f_e_0']])
            h_v_history.set_shape([None, None, point['f_v_0']])
            h_u_history.set_shape([None, None, point['f_u_0']])

            y_e = self.gru_e(h_e_history)[1]
            y_v = self.gru_v(h_v_history)[1]
            y_u = self.gru_u(h_u_history)[1]

            y_e_bar = tf.reduce_sum(
                            tf.multiply(
                                tf.tile(
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
                                    [1, 1, tf.shape(y_e)[1]]),
                                tf.tile( # (n_bonds, n_mols, d_e)
                                    tf.expand_dims(
                                        y_e, # (n_bonds, d_e)
                                        1),
                                    [1, tf.shape(bond_in_mol)[1], 1])),
                            axis=0)


            y_v_bar = tf.reduce_sum(
                    tf.multiply(
                        tf.tile(
                            tf.expand_dims(
                                tf.where( # (n_bonds, n_mols)
                                    atom_in_mol,
                                    tf.ones_like(
                                        atom_in_mol,
                                        dtype=tf.float32),
                                    tf.zeros_like(
                                        atom_in_mol,
                                        dtype=tf.float32)),
                                2),
                            [1, 1, tf.shape(y_v)[1]]),
                        tf.tile( # (n_bonds, n_mols, d_e)
                            tf.expand_dims(
                                y_v, # (n_bonds, d_e)
                                1),
                            [1, tf.shape(atom_in_mol)[1], 1])),
                    axis=0)

            y = self.d(y_e_bar, y_v_bar, y_u)

            y = tf.reshape(y, [-1])

            return y

    class f_v(tf.keras.Model):
        def __init__(self, units):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            x = tf.one_hot(x, 8)
            x.set_shape([None, 8])
            return self.d(x)

    class phi_u(tf.keras.Model):
        def __init__(self, config):
            super(phi_u, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

        @tf.function
        def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
            return self.d(h_u, h_u_0, h_e_bar, h_v_bar)

    gn = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(point['f_e_0']),

        f_v=f_v(point['f_v_0']),

        f_u=(lambda atoms, adjacency_map, batched_attr_mask: \
            tf.tile(
                tf.zeros((1, point['f_u_0'])),
                [
                     tf.math.count_nonzero(batched_attr_mask),
                     1
                ]
            )),

        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'],
             point['phi_e_a_0'],
             point['f_e_0'],
             point['phi_e_a_1'])),

        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'],
             point['phi_v_a_0'],
             point['f_v_0'],
             point['phi_v_a_1'])),

        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'],
             point['phi_u_a_0'],
             point['f_u_0'],
             point['phi_u_a_1'])),

        f_r=f_r(point['gru_unit'],
            (point['f_r_0'], point['f_r_a'], point['f_r_1'], 1)),

        repeat=5)

    optimizer = tf.keras.optimizers.Adam(point['learning_rate'])

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    n_te = int(0.2 * 0.8 * n_batched_samples_total)
    ds = ds_global_tr.shuffle(int(0.8 * n_batched_samples_total))

    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []

    for idx in range(5):
        init(point)

        y_true_train = tf.constant([-1], dtype=tf.float32)
        y_pred_train = tf.constant([-1], dtype=tf.float32)
        y_true_test = tf.constant([-1], dtype=tf.float32)
        y_pred_test = tf.constant([-1], dtype=tf.float32)

        ds_tr = ds.take(idx * n_te).concatenate(
            ds.skip((idx + 1) * n_te).take((4 - idx) * n_te))

        ds_te = ds.skip(idx * n_te).take(n_te)

        for dummy_idx in range(N_EPOCH):
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


        gn.switch(True)

        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_te:

            y_hat = gn(
                atoms,
                adjacency_map,
                atom_in_mol=atom_in_mol,
                bond_in_mol=bond_in_mol,
                batched_attr_mask=y_mask)

            y = tf.boolean_mask(
                y,
                y_mask)

            y_true_test = tf.concat(
                [
                    y_true_test,
                    tf.reshape(y, [-1])
                ],
                axis=0)

            y_pred_test = tf.concat(
                [
                    y_pred_test,
                    tf.reshape(y_hat, [-1])
                ],
                axis=0)

        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_tr:

            y_hat = gn(
                atoms,
                adjacency_map,
                atom_in_mol=atom_in_mol,
                bond_in_mol=bond_in_mol,
                batched_attr_mask=y_mask)

            y = tf.boolean_mask(
                y,
                y_mask)

            y_true_train = tf.concat(
                [
                    y_true_train,
                    tf.reshape(y, [-1])
                ],
                axis=0)

            y_pred_train = tf.concat(
                [
                    y_pred_train,
                    tf.reshape(y_hat, [-1])
                ],
                axis=0)

        y_true_train = y_true_train[1:]
        y_pred_train = y_pred_train[1:]
        y_true_test = y_true_test[1:]
        y_pred_test = y_pred_test[1:]

        r2_train.append(metrics.r2_score(y_true_train, y_pred_train))
        r2_test.append(metrics.r2_score(y_true_test, y_pred_test))
        mse_train.append(
            tf.losses.mean_squared_error(
                y_true_train,
                y_pred_train).numpy())
        mse_test.append(
            tf.losses.mean_squared_error(
                y_true_test,
                y_pred_test).numpy())



    y_true_global_test = tf.constant([-1], dtype=tf.float32)
    y_pred_global_test = tf.constant([-1], dtype=tf.float32)

    init(point)

    time0 = time.time()

    for dummy_idx in range(N_EPOCH):
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_global_tr:
            with tf.GradientTape() as tape:
                y_hat = gn.call(
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

    time1 = time.time()

    gn.switch(True)

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
        in ds_global_te:

        y_hat = gn(
            atoms,
            adjacency_map,
            atom_in_mol=atom_in_mol,
            bond_in_mol=bond_in_mol,
            batched_attr_mask=y_mask)

        y = tf.boolean_mask(
            y,
            y_mask)

        y_true_global_test = tf.concat(
            [
                y_true_global_test,
                tf.reshape(y, [-1])
            ],
            axis=0)

        y_pred_global_test = tf.concat(
            [
                y_pred_global_test,
                tf.reshape(y_hat, [-1])
            ],
            axis=0)

    y_true_global_test = y_true_global_test[1:]
    y_pred_global_test = y_pred_global_test[1:]

    mse_global_test = tf.losses.mean_squared_error(y_true_global_test,
        y_pred_global_test)
    r2_global_test = metrics.r2_score(y_true_global_test,
        y_pred_global_test)

    print(point)
    print('training time %s ' % (time1 - time0))
    print('mse_train %s +- %s' % (np.mean(mse_train), np.std(mse_train)))
    print('r2_train %s +- %s' % (np.mean(r2_train), np.std(r2_train)))
    print('mse_test %s +- %s' % (np.mean(mse_train), np.std(mse_train)))
    print('r2_test %s +- %s' % (np.mean(r2_test), np.std(r2_test)))
    print('mse_global_test %s' % mse_global_test.numpy())
    print('r2_global_test %s ' % r2_global_test)

    return mse_test


lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
