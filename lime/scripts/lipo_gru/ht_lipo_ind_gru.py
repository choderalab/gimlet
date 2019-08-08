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
from sklearn import metrics
import tensorflow as tf
import gin
import lime
import time
import pandas as pd
import numpy as np
import os


N_EPOCH = 30

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

ds_all = gin.probabilistic.gn.GraphNet.batch(ds_all, 256).cache(
    str(os.getcwd()) + '/temp')

n_batched_samples_total = gin.probabilistic.gn.GraphNet.get_number_batches(
    ds_all)
n_batched_samples_total = int(n_batched_samples_total)
n_global_te = int(0.2 * n_batched_samples_total)
ds_global_tr = ds_all.skip(n_global_te)
ds_global_te = ds_all.take(n_global_te)

config_space = {
    'D_E': [32, 64, 128],
    'D_V': [32, 64, 128],
    'D_U': [32, 64, 128],

    'phi_e_0': [32, 64, 128],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_1': [32, 64, 128],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_1': [32, 64, 128],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_1': [32, 64, 128],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_1': [32, 64, 128],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'gru_unit': [64, 128, 256, 512],

    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2]
}

def init(point):
    global gn
    global optimizer

    class f_r(tf.keras.Model):
        def __init__(
                self,
                config=(
                    point['f_r_0'],
                    point['f_r_a_0'],
                    point['f_r_1'],
                    point['f_r_a_1']
                )):
            super(f_r, self).__init__()
            self.d0 = tf.keras.layers.Dense(config[0], activation=config[1])
            self.d1 = tf.keras.layers.Dense(config[2], activation=config[3])

        @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):

            x = self.d0(h_u)
            x = self.d1(x)

            return x

    class f_e(tf.keras.Model):
        """ Featurization of edges.
        Here we split the $\sigma$ and $\pi$ component of bonds
        into two channels, and featurize them seperately.

        """
        def __init__(
                self,
                d_sigma_units=point['d_sigma_units'],
                d_pi_units=point['d_pi_units'],
                D_E=point['D_E']):

            super(f_e, self).__init__()
            self.D_E = D_E

            # sigma
            self.d_sigma_0 = tf.Variable(
                tf.zeros(
                    shape=(1, d_sigma_units),
                    dtype=tf.float32))
            self.d_sigma_1 = tf.keras.layers.Dense(
                int(self.D_E // 2))

            # pi
            self.d_pi_0 = tf.keras.layers.Dense(
                d_pi_units)
            self.d_pi_1 = tf.keras.layers.Dense(
                int(self.D_E // 2))

        @tf.function
        def call(self, x):
            # determine whether there is $\pi$ component in the bond
            has_pi = tf.greater(
                x,
                tf.constant(1, dtype=tf.float32))

            # calculate the sigma component of the bond
            x_sigma = tf.tile(
                self.d_sigma_1(self.d_sigma_0),
                [tf.shape(x, tf.int64)[0], 1])

            # calculate the pi component of the bond
            x_pi = tf.where(
                has_pi,

                # if has pi:
                self.d_pi_1(
                    self.d_pi_0(
                        tf.math.subtract(
                            x,
                            tf.constant(1, dtype=tf.float32)))),

                # else:
                tf.zeros(
                    shape=(self.D_E // 2, ),
                    dtype=tf.float32))

            x = tf.concat(
                [
                    x_sigma,
                    x_pi
                ],
                axis=1)

            return x


    class f_v(tf.keras.Model):
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            x = tf.one_hot(x, 8)
            x.set_shape([None, 8])
            return self.d(x)

    class phi_u(tf.keras.Model):
        def __init__(
                self,
                config=(
                    point['phi_u_0'],
                    point['phi_u_a_0'],
                    point['phi_u_1'],
                    point['phi_u_a_1']
                ),
                gru_units=point['D_U']):
            super(phi_u, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)
            self.gru = tf.keras.layers.GRU(
                units=gru_units,
                stateful=True)

        @tf.function
        def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
            x = self.d(h_u, h_u_0, h_e_bar, h_v_bar)
            if tf.equal(
                h_u,
                h_u_0):
                self.gru.reset_states()

            x = self.gru(
                tf.expand_dims(
                    x,
                    1))

            return x

    class phi_v(tf.keras.Model):
        def __init__(
                self,
                config=(
                    point['phi_v_0'],
                    point['phi_v_a_0'],
                    point['phi_v_1'],
                    point['phi_v_a_1']
                ),
                gru_units=point['D_V']):
            super(phi_v, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)
            self.gru = tf.keras.layers.GRU(
                units=gru_units,
                stateful=True)

        @tf.function
        def call(self, h_v, h_v_0, h_e_bar_i, h_u_i):
            x = self.d(h_v, h_v_0, h_e_bar_i, h_u_i)
            if tf.equal(
                h_v,
                h_v_0):
                self.gru.reset_states()

            x = self.gru(
                tf.expand_dims(
                    x,
                    1))

            return x


    class phi_e(tf.keras.Model):
        def __init__(
                self,
                config=(
                    point['phi_e_0'],
                    point['phi_e_a_0'],
                    point['phi_e_1'],
                    point['phi_e_a_1']
                ),
                gru_units=point['D_E']):
            super(phi_e, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)
            self.gru = tf.keras.layers.GRU(
                units=gru_units,
                stateful=True)

        @tf.function
        def call(self, h_e, h_e_0, h_left, h_right, h_u_i):
            x = self.d(h_e, h_e_0, h_left, h_right, h_u_i)
            if tf.equal(
                h_e,
                h_e_0):
                self.gru.reset_states()

            x = self.gru(
                tf.expand_dims(
                    x,
                    1))

            return x

    gn = gin.probabilistic.gn.GraphNet(
        f_e=f_e(),

        f_v=f_v(),

        f_u=(lambda atoms, adjacency_map, batched_attr_in_mol: \
            tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(batched_attr_in_mol),
                     1
                ]
            )),

        phi_e=phi_e(),

        phi_v=phi_v(),

        phi_u=phi_u(),

        f_r=f_r(),

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
                batched_attr_in_mol=y_mask)

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
                batched_attr_in_mol=y_mask)

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
            batched_attr_in_mol=y_mask)

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

    print(point, flush=True)
    print('training time %s ' % (time1 - time0), flush=True)
    print('mse_train %s +- %s' % (np.mean(mse_train), np.std(mse_train)),
        flush=True)
    print('r2_train %s +- %s' % (np.mean(r2_train), np.std(r2_train)),
        flush=True)
    print('mse_test %s +- %s' % (np.mean(mse_train), np.std(mse_train)),
        flush=True)
    print('r2_test %s +- %s' % (np.mean(r2_test), np.std(r2_test)),
        flush=True)
    print('mse_global_test %s' % mse_global_test.numpy(),
        flush=True)
    print('r2_global_test %s ' % r2_global_test, flush=True)

    return mse_test


lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
