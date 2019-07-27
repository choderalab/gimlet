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

df = pd.read_csv('data/delaney-processed.csv')
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))
n_samples = y_array.shape[0]
print(n_samples)

ds_all = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds_all = ds_all.shuffle(n_samples)

ds_all = gin.probabilistic.gn.GraphNet.batch(ds_all, 64)

n_global_te = int(0.2 * (n_samples // 64))
ds_global_tr = ds_all.skip(n_global_te)
ds_global_te = ds_all.take(n_global_te)

config_space = {
    'f_e_0': [32, 64, 128, 256],
    'f_v_0': [32, 64, 128, 256],
    'f_u_0': [32, 64, 128, 256],

    'phi_e_0': [32, 64, 128, 256],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128, 256],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128, 256],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128, 256],
    'f_r_a': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_1': [32, 64, 128, 256],

    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2]
}


def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    n_te = int(0.2 * 0.8 * n_samples // 64)
    ds = ds_global_tr.shuffle(int(0.8 * n_samples // 64))

    mse_train = []
    mse_test = []

    for idx in range(5):
        ds_tr = ds.take(idx * n_te).concatenate(
            ds.skip((idx + 1) * n_te).take((4 - idx) * n_te))

        ds_te = ds.skip(idx * n_te).take((idx + 1) * n_te)

        class f_r(tf.keras.Model):
            def __init__(self, config):
                super(f_r, self).__init__()
                self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

            @tf.function
            def call(self, h_e, h_v, h_u,
                    h_e_history, h_v_history, h_u_history,
                    atom_in_mol, bond_in_mol):
                y = self.d(h_u)[0]
                return y

        class f_v(tf.keras.Model):
            def __init__(self, units):
                super(f_v, self).__init__()
                self.d = tf.keras.layers.Dense(units)

            @tf.function
            def call(self, x):
                return self.d(tf.one_hot(x, 8))

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

            f_u=(lambda x, y: tf.zeros((16, point['f_u_0']), dtype=tf.float32)),

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

            f_r=f_r((point['f_r_0'], point['f_r_a'], point['f_r_1'], 1)))

        optimizer = tf.keras.optimizers.Adam(point['learning_rate'])
        n_epoch = 10
        loss = 0
        tape = tf.GradientTape()

        for dummy_idx in range(n_epoch):
            print('=========================')
            print('epoch %s' % dummy_idx)
            for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
                in ds_tr:
                with tf.GradientTape() as tape:
                    y_bar = gn.call(
                        atoms,
                        adjacency_map,
                        atom_in_mol=atom_in_mol,
                        bond_in_mol=bond_in_mol)
                    loss = tf.losses.mean_squared_error(y, y_bar)
                print(loss)
                variables = gn.variables
                grad = tape.gradient(loss, variables)
                optimizer.apply_gradients(
                    zip(grad, variables))

        gn.switch(True)

        # test on train data
        mse_train.append(tf.reduce_mean(
            [tf.losses.mean_squared_error(y, y_bar) \
                for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask\
                in ds_tr]))
        mse_test.append(tf.reduce_mean(
            [tf.losses.mean_squared_error(y, y_bar) \
                for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask\
                in ds_te]))

    class f_r(tf.keras.Model):
        def __init__(self, config):
            super(f_r, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

        @tf.function
        def call(self, h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):
            y = self.d(h_u)[0]
            return y

    class f_v(tf.keras.Model):
        def __init__(self, units):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            return self.d(tf.one_hot(x, 8))

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

        f_u=(lambda x, y: tf.zeros((16, point['f_u_0']), dtype=tf.float32)),

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

        f_r=f_r((point['f_r_0'], point['f_r_a'], point['f_r_1'], 1)))


    optimizer = tf.keras.optimizers.Adam(point['learning_rate'])
    n_epoch = 10
    loss = 0
    tape = tf.GradientTape()

    time0 = time.time()
    for _ in range(n_epoch):
        for epoch in range(n_epoch):
            print('=========================')
            print('epoch %s' % epoch)
            for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
                in ds_global_tr:
                with tf.GradientTape() as tape:
                    y_bar = gn.call(
                        atoms,
                        adjacency_map,
                        atom_in_mol=atom_in_mol,
                        bond_in_mol=bond_in_mol)
                    loss = tf.losses.mean_squared_error(y, y_bar)
                print(loss)
                variables = gn.variables
                grad = tape.gradient(loss, variables)
                optimizer.apply_gradients(
                    zip(grad, variables))

    time1 = time.time()

    mse_global_test = tf.reduce_mean(
        [tf.losses.mean_squared_error(y, y_bar) \
            for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask\
            in ds_global_te])

    mse_train = tf.reduce_mean(mse_train)
    mse_test = tf.reduce_mean(mse_test)

    print(point)
    print('training time %s ' % (time1 - time0))
    print('mse_train %s' % mse_train.numpy())
    print('mse_test %s' % mse_test.numpy())
    print('mse_global_test %s' % mse_global_test.numpy())

    return mse_test


lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
