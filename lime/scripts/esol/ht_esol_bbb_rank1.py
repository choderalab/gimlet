# =============================================================================
# imports
# =============================================================================
from sklearn import metrics
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import gin
import lime
import pandas as pd
import numpy as np
import math


df = pd.read_csv('delaney-processed.csv')
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
df = df[df['smiles'].str.len() > 1]
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))

n_samples = y_array.shape[0]

# ds_all = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds = gin.i_o.from_smiles.to_mols(x_array)
ds_attr = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(y_array, tf.float32))

ds_all = tf.data.Dataset.zip((ds, ds_attr))

ds_all = ds_all.map(
    lambda mol, attr: (mol[0], mol[1], attr))

ds_all = gin.probabilistic.gn.GraphNet.batch(ds_all, 128).cache(
    str(os.getcwd()) + '/temp')

ds_all = ds_all.shuffle(n_samples, seed=2666)

n_batches = int(gin.probabilistic.gn.GraphNet.get_number_batches(ds_all))
n_te = n_batches // 10

ds_te = ds_all.take(n_te)
ds_vl = ds_all.skip(n_te).take(n_te)
ds_tr = ds_all.skip(2 * n_te)

config_space = {
    'D_V': [16, 32],
    'D_E': [16, 32],
    'D_U': [16, 32],

    'phi_e_0': [16, 32],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [16, 32],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [16, 32],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [16, 32],
    'f_r_1': [16, 32],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}


def init(point):
    global gn_mu
    global gn_sigma
    global gn_theta
    global gn_u
    global optimizer_mu
    global optimizer_sigma
    global optimizer_u

    class f_v(tf.keras.Model):
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            x = tf.one_hot(
                tf.cast(
                    x,
                    tf.int64),
                8)

            x.set_shape([None, 8])
            return self.d(x)

    class f_r(tf.keras.Model):
        def __init__(self, config=[
          point['f_r_0'],
          point['f_r_a_0'],
          point['f_r_1'],
          point['f_r_a_1'], 1],

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

            y = tf.reshape(y, [-1])

            return y

    gn_theta = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(point['D_E']),

        f_v=f_v(),

        f_u=lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(
                         batched_attr_in_mol),
                    1
                ]
            ),

        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'],
             point['phi_e_a_0'],
             point['D_E'],
             point['phi_e_a_1'])),

        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'],
             point['phi_v_a_0'],
             point['D_V'],
             point['phi_v_a_1'])),

        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'],
             point['phi_u_a_0'],
             point['D_U'],
             point['phi_u_a_1'])),

        f_r=f_r(),

        repeat=5)


    gn_sigma = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(point['D_E']),

        f_v=f_v(),

        f_u=lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(
                         batched_attr_in_mol),
                    1
                ]
            ),

        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'],
             point['phi_e_a_0'],
             point['D_E'],
             point['phi_e_a_1'])),

        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'],
             point['phi_v_a_0'],
             point['D_V'],
             point['phi_v_a_1'])),

        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'],
             point['phi_u_a_0'],
             point['D_U'],
             point['phi_u_a_1'])),

        f_r=f_r(),

        repeat=5)

    gn_u = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(point['D_E']),

        f_v=f_v(),

        f_u=lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(
                         batched_attr_in_mol),
                    1
                ]
            ),

        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'],
             point['phi_e_a_0'],
             point['D_E'],
             point['phi_e_a_1'])),

        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'],
             point['phi_v_a_0'],
             point['D_V'],
             point['phi_v_a_1'])),

        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'],
             point['phi_u_a_0'],
             point['D_U'],
             point['phi_u_a_1'])),

        f_r=f_r(),

        repeat=5)


    gn_mu = gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(point['D_E']),

        f_v=f_v(),

        f_u=lambda atoms, adjacency_map, batched_attr_in_mol: tf.tile(
                tf.zeros((1, point['D_U'])),
                [
                     tf.math.count_nonzero(
                         batched_attr_in_mol),
                    1
                ]
            ),

        phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_e_0'],
             point['phi_e_a_0'],
             point['D_E'],
             point['phi_e_a_1'])),

        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'],
             point['phi_v_a_0'],
             point['D_V'],
             point['phi_v_a_1'])),

        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'],
             point['phi_u_a_0'],
             point['D_U'],
             point['phi_u_a_1'])),

        f_r=f_r(),

        repeat=5)


    optimizer_mu = tf.keras.optimizers.Adam(1e-5)
    optimizer_sigma = tf.keras.optimizers.Adam(1e-5)
    optimizer_u = tf.keras.optimizers.Adam(1e-5)

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for x in ds_tr:
        gn_mu(x[0], x[1])
        gn_sigma(x[0], x[1])
        gn_theta(x[0], x[1])
        gn_u(x[0], x[1])
        break

    gn_sigma.set_weights(
        [
            tf.cond(
                lambda: tf.reduce_all(tf.equal(weight, tf.constan(0,
                  dtype=tf.float32))),
                lambda: tf.random.normal(shape=tf.shape(weight), stddev=1e-3),
                lambda: weight)\
            for weight in gn_sigma.get_weights()
        ])

    gn_u.set_weights(
        [
            tf.random.normal(
                shape=tf.shape(weight),
                stddev=1e-3)\
            for weight in gn_u.get_weights()
        ])

    for dummy_idx in range(60):
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
                in ds_tr:

            mu = [tf.convert_to_tensor(v) for v in gn_mu.get_weights()]
            sigma = [tf.convert_to_tensor(v) for v in gn_sigma.get_weights()]
            u = [tf.convert_to_tensor(v) for v in gn_u.get_weights()]

            epsilon = [
                        tf.random.normal(
                                shape=tf.shape(sigma[idx]))\
                        for idx in range(len(gn_theta.get_weights()))
                    ]

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(sigma)
                tape.watch(mu)
                tape.watch(u)
                # $$
                # w_k = L_k^{-1} u
                # $$
                w = [
                    tf.reshape(
                        tf.matmul(
                            tf.linalg.diag(
                                tf.pow(
                                    tf.reshape(
                                        sigma[idx],
                                        [-1]),
                                    tf.constant(-1, dtype=tf.float32))),
                            tf.reshape(
                                u[idx],
                                [-1, 1])),
                        [-1])\
                    for idx in range(len(sigma))
                    ]

                # $$
                # \gamma = \sqrt{1 + ||w_k||^2} - 1
                # $$
                gamma = [
                    tf.math.subtract(
                        tf.math.sqrt(
                            tf.math.add(
                                tf.constant(1, dtype=tf.float32),
                                tf.reduce_sum(
                                    tf.square(
                                        w[idx])))),
                        tf.constant(1, dtype=tf.float32))\
                    for idx in range(len(w))
                    ]
                

                # $$
                # L = L_k (I + \frac{ww^T}{||w||^2})
                # $$
                l = [
                        tf.matmul(
                            tf.linalg.diag(
                                tf.reshape(
                                    sigma[idx],
                                    [-1])),
                            tf.math.add(
                                tf.eye(
                                    tf.shape(w[idx])[0]),
                                tf.math.multiply(
                                    gamma[idx],
                                    tf.math.divide_no_nan(
                                        tf.matmul(
                                            tf.reshape(
                                                w[idx],
                                                [-1, 1]),
                                            tf.reshape(
                                                w[idx],
                                                [1, -1])),
                                        tf.reduce_sum(
                                            tf.square(
                                                w[idx]))))))\
                    for idx in range(len(w))
                ]
                

                # $$
                # l^{-1} = (I - \frac{\gamma}{\gamma + 1}
                # \frac{ww^T}{||w||^2}) \sigma^{-1}
                # $$
                l_inverse = [
                    tf.matmul(
                        tf.math.subtract(
                            tf.eye(
                                tf.shape(l[idx])[0]),
                            tf.math.multiply(
                                tf.math.divide_no_nan(
                                    gamma[idx],
                                    gamma[idx] + 1),
                                tf.math.divide_no_nan(
                                    tf.matmul(
                                        tf.reshape(
                                            w[idx],
                                            [-1, 1]),
                                        tf.reshape(
                                            w[idx],
                                            [1, -1])),
                                    tf.reduce_sum(
                                        tf.square(
                                            w[idx]))))),
                            tf.linalg.diag(
                                tf.reshape(
                                    tf.pow(
                                        sigma[idx],
                                        -1),
                                    [-1])))\
                    for idx in range(len(l))
                ]

                # $$
                # \theta = L  epsilon + \mu
                # $$
                theta = [
                    tf.math.add(
                        tf.reshape(
                            tf.matmul(
                                l[idx],
                                tf.reshape(
                                    epsilon[idx],
                                    [-1, 1])),
                            tf.shape(epsilon[idx])),

                        mu[idx])\
                    for idx in range(len(l))
                ]

                # $$
                # k = l l^T
                # $$
                k = [
                    tf.matmul(
                        l[idx],
                        tf.transpose(
                            l[idx]))\
                    for idx in range(len(l))
                ]


                k_inverse = [
                    tf.matmul(
                        l_inverse[idx],
                        tf.transpose(
                            l_inverse[idx]))\
                    for idx in range(len(l_inverse))
                ]
                
                det_k = [
                    tf.multiply(
                        tf.squeeze(tf.math.add(
                            tf.eye(tf.shape(k[idx])[0]),
                            tf.matmul(
                                tf.matmul(
                                    tf.reshape(
                                        u[idx],
                                        [1, -1]),
                                    k_inverse[idx]),
                                tf.reshape(
                                    u[idx],
                                    [-1 ,1])))),
                        tf.math.log(
                            tf.reduce_sum(
                                tf.math.exp(
                                    tf.math.square(
                                        sigma[idx])))))\
                  for idx in range(len(k))]

                tape.watch(theta)

                f = [
                        tf.math.add(
                            tf.math.add(
                                tf.math.add(
                                    tf.math.multiply(
                                        tf.math.multiply(
                                            tf.constant(-0.5, dtype=tf.float32),
                                            tf.cast(
                                                tf.shape(
                                                    k[idx])[0],
                                                tf.float32)),
                                        tf.math.log(
                                            tf.constant(2, dtype=tf.float32),
                                            math.pi)),
                                    tf.math.multiply(
                                        tf.constant(-0.5, dtype=tf.float32),
                                        tf.math.log(
                                            det_k[idx]))),
                                tf.math.multiply(
                                    tf.constant(-0.5, dtype=tf.float32),
                                    tf.matmul(
                                        tf.matmul(
                                            tf.reshape(
                                                tf.math.subtract(
                                                    theta[idx],
                                                    mu[idx]),
                                                [1, -1]),
                                            k_inverse[idx]),
                                        tf.reshape(
                                            tf.subtract(
                                                theta[idx],
                                                mu[idx]),
                                            [-1, 1])))),

                            tf.math.multiply(
                                tf.constant(1e-10, dtype=tf.float32),
                                tf.reduce_sum(tf.math.square(theta[idx]))))\
                            for idx in range(len(gn_theta.get_weights()))
                ]
            

            g_theta = tape.gradient(f, theta)
            g_mu = tape.gradient(f, mu)
            g_sigma = tape.gradient(f, sigma)
            g_u = tape.gradient(f, u)

            d_theta_d_sigma = tape.gradient(theta, sigma)
            d_theta_d_u = tape.gradient(theta, u)
            
            
            print(f)

            gn_theta.set_weights(theta)

            with tf.GradientTape() as tape:
                y_hat = gn_theta(
                    atoms,
                    adjacency_map,
                    atom_in_mol=atom_in_mol,
                    bond_in_mol=bond_in_mol,
                    batched_attr_in_mol=y_mask)

                y_ = tf.boolean_mask(
                    y,
                    y_mask)


                loss = tf.losses.mean_squared_error(y_, y_hat)

            g = tape.gradient(loss, gn_theta.variables)

            g_mu_ = [g[idx] + g_theta[idx] + g_mu[idx] for idx in range(len(g))]
            g_sigma_ = [g[idx] * d_theta_d_sigma[idx] + g_theta[idx] * d_theta_d_sigma[idx] + g_sigma[idx] for idx in range(len(g))]
            g_u_ = [g[idx] * d_theta_d_u[idx] + g_theta[idx] * d_theta_d_u[idx] + g_u[idx] for idx in range(len(g))]

            optimizer_mu.apply_gradients(zip(g_mu_, gn_mu.variables))
            optimizer_sigma.apply_gradients(zip(g_sigma_, gn_sigma.variables))
            optimizer_u.apply_gradients(zip(g_u_, gn_u.variables))

    y_true_tr = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, ], dtype=tf.float32)

    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask \
            in ds_tr:

        y_hat = gn_mu(
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

        y_hat = gn_mu(
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

        y_hat = gn_mu(
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
