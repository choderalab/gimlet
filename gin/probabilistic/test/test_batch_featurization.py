import pytest
import gin
import tonic
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import pandas as pd

# read data
df = pd.read_csv('data/delaney-processed.csv')
df = df[:1024]
x_array = df[['smiles']].values.flatten()
y_array = \
    df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))

ds = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds = ds.map(
    (lambda atoms, adjacency_map, y: \
        (
            tf.py_function(
                gin.probabilistic.featurization.featurize_atoms,
                [atoms, adjacency_map],
                tf.float32),
            adjacency_map,
            y)))

ds = gin.probabilistic.gn.GraphNet.batch(ds, 256, feature_dimension=11)


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
        return self.d(x)

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

for epoch_idx in range(50):
    for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask in ds:
        with tf.GradientTape() as tape:
            y_hat = gn.call(
                atoms,
                adjacency_map,
                atom_in_mol=atom_in_mol,
                bond_in_mol=bond_in_mol,
                batched_attr_mask=y_mask)

            y = tf.boolean_mask(
                y,
                y_mask)

            loss = tf.losses.mean_squared_error(y, y_hat)

            print(loss)

        variables = gn.variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            zip(grad, variables))
