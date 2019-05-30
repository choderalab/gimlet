import gin
import tonic
import tensorflow as tf
# tf.enable_eager_execution()
import pandas as pd
import numpy as np

# read data
df = pd.read_csv('data/delaney-processed.csv')
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))

ds = gin.i_o.from_smiles.smiles_to_mols_with_attributes(x_array, y_array)


class f_r(tf.keras.Model):
    def __init__(self, config):
        super(f_r, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

    # @tf.function
    def call(self, h_e, h_v, h_u):
        y = self.d(h_u)[0][0]
        return y

class f_v(tf.keras.Model):
    def __init__(self, units):
        super(f_v, self).__init__()
        self.d = tf.keras.layers.Dense(units)

    @tf.function
    def call(self, x):
        return self.d(tf.one_hot(x, 8))

gn = gin.probabilistic.gn.GraphNet(
    f_e=tf.keras.layers.Dense(128),

    f_v=f_v(128),

    f_u=(lambda x, y: tf.zeros((1, 128), dtype=tf.float32)),

    phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_u=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    rho_e_v= lambda h_e, atom_is_connected_to_bonds: tf.reduce_sum(
            tf.where(
                tf.tile(
                    tf.expand_dims(
                        atom_is_connected_to_bonds,
                        2),
                    [1, 1, tf.shape(h_e)[1]]),
                tf.tile(
                    tf.expand_dims(
                        h_e,
                        0),
                    [
                        tf.shape(atom_is_connected_to_bonds)[0], # n_atoms
                        1,
                        1
                    ]),
                tf.zeros((
                    tf.shape(atom_is_connected_to_bonds)[0],
                    tf.shape(h_e)[0],
                    tf.shape(h_e)[1]))),
            axis=1),

    rho_e_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    rho_v_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    f_r=f_r((128, 'tanh', 128, 1)),

    repeat=3)


optimizer = tf.keras.optimizers.Adam(1e-5)
n_epoch = 50
batch_size = 128
batch_idx = 0
loss = 0
tape = tf.GradientTape()
mols = []

for dummy_idx in range(n_epoch):
    for atoms, adjacency_map, y in ds:
        mol = [atoms, adjacency_map]
        mols.append(mol)
        batch_idx += 1

        if batch_idx == batch_size:
            loss = 0
            idx = 0

            def loop_body(idx, loss):
                mol = mols[idx]
                y_hat = gn(mol)
                loss += tf.clip_by_norm(
                    tf.pow(y - y_hat, 2),
                    1e8)
                return idx + 1, loss

            with tf.GradientTape() as tape:
                _, loss= tf.while_loop(
                    lambda idx, loss: tf.less(idx, batch_size),
                    loop_body,
                    [idx, loss],
                    parallel_iterations=batch_size)
                print(loss)

            variables = gn.variables
            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                zip(grad, variables))
            batch_idx = 0
            mols = []
