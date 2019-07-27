import gin
import lime
import logging
import tensorflow as tf
# tf.enable_eager_execution()
import pandas as pd
import numpy as np
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = tf.get_logger()

# read data
df = pd.read_csv('data/delaney-processed.csv')
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))

ds = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds = gin.probabilistic.gn.GraphNet.batch(ds, 256)

class f_r(tf.keras.Model):
    def __init__(self, gru_unit, d_config):
        super(f_r, self).__init__()

        # edges
        self.gru_e = tf.keras.layers.GRU(
            gru_unit,
            return_state=True,
            kernel_initializer='RandomUniform')

        # vertices
        self.gru_v = tf.keras.layers.GRU(
            gru_unit,
            return_state=True,
            kernel_initializer='RandomUniform')

        # universal
        self.gru_u = tf.keras.layers.GRU(
            gru_unit,
            return_state=True,
            kernel_initializer='RandomUniform')

        self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(d_config)

    @tf.function
    def call(self, h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history,
            atom_in_mol, bond_in_mol):

        h_e_history.set_shape([None, None, 128])
        h_v_history.set_shape([None, None, 128])
        h_u_history.set_shape([None, None, 128])

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

    # @tf.function
    def call(self, x):
        x = tf.one_hot(x, 8)
        x.set_shape([None, 8])
        return self.d(x)

class phi_u(tf.keras.Model):
    def __init__(self):
        super(phi_u, self).__init__()
        self.d = lime.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu'))

    def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
        return self.d(h_u, h_u_0, h_e_bar, h_v_bar)

gn = gin.probabilistic.gn.GraphNet(
    f_e=tf.keras.layers.Dense(128),

    f_v=f_v(128),

    f_u=(lambda atoms, adjacency_map, batched_attr_mask: \
        tf.tile(
            tf.zeros((1, 128)),
            [
                 tf.math.count_nonzero(batched_attr_mask),
                 1
            ]
        )),

    phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_u=phi_u(),

    f_r=f_r(128, (128, 'elu', 128, 1)),

    repeat=3)


optimizer = tf.keras.optimizers.Adam(1e-5)
@tf.function
def train():
    n_epoch = 50

    for epoch_idx in range(50):
        for atoms, adjacency_map, atom_in_mol, bond_in_mol, y, y_mask in ds:

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

            logger.debug(loss)
            variables = gn.variables
            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                zip(grad, variables))

train()
