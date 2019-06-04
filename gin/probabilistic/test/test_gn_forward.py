import gin
import tonic
import tensorflow as tf
# tf.enable_eager_execution()
import pandas as pd

caffeine = gin.i_o.from_smiles.smiles_to_mol('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')



class f_r(tf.keras.Model):
    def __init__(self, config):
        super(f_r, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

    @tf.function
    def call(self, h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history):
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

print(gn(caffeine))
