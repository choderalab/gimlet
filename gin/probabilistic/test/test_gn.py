import gin
import tonic
import tensorflow as tf
tf.enable_eager_execution()
import pandas as pd

caffeine = gin.i_o.from_smiles.smiles_to_mol('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')

def f_e(x):
    print('=======================================')
    print('Testing forward propagation of f_e')
    print('Current bond order is:')
    print(x)
    x = tf.keras.layers.Dense(128)(x)
    print('Now h_e is')
    print(x)
    print('=======================================')

    return x

def f_v(x):
    print('=======================================')
    print('Testing forward propagation of f_v')
    print('Current atom is:')
    print(x)
    x = tf.keras.layers.Dense(128)(tf.one_hot(x, 8))
    print('Now h_v is')
    print(x)
    print('=======================================')
    return x

class f_r(tf.keras.Model):
    def __init__(self, config):
        super(f_r, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

    def call(self, h_e, h_v, h_u):
        y = self.d(h_u)[0][0]
        return y

def rho_e_v(h_e, atom_is_connected_to_bonds):
    print('=======================================')
    print('Testing forward propagation of rho_e_v')
    print('Current h_e:')
    print(h_e)
    print('Atom is connected to bonds: ')
    print(atom_is_connected_to_bonds)
    x = tf.reduce_sum(
        tf.where(
            tf.tile(
                tf.expand_dims(
                    atom_is_connected_to_bonds,
                    2),
                [1, 1, h_e.shape[1]]),
            tf.tile(
                tf.expand_dims(
                    h_e,
                    0),
                [
                    atom_is_connected_to_bonds.shape[0], # n_atoms
                    1,
                    1
                ]),
            tf.zeros((
                atom_is_connected_to_bonds.shape[0],
                h_e.shape[0],
                h_e.shape[1]))),
        axis=1)
    print('New v_bar')
    print(x)
    return x



gn = gin.probabilistic.gn.GraphNet(
    f_e=f_e,

    f_v=f_v,

    f_u=(lambda x, y: tf.zeros((1, 128), dtype=tf.float32)),

    phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    phi_u=tonic.nets.for_gn.ConcatenateThenFullyConnect((128, 'elu', 128, 'elu')),

    rho_e_v=rho_e_v,

    rho_e_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    rho_v_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    f_r=f_r((128, 'tanh', 128, 1)),

    repeat=3)

print(gn(caffeine))
