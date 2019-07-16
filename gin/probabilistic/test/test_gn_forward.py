import gin
import tonic
import tensorflow as tf
# tf.enable_eager_execution()
import pandas as pd

caffeine = gin.i_o.from_smiles.to_mol('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')

class f_r(tf.keras.Model):
    def __init__(self, config):
        super(f_r, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

    @tf.function
    def call(self, h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history,
            atom_in_mol, bond_in_mol):
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

    f_r=f_r((128, 'tanh', 128, 1)),

    repeat=3)

print(gn(caffeine[0], caffeine[1], atom_in_mol=False, bond_in_mol=False))
