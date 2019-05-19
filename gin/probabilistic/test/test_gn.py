import gin
import tonic
import tensorflow as tf
tf.enable_eager_execution()
import pandas as pd


# read data
df = pd.read_csv('data/delaney-processed.csv')
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()

ds = gin.i_o.from_smiles.smiles_to_mols_with_attributes(x_array, y_array)

class f_r(tf.keras.Model):
    def __init__(self, config):
        super(f_r, self).__init__()
        self.d = tonic.nets.for_gn.ConcatenateThenFullyConnect(config)

    def call(self, h_e, h_v, h_u):
        y = self.d(h_u)[0][0]
        return y

gn = gin.probabilistic.gn.GraphNet(
    f_e=tf.keras.layers.Dense(16),

    f_v=tf.keras.layers.Lambda(
        lambda x: tf.keras.layers.Dense(16)(tf.one_hot(x, 8))),

    f_u=(lambda x, y: tf.zeros((1, 16), dtype=tf.float32)),

    phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

    phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

    phi_u=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

    rho_e_v=(lambda h_e, atom_is_connected_to_bonds: tf.reduce_sum(
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
        axis=1)),

    rho_e_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    rho_v_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

    f_r=f_r((4, 'tanh', 4, 'elu', 1)),

    repeat=3)



optimizer = tf.train.AdamOptimizer(1e-5)
n_epoch = 10
batch_size = 5
batch_idx = 0
loss = 0
tape = tf.GradientTape()

for dummy_idx in range(n_epoch):
    for atoms, adjacency_map, y in ds:
        mol = [atoms, adjacency_map]

        with tape:
            y_hat = gn(mol)
            loss += tf.clip_by_norm(
                tf.losses.mean_squared_error(y, y_hat),
                1e8)

            batch_idx += 1

        if batch_idx == batch_size:
            print(loss)
            vars = gn.variables
            grad = tape.gradient(loss, vars)
            optimizer.apply_gradients(
                zip(grad, vars),
                tf.train.get_or_create_global_step())
            loss = 0
            batch_idx = 0
            tape = tf.GradientTape()
