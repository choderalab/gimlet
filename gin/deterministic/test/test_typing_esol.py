import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)
import gin
import lime
import pandas as pd
import numpy as np

config_space = {
    'D_V': [16, 32, 64, 128, 256],
    'D_E': [16, 32, 64, 128, 256],
    'D_A': [16, 32, 64, 128, 256],
    'D_T': [16, 32, 64, 128, 256],
    'D_U': [16, 32, 64, 128, 256],


    'phi_e_0': [32, 64, 128],
    'phi_e_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128],
    'phi_v_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_a_0': [32, 64, 128],
    'phi_a_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_a_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_t_0': [32, 64, 128],
    'phi_t_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_t_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128],
    'phi_u_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'f_e_0': [32, 64, 128],
    'f_e_a_0': ['elu', 'relu', 'tanh', 'sigmoid'],
    'f_e_a_1': ['elu', 'relu', 'tanh', 'sigmoid'],

    'f_r': [32, 64, 128],
    'f_r_a': ['elu', 'relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}

point = dict([[k, v[0]] for k, v in config_space.items()])

class f_v(tf.keras.Model):
    """ Featurization of nodes.
    Here we simply featurize atoms using one-hot encoding.
    """
    def __init__(self, units=point['D_V']):
        super(f_v, self).__init__()
        self.d = tf.keras.layers.Dense(units)

    # @tf.function
    def call(self, x):
        return self.d(tf.one_hot(x, 10))

class f_r(tf.keras.Model):
    """ Readout function
    """
    def __init__(self, units=point['f_r'], f_r_a=point['f_r_a']):
        super(f_r, self).__init__()

        self.d_q_0 = tf.keras.layers.Dense(units, activation='relu')
        self.d_q_1 = tf.keras.layers.Dense(1)

        self.d_sigma_0 = tf.keras.layers.Dense(units, activation='relu')
        self.d_sigma_1 = tf.keras.layers.Dense(1, activation='relu')

        self.d_epislon_0 = tf.keras.layers.Dense(units, activation='relu')
        self.d_epsilon_1 = tf.keras.layers.Dense(1, activation='relu')

        self.d_e_1 = tf.keras.layers.Dense(2,
            kernel_initializer='random_uniform')

        self.d_e_0 = tf.keras.layers.Dense(units, activation='relu')

        self.d_a_1 = tf.keras.layers.Dense(2,
            kernel_initializer='random_uniform')
        self.d_a_0 = tf.keras.layers.Dense(units, activation='relu')

        self.d_t_1 = tf.keras.layers.Dense(2,
            kernel_initializer='random_uniform')
        self.d_t_0 = tf.keras.layers.Dense(units, activation='relu')

        self.d_e0_0 = lime.nets.for_gn.ConcatenateThenFullyConnect((units,
            'relu', units, 'relu'))

        self.d_e0_1 = tf.keras.layers.Dense(1)

        self.units = units
        self.d_v = point['D_V']
        self.d_e = point['D_E']
        self.d_a = point['D_A']
        self.d_t = point['D_T']
        self.d_u = point['D_U']

    # @tf.function
    def call(self, h_v, h_e, h_a, h_t, h_u,
        h_v_history, h_e_history, h_a_history,
        h_t_history, h_u_history,
        atom_in_mol, bond_in_mol, angle_in_mol, torsion_in_mol,
        adjacency_map, coordinates):


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

        e0 = tf.squeeze(self.d_e0_1(self.d_e0_0(
            tf.reshape(
                h_v_bar_history,
                [-1, 6 * self.d_v]),
            tf.reshape(
                h_e_bar_history,
                [-1, 6 * self.d_e]),
            tf.reshape(
                h_u_history,
                [-1, 6 * self.d_u]))))

        adjacency_map_full = tf.math.add(
            tf.transpose(
                adjacency_map),
            adjacency_map)

        per_mol_mask = tf.matmul(
            tf.where(
                atom_in_mol,
                tf.ones_like(atom_in_mol, dtype=tf.float32),
                tf.zeros_like(atom_in_mol, dtype=tf.float32)),
            tf.transpose(
                tf.where(
                    atom_in_mol,
                    tf.ones_like(atom_in_mol, dtype=tf.float32),
                    tf.zeros_like(atom_in_mol, dtype=tf.float32))))

        # get distance matrix
        distance = gin.deterministic.md.get_distance_matrix(coordinates)

        distance = tf.expand_dims(
            distance,
            2)

        n_atoms = tf.shape(distance, tf.int64)[0]


        # (n_atoms, n_atoms)
        q = tf.squeeze(
            self.d_q_1(
                self.d_q_0(
                    h_v)))

        # (n_atoms, n_atoms)
        sigma = tf.squeeze(
            self.d_sigma_1(
                self.d_sigma_0(
                    h_v)))

        # (n_atoms, n_atoms)
        epsilon = tf.squeeze(
            self.d_epsilon_1(
                self.d_epislon_0(
                    h_v)))

        # (n_atoms, n_atoms)
        q_pair = tf.multiply(
            q,
            tf.transpose(
                q))

        # (n_atoms, n_atoms)
        sigma_pair = tf.math.multiply(
            tf.constant(0.5, dtype=tf.float32),
            tf.math.add(
                sigma,
                tf.transpose(sigma)))

        # (n_atoms, n_atoms)
        epsilon_pair = tf.math.square(
            tf.math.multiply(
                epsilon,
                tf.transpose(epsilon)))

        y_a = self.d_a_1(
            self.d_a_0(
                tf.reshape(
                    h_a_history,
                    [
                        tf.shape(h_a_history)[0],
                        6 * self.d_a
                    ])))

        y_e = self.d_e_1(
            self.d_e_0(
                tf.reshape(
                    h_e_history,
                    [
                        tf.shape(h_e_history)[0],
                        6 * self.d_e
                    ])))


        y_t = self.d_t_1(
            self.d_t_0(
                tf.reshape(
                    h_t_history,
                    [
                        tf.shape(h_t_history)[0],
                        6 * self.d_t
                    ])))

        return e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol

gn = gin.probabilistic.gn_hyper.HyperGraphNet(
    f_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['f_e_0'], 'elu', point['D_E'], 'tanh')),
    f_a=tf.keras.layers.Dense(point['D_A'], activation='tanh'),
    f_t=tf.keras.layers.Dense(point['D_T'], activation='tanh'),
    f_v=f_v(),
    f_u=(lambda atoms, adjacency_map, batched_attr_in_mol: \
        tf.tile(
            tf.zeros((1, point['D_U'])),
            [
                    tf.math.count_nonzero(batched_attr_in_mol),
                    1
            ]
        )),
    phi_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_e_0'], point['phi_e_a_0'], point['D_E'],
        point['phi_e_a_1'])),
    phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_u_0'], point['phi_u_a_0'], point['D_U'],
        point['phi_u_a_1'])),
    phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_v_0'], point['phi_v_a_0'], point['D_V'],
        point['phi_v_a_1'])),
    phi_a=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_a_0'], point['phi_a_a_0'], point['D_A'],
        point['phi_a_a_1'])),
    phi_t=lime.nets.for_gn.ConcatenateThenFullyConnect(
        (point['phi_t_0'], point['phi_t_a_0'], point['D_T'],
        point['phi_t_a_1'])),
    f_r=f_r(),
    repeat=5)

df = pd.read_csv('data/delaney-processed.csv')
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
df = df[df['smiles'].str.len() > 1]
x_array = df[['smiles']].values.flatten()
y_array = df[['measured log solubility in mols per litre']].values.flatten()
y_array = (y_array - np.mean(y_array) / np.std(y_array))
n_samples = y_array.shape[0]

# ds_all = gin.i_o.from_smiles.to_mols_with_attributes(x_array, y_array)
ds = gin.i_o.from_smiles.to_mols(x_array)
ds_attr = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(y_array, tf.float32))

ds = ds.map(lambda atoms, adjacency_map:
        tf.py_function(
            lambda atoms, adjacency_map: gin.deterministic.hydrogen.add_hydrogen([atoms, adjacency_map]),
            [atoms, adjacency_map],
            [tf.int64, tf.float32]))

ds_all = tf.data.Dataset.zip((ds, ds_attr))

ds_all = ds_all.map(
    lambda mol, attr: (mol[0], mol[1], attr)).cache(
    str(os.getcwd()) + '/temp')

with tf.GradientTape() as tape1:
    for atoms, adjacency_map, attr in ds_all:
            with tf.GradientTape() as tape:

                mol_sys = gin.deterministic.md.SingleMoleculeMechanicsSystem([atoms, adjacency_map])

                e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol = \
                    gn(atoms, adjacency_map)

                loss = tf.reduce_sum(
                    [
                        tf.reduce_sum(tf.math.square(
                            tf.math.subtract(
                                y_e,
                                tf.stack(
                                    [
                                        tf.stop_gradient(mol_sys.bond_length),
                                        tf.stop_gradient(mol_sys.bond_k)
                                    ],
                                    axis=1)))),

                        tf.reduce_sum(tf.math.square(
                            tf.math.subtract(
                                y_a,
                                tf.stack(
                                    [
                                        tf.stop_gradient(mol_sys.angle_angle),
                                        tf.stop_gradient(mol_sys.angle_k)
                                    ],
                                    axis=1)))),

                        tf.reduce_sum(tf.math.square(
                            tf.math.subtract(
                                y_t,
                                tf.stack(
                                    [
                                        tf.stop_gradient(mol_sys.torsion_proper_periodicity1),
                                        tf.stop_gradient(mol_sys.torsion_proper_k1)
                                    ],
                                    axis=1)))),
                    ],
                    axis=0
                )

                print(loss)
                print(gn.variables)


            grad = tape.gradient(loss, gn.variables)
            optimizer.apply(zip(grad, gn.variables))
            print(loss)

        # break
