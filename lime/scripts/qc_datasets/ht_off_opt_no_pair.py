# =============================================================================
# imports
# =============================================================================
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)
from sklearn import metrics
import gin
import lime
import pandas as pd
import numpy as np
# import qcportal as ptl
# client = ptl.FractalClient()


TRANSLATION = {
    6: 0,
    7: 1,
    8: 2,
    16: 3,
    15: 4,
    9: 5,
    17: 6,
    35: 7,
    53: 8,
    1: 9
}


# ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")
# ds_name = tf.data.Dataset.from_tensor_slices(list(ds_qc.data.records))

def data_generator():
    for record_name in list(ds_qc.data.records):
        r = ds_qc.get_record(record_name, specification='default')
        if r is not None:
            traj = r.get_trajectory()
            if traj is not None:
                for snapshot in traj:
                    energy = tf.convert_to_tensor(
                        snapshot.properties.scf_total_energy,
                        dtype=tf.float32)

                    mol = snapshot.get_molecule()

                    atoms = tf.convert_to_tensor(
                        [TRANSLATION[atomic_number] for atomic_number in mol.atomic_numbers],
                        dtype=tf.int64)

                    adjacency_map = tf.tensor_scatter_nd_update(
                        tf.zeros(
                            (
                                tf.shape(atoms, tf.int64)[0],
                                tf.shape(atoms, tf.int64)[0]
                            ),
                            dtype=tf.float32),
                        tf.convert_to_tensor(
                            np.array(mol.connectivity)[:, :2],
                            dtype=tf.int64),
                        tf.convert_to_tensor(
                            np.array(mol.connectivity)[:, 2],
                            dtype=tf.float32))

                    features = gin.probabilistic.featurization.featurize_atoms(
                        atoms, adjacency_map)

                    xyz = tf.convert_to_tensor(
                        mol.geometry,
                        dtype=tf.float32)

                    jacobian = tf.convert_to_tensor(
                        snapshot.return_result,
                        dtype=tf.float32)

                    atoms = tf.concat(
                        [
                            features,
                            xyz,
                            jacobian
                        ],
                    axis=1)

                    yield(atoms, adjacency_map, energy)


def data_loader(idx):
    atoms_path = 'data/atoms/' + str(idx.numpy()) + '.npy'
    adjacency_map_path = 'data/adjacency_map/' + str(idx.numpy()) + '.npy'
    energy_path = 'data/energy/' + str(idx.numpy()) + '.npy'

    atoms = tf.convert_to_tensor(
        np.load(atoms_path))

    adjacency_map = tf.convert_to_tensor(
        np.load(adjacency_map_path))

    energy = tf.convert_to_tensor(
        np.load(energy_path))

    return atoms, adjacency_map, energy


'''
ds = tf.data.Dataset.from_generator(
    data_generator,
    (tf.float32, tf.float32, tf.float32))
'''

ds_path = tf.data.Dataset.from_tensor_slices(list(range(5000)))

ds = ds_path.map(
    lambda idx: tf.py_function(
        data_loader,
        [idx],
        [tf.float32, tf.float32, tf.float32]))


ds = ds.shuffle(100000, seed=2666)


ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 128, feature_dimension=18, atom_dtype=tf.float32).cache(
            str(os.getcwd()) + '/temp')

n_batches = int(gin.probabilistic.gn.GraphNet.get_number_batches(ds))
n_te = n_batches // 10

ds_te = ds.take(n_te)
ds_vl = ds.skip(n_te).take(n_te)
ds_tr = ds.skip(2 * n_te)

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

# @tf.function
def flow(y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, atoms, adjacency_map, coordinates, atom_in_mol,
    bond_in_mol, angle_in_mol, torsion_in_mol, attr_in_mol):

    per_mol_mask = tf.stop_gradient(tf.matmul(
        tf.where(
            atom_in_mol,
            tf.ones_like(atom_in_mol, dtype=tf.float32),
            tf.zeros_like(atom_in_mol, dtype=tf.float32),
            name='per_mol_mask_0'),
        tf.transpose(
            tf.where(
                atom_in_mol,
                tf.ones_like(atom_in_mol, dtype=tf.float32),
                tf.zeros_like(atom_in_mol, dtype=tf.float32),
                name='per_mol_mask_1'))))

    bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
        .get_geometric_idxs(atoms, adjacency_map)

    is_bond = tf.stop_gradient(tf.greater(
        adjacency_map,
        tf.constant(0, dtype=tf.float32)))

    distance_matrix = gin.deterministic.md.get_distance_matrix(
        coordinates)

    bond_distances = tf.boolean_mask(
        distance_matrix,
        is_bond,
        name='bond_mask')

    angle_angles = gin.deterministic.md.get_angles_cos(
        coordinates,
        angle_idxs)

    torsion_dihedrals = gin.deterministic.md.get_dihedrals_cos(
        coordinates,
        torsion_idxs)

    y_e_0, y_e_1 = tf.split(y_e, 2, 1)
    y_e_0 = tf.squeeze(y_e_0)
    y_e_1 = tf.squeeze(y_e_1)
    u_bond = tf.math.multiply(
            tf.math.exp(y_e_1),
            tf.math.pow(
                tf.math.subtract(
                    bond_distances,
                    tf.math.exp(y_e_0)),
                tf.constant(2, dtype=tf.float32)))

    y_a_0, y_a_1 = tf.split(y_a, 2, 1)
    y_a_0 = tf.squeeze(y_a_0)
    y_a_1 = tf.squeeze(y_a_1)
    u_angle = tf.math.multiply(
        tf.math.exp(y_a_1),
        tf.math.pow(
            tf.math.subtract(
                angle_angles,
                tf.tanh(
                    y_a_1)),
            tf.constant(2, dtype=tf.float32)))

    y_t_0, y_t_1 = tf.split(y_t, 2, 1)
    y_t_0 = tf.squeeze(y_t_0)
    y_t_1 = tf.squeeze(y_t_1)
    u_dihedral = tf.math.multiply(
        tf.math.exp(y_t_1),
        tf.math.pow(
            tf.math.subtract(
                torsion_dihedrals,
                tf.tanh(
                    y_t_0)),
            tf.constant(2, dtype=tf.float32)))

    u_pair_mask = tf.linalg.band_part(
            tf.nn.relu(
                tf.subtract(
                    tf.subtract(
                        per_mol_mask,
                        adjacency_map),
                    tf.eye(
                        tf.shape(per_mol_mask)[0]))),
                0, -1)

    _distance_matrix = tf.where(
        tf.greater(
            u_pair_mask,
            tf.constant(0, dtype=tf.float32)),
        distance_matrix,
        tf.ones_like(distance_matrix))

    _distance_matrix_inverse = tf.multiply(
        u_pair_mask,
        tf.pow(
            tf.math.add(
                _distance_matrix,
                tf.constant(1e-2, dtype=tf.float32)),
            tf.constant(-1, dtype=tf.float32)))

    sigma_over_r = tf.multiply(
        sigma_pair,
        _distance_matrix_inverse)


    u_pair = tf.reduce_sum(
        [
            tf.multiply(
                tf.pow(
                    _distance_matrix_inverse,
                    tf.constant(2, dtype=tf.float32)),
                q_pair),
            tf.multiply(
                epsilon_pair,
                tf.math.subtract(
                    tf.pow(
                        sigma_over_r,
                        tf.constant(12, dtype=tf.float32)),
                    tf.pow(
                        sigma_over_r,
                        tf.constant(6, dtype=tf.float32))))

        ],
        axis=0)

    u_bond_tot = tf.matmul(
        tf.transpose(
            tf.where(
                bond_in_mol,
                tf.ones_like(bond_in_mol, dtype=tf.float32),
                tf.zeros_like(bond_in_mol, dtype=tf.float32))),
        tf.expand_dims(
            u_bond,
            axis=1))

    u_angle_tot = tf.matmul(
        tf.transpose(
            tf.where(
                angle_in_mol,
                tf.ones_like(angle_in_mol, dtype=tf.float32),
                tf.zeros_like(angle_in_mol, dtype=tf.float32))),
        tf.expand_dims(
            u_angle,
            axis=1))

    u_dihedral_tot = tf.matmul(
        tf.transpose(
            tf.where(
                torsion_in_mol,
                tf.ones_like(torsion_in_mol, dtype=tf.float32),
                tf.zeros_like(torsion_in_mol, dtype=tf.float32))),
        tf.expand_dims(
            u_dihedral,
            axis=1))

    u_pair_tot = tf.boolean_mask(
        tf.matmul(
            tf.transpose(
                tf.where(
                    atom_in_mol,
                    tf.ones_like(atom_in_mol, dtype=tf.float32),
                    tf.zeros_like(atom_in_mol, dtype=tf.float32))),
            tf.reduce_sum(
                u_pair,
                axis=1,
                keepdims=True)),
        attr_in_mol)

    u_tot = tf.squeeze(
        u_bond_tot + u_angle_tot + u_dihedral_tot + u_pair_tot)

    return u_tot

def init(point):
    global gn
    global optimizer

    class f_v(tf.keras.Model):
        """ Featurization of nodes.
        Here we simply featurize atoms using one-hot encoding.

        """
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        # @tf.function
        def call(self, x):
            return self.d(x)

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

    optimizer = tf.keras.optimizers.Adam(1e-4)

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(10):
        for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_tr:
            atoms = atoms_[:, :12]
            coordinates = tf.Variable(atoms_[:, 12:15])
            jacobian = atoms_[:, 15:]
            with tf.GradientTape() as tape:
                e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                        atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)


                with tf.GradientTape() as tape1:

                    u_hat = flow(y_e, y_a, y_t, q_pair, sigma_pair,
                        epsilon_pair, atoms, adjacency_map,
                            coordinates, atom_in_mol, bond_in_mol, angle_in_mol,
                            torsion_in_mol, attr_in_mol)

                jacobian_hat = tape1.gradient(u_hat, coordinates)

                jacobian_hat = tf.boolean_mask(
                    jacobian_hat,
                    tf.reduce_any(
                        atom_in_mol,
                        axis=1))

                jacobian = tf.boolean_mask(
                    jacobian,
                    tf.reduce_any(
                        atom_in_mol,
                        axis=1))

                u = tf.boolean_mask(
                    u,
                    attr_in_mol)
                
                '''
                loss = tf.math.add(
                    tf.reduce_sum(
                        tf.keras.losses.MSE(
                            tf.math.log(
                                tf.norm(
                                    jacobian,
                                    axis=1)),
                            tf.math.log(
                                tf.norm(
                                    jacobian_hat,
                                    axis=1)))),
                    tf.reduce_sum(
                        tf.losses.cosine_similarity(
                            jacobian,
                            jacobian_hat,
                            axis=1)))
                '''

                loss = tf.reduce_sum(tf.keras.losses.MAPE(
                    jacobian,
                    jacobian_hat))
            
            variables = gn.variables
            grad = tape.gradient(loss, variables)

            # if not tf.reduce_any([tf.reduce_any(tf.math.is_nan(_grad)) for _grad in grad]).numpy():

            optimizer.apply_gradients(
                    zip(grad, variables))

            print(loss)

            del loss
            del coordinates
            del tape
            del tape1

    y_true_tr = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, ], dtype=tf.float32)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_tr:
        atoms = atoms_[:, :12]
        coordinates = tf.Variable(atoms_[:, 12:15])
        jacobian = atoms_[:, 15:]
        with tf.GradientTape() as tape:
            e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                    atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)


            with tf.GradientTape() as tape1:

                u_hat = flow(y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, atoms, adjacency_map,
                        coordinates, atom_in_mol, bond_in_mol, angle_in_mol,
                        torsion_in_mol, attr_in_mol)

            jacobian_hat = tape1.gradient(u_hat, coordinates)

            jacobian_hat = tf.boolean_mask(
                jacobian_hat,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

            jacobian = tf.boolean_mask(
                jacobian,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

        y_true_tr = tf.concat([y_true_tr, tf.reshape(jacobian, [-1])], axis=0)
        y_pred_tr = tf.concat([y_pred_tr, tf.reshape(jacobian_hat, [-1])], axis=0)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_te:
        atoms = atoms_[:, :12]
        coordinates = tf.Variable(atoms_[:, 12:15])
        jacobian = atoms_[:, 15:]
        with tf.GradientTape() as tape:
            e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                    atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

            with tf.GradientTape() as tape1:

                u_hat = flow(y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, atoms, adjacency_map,
                        coordinates, atom_in_mol, bond_in_mol, angle_in_mol,
                        torsion_in_mol, attr_in_mol)

            jacobian_hat = tape1.gradient(u_hat, coordinates)

            jacobian_hat = tf.boolean_mask(
                jacobian_hat,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

            jacobian = tf.boolean_mask(
                jacobian,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

        y_true_te = tf.concat([y_true_te, tf.reshape(jacobian, [-1])], axis=0)
        y_pred_te = tf.concat([y_pred_te, tf.reshape(jacobian_hat, [-1])], axis=0)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_vl:
        atoms = atoms_[:, :12]
        coordinates = tf.Variable(atoms_[:, 12:15])
        jacobian = atoms_[:, 15:]
        with tf.GradientTape() as tape:
            e0, y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                    atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)


            with tf.GradientTape() as tape1:

                u_hat = flow(y_e, y_a, y_t, q_pair, sigma_pair, epsilon_pair, atoms, adjacency_map,
                        coordinates, atom_in_mol, bond_in_mol, angle_in_mol,
                        torsion_in_mol, attr_in_mol)

            jacobian_hat = tape1.gradient(u_hat, coordinates)

            jacobian_hat = tf.boolean_mask(
                jacobian_hat,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

            jacobian = tf.boolean_mask(
                jacobian,
                tf.reduce_any(
                    atom_in_mol,
                    axis=1))

        y_true_vl = tf.concat([y_true_vl, tf.reshape(jacobian, [-1])], axis=0)
        y_pred_vl = tf.concat([y_pred_vl, tf.reshape(jacobian_hat, [-1])], axis=0)

    try:
        r2_tr = metrics.r2_score(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())
        rmse_tr = metrics.mean_squared_error(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())

        r2_vl = metrics.r2_score(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())
        rmse_vl = metrics.mean_squared_error(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())

        r2_te = metrics.r2_score(y_true_te[1:].numpy(), y_pred_te[1:].numpy())
        rmse_te = metrics.mean_squared_error(y_true_te[1:].numpy(), y_pred_te[1:].numpy())


        np.save('y_true_tr', y_true_tr[1:].numpy())
        np.save('y_pred_tr', y_pred_tr[1:].numpy())
        np.save('y_true_te', y_true_te[1:].numpy())
        np.save('y_pred_te', y_pred_te[1:].numpy())
        np.save('y_true_vl', y_true_vl[1:].numpy())
        np.save('y_pred_vl', y_pred_vl[1:].numpy())

        print(tf.stack([y_true_tr, y_pred_tr], axis=1))

        print(point, flush=True)
        print(r2_tr, flush=True)
        print(rmse_tr, flush=True)
        print(r2_vl, flush=True)
        print(rmse_vl, flush=True)
        print(r2_te, flush=True)
        print(rmse_te, flush=True)

        gn.save_weights('gn.h5')

        return rmse_vl

    except:
        print('nan')
        return None

lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
