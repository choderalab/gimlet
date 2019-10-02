# =============================================================================
# imports
# =============================================================================
from sklearn import metrics
import os
import sys
import tensorflow as tf
import gin
import lime
import pandas as pd
import numpy as np
import qcportal as ptl
client = ptl.FractalClient()

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


ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")
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
    atoms_path = 'data/atoms' + str(idx)
    adjacency_map_path = 'data/adjacency_map' + str(idx)
    energy_path = '/data/energy' + str(idx)

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

ds = ds_path.map(data_loader)

ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 256, feature_dimension=18, atom_dtype=tf.float32).shuffle(
        100000,
        seed=2666).cache(
            str(os.getcwd()) + '/temp')


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

@tf.function
def flow(y_e, y_a, y_t, y_pair, atoms, adjacency_map, coordinates, atom_in_mol,
    bond_in_mol, angle_in_mol, torsion_in_mol, attr_in_mol):

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

    bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
        .get_geometric_idxs(atoms, adjacency_map)

    is_bond = tf.greater(
        adjacency_map,
        tf.constant(0, dtype=tf.float32))

    distance_matrix = gin.deterministic.md.get_distance_matrix(
        coordinates)

    bond_distances = tf.boolean_mask(
        distance_matrix,
        is_bond)

    angle_angles = gin.deterministic.md.get_angles(
        coordinates,
        angle_idxs)

    torsion_dihedrals = gin.deterministic.md.get_dihedrals(
        coordinates,
        torsion_idxs)

    u_bond = tf.math.reduce_sum(
        tf.math.multiply(
            y_e,
            tf.math.pow(
                tf.expand_dims(
                    bond_distances,
                    1),
                tf.range(16, dtype=tf.float32))),
        axis=1)

    u_angle = tf.math.reduce_sum(
        tf.math.multiply(
            y_a,
            tf.math.pow(
                tf.expand_dims(
                    angle_angles,
                    1),
                tf.range(16, dtype=tf.float32))),
        axis=1)

    u_dihedral = tf.math.reduce_sum(
        tf.math.multiply(
            y_t,
            tf.math.pow(
                tf.expand_dims(
                    torsion_dihedrals,
                    1),
                tf.range(16, dtype=tf.float32))),
        axis=1)

    u_pair = tf.reduce_sum(
            tf.multiply(
                y_pair,
                tf.math.pow(
                    tf.expand_dims(
                            tf.where(
                                tf.logical_and(
                                    tf.equal(
                                        tf.eye(
                                            tf.shape(
                                                distance_matrix)[0],
                                            dtype=tf.float32),
                                        tf.constant(0, dtype=tf.float32)),
                                    tf.greater(
                                        distance_matrix,
                                        tf.constant(0, dtype=tf.float32))),
                                tf.pow(
                                    distance_matrix + 1e-2,
                                    -1),
                                distance_matrix),

                        axis=2),
                    tf.range(1, 16, dtype=tf.float32))),
            axis=2)

    u_pair_mask = tf.linalg.band_part(
        tf.nn.relu(
            tf.subtract(
                tf.subtract(
                    per_mol_mask,
                    adjacency_map),
                tf.eye(
                    tf.shape(per_mol_mask)[0]))),
        0, -1)

    u_pair = tf.multiply(
        u_pair_mask,
        u_pair)

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

        @tf.function
        def call(self, x):
            x = tf.one_hot(x, 10)
            # set shape because Dense doesn't like variation
            x.set_shape([None, 10])
            return self.d(x)

    class f_r(tf.keras.Model):
        """ Readout function
        """
        def __init__(self, units=point['f_r'], f_r_a=point['f_r_a']):
            super(f_r, self).__init__()
            self.d_k = tf.keras.layers.Dense(units, activation='tanh')
            self.d_q = tf.keras.layers.Dense(units, activation='tanh')
            self.d_pair_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_pair_1 = tf.keras.layers.Dense(15, kernel_initializer='zeros', activity_regularizer=tf.keras.regularizers.l2(0.1))

            self.d_e_1 = tf.keras.layers.Dense(16, kernel_initializer='zeros', activity_regularizer=tf.keras.regularizers.l2(0.1))
            self.d_e_0 = tf.keras.layers.Dense(units, activation='tanh')

            self.d_a_1 = tf.keras.layers.Dense(16, kernel_initializer='zeros', activity_regularizer=tf.keras.regularizers.l2(0.1))
            self.d_a_0 = tf.keras.layers.Dense(units, activation='tanh')

            self.d_t_1 = tf.keras.layers.Dense(16, kernel_initializer='zeros', activity_regularizer=tf.keras.regularizers.l2(0.1))
            self.d_t_0 = tf.keras.layers.Dense(units, activation='tanh')

            self.units = units
            self.d_v = point['D_V']
            self.d_e = point['D_E']
            self.d_a = point['D_A']
            self.d_t = point['D_T']
            self.d_u = point['D_U']

        @tf.function
        def call(self, h_v, h_e, h_a, h_t, h_u,
            h_v_history, h_e_history, h_a_history,
            h_t_history, h_u_history,
            atom_in_mol, bond_in_mol, angle_in_mol, torsion_in_mol,
            adjacency_map, coordinates):

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

            # (n_atoms, n_atoms, units)
            k = tf.multiply(
                tf.tile(
                    tf.expand_dims(
                        per_mol_mask,
                        2),
                    [1, 1, self.units]),
                tf.tile(
                    tf.expand_dims(
                        self.d_k(h_v),
                        1),
                    [1, n_atoms, 1]))

            # (n_atoms, n_atoms, units)
            q = tf.multiply(
                tf.tile(
                    tf.expand_dims(
                        per_mol_mask,
                        2),
                    [1, 1, self.units]),
                tf.tile(
                    tf.expand_dims(
                        self.d_q(h_v),
                        0),
                    [n_atoms, 1, 1]))

            h_pair = tf.concat(
                [
                    k,
                    q,
                ],
                axis=2)

            h_pair = tf.math.multiply(
                tf.tile(
                    tf.expand_dims(
                        tf.math.multiply(
                            tf.math.subtract(
                                per_mol_mask,
                                tf.eye(
                                    tf.shape(per_mol_mask)[0])),
                            tf.where(
                                tf.equal(
                                    adjacency_map_full,
                                    tf.constant(0, dtype=tf.float32)),
                                tf.ones_like(adjacency_map),
                                tf.zeros_like(adjacency_map))),
                        2),
                    [1, 1, 15]),
                self.d_pair_1(self.d_pair_0(h_pair)))

            y_pair = h_pair

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

            return y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol


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

    optimizer = tf.keras.optimizers.Ftrl(1e-4)

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(10):
        for atoms_, adjacency_map, attr, atom_in_mol, attr_in_mol in ds_tr:
            atoms = atoms_[:, :12]
            coordinates = atoms[:, 12:25]
            jacobian = atoms[:, 15:]

            with tf.GradientTape() as tape:
                y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                    atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

                u_tot = flow(y_e, y_a, y_t, y_pair, atoms, adjacency_map,
                    coordinates, atom_in_mol, bond_in_mol, angle_in_mol,
                    torsion_in_mol, attr_in_mol)

                jaboian_hat = tape.gradient(
                    u_tot,
                    tf.boolean_mask(
                        jacobian,
                        tf.reduce_any(
                            atom_in_mol,
                            axis=1)))[0]

                loss = tf.keras.losses.MSE(u, u_tot) + tf.keras.losses.MSE(
                    jacobian, jacobian_hat)

            variables = gn.variables
            grad = tape.gradient(loss, variables)

            if not tf.reduce_any([tf.reduce_any(tf.math.is_nan(_grad)) for _grad in grad]).numpy():

                optimizer.apply_gradients(
                    zip(grad, variables))

lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
