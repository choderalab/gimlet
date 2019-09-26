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

mols_ds = gin.i_o.from_sdf.to_ds('gdb9.sdf', has_charge=False)

attr_ds = pd.read_csv('gdb9.sdf.csv').values[:, 1:].astype(np.float32)

attr_ds = attr_ds / np.linalg.norm(attr_ds, axis=0) \\
    - np.std(attr_ds, axis=0)

attr_ds = tf.data.Dataset.from_tensor_slices(attr_ds)

ds = tf.data.Dataset.zip((mols_ds, attr_ds))

ds = ds.map(
    lambda mol, attr: (mol[0], mol[1], mol[2], attr))

ds = gin.probabilistic.gn_hyper.HyperGraphNet.batch(
    ds,
    128,
    attr_dimension=19).cache(
        str(os.getcwd()) + 'temp')

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

@tf.function
def flow(y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol):
    u = tf.boolean_mask(
        attr[:, 11],
        attr_in_mol)

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
                                    distance_matrix + 1e-3,
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
            (point['f_e_0'], point['f_e_a_0'], point['D_E'], point['f_e_a_1'])),
        f_a=tf.keras.layers.Dense(point['D_A'], activation='elu'),
        f_t=tf.keras.layers.Dense(point['D_T'], activation='elu'),
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
            (point['phi_e_0'], 'elu', point['D_E'], 'tanh')),
        phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_u_0'], 'elu', point['D_U'], 'tanh')),
        phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_v_0'], 'elu', point['D_V'], 'tanh')),
        phi_a=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_a_0'], 'elu', point['D_A'], 'tanh')),
        phi_t=lime.nets.for_gn.ConcatenateThenFullyConnect(
            (point['phi_t_0'], 'elu', point['D_T'], 'tanh')),
        f_r=f_r(),
        repeat=5)

    optimizer = tf.keras.optimizers.Adam(1e-5)

def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(10):
        for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_tr:
            with tf.GradientTape() as tape:
                y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
                    atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

                u_tot = flow(y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol)

                u = tf.boolean_mask(
                    attr[:, 11],
                    attr_in_mol)

                loss = tf.keras.losses.MSE(u, u_tot)


            variables = gn.variables
            grad = tape.gradient(loss, variables)

            if not tf.reduce_any([tf.reduce_any(tf.math.is_nan(_grad)) for _grad in grad]).numpy():

                optimizer.apply_gradients(
                    zip(grad, variables))

    y_true_tr = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, ], dtype=tf.float32)


    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_tr:
        y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
            atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        u_tot = flow(y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol)

        u = tf.boolean_mask(
            attr[:, 11],
            attr_in_mol)

        y_true_tr = tf.concat([y_true_tr, u], axis=0)
        y_pred_tr = tf.concat([y_pred_tr, u_tot], axis=0)

    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_vl:
        y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
            atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        u_tot = flow(y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol)

        u = tf.boolean_mask(
            attr[:, 11],
            attr_in_mol)

        y_true_vl = tf.concat([y_true_vl, u], axis=0)
        y_pred_vl = tf.concat([y_pred_vl, u_tot], axis=0)

    for atoms, adjacency_map, coordinates, attr, atom_in_mol, attr_in_mol in ds_te:
        y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol = gn(
            atoms, adjacency_map, coordinates, atom_in_mol, attr_in_mol)

        u_tot = flow(y_e, y_a, y_t, y_pair, bond_in_mol, angle_in_mol, torsion_in_mol)

        u = tf.boolean_mask(
            attr[:, 11],
            attr_in_mol)

        y_true_te = tf.concat([y_true_te, u], axis=0)
        y_pred_te = tf.concat([y_pred_te, u_tot], axis=0)


    r2_tr = metrics.r2_score(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())
    rmse_tr = metrics.mean_squared_error(y_true_tr[1:].numpy(), y_pred_tr[1:].numpy())

    r2_vl = metrics.r2_score(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())
    rmse_vl = metrics.mean_squared_error(y_true_vl[1:].numpy(), y_pred_vl[1:].numpy())

    r2_te = metrics.r2_score(y_true_te[1:].numpy(), y_pred_te[1:].numpy())
    rmse_te = metrics.mean_squared_error(y_true_te[1:].numpy(), y_pred_te[1:].numpy())

    print(point, flush=True)
    print(r2_tr, flush=True)
    print(rmse_tr, flush=True)
    print(r2_vl, flush=True)
    print(rmse_vl, flush=True)
    print(r2_te, flush=True)
    print(rmse_te, flush=True)

    return rmse_vl

lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
