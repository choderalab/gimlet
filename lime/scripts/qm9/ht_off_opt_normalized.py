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


mols_ds = gin.i_o.from_sdf.to_ds('gdb9.sdf', has_charge=False)

attr_ds = tf.data.Dataset.from_tensor_slices(
    pd.read_csv('gdb9.sdf.csv').values[:, 1:].astype(np.float32))

mols_ds = mols_ds.map(
    lambda atoms, adjacency_map, coordinates, charges:\
        (tf.cast(atoms, tf.float32), adjacency_map, coordinates))

ds = tf.data.Dataset.zip((mols_ds, attr_ds))



ds = ds.map(
    lambda mol, attr:\
        (
            tf.concat(
                [
                    # gin.probabilistic.featurization.featurize_atoms(
                    #    mol[0],
                    #    mol[1]),
                    tf.expand_dims(
                        mol[0],
                        0),
                    mol[2]
                ],
                axis=1),
            mol[1],
            attr
        )).shuffle(
            buffer_size=10000,
            seed=2666)

ds = gin.probabilistic.gn.GraphNet.batch(
    ds, 128, attr_dimension=19, feature_dimension=4, atom_dtype=tf.float32).cache(
    str(os.getcwd()) + '/temp')


n_batches = int(gin.probabilistic.gn.GraphNet.get_number_batches(ds))
n_te = n_batches // 10


n_te = 100
ds_te = ds.take(n_te)
ds_vl = ds.skip(n_te).take(n_te)
ds_tr = ds.skip(2 * n_te).take(8 * n_te)

config_space = {
    'D_V': [16, 32, 64, 128, 256, 512],
    'D_E': [16, 32, 64, 128, 256, 512],
    'D_U': [16, 32, 64, 128, 256, 512],

    'phi_e_0': [32, 64, 128, 256, 512],
    'phi_e_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_e_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_v_0': [32, 64, 128, 256, 512],
    'phi_v_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_v_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'phi_u_0': [32, 64, 128, 256, 512],
    'phi_u_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'phi_u_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'f_r_0': [32, 64, 128, 256, 512, 1024],
    'f_r_1': [32, 64, 128, 256, 512, 1024],
    'f_r_a_0': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],
    'f_r_a_1': ['elu', 'relu', 'leaky_relu', 'tanh', 'sigmoid'],

    'learning_rate': [1e-5, 1e-4, 1e-3]

}

def init(point):
    global gn
    global opt

    class f_v(tf.keras.Model):
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            return self.d(
                    tf.one_hot(
                        tf.cast(
                            x,
                            tf.int64),
                        10))

    class f_r(tf.keras.Model):
        """ Readout function
        """
        def __init__(self, units=point['f_r_0'], f_r_a=point['f_r_a_0']):
            super(f_r, self).__init__()

            self.d_q_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_q_1 = tf.keras.layers.Dense(
                1)

            self.d_sigma_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_sigma_1 = tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.random_normal_initializer(0, 1e-5),
                bias_initializer=tf.random_normal_initializer(0, 1e-5))

            self.d_epislon_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_epsilon_1 = tf.keras.layers.Dense(
                1,
                bias_initializer=tf.constant_initializer(-15),
                kernel_initializer=tf.random_normal_initializer(0, 1e-5))
                
            self.d_e_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_e_k = tf.keras.layers.Dense(
                1, 
                bias_initializer=tf.constant_initializer(-5))
            self.d_e_l = tf.keras.layers.Dense(1)

            self.d_t_1 = tf.keras.layers.Dense(2,
                kernel_initializer=tf.random_normal_initializer(0, 1e-1),
                bias_initializer=tf.random_normal_initializer(0, 1e-3))

            self.d_t_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_t_k = tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.random_normal_initializer(0, 1e-5),
                bias_initializer=tf.constant_initializer(-10))
            self.d_t_l = tf.keras.layers.Dense(1)

            self.d_a_0 = tf.keras.layers.Dense(units, activation='tanh')
            self.d_a_k = tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.random_normal_initializer(0, 1e-5),
                bias_initializer=tf.constant_initializer(-10))
            self.d_a_l = tf.keras.layers.Dense(1)

            self.d_e0_0 = lime.nets.for_gn.ConcatenateThenFullyConnect((units,
              'relu', units, 'relu'))

            self.d_e0_1 = tf.keras.layers.Dense(1)

            self.units = units
            self.d_v = point['D_V']
            self.d_e = point['D_E']
            self.d_a = point['D_E']
            self.d_t = point['D_E']
            self.d_u = point['D_U']

        # @tf.function
        def call(self,
                 h_e, h_v, h_u,
                 h_e_history, h_v_history, h_u_history,
                 atom_in_mol, bond_in_mol, attr_in_mol,
                 bond_idxs, angle_idxs, torsion_idxs,
                 coordinates):

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


            distance_matrix = gin.deterministic.md.get_distance_matrix(
                coordinates)

            bond_distances = tf.gather_nd(
                distance_matrix,
                bond_idxs)

            angle_angles = gin.deterministic.md.get_angles_cos(
                coordinates,
                angle_idxs)

            torsion_dihedrals = gin.deterministic.md.get_dihedrals_cos(
                coordinates,
                torsion_idxs)

            h_v_history.set_shape([None, 6, self.d_v])

            h_v = tf.reshape(
                h_v_history,
               [-1, 6 * self.d_v])

            h_e = tf.math.add(
                tf.gather(
                    h_v,
                    bond_idxs[:, 0]),
                tf.gather(
                    h_v,
                    bond_idxs[:, 1]))

            h_a = tf.concat(
                [
                    tf.gather(
                        h_v,
                        angle_idxs[:, 1]),
                    tf.math.add(
                        tf.gather(
                            h_v,
                            angle_idxs[:, 0]),
                        tf.gather(
                            h_v,
                            angle_idxs[:, 2]))
                ],
                axis=1)

            h_t = tf.concat(
                [
                    tf.math.add(
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 0]),
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 3])),
                    tf.math.add(
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 1]),
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 2])),
                ],
                axis=1)

            # (n_atoms, n_atoms)
            q = self.d_q_1(
                    self.d_q_0(
                        h_v))

            # (n_atoms, n_atoms)
            sigma = tf.exp(self.d_sigma_1(
                    self.d_sigma_0(
                        h_v)))
            
        
            # (n_atoms, n_atoms)
            epsilon = tf.exp(self.d_epsilon_1(
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
            epsilon_pair = tf.math.sqrt(
                tf.math.multiply(
                    epsilon,
                    tf.transpose(epsilon)))
 
            y_e_l = tf.squeeze(tf.math.exp(self.d_e_l(self.d_e_0(h_e))))
            y_e_k = tf.squeeze(tf.math.exp(self.d_e_k(self.d_e_0(h_e))))
            
            u_bond = tf.math.multiply(
                    y_e_k,
                    tf.math.pow(
                        tf.math.subtract(
                            bond_distances,
                            y_e_l),
                        tf.constant(2, dtype=tf.float32)))

            y_a_k = tf.squeeze(tf.math.exp(self.d_a_k(self.d_a_0(h_a))))
            y_a_l = tf.squeeze(tf.nn.tanh(self.d_a_l(self.d_a_0(h_a))))
            
            u_angle = tf.math.multiply(
                y_a_k,
                tf.math.pow(
                    tf.math.subtract(
                        angle_angles,
                        y_a_l),
                    tf.constant(2, dtype=tf.float32)))

            y_t_k = tf.squeeze(tf.math.exp(self.d_t_k(self.d_t_0(h_t))))
            y_t_l = tf.squeeze(tf.nn.tanh(self.d_t_l(self.d_t_0(h_t))))
            u_dihedral = tf.math.multiply(
                y_t_k,
                tf.math.pow(
                    tf.math.subtract(
                        torsion_dihedrals,
                        y_t_l),
                    tf.constant(2, dtype=tf.float32)))


            n_atoms = tf.shape(h_v, tf.int64)[0]
            n_angles = tf.shape(angle_idxs, tf.int64)[0]
            n_torsions = tf.shape(torsion_idxs, tf.int64)[0]

            # (n_angles, n_atoms)
            angle_is_connected_to_atoms = tf.reduce_any(
                [
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_angles, 1]),
                        tf.tile(
                            tf.expand_dims(
                                angle_idxs[:, 0],
                                1),
                            [1, n_atoms])),
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_angles, 1]),
                        tf.tile(
                            tf.expand_dims(
                                angle_idxs[:, 1],
                                1),
                            [1, n_atoms])),
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_angles, 1]),
                        tf.tile(
                            tf.expand_dims(
                                angle_idxs[:, 2],
                                1),
                            [1, n_atoms]))
                ],
                axis=0)

            # (n_torsions, n_atoms)
            torsion_is_connected_to_atoms = tf.reduce_any(
                [
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_torsions, 1]),
                        tf.tile(
                            tf.expand_dims(
                                torsion_idxs[:, 0],
                                1),
                            [1, n_atoms])),
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_torsions, 1]),
                        tf.tile(
                            tf.expand_dims(
                                torsion_idxs[:, 1],
                                1),
                            [1, n_atoms])),
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_torsions, 1]),
                        tf.tile(
                            tf.expand_dims(
                                torsion_idxs[:, 2],
                                1),
                            [1, n_atoms])),
                    tf.equal(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(n_atoms),
                                0),
                            [n_torsions, 1]),
                        tf.tile(
                            tf.expand_dims(
                                torsion_idxs[:, 3],
                                1),
                            [1, n_atoms]))
                ],
                axis=0)


            angle_in_mol = tf.greater(
                tf.matmul(
                    tf.where(
                        angle_is_connected_to_atoms,
                        tf.ones_like(
                            angle_is_connected_to_atoms,
                            tf.int64),
                        tf.zeros_like(
                            angle_is_connected_to_atoms,
                            tf.int64)),
                    tf.where(
                        atom_in_mol,
                        tf.ones_like(
                            atom_in_mol,
                            tf.int64),
                        tf.zeros_like(
                            atom_in_mol,
                            tf.int64))),
                tf.constant(0, dtype=tf.int64))

            torsion_in_mol = tf.greater(
                tf.matmul(
                    tf.where(
                        torsion_is_connected_to_atoms,
                        tf.ones_like(
                            torsion_is_connected_to_atoms,
                            tf.int64),
                        tf.zeros_like(
                            torsion_is_connected_to_atoms,
                            tf.int64)),
                    tf.where(
                        atom_in_mol,
                        tf.ones_like(
                            atom_in_mol,
                            tf.int64),
                        tf.zeros_like(
                            atom_in_mol,
                            tf.int64))),
                tf.constant(0, dtype=tf.int64))


            u_pair_mask = tf.tensor_scatter_nd_update(
                per_mol_mask,
                bond_idxs,
                tf.zeros(
                    shape=(
                        tf.shape(bond_idxs, tf.int32)[0]),
                    dtype=tf.float32))

            u_pair_mask = tf.tensor_scatter_nd_update(
                u_pair_mask,
                tf.stack(
                    [
                        angle_idxs[:, 0],
                        angle_idxs[:, 2]
                    ],
                    axis=1),
                tf.zeros(
                    shape=(
                        tf.shape(angle_idxs, tf.int32)[0]),
                    dtype=tf.float32))

            u_pair_mask = tf.linalg.set_diag(
                u_pair_mask,
                tf.zeros(
                    shape=tf.shape(u_pair_mask)[0],
                    dtype=tf.float32))

            u_pair_mask = tf.linalg.band_part(
                u_pair_mask,
                -1, 0)

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
                        tf.constant(1e-5, dtype=tf.float32)),
                    tf.constant(-1, dtype=tf.float32)))
            
            sigma_over_r = tf.multiply(
                sigma_pair,
                _distance_matrix_inverse)
            

            u_pair = tf.math.add(
                    tf.multiply(
                        _distance_matrix_inverse,
                        q_pair),
                    tf.multiply(
                        tf.where(
                            tf.greater(
                                _distance_matrix,
                                0.1),
                            epsilon_pair,
                            tf.zeros_like(epsilon_pair)),
                        tf.math.subtract(
                            tf.pow(
                                sigma_over_r,
                                tf.constant(12, dtype=tf.float32)),
                            tf.pow(
                                sigma_over_r,
                                tf.constant(6, dtype=tf.float32)))))


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

            u_pair_tot = tf.matmul(
                    tf.transpose(
                        tf.where(
                            atom_in_mol,
                            tf.ones_like(atom_in_mol, dtype=tf.float32),
                            tf.zeros_like(atom_in_mol, dtype=tf.float32))),
                    tf.reduce_sum(
                        u_pair,
                        axis=1,
                        keepdims=True))

            u0_tot = tf.matmul(
                    tf.transpose(
                        tf.where(
                            atom_in_mol,
                            tf.ones_like(atom_in_mol, dtype=tf.float32),
                            tf.zeros_like(atom_in_mol, dtype=tf.float32))),
                    self.d_e0_1(
                        self.d_e0_0(
                            h_v)))

            u_tot = tf.squeeze(u0_tot +\
                u_bond_tot + u_angle_tot + u_dihedral_tot + u_pair_tot)
            
            # print(u_pair_tot, u_bond_tot, u_angle_tot, u_dihedral_tot)
            return u_tot

    gn = gin.probabilistic.gn_plus.GraphNet(
            f_e=lime.nets.for_gn.ConcatenateThenFullyConnect(
                (point['D_E'], 'elu', point['D_E'], 'tanh')),
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
            phi_v=lime.nets.for_gn.ConcatenateThenFullyConnect(
                (point['phi_v_0'], point['phi_v_a_0'], point['D_V'],
                point['phi_v_a_1'])),

            phi_u=lime.nets.for_gn.ConcatenateThenFullyConnect(
                (point['phi_u_0'], point['phi_u_a_0'], point['D_U'],
                point['phi_v_a_1'])),
            f_r=f_r(),
            repeat=5)

    opt = tf.keras.optimizers.Adam(1e-3)


def obj_fn(point):
    point = dict(zip(config_space.keys(), point))
    init(point)

    for dummy_idx in range(100):
        for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_tr:
            atoms = atoms_[:, 0]
            coordinates = tf.Variable(atoms_[:, 1:4])
            with tf.GradientTape() as tape:
                bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
                                .get_geometric_idxs(atoms, adjacency_map)

                u_hat = gn(
                        atoms, adjacency_map, atom_in_mol, 
                        bond_in_mol, 
                        attr_in_mol,
                        attr_in_mol=attr_in_mol,
                        bond_idxs=bond_idxs,
                        angle_idxs=angle_idxs,
                        torsion_idxs=torsion_idxs,
                        coordinates=coordinates)

                u = tf.boolean_mask(
                    u[:, -1],
                    attr_in_mol)
                
                loss = tf.keras.losses.MAPE(u, u_hat)

            # print(loss, flush=True)
            
            variables = gn.variables
            grad = tape.gradient(loss, variables)
            # if not tf.reduce_any([tf.reduce_any(tf.math.is_nan(_grad)) for _grad in grad]).numpy():

            opt.apply_gradients(
                    zip(grad, variables))


    y_true_tr = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_tr = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_vl = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_vl = -1. * tf.ones([1, ], dtype=tf.float32)

    y_true_te = -1. * tf.ones([1, ], dtype=tf.float32)
    y_pred_te = -1. * tf.ones([1, ], dtype=tf.float32)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_tr:
            atoms = atoms_[:, 0]
            coordinates = tf.Variable(atoms_[:, 1:4])

            bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
                            .get_geometric_idxs(atoms, adjacency_map)

            u_hat = gn(
                    atoms, adjacency_map, atom_in_mol, 
                    bond_in_mol, 
                    attr_in_mol,
                    attr_in_mol=attr_in_mol,
                    bond_idxs=bond_idxs,
                    angle_idxs=angle_idxs,
                    torsion_idxs=torsion_idxs,
                    coordinates=coordinates)

            u = tf.boolean_mask(
                u[:, -1],
                attr_in_mol)


            y_true_tr = tf.concat([y_true_tr, tf.reshape(u, [-1])], axis=0)
            y_pred_tr = tf.concat([y_pred_tr, tf.reshape(u_hat, [-1])], axis=0)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_te:
            atoms = atoms_[:, 0]
            coordinates = tf.Variable(atoms_[:, 1:4])

            bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
                            .get_geometric_idxs(atoms, adjacency_map)

            u_hat = gn(
                    atoms, adjacency_map, atom_in_mol, 
                    bond_in_mol, 
                    attr_in_mol,
                    attr_in_mol=attr_in_mol,
                    bond_idxs=bond_idxs,
                    angle_idxs=angle_idxs,
                    torsion_idxs=torsion_idxs,
                    coordinates=coordinates)

            u = tf.boolean_mask(
                u[:, -1],
                attr_in_mol)

            y_true_te = tf.concat([y_true_te, tf.reshape(u, [-1])], axis=0)
            y_pred_te = tf.concat([y_pred_te, tf.reshape(u_hat, [-1])], axis=0)

    for atoms_, adjacency_map, atom_in_mol, bond_in_mol, u, attr_in_mol in ds_vl:
            atoms = atoms_[:, 0]
            coordinates = tf.Variable(atoms_[:, 1:4])

            bond_idxs, angle_idxs, torsion_idxs = gin.probabilistic.gn_hyper\
                            .get_geometric_idxs(atoms, adjacency_map)

            u_hat = gn(
                    atoms, adjacency_map, atom_in_mol, 
                    bond_in_mol, 
                    attr_in_mol,
                    attr_in_mol=attr_in_mol,
                    bond_idxs=bond_idxs,
                    angle_idxs=angle_idxs,
                    torsion_idxs=torsion_idxs,
                    coordinates=coordinates)

            u = tf.boolean_mask(
                u[:, -1],
                attr_in_mol)

            y_true_vl = tf.concat([y_true_vl, tf.reshape(u, [-1])], axis=0)
            y_pred_vl = tf.concat([y_pred_vl, tf.reshape(u_hat, [-1])], axis=0)

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


lime.optimize.dummy.optimize(obj_fn, config_space.values(), 1000)
