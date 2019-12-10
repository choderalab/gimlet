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

HARTREE_TO_KCAL_PER_MOL = 627.509
BORN_TO_ANGSTROM = 0.529177
HARTREE_PER_BORN_TO_KCAL_PER_MOL_PER_ANGSTROM = 1185.993

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
                        snapshot.properties.scf_total_energy * HARTREE_TO_KCAL_PER_MOL,
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
                        mol.geometry * BORN_TO_ANGSTROM,
                        dtype=tf.float32)

                    jacobian = tf.convert_to_tensor(
                        snapshot.return_result * HARTREE_PER_BORN_TO_KCAL_PER_MOL_PER_ANGSTROM,
                        dtype=tf.float32)

                    atoms = tf.concat(
                        [
                            features,
                            xyz,
                            jacobian
                        ],
                    axis=1)

                    yield(atoms, adjacency_map, energy)

def params_to_potential(
        q, sigma, epsilon,
        e_l, e_k,
        a_l, a_k,
        t_l, t_k,
        bond_idxs, angle_idxs, torsion_idxs,
        coordinates,
        atom_in_mol=False,
        bond_in_mol=False,
        attr_in_mol=False):

    if tf.logical_not(tf.reduce_any(atom_in_mol)):
        atom_in_mol = tf.tile(
            [[True]],
            [n_atoms, 1])

    if tf.logical_not(tf.reduce_any(bond_in_mol)):
        bond_in_mol = tf.tile(
            [[True]],
            [n_bonds, 1])

    if tf.logical_not(tf.reduce_any(attr_in_mol)):
        attr_in_mol = tf.constant([[True]])

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


    u_bond = tf.math.multiply(
            e_k,
            tf.math.pow(
                tf.math.subtract(
                    bond_distances,
                    e_l),
                tf.constant(2, dtype=tf.float32)))

    u_angle = tf.math.multiply(
        a_k,
        tf.math.pow(
            tf.math.subtract(
                angle_angles,
                a_l),
            tf.constant(2, dtype=tf.float32)))

    u_dihedral = tf.math.multiply(
        t_k,
        tf.math.pow(
            tf.math.subtract(
                torsion_dihedrals,
                t_l),
            tf.constant(2, dtype=tf.float32)))

    n_atoms = tf.shape(q, tf.int64)[0]
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

    u_pair_mask = tf.linalg.set_diag(
        u_pair_mask,
        tf.zeros(
            shape=tf.shape(u_pair_mask)[0],
            dtype=tf.float32))

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

    '''
    u_pair = tf.math.add(
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
                        tf.constant(6, dtype=tf.float32)))))
    '''

    u_pair = tf.multiply(
        epsilon_pair,
        tf.math.subtract(
            tf.pow(
                sigma_over_r,
                tf.constant(12, dtype=tf.float32)),
            tf.pow(
                sigma_over_r,
                tf.constant(6, dtype=tf.float32))))

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
    u_tot = tf.squeeze(
        u_pair_tot + u_bond_tot + u_angle_tot + u_dihedral_tot)

    return u_tot

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

ds_path = tf.data.Dataset.from_tensor_slices(list(range(500)))

ds = ds_path.map(
    lambda idx: tf.py_function(
        data_loader,
        [idx],
        [tf.float32, tf.float32, tf.float32]))

for atoms, adjacency_map, energy in ds:
    print(atoms)
