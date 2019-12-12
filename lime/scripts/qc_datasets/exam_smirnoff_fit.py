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
import qcportal as ptl
client = ptl.FractalClient()
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')
import cmiles
from simtk import openmm

HARTREE_TO_KCAL_PER_MOL = 627.509
BOHR_TO_NM = 0.0529177
HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_NM = 10587.30

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

TRANSLATION_TO_ELEMENT = {
    0: 6,
    1: 7,
    2: 8,
    3: 16,
    4: 9,
    5: 17,
    6: 53,
    7: 1}



ds_qc = client.get_collection("OptimizationDataset", "OpenFF Full Optimization Benchmark 1")
ds_name = tf.data.Dataset.from_tensor_slices(list(ds_qc.data.records))

def data_generator():
    for record_name in list(ds_qc.data.records):
        print(record_name)
        r = ds_qc.get_record(record_name, specification='default')
        if r is not None:
            traj = r.get_trajectory()
            if traj is not None:
                for snapshot in traj:
                    energy = tf.convert_to_tensor(
                        snapshot.properties.scf_total_energy * HARTREE_TO_KCAL_PER_MOL,
                        dtype=tf.float32)

                    mol = snapshot.get_molecule()
                    # mol = snapshot.get_molecule().dict(encoding='json')
                    
                    atoms = tf.convert_to_tensor(
                        [TRANSLATION[atomic_number] for atomic_number in mol.atomic_numbers],
                        dtype=tf.int64)
                    
                    
                    zeros = tf.zeros(
                        (
                            tf.shape(atoms, tf.int64)[0],
                            tf.shape(atoms, tf.int64)[0]
                        ),
                        dtype=tf.float32)
                    

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

                    xyz = tf.convert_to_tensor(
                        mol.geometry * BOHR_TO_NM,
                        dtype=tf.float32)

                    jacobian = tf.convert_to_tensor(
                        snapshot.return_result\
                        * HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_NM,
                        dtype=tf.float32)

                    mol = cmiles.utils.load_molecule(mol.dict(encoding='json'))

                    top = Topology.from_molecules(Molecule.from_openeye(mol))
                    sys = FF.create_openmm_system(top)

                    angles = tf.convert_to_tensor(
                            [[x[0], x[1], x[2], 
                                x[3]._value, 
                                x[4]._value] for x in\
                            [sys.getForces(
                                )[0].getAngleParameters(idx)\
                                for idx in range(sys.getForces(
                                    )[0].getNumAngles())]],
                            dtype=tf.float32)
                    

                    bonds = tf.convert_to_tensor([[x[0], x[1], 
                                x[2]._value, 
                                x[3]._value]  for x in\
                            [sys.getForces(
                                )[1].getBondParameters(idx)\
                                for idx in range(sys.getForces(
                                    )[1].getNumBonds())]],
                            dtype=tf.float32)


                    torsions = tf.convert_to_tensor([
                        [x[0], x[1], x[2], x[3], x[4], x[5]._value, x[6]._value] for x in\
                            [sys.getForces(
                                )[3].getTorsionParameters(idx)\
                                for idx in range(sys.getForces(
                                    )[3].getNumTorsions())]],
                            dtype=tf.float32)


                    particle_params = tf.convert_to_tensor([[
                            x[0]._value,
                            x[1]._value,
                            x[2]._value
                            ] for x in\
                            [sys.getForces(
                                )[2].getParticleParameters(idx)\
                                for idx in range(sys.getForces(
                                    )[2].getNumParticles())]])

                    
                    yield(
                        atoms,
                        adjacency_map,
                        energy,
                        xyz,
                        jacobian,
                        angles,
                        bonds,
                        torsions,
                        particle_params,
                        sys)
    

# @tf.function
def params_to_potential(
        q, sigma, epsilon,
        e_l, e_k,
        a_l, a_k,
        t_l, t_k,
        bond_idxs, angle_idxs, torsion_idxs,
        coordinates,
        atom_in_mol=tf.constant(False),
        bond_in_mol=tf.constant(False),
        attr_in_mol=tf.constant(False)):
    
    n_atoms = tf.shape(q, tf.int64)[0]
    n_angles = tf.shape(angle_idxs, tf.int64)[0]
    n_torsions = tf.shape(torsion_idxs, tf.int64)[0]
    n_bonds = tf.shape(bond_idxs, tf.int64)[0]
    
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


    u_bond = 0.5 * tf.math.multiply(
            e_k,
            tf.math.pow(
                tf.math.subtract(
                    bond_distances,
                    e_l),
                tf.constant(2, dtype=tf.float32)))

    u_angle = 0.5 * tf.math.multiply(
        a_k,
        tf.math.pow(
            tf.math.subtract(
                tf.math.acos(angle_angles),
                a_l),
            tf.constant(2, dtype=tf.float32)))
    
    u_dihedral = tf.math.multiply(
        t_k,
        tf.math.pow(
            tf.math.subtract(
                torsion_dihedrals,
                t_l),
            tf.constant(2, dtype=tf.float32)))

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
                tf.constant(1e-5, dtype=tf.float32)),
            tf.constant(-1, dtype=tf.float32)))

    sigma_over_r = tf.multiply(
        sigma_pair,
        _distance_matrix_inverse)
    
    u_coulomb = tf.multiply(
                tf.pow(
                    _distance_matrix_inverse,
                    tf.constant(2, dtype=tf.float32)),
                tf.multiply(
                    138.93 * q_pair,
                    tf.tensor_scatter_nd_update(
                        tf.ones_like(q_pair),
                        tf.stack(
                            [
                                torsion_idxs[:, 0],
                                torsion_idxs[:, 3]
                            ],
                            axis=1),
                        tf.constant(
                            0.833,
                            shape=(
                                tf.shape(torsion_idxs)[0],
                            ),
                            dtype=tf.float32))))

    u_lj = tf.multiply(
                tf.where(
                    tf.less(
                        _distance_matrix,
                        0.1),
                    tf.zeros_like(epsilon_pair),
                    tf.multiply(
                        epsilon_pair,
                        tf.tensor_scatter_nd_update(
                            tf.ones_like(epsilon_pair),
                            tf.stack(
                                [
                                    torsion_idxs[:, 0],
                                    torsion_idxs[:, 3]
                                ],
                                axis=1),
                            tf.constant(
                                0.5,
                                shape=(
                                    tf.shape(torsion_idxs)[0],
                            ),
                            dtype=tf.float32)))),
                tf.math.subtract(
                    tf.pow(
                        sigma_over_r,
                        tf.constant(12, dtype=tf.float32)),
                    tf.pow(
                        sigma_over_r,
                        tf.constant(6, dtype=tf.float32))))
    
    
    # print(tf.reduce_sum(u_coulomb))
    u_pair = u_coulomb + u_lj
    
    print(_distance_matrix_inverse)
    print(q_pair)

    print(tf.reduce_sum(u_coulomb))
    print(tf.reduce_sum(u_lj))

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
    
    # print(u_angle_tot, u_bond_tot, u_pair_tot)
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
    

for atoms, adjacency_map, energy, xyz, jacobian, angles, bonds, torsions,\
        particle_params, sys\
    in data_generator():

    q, sigma, epsilon = tf.split(particle_params, 3, 1)
    e_l = bonds[:, 2]
    e_k = bonds[:, 3]
    bond_idxs = tf.cast(bonds[:, :2], tf.int64)
   
    a_l = angles[:, 3]
    a_k = angles[:, 4]
    angle_idxs = tf.cast(angles[:, :3], tf.int64)
 
    # xyz = tf.Variable(xyz * BOHR_TO_ANGSTROM)
    # jacobian = jacobian * HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM
    
    xyz = tf.Variable(xyz)

    with tf.GradientTape() as tape:
        u = -params_to_potential(
            q,
            sigma,
            epsilon,
            e_l, e_k,
            a_l, a_k,
            tf.constant([0.0], dtype=tf.float32),
            tf.constant([0.0], dtype=tf.float32),
            bond_idxs, angle_idxs, tf.constant([[0, 0, 0, 0]], dtype=tf.int64),
            xyz)

    
    jacobian_hat = tape.gradient(u, xyz)
    
    for idx in range(sys.getNumForces()):
        force = sys.getForce(idx)
        force.setForceGroup(idx)
    
    context = openmm.Context(sys, openmm.VerletIntegrator(0.001))
    
    context.setPositions(xyz * 1.0)

    force = sys.getForce(2)
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.updateParametersInContext(context)
    print(context.getState(getEnergy=True, groups=1<<2).getPotentialEnergy())
    
    print(tf.stack(
        [
            jacobian_hat,
            context.getState(
                getVelocities=True,
                getForces=True).getForces(asNumpy=True)._value,
            jacobian
        ],
        axis=1))
