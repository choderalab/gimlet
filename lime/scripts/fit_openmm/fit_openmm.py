# =============================================================================
# imports
# =============================================================================
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)
# from sklearn import metrics
import gin
import lime
import pandas as pd
import numpy as np
from openforcefield.topology import Molecule
from openforcefield.topology import Topology
from openforcefield.typing.engines.smirnoff import ForceField
FF = ForceField('test_forcefields/smirnoff99Frosst.offxml')

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

df = pd.read_csv('delaney-processed.csv')
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

def data_generator():
    for smiles in x_array:
        mol = Molecule.from_smiles(smiles)
        topology = Topology.from_molecules(mol)
        mol_sys = FF.create_openmm_system(topology)
        n_atoms = topology.n_topology_atoms
        atoms = [TRANSLATION[atom._atomic_number] for atom in mol.atoms]

        adjacency_map = np.zeros(n_atoms, n_atoms)

        for bond in mol.bonds:
            assert bond.atom1_index < bond.atom2_index

            adjacency_map[bond.atom1_index, bond.atom2_index] = \
                bond.bond_order

        print(atoms, adjacency_map)


data_generator()
