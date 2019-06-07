import gin
import rdkit
from rdkit import Chem
import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

BONDS = {
    Chem.BondType.SINGLE:1.0,
    Chem.BondType.DOUBLE:2.0,
    Chem.BondType.TRIPLE:3.0,
    Chem.BondType.AROMATIC:1.5,
    Chem.BondType.UNSPECIFIED:0.0
}

def get_adjacency_matrix_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    # initialize an adjacency_map
    adjacency_map = np.zeros((n_atoms, n_atoms))

    # get a list of bonds
    bonds = mol.GetBonds()

    # loop through these bonds
    for bond in bonds:
        # order = BONDS[bond.GetBondType()]
        atom0_idx = bond.GetBeginAtomIdx()
        atom1_idx = bond.GetEndAtomIdx()
        adjacency_map[atom0_idx, atom1_idx] = 1.
        adjacency_map[atom1_idx, atom0_idx] = 1.

    # adjacency_map = np.triu(adjacency_map)

    return adjacency_map

def get_num_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.rdmolops.RemoveHs(mol)
    bonds = mol.GetBonds()
    return len(bonds)

def get_eigenvalues_from_adjacency_map(adjacency_map):
    eigen_values, _ = np.linalg.eigh(adjacency_map)
    return eigen_values


df = pd.read_csv('data/delaney-processed.csv')
smiles_array = df[['smiles']].values.flatten()

'''
@pytest.mark.parametrize('smiles', smiles_array)
def test_num_bonds(smiles):
    npt.assert_almost_equal(
        get_num_bonds(smiles),
        np.count_nonzero(
            gin.i_o.from_smiles.smiles_to_mol(
                smiles)[1]))
'''

@pytest.mark.parametrize('smiles', smiles_array)
def test_adjacency_map(smiles):
    adjacency_map_rdkit = get_adjacency_matrix_rdkit(smiles)
    adjacency_map_gin = gin.i_o.from_smiles.smiles_to_mol(
        smiles)[1]

    adjacency_map_gin = tf.where(
        tf.greater(
            adjacency_map_gin,
            tf.constant(0, dtype=tf.float32)),
        tf.ones_like(adjacency_map_gin),
        tf.zeros_like(adjacency_map_gin))

    adjacency_map_gin = adjacency_map_gin + tf.transpose(adjacency_map_gin)

    eighs_rdkit = get_eigenvalues_from_adjacency_map(
        adjacency_map_rdkit)

    eighs_gin = get_eigenvalues_from_adjacency_map(
        adjacency_map_gin)

    err_msg = str(adjacency_map_rdkit) + str(adjacency_map_gin)
    npt.assert_almost_equal(
        eighs_rdkit,
        eighs_gin,
        err_msg = err_msg)
