import gin
from gin.i_o.from_smiles import *
from gin.deterministic import typing


atoms, adjacency_map = smiles_to_organic_topological_molecule('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
mol = atoms, adjacency_map
mol_type = typing.TypingGAFF(mol)

print(mol_type.is_in_conjugate_system)
