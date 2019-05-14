import gin
from gin.i_o.from_smiles import *
from gin.deterministic import typing


atoms, adjacency_map = smiles_to_organic_topological_molecule('Cc1occc1C(=O)Nc2ccccc2')
mol = gin.molecule.Molecule(atoms, adjacency_map)
mol_type = typing.TypingGAFF(mol)
print(mol_type.get_assignment())
