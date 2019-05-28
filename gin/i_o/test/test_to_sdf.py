import gin

atoms, adjacency_map = gin.i_o.from_smiles.smiles_to_mol(
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C')

'''
conformer = gin.deterministic.conformer.Conformers(
    [atoms, adjacency_map],
    gin.deterministic.forcefields.gaff,
    gin.deterministic.typing.TypingGAFF
).get_conformers_from_distance_geometry(1)[0]
'''

mols = [[atoms, adjacency_map]]

gin.i_o.to_sdf.write_sdf(mols, 'caffeine_out.sdf')
