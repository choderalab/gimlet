import gin

caffeine = gin.i_o.from_smiles.smiles_to_organic_topological_molecule('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
caffeine_conformers = gin.deterministic.conformer.Conformers(
    caffeine,
    gin.deterministic.forcefields.gaff,
    gin.deterministic.typing.TypingGAFF)

print(caffeine_conformers.get_conformers_from_distance_geometry(5))
