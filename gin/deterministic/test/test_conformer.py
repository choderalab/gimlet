import gin

caffeine = gin.i_o.from_smiles.smiles_to_organic_topological_molecule('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
caffeine = gin.deterministic.hydrogen.add_hydrogen(caffeine)

caffeine_conformers = gin.deterministic.conformer.Conformers(
    caffeine,
    gin.deterministic.forcefields.gaff,
    gin.deterministic.typing.TypingGAFF)

conformers = caffeine_conformers.get_conformers_from_distance_geometry(5)

gin.i_o.to_sdf.write_sdf(
    [
        [caffeine[0], caffeine[1], conformers[0]],
        # [caffeine[0], caffeine[1], conformers[1]],
        # [caffeine[0], caffeine[1], conformers[2]],
        # [caffeine[0], caffeine[1], conformers[3]],
        # [caffeine[0], caffeine[1], conformers[4]],
    ], 'caffeine_out.sdf')
