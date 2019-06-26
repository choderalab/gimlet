import tensorflow as tf
import gin

caffeine = gin.i_o.from_smiles.smiles_to_organic_topological_molecule('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
caffeine = gin.deterministic.hydrogen.add_hydrogen(caffeine)
caffeine_md_system = gin.deterministic.md.SingleMoleculeMechanicsSystem(caffeine)

caffeine_md_system.minimize()
