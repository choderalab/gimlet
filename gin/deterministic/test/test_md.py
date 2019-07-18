import tensorflow as tf
import gin

caffeine = list(gin.i_o.from_sdf.to_ds('data/caffeine.sdf'))[0]
print(caffeine)
# caffeine = gin.i_o.from_smiles.smiles_to_mol('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
# caffeine = gin.deterministic.hydrogen.add_hydrogen(caffeine)
caffeine_md_system = gin.deterministic.md.SingleMoleculeMechanicsSystem(caffeine)

caffeine_md_system.minimize()
