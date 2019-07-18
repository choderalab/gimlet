from gin.i_o.from_smiles import to_mols
import pandas as pd

df = pd.read_csv('data/delaney-processed.csv')
smiles_array = df[['smiles']].values.flatten()
mols = to_mols(smiles_array)

for mol in mols:
    print(mol)
