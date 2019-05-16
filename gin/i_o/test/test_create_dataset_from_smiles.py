from gin.i_o.from_smiles import *
import pandas as pd

df = pd.read_csv('data/delaney-processed.csv')
smiles_array = df[['smiles']].values.flatten()
mols = smiles_to_mols(smiles_array)
